"""Offline feature store for Parquet/S3 history, training data, and lineage."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    import pandas as pd  # pyright: ignore[reportMissingImports]

    _pandas_available = True
except Exception:
    pd = None  # type: ignore[assignment]
    _pandas_available = False

try:
    import pyarrow  # pyright: ignore[reportMissingImports]  # noqa: F401

    _parquet_available = True
except Exception:
    _parquet_available = False

try:
    import boto3  # pyright: ignore[reportMissingImports]

    _boto3_available = True
except Exception:
    boto3 = None  # type: ignore[assignment]
    _boto3_available = False


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class FeatureLineageRecord:
    feature_name: str
    entity_id: str
    source: str
    computation_version: str
    event_ts: datetime
    stored_at: datetime = field(default_factory=_utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class OfflineFeatureStore:
    """Persists historical features for backfills and training-set generation."""

    def __init__(
        self,
        *,
        base_path: str = "data/feature_store_streaming/offline",
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "feature-store-streaming",
    ) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix.strip("/")
        self._lock = threading.RLock()
        self._lineage_path = self.base_path / "lineage.jsonl"
        self._history_path = self.base_path / "historical_features.jsonl"
        boto3_module = boto3
        self._s3_client = boto3_module.client("s3") if _boto3_available and boto3_module is not None and s3_bucket else None

    def write_feature_batch(self, rows: Sequence[Mapping[str, Any]]) -> Path:
        if not rows:
            return self._history_path
        normalized = [self._normalize_row(row) for row in rows]
        with self._lock:
            parquet_path = self.base_path / f"features_{normalized[-1]['event_ts'].replace(':', '-')}.parquet"
            pandas_module = pd
            if _pandas_available and _parquet_available and pandas_module is not None:
                frame = pandas_module.DataFrame(normalized)
                frame.to_parquet(parquet_path, index=False)
                output_path = parquet_path
            else:
                with self._history_path.open("a", encoding="utf-8") as handle:
                    for row in normalized:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                output_path = self._history_path
            self._upload_if_configured(output_path)
            return output_path

    def generate_training_data(
        self,
        *,
        entity_ids: Optional[Sequence[str]] = None,
        feature_names: Optional[Sequence[str]] = None,
        start_ts: Optional[datetime] = None,
        end_ts: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        rows = self._load_rows()
        result: List[Dict[str, Any]] = []
        for row in rows:
            if entity_ids and row["entity_id"] not in entity_ids:
                continue
            if feature_names and row["feature_name"] not in feature_names:
                continue
            event_ts = datetime.fromisoformat(row["event_ts"])
            if start_ts and event_ts < start_ts:
                continue
            if end_ts and event_ts > end_ts:
                continue
            result.append(row)
        return result

    def record_lineage(self, records: Sequence[FeatureLineageRecord]) -> None:
        if not records:
            return
        with self._lock:
            with self._lineage_path.open("a", encoding="utf-8") as handle:
                for record in records:
                    payload = asdict(record)
                    payload["event_ts"] = record.event_ts.isoformat()
                    payload["stored_at"] = record.stored_at.isoformat()
                    handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def get_lineage(self, feature_name: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self._lineage_path.exists():
            return []
        with self._lineage_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if feature_name is None:
            return rows
        return [row for row in rows if row.get("feature_name") == feature_name]

    def _load_rows(self) -> List[Dict[str, Any]]:
        parquet_rows: List[Dict[str, Any]] = []
        pandas_module = pd
        if _pandas_available and _parquet_available and pandas_module is not None:
            for parquet_file in sorted(self.base_path.glob("features_*.parquet")):
                try:
                    frame = pandas_module.read_parquet(parquet_file)
                except Exception:
                    logger.warning("Failed to read parquet file %s", parquet_file, exc_info=True)
                    continue
                parquet_rows.extend(frame.to_dict(orient="records"))
        jsonl_rows: List[Dict[str, Any]] = []
        if self._history_path.exists():
            with self._history_path.open("r", encoding="utf-8") as handle:
                jsonl_rows.extend(json.loads(line) for line in handle if line.strip())
        return parquet_rows + jsonl_rows

    def _normalize_row(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        event_ts = row.get("event_ts")
        if isinstance(event_ts, datetime):
            event_ts = event_ts if event_ts.tzinfo else event_ts.replace(tzinfo=timezone.utc)
            event_ts_str = event_ts.isoformat()
        elif isinstance(event_ts, str) and event_ts:
            event_ts_str = datetime.fromisoformat(event_ts.replace("Z", "+00:00")).isoformat()
        else:
            event_ts_str = _utc_now().isoformat()
        return {
            "entity_id": str(row.get("entity_id", "UNKNOWN")),
            "feature_name": str(row.get("feature_name", "unknown_feature")),
            "value": row.get("value"),
            "event_ts": event_ts_str,
            "source": str(row.get("source", "streaming")),
            "version": int(row.get("version", 1)),
        }

    def _upload_if_configured(self, path: Path) -> None:
        if self._s3_client is None or self.s3_bucket is None:
            return
        try:
            key = f"{self.s3_prefix}/{path.name}"
            self._s3_client.upload_file(str(path), self.s3_bucket, key)
        except Exception as exc:
            logger.warning("Failed to upload offline feature file to S3 (%s)", exc)
