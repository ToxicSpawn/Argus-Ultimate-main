"""
Feast-compatible real-time feature store for ARGUS.

Provides OnlineStore, OfflineStore, FeatureRegistry, FeastFeatureStore,
and StreamingFeatureProcessor with SQLite backend, NumPy/pandas ops,
and TTL-aware caching for institutional-grade ML feature management.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_VALID_VALUE_TYPES = {"INT64", "FLOAT", "DOUBLE", "STRING", "BYTES", "BOOL"}
_VALID_ENTITY_TYPES = {"STRING", "INT64"}


@dataclass(slots=True)
class Feature:
    name: str
    dtype: str
    description: str = ""

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        self.dtype = str(self.dtype).strip().upper()
        self.description = str(self.description).strip()
        if not self.name:
            raise ValueError("Feature name must be non-empty")
        if self.dtype not in _VALID_VALUE_TYPES:
            raise ValueError(f"Invalid Feature dtype: {self.dtype!r}")


@dataclass(slots=True)
class FeatureView:
    name: str
    entities: List[str]
    features: List[Feature]
    ttl_seconds: int = 3600
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        self.entities = [str(e).strip() for e in self.entities]
        self.features = [f if isinstance(f, Feature) else Feature(**f) if isinstance(f, dict) else f for f in self.features]
        self.ttl_seconds = int(self.ttl_seconds)
        self.description = str(self.description).strip()
        if not self.name:
            raise ValueError("FeatureView name must be non-empty")
        if not self.entities:
            raise ValueError("FeatureView must have at least one entity")
        if not self.features:
            raise ValueError("FeatureView must have at least one feature")
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entities": self.entities,
            "features": [{"name": f.name, "dtype": f.dtype, "description": f.description} for f in self.features],
            "ttl_seconds": self.ttl_seconds,
            "description": self.description,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureView":
        features = [Feature(name=f["name"], dtype=f["dtype"], description=f.get("description", "")) for f in data["features"]]
        return cls(
            name=data["name"],
            entities=data["entities"],
            features=features,
            ttl_seconds=data.get("ttl_seconds", 3600),
            description=data.get("description", ""),
            tags=data.get("tags", {}),
        )


class OnlineStore:
    """Low-latency online feature store backed by SQLite with TTL expiration."""

    def __init__(self, db_path: str = "data/feast_online.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        logger.info("OnlineStore initialized at %s", self.db_path)

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS feature_values (
                entity_key TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                feature_view TEXT NOT NULL,
                value TEXT,
                event_ts REAL NOT NULL,
                created_ts REAL NOT NULL,
                PRIMARY KEY (entity_key, feature_name, feature_view)
            );
            CREATE INDEX IF NOT EXISTS idx_entity_feature
                ON feature_values(entity_key, feature_name);
            CREATE INDEX IF NOT EXISTS idx_feature_view
                ON feature_values(feature_view);
        """)
        self._conn.commit()

    def get(self, entity_key: str, feature_name: str, feature_view: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT value, event_ts FROM feature_values WHERE entity_key=? AND feature_name=? AND feature_view=?",
                (entity_key, feature_name, feature_view),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            value, event_ts = row
            if self._is_expired(event_ts, feature_view):
                self._delete(entity_key, feature_name, feature_view)
                return None
            return {"value": json.loads(value), "event_ts": datetime.fromtimestamp(event_ts, tz=timezone.utc)}

    def write(self, entity_key: str, feature_view: str, values: Dict[str, Any], event_ts: Optional[datetime] = None) -> None:
        now = datetime.now(timezone.utc)
        ts = event_ts if event_ts else now
        ts_epoch = ts.timestamp()
        created_epoch = now.timestamp()

        with self._lock:
            for fname, fval in values.items():
                serialized = json.dumps(fval)
                self._conn.execute(
                    """INSERT OR REPLACE INTO feature_values
                       (entity_key, feature_name, feature_view, value, event_ts, created_ts)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (entity_key, fname, feature_view, serialized, ts_epoch, created_epoch),
                )
            self._conn.commit()
        logger.debug("OnlineStore wrote %d features for entity %s view %s", len(values), entity_key, feature_view)

    def batch_get(self, entity_keys: List[str], feature_names: List[str], feature_view: str) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            for ek in entity_keys:
                results[ek] = {}
                for fn in feature_names:
                    cursor = self._conn.execute(
                        "SELECT value, event_ts FROM feature_values WHERE entity_key=? AND feature_name=? AND feature_view=?",
                        (ek, fn, feature_view),
                    )
                    row = cursor.fetchone()
                    if row is not None:
                        value, event_ts = row
                        if not self._is_expired(event_ts, feature_view):
                            results[ek][fn] = json.loads(value)
                        else:
                            self._delete(ek, fn, feature_view)
        return results

    def _is_expired(self, event_ts_epoch: float, feature_view: str) -> bool:
        ttl = self._get_ttl(feature_view)
        if ttl <= 0:
            return False
        return time.time() > event_ts_epoch + ttl

    def _get_ttl(self, feature_view: str) -> int:
        cursor = self._conn.execute(
            "SELECT value FROM feature_values WHERE feature_view=? LIMIT 1",
            (feature_view,),
        )
        row = cursor.fetchone()
        if row is None:
            return 0
        return 3600

    def _delete(self, entity_key: str, feature_name: str, feature_view: str) -> None:
        self._conn.execute(
            "DELETE FROM feature_values WHERE entity_key=? AND feature_name=? AND feature_view=?",
            (entity_key, feature_name, feature_view),
        )
        self._conn.commit()

    def scan_all(self, feature_view: str, limit: int = 10000) -> List[Dict[str, Any]]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT entity_key, feature_name, value, event_ts FROM feature_values WHERE feature_view=? LIMIT ?",
                (feature_view, limit),
            )
            rows = cursor.fetchall()
        result = []
        for ek, fn, val, ets in rows:
            if not self._is_expired(ets, feature_view):
                result.append({"entity_key": ek, "feature_name": fn, "value": json.loads(val), "event_ts": ets})
        return result

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        logger.info("OnlineStore closed")


class OfflineStore:
    """Historical feature store for training dataset generation."""

    def __init__(self, db_path: str = "data/feast_offline.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        logger.info("OfflineStore initialized at %s", self.db_path)

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS historical_features (
                entity_key TEXT NOT NULL,
                feature_view TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                value TEXT,
                event_ts REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_historical_entity
                ON historical_features(entity_key, event_ts);
            CREATE INDEX IF NOT EXISTS idx_historical_view
                ON historical_features(feature_view, event_ts);
        """)
        self._conn.commit()

    def write_historical(self, feature_view: str, records: List[Dict[str, Any]]) -> None:
        with self._lock:
            for rec in records:
                ek = rec["entity_key"]
                ets = rec["event_ts"]
                if isinstance(ets, datetime):
                    ets = ets.timestamp()
                for fn, fv in rec["values"].items():
                    serialized = json.dumps(fv)
                    self._conn.execute(
                        "INSERT INTO historical_features (entity_key, feature_view, feature_name, value, event_ts) VALUES (?, ?, ?, ?, ?)",
                        (ek, feature_view, fn, serialized, ets),
                    )
            self._conn.commit()
        logger.info("OfflineStore wrote %d records for view %s", len(records), feature_view)

    def get_historical_features(
        self,
        feature_view: str,
        entity_keys: List[str],
        feature_names: List[str],
        start_ts: datetime,
        end_ts: datetime,
    ) -> pd.DataFrame:
        start_epoch = start_ts.timestamp()
        end_epoch = end_ts.timestamp()

        with self._lock:
            placeholders = ",".join("?" * len(entity_keys))
            fn_placeholders = ",".join("?" * len(feature_names))
            query = f"""
                SELECT entity_key, feature_name, value, event_ts
                FROM historical_features
                WHERE feature_view=?
                  AND entity_key IN ({placeholders})
                  AND feature_name IN ({fn_placeholders})
                  AND event_ts BETWEEN ? AND ?
                ORDER BY entity_key, event_ts
            """
            params = [feature_view] + entity_keys + feature_names + [start_epoch, end_epoch]
            cursor = self._conn.execute(query, params)
            rows = cursor.fetchall()

        if not rows:
            return pd.DataFrame()

        records = []
        for ek, fn, val, ets in rows:
            records.append({
                "entity_key": ek,
                "feature_name": fn,
                "value": json.loads(val),
                "event_ts": datetime.fromtimestamp(ets, tz=timezone.utc),
            })

        df = pd.DataFrame(records)
        if df.empty:
            return df

        pivot = df.pivot_table(index=["entity_key", "event_ts"], columns="feature_name", values="value", aggfunc="last")
        pivot = pivot.reset_index()
        pivot.columns.name = None
        return pivot

    def create_training_dataset(
        self,
        feature_view: str,
        entity_keys: List[str],
        feature_names: List[str],
        label_name: str,
        labels: Dict[str, List[Tuple[datetime, float]]],
        start_ts: datetime,
        end_ts: datetime,
    ) -> pd.DataFrame:
        features_df = self.get_historical_features(
            feature_view, entity_keys, feature_names, start_ts, end_ts,
        )
        if features_df.empty:
            logger.warning("No historical features found for training dataset")
            return pd.DataFrame()

        label_rows = []
        for ek in entity_keys:
            if ek not in labels:
                continue
            for ts, label_val in labels[ek]:
                label_rows.append({"entity_key": ek, "event_ts": ts, label_name: label_val})

        if not label_rows:
            logger.warning("No labels provided for training dataset")
            return features_df

        labels_df = pd.DataFrame(label_rows)
        labels_df["event_ts"] = pd.to_datetime(labels_df["event_ts"])
        features_df["event_ts"] = pd.to_datetime(features_df["event_ts"])

        merged = pd.merge_asof(
            labels_df.sort_values("event_ts"),
            features_df.sort_values("event_ts"),
            on="event_ts",
            by="entity_key",
            direction="backward",
        )

        merged = merged.dropna(subset=feature_names)
        logger.info("Created training dataset with %d rows, %d features", len(merged), len(feature_names))
        return merged

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        logger.info("OfflineStore closed")


class FeatureRegistry:
    """Registry for FeatureView definitions with JSON persistence."""

    def __init__(self, path: str = "data/feast_registry.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._views: Dict[str, FeatureView] = {}
        self._load()

    def register(self, view: FeatureView) -> None:
        with self._lock:
            self._views[view.name] = view
        logger.info("Registered FeatureView: %s (entities=%s, features=%d)", view.name, view.entities, len(view.features))

    def get(self, name: str) -> Optional[FeatureView]:
        return self._views.get(name)

    def list_views(self) -> List[str]:
        return list(self._views.keys())

    def save(self) -> None:
        with self._lock:
            data = {name: view.to_dict() for name, view in self._views.items()}
        self.path.write_text(json.dumps(data, indent=2))
        logger.info("FeatureRegistry saved to %s (%d views)", self.path, len(data))

    def _load(self) -> None:
        if not self.path.exists():
            logger.info("FeatureRegistry file not found, starting empty")
            return
        try:
            data = json.loads(self.path.read_text())
            with self._lock:
                for name, view_data in data.items():
                    self._views[name] = FeatureView.from_dict(view_data)
            logger.info("FeatureRegistry loaded %d views from %s", len(self._views), self.path)
        except Exception:
            logger.exception("Failed to load FeatureRegistry from %s", self.path)
            self._views = {}


class FeastFeatureStore:
    """Unified Feast-compatible feature store coordinating online/offline stores."""

    def __init__(
        self,
        registry: Optional[FeatureRegistry] = None,
        online_store: Optional[OnlineStore] = None,
        offline_store: Optional[OfflineStore] = None,
    ) -> None:
        self.registry = registry or FeatureRegistry()
        self.online_store = online_store or OnlineStore()
        self.offline_store = offline_store or OfflineStore()
        self._lock = threading.RLock()
        logger.info("FeastFeatureStore initialized")

    def get_online_features(self, feature_view_name: str, entity_rows: List[Dict[str, str]]) -> Dict[str, List[Any]]:
        view = self.registry.get(feature_view_name)
        if view is None:
            raise ValueError(f"FeatureView {feature_view_name!r} not registered")

        feature_names = [f.name for f in view.features]
        result: Dict[str, List[Any]] = {fn: [] for fn in feature_names}

        for row in entity_rows:
            ek = self._make_entity_key(view.entities, row)
            batch = self.online_store.batch_get([ek], feature_names, feature_view_name)
            values = batch.get(ek, {})
            for fn in feature_names:
                result[fn].append(values.get(fn))

        return result

    def get_historical_features(
        self,
        feature_view_name: str,
        entity_keys: List[str],
        start_ts: datetime,
        end_ts: datetime,
    ) -> pd.DataFrame:
        view = self.registry.get(feature_view_name)
        if view is None:
            raise ValueError(f"FeatureView {feature_view_name!r} not registered")

        feature_names = [f.name for f in view.features]
        return self.offline_store.get_historical_features(
            feature_view_name, entity_keys, feature_names, start_ts, end_ts,
        )

    def materialize(self, feature_view_name: str, start_ts: datetime, end_ts: datetime) -> int:
        view = self.registry.get(feature_view_name)
        if view is None:
            raise ValueError(f"FeatureView {feature_view_name!r} not registered")

        feature_names = [f.name for f in view.features]
        historical_df = self.offline_store.get_historical_features(
            feature_view_name, ["*"], feature_names, start_ts, end_ts,
        )
        if historical_df.empty:
            logger.warning("No historical data to materialize for %s", feature_view_name)
            return 0

        count = 0
        for _, row in historical_df.iterrows():
            ek = str(row.get("entity_key", ""))
            values = {fn: row.get(fn) for fn in feature_names if fn in row.index}
            event_ts = row.get("event_ts")
            if isinstance(event_ts, (int, float)):
                event_ts = datetime.fromtimestamp(event_ts, tz=timezone.utc)
            self.online_store.write(ek, feature_view_name, values, event_ts)
            count += 1

        logger.info("Materialized %d rows from %s to online store", count, feature_view_name)
        return count

    def push(self, feature_view_name: str, entity_key: str, values: Dict[str, Any], event_ts: Optional[datetime] = None) -> None:
        view = self.registry.get(feature_view_name)
        if view is None:
            raise ValueError(f"FeatureView {feature_view_name!r} not registered")

        now = event_ts or datetime.now(timezone.utc)
        self.online_store.write(entity_key, feature_view_name, values, now)

        historical_record = {
            "entity_key": entity_key,
            "event_ts": now,
            "values": values,
        }
        self.offline_store.write_historical(feature_view_name, [historical_record])
        logger.debug("Pushed %d features for entity %s to view %s", len(values), entity_key, feature_view_name)

    @staticmethod
    def _make_entity_key(entities: List[str], row: Dict[str, str]) -> str:
        parts = [str(row.get(e, "")) for e in entities]
        return "|".join(parts)

    def close(self) -> None:
        self.online_store.close()
        self.offline_store.close()
        self.registry.save()
        logger.info("FeastFeatureStore shut down cleanly")


class StreamingFeatureProcessor:
    """Real-time streaming feature computation with rolling windows and EMA."""

    def __init__(
        self,
        window_sizes: Optional[List[int]] = None,
        ema_spans: Optional[List[int]] = None,
    ) -> None:
        self.window_sizes = window_sizes or [5, 10, 20, 50]
        self.ema_spans = ema_spans or [5, 12, 26, 50]
        self._buffers: Dict[str, Deque[float]] = {}
        self._ema_states: Dict[str, Dict[int, float]] = {}
        self._lock = threading.RLock()
        logger.info(
            "StreamingFeatureProcessor initialized (windows=%s, ema_spans=%s)",
            self.window_sizes, self.ema_spans,
        )

    def update(self, symbol: str, value: float) -> Dict[str, float]:
        with self._lock:
            if symbol not in self._buffers:
                self._buffers[symbol] = deque(maxlen=max(self.window_sizes) * 2)
                self._ema_states[symbol] = {}
            self._buffers[symbol].append(value)

        features = {}
        features.update(self._compute_rolling(symbol, value))
        features.update(self._compute_ema(symbol, value))
        return features

    def _compute_rolling(self, symbol: str, value: float) -> Dict[str, float]:
        buf = list(self._buffers.get(symbol, []))
        if not buf:
            return {}

        arr = np.array(buf, dtype=float)
        features: Dict[str, float] = {}

        for w in self.window_sizes:
            if len(arr) < w:
                continue
            window = arr[-w:]
            features[f"{symbol}_rolling_mean_{w}"] = float(np.mean(window))
            features[f"{symbol}_rolling_std_{w}"] = float(np.std(window))
            features[f"{symbol}_rolling_min_{w}"] = float(np.min(window))
            features[f"{symbol}_rolling_max_{w}"] = float(np.max(window))
            features[f"{symbol}_rolling_range_{w}"] = float(np.max(window) - np.min(window))

            if len(window) >= 2:
                features[f"{symbol}_rolling_skew_{w}"] = float(self._safe_skew(window))
                features[f"{symbol}_rolling_kurt_{w}"] = float(self._safe_kurtosis(window))

        if len(arr) >= 2:
            features[f"{symbol}_returns"] = float((arr[-1] - arr[-2]) / max(abs(arr[-2]), 1e-12))
            features[f"{symbol}_momentum"] = float((arr[-1] - arr[-max(len(arr), 1)]) / max(abs(arr[-max(len(arr), 1)]), 1e-12))

        return features

    def _compute_ema(self, symbol: str, value: float) -> Dict[str, float]:
        features: Dict[str, float] = {}
        for span in self.ema_spans:
            alpha = 2.0 / (span + 1)
            prev = self._ema_states[symbol].get(span)
            if prev is None:
                self._ema_states[symbol][span] = value
                features[f"{symbol}_ema_{span}"] = value
            else:
                new_ema = alpha * value + (1 - alpha) * prev
                self._ema_states[symbol][span] = new_ema
                features[f"{symbol}_ema_{span}"] = float(new_ema)
                features[f"{symbol}_ema_diff_{span}"] = float(value - new_ema)

        if len(self.ema_spans) >= 2:
            fast_span = self.ema_spans[0]
            slow_span = self.ema_spans[-1]
            fast_ema = self._ema_states[symbol].get(fast_span)
            slow_ema = self._ema_states[symbol].get(slow_span)
            if fast_ema is not None and slow_ema is not None:
                features[f"{symbol}_ema_crossover"] = float(fast_ema - slow_ema)

        return features

    @staticmethod
    def _safe_skew(arr: np.ndarray) -> float:
        n = len(arr)
        if n < 3:
            return 0.0
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-12:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 3))

    @staticmethod
    def _safe_kurtosis(arr: np.ndarray) -> float:
        n = len(arr)
        if n < 4:
            return 0.0
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-12:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 4) - 3.0)

    def get_state(self, symbol: str) -> Dict[str, Any]:
        with self._lock:
            buf = list(self._buffers.get(symbol, []))
            ema = dict(self._ema_states.get(symbol, {}))
        return {"buffer_length": len(buf), "last_value": buf[-1] if buf else None, "ema_states": ema}

    def reset(self, symbol: Optional[str] = None) -> None:
        with self._lock:
            if symbol is None:
                self._buffers.clear()
                self._ema_states.clear()
            else:
                self._buffers.pop(symbol, None)
                self._ema_states.pop(symbol, None)
        logger.info("StreamingFeatureProcessor reset for symbol=%s", symbol)
