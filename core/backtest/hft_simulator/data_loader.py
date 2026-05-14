"""Data loading utilities for HFT replay datasets."""
# pyright: reportMissingImports=false

from __future__ import annotations

import bz2
import csv
import gzip
import logging
import lzma
import pickle
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when replay data does not satisfy minimum requirements."""


@dataclass
class DataLoader:
    cache_dir: str | None = None

    def load_csv(self, path: str) -> list[dict[str, Any]]:
        with Path(path).open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.validate_data(rows)
        return rows

    def load_parquet(self, path: str) -> list[dict[str, Any]]:
        file_path = Path(path)
        try:
            import pandas as pd  # type: ignore
        except Exception:
            try:
                import pyarrow.parquet as pq  # type: ignore
            except Exception as exc:
                raise ImportError("Parquet loading requires pandas or pyarrow") from exc
            table = pq.read_table(file_path)
            rows = table.to_pylist()
        else:
            rows = pd.read_parquet(file_path).to_dict(orient="records")
        self.validate_data(rows)
        return rows

    def decompress_itch_data(self, path: str, output_path: str | None = None) -> str:
        source = Path(path)
        target = Path(output_path) if output_path else source.with_suffix("")
        readers = {
            ".gz": gzip.open,
            ".bz2": bz2.open,
            ".xz": lzma.open,
        }
        opener = readers.get(source.suffix.lower())
        if opener is None:
            raise ValueError(f"unsupported compressed format: {source.suffix}")
        with opener(source, "rb") as src, target.open("wb") as dst:
            dst.write(src.read())
        return str(target)

    def validate_data(self, rows: Iterable[dict[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            raise DataValidationError("dataset is empty")
        required = {"timestamp_ns", "message_type", "order_id", "side", "price", "quantity", "symbol"}
        missing = required.difference(rows[0].keys())
        if missing:
            raise DataValidationError(f"dataset missing required columns: {sorted(missing)}")
        previous_ts = -1
        for index, row in enumerate(rows, start=1):
            timestamp_ns = int(row["timestamp_ns"])
            if timestamp_ns < previous_ts:
                raise DataValidationError(f"timestamps not monotonic at row {index}")
            if float(row["price"]) < 0 or float(row["quantity"]) < 0:
                raise DataValidationError(f"negative price/quantity at row {index}")
            previous_ts = timestamp_ns

    def cache_processed_data(self, cache_key: str, data: Any) -> str:
        if not self.cache_dir:
            raise ValueError("cache_dir is not configured")
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_key}.pkl"
        with cache_path.open("wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return str(cache_path)

    def load_cached_data(self, cache_key: str) -> Any:
        if not self.cache_dir:
            return None
        cache_path = Path(self.cache_dir) / f"{cache_key}.pkl"
        if not cache_path.exists():
            return None
        with cache_path.open("rb") as handle:
            return pickle.load(handle)
