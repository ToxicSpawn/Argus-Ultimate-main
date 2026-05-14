"""DataFeed — OHLCV bar iterator for backtesting — Push 59.

Supports:
  - In-memory list of BarData (for tests / synthetic data)
  - CSV files with columns: timestamp, open, high, low, close, volume
  - Parquet files (requires pyarrow or fastparquet)
  - Optional resampling via pandas resample
  - warmup_bars: skip first N bars from signal dispatch
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BarData:
    """A single OHLCV bar."""
    ts: float          # Unix timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.ts, tz=timezone.utc)


class DataFeed:
    """Yields BarData objects in chronological order.

    Parameters
    ----------
    bars : list of BarData, optional
        Pre-loaded bars (in-memory). Mutually exclusive with path.
    path : Path or str, optional
        CSV or Parquet file to load.
    symbol : str
        Symbol tag applied when loading from file.
    warmup_bars : int
        Number of leading bars to emit with is_warmup=True flag
        (skipped in signal dispatch but used to prime indicators).
    start_ts : float, optional
        Unix timestamp filter — skip bars before this.
    end_ts : float, optional
        Unix timestamp filter — skip bars after this.
    """

    def __init__(
        self,
        bars: Optional[List[BarData]] = None,
        path: Optional[Union[str, Path]] = None,
        symbol: str = "BTCUSDT",
        warmup_bars: int = 0,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
    ) -> None:
        if bars is not None:
            self._bars = bars
        elif path is not None:
            self._bars = self._load(Path(path), symbol)
        else:
            self._bars = []

        self._warmup_bars = warmup_bars
        self._start_ts = start_ts
        self._end_ts = end_ts
        self._total = len(self._bars)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Generator[tuple, None, None]:
        """Yields (bar, is_warmup) tuples."""
        count = 0
        for i, bar in enumerate(self._bars):
            if self._start_ts and bar.ts < self._start_ts:
                continue
            if self._end_ts and bar.ts > self._end_ts:
                break
            is_warmup = i < self._warmup_bars
            yield bar, is_warmup
            count += 1
        logger.debug("DataFeed: yielded %d bars", count)

    def __len__(self) -> int:
        return self._total

    # ------------------------------------------------------------------
    # File loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: Path, symbol: str) -> List[BarData]:
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return DataFeed._load_csv(path, symbol)
        elif suffix in {".parquet", ".pq"}:
            return DataFeed._load_parquet(path, symbol)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    @staticmethod
    def _load_csv(path: Path, symbol: str) -> List[BarData]:
        bars: List[BarData] = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts_raw = row.get("timestamp") or row.get("ts") or row.get("time")
                    ts = float(ts_raw) if ts_raw and ts_raw.replace(".", "").isdigit() \
                        else datetime.fromisoformat(str(ts_raw)).timestamp()
                    bars.append(BarData(
                        ts=ts,
                        symbol=symbol,
                        open=float(row.get("open", 0)),
                        high=float(row.get("high", 0)),
                        low=float(row.get("low", 0)),
                        close=float(row.get("close", 0)),
                        volume=float(row.get("volume", 0)),
                    ))
                except Exception as exc:
                    logger.warning("DataFeed: skipping malformed row: %s", exc)
        bars.sort(key=lambda b: b.ts)
        return bars

    @staticmethod
    def _load_parquet(path: Path, symbol: str) -> List[BarData]:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas required to load Parquet files") from exc
        df = pd.read_parquet(path)
        df = df.sort_values("timestamp") if "timestamp" in df.columns else df.sort_index()
        bars = []
        for _, row in df.iterrows():
            ts = float(row.get("timestamp", row.name))
            bars.append(BarData(
                ts=ts, symbol=symbol,
                open=float(row["open"]), high=float(row["high"]),
                low=float(row["low"]), close=float(row["close"]),
                volume=float(row.get("volume", 0)),
            ))
        return bars

    # ------------------------------------------------------------------
    # Synthetic generator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def synthetic(
        n: int = 500,
        start_price: float = 65_000.0,
        volatility: float = 0.005,
        symbol: str = "BTCUSDT",
        start_ts: float = 1_700_000_000.0,
        interval: float = 3600.0,
        seed: int = 42,
    ) -> "DataFeed":
        """Generate synthetic random-walk OHLCV bars."""
        import random
        rng = random.Random(seed)
        bars: List[BarData] = []
        price = start_price
        ts = start_ts
        for _ in range(n):
            ret = rng.gauss(0, volatility)
            close = price * (1 + ret)
            high = close * (1 + abs(rng.gauss(0, volatility / 2)))
            low = close * (1 - abs(rng.gauss(0, volatility / 2)))
            bars.append(BarData(
                ts=ts, symbol=symbol,
                open=price, high=high, low=low, close=close,
                volume=rng.uniform(10, 1000),
            ))
            price = close
            ts += interval
        return DataFeed(bars=bars)
