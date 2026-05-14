"""
risk/correlation_regime_learner.py — Correlation Regime Learning

Tracks per-symbol returns in memory, computes rolling correlation
matrices, detects decorrelation events, and scores portfolio
diversification.  Periodically flushes to SQLite so state survives
restarts.

Classes
-------
CorrelationRegimeLearner
    Update returns, compute correlations, detect regime shifts, identify
    stress hedges.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS return_observations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol     TEXT    NOT NULL,
    return_pct REAL    NOT NULL,
    timestamp  TEXT    NOT NULL
);
"""

_CORR_SQL = """
CREATE TABLE IF NOT EXISTS correlation_snapshots (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    matrix     TEXT    NOT NULL,
    symbols    TEXT    NOT NULL,
    timestamp  TEXT    NOT NULL
);
"""

_IDX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_ro_symbol ON return_observations(symbol);",
    "CREATE INDEX IF NOT EXISTS idx_ro_ts ON return_observations(timestamp);",
]


class CorrelationRegimeLearner:
    """Learn correlation regimes from streaming return data.

    Parameters
    ----------
    db_path : str, optional
        Override the default ``data/correlation_regimes.db`` path.
    flush_interval : int
        Seconds between automatic flushes of in-memory data to SQLite.
    max_memory_points : int
        Maximum in-memory data points per symbol before oldest are trimmed.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        flush_interval: int = 300,
        max_memory_points: int = 10_000,
    ) -> None:
        self._db_path = db_path or os.path.join(_DB_DIR, "correlation_regimes.db")
        self._flush_interval = flush_interval
        self._max_mem = max_memory_points
        self._lock = threading.Lock()

        # In-memory buffers: symbol → list of (timestamp_iso, return_pct)
        self._returns: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self._last_flush = time.monotonic()

        # Cache the most recent full-window correlation matrix
        self._cached_corr: Optional[Dict[str, Dict[str, float]]] = None
        self._cached_symbols: List[str] = []

        # Previous correlation for decorrelation detection
        self._prev_corr: Optional[Dict[str, Dict[str, float]]] = None

        self._ensure_schema()
        log.info("CorrelationRegimeLearner initialised  db=%s", self._db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        with self._lock:
            con = sqlite3.connect(self._db_path)
            try:
                con.execute(_CREATE_SQL)
                con.execute(_CORR_SQL)
                for idx in _IDX_SQL:
                    con.execute(idx)
                con.commit()
            finally:
                con.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=10)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_returns(
        self,
        symbol: str,
        return_pct: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a return observation for *symbol*.

        Parameters
        ----------
        symbol : str
            Asset identifier (e.g. ``"BTC/USD"``).
        return_pct : float
            Percentage return for this period.
        timestamp : datetime, optional
            Observation time; defaults to now UTC.
        """
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        with self._lock:
            buf = self._returns[symbol]
            buf.append((ts, return_pct))
            # Trim oldest if over limit
            if len(buf) > self._max_mem:
                self._returns[symbol] = buf[-self._max_mem:]

        # Auto-flush if interval elapsed
        if time.monotonic() - self._last_flush > self._flush_interval:
            self._flush_to_db()

    def compute_correlation_matrix(
        self,
        lookback_hours: int = 168,
    ) -> Dict[str, Dict[str, float]]:
        """Compute pairwise correlations from recent return data.

        Parameters
        ----------
        lookback_hours : int
            Only use observations from the last N hours.

        Returns
        -------
        dict of dict
            ``{symbol_a: {symbol_b: correlation, ...}, ...}``
        """
        cutoff = datetime.now(timezone.utc).timestamp() - lookback_hours * 3600
        cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()

        # Collect aligned return vectors
        with self._lock:
            symbols = sorted(self._returns.keys())
            filtered: Dict[str, List[float]] = {}
            for sym in symbols:
                vals = [r for ts, r in self._returns[sym] if ts >= cutoff_iso]
                if vals:
                    filtered[sym] = vals

        symbols = sorted(filtered.keys())
        if len(symbols) < 2:
            self._cached_corr = {s: {s: 1.0} for s in symbols}
            self._cached_symbols = symbols
            return self._cached_corr

        # Align lengths to minimum
        min_len = min(len(filtered[s]) for s in symbols)
        aligned = {s: filtered[s][-min_len:] for s in symbols}

        # Compute correlation matrix
        if _HAS_NUMPY:
            matrix = self._compute_corr_numpy(symbols, aligned)
        else:
            matrix = self._compute_corr_pure(symbols, aligned)

        # Store previous for decorrelation detection
        self._prev_corr = self._cached_corr
        self._cached_corr = matrix
        self._cached_symbols = symbols

        return matrix

    def detect_decorrelation_events(
        self,
        threshold: float = 0.3,
    ) -> List[Tuple[str, str, float, float]]:
        """Detect symbol pairs whose correlation changed by more than *threshold*.

        You must call ``compute_correlation_matrix()`` at least twice before
        this returns meaningful results.

        Returns
        -------
        list of (symbol_a, symbol_b, old_corr, new_corr)
        """
        if self._prev_corr is None or self._cached_corr is None:
            return []

        events: List[Tuple[str, str, float, float]] = []
        symbols = sorted(set(self._prev_corr.keys()) & set(self._cached_corr.keys()))

        seen = set()
        for i, a in enumerate(symbols):
            for b in symbols[i + 1:]:
                pair = (a, b)
                if pair in seen:
                    continue
                seen.add(pair)

                old = self._prev_corr.get(a, {}).get(b, 0.0)
                new = self._cached_corr.get(a, {}).get(b, 0.0)

                if abs(new - old) >= threshold:
                    events.append((a, b, round(old, 4), round(new, 4)))
                    log.info(
                        "Decorrelation event: %s / %s  corr %.3f → %.3f",
                        a, b, old, new,
                    )

        return events

    def get_diversification_score(
        self,
        portfolio_symbols: List[str],
    ) -> float:
        """Score portfolio diversification from 0 (fully correlated) to 1 (diversified).

        Uses the average absolute pairwise correlation: score = 1 - avg(|corr|).
        """
        if len(portfolio_symbols) < 2:
            return 1.0

        if self._cached_corr is None:
            self.compute_correlation_matrix()

        if self._cached_corr is None:
            return 1.0

        total_abs_corr = 0.0
        count = 0
        for i, a in enumerate(portfolio_symbols):
            for b in portfolio_symbols[i + 1:]:
                corr_val = self._cached_corr.get(a, {}).get(b, 0.0)
                total_abs_corr += abs(corr_val)
                count += 1

        if count == 0:
            return 1.0

        avg_abs_corr = total_abs_corr / count
        return round(max(0.0, min(1.0, 1.0 - avg_abs_corr)), 4)

    def get_stress_hedges(
        self,
        symbol: str,
        n: int = 5,
    ) -> List[str]:
        """Return symbols that are most negatively correlated with *symbol*.

        These are the best candidates for hedging during stress events.

        Parameters
        ----------
        symbol : str
        n : int
            Maximum number of hedge candidates to return.
        """
        if self._cached_corr is None:
            self.compute_correlation_matrix()

        if self._cached_corr is None or symbol not in self._cached_corr:
            return []

        correlations = self._cached_corr[symbol]
        # Sort by correlation ascending (most negative first)
        candidates = [
            (s, c) for s, c in correlations.items()
            if s != symbol
        ]
        candidates.sort(key=lambda x: x[1])
        return [s for s, _ in candidates[:n]]

    def flush(self) -> None:
        """Force flush in-memory data to SQLite."""
        self._flush_to_db()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _flush_to_db(self) -> None:
        """Persist in-memory return observations to SQLite."""
        with self._lock:
            if not any(self._returns.values()):
                return

            con = self._connect()
            try:
                for symbol, entries in self._returns.items():
                    for ts, ret in entries:
                        con.execute(
                            "INSERT INTO return_observations (symbol, return_pct, timestamp) "
                            "VALUES (?, ?, ?)",
                            (symbol, ret, ts),
                        )

                # Also persist correlation snapshot if available
                if self._cached_corr and self._cached_symbols:
                    con.execute(
                        "INSERT INTO correlation_snapshots (matrix, symbols, timestamp) "
                        "VALUES (?, ?, ?)",
                        (
                            json.dumps(self._cached_corr),
                            json.dumps(self._cached_symbols),
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                con.commit()
            finally:
                con.close()

            self._last_flush = time.monotonic()
            log.debug("Flushed return observations to SQLite")

    @staticmethod
    def _compute_corr_numpy(
        symbols: List[str],
        aligned: Dict[str, List[float]],
    ) -> Dict[str, Dict[str, float]]:
        """Compute correlation using numpy."""
        data = np.array([aligned[s] for s in symbols], dtype=np.float64)
        # Handle constant columns
        stds = np.std(data, axis=1)
        # Replace zero-std rows with tiny noise to avoid nan
        for i in range(len(stds)):
            if stds[i] == 0:
                data[i] += np.random.normal(0, 1e-12, data.shape[1])

        corr_matrix = np.corrcoef(data)
        # Replace nan with 0
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        result: Dict[str, Dict[str, float]] = {}
        for i, a in enumerate(symbols):
            result[a] = {}
            for j, b in enumerate(symbols):
                result[a][b] = round(float(corr_matrix[i, j]), 6)
        return result

    @staticmethod
    def _compute_corr_pure(
        symbols: List[str],
        aligned: Dict[str, List[float]],
    ) -> Dict[str, Dict[str, float]]:
        """Pure-Python Pearson correlation (no numpy dependency)."""
        def _pearson(xs: List[float], ys: List[float]) -> float:
            n = len(xs)
            if n == 0:
                return 0.0
            mx = sum(xs) / n
            my = sum(ys) / n
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
            dy = math.sqrt(sum((y - my) ** 2 for y in ys))
            if dx == 0 or dy == 0:
                return 0.0
            return num / (dx * dy)

        result: Dict[str, Dict[str, float]] = {}
        for a in symbols:
            result[a] = {}
            for b in symbols:
                if a == b:
                    result[a][b] = 1.0
                elif b in result and a in result[b]:
                    result[a][b] = result[b][a]
                else:
                    result[a][b] = round(_pearson(aligned[a], aligned[b]), 6)
        return result
