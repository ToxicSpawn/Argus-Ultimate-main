"""
execution/execution_quality_tracker.py — Execution Quality Feedback Loop

Records every fill and computes slippage statistics by symbol, strategy, and
hour-of-day.  Provides a size-adjustment multiplier that reduces position size
when execution quality is poor (high slippage), creating a negative-feedback
loop that automatically throttles toxic flow.

Persistence: SQLite at ``data/execution_quality.db``.

Thread-safe: all public methods are guarded by ``threading.Lock``.

Usage::

    tracker = ExecutionQualityTracker()
    tracker.record_fill("BTC/USD", "momentum", 14, 65000.0, 65012.0, 500.0)
    adj = tracker.get_size_adjustment("BTC/USD", "momentum", 14)
    # adj in [0.0, 1.0] — multiply your order size by this
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_DEFAULT_DB_PATH = os.path.join("data", "execution_quality.db")

# Slippage thresholds for size adjustment (basis points)
_SLIPPAGE_GOOD_BPS = 5.0        # <= 5 bps → no reduction
_SLIPPAGE_BAD_BPS = 30.0        # >= 30 bps → maximum reduction
_MIN_SIZE_MULT = 0.25           # floor multiplier at worst slippage


class ExecutionQualityTracker:
    """Tracks fill quality and provides execution-aware size adjustments.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    lookback_days : int
        Default lookback window for statistics.
    """

    def __init__(
        self,
        db_path: str = _DEFAULT_DB_PATH,
        lookback_days: int = 7,
    ) -> None:
        self._db_path = db_path
        self._lookback_days = lookback_days
        self._lock = threading.Lock()

        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        log.info("ExecutionQualityTracker initialised — db=%s", db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fills (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          REAL    NOT NULL,
                    symbol      TEXT    NOT NULL,
                    strategy    TEXT    NOT NULL,
                    hour        INTEGER NOT NULL,
                    expected_px REAL    NOT NULL,
                    fill_px     REAL    NOT NULL,
                    size_usd    REAL    NOT NULL,
                    slippage_bps REAL   NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fills_sym_ts ON fills(symbol, ts)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fills_strat ON fills(strategy, ts)"
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_fill(
        self,
        symbol: str,
        strategy: str,
        hour: int,
        expected_price: float,
        fill_price: float,
        size_usd: float,
    ) -> float:
        """Record a single fill and return the slippage in basis points.

        Slippage is calculated as the absolute price difference relative
        to the expected price, expressed in basis points (1 bp = 0.01%).

        Parameters
        ----------
        symbol : str
            Trading pair (e.g. ``"BTC/USD"``).
        strategy : str
            Strategy that generated the order.
        hour : int
            Hour of day (0-23) when the fill occurred.
        expected_price : float
            The price the strategy expected to fill at.
        fill_price : float
            The actual fill price.
        size_usd : float
            Notional fill size in USD.

        Returns
        -------
        float
            Slippage in basis points (always >= 0).
        """
        if expected_price <= 0:
            log.warning("ExecutionQualityTracker: expected_price <= 0 — skipping")
            return 0.0

        slippage_bps = abs(fill_price - expected_price) / expected_price * 10_000
        ts = time.time()

        with self._lock:
            self._conn.execute(
                "INSERT INTO fills (ts, symbol, strategy, hour, expected_px, fill_px, size_usd, slippage_bps) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, symbol, strategy, hour, expected_price, fill_price, size_usd, slippage_bps),
            )
            self._conn.commit()

        log.debug(
            "Fill recorded: %s/%s hour=%d slippage=%.2f bps size=$%.0f",
            symbol,
            strategy,
            hour,
            slippage_bps,
            size_usd,
        )
        return slippage_bps

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_avg_slippage_bps(
        self,
        symbol: str,
        strategy: Optional[str] = None,
        hour: Optional[int] = None,
        lookback_days: Optional[int] = None,
    ) -> float:
        """Return average slippage in bps for the given filters.

        Parameters
        ----------
        symbol : str
            Trading pair.
        strategy : str, optional
            Filter to a specific strategy.
        hour : int, optional
            Filter to a specific hour-of-day.
        lookback_days : int, optional
            Override default lookback window.

        Returns
        -------
        float
            Average slippage in basis points, or ``0.0`` if no data.
        """
        days = lookback_days or self._lookback_days
        cutoff = time.time() - days * 86400

        query = "SELECT AVG(slippage_bps) FROM fills WHERE symbol = ? AND ts >= ?"
        params: list = [symbol, cutoff]

        if strategy is not None:
            query += " AND strategy = ?"
            params.append(strategy)
        if hour is not None:
            query += " AND hour = ?"
            params.append(hour)

        with self._lock:
            row = self._conn.execute(query, params).fetchone()
        return row[0] if row and row[0] is not None else 0.0

    def get_size_adjustment(
        self,
        symbol: str,
        strategy: str,
        hour: int,
        lookback_days: Optional[int] = None,
    ) -> float:
        """Return a size multiplier in ``[0.25, 1.0]`` based on recent slippage.

        A value of ``1.0`` means execution quality is good (low slippage);
        ``0.25`` means slippage is severe and size should be heavily reduced.

        The multiplier is linearly interpolated between the "good" and "bad"
        slippage thresholds.

        Parameters
        ----------
        symbol, strategy, hour : filter keys
        lookback_days : int, optional

        Returns
        -------
        float
            Multiplier in ``[0.25, 1.0]``.
        """
        avg = self.get_avg_slippage_bps(symbol, strategy, hour, lookback_days)
        if avg <= _SLIPPAGE_GOOD_BPS:
            return 1.0
        if avg >= _SLIPPAGE_BAD_BPS:
            return _MIN_SIZE_MULT
        # Linear interpolation
        frac = (avg - _SLIPPAGE_GOOD_BPS) / (_SLIPPAGE_BAD_BPS - _SLIPPAGE_GOOD_BPS)
        return max(_MIN_SIZE_MULT, 1.0 - frac * (1.0 - _MIN_SIZE_MULT))

    def get_worst_hours(
        self,
        symbol: str,
        top_n: int = 3,
        lookback_days: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Return the hours with the worst average slippage.

        Parameters
        ----------
        symbol : str
            Trading pair.
        top_n : int
            How many hours to return.
        lookback_days : int, optional

        Returns
        -------
        list of (hour, avg_slippage_bps)
            Sorted worst-first.
        """
        days = lookback_days or self._lookback_days
        cutoff = time.time() - days * 86400

        query = (
            "SELECT hour, AVG(slippage_bps) as avg_slip "
            "FROM fills WHERE symbol = ? AND ts >= ? "
            "GROUP BY hour ORDER BY avg_slip DESC LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(query, (symbol, cutoff, top_n)).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()
        log.info("ExecutionQualityTracker: database closed")
