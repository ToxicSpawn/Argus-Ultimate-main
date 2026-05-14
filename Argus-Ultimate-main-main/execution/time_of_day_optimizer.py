#!/usr/bin/env python3
"""
Time-of-Day Execution Optimizer — learns when slippage/spread is cheapest.

Records execution metrics (slippage, spread, fill time) bucketed by UTC hour,
and after accumulating sufficient data per bucket (default 50 fills) provides
recommendations on optimal and worst trading hours.

Features:
- ``record_execution(symbol, hour_utc, slippage_bps, spread_bps, fill_time_ms)``
- ``get_optimal_hours(symbol, top_n=3)`` → cheapest hours
- ``get_worst_hours(symbol, top_n=3)`` → most expensive hours
- ``should_delay(symbol, current_hour)`` → whether to wait for a better window

Persistence: SQLite at ``data/tod_optimizer.db``.

Usage::

    opt = TimeOfDayOptimizer()
    opt.record_execution("BTC/USD", 14, 2.5, 8.0, 120)
    best = opt.get_optimal_hours("BTC/USD")
    delay, wait_min = opt.should_delay("BTC/USD", 3)
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_DB_DIR = Path("data")
_DEFAULT_DB_NAME = "tod_optimizer.db"

# Minimum observations per hour-bucket before using that bucket's data
_MIN_OBSERVATIONS = 50

# Hours considered "market quiet" — typically low liquidity for crypto
# (weekend/holiday logic not needed; crypto trades 24/7 but volume dips)
_LOW_VOLUME_HOURS_UTC = frozenset({2, 3, 4, 5})  # ~2–5 UTC is Asian early morning


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HourStats:
    """Aggregated execution statistics for a single UTC hour."""

    hour: int
    count: int
    avg_slippage_bps: float
    avg_spread_bps: float
    avg_fill_time_ms: float
    composite_score: float  # lower is better


@dataclass
class DelayRecommendation:
    """Whether to delay an execution and for how long."""

    should_delay: bool
    suggested_delay_minutes: int
    reason: str


# ---------------------------------------------------------------------------
# Time-of-Day Optimizer
# ---------------------------------------------------------------------------


class TimeOfDayOptimizer:
    """Learns per-hour execution quality and recommends timing.

    Parameters
    ----------
    db_path : str or Path, optional
        SQLite database path.  Defaults to ``data/tod_optimizer.db``.
    min_observations : int
        Minimum fills per hour-bucket before making recommendations.
    slippage_weight : float
        Weight of slippage in the composite score (default 0.6).
    spread_weight : float
        Weight of spread in the composite score (default 0.3).
    fill_time_weight : float
        Weight of fill time in the composite score (default 0.1).
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        min_observations: int = _MIN_OBSERVATIONS,
        slippage_weight: float = 0.6,
        spread_weight: float = 0.3,
        fill_time_weight: float = 0.1,
    ) -> None:
        if db_path is None:
            self._db_path = _DEFAULT_DB_DIR / _DEFAULT_DB_NAME
        else:
            self._db_path = Path(db_path)

        self._min_obs = max(1, min_observations)
        self._w_slip = slippage_weight
        self._w_spread = spread_weight
        self._w_fill = fill_time_weight
        self._lock = threading.Lock()

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(
            "TimeOfDayOptimizer initialised — db=%s, min_obs=%d",
            self._db_path, self._min_obs,
        )

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tod_executions (
                        id            INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts            REAL    NOT NULL,
                        symbol        TEXT    NOT NULL,
                        hour_utc      INTEGER NOT NULL,
                        slippage_bps  REAL    NOT NULL,
                        spread_bps    REAL    NOT NULL,
                        fill_time_ms  REAL    NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_tod_symbol_hour
                    ON tod_executions (symbol, hour_utc)
                """)
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_execution(
        self,
        symbol: str,
        hour_utc: int,
        slippage_bps: float,
        spread_bps: float,
        fill_time_ms: float,
    ) -> None:
        """Record a single execution's timing metrics.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. ``"BTC/USD"``.
        hour_utc : int
            UTC hour 0–23.
        slippage_bps : float
            Realised slippage in basis points.
        spread_bps : float
            Bid-ask spread at time of fill in basis points.
        fill_time_ms : float
            Time from order submission to fill in milliseconds.
        """
        hour_utc = int(hour_utc) % 24
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                conn.execute(
                    """
                    INSERT INTO tod_executions
                        (ts, symbol, hour_utc, slippage_bps, spread_bps, fill_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (time.time(), symbol, hour_utc, slippage_bps, spread_bps, fill_time_ms),
                )
                conn.commit()
            finally:
                conn.close()

        logger.debug(
            "TimeOfDayOptimizer: recorded %s hour=%d slip=%.1f spread=%.1f fill=%.0fms",
            symbol, hour_utc, slippage_bps, spread_bps, fill_time_ms,
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def _get_hour_stats(self, symbol: str) -> List[HourStats]:
        """Return per-hour aggregated stats for a symbol, filtered by min_obs."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            rows = conn.execute(
                """
                SELECT hour_utc,
                       COUNT(*),
                       AVG(slippage_bps),
                       AVG(spread_bps),
                       AVG(fill_time_ms)
                FROM tod_executions
                WHERE symbol = ?
                GROUP BY hour_utc
                HAVING COUNT(*) >= ?
                ORDER BY hour_utc
                """,
                (symbol, self._min_obs),
            ).fetchall()
        finally:
            conn.close()

        stats: List[HourStats] = []
        for hour, count, avg_slip, avg_spread, avg_fill in rows:
            # Normalise fill time to bps-equivalent scale (1 second = ~1 bps penalty)
            fill_norm = avg_fill / 1000.0
            composite = (
                self._w_slip * avg_slip
                + self._w_spread * avg_spread
                + self._w_fill * fill_norm
            )
            stats.append(
                HourStats(
                    hour=hour,
                    count=count,
                    avg_slippage_bps=round(avg_slip, 2),
                    avg_spread_bps=round(avg_spread, 2),
                    avg_fill_time_ms=round(avg_fill, 1),
                    composite_score=round(composite, 3),
                )
            )
        return stats

    def get_optimal_hours(
        self, symbol: str, top_n: int = 3
    ) -> List[Tuple[int, float]]:
        """Return the cheapest hours for executing a symbol.

        Parameters
        ----------
        symbol : str
            Trading pair.
        top_n : int
            Number of hours to return.

        Returns
        -------
        list of (hour_utc, avg_slippage_bps)
            Sorted best-first.  Empty if insufficient data.
        """
        stats = self._get_hour_stats(symbol)
        if not stats:
            logger.info(
                "TimeOfDayOptimizer: insufficient data for %s optimal hours", symbol,
            )
            return []

        stats.sort(key=lambda s: s.composite_score)
        result = [(s.hour, s.avg_slippage_bps) for s in stats[:top_n]]
        logger.info("TimeOfDayOptimizer: optimal hours for %s → %s", symbol, result)
        return result

    def get_worst_hours(
        self, symbol: str, top_n: int = 3
    ) -> List[Tuple[int, float]]:
        """Return the most expensive hours for executing a symbol.

        Parameters
        ----------
        symbol : str
            Trading pair.
        top_n : int
            Number of hours to return.

        Returns
        -------
        list of (hour_utc, avg_slippage_bps)
            Sorted worst-first.  Empty if insufficient data.
        """
        stats = self._get_hour_stats(symbol)
        if not stats:
            logger.info(
                "TimeOfDayOptimizer: insufficient data for %s worst hours", symbol,
            )
            return []

        stats.sort(key=lambda s: -s.composite_score)
        result = [(s.hour, s.avg_slippage_bps) for s in stats[:top_n]]
        logger.info("TimeOfDayOptimizer: worst hours for %s → %s", symbol, result)
        return result

    def should_delay(
        self, symbol: str, current_hour: int
    ) -> Tuple[bool, int]:
        """Determine whether to delay execution based on time-of-day data.

        Parameters
        ----------
        symbol : str
            Trading pair.
        current_hour : int
            Current UTC hour (0–23).

        Returns
        -------
        (should_delay, suggested_delay_minutes)
            ``should_delay`` is True if the current hour is significantly worse
            than the optimal hour.  ``suggested_delay_minutes`` is the wait time
            until the next good window (0 if no delay).
        """
        current_hour = int(current_hour) % 24
        stats = self._get_hour_stats(symbol)

        if not stats:
            # Not enough data — don't delay
            return (False, 0)

        stats.sort(key=lambda s: s.composite_score)
        best_score = stats[0].composite_score
        best_hour = stats[0].hour

        # Find current hour's score
        current_stat = next((s for s in stats if s.hour == current_hour), None)
        if current_stat is None:
            # No data for current hour — don't delay
            return (False, 0)

        # Delay if current hour is >30% worse than the best
        threshold_ratio = 1.3
        if current_stat.composite_score <= best_score * threshold_ratio:
            return (False, 0)

        # Calculate minutes until the best hour
        hours_until = (best_hour - current_hour) % 24
        if hours_until == 0:
            return (False, 0)

        delay_minutes = hours_until * 60
        logger.info(
            "TimeOfDayOptimizer: suggest delay for %s — current hour %d "
            "(score %.1f) vs best hour %d (score %.1f), delay %d min",
            symbol, current_hour, current_stat.composite_score,
            best_hour, best_score, delay_minutes,
        )
        return (True, delay_minutes)

    def get_all_hour_stats(self, symbol: str) -> List[HourStats]:
        """Return all hour stats (including below min_obs) for diagnostics.

        Returns
        -------
        list of HourStats
            All 24 hours with whatever data is available.
        """
        conn = sqlite3.connect(str(self._db_path))
        try:
            rows = conn.execute(
                """
                SELECT hour_utc,
                       COUNT(*),
                       AVG(slippage_bps),
                       AVG(spread_bps),
                       AVG(fill_time_ms)
                FROM tod_executions
                WHERE symbol = ?
                GROUP BY hour_utc
                ORDER BY hour_utc
                """,
                (symbol,),
            ).fetchall()
        finally:
            conn.close()

        result: List[HourStats] = []
        for hour, count, avg_slip, avg_spread, avg_fill in rows:
            fill_norm = avg_fill / 1000.0
            composite = (
                self._w_slip * avg_slip
                + self._w_spread * avg_spread
                + self._w_fill * fill_norm
            )
            result.append(
                HourStats(
                    hour=hour,
                    count=count,
                    avg_slippage_bps=round(avg_slip, 2),
                    avg_spread_bps=round(avg_spread, 2),
                    avg_fill_time_ms=round(avg_fill, 1),
                    composite_score=round(composite, 3),
                )
            )
        return result
