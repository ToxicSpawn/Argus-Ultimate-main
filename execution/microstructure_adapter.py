#!/usr/bin/env python3
"""
Market Microstructure Adaptation — Tier 3 Self-Improvement Module.

Records execution quality metrics per symbol/exchange/hour bucket and derives
optimal order type, timing, and venue preferences from historical performance.

Persists all data in a local SQLite database so recommendations improve over
time without requiring external infrastructure.

Usage (standalone)::

    adapter = MicrostructureAdapter()
    adapter.record_execution("BTC/AUD", "kraken", "limit", 14, 1.2, 85, 500.0)
    rec = adapter.get_execution_recommendation("BTC/AUD", "kraken", 500.0)
"""

from __future__ import annotations

import logging
import os
import sqlite3
import statistics
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Minimum number of executions in a bucket before making recommendations.
_MIN_EXECUTIONS = 30


@dataclass
class ExecutionRecommendation:
    """Recommendation for how to execute a given order."""

    order_type: str  # "limit" or "market"
    urgency: str  # "low", "medium", "high"
    split_count: int
    timing_advice: str
    expected_slippage_bps: float


class MicrostructureAdapter:
    """
    Learns optimal execution parameters from historical fill data.

    Records slippage, fill time, and size for each execution event, bucketed
    by symbol, exchange, order type, and hour-of-day.  Once enough data
    accumulates (>= 30 observations per bucket), the adapter can recommend:

    * Best order type (limit vs market) per symbol/exchange
    * Best / worst hours for execution
    * Venue ranking by execution quality
    * Composite execution recommendation with split and timing advice

    All state is stored in ``data/microstructure.db`` (SQLite).
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            db_path = str(Path(__file__).resolve().parent.parent / "data" / "microstructure.db")
        self._db_path = db_path
        self._lock = threading.Lock()
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        log.info("MicrostructureAdapter initialised — db=%s", self._db_path)

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the executions table if it does not already exist."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS executions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        exchange TEXT NOT NULL,
                        order_type TEXT NOT NULL,
                        hour INTEGER NOT NULL,
                        slippage_bps REAL NOT NULL,
                        fill_time_ms REAL NOT NULL,
                        size_usd REAL NOT NULL,
                        recorded_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_exec_sym_exch ON executions(symbol, exchange)"
                )
                conn.commit()
            finally:
                conn.close()

    def _connect(self) -> sqlite3.Connection:
        """Return a new SQLite connection."""
        return sqlite3.connect(self._db_path)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_execution(
        self,
        symbol: str,
        exchange: str,
        order_type: str,
        hour: int,
        slippage_bps: float,
        fill_time_ms: float,
        size_usd: float,
    ) -> None:
        """
        Record an execution event for learning.

        Parameters
        ----------
        symbol:
            Trading pair, e.g. ``"BTC/AUD"``.
        exchange:
            Exchange name, e.g. ``"kraken"``.
        order_type:
            ``"limit"`` or ``"market"``.
        hour:
            Hour of day (0-23 UTC) when the execution occurred.
        slippage_bps:
            Realised slippage in basis points.
        fill_time_ms:
            Time to fill in milliseconds.
        size_usd:
            Notional size of the order in USD.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO executions (symbol, exchange, order_type, hour,
                                            slippage_bps, fill_time_ms, size_usd, recorded_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (symbol, exchange, order_type, hour, slippage_bps, fill_time_ms, size_usd, now),
                )
                conn.commit()
            finally:
                conn.close()
        log.debug(
            "Recorded execution: %s/%s type=%s hour=%d slip=%.1fbps fill=%dms size=$%.0f",
            symbol,
            exchange,
            order_type,
            hour,
            slippage_bps,
            fill_time_ms,
            size_usd,
        )

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def get_optimal_order_type(
        self,
        symbol: str,
        exchange: str,
        hour: Optional[int] = None,
    ) -> str:
        """
        Determine the best order type for a symbol/exchange pair.

        Compares average slippage for limit vs market orders.  If fewer than
        ``_MIN_EXECUTIONS`` observations exist for a type, defaults to
        ``"limit"`` (conservative).

        Parameters
        ----------
        symbol:
            Trading pair.
        exchange:
            Exchange name.
        hour:
            Optional hour filter (0-23).  If *None*, all hours are included.

        Returns
        -------
        str
            ``"limit"`` or ``"market"``.
        """
        conn = self._connect()
        try:
            if hour is not None:
                rows = conn.execute(
                    """
                    SELECT order_type, AVG(slippage_bps), COUNT(*)
                    FROM executions
                    WHERE symbol = ? AND exchange = ? AND hour = ?
                    GROUP BY order_type
                    """,
                    (symbol, exchange, hour),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT order_type, AVG(slippage_bps), COUNT(*)
                    FROM executions
                    WHERE symbol = ? AND exchange = ?
                    GROUP BY order_type
                    """,
                    (symbol, exchange),
                ).fetchall()
        finally:
            conn.close()

        best_type = "limit"
        best_slip = float("inf")
        for otype, avg_slip, count in rows:
            if count >= _MIN_EXECUTIONS and avg_slip < best_slip:
                best_slip = avg_slip
                best_type = otype

        return best_type

    def get_optimal_timing(self, symbol: str) -> Dict[str, List[int]]:
        """
        Identify the best and worst hours for executing a given symbol.

        Ranks hours by average slippage (lower is better).  Only hours with
        >= ``_MIN_EXECUTIONS`` observations are included.

        Parameters
        ----------
        symbol:
            Trading pair.

        Returns
        -------
        dict
            ``{"best_hours": [...], "worst_hours": [...]}`` where each list
            contains up to 4 hours sorted by preference.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT hour, AVG(slippage_bps) as avg_slip, COUNT(*) as cnt
                FROM executions
                WHERE symbol = ?
                GROUP BY hour
                HAVING cnt >= ?
                ORDER BY avg_slip ASC
                """,
                (symbol, _MIN_EXECUTIONS),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return {"best_hours": [], "worst_hours": []}

        hours_sorted = [r[0] for r in rows]
        best = hours_sorted[:4]
        worst = list(reversed(hours_sorted[-4:])) if len(hours_sorted) >= 2 else []
        return {"best_hours": best, "worst_hours": worst}

    def get_venue_preference(self, symbol: str) -> List[str]:
        """
        Rank exchanges by execution quality for a given symbol.

        Quality is measured as a composite score of average slippage (70%
        weight) and average fill time (30% weight), both normalised.

        Parameters
        ----------
        symbol:
            Trading pair.

        Returns
        -------
        list[str]
            Exchange names ordered from best to worst execution quality.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT exchange, AVG(slippage_bps) as avg_slip,
                       AVG(fill_time_ms) as avg_fill, COUNT(*) as cnt
                FROM executions
                WHERE symbol = ?
                GROUP BY exchange
                HAVING cnt >= ?
                """,
                (symbol, _MIN_EXECUTIONS),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return []

        # Normalise and score
        max_slip = max(r[1] for r in rows) or 1.0
        max_fill = max(r[2] for r in rows) or 1.0

        scored = []
        for exchange, avg_slip, avg_fill, _ in rows:
            # Lower is better — invert
            score = 0.7 * (1 - avg_slip / max_slip) + 0.3 * (1 - avg_fill / max_fill)
            scored.append((exchange, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored]

    def get_execution_recommendation(
        self,
        symbol: str,
        exchange: str,
        size_usd: float,
    ) -> ExecutionRecommendation:
        """
        Produce a composite execution recommendation.

        Combines order type preference, timing data, and historical slippage
        to advise on how to execute a trade of *size_usd* for *symbol* on
        *exchange*.

        Parameters
        ----------
        symbol:
            Trading pair.
        exchange:
            Target exchange.
        size_usd:
            Notional order size in USD.

        Returns
        -------
        ExecutionRecommendation
            Composite recommendation with order type, urgency, split count,
            timing advice, and expected slippage.
        """
        order_type = self.get_optimal_order_type(symbol, exchange)
        timing = self.get_optimal_timing(symbol)

        # Expected slippage from historical data
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT AVG(slippage_bps), COUNT(*)
                FROM executions
                WHERE symbol = ? AND exchange = ? AND order_type = ?
                """,
                (symbol, exchange, order_type),
            ).fetchone()
        finally:
            conn.close()

        avg_slippage = row[0] if row and row[0] is not None else 5.0  # default 5 bps
        count = row[1] if row else 0

        # Urgency: based on slippage — high slippage implies use limit and be patient
        if avg_slippage > 10:
            urgency = "low"
        elif avg_slippage > 5:
            urgency = "medium"
        else:
            urgency = "high"

        # Split count: split large orders to reduce market impact
        if size_usd > 5000:
            split_count = 5
        elif size_usd > 2000:
            split_count = 3
        elif size_usd > 500:
            split_count = 2
        else:
            split_count = 1

        # Timing advice
        if timing["best_hours"]:
            best_str = ", ".join(str(h) for h in timing["best_hours"])
            timing_advice = f"Prefer hours (UTC): {best_str}"
        else:
            timing_advice = "Insufficient data for timing recommendation"

        # Adjust expected slippage if we don't have enough data
        if count < _MIN_EXECUTIONS:
            expected_slippage = 5.0  # conservative default
            timing_advice = "Insufficient execution history — using defaults"
        else:
            expected_slippage = round(avg_slippage, 2)

        return ExecutionRecommendation(
            order_type=order_type,
            urgency=urgency,
            split_count=split_count,
            timing_advice=timing_advice,
            expected_slippage_bps=expected_slippage,
        )
