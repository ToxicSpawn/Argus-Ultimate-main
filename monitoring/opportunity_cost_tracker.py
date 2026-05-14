"""
Opportunity Cost Tracker — records signals the system chose to skip,
then tracks what would have happened so the operator can quantify
missed P&L and identify over-filtering.

Persists to SQLite at ``data/opportunity_cost.db`` with WAL journaling
for concurrent read safety.

Usage
-----
>>> tracker = OpportunityCostTracker()
>>> tracker.record_skipped_signal("BTC/AUD", "long", 0.72,
...     reason="confidence_too_low", price_at_skip=98500.0)
>>> tracker.update_prices("BTC/AUD", 101000.0)
>>> missed = tracker.get_missed_pnl(lookback_days=7)
>>> print(missed.total_missed_usd)
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MissedPnL:
    """Summary of missed P&L over a lookback window."""
    total_missed_usd: float
    best_missed: float       # single best missed trade P&L (highest positive)
    worst_missed: float      # worst missed trade P&L (most negative / smallest)
    avg_missed: float
    count: int


@dataclass
class SkippedSignal:
    """Record of a single skipped signal."""
    signal_id: int
    symbol: str
    direction: str
    confidence: float
    reason: str
    price_at_skip: float
    current_price: Optional[float]
    hypothetical_pnl_pct: Optional[float]
    timestamp: float


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS skipped_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence REAL NOT NULL,
    reason TEXT NOT NULL,
    price_at_skip REAL NOT NULL,
    current_price REAL,
    hypothetical_pnl_pct REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_skipped_symbol ON skipped_signals(symbol);
CREATE INDEX IF NOT EXISTS idx_skipped_created ON skipped_signals(created_at);
CREATE INDEX IF NOT EXISTS idx_skipped_reason ON skipped_signals(reason);
"""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OpportunityCostTracker:
    """
    Records skipped trading signals and tracks their hypothetical outcomes.

    Parameters
    ----------
    db_path : str
        Path to SQLite database file.
    position_size_usd : float
        Notional position size used to convert percentage moves to USD P&L.
    """

    def __init__(self, db_path: str = "data/opportunity_cost.db",
                 position_size_usd: float = 100.0) -> None:
        self._db_path = db_path
        self._position_size_usd = position_size_usd
        self._lock = threading.Lock()

        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

        logger.info("OpportunityCostTracker initialised (db=%s, position_size=$%.0f)",
                     db_path, position_size_usd)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_skipped_signal(self, symbol: str, direction: str,
                              confidence: float, reason: str,
                              price_at_skip: float) -> int:
        """
        Record a signal that was generated but not executed.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. "BTC/AUD".
        direction : str
            "long" or "short".
        confidence : float
            Signal confidence at time of skip.
        reason : str
            Why the signal was skipped (e.g. "confidence_too_low",
            "risk_limit_breach", "position_limit_reached").
        price_at_skip : float
            Market price at the time the signal was skipped.

        Returns
        -------
        int
            Row ID of the inserted record.
        """
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO skipped_signals "
                "(symbol, direction, confidence, reason, price_at_skip, "
                " current_price, hypothetical_pnl_pct, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, ?)",
                (symbol, direction.lower(), confidence, reason, price_at_skip, now, now),
            )
            self._conn.commit()
            row_id = cur.lastrowid

        logger.info("record_skipped_signal: %s %s conf=%.2f reason='%s' price=%.2f [id=%d]",
                     symbol, direction, confidence, reason, price_at_skip, row_id)
        return row_id  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Price updates
    # ------------------------------------------------------------------

    def update_prices(self, symbol: str, current_price: float) -> int:
        """
        Update hypothetical P&L for all open skipped signals of *symbol*.

        Parameters
        ----------
        symbol : str
        current_price : float

        Returns
        -------
        int
            Number of records updated.
        """
        now = time.time()
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, direction, price_at_skip FROM skipped_signals "
                "WHERE symbol = ?",
                (symbol,),
            ).fetchall()

            count = 0
            for row_id, direction, price_at_skip in rows:
                if price_at_skip <= 0:
                    continue
                if direction == "long":
                    pnl_pct = (current_price - price_at_skip) / price_at_skip * 100.0
                else:
                    pnl_pct = (price_at_skip - current_price) / price_at_skip * 100.0

                self._conn.execute(
                    "UPDATE skipped_signals SET current_price = ?, "
                    "hypothetical_pnl_pct = ?, updated_at = ? WHERE id = ?",
                    (current_price, pnl_pct, now, row_id),
                )
                count += 1

            self._conn.commit()

        logger.debug("update_prices(%s, %.2f): %d records updated", symbol, current_price, count)
        return count

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def get_missed_pnl(self, lookback_days: int = 7) -> MissedPnL:
        """
        Summarise missed P&L over the last *lookback_days*.

        Returns
        -------
        MissedPnL
        """
        cutoff = time.time() - lookback_days * 86400

        with self._lock:
            rows = self._conn.execute(
                "SELECT hypothetical_pnl_pct FROM skipped_signals "
                "WHERE created_at >= ? AND hypothetical_pnl_pct IS NOT NULL",
                (cutoff,),
            ).fetchall()

        if not rows:
            return MissedPnL(total_missed_usd=0.0, best_missed=0.0,
                             worst_missed=0.0, avg_missed=0.0, count=0)

        pnls_pct = [r[0] for r in rows]
        # Convert pct to USD using position size
        pnls_usd = [p / 100.0 * self._position_size_usd for p in pnls_pct]

        result = MissedPnL(
            total_missed_usd=round(sum(pnls_usd), 2),
            best_missed=round(max(pnls_usd), 2),
            worst_missed=round(min(pnls_usd), 2),
            avg_missed=round(sum(pnls_usd) / len(pnls_usd), 2),
            count=len(pnls_usd),
        )

        logger.info("get_missed_pnl(days=%d): total=$%.2f best=$%.2f worst=$%.2f count=%d",
                    lookback_days, result.total_missed_usd, result.best_missed,
                    result.worst_missed, result.count)
        return result

    def get_skip_reason_analysis(self) -> Dict[str, float]:
        """
        Average hypothetical P&L (USD) per skip reason.

        Returns
        -------
        dict
            reason → avg_missed_pnl_usd.  Positive means the system missed
            profitable trades; negative means skipping was correct.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT reason, AVG(hypothetical_pnl_pct) "
                "FROM skipped_signals "
                "WHERE hypothetical_pnl_pct IS NOT NULL "
                "GROUP BY reason",
            ).fetchall()

        analysis: Dict[str, float] = {}
        for reason, avg_pct in rows:
            avg_usd = (avg_pct / 100.0) * self._position_size_usd
            analysis[reason] = round(avg_usd, 4)

        logger.info("get_skip_reason_analysis: %d reasons", len(analysis))
        return analysis

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_recent_skips(self, limit: int = 20) -> List[SkippedSignal]:
        """Return the most recent skipped signals."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, symbol, direction, confidence, reason, price_at_skip, "
                "current_price, hypothetical_pnl_pct, created_at "
                "FROM skipped_signals ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [
            SkippedSignal(
                signal_id=r[0], symbol=r[1], direction=r[2], confidence=r[3],
                reason=r[4], price_at_skip=r[5], current_price=r[6],
                hypothetical_pnl_pct=r[7], timestamp=r[8],
            )
            for r in rows
        ]

    @property
    def total_skipped(self) -> int:
        """Total number of skipped signals in the database."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM skipped_signals").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
