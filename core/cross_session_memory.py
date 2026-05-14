#!/usr/bin/env python3
"""
Cross-Session Knowledge Persistence — Tier 3 Self-Improvement Module.

Maintains a persistent store of system learnings (insights) across trading
sessions.  Each insight belongs to a category, carries a confidence score,
and tracks how many times it has been independently confirmed.

On startup, ``get_startup_briefing()`` summarises key learnings so the
system can immediately benefit from past experience.

Usage (standalone)::

    mem = CrossSessionMemory()
    mem.record_insight("strategy_performance", "rsi_bounce_sharpe", 1.42, confidence=0.9)
    briefing = mem.get_startup_briefing()
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Valid insight categories
CATEGORIES = frozenset(
    {
        "strategy_performance",
        "market_pattern",
        "execution_quality",
        "risk_event",
        "regime_transition",
        "model_drift",
    }
)


@dataclass
class Insight:
    """A single piece of persisted knowledge."""

    id: int
    category: str
    key: str
    value: Any
    confidence: float
    source: str
    timestamp: str
    times_confirmed: int
    invalidated: bool = False


class CrossSessionMemory:
    """
    Persistent cross-session knowledge base backed by SQLite.

    Records categorised insights with confidence scores and confirmation
    counts.  Insights that are repeatedly confirmed have their confidence
    boosted automatically.  Stale or disproved insights can be pruned or
    invalidated.

    Categories
    ----------
    * ``strategy_performance`` — per-strategy Sharpe, win-rate, etc.
    * ``market_pattern``       — recurring price/volume patterns
    * ``execution_quality``    — slippage/fill quality observations
    * ``risk_event``           — circuit breaker trips, drawdown events
    * ``regime_transition``    — detected regime changes and durations
    * ``model_drift``          — ML model accuracy degradation signals
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            db_path = str(
                Path(__file__).resolve().parent.parent / "data" / "cross_session_memory.db"
            )
        self._db_path = db_path
        self._lock = threading.Lock()
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        log.info("CrossSessionMemory initialised — db=%s", self._db_path)

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the insights table if it does not exist."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS insights (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        category TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        confidence REAL NOT NULL DEFAULT 1.0,
                        source TEXT NOT NULL DEFAULT 'system',
                        timestamp TEXT NOT NULL,
                        times_confirmed INTEGER NOT NULL DEFAULT 0,
                        invalidated INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_insight_cat_key ON insights(category, key)"
                )
                conn.commit()
            finally:
                conn.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def record_insight(
        self,
        category: str,
        key: str,
        value: Any,
        confidence: float = 1.0,
        source: str = "system",
    ) -> int:
        """
        Store a new learning.

        Parameters
        ----------
        category:
            One of the valid ``CATEGORIES``.
        key:
            A short identifier for this insight within the category.
        value:
            The insight payload — any JSON-serialisable value.
        confidence:
            Confidence score between 0.0 and 1.0.
        source:
            Origin of the insight (e.g. ``"backtest"``, ``"live"``).

        Returns
        -------
        int
            Row ID of the newly created insight.

        Raises
        ------
        ValueError
            If *category* is not a recognised category.
        """
        if category not in CATEGORIES:
            raise ValueError(f"Unknown insight category '{category}'. Must be one of {sorted(CATEGORIES)}")
        confidence = max(0.0, min(1.0, confidence))
        now = datetime.now(timezone.utc).isoformat()
        value_json = json.dumps(value, default=str)

        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO insights (category, key, value, confidence, source, timestamp, times_confirmed, invalidated)
                    VALUES (?, ?, ?, ?, ?, ?, 0, 0)
                    """,
                    (category, key, value_json, confidence, source, now),
                )
                conn.commit()
                row_id = cur.lastrowid
            finally:
                conn.close()

        log.info("Recorded insight [%s] %s = %s (confidence=%.2f)", category, key, value_json[:80], confidence)
        return row_id  # type: ignore[return-value]

    def get_insights(
        self,
        category: Optional[str] = None,
        min_confidence: float = 0.5,
        lookback_days: int = 30,
    ) -> List[Insight]:
        """
        Retrieve insights matching the given filters.

        Parameters
        ----------
        category:
            Filter by category.  *None* returns all categories.
        min_confidence:
            Minimum confidence threshold.
        lookback_days:
            Only return insights recorded within this many days.

        Returns
        -------
        list[Insight]
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
        conn = self._connect()
        try:
            if category:
                rows = conn.execute(
                    """
                    SELECT id, category, key, value, confidence, source, timestamp, times_confirmed, invalidated
                    FROM insights
                    WHERE category = ? AND confidence >= ? AND timestamp >= ? AND invalidated = 0
                    ORDER BY timestamp DESC
                    """,
                    (category, min_confidence, cutoff),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, category, key, value, confidence, source, timestamp, times_confirmed, invalidated
                    FROM insights
                    WHERE confidence >= ? AND timestamp >= ? AND invalidated = 0
                    ORDER BY timestamp DESC
                    """,
                    (min_confidence, cutoff),
                ).fetchall()
        finally:
            conn.close()

        return [
            Insight(
                id=r[0],
                category=r[1],
                key=r[2],
                value=json.loads(r[3]),
                confidence=r[4],
                source=r[5],
                timestamp=r[6],
                times_confirmed=r[7],
                invalidated=bool(r[8]),
            )
            for r in rows
        ]

    def confirm_insight(self, category: str, key: str) -> bool:
        """
        Increment the confirmation count and boost confidence for an insight.

        Confidence is boosted by ``0.05`` per confirmation, capped at ``1.0``.

        Parameters
        ----------
        category:
            Insight category.
        key:
            Insight key.

        Returns
        -------
        bool
            True if an insight was found and updated, False otherwise.
        """
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    UPDATE insights
                    SET times_confirmed = times_confirmed + 1,
                        confidence = MIN(1.0, confidence + 0.05)
                    WHERE category = ? AND key = ? AND invalidated = 0
                    """,
                    (category, key),
                )
                conn.commit()
                updated = cur.rowcount > 0
            finally:
                conn.close()
        if updated:
            log.info("Confirmed insight [%s] %s", category, key)
        return updated

    def invalidate_insight(self, category: str, key: str) -> bool:
        """
        Mark an insight as invalidated so it is excluded from queries.

        Parameters
        ----------
        category:
            Insight category.
        key:
            Insight key.

        Returns
        -------
        bool
            True if an insight was found and invalidated.
        """
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    UPDATE insights SET invalidated = 1
                    WHERE category = ? AND key = ? AND invalidated = 0
                    """,
                    (category, key),
                )
                conn.commit()
                updated = cur.rowcount > 0
            finally:
                conn.close()
        if updated:
            log.info("Invalidated insight [%s] %s", category, key)
        return updated

    # ------------------------------------------------------------------
    # Startup briefing
    # ------------------------------------------------------------------

    def get_startup_briefing(self) -> Dict[str, Any]:
        """
        Produce a summary of key learnings for system startup.

        Returns
        -------
        dict
            Keys: ``best_strategies``, ``worst_execution_hours``,
            ``regime_patterns``, ``risk_events``, ``model_drift_warnings``,
            ``total_insights``.
        """
        conn = self._connect()
        try:
            # Best strategies (top 5 by confidence, confirmed)
            strats = conn.execute(
                """
                SELECT key, value, confidence, times_confirmed
                FROM insights
                WHERE category = 'strategy_performance' AND invalidated = 0
                ORDER BY confidence DESC, times_confirmed DESC
                LIMIT 5
                """,
            ).fetchall()

            # Execution quality insights (worst hours)
            exec_q = conn.execute(
                """
                SELECT key, value, confidence
                FROM insights
                WHERE category = 'execution_quality' AND invalidated = 0
                ORDER BY timestamp DESC
                LIMIT 5
                """,
            ).fetchall()

            # Regime patterns
            regimes = conn.execute(
                """
                SELECT key, value, confidence
                FROM insights
                WHERE category = 'regime_transition' AND invalidated = 0
                ORDER BY timestamp DESC
                LIMIT 5
                """,
            ).fetchall()

            # Risk events (recent)
            risks = conn.execute(
                """
                SELECT key, value, timestamp
                FROM insights
                WHERE category = 'risk_event' AND invalidated = 0
                ORDER BY timestamp DESC
                LIMIT 5
                """,
            ).fetchall()

            # Model drift warnings
            drifts = conn.execute(
                """
                SELECT key, value, confidence
                FROM insights
                WHERE category = 'model_drift' AND invalidated = 0
                ORDER BY timestamp DESC
                LIMIT 3
                """,
            ).fetchall()

            # Total count
            total = conn.execute(
                "SELECT COUNT(*) FROM insights WHERE invalidated = 0"
            ).fetchone()[0]
        finally:
            conn.close()

        briefing: Dict[str, Any] = {
            "best_strategies": [
                {"key": r[0], "value": json.loads(r[1]), "confidence": r[2], "confirmations": r[3]}
                for r in strats
            ],
            "worst_execution_hours": [
                {"key": r[0], "value": json.loads(r[1]), "confidence": r[2]}
                for r in exec_q
            ],
            "regime_patterns": [
                {"key": r[0], "value": json.loads(r[1]), "confidence": r[2]}
                for r in regimes
            ],
            "risk_events": [
                {"key": r[0], "value": json.loads(r[1]), "timestamp": r[2]}
                for r in risks
            ],
            "model_drift_warnings": [
                {"key": r[0], "value": json.loads(r[1]), "confidence": r[2]}
                for r in drifts
            ],
            "total_insights": total,
        }
        log.info(
            "Startup briefing: %d total insights, %d strategies, %d risk events",
            total,
            len(strats),
            len(risks),
        )
        return briefing

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def prune_stale(self, max_age_days: int = 90) -> int:
        """
        Remove old, unconfirmed insights.

        Deletes insights older than *max_age_days* that have never been
        confirmed (``times_confirmed == 0``).

        Parameters
        ----------
        max_age_days:
            Age threshold in days.

        Returns
        -------
        int
            Number of pruned rows.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    DELETE FROM insights
                    WHERE timestamp < ? AND times_confirmed = 0
                    """,
                    (cutoff,),
                )
                conn.commit()
                count = cur.rowcount
            finally:
                conn.close()
        log.info("Pruned %d stale insights older than %d days", count, max_age_days)
        return count
