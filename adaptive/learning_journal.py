#!/usr/bin/env python3
"""
Structured Learning Journal — Tier 3 Self-Improvement Module.

Records trading events with attached lessons, tracks which lessons have
been acted upon, and generates daily summaries.  Designed to build an
institutional memory of what works and what does not.

Usage (standalone)::

    journal = LearningJournal()
    journal.record_event(
        "trade_loss", "BTC/AUD short stopped out",
        {"loss_pct": -1.8, "regime": "bull"},
        "Avoid shorting in strong bull regimes without momentum confirmation"
    )
    summary = journal.generate_daily_summary()
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

# Valid event types
EVENT_TYPES = frozenset(
    {
        "trade_win",
        "trade_loss",
        "regime_change",
        "risk_breach",
        "model_retrain",
        "strategy_decay",
        "new_high",
        "drawdown",
    }
)


@dataclass
class JournalEntry:
    """A single learning journal entry."""

    id: int
    event_type: str
    description: str
    metrics: Dict[str, Any]
    lesson: str
    timestamp: str
    actionable: bool
    resolved: bool = False


class LearningJournal:
    """
    Persistent structured learning journal backed by SQLite.

    Each entry records a trading event alongside a human-readable lesson
    derived from that event.  Entries can be flagged as *actionable* (the
    default) and later marked *resolved* once the lesson has been applied.

    Event types
    -----------
    * ``trade_win``       — a profitable trade closed
    * ``trade_loss``      — a losing trade closed
    * ``regime_change``   — market regime transition detected
    * ``risk_breach``     — risk limit or circuit breaker triggered
    * ``model_retrain``   — ML model retrained (with quality metrics)
    * ``strategy_decay``  — strategy performance degradation observed
    * ``new_high``        — portfolio equity new high-water mark
    * ``drawdown``        — notable drawdown event
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            db_path = str(
                Path(__file__).resolve().parent.parent / "data" / "learning_journal.db"
            )
        self._db_path = db_path
        self._lock = threading.Lock()
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        log.info("LearningJournal initialised — db=%s", self._db_path)

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the journal table if it does not exist."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS journal (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        description TEXT NOT NULL,
                        metrics TEXT NOT NULL,
                        lesson TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        actionable INTEGER NOT NULL DEFAULT 1,
                        resolved INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_journal_type ON journal(event_type)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_journal_ts ON journal(timestamp)"
                )
                conn.commit()
            finally:
                conn.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_event(
        self,
        event_type: str,
        description: str,
        metrics: Dict[str, Any],
        lesson: str,
        *,
        actionable: bool = True,
    ) -> int:
        """
        Record a trading event and its associated lesson.

        Parameters
        ----------
        event_type:
            One of the valid ``EVENT_TYPES``.
        description:
            Free-text description of what happened.
        metrics:
            Quantitative data associated with the event (JSON-serialisable).
        lesson:
            The lesson learned from this event.
        actionable:
            Whether this lesson requires follow-up action.

        Returns
        -------
        int
            Row ID of the new journal entry.

        Raises
        ------
        ValueError
            If *event_type* is not recognised.
        """
        if event_type not in EVENT_TYPES:
            raise ValueError(
                f"Unknown event type '{event_type}'. Must be one of {sorted(EVENT_TYPES)}"
            )
        now = datetime.now(timezone.utc).isoformat()
        metrics_json = json.dumps(metrics, default=str)

        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO journal (event_type, description, metrics, lesson, timestamp, actionable, resolved)
                    VALUES (?, ?, ?, ?, ?, ?, 0)
                    """,
                    (event_type, description, metrics_json, lesson, now, int(actionable)),
                )
                conn.commit()
                row_id = cur.lastrowid
            finally:
                conn.close()

        log.info("Journal [%s]: %s — lesson: %s", event_type, description[:60], lesson[:60])
        return row_id  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_lessons(
        self,
        event_type: Optional[str] = None,
        lookback_days: int = 30,
    ) -> List[JournalEntry]:
        """
        Retrieve journal entries matching the given filters.

        Parameters
        ----------
        event_type:
            Filter by event type.  *None* returns all types.
        lookback_days:
            Only return entries from the last N days.

        Returns
        -------
        list[JournalEntry]
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
        conn = self._connect()
        try:
            if event_type:
                rows = conn.execute(
                    """
                    SELECT id, event_type, description, metrics, lesson, timestamp, actionable, resolved
                    FROM journal
                    WHERE event_type = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                    """,
                    (event_type, cutoff),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, event_type, description, metrics, lesson, timestamp, actionable, resolved
                    FROM journal
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                    """,
                    (cutoff,),
                ).fetchall()
        finally:
            conn.close()

        return [
            JournalEntry(
                id=r[0],
                event_type=r[1],
                description=r[2],
                metrics=json.loads(r[3]),
                lesson=r[4],
                timestamp=r[5],
                actionable=bool(r[6]),
                resolved=bool(r[7]),
            )
            for r in rows
        ]

    def get_actionable_items(self) -> List[JournalEntry]:
        """
        Return all unresolved actionable lessons.

        Returns
        -------
        list[JournalEntry]
            Entries where ``actionable=True`` and ``resolved=False``,
            ordered newest first.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT id, event_type, description, metrics, lesson, timestamp, actionable, resolved
                FROM journal
                WHERE actionable = 1 AND resolved = 0
                ORDER BY timestamp DESC
                """,
            ).fetchall()
        finally:
            conn.close()

        return [
            JournalEntry(
                id=r[0],
                event_type=r[1],
                description=r[2],
                metrics=json.loads(r[3]),
                lesson=r[4],
                timestamp=r[5],
                actionable=bool(r[6]),
                resolved=bool(r[7]),
            )
            for r in rows
        ]

    def mark_resolved(self, entry_id: int) -> bool:
        """
        Mark a journal entry's lesson as resolved.

        Parameters
        ----------
        entry_id:
            Row ID of the entry to resolve.

        Returns
        -------
        bool
            True if the entry was found and updated.
        """
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "UPDATE journal SET resolved = 1 WHERE id = ?",
                    (entry_id,),
                )
                conn.commit()
                updated = cur.rowcount > 0
            finally:
                conn.close()
        if updated:
            log.info("Resolved journal entry #%d", entry_id)
        return updated

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def generate_daily_summary(self) -> str:
        """
        Generate a text summary of today's journal entries.

        Returns
        -------
        str
            A human-readable summary grouped by event type with counts and
            key lessons.
        """
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat()

        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT event_type, description, lesson, metrics
                FROM journal
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
                """,
                (today_start,),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return "No journal entries recorded today."

        # Group by event type
        by_type: Dict[str, List[tuple]] = {}
        for event_type, desc, lesson, metrics_json in rows:
            by_type.setdefault(event_type, []).append((desc, lesson, metrics_json))

        lines: List[str] = [f"Daily Learning Journal — {len(rows)} events"]
        lines.append("=" * 50)

        for etype, entries in sorted(by_type.items()):
            lines.append(f"\n{etype.upper()} ({len(entries)} events):")
            for desc, lesson, _ in entries[:5]:  # cap at 5 per type
                lines.append(f"  - {desc}")
                lines.append(f"    Lesson: {lesson}")
            if len(entries) > 5:
                lines.append(f"  ... and {len(entries) - 5} more")

        # Actionable count
        conn = self._connect()
        try:
            actionable_count = conn.execute(
                "SELECT COUNT(*) FROM journal WHERE actionable = 1 AND resolved = 0"
            ).fetchone()[0]
        finally:
            conn.close()

        lines.append(f"\nUnresolved actionable items: {actionable_count}")
        summary = "\n".join(lines)
        log.info("Generated daily summary: %d events, %d actionable", len(rows), actionable_count)
        return summary
