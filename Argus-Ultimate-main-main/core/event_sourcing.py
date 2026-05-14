#!/usr/bin/env python3
"""
Event Sourcing Store — append-only event log with snapshot support.

Provides an immutable event stream per aggregate/stream_id, with replay
capability and snapshot caching for fast state reconstruction.

Persistence: SQLite at ``data/event_store.db`` (configurable).
Thread-safe via SQLite WAL mode + Python threading locks.

Usage::

    store = EventStore()
    eid = store.append("account-1", "credited", {"amount": 100})
    events = store.get_events("account-1")

    def reducer(state, event):
        state.setdefault("balance", 0)
        if event.event_type == "credited":
            state["balance"] += event.data["amount"]
        return state

    state = store.rebuild_state("account-1", reducer)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """Immutable representation of a single domain event."""

    event_id: int
    stream_id: str
    event_type: str
    data: Dict[str, Any]
    version: int
    timestamp: float


# ---------------------------------------------------------------------------
# Event Store
# ---------------------------------------------------------------------------


class EventStore:
    """Append-only event sourcing store backed by SQLite.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Parent directories are created
        automatically.  Defaults to ``data/event_store.db`` relative to the
        repository root.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            repo_root = Path(__file__).resolve().parent.parent
            db_path = str(repo_root / "data" / "event_store.db")

        self._db_path = db_path
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._conn = self._connect()
        self._create_tables()
        logger.info("EventStore initialised — db=%s", self._db_path)

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with WAL mode for concurrency."""
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _create_tables(self) -> None:
        """Create event and snapshot tables if they don't exist."""
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    stream_id   TEXT    NOT NULL,
                    event_type  TEXT    NOT NULL,
                    data        TEXT    NOT NULL,
                    version     INTEGER NOT NULL,
                    timestamp   REAL    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_stream
                    ON events (stream_id, version);

                CREATE TABLE IF NOT EXISTS snapshots (
                    stream_id   TEXT    PRIMARY KEY,
                    state       TEXT    NOT NULL,
                    version     INTEGER NOT NULL,
                    timestamp   REAL    NOT NULL
                );
                """
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(
        self,
        stream_id: str,
        event_type: str,
        data: Dict[str, Any],
    ) -> int:
        """Append a new event to *stream_id* and return its ``event_id``.

        Parameters
        ----------
        stream_id:
            Logical aggregate identifier (e.g. ``"order-123"``).
        event_type:
            Domain event name (e.g. ``"order_placed"``).
        data:
            Arbitrary JSON-serialisable payload.

        Returns
        -------
        int
            The auto-incremented ``event_id`` of the newly stored event.
        """
        ts = time.time()

        with self._lock:
            # Determine next version for this stream
            cur = self._conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM events WHERE stream_id = ?",
                (stream_id,),
            )
            max_version: int = cur.fetchone()[0]
            next_version = max_version + 1

            cur = self._conn.execute(
                "INSERT INTO events (stream_id, event_type, data, version, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (stream_id, event_type, json.dumps(data), next_version, ts),
            )
            self._conn.commit()
            event_id: int = cur.lastrowid  # type: ignore[assignment]

        logger.debug(
            "EventStore.append stream=%s type=%s version=%d eid=%d",
            stream_id,
            event_type,
            next_version,
            event_id,
        )
        return event_id

    def get_events(self, stream_id: str, after_version: int = 0) -> List[Event]:
        """Return all events for *stream_id* with ``version > after_version``.

        Parameters
        ----------
        stream_id:
            The aggregate stream to query.
        after_version:
            Only return events whose version is strictly greater than this.
            Defaults to ``0`` (all events).

        Returns
        -------
        list[Event]
            Ordered by version ascending.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT event_id, stream_id, event_type, data, version, timestamp "
                "FROM events WHERE stream_id = ? AND version > ? ORDER BY version",
                (stream_id, after_version),
            ).fetchall()

        events = [
            Event(
                event_id=row[0],
                stream_id=row[1],
                event_type=row[2],
                data=json.loads(row[3]),
                version=row[4],
                timestamp=row[5],
            )
            for row in rows
        ]
        logger.debug(
            "EventStore.get_events stream=%s after_version=%d → %d events",
            stream_id,
            after_version,
            len(events),
        )
        return events

    def rebuild_state(
        self,
        stream_id: str,
        reducer_fn: Callable[[Any, Event], Any],
        *,
        initial_state: Any = None,
    ) -> Any:
        """Replay all events through *reducer_fn* to reconstruct aggregate state.

        If a snapshot exists, replay starts from the snapshot version to avoid
        replaying the full history.

        Parameters
        ----------
        stream_id:
            The aggregate stream to replay.
        reducer_fn:
            ``(state, event) -> state`` — a pure function that folds each
            event into the running state.
        initial_state:
            The seed state passed to the first reducer call.  When a snapshot
            exists, the snapshot state is used instead.

        Returns
        -------
        Any
            The final state after replaying all events.
        """
        # Try snapshot first for fast rebuild
        snapshot = self.get_snapshot(stream_id)
        if snapshot is not None:
            state = snapshot["state"]
            after_version = snapshot["version"]
            logger.debug(
                "EventStore.rebuild_state using snapshot v%d for stream=%s",
                after_version,
                stream_id,
            )
        else:
            state = initial_state if initial_state is not None else {}
            state = initial_state
            after_version = 0

        events = self.get_events(stream_id, after_version=after_version)
        for event in events:
            state = reducer_fn(state, event)

        logger.debug(
            "EventStore.rebuild_state stream=%s replayed %d events from v%d",
            stream_id,
            len(events),
            after_version,
        )
        return state

    def get_snapshot(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Return the latest cached snapshot for *stream_id*, or ``None``.

        Returns
        -------
        dict or None
            ``{"state": ..., "version": int, "timestamp": float}`` if a
            snapshot exists, else ``None``.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT state, version, timestamp FROM snapshots WHERE stream_id = ?",
                (stream_id,),
            ).fetchone()

        if row is None:
            return None

        return {
            "state": json.loads(row[0]),
            "version": row[1],
            "timestamp": row[2],
        }

    def save_snapshot(
        self, stream_id: str, state: Dict[str, Any], version: int
    ) -> None:
        """Cache a state snapshot so future rebuilds can skip earlier events.

        Parameters
        ----------
        stream_id:
            The aggregate whose state is being cached.
        state:
            JSON-serialisable state dictionary.
        version:
            The event version this snapshot is valid up to (inclusive).
        """
        ts = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO snapshots (stream_id, state, version, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (stream_id, json.dumps(state), version, ts),
            )
            self._conn.commit()

        logger.info(
            "EventStore.save_snapshot stream=%s version=%d", stream_id, version
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_stream_ids(self) -> List[str]:
        """Return all distinct stream IDs in the store."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT stream_id FROM events ORDER BY stream_id"
            ).fetchall()
        return [row[0] for row in rows]

    def get_stream_version(self, stream_id: str) -> int:
        """Return the current (latest) version for *stream_id*, or 0."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM events WHERE stream_id = ?",
                (stream_id,),
            ).fetchone()
        return row[0]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()
        logger.info("EventStore closed — db=%s", self._db_path)
