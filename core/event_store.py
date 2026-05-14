"""SQLite-backed event sourcing audit trail for trading aggregates."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

SUPPORTED_EVENT_TYPES = {
    "OrderSubmitted",
    "OrderFilled",
    "OrderCancelled",
    "PositionOpened",
    "PositionClosed",
    "PositionModified",
    "TradeExecuted",
    "TradeSettled",
    "RiskLimitBreached",
    "CircuitBreakerTriggered",
    "SignalGenerated",
    "SignalExpired",
}

_DEFAULT_SCHEMA_VERSION = 1


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class DomainEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    aggregate_id: str = ""
    event_type: str = ""
    event_version: int = 0
    timestamp: datetime = field(default_factory=_utc_now)
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _ensure_utc(self.timestamp))
        if not self.aggregate_id:
            raise ValueError("aggregate_id is required")
        if self.event_type not in SUPPORTED_EVENT_TYPES:
            raise ValueError(f"Unsupported event_type: {self.event_type}")
        if self.event_version < 1:
            raise ValueError("event_version must be >= 1")

    @property
    def schema_version(self) -> int:
        value = self.metadata.get("schema_version", _DEFAULT_SCHEMA_VERSION)
        try:
            return int(value)
        except (TypeError, ValueError):
            return _DEFAULT_SCHEMA_VERSION

    def to_record(self) -> tuple[str, str, str, int, str, str, str, int]:
        return (
            self.event_id,
            self.aggregate_id,
            self.event_type,
            self.event_version,
            self.timestamp.isoformat(),
            json.dumps(self.payload, sort_keys=True, default=str),
            json.dumps(self.metadata, sort_keys=True, default=str),
            self.schema_version,
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "DomainEvent":
        return cls(
            event_id=row["event_id"],
            aggregate_id=row["aggregate_id"],
            event_type=row["event_type"],
            event_version=row["event_version"],
            timestamp=_ensure_utc(datetime.fromisoformat(row["timestamp"])),
            payload=json.loads(row["payload_json"]),
            metadata=json.loads(row["metadata_json"]),
        )


class EventStore:
    """Persists domain events and snapshots with optimistic concurrency."""

    def __init__(self, db_path: str = "data/event_store.db") -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        self._subscribers: List[Callable[[DomainEvent], None]] = []
        self._upcasters: Dict[str, Dict[int, Callable[[DomainEvent], DomainEvent]]] = {}
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()
        logger.info("EventStore initialised — db=%s", db_path)

    def subscribe(self, subscriber: Callable[[DomainEvent], None]) -> None:
        if subscriber not in self._subscribers:
            self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber: Callable[[DomainEvent], None]) -> None:
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)

    def register_upcaster(
        self,
        event_type: str,
        from_schema_version: int,
        upcaster: Callable[[DomainEvent], DomainEvent],
    ) -> None:
        self._upcasters.setdefault(event_type, {})[from_schema_version] = upcaster

    def append_events(
        self,
        aggregate_id: str,
        events: List[DomainEvent],
        expected_version: int,
    ) -> bool:
        if not events:
            logger.debug("No events to append for aggregate=%s", aggregate_id)
            return True

        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                current_version_row = conn.execute(
                    "SELECT COALESCE(MAX(event_version), 0) AS version FROM events WHERE aggregate_id = ?",
                    (aggregate_id,),
                ).fetchone()
                current_version = int(current_version_row["version"])
                if current_version != expected_version:
                    conn.rollback()
                    logger.warning(
                        "Optimistic concurrency conflict for aggregate=%s expected=%s actual=%s",
                        aggregate_id,
                        expected_version,
                        current_version,
                    )
                    return False

                next_version = expected_version + 1
                for event in events:
                    self._validate_event_for_append(aggregate_id, event, next_version)
                    conn.execute(
                        """
                        INSERT INTO events (
                            event_id,
                            aggregate_id,
                            event_type,
                            event_version,
                            timestamp,
                            payload_json,
                            metadata_json,
                            schema_version
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        event.to_record(),
                    )
                    next_version += 1

                conn.commit()
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                logger.warning("Failed to append events for aggregate=%s: %s", aggregate_id, exc)
                return False
            finally:
                conn.close()

        for event in events:
            self._publish_event(event)

        logger.debug("Appended %d events for aggregate=%s", len(events), aggregate_id)
        return True

    def get_events(self, aggregate_id: str, from_version: int = 0) -> List[DomainEvent]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT event_id, aggregate_id, event_type, event_version, timestamp,
                       payload_json, metadata_json, schema_version
                FROM events
                WHERE aggregate_id = ? AND event_version > ?
                ORDER BY event_version ASC
                """,
                (aggregate_id, from_version),
            ).fetchall()
        finally:
            conn.close()
        return [self._apply_upcasters(DomainEvent.from_row(row)) for row in rows]

    def get_events_by_type(
        self,
        event_type: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[DomainEvent]:
        if event_type not in SUPPORTED_EVENT_TYPES:
            raise ValueError(f"Unsupported event_type: {event_type}")

        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT event_id, aggregate_id, event_type, event_version, timestamp,
                       payload_json, metadata_json, schema_version
                FROM events
                WHERE event_type = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC, position ASC
                """,
                (
                    event_type,
                    _ensure_utc(start_time).isoformat(),
                    _ensure_utc(end_time).isoformat(),
                ),
            ).fetchall()
        finally:
            conn.close()
        return [self._apply_upcasters(DomainEvent.from_row(row)) for row in rows]

    def get_all_events(self, from_position: int = 0, limit: int = 100) -> List[DomainEvent]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT position, event_id, aggregate_id, event_type, event_version, timestamp,
                       payload_json, metadata_json, schema_version
                FROM events
                WHERE position > ?
                ORDER BY position ASC
                LIMIT ?
                """,
                (from_position, limit),
            ).fetchall()
        finally:
            conn.close()
        return [self._apply_upcasters(DomainEvent.from_row(row)) for row in rows]

    def snapshot_aggregate(self, aggregate_id: str, version: int, state: dict) -> None:
        payload = json.dumps(state, sort_keys=True, default=str)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO snapshots (aggregate_id, version, state_json, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(aggregate_id) DO UPDATE SET
                        version = excluded.version,
                        state_json = excluded.state_json,
                        created_at = excluded.created_at
                    """,
                    (aggregate_id, version, payload, _utc_now().isoformat()),
                )
                conn.commit()
            finally:
                conn.close()

    def get_snapshot(self, aggregate_id: str) -> Optional[dict]:
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT aggregate_id, version, state_json, created_at
                FROM snapshots
                WHERE aggregate_id = ?
                """,
                (aggregate_id,),
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None

        return {
            "aggregate_id": row["aggregate_id"],
            "version": row["version"],
            "state": json.loads(row["state_json"]),
            "created_at": _ensure_utc(datetime.fromisoformat(row["created_at"])),
        }

    def replay_events(self, aggregate_id: str, handler: Callable) -> Any:
        snapshot = self.get_snapshot(aggregate_id)
        state: Any = snapshot["state"] if snapshot else None
        from_version = int(snapshot["version"]) if snapshot else 0

        for event in self.get_events(aggregate_id, from_version=from_version):
            state = handler(state, event)
        return state

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    position INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    aggregate_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_version INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    schema_version INTEGER NOT NULL DEFAULT 1,
                    UNIQUE(aggregate_id, event_version)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    aggregate_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_aggregate ON events(aggregate_id, event_version)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_type_time ON events(event_type, timestamp)"
            )
        finally:
            conn.close()

    def _validate_event_for_append(
        self,
        aggregate_id: str,
        event: DomainEvent,
        expected_event_version: int,
    ) -> None:
        if event.aggregate_id != aggregate_id:
            raise ValueError(
                f"Event aggregate_id mismatch: expected {aggregate_id}, got {event.aggregate_id}"
            )
        if event.event_version != expected_event_version:
            raise ValueError(
                f"Event version mismatch: expected {expected_event_version}, got {event.event_version}"
            )

    def _publish_event(self, event: DomainEvent) -> None:
        for subscriber in tuple(self._subscribers):
            try:
                subscriber(event)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Event subscriber failed for event_id=%s event_type=%s: %s",
                    event.event_id,
                    event.event_type,
                    exc,
                )

    def _apply_upcasters(self, event: DomainEvent) -> DomainEvent:
        upcasters = self._upcasters.get(event.event_type, {})
        current = event
        visited: set[int] = set()

        while current.schema_version in upcasters and current.schema_version not in visited:
            visited.add(current.schema_version)
            current = upcasters[current.schema_version](current)

        return current


__all__ = ["DomainEvent", "EventStore", "SUPPORTED_EVENT_TYPES"]
