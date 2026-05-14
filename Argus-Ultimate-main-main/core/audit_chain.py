#!/usr/bin/env python3
"""
Immutable audit chain for Argus trading events.
Provides cryptographic proof of event ordering and integrity.
"""

from dataclasses import dataclass
import hashlib
import sqlite3
import json
import time
from typing import Optional, List


def sha256_hex(data: bytes) -> str:
    """Compute SHA256 hash and return as hex string"""
    return hashlib.sha256(data).hexdigest()


def compute_event_hash(prev_hash: str, ts: float, kind: str, payload_json: str) -> str:
    """Compute hash for an audit event"""
    # Stable encoding: prev|ts|kind|payload
    s = f"{prev_hash}|{ts:.6f}|{kind}|{payload_json}".encode()
    return sha256_hex(s)


@dataclass(frozen=True)
class ChainHead:
    """Current head of the audit chain"""

    event_hash: str
    sequence: int
    timestamp: float


@dataclass(frozen=True)
class AuditEvent:
    """An auditable event in the chain"""

    sequence: int
    timestamp: float
    kind: str
    payload: dict
    event_hash: str
    prev_hash: str


class AuditChain:
    """
    Immutable audit chain stored in SQLite.
    Every event is cryptographically linked to prevent tampering.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the audit chain database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
    sequence INTEGER PRIMARY KEY,
    timestamp REAL NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    event_hash TEXT NOT NULL,
    prev_hash TEXT NOT NULL
                )
            """
            )

            # Create index for efficient queries
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON audit_events(timestamp)
            """
            )

            # Insert genesis block if not exists
            cursor = conn.execute("SELECT COUNT(*) FROM audit_events WHERE sequence = 0")
            if cursor.fetchone()[0] == 0:
                genesis_hash = sha256_hex(b"argus_audit_chain_genesis")
                conn.execute(
                    """
    INSERT INTO audit_events (sequence, timestamp, kind, payload, event_hash, prev_hash)
    VALUES (0, ?, 'genesis', '{}', ?, '0000000000000000000000000000000000000000000000000000000000000000')
                """,
                    (time.time(), genesis_hash),
                )

    def append_event(self, kind: str, payload: dict) -> AuditEvent:
        """
        Append a new event to the audit chain.

        Args:
            kind: Event type identifier
            payload: Event data (must be JSON serializable)

        Returns:
            The newly created audit event
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get current head
            cursor = conn.execute(
                """
                SELECT sequence, event_hash
                FROM audit_events
                ORDER BY sequence DESC
                LIMIT 1
            """
            )

            row = cursor.fetchone()
            if not row:
                raise RuntimeError("Audit chain corrupted - no genesis block")

            prev_sequence, prev_hash = row
            new_sequence = prev_sequence + 1

            # Create event
            timestamp = time.time()
            payload_json = json.dumps(payload, sort_keys=True)
            event_hash = compute_event_hash(prev_hash, timestamp, kind, payload_json)

            # Insert event
            conn.execute(
                """
                INSERT INTO audit_events (sequence, timestamp, kind, payload, event_hash, prev_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (new_sequence, timestamp, kind, payload_json, event_hash, prev_hash),
            )

            return AuditEvent(
                sequence=new_sequence,
                timestamp=timestamp,
                kind=kind,
                payload=payload,
                event_hash=event_hash,
                prev_hash=prev_hash,
            )

    def get_head(self) -> ChainHead:
        """Get the current chain head"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT sequence, event_hash, timestamp
                FROM audit_events
                ORDER BY sequence DESC
                LIMIT 1
            """
            )

            row = cursor.fetchone()
            if not row:
                raise RuntimeError("Audit chain corrupted - no events")

            sequence, event_hash, timestamp = row
            return ChainHead(event_hash=event_hash, sequence=sequence, timestamp=timestamp)

    def get_event(self, sequence: int) -> Optional[AuditEvent]:
        """Get a specific event by sequence number"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT sequence, timestamp, kind, payload, event_hash, prev_hash
                FROM audit_events
                WHERE sequence = ?
            """,
                (sequence,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            sequence, timestamp, kind, payload_json, event_hash, prev_hash = row
            payload = json.loads(payload_json)

            return AuditEvent(
                sequence=sequence,
                timestamp=timestamp,
                kind=kind,
                payload=payload,
                event_hash=event_hash,
                prev_hash=prev_hash,
            )

    def verify_chain_integrity(self) -> bool:
        """
        Verify the entire chain's cryptographic integrity.
        Returns True if chain is valid, False if tampered with.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT sequence, timestamp, kind, payload, event_hash, prev_hash
                FROM audit_events
                ORDER BY sequence
            """
            )

            prev_hash = "0000000000000000000000000000000000000000000000000000000000000000"

            for row in cursor:
                sequence, timestamp, kind, payload_json, event_hash, stored_prev_hash = row

                # Verify previous hash continuity
                if stored_prev_hash != prev_hash:
                    return False

                # Verify event hash
                expected_hash = compute_event_hash(prev_hash, timestamp, kind, payload_json)
                if expected_hash != event_hash:
                    return False

                prev_hash = event_hash

            return True

    def get_events_since(self, timestamp: float) -> List[AuditEvent]:
        """Get all events since a given timestamp"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT sequence, timestamp, kind, payload, event_hash, prev_hash
                FROM audit_events
                WHERE timestamp >= ?
                ORDER BY sequence
            """,
                (timestamp,),
            )

            events = []
            for row in cursor:
                sequence, timestamp, kind, payload_json, event_hash, prev_hash = row
                payload = json.loads(payload_json)

                events.append(
                    AuditEvent(
                        sequence=sequence,
                        timestamp=timestamp,
                        kind=kind,
                        payload=payload,
                        event_hash=event_hash,
                        prev_hash=prev_hash,
                    )
                )

            return events
