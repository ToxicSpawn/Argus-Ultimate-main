"""
Immutable audit trail: every order, fill, cancel, risk event; tamper-evident and queryable.

Wraps or extends core/firm/audit_chain for order/fill/risk events used by execution and risk.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _hash_event(prev_hash: str, ts: float, kind: str, payload_json: str) -> str:
    s = f"{prev_hash}|{ts:.6f}|{kind}|{payload_json}".encode()
    return hashlib.sha256(s).hexdigest()


class AuditTrail:
    """Append-only audit log for orders, fills, cancels, risk events. Queryable by time/kind."""

    def __init__(self, db_path: str = "data/audit_trail.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    seq INTEGER NOT NULL,
                    ts REAL NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    event_hash TEXT NOT NULL,
                    prev_hash TEXT NOT NULL
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_events(ts)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_kind ON audit_events(kind)")
            conn.commit()

    def append(self, kind: str, payload: Dict[str, Any]) -> str:
        """Append event; returns event_hash. Thread-safe via BEGIN IMMEDIATE."""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.cursor()
            cur.execute("SELECT seq, event_hash FROM audit_events ORDER BY seq DESC LIMIT 1")
            row = cur.fetchone()
            prev_hash = row["event_hash"] if row else "0"
            seq = (row["seq"] + 1) if row else 1
            ts = time.time()
            payload_json = json.dumps(payload, sort_keys=True)
            event_hash = _hash_event(prev_hash, ts, kind, payload_json)
            cur.execute(
                "INSERT INTO audit_events (seq, ts, kind, payload_json, event_hash, prev_hash) VALUES (?,?,?,?,?,?)",
                (seq, ts, kind, payload_json, event_hash, prev_hash),
            )
            conn.commit()
            return event_hash
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def query(self, start_ts: Optional[float] = None, end_ts: Optional[float] = None, kind: Optional[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """Query events by time and optional kind."""
        with self._connect() as conn:
            cur = conn.cursor()
            sql = "SELECT seq, ts, kind, payload_json, event_hash FROM audit_events WHERE 1=1"
            params: List[Any] = []
            if start_ts is not None:
                sql += " AND ts >= ?"
                params.append(start_ts)
            if end_ts is not None:
                sql += " AND ts <= ?"
                params.append(end_ts)
            if kind is not None:
                sql += " AND kind = ?"
                params.append(kind)
            sql += " ORDER BY seq LIMIT ?"
            params.append(limit)
            cur.execute(sql, params)
            rows = cur.fetchall()
        results = []
        for r in rows:
            try:
                payload = json.loads(r["payload_json"])
            except Exception as _e:
                logger.warning("audit_trail.query: corrupted payload at seq=%s: %s", r["seq"], _e)
                payload = {}
            results.append({
                "seq": r["seq"],
                "ts": r["ts"],
                "kind": r["kind"],
                "payload": payload,
                "event_hash": r["event_hash"],
            })
        return results

    def verify_chain(self) -> Dict[str, Any]:
        """Walk the entire audit chain and verify each event's hash linkage.

        Returns {"ok": bool, "total": int, "first_bad_seq": Optional[int], "error": Optional[str]}.
        """
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT seq, ts, kind, payload_json, event_hash, prev_hash FROM audit_events ORDER BY seq"
            )
            rows = cur.fetchall()

        prev_hash = "0"
        expected_seq = None
        for row in rows:
            # FIX #15: Check sequence contiguity — detect deleted events
            cur_seq = row["seq"]
            if expected_seq is not None and cur_seq != expected_seq:
                return {"ok": False, "total": len(rows), "first_bad_seq": cur_seq, "error": f"gap_detected: expected seq {expected_seq}, got {cur_seq}"}
            expected_seq = cur_seq + 1

            expected = _hash_event(row["prev_hash"], row["ts"], row["kind"], row["payload_json"])
            if expected != row["event_hash"]:
                return {"ok": False, "total": len(rows), "first_bad_seq": cur_seq, "error": "hash_mismatch"}
            if row["prev_hash"] != prev_hash:
                return {"ok": False, "total": len(rows), "first_bad_seq": cur_seq, "error": "chain_break"}
            prev_hash = row["event_hash"]
        return {"ok": True, "total": len(rows), "first_bad_seq": None, "error": None}


def get_audit_trail(db_path: str = "data/audit_trail.db") -> AuditTrail:
    return AuditTrail(db_path=db_path)
