#!/usr/bin/env python3
"""
Signal Subscription Service — fan-out trading signals to registered webhooks.

Subscribers register a callback URL; on each signal emission the service
delivers a JSON payload to all active subscribers. A SQLite store keeps
delivery receipts and subscription records.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import threading
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS subscriptions (
    sub_id      TEXT PRIMARY KEY,
    callback_url TEXT NOT NULL,
    name        TEXT,
    active      INTEGER NOT NULL DEFAULT 1,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS signal_deliveries (
    delivery_id TEXT PRIMARY KEY,
    sub_id      TEXT NOT NULL,
    signal_json TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    attempts    INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    delivered_at REAL
);
"""


@dataclass
class Subscription:
    sub_id: str
    callback_url: str
    name: str = ""
    active: bool = True
    created_at: float = field(default_factory=time.time)


class SignalService:
    """Thread-safe signal subscription and fan-out service."""

    def __init__(self, db_path: str = "data/signal_service.db", max_retries: int = 3, timeout: float = 5.0):
        self.db_path = str(db_path)
        self.max_retries = max(1, int(max_retries))
        self.timeout = max(1.0, float(timeout))
        Path(Path(self.db_path).parent).mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    # ---------------------------------------------------------------- schema

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.execute("PRAGMA journal_mode=WAL")
        con.row_factory = sqlite3.Row
        return con

    def _init_schema(self) -> None:
        try:
            con = self._connect()
            con.executescript(_SCHEMA)
            con.commit()
            con.close()
        except Exception as e:
            logger.warning("SignalService schema: %s", e)

    # ---------------------------------------------------------------- subscriptions

    def add_subscription(self, callback_url: str, *, name: str = "") -> Subscription:
        sub_id = uuid.uuid4().hex[:12]
        sub = Subscription(sub_id=sub_id, callback_url=callback_url, name=name)
        try:
            con = self._connect()
            con.execute(
                "INSERT OR IGNORE INTO subscriptions (sub_id,callback_url,name,active,created_at) VALUES (?,?,?,1,?)",
                (sub.sub_id, sub.callback_url, sub.name, sub.created_at),
            )
            con.commit()
            con.close()
        except Exception as e:
            logger.warning("SignalService add_subscription: %s", e)
        logger.info("SignalService: subscription added sub_id=%s url=%s", sub_id, callback_url)
        return sub

    def remove_subscription(self, sub_id: str) -> None:
        try:
            con = self._connect()
            con.execute("UPDATE subscriptions SET active=0 WHERE sub_id=?", (sub_id,))
            con.commit()
            con.close()
        except Exception as e:
            logger.warning("SignalService remove_subscription: %s", e)

    def list_subscriptions(self) -> List[Subscription]:
        try:
            con = self._connect()
            rows = con.execute("SELECT * FROM subscriptions WHERE active=1").fetchall()
            con.close()
            return [Subscription(sub_id=r["sub_id"], callback_url=r["callback_url"], name=r["name"] or "", created_at=r["created_at"]) for r in rows]
        except Exception as e:
            logger.warning("SignalService list: %s", e)
            return []

    # ---------------------------------------------------------------- emit

    def emit(self, signal: Any) -> int:
        """Fan-out a signal dict/object to all active subscribers. Returns delivery count."""
        subs = self.list_subscriptions()
        if not subs:
            return 0
        payload = self._to_payload(signal)
        delivered = 0
        for sub in subs:
            if self._deliver(sub, payload):
                delivered += 1
        return delivered

    def emit_async(self, signal: Any) -> None:
        """Fire-and-forget async emit in a daemon thread."""
        t = threading.Thread(target=self.emit, args=(signal,), daemon=True)
        t.start()

    def _deliver(self, sub: Subscription, payload: bytes) -> bool:
        delivery_id = uuid.uuid4().hex[:12]
        now = time.time()
        try:
            con = self._connect()
            con.execute(
                "INSERT INTO signal_deliveries (delivery_id,sub_id,signal_json,status,attempts,created_at) VALUES (?,?,?,?,0,?)",
                (delivery_id, sub.sub_id, payload.decode("utf-8"), "pending", now),
            )
            con.commit()
            con.close()
        except Exception:
            pass

        ok = False
        for attempt in range(self.max_retries):
            try:
                import urllib.request
                req = urllib.request.Request(
                    sub.callback_url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    ok = 200 <= resp.status < 300
                if ok:
                    break
            except Exception as e:
                logger.debug("SignalService delivery attempt %d to %s: %s", attempt + 1, sub.callback_url, e)
        status = "delivered" if ok else "failed"
        try:
            con = self._connect()
            con.execute(
                "UPDATE signal_deliveries SET status=?,attempts=?,delivered_at=? WHERE delivery_id=?",
                (status, self.max_retries if not ok else 1, time.time(), delivery_id),
            )
            con.commit()
            con.close()
        except Exception:
            pass
        return ok

    @staticmethod
    def _to_payload(signal: Any) -> bytes:
        if isinstance(signal, (bytes, bytearray)):
            return bytes(signal)
        if isinstance(signal, str):
            return signal.encode("utf-8")
        if isinstance(signal, dict):
            return json.dumps(signal, default=str).encode("utf-8")
        try:
            return json.dumps(vars(signal), default=str).encode("utf-8")
        except Exception:
            return json.dumps({"signal": str(signal)}).encode("utf-8")
