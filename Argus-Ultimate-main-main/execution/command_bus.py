from __future__ import annotations

import hashlib
import hmac
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


def _stable_payload(payload: Dict[str, Any]) -> str:
    """Canonical JSON for deterministic signing."""
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sign_instruction_payload(payload: Dict[str, Any], secret: str) -> str:
    msg = _stable_payload(dict(payload or {})).encode("utf-8")
    key = str(secret or "").encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def verify_instruction_payload(payload: Dict[str, Any], signature: str, secret: str) -> bool:
    if not signature:
        return False
    expected = sign_instruction_payload(payload, secret)
    return hmac.compare_digest(str(signature).lower(), str(expected).lower())


def deterministic_instruction_id(payload: Dict[str, Any]) -> str:
    symbol = str(payload.get("symbol", "") or "")
    action = str(payload.get("action", "") or "")
    quantity = float(payload.get("quantity", 0.0) or 0.0)
    entry_price = float(payload.get("entry_price", 0.0) or 0.0)
    cycle_id = int(payload.get("cycle_id", 0) or 0)
    trace_id = str(payload.get("trace_id", "") or "")
    run_id = str(payload.get("run_id", "") or "")
    basis = f"{run_id}|{trace_id}|{cycle_id}|{symbol}|{action}|{quantity:.10f}|{entry_price:.10f}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class BusValidationResult:
    ok: bool
    reason: str


def validate_instruction_payload(
    payload: Dict[str, Any],
    *,
    now_ts: Optional[float] = None,
    max_notional_aud: float = 0.0,
    aud_to_usd: float = 0.65,
) -> BusValidationResult:
    now = float(now_ts if now_ts is not None else time.time())
    symbol = str(payload.get("symbol", "") or "").strip()
    action = str(payload.get("action", "") or "").upper().strip()
    if not symbol:
        return BusValidationResult(False, "missing_symbol")
    if action not in {"BUY", "SELL"}:
        return BusValidationResult(False, "invalid_action")
    try:
        quantity = float(payload.get("quantity", 0.0) or 0.0)
        entry_price = float(payload.get("entry_price", 0.0) or 0.0)
    except Exception:
        return BusValidationResult(False, "invalid_numeric")
    if quantity <= 0.0:
        return BusValidationResult(False, "invalid_quantity")
    if entry_price <= 0.0:
        return BusValidationResult(False, "invalid_entry_price")
    try:
        expires_ts = float(payload.get("expires_ts", 0.0) or 0.0)
    except Exception:
        return BusValidationResult(False, "invalid_expires_ts")
    if expires_ts > 0.0 and now > expires_ts:
        return BusValidationResult(False, "stale_instruction")

    if max_notional_aud > 0.0:
        quote = symbol.split("/")[-1].upper() if "/" in symbol else "USD"
        notional_quote = quantity * entry_price
        notional_aud = notional_quote if quote == "AUD" else (notional_quote / max(float(aud_to_usd), 1e-9))
        if notional_aud > float(max_notional_aud):
            return BusValidationResult(False, "max_notional_exceeded")
    return BusValidationResult(True, "ok")


class LocalInstructionBus:
    """
    SQLite-backed local command bus for deterministic strategy->execution handoff.

    States:
      - PENDING: published and awaiting consumer claim
      - CLAIMED: atomically claimed by consumer; must transition to CONSUMED/REJECTED
      - CONSUMED: processed by execution node
      - REJECTED: consumed but rejected (invalid signature/stale/etc)
    """

    def __init__(self, db_path: str = "data/command_bus.db", queue: str = "default") -> None:
        self.db_path = str(db_path)
        self.queue = str(queue or "default")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=15000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS command_instructions (
                    instruction_id TEXT PRIMARY KEY,
                    queue_name TEXT NOT NULL,
                    created_ts REAL NOT NULL,
                    expires_ts REAL NOT NULL,
                    status TEXT NOT NULL,
                    producer_role TEXT NOT NULL,
                    consumer_role TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    signature TEXT,
                    claimed_ts REAL,
                    consumed_ts REAL,
                    rejected_reason TEXT
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_command_instructions_queue_status "
                "ON command_instructions(queue_name, status, created_ts)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_command_instructions_expires "
                "ON command_instructions(expires_ts)"
            )
            conn.commit()

    def publish(
        self,
        *,
        payload: Dict[str, Any],
        signature: str = "",
        instruction_id: str = "",
        producer_role: str = "strategy-node",
        consumer_role: str = "execution-node",
    ) -> str:
        now = float(time.time())
        pid = str(instruction_id or deterministic_instruction_id(payload))
        expires_ts = float(payload.get("expires_ts", 0.0) or 0.0)
        if expires_ts <= 0:
            expires_ts = now + 5.0
        row_payload = dict(payload or {})
        row_payload["instruction_id"] = pid
        row_payload.setdefault("created_ts", now)
        row_payload["expires_ts"] = expires_ts
        payload_json = _stable_payload(row_payload)
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT OR IGNORE INTO command_instructions (
                    instruction_id, queue_name, created_ts, expires_ts, status,
                    producer_role, consumer_role, payload_json, signature
                ) VALUES (?, ?, ?, ?, 'PENDING', ?, ?, ?, ?)
                """,
                (
                    pid,
                    self.queue,
                    now,
                    expires_ts,
                    str(producer_role or "strategy-node"),
                    str(consumer_role or "execution-node"),
                    payload_json,
                    str(signature or ""),
                ),
            )
            conn.commit()
        return pid

    def claim_pending(self, *, limit: int = 50, now_ts: Optional[float] = None) -> List[Dict[str, Any]]:
        now = float(now_ts if now_ts is not None else time.time())
        rows: List[Dict[str, Any]] = []
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("BEGIN IMMEDIATE TRANSACTION")
            cur.execute(
                """
                SELECT instruction_id, payload_json, signature, producer_role, consumer_role, expires_ts
                FROM command_instructions
                WHERE queue_name = ?
                  AND status = 'PENDING'
                  AND expires_ts >= ?
                ORDER BY created_ts ASC
                LIMIT ?
                """,
                (self.queue, now, int(max(1, limit))),
            )
            candidates = cur.fetchall()
            for c in candidates:
                iid = str(c["instruction_id"])
                cur.execute(
                    """
                    UPDATE command_instructions
                    SET status = 'CLAIMED',
                        claimed_ts = ?
                    WHERE instruction_id = ?
                      AND status = 'PENDING'
                    """,
                    (now, iid),
                )
                if cur.rowcount <= 0:
                    continue
                try:
                    payload = json.loads(c["payload_json"] or "{}")
                except Exception:
                    payload = {}
                if not isinstance(payload, dict):
                    payload = {}
                payload["instruction_id"] = iid
                rows.append(
                    {
                        "instruction_id": iid,
                        "payload": payload,
                        "signature": str(c["signature"] or ""),
                        "producer_role": str(c["producer_role"] or ""),
                        "consumer_role": str(c["consumer_role"] or ""),
                        "expires_ts": float(c["expires_ts"] or 0.0),
                    }
                )
            conn.commit()
        return rows

    def mark_consumed(self, instruction_id: str) -> None:
        iid = str(instruction_id or "")
        if not iid:
            return
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE command_instructions
                SET status = 'CONSUMED',
                    consumed_ts = ?,
                    rejected_reason = NULL
                WHERE instruction_id = ?
                """,
                (float(time.time()), iid),
            )
            conn.commit()

    def mark_rejected(self, instruction_id: str, reason: str) -> None:
        iid = str(instruction_id or "")
        if not iid:
            return
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE command_instructions
                SET status = 'REJECTED',
                    consumed_ts = ?,
                    rejected_reason = ?
                WHERE instruction_id = ?
                """,
                (float(time.time()), str(reason or "rejected"), iid),
            )
            conn.commit()

    def metrics(self) -> Dict[str, int]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT status, COUNT(*) AS n
                FROM command_instructions
                WHERE queue_name = ?
                GROUP BY status
                """,
                (self.queue,),
            )
            rows = cur.fetchall()
        out = {"PENDING": 0, "CLAIMED": 0, "CONSUMED": 0, "REJECTED": 0}
        for r in rows:
            out[str(r["status"])] = int(r["n"] or 0)
        return out

    def purge_old(self, *, max_age_seconds: float = 86400.0) -> int:
        cutoff = float(time.time()) - float(max(max_age_seconds, 60.0))
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                DELETE FROM command_instructions
                WHERE queue_name = ?
                  AND status IN ('CONSUMED', 'REJECTED')
                  AND consumed_ts IS NOT NULL
                  AND consumed_ts < ?
                """,
                (self.queue, cutoff),
            )
            n = int(cur.rowcount or 0)
            conn.commit()
        return n


def instruction_to_signal(payload: Dict[str, Any]) -> Tuple[Optional[Any], str]:
    """Convert instruction payload into SimpleNamespace signal shape expected by execution engine."""
    try:
        from types import SimpleNamespace

        symbol = str(payload.get("symbol", "") or "").strip()
        action = str(payload.get("action", "") or "").upper().strip()
        qty = float(payload.get("quantity", 0.0) or 0.0)
        entry_price = float(payload.get("entry_price", 0.0) or 0.0)
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        strategy = str(payload.get("strategy", "") or payload.get("source_strategy", "") or "bus_instruction")
        if not symbol or action not in {"BUY", "SELL"} or qty <= 0.0 or entry_price <= 0.0:
            return None, "invalid_signal_payload"
        sig = SimpleNamespace(
            symbol=symbol,
            action=action,
            side=action,
            quantity=qty,
            entry_price=entry_price,
            confidence=confidence,
            strategy=strategy,
            source_strategy=strategy,
            trace_id=str(payload.get("trace_id", "") or ""),
            reason=str(payload.get("reason", "") or ""),
            max_notional_aud=float(payload.get("max_notional_aud", 0.0) or 0.0),
        )
        return sig, "ok"
    except Exception as e:
        return None, f"signal_conversion_error:{e}"
