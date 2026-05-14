"""
Execution state store (SQLite).

This is the canonical persistence for:
- order lifecycle (placed/reconciled/filled/rejected/cancelled)
- positions (quantity + avg price)

It is intentionally lightweight and dependency-free.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_INTENT_ALLOWED_TRANSITIONS: Dict[str, set[str]] = {
    "CREATED": {"CREATED", "SENT", "FAILED", "CANCELED", "RECON_REQUIRED"},
    "SENT": {"SENT", "ACKED", "PARTIALLY_FILLED", "FILLED", "CANCELED", "FAILED", "UNKNOWN", "RECON_REQUIRED"},
    "ACKED": {"ACKED", "PARTIALLY_FILLED", "FILLED", "CANCELED", "FAILED", "UNKNOWN", "RECON_REQUIRED"},
    "PARTIALLY_FILLED": {"PARTIALLY_FILLED", "FILLED", "CANCELED", "FAILED", "UNKNOWN", "RECON_REQUIRED"},
    "UNKNOWN": {"UNKNOWN", "RECON_REQUIRED", "FAILED", "CANCELED"},
    "RECON_REQUIRED": {"RECON_REQUIRED", "ACKED", "PARTIALLY_FILLED", "FILLED", "CANCELED", "FAILED"},
    "FAILED": {"FAILED"},
    "CANCELED": {"CANCELED"},
    "FILLED": {"FILLED"},
}


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    current_price: float = 0.0


class ExecutionStateStore:
    def __init__(self, db_path: str = "data/unified_state.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
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
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    order_id TEXT NOT NULL,
                    client_order_id TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    exchange TEXT,
                    type TEXT,
                    status TEXT NOT NULL,
                    amount REAL,
                    filled REAL,
                    price REAL,
                    raw_json TEXT
                )
                """
            )
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_orders_order_id ON orders(order_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders(timestamp)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity REAL NOT NULL,
                    avg_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    updated_ts REAL NOT NULL
                )
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS account_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_ts REAL NOT NULL
                )
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS idempotency (
                    key TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS order_intents (
                    intent_id TEXT PRIMARY KEY,
                    created_ts REAL NOT NULL,
                    updated_ts REAL NOT NULL,
                    state TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    expected_price REAL NOT NULL,
                    exchange TEXT,
                    run_id TEXT,
                    trace_id TEXT,
                    client_order_id TEXT,
                    exchange_order_id TEXT,
                    execution_plan_json TEXT,
                    details_json TEXT,
                    last_error TEXT
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_order_intents_state ON order_intents(state)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_order_intents_trace ON order_intents(trace_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_order_intents_exchange_order ON order_intents(exchange_order_id)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS order_intent_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    intent_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    from_state TEXT,
                    to_state TEXT NOT NULL,
                    reason TEXT,
                    details_json TEXT
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_order_intent_transitions_intent ON order_intent_transitions(intent_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_order_intent_transitions_ts ON order_intent_transitions(timestamp)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS recon_recovery_state (
                    intent_id TEXT PRIMARY KEY,
                    retry_count INTEGER NOT NULL,
                    recovery_status TEXT NOT NULL,
                    last_retry_ts REAL NOT NULL,
                    resolution_reason TEXT
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_recon_recovery_status ON recon_recovery_state(recovery_status)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_recon_recovery_retry_ts ON recon_recovery_state(last_retry_ts)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS recon_recovery_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    intent_id TEXT NOT NULL,
                    retry_count INTEGER NOT NULL,
                    recovery_status TEXT NOT NULL,
                    resolution_reason TEXT,
                    recovery_classification TEXT
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_recon_recovery_history_intent ON recon_recovery_history(intent_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_recon_recovery_history_ts ON recon_recovery_history(timestamp)")
            conn.commit()

    def upsert_order(self, order: Dict[str, Any]) -> None:
        ts = float(order.get("timestamp") or time.time())
        order_id = str(order.get("order_id") or order.get("id") or "")
        if not order_id:
            return
        symbol = str(order.get("symbol") or "")
        side = str(order.get("side") or "")
        status = str(order.get("status") or "unknown")
        exchange = str(order.get("exchange") or "")
        typ = str(order.get("type") or order.get("order_type") or "")
        client_order_id = str(order.get("client_order_id") or "")
        amount = _to_float(order.get("amount") or order.get("quantity"))
        filled = _to_float(order.get("filled"))
        price = _to_float(order.get("price"))
        raw_json = None
        try:
            raw_json = json.dumps(order.get("raw") or order, ensure_ascii=True, default=str)
        except Exception:
            raw_json = None

        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO orders (timestamp, order_id, client_order_id, symbol, side, exchange, type, status, amount, filled, price, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(order_id) DO UPDATE SET
                    timestamp=excluded.timestamp,
                    client_order_id=excluded.client_order_id,
                    status=excluded.status,
                    filled=COALESCE(excluded.filled, orders.filled),
                    price=COALESCE(excluded.price, orders.price),
                    raw_json=COALESCE(excluded.raw_json, orders.raw_json)
                """,
                (ts, order_id, client_order_id, symbol, side, exchange, typ, status, amount, filled, price, raw_json),
            )
            conn.commit()

    def update_position_price(self, symbol: str, current_price: float) -> None:
        sym = str(symbol or "")
        if not sym:
            return
        px = float(current_price or 0.0)
        now = float(time.time())
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO positions (symbol, quantity, avg_price, current_price, updated_ts)
                VALUES (?, 0.0, 0.0, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    current_price=excluded.current_price,
                    updated_ts=excluded.updated_ts
                """,
                (sym, px, now),
            )
            conn.commit()

    def apply_fill(self, *, symbol: str, side: str, quantity: float, price: float) -> None:
        sym = str(symbol or "")
        if not sym:
            return
        side_u = str(side or "").upper()
        q = float(quantity or 0.0)
        px = float(price or 0.0)
        if q <= 0 or px <= 0:
            return

        now = float(time.time())
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT symbol, quantity, avg_price, current_price FROM positions WHERE symbol = ?", (sym,))
            row = cur.fetchone()
            held = float(row["quantity"]) if row else 0.0
            avg = float(row["avg_price"]) if row else 0.0
            cur_px = float(row["current_price"]) if row else 0.0

            if side_u == "BUY":
                new_qty = held + q
                new_avg = ((held * avg) + (q * px)) / new_qty if new_qty > 0 else 0.0
                cur.execute(
                    """
                    INSERT INTO positions (symbol, quantity, avg_price, current_price, updated_ts)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        quantity=excluded.quantity,
                        avg_price=excluded.avg_price,
                        current_price=excluded.current_price,
                        updated_ts=excluded.updated_ts
                    """,
                    (sym, new_qty, new_avg, (cur_px or px), now),
                )

            elif side_u == "SELL":
                sell_qty = min(q, held)
                new_qty = max(0.0, held - sell_qty)
                new_avg = avg if new_qty > 0 else 0.0
                cur.execute(
                    """
                    INSERT INTO positions (symbol, quantity, avg_price, current_price, updated_ts)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        quantity=excluded.quantity,
                        avg_price=excluded.avg_price,
                        current_price=excluded.current_price,
                        updated_ts=excluded.updated_ts
                    """,
                    (sym, new_qty, new_avg, (cur_px or px), now),
                )
            else:
                return

            conn.commit()

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT symbol, quantity, avg_price, current_price FROM positions")
            rows = cur.fetchall()
        out: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            sym = str(r["symbol"])
            out[sym] = {
                "quantity": float(r["quantity"] or 0.0),
                "avg_price": float(r["avg_price"] or 0.0),
                "current_price": float(r["current_price"] or 0.0),
            }
        return out

    def set_position(self, symbol: str, quantity: float, avg_price: float, current_price: float) -> None:
        sym = str(symbol or "")
        if not sym:
            return
        now = float(time.time())
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO positions (symbol, quantity, avg_price, current_price, updated_ts)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    quantity=excluded.quantity,
                    avg_price=excluded.avg_price,
                    current_price=excluded.current_price,
                    updated_ts=excluded.updated_ts
                """,
                (sym, float(quantity), float(avg_price), float(current_price), now),
            )
            conn.commit()

    def set_account_value(self, key: str, value: Any) -> None:
        k = str(key or "")
        if not k:
            return
        now = float(time.time())
        try:
            v = json.dumps(value, ensure_ascii=True, default=str)
        except Exception:
            v = str(value)
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO account_state (key, value, updated_ts)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_ts=excluded.updated_ts
                """,
                (k, v, now),
            )
            conn.commit()

    def get_account_value(self, key: str, default: Any = None) -> Any:
        k = str(key or "")
        if not k:
            return default
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT value FROM account_state WHERE key = ?", (k,))
            row = cur.fetchone()
        if not row:
            return default
        raw = row["value"]
        try:
            return json.loads(raw)
        except Exception:
            return raw

    def seen_or_mark(self, key: str) -> bool:
        """
        Idempotency primitive:
        - returns True if key already exists
        - otherwise inserts and returns False
        """
        k = str(key or "")
        if not k:
            return False
        now = float(time.time())
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute("INSERT INTO idempotency (key, timestamp) VALUES (?, ?)", (k, now))
                conn.commit()
                return False
            except sqlite3.IntegrityError:
                return True

    def get_order_by_client_order_id(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        cid = str(client_order_id or "")
        if not cid:
            return None
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM orders WHERE client_order_id = ? ORDER BY timestamp DESC LIMIT 1", (cid,))
            row = cur.fetchone()
        return dict(row) if row else None

    def get_open_orders(self, *, exchange: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return latest known open/pending orders (deduplicated by order_id).
        """
        params: List[Any] = []
        where = "LOWER(status) IN ('open', 'pending', 'partially_filled', 'partial')"
        if exchange:
            where += " AND exchange = ?"
            params.append(str(exchange))
        query = f"""
            SELECT o.*
            FROM orders o
            JOIN (
                SELECT order_id, MAX(timestamp) AS max_ts
                FROM orders
                GROUP BY order_id
            ) latest
              ON latest.order_id = o.order_id AND latest.max_ts = o.timestamp
            WHERE {where}
            ORDER BY o.timestamp DESC
        """
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def get_recent_fills(self, *, exchange: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Return latest known filled/closed orders, newest first.
        """
        lim = max(1, int(limit or 200))
        params: List[Any] = []
        where = "LOWER(status) IN ('filled', 'closed')"
        if exchange:
            where += " AND exchange = ?"
            params.append(str(exchange))
        query = f"""
            SELECT o.*
            FROM orders o
            JOIN (
                SELECT order_id, MAX(timestamp) AS max_ts
                FROM orders
                GROUP BY order_id
            ) latest
              ON latest.order_id = o.order_id AND latest.max_ts = o.timestamp
            WHERE {where}
            ORDER BY o.timestamp DESC
            LIMIT ?
        """
        params.append(lim)
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def create_intent(
        self,
        *,
        intent_id: str,
        symbol: str,
        side: str,
        quantity: float,
        expected_price: float,
        exchange: str = "",
        run_id: str = "",
        trace_id: str = "",
        client_order_id: str = "",
        execution_plan: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        iid = str(intent_id or "")
        if not iid:
            raise ValueError("intent_id is required")
        now = float(time.time())
        ep_json = json.dumps(execution_plan or {}, ensure_ascii=True, default=str)
        details_json = json.dumps(details or {}, ensure_ascii=True, default=str)
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM order_intents WHERE intent_id = ?", (iid,))
            existing = cur.fetchone()
            if existing:
                return dict(existing)
            cur.execute(
                """
                INSERT INTO order_intents (
                    intent_id, created_ts, updated_ts, state, symbol, side, quantity, expected_price, exchange,
                    run_id, trace_id, client_order_id, execution_plan_json, details_json, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    iid,
                    now,
                    now,
                    "CREATED",
                    str(symbol or ""),
                    str(side or ""),
                    float(quantity or 0.0),
                    float(expected_price or 0.0),
                    str(exchange or ""),
                    str(run_id or ""),
                    str(trace_id or ""),
                    str(client_order_id or ""),
                    ep_json,
                    details_json,
                    None,
                ),
            )
            cur.execute(
                """
                INSERT INTO order_intent_transitions (intent_id, timestamp, from_state, to_state, reason, details_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    iid,
                    now,
                    None,
                    "CREATED",
                    "intent_created",
                    details_json,
                ),
            )
            conn.commit()
            cur.execute("SELECT * FROM order_intents WHERE intent_id = ?", (iid,))
            row = cur.fetchone()
        return dict(row) if row else {}

    def _validate_intent_transition(self, from_state: str, to_state: str) -> None:
        src = str(from_state or "").upper()
        dst = str(to_state or "").upper()
        if not src or not dst:
            raise ValueError("intent transition requires non-empty states")
        allowed = _INTENT_ALLOWED_TRANSITIONS.get(src)
        if allowed is None:
            # Backward compatibility for legacy/custom states: allow transition but warn.
            logger.warning("Unknown intent state '%s'; allowing transition to '%s'", src, dst)
            return
        if dst not in allowed:
            raise ValueError(f"invalid intent transition: {src} -> {dst}")

    def get_intent(self, intent_id: str) -> Optional[Dict[str, Any]]:
        iid = str(intent_id or "")
        if not iid:
            return None
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM order_intents WHERE intent_id = ?", (iid,))
            row = cur.fetchone()
        return dict(row) if row else None

    def update_intent_state(
        self,
        intent_id: str,
        state: str,
        *,
        exchange_order_id: Optional[str] = None,
        last_error: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        iid = str(intent_id or "")
        st = str(state or "").upper()
        if not iid or not st:
            return None
        now = float(time.time())
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT state, details_json FROM order_intents WHERE intent_id = ?", (iid,))
            row = cur.fetchone()
            if not row:
                return None
            old_state = str(row["state"] or "").upper()
            self._validate_intent_transition(old_state, st)
            existing_details: Dict[str, Any] = {}
            try:
                existing_details = json.loads(row["details_json"] or "{}")
                if not isinstance(existing_details, dict):
                    existing_details = {}
            except Exception:
                existing_details = {}
            if details:
                existing_details.update(details)
            cur.execute(
                """
                UPDATE order_intents
                SET state = ?,
                    updated_ts = ?,
                    exchange_order_id = COALESCE(?, exchange_order_id),
                    last_error = COALESCE(?, last_error),
                    details_json = ?
                WHERE intent_id = ?
                """,
                (
                    st,
                    now,
                    str(exchange_order_id) if exchange_order_id else None,
                    str(last_error) if last_error else None,
                    json.dumps(existing_details, ensure_ascii=True, default=str),
                    iid,
                ),
            )
            cur.execute(
                """
                INSERT INTO order_intent_transitions (intent_id, timestamp, from_state, to_state, reason, details_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    iid,
                    now,
                    old_state,
                    st,
                    str(last_error or ""),
                    json.dumps(existing_details, ensure_ascii=True, default=str),
                ),
            )
            conn.commit()
            cur.execute("SELECT * FROM order_intents WHERE intent_id = ?", (iid,))
            updated = cur.fetchone()
        return dict(updated) if updated else None

    def get_intent_transitions(self, intent_id: str) -> List[Dict[str, Any]]:
        iid = str(intent_id or "")
        if not iid:
            return []
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT intent_id, timestamp, from_state, to_state, reason, details_json
                FROM order_intent_transitions
                WHERE intent_id = ?
                ORDER BY id ASC
                """,
                (iid,),
            )
            rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            out.append(dict(row))
        return out

    def count_intents(self, intent_id: Optional[str] = None) -> int:
        with self._connect() as conn:
            cur = conn.cursor()
            if intent_id:
                cur.execute("SELECT COUNT(*) AS n FROM order_intents WHERE intent_id = ?", (str(intent_id),))
            else:
                cur.execute("SELECT COUNT(*) AS n FROM order_intents")
            row = cur.fetchone()
        return int(row["n"]) if row else 0

    def has_recon_required_intent(self, symbol: Optional[str] = None) -> bool:
        with self._connect() as conn:
            cur = conn.cursor()
            if symbol:
                cur.execute(
                    "SELECT 1 FROM order_intents WHERE state = 'RECON_REQUIRED' AND symbol = ? LIMIT 1",
                    (str(symbol),),
                )
            else:
                cur.execute("SELECT 1 FROM order_intents WHERE state = 'RECON_REQUIRED' LIMIT 1")
            row = cur.fetchone()
        return bool(row)

    def intent_is_recon_required(self, intent_id: str) -> bool:
        iid = str(intent_id or "")
        if not iid:
            return False
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT state FROM order_intents WHERE intent_id = ?", (iid,))
            row = cur.fetchone()
        return str((row["state"] if row else "") or "").upper() == "RECON_REQUIRED"

    def list_recon_required_intents(
        self,
        *,
        symbol: Optional[str] = None,
        stale_after_seconds: float = 0.0,
        now_ts: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        now = float(now_ts or time.time())
        stale = max(0.0, float(stale_after_seconds or 0.0))
        with self._connect() as conn:
            cur = conn.cursor()
            if symbol:
                cur.execute(
                    """
                    SELECT intent_id, symbol, state, last_error, updated_ts, exchange, exchange_order_id
                    FROM order_intents
                    WHERE state = 'RECON_REQUIRED' AND symbol = ?
                    """,
                    (str(symbol),),
                )
            else:
                cur.execute(
                    """
                    SELECT intent_id, symbol, state, last_error, updated_ts, exchange, exchange_order_id
                    FROM order_intents
                    WHERE state = 'RECON_REQUIRED'
                    """
                )
            rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            updated = float(d.get("updated_ts") or 0.0)
            if stale > 0.0 and (now - updated) <= stale:
                continue
            out.append(d)
        return out

    def upsert_recon_recovery_state(
        self,
        *,
        intent_id: str,
        retry_count: int,
        recovery_status: str,
        last_retry_ts: float,
        resolution_reason: Optional[str] = None,
        recovery_classification: Optional[str] = None,
    ) -> None:
        iid = str(intent_id or "")
        if not iid:
            return
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO recon_recovery_state (
                    intent_id, retry_count, recovery_status, last_retry_ts, resolution_reason
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(intent_id) DO UPDATE SET
                    retry_count=excluded.retry_count,
                    recovery_status=excluded.recovery_status,
                    last_retry_ts=excluded.last_retry_ts,
                    resolution_reason=excluded.resolution_reason
                """,
                (
                    iid,
                    int(retry_count or 0),
                    str(recovery_status or "pending"),
                    float(last_retry_ts or 0.0),
                    str(resolution_reason) if resolution_reason is not None else None,
                ),
            )
            cur.execute(
                """
                INSERT INTO recon_recovery_history (
                    timestamp, intent_id, retry_count, recovery_status, resolution_reason, recovery_classification
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    float(time.time()),
                    iid,
                    int(retry_count or 0),
                    str(recovery_status or "pending"),
                    str(resolution_reason) if resolution_reason is not None else None,
                    str(recovery_classification) if recovery_classification is not None else None,
                ),
            )
            conn.commit()

    def get_recon_recovery_state(self, intent_id: str) -> Optional[Dict[str, Any]]:
        iid = str(intent_id or "")
        if not iid:
            return None
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT rr.intent_id, rr.retry_count, rr.recovery_status, rr.last_retry_ts, rr.resolution_reason,
                       oi.symbol, oi.state, oi.last_error
                FROM recon_recovery_state rr
                LEFT JOIN order_intents oi ON oi.intent_id = rr.intent_id
                WHERE rr.intent_id = ?
                """,
                (iid,),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def get_recon_recovery_states(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        n = max(1, int(limit or 100))
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT rr.intent_id, rr.retry_count, rr.recovery_status, rr.last_retry_ts, rr.resolution_reason,
                       oi.symbol, oi.state, oi.last_error
                FROM recon_recovery_state rr
                LEFT JOIN order_intents oi ON oi.intent_id = rr.intent_id
                ORDER BY rr.last_retry_ts DESC
                LIMIT ?
                """,
                (n,),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def get_recon_recovery_history(self, *, intent_id: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        n = max(1, int(limit or 200))
        iid = str(intent_id or "").strip()
        with self._connect() as conn:
            cur = conn.cursor()
            if iid:
                cur.execute(
                    """
                    SELECT id, timestamp, intent_id, retry_count, recovery_status, resolution_reason, recovery_classification
                    FROM recon_recovery_history
                    WHERE intent_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (iid, n),
                )
            else:
                cur.execute(
                    """
                    SELECT id, timestamp, intent_id, retry_count, recovery_status, resolution_reason, recovery_classification
                    FROM recon_recovery_history
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (n,),
                )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def clear_recon_required_intents(self, *, symbol: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> int:
        now = float(time.time())
        match_symbol = str(symbol or "")
        updated = 0
        with self._connect() as conn:
            cur = conn.cursor()
            if match_symbol:
                cur.execute(
                    "SELECT intent_id, details_json FROM order_intents WHERE state = 'RECON_REQUIRED' AND symbol = ?",
                    (match_symbol,),
                )
            else:
                cur.execute("SELECT intent_id, details_json FROM order_intents WHERE state = 'RECON_REQUIRED'")
            rows = cur.fetchall()
            for row in rows:
                intent_id = str(row["intent_id"] or "")
                existing_details: Dict[str, Any] = {}
                try:
                    existing_details = json.loads(row["details_json"] or "{}")
                    if not isinstance(existing_details, dict):
                        existing_details = {}
                except Exception:
                    existing_details = {}
                existing_details["reconciliation_cleared"] = True
                existing_details["reconciliation_cleared_ts"] = now
                if details:
                    existing_details.update(details)
                cur.execute(
                    """
                    UPDATE order_intents
                    SET state = 'FAILED',
                        updated_ts = ?,
                        details_json = ?,
                        last_error = COALESCE(last_error, 'reconciliation_required')
                    WHERE intent_id = ?
                    """,
                    (now, json.dumps(existing_details, ensure_ascii=True, default=str), intent_id),
                )
                updated += 1
            conn.commit()
        return updated


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None
