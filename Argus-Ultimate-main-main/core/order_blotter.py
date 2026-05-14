"""
Order Blotter — real-time queryable view of all orders.

Provides filterable access to orders by symbol, strategy, status, venue,
and time range. Backed by SQLite for persistence and fast queries.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BlotterEntry:
    """Single order in the blotter."""
    order_id: str
    symbol: str
    side: str
    strategy: str
    venue: str
    status: str  # "open", "filled", "partial", "cancelled", "rejected"
    quantity: float
    filled_qty: float
    price: float
    fill_price: float
    commission: float
    slippage_bps: float
    created_at: float
    updated_at: float
    reason: str = ""


class OrderBlotter:
    """
    SQLite-backed order blotter with real-time filtering.

    Usage:
        blotter = OrderBlotter()
        blotter.record(BlotterEntry(...))
        open_orders = blotter.query(status="open")
        btc_fills = blotter.query(symbol="BTC/USD", status="filled")
        recent = blotter.query(since_seconds=3600)  # last hour
    """

    def __init__(self, db_path: str = "data/order_blotter.db"):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS blotter (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    venue TEXT NOT NULL,
                    status TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    filled_qty REAL DEFAULT 0,
                    price REAL NOT NULL,
                    fill_price REAL DEFAULT 0,
                    commission REAL DEFAULT 0,
                    slippage_bps REAL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    reason TEXT DEFAULT ''
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_blotter_symbol ON blotter(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_blotter_status ON blotter(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_blotter_strategy ON blotter(strategy)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_blotter_created ON blotter(created_at)")

    def record(self, entry: BlotterEntry) -> None:
        """Record or update an order in the blotter."""
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO blotter
                (order_id, symbol, side, strategy, venue, status, quantity,
                 filled_qty, price, fill_price, commission, slippage_bps,
                 created_at, updated_at, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.order_id, entry.symbol, entry.side, entry.strategy,
                entry.venue, entry.status, entry.quantity, entry.filled_qty,
                entry.price, entry.fill_price, entry.commission,
                entry.slippage_bps, entry.created_at, entry.updated_at,
                entry.reason,
            ))

    def update_status(self, order_id: str, status: str, fill_price: float = 0,
                      filled_qty: float = 0, reason: str = "") -> None:
        """Update order status."""
        with self._connect() as conn:
            conn.execute("""
                UPDATE blotter SET status=?, fill_price=?, filled_qty=?,
                updated_at=?, reason=? WHERE order_id=?
            """, (status, fill_price, filled_qty, time.time(), reason, order_id))

    def query(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        venue: Optional[str] = None,
        status: Optional[str] = None,
        since_seconds: Optional[float] = None,
        limit: int = 100,
    ) -> List[BlotterEntry]:
        """Query orders with filters."""
        conditions = []
        params = []

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if strategy:
            conditions.append("strategy = ?")
            params.append(strategy)
        if venue:
            conditions.append("venue = ?")
            params.append(venue)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if since_seconds:
            conditions.append("created_at > ?")
            params.append(time.time() - since_seconds)

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM blotter WHERE {where} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            BlotterEntry(
                order_id=r["order_id"], symbol=r["symbol"], side=r["side"],
                strategy=r["strategy"], venue=r["venue"], status=r["status"],
                quantity=r["quantity"], filled_qty=r["filled_qty"],
                price=r["price"], fill_price=r["fill_price"],
                commission=r["commission"], slippage_bps=r["slippage_bps"],
                created_at=r["created_at"], updated_at=r["updated_at"],
                reason=r["reason"],
            )
            for r in rows
        ]

    def summary(self) -> Dict[str, Any]:
        """Get blotter summary stats."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM blotter").fetchone()[0]
            by_status = {}
            for row in conn.execute("SELECT status, COUNT(*) as cnt FROM blotter GROUP BY status"):
                by_status[row["status"]] = row["cnt"]
        return {"total_orders": total, "by_status": by_status}
