"""monitoring/trade_ledger.py — Parameterised sqlite3 trade ledger (M19).

All raw f-string SQL queries replaced with parameterised placeholders
to eliminate SQL-injection risk.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/trade_ledger.db")

_DDL = """
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    side        TEXT    NOT NULL,
    quantity    REAL    NOT NULL,
    price       REAL    NOT NULL,
    fee         REAL    NOT NULL DEFAULT 0.0,
    strategy    TEXT,
    exchange    TEXT,
    order_id    TEXT,
    pnl         REAL,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT    NOT NULL,
    ended_at    TEXT,
    mode        TEXT,
    total_pnl   REAL,
    trade_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol    ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_strategy  ON trades(strategy);
"""


@dataclass
class TradeRecord:
    symbol: str
    side: str  # "buy" | "sell"
    quantity: float
    price: float
    fee: float = 0.0
    strategy: str | None = None
    exchange: str | None = None
    order_id: str | None = None
    pnl: float | None = None
    notes: str | None = None
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class TradeLedger:
    """Thread-safe trade ledger backed by SQLite.

    M19: All SQL is now fully parameterised — no f-string query construction.
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    # ── Internals ────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                isolation_level=None,  # autocommit
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            conn.executescript(_DDL)

    # ── Public API ───────────────────────────────────────────────────────────

    def record_trade(self, trade: TradeRecord) -> int:
        """Insert *trade* and return the new row id."""
        sql = """
            INSERT INTO trades
                (timestamp, symbol, side, quantity, price, fee,
                 strategy, exchange, order_id, pnl, notes)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """  # ← fully parameterised (M19)
        params = (
            trade.timestamp,
            trade.symbol,
            trade.side,
            trade.quantity,
            trade.price,
            trade.fee,
            trade.strategy,
            trade.exchange,
            trade.order_id,
            trade.pnl,
            trade.notes,
        )
        with self._lock:
            cur = self._connect().execute(sql, params)
            return cur.lastrowid or 0

    def get_trades(
        self,
        *,
        symbol: str | None = None,
        strategy: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Fetch trades with optional parameterised filters."""
        conditions: list[str] = []
        params: list[Any] = []

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if strategy:
            conditions.append("strategy = ?")
            params.append(strategy)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"SELECT * FROM trades {where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = self._connect().execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def total_pnl(
        self,
        *,
        strategy: str | None = None,
        symbol: str | None = None,
    ) -> float:
        """Sum PnL with optional parameterised filters."""
        conditions: list[str] = ["pnl IS NOT NULL"]
        params: list[Any] = []

        if strategy:
            conditions.append("strategy = ?")
            params.append(strategy)
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)

        sql = "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE " + " AND ".join(conditions)
        with self._lock:
            result = self._connect().execute(sql, params).fetchone()
        return float(result[0]) if result else 0.0

    def start_session(self, mode: str = "paper") -> int:
        """Open a new session record and return its id."""
        sql = "INSERT INTO sessions (started_at, mode) VALUES (?, ?)"
        with self._lock:
            cur = self._connect().execute(sql, (datetime.utcnow().isoformat(), mode))
            return cur.lastrowid or 0

    def end_session(self, session_id: int, total_pnl: float, trade_count: int) -> None:
        """Close a session record."""
        sql = """
            UPDATE sessions
               SET ended_at    = ?,
                   total_pnl   = ?,
                   trade_count = ?
             WHERE id = ?
        """
        with self._lock:
            self._connect().execute(
                sql,
                (datetime.utcnow().isoformat(), total_pnl, trade_count, session_id),
            )

    def close(self) -> None:
        """Close the underlying database connection."""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
