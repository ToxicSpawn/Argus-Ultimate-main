"""
Push 86 — Argus Trade Ledger
Persistent SQLite-backed auditable PnL tracker.
Records every fill, computes running equity, and exposes
query/export methods for reporting and auditing.
"""

from __future__ import annotations

import sqlite3
import csv
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Fill:
    """
    Represents a single filled order.

    Fields
    ------
    symbol      : trading pair, e.g. 'BTC/USDT'
    side        : 'buy' or 'sell'
    qty         : base asset quantity filled
    price       : fill price in quote currency
    fee         : fee paid in quote currency
    fee_currency: currency fee was charged in
    exchange    : exchange name
    order_id    : exchange order ID
    strategy    : strategy name that generated the signal
    tags        : free-form JSON string for extra metadata
    timestamp_ms: fill timestamp in Unix milliseconds (UTC)
    """
    symbol: str
    side: str
    qty: float
    price: float
    fee: float
    fee_currency: str
    exchange: str
    order_id: str
    strategy: str
    tags: str = "{}"
    timestamp_ms: Optional[int] = None

    def __post_init__(self):
        if self.timestamp_ms is None:
            self.timestamp_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        self.side = self.side.lower()
        assert self.side in ("buy", "sell"), f"Invalid side: {self.side}"
        assert self.qty > 0, "qty must be positive"
        assert self.price > 0, "price must be positive"

    @property
    def notional(self) -> float:
        """Quote value of the fill (before fees)."""
        return self.qty * self.price

    @property
    def net_value(self) -> float:
        """Net quote value after fees (negative = cost for buys)."""
        if self.side == "buy":
            return -(self.notional + self.fee)
        return self.notional - self.fee


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

CREATE_FILLS_SQL = """
CREATE TABLE IF NOT EXISTS fills (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ms    INTEGER NOT NULL,
    symbol          TEXT    NOT NULL,
    side            TEXT    NOT NULL,
    qty             REAL    NOT NULL,
    price           REAL    NOT NULL,
    fee             REAL    NOT NULL DEFAULT 0.0,
    fee_currency    TEXT    NOT NULL DEFAULT 'USDT',
    exchange        TEXT    NOT NULL DEFAULT '',
    order_id        TEXT    NOT NULL DEFAULT '',
    strategy        TEXT    NOT NULL DEFAULT '',
    tags            TEXT    NOT NULL DEFAULT '{}',
    notional        REAL    GENERATED ALWAYS AS (qty * price) STORED,
    net_value       REAL    GENERATED ALWAYS AS (
                        CASE side
                            WHEN 'buy'  THEN -(qty * price + fee)
                            WHEN 'sell' THEN  (qty * price - fee)
                            ELSE 0
                        END
                    ) STORED
);
"""

CREATE_IDX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_fills_symbol   ON fills(symbol);",
    "CREATE INDEX IF NOT EXISTS idx_fills_ts       ON fills(timestamp_ms);",
    "CREATE INDEX IF NOT EXISTS idx_fills_strategy ON fills(strategy);",
    "CREATE INDEX IF NOT EXISTS idx_fills_exchange ON fills(exchange);",
]

INSERT_FILL_SQL = """
INSERT INTO fills
    (timestamp_ms, symbol, side, qty, price, fee, fee_currency,
     exchange, order_id, strategy, tags)
VALUES
    (:timestamp_ms, :symbol, :side, :qty, :price, :fee, :fee_currency,
     :exchange, :order_id, :strategy, :tags)
"""


class TradeLedger:
    """
    Persistent, thread-safe (WAL mode) trade ledger backed by SQLite.

    Usage
    -----
        ledger = TradeLedger("data/trades.db")
        ledger.record(Fill(symbol="BTC/USDT", side="buy", ...))
        report = ledger.pnl_report(symbol="BTC/USDT")
    """

    def __init__(self, db_path: str | Path = "data/trade_ledger.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._tx() as conn:
            conn.execute(CREATE_FILLS_SQL)
            for idx_sql in CREATE_IDX_SQL:
                conn.execute(idx_sql)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(self, fill: Fill) -> int:
        """
        Persist a Fill to the ledger.
        Returns the auto-incremented row ID.
        """
        row = asdict(fill)
        with self._tx() as conn:
            cur = conn.execute(INSERT_FILL_SQL, row)
            return cur.lastrowid

    def record_many(self, fills: list[Fill]) -> int:
        """Bulk-insert fills. Returns number of rows inserted."""
        rows = [asdict(f) for f in fills]
        with self._tx() as conn:
            cur = conn.executemany(INSERT_FILL_SQL, rows)
            return cur.rowcount

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        conn = self._connect()
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    def get_fills(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        exchange: Optional[str] = None,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Flexible fill query with optional filters."""
        clauses = []
        params: list = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if strategy:
            clauses.append("strategy = ?")
            params.append(strategy)
        if exchange:
            clauses.append("exchange = ?")
            params.append(exchange)
        if since_ms is not None:
            clauses.append("timestamp_ms >= ?")
            params.append(since_ms)
        if until_ms is not None:
            clauses.append("timestamp_ms <= ?")
            params.append(until_ms)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM fills {where} ORDER BY timestamp_ms ASC LIMIT ?"
        params.append(limit)
        rows = self._query(sql, tuple(params))
        return [dict(r) for r in rows]

    def pnl_report(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        exchange: Optional[str] = None,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
    ) -> dict:
        """
        Compute aggregated PnL statistics.

        Returns
        -------
        dict with keys:
            symbol, strategy, exchange, total_fills, total_volume,
            total_fees, realised_pnl, buy_count, sell_count
        """
        fills = self.get_fills(
            symbol=symbol, strategy=strategy, exchange=exchange,
            since_ms=since_ms, until_ms=until_ms, limit=1_000_000
        )
        if not fills:
            return {"error": "no fills found", "filters": {
                "symbol": symbol, "strategy": strategy,
                "exchange": exchange, "since_ms": since_ms, "until_ms": until_ms
            }}

        total_volume = sum(f["notional"] for f in fills)
        total_fees = sum(f["fee"] for f in fills)
        realised_pnl = sum(f["net_value"] for f in fills)
        buy_count = sum(1 for f in fills if f["side"] == "buy")
        sell_count = sum(1 for f in fills if f["side"] == "sell")

        return {
            "symbol": symbol or "all",
            "strategy": strategy or "all",
            "exchange": exchange or "all",
            "total_fills": len(fills),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "total_volume_usd": round(total_volume, 4),
            "total_fees_usd": round(total_fees, 4),
            "realised_pnl_usd": round(realised_pnl, 4),
        }

    def equity_curve(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> list[dict]:
        """
        Running cumulative PnL curve for charting/reporting.
        Returns list of {timestamp_ms, cumulative_pnl} dicts.
        """
        fills = self.get_fills(symbol=symbol, strategy=strategy, limit=1_000_000)
        cumulative = 0.0
        curve = []
        for f in fills:
            cumulative += f["net_value"]
            curve.append({"timestamp_ms": f["timestamp_ms"],
                           "cumulative_pnl": round(cumulative, 4)})
        return curve

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, out_path: str | Path,
                   symbol: Optional[str] = None,
                   strategy: Optional[str] = None) -> int:
        """Export fills to CSV. Returns number of rows written."""
        fills = self.get_fills(symbol=symbol, strategy=strategy, limit=1_000_000)
        if not fills:
            return 0
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fills[0].keys())
            writer.writeheader()
            writer.writerows(fills)
        return len(fills)

    def export_json(self, out_path: str | Path,
                    symbol: Optional[str] = None,
                    strategy: Optional[str] = None) -> int:
        """Export fills to JSON. Returns number of rows written."""
        fills = self.get_fills(symbol=symbol, strategy=strategy, limit=1_000_000)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(fills, f, indent=2)
        return len(fills)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Total number of fills in the ledger."""
        rows = self._query("SELECT COUNT(*) as n FROM fills")
        return rows[0]["n"]

    def symbols(self) -> list[str]:
        """All distinct symbols in the ledger."""
        rows = self._query("SELECT DISTINCT symbol FROM fills ORDER BY symbol")
        return [r["symbol"] for r in rows]

    def strategies(self) -> list[str]:
        """All distinct strategy names in the ledger."""
        rows = self._query("SELECT DISTINCT strategy FROM fills ORDER BY strategy")
        return [r["strategy"] for r in rows]

    def __repr__(self) -> str:
        return f"TradeLedger(db={self.db_path}, fills={self.count()})"
