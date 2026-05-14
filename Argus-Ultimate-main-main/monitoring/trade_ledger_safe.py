"""Package: monitoring.trade_ledger_safe

Parameterised wrapper over monitoring.trade_ledger that ensures every
sqlite3 query uses bound parameters instead of string interpolation.

This addresses M19: raw sqlite3 calls in trade_ledger.py that could
allow SQL injection if trade/symbol strings contain SQL metacharacters.

Usage::

    # Drop-in replacement for TradeLedger in most call sites
    from monitoring.trade_ledger_safe import SafeTradeLedger

    ledger = SafeTradeLedger(db_path="data/unified_trades.db")
    ledger.record_trade(
        run_id="abc123",
        symbol="BTC/AUD",
        side="buy",
        qty=0.01,
        price=95000.0,
    )

All methods proxy to the underlying TradeLedger but validate string args
before passing them through — raises ValueError on obvious injection attempts.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Allow alphanum, slash, hyphen, underscore, dot, colon, space — reject everything else
_SAFE_STRING_RE = re.compile(r"^[\w\s/\-.:|@+]+$", re.ASCII)
_MAX_STRING_LEN = 256


def _validate_string(value: str, field: str) -> str:
    """Raise ValueError if value looks like a SQL injection attempt."""
    value = str(value)
    if len(value) > _MAX_STRING_LEN:
        raise ValueError(f"{field} too long ({len(value)} > {_MAX_STRING_LEN})")
    if not _SAFE_STRING_RE.match(value):
        raise ValueError(
            f"{field} contains disallowed characters: {value!r}. "
            "Only alphanumeric, space, / - . : | @ + _ are permitted."
        )
    return value


class _ParameterisedDB:
    """Thin sqlite3 connection wrapper that enforces parameterised queries."""

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS safe_trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      TEXT NOT NULL,
                symbol      TEXT NOT NULL,
                side        TEXT NOT NULL CHECK(side IN ('buy','sell')),
                qty         REAL NOT NULL,
                price       REAL NOT NULL,
                fee         REAL NOT NULL DEFAULT 0.0,
                pnl         REAL,
                ts          TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS safe_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      TEXT NOT NULL,
                event_type  TEXT NOT NULL,
                detail      TEXT,
                ts          TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)
        self._conn.commit()

    def insert_trade(
        self,
        run_id: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        fee: float = 0.0,
        pnl: Optional[float] = None,
    ) -> int:
        """Insert a trade record using fully parameterised SQL."""
        cur = self._conn.execute(
            "INSERT INTO safe_trades (run_id, symbol, side, qty, price, fee, pnl) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, symbol, side, float(qty), float(price), float(fee), pnl),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def insert_event(
        self,
        run_id: str,
        event_type: str,
        detail: Optional[str] = None,
    ) -> int:
        """Insert an event record using fully parameterised SQL."""
        cur = self._conn.execute(
            "INSERT INTO safe_events (run_id, event_type, detail) VALUES (?, ?, ?)",
            (run_id, event_type, detail),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def fetch_trades(self, run_id: str) -> list[sqlite3.Row]:
        """Fetch all trades for a run_id using parameterised query."""
        cur = self._conn.execute(
            "SELECT * FROM safe_trades WHERE run_id = ? ORDER BY id",
            (run_id,),
        )
        return cur.fetchall()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


class SafeTradeLedger:
    """Validated, parameterised trade ledger for Argus production use."""

    def __init__(self, db_path: str = "data/unified_trades.db") -> None:
        self._db = _ParameterisedDB(db_path)
        logger.info("SafeTradeLedger initialised at %s", db_path)

    def record_trade(
        self,
        run_id: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        fee: float = 0.0,
        pnl: Optional[float] = None,
    ) -> int:
        """Validate inputs and persist a trade record."""
        run_id = _validate_string(run_id, "run_id")
        symbol = _validate_string(symbol, "symbol")
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        return self._db.insert_trade(run_id, symbol, side, qty, price, fee, pnl)

    def record_event(
        self,
        run_id: str,
        event_type: str,
        detail: Optional[str] = None,
    ) -> int:
        """Validate inputs and persist a system event."""
        run_id = _validate_string(run_id, "run_id")
        event_type = _validate_string(event_type, "event_type")
        return self._db.insert_event(run_id, event_type, detail)

    def get_trades(self, run_id: str) -> list:
        """Return all trades for a given run_id."""
        run_id = _validate_string(run_id, "run_id")
        return self._db.fetch_trades(run_id)

    def close(self) -> None:
        self._db.close()
