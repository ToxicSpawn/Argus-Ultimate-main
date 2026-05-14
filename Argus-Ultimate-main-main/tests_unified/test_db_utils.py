"""
Tests for core/db_utils.py — Issue #20 SQL injection guard.
"""
from __future__ import annotations

import sqlite3
import pytest

from core.db_utils import execute, executemany, validate_table_name


@pytest.fixture()
def mem_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE trades (id INTEGER PRIMARY KEY, symbol TEXT, pnl REAL)"
    )
    conn.execute("INSERT INTO trades (symbol, pnl) VALUES (?, ?)", ("BTC/USD", 12.5))
    conn.execute("INSERT INTO trades (symbol, pnl) VALUES (?, ?)", ("ETH/USD", -3.2))
    conn.commit()
    return conn


def test_execute_parameterised(mem_db: sqlite3.Connection) -> None:
    rows = execute(mem_db, "SELECT * FROM trades WHERE symbol = ?", ("BTC/USD",))
    assert len(rows) == 1
    assert rows[0]["symbol"] == "BTC/USD"


def test_execute_injection_attempt_blocked(mem_db: sqlite3.Connection) -> None:
    """A classic injection payload is treated as a literal value, returning no rows."""
    payload = "' OR '1'='1"
    rows = execute(mem_db, "SELECT * FROM trades WHERE symbol = ?", (payload,))
    assert rows == []


def test_execute_returns_dict_rows(mem_db: sqlite3.Connection) -> None:
    rows = execute(mem_db, "SELECT * FROM trades")
    assert all(isinstance(r, dict) for r in rows)
    assert {"id", "symbol", "pnl"} == set(rows[0].keys())


def test_executemany_bulk_insert(mem_db: sqlite3.Connection) -> None:
    pairs = [("SOL/USD", 5.0), ("XRP/USD", 1.5)]
    executemany(mem_db, "INSERT INTO trades (symbol, pnl) VALUES (?, ?)", pairs)
    mem_db.commit()
    rows = execute(mem_db, "SELECT * FROM trades")
    assert len(rows) == 4


def test_validate_table_name_allows_known_tables() -> None:
    assert validate_table_name("trades") == "trades"
    assert validate_table_name("orders") == "orders"


def test_validate_table_name_blocks_unknown() -> None:
    with pytest.raises(ValueError, match="Invalid table name"):
        validate_table_name("user_secrets")


def test_validate_table_name_custom_allowlist() -> None:
    result = validate_table_name("my_custom_table", allowed={"my_custom_table"})
    assert result == "my_custom_table"


def test_validate_table_name_blocks_injection() -> None:
    with pytest.raises(ValueError):
        validate_table_name("trades; DROP TABLE trades --")
