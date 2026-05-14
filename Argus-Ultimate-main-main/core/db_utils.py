"""
db_utils — safe SQLite query helpers.

Problem addressed (Issue #20):
  Any query built with f-strings is a SQL injection vector, even when the
  source data comes from internal config or signals.

Rules:
  1. Never build WHERE clause values via f-string — use parameterised queries.
  2. Never interpolate table names via f-string — use validate_table_name().

Usage:
    from core.db_utils import execute, validate_table_name

    conn = sqlite3.connect("trades.db")
    rows = execute(conn, "SELECT * FROM trades WHERE symbol = ?", (symbol,))

    # Dynamic table names:
    tbl = validate_table_name(table_arg, allowed={"trades", "orders", "fills"})
    rows = execute(conn, f"SELECT * FROM {tbl} LIMIT 100")  # safe — tbl allowlisted
"""
from __future__ import annotations

import sqlite3
from typing import Any, Sequence, Set

# Default allowlist for production tables.  Extend per-module by passing
# the `allowed` argument to validate_table_name().
_DEFAULT_ALLOWED_TABLES: Set[str] = {
    "trades",
    "orders",
    "fills",
    "equity_snapshots",
    "signals",
    "paper_trades",
    "audit_log",
}


def validate_table_name(
    name: str,
    allowed: Set[str] | None = None,
) -> str:
    """
    Return `name` if it is in the allowlist, otherwise raise ValueError.

    This is the safe way to handle dynamic table names where SQLite
    parameterisation is not available.
    """
    table_set = allowed if allowed is not None else _DEFAULT_ALLOWED_TABLES
    if name not in table_set:
        raise ValueError(
            f"Invalid table name '{name}'. Allowed: {sorted(table_set)}"
        )
    return name


def execute(
    conn: sqlite3.Connection,
    sql: str,
    params: Sequence[Any] = (),
) -> list[dict[str, Any]]:
    """
    Execute a parameterised SQL query and return rows as a list of dicts.

    Always use `?` placeholders for values — never f-strings.
    """
    cursor = conn.execute(sql, params)
    if cursor.description is None:
        return []
    columns = [col[0] for col in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def executemany(
    conn: sqlite3.Connection,
    sql: str,
    params_seq: Sequence[Sequence[Any]],
) -> None:
    """
    Execute a parameterised SQL statement for each item in params_seq.
    Used for bulk INSERT/UPDATE.
    """
    conn.executemany(sql, params_seq)
