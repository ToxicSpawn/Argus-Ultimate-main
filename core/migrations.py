"""Simple SQLite migration system for ARGUS.

No external dependencies required (no Alembic). Migrations are registered
in Python and applied/rolled-back transactionally.

Usage::

    from core.migrations import MigrationManager, Migration

    mgr = MigrationManager("argus.db")
    mgr.register(Migration(
        version=1,
        description="initial schema",
        up_sql="CREATE TABLE ...",
        down_sql="DROP TABLE ...",
    ))
    mgr.apply_all()
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List

logger = logging.getLogger(__name__)

# ── dataclass ────────────────────────────────────────────────────────────────

@dataclass
class Migration:
    """A single migration step."""
    version: int
    description: str
    up_sql: str
    down_sql: str


# ── manager ──────────────────────────────────────────────────────────────────

class MigrationManager:
    """Apply / rollback ordered SQL migrations against an SQLite database."""

    _INIT_SQL = """\
CREATE TABLE IF NOT EXISTS _migrations (
    version     INTEGER PRIMARY KEY,
    description TEXT    NOT NULL,
    applied_at  TEXT    NOT NULL
);
"""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._registry: List[Migration] = []
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(self._INIT_SQL)
        logger.info("MigrationManager initialised for %s", db_path)

    # ── public API ───────────────────────────────────────────────────────

    def register(self, migration: Migration) -> None:
        """Add a migration to the registry (must be unique version)."""
        for m in self._registry:
            if m.version == migration.version:
                raise ValueError(
                    f"Duplicate migration version {migration.version}"
                )
        self._registry.append(migration)
        self._registry.sort(key=lambda m: m.version)
        logger.debug("Registered migration v%d: %s", migration.version, migration.description)

    def get_current_version(self) -> int:
        """Return the highest applied migration version, or 0 if none."""
        row = self._conn.execute(
            "SELECT MAX(version) FROM _migrations"
        ).fetchone()
        return row[0] if row[0] is not None else 0

    def pending(self) -> List[Migration]:
        """Return list of migrations not yet applied, in order."""
        current = self.get_current_version()
        return [m for m in self._registry if m.version > current]

    def apply_all(self) -> int:
        """Apply all pending migrations. Returns count of applied migrations."""
        to_apply = self.pending()
        if not to_apply:
            logger.info("No pending migrations")
            return 0

        applied = 0
        for migration in to_apply:
            self._apply_one(migration)
            applied += 1

        logger.info("Applied %d migration(s), now at v%d", applied, self.get_current_version())
        return applied

    def rollback(self, target_version: int) -> int:
        """Rollback to *target_version* (inclusive — that version stays applied).

        Returns count of rolled-back migrations.
        """
        current = self.get_current_version()
        if target_version >= current:
            logger.info("Already at v%d, nothing to rollback", current)
            return 0

        # Build lookup of registered migrations by version
        by_version = {m.version: m for m in self._registry}

        # Get applied versions > target in descending order
        rows = self._conn.execute(
            "SELECT version FROM _migrations WHERE version > ? ORDER BY version DESC",
            (target_version,),
        ).fetchall()

        rolled_back = 0
        for (ver,) in rows:
            migration = by_version.get(ver)
            if migration is None:
                raise RuntimeError(
                    f"Cannot rollback v{ver}: migration not in registry"
                )
            self._rollback_one(migration)
            rolled_back += 1

        logger.info(
            "Rolled back %d migration(s), now at v%d",
            rolled_back,
            self.get_current_version(),
        )
        return rolled_back

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    # ── internals ────────────────────────────────────────────────────────

    def _apply_one(self, migration: Migration) -> None:
        logger.info("Applying migration v%d: %s", migration.version, migration.description)
        try:
            self._conn.execute("BEGIN")
            self._conn.executescript(migration.up_sql)
            self._conn.execute(
                "INSERT INTO _migrations (version, description, applied_at) VALUES (?, ?, ?)",
                (
                    migration.version,
                    migration.description,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            logger.exception("Migration v%d failed, rolled back", migration.version)
            raise

    def _rollback_one(self, migration: Migration) -> None:
        logger.info("Rolling back migration v%d: %s", migration.version, migration.description)
        try:
            self._conn.execute("BEGIN")
            self._conn.executescript(migration.down_sql)
            self._conn.execute(
                "DELETE FROM _migrations WHERE version = ?",
                (migration.version,),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            logger.exception("Rollback of v%d failed", migration.version)
            raise


# ── Built-in migration v1: core ARGUS schema (SQLite-compatible) ─────────

MIGRATION_V1 = Migration(
    version=1,
    description="Core ARGUS schema — trades, audit_events, fills, regime_history",
    up_sql="""\
-- trades (monitoring/trade_ledger.py)
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    order_id    TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    side        TEXT    NOT NULL,
    exchange    TEXT,
    size        REAL    NOT NULL,
    price       REAL    NOT NULL,
    status      TEXT    NOT NULL,
    commission  REAL,
    slippage    REAL,
    pnl         REAL,
    value       REAL    NOT NULL,
    strategy    TEXT,
    raw_json    TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_sym_ts   ON trades (symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_order_id ON trades (order_id);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades (strategy, timestamp);

-- audit_events (monitoring/audit_trail.py)
CREATE TABLE IF NOT EXISTS audit_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    seq         INTEGER NOT NULL UNIQUE,
    ts          TEXT    NOT NULL,
    kind        TEXT    NOT NULL,
    payload_json TEXT   NOT NULL,
    event_hash  TEXT    NOT NULL,
    prev_hash   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_ts   ON audit_events (ts);
CREATE INDEX IF NOT EXISTS idx_audit_kind ON audit_events (kind, ts);
CREATE INDEX IF NOT EXISTS idx_audit_seq  ON audit_events (seq);

-- fills (execution/fill_tracker.py)
CREATE TABLE IF NOT EXISTS fills (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    order_id        TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    exchange        TEXT    NOT NULL,
    side            TEXT    NOT NULL,
    expected_price  REAL    NOT NULL,
    actual_price    REAL    NOT NULL,
    quantity        REAL    NOT NULL,
    slippage_bps    REAL    NOT NULL,
    maker           INTEGER NOT NULL DEFAULT 0,
    strategy        TEXT,
    latency_ms      REAL
);
CREATE INDEX IF NOT EXISTS idx_fills_sym_ts ON fills (symbol, timestamp);

-- regime_history (core/regime_store.py)
CREATE TABLE IF NOT EXISTS regime_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL DEFAULT (datetime('now')),
    symbol      TEXT    NOT NULL,
    regime      TEXT    NOT NULL,
    confidence  REAL,
    source      TEXT
);
CREATE INDEX IF NOT EXISTS idx_regime_sym_ts ON regime_history (symbol, timestamp);

-- journal_entries (monitoring/trade_journal.py)
CREATE TABLE IF NOT EXISTS journal_entries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    trade_id    TEXT,
    symbol      TEXT    NOT NULL,
    side        TEXT    NOT NULL,
    pnl         REAL    NOT NULL,
    strategy    TEXT,
    regime      TEXT,
    notes       TEXT,
    tags        TEXT
);
CREATE INDEX IF NOT EXISTS idx_journal_sym ON journal_entries (symbol, timestamp);

-- latency_measurements (monitoring/latency_tracker.py)
CREATE TABLE IF NOT EXISTS latency_measurements (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL DEFAULT (datetime('now')),
    operation   TEXT    NOT NULL,
    duration_ms REAL    NOT NULL,
    exchange    TEXT
);
CREATE INDEX IF NOT EXISTS idx_latency_op_ts ON latency_measurements (operation, timestamp);
""",
    down_sql="""\
DROP TABLE IF EXISTS latency_measurements;
DROP TABLE IF EXISTS journal_entries;
DROP TABLE IF EXISTS regime_history;
DROP TABLE IF EXISTS fills;
DROP TABLE IF EXISTS audit_events;
DROP TABLE IF EXISTS trades;
""",
)


def get_default_migrations() -> List[Migration]:
    """Return the built-in migration list."""
    return [MIGRATION_V1]
