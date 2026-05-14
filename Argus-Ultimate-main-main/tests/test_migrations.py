"""Tests for core.migrations — SQLite migration system."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from core.migrations import (
    Migration,
    MigrationManager,
    MIGRATION_V1,
    get_default_migrations,
)


class TestMigrationDataclass(unittest.TestCase):
    """Basic Migration dataclass tests."""

    def test_fields(self):
        m = Migration(version=1, description="test", up_sql="CREATE TABLE t(id INT)", down_sql="DROP TABLE t")
        self.assertEqual(m.version, 1)
        self.assertEqual(m.description, "test")
        self.assertIn("CREATE TABLE", m.up_sql)
        self.assertIn("DROP TABLE", m.down_sql)


class TestMigrationManager(unittest.TestCase):
    """Core MigrationManager behaviour."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db_path = self.tmp.name
        self.mgr = MigrationManager(self.db_path)

    def tearDown(self):
        self.mgr.close()
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    # ── helpers ──────────────────────────────────────────────────────────

    def _make_migration(self, version: int, table_name: str = "t") -> Migration:
        return Migration(
            version=version,
            description=f"create {table_name}_{version}",
            up_sql=f"CREATE TABLE {table_name}_{version} (id INTEGER PRIMARY KEY);",
            down_sql=f"DROP TABLE IF EXISTS {table_name}_{version};",
        )

    # ── tests ────────────────────────────────────────────────────────────

    def test_initial_version_is_zero(self):
        self.assertEqual(self.mgr.get_current_version(), 0)

    def test_register_and_pending(self):
        m1 = self._make_migration(1)
        m2 = self._make_migration(2)
        self.mgr.register(m1)
        self.mgr.register(m2)
        self.assertEqual(len(self.mgr.pending()), 2)

    def test_register_duplicate_raises(self):
        self.mgr.register(self._make_migration(1))
        with self.assertRaises(ValueError):
            self.mgr.register(self._make_migration(1))

    def test_apply_all(self):
        self.mgr.register(self._make_migration(1))
        self.mgr.register(self._make_migration(2))
        applied = self.mgr.apply_all()
        self.assertEqual(applied, 2)
        self.assertEqual(self.mgr.get_current_version(), 2)
        self.assertEqual(len(self.mgr.pending()), 0)

    def test_apply_all_idempotent(self):
        self.mgr.register(self._make_migration(1))
        self.mgr.apply_all()
        second = self.mgr.apply_all()
        self.assertEqual(second, 0)

    def test_apply_creates_tables(self):
        self.mgr.register(self._make_migration(1, "widgets"))
        self.mgr.apply_all()
        # Verify table exists by inserting
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO widgets_1 (id) VALUES (42)")
        row = conn.execute("SELECT id FROM widgets_1").fetchone()
        conn.close()
        self.assertEqual(row[0], 42)

    def test_rollback_to_zero(self):
        self.mgr.register(self._make_migration(1))
        self.mgr.register(self._make_migration(2))
        self.mgr.apply_all()
        rolled = self.mgr.rollback(0)
        self.assertEqual(rolled, 2)
        self.assertEqual(self.mgr.get_current_version(), 0)

    def test_rollback_partial(self):
        self.mgr.register(self._make_migration(1))
        self.mgr.register(self._make_migration(2))
        self.mgr.register(self._make_migration(3))
        self.mgr.apply_all()
        rolled = self.mgr.rollback(1)
        self.assertEqual(rolled, 2)
        self.assertEqual(self.mgr.get_current_version(), 1)

    def test_rollback_drops_tables(self):
        self.mgr.register(self._make_migration(1, "foo"))
        self.mgr.apply_all()
        self.mgr.rollback(0)
        conn = sqlite3.connect(self.db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='foo_1'"
        ).fetchall()
        conn.close()
        self.assertEqual(tables, [])

    def test_rollback_noop_when_at_target(self):
        self.mgr.register(self._make_migration(1))
        self.mgr.apply_all()
        rolled = self.mgr.rollback(1)
        self.assertEqual(rolled, 0)
        self.assertEqual(self.mgr.get_current_version(), 1)

    def test_rollback_unregistered_raises(self):
        """Rollback fails if the migration is not in the registry."""
        self.mgr.register(self._make_migration(1))
        self.mgr.apply_all()
        # Remove from registry by creating a fresh manager with no registrations
        self.mgr.close()
        mgr2 = MigrationManager(self.db_path)
        with self.assertRaises(RuntimeError):
            mgr2.rollback(0)
        mgr2.close()

    def test_apply_bad_sql_rolls_back(self):
        bad = Migration(
            version=1,
            description="bad",
            up_sql="THIS IS NOT VALID SQL;",
            down_sql="SELECT 1;",
        )
        self.mgr.register(bad)
        with self.assertRaises(Exception):
            self.mgr.apply_all()
        # Version should still be 0
        self.assertEqual(self.mgr.get_current_version(), 0)

    def test_migrations_table_records(self):
        self.mgr.register(self._make_migration(1))
        self.mgr.apply_all()
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT version, description, applied_at FROM _migrations WHERE version=1"
        ).fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 1)
        self.assertIn("create", row[1])
        self.assertIsNotNone(row[2])  # applied_at timestamp

    def test_pending_after_partial_apply(self):
        self.mgr.register(self._make_migration(1))
        self.mgr.register(self._make_migration(2))
        self.mgr.register(self._make_migration(3))
        # Manually apply only v1
        self.mgr._apply_one(self._make_migration(1))
        pending = self.mgr.pending()
        self.assertEqual(len(pending), 2)
        self.assertEqual(pending[0].version, 2)

    def test_register_order_independent(self):
        """Migrations are sorted by version regardless of registration order."""
        self.mgr.register(self._make_migration(3))
        self.mgr.register(self._make_migration(1))
        self.mgr.register(self._make_migration(2))
        self.mgr.apply_all()
        self.assertEqual(self.mgr.get_current_version(), 3)


class TestBuiltInMigrationV1(unittest.TestCase):
    """Verify the built-in v1 migration creates the expected tables."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db_path = self.tmp.name
        self.mgr = MigrationManager(self.db_path)
        self.mgr.register(MIGRATION_V1)

    def tearDown(self):
        self.mgr.close()
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def test_v1_creates_all_tables(self):
        self.mgr.apply_all()
        conn = sqlite3.connect(self.db_path)
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        expected = {
            "trades",
            "audit_events",
            "fills",
            "regime_history",
            "journal_entries",
            "latency_measurements",
            "_migrations",
        }
        self.assertTrue(expected.issubset(tables), f"Missing: {expected - tables}")

    def test_v1_rollback_drops_tables(self):
        self.mgr.apply_all()
        self.mgr.rollback(0)
        conn = sqlite3.connect(self.db_path)
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        # Only _migrations should remain
        self.assertNotIn("trades", tables)
        self.assertNotIn("audit_events", tables)

    def test_v1_trades_insert(self):
        self.mgr.apply_all()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO trades (timestamp, order_id, symbol, side, size, price, status, value) "
            "VALUES ('2026-01-01T00:00:00Z', 'ord-1', 'BTC/AUD', 'buy', 0.01, 50000, 'filled', 500)"
        )
        row = conn.execute("SELECT symbol, side FROM trades WHERE order_id='ord-1'").fetchone()
        conn.close()
        self.assertEqual(row, ("BTC/AUD", "buy"))

    def test_v1_audit_events_insert(self):
        self.mgr.apply_all()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO audit_events (seq, ts, kind, payload_json, event_hash, prev_hash) "
            "VALUES (1, '2026-01-01', 'TRADE', '{}', 'abc', '000')"
        )
        row = conn.execute("SELECT kind FROM audit_events WHERE seq=1").fetchone()
        conn.close()
        self.assertEqual(row[0], "TRADE")


class TestGetDefaultMigrations(unittest.TestCase):
    def test_returns_list_with_v1(self):
        defaults = get_default_migrations()
        self.assertIsInstance(defaults, list)
        self.assertTrue(len(defaults) >= 1)
        self.assertEqual(defaults[0].version, 1)


if __name__ == "__main__":
    unittest.main()
