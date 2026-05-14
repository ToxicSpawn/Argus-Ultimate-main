from __future__ import annotations

import logging
import sqlite3
import tempfile
import unittest
from pathlib import Path

from metrics.system_health_metrics import SystemHealthMetricsCollector
from unified_trading_system import OmegaSQLiteStore, UnifiedConfig, UnifiedSystemArchitecture


class TestSystemHealthMetricsCollector(unittest.IsolatedAsyncioTestCase):
    async def test_metrics_collection_and_snapshot(self) -> None:
        collector = SystemHealthMetricsCollector(enabled=True, snapshot_interval_cycles=2)
        logger = logging.getLogger("argus.tests.system_health")
        logger.warning("health warning test")
        logger.error("health error test")

        d0 = await collector.sample_event_loop_delay_ms()
        collector.record_cycle(cycle_latency_ms=100.0, event_loop_delay_ms=d0)
        d1 = await collector.sample_event_loop_delay_ms()
        collector.record_cycle(cycle_latency_ms=200.0, event_loop_delay_ms=d1)

        self.assertTrue(collector.should_snapshot(cycles_completed=2))
        snapshot = collector.build_snapshot(cycles_completed=2)
        self.assertEqual(int(snapshot.cycles_completed), 2)
        self.assertAlmostEqual(float(snapshot.avg_latency_ms), 150.0, places=6)
        self.assertGreaterEqual(int(snapshot.errors_last_hour), 1)
        self.assertGreaterEqual(int(snapshot.warnings_last_hour), 1)
        self.assertGreaterEqual(float(snapshot.event_loop_delay_ms), 0.0)
        self.assertGreaterEqual(float(snapshot.engine_uptime_seconds), 0.0)
        self.assertGreaterEqual(float(snapshot.memory_rss_mb), 0.0)
        self.assertGreaterEqual(float(snapshot.memory_python_mb), 0.0)

    async def test_runtime_hook_persists_snapshot_row(self) -> None:
        cfg = UnifiedConfig()
        cfg.system_health_metrics_enabled = True
        cfg.system_health_metrics_snapshot_interval_cycles = 1
        system = UnifiedSystemArchitecture(cfg)

        td = Path(tempfile.mkdtemp())
        db_path = td / "system_health_runtime.db"
        system.omega_store = OmegaSQLiteStore(str(db_path))
        system.omega_store.init_schema()

        await system._record_system_health_snapshot(cycles_completed=1, cycle_duration_seconds=0.25)

        with sqlite3.connect(db_path) as con:
            row = con.execute(
                "SELECT cycles_completed, avg_latency_ms, errors_last_hour, warnings_last_hour, event_loop_delay_ms, memory_rss_mb, memory_python_mb "
                "FROM system_health_snapshots ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(int(row[0]), 1)
        self.assertGreaterEqual(float(row[1]), 0.0)
        self.assertGreaterEqual(int(row[2]), 0)
        self.assertGreaterEqual(int(row[3]), 0)
        self.assertGreaterEqual(float(row[4]), 0.0)
        self.assertGreaterEqual(float(row[5]), 0.0)
        self.assertGreaterEqual(float(row[6]), 0.0)


class TestSystemHealthMetricsStore(unittest.TestCase):
    def test_db_persistence(self) -> None:
        td = Path(tempfile.mkdtemp())
        db_path = td / "system_health_store.db"
        store = OmegaSQLiteStore(str(db_path))
        store.init_schema()

        store.record_system_health_snapshot(
            {
                "timestamp": "2026-03-08T00:00:00Z",
                "cycles_completed": 12,
                "avg_latency_ms": 42.5,
                "errors_last_hour": 1,
                "warnings_last_hour": 2,
                "event_loop_delay_ms": 0.9,
                "memory_rss_mb": 123.4,
                "memory_python_mb": 12.3,
            }
        )

        with sqlite3.connect(db_path) as con:
            row = con.execute(
                "SELECT timestamp, cycles_completed, avg_latency_ms, errors_last_hour, warnings_last_hour, event_loop_delay_ms, memory_rss_mb, memory_python_mb "
                "FROM system_health_snapshots ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(str(row[0]), "2026-03-08T00:00:00Z")
        self.assertEqual(int(row[1]), 12)
        self.assertAlmostEqual(float(row[2]), 42.5, places=6)
        self.assertEqual(int(row[3]), 1)
        self.assertEqual(int(row[4]), 2)
        self.assertAlmostEqual(float(row[5]), 0.9, places=6)
        self.assertAlmostEqual(float(row[6]), 123.4, places=6)
        self.assertAlmostEqual(float(row[7]), 12.3, places=6)
