from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from unified_trading_system import OmegaSQLiteStore


class TestOmegaStoreCompat(unittest.TestCase):
    def test_record_decision_writes_to_legacy_schema(self) -> None:
        td = Path(tempfile.mkdtemp())
        db_path = td / "legacy_snapshots.db"
        con = sqlite3.connect(str(db_path))
        con.executescript(
            """
            CREATE TABLE decision_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                run_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                cycle_id INTEGER,
                symbol TEXT,
                side TEXT,
                strategy TEXT,
                allowed INTEGER NOT NULL,
                reason_code TEXT NOT NULL,
                details_json TEXT,
                cost_json TEXT,
                execution_plan_json TEXT
            );
            """
        )
        con.commit()
        con.close()

        store = OmegaSQLiteStore(str(db_path))
        store.record_decision(
            run_id="run_1",
            trace_id="trace_1",
            cycle_id=1,
            correlation_id="corr_1",
            symbol="BTC/USD",
            strategy="momentum",
            side="BUY",
            signal_score=0.9,
            allowed=False,
            reason_code="PRE_TRADE_RISK_BLOCK",
            details={
                "spread_bps": 2.0,
                "order_book_imbalance": 0.1,
                "microprice": 100.0,
                "trade_velocity": 3.0,
                "liquidity_vacuum_flag": False,
                "adverse_selection_risk": 0.2,
            },
            cost={"net_edge_bps": 1.0},
            exec_plan={"order_type": "maker"},
        )

        con = sqlite3.connect(str(db_path))
        row = con.execute(
            "SELECT run_id, trace_id, cycle_id, symbol, reason_code, details_json, cost_json, execution_plan_json "
            "FROM decision_snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()
        con.close()
        self.assertIsNotNone(row)
        self.assertEqual(str(row[0]), "run_1")
        self.assertEqual(str(row[1]), "trace_1")
        self.assertEqual(int(row[2]), 1)
        self.assertEqual(str(row[3]), "BTC/USD")
        self.assertEqual(str(row[4]), "PRE_TRADE_RISK_BLOCK")
        details = json.loads(str(row[5] or "{}"))
        self.assertIn("spread_bps", details)
        self.assertIn("microprice", details)
        self.assertIn("adverse_selection_risk", details)
        plan = json.loads(str(row[7] or "{}"))
        self.assertEqual(str(plan.get("order_type", "")), "maker")

    def test_record_decision_infers_min_exec_plan(self) -> None:
        td = Path(tempfile.mkdtemp())
        db_path = td / "legacy_snapshots_infer.db"
        con = sqlite3.connect(str(db_path))
        con.executescript(
            """
            CREATE TABLE decision_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                run_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                cycle_id INTEGER,
                symbol TEXT,
                side TEXT,
                strategy TEXT,
                allowed INTEGER NOT NULL,
                reason_code TEXT NOT NULL,
                details_json TEXT,
                cost_json TEXT,
                execution_plan_json TEXT
            );
            """
        )
        con.commit()
        con.close()

        store = OmegaSQLiteStore(str(db_path))
        store.record_decision(
            run_id="run_2",
            trace_id="trace_2",
            cycle_id=2,
            correlation_id="corr_2",
            symbol="ETH/USD",
            strategy="target_rebalance",
            side="SELL",
            signal_score=0.4,
            allowed=False,
            reason_code="PRE_TRADE_RISK_BLOCK",
            details={"suppression_reason": "small_delta"},
            cost={},
            exec_plan={},
        )

        con = sqlite3.connect(str(db_path))
        row = con.execute(
            "SELECT execution_plan_json FROM decision_snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()
        con.close()
        self.assertIsNotNone(row)
        plan = json.loads(str(row[0] or "{}"))
        self.assertEqual(str(plan.get("order_type", "")), "none")


if __name__ == "__main__":
    unittest.main()
