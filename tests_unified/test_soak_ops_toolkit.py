from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from execution.state_store import ExecutionStateStore
from monitoring.trade_ledger import TradeLedger


class TestSoakOpsToolkit(unittest.TestCase):
    def _make_sample_dbs(self, base: Path) -> tuple[Path, Path, Path]:
        trades_db = base / "unified_trades.db"
        state_db = base / "unified_state.db"
        meta_db = base / "meta_weights.db"

        ledger = TradeLedger(db_path=str(trades_db))
        now = float(time.time())
        ledger.record_trade(
            {
                "timestamp": now - 30.0,
                "order_id": "t1",
                "symbol": "BTC/USD",
                "side": "BUY",
                "exchange": "kraken",
                "size": 0.01,
                "price": 100.0,
                "status": "filled",
                "commission": 0.1,
                "slippage": 0.0015,
                "pnl": 0.0,
                "value": 1.0,
            }
        )
        ledger.record_decision_snapshot(
            run_id="run1",
            trace_id="trace1",
            cycle_id=1,
            symbol="BTC/USD",
            side="BUY",
            strategy="momentum",
            allowed=False,
            reason_code="RECON_REQUIRED_LOCK",
            details_json=json.dumps({"liquidity_clamp_flag": True}, ensure_ascii=True),
            cost_json="{}",
            execution_plan_json="{}",
            timestamp=now - 20.0,
        )
        ledger.record_decision_snapshot(
            run_id="run1",
            trace_id="trace2",
            cycle_id=2,
            symbol="ETH/USD",
            side="BUY",
            strategy="breakout",
            allowed=True,
            reason_code="ALLOWED",
            details_json=json.dumps({"liquidity_clamp_flag": False}, ensure_ascii=True),
            cost_json="{}",
            execution_plan_json="{}",
            timestamp=now - 10.0,
        )

        store = ExecutionStateStore(db_path=str(state_db))
        store.create_intent(
            intent_id="i1",
            symbol="BTC/USD",
            side="BUY",
            quantity=1.0,
            expected_price=100.0,
            exchange="kraken",
            run_id="run1",
            trace_id="trace1",
            client_order_id="cid1",
            execution_plan={},
        )
        store.update_intent_state("i1", "RECON_REQUIRED", last_error="timeout")
        store.upsert_recon_recovery_state(
            intent_id="i1",
            retry_count=2,
            recovery_status="retrying",
            last_retry_ts=now - 5.0,
            resolution_reason="exchange_state_uncertain:timeout",
        )

        con = sqlite3.connect(str(meta_db))
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS meta_weight_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    ts REAL NOT NULL,
                    regime_label TEXT,
                    weights_json TEXT NOT NULL,
                    reasons_json TEXT NOT NULL,
                    source_metrics_json TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                INSERT INTO meta_weight_snapshots (run_id, trace_id, ts, regime_label, weights_json, reasons_json, source_metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "run1",
                    "trace1",
                    now - 60.0,
                    "trend:mid_vol",
                    json.dumps({"momentum": 0.5, "breakout": 0.5}, ensure_ascii=True),
                    "{}",
                    "{}",
                ),
            )
            con.execute(
                """
                INSERT INTO meta_weight_snapshots (run_id, trace_id, ts, regime_label, weights_json, reasons_json, source_metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "run1",
                    "trace2",
                    now - 5.0,
                    "trend:mid_vol",
                    json.dumps({"momentum": 0.7, "breakout": 0.3}, ensure_ascii=True),
                    "{}",
                    "{}",
                ),
            )
            con.commit()
        finally:
            con.close()

        return trades_db, state_db, meta_db

    def test_generate_daily_report_script(self) -> None:
        td = Path(tempfile.mkdtemp(prefix="argus_daily_report_"))
        trades_db, state_db, meta_db = self._make_sample_dbs(td)
        out = td / "daily_runtime_summary.json"
        cmd = [
            sys.executable,
            "scripts/generate_daily_report.py",
            "--trades-db",
            str(trades_db),
            "--state-db",
            str(state_db),
            "--meta-db",
            str(meta_db),
            "--lookback-hours",
            "24",
            "--output",
            str(out),
        ]
        proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertTrue(out.exists())
        data = json.loads(out.read_text(encoding="utf-8"))
        self.assertIn("cycles_executed", data)
        self.assertIn("decisions_generated", data)
        self.assertIn("intent_count_by_state", data)
        self.assertIn("liquidity_clamp_count", data)
        self.assertIn("slippage_summary", data)
        self.assertIn("cycle_latency_ms", data)
        self.assertIn("strategy_signal_counts", data)
        self.assertIn("top_strategies_by_activity", data)
        self.assertIn("recon_required_counts", data)
        self.assertIn("meta_weight_changes", data)
        self.assertIn("emergency_stop_count", data)

    def test_db_health_check_script(self) -> None:
        td = Path(tempfile.mkdtemp(prefix="argus_db_health_"))
        trades_db, state_db, meta_db = self._make_sample_dbs(td)
        out = td / "db_health.json"
        cmd = [
            sys.executable,
            "scripts/db_health_check.py",
            "--db",
            str(trades_db),
            "--db",
            str(state_db),
            "--db",
            str(meta_db),
            "--output",
            str(out),
        ]
        proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        data = json.loads(out.read_text(encoding="utf-8"))
        self.assertIn("overall_ok", data)
        self.assertTrue(bool(data["overall_ok"]))
        self.assertIn("databases", data)
        self.assertEqual(len(data["databases"]), 3)

    def test_run_soak_powershell_once(self) -> None:
        pwsh = shutil.which("pwsh") or shutil.which("powershell")
        if not pwsh:
            self.skipTest("PowerShell not available")
        td = Path(tempfile.mkdtemp(prefix="argus_soak_ps1_"))
        script = Path(__file__).resolve().parents[1] / "scripts" / "Run-Soak.ps1"
        # One-shot lightweight command for deterministic test.
        cmd = [
            pwsh,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Once",
            "-PaperCommand",
            "py -3 -V",
            "-RestartDelaySeconds",
            "1",
        ]
        proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)


if __name__ == "__main__":
    unittest.main()
