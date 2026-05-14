from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _write_sample_dbs(base: Path) -> tuple[Path, Path, Path, Path]:
    trades_db = base / "unified_trades.db"
    state_db = base / "unified_state.db"
    meta_db = base / "meta_weights.db"
    omega_db = base / "unified_omega.db"

    with sqlite3.connect(str(trades_db)) as con:
        con.execute(
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
            )
            """
        )
        con.execute(
            """
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                exchange TEXT,
                size REAL NOT NULL,
                price REAL NOT NULL,
                status TEXT NOT NULL,
                commission REAL,
                slippage REAL,
                pnl REAL,
                value REAL NOT NULL,
                raw_json TEXT
            )
            """
        )
        details = json.dumps(
            {
                "symbol": "BTC/USD",
                "regime_label": "trend:mid_vol",
                "target_exposure_pct": 0.18,
                "current_exposure_pct": 0.10,
                "delta_exposure_pct": 0.08,
                "liquidity_score": 0.82,
                "liquidity_clamp_flag": False,
                "spread_bps": 3.0,
                "order_book_imbalance": 0.25,
                "microprice": 100.5,
                "trade_velocity": 12.0,
                "liquidity_vacuum_flag": False,
                "adverse_selection_risk": 0.2,
                "microstructure_bias": "up",
            },
            ensure_ascii=True,
        )
        plan = json.dumps({"order_type": "maker", "planned_order_size": 0.01}, ensure_ascii=True)
        con.execute(
            """
            INSERT INTO decision_snapshots
            (timestamp, run_id, trace_id, cycle_id, symbol, side, strategy, allowed, reason_code, details_json, cost_json, execution_plan_json)
            VALUES (strftime('%s','now'), 'run', 'trace', 1, 'BTC/USD', 'BUY', 'momentum', 1, 'ALLOWED', ?, '{}', ?)
            """,
            (details, plan),
        )
        con.execute(
            """
            INSERT INTO trades
            (timestamp, order_id, symbol, side, exchange, size, price, status, commission, slippage, pnl, value, raw_json)
            VALUES (strftime('%s','now'), 'o1', 'BTC/USD', 'BUY', 'kraken', 0.01, 100.0, 'filled', 0.1, 0.001, 0.0, 1.0, '{}')
            """
        )
        con.commit()

    with sqlite3.connect(str(state_db)) as con:
        con.execute(
            """
            CREATE TABLE order_intents (
                intent_id TEXT PRIMARY KEY,
                state TEXT NOT NULL
            )
            """
        )
        con.execute("INSERT INTO order_intents (intent_id, state) VALUES ('i1', 'FILLED')")
        con.execute("INSERT INTO order_intents (intent_id, state) VALUES ('i2', 'RECON_REQUIRED')")
        con.execute(
            """
            CREATE TABLE positions (
                symbol TEXT PRIMARY KEY,
                quantity REAL NOT NULL,
                avg_price REAL NOT NULL,
                current_price REAL NOT NULL,
                updated_ts REAL NOT NULL
            )
            """
        )
        con.execute(
            "INSERT INTO positions (symbol, quantity, avg_price, current_price, updated_ts) VALUES ('BTC/USD', 0.02, 99.0, 101.0, strftime('%s','now'))"
        )
        con.commit()

    with sqlite3.connect(str(meta_db)) as con:
        con.execute(
            """
            CREATE TABLE meta_weight_snapshots (
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
            VALUES ('run', 't1', strftime('%s','now')-60, 'trend:mid_vol', '{"momentum":0.5,"breakout":0.5}', '{}', '{}')
            """
        )
        con.execute(
            """
            INSERT INTO meta_weight_snapshots (run_id, trace_id, ts, regime_label, weights_json, reasons_json, source_metrics_json)
            VALUES ('run', 't2', strftime('%s','now'), 'trend:mid_vol', '{"momentum":0.6,"breakout":0.4}', '{}', '{}')
            """
        )
        con.commit()

    with sqlite3.connect(str(omega_db)) as con:
        con.execute(
            """
            CREATE TABLE system_health_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cycles_completed INTEGER NOT NULL,
                avg_latency_ms REAL NOT NULL,
                errors_last_hour INTEGER NOT NULL,
                warnings_last_hour INTEGER NOT NULL,
                event_loop_delay_ms REAL NOT NULL
            )
            """
        )
        con.execute(
            """
            INSERT INTO system_health_snapshots
            (timestamp, cycles_completed, avg_latency_ms, errors_last_hour, warnings_last_hour, event_loop_delay_ms)
            VALUES ('2026-03-08T00:00:00Z', 100, 120.0, 0, 1, 0.7)
            """
        )
        con.commit()

    return trades_db, state_db, meta_db, omega_db


class TestMonitoringExports(unittest.TestCase):
    def test_export_runtime_dashboard_script(self) -> None:
        td = Path(tempfile.mkdtemp(prefix="argus_dashboard_"))
        trades_db, state_db, meta_db, omega_db = _write_sample_dbs(td)
        out = td / "runtime_dashboard.json"
        cmd = [
            sys.executable,
            "scripts/export_runtime_dashboard.py",
            "--omega-db",
            str(omega_db),
            "--trades-db",
            str(trades_db),
            "--state-db",
            str(state_db),
            "--meta-db",
            str(meta_db),
            "--output",
            str(out),
        ]
        proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertIn("system_health", payload)
        self.assertIn("trading_intelligence", payload)
        self.assertIn("execution_risk", payload)

    def test_overnight_verification_script(self) -> None:
        td = Path(tempfile.mkdtemp(prefix="argus_overnight_"))
        trades_db, state_db, meta_db, omega_db = _write_sample_dbs(td)
        out = td / "overnight_verification_summary.json"
        cmd = [
            sys.executable,
            "scripts/overnight_verification.py",
            "--trades-db",
            str(trades_db),
            "--state-db",
            str(state_db),
            "--meta-db",
            str(meta_db),
            "--omega-db",
            str(omega_db),
            "--output",
            str(out),
            "--max-recon-required",
            "3",
        ]
        proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(str(payload.get("overall_status", "")), "PASS")
        self.assertTrue(isinstance(payload.get("checks"), list))


if __name__ == "__main__":
    unittest.main()
