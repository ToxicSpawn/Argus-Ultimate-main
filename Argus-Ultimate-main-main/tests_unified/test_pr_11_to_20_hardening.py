from __future__ import annotations

import json
import sqlite3
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import main
from execution.reason_codes import ReasonCode
from execution.state_store import ExecutionStateStore
from monitoring.soak_gate import SoakGateThresholds, evaluate_soak_gate
from monitoring.trade_ledger import TradeLedger
from unified_execution_engine import ExecutionRiskManager, KrakenDCAExecutionEngine


def _init_ledger(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
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
    cur.execute(
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
    cur.execute(
        """
        CREATE TABLE decision_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            cycle_id INTEGER,
            stage TEXT NOT NULL,
            correlation_id TEXT,
            payload_json TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE reconciliation_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            run_id TEXT NOT NULL,
            cycle_id INTEGER,
            trace_id TEXT,
            freeze_id TEXT,
            event_type TEXT NOT NULL,
            reason_code TEXT,
            details_json TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def _insert_snapshot(db_path: Path, ts: float, reason_code: str, allowed: int = 1) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO decision_snapshots
        (timestamp, run_id, trace_id, cycle_id, symbol, side, strategy, allowed, reason_code, details_json, cost_json, execution_plan_json)
        VALUES (?, 'run', 'trace', 1, 'BTC/USD', 'BUY', 's', ?, ?, '{}', '{}', '{}')
        """,
        (float(ts), int(allowed), str(reason_code)),
    )
    conn.commit()
    conn.close()


def _insert_trade(db_path: Path, ts: float) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO trades
        (timestamp, order_id, symbol, side, exchange, size, price, status, commission, slippage, pnl, value, raw_json)
        VALUES (?, 'ord-1', 'BTC/USD', 'BUY', 'kraken', 1.0, 100.0, 'filled', 0.1, 0.0, 0.0, 100.0, '{}')
        """,
        (float(ts),),
    )
    conn.commit()
    conn.close()


def _insert_portfolio_event(db_path: Path, ts: float, portfolio_value_aud: float) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO decision_events (timestamp, cycle_id, stage, correlation_id, payload_json)
        VALUES (?, 1, 'portfolio_update', 'corr', ?)
        """,
        (float(ts), json.dumps({"portfolio_value_aud": float(portfolio_value_aud)})),
    )
    conn.commit()
    conn.close()


class TestPR11SoakGate(unittest.TestCase):
    def test_soak_gate_pass(self) -> None:
        td = Path(tempfile.mkdtemp())
        db = td / "ledger.db"
        _init_ledger(db)
        start_ts = 1000.0
        for i in range(60):
            _insert_snapshot(db, ts=start_ts + i + 1, reason_code="OK", allowed=1)
        _insert_trade(db, ts=start_ts + 5)

        report = evaluate_soak_gate(
            thresholds=SoakGateThresholds(
                min_duration_seconds=30.0,
                min_decision_count=50,
                max_error_rate=0.10,
                max_timeout_rate=0.10,
                max_reconciliation_halts=0,
                max_duplicate_intents=0,
            ),
            db_path=str(db),
            start_ts=start_ts,
            end_ts=start_ts + 120.0,
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(int(report["metrics"]["decision_count"]), 60)
        self.assertEqual(int(report["metrics"]["trade_count"]), 1)

    def test_soak_gate_fail(self) -> None:
        td = Path(tempfile.mkdtemp())
        db = td / "ledger.db"
        _init_ledger(db)
        start_ts = 2000.0
        for _ in range(3):
            _insert_snapshot(db, ts=start_ts + 1, reason_code="ORDER_TIMEOUT", allowed=0)
        _insert_snapshot(db, ts=start_ts + 2, reason_code="RECONCILIATION_HALT", allowed=0)
        _insert_snapshot(db, ts=start_ts + 3, reason_code="DUPLICATE_SIGNAL", allowed=0)
        _insert_snapshot(db, ts=start_ts + 4, reason_code="INTERNAL_ERROR", allowed=0)

        report = evaluate_soak_gate(
            thresholds=SoakGateThresholds(
                min_duration_seconds=300.0,
                min_decision_count=10,
                max_error_rate=0.0,
                max_timeout_rate=0.0,
                max_reconciliation_halts=0,
                max_duplicate_intents=0,
            ),
            db_path=str(db),
            start_ts=start_ts,
            end_ts=start_ts + 30.0,
        )
        self.assertEqual(report["status"], "FAIL")
        reasons = " | ".join(report.get("fail_reasons") or [])
        self.assertIn("min_duration_seconds", reasons)
        self.assertIn("min_decision_count", reasons)
        self.assertIn("max_timeout_rate", reasons)
        self.assertIn("max_reconciliation_halts", reasons)
        self.assertIn("max_duplicate_intents", reasons)

    def test_live_gate_enforced_by_report(self) -> None:
        td = Path(tempfile.mkdtemp())
        report_path = td / "soak_gate_latest.json"
        config_path = td / "cfg.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "config_version: 1",
                    "runtime:",
                    "  mode: live",
                    "  soak_gate:",
                    "    enabled: true",
                    f"    report_path: \"{str(report_path).replace(chr(92), '/')}\"",
                    "    max_age_hours: 24.0",
                ]
            ),
            encoding="utf-8",
        )

        fresh = {
            "status": "PASS",
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "thresholds": {
                "min_duration_seconds": 1800.0,
                "min_decision_count": 50,
                "max_error_rate": 0.05,
                "max_timeout_rate": 0.05,
                "max_reconciliation_halts": 0,
                "max_duplicate_intents": 0,
            },
            "fail_reasons": [],
        }
        report_path.write_text(json.dumps(fresh), encoding="utf-8")
        main.enforce_soak_gate_or_exit(str(config_path))

        stale_or_fail = {
            "status": "FAIL",
            "checked_at": fresh["checked_at"],
            "fail_reasons": ["test fail"],
        }
        report_path.write_text(json.dumps(stale_or_fail), encoding="utf-8")
        with self.assertRaises(SystemExit):
            main.enforce_soak_gate_or_exit(str(config_path))

    def test_live_gate_threshold_mismatch_blocks(self) -> None:
        td = Path(tempfile.mkdtemp())
        report_path = td / "soak_gate_latest.json"
        config_path = td / "cfg.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "config_version: 1",
                    "runtime:",
                    "  mode: live",
                    "  soak_gate:",
                    "    enabled: true",
                    f"    report_path: \"{str(report_path).replace(chr(92), '/')}\"",
                    "    max_age_hours: 24.0",
                    "    min_decision_count: 50",
                ]
            ),
            encoding="utf-8",
        )

        mismatched = {
            "status": "PASS",
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "thresholds": {
                "min_duration_seconds": 1800.0,
                "min_decision_count": 0,
                "max_error_rate": 0.05,
                "max_timeout_rate": 0.05,
                "max_reconciliation_halts": 0,
                "max_duplicate_intents": 0,
            },
            "fail_reasons": [],
        }
        report_path.write_text(json.dumps(mismatched), encoding="utf-8")
        with self.assertRaises(SystemExit):
            main.enforce_soak_gate_or_exit(str(config_path))

    def test_soak_gate_v2_drawdown_latency_and_trade_thresholds(self) -> None:
        td = Path(tempfile.mkdtemp())
        db = td / "ledger.db"
        _init_ledger(db)
        start_ts = 3000.0

        for i in range(70):
            _insert_snapshot(db, ts=start_ts + i + 1, reason_code="OK", allowed=1)
        for i in range(20):
            _insert_trade(db, ts=start_ts + 10 + i)

        # Drawdown from 1,020 -> 980 is ~3.92%; cycle latency is ~1,000ms.
        for i, pv in enumerate([1000.0, 1010.0, 1020.0, 980.0, 990.0, 995.0]):
            _insert_portfolio_event(db, ts=start_ts + 5 + i, portfolio_value_aud=pv)

        pass_report = evaluate_soak_gate(
            thresholds=SoakGateThresholds(
                min_duration_seconds=20.0,
                min_decision_count=50,
                min_trade_count=20,
                max_error_rate=0.10,
                max_timeout_rate=0.10,
                max_reconciliation_halts=0,
                max_duplicate_intents=0,
                max_drawdown_pct=0.05,
                max_cycle_latency_p90_ms=1500.0,
            ),
            db_path=str(db),
            start_ts=start_ts,
            end_ts=start_ts + 80.0,
        )
        self.assertEqual(pass_report["status"], "PASS")

        fail_report = evaluate_soak_gate(
            thresholds=SoakGateThresholds(
                min_duration_seconds=20.0,
                min_decision_count=50,
                min_trade_count=20,
                max_error_rate=0.10,
                max_timeout_rate=0.10,
                max_reconciliation_halts=0,
                max_duplicate_intents=0,
                max_drawdown_pct=0.02,
                max_cycle_latency_p90_ms=900.0,
            ),
            db_path=str(db),
            start_ts=start_ts,
            end_ts=start_ts + 80.0,
        )
        self.assertEqual(fail_report["status"], "FAIL")
        reasons = " | ".join(fail_report.get("fail_reasons") or [])
        self.assertIn("max_drawdown_pct", reasons)
        self.assertIn("cycle_latency_p90_ms", reasons)


class _PR12TimeoutExchange:
    def __init__(self) -> None:
        self.calls = 0

    async def create_order(self, symbol: str, type: str, side: str, amount: float, price: float | None = None, params=None):
        self.calls += 1
        raise TimeoutError("simulated timeout")

    async def fetch_balance(self):
        return {"BTC": {"total": 0.0, "free": 0.0}}


class _PR12HealthExchange:
    def __init__(self, *, fail_balance: bool = False, fail_ticker: bool = False) -> None:
        self.fail_balance = bool(fail_balance)
        self.fail_ticker = bool(fail_ticker)
        self.order_calls = 0

    async def fetch_balance(self):
        if self.fail_balance:
            raise TimeoutError("health_probe_timeout")
        return {"BTC": {"total": 0.0, "free": 0.0}}

    async def fetch_ticker(self, _symbol: str):
        if self.fail_ticker:
            raise TimeoutError("ticker_probe_timeout")
        return {"last": 100.0, "timestamp": int(time.time() * 1000)}

    async def create_order(self, symbol: str, type: str, side: str, amount: float, price: float | None = None, params=None):
        self.order_calls += 1
        return {
            "id": f"ord_{self.order_calls}",
            "status": "filled",
            "price": price or 100.0,
            "amount": amount,
            "filled": amount,
            "symbol": symbol,
            "side": side,
        }


class TestPR12FailClosedAdapter(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_moves_intent_to_recon_required_and_reconcile_clears(self) -> None:
        td = Path(tempfile.mkdtemp())
        cfg = SimpleNamespace(
            retry_attempts=2,
            retry_delay_seconds=0.0,
            order_type="market",
            order_fill_timeout_seconds=0.0,
            execution_maker_timeout_seconds=0.0,
            exchange_call_timeout_seconds=1.0,
            primary_exchange="kraken",
            secondary_exchange="coinbase_advanced",
            _run_id="pr12_run",
            _cycle_id=1,
            kill_switch_file=str(td / "KILL_SWITCH"),
            reconciliation_small_drift_pct=0.01,
            reconciliation_halt_drift_pct=0.10,
        )
        ex = _PR12TimeoutExchange()
        engine = KrakenDCAExecutionEngine(cfg, {"kraken": ex})
        engine.state_store = ExecutionStateStore(db_path=str(td / "state.db"))
        engine.state_store.create_intent(
            intent_id="intent_pr12",
            symbol="BTC/USD",
            side="BUY",
            quantity=1.0,
            expected_price=100.0,
            exchange="kraken",
            run_id="run",
            trace_id="trace",
            client_order_id="cid_pr12",
            execution_plan={"maker_first": False},
        )
        out = await engine._place_order_with_retries(
            exchange=ex,
            exchange_name="kraken",
            symbol="BTC/USD",
            side="BUY",
            quantity=1.0,
            expected_price=100.0,
            client_order_id="cid_pr12",
            intent_id="intent_pr12",
            execution_plan={"maker_first": False, "primary_order_type": "market"},
        )
        self.assertIsNone(out)
        state = str((engine.state_store.get_intent("intent_pr12") or {}).get("state") or "")
        self.assertEqual(state, "RECON_REQUIRED")
        self.assertTrue(engine.state_store.has_recon_required_intent("BTC/USD"))

        ok, payload = await engine.reconcile_state(cycle_id=2, trace_id="trace_recon")
        self.assertTrue(ok)
        self.assertGreaterEqual(int(payload.get("recon_required_cleared", 0)), 1)
        self.assertFalse(engine.state_store.has_recon_required_intent("BTC/USD"))

    async def test_runtime_fail_closed_blocks_on_exchange_heartbeat_failure(self) -> None:
        td = Path(tempfile.mkdtemp())
        kill_switch = td / "KILL_SWITCH"
        cfg = SimpleNamespace(
            retry_attempts=1,
            retry_delay_seconds=0.0,
            order_type="market",
            order_fill_timeout_seconds=0.0,
            execution_maker_timeout_seconds=0.0,
            exchange_call_timeout_seconds=1.0,
            primary_exchange="kraken",
            secondary_exchange="coinbase_advanced",
            _run_id="pr12_fail_closed_exchange",
            _cycle_id=1,
            run_mode="live",
            kill_switch_file=str(kill_switch),
            fail_closed_runtime_enabled=True,
            fail_closed_runtime_enforce_paper=False,
            fail_closed_max_exchange_heartbeat_age_seconds=0.0,
            fail_closed_max_market_data_age_seconds=30.0,
            fail_closed_max_ticker_age_seconds=30.0,
            fail_closed_max_clock_skew_seconds=5.0,
            fail_closed_require_clock_sync=False,
            min_signal_confidence=0.1,
            max_trades_per_day=0,
            symbol_cooldown_cycles=0,
            loss_streak_cooldown_trigger=0,
            loss_streak_cooldown_cycles=0,
            max_position_size_aud=1_000_000.0,
            max_total_exposure_pct=0.95,
            aud_to_usd=1.0,
            starting_capital_aud=1000.0,
            current_equity_aud=1000.0,
            take_profit_pct=0.05,
            kraken_taker_fee=0.0026,
            coinbase_taker_fee=0.005,
            slippage_pct=0.001,
            edge_cost_gate_enabled=False,
        )
        ex = _PR12HealthExchange(fail_balance=True)
        engine = KrakenDCAExecutionEngine(cfg, {"kraken": ex})
        engine.trade_ledger = TradeLedger(db_path=str(td / "ledger.db"))
        engine.state_store = ExecutionStateStore(db_path=str(td / "state.db"))
        engine.risk_manager = ExecutionRiskManager(cfg)

        sig = SimpleNamespace(symbol="BTC/USD", action="BUY", confidence=1.0, quantity=0.01, entry_price=100.0, strategy="pr12_fc")
        out = await engine.execute_signals([sig], correlation_id="corr_fc_exchange")
        self.assertEqual(out, [])
        self.assertEqual(ex.order_calls, 0)
        self.assertTrue(kill_switch.exists())
        rows = engine.trade_ledger.get_decision_snapshots(limit=20)
        self.assertTrue(any(str(r.get("reason_code", "")) == ReasonCode.EXCHANGE_HEARTBEAT_STALE.value for r in rows))

    async def test_runtime_fail_closed_blocks_on_market_data_probe_failure(self) -> None:
        td = Path(tempfile.mkdtemp())
        kill_switch = td / "KILL_SWITCH"
        cfg = SimpleNamespace(
            retry_attempts=1,
            retry_delay_seconds=0.0,
            order_type="market",
            order_fill_timeout_seconds=0.0,
            execution_maker_timeout_seconds=0.0,
            exchange_call_timeout_seconds=1.0,
            primary_exchange="kraken",
            secondary_exchange="coinbase_advanced",
            _run_id="pr12_fail_closed_md",
            _cycle_id=1,
            run_mode="live",
            kill_switch_file=str(kill_switch),
            fail_closed_runtime_enabled=True,
            fail_closed_runtime_enforce_paper=False,
            fail_closed_max_exchange_heartbeat_age_seconds=30.0,
            fail_closed_max_market_data_age_seconds=0.0,
            fail_closed_max_ticker_age_seconds=30.0,
            fail_closed_max_clock_skew_seconds=5.0,
            fail_closed_require_clock_sync=False,
            min_signal_confidence=0.1,
            max_trades_per_day=0,
            symbol_cooldown_cycles=0,
            loss_streak_cooldown_trigger=0,
            loss_streak_cooldown_cycles=0,
            max_position_size_aud=1_000_000.0,
            max_total_exposure_pct=0.95,
            aud_to_usd=1.0,
            starting_capital_aud=1000.0,
            current_equity_aud=1000.0,
            take_profit_pct=0.05,
            kraken_taker_fee=0.0026,
            coinbase_taker_fee=0.005,
            slippage_pct=0.001,
            edge_cost_gate_enabled=False,
        )
        ex = _PR12HealthExchange(fail_ticker=True)
        engine = KrakenDCAExecutionEngine(cfg, {"kraken": ex})
        engine.trade_ledger = TradeLedger(db_path=str(td / "ledger.db"))
        engine.state_store = ExecutionStateStore(db_path=str(td / "state.db"))
        engine.risk_manager = ExecutionRiskManager(cfg)

        sig = SimpleNamespace(symbol="BTC/USD", action="BUY", confidence=1.0, quantity=0.01, entry_price=100.0, strategy="pr12_fc_md")
        out = await engine.execute_signals([sig], correlation_id="corr_fc_md")
        self.assertEqual(out, [])
        self.assertEqual(ex.order_calls, 0)
        self.assertTrue(kill_switch.exists())
        rows = engine.trade_ledger.get_decision_snapshots(limit=20)
        self.assertTrue(any(str(r.get("reason_code", "")) == ReasonCode.MARKET_DATA_STALE.value for r in rows))

    async def test_runtime_fail_closed_blocks_on_clock_skew(self) -> None:
        td = Path(tempfile.mkdtemp())
        kill_switch = td / "KILL_SWITCH"
        cfg = SimpleNamespace(
            retry_attempts=1,
            retry_delay_seconds=0.0,
            order_type="market",
            order_fill_timeout_seconds=0.0,
            execution_maker_timeout_seconds=0.0,
            exchange_call_timeout_seconds=1.0,
            primary_exchange="kraken",
            secondary_exchange="coinbase_advanced",
            _run_id="pr12_fail_closed_clock",
            _cycle_id=1,
            run_mode="live",
            kill_switch_file=str(kill_switch),
            fail_closed_runtime_enabled=True,
            fail_closed_runtime_enforce_paper=False,
            fail_closed_max_exchange_heartbeat_age_seconds=30.0,
            fail_closed_max_market_data_age_seconds=30.0,
            fail_closed_max_ticker_age_seconds=30.0,
            fail_closed_max_clock_skew_seconds=2.0,
            fail_closed_require_clock_sync=True,
            _clock_skew_seconds=9.5,
            min_signal_confidence=0.1,
            max_trades_per_day=0,
            symbol_cooldown_cycles=0,
            loss_streak_cooldown_trigger=0,
            loss_streak_cooldown_cycles=0,
            max_position_size_aud=1_000_000.0,
            max_total_exposure_pct=0.95,
            aud_to_usd=1.0,
            starting_capital_aud=1000.0,
            current_equity_aud=1000.0,
            take_profit_pct=0.05,
            kraken_taker_fee=0.0026,
            coinbase_taker_fee=0.005,
            slippage_pct=0.001,
            edge_cost_gate_enabled=False,
        )
        ex = _PR12HealthExchange()
        engine = KrakenDCAExecutionEngine(cfg, {"kraken": ex})
        engine.trade_ledger = TradeLedger(db_path=str(td / "ledger.db"))
        engine.state_store = ExecutionStateStore(db_path=str(td / "state.db"))
        engine.risk_manager = ExecutionRiskManager(cfg)

        sig = SimpleNamespace(symbol="BTC/USD", action="BUY", confidence=1.0, quantity=0.01, entry_price=100.0, strategy="pr12_fc_clock")
        out = await engine.execute_signals([sig], correlation_id="corr_fc_clock")
        self.assertEqual(out, [])
        self.assertEqual(ex.order_calls, 0)
        self.assertTrue(kill_switch.exists())
        rows = engine.trade_ledger.get_decision_snapshots(limit=20)
        self.assertTrue(any(str(r.get("reason_code", "")) == ReasonCode.CLOCK_SKEW_EXCEEDED.value for r in rows))


class _PR13ReconExchange:
    def __init__(self) -> None:
        self.cancelled = 0
        self.order_calls = 0
        self.balance_total = 0.0

    async def fetch_balance(self):
        return {"BTC": {"total": float(self.balance_total), "free": float(self.balance_total)}}

    async def fetch_open_orders(self, _symbol=None):
        return [{"id": "open_1", "symbol": "BTC/USD"}]

    async def cancel_order(self, _order_id: str, _symbol: str | None = None):
        self.cancelled += 1
        return {"id": _order_id, "status": "canceled"}

    async def fetch_ticker(self, _symbol: str):
        return {"last": 100.0}

    async def create_order(self, symbol: str, type: str, side: str, amount: float, price: float | None = None, params=None):
        self.order_calls += 1
        return {
            "id": f"ord_{self.order_calls}",
            "status": "filled",
            "price": price or 100.0,
            "amount": amount,
            "filled": amount,
            "symbol": symbol,
            "side": side,
        }


class TestPR13ReconOwnership(unittest.IsolatedAsyncioTestCase):
    def test_reconcile_ack_command_updates_freeze_and_audits(self) -> None:
        td = Path(tempfile.mkdtemp())
        freeze_path = td / "freeze.json"
        kill_switch = td / "KILL_SWITCH"
        ledger_db = td / "ledger.db"
        cfg_path = td / "cfg.yaml"
        cfg_path.write_text(
            "\n".join(
                [
                    "config_version: 1",
                    "runtime:",
                    "  mode: live",
                    "reconciliation:",
                    "  require_operator_ack: true",
                    f"  freeze_file: \"{str(freeze_path).replace(chr(92), '/')}\"",
                    "execution_engine:",
                    "  trade_ledger:",
                    f"    db_path: \"{str(ledger_db).replace(chr(92), '/')}\"",
                ]
            ),
            encoding="utf-8",
        )
        freeze_path.write_text(
            json.dumps({"active": True, "acknowledged": False, "freeze_id": "f_ack_cli"}),
            encoding="utf-8",
        )
        kill_switch.touch()
        payload = main.run_reconcile_ack(
            config_file=str(cfg_path),
            operator_id="oncall_cli",
            note="resolved via cli",
            yes_ack=True,
        )
        self.assertTrue(bool(payload.get("acknowledged", False)))
        self.assertFalse(kill_switch.exists())
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        self.assertFalse(bool(freeze.get("active", True)))
        self.assertTrue(bool(freeze.get("acknowledged", False)))
        self.assertEqual(str(freeze.get("ack_operator_id", "")), "oncall_cli")

        ledger = TradeLedger(db_path=str(ledger_db))
        events = ledger.get_reconciliation_events(limit=20, freeze_id="f_ack_cli")
        self.assertTrue(any(str(e.get("event_type", "")) == "ACKNOWLEDGED" for e in events))

    def test_reconcile_ack_command_requires_exact_yes(self) -> None:
        td = Path(tempfile.mkdtemp())
        freeze_path = td / "freeze.json"
        cfg_path = td / "cfg.yaml"
        cfg_path.write_text(
            "\n".join(
                [
                    "config_version: 1",
                    "runtime:",
                    "  mode: live",
                    "reconciliation:",
                    "  require_operator_ack: true",
                    f"  freeze_file: \"{str(freeze_path).replace(chr(92), '/')}\"",
                ]
            ),
            encoding="utf-8",
        )
        freeze_path.write_text(
            json.dumps({"active": True, "acknowledged": False, "freeze_id": "f_ack_token"}),
            encoding="utf-8",
        )
        with mock.patch("builtins.input", return_value="yes"):
            with self.assertRaises(SystemExit):
                main.run_reconcile_ack(
                    config_file=str(cfg_path),
                    operator_id="oncall_cli",
                    note="bad token",
                    yes_ack=False,
                )
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        self.assertFalse(bool(freeze.get("acknowledged", False)))

    def test_live_gate_blocks_unacknowledged_freeze(self) -> None:
        td = Path(tempfile.mkdtemp())
        freeze_path = td / "freeze.json"
        cfg_path = td / "cfg.yaml"
        cfg_path.write_text(
            "\n".join(
                [
                    "config_version: 1",
                    "runtime:",
                    "  mode: live",
                    "reconciliation:",
                    "  require_operator_ack: true",
                    f"  freeze_file: \"{str(freeze_path).replace(chr(92), '/')}\"",
                ]
            ),
            encoding="utf-8",
        )
        freeze_path.write_text(
            json.dumps({"active": True, "acknowledged": False, "freeze_id": "f1"}),
            encoding="utf-8",
        )
        with self.assertRaises(SystemExit):
            main.enforce_reconciliation_ack_or_exit(str(cfg_path))

        freeze_path.write_text(
            json.dumps({"active": False, "acknowledged": True, "freeze_id": "f1"}),
            encoding="utf-8",
        )
        main.enforce_reconciliation_ack_or_exit(str(cfg_path))

    async def test_reconciliation_freeze_requires_operator_ack(self) -> None:
        td = Path(tempfile.mkdtemp())
        kill_switch = td / "KILL_SWITCH"
        freeze_file = td / "reconciliation_freeze.json"
        ops_jsonl = td / "ops_events.jsonl"
        cfg = SimpleNamespace(
            retry_attempts=1,
            retry_delay_seconds=0.0,
            order_type="market",
            order_fill_timeout_seconds=0.0,
            execution_maker_first=False,
            execution_maker_timeout_seconds=0.0,
            execution_maker_offset_bps=1.0,
            execution_time_in_force="GTC",
            execution_twap_threshold_aud=999999.0,
            execution_twap_slices=1,
            execution_twap_duration_seconds=0.0,
            execution_adverse_spread_bps=9999.0,
            exchange_call_timeout_seconds=2.0,
            primary_exchange="kraken",
            secondary_exchange="coinbase_advanced",
            _run_id="pr13_run",
            _cycle_id=1,
            kill_switch_file=str(kill_switch),
            reconciliation_small_drift_pct=0.01,
            reconciliation_halt_drift_pct=0.05,
            reconciliation_require_operator_ack=True,
            reconciliation_freeze_file=str(freeze_file),
            ops_jsonl_path=str(ops_jsonl),
        )
        ex = _PR13ReconExchange()
        engine = KrakenDCAExecutionEngine(cfg, {"kraken": ex})
        engine.state_store = ExecutionStateStore(db_path=str(td / "state.db"))
        engine.trade_ledger = TradeLedger(db_path=str(td / "ledger.db"))
        engine.state_store.set_position("BTC/USD", quantity=1.0, avg_price=100.0, current_price=100.0)

        ok, payload = await engine.reconcile_state(cycle_id=1, trace_id="trace_pr13_halt")
        self.assertFalse(ok)
        self.assertTrue(bool(payload.get("halted")))
        self.assertTrue(bool(payload.get("drift_details")))
        self.assertEqual(ex.cancelled, 1)
        self.assertTrue(kill_switch.exists())
        self.assertTrue(freeze_file.exists())

        blocked_signal = SimpleNamespace(
            symbol="BTC/USD",
            action="BUY",
            confidence=1.0,
            quantity=0.01,
            entry_price=100.0,
            strategy="pr13_blocked",
        )
        blocked_results = await engine.execute_signals([blocked_signal], correlation_id="corr_pr13_blocked")
        self.assertEqual(blocked_results, [])
        rows = engine.trade_ledger.get_decision_snapshots(limit=20)
        self.assertTrue(any(str(r.get("reason_code", "")) == ReasonCode.OPERATOR_ACK_REQUIRED.value for r in rows))

        ops_text = ops_jsonl.read_text(encoding="utf-8")
        self.assertIn("reconciliation_halt_alert", ops_text)

        ack = engine.acknowledge_reconciliation_halt(operator_id="oncall_1", note="mismatch resolved", ack_token="YES")
        self.assertTrue(bool(ack.get("acknowledged", False)))
        self.assertFalse(kill_switch.exists())
        recon_events = engine.trade_ledger.get_reconciliation_events(limit=20)
        event_types = {str(ev.get("event_type", "")) for ev in recon_events}
        self.assertIn("HALT_TRIGGERED", event_types)
        self.assertIn("ACKNOWLEDGED", event_types)

        engine.state_store.set_position("BTC/USD", quantity=0.0, avg_price=100.0, current_price=100.0)
        allowed_signal = SimpleNamespace(
            symbol="BTC/USD",
            action="BUY",
            confidence=1.0,
            quantity=0.02,
            entry_price=100.0,
            strategy="pr13_allowed",
        )
        allowed_results = await engine.execute_signals([allowed_signal], correlation_id="corr_pr13_allowed")
        self.assertTrue(len(allowed_results) >= 1)
        self.assertGreaterEqual(ex.order_calls, 1)


class TestPR14PortfolioRisk(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _base_cfg() -> SimpleNamespace:
        return SimpleNamespace(
            min_signal_confidence=0.1,
            max_trades_per_day=0,
            symbol_cooldown_cycles=0,
            loss_streak_cooldown_trigger=0,
            loss_streak_cooldown_cycles=0,
            max_position_size_aud=1_000_000.0,
            max_total_exposure_pct=0.90,
            aud_to_usd=1.0,
            starting_capital_aud=1000.0,
            current_equity_aud=1000.0,
            take_profit_pct=0.06,
            kraken_taker_fee=0.0026,
            coinbase_taker_fee=0.005,
            slippage_pct=0.001,
            edge_cost_gate_enabled=False,
            portfolio_var_limit_pct=0.0,
            portfolio_cvar_limit_pct=0.0,
            portfolio_var_confidence=0.95,
            portfolio_var_lookback_trades=50,
            cluster_drawdown_brake_pct=0.0,
            target_cluster_cap_pct=0.4,
            target_cluster_map={},
            risk_cluster_map={},
            realized_vol_pct=0.0,
            portfolio_vol_target_pct=2.0,
            portfolio_liquidity_spread_ref_bps=20.0,
            portfolio_exposure_min_scale=0.30,
            run_mode="paper",
            _run_id="pr14_run",
            _cycle_id=1,
            retry_attempts=1,
            retry_delay_seconds=0.0,
            order_type="market",
            order_fill_timeout_seconds=0.0,
            execution_maker_first=False,
            execution_maker_timeout_seconds=0.0,
            execution_maker_offset_bps=1.0,
            execution_time_in_force="GTC",
            execution_twap_threshold_aud=999999.0,
            execution_twap_slices=1,
            execution_twap_duration_seconds=0.0,
            execution_adverse_spread_bps=9999.0,
            exchange_call_timeout_seconds=2.0,
            primary_exchange="kraken",
            secondary_exchange="coinbase_advanced",
        )

    @staticmethod
    def _signal(symbol: str = "BTC/USD", qty: float = 1.0, px: float = 100.0, action: str = "BUY") -> SimpleNamespace:
        return SimpleNamespace(
            symbol=symbol,
            action=str(action),
            confidence=1.0,
            quantity=float(qty),
            entry_price=float(px),
            strategy="pr14_test",
            spread_bps=5.0,
        )

    def test_var_limit_reject(self) -> None:
        cfg = self._base_cfg()
        cfg.portfolio_var_limit_pct = 0.02
        risk = ExecutionRiskManager(cfg)
        for _ in range(50):
            risk._portfolio_returns.append(-0.05)
        allowed, reason, details, _ = risk.evaluate_signal(self._signal(), cycle_id=1)
        self.assertFalse(allowed)
        self.assertEqual(reason, ReasonCode.PORTFOLIO_VAR_REJECT)
        self.assertGreater(float(details.get("portfolio_var_pct", 0.0) or 0.0), 0.0)

    def test_cluster_drawdown_brake_reject(self) -> None:
        cfg = self._base_cfg()
        cfg.cluster_drawdown_brake_pct = 0.05
        cfg.risk_cluster_map = {"BTC/USD": "majors"}
        risk = ExecutionRiskManager(cfg)
        risk._cluster_peak_pnl["majors"] = 100.0
        risk._cluster_pnl["majors"] = -50.0
        allowed, reason, details, _ = risk.evaluate_signal(self._signal(), cycle_id=1)
        self.assertFalse(allowed)
        self.assertEqual(reason, ReasonCode.CLUSTER_DRAWDOWN_REJECT)
        self.assertEqual(str(details.get("cluster")), "majors")

    def test_vol_liquidity_dynamic_cap_reject(self) -> None:
        cfg = self._base_cfg()
        cfg.max_total_exposure_pct = 0.50
        cfg.realized_vol_pct = 8.0
        cfg.portfolio_vol_target_pct = 2.0
        cfg.portfolio_liquidity_spread_ref_bps = 20.0
        cfg.portfolio_exposure_min_scale = 0.10
        risk = ExecutionRiskManager(cfg)
        sig = self._signal(qty=2.0, px=100.0)
        sig.spread_bps = 100.0
        allowed, reason, details, _ = risk.evaluate_signal(sig, cycle_id=1)
        self.assertFalse(allowed)
        self.assertEqual(reason, ReasonCode.VOL_LIQUIDITY_CAP_REJECT)
        self.assertGreater(float(details.get("projected_exposure_aud", 0.0) or 0.0), float(details.get("dynamic_cap_aud", 0.0) or 0.0))

    def test_var_limit_does_not_block_sell_derisk(self) -> None:
        cfg = self._base_cfg()
        cfg.portfolio_var_limit_pct = 0.02
        risk = ExecutionRiskManager(cfg)
        for _ in range(50):
            risk._portfolio_returns.append(-0.05)
        allowed, reason, _, _ = risk.evaluate_signal(self._signal(action="SELL"), cycle_id=1)
        self.assertTrue(allowed)
        self.assertEqual(reason, ReasonCode.ALLOWED)

    def test_vol_liquidity_cap_does_not_block_sell_derisk(self) -> None:
        cfg = self._base_cfg()
        cfg.max_total_exposure_pct = 0.50
        cfg.realized_vol_pct = 8.0
        cfg.portfolio_vol_target_pct = 2.0
        cfg.portfolio_liquidity_spread_ref_bps = 20.0
        cfg.portfolio_exposure_min_scale = 0.10
        risk = ExecutionRiskManager(cfg)
        sig = self._signal(qty=2.0, px=100.0, action="SELL")
        sig.spread_bps = 100.0
        allowed, reason, _, _ = risk.evaluate_signal(sig, cycle_id=1)
        self.assertTrue(allowed)
        self.assertEqual(reason, ReasonCode.ALLOWED)

    async def test_reason_code_persisted_in_decision_snapshot(self) -> None:
        td = Path(tempfile.mkdtemp())
        cfg = self._base_cfg()
        cfg.portfolio_var_limit_pct = 0.02
        cfg.ops_jsonl_path = str(td / "ops_events.jsonl")
        engine = KrakenDCAExecutionEngine(cfg, {"kraken": _PR13ReconExchange()})
        engine.trade_ledger = TradeLedger(db_path=str(td / "ledger.db"))
        engine.risk_manager = ExecutionRiskManager(cfg)
        for _ in range(50):
            engine.risk_manager._portfolio_returns.append(-0.05)

        out = await engine.execute_signals([self._signal()], correlation_id="corr_pr14")
        self.assertEqual(out, [])
        rows = engine.trade_ledger.get_decision_snapshots(limit=20)
        self.assertTrue(any(str(r.get("reason_code", "")) == ReasonCode.PORTFOLIO_VAR_REJECT.value for r in rows))


class _PR15FeedbackExchange:
    def __init__(self) -> None:
        self.order_calls = 0
        self._orders: dict[str, dict] = {}

    async def fetch_balance(self):
        return {"USD": {"total": 10_000.0, "free": 10_000.0}}

    async def fetch_ticker(self, _symbol: str):
        return {"last": 100.0, "timestamp": int(time.time() * 1000)}

    async def fetch_order_book(self, _symbol: str, _limit: int = 5):
        return {"bids": [[99.99, 1.0]], "asks": [[100.01, 1.0]]}

    async def create_order(self, symbol: str, type: str, side: str, amount: float, price: float | None = None, params=None):
        self.order_calls += 1
        order_id = f"pr15_{self.order_calls}"
        t = str(type or "").lower()
        if t == "limit":
            row = {
                "id": order_id,
                "status": "open",
                "price": float(price or 100.0),
                "amount": float(amount),
                "filled": 0.0,
                "symbol": symbol,
                "side": side,
            }
            self._orders[order_id] = dict(row)
            return row
        row = {
            "id": order_id,
            "status": "filled",
            "price": 105.0,
            "amount": float(amount),
            "filled": float(amount),
            "symbol": symbol,
            "side": side,
        }
        self._orders[order_id] = dict(row)
        return row

    async def fetch_order(self, order_id: str, _symbol: str):
        return dict(self._orders.get(str(order_id), {"id": str(order_id), "status": "unknown"}))

    async def cancel_order(self, order_id: str, _symbol: str | None = None):
        row = dict(self._orders.get(str(order_id), {"id": str(order_id)}))
        row["status"] = "canceled"
        row["filled"] = float(row.get("filled") or 0.0)
        row["amount"] = float(row.get("amount") or 0.0)
        self._orders[str(order_id)] = dict(row)
        return row


class TestPR15ExecutionFeedback(unittest.IsolatedAsyncioTestCase):
    async def test_poor_maker_quality_increases_fallback_aggressiveness(self) -> None:
        td = Path(tempfile.mkdtemp())
        cfg = TestPR14PortfolioRisk._base_cfg()
        cfg._run_id = "pr15_run"
        cfg.execution_maker_first = True
        cfg.execution_maker_timeout_seconds = 4.0
        cfg.execution_maker_offset_bps = 1.0
        cfg.execution_twap_threshold_aud = 999999.0
        cfg.execution_twap_slices = 2
        cfg.execution_twap_duration_seconds = 0.0
        cfg.execution_adverse_spread_bps = 9999.0
        cfg.execution_quality_feedback_enabled = True
        cfg.execution_quality_lookback = 20
        cfg.execution_quality_min_samples = 1
        cfg.execution_quality_target_slippage_bps = 10.0
        cfg.execution_quality_adverse_rate_threshold = 0.20
        cfg.execution_quality_disable_maker_adverse_rate = 0.95
        cfg.execution_quality_timeout_adjust_down_pct = 0.50
        cfg.execution_quality_timeout_adjust_up_pct = 0.10
        cfg.execution_quality_min_timeout_scale = 0.20
        cfg.execution_quality_max_timeout_scale = 1.50
        cfg.execution_quality_max_slice_multiplier = 4
        cfg.max_slippage_pct = 0.20
        cfg.ops_jsonl_path = str(td / "ops_events.jsonl")

        ex = _PR15FeedbackExchange()
        engine = KrakenDCAExecutionEngine(cfg, {"kraken": ex})
        engine.trade_ledger = TradeLedger(db_path=str(td / "ledger.db"))
        engine.state_store = ExecutionStateStore(db_path=str(td / "state.db"))
        engine.risk_manager = ExecutionRiskManager(cfg)

        sig1 = SimpleNamespace(
            symbol="BTC/USD",
            action="BUY",
            confidence=1.0,
            quantity=0.05,
            entry_price=100.0,
            strategy="pr15_quality",
        )
        sig2 = SimpleNamespace(
            symbol="BTC/USD",
            action="BUY",
            confidence=0.9999,
            quantity=0.05,
            entry_price=100.0,
            strategy="pr15_quality",
        )

        out1 = await engine.execute_signals([sig1], correlation_id="corr_pr15_1")
        out2 = await engine.execute_signals([sig2], correlation_id="corr_pr15_2")
        self.assertEqual(len(out1), 1)
        self.assertEqual(len(out2), 1)

        rows = engine.trade_ledger.get_decision_snapshots(limit=50)
        plans: list[dict] = []
        for row in sorted(rows, key=lambda r: float(r.get("timestamp", 0.0) or 0.0)):
            if int(row.get("allowed", 0) or 0) != 1:
                continue
            if str(row.get("strategy", "")) != "pr15_quality":
                continue
            raw = str(row.get("execution_plan_json", "") or "{}")
            try:
                plans.append(json.loads(raw))
            except Exception:
                plans.append({})
        self.assertGreaterEqual(len(plans), 2)

        first_timeout = float((plans[0].get("fallback") or {}).get("after_seconds", 0.0) or 0.0)
        second_timeout = float((plans[1].get("fallback") or {}).get("after_seconds", 0.0) or 0.0)
        self.assertLess(second_timeout, first_timeout)
        second_feedback = (plans[1].get("feedback") or {}).get("tuned_params") or {}
        self.assertLess(float(second_feedback.get("maker_timeout_seconds", second_timeout) or second_timeout), first_timeout)

        events = engine.trade_ledger.get_events(limit=100)
        tuning_events = [e for e in events if str(e.get("stage", "")) == "execution_quality_tuning"]
        self.assertTrue(tuning_events)

        conn = sqlite3.connect(str(td / "state.db"))
        cur = conn.cursor()
        cur.execute(
            "SELECT execution_plan_json FROM order_intents WHERE run_id = ? ORDER BY created_ts ASC",
            ("pr15_run",),
        )
        intent_plan_rows = [str(r[0] or "{}") for r in cur.fetchall()]
        conn.close()
        self.assertGreaterEqual(len(intent_plan_rows), 2)
        first_intent_plan = json.loads(intent_plan_rows[0])
        second_intent_plan = json.loads(intent_plan_rows[1])
        first_intent_timeout = float((first_intent_plan.get("fallback") or {}).get("after_seconds", 0.0) or 0.0)
        second_intent_timeout = float((second_intent_plan.get("fallback") or {}).get("after_seconds", 0.0) or 0.0)
        self.assertLess(second_intent_timeout, first_intent_timeout)


if __name__ == "__main__":
    unittest.main()
