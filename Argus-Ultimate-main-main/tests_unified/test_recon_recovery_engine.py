from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
import unittest
from types import SimpleNamespace

from execution.recon_recovery_engine import ReconRecoveryEngine
from execution.reason_codes import ReasonCode
from execution.state_store import ExecutionStateStore
from monitoring.trade_ledger import TradeLedger
from unified_execution_engine import ExecutionRiskManager, KrakenDCAExecutionEngine


class _Signal(SimpleNamespace):
    pass


class _OrderNotFoundExchange:
    async def fetch_order(self, _order_id: str, _symbol: str):
        raise Exception("order not found")


class _UncertainExchange:
    async def fetch_order(self, _order_id: str, _symbol: str):
        raise TimeoutError("exchange state uncertain")


class _NoOpExchange:
    async def create_order(self, symbol: str, type: str, side: str, amount: float, price: float | None = None, params=None):
        _ = (symbol, type, side, amount, price, params)
        return {"id": "noop_1", "status": "filled", "price": 100.0, "amount": amount, "filled": amount}


def _base_cfg(**overrides):
    cfg = SimpleNamespace(
        run_mode="paper",
        min_signal_confidence=0.0,
        max_position_size_aud=1_000_000.0,
        max_position_pct=1.0,
        max_total_exposure_pct=1.0,
        aud_to_usd=0.65,
        kraken_taker_fee=0.0026,
        coinbase_taker_fee=0.005,
        slippage_pct=0.001,
        take_profit_pct=0.03,
        retry_attempts=2,
        retry_delay_seconds=0.0,
        order_type="market",
        order_fill_timeout_seconds=0.0,
        max_slippage_pct=1.0,
        edge_cost_gate_enabled=True,
        min_net_edge_bps=0.0,
        execution_maker_first=True,
        execution_maker_timeout_seconds=0.0,
        execution_maker_offset_bps=1.0,
        execution_time_in_force="GTC",
        execution_twap_threshold_aud=80.0,
        execution_twap_slices=2,
        execution_twap_duration_seconds=5.0,
        execution_adverse_spread_bps=9999.0,
        primary_exchange="kraken",
        secondary_exchange="coinbase_advanced",
        _run_id="test_run",
        _cycle_id=1,
        kill_switch_file="KILL_SWITCH",
        reconciliation_small_drift_pct=0.01,
        reconciliation_halt_drift_pct=0.05,
        recon_recovery_enabled=True,
        recon_recovery_stale_threshold_seconds=60.0,
        recon_recovery_base_retry_delay_seconds=5.0,
        recon_recovery_max_retries=5,
        recon_recovery_halt_on_retry_exhausted=True,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _create_recon_required_intent(store: ExecutionStateStore, *, intent_id: str, symbol: str = "BTC/USD") -> None:
    store.create_intent(
        intent_id=intent_id,
        symbol=symbol,
        side="BUY",
        quantity=1.0,
        expected_price=100.0,
        exchange="kraken",
        run_id="run",
        trace_id="trace",
        client_order_id=f"cid_{intent_id}",
        execution_plan={"maker_first": False},
    )
    store.update_intent_state(
        intent_id,
        "RECON_REQUIRED",
        exchange_order_id=f"oid_{intent_id}",
        last_error="timeout_ambiguous",
    )


def _age_intent(store: ExecutionStateStore, *, intent_id: str, seconds: float) -> None:
    ts = float(time.time()) - float(seconds)
    with sqlite3.connect(store.db_path) as con:
        con.execute("UPDATE order_intents SET updated_ts = ? WHERE intent_id = ?", (ts, str(intent_id)))
        con.commit()


def _age_recovery_retry(store: ExecutionStateStore, *, intent_id: str, seconds: float) -> None:
    ts = float(time.time()) - float(seconds)
    with sqlite3.connect(store.db_path) as con:
        con.execute("UPDATE recon_recovery_state SET last_retry_ts = ? WHERE intent_id = ?", (ts, str(intent_id)))
        con.commit()


class TestReconRecoveryEngine(unittest.IsolatedAsyncioTestCase):
    async def test_stale_detection_and_clear_on_not_found(self) -> None:
        td = tempfile.mkdtemp()
        state = ExecutionStateStore(db_path=os.path.join(td, "state.db"))
        _create_recon_required_intent(state, intent_id="stale_1", symbol="BTC/USD")
        _create_recon_required_intent(state, intent_id="fresh_1", symbol="ETH/USD")
        _age_intent(state, intent_id="stale_1", seconds=120.0)
        _age_intent(state, intent_id="fresh_1", seconds=10.0)

        cfg = _base_cfg(recon_recovery_stale_threshold_seconds=60.0, recon_recovery_base_retry_delay_seconds=0.1)
        engine = ReconRecoveryEngine(cfg, state)
        summary = await engine.run_cycle(
            exchanges={"kraken": _OrderNotFoundExchange()},
            cycle_id=1,
            trace_id="t1",
            reconcile_fn=None,
        )
        self.assertEqual(int(summary.get("stale_count", 0)), 1)
        self.assertEqual(int(summary.get("cleared", 0)), 1)
        self.assertTrue(state.has_recon_required_intent("ETH/USD"))
        self.assertFalse(state.has_recon_required_intent("BTC/USD"))
        row = state.get_recon_recovery_state("stale_1") or {}
        self.assertEqual(str(row.get("recovery_status", "")), "cleared")

    async def test_retry_logic_uses_backoff(self) -> None:
        td = tempfile.mkdtemp()
        state = ExecutionStateStore(db_path=os.path.join(td, "state.db"))
        _create_recon_required_intent(state, intent_id="retry_1", symbol="BTC/USD")
        _age_intent(state, intent_id="retry_1", seconds=120.0)

        cfg = _base_cfg(
            recon_recovery_stale_threshold_seconds=60.0,
            recon_recovery_base_retry_delay_seconds=5.0,
            recon_recovery_max_retries=5,
        )
        engine = ReconRecoveryEngine(cfg, state)

        s1 = await engine.run_cycle(
            exchanges={"kraken": _UncertainExchange()},
            cycle_id=1,
            trace_id="t2",
            reconcile_fn=None,
        )
        self.assertEqual(int(s1.get("retried", 0)), 1)
        row1 = state.get_recon_recovery_state("retry_1") or {}
        self.assertEqual(int(row1.get("retry_count", 0)), 1)
        self.assertEqual(str(row1.get("recovery_status", "")), "retrying")

        s2 = await engine.run_cycle(
            exchanges={"kraken": _UncertainExchange()},
            cycle_id=2,
            trace_id="t3",
            reconcile_fn=None,
        )
        self.assertEqual(int(s2.get("retried", 0)), 0)

        _age_recovery_retry(state, intent_id="retry_1", seconds=20.0)
        s3 = await engine.run_cycle(
            exchanges={"kraken": _UncertainExchange()},
            cycle_id=3,
            trace_id="t4",
            reconcile_fn=None,
        )
        self.assertEqual(int(s3.get("retried", 0)), 1)
        row3 = state.get_recon_recovery_state("retry_1") or {}
        self.assertEqual(int(row3.get("retry_count", 0)), 2)

    async def test_escalates_after_max_retries(self) -> None:
        td = tempfile.mkdtemp()
        state = ExecutionStateStore(db_path=os.path.join(td, "state.db"))
        _create_recon_required_intent(state, intent_id="halt_1", symbol="BTC/USD")
        _age_intent(state, intent_id="halt_1", seconds=120.0)

        cfg = _base_cfg(
            recon_recovery_stale_threshold_seconds=60.0,
            recon_recovery_base_retry_delay_seconds=0.1,
            recon_recovery_max_retries=1,
        )
        engine = ReconRecoveryEngine(cfg, state)
        s1 = await engine.run_cycle(
            exchanges={"kraken": _UncertainExchange()},
            cycle_id=1,
            trace_id="t5",
            reconcile_fn=None,
        )
        self.assertFalse(bool(s1.get("halt_required")))

        _age_recovery_retry(state, intent_id="halt_1", seconds=10.0)
        s2 = await engine.run_cycle(
            exchanges={"kraken": _UncertainExchange()},
            cycle_id=2,
            trace_id="t6",
            reconcile_fn=None,
        )
        self.assertTrue(bool(s2.get("halt_required")))
        row = state.get_recon_recovery_state("halt_1") or {}
        self.assertEqual(str(row.get("recovery_status", "")), "halted")
        self.assertIn("HALT_REQUIRED", str(row.get("resolution_reason", "")))

    async def test_retry_exhausted_can_fall_back_to_operator_review(self) -> None:
        td = tempfile.mkdtemp()
        state = ExecutionStateStore(db_path=os.path.join(td, "state.db"))
        _create_recon_required_intent(state, intent_id="review_1", symbol="BTC/USD")
        _age_intent(state, intent_id="review_1", seconds=120.0)

        cfg = _base_cfg(
            recon_recovery_stale_threshold_seconds=60.0,
            recon_recovery_base_retry_delay_seconds=0.1,
            recon_recovery_max_retries=1,
            recon_recovery_halt_on_retry_exhausted=False,
        )
        engine = ReconRecoveryEngine(cfg, state)
        await engine.run_cycle(
            exchanges={"kraken": _UncertainExchange()},
            cycle_id=1,
            trace_id="t5b",
            reconcile_fn=None,
        )

        _age_recovery_retry(state, intent_id="review_1", seconds=10.0)
        s2 = await engine.run_cycle(
            exchanges={"kraken": _UncertainExchange()},
            cycle_id=2,
            trace_id="t6b",
            reconcile_fn=None,
        )
        self.assertFalse(bool(s2.get("halt_required")))
        self.assertGreaterEqual(int(s2.get("operator_review", 0)), 1)
        row = state.get_recon_recovery_state("review_1") or {}
        self.assertEqual(str(row.get("recovery_status", "")), "pending")
        self.assertIn("OPERATOR_REVIEW", str(row.get("resolution_reason", "")))

    async def test_persistence_table_rows(self) -> None:
        td = tempfile.mkdtemp()
        state = ExecutionStateStore(db_path=os.path.join(td, "state.db"))
        _create_recon_required_intent(state, intent_id="persist_1", symbol="BTC/USD")
        _age_intent(state, intent_id="persist_1", seconds=120.0)
        cfg = _base_cfg(recon_recovery_stale_threshold_seconds=60.0, recon_recovery_base_retry_delay_seconds=0.1)
        engine = ReconRecoveryEngine(cfg, state)
        await engine.run_cycle(
            exchanges={"kraken": _UncertainExchange()},
            cycle_id=1,
            trace_id="t7",
            reconcile_fn=None,
        )
        rows = state.get_recon_recovery_states(limit=10)
        self.assertTrue(rows)
        self.assertIn("intent_id", rows[0])
        self.assertIn("retry_count", rows[0])
        self.assertIn("recovery_status", rows[0])
        self.assertIn("last_retry_ts", rows[0])
        self.assertIn("resolution_reason", rows[0])
        history_rows = state.get_recon_recovery_history(intent_id="persist_1", limit=10)
        self.assertTrue(history_rows)
        self.assertIn("recovery_classification", history_rows[0])


class TestReconRecoverySnapshotIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_includes_recon_recovery_fields(self) -> None:
        td = tempfile.mkdtemp()
        cfg = _base_cfg(
            recon_recovery_stale_threshold_seconds=3600.0,
            recon_recovery_base_retry_delay_seconds=0.1,
            recon_recovery_max_retries=5,
        )
        ex = _NoOpExchange()
        engine = KrakenDCAExecutionEngine(cfg, {"kraken": ex})
        engine.state_store = ExecutionStateStore(db_path=os.path.join(td, "state.db"))
        engine.trade_ledger = TradeLedger(db_path=os.path.join(td, "trades.db"))
        engine.risk_manager = ExecutionRiskManager(cfg)

        _create_recon_required_intent(engine.state_store, intent_id="snap_1", symbol="BTC/USD")
        _age_intent(engine.state_store, intent_id="snap_1", seconds=120.0)

        sig = _Signal(symbol="BTC/USD", action="BUY", confidence=1.0, quantity=0.01, entry_price=100.0, strategy="s1")
        out = await engine.execute_signals([sig], correlation_id="corr_recon_snapshot")
        self.assertEqual(out, [])

        rows = engine.trade_ledger.get_decision_snapshots(limit=10)
        self.assertTrue(rows)
        self.assertEqual(rows[0]["reason_code"], ReasonCode.RECON_REQUIRED_LOCK.value)
        details = json.loads(str(rows[0].get("details_json") or "{}"))
        self.assertIn("recon_recovery_status", details)
        self.assertIn("recon_retry_count", details)
        self.assertIn("recon_resolution_reason", details)

        # Snapshot integration should not require recovery persistence to be written in this path.


if __name__ == "__main__":
    unittest.main()
