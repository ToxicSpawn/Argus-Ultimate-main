from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace

from execution.reason_codes import ReasonCode
from execution.state_store import ExecutionStateStore
from monitoring.trade_ledger import TradeLedger
from unified_execution_engine import ExecutionRiskManager, KrakenDCAExecutionEngine


class _Signal(SimpleNamespace):
    pass


class _TimeoutThenMaybeFillExchange:
    def __init__(self) -> None:
        self.calls = 0

    async def create_order(self, symbol: str, type: str, side: str, amount: float, price: float | None = None, params=None):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("simulated timeout after send")
        return {
            "id": f"ord_{self.calls}",
            "status": "filled",
            "price": price or 100.0,
            "amount": amount,
            "filled": amount,
            "symbol": symbol,
            "side": side,
        }


class _MakerFallbackExchange:
    def __init__(self) -> None:
        self.order_types: list[str] = []

    async def create_order(self, symbol: str, type: str, side: str, amount: float, price: float | None = None, params=None):
        self.order_types.append(type)
        if type == "limit":
            return {"id": "maker_1", "status": "open", "price": price, "amount": amount, "filled": 0.0}
        return {"id": "taker_1", "status": "filled", "price": price or 100.0, "amount": amount, "filled": amount}


class _ReconExchange:
    def __init__(self) -> None:
        self.cancelled = 0

    async def fetch_balance(self):
        return {"BTC": {"total": 0.0, "free": 0.0}}

    async def fetch_open_orders(self, _symbol=None):
        return [{"id": "open_1", "symbol": "BTC/USD"}]

    async def cancel_order(self, _order_id: str, _symbol: str | None = None):
        self.cancelled += 1
        return {"id": _order_id, "status": "canceled"}


class _OpenOrderDriftExchange:
    def __init__(self) -> None:
        self.cancelled = 0

    async def fetch_balance(self):
        return {"BTC": {"total": 0.0, "free": 0.0}}

    async def fetch_open_orders(self, _symbol=None):
        # Exchange truth: no open orders, while internal state may still have one.
        return []

    async def cancel_order(self, _order_id: str, _symbol: str | None = None):
        self.cancelled += 1
        return {"id": _order_id, "status": "canceled"}


class _AmbiguousThenFillExchange:
    def __init__(self) -> None:
        self.calls = 0

    async def fetch_ticker(self, _symbol: str):
        return {"last": 100.0}

    async def create_order(self, symbol: str, type: str, side: str, amount: float, price: float | None = None, params=None):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("simulated timeout")
        return {
            "id": f"ord_{self.calls}",
            "status": "filled",
            "price": price or 100.0,
            "amount": amount,
            "filled": amount,
            "symbol": symbol,
            "side": side,
        }

    async def fetch_balance(self):
        return {"BTC": {"total": 0.0, "free": 0.0}}


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
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestPR01ToPR06ExecutionControls(unittest.IsolatedAsyncioTestCase):
    async def test_intent_id_is_deterministic_across_trace_ids(self) -> None:
        cfg = _base_cfg()
        engine = KrakenDCAExecutionEngine(cfg, {"kraken": _MakerFallbackExchange()})
        signal = _Signal(symbol="BTC/USD", action="BUY", confidence=0.8, quantity=0.01, entry_price=100.0, strategy="s1")
        a = engine._intent_id(signal, quantity=0.01, expected_price=100.0, trace_id="trace_a")
        b = engine._intent_id(signal, quantity=0.01, expected_price=100.0, trace_id="trace_b")
        self.assertEqual(a, b)

    async def test_timeout_retry_produces_single_send_and_single_intent(self) -> None:
        td = tempfile.mkdtemp()
        cfg = _base_cfg()
        ex = _TimeoutThenMaybeFillExchange()
        engine = KrakenDCAExecutionEngine(cfg, {"kraken": ex})
        engine.state_store = ExecutionStateStore(db_path=os.path.join(td, "state.db"))
        engine.state_store.create_intent(
            intent_id="intent_timeout",
            symbol="BTC/USD",
            side="BUY",
            quantity=1.0,
            expected_price=100.0,
            exchange="kraken",
            run_id="run",
            trace_id="trace",
            client_order_id="cid_1",
            execution_plan={"maker_first": False},
        )
        out = await engine._place_order_with_retries(
            exchange=ex,
            exchange_name="kraken",
            symbol="BTC/USD",
            side="BUY",
            quantity=1.0,
            expected_price=100.0,
            client_order_id="cid_1",
            intent_id="intent_timeout",
            execution_plan={"maker_first": False, "primary_order_type": "market"},
        )
        self.assertIsNone(out)
        self.assertEqual(ex.calls, 1)
        self.assertEqual(engine.state_store.count_intents("intent_timeout"), 1)
        state = (engine.state_store.get_intent("intent_timeout") or {}).get("state")
        self.assertEqual(state, "RECON_REQUIRED")
        # Retry-safe: second execution attempt must consult intent state first and avoid re-send.
        out2 = await engine._place_order_with_retries(
            exchange=ex,
            exchange_name="kraken",
            symbol="BTC/USD",
            side="BUY",
            quantity=1.0,
            expected_price=100.0,
            client_order_id="cid_1",
            intent_id="intent_timeout",
            execution_plan={"maker_first": False, "primary_order_type": "market"},
        )
        self.assertIsNone(out2)
        self.assertEqual(ex.calls, 1)
        transitions = engine.state_store.get_intent_transitions("intent_timeout")
        self.assertEqual([t.get("to_state") for t in transitions], ["CREATED", "SENT", "UNKNOWN", "RECON_REQUIRED"])

    async def test_reconciliation_halts_and_cancels(self) -> None:
        td = tempfile.mkdtemp()
        kill_switch = os.path.join(td, "KILL_SWITCH")
        cfg = _base_cfg(kill_switch_file=kill_switch, reconciliation_small_drift_pct=0.01, reconciliation_halt_drift_pct=0.05)
        ex = _ReconExchange()
        engine = KrakenDCAExecutionEngine(cfg, {"kraken": ex})
        engine.state_store = ExecutionStateStore(db_path=os.path.join(td, "state.db"))
        engine.state_store.set_position("BTC/USD", quantity=1.0, avg_price=100.0, current_price=100.0)
        ok, payload = await engine.reconcile_state(cycle_id=1, trace_id="trace_r")
        self.assertFalse(ok)
        self.assertTrue(payload.get("halted"))
        self.assertTrue(engine._reconciliation_halted)
        self.assertEqual(ex.cancelled, 1)
        self.assertTrue(os.path.exists(kill_switch))

    async def test_reconciliation_open_order_mismatch_halts_and_logs_reason(self) -> None:
        td = tempfile.mkdtemp()
        kill_switch = os.path.join(td, "KILL_SWITCH")
        cfg = _base_cfg(kill_switch_file=kill_switch, reconciliation_small_drift_pct=0.01, reconciliation_halt_drift_pct=0.05)
        ex = _OpenOrderDriftExchange()
        engine = KrakenDCAExecutionEngine(cfg, {"kraken": ex})
        engine.state_store = ExecutionStateStore(db_path=os.path.join(td, "state.db"))
        engine.trade_ledger = TradeLedger(db_path=os.path.join(td, "trades.db"))
        engine.state_store.upsert_order(
            {
                "timestamp": 1.0,
                "order_id": "stale_open_1",
                "client_order_id": "cid_stale_1",
                "symbol": "BTC/USD",
                "side": "BUY",
                "exchange": "kraken",
                "type": "limit",
                "status": "open",
                "amount": 0.5,
                "filled": 0.0,
                "price": 100.0,
                "raw": {"id": "stale_open_1"},
            }
        )

        ok, payload = await engine.reconcile_state(cycle_id=2, trace_id="trace_open_drift")
        self.assertFalse(ok)
        self.assertTrue(payload.get("halted"))
        self.assertIn("open_orders drift", str(payload.get("halt_reason", "")))
        self.assertTrue(engine._reconciliation_halted)
        self.assertTrue(os.path.exists(kill_switch))

        events = engine.trade_ledger.get_reconciliation_events(limit=20)
        halt_events = [e for e in events if str(e.get("event_type", "")).upper() == "HALT_TRIGGERED"]
        self.assertTrue(halt_events)
        self.assertTrue(any(str(e.get("reason_code", "")) == ReasonCode.RECONCILIATION_HALT.value for e in halt_events))

    async def test_maker_then_taker_fallback(self) -> None:
        td = tempfile.mkdtemp()
        cfg = _base_cfg()
        ex = _MakerFallbackExchange()
        engine = KrakenDCAExecutionEngine(cfg, {"kraken": ex})
        engine.state_store = ExecutionStateStore(db_path=os.path.join(td, "state.db"))
        engine.state_store.create_intent(
            intent_id="intent_fallback",
            symbol="BTC/USD",
            side="BUY",
            quantity=1.0,
            expected_price=100.0,
            exchange="kraken",
            run_id="run",
            trace_id="trace",
            client_order_id="cid_fallback",
            execution_plan={"maker_first": True},
        )
        out = await engine._place_order_with_retries(
            exchange=ex,
            exchange_name="kraken",
            symbol="BTC/USD",
            side="BUY",
            quantity=1.0,
            expected_price=100.0,
            client_order_id="cid_fallback",
            intent_id="intent_fallback",
            execution_plan={
                "maker_first": True,
                "primary_order_type": "limit",
                "primary_price": 99.9,
                "fallback": {"to": "market", "after_seconds": 0.0},
            },
        )
        self.assertIsNotNone(out)
        self.assertEqual(ex.order_types, ["limit", "market"])

    async def test_decision_snapshot_written_for_rejected_candidate(self) -> None:
        td = tempfile.mkdtemp()
        cfg = _base_cfg(min_signal_confidence=0.9)
        engine = KrakenDCAExecutionEngine(cfg, {"kraken": _MakerFallbackExchange()})
        engine.trade_ledger = TradeLedger(db_path=os.path.join(td, "trades.db"))
        engine.risk_manager = ExecutionRiskManager(cfg)
        signal = _Signal(symbol="BTC/USD", action="BUY", confidence=0.1, quantity=0.01, entry_price=100.0, strategy="s1")
        res = await engine.execute_signals([signal], correlation_id="corr_1")
        self.assertEqual(res, [])
        rows = engine.trade_ledger.get_decision_snapshots(limit=10)
        self.assertTrue(rows)
        self.assertEqual(rows[0]["reason_code"], ReasonCode.MIN_CONFIDENCE.value)
        self.assertTrue(rows[0]["trace_id"])

    async def test_reconciliation_halt_reason_is_snapshotted(self) -> None:
        td = tempfile.mkdtemp()
        cfg = _base_cfg(min_signal_confidence=0.0)
        engine = KrakenDCAExecutionEngine(cfg, {"kraken": _MakerFallbackExchange()})
        engine.trade_ledger = TradeLedger(db_path=os.path.join(td, "trades.db"))
        engine.risk_manager = ExecutionRiskManager(cfg)
        engine._reconciliation_halted = True
        engine._reconciliation_halt_reason = "forced_test_halt"
        signal = _Signal(symbol="BTC/USD", action="BUY", confidence=1.0, quantity=0.01, entry_price=100.0, strategy="s1")
        _ = await engine.execute_signals([signal], correlation_id="corr_halt")
        rows = engine.trade_ledger.get_decision_snapshots(limit=10)
        self.assertTrue(rows)
        self.assertEqual(rows[0]["reason_code"], ReasonCode.RECONCILIATION_HALT.value)

    async def test_recon_required_lock_blocks_then_clears_after_reconcile(self) -> None:
        td = tempfile.mkdtemp()
        cfg = _base_cfg(min_signal_confidence=0.0)
        ex = _AmbiguousThenFillExchange()
        engine = KrakenDCAExecutionEngine(cfg, {"kraken": ex})
        engine.state_store = ExecutionStateStore(db_path=os.path.join(td, "state.db"))
        engine.trade_ledger = TradeLedger(db_path=os.path.join(td, "trades.db"))
        engine.risk_manager = ExecutionRiskManager(cfg)

        sig1 = _Signal(symbol="BTC/USD", action="BUY", confidence=1.0, quantity=0.01, entry_price=100.0, strategy="s1")
        res1 = await engine.execute_signals([sig1], correlation_id="corr_amb_1")
        self.assertEqual(res1, [])
        self.assertEqual(ex.calls, 1)
        self.assertTrue(engine.state_store.has_recon_required_intent("BTC/USD"))

        sig2 = _Signal(symbol="BTC/USD", action="BUY", confidence=1.0, quantity=0.02, entry_price=100.0, strategy="s2")
        res2 = await engine.execute_signals([sig2], correlation_id="corr_amb_2")
        self.assertEqual(res2, [])
        self.assertEqual(ex.calls, 1)
        rows = engine.trade_ledger.get_decision_snapshots(limit=10)
        self.assertTrue(any(r["reason_code"] == ReasonCode.RECON_REQUIRED_LOCK.value for r in rows))

        ok, payload = await engine.reconcile_state(cycle_id=3, trace_id="recon_clear")
        self.assertTrue(ok)
        self.assertGreaterEqual(int(payload.get("recon_required_cleared", 0)), 1)
        self.assertFalse(engine.state_store.has_recon_required_intent("BTC/USD"))

        sig3 = _Signal(symbol="BTC/USD", action="BUY", confidence=1.0, quantity=0.03, entry_price=100.0, strategy="s3")
        _ = await engine.execute_signals([sig3], correlation_id="corr_amb_3")
        self.assertEqual(ex.calls, 2)


class TestThrottleReasonCodes(unittest.TestCase):
    def test_trade_limit_reason_code(self) -> None:
        cfg = _base_cfg(max_trades_per_day=1)
        rm = ExecutionRiskManager(cfg)
        rm.record_execution({"status": "filled", "commission": 1.0, "symbol": "BTC/USD", "pnl": 1.0}, cycle_id=1)
        sig = _Signal(symbol="BTC/USD", action="BUY", confidence=1.0, quantity=0.01, entry_price=100.0)
        allowed, reason, details, _cost = rm.evaluate_signal(sig, cycle_id=2)
        self.assertFalse(allowed)
        self.assertEqual(reason, ReasonCode.TRADE_LIMIT_REACHED)
        self.assertIn("max_trades_per_day", details)

    def test_net_edge_gate_reason_code(self) -> None:
        cfg = _base_cfg(min_net_edge_bps=200.0, take_profit_pct=0.005, slippage_pct=0.002, kraken_taker_fee=0.003, coinbase_taker_fee=0.003)
        rm = ExecutionRiskManager(cfg)
        sig = _Signal(symbol="BTC/USD", action="BUY", confidence=1.0, quantity=0.01, entry_price=100.0)
        allowed, reason, _details, cost = rm.evaluate_signal(sig, cycle_id=1)
        self.assertFalse(allowed)
        self.assertEqual(reason, ReasonCode.EDGE_COST_REJECT)
        self.assertIn("net_edge_bps", cost)

    def test_symbol_cooldown_reason_code(self) -> None:
        cfg = _base_cfg(symbol_cooldown_cycles=3)
        rm = ExecutionRiskManager(cfg)
        rm._last_trade_cycle_by_symbol["BTC/USD"] = 5
        sig = _Signal(symbol="BTC/USD", action="BUY", confidence=1.0, quantity=0.01, entry_price=100.0)
        allowed, reason, details, _cost = rm.evaluate_signal(sig, cycle_id=6)
        self.assertFalse(allowed)
        self.assertEqual(reason, ReasonCode.SYMBOL_COOLDOWN)
        self.assertEqual(details.get("symbol"), "BTC/USD")

    def test_fee_budget_reason_code(self) -> None:
        cfg = _base_cfg(
            max_fees_equity_pct=0.01,
            current_equity_aud=1000.0,
            kraken_taker_fee=0.005,
            coinbase_taker_fee=0.005,
        )
        rm = ExecutionRiskManager(cfg)
        sig = _Signal(symbol="BTC/USD", action="BUY", confidence=1.0, quantity=1.0, entry_price=100.0)
        allowed, reason, details, _cost = rm.evaluate_signal(sig, cycle_id=1)
        self.assertFalse(allowed)
        self.assertEqual(reason, ReasonCode.FEE_BUDGET_EXCEEDED)
        self.assertIn("fee_cap_aud", details)

    def test_loss_streak_cooldown_reason_code(self) -> None:
        cfg = _base_cfg(
            loss_streak_cooldown_trigger=2,
            loss_streak_cooldown_cycles=5,
        )
        rm = ExecutionRiskManager(cfg)
        rm._loss_streak = 2
        rm._loss_streak_cooldown_until = 10
        sig = _Signal(symbol="BTC/USD", action="BUY", confidence=1.0, quantity=0.01, entry_price=100.0)
        allowed, reason, details, _cost = rm.evaluate_signal(sig, cycle_id=7)
        self.assertFalse(allowed)
        self.assertEqual(reason, ReasonCode.LOSS_STREAK_COOLDOWN)
        self.assertIn("cooldown_until_cycle", details)
