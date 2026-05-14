from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from execution.state_store import ExecutionStateStore
from monitoring.trade_ledger import TradeLedger
from unified_execution_engine import ExecutionRiskManager, KrakenDCAExecutionEngine


class _Signal(SimpleNamespace):
    pass


class _PlanExchange:
    def __init__(self, *, bid: float = 99.98, ask: float = 100.02, maker_open: bool = False) -> None:
        self.bid = float(bid)
        self.ask = float(ask)
        self.maker_open = bool(maker_open)
        self.calls: list[str] = []
        self._orders: dict[str, dict] = {}

    async def fetch_balance(self):
        return {"USD": {"total": 10_000.0, "free": 10_000.0}}

    async def fetch_ticker(self, _symbol: str):
        return {"last": 100.0}

    async def fetch_order_book(self, _symbol: str, _limit: int = 5):
        return {"bids": [[self.bid, 1.0]], "asks": [[self.ask, 1.0]]}

    async def create_order(self, symbol: str, type: str, side: str, amount: float, price: float | None = None, params=None):
        self.calls.append(str(type))
        oid = f"alpha_{len(self.calls)}"
        if str(type).lower() == "limit" and self.maker_open:
            row = {"id": oid, "status": "open", "symbol": symbol, "side": side, "amount": amount, "filled": 0.0, "price": float(price or 100.0)}
        else:
            row = {"id": oid, "status": "filled", "symbol": symbol, "side": side, "amount": amount, "filled": amount, "price": float(price or 100.0)}
        self._orders[oid] = dict(row)
        return row

    async def fetch_order(self, order_id: str, _symbol: str):
        return dict(self._orders.get(str(order_id), {"id": str(order_id), "status": "unknown"}))

    async def cancel_order(self, order_id: str, _symbol: str | None = None):
        row = dict(self._orders.get(str(order_id), {"id": str(order_id)}))
        row["status"] = "canceled"
        self._orders[str(order_id)] = dict(row)
        return row


def _base_cfg(**overrides) -> SimpleNamespace:
    cfg = SimpleNamespace(
        run_mode="paper",
        min_signal_confidence=0.0,
        max_position_size_aud=1_000_000.0,
        max_position_pct=1.0,
        max_total_exposure_pct=1.0,
        aud_to_usd=0.65,
        starting_capital_aud=1000.0,
        current_equity_aud=1000.0,
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
        execution_maker_timeout_seconds=4.0,
        execution_maker_offset_bps=1.0,
        execution_time_in_force="GTC",
        execution_twap_threshold_aud=80.0,
        execution_twap_slices=2,
        execution_twap_duration_seconds=0.0,
        execution_adverse_spread_bps=9999.0,
        execution_alpha_enabled=True,
        execution_alpha_maker_spread_threshold_bps=2.0,
        execution_alpha_min_fill_probability=0.35,
        execution_alpha_slice_threshold_pct=0.03,
        execution_alpha_maker_fallback_seconds=8.0,
        execution_alpha_telemetry_window=200,
        primary_exchange="kraken",
        secondary_exchange="coinbase_advanced",
        _run_id="alpha_v2_run",
        _cycle_id=1,
        kill_switch_file="KILL_SWITCH",
        reconciliation_small_drift_pct=0.01,
        reconciliation_halt_drift_pct=0.05,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestExecutionAlphaV2(unittest.IsolatedAsyncioTestCase):
    async def test_maker_vs_taker_routing(self) -> None:
        cfg = _base_cfg()
        engine = KrakenDCAExecutionEngine(cfg, {"kraken": _PlanExchange()})
        signal = _Signal(
            symbol="BTC/USD",
            action="BUY",
            confidence=1.0,
            quantity=0.02,
            entry_price=100.0,
            strategy="alpha_v2",
            delta_exposure_pct=0.02,
            target_exposure_pct=0.02,
        )
        maker_plan = engine._build_execution_plan(
            signal=signal,
            exchange_name="kraken",
            quantity=0.02,
            current_price=100.0,
            spread_bps=3.0,
        )
        self.assertEqual(str(maker_plan.get("order_type")), "maker")
        self.assertEqual(str(maker_plan.get("primary_order_type")), "limit")

        for _ in range(5):
            engine.execution_alpha_engine.update_telemetry(
                symbol="BTC/USD",
                maker_planned=True,
                maker_filled=False,
                slippage_bps=5.0,
                fill_time_seconds=1.0,
                adverse_selection=False,
                fallback_from_maker=False,
            )
        taker_plan = engine._build_execution_plan(
            signal=signal,
            exchange_name="kraken",
            quantity=0.02,
            current_price=100.0,
            spread_bps=3.0,
        )
        self.assertEqual(str(taker_plan.get("order_type")), "taker")
        self.assertEqual(str(taker_plan.get("primary_order_type")), "market")

    async def test_slicing_logic_applies_for_large_delta(self) -> None:
        cfg = _base_cfg(execution_alpha_slice_threshold_pct=0.02)
        engine = KrakenDCAExecutionEngine(cfg, {"kraken": _PlanExchange()})
        signal = _Signal(
            symbol="ETH/USD",
            action="BUY",
            confidence=1.0,
            quantity=3.0,
            entry_price=100.0,
            strategy="alpha_v2",
            delta_exposure_pct=0.30,
            target_exposure_pct=0.30,
        )
        plan = engine._build_execution_plan(
            signal=signal,
            exchange_name="kraken",
            quantity=3.0,
            current_price=100.0,
            spread_bps=3.0,
        )
        self.assertEqual(str(plan.get("order_type")), "slice")
        self.assertGreater(int(plan.get("slice_count", 1) or 1), 1)
        self.assertEqual(str((plan.get("slicing") or {}).get("mode", "")), "twap")

    async def test_maker_fallback_behavior(self) -> None:
        td = tempfile.mkdtemp()
        cfg = _base_cfg(execution_maker_timeout_seconds=0.0)
        ex = _PlanExchange(maker_open=True)
        engine = KrakenDCAExecutionEngine(cfg, {"kraken": ex})
        engine.state_store = ExecutionStateStore(db_path=os.path.join(td, "state.db"))
        engine.state_store.create_intent(
            intent_id="intent_alpha_fallback",
            symbol="BTC/USD",
            side="BUY",
            quantity=1.0,
            expected_price=100.0,
            exchange="kraken",
            run_id="run",
            trace_id="trace",
            client_order_id="cid_alpha_fb",
            execution_plan={"maker_first": True},
        )
        out = await engine._place_order_with_retries(
            exchange=ex,
            exchange_name="kraken",
            symbol="BTC/USD",
            side="BUY",
            quantity=1.0,
            expected_price=100.0,
            client_order_id="cid_alpha_fb",
            intent_id="intent_alpha_fallback",
            execution_plan={
                "maker_first": True,
                "primary_order_type": "limit",
                "primary_price": 99.9,
                "fallback": {"to": "market", "after_seconds": 0.0},
            },
        )
        self.assertIsNotNone(out)
        self.assertEqual(ex.calls, ["limit", "market"])
        self.assertTrue(bool((out or {}).get("_fallback_from_maker", False)))

    async def test_telemetry_updates_and_snapshot_stats(self) -> None:
        cfg = _base_cfg()
        engine = KrakenDCAExecutionEngine(cfg, {"kraken": _PlanExchange()})
        engine.execution_alpha_engine.update_telemetry(
            symbol="BTC/USD",
            maker_planned=True,
            maker_filled=True,
            slippage_bps=1.0,
            fill_time_seconds=0.5,
            adverse_selection=False,
            fallback_from_maker=False,
        )
        engine.execution_alpha_engine.update_telemetry(
            symbol="BTC/USD",
            maker_planned=True,
            maker_filled=False,
            slippage_bps=3.0,
            fill_time_seconds=1.5,
            adverse_selection=True,
            fallback_from_maker=True,
        )
        snap = engine.execution_alpha_engine.snapshot("BTC/USD")
        self.assertAlmostEqual(float(snap.get("maker_fill_ratio", 0.0)), 0.5, places=6)
        self.assertGreaterEqual(float(snap.get("slippage_p90", 0.0)), float(snap.get("slippage_p50", 0.0)))
        self.assertGreater(float(snap.get("avg_fill_time", 0.0)), 0.0)
        self.assertGreater(float(snap.get("adverse_selection_rate", 0.0)), 0.0)

    async def test_execution_plan_persisted_in_snapshot(self) -> None:
        td = Path(tempfile.mkdtemp())
        cfg = _base_cfg(ops_jsonl_path=str(td / "ops_events.jsonl"))
        ex = _PlanExchange()
        engine = KrakenDCAExecutionEngine(cfg, {"kraken": ex})
        engine.state_store = ExecutionStateStore(db_path=str(td / "state.db"))
        engine.trade_ledger = TradeLedger(db_path=str(td / "ledger.db"))
        engine.risk_manager = ExecutionRiskManager(cfg)
        signal = _Signal(
            symbol="BTC/USD",
            action="BUY",
            confidence=1.0,
            quantity=0.02,
            entry_price=100.0,
            strategy="alpha_v2_snapshot",
            delta_exposure_pct=0.05,
            target_exposure_pct=0.05,
            priority_score=0.9,
        )
        out = await engine.execute_signals([signal], correlation_id="corr_alpha_v2")
        self.assertEqual(len(out), 1)
        rows = engine.trade_ledger.get_decision_snapshots(limit=20)
        self.assertTrue(rows)
        row = rows[0]
        plan = json.loads(str(row.get("execution_plan_json") or "{}"))
        self.assertIn("order_type", plan)
        self.assertIn("planned_order_size", plan)
        self.assertIn("slice_count", plan)
        self.assertIn("expected_slippage_bps", plan)
        self.assertIn("expected_fill_probability", plan)
        details = json.loads(str(row.get("details_json") or "{}"))
        self.assertIn("order_type", details)
        self.assertIn("planned_order_size", details)
        self.assertIn("expected_fill_probability", details)


if __name__ == "__main__":
    unittest.main()
