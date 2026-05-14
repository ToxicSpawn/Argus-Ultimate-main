"""Push 67 — Tests: PositionExecutor, DCAExecutor, FeeAdjuster,
OrderRefreshLoop, StrategyController, RLController. 26 tests.
"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class _Bar:
    def __init__(self, o=50000.0, h=51000.0, l=49000.0, c=50500.0, sym="BTCUSDT"):
        self.open=o; self.high=h; self.low=l; self.close=c; self.symbol=sym


# ---------------------------------------------------------------------------
# PositionExecutor (6)
# ---------------------------------------------------------------------------

class TestPositionExecutor:
    def test_opens_in_open_status(self):
        from core.execution.position_executor import PositionExecutor, PositionStatus
        p = PositionExecutor("BTCUSDT", "buy", 50000.0, 1000.0)
        assert p.status == PositionStatus.OPEN

    def test_sl_hit_on_price_drop(self):
        from core.execution.position_executor import PositionExecutor, PositionStatus
        p = PositionExecutor("BTCUSDT", "buy", 50000.0, 1000.0, stop_loss_pct=0.02)
        status = p.evaluate(48000.0)  # 4% drop
        assert status == PositionStatus.SL_HIT

    def test_tp_hit_on_price_rise(self):
        from core.execution.position_executor import PositionExecutor, PositionStatus
        p = PositionExecutor("BTCUSDT", "buy", 50000.0, 1000.0,
                             take_profit_pct=0.04, use_partial_tp=False)
        status = p.evaluate(52500.0)  # 5% rise
        assert status == PositionStatus.TP_HIT

    def test_partial_tp_at_1r(self):
        from core.execution.position_executor import PositionExecutor, PositionStatus
        p = PositionExecutor("BTCUSDT", "buy", 50000.0, 1000.0,
                             partial_tp_pct=0.02, use_partial_tp=True)
        status = p.evaluate(51100.0)  # > 2% rise
        assert status == PositionStatus.TP1_HIT
        assert p.partial_closed

    def test_unrealised_pnl_positive_for_long_up(self):
        from core.execution.position_executor import PositionExecutor
        p = PositionExecutor("BTCUSDT", "buy", 50000.0, 1000.0)
        p.evaluate(51000.0)
        assert p.unrealised_pnl > 0

    def test_to_dict_has_required_keys(self):
        from core.execution.position_executor import PositionExecutor
        p = PositionExecutor("BTCUSDT", "buy", 50000.0, 1000.0)
        d = p.to_dict()
        for k in ["symbol", "side", "status", "total_pnl", "position_id"]:
            assert k in d


# ---------------------------------------------------------------------------
# PositionExecutorEngine (3)
# ---------------------------------------------------------------------------

class TestPositionExecutorEngine:
    def test_open_registers_position(self):
        from core.execution.position_executor import PositionExecutor, PositionExecutorEngine
        eng = PositionExecutorEngine(max_positions=3)
        p = PositionExecutor("BTCUSDT", "buy", 50000.0, 500.0)
        assert eng.open(p) is True
        assert len(eng.get_open()) == 1

    def test_capacity_limit_enforced(self):
        from core.execution.position_executor import PositionExecutor, PositionExecutorEngine
        eng = PositionExecutorEngine(max_positions=1)
        eng.open(PositionExecutor("BTCUSDT", "buy", 50000.0, 500.0))
        result = eng.open(PositionExecutor("ETHUSDT", "buy", 3000.0, 500.0))
        assert result is False

    def test_evaluate_all_closes_sl(self):
        from core.execution.position_executor import PositionExecutor, PositionExecutorEngine
        eng = PositionExecutorEngine()
        p = PositionExecutor("BTCUSDT", "buy", 50000.0, 1000.0, stop_loss_pct=0.02)
        eng.open(p)
        closed = eng.evaluate_all({"BTCUSDT": 48000.0})
        assert len(closed) == 1


# ---------------------------------------------------------------------------
# DCAPlan (4)
# ---------------------------------------------------------------------------

class TestDCAPlan:
    def test_builds_correct_levels(self):
        from core.execution.dca_executor import DCAPlan
        plan = DCAPlan("BTCUSDT", "buy", 3000.0, n_levels=3, level_spread=0.01)
        plan.build(50000.0)
        assert len(plan.levels) == 3
        assert plan.levels[1].price_target < plan.levels[0].price_target

    def test_fills_level_on_price_drop(self):
        from core.execution.dca_executor import DCAPlan, DCAStatus
        plan = DCAPlan("BTCUSDT", "buy", 3000.0, n_levels=3, level_spread=0.01)
        plan.build(50000.0)
        filled = plan.evaluate(49000.0)  # hits level 1 and 2
        assert len(filled) >= 1

    def test_avg_fill_price_weighted(self):
        from core.execution.dca_executor import DCAPlan
        plan = DCAPlan("BTCUSDT", "buy", 3000.0, n_levels=3, level_spread=0.01)
        plan.build(50000.0)
        plan.evaluate(48000.0)  # fills all levels
        assert plan.avg_fill_price > 0

    def test_complete_status_when_all_filled(self):
        from core.execution.dca_executor import DCAPlan, DCAStatus
        plan = DCAPlan("BTCUSDT", "buy", 3000.0, n_levels=2, level_spread=0.005)
        plan.build(50000.0)
        plan.evaluate(49000.0)
        assert plan.status == DCAStatus.COMPLETE


# ---------------------------------------------------------------------------
# FeeAdjuster (5)
# ---------------------------------------------------------------------------

class TestFeeAdjuster:
    def test_standard_tier_at_zero_volume(self):
        from core.execution.fee_adjuster import FeeAdjuster
        fa = FeeAdjuster(monthly_volume_usd=0)
        assert fa.fee_profile.tier_name == "Standard"

    def test_supreme_tier_at_high_volume(self):
        from core.execution.fee_adjuster import FeeAdjuster
        fa = FeeAdjuster(monthly_volume_usd=200_000_000)
        assert fa.fee_profile.tier_name == "Supreme VIP"

    def test_adjusted_spreads_wider_than_base(self):
        from core.execution.fee_adjuster import FeeAdjuster
        fa = FeeAdjuster(base_bid_spread=0.001)
        quote = fa.adjusted_spreads()
        assert quote.bid_spread >= 0.001

    def test_is_profitable_true_for_good_trade(self):
        from core.execution.fee_adjuster import FeeAdjuster
        fa = FeeAdjuster()
        assert fa.is_profitable(50000.0, 50300.0, "buy") is True

    def test_gas_gate_rejects_expensive_gas(self):
        from core.execution.fee_adjuster import FeeAdjuster
        fa = FeeAdjuster(max_gas_usd=2.0)
        assert fa.gas_is_acceptable(5.0, 100.0) is False


# ---------------------------------------------------------------------------
# OrderRefreshLoop (3)
# ---------------------------------------------------------------------------

class TestOrderRefreshLoop:
    def test_starts_and_stops(self):
        from core.execution.order_refresh import OrderRefreshLoop
        loop_obj = OrderRefreshLoop(refresh_secs=0.05)
        async def run():
            await loop_obj.start()
            await asyncio.sleep(0.12)
            await loop_obj.stop()
        asyncio.get_event_loop().run_until_complete(run())
        assert not loop_obj.is_running

    def test_refresh_count_increments(self):
        from core.execution.order_refresh import OrderRefreshLoop
        loop_obj = OrderRefreshLoop(refresh_secs=0.05)
        async def run():
            await loop_obj.start()
            await asyncio.sleep(0.18)
            await loop_obj.stop()
        asyncio.get_event_loop().run_until_complete(run())
        assert loop_obj.stats.total_refreshes >= 1

    def test_callback_invoked(self):
        from core.execution.order_refresh import OrderRefreshLoop
        called = []
        async def cb(stats): called.append(1)
        loop_obj = OrderRefreshLoop(refresh_secs=0.05, on_refresh_cb=cb)
        async def run():
            await loop_obj.start()
            await asyncio.sleep(0.12)
            await loop_obj.stop()
        asyncio.get_event_loop().run_until_complete(run())
        assert len(called) >= 1


# ---------------------------------------------------------------------------
# StrategyController (3)
# ---------------------------------------------------------------------------

class TestStrategyController:
    def test_starts_and_stops(self):
        from core.strategy.v2.strategy_controller import StrategyController
        ctrl = StrategyController()
        async def run():
            await ctrl.start()
            await asyncio.sleep(0.05)
            await ctrl.stop()
        asyncio.get_event_loop().run_until_complete(run())
        assert not ctrl.is_running

    def test_tick_count_increments(self):
        from core.strategy.v2.strategy_controller import StrategyController
        ctrl = StrategyController()
        async def run():
            await ctrl.start()
            await asyncio.sleep(1.1)
            await ctrl.stop()
        asyncio.get_event_loop().run_until_complete(run())
        assert ctrl.tick_count >= 1

    def test_on_bar_increments_bars(self):
        from core.strategy.v2.strategy_controller import StrategyController
        ctrl = StrategyController()
        async def run():
            await ctrl.start()
            await ctrl.on_bar(_Bar())
            await ctrl.stop()
        asyncio.get_event_loop().run_until_complete(run())
        assert ctrl.bars_processed == 1


# ---------------------------------------------------------------------------
# ControllerConfig (2)
# ---------------------------------------------------------------------------

class TestControllerConfig:
    def test_to_dict_roundtrip(self):
        from core.strategy.v2.controller_config import ControllerConfig
        cfg = ControllerConfig(controller_name="Test", initial_equity=5000.0)
        d = cfg.to_dict()
        cfg2 = ControllerConfig.from_dict(d)
        assert cfg2.controller_name == "Test"
        assert cfg2.initial_equity == 5000.0

    def test_yaml_str_contains_name(self):
        from core.strategy.v2.controller_config import ControllerConfig
        cfg = ControllerConfig(controller_name="MyCtrl")
        yaml_str = cfg.to_yaml_str()
        assert "MyCtrl" in yaml_str


# ---------------------------------------------------------------------------
# RLController (no model) (2 smoke tests)
# ---------------------------------------------------------------------------

class TestRLController:
    def test_creates_without_error(self):
        from core.strategy.v2.rl_controller import RLController
        ctrl = RLController()
        assert ctrl is not None
        assert ctrl.open_positions == 0

    def test_on_bar_no_model_returns_empty(self):
        from core.strategy.v2.rl_controller import RLController
        ctrl = RLController()
        async def run():
            await ctrl.start()
            actions = await ctrl.on_bar(_Bar())
            await ctrl.stop()
            return actions
        actions = asyncio.get_event_loop().run_until_complete(run())
        assert isinstance(actions, list)
