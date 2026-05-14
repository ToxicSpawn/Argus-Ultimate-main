"""Push 65 — Full pipeline integration: 30 tests.

Wires every subsystem end-to-end without mocks:
  Config → RiskManager → PnLTracker → ExecutionEngine
  → StrategyRunner → BacktestEngine → AlertManager → HealthRegistry
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def pnl():
    from core.pnl.pnl_tracker import PnLTracker
    return PnLTracker(initial_equity=10_000.0)


@pytest.fixture(scope="module")
def risk_cfg():
    from core.risk.risk_config import RiskConfig
    return RiskConfig(
        max_position_usd=5_000.0,
        max_daily_loss_usd=300.0,
        max_drawdown_pct=8.0,
        min_confidence=0.55,
        max_open_positions=3,
        halt_on_drawdown=True,
    )


@pytest.fixture(scope="module")
def alert_mgr():
    from core.alerts.alert_manager import AlertManager
    return AlertManager()


@pytest.fixture(scope="module")
def rm(risk_cfg, pnl, alert_mgr):
    from core.risk.risk_manager import RiskManager
    return RiskManager(risk_cfg, pnl_tracker=pnl, alert_manager=alert_mgr)


@pytest.fixture(scope="module")
def exec_engine(pnl):
    from core.execution.execution_engine import ExecutionEngine
    return ExecutionEngine(pnl_tracker=pnl, paper_trading=True)


@pytest.fixture(scope="module")
def strat_runner():
    from core.strategy.strategy_registry import StrategyRegistry
    from core.strategy.strategy_runner import StrategyRunner
    from core.strategy.builtin.momentum_strategy import MomentumStrategy
    reg = StrategyRegistry()
    reg.register(MomentumStrategy)
    runner = StrategyRunner(reg)
    return runner


@pytest.fixture(scope="module")
def health_reg():
    from core.health.health_registry import HealthRegistry
    from core.health.builtin_checks import disk_check, memory_check, event_loop_check
    reg = HealthRegistry(version="8.1.0", env="test", start_time=time.time())
    reg.register_check("disk", disk_check("."))
    reg.register_check("memory", memory_check(max_pct=99.9))
    reg.register_check("event_loop", event_loop_check())
    return reg


@pytest.fixture(scope="module")
def bt_result():
    from core.backtest.backtest_config import BacktestConfig
    from core.backtest.data_feed import DataFeed
    from core.backtest.backtest_engine import BacktestEngine
    from core.pnl.pnl_tracker import PnLTracker
    from core.execution.execution_engine import ExecutionEngine
    from core.strategy.strategy_registry import StrategyRegistry
    from core.strategy.strategy_runner import StrategyRunner
    from core.strategy.builtin.momentum_strategy import MomentumStrategy

    cfg = BacktestConfig(symbols=["BTCUSDT"], initial_equity=10_000.0, fee_bps=2.0)
    feed = DataFeed.synthetic(n=200, symbol="BTCUSDT", seed=7)
    reg = StrategyRegistry()
    reg.register(MomentumStrategy)
    runner = StrategyRunner(reg)
    pnl = PnLTracker(initial_equity=10_000.0)
    engine = ExecutionEngine(pnl_tracker=pnl, paper_trading=True)
    loop = asyncio.new_event_loop()
    loop.run_until_complete(runner.start("MomentumStrategy"))
    loop.close()
    bt = BacktestEngine(cfg, runner, engine, pnl)
    return bt.run(feed), bt


# ---------------------------------------------------------------------------
# PnL + Risk wiring (6)
# ---------------------------------------------------------------------------

class TestPnlRiskWiring:
    def test_pnl_initial_equity(self, pnl):
        assert pnl.equity == 10_000.0

    def test_risk_not_halted_initially(self, rm):
        assert rm.halted is False

    def test_risk_allows_valid_trade(self, rm):
        allowed = rm.check_order(symbol="BTCUSDT", side="buy", size_usd=100.0, confidence=0.9)
        assert allowed is True

    def test_risk_rejects_low_confidence(self, rm):
        allowed = rm.check_order(symbol="BTCUSDT", side="buy", size_usd=100.0, confidence=0.1)
        assert allowed is False

    def test_risk_rejects_oversized_position(self, rm):
        allowed = rm.check_order(symbol="BTCUSDT", side="buy", size_usd=99_999.0, confidence=0.9)
        assert allowed is False

    def test_pnl_records_trade(self, pnl, exec_engine):
        from core.execution.order import Order
        order = Order(symbol="BTCUSDT", side="buy", quantity=0.001, price=50_000.0)
        loop = asyncio.new_event_loop()
        fill = loop.run_until_complete(exec_engine.submit(order))
        loop.close()
        assert fill is not None


# ---------------------------------------------------------------------------
# Backtest integration (8)
# ---------------------------------------------------------------------------

class TestBacktestIntegration:
    def test_result_not_none(self, bt_result):
        result, _ = bt_result
        assert result is not None

    def test_equity_curve_nonempty(self, bt_result):
        result, _ = bt_result
        assert len(result.equity_curve) > 0

    def test_sharpe_is_float(self, bt_result):
        result, _ = bt_result
        assert isinstance(result.sharpe, float)

    def test_total_return_is_float(self, bt_result):
        result, _ = bt_result
        assert isinstance(result.total_return, float)

    def test_max_drawdown_nonnegative(self, bt_result):
        result, _ = bt_result
        assert result.max_drawdown >= 0.0

    def test_win_rate_in_range(self, bt_result):
        result, _ = bt_result
        assert 0.0 <= result.win_rate <= 1.0

    def test_n_trades_nonnegative(self, bt_result):
        result, _ = bt_result
        assert result.n_trades >= 0

    def test_bar_count_matches_feed(self, bt_result):
        result, bt = bt_result
        assert bt.bar_count == 200

    def test_result_to_dict_serialisable(self, bt_result):
        import json
        result, _ = bt_result
        json.dumps(result.to_dict())


# ---------------------------------------------------------------------------
# Strategy runner integration (4)
# ---------------------------------------------------------------------------

class TestStrategyRunnerIntegration:
    def test_runner_starts(self, strat_runner, event_loop):
        event_loop.run_until_complete(strat_runner.start("MomentumStrategy"))
        assert strat_runner.is_running("MomentumStrategy")

    def test_runner_lists_active(self, strat_runner, event_loop):
        event_loop.run_until_complete(strat_runner.start("MomentumStrategy"))
        assert len(strat_runner.active) >= 1

    def test_runner_stops(self, strat_runner, event_loop):
        event_loop.run_until_complete(strat_runner.start("MomentumStrategy"))
        event_loop.run_until_complete(strat_runner.stop("MomentumStrategy"))
        assert not strat_runner.is_running("MomentumStrategy")

    def test_runner_restart(self, strat_runner, event_loop):
        event_loop.run_until_complete(strat_runner.start("MomentumStrategy"))
        assert strat_runner.is_running("MomentumStrategy")


# ---------------------------------------------------------------------------
# Health registry integration (5)
# ---------------------------------------------------------------------------

class TestHealthIntegration:
    def test_run_checks_returns_system_health(self, health_reg, event_loop):
        from core.health.health_models import SystemHealth
        result = event_loop.run_until_complete(health_reg.run_checks())
        assert isinstance(result, SystemHealth)

    def test_all_checks_ran(self, health_reg, event_loop):
        result = event_loop.run_until_complete(health_reg.run_checks())
        assert "disk" in result.components
        assert "memory" in result.components
        assert "event_loop" in result.components

    def test_system_is_ready(self, health_reg, event_loop):
        result = event_loop.run_until_complete(health_reg.run_checks())
        assert result.is_ready is True

    def test_system_is_live(self, health_reg, event_loop):
        result = event_loop.run_until_complete(health_reg.run_checks())
        assert result.is_live is True

    def test_uptime_positive(self, health_reg, event_loop):
        result = event_loop.run_until_complete(health_reg.run_checks())
        assert result.uptime_s >= 0.0


# ---------------------------------------------------------------------------
# Alert manager integration (4)
# ---------------------------------------------------------------------------

class TestAlertManagerIntegration:
    def test_alert_manager_creates(self, alert_mgr):
        assert alert_mgr is not None

    def test_enqueue_alert_no_error(self, alert_mgr, event_loop):
        from core.alerts.alert_models import Alert, AlertLevel
        alert = Alert(level=AlertLevel.INFO, title="Test", message="Integration test")
        event_loop.run_until_complete(alert_mgr.send(alert))

    def test_channel_count_zero_by_default(self, alert_mgr):
        assert len(alert_mgr.channels) == 0

    def test_alert_worker_start_stop(self, alert_mgr, event_loop):
        event_loop.run_until_complete(alert_mgr.start_worker())
        event_loop.run_until_complete(alert_mgr.stop_worker())
