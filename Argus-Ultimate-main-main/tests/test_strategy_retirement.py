"""Unit tests for core.strategy_retirement — Push 100 StatePersist."""
from __future__ import annotations

import pytest

from core.strategy_retirement import (
    RetirementConfig,
    StrategyRetirementManager,
    StrategyStats,
)


@pytest.fixture()
def mgr() -> StrategyRetirementManager:
    cfg = RetirementConfig(
        min_sharpe=0.3,
        max_drawdown=0.25,
        min_win_rate=0.35,
        min_trade_count=5,
        max_cons_loss=10,
        cooldown_secs=0.0,   # disable cooldown in tests
    )
    return StrategyRetirementManager(config=cfg)


class TestRetirementGates:
    def test_healthy_not_retired(self, mgr):
        stats = StrategyStats(
            strategy="momentum_v2",
            sharpe=1.2, drawdown=0.08, win_rate=0.55,
            trade_count=100, consecutive_losses=2,
        )
        result = mgr.evaluate(stats)
        assert result is None
        assert not mgr.is_retired("momentum_v2")

    def test_low_sharpe_retires(self, mgr):
        stats = StrategyStats(
            strategy="bad_sharpe",
            sharpe=0.1, drawdown=0.10, win_rate=0.50,
            trade_count=50, consecutive_losses=1,
        )
        reason = mgr.evaluate(stats)
        assert reason is not None and "sharpe" in reason
        assert mgr.is_retired("bad_sharpe")

    def test_high_drawdown_retires(self, mgr):
        stats = StrategyStats(
            strategy="bad_dd",
            sharpe=1.0, drawdown=0.40, win_rate=0.50,
            trade_count=50, consecutive_losses=1,
        )
        reason = mgr.evaluate(stats)
        assert reason is not None and "drawdown" in reason

    def test_low_win_rate_retires(self, mgr):
        stats = StrategyStats(
            strategy="bad_wr",
            sharpe=1.0, drawdown=0.10, win_rate=0.20,
            trade_count=50, consecutive_losses=1,
        )
        reason = mgr.evaluate(stats)
        assert reason is not None and "win_rate" in reason

    def test_cons_loss_retires(self, mgr):
        stats = StrategyStats(
            strategy="loss_streak",
            sharpe=1.0, drawdown=0.10, win_rate=0.50,
            trade_count=50, consecutive_losses=15,
        )
        reason = mgr.evaluate(stats)
        assert reason is not None and "consecutive_losses" in reason

    def test_sparse_skipped(self, mgr):
        stats = StrategyStats(
            strategy="new_strat",
            sharpe=0.05, drawdown=0.50, win_rate=0.10,
            trade_count=2, consecutive_losses=2,
        )
        result = mgr.evaluate(stats)
        assert result is None   # too few trades — not retired

    def test_reinstate(self, mgr):
        stats = StrategyStats(
            strategy="to_reinstate",
            sharpe=0.1, drawdown=0.10, win_rate=0.50,
            trade_count=50, consecutive_losses=1,
        )
        mgr.evaluate(stats)
        assert mgr.is_retired("to_reinstate")
        mgr.reinstate("to_reinstate")
        assert not mgr.is_retired("to_reinstate")

    def test_active_filter(self, mgr):
        stats = StrategyStats(
            strategy="retiring",
            sharpe=0.1, drawdown=0.10, win_rate=0.50,
            trade_count=50, consecutive_losses=1,
        )
        mgr.evaluate(stats)
        all_strats = ["retiring", "healthy_a", "healthy_b"]
        active = mgr.active_strategies(all_strats)
        assert active == ["healthy_a", "healthy_b"]
