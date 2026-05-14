"""
tests/test_unified_strategy_orchestrator.py — Tests for Unified Strategy Orchestrator

Tests the integration layer that connects strategies to adaptive systems.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock

from adaptive.unified_strategy_orchestrator import (
    UnifiedStrategyOrchestrator,
    StrategyMetrics,
    StrategyState,
    AdaptationCycleResult,
    STRATEGY_REGIME_AFFINITY,
    create_unified_orchestrator,
)


class TestStrategyMetrics:
    """Tests for StrategyMetrics dataclass."""
    
    def test_defaults(self):
        """Should have sensible defaults."""
        metrics = StrategyMetrics(strategy_name="test")
        
        assert metrics.strategy_name == "test"
        assert metrics.sharpe == 0.0
        assert metrics.win_rate == 0.5
        assert metrics.is_active is True
    
    def test_to_dict(self):
        """Should convert to dict for adaptive system compatibility."""
        metrics = StrategyMetrics(
            strategy_name="test",
            sharpe=1.5,
            win_rate=0.6,
            trades_7d=10,
        )
        
        d = metrics.to_dict()
        
        assert d["sharpe"] == 1.5
        assert d["win_rate"] == 0.6
        assert d["trades_7d"] == 10


class TestStrategyState:
    """Tests for StrategyState dataclass."""
    
    def test_defaults(self):
        """Should have sensible defaults."""
        state = StrategyState(strategy_name="test")
        
        assert state.is_enabled is True
        assert state.current_weight == 0.1
        assert state.risk_multiplier == 1.0
        assert state.consecutive_losses == 0


class TestUnifiedStrategyOrchestrator:
    """Tests for UnifiedStrategyOrchestrator."""
    
    def test_init_defaults(self):
        """Should initialize with sensible defaults."""
        orchestrator = UnifiedStrategyOrchestrator()
        
        assert orchestrator.enable_regime_routing is True
        assert orchestrator.enable_meta_learning is True
        assert orchestrator.enable_risk_adaptation is True
        assert orchestrator.min_regime_confidence == 0.6
    
    def test_register_strategy(self):
        """Should register a strategy."""
        orchestrator = UnifiedStrategyOrchestrator()
        mock_strategy = MagicMock()
        
        orchestrator.register_strategy("test_strategy", mock_strategy, strategy_type="momentum")
        
        assert "test_strategy" in orchestrator._strategies
        assert "test_strategy" in orchestrator._strategy_states
        assert "test_strategy" in orchestrator._strategy_metrics
    
    def test_unregister_strategy(self):
        """Should unregister a strategy."""
        orchestrator = UnifiedStrategyOrchestrator()
        mock_strategy = MagicMock()
        
        orchestrator.register_strategy("test_strategy", mock_strategy)
        orchestrator.unregister_strategy("test_strategy")
        
        assert "test_strategy" not in orchestrator._strategies
    
    def test_update_strategy_metrics(self):
        """Should update strategy metrics."""
        orchestrator = UnifiedStrategyOrchestrator()
        mock_strategy = MagicMock()
        
        orchestrator.register_strategy("test_strategy", mock_strategy)
        orchestrator.update_strategy_metrics("test_strategy", {
            "sharpe": 1.5,
            "win_rate": 0.65,
            "trades_7d": 20,
        })
        
        metrics = orchestrator._strategy_metrics["test_strategy"]
        assert metrics.sharpe == 1.5
        assert metrics.win_rate == 0.65
        assert metrics.trades_7d == 20
    
    def test_update_regime(self):
        """Should update current regime."""
        orchestrator = UnifiedStrategyOrchestrator()
        
        orchestrator.update_regime("trending", 0.85)
        
        assert orchestrator._current_regime == "trending"
        assert orchestrator._regime_confidence == 0.85
    
    def test_get_strategy_weights(self):
        """Should return current strategy weights."""
        orchestrator = UnifiedStrategyOrchestrator()
        
        orchestrator.register_strategy("strat1", MagicMock(), initial_weight=0.3)
        orchestrator.register_strategy("strat2", MagicMock(), initial_weight=0.7)
        
        weights = orchestrator.get_strategy_weights()
        
        assert weights["strat1"] == 0.3
        assert weights["strat2"] == 0.7
    
    def test_should_skip_strategy_disabled(self):
        """Should skip disabled strategies."""
        orchestrator = UnifiedStrategyOrchestrator()
        
        orchestrator.register_strategy("test_strategy", MagicMock())
        orchestrator._strategy_states["test_strategy"].is_enabled = False
        
        assert orchestrator.should_skip_strategy("test_strategy") is True
    
    def test_should_skip_strategy_regime(self):
        """Should skip strategies based on regime."""
        orchestrator = UnifiedStrategyOrchestrator()
        
        orchestrator.register_strategy("trend_following", MagicMock())
        orchestrator.update_regime("crisis", 0.9)
        
        # trend_following should be skipped in crisis
        assert orchestrator.should_skip_strategy("trend_following") is True
    
    def test_run_adaptation_cycle_basic(self):
        """Should run a basic adaptation cycle."""
        orchestrator = UnifiedStrategyOrchestrator()
        
        orchestrator.register_strategy("strat1", MagicMock(), initial_weight=0.5)
        orchestrator.register_strategy("strat2", MagicMock(), initial_weight=0.5)
        orchestrator.update_regime("trending", 0.8)
        
        result = orchestrator.run_adaptation_cycle()
        
        assert isinstance(result, AdaptationCycleResult)
        assert result.regime == "trending"
        assert result.regime_confidence == 0.8
        assert "strat1" in result.strategy_weights
        assert "strat2" in result.strategy_weights
    
    def test_adaptation_cycle_regime_routing(self):
        """Should apply regime-based routing."""
        orchestrator = UnifiedStrategyOrchestrator(
            enable_meta_learning=False, 
            enable_risk_adaptation=False,
            max_strategy_allocation=0.9,  # Allow larger allocation to see regime effects
        )
        
        orchestrator.register_strategy("momentum", MagicMock(), initial_weight=0.5)
        orchestrator.register_strategy("mean_reversion", MagicMock(), initial_weight=0.5)
        
        # Trending regime should boost momentum, penalize mean_reversion
        orchestrator.update_regime("trending", 0.9)
        result = orchestrator.run_adaptation_cycle()
        
        # Momentum should have higher weight than mean_reversion
        # (after normalization, trending boosts momentum)
        assert result.strategy_weights["momentum"] >= result.strategy_weights["mean_reversion"]
    
    def test_adaptation_cycle_risk_adjustment(self):
        """Should apply risk adjustments for high drawdown."""
        orchestrator = UnifiedStrategyOrchestrator(enable_meta_learning=False)
        
        orchestrator.register_strategy("risky_strat", MagicMock(), initial_weight=0.5)
        orchestrator.register_strategy("safe_strat", MagicMock(), initial_weight=0.5)
        
        # Set high drawdown for risky strategy
        orchestrator.update_strategy_metrics("risky_strat", {"drawdown_pct": 15.0})
        orchestrator.update_regime("volatile", 0.8)
        
        result = orchestrator.run_adaptation_cycle()
        
        # Risky strategy should have lower weight due to drawdown
        assert result.risk_adjustments.get("risky_strat", 1.0) < 1.0
    
    def test_adaptation_cycle_consecutive_losses(self):
        """Should reduce weight after consecutive losses."""
        orchestrator = UnifiedStrategyOrchestrator(
            enable_meta_learning=False,
            max_strategy_allocation=0.9,
        )
        
        orchestrator.register_strategy("losing_strat", MagicMock(), initial_weight=0.5)
        orchestrator.register_strategy("winning_strat", MagicMock(), initial_weight=0.5)
        
        # Set consecutive losses
        orchestrator._strategy_states["losing_strat"].consecutive_losses = 6
        orchestrator.update_regime("range", 0.8)
        
        result = orchestrator.run_adaptation_cycle()
        
        # Losing strategy should have reduced weight due to consecutive losses
        assert result.strategy_weights["losing_strat"] <= result.strategy_weights["winning_strat"]
    
    def test_get_regime_affinity(self):
        """Should return regime affinity for strategies."""
        orchestrator = UnifiedStrategyOrchestrator()
        orchestrator.update_regime("trending", 0.8)
        
        # Momentum should have high affinity in trending
        affinity = orchestrator.get_regime_affinity("momentum")
        assert affinity > 1.0
        
        # Mean reversion should have low affinity in trending
        affinity = orchestrator.get_regime_affinity("mean_reversion")
        assert affinity < 1.0
    
    def test_get_orchestration_stats(self):
        """Should return orchestration statistics."""
        orchestrator = UnifiedStrategyOrchestrator()
        
        orchestrator.register_strategy("strat1", MagicMock())
        orchestrator.register_strategy("strat2", MagicMock())
        orchestrator.update_regime("trending", 0.8)
        orchestrator.run_adaptation_cycle()
        
        stats = orchestrator.get_orchestration_stats()
        
        assert stats["total_strategies"] == 2
        assert stats["enabled_strategies"] == 2
        assert stats["current_regime"] == "trending"
        assert stats["adaptation_count"] == 1
    
    def test_reset_strategy_state(self):
        """Should reset strategy adaptive state."""
        orchestrator = UnifiedStrategyOrchestrator()
        
        orchestrator.register_strategy("test_strategy", MagicMock())
        state = orchestrator._strategy_states["test_strategy"]
        state.is_enabled = False
        state.consecutive_losses = 10
        state.risk_multiplier = 0.5
        
        orchestrator.reset_strategy_state("test_strategy")
        
        assert state.is_enabled is True
        assert state.consecutive_losses == 0
        assert state.risk_multiplier == 1.0


class TestStrategyRegimeAffinity:
    """Tests for strategy regime affinity mappings."""
    
    def test_momentum_affinity(self):
        """Momentum should favor trending regimes."""
        affinity = STRATEGY_REGIME_AFFINITY.get("momentum", {})
        
        assert affinity.get("trending", 1.0) > 1.0
        assert affinity.get("mean_revert", 1.0) < 1.0
    
    def test_mean_reversion_affinity(self):
        """Mean reversion should favor range-bound regimes."""
        affinity = STRATEGY_REGIME_AFFINITY.get("mean_reversion", {})
        
        assert affinity.get("mean_revert", 1.0) > 1.0
        assert affinity.get("trending", 1.0) < 1.0
    
    def test_grid_mean_reversion_affinity(self):
        """Grid mean reversion should favor range regimes."""
        affinity = STRATEGY_REGIME_AFFINITY.get("grid_mean_reversion", {})
        
        assert affinity.get("range", 1.0) > 1.0
        assert affinity.get("mean_revert", 1.0) > 1.0
    
    def test_mev_sandwich_affinity(self):
        """MEV sandwich should work in most regimes but struggle in crisis."""
        affinity = STRATEGY_REGIME_AFFINITY.get("mev_sandwich", {})
        
        assert affinity.get("crisis", 1.0) < 0.6
        assert affinity.get("calm", 1.0) >= 1.0
    
    def test_options_vol_arb_affinity(self):
        """Options vol arb should favor volatile regimes."""
        affinity = STRATEGY_REGIME_AFFINITY.get("options_vol_arb", {})
        
        assert affinity.get("volatile", 1.0) > 1.0
        assert affinity.get("crisis", 1.0) > 1.0
        assert affinity.get("calm", 1.0) < 1.0


class TestFactoryFunction:
    """Tests for factory functions."""
    
    def test_create_unified_orchestrator(self):
        """Should create orchestrator with strategies."""
        strategies = {
            "strat1": MagicMock(),
            "strat2": MagicMock(),
        }
        
        orchestrator = create_unified_orchestrator(strategies)
        
        assert "strat1" in orchestrator._strategies
        assert "strat2" in orchestrator._strategies
    
    def test_create_with_config(self):
        """Should create orchestrator with custom config."""
        orchestrator = create_unified_orchestrator(
            enable_regime_routing=False,
            min_regime_confidence=0.8,
            max_strategy_allocation=0.3,
        )
        
        assert orchestrator.enable_regime_routing is False
        assert orchestrator.min_regime_confidence == 0.8
        assert orchestrator.max_strategy_allocation == 0.3


class TestIntegrationWithRegimeRouter:
    """Integration tests with regime strategy router."""
    
    def test_regime_router_includes_new_strategies(self):
        """Regime router should include new strategies."""
        from adaptive.regime_strategy_router import RegimeStrategyRouter
        
        router = RegimeStrategyRouter()
        
        # Check that new strategies are in the regime map
        base_weights = {
            "momentum": 1.0,
            "mean_reversion": 1.0,
            "mev_sandwich": 1.0,
            "triangular_arb": 1.0,
            "options_vol_arb": 1.0,
            "grid_mean_reversion": 1.0,
            "oracle_deviation": 1.0,
            "cross_chain_arb": 1.0,
        }
        
        # Trending should boost MEV
        weights = router.get_weights("trending", base_weights)
        assert weights["mev_sandwich"] > weights["mean_reversion"]
        
        # Mean revert should boost grid
        weights = router.get_weights("mean_revert", base_weights)
        assert weights["grid_mean_reversion"] > weights["momentum"]
        
        # Crisis should boost options vol arb
        weights = router.get_weights("crisis", base_weights)
        assert weights["options_vol_arb"] > weights["mev_sandwich"]
    
    def test_calm_regime_boosts_arb_strategies(self):
        """Calm regime should boost arbitrage strategies."""
        from adaptive.regime_strategy_router import RegimeStrategyRouter
        
        router = RegimeStrategyRouter()
        
        base_weights = {
            "triangular_arb": 1.0,
            "oracle_deviation": 1.0,
            "momentum": 1.0,
        }
        
        weights = router.get_weights("calm", base_weights)
        
        # Arb strategies should be boosted
        assert weights["triangular_arb"] > 1.0
        assert weights["oracle_deviation"] > 1.0


class TestEndToEndAdaptation:
    """End-to-end tests for complete adaptation workflow."""
    
    def test_full_adaptation_workflow(self):
        """Should complete full adaptation workflow."""
        orchestrator = UnifiedStrategyOrchestrator()
        
        # Register all new strategies
        strategies = {
            "momentum": MagicMock(),
            "mean_reversion": MagicMock(),
            "mev_sandwich": MagicMock(),
            "triangular_arb": MagicMock(),
            "options_vol_arb": MagicMock(),
            "grid_mean_reversion": MagicMock(),
            "oracle_deviation": MagicMock(),
            "cross_chain_arb": MagicMock(),
        }
        
        for name, strategy in strategies.items():
            orchestrator.register_strategy(name, strategy, initial_weight=0.125)
        
        # Simulate different regimes and verify adaptation
        regimes = ["trending", "mean_revert", "volatile", "crisis", "calm"]
        
        for regime in regimes:
            orchestrator.update_regime(regime, 0.85)
            result = orchestrator.run_adaptation_cycle()
            
            assert result.regime == regime
            assert len(result.strategy_weights) == len(strategies)
            
            # Verify total allocation is normalized
            total = sum(result.strategy_weights.values())
            assert 0.99 < total < 1.01  # Allow small floating point error
    
    def test_strategy_disabled_in_crisis(self):
        """Trend following should be disabled in crisis."""
        orchestrator = UnifiedStrategyOrchestrator()
        
        orchestrator.register_strategy("trend_following", MagicMock(), initial_weight=0.5)
        orchestrator.register_strategy("tail_hedge", MagicMock(), initial_weight=0.5)
        
        orchestrator.update_regime("crisis", 0.95)
        result = orchestrator.run_adaptation_cycle()
        
        # trend_following should be skipped in crisis
        assert orchestrator.should_skip_strategy("trend_following") is True
    
    def test_adaptation_respects_min_max_allocation(self):
        """Should respect min/max allocation bounds."""
        orchestrator = UnifiedStrategyOrchestrator(
            min_strategy_allocation=0.05,
            max_strategy_allocation=0.4,
        )
        
        orchestrator.register_strategy("dominant", MagicMock(), initial_weight=0.8)
        orchestrator.register_strategy("minor", MagicMock(), initial_weight=0.2)
        
        orchestrator.update_regime("trending", 0.8)
        result = orchestrator.run_adaptation_cycle()
        
        for weight in result.strategy_weights.values():
            assert weight >= 0.05
            assert weight <= 0.4
