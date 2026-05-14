"""
Test Strategy Learning Adapter
================================
Tests for the strategy_learning_adapter module.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from strategies.strategy_learning_adapter import (
    StrategyLearningAdapter,
    StrategyLearningManager,
    LearnableStrategyParams,
    StrategyType,
    StrategyPerformance,
    wire_all_strategies,
)


class MockMomentumStrategy:
    """Mock momentum strategy for testing."""
    def __init__(self):
        self.short_window = 10
        self.long_window = 40
        self.min_strength = 0.002


class MockMeanReversionStrategy:
    """Mock mean reversion strategy for testing."""
    def __init__(self):
        self.lookback = 50
        self.base_threshold = 1.5
        self.vol_scale = 1.0


class TestLearnableStrategyParams:
    """Test suite for LearnableStrategyParams."""
    
    def test_default_values(self):
        """Test default parameter values."""
        params = LearnableStrategyParams()
        assert params.momentum_short_window == 10
        assert params.momentum_long_window == 40
        assert params.mr_lookback == 50
        assert params.tf_fast_window == 12
        assert params.position_size_pct == 0.1
        assert params.stop_loss_pct == 0.02


class TestStrategyType:
    """Test suite for StrategyType enum."""
    
    def test_strategy_types(self):
        """Test all strategy types exist."""
        assert StrategyType.MOMENTUM.value == "momentum"
        assert StrategyType.MEAN_REVERSION.value == "mean_reversion"
        assert StrategyType.TREND_FOLLOWING.value == "trend_following"
        assert StrategyType.BREAKOUT.value == "breakout"
        assert StrategyType.SCALPING.value == "scalping"


class TestStrategyPerformance:
    """Test suite for StrategyPerformance."""
    
    def test_initial_values(self):
        """Test initial performance values."""
        perf = StrategyPerformance(strategy_name="test")
        assert perf.total_trades == 0
        assert perf.win_rate == 0.0
        # profit_factor returns 0.0 when no trades (division by zero protection)
        assert perf.profit_factor == 0.0
    
    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        perf = StrategyPerformance(strategy_name="test")
        perf.total_trades = 10
        perf.winning_trades = 6
        
        assert perf.win_rate == 0.6
    
    def test_profit_factor_calculation(self):
        """Test profit factor calculation."""
        perf = StrategyPerformance(strategy_name="test")
        perf.return_history = [100.0, -50.0, 30.0, -20.0]
        
        # Wins: 100 + 30 = 130
        # Losses: 50 + 20 = 70
        # Profit factor: 130 / 70
        assert abs(perf.profit_factor - 130/70) < 0.01


class TestStrategyLearningAdapter:
    """Test suite for StrategyLearningAdapter."""
    
    def setup_method(self):
        self.mock_strategy = MockMomentumStrategy()
        # Use real orchestrator to avoid MagicMock attribute issues
        from learning.learning_orchestrator import LearningOrchestrator
        self.orchestrator = LearningOrchestrator()
        self.adapter = StrategyLearningAdapter(
            strategy=self.mock_strategy,
            strategy_type=StrategyType.MOMENTUM,
            learning_orchestrator=self.orchestrator,
            name="test_momentum"
        )
    
    def test_initialization(self):
        """Test adapter initialization."""
        assert self.adapter.name == "test_momentum"
        assert self.adapter.strategy_type == StrategyType.MOMENTUM
        assert self.adapter.performance.total_trades == 0
    
    def test_generate_signal_buy(self):
        """Test buy signal generation."""
        # Create prices with clear upward momentum
        prices = [100.0] * 50
        for i in range(50):
            prices[i] = 100.0 + i * 0.5  # Strong uptrend
        
        signal = self.adapter.generate_signal(prices, regime="trending_up")
        
        assert signal["action"] in ["buy", "hold"]
        assert 0.0 <= signal["confidence"] <= 1.0
    
    def test_generate_signal_sell(self):
        """Test sell signal generation."""
        # Create prices with clear downward momentum
        prices = [100.0] * 50
        for i in range(50):
            prices[i] = 100.0 - i * 0.5  # Strong downtrend
        
        signal = self.adapter.generate_signal(prices, regime="trending_down")
        
        assert signal["action"] in ["sell", "hold"]
        assert 0.0 <= signal["confidence"] <= 1.0
    
    def test_record_outcome_updates_performance(self):
        """Test that recording outcomes updates performance."""
        self.adapter.record_outcome(pnl=100.0, regime="trending_up")
        
        assert self.adapter.performance.total_trades == 1
        assert self.adapter.performance.total_pnl == 100.0
        assert self.adapter.performance.winning_trades == 1
    
    def test_record_outcome_feeds_orchestrator(self):
        """Test that outcomes are reported to orchestrator."""
        self.adapter.last_params_used = {"momentum_short_window": 10.0}
        
        self.adapter.record_outcome(pnl=50.0, regime="ranging")
        
        # Verify the orchestrator recorded the trade
        assert self.orchestrator.state.total_updates == 1
    
    def test_regime_specific_performance(self):
        """Test regime-specific performance tracking."""
        self.adapter.record_outcome(pnl=100.0, regime="trending_up")
        self.adapter.record_outcome(pnl=80.0, regime="trending_up")
        self.adapter.record_outcome(pnl=-20.0, regime="ranging")
        
        assert "trending_up" in self.adapter.regime_performance
        assert "ranging" in self.adapter.regime_performance
        assert self.adapter.regime_performance["trending_up"].total_trades == 2
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        self.adapter.record_outcome(pnl=100.0, regime="trending_up")
        self.adapter.record_outcome(pnl=-50.0, regime="trending_up")
        
        stats = self.adapter.get_stats()
        
        assert stats["name"] == "test_momentum"
        assert stats["total_trades"] == 2
        assert stats["win_rate"] == 0.5
        assert stats["total_pnl"] == 50.0


class TestStrategyLearningManager:
    """Test suite for StrategyLearningManager."""
    
    def setup_method(self):
        self.mock_orchestrator = MagicMock()
        self.manager = StrategyLearningManager(learning_orchestrator=self.mock_orchestrator)
    
    def test_register_strategy(self):
        """Test strategy registration."""
        mock_strategy = MockMomentumStrategy()
        
        adapter = self.manager.register_strategy(
            name="momentum",
            strategy=mock_strategy,
            strategy_type=StrategyType.MOMENTUM
        )
        
        assert "momentum" in self.manager.adapters
        assert self.manager.strategy_weights["momentum"] == 1.0
    
    def test_generate_all_signals(self):
        """Test generating signals from all strategies."""
        self.manager.register_strategy(
            "momentum", MockMomentumStrategy(), StrategyType.MOMENTUM
        )
        self.manager.register_strategy(
            "mean_reversion", MockMeanReversionStrategy(), StrategyType.MEAN_REVERSION
        )
        
        prices = [100.0] * 60
        signals = self.manager.generate_all_signals(prices, regime="ranging")
        
        assert "momentum" in signals
        assert "mean_reversion" in signals
    
    def test_get_best_signal(self):
        """Test getting best signal from ensemble."""
        self.manager.register_strategy(
            "momentum", MockMomentumStrategy(), StrategyType.MOMENTUM
        )
        
        prices = [100.0] * 60
        signal = self.manager.get_best_signal(prices, regime="trending_up")
        
        assert "action" in signal
        assert "confidence" in signal
        assert "source" in signal
    
    def test_record_outcome_updates_weight(self):
        """Test that recording outcomes updates strategy weights."""
        self.manager.register_strategy(
            "momentum", MockMomentumStrategy(), StrategyType.MOMENTUM
        )
        
        # Record positive outcomes
        for _ in range(10):
            self.manager.record_outcome("momentum", pnl=100.0, regime="trending_up")
        
        # Weight should increase
        assert self.manager.strategy_weights["momentum"] > 1.0
    
    def test_get_best_strategy(self):
        """Test getting best performing strategy."""
        self.manager.register_strategy(
            "momentum", MockMomentumStrategy(), StrategyType.MOMENTUM
        )
        self.manager.register_strategy(
            "mean_reversion", MockMeanReversionStrategy(), StrategyType.MEAN_REVERSION
        )
        
        # Make momentum perform better
        for _ in range(10):
            self.manager.record_outcome("momentum", pnl=100.0, regime="trending_up")
        
        for _ in range(10):
            self.manager.record_outcome("mean_reversion", pnl=10.0, regime="trending_up")
        
        best = self.manager.get_best_strategy("trending_up")
        assert best == "momentum"
    
    def test_get_all_stats(self):
        """Test getting stats for all strategies."""
        self.manager.register_strategy(
            "momentum", MockMomentumStrategy(), StrategyType.MOMENTUM
        )
        
        self.manager.record_outcome("momentum", pnl=50.0, regime="ranging")
        
        stats = self.manager.get_all_stats()
        
        assert "momentum" in stats
        assert stats["momentum"]["total_trades"] == 1


class TestGlobalFunctions:
    """Test suite for global functions."""
    
    def test_wire_all_strategies(self):
        """Test wire_all_strategies returns valid manager."""
        mock_orchestrator = MagicMock()
        manager = wire_all_strategies(mock_orchestrator)
        
        assert manager is not None
        assert isinstance(manager, StrategyLearningManager)
        assert manager.learning_orchestrator == mock_orchestrator


class TestIntegrationWithLearningOrchestrator:
    """Integration tests with LearningOrchestrator."""
    
    def test_adapter_uses_learned_parameters(self):
        """Test that adapter uses parameters from orchestrator."""
        from learning.learning_orchestrator import LearningOrchestrator
        
        orchestrator = LearningOrchestrator()
        mock_strategy = MockMomentumStrategy()
        
        adapter = StrategyLearningAdapter(
            strategy=mock_strategy,
            strategy_type=StrategyType.MOMENTUM,
            learning_orchestrator=orchestrator,
            name="test"
        )
        
        # Generate signal (should use learned params)
        prices = [100.0] * 60
        signal = adapter.generate_signal(prices, regime="trending_up")
        
        # Record outcome
        adapter.record_outcome(pnl=100.0, regime="trending_up")
        
        # Orchestrator should have recorded it
        assert orchestrator.state.total_updates == 1
    
    def test_full_workflow(self):
        """Test complete workflow: generate → trade → learn → adapt."""
        from learning.learning_orchestrator import LearningOrchestrator
        
        orchestrator = LearningOrchestrator()
        orchestrator.enable_market_speed()
        
        manager = StrategyLearningManager(orchestrator)
        manager.register_strategy("momentum", MockMomentumStrategy(), StrategyType.MOMENTUM)
        manager.register_strategy("mean_reversion", MockMeanReversionStrategy(), StrategyType.MEAN_REVERSION)
        
        # Simulate multiple trades
        prices = [100.0] * 60
        for i in range(20):
            # Generate signals
            signals = manager.generate_all_signals(prices, regime="trending_up")
            
            # Simulate trade outcome
            pnl = np.random.uniform(-50, 100)
            
            # Record outcome for both strategies
            manager.record_outcome("momentum", pnl, "trending_up")
            manager.record_outcome("mean_reversion", pnl * 0.8, "trending_up")
            
            # Add some price movement
            prices.append(prices[-1] + np.random.uniform(-1, 1))
            prices = prices[-60:]  # Keep bounded
        
        # Verify learning occurred
        assert orchestrator.state.total_updates == 40  # 20 cycles * 2 strategies
        stats = manager.get_all_stats()
        assert stats["momentum"]["total_trades"] == 20
        assert stats["mean_reversion"]["total_trades"] == 20
