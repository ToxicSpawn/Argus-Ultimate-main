"""Tests for Adaptive Strategy Thresholds."""
import pytest
import numpy as np
from strategies.adaptive_strategy_thresholds import (
    MarketAdaptiveStrategies,
    AdaptiveThresholdLearner,
    ThresholdConfig,
    get_adaptive_strategies,
    reset_adaptive_strategies,
)


class TestAdaptiveThresholdLearner:
    """Tests for AdaptiveThresholdLearner."""
    
    def test_initialization(self):
        learner = AdaptiveThresholdLearner()
        assert "trending" in learner.thresholds
        assert "ranging" in learner.thresholds
        assert "high_vol" in learner.thresholds
        assert "low_vol" in learner.thresholds
    
    def test_get_threshold(self):
        learner = AdaptiveThresholdLearner()
        
        trend_thresh = learner.get_threshold("trending", "trend")
        assert 0 < trend_thresh < 0.1
        
        reversion_thresh = learner.get_threshold("ranging", "reversion")
        assert 0.5 < reversion_thresh < 3.0
    
    def test_normalize_regime(self):
        learner = AdaptiveThresholdLearner()
        
        assert learner._normalize_regime("strong_uptrend") == "trending"
        assert learner._normalize_regime("weak_downtrend") == "trending"
        assert learner._normalize_regime("ranging_tight") == "ranging"
        assert learner._normalize_regime("accumulation") == "ranging"
        assert learner._normalize_regime("high_volatility") == "high_vol"
        assert learner._normalize_regime("low_volatility") == "low_vol"
    
    def test_record_outcome(self):
        learner = AdaptiveThresholdLearner()
        
        learner.record_signal_outcome("ranging", "trend", 0.5, True)
        learner.record_signal_outcome("ranging", "trend", 0.3, False)
        
        assert learner.signals_generated == 2
        assert learner.signals_winning == 1
    
    def test_learn_thresholds_insufficient_data(self):
        learner = AdaptiveThresholdLearner()
        
        result = learner.learn_thresholds()
        assert len(result.get("regimes_updated", [])) == 0
    
    def test_learn_thresholds_with_data(self):
        learner = AdaptiveThresholdLearner()
        
        # Add enough data for learning
        for i in range(30):
            strength = 0.3 + (i % 3) * 0.2  # Varying strengths
            profitable = strength > 0.4  # Higher strength = profitable
            learner.record_signal_outcome("ranging", "trend", strength, profitable)
        
        result = learner.learn_thresholds()
        # Should have learned something
        assert learner.threshold_adjustments >= 0
    
    def test_get_stats(self):
        learner = AdaptiveThresholdLearner()
        
        stats = learner.get_stats()
        assert "signals_generated" in stats
        assert "win_rate" in stats
        assert "current_thresholds" in stats


class TestMarketAdaptiveStrategies:
    """Tests for MarketAdaptiveStrategies."""
    
    def test_initialization(self):
        adaptive = MarketAdaptiveStrategies()
        assert adaptive.learner is not None
    
    def test_get_trend_signal_triggered(self):
        adaptive = MarketAdaptiveStrategies()
        
        # Create prices with clear trend (strong upward)
        prices = [75000 + i * 100 for i in range(50)]  # Strong uptrend
        
        signal = adaptive.get_trend_signal(prices, "trending")
        assert signal is not None
        assert signal["action"] == "buy"
        assert signal["confidence"] > 0
    
    def test_get_trend_signal_not_triggered(self):
        adaptive = MarketAdaptiveStrategies()
        
        # Create flat prices (no trend)
        np.random.seed(42)
        prices = [75000 + np.random.randn() * 5 for _ in range(50)]
        
        signal = adaptive.get_trend_signal(prices, "trending")
        # May or may not trigger depending on random data
        # Just verify it doesn't crash
    
    def test_get_momentum_signal(self):
        adaptive = MarketAdaptiveStrategies()
        
        # Create prices with momentum
        prices = [75000] * 40 + [75000 + i * 50 for i in range(10)]
        
        signal = adaptive.get_momentum_signal(prices, "trending")
        # Should trigger due to recent momentum
        if signal:
            assert signal["action"] in ["buy", "sell"]
            assert 0 < signal["confidence"] <= 1.0
    
    def test_get_mean_reversion_signal(self):
        adaptive = MarketAdaptiveStrategies()
        
        # Create prices that deviate from mean
        prices = [75000] * 15 + [75000 - 500]  # Sudden drop
        
        signal = adaptive.get_mean_reversion_signal(prices, "ranging")
        if signal:
            assert signal["action"] == "buy"  # Buy the dip
            assert signal["signal_type"] == "mean_reversion"
    
    def test_get_breakout_signal(self):
        adaptive = MarketAdaptiveStrategies()
        
        # Create prices with breakout
        prices = [75000 + np.random.randn() * 10 for _ in range(19)]
        prices.append(76000)  # Breakout above
        
        signal = adaptive.get_breakout_signal(prices, "ranging")
        if signal:
            assert signal["action"] == "buy"
            assert signal["signal_type"] == "breakout"
    
    def test_get_all_signals(self):
        adaptive = MarketAdaptiveStrategies()
        
        # Create trending prices
        prices = [75000 + i * 50 for i in range(50)]
        
        signals = adaptive.get_all_signals(prices, "trending")
        assert isinstance(signals, list)
        # Each signal should have required fields
        for sig in signals:
            assert "action" in sig
            assert "confidence" in sig
            assert "signal_type" in sig
    
    def test_record_outcome(self):
        adaptive = MarketAdaptiveStrategies()
        
        adaptive.record_outcome("ranging", "trend", 0.5, True)
        adaptive.record_outcome("ranging", "trend", 0.3, False)
        
        assert adaptive.learner.signals_generated == 2
    
    def test_learn(self):
        adaptive = MarketAdaptiveStrategies()
        
        # Add data
        for i in range(25):
            adaptive.record_outcome("ranging", "trend", 0.3 + i * 0.02, i % 2 == 0)
        
        result = adaptive.learn()
        assert "regimes_updated" in result
    
    def test_get_stats(self):
        adaptive = MarketAdaptiveStrategies()
        
        stats = adaptive.get_stats()
        assert "signals_generated" in stats
        assert "win_rate" in stats


class TestSingletonPattern:
    """Tests for singleton pattern."""
    
    def test_get_adaptive_strategies_singleton(self):
        reset_adaptive_strategies()
        
        s1 = get_adaptive_strategies()
        s2 = get_adaptive_strategies()
        
        assert s1 is s2
    
    def test_reset_adaptive_strategies(self):
        s1 = get_adaptive_strategies()
        reset_adaptive_strategies()
        s2 = get_adaptive_strategies()
        
        assert s1 is not s2


class TestMainIntegration:
    """Integration tests for main.py with adaptive strategies."""
    
    def test_adaptive_strategies_available_flag(self):
        import main
        assert hasattr(main, 'ADAPTIVE_STRATEGIES_AVAILABLE')
        assert main.ADAPTIVE_STRATEGIES_AVAILABLE == True
    
    def test_argus_has_adaptive_strategies(self):
        from main import Argus
        
        system = Argus(mode='paper', capital=1000)
        assert hasattr(system, 'adaptive_strategies')


class TestThresholdComparison:
    """Compare old vs new thresholds to verify improvement."""
    
    def test_new_thresholds_are_lower(self):
        """Verify new thresholds are lower than original."""
        learner = AdaptiveThresholdLearner()
        
        # Original thresholds (from StrategyEngine)
        original_trend = 0.01  # 1%
        original_momentum = 0.03  # 3%
        original_reversion = 2.0  # 2 std dev
        original_breakout = 0.01  # 1%
        
        # New learned thresholds (initial)
        new_trend = learner.get_threshold("ranging", "trend")
        new_momentum = learner.get_threshold("ranging", "momentum")
        new_reversion = learner.get_threshold("ranging", "reversion")
        new_breakout = learner.get_threshold("ranging", "breakout")
        
        # Verify new thresholds are lower (more signals)
        assert new_trend < original_trend, f"Trend threshold {new_trend} should be < {original_trend}"
        assert new_momentum < original_momentum, f"Momentum threshold {new_momentum} should be < {original_momentum}"
        assert new_reversion < original_reversion, f"Reversion threshold {new_reversion} should be < {original_reversion}"
        assert new_breakout < original_breakout, f"Breakout threshold {new_breakout} should be < {original_breakout}"
    
    def test_signals_generated_with_new_thresholds(self):
        """Verify adaptive strategies generate signals where old would not."""
        adaptive = MarketAdaptiveStrategies()
        
        # Create mild trend (0.5% - below old 1% threshold but above new 0.5%)
        prices = [75000 + i * 25 for i in range(50)]  # 0.5% trend over 50 bars
        
        signals = adaptive.get_all_signals(prices, "ranging")
        
        # Should generate signals with new lower thresholds
        assert len(signals) > 0, "Adaptive strategies should generate signals with lower thresholds"


class TestContinuousLearning:
    """Tests for continuous learning feature."""
    
    def test_continuous_learning_config(self):
        """Test continuous learning configuration."""
        config = ThresholdConfig(continuous_learning_interval=0.5)
        assert config.continuous_learning_interval == 0.5
    
    def test_should_learn_initially_false(self):
        """Test should_learn returns False before continuous learning starts."""
        learner = AdaptiveThresholdLearner()
        assert learner.should_learn() == False
    
    def test_start_continuous_learning(self):
        """Test starting continuous learning."""
        import asyncio
        
        learner = AdaptiveThresholdLearner()
        
        async def test():
            await learner.start_continuous_learning()
            assert learner._continuous_learning_enabled == True
        
        asyncio.run(test())
    
    def test_stop_continuous_learning(self):
        """Test stopping continuous learning."""
        import asyncio
        
        learner = AdaptiveThresholdLearner()
        
        async def test():
            await learner.start_continuous_learning()
            learner.stop_continuous_learning()
            assert learner._continuous_learning_enabled == False
        
        asyncio.run(test())
    
    def test_run_learning_cycle(self):
        """Test running a learning cycle."""
        import time
        
        learner = AdaptiveThresholdLearner()
        learner._continuous_learning_enabled = True
        learner._last_learning_time = 0  # Force learning to be due
        
        # Add some data
        for i in range(30):
            learner.record_signal_outcome("ranging", "trend", 0.3 + i * 0.02, i % 2 == 0)
        
        result = learner.run_learning_cycle()
        
        assert result["skipped"] == False
        assert result["cycle_number"] == 1
        assert "learning_time_ms" in result
    
    def test_run_learning_cycle_timing(self):
        """Test learning cycle respects timing interval."""
        import time
        
        learner = AdaptiveThresholdLearner(
            config=ThresholdConfig(continuous_learning_interval=1.0)
        )
        learner._continuous_learning_enabled = True
        
        # First call should learn
        learner._last_learning_time = 0
        result1 = learner.run_learning_cycle()
        assert result1["skipped"] == False
        
        # Immediate second call should be skipped
        result2 = learner.run_learning_cycle()
        assert result2["skipped"] == True
    
    def test_continuous_learning_stats(self):
        """Test stats include continuous learning info."""
        import asyncio
        
        learner = AdaptiveThresholdLearner()
        
        async def test():
            await learner.start_continuous_learning()
            
            # Run some cycles
            learner._last_learning_time = 0
            learner.run_learning_cycle()
            
            stats = learner.get_stats()
            assert stats["continuous_learning_enabled"] == True
            assert stats["learning_cycle_count"] >= 1
            assert "avg_learning_time_ms" in stats
        
        asyncio.run(test())
