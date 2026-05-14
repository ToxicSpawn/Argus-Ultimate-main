"""Tests for Learning Risk Manager."""
import pytest
import numpy as np
from collections import deque
from learning.learning_risk_manager import (
    LearningRiskManager,
    RiskLearningConfig,
    TradeOutcome,
    VolatilityRiskScaler,
    StopLossOptimizer,
    PositionSizingLearner,
)


class TestVolatilityRiskScaler:
    """Tests for VolatilityRiskScaler."""
    
    def test_initialization(self):
        scaler = VolatilityRiskScaler(window=50)
        assert scaler.window == 50
        assert len(scaler.returns) == 0
    
    def test_volatility_score_default(self):
        scaler = VolatilityRiskScaler()
        # With no data, should return 0.5
        assert scaler.get_volatility_score() == 0.5
    
    def test_update_and_score(self):
        scaler = VolatilityRiskScaler(window=50)
        
        # Add stable prices (low volatility)
        base_price = 75000
        for i in range(60):
            price = base_price + np.random.randn() * 10  # Very small moves
            prev_price = base_price + np.random.randn() * 10 if i > 0 else base_price
            scaler.update(price, prev_price)
        
        score = scaler.get_volatility_score()
        assert 0.0 <= score <= 1.0
    
    def test_risk_multiplier_low_vol(self):
        scaler = VolatilityRiskScaler()
        scaler.returns = [0.001] * 50  # Low volatility
        multiplier = scaler.get_risk_multiplier()
        assert multiplier >= 1.0  # Should increase risk in low vol
    
    def test_risk_multiplier_high_vol(self):
        scaler = VolatilityRiskScaler()
        # Add varying high volatility returns
        np.random.seed(42)
        scaler.returns = deque([abs(np.random.randn() * 0.05) for _ in range(50)], maxlen=50)
        multiplier = scaler.get_risk_multiplier()
        # High volatility should reduce risk multiplier
        assert multiplier <= 1.0  # Should reduce or maintain risk in high vol


class TestStopLossOptimizer:
    """Tests for StopLossOptimizer."""
    
    def test_initialization(self):
        config = RiskLearningConfig()
        optimizer = StopLossOptimizer(config)
        assert "trending" in optimizer.regime_stops
        assert "ranging" in optimizer.regime_stops
    
    def test_get_optimal_stop(self):
        config = RiskLearningConfig()
        optimizer = StopLossOptimizer(config)
        
        stop = optimizer.get_optimal_stop("trending", volatility=0.5)
        assert config.min_stop_loss_pct <= stop <= config.max_stop_loss_pct
    
    def test_record_trade(self):
        config = RiskLearningConfig()
        optimizer = StopLossOptimizer(config)
        
        outcome = TradeOutcome(
            timestamp=1000.0,
            entry_price=75000.0,
            exit_price=76000.0,
            position_size=1000.0,
            stop_loss=74000.0,
            take_profit=78000.0,
            pnl=100.0,
            pnl_pct=0.01,
            regime="trending",
            volatility=0.5,
            duration_seconds=3600.0,
            was_stopped_out=False,
            was_take_profit=False,
        )
        
        optimizer.record_trade(outcome)
        assert len(optimizer.outcomes) == 1
    
    def test_learning_no_data(self):
        config = RiskLearningConfig()
        optimizer = StopLossOptimizer(config)
        
        result = optimizer.learn_from_outcomes()
        assert result["updates"] == 0


class TestPositionSizingLearner:
    """Tests for PositionSizingLearner."""
    
    def test_initialization(self):
        config = RiskLearningConfig()
        learner = PositionSizingLearner(config)
        assert learner.learned_position_pct == config.base_position_pct
    
    def test_record_trade_win(self):
        config = RiskLearningConfig()
        learner = PositionSizingLearner(config)
        
        learner.record_trade(pnl=100.0, pnl_pct=0.01)
        assert learner.consecutive_wins == 1
        assert learner.consecutive_losses == 0
    
    def test_record_trade_loss(self):
        config = RiskLearningConfig()
        learner = PositionSizingLearner(config)
        
        learner.record_trade(pnl=-50.0, pnl_pct=-0.005)
        assert learner.consecutive_wins == 0
        assert learner.consecutive_losses == 1
    
    def test_position_multiplier_consecutive_losses(self):
        config = RiskLearningConfig()
        learner = PositionSizingLearner(config)
        
        # Simulate consecutive losses
        for _ in range(4):
            learner.record_trade(pnl=-50.0, pnl_pct=-0.005)
        
        multiplier = learner.get_position_multiplier("trending", confidence=0.8)
        # Should be reduced due to consecutive losses
        assert multiplier < 1.0


class TestLearningRiskManager:
    """Tests for LearningRiskManager."""
    
    def test_initialization(self):
        manager = LearningRiskManager(capital=10000.0)
        assert manager.capital == 10000.0
        assert manager.peak_capital == 10000.0
    
    def test_can_trade_initially(self):
        manager = LearningRiskManager(capital=10000.0)
        can_trade, reason = manager.can_trade("trending")
        assert can_trade == True
        assert reason == "OK"
    
    def test_can_trade_max_drawdown(self):
        config = RiskLearningConfig(base_drawdown_limit=0.10)
        manager = LearningRiskManager(capital=10000.0, config=config)
        manager.capital = 8000.0  # 20% drawdown
        
        can_trade, reason = manager.can_trade("trending")
        assert can_trade == False
        assert "Drawdown" in reason
    
    def test_calculate_position_size(self):
        manager = LearningRiskManager(capital=10000.0)
        
        size = manager.calculate_position_size(
            regime="trending",
            confidence=0.8,
            signal_strength=1.0,
        )
        
        assert size >= 0
        assert size <= manager.capital * manager.config.max_position_pct
    
    def test_calculate_stop_loss_buy(self):
        manager = LearningRiskManager(capital=10000.0)
        
        stop = manager.calculate_stop_loss(
            entry_price=75000.0,
            side="buy",
            regime="trending",
        )
        
        assert stop < 75000.0  # Stop below entry for buy
    
    def test_calculate_stop_loss_sell(self):
        manager = LearningRiskManager(capital=10000.0)
        
        stop = manager.calculate_stop_loss(
            entry_price=75000.0,
            side="sell",
            regime="trending",
        )
        
        assert stop > 75000.0  # Stop above entry for sell
    
    def test_calculate_take_profit(self):
        manager = LearningRiskManager(capital=10000.0)
        
        entry = 75000.0
        stop = 74000.0
        tp = manager.calculate_take_profit(entry, stop, "buy", min_rr=2.0)
        
        risk = entry - stop
        reward = tp - entry
        assert reward >= risk * 2.0  # At least 2:1 R:R
    
    def test_record_trade_win(self):
        manager = LearningRiskManager(capital=10000.0)
        
        outcome = TradeOutcome(
            timestamp=1000.0,
            entry_price=75000.0,
            exit_price=76000.0,
            position_size=1000.0,
            stop_loss=74000.0,
            take_profit=77000.0,
            pnl=100.0,
            pnl_pct=0.01,
            regime="trending",
            volatility=0.5,
            duration_seconds=3600.0,
            was_stopped_out=False,
            was_take_profit=False,
        )
        
        manager.record_trade(outcome)
        assert manager.winning_trades == 1
        assert manager.consecutive_losses == 0
    
    def test_record_trade_loss(self):
        manager = LearningRiskManager(capital=10000.0)
        
        outcome = TradeOutcome(
            timestamp=1000.0,
            entry_price=75000.0,
            exit_price=74000.0,
            position_size=1000.0,
            stop_loss=74000.0,
            take_profit=77000.0,
            pnl=-100.0,
            pnl_pct=-0.01,
            regime="trending",
            volatility=0.5,
            duration_seconds=3600.0,
            was_stopped_out=True,
            was_take_profit=False,
        )
        
        manager.record_trade(outcome)
        assert manager.losing_trades == 1
        assert manager.consecutive_losses == 1
    
    def test_learn_with_insufficient_data(self):
        manager = LearningRiskManager(capital=10000.0)
        
        result = manager.learn()
        # Should return empty results with no data
        assert result.get("stop_learning", {}).get("updates", 0) == 0
    
    def test_update_market_data(self):
        manager = LearningRiskManager(capital=10000.0)
        
        # Update with price data
        manager.update_market_data(75000.0, 74900.0, "trending")
        manager.update_market_data(75100.0, 75000.0, "trending")
        
        # Volatility should be calculated
        vol = manager.volatility_scaler.get_volatility_score()
        assert 0.0 <= vol <= 1.0
    
    def test_get_stats(self):
        manager = LearningRiskManager(capital=10000.0)
        
        stats = manager.get_stats()
        assert stats["capital"] == 10000.0
        assert stats["trade_count"] == 0
        assert "learned_position_pct" in stats
        assert "learned_stop_losses" in stats
        assert "volatility_score" in stats
    
    def test_consecutive_loss_threshold(self):
        config = RiskLearningConfig()
        manager = LearningRiskManager(capital=10000.0, config=config)
        
        # Simulate 5 consecutive losses
        for _ in range(5):
            outcome = TradeOutcome(
                timestamp=1000.0,
                entry_price=75000.0,
                exit_price=74000.0,
                position_size=1000.0,
                stop_loss=74000.0,
                take_profit=77000.0,
                pnl=-100.0,
                pnl_pct=-0.01,
                regime="trending",
                volatility=0.5,
                duration_seconds=60.0,
                was_stopped_out=True,
                was_take_profit=False,
            )
            manager.record_trade(outcome)
        
        # Should block trading after 5 consecutive losses
        can_trade, reason = manager.can_trade("trending")
        assert can_trade == False
        assert "consecutive losses" in reason


class TestMainIntegration:
    """Integration tests for main.py with Learning Risk Manager."""
    
    def test_learning_risk_available_flag(self):
        import main
        assert hasattr(main, 'LEARNING_RISK_AVAILABLE')
    
    def test_argus_has_learning_risk(self):
        from main import Argus
        
        system = Argus(mode='paper', capital=1000)
        assert hasattr(system, 'learning_risk')
