"""
Test Walk-Forward Validator Module
====================================
Tests for walk-forward validation framework.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from validation.walk_forward_validator import (
    ValidationConfig,
    TradeResult,
    WindowResult,
    PerformanceMetrics,
    StrategySimulator,
    WalkForwardValidator,
)


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics."""
    
    def test_sharpe_ratio_positive(self):
        # Positive returns
        returns = [0.01] * 50  # Consistent 1% returns
        sharpe = PerformanceMetrics.calculate_sharpe(returns)
        
        assert sharpe > 0
    
    def test_sharpe_ratio_negative(self):
        # Negative returns (consistent losses)
        returns = [-0.005 + np.random.randn() * 0.001 for _ in range(50)]
        sharpe = PerformanceMetrics.calculate_sharpe(returns)
        
        assert sharpe < 0
    
    def test_sharpe_ratio_zero_std(self):
        # Zero volatility
        returns = [0.0] * 50
        sharpe = PerformanceMetrics.calculate_sharpe(returns)
        
        assert sharpe == 0.0
    
    def test_sortino_ratio(self):
        returns = [0.02, 0.01, -0.01, 0.03, 0.01] * 10
        sortino = PerformanceMetrics.calculate_sortino(returns)
        
        assert isinstance(sortino, float)
    
    def test_max_drawdown(self):
        equity = [10000, 10500, 10200, 10800, 10300, 11000]
        dd = PerformanceMetrics.calculate_max_drawdown(equity)
        
        assert dd > 0
        assert dd < 1.0  # Should be less than 100%
    
    def test_max_drawdown_empty(self):
        dd = PerformanceMetrics.calculate_max_drawdown([])
        assert dd == 0.0
    
    def test_win_rate(self):
        trades = [
            TradeResult(0, "sell", 100, 110, 1, 10, 0.1, 1, "trending"),
            TradeResult(0, "sell", 100, 95, 1, -5, -0.05, 1, "trending"),
            TradeResult(0, "sell", 100, 105, 1, 5, 0.05, 1, "trending"),
        ]
        
        win_rate = PerformanceMetrics.calculate_win_rate(trades)
        
        assert win_rate == 2/3
    
    def test_win_rate_empty(self):
        win_rate = PerformanceMetrics.calculate_win_rate([])
        assert win_rate == 0.0
    
    def test_total_return(self):
        trades = [
            TradeResult(0, "sell", 100, 110, 10, 100, 0.1, 1, "trending"),
            TradeResult(0, "sell", 100, 95, 10, -50, -0.05, 1, "trending"),
        ]
        
        total_return = PerformanceMetrics.calculate_total_return(trades, 10000)
        
        assert total_return == 0.005  # (100 - 50) / 10000


class TestStrategySimulator:
    """Test suite for StrategySimulator."""
    
    def setup_method(self):
        self.simulator = StrategySimulator(initial_capital=10000.0)
    
    def test_buy_sell_cycle(self):
        prices = [100.0, 101.0, 102.0, 103.0, 102.0, 101.0]
        signals = [
            {"action": "buy", "confidence": 0.8},
            {"action": "hold", "confidence": 0.0},
            {"action": "hold", "confidence": 0.0},
            {"action": "hold", "confidence": 0.0},
            {"action": "sell", "confidence": 0.8},
            {"action": "hold", "confidence": 0.0},
        ]
        
        trades, equity = self.simulator.simulate(prices, signals)
        
        assert len(trades) == 1
        assert trades[0].pnl > 0  # Should be profitable
    
    def test_loss_trade(self):
        prices = [100.0, 99.0, 98.0, 97.0, 96.0, 95.0]
        signals = [
            {"action": "buy", "confidence": 0.8},
            {"action": "hold", "confidence": 0.0},
            {"action": "hold", "confidence": 0.0},
            {"action": "hold", "confidence": 0.0},
            {"action": "sell", "confidence": 0.8},
            {"action": "hold", "confidence": 0.0},
        ]
        
        trades, equity = self.simulator.simulate(prices, signals)
        
        assert len(trades) == 1
        assert trades[0].pnl < 0  # Should be loss
    
    def test_no_trades(self):
        prices = [100.0] * 10
        signals = [{"action": "hold", "confidence": 0.0}] * 10
        
        trades, equity = self.simulator.simulate(prices, signals)
        
        assert len(trades) == 0
    
    def test_equity_curve(self):
        prices = [100.0, 101.0, 102.0, 103.0]
        signals = [
            {"action": "buy", "confidence": 0.5},
            {"action": "hold", "confidence": 0.0},
            {"action": "hold", "confidence": 0.0},
            {"action": "hold", "confidence": 0.0},
        ]
        
        trades, equity = self.simulator.simulate(prices, signals)
        
        assert len(equity) > 0
        assert equity[0] == 10000.0  # Initial capital


class TestWalkForwardValidator:
    """Test suite for WalkForwardValidator."""
    
    def setup_method(self):
        self.config = ValidationConfig(
            train_days=30,
            validation_days=10,
            step_days=10,
            min_train_samples=100,
            min_validation_samples=50,
        )
        self.validator = WalkForwardValidator(self.config)
    
    def test_initialization(self):
        assert self.validator is not None
    
    def test_validation_with_profitable_strategy(self):
        # Generate synthetic data with uptrend
        np.random.seed(42)
        n_bars = 1000
        prices = [100.0]
        for _ in range(n_bars - 1):
            prices.append(prices[-1] * (1 + np.random.randn() * 0.005 + 0.001))
        
        # Simple trend-following signal generator
        def signal_generator(prices, regimes):
            signals = []
            for i in range(len(prices)):
                if i < 20:
                    signals.append({"action": "hold", "confidence": 0.0})
                    continue
                
                ma = np.mean(prices[i-20:i])
                if prices[i] > ma:
                    signals.append({"action": "buy", "confidence": 0.7, "signal_type": "trend"})
                else:
                    signals.append({"action": "sell", "confidence": 0.7, "signal_type": "trend"})
            
            return signals
        
        results = self.validator.validate(
            prices=prices,
            signal_generator=signal_generator,
        )
        
        assert results.total_windows > 0
        assert isinstance(results.overall_passed, bool)
    
    def test_validation_report(self):
        # Create a mock summary
        summary = type('obj', (object,), {
            'total_windows': 10,
            'profitable_windows': 7,
            'profitability_rate': 0.7,
            'avg_train_sharpe': 1.5,
            'avg_val_sharpe': 1.2,
            'avg_train_win_rate': 0.55,
            'avg_val_win_rate': 0.52,
            'avg_max_drawdown': 0.12,
            'passed_min_sharpe': True,
            'passed_profitability_rate': True,
            'passed_drawdown': True,
            'passed_degradation': True,
            'overall_passed': True,
            'window_results': [],
        })()
        
        report = self.validator.get_report(summary)
        
        assert "WALK-FORWARD VALIDATION REPORT" in report
        assert "STRATEGY VALIDATED" in report
    
    def test_validation_report_failure(self):
        summary = type('obj', (object,), {
            'total_windows': 10,
            'profitable_windows': 3,
            'profitability_rate': 0.3,
            'avg_train_sharpe': 0.5,
            'avg_val_sharpe': 0.3,
            'avg_train_win_rate': 0.45,
            'avg_val_win_rate': 0.40,
            'avg_max_drawdown': 0.35,
            'passed_min_sharpe': False,
            'passed_profitability_rate': False,
            'passed_drawdown': False,
            'passed_degradation': True,
            'overall_passed': False,
            'window_results': [],
        })()
        
        report = self.validator.get_report(summary)
        
        assert "STRATEGY FAILED" in report
