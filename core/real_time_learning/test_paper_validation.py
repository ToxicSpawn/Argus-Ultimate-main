"""
Test cases for PaperValidationEngine
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone
from core.real_time_learning.paper_validation import PaperValidationEngine, ValidationResult


@pytest.fixture
def validation_engine():
    """Create a test validation engine"""
    return PaperValidationEngine()


def test_initialization(validation_engine):
    """Test that validation engine is properly initialized"""
    assert len(validation_engine.component_thresholds) == 4
    assert 'strategy_allocator' in validation_engine.component_thresholds
    assert 'correlation_matrix' in validation_engine.component_thresholds
    assert 'order_router' in validation_engine.component_thresholds
    assert 'regime_parameters' in validation_engine.component_thresholds


def test_validation_result(validation_engine):
    """Test ValidationResult class"""
    result = ValidationResult(
        component="strategy_allocator",
        parameter_changes={"strategy_weights": {"momentum": 0.4, "mean_reversion": 0.6}},
        test_passed=True,
        metrics={
            "sharpe_ratio": 2.0,
            "win_rate": 0.6,
            "max_drawdown": 0.15
        },
        required_metrics={
            "sharpe_ratio": (1.2, ">"),
            "win_rate": (0.5, ">="),
            "max_drawdown": (0.2, "<")
        },
        statistical_results={
            "sharpe_improvement": {
                "test": "one_sample_t_test",
                "statistic": 3.2,
                "p_value": 0.001,
                "significant": True,
                "threshold": 0.05
            }
        }
    )
    
    assert result.is_valid() == True
    assert result.component == "strategy_allocator"
    assert result.test_passed == True
    assert result.metrics["sharpe_ratio"] == 2.0
    assert result.all_statistical_tests_passed() == True
    
    # Test failure case
    bad_result = ValidationResult(
        component="strategy_allocator",
        parameter_changes={"strategy_weights": {"momentum": 0.4, "mean_reversion": 0.6}},
        test_passed=False,
        metrics={"sharpe_ratio": 1.0},  # Below threshold
        required_metrics={"sharpe_ratio": (1.2, ">")},
        statistical_results={
            "sharpe_improvement": {
                "test": "one_sample_t_test",
                "statistic": 1.2,
                "p_value": 0.25,  # Not significant
                "significant": False,
                "threshold": 0.05
            }
        }
    )
    
    assert bad_result.is_valid() == False
    assert bad_result.all_statistical_tests_passed() == False
    failure_reasons = bad_result.get_failure_reasons()
    assert "sharpe_ratio (1.000) not > 1.200" in failure_reasons
    assert any("sharpe_improvement not statistically significant" in reason for reason in failure_reasons)


def test_strategy_allocator_validation(validation_engine):
    """Test validation for strategy allocator changes"""
    # Create test data
    data = {
        'component_name': 'strategy_allocator',
        'proposed_changes': {
            'strategy_weights': {
                'momentum': 0.4,
                'mean_reversion': 0.4,
                'breakout': 0.2
            }
        },
        'market_data': {
            'regime': 'volatile'
        }
    }
    
    # Mock the backtest results
    backtest_results = {
        'sharpe_ratio': 2.0,
        'win_rate': 0.6,
        'max_drawdown': 0.15,
        'profit_factor': 2.0,
        'total_trades': 50,
        'original_sharpe': 1.8,
        'original_drawdown': 0.17
    }
    
    # Mock the _run_paper_trading method
    validation_engine._run_paper_trading = MagicMock(return_value=backtest_results)
    
    # Run validation
    result = validation_engine.learn(data)
    
    # Check results
    assert result["status"] == "success"
    validation_result = result["validation_result"]
    assert validation_result.is_valid() == True
    assert validation_result.metrics["sharpe_ratio"] == 2.0
    assert abs(validation_result.metrics["sharpe_improvement"] - 0.2) < 0.001
    assert abs(validation_result.metrics["drawdown_change"] - (-0.02)) < 0.001
    assert validation_result.all_statistical_tests_passed() == True


def test_correlation_matrix_validation(validation_engine):
    """Test validation for correlation matrix changes"""
    # Create test data
    data = {
        'component_name': 'correlation_matrix',
        'proposed_changes': {
            'current_matrix': {
                'BTC/USDT:ETH/USDT': 0.7,
                'BTC/USDT:SOL/USDT': 0.5,
                'ETH/USDT:SOL/USDT': 0.6
            }
        },
        'market_data': {
            'regime': 'stable'
        }
    }
    
    # Mock the backtest results
    backtest_results = {
        'diversification_score': 0.75,
        'sharpe_ratio': 2.1,
        'max_drawdown': 0.12,
        'portfolio_concentration': 0.3,
        'total_trades': 40,
        'original_diversification': 0.65
    }
    
    # Mock the _run_paper_trading method
    validation_engine._run_paper_trading = MagicMock(return_value=backtest_results)
    
    # Run validation
    result = validation_engine.learn(data)
    
    # Check results
    assert result["status"] == "success"
    validation_result = result["validation_result"]
    assert validation_result.is_valid() == True
    assert validation_result.metrics["diversification_score"] == 0.75
    assert abs(validation_result.metrics["diversification_improvement"] - 0.1) < 0.001
    assert validation_result.all_statistical_tests_passed() == True


def test_order_router_validation(validation_engine):
    """Test validation for order router changes"""
    # Create test data
    data = {
        'component_name': 'order_router',
        'proposed_changes': {
            'venue_performance': {
                'binance': {'fill_ratio': 0.9, 'avg_slippage': 0.0008, 'latency': 35},
                'bybit': {'fill_ratio': 0.88, 'avg_slippage': 0.0009, 'latency': 40}
            }
        },
        'market_data': {
            'regime': 'trending'
        }
    }
    
    # Mock the backtest results
    backtest_results = {
        'avg_fill_ratio': 0.89,
        'avg_slippage': 0.00085,
        'avg_latency': 37,
        'execution_quality_score': 0.85,
        'total_trades': 30,
        'original_fill': 0.85,
        'original_slippage': 0.001,
        'original_latency': 45
    }
    
    # Mock the _run_paper_trading method
    validation_engine._run_paper_trading = MagicMock(return_value=backtest_results)
    
    # Run validation
    result = validation_engine.learn(data)
    
    # Check results
    assert result["status"] == "success"
    validation_result = result["validation_result"]
    assert validation_result.is_valid() == True
    assert validation_result.metrics["avg_fill_ratio"] == 0.89
    assert abs(validation_result.metrics["fill_improvement"] - 0.04) < 0.001
    assert validation_result.all_statistical_tests_passed() == True


def test_regime_parameters_validation(validation_engine):
    """Test validation for regime parameters changes"""
    # Create test data
    data = {
        'component_name': 'regime_parameters',
        'proposed_changes': {
            'current_parameters': {
                'position_size_pct': 0.04,
                'max_leverage': 1.5,
                'take_profit_pct': 0.06,
                'stop_loss_pct': 0.03
            }
        },
        'market_data': {
            'regime': 'volatile'
        }
    }
    
    # Mock the backtest results
    backtest_results = {
        'sharpe_ratio': 2.2,
        'win_rate': 0.6,
        'max_drawdown': 0.13,
        'profit_factor': 2.1,
        'total_trades': 50,
        'original_sharpe': 1.8,
        'original_drawdown': 0.17
    }
    
    # Mock the _run_paper_trading method
    validation_engine._run_paper_trading = MagicMock(return_value=backtest_results)
    
    # Run validation
    result = validation_engine.learn(data)
    
    # Check results
    assert result["status"] == "success"
    validation_result = result["validation_result"]
    assert validation_result.is_valid() == True
    assert validation_result.metrics["sharpe_ratio"] == 2.2
    assert abs(validation_result.metrics["sharpe_improvement"] - 0.4) < 0.001
    assert abs(validation_result.metrics["drawdown_change"] - (-0.04)) < 0.001
    assert validation_result.all_statistical_tests_passed() == True


def test_insufficient_trades(validation_engine):
    """Test validation failure due to insufficient trades"""
    # Create test data with too few trades
    data = {
        'component_name': 'strategy_allocator',
        'proposed_changes': {
            'strategy_weights': {
                'momentum': 0.4,
                'mean_reversion': 0.6
            }
        },
        'market_data': {
            'regime': 'stable'
        }
    }
    
    # Mock the backtest results with insufficient trades
    backtest_results = {
        'sharpe_ratio': 2.0,
        'win_rate': 0.6,
        'max_drawdown': 0.15,
        'profit_factor': 2.0,
        'total_trades': 10,  # Too few trades
        'original_sharpe': 1.8,
        'original_drawdown': 0.17
    }
    
    # Mock the _run_paper_trading method
    validation_engine._run_paper_trading = MagicMock(return_value=backtest_results)
    
    # Run validation
    result = validation_engine.learn(data)
    
    # Check results
    assert result["status"] == "failed"
    validation_result = result["validation_result"]
    assert validation_result.is_valid() == False
    failure_reasons = validation_result.get_failure_reasons()
    assert any("total_trades (10.000) not >= 30.000" in reason for reason in failure_reasons)


def test_insignificant_improvement(validation_engine):
    """Test validation failure due to statistically insignificant improvement"""
    # Create test data
    data = {
        'component_name': 'strategy_allocator',
        'proposed_changes': {
            'strategy_weights': {
                'momentum': 0.4,
                'mean_reversion': 0.6
            }
        },
        'market_data': {
            'regime': 'stable'
        }
    }
    
    # Mock the backtest results with insignificant improvement
    # We need to mock the statistical test directly to force an insignificant result
    backtest_results = {
        'sharpe_ratio': 1.9,  # Meets threshold
        'win_rate': 0.6,
        'max_drawdown': 0.15,  # Meets threshold
        'profit_factor': 1.9,
        'total_trades': 50,  # Meets minimum trades
        'original_sharpe': 1.8,
        'original_drawdown': 0.17
    }
    
    # Mock the _run_paper_trading method
    validation_engine._run_paper_trading = MagicMock(return_value=backtest_results)
    
    # Mock the statistical test to return insignificant result
    original_statistical_tests = validation_engine._run_statistical_tests
    def mock_statistical_tests(component_name, metrics, backtest_results):
        return {
            'sharpe_improvement': {
                'test': 'one_sample_t_test',
                'statistic': 1.5,
                'p_value': 0.15,  # Not significant (p > 0.05)
                'significant': False,
                'threshold': 0.05
            }
        }
    
    validation_engine._run_statistical_tests = mock_statistical_tests
    
    # Run validation
    result = validation_engine.learn(data)
    
    # Restore original method
    validation_engine._run_statistical_tests = original_statistical_tests
    
    # Check results
    assert result["status"] == "failed"
    validation_result = result["validation_result"]
    assert validation_result.is_valid() == False
    failure_reasons = validation_result.get_failure_reasons()
    print("Failure reasons:", failure_reasons)
    assert any("sharpe_improvement not statistically significant" in reason for reason in failure_reasons)


def test_cache_functionality(validation_engine):
    """Test backtest result caching"""
    # Create test data
    data = {
        'component_name': 'strategy_allocator',
        'proposed_changes': {
            'strategy_weights': {
                'momentum': 0.4,
                'mean_reversion': 0.6
            }
        },
        'market_data': {
            'regime': 'stable'
        }
    }
    
    # Mock the backtest results
    backtest_results = {
        'sharpe_ratio': 2.0,
        'win_rate': 0.6,
        'max_drawdown': 0.15,
        'profit_factor': 2.0,
        'total_trades': 50,
        'original_sharpe': 1.8,
        'original_drawdown': 0.17
    }
    
    # Mock the _run_paper_trading method
    validation_engine._run_paper_trading = MagicMock(return_value=backtest_results)
    
    # Run validation twice with same parameters
    result1 = validation_engine.learn(data)
    result2 = validation_engine.learn(data)
    
    # Check that cache was used on second call
    assert validation_engine._run_paper_trading.call_count == 1
    assert result1["status"] == "success"
    assert result2["status"] == "success"
    
    # Verify results are identical
    assert result1["validation_result"].metrics["sharpe_ratio"] == result2["validation_result"].metrics["sharpe_ratio"]