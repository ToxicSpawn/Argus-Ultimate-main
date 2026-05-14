"""
Test cases for RegimeSpecificParameters
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone
from core.real_time_learning.regime_parameters import RegimeSpecificParameters, RegimeParameters


@pytest.fixture
def regime_parameters():
    """Create a test regime parameters component"""
    params = RegimeSpecificParameters()
    regimes = ["stable", "volatile", "trending", "range"]
    params.initialize_regimes(regimes)
    return params


def test_initialization(regime_parameters):
    """Test that regimes are properly initialized"""
    assert len(regime_parameters.regime_parameters) == 4
    assert "stable" in regime_parameters.regime_parameters
    assert "volatile" in regime_parameters.regime_parameters
    assert "trending" in regime_parameters.regime_parameters
    assert "range" in regime_parameters.regime_parameters
    
    # Check default parameters
    assert regime_parameters.current_parameters['position_size_pct'] == 0.05
    assert regime_parameters.current_parameters['max_leverage'] == 2.0
    assert regime_parameters.current_parameters['stop_loss_pct'] == 0.03


def test_regime_detection(regime_parameters):
    """Test regime detection and its impact on parameters"""
    # Test volatile regime
    volatile_data = {
        'market_data': {
            'volatility': 0.03,  # High volatility
            'trend_strength': 0.3
        }
    }
    regime_parameters.learn(volatile_data)
    assert regime_parameters.current_regime == "volatile"
    
    # Check that parameters were adjusted for volatile regime
    assert regime_parameters.current_parameters['position_size_pct'] < 0.05  # Should be reduced
    assert regime_parameters.current_parameters['max_leverage'] < 2.0      # Should be reduced
    assert regime_parameters.current_parameters['stop_loss_pct'] > 0.03   # Should be widened
    
    # Test trending regime
    trending_data = {
        'market_data': {
            'volatility': 0.01,
            'trend_strength': 0.6  # Strong trend
        }
    }
    regime_parameters.learn(trending_data)
    assert regime_parameters.current_regime == "trending"
    
    # Check that parameters were adjusted for trending regime
    assert regime_parameters.current_parameters['position_size_pct'] > 0.05  # Should be increased
    assert regime_parameters.current_parameters['max_leverage'] > 2.0      # Should be increased
    assert regime_parameters.current_parameters['stop_loss_pct'] < 0.03   # Should be tightened
    
    # Test range regime
    range_data = {
        'market_data': {
            'volatility': 0.005,  # Low volatility
            'trend_strength': 0.05  # Weak trend
        }
    }
    regime_parameters.learn(range_data)
    assert regime_parameters.current_regime == "range"
    
    # Check that parameters were adjusted for range regime
    assert regime_parameters.current_parameters['position_size_pct'] > 0.05  # Should be slightly increased
    assert regime_parameters.current_parameters['entry_aggressiveness'] > 0.5  # Should be more aggressive


def test_parameter_adjustments(regime_parameters):
    """Test that parameters are adjusted correctly for each regime"""
    # Test volatile regime adjustments
    regime_parameters.current_regime = "volatile"
    regime_parameters._adjust_parameters_for_regime()
    
    assert regime_parameters.current_parameters['position_size_pct'] == 0.035  # 0.05 * 0.7
    assert regime_parameters.current_parameters['max_leverage'] == 1.0       # 2.0 * 0.5
    assert regime_parameters.current_parameters['risk_per_trade_pct'] == 0.006  # 0.01 * 0.6
    
    # Test trending regime adjustments
    regime_parameters.current_regime = "trending"
    regime_parameters._adjust_parameters_for_regime()
    
    assert regime_parameters.current_parameters['position_size_pct'] == 0.065  # 0.05 * 1.3
    assert regime_parameters.current_parameters['max_leverage'] == 2.4       # 2.0 * 1.2
    assert regime_parameters.current_parameters['take_profit_pct'] == 0.075   # 0.05 * 1.5
    
    # Test range regime adjustments
    regime_parameters.current_regime = "range"
    regime_parameters._adjust_parameters_for_regime()
    
    assert regime_parameters.current_parameters['position_size_pct'] == 0.055  # 0.05 * 1.1
    assert regime_parameters.current_parameters['entry_aggressiveness'] == 0.45  # 0.5 * 0.9


def test_performance_tracking(regime_parameters):
    """Test performance tracking for regimes"""
    # Add performance metrics for volatile regime
    regime_parameters.current_regime = "volatile"
    metrics = {
        'sharpe_ratio': 1.5,
        'win_rate': 0.6,
        'max_drawdown': 0.10,
        'profit_factor': 1.8
    }
    
    regime_parameters.learn_from_performance(metrics)
    
    # Check that performance was recorded
    performance = regime_parameters.get_regime_performance("volatile")
    assert performance['sharpe_ratio'] == 1.5
    assert performance['win_rate'] == 0.6
    
    # Add more metrics to test averaging
    metrics2 = {
        'sharpe_ratio': 1.8,
        'win_rate': 0.65,
        'max_drawdown': 0.08,
        'profit_factor': 2.0
    }
    
    regime_parameters.learn_from_performance(metrics2)
    
    # Should average the two metrics
    performance = regime_parameters.get_regime_performance("volatile")
    assert 1.5 < performance['sharpe_ratio'] < 1.8
    assert 0.6 < performance['win_rate'] < 0.65


def test_parameter_bounds(regime_parameters):
    """Test that parameters stay within bounds"""
    # Try to set a parameter outside bounds
    regime_parameters.current_parameters['position_size_pct'] = 0.2  # Above max of 0.1
    regime_parameters._adjust_parameters_for_regime()
    
    # Should be bounded to max
    assert regime_parameters.current_parameters['position_size_pct'] == 0.1
    
    # Try to set a parameter below min
    regime_parameters.current_parameters['position_size_pct'] = 0.001  # Below min of 0.01
    regime_parameters._adjust_parameters_for_regime()
    
    # Should be bounded to min
    assert regime_parameters.current_parameters['position_size_pct'] == 0.01


def test_rollback(regime_parameters):
    """Test rollback functionality"""
    # Set current regime and parameters
    regime_parameters.current_regime = "volatile"
    regime_parameters._adjust_parameters_for_regime()
    
    # Add some performance history
    regime_parameters.performance_history = [
        {'regime': 'stable', 'timestamp': '2023-01-01T00:00:00', 'sharpe_ratio': 1.2}
    ]
    
    # Change parameters manually
    regime_parameters.current_parameters['position_size_pct'] = 0.2
    regime_parameters.current_regime = "trending"
    
    # Perform rollback
    regime_parameters.rollback()
    
    # Should revert to stable regime parameters
    assert regime_parameters.current_regime == "stable"
    assert regime_parameters.current_parameters['position_size_pct'] == 0.05


def test_validation(regime_parameters):
    """Test parameter validation"""
    # Test valid parameters
    valid_params = {
        'current_regime': 'stable',
        'current_parameters': {
            'position_size_pct': 0.05,
            'max_leverage': 2.0,
            'stop_loss_pct': 0.03
        },
        'regime_adjustments': {
            'stable': {
                'position_size_pct': 1.0,
                'max_leverage': 1.0
            }
        }
    }
    assert regime_parameters.validate(valid_params) == True
    
    # Test invalid parameter (out of bounds)
    invalid_params = {
        'current_parameters': {
            'position_size_pct': 0.15  # Above max of 0.1
        }
    }
    assert regime_parameters.validate(invalid_params) == False
    
    # Test invalid adjustment factor
    invalid_params = {
        'regime_adjustments': {
            'stable': {
                'position_size_pct': -0.5  # Negative factor
            }
        }
    }
    assert regime_parameters.validate(invalid_params) == False