"""
Test cases for DynamicCorrelationMatrix
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from core.real_time_learning.correlation_matrix import DynamicCorrelationMatrix, AssetCorrelationData


@pytest.fixture
def correlation_matrix():
    """Create a test correlation matrix with 3 assets"""
    matrix = DynamicCorrelationMatrix()
    assets = ["BTC", "ETH", "SOL"]
    matrix.initialize_assets(assets)
    return matrix


def test_initialization(correlation_matrix):
    """Test that assets are properly initialized"""
    assert len(correlation_matrix.assets) == 3
    assert ("BTC", "ETH") in correlation_matrix.correlation_pairs
    assert ("BTC", "SOL") in correlation_matrix.correlation_pairs
    assert ("ETH", "SOL") in correlation_matrix.correlation_pairs
    
    # Check initial correlations are 0
    assert correlation_matrix.current_matrix[("BTC", "ETH")] == 0.0
    assert correlation_matrix.current_matrix[("ETH", "BTC")] == 0.0


def test_correlation_update(correlation_matrix):
    """Test that correlations are updated correctly"""
    # Create mock returns data
    returns_data = {
        "BTC": [0.01, -0.005, 0.02, 0.015, -0.01],
        "ETH": [0.015, -0.003, 0.025, 0.01, -0.008],
        "SOL": [0.02, 0.005, 0.01, 0.03, -0.02]
    }
    
    # Create market data with regime
    data = {
        'asset_returns': returns_data,
        'market_data': {
            'volatility': 0.015,
            'trend_strength': 0.3
        }
    }
    
    # Update correlations
    correlation_matrix.learn(data)
    
    # Check that correlations were updated
    btc_eth_corr = correlation_matrix.current_matrix[("BTC", "ETH")]
    assert abs(btc_eth_corr) <= 1.0  # Should be a valid correlation
    
    # Check symmetry
    assert abs(btc_eth_corr - correlation_matrix.current_matrix[("ETH", "BTC")]) < 0.001


def test_regime_adjustment(correlation_matrix):
    """Test that correlations are adjusted based on regime"""
    # Set volatile regime
    correlation_matrix.current_regime = "volatile"
    
    # Create returns data
    returns_data = {
        "BTC": [0.01, -0.005, 0.02, 0.015, -0.01],
        "ETH": [0.012, -0.004, 0.022, 0.012, -0.009]
    }
    
    # Calculate expected raw correlation
    df = pd.DataFrame({
        "BTC": returns_data["BTC"],
        "ETH": returns_data["ETH"]
    })
    raw_corr = df.corr(method='spearman').loc["BTC", "ETH"]
    
    # Update correlations
    data = {
        'asset_returns': returns_data,
        'market_data': {
            'volatility': 0.03,  # High volatility -> volatile regime
            'trend_strength': 0.2
        }
    }
    
    correlation_matrix.learn(data)
    
    # Get adjusted correlation
    adjusted_corr = correlation_matrix.current_matrix[("BTC", "ETH")]
    
    # In volatile regime, correlations should be increased (multiplied by 1.2)
    expected_adjusted = raw_corr * 1.2
    assert abs(adjusted_corr - expected_adjusted) < 0.05  # Allow some tolerance


def test_concentration_metrics(correlation_matrix):
    """Test portfolio concentration metrics"""
    # Set up some correlations
    correlation_matrix.current_matrix[("BTC", "ETH")] = 0.8
    correlation_matrix.current_matrix[("ETH", "BTC")] = 0.8
    correlation_matrix.current_matrix[("BTC", "SOL")] = 0.3
    correlation_matrix.current_matrix[("SOL", "BTC")] = 0.3
    correlation_matrix.current_matrix[("ETH", "SOL")] = 0.5
    correlation_matrix.current_matrix[("SOL", "ETH")] = 0.5
    
    # Calculate concentration
    correlation_matrix._calculate_diversification_metrics()
    
    # Should be high concentration with these correlations
    assert correlation_matrix.portfolio_concentration > 0.5
    
    # Test concentration alert
    alerts = correlation_matrix.get_concentration_alerts()
    assert len(alerts) > 0
    assert alerts[0]['type'] == 'concentration'


def test_diversification_score(correlation_matrix):
    """Test diversification score calculation"""
    # Test with low correlation (high diversification)
    correlation_matrix.current_matrix[("BTC", "ETH")] = 0.2
    correlation_matrix.current_matrix[("ETH", "BTC")] = 0.2
    correlation_matrix.current_matrix[("BTC", "SOL")] = 0.1
    correlation_matrix.current_matrix[("SOL", "BTC")] = 0.1
    correlation_matrix.current_matrix[("ETH", "SOL")] = 0.3
    correlation_matrix.current_matrix[("SOL", "ETH")] = 0.3
    
    score = correlation_matrix.get_diversification_score()
    assert score > 0.7  # Should be well diversified
    
    # Test with high correlation (low diversification)
    correlation_matrix.current_matrix[("BTC", "ETH")] = 0.9
    correlation_matrix.current_matrix[("ETH", "BTC")] = 0.9
    correlation_matrix.current_matrix[("BTC", "SOL")] = 0.8
    correlation_matrix.current_matrix[("SOL", "BTC")] = 0.8
    correlation_matrix.current_matrix[("ETH", "SOL")] = 0.95
    correlation_matrix.current_matrix[("SOL", "ETH")] = 0.95
    
    score = correlation_matrix.get_diversification_score()
    assert score < 0.3  # Should be poorly diversified


def test_rollback(correlation_matrix):
    """Test rollback functionality"""
    # Set some initial correlations
    correlation_matrix.current_matrix[("BTC", "ETH")] = 0.5
    correlation_matrix.current_matrix[("ETH", "BTC")] = 0.5
    correlation_matrix.portfolio_concentration = 0.4
    
    # Add to history
    correlation_matrix.matrix_history = [
        {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'regime': 'stable',
            'matrix': {("BTC", "ETH"): 0.3, ("ETH", "BTC"): 0.3},
            'concentration': 0.3
        }
    ]
    
    # Change current state
    correlation_matrix.current_matrix[("BTC", "ETH")] = 0.8
    correlation_matrix.current_matrix[("ETH", "BTC")] = 0.8
    correlation_matrix.portfolio_concentration = 0.7
    
    # Perform rollback
    correlation_matrix.rollback()
    
    # Should revert to previous state
    assert abs(correlation_matrix.current_matrix[("BTC", "ETH")] - 0.3) < 0.01
    assert abs(correlation_matrix.portfolio_concentration - 0.3) < 0.01


def test_validation(correlation_matrix):
    """Test parameter validation"""
    # Test valid parameters
    valid_params = {
        'current_matrix': {("BTC", "ETH"): 0.5, ("ETH", "BTC"): 0.5},
        'portfolio_concentration': 0.3,
        'current_regime': 'stable',
        'max_concentration_threshold': 0.4,
        'correlation_threshold': 0.7
    }
    assert correlation_matrix.validate(valid_params) == True
    
    # Test invalid correlation (not symmetric)
    invalid_params = {
        'current_matrix': {("BTC", "ETH"): 0.5, ("ETH", "BTC"): 0.6},
        'portfolio_concentration': 0.3
    }
    assert correlation_matrix.validate(invalid_params) == False
    
    # Test out of bounds correlation
    invalid_params = {
        'current_matrix': {("BTC", "ETH"): 1.2, ("ETH", "BTC"): 1.2},
        'portfolio_concentration': 0.3
    }
    assert correlation_matrix.validate(invalid_params) == False
    
    # Test invalid concentration threshold
    invalid_params = {
        'current_matrix': {("BTC", "ETH"): 0.5, ("ETH", "BTC"): 0.5},
        'max_concentration_threshold': 1.2
    }
    assert correlation_matrix.validate(invalid_params) == False