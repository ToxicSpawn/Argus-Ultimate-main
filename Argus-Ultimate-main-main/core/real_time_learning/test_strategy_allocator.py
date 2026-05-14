"""
Test cases for AdaptiveStrategyAllocator
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone
from core.real_time_learning.strategy_allocator import AdaptiveStrategyAllocator, StrategyPerformance


@pytest.fixture
def allocator():
    """Create a test allocator with 3 strategies"""
    allocator = AdaptiveStrategyAllocator()
    allocator.initialize_strategies(["strategy1", "strategy2", "strategy3"])
    return allocator


def test_initialization(allocator):
    """Test that strategies are properly initialized"""
    assert len(allocator.strategy_performance) == 3
    assert "strategy1" in allocator.strategy_performance
    assert "strategy2" in allocator.strategy_performance
    assert "strategy3" in allocator.strategy_performance
    
    # Check initial equal weights
    weights = allocator.get_params()['strategy_weights']
    assert len(weights) == 3
    assert abs(weights["strategy1"] - 1/3) < 0.01
    assert abs(weights["strategy2"] - 1/3) < 0.01
    assert abs(weights["strategy3"] - 1/3) < 0.01


def test_performance_update(allocator):
    """Test that strategy performance metrics are updated correctly"""
    # Simulate trade results
    trade_data = {
        'trade_results': [
            {
                'strategy': 'strategy1',
                'pnl': 100,
                'return_pct': 0.01,
                'regime': 'stable'
            },
            {
                'strategy': 'strategy1',
                'pnl': -50,
                'return_pct': -0.005,
                'regime': 'stable'
            },
            {
                'strategy': 'strategy2',
                'pnl': 200,
                'return_pct': 0.02,
                'regime': 'stable'
            }
        ]
    }
    
    # Update performance
    allocator.learn(trade_data)
    
    # Check metrics were updated
    strat1 = allocator.strategy_performance["strategy1"]
    assert len(strat1.recent_trades) == 2
    assert strat1.win_rate == 0.5  # 1 win, 1 loss
    
    strat2 = allocator.strategy_performance["strategy2"]
    assert len(strat2.recent_trades) == 1
    assert strat2.win_rate == 1.0  # 1 win, 0 losses


def test_regime_detection(allocator):
    """Test regime detection and its impact on allocations"""
    # Test volatile regime
    volatile_data = {
        'market_data': {
            'volatility': 0.03,  # High volatility
            'trend_strength': 0.2
        }
    }
    allocator.learn(volatile_data)
    assert allocator.current_regime == "volatile"
    
    # Test trending regime
    trending_data = {
        'market_data': {
            'volatility': 0.01,
            'trend_strength': 0.6  # Strong trend
        }
    }
    allocator.learn(trending_data)
    assert allocator.current_regime == "trending"
    
    # Test range regime
    range_data = {
        'market_data': {
            'volatility': 0.004,  # Low volatility
            'trend_strength': 0.05  # Weak trend
        }
    }
    allocator.learn(range_data)
    assert allocator.current_regime == "range"


def test_weight_constraints(allocator):
    """Test that weight constraints are properly applied"""
    # Test min/max weight constraints
    allocator.min_weight = 0.1
    allocator.max_weight = 0.4
    
    # Create extreme weights that should be constrained
    extreme_weights = {
        'strategy1': 0.01,  # Below min
        'strategy2': 0.6,   # Above max  
        'strategy3': 0.39   # Should be adjusted due to others
    }
    
    # Apply constraints
    constrained = allocator._apply_constraints(extreme_weights)
    
    # Check constraints
    assert constrained['strategy1'] >= 0.1  # Should be lifted to min
    assert constrained['strategy2'] <= 0.4  # Should be capped at max
    assert abs(sum(constrained.values()) - 1.0) < 0.01  # Should sum to 1


def test_rollback(allocator):
    """Test rollback functionality"""
    # Set some initial state
    initial_weights = {'strategy1': 0.4, 'strategy2': 0.3, 'strategy3': 0.3}
    allocator.last_allocation = initial_weights.copy()
    allocator.allocation_history = [
        {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'regime': 'stable',
            'allocation': {'strategy1': 0.33, 'strategy2': 0.33, 'strategy3': 0.33}
        }
    ]
    
    # Change current allocation
    allocator.last_allocation = {'strategy1': 0.8, 'strategy2': 0.1, 'strategy3': 0.1}
    
    # Perform rollback
    allocator.rollback()
    
    # Should revert to previous allocation
    assert abs(allocator.last_allocation['strategy1'] - 0.33) < 0.01
    assert abs(allocator.last_allocation['strategy2'] - 0.33) < 0.01
    assert abs(allocator.last_allocation['strategy3'] - 0.33) < 0.01


def test_validation(allocator):
    """Test parameter validation"""
    # Test valid parameters
    valid_params = {
        'strategy_weights': {'strategy1': 0.4, 'strategy2': 0.3, 'strategy3': 0.3},
        'current_regime': 'stable'
    }
    assert allocator.validate(valid_params) == True
    
    # Test invalid weights (don't sum to 1)
    invalid_weights = {
        'strategy_weights': {'strategy1': 0.9, 'strategy2': 0.2, 'strategy3': 0.2}
    }
    assert allocator.validate(invalid_weights) == False
    
    # Test invalid regime
    invalid_regime = {
        'strategy_weights': {'strategy1': 0.4, 'strategy2': 0.3, 'strategy3': 0.3},
        'current_regime': 'invalid_regime'
    }
    assert allocator.validate(invalid_regime) == False


def test_correlation_penalty(allocator):
    """Test that correlation between strategies affects weights"""
    # Set up some correlation between strategies
    allocator.strategy_correlation = {
        ('strategy1', 'strategy2'): 0.9,  # High correlation
        ('strategy1', 'strategy3'): 0.1,  # Low correlation
        ('strategy2', 'strategy1'): 0.9,
        ('strategy2', 'strategy3'): 0.2,
        ('strategy3', 'strategy1'): 0.1,
        ('strategy3', 'strategy2'): 0.2
    }
    
    # Equal initial weights
    initial_weights = {'strategy1': 0.33, 'strategy2': 0.33, 'strategy3': 0.33}
    
    # Apply correlation penalty
    adjusted = allocator._apply_correlation_penalty(initial_weights)
    
    # strategy1 and strategy2 are highly correlated - strategy2 should get penalized more
    # because it comes second in the iteration
    assert adjusted['strategy1'] > initial_weights['strategy1'] - 0.05
    assert adjusted['strategy2'] < initial_weights['strategy2'] + 0.05
    assert adjusted['strategy3'] > initial_weights['strategy3']  # Low correlation should benefit


def test_regime_specific_allocation(allocator):
    """Test that allocations adapt to different regimes"""
    # Set up different performance in different regimes
    strat1 = allocator.strategy_performance["strategy1"]
    strat2 = allocator.strategy_performance["strategy2"]
    
    # Strategy1 performs well in volatile, strategy2 in stable
    strat1.regime_performance = {
        'volatile': {'win_rate': 0.8, 'avg_return': 0.02, 'trades': 10},
        'stable': {'win_rate': 0.4, 'avg_return': 0.005, 'trades': 10}
    }
    
    strat2.regime_performance = {
        'volatile': {'win_rate': 0.3, 'avg_return': -0.01, 'trades': 10},
        'stable': {'win_rate': 0.7, 'avg_return': 0.015, 'trades': 10}
    }
    
    # Test volatile regime
    allocator.current_regime = 'volatile'
    allocation = allocator._calculate_optimal_allocation()
    assert allocation['strategy1'] > allocation['strategy2']  # strategy1 should get more weight
    
    # Test stable regime
    allocator.current_regime = 'stable'
    allocation = allocator._calculate_optimal_allocation()
    assert allocation['strategy2'] > allocation['strategy1']  # strategy2 should get more weight