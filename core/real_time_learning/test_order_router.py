"""
Test cases for SmartOrderRouter
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone
from core.real_time_learning.order_router import SmartOrderRouter, VenuePerformance, OrderRoutingDecision


@pytest.fixture
def order_router():
    """Create a test order router with 3 venues"""
    router = SmartOrderRouter()
    venues = ["binance", "bybit", "okx"]
    router.initialize_venues(venues)
    return router


def test_initialization(order_router):
    """Test that venues are properly initialized"""
    assert len(order_router.available_venues) == 3
    assert "binance" in order_router.venues
    assert "bybit" in order_router.venues
    assert "okx" in order_router.venues


def test_venue_performance_update(order_router):
    """Test that venue performance metrics are updated correctly"""
    # Create execution reports
    reports = [
        {
            'venue': 'binance',
            'order_type': 'limit',
            'order_size': 'medium',
            'fill_ratio': 0.95,
            'slippage': 0.0005,
            'latency': 30,
            'success': True
        },
        {
            'venue': 'bybit',
            'order_type': 'market',
            'order_size': 'large',
            'fill_ratio': 0.85,
            'slippage': 0.0015,
            'latency': 45,
            'success': True
        },
        {
            'venue': 'binance',
            'order_type': 'market',
            'order_size': 'small',
            'fill_ratio': 0.9,
            'slippage': 0.001,
            'latency': 35,
            'success': True
        }
    ]
    
    # Update performance
    order_router._update_venue_performance(reports)
    
    # Check metrics were updated
    binance = order_router.venues["binance"]
    assert binance.fill_ratio > 0.9
    assert binance.avg_slippage < 0.001
    assert binance.latency < 40
    assert binance.liquidity_score > 0.8
    
    bybit = order_router.venues["bybit"]
    assert bybit.fill_ratio == 0.85
    assert abs(bybit.avg_slippage - 0.0015) < 0.0001
    assert bybit.latency == 45


def test_regime_detection(order_router):
    """Test regime detection and its impact on routing"""
    # Test volatile regime
    volatile_data = {
        'market_data': {
            'volatility': 0.03,  # High volatility
            'trend_strength': 0.3
        }
    }
    order_router.learn(volatile_data)
    assert order_router.current_regime == "volatile"
    
    # Test trending regime
    trending_data = {
        'market_data': {
            'volatility': 0.01,
            'trend_strength': 0.6  # Strong trend
        }
    }
    order_router.learn(trending_data)
    assert order_router.current_regime == "trending"
    
    # Test range regime
    range_data = {
        'market_data': {
            'volatility': 0.005,  # Low volatility
            'trend_strength': 0.05  # Weak trend
        }
    }
    order_router.learn(range_data)
    assert order_router.current_regime == "range"


def test_order_routing(order_router):
    """Test order routing decisions"""
    # Set up venue performance
    order_router.venues["binance"].liquidity_score = 0.9
    order_router.venues["binance"].fill_ratio = 0.95
    order_router.venues["binance"].avg_slippage = 0.0005
    order_router.venues["binance"].latency = 30
    
    order_router.venues["bybit"].liquidity_score = 0.8
    order_router.venues["bybit"].fill_ratio = 0.9
    order_router.venues["bybit"].avg_slippage = 0.001
    order_router.venues["bybit"].latency = 40
    
    order_router.venues["okx"].liquidity_score = 0.7
    order_router.venues["okx"].fill_ratio = 0.85
    order_router.venues["okx"].avg_slippage = 0.0015
    order_router.venues["okx"].latency = 50
    
    # Test routing in stable regime
    order_router.current_regime = "stable"
    order = {
        'order_id': 'test123',
        'order_type': 'market',
        'order_size': 'medium',
        'symbol': 'BTCUSDT'
    }
    
    decision = order_router.route_order(order)
    
    # Binance should be selected as it has the best metrics
    assert decision.venue == "binance"
    assert decision.confidence > 0.8
    assert decision.expected_fill_ratio > 0.9
    
    # Test routing in volatile regime (liquidity more important)
    order_router.current_regime = "volatile"
    # Make bybit have better liquidity but worse latency
    order_router.venues["bybit"].liquidity_score = 0.95
    order_router.venues["bybit"].latency = 60
    
    decision = order_router.route_order(order)
    # Bybit should be selected due to higher liquidity weight in volatile regime
    assert decision.venue == "bybit"


def test_venue_performance_metrics(order_router):
    """Test getting venue performance metrics"""
    # Set some performance metrics
    order_router.venues["binance"].liquidity_score = 0.9
    order_router.venues["binance"].fill_ratio = 0.95
    order_router.venues["binance"].avg_slippage = 0.0005
    order_router.venues["binance"].latency = 30
    
    # Get performance metrics
    metrics = order_router.get_venue_performance("binance")
    
    assert metrics['liquidity_score'] == 0.9
    assert metrics['fill_ratio'] == 0.95
    assert metrics['avg_slippage'] == 0.0005
    assert metrics['latency'] == 30


def test_routing_history(order_router):
    """Test routing history tracking"""
    # Route some orders
    order1 = {'order_id': 'order1', 'order_type': 'market', 'order_size': 'medium', 'symbol': 'BTCUSDT'}
    order2 = {'order_id': 'order2', 'order_type': 'limit', 'order_size': 'large', 'symbol': 'ETHUSDT'}
    
    # Set up venues for consistent routing
    order_router.venues["binance"].liquidity_score = 0.9
    order_router.venues["bybit"].liquidity_score = 0.7
    
    decision1 = order_router.route_order(order1)
    decision2 = order_router.route_order(order2)
    
    # Check history
    history = order_router.get_routing_history()
    assert len(history) == 2
    assert history[0].order_id == 'order1'
    assert history[1].order_id == 'order2'


def test_rollback(order_router):
    """Test rollback functionality"""
    # Create some routing history
    order1 = {'order_id': 'order1', 'order_type': 'market', 'order_size': 'medium', 'symbol': 'BTCUSDT'}
    order2 = {'order_id': 'order2', 'order_type': 'limit', 'order_size': 'large', 'symbol': 'ETHUSDT'}
    
    # Set up venues
    order_router.venues["binance"].liquidity_score = 0.9
    order_router.venues["bybit"].liquidity_score = 0.7
    
    decision1 = order_router.route_order(order1)
    decision2 = order_router.route_order(order2)
    
    # Change last decision
    order_router.last_routing_decision = OrderRoutingDecision(
        order_id="bad_order",
        venue="bad_venue",
        reason="test",
        expected_fill_ratio=0.5,
        expected_slippage=0.01,
        expected_latency=100,
        confidence=0.2
    )
    
    # Perform rollback
    order_router.rollback()
    
    # Should revert to previous decision
    assert order_router.last_routing_decision.order_id == 'order2'


def test_validation(order_router):
    """Test parameter validation"""
    # Test valid parameters
    valid_params = {
        'current_regime': 'stable',
        'regime_adjustments': {
            'stable': {'liquidity_weight': 0.5, 'latency_weight': 0.3, 'slippage_weight': 0.2}
        },
        'available_venues': ['binance', 'bybit']
    }
    assert order_router.validate(valid_params) == True
    
    # Test invalid regime weights (don't sum to 1)
    invalid_params = {
        'regime_adjustments': {
            'stable': {'liquidity_weight': 0.8, 'latency_weight': 0.3, 'slippage_weight': 0.2}
        }
    }
    assert order_router.validate(invalid_params) == False
    
    # Test invalid weight range
    invalid_params = {
        'regime_adjustments': {
            'stable': {'liquidity_weight': 1.2, 'latency_weight': 0.3, 'slippage_weight': -0.1}
        }
    }
    assert order_router.validate(invalid_params) == False
    
    # Test unknown venue
    invalid_params = {
        'available_venues': ['binance', 'unknown_venue']
    }
    assert order_router.validate(invalid_params) == False