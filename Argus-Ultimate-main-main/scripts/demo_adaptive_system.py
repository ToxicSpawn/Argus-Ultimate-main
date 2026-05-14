"""
Comprehensive Demonstration of Adaptive Trading System

This script demonstrates all adaptive components working together:
1. AdaptiveStrategyAllocator - Dynamically adjusts strategy weights
2. DynamicCorrelationMatrix - Continuously updates asset correlations
3. SmartOrderRouter - Routes orders to optimal venues
4. RegimeSpecificParameters - Adjusts parameters based on market regime
"""

import sys
import os
import logging
from datetime import datetime, timezone
import random
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.real_time_learning.orchestrator import RealTimeLearningOrchestrator
from core.real_time_learning.strategy_allocator import AdaptiveStrategyAllocator
from core.real_time_learning.correlation_matrix import DynamicCorrelationMatrix
from core.real_time_learning.order_router import SmartOrderRouter
from core.real_time_learning.regime_parameters import RegimeSpecificParameters

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_market_data(regime):
    """Generate synthetic market data for a regime"""
    base_data = {
        'volatility': 0.01,
        'trend_strength': 0.2
    }
    
    if regime == 'volatile':
        base_data['volatility'] = 0.03
    elif regime == 'trending':
        base_data['trend_strength'] = 0.6
    elif regime == 'range':
        base_data['volatility'] = 0.005
    
    return base_data


def generate_strategy_performance(regime):
    """Generate synthetic strategy performance for a regime"""
    base_perf = {
        'momentum': {'win_rate': 0.5, 'avg_return': 0.005},
        'mean_reversion': {'win_rate': 0.5, 'avg_return': 0.005},
        'breakout': {'win_rate': 0.5, 'avg_return': 0.005}
    }
    
    if regime == 'volatile':
        base_perf['momentum']['win_rate'] = 0.7
        base_perf['momentum']['avg_return'] = 0.02
        base_perf['mean_reversion']['win_rate'] = 0.3
        base_perf['mean_reversion']['avg_return'] = -0.01
    elif regime == 'trending':
        base_perf['momentum']['win_rate'] = 0.8
        base_perf['momentum']['avg_return'] = 0.03
        base_perf['breakout']['win_rate'] = 0.9
        base_perf['breakout']['avg_return'] = 0.04
    elif regime == 'range':
        base_perf['mean_reversion']['win_rate'] = 0.7
        base_perf['mean_reversion']['avg_return'] = 0.015
    
    return base_perf


def generate_asset_returns(regime):
    """Generate synthetic asset returns for correlation analysis"""
    assets = ["BTC", "ETH", "SOL"]
    
    # Base correlation matrix
    if regime == 'volatile':
        corr_matrix = np.array([
            [1.0, 0.8, 0.7],
            [0.8, 1.0, 0.85],
            [0.7, 0.85, 1.0]
        ])
    elif regime == 'trending':
        corr_matrix = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])
    else:  # stable and range
        corr_matrix = np.array([
            [1.0, 0.6, 0.4],
            [0.6, 1.0, 0.5],
            [0.4, 0.5, 1.0]
        ])
    
    # Generate correlated returns
    L = np.linalg.cholesky(corr_matrix)
    shocks = np.random.normal(0, 1, (30, 3))
    correlated = np.dot(shocks, L.T)
    
    # Convert to returns
    returns = {}
    for i, asset in enumerate(assets):
        returns[asset] = list(correlated[:,i] * 0.01)  # Scale to 1% std
    
    return returns


def generate_execution_reports():
    """Generate synthetic execution reports"""
    venues = ["binance", "bybit", "okx"]
    reports = []
    
    for i in range(10):
        venue = random.choice(venues)
        reports.append({
            'venue': venue,
            'order_type': random.choice(['market', 'limit']),
            'order_size': random.choice(['small', 'medium', 'large']),
            'fill_ratio': 0.8 + random.random() * 0.2,
            'slippage': 0.0005 + random.random() * 0.001,
            'latency': 20 + random.randint(0, 60),
            'success': True
        })
    
    return reports


def generate_performance_metrics(regime):
    """Generate synthetic performance metrics"""
    base_metrics = {
        'sharpe_ratio': 1.2,
        'win_rate': 0.55,
        'max_drawdown': 0.15,
        'profit_factor': 1.5
    }
    
    if regime == 'volatile':
        base_metrics['sharpe_ratio'] = 0.8
        base_metrics['win_rate'] = 0.45
        base_metrics['max_drawdown'] = 0.25
    elif regime == 'trending':
        base_metrics['sharpe_ratio'] = 2.0
        base_metrics['win_rate'] = 0.7
        base_metrics['max_drawdown'] = 0.10
    elif regime == 'range':
        base_metrics['sharpe_ratio'] = 1.5
        base_metrics['win_rate'] = 0.6
        base_metrics['max_drawdown'] = 0.12
    
    return base_metrics


def run_demo():
    """Run the comprehensive adaptive system demonstration"""
    
    # Create orchestrator
    orchestrator = RealTimeLearningOrchestrator()
    
    # Create and register all adaptive components
    strategy_allocator = AdaptiveStrategyAllocator()
    strategy_allocator.update_frequency = 1
    
    correlation_matrix = DynamicCorrelationMatrix()
    correlation_matrix.update_frequency = 1
    
    order_router = SmartOrderRouter()
    order_router.update_frequency = 1
    
    regime_params = RegimeSpecificParameters()
    regime_params.update_frequency = 1
    
    # Register components
    orchestrator.register_component(strategy_allocator)
    orchestrator.register_component(correlation_matrix)
    orchestrator.register_component(order_router)
    orchestrator.register_component(regime_params)
    
    # Initialize components
    strategies = ["momentum", "mean_reversion", "breakout"]
    strategy_allocator.initialize_strategies(strategies)
    
    assets = ["BTC", "ETH", "SOL"]
    correlation_matrix.initialize_assets(assets)
    
    venues = ["binance", "bybit", "okx"]
    order_router.initialize_venues(venues)
    
    regimes = ["stable", "volatile", "trending", "range"]
    regime_params.initialize_regimes(regimes)
    
    print("=== Comprehensive Adaptive Trading System Demo ===\n")
    
    # Test different market regimes
    for regime in regimes:
        print(f"\n--- Testing {regime.upper()} Regime ---")
        
        # Generate data for this regime
        market_data = generate_market_data(regime)
        strategy_perf = generate_strategy_performance(regime)
        asset_returns = generate_asset_returns(regime)
        execution_reports = generate_execution_reports()
        performance_metrics = generate_performance_metrics(regime)
        
        # Create trade results for strategy allocator
        trade_results = []
        for strategy, perf in strategy_perf.items():
            for i in range(3):
                is_winner = i < int(perf['win_rate'] * 3)
                pnl = 100 if is_winner else -50
                return_pct = perf['avg_return'] if is_winner else perf['avg_return'] * -0.5
                
                trade_results.append({
                    'strategy': strategy,
                    'pnl': pnl,
                    'return_pct': return_pct,
                    'regime': regime
                })
        
        # Create input data
        data = {
            'market_data': market_data,
            'trade_results': trade_results,
            'asset_returns': asset_returns,
            'execution_reports': execution_reports,
            'performance_metrics': performance_metrics
        }
        
        # Process through orchestrator
        results = orchestrator.on_market_data(data)
        
        # Get component states
        allocator_state = strategy_allocator.get_params()
        correlation_state = correlation_matrix.get_params()
        router_state = order_router.get_params()
        regime_state = regime_params.get_params()
        
        print(f"Detected regime: {regime_state['current_regime']}")
        
        # Show strategy allocation
        print("\nStrategy Allocation:")
        for strategy, weight in sorted(allocator_state['strategy_weights'].items(), key=lambda x: -x[1]):
            print(f"  {strategy}: {weight:.1%}")
        
        # Show correlation matrix
        print("\nAsset Correlations:")
        for (a1, a2), corr in sorted(correlation_state['current_matrix'].items()):
            if a1 < a2:  # Avoid duplicate pairs
                print(f"  {a1}-{a2}: {corr:.2f}")
        print(f"Diversification score: {correlation_matrix.get_diversification_score():.2f}")
        
        # Show order routing
        print("\nVenue Performance:")
        for venue, perf in router_state['venue_performance'].items():
            print(f"  {venue}: Liquidity={perf['liquidity_score']:.2f}, Fill={perf['fill_ratio']:.2f}")
        
        # Show regime parameters
        print("\nTrading Parameters:")
        for param, value in sorted(regime_state['current_parameters'].items()):
            print(f"  {param}: {value:.3f}")
        
        # Route a test order
        test_order = {
            'order_id': f'test_{regime}',
            'order_type': 'market',
            'order_size': 'medium',
            'symbol': 'BTCUSDT'
        }
        
        decision = order_router.route_order(test_order)
        print(f"\nRouting Decision: {decision.venue} (confidence: {decision.confidence:.2f})")
        
        print("\n" + "="*60)
    
    # Show final system state
    print("\n=== Final System State ===")
    print(f"Components registered: {len(orchestrator.components)}")
    print(f"Recent changes: {len(orchestrator.audit.get_recent_changes())}")
    
    # Show audit trail
    print("\nRecent Adaptations:")
    for change in orchestrator.audit.get_recent_changes(hours=1):
        print(f"  {change['timestamp']}: {change['component']} - {len(change['changes'])} parameters changed")


if __name__ == "__main__":
    run_demo()