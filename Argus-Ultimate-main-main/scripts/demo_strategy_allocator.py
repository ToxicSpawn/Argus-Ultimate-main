"""
Demonstration of AdaptiveStrategyAllocator

This script shows how the AdaptiveStrategyAllocator works with:
1. Different market regimes
2. Strategy performance tracking
3. Dynamic weight allocation
4. Safety validation
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Any
from core.real_time_learning.orchestrator import RealTimeLearningOrchestrator
from core.real_time_learning.strategy_allocator import AdaptiveStrategyAllocator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_demo_data(regime: str, strategy_performance: Dict[str, Dict], allocator: AdaptiveStrategyAllocator) -> Dict:
    """Create demo market data for a specific regime"""
    
    # Base market data
    market_data = {
        'volatility': 0.01,
        'trend_strength': 0.2
    }
    
    # Adjust based on regime using the allocator's thresholds
    thresholds = allocator.regime_thresholds
    if regime == 'volatile':
        market_data['volatility'] = thresholds['volatile']['volatility'] + 0.01
        market_data['trend_strength'] = thresholds['volatile']['trend_strength'] + 0.1
    elif regime == 'trending':
        market_data['volatility'] = thresholds['trending']['volatility']
        market_data['trend_strength'] = thresholds['trending']['trend_strength'] + 0.2
    elif regime == 'range':
        market_data['volatility'] = thresholds['range']['volatility'] - 0.001
        market_data['trend_strength'] = thresholds['range']['trend_strength'] - 0.05
    
    # Create trade results based on strategy performance
    trade_results = []
    for strategy, perf in strategy_performance.items():
        regime_perf = perf.get(regime, perf['stable'])
        
        # Create 3 trades for this strategy
        for i in range(3):
            is_winner = i < int(regime_perf['win_rate'] * 3)  # Simulate win rate
            pnl = 100 if is_winner else -50
            return_pct = regime_perf['avg_return'] if is_winner else regime_perf['avg_return'] * -0.5
            
            trade_results.append({
                'strategy': strategy,
                'pnl': pnl,
                'return_pct': return_pct,
                'regime': regime
            })
    
    return {
        'market_data': market_data,
        'trade_results': trade_results
    }


def run_demo():
    """Run the adaptive strategy allocator demonstration"""
    
    # Create orchestrator and allocator
    orchestrator = RealTimeLearningOrchestrator()
    allocator = AdaptiveStrategyAllocator()
    
    # Adjust regime thresholds for demo purposes
    allocator.regime_thresholds = {
        'volatile': {'volatility': 0.02, 'trend_strength': 0.3},
        'range': {'volatility': 0.005, 'trend_strength': 0.1},
        'trending': {'volatility': 0.01, 'trend_strength': 0.5}
    }
    
    # Set update frequency to 1 for demo purposes
    allocator.update_frequency = 1
    
    # Register the component
    orchestrator.register_component(allocator)
    
    # Initialize with 3 strategies
    strategies = ["momentum", "mean_reversion", "breakout"]
    allocator.initialize_strategies(strategies)
    
    # Define strategy performance in different regimes (moved here for scope)
    strategy_performance = {
        "momentum": {
            'stable': {'win_rate': 0.5, 'avg_return': 0.005},
            'volatile': {'win_rate': 0.7, 'avg_return': 0.02},
            'trending': {'win_rate': 0.8, 'avg_return': 0.03},
            'range': {'win_rate': 0.4, 'avg_return': 0.002}
        },
        "mean_reversion": {
            'stable': {'win_rate': 0.6, 'avg_return': 0.01},
            'volatile': {'win_rate': 0.3, 'avg_return': -0.01},
            'trending': {'win_rate': 0.4, 'avg_return': 0.005},
            'range': {'win_rate': 0.7, 'avg_return': 0.015}
        },
        "breakout": {
            'stable': {'win_rate': 0.4, 'avg_return': 0.003},
            'volatile': {'win_rate': 0.6, 'avg_return': 0.015},
            'trending': {'win_rate': 0.9, 'avg_return': 0.04},
            'range': {'win_rate': 0.3, 'avg_return': -0.005}
        }
    }
    
    # Pre-populate some performance data to make the demo more interesting
    for strategy, perf_data in strategy_performance.items():
        strat_perf = allocator.strategy_performance[strategy]
        for regime, metrics in perf_data.items():
            strat_perf.regime_performance[regime] = {
                'trades': 20,
                'win_rate': metrics['win_rate'],
                'avg_return': metrics['avg_return']
            }
            
        # Set some base metrics
        strat_perf.sharpe_ratio = 1.5
        strat_perf.win_rate = 0.55
        strat_perf.max_drawdown = 0.10
    
    # Define strategy performance in different regimes
    strategy_performance = {
        "momentum": {
            'stable': {'win_rate': 0.5, 'avg_return': 0.005},
            'volatile': {'win_rate': 0.7, 'avg_return': 0.02},
            'trending': {'win_rate': 0.8, 'avg_return': 0.03},
            'range': {'win_rate': 0.4, 'avg_return': 0.002}
        },
        "mean_reversion": {
            'stable': {'win_rate': 0.6, 'avg_return': 0.01},
            'volatile': {'win_rate': 0.3, 'avg_return': -0.01},
            'trending': {'win_rate': 0.4, 'avg_return': 0.005},
            'range': {'win_rate': 0.7, 'avg_return': 0.015}
        },
        "breakout": {
            'stable': {'win_rate': 0.4, 'avg_return': 0.003},
            'volatile': {'win_rate': 0.6, 'avg_return': 0.015},
            'trending': {'win_rate': 0.9, 'avg_return': 0.04},
            'range': {'win_rate': 0.3, 'avg_return': -0.005}
        }
    }
    
    # Test different market regimes
    regimes = ['stable', 'volatile', 'trending', 'range']
    
    print("=== Adaptive Strategy Allocator Demo ===\n")
    
    for regime in regimes:
        print(f"\n--- Testing {regime.upper()} Regime ---")
        
        # Create demo data for this regime
        data = create_demo_data(regime, strategy_performance, allocator)
        
        # First update the regime directly to ensure it's set correctly
        allocator.current_regime = regime
        
        # Process through orchestrator
        results = orchestrator.on_market_data(data)
        
        # Get current allocation
        params = allocator.get_params()
        allocation = params['strategy_weights']
        current_regime = params['current_regime']
        
        print(f"Detected regime: {current_regime}")
        print("Strategy allocation:")
        for strategy, weight in sorted(allocation.items(), key=lambda x: -x[1]):
            print(f"  {strategy}: {weight:.1%}")
        
        # Show performance metrics
        print("\nStrategy performance in this regime:")
        for strategy in strategies:
            perf = allocator.strategy_performance[strategy]
            regime_metrics = perf.get_regime_metrics(regime)
            print(f"  {strategy}:")
            print(f"    Win rate: {regime_metrics['win_rate']:.0%}")
            print(f"    Avg return: {regime_metrics['avg_return']:.1%}")
            print(f"    Overall Sharpe: {perf.sharpe_ratio:.2f}")
            print(f"    Overall Drawdown: {perf.max_drawdown:.1%}")
        
        print("\n" + "="*50)
    
    # Show final state
    print("\n=== Final System State ===")
    print("Allocation history:")
    for i, entry in enumerate(allocator.allocation_history[-4:]):
        print(f"Cycle {i+1}:")
        print(f"  Regime: {entry['regime']}")
        for strat, weight in entry['allocation'].items():
            print(f"  {strat}: {weight:.1%}")
        print()


if __name__ == "__main__":
    run_demo()