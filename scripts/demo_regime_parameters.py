"""
Demonstration of RegimeSpecificParameters

This script shows how the RegimeSpecificParameters component works with:
1. Different market regimes
2. Parameter adjustments based on regime
3. Performance tracking
4. Parameter bounds enforcement
"""

import logging
from datetime import datetime, timezone
from core.real_time_learning.orchestrator import RealTimeLearningOrchestrator
from core.real_time_learning.regime_parameters import RegimeSpecificParameters

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_performance_metrics(regime):
    """Generate synthetic performance metrics for a regime"""
    # Base metrics
    metrics = {
        'sharpe_ratio': 1.2,
        'win_rate': 0.55,
        'max_drawdown': 0.15,
        'profit_factor': 1.5
    }
    
    # Regime-specific adjustments
    if regime == 'volatile':
        metrics['sharpe_ratio'] = 0.8
        metrics['win_rate'] = 0.45
        metrics['max_drawdown'] = 0.25
        metrics['profit_factor'] = 1.2
    elif regime == 'trending':
        metrics['sharpe_ratio'] = 2.0
        metrics['win_rate'] = 0.7
        metrics['max_drawdown'] = 0.10
        metrics['profit_factor'] = 2.5
    elif regime == 'range':
        metrics['sharpe_ratio'] = 1.5
        metrics['win_rate'] = 0.6
        metrics['max_drawdown'] = 0.12
        metrics['profit_factor'] = 1.8
    
    return metrics


def run_demo():
    """Run the regime-specific parameters demonstration"""
    
    # Create orchestrator and regime parameters
    orchestrator = RealTimeLearningOrchestrator()
    regime_params = RegimeSpecificParameters()
    
    # Set update frequency to 1 for demo purposes
    regime_params.update_frequency = 1
    
    # Register the component
    orchestrator.register_component(regime_params)
    
    # Initialize with 4 regimes
    regimes = ["stable", "volatile", "trending", "range"]
    regime_params.initialize_regimes(regimes)
    
    print("=== Regime-Specific Parameters Demo ===\n")
    
    # Test different market regimes
    for regime in regimes:
        print(f"\n--- Testing {regime.upper()} Regime ---")
        
        # Create market data for this regime
        market_data = {
            'volatility': 0.01,
            'trend_strength': 0.2
        }
        
        if regime == 'volatile':
            market_data['volatility'] = 0.03
        elif regime == 'trending':
            market_data['trend_strength'] = 0.6
        elif regime == 'range':
            market_data['volatility'] = 0.005
        
        # Create performance metrics
        performance_metrics = generate_performance_metrics(regime)
        
        # Create input data
        data = {
            'market_data': market_data,
            'performance_metrics': performance_metrics
        }
        
        # Process through orchestrator
        results = orchestrator.on_market_data(data)
        
        # Get current state
        params = regime_params.get_params()
        current_regime = params['current_regime']
        
        print(f"Detected regime: {current_regime}")
        
        # Show current parameters
        print("\nCurrent trading parameters:")
        for param, value in sorted(params['current_parameters'].items()):
            print(f"  {param}: {value:.3f}")
        
        # Show regime performance
        regime_perf = regime_params.get_regime_performance(current_regime)
        print(f"\nRegime performance ({current_regime}):")
        print(f"  Sharpe ratio: {regime_perf['sharpe_ratio']:.2f}")
        print(f"  Win rate: {regime_perf['win_rate']:.2f}")
        print(f"  Max drawdown: {regime_perf['max_drawdown']:.2f}")
        print(f"  Profit factor: {regime_perf['profit_factor']:.2f}")
        
        # Show regime adjustments
        adjustments = regime_params.regime_adjustments[current_regime]
        print(f"\nRegime adjustments for {current_regime}:")
        for param, factor in adjustments.items():
            print(f"  {param}: x{factor:.1f}")
        
        print("\n" + "="*50)
    
    # Show final state
    print("\n=== Final System State ===")
    print(f"Regime history (last 5): {regime_params.get_params()['regime_history']}")
    
    # Test parameter retrieval
    print(f"\nCurrent position size: {regime_params.get_parameter('position_size_pct'):.3f}")
    print(f"Current max leverage: {regime_params.get_parameter('max_leverage'):.1f}")
    
    # Show performance history
    print(f"\nPerformance history length: {len(regime_params.performance_history)}")
    if regime_params.performance_history:
        last_perf = regime_params.performance_history[-1]
        print(f"Last performance record:")
        print(f"  Regime: {last_perf['regime']}")
        print(f"  Sharpe: {last_perf['sharpe_ratio']:.2f}")
        print(f"  Win rate: {last_perf['win_rate']:.2f}")


if __name__ == "__main__":
    run_demo()