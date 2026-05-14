"""
Demonstration of SmartOrderRouter

This script shows how the SmartOrderRouter works with:
1. Different market regimes
2. Venue performance tracking
3. Dynamic order routing
4. Execution quality analysis
"""

import logging
from datetime import datetime, timezone
import random
from core.real_time_learning.orchestrator import RealTimeLearningOrchestrator
from core.real_time_learning.order_router import SmartOrderRouter, OrderRoutingDecision

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_execution_reports(venues, num_reports=20):
    """Generate synthetic execution reports for testing"""
    reports = []
    
    for i in range(num_reports):
        venue = random.choice(venues)
        order_type = random.choice(['market', 'limit', 'stop'])
        order_size = random.choice(['small', 'medium', 'large'])
        
        # Base metrics
        base_fill = 0.8 + random.random() * 0.2
        base_slippage = 0.0005 + random.random() * 0.001
        base_latency = 20 + random.randint(0, 60)
        
        # Venue-specific adjustments
        if venue == 'binance':
            base_fill += 0.05
            base_slippage -= 0.0002
            base_latency -= 10
        elif venue == 'bybit':
            base_fill -= 0.02
            base_slippage += 0.0003
            base_latency += 5
        elif venue == 'okx':
            base_fill -= 0.03
            base_slippage += 0.0005
            base_latency += 15
        
        report = {
            'venue': venue,
            'order_type': order_type,
            'order_size': order_size,
            'fill_ratio': min(1.0, max(0.5, base_fill)),
            'slippage': max(0.0, base_slippage),
            'latency': max(10, base_latency),
            'success': random.random() > 0.05  # 95% success rate
        }
        
        reports.append(report)
    
    return reports


def run_demo():
    """Run the smart order router demonstration"""
    
    # Create orchestrator and order router
    orchestrator = RealTimeLearningOrchestrator()
    order_router = SmartOrderRouter()
    
    # Set update frequency to 1 for demo purposes
    order_router.update_frequency = 1
    
    # Register the component
    orchestrator.register_component(order_router)
    
    # Initialize with 3 venues
    venues = ["binance", "bybit", "okx"]
    order_router.initialize_venues(venues)
    
    # Generate synthetic execution reports
    execution_reports = generate_execution_reports(venues)
    
    print("=== Smart Order Router Demo ===\n")
    
    # Test different market regimes
    regimes = ['stable', 'volatile', 'trending', 'range']
    
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
        
        # Create input data
        data = {
            'market_data': market_data,
            'execution_reports': execution_reports
        }
        
        # Process through orchestrator
        results = orchestrator.on_market_data(data)
        
        # Get current state
        params = order_router.get_params()
        current_regime = params['current_regime']
        
        print(f"Detected regime: {current_regime}")
        
        # Show venue performance
        print("\nVenue performance:")
        for venue, perf in params['venue_performance'].items():
            print(f"  {venue}:")
            print(f"    Liquidity: {perf['liquidity_score']:.2f}")
            print(f"    Fill ratio: {perf['fill_ratio']:.2f}")
            print(f"    Slippage: {perf['avg_slippage']:.3f}")
            print(f"    Latency: {perf['latency']}ms")
        
        # Route a test order
        test_order = {
            'order_id': f'test_{regime}',
            'order_type': 'market',
            'order_size': 'medium',
            'symbol': 'BTCUSDT'
        }
        
        decision = order_router.route_order(test_order)
        
        print(f"\nRouting decision for {test_order['symbol']}:")
        print(f"  Selected venue: {decision.venue}")
        print(f"  Reason: {decision.reason}")
        print(f"  Expected fill ratio: {decision.expected_fill_ratio:.2f}")
        print(f"  Expected slippage: {decision.expected_slippage:.3f}")
        print(f"  Expected latency: {decision.expected_latency}ms")
        print(f"  Confidence: {decision.confidence:.2f}")
        
        # Show regime-specific weights
        regime_weights = order_router.regime_adjustments.get(current_regime, 
                                                           order_router.regime_adjustments['stable'])
        print(f"\nRegime-specific weights for {current_regime}:")
        print(f"  Liquidity: {regime_weights['liquidity_weight']:.1f}")
        print(f"  Latency: {regime_weights['latency_weight']:.1f}")
        print(f"  Slippage: {regime_weights['slippage_weight']:.1f}")
        
        print("\n" + "="*50)
    
    # Show final state
    print("\n=== Final System State ===")
    print(f"Routing history length: {len(order_router.routing_history)}")
    print("Recent routing decisions:")
    for i, decision in enumerate(order_router.get_routing_history(4)):
        print(f"  Decision {i+1}:")
        print(f"    Order: {decision.order_id}")
        print(f"    Venue: {decision.venue}")
        print(f"    Confidence: {decision.confidence:.2f}")


if __name__ == "__main__":
    run_demo()