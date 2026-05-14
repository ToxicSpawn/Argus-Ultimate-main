#!/usr/bin/env python3
"""Test quantum adaptation system."""
import asyncio
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from adaptive.quantum_adaptation import get_quantum_adaptation


async def test():
    print("=" * 70)
    print("QUANTUM-ENHANCED ADAPTATION SYSTEM")
    print("=" * 70)
    
    system = get_quantum_adaptation(qubits=16)
    
    # Generate test data
    base_price = 50000
    
    timeframe_data = {}
    for tf in ['1s', '1m', '5m', '1h', '1d']:
        n = {'1s': 100, '1m': 100, '5m': 100, '1h': 100, '1d': 50}[tf]
        prices = [base_price + np.random.randn() * 100 for _ in range(n)]
        volumes = [1000 + np.random.randn() * 200 for _ in range(n)]
        timeframe_data[tf] = {'prices': prices, 'volumes': volumes}
    
    cross_asset_data = {
        'BTC': [base_price + np.random.randn() * 100 for _ in range(100)],
        'ETH': [base_price * 0.05 + np.random.randn() * 5 for _ in range(100)],
        'SPY': [450 + np.random.randn() * 5 for _ in range(100)],
    }
    
    market_data = {'prices': timeframe_data['5m']['prices']}
    
    # Run quantum adaptation
    result = await system.quantum_adapt(market_data, timeframe_data, cross_asset_data)
    
    print("\nQUANTUM REGIME DETECTION (Superposition)")
    print("-" * 50)
    print(f"Detected Regime: {result['quantum_regime']}")
    print(f"Quantum Confidence: {result['quantum_confidence']:.2%}")
    print(f"Timeframe Agreement: {result['timeframe_agreement']:.2%}")
    
    print("\nQUANTUM STRATEGY OPTIMIZATION (QAOA)")
    print("-" * 50)
    for strategy, weight in sorted(result['strategy_weights'].items(), key=lambda x: -x[1]):
        bar = "#" * int(weight * 40)
        print(f"  {strategy:15s}: {weight:.1%} {bar}")
    
    print("\nQUANTUM CORRELATION ANALYSIS (Entanglement)")
    print("-" * 50)
    for pair, data in result['correlation_signals'].items():
        print(f"  {pair}:")
        print(f"    Long-term Corr: {data['long_correlation']:.3f}")
        print(f"    Short-term Corr: {data['short_correlation']:.3f}")
        print(f"    Signal: {data['signal']}")
    
    print("\nQUANTUM RISK CALCULATION (Monte Carlo - 1000 Universes)")
    print("-" * 50)
    risk = result['quantum_risk']
    print(f"  VaR (95%): ${risk['var_95']:.2f}")
    print(f"  CVaR (95%): ${risk['cvar_95']:.2f}")
    print(f"  Worst Case: ${risk['worst_case']:.2f}")
    print(f"  Best Case: ${risk['best_case']:.2f}")
    
    print("\nQUANTUM ADAPTATION OUTPUT")
    print("-" * 50)
    print(f"  Position Multiplier: {result['position_multiplier']:.2f}")
    print(f"  Quantum Advantage: {result['quantum_advantage']:.2f}x")
    print(f"  Qubits Used: {result['qubits_used']}")
    print(f"  State Space: {2**result['qubits_used']:,} states")
    
    print("\n" + "=" * 70)
    print("QUANTUM ADVANTAGES:")
    print("  1. Superposition: Tests ALL regimes simultaneously")
    print("  2. Entanglement: Correlates timeframes instantly")
    print("  3. Interference: Amplifies correct signals")
    print("  4. Parallel Universes: Risk calculated across 1000 scenarios")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test())
