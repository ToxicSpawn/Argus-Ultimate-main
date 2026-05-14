#!/usr/bin/env python3
"""Test advanced adaptation system."""
import asyncio
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from adaptive.advanced_adaptation import get_advanced_adaptation


async def test():
    system = get_advanced_adaptation()
    
    # Simulate multi-timeframe data
    base_price = 50000
    data = {}
    for tf in ['1s', '1m', '5m', '1h', '1d']:
        n = {'1s': 100, '1m': 100, '5m': 100, '1h': 100, '1d': 50}[tf]
        prices = [base_price + np.random.randn() * 100 for _ in range(n)]
        volumes = [1000 + np.random.randn() * 200 for _ in range(n)]
        data[tf] = {'prices': prices, 'volumes': volumes}
    
    orderbook = {
        'bids': [[base_price - i * 10, 10 + i] for i in range(1, 21)],
        'asks': [[base_price + i * 10, 10 + i] for i in range(1, 21)],
    }
    
    trades = [{'price': base_price + np.random.randn() * 10, 'size': np.random.uniform(0.1, 10), 'side': np.random.choice(['buy', 'sell']), 'timestamp': 0} for _ in range(50)]
    
    cross_asset = {
        'BTC': [base_price + np.random.randn() * 100 for _ in range(100)],
        'ETH': [base_price * 0.05 + np.random.randn() * 5 for _ in range(100)],
        'SPY': [450 + np.random.randn() * 5 for _ in range(100)],
    }
    
    result = await system.analyze(data, orderbook, trades, cross_asset)
    summary = system.get_adaptation_summary()
    
    print('ADVANCED ADAPTATION SYSTEM - MAXIMUM LEVEL')
    print('=' * 60)
    print(f"Regime: {summary['regime']}")
    print(f"Predicted: {summary['predicted_regime']}")
    print(f"Confidence: {summary['confidence']:.0%}")
    print(f"Dominant TF: {summary['dominant_timeframe']}")
    print()
    print('Adaptation Parameters:')
    print(f"  Position Multiplier: {summary['adaptation']['position_multiplier']:.2f}")
    print(f"  Aggressiveness: {summary['adaptation']['aggressiveness']:.2f}")
    print(f"  Risk Multiplier: {summary['adaptation']['risk_multiplier']:.2f}")
    print()
    print('Volatility:')
    print(f"  Realized: {summary['volatility']['realized']:.2%}")
    print(f"  Regime: {summary['volatility']['regime']}")
    print()
    print('Liquidity:')
    print(f"  Regime: {summary['liquidity']['regime']}")
    print(f"  Spread: {summary['liquidity']['spread_bps']:.1f} bps")
    print()
    print('Strategies:')
    for s, w in summary['strategies'].items():
        print(f"  {s}: {w:.0%}")


if __name__ == "__main__":
    asyncio.run(test())
