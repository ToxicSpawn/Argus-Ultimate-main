#!/usr/bin/env python3
"""Test Omega Adaptation System - The Pinnacle."""
import asyncio
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from adaptive.omega_adaptation import get_omega_adaptation


async def test():
    print("=" * 70)
    print("OMEGA ADAPTATION SYSTEM - THE ABSOLUTE PINNACLE")
    print("=" * 70)
    
    system = get_omega_adaptation(qubits=28)
    
    # Generate comprehensive test data
    base_price = 50000
    
    # Multi-timeframe
    multi_timeframe = {}
    for tf in ['1s', '1m', '5m', '1h', '1d']:
        n = 100
        prices = [base_price + np.random.randn() * 100 for _ in range(n)]
        volumes = [1000 + np.random.randn() * 200 for _ in range(n)]
        multi_timeframe[tf] = {'prices': prices, 'volumes': volumes}
    
    # Cross-asset
    cross_asset = {
        'BTC': [base_price + np.random.randn() * 100 for _ in range(100)],
        'ETH': [base_price * 0.05 + np.random.randn() * 5 for _ in range(100)],
        'SPY': [450 + np.random.randn() * 5 for _ in range(100)],
        'DXY': [100 + np.random.randn() * 1 for _ in range(100)],
        'Gold': [2000 + np.random.randn() * 20 for _ in range(100)],
    }
    
    # Options data
    options_data = {
        'put_call_ratio': 0.8,
        'iv_rank': 0.6,
        'skew': -0.2,
    }
    
    # On-chain data
    onchain_data = {
        'whale_movement': 0.3,
        'exchange_flow': -0.2,
        'miner_flow': 0.1,
    }
    
    # Sentiment data
    sentiment_data = {
        'fear_greed': 0.65,
        'social_sentiment': 0.55,
        'news_sentiment': 0.5,
    }
    
    # Market data
    market_data = {'prices': multi_timeframe['5m']['prices']}
    
    print("\nRunning Omega Adaptation Analysis...")
    print("-" * 70)
    
    result = await system.omega_adapt(
        market_data=market_data,
        multi_timeframe_data=multi_timeframe,
        cross_asset_data=cross_asset,
        options_data=options_data,
        onchain_data=onchain_data,
        sentiment_data=sentiment_data,
    )
    
    status = system.get_omega_status()
    
    print(f"\nREGIME DETECTION")
    print(f"  Current Regime: {status['regime']}")
    print(f"  Confidence: {status['confidence']:.1%}")
    print(f"  Predicted Regime: {status['predicted_regime']}")
    print(f"  Transition in: {status['transition_in_seconds']:.0f} seconds")
    
    print(f"\nENSEMBLE INTELLIGENCE")
    print(f"  Models Active: {status['models_active']}")
    print(f"  Ensemble Confidence: {status['ensemble_confidence']:.1%}")
    
    print(f"\nADAPTATION OUTPUT")
    print(f"  Position Multiplier: {status['position_multiplier']:.2f}")
    print(f"  Adaptation Score: {status['adaptation_score']:.1%}")
    
    print(f"\nRISK ASSESSMENT")
    print(f"  Black Swan Risk: {status['black_swan_risk']:.1%}")
    
    print(f"\nADAPTIVE LEARNING")
    print(f"  Decisions Made: {status['learning']['decisions']}")
    print(f"  Correct Rate: {status['learning']['correct_rate']:.1%}")
    
    print(f"\nSTRATEGY WEIGHTS")
    for strategy, weight in sorted(result.strategy_weights.items(), key=lambda x: -x[1]):
        bar = "#" * int(weight * 40)
        print(f"  {strategy:15s}: {weight:.1%} {bar}")
    
    print(f"\nSYSTEM INFO")
    print(f"  Qubits: {status['qubits']}")
    print(f"  State Space: {2**status['qubits']:,} states")
    print(f"  Cycles: {status['cycle']}")
    
    print("\n" + "=" * 70)
    print("OMEGA PINNACLE FEATURES:")
    print("  1. Transformer Prediction (regime + timing)")
    print("  2. 10-Model Ensemble Voting")
    print("  3. Adaptive Meta-Learning")
    print("  4. Black Swan Early Detection")
    print("  5. 28-Qubit Quantum Enhancement")
    print("  6. Options Flow Integration")
    print("  7. On-Chain Analytics")
    print("  8. Sentiment Analysis")
    print("  9. Cross-Asset Correlation")
    print(" 10. Self-Optimizing Parameters")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test())
