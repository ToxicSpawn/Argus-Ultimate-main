#!/usr/bin/env py
"""
Smoke test for Maximum Trading Engine.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import math
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s')
logger = logging.getLogger(__name__)

from trading.maximum_trading_engine import MaximumTradingEngine, SignalDirection


def test_maximum_trading_engine():
    """Test the Maximum Trading Engine."""
    print("\n" + "=" * 70)
    print("  MAXIMUM TRADING ENGINE - SMOKE TEST")
    print("=" * 70 + "\n")

    engine = MaximumTradingEngine()

    # Generate synthetic market data
    print("1. Generating synthetic market data...")
    random.seed(42)
    
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "DOGE/USDT"]
    market_data = {}
    
    for symbol in symbols:
        base_price = {"BTC/USDT": 65000, "ETH/USDT": 3500, "SOL/USDT": 150, 
                      "AVAX/USDT": 40, "DOGE/USDT": 0.15}[symbol]
        
        # Generate price history (enough for all timeframes)
        prices = [base_price]
        for _ in range(1500):
            prices.append(prices[-1] * (1 + random.gauss(0, 0.01)))
        
        # Generate signals
        signals = []
        for i in range(5):
            conf = random.uniform(0.3, 0.9)
            if random.random() > 0.5:
                direction = "buy"
            else:
                direction = "sell"
            signals.append({
                "strategy": f"strategy_{i}",
                "direction": direction,
                "confidence": conf,
            })
        
        market_data[symbol] = {
            "close": prices[-1],
            "open": prices[-2],
            "high": prices[-1] * 1.005,
            "low": prices[-1] * 0.995,
            "volume": random.uniform(1000000, 10000000),
            "avg_volume": 5000000,
            "spread_bps": random.uniform(1, 15),
            "signals": signals,
            "timeframes": {
                "1m": {"close": prices[-1], "open": prices[-5], "high": prices[-1]*1.001, "low": prices[-1]*0.999},
                "5m": {"close": prices[-1], "open": prices[-25], "high": max(prices[-5:]), "low": min(prices[-5:])},
                "15m": {"close": prices[-1], "open": prices[-75], "high": max(prices[-15:]), "low": min(prices[-15:])},
                "1h": {"close": prices[-1], "open": prices[-300], "high": max(prices[-60:]), "low": min(prices[-60:])},
                "4h": {"close": prices[-1], "open": prices[-1200], "high": max(prices[-240:]), "low": min(prices[-240:])},
                "1d": {"close": prices[-1], "open": prices[0], "high": max(prices), "low": min(prices)},
            },
        }
    
    print(f"   Generated data for {len(symbols)} symbols")

    # Test asset selection
    print("\n2. Testing Asset Selection (Pillar 3)...")
    rankings = engine.selection_engine.rank_assets(symbols, market_data)
    for r in rankings[:3]:
        print(f"   #{r.rank} {r.symbol}: opportunity={r.opportunity_score:.3f}, "
              f"momentum={r.momentum_score:.3f}, flow={r.flow_score:.3f}")
        print(f"      Reason: {r.selection_reason}")

    # Test signal quality
    print("\n3. Testing Signal Quality (Pillar 1)...")
    best_symbol = rankings[0].symbol
    best_data = market_data[best_symbol]
    
    market_state = {
        "volume_ratio": 1.5,
        "spread_bps": best_data["spread_bps"],
        "regime": "ranging",
        "hour": 14,
        "correlation_with_portfolio": 0.0,
    }
    
    signal_quality = engine.signal_engine.assess_signal(
        best_data["signals"], market_state
    )
    print(f"   Symbol: {best_symbol}")
    print(f"   Raw confidence: {signal_quality.raw_confidence:.3f}")
    print(f"   Calibrated confidence: {signal_quality.calibrated_confidence:.3f}")
    print(f"   Quality score: {signal_quality.quality_score:.3f}")
    print(f"   Ensemble agreement: {signal_quality.ensemble_agreement:.3f}")
    print(f"   Meta-label probability: {signal_quality.meta_label_prob:.3f}")
    print(f"   Passed quality gate: {signal_quality.passed_quality_gate}")
    print(f"   Quality factors:")
    for factor, score in signal_quality.quality_factors.items():
        print(f"      {factor}: {score:.3f}")

    # Test market timing
    print("\n4. Testing Market Timing (Pillar 2)...")
    timing = engine.timing_engine.assess_timing(
        best_symbol,
        best_data["timeframes"],
        time.time(),
    )
    print(f"   Regime: {timing.current_regime.value}")
    print(f"   Regime confidence: {timing.regime_confidence:.3f}")
    print(f"   Timing window: {timing.optimal_timing_window.value}")
    print(f"   Timing score: {timing.timing_score:.3f}")
    print(f"   MTF confluence: {timing.mtf_confluence:.3f}")
    print(f"   Session edge: {timing.session_edge:.3f}")
    print(f"   Vol regime: {timing.vol_regime}")
    print(f"   Should trade now: {timing.should_trade_now}")

    # Test execution planning
    print("\n5. Testing Execution Planning (Pillar 4)...")
    execution = engine.execution_engine.create_execution_plan(
        symbol=best_symbol,
        side="buy",
        size_usd=10000,
        current_price=best_data["close"],
        spread_bps=best_data["spread_bps"],
        urgency="normal",
    )
    print(f"   Algorithm: {execution.algorithm.value}")
    print(f"   Optimal venue: {execution.optimal_venue}")
    print(f"   Estimated fill price: ${execution.estimated_fill_price:,.2f}")
    print(f"   Expected slippage: {execution.expected_slippage_bps:.2f}bps")
    print(f"   Expected fill rate: {execution.expected_fill_rate:.2%}")
    print(f"   Maker/Taker: {execution.maker_taker_decision}")
    print(f"   Slice count: {execution.slice_count}")
    print(f"   Duration: {execution.duration_seconds}s")
    print(f"   Fee optimization: {execution.fee_optimization_bps:.2f}bps saved")

    # Test full trade generation
    print("\n6. Testing Full Trade Generation...")
    decision = engine.generate_trade(
        symbols=symbols,
        market_data=market_data,
        portfolio_equity=100000,
        existing_positions={},
        current_time_utc=time.time(),
    )
    
    print(f"\n   === TRADE DECISION ===")
    print(f"   Should trade: {decision.should_trade}")
    print(f"   Direction: {decision.direction.value}")
    print(f"   Symbol: {decision.symbol}")
    print(f"   Size: ${decision.size_usd:,.2f}")
    print(f"   Composite score: {decision.composite_score:.1f}/100")
    print(f"   Expected edge: {decision.expected_edge_bps:.1f}bps")
    print(f"   Confidence: {decision.confidence:.2%}")
    print(f"   Kelly fraction: {decision.kelly_fraction:.3f}")
    print(f"   Execute immediately: {decision.execute_immediately}")
    print(f"\n   Reasoning:")
    for reason in decision.reasoning:
        print(f"      - {reason}")

    # Test with different market conditions
    print("\n7. Testing Different Market Conditions...")
    
    # Strong bullish signal
    market_data["BTC/USDT"]["signals"] = [
        {"strategy": f"strat_{i}", "direction": "buy", "confidence": 0.8 + random.random() * 0.15}
        for i in range(7)
    ]
    
    decision_bull = engine.generate_trade(
        symbols=["BTC/USDT"],
        market_data={"BTC/USDT": market_data["BTC/USDT"]},
        portfolio_equity=100000,
    )
    print(f"   Strong bullish signals:")
    print(f"      Should trade: {decision_bull.should_trade}")
    print(f"      Direction: {decision_bull.direction.value}")
    print(f"      Composite score: {decision_bull.composite_score:.1f}")
    
    # Mixed signals
    market_data["BTC/USDT"]["signals"] = [
        {"strategy": "strat_0", "direction": "buy", "confidence": 0.7},
        {"strategy": "strat_1", "direction": "sell", "confidence": 0.6},
        {"strategy": "strat_2", "direction": "buy", "confidence": 0.5},
        {"strategy": "strat_3", "direction": "sell", "confidence": 0.8},
    ]
    
    decision_mixed = engine.generate_trade(
        symbols=["BTC/USDT"],
        market_data={"BTC/USDT": market_data["BTC/USDT"]},
        portfolio_equity=100000,
    )
    print(f"\n   Mixed signals:")
    print(f"      Should trade: {decision_mixed.should_trade}")
    print(f"      Composite score: {decision_mixed.composite_score:.1f}")

    print("\n" + "=" * 70)
    print("  MAXIMUM TRADING ENGINE - TEST COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_maximum_trading_engine()
