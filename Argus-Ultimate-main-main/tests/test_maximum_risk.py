#!/usr/bin/env py
"""
Smoke test for Maximum Risk Engine.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import math
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s')
logger = logging.getLogger(__name__)

from risk.maximum_risk_engine import MaximumRiskEngine, RiskLevel, PositionAction


def test_maximum_risk_engine():
    """Test the Maximum Risk Engine."""
    print("\n" + "=" * 70)
    print("  MAXIMUM RISK ENGINE - SMOKE TEST")
    print("=" * 70 + "\n")

    # Initialize engine
    engine = MaximumRiskEngine({
        "max_daily_loss_pct": 5.0,
        "max_drawdown_pct": 15.0,
        "max_position_pct": 25.0,
        "max_leverage": 5.0,
        "kelly_fraction": 0.5,
    })

    # Simulate market data
    print("1. Simulating market data (100 ticks)...")
    price = 65000.0
    random.seed(42)

    for i in range(100):
        # Random walk with drift
        shock = random.gauss(0, 1)
        ret = 0.0001 + 0.01 * shock  # small drift + volatility
        price = price * (1 + ret)
        volume = random.uniform(100, 1000)
        high = price * (1 + abs(random.gauss(0, 0.002)))
        low = price * (1 - abs(random.gauss(0, 0.002)))

        engine.update_market(price, volume, high, low)
        engine.update_equity(100000 + random.gauss(0, 500))

    print(f"   Price: ${price:,.2f}")

    # Test risk prediction
    print("\n2. Testing Risk Prediction...")
    prediction = engine.risk_predictor.predict()
    print(f"   VaR 5m:  {prediction.predicted_var_5m:.4%}")
    print(f"   VaR 15m: {prediction.predicted_var_15m:.4%}")
    print(f"   VaR 30m: {prediction.predicted_var_30m:.4%}")
    print(f"   Vol Regime: {prediction.volatility_regime}")
    print(f"   Crash Prob: {prediction.crash_probability:.1%}")
    print(f"   Reversal Prob: {prediction.trend_reversal_probability:.1%}")
    print(f"   Confidence: {prediction.confidence:.1%}")

    # Test drawdown forecast
    print("\n3. Testing Drawdown Forecast...")
    dd_forecast = engine.drawdown_forecaster.forecast(hours=24)
    print(f"   Current DD: {dd_forecast.current_drawdown_pct:.2f}%")
    print(f"   Predicted Max DD 1h: {dd_forecast.predicted_max_dd_1h:.2f}%")
    print(f"   Predicted Max DD 4h: {dd_forecast.predicted_max_dd_4h:.2f}%")
    print(f"   Predicted Max DD 24h: {dd_forecast.predicted_max_dd_24h:.2f}%")
    print(f"   Recovery Prob: {dd_forecast.recovery_probability:.1%}")

    # Test adaptive stop-loss
    print("\n4. Testing Adaptive Stop-Loss...")
    engine.adaptive_stop.set_entry(price, "buy")  # Set entry first
    for regime in ["low", "normal", "high", "extreme"]:
        stop = engine.adaptive_stop.get_stop("buy", price, regime)
        print(f"   {regime:8s}: Stop={stop.stop_loss_price:,.2f} ({stop.stop_loss_pct:.2f}%), "
              f"Trailing={'Y' if stop.trailing_stop_active else 'N'}")

    # Test position sizing
    print("\n5. Testing Position Sizing...")
    for risk_level_name in ["MINIMAL", "LOW", "MODERATE", "HIGH"]:
        # Simulate different risk levels by adjusting equity
        if risk_level_name == "MINIMAL":
            engine.update_equity(105000)  # profit
        elif risk_level_name == "LOW":
            engine.update_equity(100000)  # flat
        elif risk_level_name == "MODERATE":
            engine.update_equity(95000)   # 5% drawdown
        else:
            engine.update_equity(88000)   # 12% drawdown

        sizing = engine.calculate_position_size("BTC/USDT", "buy", 10000)
        print(f"   {risk_level_name:10s}: Kelly={sizing.kelly_fraction:.3f}, "
              f"Adjusted={sizing.adjusted_fraction:.3f}, "
              f"Size=${sizing.recommended_size_usd:,.0f}")

    # Test trade evaluation
    print("\n6. Testing Trade Evaluation...")
    engine.update_equity(100000)

    # Normal trade
    decision = engine.evaluate_trade(
        symbol="BTC/USDT",
        side="buy",
        size_usd=5000,
        current_price=65000,
        portfolio_equity=100000,
        win_rate=0.55,
        win_loss_ratio=1.5,
    )
    print(f"   Normal Trade:")
    print(f"     Approved: {decision.approved}")
    print(f"     Action: {decision.action.value}")
    print(f"     Size: ${decision.approved_size_usd:,.0f}")
    print(f"     Stop: ${decision.stop_loss.stop_loss_price:,.2f} ({decision.stop_loss.stop_loss_pct:.2f}%)")
    print(f"     Take Profit: {decision.take_profit_pct:.2f}%")
    print(f"     Risk Score: {decision.risk_score:.1f}/10")
    print(f"     Risk Level: {decision.risk_level.value}")
    if decision.conditions:
        print(f"     Conditions: {', '.join(decision.conditions)}")

    # Test system status
    print("\n7. Testing System Status...")
    status = engine.get_system_status()
    print(f"   Risk Level: {status.risk_level.value}")
    print(f"   Risk Score: {status.risk_score:.1f}/10")
    print(f"   Can Trade: {status.can_trade}")
    print(f"   Circuit Breaker: {status.circuit_breaker_active}")
    print(f"   Position Multiplier: {status.position_multiplier:.2f}")
    if status.recommendations:
        print(f"   Recommendations:")
        for rec in status.recommendations:
            print(f"     - {rec}")

    # Test halt scenario
    print("\n8. Testing Halt Scenario...")
    engine.update_equity(80000)  # 20% drawdown - should trigger halt
    engine.update_equity(80000)
    decision = engine.evaluate_trade(
        symbol="BTC/USDT",
        side="buy",
        size_usd=5000,
        current_price=65000,
        portfolio_equity=80000,
    )
    print(f"   After 20% DD:")
    print(f"     Approved: {decision.approved}")
    print(f"     Halted: {not decision.approved}")
    print(f"     Conditions: {decision.conditions}")

    print("\n" + "=" * 70)
    print("  SMOKE TEST COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_maximum_risk_engine()
