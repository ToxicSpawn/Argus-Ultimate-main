"""Test unified integration with adjusted parameters."""
import sys
sys.path.insert(0, '.')

from scripts.unified_integration import UnifiedTradingIntegration
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

print("=" * 60)
print("UNIFIED INTEGRATION TEST")
print("=" * 60)
print()

# Create with faster settings for testing
integration = UnifiedTradingIntegration(
    min_confidence=0.50,
    min_trade_interval=1,
    max_trades_per_hour=10
)

# Simulate data
np.random.seed(42)
prices = 50000 + np.cumsum(np.random.randn(100) * 100)
df = pd.DataFrame({
    'open': prices,
    'high': prices + np.random.rand(100) * 50,
    'low': prices - np.random.rand(100) * 50,
    'close': prices,
    'volume': np.random.rand(100) * 1000
})

# Generate signals
trades = 0
for i in range(24, len(df)):
    signal = integration.generate_signal(df.iloc[:i+1], df['close'].iloc[i])
    
    # Check if should trade
    should, reason = integration.should_trade(signal)
    
    if should:
        # Simulate trade
        size = integration.optimize_position_size(signal)
        pnl = size * np.random.randn() * 0.001
        actual_return = np.random.randn() * 0.01
        
        # Record outcome
        result = integration.record_outcome(signal, pnl, actual_return)
        trades += 1
        
        if trades <= 5:
            print("Trade {}: {} conf={:.0%} correct={} acc={:.0%}".format(
                trades, signal['action'], signal['confidence'], 
                result['correct'], result['accuracy']))

print()
perf = integration.get_performance()
print("Total trades: {}".format(perf['total_trades']))
print("Win rate: {:.0%}".format(perf['win_rate']))
print("Total PnL: ${:.2f}".format(perf['total_pnl']))
print("Accuracy: {:.0%}".format(perf['accuracy']))
print("Drift events: {}".format(perf.get('drift_count', 0)))
print("Ultimate learner: {}".format(perf.get('ultimate_learner', False)))