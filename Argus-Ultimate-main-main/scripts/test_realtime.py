"""Quick test for realtime learning."""
import logging
logging.basicConfig(level=logging.WARNING)
import pandas as pd
import numpy as np
from scripts.realtime_learning import RealTimeLearning

# Create
rt = RealTimeLearning(min_confidence=0.45, min_trade_interval=1, max_trades_per_hour=100, offline=True)

# Quick simulate
np.random.seed(42)
prices = 50000 + np.cumsum(np.random.randn(100) * 100)
df = pd.DataFrame({
    'close': prices,
    'high': prices * 1.01,
    'low': prices * 0.99,
    'volume': np.random.rand(100) * 1000 + 500
})

trades = 0
for i in range(25, len(df)):
    rt.on_bar(df.iloc[:i+1], df['close'].iloc[i])
    signal = rt.get_signal()
    if rt.should_trade(signal)[0]:
        ret = np.random.randn() * 0.01
        trades += 1
        rt.on_trade(1000 * ret, ret)

perf = rt.get_performance()
print("Trades: {} PnL: ${:.0f} Acc: {:.0%} Equity: ${:.0f}".format(
    perf['total_trades'], perf['total_pnl'], perf['accuracy'], perf['equity']))