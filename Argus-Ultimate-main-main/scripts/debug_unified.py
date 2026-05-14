"""Debug unified integration."""
import sys
sys.path.insert(0, '.')

from scripts.unified_integration import UnifiedTradingIntegration
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

# Create with faster settings
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
for i in range(24, 30):
    signal = integration.generate_signal(df.iloc[:i+1], df['close'].iloc[i])
    
    print("i={} action={} dir={} conf={:.2f}".format(
        i, signal['action'], signal['direction'], signal['confidence']))
    
    should, reason = integration.should_trade(signal)
    print("  should_trade={} reason={}".format(should, reason))