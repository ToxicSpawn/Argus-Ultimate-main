"""Quick test of complete trading system."""
import sys
sys.path.insert(0, '.')

from scripts.complete_trading_system import CompleteTradingSystem
import numpy as np

# Test with cycle-based intervals
system = CompleteTradingSystem(initial_capital=10000, min_trade_interval=3, max_trades_per_hour=10, min_confidence=0.50)

trades_count = 0
for i in range(20):
    features = np.random.randn(9)
    signal = system.generate_signal(features, 'sideways')
    
    should_trade, reason = system.should_trade(signal, {'volatility': 0.01, 'volume_ratio': 1.0})
    
    if i < 5:
        print("i={} action={} conf={:.2f} trade={} reason={}".format(i, signal['action'], signal['confidence'], should_trade, reason))
    
    if should_trade:
        trade = system.execute_trade(signal, {'volatility': 0.01, 'volume_ratio': 1.0, 'price': 50000}, 50000, 1000)
        if trade:
            trades_count += 1
            system.close_position(50000 * 1.01)
            system.record_outcome(signal, 0.01)
    
    system.advance_cycle()

print("Total: {}".format(trades_count))