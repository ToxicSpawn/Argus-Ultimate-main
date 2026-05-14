"""
Enhanced Trading Integration for Argus

Connects EnhancedTradingSystem to live Argus execution.
Works with paper trading or live trading.

Run: py main.py paper (with this module integrated)

Usage:
1. Add to your strategy
2. Or use as standalone with simulated execution
"""

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Import the enhanced trading system
try:
    from scripts.enhanced_trading import EnhancedTradingSystem
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    logger.warning("Enhanced trading not available")


class ArgusTradingIntegration:
    """
    Integration between enhanced trading and Argus execution.
    
    Handles:
    - Data feed from Argus
    - Trade execution (paper or live)
    - Position management
    - PnL tracking
    - Learning updates
    """

    def __init__(
        self,
        symbols: List[str] = None,
        portfolio_value: float = 10000,
        trading_mode: str = 'paper',  # 'paper' or 'live'
    ):
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        self.portfolio_value = portfolio_value
        self.trading_mode = trading_mode
        
        # Enhanced trading system
        if ENHANCED_AVAILABLE:
            self.trading = EnhancedTradingSystem(
                symbols=self.symbols,
                portfolio_value=portfolio_value
            )
        else:
            logger.error("Enhanced trading not available!")
            return
        
        # Position tracking
        self.positions: Dict[str, Dict] = {}
        
        # Trade history
        self.trade_history: deque = deque(maxlen=500)
        
        # Stats
        self.total_pnl = 0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Current prices
        self.prices: Dict[str, float] = {s: 0 for s in self.symbols}
        
        logger.info("=" * 60)
        logger.info("ARGUS TRADING INTEGRATION")
        logger.info("=" * 60)
        logger.info("Mode: {}".format(trading_mode))
        logger.info("Symbols: {}".format(self.symbols))
        logger.info("Portfolio: ${}".format(portfolio_value))
        logger.info("=" * 60)

    def on_data(self, symbol: str, price: float, volume: float, timestamp: datetime = None):
        """
        Called when new market data arrives.
        
        Args:
            symbol: e.g., 'BTC/USDT'
            price: Current price
            volume: Volume
            timestamp: Data timestamp
        """
        if symbol not in self.symbols:
            return
        
        # Update price
        self.prices[symbol] = price
        
        # Update trading system
        self.trading.update(symbol, price)
        
        # Check for trade opportunity
        should, reason = self.trading.should_trade()
        
        # Execute if conditions met
        if should and symbol == self.trading.best_symbol:
            self._execute_trade(symbol)

    def _execute_trade(self, symbol: str):
        """Execute a trade."""
        signal = self.trading.best_signal
        confidence = self.trading.best_confidence
        
        # Calculate position size
        size = self.trading.get_position_size()
        
        if self.trading_mode == 'paper':
            # Paper trade
            self._paper_trade(symbol, signal, size)
        else:
            # Live trade (would call exchange)
            self._live_trade(symbol, signal, size)

    def _paper_trade(self, symbol: str, signal: str, size: float):
        """Simulate paper trade."""
        price = self.prices[symbol]
        
        # Calculate costs
        fee = size * 0.001  # 0.1% fee
        
        # Determine entry
        if signal == 'buy':
            # Long position
            self.positions[symbol] = {
                'type': 'long',
                'entry_price': price,
                'size': size,
                'entry_time': datetime.now(timezone.utc)
            }
            action = "LONG"
        elif signal == 'sell':
            # Short position
            self.positions[symbol] = {
                'type': 'short',
                'entry_price': price,
                'size': size,
                'entry_time': datetime.now(timezone.utc)
            }
            action = "SHORT"
        
        logger.info("📝 PAPER TRADE: {} {} {} @ ${:.0f} (${:.0f})".format(
            action, symbol, size, price, size))

    def _live_trade(self, symbol: str, signal: str, size: float):
        """Execute live trade (placeholder for exchange API)."""
        # This would integrate with Bybit or other exchange
        logger.info("🔴 LIVE TRADE: {} {} @ ${}".format(
            signal.upper(), symbol, self.prices[symbol]))
        
        # Place exchange API call here
        # exchange.create_order(symbol, signal, size)

    def close_position(self, symbol: str, reason: str = "signal"):
        """Close a position."""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        current_price = self.prices.get(symbol, 0)
        
        if current_price == 0:
            logger.warning("No price for {}".format(symbol))
            return
        
        entry_price = position['entry_price']
        size = position['size']
        
        # Calculate PnL
        if position['type'] == 'long':
            pnl = size * (current_price - entry_price) / entry_price
        else:  # short
            pnl = size * (entry_price - current_price) / entry_price
        
        # Subtract fees
        fee = size * 0.001 * 2  # Entry + exit
        pnl -= fee
        
        # Record in system
        self.trading.record_trade(symbol, pnl)
        
        # Update stats
        self.total_pnl += pnl
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        
        # Record history
        self.trade_history.append({
            'symbol': symbol,
            'type': position['type'],
            'entry_price': entry_price,
            'exit_price': current_price,
            'pnl': pnl,
            'reason': reason,
            'timestamp': datetime.now(timezone.utc)
        })
        
        # Clear position
        del self.positions[symbol]
        
        logger.info("📊 CLOSED: {} {} PnL: ${:.2f}".format(
            symbol, position['type'], pnl))
        
        return pnl

    def check_stops(self):
        """Check stop losses and take profits."""
        stop_loss_pct = 0.05  # 5%
        
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            current_price = self.prices.get(symbol, 0)
            
            if current_price == 0:
                continue
            
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            
            if position['type'] == 'short':
                pnl_pct = -pnl_pct
            
            # Stop loss
            if pnl_pct < -stop_loss_pct:
                self.close_position(symbol, "stop_loss")
            # Take profit (optional)
            elif pnl_pct > 0.10:  # 10% profit
                self.close_position(symbol, "take_profit")

    def get_status(self) -> Dict:
        """Get system status."""
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        return {
            'trading_mode': self.trading_mode,
            'portfolio': self.portfolio_value,
            'current_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'positions': {
                s: {
                    'type': p['type'],
                    'entry': p['entry_price'],
                    'current': self.prices.get(s, 0)
                }
                for s, p in self.positions.items()
            },
            'best_opportunity': {
                'symbol': self.trading.best_symbol,
                'signal': self.trading.best_signal,
                'confidence': self.trading.best_confidence
            }
        }

    def get_performance(self) -> Dict:
        """Get detailed performance."""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_win': 0
            }
        
        pnls = [t['pnl'] for t in self.trade_history]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        return {
            'total_trades': len(pnls),
            'winning_trades': len(wins),
            'win_rate': len(wins) / len(pnls),
            'total_pnl': sum(pnls),
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0
        }


# ============================================================================
# SIMPLE USAGE EXAMPLES
# ============================================================================

def example_basic():
    """Basic usage example."""
    print("=" * 50)
    print("BASIC USAGE EXAMPLE")
    print("=" * 50)
    
    # Create integration
    argus = ArgusTradingIntegration(
        symbols=['BTC/USDT', 'ETH/USDT'],
        portfolio_value=10000,
        trading_mode='paper'
    )
    
    # Generate some test data
    prices = {'BTC/USDT': 50000, 'ETH/USDT': 3000}
    
    for i in range(100):
        for symbol in argus.symbols:
            # Random price movement
            prices[symbol] *= 1 + np.random.randn() * 0.002
            prices[symbol] = max(prices[symbol], 100)
            
            # Update with data
            argus.on_data(symbol, prices[symbol], 1000)
    
    # Check status
    status = argus.get_status()
    print("Status: {}".format(status))
    
    perf = argus.get_performance()
    print("Performance: {}".format(perf))


def example_full():
    """Full usage with data feed simulation."""
    print("=" * 50)
    print("FULL USAGE EXAMPLE")
    print("=" * 50)
    
    argus = ArgusTradingIntegration(
        symbols=['BTC/USDT'],
        portfolio_value=10000,
        trading_mode='paper'
    )
    
    price = 50000
    
    # Simulate real-time data
    for second in range(60):  # 60 seconds
        # Update price
        price *= 1 + np.random.randn() * 0.001
        price = max(price, 100)
        
        # Send to system (every 0.5s = 2 updates per second)
        argus.on_data('BTC/USDT', price, 1000)
        
        # Check stops every second
        if second % 1 == 0:
            argus.check_stops()
        
        if second % 10 == 0:
            status = argus.get_status()
            print("{}s: {} {} (${:.0f}) | Trades: {} | PnL: ${:.2f}".format(
                second,
                status['best_opportunity']['symbol'] or '-',
                status['best_opportunity']['signal'],
                status['portfolio'],
                status['total_trades'],
                status['current_pnl']
            ))
    
    print()
    perf = argus.get_performance()
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print("Total trades: {}".format(perf['total_trades']))
    print("Win rate: {:.0%}".format(perf['win_rate']))
    print("Total PnL: ${:.2f}".format(perf['total_pnl']))


if __name__ == "__main__":
    # Run basic example
    example_basic()
    
    print()
    
    # Run full example
    example_full()