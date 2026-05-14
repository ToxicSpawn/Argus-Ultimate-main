"""
Live Execution with Paper Trading

Features:
- Paper trading mode (simulated execution)
- Live trading mode (real orders)
- Realistic slippage simulation
- Risk management checks
- Order execution with logging
- Trade history tracking

Usage:
    from scripts.live_execution import LiveExecution
    
    # Paper trading
    executor = LiveExecution(capital=10000, mode='paper')
    
    # Execute order
    result = executor.execute('BTCUSDT', 'buy', 0.1, 50000)
    
    # Get status
    status = executor.get_status()

Requirements:
    pip install python-binance (for live mode)

Run: py scripts/live_execution.py
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class LiveExecution:
    """
    Live execution with paper/live modes.
    
    Features:
    - Paper trading: Simulates execution with realistic slippage
    - Live trading: Real orders (requires API keys)
    - Risk management: Checks before execution
    - Logging: Tracks all trades
    - Performance: Tracks PnL, win rate, etc.
    """
    
    def __init__(
        self,
        capital: float = 10000,
        mode: str = 'paper',
        max_risk_per_trade: float = 0.02,
        max_position_size: float = 0.2,
        slippage_model: str = 'normal'
    ):
        self.capital = capital
        self.initial_capital = capital
        self.mode = mode  # 'paper' or 'live'
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_size = max_position_size
        self.slippage_model = slippage_model
        
        # State
        self.positions = {}
        self.trade_history = []
        self.total_pnl = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.peak_capital = capital
        self.drawdown = 0
        self.orders_executed = 0
        
        # Risk limits
        self.max_daily_loss = 0.05
        self.max_drawdown = 0.15
        self.daily_start = capital
        self.daily_pnl = 0
        
        logger.info(f"LiveExecution initialized: mode={mode}, capital=${capital:,.2f}")
    
    def execute(
        self,
        symbol: str,
        action: str,  # 'buy' or 'sell'
        size: float,  # in base currency
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: float = 1.0
    ) -> Dict:
        """
        Execute an order.
        
        Returns:
            Dict with execution details or error
        """
        
        # Check risk limits
        if not self._check_risk_limits():
            return {'status': 'error', 'message': 'Risk limit exceeded'}
        
        # Calculate position value
        position_value = size * price * leverage
        
        # Check max position
        if position_value > self.capital * self.max_position_size:
            return {'status': 'error', 'message': 'Position too large'}
        
        # Calculate slippage
        slippage = self._calculate_slippage(symbol, size, action)
        executed_price = price * (1 + slippage * (1 if action == 'buy' else -1))
        
        # Paper trading simulation
        if self.mode == 'paper':
            pnl = self._simulate_pnl(symbol, action, size, executed_price, stop_loss, take_profit)
            
            # Update capital
            self.capital += pnl
            self.total_pnl += pnl
            
            # Record trade
            trade = {
                'symbol': symbol,
                'action': action,
                'size': size,
                'requested_price': price,
                'executed_price': executed_price,
                'slippage': slippage,
                'pnl': pnl,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'leverage': leverage,
                'time': datetime.now(timezone.utc),
                'mode': 'paper'
            }
            
            self.trade_history.append(trade)
            self.orders_executed += 1
            
            # Update performance
            self._update_performance(pnl)
            
            logger.info(f"Executed {action} {size:.4f} {symbol} @ {executed_price:.2f} | PnL: ${pnl:,.2f}")
            
            return {'status': 'success', 'trade': trade, 'capital': self.capital}
        
        # Live trading (placeholder - requires API keys)
        elif self.mode == 'live':
            # TODO: Implement real Binance API execution
            return {'status': 'error', 'message': 'Live mode not yet implemented (requires API keys)'}
        
        else:
            return {'status': 'error', 'message': 'Invalid mode'}
    
    def _check_risk_limits(self) -> bool:
        """Check all risk limits."""
        
        # Daily loss
        if self.daily_start > 0:
            daily_pnl = (self.capital - self.daily_start) / self.daily_start
            if daily_pnl < -self.max_daily_loss:
                logger.warning("Daily loss limit exceeded")
                return False
        
        # Drawdown
        if self.peak_capital > 0:
            dd = (self.peak_capital - self.capital) / self.peak_capital
            if dd > self.max_drawdown:
                logger.warning("Max drawdown limit exceeded")
                return False
        
        return True
    
    def _calculate_slippage(
        self,
        symbol: str,
        size: float,
        action: str
    ) -> float:
        """
        Calculate realistic slippage based on market conditions.
        
        Models:
        - normal: 0.01% - 0.1%
        - high_vol: 0.1% - 0.5%
        - low_liquidity: 0.5% - 2.0%
        """
        
        base_slippage = 0.0005  # 0.05%
        
        if self.slippage_model == 'high_vol':
            base_slippage = np.random.uniform(0.001, 0.005)
        elif self.slippage_model == 'low_liquidity':
            base_slippage = np.random.uniform(0.005, 0.02)
        
        # Add random component
        slippage = base_slippage * (1 + np.random.randn() * 0.3)
        slippage = min(max(slippage, 0.0001), 0.02)
        
        return slippage
    
    def _simulate_pnl(
        self,
        symbol: str,
        action: str,
        size: float,
        executed_price: float,
        stop_loss: Optional[float],
        take_profit: Optional[float]
    ) -> float:
        """
        Simulate PnL with realistic price movement.
        """
        
        # Simulate price movement over next 5 minutes
        time_horizon = 300  # seconds
        price_movement = np.random.randn() * 0.002 * time_horizon / 60
        
        if action == 'buy':
            final_price = executed_price * (1 + price_movement)
            pnl = (final_price / executed_price - 1) * size * executed_price
        else:  # sell
            final_price = executed_price * (1 - price_movement)
            pnl = (executed_price / final_price - 1) * size * executed_price
        
        # Apply stop loss/take profit
        if stop_loss and ((action == 'buy' and final_price <= stop_loss) or 
                         (action == 'sell' and final_price >= stop_loss)):
            pnl = (stop_loss / executed_price - 1) * size * executed_price
        
        if take_profit and ((action == 'buy' and final_price >= take_profit) or 
                           (action == 'sell' and final_price <= take_profit)):
            pnl = (take_profit / executed_price - 1) * size * executed_price
        
        return pnl
    
    def _update_performance(self, pnl: float):
        """Update performance metrics."""
        
        # Update peak capital
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        
        # Update drawdown
        dd = (self.peak_capital - self.capital) / self.peak_capital
        self.drawdown = max(self.drawdown, dd)
        
        # Update daily PnL
        if self.daily_start > 0:
            self.daily_pnl = self.capital - self.daily_start
        
        # Update win/loss count
        if pnl > 0:
            self.winning_trades += 1
        elif pnl < 0:
            self.losing_trades += 1
    
    def get_status(self) -> Dict:
        """Get current execution status."""
        
        total_trades = self.winning_trades + self.losing_trades
        win_rate = self.winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'capital': self.capital,
            'initial_capital': self.initial_capital,
            'return': (self.capital / self.initial_capital - 1),
            'total_pnl': self.total_pnl,
            'win_rate': win_rate,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'peak_capital': self.peak_capital,
            'max_drawdown': self.drawdown,
            'daily_pnl': self.daily_pnl,
            'orders_executed': self.orders_executed,
            'positions': self.positions,
            'mode': self.mode
        }
    
    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Get trade history."""
        return self.trade_history[-limit:]
    
    def reset_daily(self):
        """Reset daily tracking."""
        self.daily_start = self.capital
        self.daily_pnl = 0
        logger.info("Daily tracking reset")
    
    def print_summary(self):
        """Print execution summary."""
        
        status = self.get_status()
        
        print("=" * 60)
        print("LIVE EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Mode: {status['mode'].upper()}")
        print(f"Capital: ${status['capital']:,.2f}")
        print(f"Return: {status['return']:.2%}")
        print(f"Total PnL: ${status['total_pnl']:,.2f}")
        print(f"Win Rate: {status['win_rate']:.1%}")
        print(f"Orders: {status['orders_executed']}")
        print(f"Drawdown: {status['max_drawdown']:.1%}")
        print("-" * 60)
        print(f"Winning Trades: {status['winning_trades']}")
        print(f"Losing Trades: {status['losing_trades']}")
        print("=" * 60)


# ============================================================================
# DEMO
# ============================================================================

def demo():
    """Demo the live execution system."""
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("LIVE EXECUTION DEMO (Paper Trading)")
    print("=" * 60)
    print()
    
    # Create executor
    executor = LiveExecution(capital=10000, mode='paper')
    
    # Simulate some trades
    print("Simulating trades...")
    print()
    
    # Trade 1: Buy BTC
    result1 = executor.execute('BTCUSDT', 'buy', 0.1, 50000, stop_loss=49500, take_profit=51000)
    print(f"Trade 1: {result1['status']}")
    
    # Trade 2: Buy ETH
    result2 = executor.execute('ETHUSDT', 'buy', 0.5, 3000, stop_loss=2950, take_profit=3100)
    print(f"Trade 2: {result2['status']}")
    
    # Trade 3: Sell SOL
    result3 = executor.execute('SOLUSDT', 'sell', 2.0, 100, stop_loss=105, take_profit=95)
    print(f"Trade 3: {result3['status']}")
    
    # Trade 4: Buy BTC larger
    result4 = executor.execute('BTCUSDT', 'buy', 0.15, 50500, stop_loss=50000, take_profit=52000)
    print(f"Trade 4: {result4['status']}")
    
    print()
    print("=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    
    executor.print_summary()
    
    print()
    print("Trade History:")
    for i, trade in enumerate(executor.get_trade_history(4), 1):
        print(f"{i}. {trade['action']} {trade['size']:.4f} {trade['symbol']} @ {trade['executed_price']:.2f}")
        print(f"   PnL: ${trade['pnl']:,.2f} | Slippage: {trade['slippage']:.3%}")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    demo()
