"""
Adaptive Risk & Execution System

Learns to adapt:
1. Risk Parameters (position size, stop loss, take profit, max risk)
2. Execution Parameters (order size, order type, slippage, speed)

Features:
- Learns optimal risk from performance
- Adapts execution to market conditions
- Dynamic parameter adjustment
- Real-time learning (0.5s updates)

Usage:
    from scripts.adaptive_risk import AdaptiveRisk
    
    ar = AdaptiveRisk(symbols=['BTC/USDT', 'ETH/USDT'])
    
    # Update with price (every 0.5s)
    ar.on_price('BTC/USDT', 50000)
    
    # Get adapted risk
    risk = ar.get_risk('BTC/USDT')
    
    # Get adapted execution
    exec = ar.get_execution('BTC/USDT')
    
    # Record trade result
    ar.on_trade('BTC/USDT', 100, 50000, 51000)

Run: py scripts/adaptive_risk.py
"""

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# ADAPTIVE RISK SYSTEM
# ============================================================================

class AdaptiveRisk:
    """
    Adaptive Risk & Execution - Learns optimal parameters.
    
    Adapts:
    - Position size based on confidence & performance
    - Stop loss based on volatility
    - Take profit based on patterns
    - Max risk based on drawdown
    - Execution based on spread/liquidity
    """
    
    def __init__(
        self,
        capital: float = 10000,
        symbols: List[str] = None,
        initial_max_risk: float = 0.02,
        initial_stop: float = 0.02,
        initial_target: float = 0.03
    ):
        self.capital = capital
        self.initial_capital = capital
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        
        # ========================================
        # INITIAL PARAMETERS
        # ========================================
        self.initial_max_risk = initial_max_risk
        self.initial_stop = initial_stop
        self.initial_target = initial_target
        
        # ========================================
        # STATE
        # ========================================
        self.prices = {s: 0 for s in self.symbols}
        self.price_history = {s: deque(maxlen=500) for s in self.symbols}
        self.volatility = {s: 0.02 for s in self.symbols}
        self.spread = {s: 0.0005 for s in self.symbols}
        
        # ========================================
        # LEARNED PARAMETERS (per symbol)
        # ========================================
        self.max_risk = {s: initial_max_risk for s in self.symbols}
        self.stop_loss = {s: initial_stop for s in self.symbols}
        self.take_profit = {s: initial_target for s in self.symbols}
        self.confidence = {s: 0.5 for s in self.symbols}
        
        # Execution parameters
        self.order_type = {s: 'limit' for s in self.symbols}
        self.order_size_pct = {s: 0.1 for s in self.symbols}
        self.slippageExpectation = {s: 0.0005 for s in self.symbols}
        self.execution_speed = {s: 'normal' for s in self.symbols}
        
        # ========================================
        # PERFORMANCE TRACKING
        # ========================================
        self.trade_history = deque(maxlen=100)
        self.wins = {s: 0 for s in self.symbols}
        self.losses = {s: 0 for s in self.symbols}
        self.total_pnl = 0
        self.peak_capital = capital
        self.drawdown = 0
        
        # Strategy tracking
        self.win_rate = 0.55
        self.avg_win = 0
        self.avg_loss = 0
        self.recent_returns = deque(maxlen=50)
        
        logger.info("=" * 60)
        logger.info("ADAPTIVE RISK & EXECUTION SYSTEM")
        logger.info("=" * 60)
        logger.info(f"Capital: ${capital:,.2f}")
        logger.info(f"Initial Risk: {initial_max_risk:.1%}")
        logger.info(f"Symbols: {self.symbols}")
        logger.info("=" * 60)
    
    # ========================================
    # PRICE UPDATE (0.5s)
    # ========================================
    
    def on_price(self, symbol: str, price: float):
        """Update price and adapt parameters."""
        
        if symbol not in self.symbols:
            return
        
        self.prices[symbol] = price
        self.price_history[symbol].append(price)
        
        # Update volatility
        if len(self.price_history[symbol]) >= 25:
            self._update_volatility(symbol)
            self._update_spread(symbol)
            self._adapt_parameters(symbol)
    
    def _update_volatility(self, symbol: str):
        """Update volatility estimate."""
        
        prices = list(self.price_history[symbol])
        
        if len(prices) >= 25:
            returns = np.diff(np.log(prices[-25:]))
            self.volatility[symbol] = np.std(returns)
    
    def _update_spread(self, symbol: str):
        """Update spread estimate."""
        
        # Simulate spread (in real system, get from order book)
        self.spread[symbol] = 0.0005 + np.random.uniform(0, 0.001)
    
    def _adapt_parameters(self, symbol: str):
        """Adapt all parameters based on learning."""
        
        self._adapt_risk(symbol)
        self._adapt_stop_loss(symbol)
        self._adapt_take_profit(symbol)
        self._adapt_execution(symbol)
    
    # ========================================
    # ADAPT RISK PARAMETERS
    # ========================================
    
    def _adapt_risk(self, symbol: str):
        """Adapt max risk based on performance."""
        
        # Start with initial
        max_r = self.initial_max_risk
        
        # Adjust based on win rate
        if self.win_rate > 0.6:
            max_r *= 1.2  # Increase when winning
        elif self.win_rate < 0.4:
            max_r *= 0.7  # Decrease when losing
        
        # Adjust based on drawdown
        if self.drawdown > 0.1:
            max_r *= 0.5
        elif self.drawdown > 0.05:
            max_r *= 0.75
        
        # Adjust based on confidence
        conf = self.confidence[symbol]
        max_r *= (0.5 + conf)
        
        # Limit
        max_r = min(max_r, 0.25)
        max_r = max(max_r, 0.005)
        
        self.max_risk[symbol] = max_r
    
    def _adapt_stop_loss(self, symbol: str):
        """Adapt stop loss based on volatility."""
        
        vol = self.volatility[symbol]
        
        # Stop at 1.5x volatility
        stop = vol * 1.5
        
        # But min 1%, max 5%
        stop = min(max(stop, 0.01), 0.05)
        
        self.stop_loss[symbol] = stop
    
    def _adapt_take_profit(self, symbol: str):
        """Adapt take profit based on patterns."""
        
        vol = self.volatility[symbol]
        
        # Target 2x volatility minimum
        target = vol * 2
        
        # Adjust based on recent returns
        if self.recent_returns:
            recent = np.mean(self.recent_returns)
            if recent > 0:
                target *= 1.2
        
        # But min 1.5%, max 10%
        target = min(max(target, 0.015), 0.10)
        
        self.take_profit[symbol] = target
    
    # ========================================
    # ADAPT EXECUTION PARAMETERS
    # ========================================
    
    def _adapt_execution(self, symbol: str):
        """Adapt execution parameters."""
        
        vol = self.volatility[symbol]
        spread = self.spread[symbol]
        
        # Order type: limit for low vol, market for high vol
        if vol > 0.03 or spread > 0.002:
            self.order_type[symbol] = 'market'
        else:
            self.order_type[symbol] = 'limit'
        
        # Order size: smaller for high vol
        size_pct = 0.1
        if vol > 0.03:
            size_pct *= 0.7
        elif vol < 0.01:
            size_pct *= 1.2
        
        self.order_size_pct[symbol] = min(size_pct, 0.2)
        
        # Slippage expectation
        self.slippageExpectation[symbol] = spread * 2 + vol / 2
        
        # Execution speed
        if vol > 0.03:
            self.execution_speed[symbol] = 'fast'
        elif spread > 0.001:
            self.execution_speed[symbol] = 'fast'
        else:
            self.execution_speed[symbol] = 'normal'
    
    # ========================================
    # GET PARAMETERS
    # ========================================
    
    def get_risk(self, symbol: str) -> Dict:
        """Get adapted risk parameters."""
        
        return {
            'max_risk': self.max_risk.get(symbol, self.initial_max_risk),
            'stop_loss': self.stop_loss.get(symbol, self.initial_stop),
            'take_profit': self.take_profit.get(symbol, self.initial_target),
            'confidence': self.confidence.get(symbol, 0.5),
            'win_rate': self.win_rate
        }
    
    def get_execution(self, symbol: str) -> Dict:
        """Get adapted execution parameters."""
        
        return {
            'order_type': self.order_type.get(symbol, 'limit'),
            'order_size_pct': self.order_size_pct.get(symbol, 0.1),
            'slippage': self.slippageExpectation.get(symbol, 0.0005),
            'speed': self.execution_speed.get(symbol, 'normal'),
            'spread': self.spread.get(symbol, 0.0005),
            'volatility': self.volatility.get(symbol, 0.02)
        }
    
    def get_all_parameters(self, symbol: str) -> Dict:
        """Get all adapted parameters."""
        
        return {
            'risk': self.get_risk(symbol),
            'execution': self.get_execution(symbol)
        }
    
    # ========================================
    # TRADE RECORDING
    # ========================================
    
    def on_trade(
        self,
        symbol: str,
        pnl: float,
        entry_price: float,
        exit_price: float
    ):
        """Record trade and update learning."""
        
        # Record trade
        self.trade_history.append({
            'symbol': symbol,
            'pnl': pnl,
            'entry': entry_price,
            'exit': exit_price,
            'time': datetime.now(timezone.utc)
        })
        
        # Update wins/losses
        if pnl > 0:
            self.wins[symbol] += 1
            self.total_pnl += pnl
        else:
            self.losses[symbol] += 1
            self.total_pnl += pnl
        
        # Update returns
        ret = pnl / self.initial_capital
        self.recent_returns.append(ret)
        
        # Update performance metrics
        self._update_performance()
        
        # Update peak/drawdown
        if self.capital + pnl > self.peak_capital:
            self.peak_capital = self.capital + pnl
        
        dd = (self.peak_capital - (self.capital + pnl)) / self.peak_capital
        self.drawdown = max(self.drawdown, dd)
        
        # Log
        logger.info(f"Trade: {symbol} PnL: ${pnl:,.2f} | Risk: {self.max_risk[symbol]:.1%}")
    
    def _update_performance(self):
        """Update performance metrics."""
        
        total_wins = sum(self.wins.values())
        total_losses = sum(self.losses.values())
        total = total_wins + total_losses
        
        if total > 0:
            self.win_rate = total_wins / total
        
        # Calculate avg win/loss from recent trades
        if len(self.trade_history) > 0:
            recent = list(self.trade_history)[-20:]
            wins = [t['pnl'] for t in recent if t['pnl'] > 0]
            losses = [t['pnl'] for t in recent if t['pnl'] < 0]
            
            self.avg_win = np.mean(wins) if wins else 0
            self.avg_loss = np.mean(losses) if losses else 0
    
    # ========================================
    # TRADE RECOMMENDATION
    # ========================================
    
    def should_trade(self, symbol: str) -> bool:
        """Check if should trade based on adapted parameters."""
        
        # Check risk
        risk = self.get_risk(symbol)
        
        if risk['max_risk'] < 0.01:
            return False
        
        if self.drawdown > 0.15:
            return False
        
        # Check confidence
        if risk['confidence'] < 0.5:
            return False
        
        return True
    
    def get_trade_parameters(self, symbol: str) -> Dict:
        """Get full trade parameters."""
        
        risk = self.get_risk(symbol)
        exec_params = self.get_execution(symbol)
        
        # Calculate position size
        position_size = self._calculate_position_size(symbol, risk, exec_params)
        
        return {
            'symbol': symbol,
            'price': self.prices.get(symbol, 0),
            'position_size': position_size,
            'stop_loss_pct': risk['stop_loss'],
            'take_profit_pct': risk['take_profit'],
            'order_type': exec_params['order_type'],
            'order_size_pct': exec_params['order_size_pct'],
            'slippage': exec_params['slippage'],
            'speed': exec_params['speed'],
            'confidence': risk['confidence'],
            'win_rate': risk['win_rate']
        }
    
    def _calculate_position_size(
        self,
        symbol: str,
        risk: Dict,
        execution: Dict
    ) -> float:
        """Calculate position size."""
        
        # Kelly criterion
        wr = risk['win_rate']
        
        if wr > 0.5:
            kelly = (wr - (1 - wr)) / 1
            kelly = max(0, kelly * 0.25)
        else:
            kelly = 0
        
        # Apply risk limit
        size = min(kelly, risk['max_risk'])
        
        # Adjust for execution
        size *= execution['order_size_pct'] / 0.1
        
        return max(min(size, 0.25), 0.01)
    
    # ========================================
    # STATUS
    # ========================================
    
    def get_status(self) -> Dict:
        """Get system status."""
        
        return {
            'capital': self.capital,
            'total_pnl': self.total_pnl,
            'win_rate': self.win_rate,
            'drawdown': self.drawdown,
            'symbols': {
                s: {
                    'risk': self.get_risk(s),
                    'execution': self.get_execution(s)
                }
                for s in self.symbols
            }
        }
    
    def print_status(self):
        """Print status."""
        
        status = self.get_status()
        
        print("=" * 60)
        print("ADAPTIVE RISK & EXECUTION STATUS")
        print("=" * 60)
        print(f"Capital:  ${self.capital:,.2f}")
        print(f"Total PnL: ${self.total_pnl:,.2f}")
        print(f"Win Rate: {self.win_rate:.1%}")
        print(f"Drawdown: {self.drawdown:.1%}")
        print("-" * 60)
        
        print("Adapted Parameters:")
        for symbol in self.symbols:
            r = status['symbols'][symbol]
            
            print(f"\n{symbol}:")
            print(f"  Risk: max={r['risk']['max_risk']:.1%}, "
                  f"stop={r['risk']['stop_loss']:.1%}, "
                  f"target={r['risk']['take_profit']:.1%}")
            print(f"  Exec: type={r['execution']['order_type']}, "
                  f"size={r['execution']['order_size_pct']:.0%}, "
                  f"speed={r['execution']['speed']}")
        
        print("=" * 60)


# ============================================================================
# DEMO
# ============================================================================

def main():
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("ADAPTIVE RISK & EXECUTION DEMO")
    print("=" * 60)
    print()
    
    # Create system
    ar = AdaptiveRisk(
        capital=10000,
        symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
        initial_max_risk=0.02,
        initial_stop=0.02,
        initial_target=0.03
    )
    
    # Simulate price updates
    prices = {'BTC/USDT': 50000, 'ETH/USDT': 3000, 'SOL/USDT': 100}
    
    for sec in range(30):
        for symbol in prices:
            prices[symbol] *= 1 + np.random.randn() * 0.002
            prices[symbol] = max(prices[symbol], 10)
            
            ar.on_price(symbol, prices[symbol])
        
        if sec % 10 == 0:
            params = ar.get_trade_parameters('BTC/USDT')
            print(f"{sec}s: risk={params['position_size']:.0%}, "
                  f"stop={params['stop_loss_pct']:.1%}, "
                  f"target={params['take_profit_pct']:.1%}, "
                  f"type={params['order_type']}")
    
    print()
    
    # Simulate trades
    print("Simulated Trades:")
    ar.on_trade('BTC/USDT', 100, 50000, 51000)
    ar.on_trade('ETH/USDT', 50, 3000, 3100)
    ar.on_trade('SOL/USDT', -25, 100, 95)
    ar.on_trade('BTC/USDT', 150, 51000, 52500)
    
    print()
    
    # Print status
    ar.print_status()
    
    print()


if __name__ == "__main__":
    main()