"""
Portfolio + Learning Integration - Complete Trading System

Features:
- Portfolio Manager + Learning System combined
- Position sizing with learning
- Risk management with learning
- Real-time adaptation
- Complete trading workflow

Usage:
    from scripts.portfolio_learning import TradingSystem
    
    ts = TradingSystem(
        capital=10000,
        symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    )
    
    # On each price update (0.5s)
    ts.on_price('BTC/USDT', 50000)
    
    # Get signal
    signal = ts.get_signal('BTC/USDT')
    
    # On trade
    ts.on_trade('BTC/USDT', 'buy', 100, 50000, 51000)
    
    # Get status
    status = ts.get_status()

Run: py scripts/portfolio_learning.py
"""

import logging
import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# COMPLETE TRADING SYSTEM
# ============================================================================

class TradingSystem:
    """
    Complete Trading System with:
    - Portfolio Management
    - Learning System
    - Risk Management
    - Real-time Adaptation
    """
    
    def __init__(
        self,
        capital: float = 10000,
        symbols: List[str] = None,
        max_risk: float = 0.02,
        position_strategy: str = 'kelly'
    ):
        self.capital = capital
        self.initial_capital = capital
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        self.max_risk = max_risk
        self.position_strategy = position_strategy
        
        # ========================================
        # STATE
        # ========================================
        self.prices = {s: 0 for s in self.symbols}
        self.positions = {s: {'size': 0, 'entry': 0, 'pnl': 0} for s in self.symbols}
        self.weights = {s: 1.0/len(self.symbols) for s in self.symbols}
        self.target_weights = {s: 1.0/len(self.symbols) for s in self.symbols}
        
        # ========================================
        # LEARNING STATE
        # ========================================
        self.price_history = {s: deque(maxlen=500) for s in self.symbols}
        self.features = {s: deque(maxlen=500) for s in self.symbols}
        self.signals = {s: 'hold' for s in self.symbols}
        self.confidences = {s: 0.5 for s in self.symbols}
        
        # Strategy performance
        self.strategy_performance = {
            'momentum': {'wins': 0, 'losses': 0},
            'mean_reversion': {'wins': 0, 'losses': 0},
            'breakout': {'wins': 0, 'losses': 0}
        }
        
        # ========================================
        # PORTFOLIO STATE
        # ========================================
        self.total_trades = 0
        self.winning_trades = 0
        self.trade_history = deque(maxlen=100)
        self.returns = deque(maxlen=500)
        
        self.peak_capital = capital
        self.daily_pnl = 0
        self.daily_start = capital
        
        # Risk limits
        self.max_position = 0.2
        self.max_daily_loss = 0.05
        self.max_drawdown = 0.15
        
        # Stats
        self.is_running = False
        
        logger.info("=" * 60)
        logger.info("TRADING SYSTEM INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"Capital: ${capital:,.2f}")
        logger.info(f"Risk: {max_risk:.1%}")
        logger.info(f"Position: {position_strategy}")
        logger.info(f"Symbols: {self.symbols}")
        logger.info("=" * 60)
    
    # ========================================
    # PRICE UPDATE (0.5s)
    # ========================================
    
    def on_price(self, symbol: str, price: float):
        """Update price and learn."""
        
        if symbol not in self.symbols:
            return
        
        self.prices[symbol] = price
        
        # Update history
        self.price_history[symbol].append(price)
        
        # Extract features
        if len(self.price_history[symbol]) >= 25:
            self._extract_features(symbol)
            self._learn(symbol)
            self._generate_signal(symbol)
    
    def _extract_features(self, symbol: str):
        """Extract features from price history."""
        
        prices = np.array(list(self.price_history[symbol]))
        
        r1 = prices[-1] / prices[-2] - 1 if len(prices) > 1 else 0
        r4 = prices[-1] / prices[-5] - 1 if len(prices) > 5 else 0
        r12 = prices[-1] / prices[-13] - 1 if len(prices) > 13 else 0
        r24 = prices[-1] / prices[-25] - 1 if len(prices) > 25 else 0
        
        v12 = np.std(prices[-13:]) if len(prices) > 13 else 0.01
        v24 = np.std(prices[-25:]) if len(prices) > 25 else 0.01
        
        # RSI
        deltas = np.diff(prices[-25:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        rs = avg_gain / (avg_loss + 0.0001)
        rsi = 100 - (100 / (1 + rs))
        
        # Position
        pp = (prices[-1] - prices.min()) / (prices.max() - prices.min() + 0.0001)
        
        # Volume ratio
        vr = v12 / (v24 + 0.0001)
        
        feat = np.array([r1, r4, r12, r24, v12, v24, rsi, pp, vr])
        self.features[symbol].append(feat)
    
    def _learn(self, symbol: str):
        """Learn from recent data."""
        
        if len(self.features[symbol]) < 10:
            return
        
        # Simple weight update
        feat = self.features[symbol][-1]
        
        # Update based on recent performance
        for strat in self.strategy_performance:
            stats = self.strategy_performance[strat]
            total = stats['wins'] + stats['losses']
            
            if total > 0:
                win_rate = stats['wins'] / total
                
                if win_rate > 0.6:
                    self.confidences[symbol] = min(0.85, self.confidences[symbol] + 0.01)
                elif win_rate < 0.4:
                    self.confidences[symbol] = max(0.3, self.confidences[symbol] - 0.01)
    
    def _generate_signal(self, symbol: str):
        """Generate trading signal."""
        
        if len(self.features[symbol]) < 2:
            return
        
        feat = self.features[symbol][-1]
        
        r1, r4, r12, r24, v12, v24, rsi, pp, vr = feat
        
        # Momentum strategy
        if r4 > 0.01 and r24 > 0:
            self.signals[symbol] = 'buy'
        elif r4 < -0.01 and r24 < 0:
            self.signals[symbol] = 'sell'
        elif rsi > 75 or rsi < 25:
            self.signals[symbol] = 'hold'
        else:
            self.signals[symbol] = 'hold'
        
        # Adjust confidence
        confidence = 0.5 + abs(r24) * 5 + (1 - abs(vr - 1)) * 0.2
        self.confidences[symbol] = min(0.85, max(0.3, confidence))
    
    # ========================================
    # SIGNAL PROCESSING
    # ========================================
    
    def get_signal(self, symbol: str) -> Dict:
        """Get current signal for symbol."""
        
        return {
            'signal': self.signals.get(symbol, 'hold'),
            'confidence': self.confidences.get(symbol, 0.5),
            'price': self.prices.get(symbol, 0)
        }
    
    def should_trade(self, symbol: str) -> bool:
        """Check if should trade."""
        
        # Check risk limits
        if not self._check_limits():
            return False
        
        # Check signal
        signal = self.signals.get(symbol, 'hold')
        confidence = self.confidences.get(symbol, 0.5)
        
        return signal in ['buy', 'sell'] and confidence > 0.55
    
    def _check_limits(self) -> bool:
        """Check risk limits."""
        
        # Daily loss
        if self.daily_start > 0:
            daily_pnl = (self.capital - self.daily_start) / self.daily_start
            if daily_pnl < -self.max_daily_loss:
                logger.warning("Daily loss limit")
                return False
        
        # Drawdown
        if self.peak_capital > 0:
            dd = (self.peak_capital - self.capital) / self.peak_capital
            if dd > self.max_drawdown:
                logger.warning("Drawdown limit")
                return False
        
        return True
    
    # ========================================
    # POSITION SIZING
    # ========================================
    
    def calculate_position_size(self, symbol: str, confidence: float) -> float:
        """Calculate position size."""
        
        # Kelly criterion
        win_rate = self._get_win_rate()
        
        if win_rate > 0.5:
            kelly = (win_rate - (1 - win_rate)) / 1
            kelly = max(0, kelly * 0.25)
        else:
            kelly = 0
        
        # Apply confidence
        size = kelly * confidence
        
        # Limit
        size = min(size, self.max_position)
        
        return size
    
    def _get_win_rate(self) -> float:
        """Get overall win rate."""
        if self.total_trades == 0:
            return 0.55
        
        return self.winning_trades / self.total_trades
    
    # ========================================
    # TRADE MANAGEMENT
    # ========================================
    
    def on_trade(
        self,
        symbol: str,
        action: str,
        size: float,
        entry_price: float,
        exit_price: float
    ):
        """On trade execution."""
        
        pnl = 0
        
        if action == 'buy':
            pnl = (exit_price / entry_price - 1) * size * entry_price
        elif action == 'sell':
            pnl = (entry_price / exit_price - 1) * size * entry_price
        
        self.capital += pnl
        self.total_trades += 1
        
        if pnl > 0:
            self.winning_trades += 1
        
        # Record trade
        self.trade_history.append({
            'symbol': symbol,
            'action': action,
            'size': size,
            'pnl': pnl,
            'entry': entry_price,
            'exit': exit_price,
            'time': datetime.now(timezone.utc)
        })
        
        # Track returns
        ret = pnl / self.initial_capital
        self.returns.append(ret)
        
        # Track strategy performance
        strategy = self._get_strategy()
        if strategy in self.strategy_performance:
            if pnl > 0:
                self.strategy_performance[strategy]['wins'] += 1
            else:
                self.strategy_performance[strategy]['losses'] += 1
        
        # Update peak
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        
        # Log
        logger.info(f"Trade: {symbol} {action} {size:.1%} | PnL: ${pnl:,.2f} | Capital: ${self.capital:,.2f}")
    
    def _get_strategy(self) -> str:
        """Get current strategy."""
        
        r1 = 0
        for feat in list(self.features.get('BTC/USDT', []))[-5:]:
            r1 += feat[0]
        r1 /= 5
        
        if r1 > 0.005:
            return 'momentum'
        elif r1 < -0.005:
            return 'mean_reversion'
        
        return 'breakout'
    
    # ========================================
    # POSITION MANAGEMENT
    # ========================================
    
    def on_bar_close(self, symbol: str, price: float):
        """Update positions on bar close."""
        
        pos = self.positions[symbol]
        
        if pos['size'] > 0 and pos['entry'] > 0:
            pnl = (price / pos['entry'] - 1) * pos['size'] * pos['entry']
            pos['pnl'] = pnl
    
    def get_positions(self) -> Dict:
        """Get current positions."""
        
        return {
            s: {
                'size': p['size'],
                'entry': p['entry'],
                'pnl': p['pnl']
            }
            for s, p in self.positions.items()
        }
    
    # ========================================
    # RISK METRICS
    # ========================================
    
    def get_risk_metrics(self) -> Dict:
        """Get risk metrics."""
        
        returns_arr = np.array(list(self.returns)) if self.returns else np.array([0])
        
        if len(returns_arr) > 1:
            volatility = np.std(returns_arr) * np.sqrt(252)
            sharpe = np.mean(returns_arr) / (np.std(returns_arr) + 0.0001) * np.sqrt(252)
        else:
            volatility = 0
            sharpe = 0
        
        max_dd = self._max_drawdown()
        
        return {
            'capital': self.capital,
            'return': (self.capital / self.initial_capital - 1),
            'trades': self.total_trades,
            'win_rate': self._get_win_rate(),
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'exposure': sum(p['size'] for p in self.positions.values())
        }
    
    def _max_drawdown(self) -> float:
        """Calculate max drawdown."""
        
        if not self.returns:
            return 0
        
        cumulative = np.cumsum([0] + list(self.returns))
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / (peak + 1)
        
        return np.max(drawdown) if len(drawdown) > 0 else 0
    
    # ========================================
    # STATUS
    # ========================================
    
    def get_status(self) -> Dict:
        """Get complete status."""
        
        risk = self.get_risk_metrics()
        
        return {
            'capital': self.capital,
            'return': risk['return'],
            'trades': risk['trades'],
            'win_rate': risk['win_rate'],
            'sharpe': risk['sharpe'],
            'max_dd': risk['max_drawdown'],
            'signals': self.signals.copy(),
            'confidences': self.confidences.copy(),
            'prices': self.prices.copy(),
            'strategy_perf': self.strategy_performance.copy()
        }
    
    def print_status(self):
        """Print status."""
        
        status = self.get_status()
        
        print("=" * 60)
        print("TRADING SYSTEM STATUS")
        print("=" * 60)
        print(f"Capital:    ${status['capital']:,.2f}")
        print(f"Return:    {status['return']:.2%}")
        print(f"Trades:     {status['trades']}")
        print(f"Win Rate:   {status['win_rate']:.1%}")
        print(f"Sharpe:     {status['sharpe']:.2f}")
        print(f"Max DD:     {status['max_dd']:.2%}")
        print("-" * 60)
        
        print("Signals:")
        for symbol in self.symbols:
            sig = status['signals'][symbol]
            conf = status['confidences'][symbol]
            price = status['prices'][symbol]
            print(f"  {symbol}: {sig} ({conf:.0%}) @ ${price:,.0f}")
        
        print("=" * 60)


# ============================================================================
# DEMO
# ============================================================================

async def main():
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("PORTFOLIO + LEARNING TRADING SYSTEM")
    print("=" * 60)
    print()
    
    # Create system
    ts = TradingSystem(
        capital=10000,
        symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
        max_risk=0.02,
        position_strategy='kelly'
    )
    
    # Simulate price updates
    prices = {'BTC/USDT': 50000, 'ETH/USDT': 3000, 'SOL/USDT': 100}
    
    for sec in range(60):
        for symbol in prices:
            # Simulate price movement
            prices[symbol] *= 1 + np.random.randn() * 0.002
            prices[symbol] = max(prices[symbol], 10)
            
            # Update system
            ts.on_price(symbol, prices[symbol])
        
        if sec % 20 == 0:
            sig = ts.get_signal('BTC/USDT')
            print(f"{sec}s: {sig['signal']} ({sig['confidence']:.0%}) @ ${sig['price']:,.0f}")
    
    print()
    
    # Simulate trades
    print("Simulated Trades:")
    ts.on_trade('BTC/USDT', 'buy', 0.1, 50000, 51000)
    ts.on_trade('ETH/USDT', 'buy', 0.1, 3000, 3100)
    ts.on_trade('SOL/USDT', 'sell', 0.1, 100, 95)
    
    print()
    
    # Print status
    ts.print_status()
    
    print()


if __name__ == "__main__":
    asyncio.run(main())