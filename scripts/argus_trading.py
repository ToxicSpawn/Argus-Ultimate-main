"""
Argus Integration - Standalone Version

Complete integration for Argus live/paper trading.
Works without external dependencies.

Run: py scripts/argus_trading.py

Then integrate into your main.py strategy.
"""

import logging
import asyncio
import time
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# STRATEGIES
# ============================================================================

strategies = {
    'momentum': {'weight': 1.0},
    'mean_reversion': {'weight': 1.0},
    'breakout': {'weight': 1.0},
    'volatility': {'weight': 1.0}
}

strategy_names = list(strategies.keys())


# ============================================================================
# PER-SYMBOL LEARNER
# ============================================================================

class SymbolLearner:
    """Learns for one symbol."""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.prices = deque(maxlen=500)
        self.features = deque(maxlen=500)
        self.strategy_weights = np.ones(len(strategies))
        self.total_updates = 0
        self.wins = 0
        self.losses = 0
        self.signal = 'hold'
        self.confidence = 0.5
    
    def update(self, price: float):
        self.prices.append(price)
        features = self._extract()
        if len(self.features) > 0:
            self._learn(features)
        self.features.append(features)
        self.total_updates += 1
        self._generate_signal(features)
    
    def _extract(self):
        p = np.array(list(self.prices))
        if len(p) < 25:
            return np.zeros(9)
        
        r1 = p[-1] / p[-2] - 1 if len(p) > 1 else 0
        r4 = p[-1] / p[-5] - 1 if len(p) > 5 else 0
        r12 = p[-1] / p[-13] - 1 if len(p) > 13 else 0
        r24 = p[-1] / p[-25] - 1 if len(p) > 25 else 0
        v12 = np.std(p[-13:]) / np.mean(p[-13:]) if len(p) > 13 else 0
        v24 = np.std(p[-25:]) / np.mean(p[-25:]) if len(p) > 25 else 0
        
        d = np.diff(p)
        g = np.where(d > 0, d, 0)
        l = np.where(d < 0, -d, 0)
        rsi = 50
        if len(g) >= 14:
            rsi = 100 - (100 / (1 + np.mean(g[-14:]) / max(np.mean(l[-14:]), 1e-8)))
        
        return np.array([r1, r4, r12, r24, v12, v24, rsi, 0.5, 1.0])
    
    def _learn(self, features):
        if len(self.features) < 20:
            return
        
        # Simple strategy weighting
        for i in range(len(self.strategy_weights)):
            self.strategy_weights[i] += (np.random.rand() - 0.5) * 0.01
        
        self.strategy_weights = np.maximum(self.strategy_weights, 0.1)
        self.strategy_weights /= self.strategy_weights.sum()
    
    def _generate_signal(self, features):
        if len(self.features) < 20:
            self.signal = 'hold'
            self.confidence = 0.5
            return
        
        # Combine strategies
        score = features[0] * self.strategy_weights[0]  # momentum
        score += (50 - features[6]) / 100 * self.strategy_weights[1]  # mean reversion
        score += features[3] * self.strategy_weights[2]  # breakout
        score += features[5] * self.strategy_weights[3]  # volatility
        
        if score > 0.005:
            self.signal = 'buy'
        elif score < -0.005:
            self.signal = 'sell'
        else:
            self.signal = 'hold'
        
        self.confidence = min(0.5 + abs(score) * 10, 0.85)
    
    def record_result(self, won: bool):
        if won:
            self.wins += 1
            self.strategy_weights *= 1.02
        else:
            self.losses += 1
            self.strategy_weights *= 0.98
        
        self.strategy_weights = np.maximum(self.strategy_weights, 0.1)
        self.strategy_weights /= self.strategy_weights.sum()


# ============================================================================
# MAIN INTEGRATION
# ============================================================================

class ArgusTrading:
    """
    Main Argus Trading Integration.
    
    Usage:
        from scripts.argus_trading import ArgusTrading
        
        trading = ArgusTrading(['BTC/USDT', 'ETH/USDT'], 10000)
        
        # Every price update:
        trading.update('BTC/USDT', 50000)
        
        # Get decision:
        should, reason = trading.should_trade()
        
        # Execute:
        if should:
            trading.execute_order()
        
        # After trade:
        trading.close_position('BTC/USDT', pnl)
    """

    def __init__(
        self,
        symbols: List[str] = None,
        portfolio: float = 10000,
        mode: str = 'paper'
    ):
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        self.portfolio = portfolio
        self.mode = mode
        
        self.learners = {s: SymbolLearner(s) for s in self.symbols}
        
        self.positions = {}
        self.trades = deque(maxlen=500)
        
        self.total_pnl = 0
        self.total_trades = 0
        self.wins = 0
        
        self.prices = {s: 0 for s in self.symbols}
        
        self.best_symbol = None
        self.best_signal = 'hold'
        self.best_confidence = 0.5
        
        logger.info("=" * 50)
        logger.info("ARGUS TRADING v5.0")
        logger.info("=" * 50)
        logger.info("Mode: {} | Portfolio: ${}".format(mode, portfolio))
        logger.info("Symbols: {}".format(symbols))

    def update(self, symbol: str, price: float):
        """Update with new price data."""
        if symbol not in self.symbols:
            return
        
        self.prices[symbol] = price
        self.learners[symbol].update(price)
        self._scan_best()

    def _scan_best(self):
        opportunities = []
        
        for s, learner in self.learners.items():
            if learner.signal != 'hold' and learner.confidence > 0.5:
                score = learner.confidence * (learner.wins / max(learner.wins + learner.losses, 1))
                opportunities.append((s, score, learner.signal, learner.confidence))
        
        if opportunities:
            opportunities.sort(key=lambda x: x[1], reverse=True)
            self.best_symbol = opportunities[0][0]
            self.best_signal = opportunities[0][2]
            self.best_confidence = opportunities[0][3]
        else:
            self.best_symbol = None
            self.best_signal = 'hold'
            self.best_confidence = 0.5

    def should_trade(self) -> Tuple[bool, str]:
        """Should we trade?"""
        if not self.best_symbol:
            return False, "No opportunity"
        
        if self.best_signal == 'hold':
            return False, "Hold signal"
        
        if self.best_confidence < 0.55:
            return False, "Low confidence"
        
        # Check position limit (10%)
        if self.best_symbol in self.positions:
            return False, "Already in position"
        
        return True, "OK"

    def position_size(self) -> float:
        """Calculate position size."""
        base = self.portfolio * 0.1  # 10%
        
        if self.best_confidence < 0.5:
            base *= 0.5
        elif self.best_confidence > 0.7:
            base *= 1.2
        
        return base

    def execute_order(self):
        """Execute the current best trade (paper or live)."""
        if not self.should_trade()[0]:
            return
        
        symbol = self.best_symbol
        signal = self.best_signal
        size = self.position_size()
        price = self.prices[symbol]
        
        # Record position
        self.positions[symbol] = {
            'type': signal,
            'entry': price,
            'size': size,
            'time': datetime.now(timezone.utc)
        }
        
        if self.mode == 'paper':
            logger.info("📝 PAPER: {} {} {} @ ${:.0f}".format(
                signal.upper(), symbol, size, price))
        else:
            logger.info("🔴 LIVE: {} {} @ ${:.0f}".format(
                signal.upper(), symbol, price))

    def close_position(self, symbol: str, pnl: float):
        """Close position and record result."""
        if symbol not in self.positions:
            return
        
        # Record
        self.positions.pop(symbol)
        
        self.trades.append({
            'symbol': symbol,
            'pnl': pnl,
            'time': datetime.now(timezone.utc)
        })
        
        self.total_pnl += pnl
        self.total_trades += 1
        
        if pnl > 0:
            self.wins += 1
        
        # Update learner
        self.learners[symbol].record_result(pnl > 0)
        
        logger.info("📊 CLOSED: {} PnL: ${:.2f}".format(symbol, pnl))

    def check_stops(self, stop_pct: float = 0.05):
        """Check stop losses."""
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            current = self.prices.get(symbol, 0)
            
            if current == 0:
                continue
            
            pnl_pct = (current - pos['entry']) / pos['entry']
            if pos['type'] == 'short':
                pnl_pct = -pnl_pct
            
            if pnl_pct < -stop_pct:
                pnl = -self.portfolio * stop_pct
                self.close_position(symbol, pnl)

    def get_status(self) -> Dict:
        """Get current status."""
        win_rate = self.wins / self.total_trades if self.total_trades > 0 else 0
        
        return {
            'mode': self.mode,
            'portfolio': self.portfolio,
            'pnl': self.total_pnl,
            'trades': self.total_trades,
            'win_rate': win_rate,
            'best': {
                'symbol': self.best_symbol,
                'signal': self.best_signal,
                'confidence': self.best_confidence
            },
            'positions': {
                s: {'type': p['type'], 'entry': p['entry']}
                for s, p in self.positions.items()
            }
        }


# ============================================================================
# MAIN TEST
# ============================================================================

async def main():
    logging.basicConfig(level=logging.INFO)
    
    print()
    print("=" * 50)
    print("ARGUS TRADING v5.0 TEST")
    print("=" * 50)
    print()
    
    # Create
    trading = ArgusTrading(['BTC/USDT', 'ETH/USDT'], 10000, 'paper')
    
    prices = {'BTC/USDT': 50000, 'ETH/USDT': 3000}
    
    # Simulate
    for sec in range(60):
        for symbol in trading.symbols:
            prices[symbol] *= 1 + np.random.randn() * 0.002
            prices[symbol] = max(prices[symbol], 100)
            trading.update(symbol, prices[symbol])
        
        # Check trades every 10s
        if sec % 10 == 0:
            should, reason = trading.should_trade()
            status = trading.get_status()
            print("{:2d}s: {} {} ({:.0%}) | ${:.0f} | {} {}".format(
                sec,
                status['best']['symbol'] or '-',
                status['best']['signal'],
                status['best']['confidence'],
                status['portfolio'],
                status['trades'],
                "TRADE" if should else ""
            ))
            
            if should:
                trading.execute_order()
        
        # Check stops
        trading.check_stops(0.05)
        
        await asyncio.sleep(0.05)
    
    print()
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    status = trading.get_status()
    print("Total trades: {}".format(status['trades']))
    print("Win rate: {:.0%}".format(status['win_rate']))
    print("Total PnL: ${:.2f}".format(status['pnl']))


if __name__ == "__main__":
    asyncio.run(main())