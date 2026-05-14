"""
Enhanced Trading System v5.0

Complete trading system with:
1. Multi-symbol scanning (multiple pairs)
2. Advanced signal generation (multiple strategies)
3. Position sizing (confidence-based)
4. Risk management (stop loss, max risk)
5. Continuous learning (every 0.5s)

Run: py scripts/enhanced_trading.py
"""

import logging
import asyncio
import time
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import threading

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# STRATEGIES
# ============================================================================

class Strategy:
    """Base strategy."""
    
    def __init__(self, name: str):
        self.name = name
        self.weight = 1.0
        
    def score(self, features: np.ndarray) -> float:
        return 0.0
    
    def get_signal(self, features: np.ndarray) -> str:
        return 'hold'


class MomentumStrategy(Strategy):
    """Follows momentum."""
    
    def __init__(self):
        super().__init__('momentum')
    
    def score(self, features: np.ndarray) -> float:
        # r1, r4, r12 are returns
        return features[0] * 2 + features[1] + features[2] * 0.5
    
    def get_signal(self, features: np.ndarray) -> str:
        score = self.score(features)
        if score > 0.005:
            return 'buy'
        elif score < -0.005:
            return 'sell'
        return 'hold'


class MeanReversionStrategy(Strategy):
    """Mean reversion."""
    
    def __init__(self):
        super().__init__('mean_reversion')
    
    def score(self, features: np.ndarray) -> float:
        # rsi based - oversold = buy
        rsi = features[6]
        if rsi < 30:
            return 0.5  # Oversold - buy
        elif rsi > 70:
            return -0.5  # Overbought - sell
        return 0.0
    
    def get_signal(self, features: np.ndarray) -> str:
        score = self.score(features)
        if score > 0.3:
            return 'buy'
        elif score < -0.3:
            return 'sell'
        return 'hold'


class BreakoutStrategy(Strategy):
    """Breakout trading."""
    
    def __init__(self):
        super().__init__('breakout')
    
    def score(self, features: np.ndarray) -> float:
        # r24 breakout
        return features[3] * 3
    
    def get_signal(self, features: np.ndarray) -> str:
        score = self.score(features)
        if score > 0.02:
            return 'buy'
        elif score < -0.02:
            return 'sell'
        return 'hold'


class VolatilityStrategy(Strategy):
    """Low volatility - mean reversion."""
    
    def __init__(self):
        super().__init__('volatility')
    
    def score(self, features: np.ndarray) -> float:
        v24 = features[5]
        # Low volatility + mean reversion
        if v24 < 0.01:
            return -features[0]  # Short rallies
        return 0.0
    
    def get_signal(self, features: np.ndarray) -> str:
        score = self.score(features)
        if score > 0.005:
            return 'buy'
        elif score < -0.005:
            return 'sell'
        return 'hold'


# ============================================================================
# PER-SYMBOL LEARNER
# ============================================================================

class SymbolLearner:
    """Learns for one symbol."""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        
        # Data
        self.prices = deque(maxlen=500)
        self.features = deque(maxlen=500)
        
        # Strategies
        self.strategies = [
            MomentumStrategy(),
            MeanReversionStrategy(),
            BreakoutStrategy(),
            VolatilityStrategy()
        ]
        
        # Weights (learned)
        self.strategy_weights = np.ones(len(self.strategies))
        
        # Feature weights
        self.feature_weights = np.zeros(9)
        self.feature_importance = np.ones(9)
        
        # Stats
        self.total_updates = 0
        self.wins = 0
        self.losses = 0
        
        # Current state
        self.signal = 'hold'
        self.confidence = 0.5
    
    def update(self, price: float) -> np.ndarray:
        """Update with new price."""
        self.prices.append(price)
        
        features = self._extract_features()
        if len(self.features) > 0:
            self._learn(features)
        
        self.features.append(features)
        self.total_updates += 1
        
        self._generate_signal(features)
        
        return features
    
    def _extract_features(self) -> np.ndarray:
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
    
    def _learn(self, features: np.ndarray):
        """Learn from features."""
        if len(self.features) < 20:
            return
        
        # Update strategy weights based on correlation with returns
        recent = np.array(list(self.features)[-20:])
        
        for i, strat in enumerate(self.strategies):
            scores = np.array([strat.score(f) for f in recent])
            returns = np.diff(np.array(list(self.prices)[-21:])) / np.array(list(self.prices)[-21:-1])
            
            if len(returns) > 0:
                corr = np.corrcoef(scores, returns)
                if not np.isnan(corr[0, 1]):
                    self.strategy_weights[i] += 0.01 * corr[0, 1]
        
        # Normalize
        self.strategy_weights = np.maximum(self.strategy_weights, 0.1)
        self.strategy_weights /= self.strategy_weights.sum()
    
    def _generate_signal(self, features: np.ndarray):
        """Generate combined signal."""
        if len(self.features) < 20:
            self.signal = 'hold'
            self.confidence = 0.5
            return
        
        # Weighted strategy scores
        total_score = 0
        for i, strat in enumerate(self.strategies):
            score = strat.score(features)
            total_score += score * self.strategy_weights[i]
        
        if total_score > 0.005:
            self.signal = 'buy'
        elif total_score < -0.005:
            self.signal = 'sell'
        else:
            self.signal = 'hold'
        
        # Confidence from weight
        self.confidence = min(0.5 + abs(total_score) * 5, 0.85)
    
    def record_result(self, won: bool):
        """Record trade outcome."""
        if won:
            self.wins += 1
            self.strategy_weights *= 1.02
        else:
            self.losses += 1
            self.strategy_weights *= 0.98
        
        self.strategy_weights = np.maximum(self.strategy_weights, 0.1)
        self.strategy_weights /= self.strategy_weights.sum()
    
    def get_stats(self) -> Dict:
        total = self.wins + self.losses
        return {
            'symbol': self.symbol,
            'signal': self.signal,
            'confidence': self.confidence,
            'win_rate': self.wins / total if total > 0 else 0,
            'updates': self.total_updates,
            'strategy_weights': list(self.strategy_weights)
        }


# ============================================================================
#RISK MANAGER
# ============================================================================

class RiskManager:
    """Manages risk."""
    
    def __init__(
        self,
        max_position_pct: float = 0.1,  # 10% of portfolio
        max_loss_pct: float = 0.02,  # 2% per trade
        stop_loss_pct: float = 0.05,  # 5% stop loss
    ):
        self.max_position_pct = max_position_pct
        self.max_loss_pct = max_loss_pct
        self.stop_loss_pct = stop_loss_pct
        
        self.daily_pnl = 0
        self.daily_trades = 0
        self.max_daily_loss = 0
    
    def can_trade(self, portfolio_value: float, position_value: float) -> Tuple[bool, str]:
        """Check if can trade."""
        # Max position size
        if position_value >= portfolio_value * self.max_position_pct:
            return False, "Max position"
        
        # Max daily loss
        if self.daily_pnl <= -portfolio_value * self.max_daily_loss:
            return False, "Max daily loss"
        
        return True, "OK"
    
    def position_size(self, portfolio_value: float, confidence: float) -> float:
        """Calculate position size."""
        # Base on confidence
        base = portfolio_value * self.max_position_pct
        
        # Adjust for confidence
        if confidence < 0.5:
            base *= 0.5
        elif confidence > 0.7:
            base *= 1.2
        
        return base
    
    def record_pnl(self, pnl: float):
        """Record daily PnL."""
        self.daily_pnl += pnl
        self.daily_trades += 1
    
    def reset_daily(self):
        """Reset daily tracking."""
        self.daily_pnl = 0
        self.daily_trades = 0


# ============================================================================
# MAIN TRADING SYSTEM
# ============================================================================

class EnhancedTradingSystem:
    """
    Complete enhanced trading system.
    
    Features:
    - Multi-symbol scanning
    - Multiple strategies (4)
    - Position sizing
    - Risk management
    - Continuous learning
    """

    def __init__(
        self,
        symbols: List[str] = None,
        portfolio_value: float = 10000,
    ):
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        self.portfolio_value = portfolio_value
        
        # Per-symbol learners
        self.symbol_learners = {s: SymbolLearner(s) for s in self.symbols}
        
        # Risk manager
        self.risk = RiskManager(
            max_position_pct=0.1,
            max_loss_pct=0.02,
            stop_loss_pct=0.05
        )
        
        # Best opportunity
        self.best_symbol = None
        self.best_signal = 'hold'
        self.best_confidence = 0.5
        
        # Stats
        self.total_trades = 0
        self.is_running = False
        
        logger.info("=" * 60)
        logger.info("ENHANCED TRADING SYSTEM v5.0")
        logger.info("=" * 60)
        logger.info("Symbols: {}".format(self.symbols))
        logger.info("Strategies: 4 (momentum, mean_reversion, breakout, volatility)")
        logger.info("Risk: position={}, stop_loss={}".format(
            self.risk.max_position_pct, self.risk.stop_loss_pct))
        logger.info("=" * 60)

    def update(self, symbol: str, price: float):
        """Update a symbol."""
        learner = self.symbol_learners.get(symbol)
        if learner:
            learner.update(price)
            
            # Update best opportunity
            self._scan_best()

    def _scan_best(self):
        """Find best opportunity."""
        opportunities = []
        
        for symbol, learner in self.symbol_learners.items():
            stats = learner.get_stats()
            
            if learner.signal != 'hold' and learner.confidence > 0.5:
                score = learner.confidence * learner.wins / max(learner.wins + learner.losses, 1)
                opportunities.append((symbol, score, learner.signal, learner.confidence))
        
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
            return False, "Hold"
        
        if self.best_confidence < 0.55:
            return False, "Low confidence"
        
        # Check risk
        position_value = self.portfolio_value * self.best_confidence * 0.1
        can_trade, reason = self.risk.can_trade(self.portfolio_value, position_value)
        
        return can_trade, reason

    def get_position_size(self) -> float:
        """Get position size."""
        return self.risk.position_size(
            self.portfolio_value,
            self.best_confidence
        )

    def record_trade(self, symbol: str, pnl: float):
        """Record trade outcome."""
        self.total_trades += 1
        
        # Update learner
        learner = self.symbol_learners.get(symbol)
        if learner:
            learner.record_result(pnl > 0)
        
        # Update risk
        self.risk.record_pnl(pnl)
        
        # Update portfolio
        self.portfolio_value += pnl

    def get_state(self) -> Dict:
        """Get system state."""
        return {
            'best_symbol': self.best_symbol,
            'best_signal': self.best_signal,
            'best_confidence': self.best_confidence,
            'should_trade': self.should_trade()[0],
            'portfolio': self.portfolio_value,
            'total_trades': self.total_trades,
            'daily_pnl': self.risk.daily_pnl,
            'symbols': {
                s: self.symbol_learners[s].get_stats()
                for s in self.symbols
            }
        }


# ============================================================================
# TEST
# ============================================================================

async def main():
    logging.basicConfig(level=logging.INFO)
    
    print()
    print("=" * 60)
    print("ENHANCED TRADING SYSTEM v5.0 TEST")
    print("=" * 60)
    print()
    
    # Create system
    system = EnhancedTradingSystem(
        symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
        portfolio_value=10000
    )
    
    # Simulate prices
    prices = {
        'BTC/USDT': 50000,
        'ETH/USDT': 3000,
        'SOL/USDT': 100
    }
    
    # Run simulation
    for cycle in range(40):
        for symbol in system.symbols:
            # Random walk
            prices[symbol] *= 1 + np.random.randn() * 0.002
            prices[symbol] = max(prices[symbol], 100)
            
            system.update(symbol, prices[symbol])
        
        if cycle % 10 == 0:
            state = system.get_state()
            print("Cycle {:2d}: {} {} ({:.0%}) | ${:.0f} | Trades: {}".format(
                cycle,
                state['best_symbol'] or '-',
                state['best_signal'],
                state['best_confidence'],
                state['portfolio'],
                state['total_trades']
            ))
        
        await asyncio.sleep(0.05)  # Fast simulation
    
    print()
    print("=" * 60)
    print("FINAL")
    print("=" * 60)
    state = system.get_state()
    print("Best: {} {} ({:.0%})".format(
        state['best_symbol'], state['best_signal'], state['best_confidence']))
    print("Portfolio: ${:.2f}".format(state['portfolio']))
    print("Total trades: {}".format(state['total_trades']))
    print("Daily PnL: ${:.2f}".format(state['daily_pnl']))


if __name__ == "__main__":
    asyncio.run(main())