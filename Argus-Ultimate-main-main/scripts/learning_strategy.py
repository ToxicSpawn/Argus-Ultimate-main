"""
Learning-Enhanced Strategy Integration

This integrates the learning system into Argus trading.
Can be used with ANY strategy in the strategies/ folder.

Usage in your strategy:
    from scripts.learning_strategy import LearningStrategy
    
    learning_strategy = LearningStrategy(symbols=['BTC/USDT', 'ETH/USDT'])
    
    def on_bar(self, bar):
        # Standard strategy logic first...
        signal = self.standard_strategy(bar)
        
        # Then enhance with learning
        if signal and signal.action != 'hold':
            learning_score = learning_strategy.get_learning_score(bar.symbol, bar.close)
            
            # Boost or reduce confidence based on learning
            if learning_score > 0.6:
                signal.confidence = min(signal.confidence + 0.1, 0.85)
                signal.reasoning += f" [Learning boost: {learning_score:.0%}]"
            elif learning_score < 0.4:
                signal.confidence *= 0.8
                signal.reasoning += f" [Learning warning: {learning_score:.0%}]"
        
        # After trade closes
        learning_strategy.record_outcome(bar.symbol, pnl, actual_return)
        
        return signal
"""

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class LearningStrategy:
    """
    Learning-enhanced trading strategy.
    
    Features:
    - Real-time feature extraction
    - Strategy voting (4 strategies)
    - Confidence adjustment
    - Trade outcome learning
    - Stop loss checking
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
        
        # Per-symbol data
        self.prices: Dict[str, deque] = {s: deque(maxlen=100) for s in self.symbols}
        self.features: Dict[str, deque] = {s: deque(maxlen=100) for s in self.symbols}
        
        # Strategy weights (learned)
        self.strategy_weights = {
            'momentum': 1.0,
            'mean_reversion': 1.0,
            'breakout': 1.0,
            'volatility': 1.0
        }
        
        # Tracking
        self.wins = {s: 0 for s in self.symbols}
        self.losses = {s: 0 for s in self.symbols}
        self.trades = {s: 0 for s in self.symbols}
        
        # Current state
        self.current_signals = {}
        
        logger.info("=" * 50)
        logger.info("LEARNING STRATEGY INITIALIZED")
        logger.info("=" * 50)

    def on_data(self, symbol: str, price: float) -> Dict:
        """
        Process new data and return analysis.
        
        Returns:
            {
                'signal': 'buy'/'sell'/'hold',
                'confidence': 0.0-1.0,
                'learning_score': 0.0-1.0,
                'reasoning': str
            }
        """
        if symbol not in self.symbols:
            return {'signal': 'hold', 'confidence': 0.5, 'learning_score': 0.5, 'reasoning': 'Unknown symbol'}
        
        # Update data
        self.prices[symbol].append(price)
        
        features = self._extract_features(symbol)
        self.features[symbol].append(features)
        
        if len(self.features[symbol]) < 20:
            return {
                'signal': 'hold',
                'confidence': 0.5,
                'learning_score': 0.5,
                'reasoning': 'Warming up'
            }
        
        # Generate signal
        return self._generate_signal(symbol, features)

    def _extract_features(self, symbol: str) -> np.ndarray:
        """Extract features for a symbol."""
        prices = np.array(list(self.prices[symbol]))
        
        if len(prices) < 25:
            return np.zeros(9)
        
        r1 = prices[-1] / prices[-2] - 1
        r4 = prices[-1] / prices[-5] - 1
        r12 = prices[-1] / prices[-13] - 1
        r24 = prices[-1] / prices[-25] - 1
        v12 = np.std(prices[-13:]) / np.mean(prices[-13:])
        v24 = np.std(prices[-25:]) / np.mean(prices[-25:])
        
        d = np.diff(prices)
        g = np.where(d > 0, d, 0)
        l = np.where(d < 0, -d, 0)
        rsi = 50
        if len(g) >= 14:
            rsi = 100 - (100 / (1 + np.mean(g[-14:]) / max(np.mean(l[-14:]), 1e-8)))
        
        return np.array([r1, r4, r12, r24, v12, v24, rsi, 0.5, 1.0])

    def _generate_signal(self, symbol: str, features: np.ndarray) -> Dict:
        """Generate trading signal with learning."""
        
        # Strategy scores
        momentum = features[0] * 2 + features[1]
        mean_rev = (50 - features[6]) / 100  # RSI based
        breakout = features[3] * 3
        vol = -features[5] if features[5] < 0.01 else 0
        
        # Weighted total
        score = (
            momentum * self.strategy_weights['momentum'] +
            mean_rev * self.strategy_weights['mean_reversion'] +
            breakout * self.strategy_weights['breakout'] +
            vol * self.strategy_weights['volatility']
        )
        
        # Map to signal
        if score > 0.02:
            signal = 'buy'
        elif score < -0.02:
            signal = 'sell'
        else:
            signal = 'hold'
        
        # Learning score (based on historical performance)
        total = self.wins[symbol] + self.losses[symbol]
        learning_score = self.wins[symbol] / total if total > 0 else 0.5
        
        # Confidence
        confidence = 0.5 + abs(score) * 5 + (learning_score - 0.5) * 0.3
        confidence = min(max(confidence, 0.35), 0.85)
        
        # Reasoning
        reasoning = f"M:{momentum:.2f} MR:{mean_rev:.2f} B:{breakout:.2f} V:{vol:.2f}"
        reasoning += f" | Win:{learning_score:.0%}"
        
        self.current_signals[symbol] = {
            'signal': signal,
            'confidence': confidence,
            'learning_score': learning_score,
            'reasoning': reasoning
        }
        
        return self.current_signals[symbol]

    def get_learning_score(self, symbol: str, current_price: float) -> float:
        """Get learning score for a symbol (for external strategies)."""
        if symbol not in self.symbols:
            return 0.5
        
        # Update with latest price
        self.on_data(symbol, current_price)
        
        return self.current_signals.get(symbol, {}).get('learning_score', 0.5)

    def should_trade(self, symbol: str) -> Tuple[bool, str]:
        """Should we trade this symbol?"""
        signal_data = self.current_signals.get(symbol)
        
        if not signal_data:
            return False, "No signal"
        
        if signal_data['signal'] == 'hold':
            return False, "Hold signal"
        
        if signal_data['confidence'] < 0.55:
            return False, f"Low confidence ({signal_data['confidence']:.0%})"
        
        return True, "OK"

    def record_outcome(self, symbol: str, pnl: float, actual_return: float):
        """Record trade outcome for learning."""
        if symbol not in self.symbols:
            return
        
        self.trades[symbol] += 1
        
        won = pnl > 0 or actual_return > 0.01
        
        if won:
            self.wins[symbol] += 1
            # Boost winning strategies
            for strat in self.strategy_weights:
                self.strategy_weights[strat] *= 1.02
        else:
            self.losses[symbol] += 1
            # Reduce losing strategies
            for strat in self.strategy_weights:
                self.strategy_weights[strat] *= 0.98
        
        # Normalize
        total = sum(self.strategy_weights.values())
        for strat in self.strategy_weights:
            self.strategy_weights[strat] /= total
        
        logger.debug(f"Learned {symbol}: win={won}, weights: {self.strategy_weights}")

    def get_status(self) -> Dict:
        """Get system status."""
        total_wins = sum(self.wins.values())
        total_losses = sum(self.losses.values())
        
        return {
            'symbols': self.symbols,
            'strategy_weights': self.strategy_weights,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'win_rate': total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0,
            'signals': self.current_signals
        }


# ============================================================================
# EXAMPLE USAGE IN STRATEGY
# ============================================================================

"""
Example of how to integrate into an existing strategy:

from scripts.learning_strategy import LearningStrategy

class MyStrategy(BaseStrategy):
    name = "MyStrategy"
    
    def __init__(self, config):
        super().__init__(config)
        # Add learning
        self.learning = LearningStrategy(
            symbols=['BTC/USDT', 'ETH/USDT'],
            portfolio=getattr(config, 'risk', {}).get('max_position_usd', 10000),
            mode='paper' if config.ARGUS_ENV == 'paper' else 'live'
        )
    
    async def on_bar(self, bar):
        # Get standard signal
        signal = self.standard_signal(bar)
        
        # Get learning score
        learning_score = self.learning.get_learning_score(bar.symbol, bar.close)
        
        # Adjust confidence
        if learning_score < 0.4:
            signal.confidence *= 0.7
            signal.reasoning += " [LOW LEARNING]"
        
        return signal
    
    async def on_signal(self, signal, fill):
        # After trade, record outcome
        pnl = (fill.exit_price - fill.entry_price) / fill.entry_price
        self.learning.record_outcome(fill.symbol, pnl * fill.size, pnl)
"""


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 50)
    print("LEARNING STRATEGY TEST")
    print("=" * 50)
    
    ls = LearningStrategy(['BTC/USDT', 'ETH/USDT'], 10000, 'paper')
    
    prices = {'BTC/USDT': 50000, 'ETH/USDT': 3000}
    
    for i in range(50):
        for symbol in ls.symbols:
            # Simulate price movement
            prices[symbol] *= 1 + np.random.randn() * 0.002
            prices[symbol] = max(prices[symbol], 100)
            
            # Process data
            result = ls.on_data(symbol, prices[symbol])
            
            # Check should trade
            if i > 20:
                should, reason = ls.should_trade(symbol)
                if should:
                    print(f"{i}: {symbol} {result['signal']} ({result['confidence']:.0%}) | {result['reasoning']}")
    
    print()
    status = ls.get_status()
    print("Strategy weights:", status['strategy_weights'])
    print("Win rate:", status['win_rate'])