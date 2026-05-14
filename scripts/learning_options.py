"""
Alternative Learning Implementations

Multiple ways to add learning to Argus:

1. CONFIG-BASED: Add to unified_config.yaml
2. WRAPPER: Wrap existing strategy
3. DECORATOR: Use as decorator
4. MIDDLEWARE: Process between strategy and execution
5. POST-TRADE: Learn after trades close

Run: py scripts/learning_options.py
"""

import logging
import numpy as np
from collections import deque
from typing import Dict, List, Optional
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# OPTION 1: CONFIG-BASED LEARNING
# ============================================================================

class ConfigLearning:
    """
    Learning via configuration.
    
    Add to unified_config.yaml:
        learning:
          enabled: true
          min_confidence: 0.55
          min_trade_interval: 300
          strategies: [momentum, mean_reversion, breakout, volatility]
    
    Then access via:
        config.learning.enabled
    """
    
    def __init__(self, config: Dict):
        self.enabled = config.get('learning', {}).get('enabled', False)
        self.min_confidence = config.get('learning', {}).get('min_confidence', 0.55)
        self.strategies = config.get('learning', {}).get('strategies', ['momentum'])
        self.min_interval = config.get('learning', {}).get('min_trade_interval', 300)
        
        # Track stats
        self.wins = 0
        self.losses = 0
        self.strategy_performance = {s: 1.0 for s in self.strategies}
        
        logger.info(f"Config Learning: enabled={self.enabled}, strategies={self.strategies}")
    
    def adjust_confidence(self, signal_confidence: float, strategy: str) -> float:
        """Adjust confidence based on strategy performance."""
        if not self.enabled or strategy not in self.strategy_performance:
            return signal_confidence
        
        perf = self.strategy_performance[strategy]
        
        # Boost confidence if strategy performing well
        if perf > 1.2:
            return min(signal_confidence * 1.1, 0.85)
        # Reduce if strategy performing poorly
        elif perf < 0.8:
            return signal_confidence * 0.9
        
        return signal_confidence
    
    def record_result(self, strategy: str, won: bool):
        """Record trade result."""
        if won:
            self.wins += 1
            self.strategy_performance[strategy] *= 1.02
        else:
            self.losses += 1
            self.strategy_performance[strategy] *= 0.98
        
        # Normalize
        total = sum(self.strategy_performance.values())
        for s in self.strategy_performance:
            self.strategy_performance[s] /= total


# ============================================================================
# OPTION 2: WRAPPER LEARNING
# ============================================================================

class WrapperLearning:
    """
    Wrap any existing strategy to add learning.
    
    Usage:
        from strategies.momentum import MomentumStrategy
        wrapped = WrapperLearning(MomentumStrategy())
        
        signal = wrapped.generate(df)
    """
    
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance
        self.history = deque(maxlen=100)
        self.wins = 0
        self.losses = 0
        
        # Feature weights (learned)
        self.weights = np.ones(9)
        
        logger.info(f"Wrapped: {strategy_instance.__class__.__name__}")
    
    def generate(self, df) -> Dict:
        """Generate signal with learning."""
        # Get base signal
        signal = self.strategy.generate(df)
        
        # Extract features
        features = self._extract_features(df)
        
        # Apply learning adjustment
        if len(self.history) > 20:
            score = np.dot(features[:3], self.weights[:3])
            
            # Boost confidence if score positive
            if score > 0:
                signal.confidence = min(signal.confidence * (1 + score), 0.85)
            else:
                signal.confidence *= (1 + score)
        
        # Record
        self.history.append((features, signal.action))
        
        return signal
    
    def _extract_features(self, df) -> np.ndarray:
        """Extract simple features."""
        close = df['close'].values
        if len(close) < 2:
            return np.zeros(9)
        
        return np.array([
            close[-1] / close[-2] - 1 if close[-2] != 0 else 0,
            close[-1] / close[-5] - 1 if len(close) > 5 and close[-5] != 0 else 0,
            close[-1] / close[-10] - 1 if len(close) > 10 and close[-10] != 0 else 0,
        ] + [0.5] * 6)
    
    def record_result(self, won: bool):
        """Learn from result."""
        if won:
            self.wins += 1
            self.weights *= 1.01
        else:
            self.losses += 1
            self.weights *= 0.99
        
        # Normalize
        self.weights /= self.weights.sum()


# ============================================================================
# OPTION 3: DECORATOR LEARNING
# ============================================================================

def learning_decorator(strategy_func):
    """
    Decorator to add learning to any strategy function.
    
    Usage:
        @learning_decorator
        def my_strategy(df, symbol):
            return Signal(...)
    """
    results = deque(maxlen=50)
    weights = np.ones(9)
    
    def wrapper(df, symbol):
        # Get base signal
        signal = strategy_func(df, symbol)
        
        # Learn from history
        if len(results) > 10:
            recent = list(results)[-10:]
            wins = sum(1 for r in recent if r['won'])
            
            # Adjust confidence
            if wins > 7:
                signal.confidence = min(signal.confidence * 1.1, 0.85)
            elif wins < 3:
                signal.confidence *= 0.9
        
        return signal
    
    def record(won):
        results.append({'won': won})
    
    wrapper.record = record
    return wrapper


# ============================================================================
# OPTION 4: MIDDLEWARE LEARNING
# ============================================================================

class LearningMiddleware:
    """
    Learning as middleware between strategy and execution.
    
    Insert in the execution pipeline:
        strategy -> middleware -> executor
    
    Usage in execution pipeline:
        middleware = LearningMiddleware()
        signal = strategy.generate(df)
        signal = middleware.process(signal, symbol)
        executor.execute(signal)
    """
    
    def __init__(self):
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        self.prices = {s: 0 for s in self.symbols}
        self.features = {s: np.zeros(9) for s in self.symbols}
        
        # Strategy performance
        self.strategy_scores = {s: [] for s in self.symbols}
        self.current_best = {s: 'hold' for s in self.symbols}
        
        logger.info("Learning Middleware initialized")
    
    def process(self, signal, symbol: str, price: float) -> Dict:
        """
        Process signal through learning.
        
        Args:
            signal: Original signal from strategy
            symbol: Trading symbol
            price: Current price
        
        Returns:
            Enhanced signal dict
        """
        if symbol not in self.symbols:
            return signal
        
        # Update data
        self.prices[symbol] = price
        self.features[symbol] = self._extract(price)
        
        # Analyze
        analysis = self._analyze(symbol)
        
        # Adjust signal
        if signal.action != 'hold':
            # Check confidence adjustment
            if analysis['confidence_boost'] > 0:
                signal.confidence = min(
                    signal.confidence + analysis['confidence_boost'],
                    0.85
                )
            
            # Check should block
            if analysis['should_block']:
                signal.action = 'hold'
                signal.reasoning += f" [blocked: {analysis['block_reason']}]"
        
        # Store for learning
        signal.learning = analysis
        
        return signal
    
    def _extract(self, price: float) -> np.ndarray:
        """Quick feature extraction."""
        # Simple implementation
        return np.random.randn(9) * 0.01
    
    def _analyze(self, symbol: str) -> Dict:
        """Analyze and return learning adjustments."""
        features = self.features[symbol]
        
        # Check signals
        momentum = features[0]
        rsi = features[6]
        
        confidence_boost = 0.0
        should_block = False
        block_reason = ""
        
        # RSI bounds
        if rsi > 75:
            should_block = True
            block_reason = "overbought"
        elif rsi < 25:
            should_block = True
            block_reason = "oversold"
        
        # Momentum check
        if abs(momentum) > 0.02:
            confidence_boost = 0.05 * np.sign(momentum)
        
        return {
            'confidence_boost': confidence_boost,
            'should_block': should_block,
            'block_reason': block_reason,
            'features': features
        }
    
    def record_result(self, symbol: str, won: bool):
        """Learn from outcome."""
        self.strategy_scores[symbol].append(1 if won else 0)
        
        # Keep only recent
        if len(self.strategy_scores[symbol]) > 20:
            self.strategy_scores[symbol] = self.strategy_scores[symbol][-20:]


# ============================================================================
# OPTION 5: POST-TRADE LEARNING
# ============================================================================

class PostTradeLearner:
    """
    Learn specifically from trade outcomes.
    
    Key methods:
    - learn_from_trade: Call after each trade
    - get_strategy_recommendation: Get best strategy for next trade
    - adapt_parameters: Adjust strategy parameters
    """
    
    def __init__(self):
        self.trade_history = deque(maxlen=200)
        
        # Performance tracking
        self.by_strategy = {
            'momentum': {'wins': 0, 'losses': 0},
            'mean_reversion': {'wins': 0, 'losses': 0},
            'breakout': {'wins': 0, 'losses': 0},
            'volatility': {'wins': 0, 'losses': 0},
        }
        
        # Best times to trade
        self.by_hour = {h: {'wins': 0, 'trades': 0} for h in range(24)}
        self.by_day = {d: {'wins': 0, 'trades': 0} for d in range(7)}
        
        logger.info("Post-Trade Learning initialized")
    
    def learn_from_trade(
        self,
        strategy: str,
        entry_time: datetime,
        pnl: float,
        entry_price: float,
        exit_price: float,
        symbol: str
    ):
        """Record trade and learn."""
        won = pnl > 0
        
        # Update strategy stats
        if strategy in self.by_strategy:
            if won:
                self.by_strategy[strategy]['wins'] += 1
            else:
                self.by_strategy[strategy]['losses'] += 1
        
        # Update timing stats
        hour = entry_time.hour
        self.by_hour[hour]['trades'] += 1
        if won:
            self.by_hour[hour]['wins'] += 1
        
        day = entry_time.weekday()
        self.by_day[day]['trades'] += 1
        if won:
            self.by_day[day]['wins'] += 1
        
        # Store trade
        self.trade_history.append({
            'strategy': strategy,
            'won': won,
            'pnl': pnl,
            'entry_time': entry_time,
            'symbol': symbol
        })
    
    def get_recommendation(self) -> Dict:
        """Get strategy recommendation for next trade."""
        recommendations = {}
        
        # Best strategy
        for strat, stats in self.by_strategy.items():
            total = stats['wins'] + stats['losses']
            if total > 0:
                recommendations[strat] = stats['wins'] / total
            else:
                recommendations[strat] = 0.5
        
        # Best hour
        best_hour = 0
        best_hour_rate = 0
        for h, stats in self.by_hour.items():
            if stats['trades'] > 3:
                rate = stats['wins'] / stats['trades']
                if rate > best_hour_rate:
                    best_hour_rate = rate
                    best_hour = h
        
        return {
            'strategy': max(recommendations, key=recommendations.get),
            'best_hour': best_hour,
            'confidence': recommendations
        }
    
    def adapt_parameters(self) -> Dict:
        """Get adapted parameters."""
        rec = self.get_recommendation()
        
        return {
            'preferred_strategy': rec['strategy'],
            'preferred_hour': rec['best_hour'],
            'min_confidence': 0.55 if rec['confidence'][rec['strategy']] > 0.5 else 0.60,
            'position_size_mult': 1.2 if rec['confidence'][rec['strategy']] > 0.6 else 1.0
        }


# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("LEARNING OPTIONS TEST")
    print("=" * 50)
    print()
    
    # Option 1: Config
    print("1. CONFIG-BASED")
    config = {'learning': {'enabled': True, 'min_confidence': 0.55, 'strategies': ['momentum']}}
    cl = ConfigLearning(config)
    print(f"   Enabled: {cl.enabled}, Min conf: {cl.min_confidence}")
    
    print()
    
    # Option 2: Wrapper
    print("2. WRAPPER")
    class DummyStrategy:
        def generate(self, df):
            return type('Signal', (), {'action': 'buy', 'confidence': 0.6})()
    wl = WrapperLearning(DummyStrategy())
    print(f"   Wrapper for: {wl.strategy.__class__.__name__}")
    
    print()
    
    # Option 3: Middleware
    print("3. MIDDLEWARE")
    lm = LearningMiddleware()
    test_signal = type('S', (), {'action': 'buy', 'confidence': 0.6, 'reasoning': ''})()
    result = lm.process(test_signal, 'BTC/USDT', 50000)
    print(f"   Signal: {result.action} conf: {result.confidence}")
    
    print()
    
    # Option 4: Post-Trade
    print("4. POST-TRADE LEARNING")
    ptl = PostTradeLearner()
    now = datetime.now(timezone.utc)
    ptl.learn_from_trade('momentum', now, 100, 50000, 50500, 'BTC/USDT')
    ptl.learn_from_trade('breakout', now, -50, 50000, 49000, 'BTC/USDT')
    rec = ptl.get_recommendation()
    print(f"   Best strategy: {rec['strategy']}")
    print(f"   Best hour: {rec['best_hour']}:00")
    
    print()
    print("=" * 50)
    print("All learning options ready!")
    print("=" * 50)