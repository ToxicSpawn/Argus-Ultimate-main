"""
Continuous Learning System - Learns from Every Market Cycle

Unlike batch learning, this learns continuously:
1. From every bar (not just trades)
2. From price patterns
3. From regime transitions
4. From signal outcomes

Run: py scripts/continuous_learning.py
"""

import logging
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Adaptive config integration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from adaptive_config_learning import (
    AdaptiveConfigLearner,
    ConfigObservation,
)


class ContinuousLearningSystem:
    """
    Continuous learning that adapts in REAL-TIME.
    
    Learning sources:
    1. Price movements - every bar
    2. Signal outcomes - every trade
    3. Regime changes - detected transitions
    4. Feature correlations - discovered patterns
    
    Adaptation speed:
    - Immediate: Adjust confidence after each outcome
    - Fast: Update weights after 10 trades
    - Detect drift: After 20 trades
    - Full retrain: After 100 trades
    """

    def __init__(
        self,
        initial_capital: float = 10000,
        # Learning rates
        fast_lr: float = 0.1,    # 10% fast learning from recent
        slow_lr: float = 0.01,   # 1% slow drift
        # Thresholds
        min_confidence: float = 0.50,
        drift_threshold: float = 0.10,  # 10% drift detection
        # Features
        feature_dim: int = 9,
        # Adaptive config
        enable_adaptive_config: bool = True,
        adaptive_config_path: str = "config/runtime/adaptive_overlay.json",
    ):
        # Capital
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # Adaptive config
        self.enable_adaptive_config = enable_adaptive_config
        self.adaptive_learner = AdaptiveConfigLearner(adaptive_config_path)
        self.current_overlay = {}
        self.trade_history = []  # Store trade results for adaptive analysis
        
        # Adaptive config
        self.enable_adaptive_config = enable_adaptive_config
        if self.enable_adaptive_config:
            self.adaptive_learner = AdaptiveConfigLearner(adaptive_config_path)
            self.current_overlay = self.adaptive_learner.apply_overlay({})
            logger.info("Adaptive config learner initialized with overlay: %s", adaptive_config_path)
        
        # Learning rates
        self.fast_lr = fast_lr
        self.slow_lr = slow_lr
        self.drift_threshold = drift_threshold
        
        # Thresholds
        self.min_confidence = min_confidence
        
        # Feature learned weights (start neutral)
        self.feature_weights = np.zeros(feature_dim)
        self.feature_bias = 0.0
        
        # History buffers
        self.features_buffer: deque = deque(maxlen=500)
        self.outcomes_buffer: deque = deque(maxlen=500)
        self.signals_buffer: deque = deque(maxlen=500)
        
        # Learning state (lock for thread safety)
        self._lock = threading.Lock()
        self.total_learned = 0
        self.correct_learned = 0
        
        # Accuracy tracking (rolling)
        self.window_size = 20
        self.recent_correct = deque(maxlen=self.window_size)
        
        # Regime tracking
        self.current_regime = "sideways"
        self.regime_history = deque(maxlen=50)
        self.regime_transitions = 0
        
        # Correlation tracking (feature -> outcome)
        self.feature_correlations = np.zeros(feature_dim)
        self.correlation_count = 0
        
        # Performance
        self.equity_curve = [initial_capital]
        
        logger.info("=" * 60)
        logger.info("CONTINUOUS LEARNING SYSTEM INITIALIZED")
        logger.info("=" * 60)
        logger.info("Fast LR: {:.1%}".format(fast_lr))
        logger.info("Slow LR: {:.1%}".format(slow_lr))
        logger.info("Drift threshold: {:.0%}".format(drift_threshold))
        logger.info("Min confidence: {:.0%}".format(min_confidence))
        logger.info("=" * 60)

    def extract_features(self, df) -> np.ndarray:
        """Extract 9 features from OHLCV."""
        if len(df) < 24:
            return np.zeros(9)
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Returns
        r1 = (close[-1] / close[-2] - 1) if close[-2] != 0 else 0
        r4 = (close[-1] / close[-5] - 1) if len(close) > 5 and close[-5] != 0 else 0
        r12 = (close[-1] / close[-13] - 1) if len(close) > 13 and close[-13] != 0 else 0
        r24 = (close[-1] / close[-25] - 1) if len(close) > 25 and close[-25] != 0 else 0
        
        # Volatility
        v12 = np.std(close[-13:]) / np.mean(close[-13:]) if len(close) >= 13 else 0
        v24 = np.std(close[-25:]) / np.mean(close[-25:]) if len(close) >= 25 else 0
        
        # RSI
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        if len(gain) >= 14:
            avg_gain = np.mean(gain[-14:])
            avg_loss = np.mean(loss[-14:])
            rsi = 100 - (100 / (1 + avg_gain / max(avg_loss, 1e-8)))
        else:
            rsi = 50
        
        # Position
        pp = (close[-1] - np.min(low[-25:])) / (np.max(high[-25:]) - np.min(low[-25:]) + 1e-8) if len(low) >= 25 else 0.5
        
        # Volume ratio
        vr = volume[-1] / np.mean(volume[-25:]) if len(volume) >= 25 and np.mean(volume[-25:]) != 0 else 1.0
        
        features = np.array([r1, r4, r12, r24, v12, v24, rsi, pp, vr])
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features

    def learn_from_bar(self, features: np.ndarray, actual_return: float):
        """
        Learn from EVERY bar - continuous learning!
        
        This is the key difference from batch learning.
        """
        with self._lock:
            # Store for correlation
            self.features_buffer.append(features.copy())
            self.outcomes_buffer.append(actual_return)
            self.correlation_count += 1
            
            # Update feature correlations (moving average)
            if self.correlation_count >= 10:
                recent_features = np.array(list(self.features_buffer)[-50:])
                recent_outcomes = np.array(list(self.outcomes_buffer)[-50:])
                
                # Correlation: feature direction vs outcome direction
                for i in range(len(features)):
                    if abs(recent_features[:, i]).sum() > 0:
                        corr = np.corrcoef(recent_features[:, i], recent_outcomes)[0, 1]
                        if not np.isnan(corr):
                            self.feature_correlations[i] = (
                                0.9 * self.feature_correlations[i] + 0.1 * corr
                            )
            
            # Update weights if we have enough data
            if self.correlation_count >= 20:
                self._update_weights_from_correlation()
            
            self.total_learned += 1

    def _update_weights_from_correlation(self):
        """Update model weights based on learned correlations."""
        # Convert correlations to weights
        correlations = self.feature_correlations.copy()
        
        # Positive correlation = buy signal
        # Negative correlation = sell signal
        for i in range(len(self.feature_weights)):
            if abs(correlations[i]) > 0.1:  # Significant correlation
                # Fast update for strong correlations
                lr = self.fast_lr if correlations[i] > 0 else -self.fast_lr
                self.feature_weights[i] += lr * correlations[i]
        
        # Update bias based on recent outcomes
        recent = list(self.outcomes_buffer)[-20:] if self.outcomes_buffer else []
        if recent:
            avg_return = np.mean(recent)
            # Slow update for bias
            self.feature_bias += self.slow_lr * avg_return

    def generate_signal(self, features: np.ndarray, regime: str = "sideways") -> Dict:
        """
        Generate signal using learned weights.
        
        This is fully continuous - learns from every bar!
        """
        with self._lock:
            # Calculate signal from learned weights
            signal_score = np.dot(features, self.feature_weights) + self.feature_bias
            
            # Map to action
            if signal_score > 0.005:
                action = "buy"
                direction = 2
            elif signal_score < -0.005:
                action = "sell"
                direction = 0
            else:
                action = "hold"
                direction = 1
            
            # Calculate confidence from weight magnitude + recent accuracy
            weight_magnitude = np.mean(np.abs(self.feature_weights))
            accuracy = self.get_accuracy()
            
            # Confidence starts low, increases with learning
            confidence = 0.5 + weight_magnitude * 2 + (accuracy - 0.5) * 0.3
            confidence = np.clip(confidence, 0.35, 0.85)
            
            # Check for regime change
            new_regime = regime
            if regime != self.current_regime:
                self.regime_transitions += 1
                self.regime_history.append({
                    'from': self.current_regime,
                    'to': regime,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                self.current_regime = regime
            
            return {
                'action': action,
                'direction': direction,
                'confidence': confidence,
                'regime': regime,
                'signal_score': signal_score,
                'features': features,
                'weight_magnitude': weight_magnitude,
                'regime_change': new_regime != regime
            }

    def learn_from_outcome(self, signal: Dict, pnl: float, actual_return: float):
        """
        Learn from trade outcome -强化 learning.
        
        More aggressive than bar learning.
        """
        with self._lock:
            # Store signal
            self.signals_buffer.append({
                'signal': signal,
                'pnl': pnl,
                'actual_return': actual_return,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
            # Determine if correct
            direction = signal['direction']
            if direction == 2 and actual_return > 0.01:  # buy + positive
                correct = True
            elif direction == 0 and actual_return < -0.01:  # sell + negative
                correct = True
            elif direction == 1 and abs(actual_return) <= 0.01:  # hold + small
                correct = True
            else:
                correct = False
            
            self.recent_correct.append(1 if correct else 0)
            
            if correct:
                self.correct_learned += 1
            
            # Fast learning: Update after just 5 outcomes
            if len(self.recent_correct) >= 5:
                self._fast_adjust(signal, correct, actual_return)
            
            # Drift detection after 20
            if len(self.recent_correct) >= 20:
                self._check_drift()

    def _fast_adjust(self, signal: Dict, correct: bool, actual_return: float):
        """Fast adjustment after few outcomes."""
        features = signal.get('features', np.zeros(9))
        
        # Correct = reinforce (increase weight)
        # Incorrect = anti-reinforce (decrease weight)
        delta = self.fast_lr if correct else -self.fast_lr
        
        # Direction multiplier
        dir_mult = 1 if signal['direction'] == 2 else -1 if signal['direction'] == 0 else 0
        
        # Update weights
        for i in range(len(self.feature_weights)):
            self.feature_weights[i] += delta * dir_mult * features[i] * 0.5
        
        # Update bias
        self.feature_bias += delta * actual_return * 0.1

    def _check_drift(self):
        """Check for regime drift."""
        if len(self.recent_correct) < self.window_size:
            return
        
        recent = list(self.recent_correct)
        window = self.window_size
        
        # Recent accuracy
        recent_acc = sum(recent[-window:]) / window
        
        # Older accuracy (baseline)
        if len(recent) >= window * 2:
            older_acc = sum(recent[:window]) / window
            
            # Drift detected
            if recent_acc < older_acc - self.drift_threshold:
                # Reduce confidence, reset some weights
                self.feature_weights *= 0.5  # Halve weights on drift
                logger.warning(
                    "Drift detected! Recent: {:.0%}, Older: {:.0%}".format(
                        recent_acc, older_acc))
            elif recent_acc > older_acc + self.drift_threshold:
                # Improving! Boost weights
                self.feature_weights *= 1.2
                logger.info("Improving! Boosting weights. Recent: {:.0%}".format(recent_acc))

    def get_accuracy(self) -> float:
        """Get current accuracy."""
        if len(self.recent_correct) == 0:
            return 0.5
        return sum(self.recent_correct) / len(self.recent_correct)

    def get_performance(self) -> Dict:
        """Get performance metrics."""
        return {
            'total_learned': self.total_learned,
            'correct_learned': self.correct_learned,
            'accuracy': self.get_accuracy(),
            'feature_weights': self.feature_weights.tolist(),
            'bias': self.feature_bias,
            'regime_transitions': self.regime_transitions,
            'correlation_count': self.correlation_count,
            'equity': self.capital
        }


# ============================================================================
# BACKTEST
# ============================================================================

def run_backtest(cycles: int = 200):
    """Run backtest with continuous learning."""
    import pandas as pd
    
    print()
    print("=" * 60)
    print("CONTINUOUS LEARNING - BACKTEST")
    print("=" * 60)
    print()
    
    system = ContinuousLearningSystem(
        initial_capital=10000,
        fast_lr=0.15,  # 15% fast learning
        slow_lr=0.02,  # 2% slow learning
        min_confidence=0.50,
        drift_threshold=0.15,
        enable_adaptive_config=True
    )
    
    # Simulate market data
    np.random.seed(42)
    prices = 50000 + np.cumsum(np.random.randn(cycles + 50) * 100)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(cycles + 50) * 50),
        'low': prices - np.abs(np.random.randn(cycles + 50) * 50),
        'close': prices,
        'volume': np.abs(np.random.randn(cycles + 50)) * 1000 + 500
    })
    
    trades = 0
    for i in range(50, cycles + 50):
        # Learn from EVERY bar (continuous!)
        features = system.extract_features(df.iloc[:i+1])
        
        # Actual return (ground truth)
        actual_return = (df['close'].iloc[i] / df['close'].iloc[i-1] - 1)
        
        # Learn from bar (every cycle!)
        system.learn_from_bar(features, actual_return)
        
        # Generate signal
        regime = "bull" if features[2] > 0.1 else "bear" if features[2] < -0.1 else "sideways"
        signal = system.generate_signal(features, regime)
        
        # Trade if signal is strong enough
        if signal['confidence'] >= 0.50 and signal['action'] != 'hold':
            # Simulate trade
            size = 1000
            pnl = size * actual_return * (1 if signal['direction'] == 2 else -1)
            
            # Execute
            trades += 1
            trades_pnl = pnl - size * 0.002  # Fees
            system.capital += trades_pnl

            # Store trade result for adaptive config
            system.trade_history.append({
                "pnl": trades_pnl,
                "return": actual_return,
                "direction": signal["direction"],
                "confidence": signal["confidence"],
                "regime": signal["regime"],
                "capital": system.capital,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Learn from outcome (强化!)
            system.learn_from_outcome(signal, trades_pnl, actual_return)
            


    def _check_adaptive_config(self):
        """
        Analyze recent performance and suggest adaptive config changes.
        """
        if not self.enable_adaptive_config or len(self.trade_history) < 30:
            return
            
        # Calculate performance metrics
        pnls = [t["pnl"] for t in self.trade_history[-30:]]
        returns = [t["return"] for t in self.trade_history[-30:]]
        
        # Basic metrics
        win_rate = sum(1 for t in self.trade_history[-30:] if t["pnl"] > 0) / 30
        profit_factor = sum(p for p in pnls if p > 0) / abs(sum(p for p in pnls if p < 0)) if any(p < 0 for p in pnls) else 10.0
        drawdown = max(0, (max(t["capital"] for t in self.trade_history) - self.capital) / max(t["capital"] for t in self.trade_history))
        avg_slippage = 0.001  # Simulated slippage
        
        # Determine regime
        regime = "volatile" if max(abs(r) for r in returns) > 0.02 else "range"
        
        # Create observation
        observation = ConfigObservation(
            win_rate=win_rate,
            profit_factor=profit_factor,
            drawdown=drawdown,
            average_slippage=avg_slippage,
            regime=regime,
            trades=30
        )
        
        # Generate suggestions
        suggestions = self.adaptive_learner.suggest({}, observation)
        if suggestions:
            logger.info("Adaptive config suggestions: %s", [s.__dict__ for s in suggestions])
            overlay = self.adaptive_learner.write_overlay(suggestions)
            self.current_overlay = self.adaptive_learner.apply_overlay({}, overlay)
            logger.info("Applied adaptive overlay: %s", self.current_overlay)
        
        if (i - 50) % 25 == 0:
            perf = system.get_performance()
            acc = perf['accuracy']
            print("Cycle {:3d}: Trades={:3d} Acc={:.0%} W={:.2f} Regime={}".format(
                i - 50, trades, acc, np.mean(np.abs(perf['feature_weights'])),
                perf['regime_transitions']))
    
    print()
    print("=" * 60)
    print("FINAL PERFORMANCE")
    print("=" * 60)
    
    perf = system.get_performance()
    print("Total learned: {}".format(perf['total_learned']))
    print("Correct: {}".format(perf['correct_learned']))
    print("Accuracy: {:.0%}".format(perf['accuracy']))
    print("Feature weights: {}".format(["{:.3f}".format(w) for w in perf['feature_weights'][:3]]))
    print("Bias: {:.4f}".format(perf['bias']))
    print("Regime transitions: {}".format(perf['regime_transitions']))
    print("Capital: ${:.2f}".format(perf['equity']))
    
    return perf


if __name__ == "__main__":
    run_backtest(200)