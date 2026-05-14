"""
Continuous Learning Integration for Argus

This module adds TRUE continuous learning to Argus:
- Learns from EVERY bar (not just trades)
- Real-time weight updates
- Drift detection
- Regime adaptation

Usage in Argus:
    from scripts.continuous_argus import ContinuousArgusLearning
    
    # Initialize once
    learner = ContinuousArgusLearning()
    
    # In on_bar() every cycle:
    features = learner.extract_features(df)
    actual_return = df['close'].iloc[-1] / df['close'].iloc[-2] - 1
    
    # Learn from bar - ALWAYS
    learner.learn_from_bar(features, actual_return)
    
    # Get signal
    signal = learner.generate_signal(features, regime)
    
    # Should trade?
    if signal['confidence'] >= 0.55 and signal['action'] != 'hold':
        # ... execute trade
    
    # After trade closes:
    learner.learn_from_outcome(signal, pnl, actual_return)
"""

import logging
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
import threading

import numpy as np

logger = logging.getLogger(__name__)


class ContinuousArgusLearning:
    """
    Continuous learning for live Argus trading.
    
    Key features:
    1. Learns from EVERY bar
    2. Real-time weight updates
    3. Drift detection
    4. Regime tracking
    5. Thread-safe
    
    Learning rates:
    - Fast: 10% per outcome
    - Slow: 1% drift
    - Correlation update: every 10 bars
    """

    def __init__(
        self,
        fast_lr: float = 0.10,
        slow_lr: float = 0.01,
        min_confidence: float = 0.50,
        drift_threshold: float = 0.10,
        feature_dim: int = 9,
    ):
        # Learning config
        self.fast_lr = fast_lr
        self.slow_lr = slow_lr
        self.min_confidence = min_confidence
        self.drift_threshold = drift_threshold
        
        # Model weights
        self.feature_weights = np.zeros(feature_dim)
        self.feature_bias = 0.0
        
        # History
        self.features_buffer: deque = deque(maxlen=500)
        self.outcomes_buffer: deque = deque(maxlen=500)
        self.signals_buffer: deque = deque(maxlen=200)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Accuracy tracking
        self.recent_correct = deque(maxlen=20)
        self.baseline_accuracy = None
        self.drift_detected = False
        self.drift_count = 0
        
        # Regime
        self.current_regime = "sideways"
        
        # Stats
        self.total_learned = 0
        self.total_trades = 0
        
        # Performance
        self.equity = 10000
        self.equity_curve = [10000]
        
        logger.info("Continuous Argus Learning initialized")
        logger.info("  Fast LR: {:.0%}".format(fast_lr))
        logger.info("  Slow LR: {:.0%}".format(slow_lr))
        logger.info("  Min confidence: {:.0%}".format(min_confidence))
        logger.info("  Drift threshold: {:.0%}".format(drift_threshold))

    def extract_features(self, df) -> np.ndarray:
        """Extract 9 features from OHLCV data."""
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
        if len(low) >= 25:
            pp = (close[-1] - np.min(low[-25:])) / (np.max(high[-25:]) - np.min(low[-25:]) + 1e-8)
        else:
            pp = 0.5
        
        # Volume ratio
        vr = volume[-1] / np.mean(volume[-25:]) if len(volume) >= 25 and np.mean(volume[-25:]) != 0 else 1.0
        
        features = np.array([r1, r4, r12, r24, v12, v24, rsi, pp, vr])
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features

    def learn_from_bar(self, features: np.ndarray, actual_return: float):
        """
        Learn from EVERY bar - this is continuous learning.
        
        This updates the model in real-time based on price movements.
        """
        with self._lock:
            # Store
            self.features_buffer.append(features.copy())
            self.outcomes_buffer.append(actual_return)
            
            # Update correlation tracker every 10 bars
            if len(self.features_buffer) >= 10:
                self._update_correlation()
            
            # Slow drift update every 50 bars
            if len(self.outcomes_buffer) >= 50:
                self._slow_drift()
            
            self.total_learned += 1

    def _update_correlation(self):
        """Update feature correlations."""
        recent_features = np.array(list(self.features_buffer)[-50:])
        recent_outcomes = np.array(list(self.outcomes_buffer)[-50:])
        
        # Calculate correlation for each feature
        for i in range(len(self.feature_weights)):
            if abs(recent_features[:, i]).sum() > 0.001:
                try:
                    corr = np.corrcoef(recent_features[:, i], recent_outcomes)[0, 1]
                    if not np.isnan(corr) and abs(corr) > 0.1:
                        # Adjust weight toward correlation
                        self.feature_weights[i] += self.slow_lr * corr * np.sign(corr)
                except:
                    pass

    def _slow_drift(self):
        """Slow drift adjustment."""
        recent = list(self.outcomes_buffer)[-20:]
        if recent:
            avg = np.mean(recent)
            self.feature_bias += self.slow_lr * avg

    def generate_signal(self, features: np.ndarray, regime: str = "sideways") -> Dict:
        """Generate signal using learned model."""
        with self._lock:
            # Calculate score
            score = np.dot(features, self.feature_weights) + self.feature_bias
            
            # Map to action
            if score > 0.005:
                action = "buy"
                direction = 2
            elif score < -0.005:
                action = "sell"
                direction = 0
            else:
                action = "hold"
                direction = 1
            
            # Confidence from weight magnitude
            weight_mag = np.mean(np.abs(self.feature_weights))
            accuracy = self.get_accuracy()
            
            confidence = 0.5 + weight_mag + (accuracy - 0.5) * 0.2
            confidence = np.clip(confidence, 0.35, 0.85)
            
            # Regime
            if regime != self.current_regime and regime != "sideways":
                self.current_regime = regime
            
            return {
                'action': action,
                'direction': direction,
                'confidence': confidence,
                'regime': regime,
                'score': score,
                'features': features,
                'weight_magnitude': weight_mag
            }

    def should_trade(self, signal: Dict) -> tuple:
        """Check if should trade."""
        if signal['confidence'] < self.min_confidence:
            return False, "Low confidence"
        
        if signal['action'] == 'hold':
            return False, "Hold signal"
        
        if self.drift_detected:
            return False, "Drift detected"
        
        return True, "Trade approved"

    def learn_from_outcome(self, signal: Dict, pnl: float, actual_return: float):
        """Learn强化 from trade outcome."""
        with self._lock:
            # Store
            self.signals_buffer.append({
                'signal': signal,
                'pnl': pnl,
                'actual_return': actual_return,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
            self.total_trades += 1
            
            # Correct?
            direction = signal['direction']
            if direction == 2 and actual_return > 0.01:
                correct = True
            elif direction == 0 and actual_return < -0.01:
                correct = True
            elif direction == 1 and abs(actual_return) <= 0.01:
                correct = True
            else:
                correct = False
            
            self.recent_correct.append(1 if correct else 0)
            
            # Fast learning after just 3 outcomes
            if len(self.recent_correct) >= 3:
                self._fast_learn(signal, correct, actual_return)
            
            # Drift detection after 20
            if len(self.recent_correct) >= 20:
                self._check_drift()
            
            # Update equity
            self.equity += pnl
            self.equity_curve.append(self.equity)

    def _fast_learn(self, signal: Dict, correct: bool, actual_return: float):
        """Fast learning from outcomes."""
        features = signal.get('features', np.zeros(9))
        
        # Reinforce or anti-reinforce
        delta = self.fast_lr if correct else -self.fast_lr
        dir_mult = 1 if signal['direction'] == 2 else -1 if signal['direction'] == 0 else 0
        
        # Update weights
        self.feature_weights += delta * dir_mult * features * 0.5
        self.feature_bias += delta * actual_return * 0.1

    def _check_drift(self):
        """Check for regime drift."""
        if len(self.recent_correct) < 20:
            return
        
        recent = list(self.recent_correct)[-20:]
        older = list(self.recent_correct)[:20] if len(self.recent_correct) >= 40 else list(self.recent_correct)
        
        recent_acc = sum(recent) / 20
        older_acc = sum(older) / len(older) if older else 0.5
        
        if recent_acc < older_acc - self.drift_threshold:
            self.drift_detected = True
            self.drift_count += 1
            # Halve weights on drift
            self.feature_weights *= 0.5
            logger.warning("Drift! Recent: {:.0%}, Older: {:.0%}".format(recent_acc, older_acc))
        else:
            self.drift_detected = False
            
        if recent_acc > older_acc + self.drift_threshold:
            self.feature_weights *= 1.2
            logger.info("Improving! Boosting.")

    def get_accuracy(self) -> float:
        """Get accuracy."""
        if len(self.recent_correct) == 0:
            return 0.5
        return sum(self.recent_correct) / len(self.recent_correct)

    def get_performance(self) -> Dict:
        """Get performance."""
        return {
            'total_learned': self.total_learned,
            'total_trades': self.total_trades,
            'accuracy': self.get_accuracy(),
            'weights': self.feature_weights.tolist(),
            'bias': self.feature_bias,
            'drift_detected': self.drift_detected,
            'drift_count': self.drift_count,
            'equity': self.equity
        }


# ============================================================================
# INSTANTIATE FOR ARGUS
# ============================================================================

# Global instance - import and use
_argus_learner = None


def get_argus_learner() -> ContinuousArgusLearning:
    """Get the global Argus learner instance."""
    global _argus_learner
    if _argus_learner is None:
        _argus_learner = ContinuousArgusLearning()
        logger.info("Argus learner initialized")
    return _argus_learner


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import pandas as pd
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("CONTINUOUS ARGUS LEARNING TEST")
    print("=" * 60)
    print()
    
    learner = get_argus_learner()
    
    # Simulate data
    np.random.seed(42)
    cycles = 150
    prices = 50000 + np.cumsum(np.random.randn(cycles + 30) * 100)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(cycles + 30) * 50),
        'low': prices - np.abs(np.random.randn(cycles + 30) * 50),
        'close': prices,
        'volume': np.abs(np.random.randn(cycles + 30)) * 1000 + 500
    })
    
    trades = 0
    for i in range(30, cycles + 30):
        # Extract features
        features = learner.extract_features(df.iloc[:i+1])
        
        # Actual return
        actual_return = df['close'].iloc[i] / df['close'].iloc[i-1] - 1
        
        # LEARN FROM EVERY BAR!
        learner.learn_from_bar(features, actual_return)
        
        # Generate signal
        regime = "bull" if features[2] > 0.1 else "bear" if features[2] < -0.1 else "sideways"
        signal = learner.generate_signal(features, regime)
        
        # Trade?
        if learner.should_trade(signal)[0]:
            # Simulate trade
            pnl = 1000 * actual_return * (1 if signal['direction'] == 2 else -1)
            pnl -= 2  # Fees
            
            trades += 1
            learner.learn_from_outcome(signal, pnl, actual_return)
        
        if (i - 30) % 30 == 0:
            perf = learner.get_performance()
            w = np.mean(np.abs(perf['weights']))
            print("Cycle {:3d}: Trades={:3d} Acc={:.0%} W={:.2f}".format(
                i - 30, trades, perf['accuracy'], w))
    
    print()
    perf = learner.get_performance()
    print("Total learned (bars): {}".format(perf['total_learned']))
    print("Total trades: {}".format(perf['total_trades']))
    print("Accuracy: {:.0%}".format(perf['accuracy']))
    print("Drift events: {}".format(perf['drift_count']))
    print("Equity: ${:.2f}".format(perf['equity']))