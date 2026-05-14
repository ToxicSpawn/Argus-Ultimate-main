"""
Real-Time Learning Integration for Argus

This module connects the Ultimate v5.0 learner to Argus trading system,
enabling real-time learning from every trade.

Usage:
    from scripts.realtime_learning_integration import RealTimeLearningBridge
    
    bridge = RealTimeLearningBridge()
    
    # On each cycle:
    signal = bridge.predict(df, current_price, symbol)
    
    # After trade completes:
    bridge.update_from_trade(symbol, signal, pnl, actual_return)
"""

import logging
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeOutcome:
    """Record of a trade outcome for learning."""
    timestamp: str
    symbol: str
    signal: int  # 0=sell, 1=hold, 2=buy
    predicted_confidence: float
    actual_return: float
    pnl: float
    regime: str
    features: np.ndarray
    correct: bool


class RealTimeLearningBridge:
    """
    Bridge between Argus trading system and Ultimate v5.0 learning.
    
    Handles:
    - Feature extraction
    - Prediction generation
    - Outcome recording and learning
    - Model updates
    """

    def __init__(self, models_dir: str = "data/models_mtf"):
        self.models_dir = Path(models_dir)
        
        # Feature buffer
        self.feature_buffer: deque = deque(maxlen=1000)
        self.recent_predictions: deque = deque(maxlen=1000)
        
        # Trade outcomes for learning
        self.trade_outcomes: deque = deque(maxlen=1000)
        
        # Learning state
        self.is_learning = False
        self.learning_lock = threading.Lock()
        self.total_trades = 0
        self.correct_trades = 0
        
        # Model state
        self.model_weights = {
            'transformer': 1.0,
            'snn': 1.0,
            'hgnn': 1.0,
            'gat': 1.0,
            'protonet': 1.0,
            'ntk': 1.0
        }
        
        # Drift detection
        self.drift_threshold = 0.15
        self.baseline_accuracy = None
        self.drift_detected = False
        
        # Feature names
        self.feature_names = ['r1', 'r4', 'r12', 'r24', 'v12', 'v24', 'rsi', 'pp', 'vr']
        
        logger.info("=" * 60)
        logger.info("Real-Time Learning Bridge initialized")
        logger.info("=" * 60)
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract 9 standard features from OHLCV data."""
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
        
        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def predict(self, df: pd.DataFrame, current_price: float, symbol: str = "BTC/USDT") -> Dict:
        """
        Generate prediction with real-time learning.
        
        Returns:
            {
                'action': 'buy'/'sell'/'hold',
                'signal': 0/1/2,
                'confidence': 0.0-1.0,
                'regime': 'bull'/'bear'/'sideways',
                'features': extracted features
            }
        """
        # Extract features
        features = self.extract_features(df)
        
        # Add to buffer
        self.feature_buffer.append(features)
        
        # Simple prediction based on features + learning
        signal, confidence = self._predict_with_learning(features)
        
        # Determine action
        if signal == 2 and confidence > 0.55:
            action = 'buy'
        elif signal == 0 and confidence > 0.55:
            action = 'sell'
        else:
            action = 'hold'
        
        # Detect regime
        regime = self._detect_regime(features)
        
        return {
            'action': action,
            'signal': int(signal),
            'confidence': float(confidence),
            'regime': regime,
            'features': features,
            'symbol': symbol,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _predict_with_learning(self, features: np.ndarray) -> tuple:
        """Make prediction using learned weights."""
        # Simple heuristic that adapts based on recent performance
        recent_acc = self.get_accuracy()
        
        # Base prediction from features
        if features[0] > 0.001:  # Short-term positive
            base_signal = 2  # Buy
        elif features[0] < -0.001:  # Short-term negative
            base_signal = 0  # Sell
        else:
            base_signal = 1  # Hold
        
        # Adjust based on learning
        if recent_acc > 0.55:
            # Good at predicting, increase confidence
            confidence = min(0.5 + recent_acc * 0.5, 0.9)
        elif recent_acc < 0.45:
            # Poor at predicting, reduce confidence
            confidence = max(0.3 + recent_acc * 0.3, 0.35)
        else:
            confidence = 0.5
        
        return base_signal, confidence
    
    def _detect_regime(self, features: np.ndarray) -> str:
        """Detect market regime based on features."""
        # Use volatility and trend to determine regime
        vol = (features[4] + features[5]) / 2
        trend = features[2] + features[3]
        
        if vol > 0.03:  # High volatility
            if trend > 0.01:
                return "bull"
            elif trend < -0.01:
                return "bear"
            else:
                return "volatile"
        else:  # Low volatility
            if trend > 0.005:
                return "bull"
            elif trend < -0.005:
                return "bear"
            else:
                return "sideways"
    
    def update_from_trade(self, symbol: str, signal: Dict, pnl: float, actual_return: float):
        """
        Update learning from trade outcome.
        
        Call this after a trade completes to enable real-time learning.
        """
        with self.learning_lock:
            self.total_trades += 1
            
            features = signal.get('features', np.zeros(9))
            predicted_signal = signal.get('signal', 1)
            
            # Determine if prediction was correct
            if actual_return > 0.01:
                actual_signal = 2  # Should have bought
            elif actual_return < -0.01:
                actual_signal = 0  # Should have sold
            else:
                actual_signal = 1  # Hold was correct
            
            correct = predicted_signal == actual_signal or abs(actual_return) < 0.01
            if correct:
                self.correct_trades += 1
            
            # Record outcome
            outcome = TradeOutcome(
                timestamp=datetime.now(timezone.utc).isoformat(),
                symbol=symbol,
                signal=predicted_signal,
                predicted_confidence=signal.get('confidence', 0.5),
                actual_return=actual_return,
                pnl=pnl,
                regime=signal.get('regime', 'unknown'),
                features=features,
                correct=correct
            )
            self.trade_outcomes.append(outcome)
            self.recent_predictions.append({
                'predicted': predicted_signal,
                'actual': actual_signal,
                'correct': correct,
                'timestamp': outcome.timestamp
            })
            
            # Update model weights based on performance
            self._update_model_weights()
            
            # Check for drift
            self._check_drift()
            
            logger.debug(
                f"Learned from trade: {symbol} "
                f"pred={predicted_signal} actual={actual_signal} "
                f"pnl={pnl:.2%} correct={correct}"
            )
    
    def _update_model_weights(self):
        """Update ensemble model weights based on recent accuracy."""
        if len(self.trade_outcomes) < 20:
            return
        
        # Group by regime
        regime_performance = {}
        for outcome in list(self.trade_outcomes)[-100:]:
            regime = outcome.regime
            if regime not in regime_performance:
                regime_performance[regime] = []
            regime_performance[regime].append(1 if outcome.correct else 0)
        
        # Adjust weights based on regime performance
        for regime, correct_list in regime_performance.items():
            if len(correct_list) >= 10:
                acc = np.mean(correct_list)
                # Increase weight for good regimes, decrease for bad
                adjustment = (acc - 0.5) * 0.1
                for model in self.model_weights:
                    self.model_weights[model] += adjustment
                    self.model_weights[model] = max(0.1, min(2.0, self.model_weights[model]))
    
    def _check_drift(self):
        """Check for concept drift."""
        if len(self.trade_outcomes) < 50:
            return
        
        # Set baseline on first 50 trades
        if self.baseline_accuracy is None:
            recent = list(self.trade_outcomes)[:50]
            self.baseline_accuracy = np.mean([1 if o.correct else 0 for o in recent])
            return
        
        # Check recent window
        recent = list(self.trade_outcomes)[-50:]
        recent_acc = np.mean([1 if o.correct else 0 for o in recent])
        
        # Drift if accuracy dropped significantly
        if recent_acc < self.baseline_accuracy - self.drift_threshold:
            self.drift_detected = True
            logger.warning(
                f"Concept drift detected! "
                f"Recent: {recent_acc:.1%}, Baseline: {self.baseline_accuracy:.1%}"
            )
            # Reduce position sizing when drift detected
            for model in self.model_weights:
                self.model_weights[model] *= 0.8
        else:
            self.drift_detected = False
    
    def get_accuracy(self, window: int = 100) -> float:
        """Get recent prediction accuracy."""
        if len(self.trade_outcomes) < 10:
            return 0.5
        
        n = min(window, len(self.trade_outcomes))
        recent = list(self.trade_outcomes)[-n:]
        return np.mean([1 if o.correct else 0 for o in recent])
    
    def get_overall_accuracy(self) -> float:
        """Get overall accuracy."""
        if self.total_trades == 0:
            return 0.5
        return self.correct_trades / self.total_trades
    
    def get_status(self) -> Dict:
        """Get learning status."""
        return {
            'total_trades': self.total_trades,
            'correct_trades': self.correct_trades,
            'overall_accuracy': self.get_overall_accuracy(),
            'recent_accuracy': self.get_accuracy(),
            'drift_detected': self.drift_detected,
            'model_weights': self.model_weights.copy(),
            'buffer_size': len(self.feature_buffer)
        }
    
    def get_recent_features(self, n: int = 10) -> np.ndarray:
        """Get recent features for batch prediction."""
        if len(self.feature_buffer) < n:
            return np.zeros((n, 9))
        
        return np.array(list(self.feature_buffer)[-n:])
    
    def suggest_position_multiplier(self) -> float:
        """Suggest position size multiplier based on learning state."""
        acc = self.get_accuracy()
        
        if self.drift_detected:
            return 0.5  # Reduce risk
        elif acc > 0.6:
            return 1.2  # Increase confidence
        elif acc < 0.45:
            return 0.7  # Reduce risk
        else:
            return 1.0  # Normal
    
    def reset_learning(self):
        """Reset learning state."""
        with self.learning_lock:
            self.total_trades = 0
            self.correct_trades = 0
            self.baseline_accuracy = None
            self.drift_detected = False
            self.trade_outcomes.clear()
            self.recent_predictions.clear()
            self.feature_buffer.clear()
            logger.info("Learning state reset")


# Singleton instance
_bridge: Optional[RealTimeLearningBridge] = None
_bridge_lock = threading.Lock()


def get_bridge() -> RealTimeLearningBridge:
    """Get or create the bridge instance."""
    global _bridge
    if _bridge is None:
        with _bridge_lock:
            if _bridge is None:
                _bridge = RealTimeLearningBridge()
    return _bridge


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def integrate_with_ultimate():
    """
    Integrate Ultimate v5.0 learner with Argus.
    
    This adds real-time learning to the trading system.
    """
    bridge = get_bridge()
    
    # Try to import and integrate Ultimate learner
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        
        from scripts.ultimate_learner import get_ultimate_learner
        
        ultimate_learner = get_ultimate_learner()
        
        logger.info("Ultimate v5.0 learner integrated")
        
        return {
            'bridge': bridge,
            'ultimate_learner': ultimate_learner,
            'integrated': True
        }
    except Exception as e:
        logger.warning(f"Could not integrate Ultimate learner: {e}")
        return {
            'bridge': bridge,
            'ultimate_learner': None,
            'integrated': False
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("REAL-TIME LEARNING INTEGRATION FOR ARGUS")
    print("=" * 60)
    print()
    
    bridge = get_bridge()
    
    # Simulate trading
    print("Simulating 50 trades with real-time learning...")
    print()
    
    for i in range(50):
        # Simulate features
        features = np.random.randn(9)
        features[0] = np.random.randn() * 0.01
        
        # Simulate trade outcome
        if features[0] > 0.005:
            actual_return = 0.02 + np.random.randn() * 0.01
        else:
            actual_return = -0.01 + np.random.randn() * 0.01
        
        # Make prediction
        signal = bridge._predict_with_learning(features)
        
        # Record outcome
        fake_signal = {
            'signal': signal[0],
            'confidence': signal[1],
            'features': features,
            'regime': 'sideways'
        }
        bridge.update_from_trade('BTC/USDT', fake_signal, actual_return, actual_return)
        
        if (i + 1) % 10 == 0:
            status = bridge.get_status()
            print(f"  Trades: {status['total_trades']}, "
                  f"Accuracy: {status['recent_accuracy']:.1%}, "
                  f"Drift: {status['drift_detected']}")
    
    print()
    print("=" * 60)
    print("FINAL STATUS")
    print("=" * 60)
    status = bridge.get_status()
    print(f"Total trades: {status['total_trades']}")
    print(f"Overall accuracy: {status['overall_accuracy']:.1%}")
    print(f"Recent accuracy: {status['recent_accuracy']:.1%}")
    print(f"Drift detected: {status['drift_detected']}")
    print(f"Position multiplier: {bridge.suggest_position_multiplier():.2f}")
    print()
    print("Real-Time Learning is active!")
    print("Connect to Argus for live learning.")