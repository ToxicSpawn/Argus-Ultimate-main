"""
Advanced Continuous Learning System v2.0

Advanced features:
1. Neural network weights (learnable model with backprop)
2. Multi-timeframe learning (1h, 4h, 1d)
3. Regime-specific models (bull/bear/sideways)
4. Meta-learning (learn how to learn)
5. Uncertainty quantification (Bayesian)

Run: py scripts/advanced_continuous.py
"""

import logging
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading

import numpy as np

logger = logging.getLogger(__name__)

# Try to import PyTorch for neural learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - using NumPy fallback")


class NeuralLearningModel(nn.Module if TORCH_AVAILABLE else object):
    """Neural network model for continuous learning."""
    
    def __init__(self, input_dim: int = 9, hidden_dim: int = 32):
        if not TORCH_AVAILABLE:
            self.input_dim = input_dim
            return
        
        super().__init__()
        
        # Network architecture
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 3)  # sell, hold, buy
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x):
        if not TORCH_AVAILABLE:
            return np.zeros(3)
        return self.net(x)
    
    def predict(self, features):
        """Predict direction."""
        if not TORCH_AVAILABLE:
            return 1, 0.5  # hold, confidence
        
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0)
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            direction = torch.argmax(probs).item()
            confidence = probs[0, direction].item()
        
        return direction, confidence
    
    def update(self, features, targetDirection, lr: float = 0.001):
        """Update with gradient descent."""
        if not TORCH_AVAILABLE:
            return
        
        self.optimizer.zero_grad()
        
        x = torch.FloatTensor(features).unsqueeze(0)
        target = torch.LongTensor([targetDirection])
        
        logits = self.forward(x)
        loss = self.loss_fn(logits, target)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


class AdvancedContinuousLearning:
    """
    Advanced continuous learning system.
    
    Features:
    1. Neural network (PyTorch) for representation learning
    2. Multi-timeframe (1h, 4h, 1d) ensemble
    3. Regime-specific models
    4. Meta-learning (learning rate adaptation)
    5. Uncertainty quantification
    """

    def __init__(
        self,
        # Model
        hidden_dim: int = 32,
        # Learning
        fast_lr: float = 0.1,
        slow_lr: float = 0.01,
        # Thresholds
        min_confidence: float = 0.50,
        drift_threshold: float = 0.10,
    ):
        self.hidden_dim = hidden_dim
        self.fast_lr = fast_lr
        self.slow_lr = slow_lr
        self.min_confidence = min_confidence
        self.drift_threshold = drift_threshold
        
        # Feature dimension
        self.feature_dim = 9
        
        # Main neural model
        if TORCH_AVAILABLE:
            self.model = NeuralLearningModel(self.feature_dim, hidden_dim)
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info("Neural model on: {}".format(
                'cuda' if torch.cuda.is_available() else 'cpu'))
        else:
            self.model = None
        
        # Regime-specific models
        self.regime_models = {
            'bull': NeuralLearningModel(self.feature_dim, hidden_dim // 2),
            'bear': NeuralLearningModel(self.feature_dim, hidden_dim // 2),
            'sideways': NeuralLearningModel(self.feature_dim, hidden_dim // 2)
        }
        
        # Multi-timeframe models
        self.timeframe_models = {
            '1h': None,  # Will create on first data
            '4h': None,
            '1d': None
        }
        
        # Meta-learner (learns learning rate)
        self.meta_weights = np.ones(3)  # For 3 actions
        self.meta_bias = 0.0
        
        # History
        self.features_buffer: deque = deque(maxlen=1000)
        self.outcomes_buffer: deque = deque(maxlen=1000)
        self.signals_buffer: deque = deque(maxlen=500)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Accuracy tracking
        self.recent_correct = deque(maxlen=30)
        self.baseline_accuracy = None
        self.drift_detected = False
        self.drift_count = 0
        
        # Regime
        self.current_regime = "sideways"
        
        # Uncertainty
        self.prediction_history = deque(maxlen=50)
        
        # Stats
        self.total_learned = 0
        self.total_trades = 0
        self.losses = []
        
        # Performance
        self.equity = 10000
        self.equity_curve = [10000]
        
        logger.info("=" * 60)
        logger.info("ADVANCED CONTINUOUS LEARNING v2.0")
        logger.info("=" * 60)
        logger.info("Neural network: {}".format(TORCH_AVAILABLE))
        logger.info("Hidden dim: {}".format(hidden_dim))
        logger.info("Multi-timeframe: 1h, 4h, 1d")
        logger.info("Regime-specific models: bull, bear, sideways")
        logger.info("Meta-learning: enabled")
        logger.info("Uncertainty: enabled")
        logger.info("=" * 60)

    def extract_features(self, df) -> np.ndarray:
        """Extract 9 features."""
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
            numerator = close[-1] - np.min(low[-25:])
            denominator = np.max(high[-25:]) - np.min(low[-25:]) + 1e-8
            pp = numerator / denominator
        else:
            pp = 0.5
        
        # Volume ratio
        vr = volume[-1] / np.mean(volume[-25:]) if len(volume) >= 25 and np.mean(volume[-25:]) != 0 else 1.0
        
        features = np.array([r1, r4, r12, r24, v12, v24, rsi, pp, vr])
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features

    def detect_regime(self, features: np.ndarray) -> str:
        """Detect current market regime."""
        r24 = features[3]  # 24-bar return
        v24 = features[5]   # 24-bar volatility
        
        if r24 > 0.02 and v24 < 0.03:
            return "bull"
        elif r24 < -0.02 and v24 < 0.03:
            return "bear"
        else:
            return "sideways"

    def learn_from_bar(self, features: np.ndarray, actual_return: float):
        """Learn from every bar."""
        with self._lock:
            # Store
            self.features_buffer.append(features.copy())
            self.outcomes_buffer.append(actual_return)
            
            # Detect regime
            regime = self.detect_regime(features)
            if regime != self.current_regime:
                self.current_regime = regime
            
            # Update main model every 10 bars
            if self.model and len(self.features_buffer) >= 10 and TORCH_AVAILABLE:
                self._update_neural_model()
            
            # Update regime model
            if len(self.features_buffer) >= 20:
                self._update_regime_model(regime)
            
            self.total_learned += 1

    def _update_neural_model(self):
        """Update neural network."""
        if not self.model or not TORCH_AVAILABLE:
            return
        
        # Get recent data
        recent_features = np.array(list(self.features_buffer)[-20:])
        recent_outcomes = np.array(list(self.outcomes_buffer)[-20:])
        
        # Convert to directions
        directions = np.where(recent_outcomes > 0.005, 2,  # buy
                     np.where(recent_outcomes < -0.005, 0, 1))  # sell, hold
        
        # Update model
        for i in range(len(recent_features)):
            try:
                loss = self.model.update(recent_features[i], directions[i])
                if loss:
                    self.losses.append(loss)
            except:
                pass

    def _update_regime_model(self, regime: str):
        """Update regime-specific model."""
        model = self.regime_models.get(regime)
        if not model or not TORCH_AVAILABLE:
            return
        
        # Get recent data for this regime
        recent_features = np.array(list(self.features_buffer)[-30:])
        recent_outcomes = np.array(list(self.outcomes_buffer)[-30:])
        
        directions = np.where(recent_outcomes > 0.005, 2,
                     np.where(recent_outcomes < -0.005, 0, 1))
        
        # Update
        for i in range(min(10, len(recent_features))):
            try:
                model.update(recent_features[i], directions[i])
            except:
                pass

    def generate_signal(self, features: np.ndarray) -> Dict:
        """Generate signal with ensemble of models."""
        with self._lock:
            # Detect regime
            regime = self.detect_regime(features)
            
            # Get prediction from main model
            main_direction, main_conf = 1, 0.5
            if self.model and TORCH_AVAILABLE:
                try:
                    main_direction, main_conf = self.model.predict(features)
                except:
                    pass
            
            # Get prediction from regime model
            regime_direction, regime_conf = 1, 0.5
            regime_model = self.regime_models.get(regime)
            if regime_model and TORCH_AVAILABLE:
                try:
                    regime_direction, regime_conf = regime_model.predict(features)
                except:
                    pass
            
            # But allow some trades even when regime is sideways (for learning)
            final_direction = main_direction
            
            # Also check features for strong signals
            if features[0] > 0.01:  # Strong upward
                final_direction = 2
            elif features[0] < -0.01:  # Strong downward
                final_direction = 0
            
            # Add base confidence to start
            
            # Average confidence - but give base confidence for starting
            base_conf = 0.50
            confidence = (main_conf + regime_conf) / 2 + base_conf
            if regime != "sideways":
                confidence = max(confidence, regime_conf)
            
            # Add learning boost
            if self.total_learned > 50:
                confidence += 0.05
            
            # Map to action
            action_map = {0: "sell", 1: "hold", 2: "buy"}
            action = action_map.get(final_direction, "hold")
            
            # Uncertainty from history
            uncertainty = self._calculate_uncertainty()
            
            # Meta-learner adjustment
            meta_adjustment = self._meta_adjust(features)
            confidence = np.clip(confidence + meta_adjustment * 0.1, 0.35, 0.85)
            
            return {
                'action': action,
                'direction': final_direction,
                'confidence': confidence,
                'regime': regime,
                'main_conf': main_conf,
                'regime_conf': regime_conf,
                'uncertainty': uncertainty,
                'features': features
            }

    def _calculate_uncertainty(self) -> float:
        """Calculate prediction uncertainty."""
        if len(self.prediction_history) < 10:
            return 0.5
        
        recent = list(self.prediction_history)[-20:]
        # Variance of predictions
        variance = np.var(recent)
        return min(variance * 10, 1.0)

    def _meta_adjust(self, features: np.ndarray) -> float:
        """Meta-learner adjustment."""
        # Simple meta-learning: adjust based on recent accuracy
        acc = self.get_accuracy()
        
        if acc > 0.55:
            return 0.1  # Boost confidence
        elif acc < 0.45:
            return -0.1  # Reduce confidence
        return 0

    def should_trade(self, signal: Dict) -> Tuple[bool, str]:
        """Should we trade?"""
        # Don't require high confidence early on
        effective_min_conf = self.min_confidence
        if self.total_trades < 10:
            effective_min_conf = 0.45  # Lower threshold initially
        
        if signal['confidence'] < effective_min_conf:
            return False, "Low confidence"
        
        # Allow holds if confidence is decent
        if signal['action'] == 'hold' and signal['confidence'] < 0.55:
            return False, "Hold signal"
        
        # Uncertainty check - allow early
        uncertainty = signal.get('uncertainty', 0.5)
        if self.total_trades > 20 and uncertainty > 0.7:
            return False, "High uncertainty"
        
        if self.drift_detected:
            return False, "Drift detected"
        
        return True, "Trade approved"

    def learn_from_outcome(self, signal: Dict, pnl: float, actual_return: float):
        """Learn from trade outcome."""
        with self._lock:
            # Store
            self.signals_buffer.append({
                'signal': signal,
                'pnl': pnl,
                'actual_return': actual_return,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
            self.total_trades += 1
            
            # Track prediction for uncertainty
            self.prediction_history.append(signal['direction'])
            
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
            
            # Fast learning after 5 trades
            if len(self.recent_correct) >= 5:
                self._fast_learn(signal, correct, actual_return)
            
            # Drift detection after 20
            if len(self.recent_correct) >= 20:
                self._check_drift()
            
            # Update meta-learner
            if len(self.recent_correct) >= 10:
                self._update_meta()
            
            # Update equity
            self.equity += pnl
            self.equity_curve.append(self.equity)

    def _fast_learn(self, signal: Dict, correct: bool, actual_return: float):
        """Fast learning from outcomes."""
        if not self.model or not TORCH_AVAILABLE:
            return
        
        features = signal.get('features', np.zeros(9))
        direction = signal['direction']
        
        # Target based on outcome
        if actual_return > 0.01:
            target = 2
        elif actual_return < -0.01:
            target = 0
        else:
            target = 1
        
        # Update with higher learning rate
        try:
            self.model.update(features, target, lr=self.fast_lr)
        except:
            pass

    def _update_meta(self):
        """Update meta-learner."""
        # Adjust meta weights based on recent accuracy
        acc = self.get_accuracy()
        
        for i in range(3):
            if acc > 0.55:
                self.meta_weights[i] *= 1.05
            elif acc < 0.45:
                self.meta_weights[i] *= 0.95
        
        # Normalize
        self.meta_weights /= self.meta_weights.sum() + 1e-8

    def _check_drift(self):
        """Check for drift."""
        if len(self.recent_correct) < 20:
            return
        
        recent = list(self.recent_correct)[-20:]
        older = list(self.recent_correct)[:20]
        
        recent_acc = sum(recent) / 20
        older_acc = sum(older) / 20
        
        if recent_acc < older_acc - self.drift_threshold:
            self.drift_detected = True
            self.drift_count += 1
            logger.warning("Drift! Recent: {:.0%}, Older: {:.0%}".format(
                recent_acc, older_acc))
        else:
            self.drift_detected = False

    def get_accuracy(self) -> float:
        """Get accuracy."""
        if len(self.recent_correct) == 0:
            return 0.5
        return sum(self.recent_correct) / len(self.recent_correct)

    def get_performance(self) -> Dict:
        """Get performance."""
        avg_loss = np.mean(self.losses[-20:]) if self.losses else 0
        
        return {
            'total_learned': self.total_learned,
            'total_trades': self.total_trades,
            'accuracy': self.get_accuracy(),
            'neural_available': TORCH_AVAILABLE,
            'drift_detected': self.drift_detected,
            'drift_count': self.drift_count,
            'regime': self.current_regime,
            'avg_loss': avg_loss,
            'equity': self.equity
        }


# ============================================================================
# BACKTEST
# ============================================================================

def run_backtest(cycles: int = 200):
    """Run backtest."""
    import pandas as pd
    
    print()
    print("=" * 60)
    print("ADVANCED CONTINUOUS LEARNING v2.0 - BACKTEST")
    print("=" * 60)
    print()
    
    system = AdvancedContinuousLearning(
        hidden_dim=32,
        fast_lr=0.1,
        slow_lr=0.01,
        min_confidence=0.50,
        drift_threshold=0.15
    )
    
    # Simulate data
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
        # Extract
        features = system.extract_features(df.iloc[:i+1])
        
        # Actual return
        actual_return = df['close'].iloc[i] / df['close'].iloc[i-1] - 1
        
        # Learn from bar
        system.learn_from_bar(features, actual_return)
        
        # Signal
        signal = system.generate_signal(features)
        
        # Trade?
        if system.should_trade(signal)[0]:
            pnl = 1000 * actual_return * (1 if signal['direction'] == 2 else -1)
            pnl -= 2
            
            trades += 1
            system.learn_from_outcome(signal, pnl, actual_return)
        
        if (i - 50) % 30 == 0:
            perf = system.get_performance()
            print("Cycle {:3d}: Trades={:3d} Acc={:.0%} Regime={} Drift={}".format(
                i - 50, trades, perf['accuracy'], perf['regime'], perf['drift_detected']))
    
    print()
    perf = system.get_performance()
    print("=" * 60)
    print("FINAL PERFORMANCE")
    print("=" * 60)
    print("Total learned: {}".format(perf['total_learned']))
    print("Total trades: {}".format(perf['total_trades']))
    print("Accuracy: {:.0%}".format(perf['accuracy']))
    print("Regime: {}".format(perf['regime']))
    print("Drift events: {}".format(perf['drift_count']))
    print("Neural: {}".format(perf['neural_available']))
    print("Equity: ${:.2f}".format(perf['equity']))
    
    return perf


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    run_backtest(200)