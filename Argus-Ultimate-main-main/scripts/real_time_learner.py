"""
Real-Time Learning System for Argus

Continuously:
1. Collects new market data
2. Detects concept drift
3. Updates model weights
4. Retrains models incrementally
5. Adapts to new patterns

Run alongside: py main.py paper
"""

import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


class RealTimeLearner:
    """
    Real-time learning system that continuously updates models
    based on new market data and trade outcomes.
    """
    
    def __init__(self, models_dir: str = "data/models_mtf", history_file: str = "data/real_time_history.json"):
        self.models_dir = Path(models_dir)
        self.history_file = history_file
        
        # Real-time models (lightweight, fast updates)
        self.signal_model = None
        self.regime_model = None
        
        # Scaler for real-time data
        self.scaler = StandardScaler()
        
        # Performance tracking
        self.recent_predictions = []  # [(prediction, actual, timestamp), ...]
        self.drift_detected = False
        self.performance_history = []
        
        # Initialize
        self._initialize()
        
    def _initialize(self):
        """Initialize real-time models."""
        logger.info("Initializing real-time learning system...")
        
        # Create incremental learning models
        self.signal_model = SGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=0.0001,
            learning_rate='adaptive',
            eta0=0.01,
            random_state=42,
            warm_start=True
        )
        
        self.regime_model = SGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=0.0001,
            learning_rate='adaptive',
            eta0=0.01,
            random_state=42,
            warm_start=True
        )
        
        # Fit scaler with dummy data first
        dummy_X = np.random.randn(100, 9)
        self.scaler.fit(dummy_X)
        
        # Initialize models with dummy data
        dummy_labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]) % 3
        self.signal_model.partial_fit(dummy_X[:10], dummy_labels, classes=[0, 1, 2])
        self.regime_model.partial_fit(dummy_X[:10], dummy_labels, classes=[0, 1, 2])
        
        # Try to load historical data for better initialization
        self._load_historical_data()
        
        logger.info("Real-time learning initialized")
        
    def _load_historical_data(self):
        """Load historical data for initial model."""
        try:
            with open('data/historical/historical_data.pkl', 'rb') as f:
                data = pickle.load(f)
            
            # Process first symbol
            symbol = list(data.keys())[0]
            df = pd.DataFrame(data[symbol]['1h'])
            
            # Create features
            f = pd.DataFrame()
            f['r1'] = df['close'].pct_change(1)
            f['r4'] = df['close'].pct_change(4)
            f['r12'] = df['close'].pct_change(12)
            f['r24'] = df['close'].pct_change(24)
            f['v12'] = f['r1'].rolling(12).std()
            f['v24'] = f['r1'].rolling(24).std()
            
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            f['rsi'] = 100 - (100 / (1 + gain / loss.clip(lower=1e-8)))
            
            f['pp'] = (df['close'] - df['low'].rolling(24).min()) / (df['high'].rolling(24).max() - df['low'].rolling(24).min()).clip(lower=1e-8)
            f['vr'] = df['volume'] / df['volume'].rolling(24).mean().clip(lower=1e-8)
            
            f = f.dropna()
            
            # Labels
            fwd = df['close'].pct_change(4).shift(-4)
            fwd24 = df['close'].pct_change(24).shift(-24)
            
            y_signal = pd.cut(fwd.loc[f.index], bins=[-np.inf, -0.01, 0.01, np.inf], labels=[0, 1, 2])
            y_regime = pd.cut(fwd24.loc[f.index], bins=[-np.inf, -0.03, 0.03, np.inf], labels=[0, 1, 2])
            
            X = f.values
            y_s = y_signal.values.astype(int)
            y_r = y_regime.values.astype(int)
            
            # Remove NaN labels
            valid = ~np.isnan(y_s) & ~np.isnan(y_r)
            X = X[valid]
            y_s = y_s[valid]
            y_r = y_r[valid]
            
            # Fit scaler
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Initial fit
            self.signal_model.partial_fit(X_scaled[:100], y_s[:100], classes=[0, 1, 2])
            self.regime_model.partial_fit(X_scaled[:100], y_r[:100], classes=[0, 1, 2])
            
            logger.info(f"Loaded {len(X)} historical samples for real-time learning")
            
        except Exception as e:
            logger.warning(f"Could not load historical data: {e}")
    
    def update(self, features: np.ndarray, actual_return: float = None):
        """
        Update models with new data.
        
        Args:
            features: Feature vector (9 features)
            actual_return: Actual return after prediction resolves
        """
        # Scale features
        X = self.scaler.transform(features.reshape(1, -1))
        
        # Get prediction
        signal_pred = self.signal_model.predict(X)[0]
        
        # If we have actual outcome, update models
        if actual_return is not None:
            # Determine actual label
            if actual_return > 0.01:
                actual_signal = 2  # buy correct
            elif actual_return < -0.01:
                actual_signal = 0  # sell correct
            else:
                actual_signal = 1  # hold
            
            # Partial fit (incremental update)
            self.signal_model.partial_fit(X, [actual_signal])
            
            # Track performance
            correct = int(signal_pred == actual_signal)
            self.recent_predictions.append({
                'prediction': int(signal_pred),
                'actual': int(actual_signal),
                'correct': correct,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
            # Keep last 1000 predictions
            if len(self.recent_predictions) > 1000:
                self.recent_predictions = self.recent_predictions[-1000:]
            
            # Check for drift
            self._check_drift()
        
        return signal_pred
    
    def _check_drift(self):
        """Check for concept drift."""
        if len(self.recent_predictions) < 50:
            return
        
        # Compare recent accuracy to overall
        recent = self.recent_predictions[-50:]
        overall = self.recent_predictions[-200:]
        
        recent_acc = np.mean([p['correct'] for p in recent])
        overall_acc = np.mean([p['correct'] for p in overall])
        
        # If recent accuracy dropped significantly, flag drift
        if recent_acc < overall_acc - 0.1:
            self.drift_detected = True
            logger.warning(f"Concept drift detected! Recent: {recent_acc:.1%}, Overall: {overall_acc:.1%}")
            
            # Trigger retraining
            self._trigger_retrain()
        else:
            self.drift_detected = False
    
    def _trigger_retrain(self):
        """Trigger model retraining."""
        logger.info("Triggering model retraining due to drift...")
        
        # In a full implementation, this would:
        # 1. Collect recent training data
        # 2. Retrain full models
        # 3. Update model files
        # 4. Reset incremental models
        
        logger.info("Retraining triggered - would retrain full models here")
    
    def get_performance(self) -> dict:
        """Get current performance metrics."""
        if not self.recent_predictions:
            return {'accuracy': 0.0, 'drift_detected': False}
        
        recent = self.recent_predictions[-100:]
        accuracy = np.mean([p['correct'] for p in recent])
        
        return {
            'accuracy': float(accuracy),
            'drift_detected': self.drift_detected,
            'total_updates': len(self.recent_predictions),
            'last_update': self.recent_predictions[-1]['timestamp'] if self.recent_predictions else None
        }
    
    def predict(self, features: np.ndarray) -> dict:
        """Make a prediction with current models."""
        X = self.scaler.transform(features.reshape(1, -1))
        
        signal = int(self.signal_model.predict(X)[0])
        regime = int(self.regime_model.predict(X)[0])
        
        # Get probabilities
        try:
            signal_proba = self.signal_model.predict_proba(X)[0]
            signal_conf = float(np.max(signal_proba))
        except:
            signal_conf = 0.5
        
        return {
            'signal': signal,
            'regime': regime,
            'confidence': signal_conf,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


# Global instance
_learner = None


def get_real_time_learner() -> RealTimeLearner:
    """Get or create the real-time learner."""
    global _learner
    if _learner is None:
        _learner = RealTimeLearner()
    return _learner


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("REAL-TIME LEARNING SYSTEM")
    logger.info("=" * 60)
    
    # Initialize
    learner = get_real_time_learner()
    
    # Simulate continuous learning
    logger.info("Starting continuous learning loop...")
    
    for i in range(10):
        # Generate random features (simulating new data)
        features = np.random.randn(9)
        
        # Get prediction
        pred = learner.predict(features)
        
        # Simulate actual return
        actual_return = np.random.randn() * 0.02
        
        # Update with actual
        learner.update(features, actual_return)
        
        # Print performance every iteration
        perf = learner.get_performance()
        logger.info(f"Iter {i+1}: signal={pred['signal']}, drift={perf['drift_detected']}, acc={perf['accuracy']:.1%}")
        
        time.sleep(0.1)
    
    # Final performance
    logger.info("=" * 60)
    logger.info("FINAL PERFORMANCE")
    logger.info("=" * 60)
    perf = learner.get_performance()
    logger.info(f"Accuracy: {perf['accuracy']:.1%}")
    logger.info(f"Drift detected: {perf['drift_detected']}")
    logger.info(f"Total updates: {perf['total_updates']}")
    
    logger.info("Real-time learning system ready!")