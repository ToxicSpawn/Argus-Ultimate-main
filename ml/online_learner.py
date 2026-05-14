"""
Online Learning Module

Adapts ML models in real-time as new market data arrives.
Features:
- Incremental model updates
- Concept drift detection
- Model performance tracking
- Automatic retraining triggers
- Ensemble weight adjustment based on recent performance
"""

import json
import logging
import pickle
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, r2_score

logger = logging.getLogger(__name__)


class ConceptDriftDetector:
    """Detects when model performance degrades (concept drift)."""
    
    def __init__(self, window_size: int = 100, threshold: float = 0.15):
        self.window_size = window_size
        self.threshold = threshold
        self.recent_accuracies: deque = deque(maxlen=window_size)
        self.baseline_accuracy: Optional[float] = None
        
    def update(self, accuracy: float):
        """Update with new accuracy measurement."""
        self.recent_accuracies.append(accuracy)
        
        if self.baseline_accuracy is None and len(self.recent_accuracies) >= 50:
            self.baseline_accuracy = np.mean(list(self.recent_accuracies)[:50])
    
    def check_drift(self) -> Tuple[bool, float]:
        """Check if concept drift detected."""
        if self.baseline_accuracy is None or len(self.recent_accuracies) < 20:
            return False, 0.0
        
        recent_mean = np.mean(list(self.recent_accuracies)[-20:])
        drift_magnitude = self.baseline_accuracy - recent_mean
        
        return drift_magnitude > self.threshold, drift_magnitude
    
    def reset(self, new_baseline: float):
        """Reset detector with new baseline."""
        self.baseline_accuracy = new_baseline
        self.recent_accuracies.clear()


class ModelPerformanceTracker:
    """Tracks model performance over time."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.predictions: deque = deque(maxlen=1000)
        self.actuals: deque = deque(maxlen=1000)
        self.timestamps: deque = deque(maxlen=1000)
        self.accuracies: List[float] = []
        
    def record(self, prediction, actual, timestamp: Optional[datetime] = None):
        """Record a prediction vs actual outcome."""
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.timestamps.append(timestamp or datetime.now())
        
        if len(self.predictions) >= 50:
            recent_preds = list(self.predictions)[-100:]
            recent_actuals = list(self.actuals)[-100:]
            acc = accuracy_score(recent_actuals, recent_preds)
            self.accuracies.append(acc)
    
    def get_recent_accuracy(self, window: int = 100) -> float:
        """Get accuracy over recent window."""
        if len(self.predictions) < 10:
            return 0.5
        
        n = min(window, len(self.predictions))
        recent_preds = list(self.predictions)[-n:]
        recent_actuals = list(self.actuals)[-n:]
        
        return accuracy_score(recent_actuals, recent_preds)
    
    def get_trend(self, window: int = 50) -> str:
        """Get performance trend."""
        if len(self.accuracies) < window * 2:
            return "insufficient_data"
        
        recent = np.mean(self.accuracies[-window:])
        older = np.mean(self.accuracies[-window*2:-window])
        
        diff = recent - older
        if diff > 0.02:
            return "improving"
        elif diff < -0.02:
            return "declining"
        return "stable"


class OnlineLearner:
    """
    Online learning system that adapts models as new data arrives.
    """
    
    def __init__(self, models_dir: str = "data/models_mtf", 
                 backup_dir: str = "data/model_backups"):
        self.models_dir = Path(models_dir)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict = {}
        self.scaler = None
        self.feature_names: List[str] = []
        
        self.trackers: Dict[str, ModelPerformanceTracker] = {}
        self.drift_detectors: Dict[str, ConceptDriftDetector] = {}
        
        self.update_buffer_X: List = []
        self.update_buffer_y: Dict[str, List] = {}
        self.buffer_size = 100
        self.updates_since_save = 0
        
    def load_models(self):
        """Load current production models."""
        logger.info("Loading models for online learning...")
        
        model_files = [
            'signal_classifier', 'regime_classifier', 'position_sizer',
            'volatility_model', 'trend_strength'
        ]
        
        for name in model_files:
            model_path = self.models_dir / f"{name}.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[name] = pickle.load(f)
                self.trackers[name] = ModelPerformanceTracker(name)
                self.drift_detectors[name] = ConceptDriftDetector()
                logger.info(f"  Loaded {name}")
        
        scaler_path = self.models_dir / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        features_path = self.models_dir / "feature_names.pkl"
        if features_path.exists():
            with open(features_path, 'rb') as f:
                self.feature_names = pickle.load(f)
        
        logger.info(f"Loaded {len(self.models)} models")
    
    def backup_models(self):
        """Create backup of current models."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, model in self.models.items():
            backup_path = self.backup_dir / f"{name}_{timestamp}.pkl"
            with open(backup_path, 'wb') as f:
                pickle.dump(model, f)
        
        logger.info(f"Models backed up with timestamp {timestamp}")
    
    def restore_backup(self, timestamp: str):
        """Restore models from backup."""
        for name in self.models.keys():
            backup_path = self.backup_dir / f"{name}_{timestamp}.pkl"
            if backup_path.exists():
                with open(backup_path, 'rb') as f:
                    self.models[name] = pickle.load(f)
                logger.info(f"Restored {name} from backup")
    
    def record_prediction(self, model_name: str, prediction, actual):
        """Record a prediction for performance tracking."""
        if model_name in self.trackers:
            self.trackers[model_name].record(prediction, actual)
        
        if model_name in self.drift_detectors:
            tracker = self.trackers[model_name]
            recent_acc = tracker.get_recent_accuracy(50)
            self.drift_detectors[model_name].update(recent_acc)
    
    def add_training_sample(self, X: np.ndarray, labels: Dict[str, float]):
        """Add a new training sample to the update buffer."""
        self.update_buffer_X.append(X)
        
        for name, target in labels.items():
            if name not in self.update_buffer_y:
                self.update_buffer_y[name] = []
            self.update_buffer_y[name].append(target)
        
        if len(self.update_buffer_X) >= self.buffer_size:
            self._process_buffer()
    
    def _process_buffer(self):
        """Process the update buffer and incrementally update models."""
        if not self.update_buffer_X:
            return
        
        logger.info(f"Processing update buffer ({len(self.update_buffer_X)} samples)")
        
        X_buffer = np.array(self.update_buffer_X)
        X_scaled = self.scaler.transform(X_buffer) if self.scaler else X_buffer
        
        for model_name in self.models:
            if model_name not in self.update_buffer_y:
                continue
            
            y_buffer = np.array(self.update_buffer_y[model_name])
            
            if len(np.unique(y_buffer)) < 2:
                logger.warning(f"  Skipping {model_name}: single class")
                continue
            
            model = self.models[model_name]
            
            try:
                model.fit(X_scaled, y_buffer)
                logger.info(f"  Updated {model_name}")
            except Exception as e:
                logger.warning(f"  Failed update for {model_name}: {e}")
        
        self.update_buffer_X.clear()
        self.update_buffer_y.clear()
        self.updates_since_save += 1
        
        if self.updates_since_save >= 10:
            self._save_updated_models()
            self.updates_since_save = 0
    
    def _save_updated_models(self):
        """Save updated models to disk."""
        for name, model in self.models.items():
            model_path = self.models_dir / f"{name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        logger.info("Updated models saved")
    
    def check_all_drift(self) -> Dict[str, Tuple[bool, float]]:
        """Check concept drift for all models."""
        results = {}
        for name, detector in self.drift_detectors.items():
            drift_detected, magnitude = detector.check_drift()
            results[name] = (drift_detected, magnitude)
            
            if drift_detected:
                logger.warning(f"DRIFT: {name} (magnitude: {magnitude:.3f})")
        
        return results
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }
        
        for name, tracker in self.trackers.items():
            report['models'][name] = {
                'recent_accuracy': tracker.get_recent_accuracy(100),
                'trend': tracker.get_trend(),
                'total_predictions': len(tracker.predictions),
            }
            
            if name in self.drift_detectors:
                drift_detected, magnitude = self.drift_detectors[name].check_drift()
                report['models'][name]['drift_detected'] = drift_detected
                report['models'][name]['drift_magnitude'] = magnitude
        
        return report
    
    def force_retrain(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]):
        """Force full retrain of all models."""
        logger.info("Force retraining all models...")
        
        X_scaled = self.scaler.transform(X) if self.scaler else X
        
        for model_name, model in self.models.items():
            if model_name in y_dict:
                try:
                    model.fit(X_scaled, y_dict[model_name])
                    
                    if model_name in self.drift_detectors:
                        self.drift_detectors[model_name].reset(0.7)
                    
                    logger.info(f"  Retrained {model_name}")
                except Exception as e:
                    logger.error(f"  Failed to retrain {model_name}: {e}")
        
        self._save_updated_models()
        logger.info("Force retrain complete")


class EnsembleWeightOptimizer:
    """Optimizes ensemble weights based on recent model performance."""
    
    def __init__(self, models: Dict, decay_factor: float = 0.95):
        self.models = models
        self.decay_factor = decay_factor
        self.weights: Dict[str, float] = {}
        self.performance_history: Dict[str, List[float]] = {}
        
        for name in models:
            self.weights[name] = 1.0 / len(models)
            self.performance_history[name] = []
    
    def update_weights(self, performance: Dict[str, float]):
        """Update weights based on recent performance."""
        for name, acc in performance.items():
            if name in self.performance_history:
                self.performance_history[name].append(acc)
        
        total_weight = 0
        for name in self.models:
            if name in performance:
                recent_perf = np.mean(self.performance_history[name][-20:]) if self.performance_history[name] else 0.5
                self.weights[name] = max(0.1, recent_perf)
                total_weight += self.weights[name]
        
        if total_weight > 0:
            for name in self.weights:
                self.weights[name] /= total_weight
        
        logger.info(f"Updated weights: {self.weights}")
    
    def predict_ensemble(self, X: np.ndarray, prediction_type: str = 'classifier') -> np.ndarray:
        """Make weighted ensemble prediction."""
        predictions = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    preds = model.predict_proba(X)
                else:
                    preds = model.predict(X)
                predictions[name] = preds
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
        
        if not predictions:
            return np.zeros(len(X))
        
        if prediction_type == 'classifier':
            n_classes = predictions[next(iter(predictions.keys()))].shape[1] if len(predictions[next(iter(predictions.keys()))].shape) > 1 else 3
            weighted_probs = np.zeros((len(X), n_classes))
            
            for name, probs in predictions.items():
                if len(probs.shape) == 2:
                    weighted_probs += self.weights.get(name, 0) * probs
            
            return np.argmax(weighted_probs, axis=1)
        else:
            weighted_pred = np.zeros(len(X))
            
            for name, pred in predictions.items():
                weighted_pred += self.weights.get(name, 0) * pred
            
            return weighted_pred
    
    def get_weights(self) -> Dict[str, float]:
        """Get current ensemble weights."""
        return self.weights.copy()
