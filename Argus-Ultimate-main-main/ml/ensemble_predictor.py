"""
Ensemble Prediction Pipeline — combines multiple ML models for robust predictions.

Features:
  - Dynamic weighting based on recent performance
  - Multiple combination strategies (weighted average, stacking, voting)
  - Automatic model health monitoring
  - Confidence calibration across models
  - Fallback to simpler models when complex ones fail

Usage:
    ensemble = EnsemblePredictor()
    
    # Register models
    ensemble.add_model("lstm", lstm_predictor, weight=0.3)
    ensemble.add_model("transformer", transformer_predictor, weight=0.4)
    ensemble.add_model("xgboost", xgb_predictor, weight=0.3)
    
    # Predict
    result = ensemble.predict(features)
    # result.predictions, result.confidence, result.model_weights
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class CombinationMethod(Enum):
    """Method for combining model predictions."""
    WEIGHTED_AVERAGE = "weighted_average"
    MEDIAN = "median"
    STACKING = "stacking"
    VOTING = "voting"
    DYNAMIC = "dynamic"  # Adapts based on recent performance


@dataclass
class ModelEntry:
    """Registered model in the ensemble."""
    name: str
    model: Any
    weight: float
    enabled: bool = True
    n_predictions: int = 0
    total_error: float = 0.0
    last_error: float = 0.0
    last_prediction_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def avg_error(self) -> float:
        if self.n_predictions == 0:
            return float('inf')
        return self.total_error / self.n_predictions
    
    @property
    def reliability(self) -> float:
        """Compute reliability score [0, 1] based on history."""
        if self.n_predictions < 10:
            return 0.5  # Neutral until enough data
        # Lower error = higher reliability
        return max(0.0, 1.0 - min(self.avg_error, 1.0))
    
    def update_error(self, error: float) -> None:
        """Update error tracking."""
        self.n_predictions += 1
        self.total_error += abs(error)
        self.last_error = abs(error)
        self.last_prediction_time = datetime.now(timezone.utc)


@dataclass
class EnsemblePrediction:
    """Result from ensemble prediction."""
    predictions: np.ndarray
    confidence: np.ndarray
    model_predictions: Dict[str, np.ndarray]  # Individual model predictions
    model_weights: Dict[str, float]           # Final weights used
    combination_method: str
    n_models_used: int
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "predictions": self.predictions.tolist() if isinstance(self.predictions, np.ndarray) else self.predictions,
            "confidence": self.confidence.tolist() if isinstance(self.confidence, np.ndarray) else self.confidence,
            "model_weights": self.model_weights,
            "combination_method": self.combination_method,
            "n_models_used": self.n_models_used,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class BaseModelWrapper(ABC):
    """Wrapper interface for models in the ensemble."""
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make prediction."""
        ...
    
    def predict_with_confidence(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make prediction with confidence. Override for better confidence."""
        predictions = self.predict(features)
        # Default confidence: 0.7
        confidence = np.full_like(predictions, 0.7)
        return predictions, confidence


class EnsemblePredictor:
    """
    Ensemble prediction combining multiple ML models.
    
    Features:
    - Dynamic weight adjustment based on recent performance
    - Multiple combination strategies
    - Automatic fallback when models fail
    - Confidence calibration
    
    Usage:
        ensemble = EnsemblePredictor(method=CombinationMethod.DYNAMIC)
        ensemble.add_model("lstm", lstm_model, weight=0.3)
        ensemble.add_model("xgb", xgb_model, weight=0.4)
        
        result = ensemble.predict(features)
    """
    
    def __init__(
        self,
        method: CombinationMethod = CombinationMethod.DYNAMIC,
        min_models: int = 1,
        confidence_threshold: float = 0.3,
        weight_update_rate: float = 0.1,
    ):
        self.method = method
        self.min_models = min_models
        self.confidence_threshold = confidence_threshold
        self.weight_update_rate = weight_update_rate
        
        self._models: Dict[str, ModelEntry] = {}
        self._prediction_history: List[Dict[str, Any]] = []
        
        logger.info("EnsemblePredictor initialized: method=%s", method.value)
    
    def add_model(
        self,
        name: str,
        model: Any,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a model to the ensemble."""
        self._models[name] = ModelEntry(
            name=name,
            model=model,
            weight=weight,
            metadata=metadata or {},
        )
        self._normalize_weights()
        logger.info("Model added: %s (weight=%.2f)", name, weight)
    
    def remove_model(self, name: str) -> None:
        """Remove a model from the ensemble."""
        if name in self._models:
            del self._models[name]
            self._normalize_weights()
            logger.info("Model removed: %s", name)
    
    def enable_model(self, name: str, enabled: bool = True) -> None:
        """Enable or disable a model."""
        if name in self._models:
            self._models[name].enabled = enabled
    
    def predict(
        self,
        features: np.ndarray,
        return_individual: bool = False,
    ) -> EnsemblePrediction:
        """
        Make ensemble prediction.
        
        Args:
            features: Input features
            return_individual: If True, include individual model predictions
            
        Returns:
            EnsemblePrediction with combined predictions and metadata
        """
        # Get enabled models
        active_models = {
            name: entry for name, entry in self._models.items()
            if entry.enabled
        }
        
        if len(active_models) < self.min_models:
            raise ValueError(
                f"Need at least {self.min_models} active models, got {len(active_models)}"
            )
        
        # Collect predictions from each model
        model_predictions: Dict[str, np.ndarray] = {}
        model_confidences: Dict[str, np.ndarray] = {}
        failed_models: List[str] = []
        
        for name, entry in active_models.items():
            try:
                pred, conf = self._predict_single(entry, features)
                model_predictions[name] = pred
                model_confidences[name] = conf
            except Exception as e:
                logger.warning("Model %s failed: %s", name, e)
                failed_models.append(name)
                self.enable_model(name, False)
        
        if not model_predictions:
            raise ValueError("All models failed")
        
        # Compute weights
        weights = self._compute_weights(model_predictions, features)
        
        # Combine predictions
        combined = self._combine_predictions(model_predictions, weights)
        
        # Combine confidence
        confidence = self._combine_confidence(model_confidences, weights)
        
        # Update model weights based on performance
        if self.method == CombinationMethod.DYNAMIC:
            self._update_weights_dynamic(model_predictions, features)
        
        result = EnsemblePrediction(
            predictions=combined,
            confidence=confidence,
            model_predictions=model_predictions if return_individual else {},
            model_weights=weights,
            combination_method=self.method.value,
            n_models_used=len(model_predictions),
            timestamp=datetime.now(timezone.utc),
            metadata={
                "failed_models": failed_models,
                "active_models": list(model_predictions.keys()),
            },
        )
        
        self._prediction_history.append({
            "timestamp": result.timestamp.isoformat(),
            "n_models": len(model_predictions),
            "weights": weights,
        })
        
        return result
    
    def update_feedback(
        self,
        features: np.ndarray,
        actual: np.ndarray,
    ) -> Dict[str, float]:
        """
        Update model weights based on actual outcomes.
        
        Args:
            features: Input features
            actual: Actual outcomes
            
        Returns:
            Dict of model_name → error for this update
        """
        errors = {}
        
        for name, entry in self._models.items():
            if not entry.enabled:
                continue
            
            try:
                pred, _ = self._predict_single(entry, features)
                error = float(np.mean(np.abs(pred - actual)))
                entry.update_error(error)
                errors[name] = error
            except Exception as e:
                logger.warning("Feedback update failed for %s: %s", name, e)
        
        return errors
    
    def get_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all models."""
        return {
            name: {
                "weight": entry.weight,
                "enabled": entry.enabled,
                "n_predictions": entry.n_predictions,
                "avg_error": entry.avg_error,
                "reliability": entry.reliability,
                "last_error": entry.last_error,
            }
            for name, entry in self._models.items()
        }
    
    def get_best_model(self) -> Optional[str]:
        """Get the name of the most reliable model."""
        if not self._models:
            return None
        
        enabled = {n: e for n, e in self._models.items() if e.enabled}
        if not enabled:
            return None
        
        best = max(enabled.values(), key=lambda e: e.reliability)
        return best.name
    
    def _predict_single(
        self,
        entry: ModelEntry,
        features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get prediction from a single model."""
        model = entry.model
        
        # Try predict_with_confidence first
        if hasattr(model, 'predict_with_confidence'):
            return model.predict_with_confidence(features)
        
        # Fall back to predict
        if hasattr(model, 'predict'):
            predictions = model.predict(features)
            # Estimate confidence from model reliability
            confidence = np.full_like(predictions, entry.reliability)
            return predictions, confidence
        
        # Callable
        if callable(model):
            predictions = model(features)
            confidence = np.full_like(predictions, entry.reliability)
            return predictions, confidence
        
        raise ValueError(f"Model {entry.name} has no predict method")
    
    def _compute_weights(
        self,
        predictions: Dict[str, np.ndarray],
        features: np.ndarray,
    ) -> Dict[str, float]:
        """Compute combination weights."""
        if self.method == CombinationMethod.WEIGHTED_AVERAGE:
            # Use registered weights
            total = sum(
                self._models[name].weight for name in predictions.keys()
            )
            return {
                name: self._models[name].weight / total
                for name in predictions.keys()
            }
        
        elif self.method == CombinationMethod.MEDIAN:
            # Equal weights (median doesn't use weights)
            n = len(predictions)
            return {name: 1.0 / n for name in predictions.keys()}
        
        elif self.method == CombinationMethod.VOTING:
            # Equal weights for voting
            n = len(predictions)
            return {name: 1.0 / n for name in predictions.keys()}
        
        elif self.method == CombinationMethod.DYNAMIC:
            # Weight by reliability
            reliabilities = {
                name: self._models[name].reliability
                for name in predictions.keys()
            }
            total = sum(reliabilities.values())
            if total > 0:
                return {name: r / total for name, r in reliabilities.items()}
            else:
                n = len(predictions)
                return {name: 1.0 / n for name in predictions.keys()}
        
        elif self.method == CombinationMethod.STACKING:
            # Simple stacking: weight by inverse error
            errors = {
                name: max(self._models[name].avg_error, 1e-6)
                for name in predictions.keys()
            }
            inv_errors = {name: 1.0 / e for name, e in errors.items()}
            total = sum(inv_errors.values())
            return {name: e / total for name, e in inv_errors.items()}
        
        # Default: equal weights
        n = len(predictions)
        return {name: 1.0 / n for name in predictions.keys()}
    
    def _combine_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        weights: Dict[str, float],
    ) -> np.ndarray:
        """Combine predictions using weights."""
        if self.method == CombinationMethod.MEDIAN:
            # Median combination
            stacked = np.stack(list(predictions.values()))
            return np.median(stacked, axis=0)
        
        # Weighted combination
        combined = np.zeros_like(list(predictions.values())[0])
        for name, pred in predictions.items():
            w = weights.get(name, 0.0)
            combined += w * pred
        
        return combined
    
    def _combine_confidence(
        self,
        confidences: Dict[str, np.ndarray],
        weights: Dict[str, float],
    ) -> np.ndarray:
        """Combine confidence scores."""
        combined = np.zeros_like(list(confidences.values())[0])
        for name, conf in confidences.items():
            w = weights.get(name, 0.0)
            combined += w * conf
        
        return np.clip(combined, 0.0, 1.0)
    
    def _update_weights_dynamic(
        self,
        predictions: Dict[str, np.ndarray],
        features: np.ndarray,
    ) -> None:
        """Update model weights based on recent performance."""
        for name in predictions.keys():
            if name in self._models:
                entry = self._models[name]
                # Adjust weight based on reliability
                reliability = entry.reliability
                entry.weight = entry.weight * (1 - self.weight_update_rate) + \
                              reliability * self.weight_update_rate
        
        self._normalize_weights()
    
    def _normalize_weights(self) -> None:
        """Normalize all weights to sum to 1."""
        enabled = [e for e in self._models.values() if e.enabled]
        if not enabled:
            return
        
        total = sum(e.weight for e in enabled)
        if total > 0:
            for entry in enabled:
                entry.weight /= total
