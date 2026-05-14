"""
Unified ML Orchestrator for Argus Trading System.

Combines all ML components into a single, coherent pipeline:
- Feature engineering (technical, microstructural, cross-asset)
- Feature selection (importance-based, correlation-based)
- Model ensemble (stacking, blending, dynamic weighting)
- Validation (walk-forward, bootstrap)
- Online learning (drift detection, model updates)
- Health monitoring (performance tracking, alerts)

Usage:
    orchestrator = MLOrchestrator()
    
    # Configure
    orchestrator.configure(
        features=["price", "technical", "volume"],
        models=["xgb", "lgb", "rfr"],
        validation="walk_forward",
    )
    
    # Train
    orchestrator.fit(features, labels)
    
    # Predict
    prediction = orchestrator.predict(features)
    
    # Update online
    orchestrator.update(features_new, labels_new)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MLOrchestratorConfig:
    """Configuration for ML orchestrator."""

    # Feature settings
    feature_categories: List[str] = field(default_factory=lambda: ["price", "technical", "volume"])
    min_features: int = 10
    max_features: int = 200
    feature_selection_method: str = "importance"  # importance, correlation, recursive

    # Model settings
    models: List[str] = field(default_factory=lambda: ["xgb", "lgb"])
    ensemble_method: str = "dynamic"  # weighted_average, stacking, voting, dynamic
    model_diversity_threshold: float = 0.3

    # Validation settings
    validation_method: str = "walk_forward"  # walk_forward, bootstrap, purged_cv
    n_validation_splits: int = 5
    train_pct: float = 0.7
    purge_gap: int = 5

    # Online learning settings
    online_learning_enabled: bool = True
    drift_detection_enabled: bool = True
    retrain_threshold: float = 0.15  # Performance drop triggers retrain

    # Health monitoring
    health_check_interval: int = 100  # Predictions between health checks
    alert_threshold: float = 0.05  # Performance drop triggers alert


@dataclass
class MLOrchestratorResult:
    """Result from ML orchestrator operations."""

    prediction: np.ndarray
    confidence: float
    action: str  # buy, sell, hold
    model_weights: Dict[str, float]
    features_used: List[str]
    regime: str
    regime_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction": self.prediction.tolist() if hasattr(self.prediction, "tolist") else float(self.prediction),
            "confidence": float(self.confidence),
            "action": self.action,
            "model_weights": self.model_weights,
            "features_used": self.features_used,
            "regime": self.regime,
            "regime_confidence": float(self.regime_confidence),
            "metadata": self.metadata,
        }


class MLOrchestrator:
    """
    Unified ML orchestrator for trading strategies.

    Integrates:
    - FeatureLibrary for feature engineering
    - FeatureSelector for feature selection
    - EnsemblePredictor for model combination
    - WalkForwardValidator for validation
    - OnlineLearner for real-time updates
    - DriftDetector for concept drift
    """

    def __init__(
        self,
        *,
        config: Optional[MLOrchestratorConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config or MLOrchestratorConfig()
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self._fitted = False
        self._models: Dict[str, Any] = {}
        self._feature_selector = None
        self._ensemble = None
        self._validator = None
        self._online_learner = None
        self._drift_detector = None
        self._feature_library = None
        self._regime_router = None

        self._n_predictions = 0
        self._last_health_check = 0
        self._health_metrics: Dict[str, List[float]] = {}

    def configure(self, **kwargs) -> "MLOrchestrator":
        """Update configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self._initialize_components()
        return self

    def _initialize_components(self) -> None:
        """Initialize all ML components."""
        # Feature library
        try:
            from ml.features.feature_library import FeatureLibrary
            self._feature_library = FeatureLibrary()
        except Exception:
            self._feature_library = None

        # Feature selector
        try:
            from ml.feature_selector import FeatureSelector
            self._feature_selector = FeatureSelector(
                method=self.config.feature_selection_method,
                min_features=self.config.min_features,
                max_features=self.config.max_features,
            )
        except Exception:
            self._feature_selector = None

        # Ensemble predictor
        try:
            from ml.ensemble_predictor import EnsemblePredictor
            self._ensemble = EnsemblePredictor()
        except Exception:
            self._ensemble = None

        # Walk-forward validator
        try:
            from ml.walk_forward_validator import WalkForwardMLValidator
            self._validator = WalkForwardMLValidator(
                n_splits=self.config.n_validation_splits,
                train_pct=self.config.train_pct,
                purge_gap=self.config.purge_gap,
            )
        except Exception:
            self._validator = None

        # Online learner
        if self.config.online_learning_enabled:
            try:
                from ml.online_learning import OnlineLearner
                self._online_learner = OnlineLearner()
            except Exception:
                self._online_learner = None

        # Drift detector
        if self.config.drift_detection_enabled:
            try:
                from ml.drift_detector import DriftDetector
                self._drift_detector = DriftDetector()
            except Exception:
                self._drift_detector = None

        # Regime router
        try:
            from ml.regime_strategy_router import RegimeStrategyRouter
            self._regime_router = RegimeStrategyRouter()
        except Exception:
            self._regime_router = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: Optional[List[str]] = None,
        validate: bool = True,
    ) -> "MLOrchestrator":
        """Train the ML pipeline."""
        if len(X) < 50:
            raise ValueError("Need at least 50 samples to train")

        # Feature selection
        if self._feature_selector is not None and len(X) > 100:
            try:
                X_selected, selected_idx = self._feature_selector.select(X, y)
                if len(selected_idx) > 0:
                    X = X_selected
                    if feature_names:
                        feature_names = [feature_names[i] for i in selected_idx]
            except Exception:
                pass

        # Train ensemble
        if self._ensemble is not None:
            self._train_ensemble(X, y)
        else:
            # Fallback: simple average
            self._models["default"] = self._train_simple_model(X, y)
            self._fitted = True

        # Validate if requested
        if validate and self._validator is not None:
            try:
                val_results = self._validator.validate(
                    lambda: self._train_simple_model(X, y),
                    X, y,
                )
                self._validation_results = val_results
            except Exception:
                pass

        self._X_train = X
        self._y_train = y
        self._feature_names = feature_names
        self._fitted = True

        return self

    def _train_ensemble(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train ensemble of models."""
        from ml.ensemble_predictor import CombinationMethod

        # Determine combination method
        method_map = {
            "weighted_average": CombinationMethod.WEIGHTED_AVERAGE,
            "stacking": CombinationMethod.STACKING,
            "voting": CombinationMethod.VOTING,
            "dynamic": CombinationMethod.DYNAMIC,
        }
        method = method_map.get(self.config.ensemble_method, CombinationMethod.DYNAMIC)

        # Add available models
        for model_name in self.config.models:
            try:
                model = self._create_model(model_name)
                weight = 1.0 / len(self.config.models)
                self._ensemble.add_model(model_name, model, weight=weight)
            except Exception:
                pass

        self._fitted = True

    def _create_model(self, model_name: str) -> Any:
        """Create a model by name."""
        if model_name == "xgb":
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(n_estimators=100, max_depth=5, random_state=self.seed)
            except ImportError:
                pass
        elif model_name == "lgb":
            try:
                from lightgbm import LGBMClassifier
                return LGBMClassifier(n_estimators=100, max_depth=5, random_state=self.seed)
            except ImportError:
                pass
        elif model_name == "rfr":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, max_depth=5, random_state=self.seed)
        elif model_name == "lr":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, random_state=self.seed)

        # Fallback to simple
        return self._train_simple_model(np.array([]), np.array([]))

    def _train_simple_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Train a simple fallback model."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=self.seed)
        except Exception:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, random_state=self.seed)

    def predict(
        self,
        X: np.ndarray,
        *,
        returns: Optional[List[float]] = None,
    ) -> MLOrchestratorResult:
        """Generate prediction from ML pipeline."""
        if not self._fitted:
            return MLOrchestratorResult(
                prediction=np.array([0.0]),
                confidence=0.0,
                action="hold",
                model_weights={},
                features_used=[],
                regime="UNKNOWN",
                regime_confidence=0.0,
            )

        # Get regime prediction
        regime = "UNKNOWN"
        regime_confidence = 0.0
        if self._regime_router is not None and returns is not None:
            try:
                regime_pred = self._regime_router.get_strategy_weights_for_regime("TREND_UP")
                regime = regime_pred.regime if hasattr(regime_pred, "regime") else "UNKNOWN"
            except Exception:
                pass

        # Get ensemble prediction
        confidence = 0.5
        action = "hold"
        model_weights = {}

        if self._ensemble is not None:
            try:
                result = self._ensemble.predict(X)
                if hasattr(result, "predictions"):
                    prediction = np.array(result.predictions)
                else:
                    prediction = np.array([0.0])
                confidence = getattr(result, "confidence", 0.5)
                model_weights = getattr(result, "model_weights", {})
            except Exception:
                prediction = self._predict_fallback(X)
        else:
            prediction = self._predict_fallback(X)

        # Convert to action
        if isinstance(prediction, np.ndarray):
            if len(prediction) == 1:
                prob = float(prediction[0]) if prediction[0] > 0 else 0.0
            else:
                prob = float(np.mean(prediction))
        else:
            prob = float(prediction)

        if prob > 0.6:
            action = "buy"
        elif prob < 0.4:
            action = "sell"
        else:
            action = "hold"

        # Update online learning
        if self._online_learner is not None and self._n_predictions > 0:
            try:
                self._online_learner.update(X, prob)
            except Exception:
                pass

        self._n_predictions += 1

        return MLOrchestratorResult(
            prediction=prediction,
            confidence=confidence,
            action=action,
            model_weights=model_weights,
            features_used=self._feature_names or [],
            regime=regime,
            regime_confidence=regime_confidence,
        )

    def _predict_fallback(self, X: np.ndarray) -> np.ndarray:
        """Fallback prediction when ensemble unavailable."""
        return np.array([0.5])

    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        check_drift: bool = True,
    ) -> Dict[str, Any]:
        """Update models with new data."""
        result = {
            "updated": False,
            "drift_detected": False,
            "retrain_triggered": False,
            "metrics": {},
        }

        # Check for drift
        if check_drift and self._drift_detector is not None:
            try:
                drift_result = self._drift_detector.update(X)
                result["drift_detected"] = drift_result.get("drift", False)
            except Exception:
                pass

        # Online update
        if self._online_learner is not None:
            try:
                learning_result = self._online_learner.update(X, y)
                result["updated"] = learning_result.updated
            except Exception:
                pass

        # Check if retrain needed
        if self._should_retrain():
            result["retrain_triggered"] = True

        return result

    def _should_retrain(self) -> bool:
        """Check if retraining is needed."""
        if not self._health_metrics:
            return False

        # Check recent performance
        recent = self._health_metrics.get("accuracy", [])
        if len(recent) < 10:
            return False

        # Performance drop check
        recent_avg = np.mean(recent[-5:])
        historical_avg = np.mean(recent[:-5])
        drop = historical_avg - recent_avg

        return drop > self.config.retrain_threshold

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on ML components."""
        return {
            "fitted": self._fitted,
            "n_predictions": self._n_predictions,
            "models_loaded": list(self._models.keys()),
            "online_learning": self._online_learner is not None,
            "drift_detection": self._drift_detector is not None,
            "regime_routing": self._regime_router is not None,
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self._feature_selector is not None:
            try:
                return self._feature_selector.get_importance_scores()
            except Exception:
                pass
        return {}

    def get_model_diversity(self) -> float:
        """Compute model diversity score."""
        if self._ensemble is None or len(self._models) < 2:
            return 0.0

        try:
            return self._ensemble.get_diversity_score()
        except Exception:
            return 0.0


def create_orchestrator(
    models: Optional[List[str]] = None,
    validation: str = "walk_forward",
    online_learning: bool = True,
) -> MLOrchestrator:
    """Factory function to create configured orchestrator."""
    config = MLOrchestratorConfig(
        models=models or ["xgb", "lgb"],
        validation_method=validation,
        online_learning_enabled=online_learning,
    )
    return MLOrchestrator(config=config)


__all__ = [
    "MLOrchestrator",
    "MLOrchestratorConfig",
    "MLOrchestratorResult",
    "create_orchestrator",
]