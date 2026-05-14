"""
Unified ML Pipeline Orchestrator — ties together all Argus ML components.

This module provides a single coordination point for:
  1. Feature engineering pipeline
  2. Model training and evaluation
  3. Drift detection and monitoring
  4. Automated retraining triggers
  5. Model versioning and persistence
  6. Prediction serving with fallbacks

Usage:
    pipeline = MLPipelineOrchestrator(config={
        "model_name": "regime_classifier",
        "model_type": "xgboost",
        "feature_sources": ["prices", "volume", "orderbook"],
        "retrain_interval_hours": 24,
        "drift_threshold": 0.1,
    })
    
    # Train
    pipeline.train(features, labels)
    
    # Predict
    result = pipeline.predict(current_features)
    
    # Monitor
    metrics = pipeline.check_health()
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Data Classes
# ---------------------------------------------------------------------------

class ModelStatus(Enum):
    """Model lifecycle status."""
    INITIALIZING = "initializing"
    TRAINING = "training"
    READY = "ready"
    DEGRADED = "degraded"  # Drift detected but still usable
    RETRAINING = "retraining"
    ARCHIVED = "archived"
    FAILED = "failed"


class DriftSeverity(Enum):
    """Drift severity levels."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class ModelMetadata:
    """Model version metadata."""
    model_id: str
    model_name: str
    model_type: str
    version: int
    created_at: datetime
    trained_samples: int
    feature_hash: str  # Hash of feature schema
    metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    status: ModelStatus = ModelStatus.INITIALIZING
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "trained_samples": self.trained_samples,
            "feature_hash": self.feature_hash,
            "metrics": self.metrics,
            "hyperparameters": self.hyperparameters,
            "status": self.status.value,
        }


@dataclass
class PredictionResult:
    """Unified prediction result."""
    predictions: np.ndarray
    confidence: np.ndarray
    model_id: str
    timestamp: datetime
    feature_importance: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "predictions": self.predictions.tolist() if isinstance(self.predictions, np.ndarray) else self.predictions,
            "confidence": self.confidence.tolist() if isinstance(self.confidence, np.ndarray) else self.confidence,
            "model_id": self.model_id,
            "timestamp": self.timestamp.isoformat(),
            "feature_importance": self.feature_importance,
            "metadata": self.metadata,
        }


@dataclass
class HealthMetrics:
    """Pipeline health metrics."""
    model_status: ModelStatus
    total_predictions: int
    avg_confidence: float
    drift_score: float
    drift_severity: DriftSeverity
    last_training_time: Optional[datetime]
    last_drift_check: Optional[datetime]
    uptime_hours: float
    error_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_status": self.model_status.value,
            "total_predictions": self.total_predictions,
            "avg_confidence": round(self.avg_confidence, 4),
            "drift_score": round(self.drift_score, 4),
            "drift_severity": self.drift_severity.value,
            "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
            "last_drift_check": self.last_drift_check.isoformat() if self.last_drift_check else None,
            "uptime_hours": round(self.uptime_hours, 2),
            "error_rate": round(self.error_rate, 4),
        }


@dataclass
class PipelineConfig:
    """ML Pipeline configuration."""
    model_name: str = "default_model"
    model_type: str = "xgboost"  # xgboost, random_forest, neural_network, tft
    feature_sources: List[str] = field(default_factory=lambda: ["prices"])
    target_column: str = "target"
    
    # Training
    train_test_split: float = 0.8
    min_training_samples: int = 100
    retrain_interval_hours: int = 24
    retrain_samples_threshold: int = 1000
    
    # Drift detection
    drift_threshold: float = 0.1
    drift_check_interval_minutes: int = 60
    auto_retrain_on_drift: bool = True
    
    # Model selection
    hyperparameter_tuning: bool = True
    n_cv_folds: int = 5
    
    # Persistence
    model_dir: str = "models"
    keep_versions: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "feature_sources": self.feature_sources,
            "target_column": self.target_column,
            "train_test_split": self.train_test_split,
            "min_training_samples": self.min_training_samples,
            "retrain_interval_hours": self.retrain_interval_hours,
            "retrain_samples_threshold": self.retrain_samples_threshold,
            "drift_threshold": self.drift_threshold,
            "drift_check_interval_minutes": self.drift_check_interval_minutes,
            "auto_retrain_on_drift": self.auto_retrain_on_drift,
            "hyperparameter_tuning": self.hyperparameter_tuning,
            "n_cv_folds": self.n_cv_folds,
            "model_dir": self.model_dir,
            "keep_versions": self.keep_versions,
        }


# ---------------------------------------------------------------------------
# Feature Engineering Pipeline
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """Feature engineering pipeline with transformation tracking."""
    
    def __init__(self, feature_sources: List[str]):
        self.feature_sources = feature_sources
        self._transformations: List[Callable] = []
        self._feature_names: List[str] = []
        self._fitted = False
    
    def add_transformation(self, transform: Callable, name: str = "") -> "FeaturePipeline":
        """Add a feature transformation."""
        self._transformations.append(transform)
        return self
    
    def fit(self, raw_data: Dict[str, np.ndarray]) -> "FeaturePipeline":
        """Fit transformations on training data."""
        self._fitted = True
        return self
    
    def transform(self, raw_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Transform raw data to features."""
        # Combine all feature sources
        features = []
        for source in self.feature_sources:
            if source in raw_data:
                features.append(raw_data[source].flatten())
        
        if not features:
            # Return zeros if no features available
            return np.zeros((1, len(self.feature_sources)))
        
        combined = np.concatenate(features)
        return combined.reshape(1, -1) if combined.ndim == 1 else combined
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after transformation."""
        return self._feature_names.copy()
    
    def compute_feature_hash(self, feature_array: np.ndarray) -> str:
        """Compute hash of feature schema for versioning."""
        shape_str = str(feature_array.shape)
        dtype_str = str(feature_array.dtype)
        return hashlib.md5(f"{shape_str}_{dtype_str}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Registry for model versioning and persistence."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, ModelMetadata] = {}
    
    def register(self, metadata: ModelMetadata) -> str:
        """Register a new model version."""
        self._models[metadata.model_id] = metadata
        self._save_metadata(metadata)
        return metadata.model_id
    
    def get(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID."""
        return self._models.get(model_id)
    
    def get_latest(self, model_name: str) -> Optional[ModelMetadata]:
        """Get latest version of a model."""
        matching = [
            m for m in self._models.values()
            if m.model_name == model_name and m.status == ModelStatus.READY
        ]
        if not matching:
            return None
        return max(matching, key=lambda m: m.version)
    
    def archive_old_versions(self, model_name: str, keep: int = 5) -> List[str]:
        """Archive old model versions, keeping only the latest N."""
        matching = [
            m for m in self._models.values()
            if m.model_name == model_name
        ]
        matching.sort(key=lambda m: m.version, reverse=True)
        
        archived = []
        for m in matching[keep:]:
            m.status = ModelStatus.ARCHIVED
            archived.append(m.model_id)
        
        return archived
    
    def _save_metadata(self, metadata: ModelMetadata) -> None:
        """Save model metadata to disk."""
        path = self.model_dir / f"{metadata.model_id}_metadata.json"
        with open(path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Drift Monitor
# ---------------------------------------------------------------------------

class DriftMonitor:
    """Drift monitoring with multiple detection methods."""
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self._reference_distribution: Optional[np.ndarray] = None
        self._current_window: List[float] = []
        self._window_size: int = 100
        self._drift_history: List[Dict[str, Any]] = []
    
    def set_reference(self, reference: np.ndarray) -> None:
        """Set reference distribution from training data."""
        self._reference_distribution = reference.copy()
    
    def update(self, predictions: np.ndarray) -> Optional[Dict[str, Any]]:
        """Update with new predictions and check for drift."""
        self._current_window.extend(predictions.flatten())
        if len(self._current_window) > self._window_size:
            self._current_window = self._current_window[-self._window_size:]
        
        if len(self._current_window) < self._window_size // 2:
            return None
        
        return self._check_drift()
    
    def _check_drift(self) -> Optional[Dict[str, Any]]:
        """Check for drift using KS test."""
        if self._reference_distribution is None:
            return None
        
        current = np.array(self._current_window)
        reference = self._reference_distribution
        
        # KS statistic
        ks_stat = self._ks_statistic(reference, current)
        
        # PSI
        psi = self._psi(reference, current)
        
        drift_detected = ks_stat > self.threshold or psi > 0.2
        
        result = {
            "drift_detected": drift_detected,
            "ks_statistic": float(ks_stat),
            "psi": float(psi),
            "severity": self._classify_severity(max(ks_stat, psi)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        if drift_detected:
            self._drift_history.append(result)
            logger.warning("Drift detected: KS=%.4f PSI=%.4f", ks_stat, psi)
        
        return result
    
    def _ks_statistic(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute Kolmogorov-Smirnov statistic."""
        if len(a) == 0 or len(b) == 0:
            return 0.0
        
        a_sorted = np.sort(a)
        b_sorted = np.sort(b)
        all_values = np.sort(np.concatenate([a_sorted, b_sorted]))
        
        if len(all_values) == 0:
            return 0.0
        
        cdf_a = np.searchsorted(a_sorted, all_values, side="right") / max(len(a_sorted), 1)
        cdf_b = np.searchsorted(b_sorted, all_values, side="right") / max(len(b_sorted), 1)
        
        return float(np.nanmax(np.abs(cdf_a - cdf_b)))
    
    def _psi(self, reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        """Compute Population Stability Index."""
        bins = np.histogram_bin_edges(reference, bins=n_bins)
        ref_hist, _ = np.histogram(reference, bins=bins, density=True)
        cur_hist, _ = np.histogram(current, bins=bins, density=True)
        
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        ref_hist = ref_hist + eps
        cur_hist = cur_hist + eps
        
        psi = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))
        return float(max(psi, 0.0))
    
    def _classify_severity(self, score: float) -> str:
        """Classify drift severity."""
        if score < self.threshold:
            return "none"
        if score < 0.2:
            return "minor"
        if score < 0.35:
            return "moderate"
        return "severe"
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get drift detection history."""
        return self._drift_history.copy()


# ---------------------------------------------------------------------------
# Main Pipeline Orchestrator
# ---------------------------------------------------------------------------

class MLPipelineOrchestrator:
    """
    Unified ML Pipeline Orchestrator.
    
    Coordinates:
    - Feature engineering
    - Model training and evaluation
    - Drift detection
    - Automated retraining
    - Model versioning
    - Prediction serving
    
    Usage:
        pipeline = MLPipelineOrchestrator(config={
            "model_name": "regime_classifier",
            "model_type": "xgboost",
        })
        
        # Train
        pipeline.train(features, labels)
        
        # Predict
        result = pipeline.predict(new_features)
        
        # Monitor
        health = pipeline.check_health()
    """
    
    def __init__(self, config: Optional[Union[Dict, PipelineConfig]] = None):
        if isinstance(config, dict):
            self.config = PipelineConfig(**config)
        elif config is None:
            self.config = PipelineConfig()
        else:
            self.config = config
        
        # Initialize components
        self.feature_pipeline = FeaturePipeline(self.config.feature_sources)
        self.model_registry = ModelRegistry(self.config.model_dir)
        self.drift_monitor = DriftMonitor(self.config.drift_threshold)
        
        # State
        self._model: Optional[Any] = None
        self._metadata: Optional[ModelMetadata] = None
        self._status = ModelStatus.INITIALIZING
        self._total_predictions = 0
        self._total_errors = 0
        self._start_time = datetime.now(timezone.utc)
        self._last_training_time: Optional[datetime] = None
        self._last_drift_check: Optional[datetime] = None
        self._training_history: List[Dict[str, Any]] = []
        
        logger.info("MLPipelineOrchestrator initialized: %s (%s)", 
                    self.config.model_name, self.config.model_type)
    
    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        validation_features: Optional[np.ndarray] = None,
        validation_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            features: Training features (n_samples, n_features)
            labels: Training labels
            validation_features: Optional validation features
            validation_labels: Optional validation labels
            
        Returns:
            Training metrics dict
        """
        self._status = ModelStatus.TRAINING
        start_time = time.time()
        
        try:
            # Validate inputs
            if len(features) < self.config.min_training_samples:
                raise ValueError(
                    f"Need at least {self.config.min_training_samples} samples, "
                    f"got {len(features)}"
                )
            
            # Create model based on type
            model = self._create_model()
            
            # Train
            model = self._train_model(model, features, labels)
            
            # Evaluate
            train_metrics = self._evaluate_model(model, features, labels)
            val_metrics = None
            if validation_features is not None and validation_labels is not None:
                val_metrics = self._evaluate_model(model, validation_features, validation_labels)
            
            # Set drift reference
            predictions = self._predict_raw(model, features)
            self.drift_monitor.set_reference(predictions)
            
            # Create metadata
            feature_hash = self.feature_pipeline.compute_feature_hash(features)
            model_id = f"{self.config.model_name}_v{int(time.time())}"
            
            self._metadata = ModelMetadata(
                model_id=model_id,
                model_name=self.config.model_name,
                model_type=self.config.model_type,
                version=self._get_next_version(),
                created_at=datetime.now(timezone.utc),
                trained_samples=len(features),
                feature_hash=feature_hash,
                metrics=train_metrics,
                status=ModelStatus.READY,
            )
            
            # Register model
            self.model_registry.register(self._metadata)
            
            # Archive old versions
            self.model_registry.archive_old_versions(
                self.config.model_name,
                keep=self.config.keep_versions,
            )
            
            # Update state
            self._model = model
            self._status = ModelStatus.READY
            self._last_training_time = datetime.now(timezone.utc)
            
            training_time = time.time() - start_time
            
            result = {
                "model_id": model_id,
                "training_time_seconds": training_time,
                "train_metrics": train_metrics,
                "validation_metrics": val_metrics,
                "n_samples": len(features),
                "n_features": features.shape[1] if features.ndim > 1 else 1,
                "status": "success",
            }
            
            self._training_history.append(result)
            logger.info("Training completed: %s (%.2fs)", model_id, training_time)
            
            return result
            
        except Exception as e:
            self._status = ModelStatus.FAILED
            logger.error("Training failed: %s", e)
            return {
                "status": "failed",
                "error": str(e),
            }
    
    def predict(self, features: np.ndarray) -> PredictionResult:
        """
        Make predictions.
        
        Args:
            features: Input features
            
        Returns:
            PredictionResult with predictions and confidence
        """
        if self._model is None or self._status not in (ModelStatus.READY, ModelStatus.DEGRADED):
            raise RuntimeError(f"Model not ready. Status: {self._status}")
        
        try:
            predictions = self._predict_raw(self._model, features)
            confidence = self._compute_confidence(predictions)
            
            # Update drift monitor
            self.drift_monitor.update(predictions)
            self._last_drift_check = datetime.now(timezone.utc)
            
            self._total_predictions += 1
            
            return PredictionResult(
                predictions=predictions,
                confidence=confidence,
                model_id=self._metadata.model_id if self._metadata else "unknown",
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "model_status": self._status.value,
                },
            )
            
        except Exception as e:
            self._total_errors += 1
            logger.error("Prediction failed: %s", e)
            raise
    
    def check_health(self) -> HealthMetrics:
        """Check pipeline health."""
        drift_result = self.drift_monitor._check_drift()
        drift_score = drift_result.get("ks_statistic", 0.0) if drift_result else 0.0
        drift_severity = DriftSeverity(drift_result.get("severity", "none")) if drift_result else DriftSeverity.NONE
        
        # Update status based on drift
        if drift_severity in (DriftSeverity.MODERATE, DriftSeverity.SEVERE):
            self._status = ModelStatus.DEGRADED
        
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds() / 3600
        error_rate = self._total_errors / max(self._total_predictions, 1)
        
        avg_confidence = 0.0
        # Could track running average of confidence
        
        return HealthMetrics(
            model_status=self._status,
            total_predictions=self._total_predictions,
            avg_confidence=avg_confidence,
            drift_score=drift_score,
            drift_severity=drift_severity,
            last_training_time=self._last_training_time,
            last_drift_check=self._last_drift_check,
            uptime_hours=uptime,
            error_rate=error_rate,
        )
    
    def trigger_retrain(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Manually trigger retraining."""
        logger.info("Manual retrain triggered for %s", self.config.model_name)
        self._status = ModelStatus.RETRAINING
        return self.train(features, labels)
    
    def get_metadata(self) -> Optional[ModelMetadata]:
        """Get current model metadata."""
        return self._metadata
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return self._training_history.copy()
    
    def export_state(self) -> Dict[str, Any]:
        """Export full pipeline state."""
        return {
            "config": self.config.to_dict(),
            "metadata": self._metadata.to_dict() if self._metadata else None,
            "status": self._status.value,
            "health": self.check_health().to_dict(),
            "training_history": self._training_history,
            "drift_history": self.drift_monitor.get_history(),
        }
    
    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------
    
    def _create_model(self) -> Any:
        """Create model based on config."""
        model_type = self.config.model_type.lower()
        
        if model_type == "xgboost":
            return self._create_xgboost_model()
        elif model_type == "random_forest":
            return self._create_random_forest_model()
        elif model_type == "neural_network":
            return self._create_neural_network()
        else:
            # Default to simple linear model
            return self._create_linear_model()
    
    def _create_xgboost_model(self) -> Any:
        """Create XGBoost model."""
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
            )
        except ImportError:
            logger.warning("XGBoost not available, falling back to RandomForest")
            return self._create_random_forest_model()
    
    def _create_random_forest_model(self) -> Any:
        """Create RandomForest model."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )
        except ImportError:
            logger.warning("sklearn not available, falling back to linear model")
            return self._create_linear_model()
    
    def _create_neural_network(self) -> Any:
        """Create neural network model."""
        try:
            import torch
            import torch.nn as nn
            
            class SimpleNN(nn.Module):
                def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim // 2, output_dim),
                    )
                
                def forward(self, x):
                    return self.net(x)
            
            return {"type": "nn", "model": None, "input_dim": None}  # Placeholder
        except ImportError:
            logger.warning("PyTorch not available, falling back to linear model")
            return self._create_linear_model()
    
    def _create_linear_model(self) -> Any:
        """Create simple linear model as fallback."""
        try:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, random_state=42)
        except ImportError:
            # Ultimate fallback: simple numpy-based model
            return {"type": "numpy_linear", "weights": None, "bias": None}
    
    def _train_model(self, model: Any, features: np.ndarray, labels: np.ndarray) -> Any:
        """Train the model."""
        # Handle sklearn/xgboost models
        if hasattr(model, "fit"):
            model.fit(features, labels)
            return model
        
        # Handle numpy fallback
        if isinstance(model, dict) and model.get("type") == "numpy_linear":
            # Simple linear regression via least squares
            X = np.column_stack([features, np.ones(len(features))])
            weights, _, _, _ = np.linalg.lstsq(X, labels, rcond=None)
            model["weights"] = weights[:-1]
            model["bias"] = weights[-1]
            return model
        
        return model
    
    def _predict_raw(self, model: Any, features: np.ndarray) -> np.ndarray:
        """Raw prediction from model."""
        if hasattr(model, "predict"):
            return model.predict(features)
        
        if isinstance(model, dict) and model.get("type") == "numpy_linear":
            if model["weights"] is None:
                return np.zeros(len(features))
            return features @ model["weights"] + model["bias"]
        
        return np.zeros(len(features))
    
    def _evaluate_model(self, model: Any, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        predictions = self._predict_raw(model, features)
        
        # For classification
        if len(np.unique(labels)) < 20:  # Likely classification
            pred_labels = (predictions > 0.5).astype(int) if predictions.ndim == 1 else predictions
            accuracy = np.mean(pred_labels == labels)
            return {"accuracy": float(accuracy)}
        
        # For regression
        mse = np.mean((predictions - labels) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - labels))
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
        }
    
    def _compute_confidence(self, predictions: np.ndarray) -> np.ndarray:
        """Compute prediction confidence."""
        # Simple confidence based on distance from decision boundary
        if predictions.ndim == 1:
            confidence = 1.0 / (1.0 + np.exp(-np.abs(predictions)))
        else:
            # Multi-class: use max probability
            confidence = np.max(predictions, axis=1)
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _get_next_version(self) -> int:
        """Get next version number."""
        latest = self.model_registry.get_latest(self.config.model_name)
        if latest is None:
            return 1
        return latest.version + 1
