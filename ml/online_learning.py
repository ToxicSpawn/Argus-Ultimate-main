"""
Argus Trading System - Online Learning with Concept Drift Detection
===================================================================

Implements online/incremental learning algorithms with concept drift detection
for adaptive trading models.

Classes:
    OnlineLearner: Base online learner with update/predict/partial_fit/warm_up
    IncrementalLinearRegression: RLS and SGD implementations
    DriftDetector: ADWIN and Page-Hinkley drift detection
    EnsembleOnlineLearner: Multi-learner ensemble with weighted voting
    FeatureImportanceTracker: Track and detect feature drift
    ModelPerformanceTracker: Rolling performance metrics
    AdaptiveLearningManager: Orchestrates learning with auto-retrain

Uses NumPy. Follows Argus conventions: logger, type hints, dataclasses.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PerformanceRecord:
    """Record of a single prediction performance."""
    timestamp: float
    y_true: float
    y_pred: float
    error: float
    squared_error: float

@dataclass
class LearningStatus:
    """Current status of the adaptive learning system."""
    is_drifting: bool
    drift_confidence: float
    learner_state: str
    performance_score: float
    last_update: float
    retrain_count: int
    samples_seen: int

@dataclass
class LearningResult:
    """Result of a learning cycle."""
    updated: bool
    drift_detected: bool
    drift_confidence: float
    performance_delta: float
    retrain_triggered: bool
    metrics: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# 1. OnlineLearner
# =============================================================================

class OnlineLearner:
    """
    Base online learner with incremental learning capabilities.

    Supports update(), predict(), partial_fit(), and warm_up() methods.
    Wraps an underlying incremental model (defaults to IncrementalLinearRegression).
    """

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.01,
        forgetting_factor: float = 1.0,
        method: str = "sgd",
    ):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.forgetting_factor = forgetting_factor
        self.method = method.lower()
        self._model = IncrementalLinearRegression(
            n_features=n_features,
            learning_rate=learning_rate,
            forgetting_factor=forgetting_factor,
            method=self.method,
        )
        self._samples_seen: int = 0
        self._is_warmed_up: bool = False

        logger.info(
            "OnlineLearner initialized: %d features, method=%s, lr=%.4f",
            n_features, self.method, learning_rate,
        )

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Incremental learning update with batch data.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        self._model.update_weights(X, y)
        self._samples_seen += len(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input features.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predictions of shape (n_samples,)
        """
        X = np.atleast_2d(X)
        return self._model.predict(X)

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> OnlineLearner:
        """
        Fit model incrementally with a single sample or mini-batch.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            self
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        self._model.partial_fit(X, y)
        self._samples_seen += len(y)
        self._is_warmed_up = True
        return self

    def warm_up(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Initial training on a batch to establish baseline weights.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        self._model.warm_up(X, y)
        self._samples_seen += len(y)
        self._is_warmed_up = True
        logger.info("OnlineLearner warm-up complete with %d samples", len(y))

    def get_weights(self) -> np.ndarray:
        """Return current model weights."""
        return self._model.weights.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Return learner statistics."""
        return {
            "n_features": self.n_features,
            "method": self.method,
            "samples_seen": self._samples_seen,
            "is_warmed_up": self._is_warmed_up,
            "weight_norm": float(np.linalg.norm(self._model.weights)),
        }


# =============================================================================
# 2. IncrementalLinearRegression
# =============================================================================

class IncrementalLinearRegression:
    """
    Incremental linear regression with RLS and SGD implementations.

    Supports:
    - Recursive Least Squares (RLS) with forgetting factor
    - Stochastic Gradient Descent (SGD) with adaptive learning rate
    """

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.01,
        forgetting_factor: float = 1.0,
        method: str = "sgd",
        regularization: float = 1e-4,
    ):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.forgetting_factor = forgetting_factor
        self.method = method.lower()
        self.regularization = regularization

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # RLS state
        self._P: Optional[np.ndarray] = None  # Inverse covariance matrix
        self._rls_initialized = False

        # SGD state
        self._n_updates: int = 0
        self._gradient_accumulator: np.ndarray = np.zeros(n_features)

        logger.info(
            "IncrementalLinearRegression initialized: method=%s, lambda=%.4f",
            self.method, self.forgetting_factor,
        )

    def warm_up(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Initial batch training to establish baseline.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
        """
        if self.method == "rls":
            self._init_rls(X, y)
        else:
            self._init_sgd(X, y)

    def _init_rls(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initialize RLS with batch least squares."""
        n = X.shape[0]
        delta = 1.0
        self._P = np.eye(self.n_features) / delta
        self.weights = np.zeros(self.n_features)

        for i in range(n):
            x_i = X[i]
            y_i = y[i]
            self._rls_update(x_i, y_i)

        self._rls_initialized = True

    def _init_sgd(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initialize SGD with multiple passes over data."""
        n_epochs = 10
        for _ in range(n_epochs):
            indices = np.random.permutation(len(y))
            for idx in indices:
                self._sgd_update(X[idx], y[idx])

    def update_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Update weights with batch data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        for i in range(len(y)):
            if self.method == "rls":
                self._rls_update(X[i], y[i])
            else:
                self._sgd_update(X[i], y[i])

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit with single sample or mini-batch.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        self.update_weights(X, y)

    def _rls_update(self, x: np.ndarray, y: float) -> None:
        """
        Recursive Least Squares update with forgetting factor.

        RLS equations:
            k = P x / (lambda + x^T P x)
            e = y - w^T x
            w = w + k e
            P = (P - k x^T P) / lambda
        """
        if not self._rls_initialized:
            self._P = np.eye(self.n_features)
            self._rls_initialized = True

        lambda_ = self.forgetting_factor
        Px = self._P @ x
        denominator = lambda_ + x @ Px
        k = Px / denominator

        error = y - (self.weights @ x + self.bias)
        self.weights += k * error
        self.bias += k.mean() * error

        self._P = (self._P - np.outer(k, x @ self._P)) / lambda_

    def _sgd_update(self, x: np.ndarray, y: float) -> None:
        """
        Stochastic Gradient Descent update.

        SGD equations:
            pred = w^T x + b
            error = y - pred
            w = w + lr * (error * x - reg * w)
            b = b + lr * error
        """
        pred = self.weights @ x + self.bias
        error = y - pred

        lr = self.learning_rate / (1 + self._n_updates * 1e-5)
        self.weights += lr * (error * x - self.regularization * self.weights)
        self.bias += lr * error

        self._n_updates += 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predictions (n_samples,)
        """
        return X @ self.weights + self.bias

    def get_state(self) -> Dict[str, Any]:
        """Return model state for serialization."""
        return {
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "n_updates": self._n_updates,
            "method": self.method,
            "rls_initialized": self._rls_initialized,
        }


# =============================================================================
# 3. DriftDetector
# =============================================================================

class DriftDetector:
    """
    Concept drift detector using ADWIN and Page-Hinkley tests.

    Combines two drift detection algorithms for robust detection:
    - ADWIN (Adaptive Windowing): Automatically adjusts window size
    - Page-Hinkley: Detects changes in mean of error sequence
    """

    def __init__(
        self,
        adwin_delta: float = 0.002,
        ph_delta: float = 0.005,
        ph_threshold: float = 50.0,
        min_samples: int = 30,
    ):
        self.adwin_delta = adwin_delta
        self.ph_delta = ph_delta
        self.ph_threshold = ph_threshold
        self.min_samples = min_samples

        # ADWIN state
        self._adwin_window: Deque[float] = deque()
        self._adwin_total: float = 0.0
        self._adwin_n: int = 0

        # Page-Hinkley state
        self._ph_sum: float = 0.0
        self._ph_mean: float = 0.0
        self._ph_max_sum: float = 0.0
        self._ph_n: int = 0

        # Drift tracking
        self._drift_detected: bool = False
        self._drift_confidence: float = 0.0
        self._total_samples: int = 0
        self._drift_count: int = 0

        logger.info(
            "DriftDetector initialized: ADWIN delta=%.4f, PH threshold=%.1f",
            adwin_delta, ph_threshold,
        )

    def detect_drift(self, new_sample: float) -> bool:
        """
        Check if concept drift has occurred with new sample.

        Args:
            new_sample: New error/performance sample

        Returns:
            True if drift detected
        """
        self._total_samples += 1
        self._drift_detected = False

        adwin_drift = self._check_adwin(new_sample)
        ph_drift = self._check_page_hinkley(new_sample)

        if adwin_drift or ph_drift:
            self._drift_detected = True
            self._drift_count += 1
            self._drift_confidence = min(1.0, (
                (0.6 if adwin_drift else 0.0) +
                (0.4 if ph_drift else 0.0)
            ))
            logger.warning(
                "Drift detected: ADWIN=%s, PH=%s, confidence=%.2f",
                adwin_drift, ph_drift, self._drift_confidence,
            )
        else:
            self._drift_confidence *= 0.95

        return self._drift_detected

    def _check_adwin(self, new_sample: float) -> bool:
        """
        ADWIN (Adaptive Windowing) algorithm.

        Maintains a sliding window and detects drift by comparing
        sub-window means using Hoeffding's inequality.
        """
        self._adwin_window.append(new_sample)
        self._adwin_total += new_sample
        self._adwin_n += 1

        if self._adwin_n < 2 * self.min_samples:
            return False

        n = len(self._adwin_window)
        samples = list(self._adwin_window)

        best_cut = -1
        max_epsilon = 0.0

        for cut in range(self.min_samples, n - self.min_samples, max(1, n // 50)):
            n0 = cut
            n1 = n - cut
            sum0 = sum(samples[:cut])
            sum1 = sum(samples[cut:])

            mean0 = sum0 / n0
            mean1 = sum1 / n1

            n_harmonic = 2.0 / (1.0 / n0 + 1.0 / n1)
            epsilon = np.sqrt((1.0 / n_harmonic) * np.log(2.0 / self.adwin_delta))

            if abs(mean0 - mean1) > epsilon:
                if abs(mean0 - mean1) > max_epsilon:
                    max_epsilon = abs(mean0 - mean1)
                    best_cut = cut

        if best_cut > 0:
            for _ in range(best_cut):
                removed = self._adwin_window.popleft()
                self._adwin_total -= removed
                self._adwin_n -= 1
            return True

        return False

    def _check_page_hinkley(self, new_sample: float) -> bool:
        """
        Page-Hinkley test for mean change detection.

        Detects when the cumulative sum of deviations exceeds threshold.
        """
        self._ph_n += 1
        self._ph_mean += (new_sample - self._ph_mean) / self._ph_n
        self._ph_sum += new_sample - self._ph_mean - self.ph_delta
        self._ph_max_sum = max(self._ph_max_sum, self._ph_sum)

        if self._ph_n < self.min_samples:
            return False

        ph_statistic = self._ph_max_sum - self._ph_sum
        return ph_statistic > self.ph_threshold

    def get_drift_confidence(self) -> float:
        """Return current drift confidence level [0, 1]."""
        return self._drift_confidence

    def reset(self) -> None:
        """Reset all detector state."""
        self._adwin_window.clear()
        self._adwin_total = 0.0
        self._adwin_n = 0

        self._ph_sum = 0.0
        self._ph_mean = 0.0
        self._ph_max_sum = 0.0
        self._ph_n = 0

        self._drift_detected = False
        self._drift_confidence = 0.0

        logger.info("DriftDetector reset")

    def get_stats(self) -> Dict[str, Any]:
        """Return detector statistics."""
        return {
            "total_samples": self._total_samples,
            "drift_count": self._drift_count,
            "drift_detected": self._drift_detected,
            "drift_confidence": self._drift_confidence,
            "adwin_window_size": len(self._adwin_window),
        }


# =============================================================================
# 4. EnsembleOnlineLearner
# =============================================================================

class EnsembleOnlineLearner:
    """
    Ensemble of online learners with weighted voting.

    Features:
    - Multiple base learners with different configurations
    - Weights adjusted based on recent performance
    - Dynamic learner addition/removal
    """

    def __init__(
        self,
        n_features: int,
        n_learners: int = 5,
        performance_window: int = 100,
        min_weight: float = 0.05,
        removal_threshold: float = 0.02,
    ):
        self.n_features = n_features
        self.n_learners = n_learners
        self.performance_window = performance_window
        self.min_weight = min_weight
        self.removal_threshold = removal_threshold

        self._learners: List[OnlineLearner] = []
        self._weights: List[float] = []
        self._recent_errors: List[Deque[float]] = []
        self._total_updates: int = 0

        self._init_learners()

        logger.info(
            "EnsembleOnlineLearner initialized: %d learners, window=%d",
            n_learners, performance_window,
        )

    def _init_learners(self) -> None:
        """Initialize diverse base learners."""
        configs = [
            {"method": "sgd", "learning_rate": 0.001, "forgetting_factor": 1.0},
            {"method": "sgd", "learning_rate": 0.01, "forgetting_factor": 0.99},
            {"method": "sgd", "learning_rate": 0.05, "forgetting_factor": 0.95},
            {"method": "rls", "learning_rate": 0.01, "forgetting_factor": 0.98},
            {"method": "rls", "learning_rate": 0.001, "forgetting_factor": 1.0},
        ]

        for i in range(self.n_learners):
            cfg = configs[i % len(configs)]
            learner = OnlineLearner(
                n_features=self.n_features,
                learning_rate=cfg["learning_rate"],
                forgetting_factor=cfg["forgetting_factor"],
                method=cfg["method"],
            )
            self._learners.append(learner)
            self._weights.append(1.0 / self.n_learners)
            self._recent_errors.append(deque(maxlen=self.performance_window))

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update all learners and adjust weights based on performance."""
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        predictions = []
        for learner in self._learners:
            learner.update(X, y)
            preds = learner.predict(X)
            predictions.append(preds)

        self._update_weights(y, predictions)
        self._total_updates += 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Weighted ensemble prediction.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Weighted average prediction (n_samples,)
        """
        X = np.atleast_2d(X)
        total_weight = sum(self._weights)
        if total_weight < 1e-12:
            return np.zeros(X.shape[0])

        weighted_pred = np.zeros(X.shape[0])
        for learner, weight in zip(self._learners, self._weights):
            weighted_pred += weight * learner.predict(X)

        return weighted_pred / total_weight

    def _update_weights(self, y: np.ndarray, predictions: List[np.ndarray]) -> None:
        """Update learner weights based on recent performance."""
        for i, preds in enumerate(predictions):
            errors = np.abs(y - preds)
            mean_error = float(np.mean(errors))
            self._recent_errors[i].append(mean_error)

        if len(self._recent_errors[0]) < 10:
            return

        avg_errors = []
        for err_deque in self._recent_errors:
            avg_errors.append(float(np.mean(list(err_deque))))

        min_err = min(avg_errors)
        if min_err < 1e-12:
            min_err = 1e-12

        inverse_errors = [1.0 / max(e, min_err) for e in avg_errors]
        total_inv = sum(inverse_errors)

        new_weights = [inv / total_inv for inv in inverse_errors]
        new_weights = [max(w, self.min_weight) for w in new_weights]
        total_w = sum(new_weights)
        self._weights = [w / total_w for w in new_weights]

    def add_learner(self, learner: OnlineLearner) -> None:
        """Dynamically add a new learner to the ensemble."""
        self._learners.append(learner)
        self._weights.append(self.min_weight)
        self._recent_errors.append(deque(maxlen=self.performance_window))

        total = sum(self._weights)
        self._weights = [w / total for w in self._weights]
        logger.info("Added learner to ensemble, total: %d", len(self._learners))

    def remove_poor_learners(self) -> int:
        """Remove learners with weights below threshold."""
        to_remove = []
        for i, w in enumerate(self._weights):
            if w < self.removal_threshold and len(self._learners) > 2:
                to_remove.append(i)

        for i in reversed(to_remove):
            self._learners.pop(i)
            self._weights.pop(i)
            self._recent_errors.pop(i)

        if self._weights:
            total = sum(self._weights)
            self._weights = [w / total for w in self._weights]

        if to_remove:
            logger.info("Removed %d poor learners", len(to_remove))

        return len(to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Return ensemble statistics."""
        return {
            "n_learners": len(self._learners),
            "weights": self._weights.copy(),
            "total_updates": self._total_updates,
        }


# =============================================================================
# 5. FeatureImportanceTracker
# =============================================================================

class FeatureImportanceTracker:
    """
    Track feature importance and detect feature drift.

    Monitors which features contribute most to predictions
    and detects when feature importance shifts over time.
    """

    def __init__(
        self,
        n_features: int,
        feature_names: Optional[List[str]] = None,
        history_window: int = 500,
    ):
        self.n_features = n_features
        self.feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]
        self.history_window = history_window

        self._importance_history: Deque[np.ndarray] = deque(maxlen=history_window)
        self._current_importance: np.ndarray = np.zeros(n_features)
        self._baseline_importance: Optional[np.ndarray] = None
        self._n_updates: int = 0

    def track_feature_importance(
        self,
        learner: OnlineLearner,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute and track feature importance for current batch.

        Uses weight magnitude and gradient-based importance.

        Args:
            learner: OnlineLearner instance
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)

        Returns:
            Dict mapping feature names to importance scores
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        weights = learner.get_weights()

        weight_importance = np.abs(weights)
        if weight_importance.sum() > 0:
            weight_importance = weight_importance / weight_importance.sum()

        grad_importance = np.zeros(self.n_features)
        preds = learner.predict(X)
        errors = y - preds

        for i in range(min(len(y), 100)):
            grad_importance += np.abs(errors[i] * X[i])

        if grad_importance.sum() > 0:
            grad_importance = grad_importance / grad_importance.sum()

        combined = 0.6 * weight_importance + 0.4 * grad_importance
        self._current_importance = combined
        self._importance_history.append(combined.copy())
        self._n_updates += 1

        if self._baseline_importance is None and len(self._importance_history) >= 50:
            self._baseline_importance = np.mean(
                list(self._importance_history), axis=0
            )

        return {
            name: float(importance)
            for name, importance in zip(self.feature_names, combined)
        }

    def detect_feature_drift(self) -> List[str]:
        """
        Detect features whose importance has drifted significantly.

        Returns:
            List of feature names with drifted importance
        """
        if self._baseline_importance is None or len(self._importance_history) < 100:
            return []

        recent = list(self._importance_history)[-100:]
        recent_mean = np.mean(recent, axis=0)
        recent_std = np.std(recent, axis=0) + 1e-12

        drifted = []
        for i in range(self.n_features):
            baseline_val = self._baseline_importance[i]
            recent_val = recent_mean[i]

            z_score = abs(recent_val - baseline_val) / recent_std[i]
            if z_score > 2.0:
                drifted.append(self.feature_names[i])

        if drifted:
            logger.warning(
                "Feature drift detected for: %s", ", ".join(drifted)
            )

        return drifted

    def get_important_features(self, threshold: float = 0.1) -> List[str]:
        """
        Get features with importance above threshold.

        Args:
            threshold: Minimum importance score

        Returns:
            List of important feature names
        """
        important = []
        for name, importance in zip(self.feature_names, self._current_importance):
            if importance >= threshold:
                important.append(name)

        return sorted(important, key=lambda n: self._get_importance_for_name(n), reverse=True)

    def _get_importance_for_name(self, name: str) -> float:
        """Get importance value for a feature name."""
        if name in self.feature_names:
            idx = self.feature_names.index(name)
            return float(self._current_importance[idx])
        return 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Return tracker statistics."""
        return {
            "n_features": self.n_features,
            "n_updates": self._n_updates,
            "has_baseline": self._baseline_importance is not None,
            "top_features": self.get_important_features(threshold=0.0)[:5],
        }


# =============================================================================
# 6. ModelPerformanceTracker
# =============================================================================

class ModelPerformanceTracker:
    """
    Track model performance over time with rolling metrics.

    Records predictions and computes rolling accuracy,
    detects performance degradation.
    """

    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self._records: Deque[PerformanceRecord] = deque(maxlen=max_history)
        self._cumulative_error: float = 0.0
        self._cumulative_squared_error: float = 0.0

    def track_prediction(self, y_true: float, y_pred: float) -> None:
        """
        Record a prediction result.

        Args:
            y_true: Actual value
            y_pred: Predicted value
        """
        error = y_true - y_pred
        record = PerformanceRecord(
            timestamp=time.time(),
            y_true=float(y_true),
            y_pred=float(y_pred),
            error=float(error),
            squared_error=float(error ** 2),
        )
        self._records.append(record)
        self._cumulative_error += abs(error)
        self._cumulative_squared_error += error ** 2

    def compute_rolling_accuracy(self, window: int = 100) -> float:
        """
        Compute accuracy over recent window.

        Uses 1 - normalized MAE as accuracy metric.

        Args:
            window: Number of recent predictions to consider

        Returns:
            Accuracy score in [0, 1]
        """
        if len(self._records) < window:
            if not self._records:
                return 0.0
            window = len(self._records)

        recent = list(self._records)[-window:]
        mae = np.mean([abs(r.error) for r in recent])

        y_range = max(
            abs(max(r.y_true for r in recent)),
            abs(min(r.y_true for r in recent)),
            1e-12,
        )

        return max(0.0, 1.0 - mae / y_range)

    def detect_performance_degradation(self, threshold: float = 0.1) -> bool:
        """
        Detect if performance has degraded beyond threshold.

        Compares recent performance against historical baseline.

        Args:
            threshold: Degradation threshold (fraction)

        Returns:
            True if degradation detected
        """
        if len(self._records) < 200:
            return False

        recent_window = min(100, len(self._records) // 4)
        recent = list(self._records)[-recent_window:]
        older = list(self._records)[:recent_window]

        recent_mae = np.mean([abs(r.error) for r in recent])
        older_mae = np.mean([abs(r.error) for r in older])

        if older_mae < 1e-12:
            return recent_mae > 1e-12

        degradation = (recent_mae - older_mae) / older_mae
        return degradation > threshold

    def get_performance_history(self) -> List[PerformanceRecord]:
        """Return all recorded performance records."""
        return list(self._records)

    def get_stats(self) -> Dict[str, Any]:
        """Return performance statistics."""
        if not self._records:
            return {"n_records": 0}

        errors = [abs(r.error) for r in self._records]
        return {
            "n_records": len(self._records),
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "max_error": float(np.max(errors)),
            "min_error": float(np.min(errors)),
            "rolling_accuracy_100": self.compute_rolling_accuracy(100),
            "rolling_accuracy_500": self.compute_rolling_accuracy(500),
        }


# =============================================================================
# 7. AdaptiveLearningManager
# =============================================================================

class AdaptiveLearningManager:
    """
    Manages online learning with automatic drift detection and retraining.

    Orchestrates:
    - Learner updates with drift monitoring
    - Auto-retrain when performance degrades
    - Learning status tracking
    - Drift parameter configuration
    """

    def __init__(
        self,
        n_features: int,
        retrain_threshold: float = 0.15,
        performance_window: int = 200,
        max_retrain_interval: int = 1000,
    ):
        self.n_features = n_features
        self.retrain_threshold = retrain_threshold
        self.performance_window = performance_window
        self.max_retrain_interval = max_retrain_interval

        self._learner = OnlineLearner(n_features=n_features)
        self._drift_detector = DriftDetector()
        self._performance_tracker = ModelPerformanceTracker()
        self._feature_tracker = FeatureImportanceTracker(n_features=n_features)

        self._samples_since_retrain: int = 0
        self._retrain_count: int = 0
        self._last_update_time: float = time.time()
        self._is_drifting: bool = False

        logger.info(
            "AdaptiveLearningManager initialized: features=%d, retrain_threshold=%.2f",
            n_features, retrain_threshold,
        )

    def manage_learning(self, X: np.ndarray, y: np.ndarray) -> LearningResult:
        """
        Full learning cycle: update, detect drift, track performance.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)

        Returns:
            LearningResult with cycle outcomes
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        old_accuracy = self._performance_tracker.compute_rolling_accuracy(
            self.performance_window
        )

        self._learner.update(X, y)

        predictions = self._learner.predict(X)
        errors = np.abs(y - predictions)
        mean_error = float(np.mean(errors))

        for err in errors:
            self._drift_detector.detect_drift(err)

        for yt, yp in zip(y, predictions):
            self._performance_tracker.track_prediction(float(yt), float(yp))

        self._feature_tracker.track_feature_importance(self._learner, X, y)

        new_accuracy = self._performance_tracker.compute_rolling_accuracy(
            self.performance_window
        )
        performance_delta = new_accuracy - old_accuracy

        drift_detected = self._drift_detector.detect_drift(mean_error)
        self._is_drifting = drift_detected

        retrain_triggered = self._should_retrain()
        if retrain_triggered:
            self._retrain(X, y)

        self._samples_since_retrain += len(y)
        self._last_update_time = time.time()

        return LearningResult(
            updated=True,
            drift_detected=drift_detected,
            drift_confidence=self._drift_detector.get_drift_confidence(),
            performance_delta=performance_delta,
            retrain_triggered=retrain_triggered,
            metrics={
                "accuracy": new_accuracy,
                "mean_error": mean_error,
            },
        )

    def auto_retrain_if_needed(self) -> bool:
        """
        Check if retraining is needed and perform it.

        Returns:
            True if retraining was performed
        """
        if self._should_retrain():
            logger.info("Auto-retrain triggered")
            return True
        return False

    def _should_retrain(self) -> bool:
        """Check conditions for retraining."""
        if self._performance_tracker.detect_performance_degradation(
            self.retrain_threshold
        ):
            return True

        if self._samples_since_retrain >= self.max_retrain_interval:
            return True

        if self._is_drifting and self._drift_detector.get_drift_confidence() > 0.8:
            return True

        return False

    def _retrain(self, X: np.ndarray, y: np.ndarray) -> None:
        """Perform full retraining cycle."""
        logger.info("Retraining learner with %d samples", len(y))
        self._learner.warm_up(X, y)
        self._drift_detector.reset()
        self._samples_since_retrain = 0
        self._retrain_count += 1
        self._is_drifting = False

    def get_learning_status(self) -> LearningStatus:
        """Return current learning system status."""
        accuracy = self._performance_tracker.compute_rolling_accuracy(
            self.performance_window
        )

        return LearningStatus(
            is_drifting=self._is_drifting,
            drift_confidence=self._drift_detector.get_drift_confidence(),
            learner_state="warmed_up" if self._learner._is_warmed_up else "cold",
            performance_score=accuracy,
            last_update=self._last_update_time,
            retrain_count=self._retrain_count,
            samples_seen=self._learner._samples_seen,
        )

    def configure_drift_params(self, params: Dict[str, float]) -> None:
        """
        Configure drift detection parameters.

        Args:
            params: Dict with keys like 'adwin_delta', 'ph_threshold', etc.
        """
        if "adwin_delta" in params:
            self._drift_detector.adwin_delta = params["adwin_delta"]
        if "ph_delta" in params:
            self._drift_detector.ph_delta = params["ph_delta"]
        if "ph_threshold" in params:
            self._drift_detector.ph_threshold = params["ph_threshold"]
        if "retrain_threshold" in params:
            self.retrain_threshold = params["retrain_threshold"]
        if "max_retrain_interval" in params:
            self.max_retrain_interval = int(params["max_retrain_interval"])

        logger.info("Drift parameters configured: %s", params)

    def get_learner(self) -> OnlineLearner:
        """Return the underlying online learner."""
        return self._learner

    def get_stats(self) -> Dict[str, Any]:
        """Return comprehensive system statistics."""
        return {
            "learning_status": self.get_learning_status().__dict__,
            "drift_detector": self._drift_detector.get_stats(),
            "performance": self._performance_tracker.get_stats(),
            "feature_importance": self._feature_tracker.get_stats(),
            "samples_since_retrain": self._samples_since_retrain,
        }
