"""
adaptive/self_healing_manager.py --- Self-Healing Model Management with Automatic Retraining.

Implements comprehensive model health monitoring, drift detection, automatic retraining,
version management, ensemble health tracking, and alert management for the Argus
algorithmic trading system.

Usage::

    manager = SelfHealingManager(config=config_section)
    manager.register_model("regime_detector", model, config={...})
    health = manager.monitor_all()
    actions = manager.auto_heal()
    report = manager.get_health_report()

Standalone --- no hard imports on the rest of the ARGUS tree at module load.
"""

from __future__ import annotations

import logging
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelHealth:
    """Health metrics for a single model."""

    model_name: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    prediction_drift: float = 0.0
    feature_drift: Dict[str, float] = field(default_factory=dict)
    last_trained: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    training_samples: int = 0
    health_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["last_trained"] = self.last_trained.isoformat()
        return d


class HealthStatus(Enum):
    """Model health status categories."""

    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"

    @staticmethod
    def from_score(score: float) -> "HealthStatus":
        if score > 80:
            return HealthStatus.HEALTHY
        if score >= 60:
            return HealthStatus.WARNING
        if score >= 40:
            return HealthStatus.DEGRADED
        return HealthStatus.CRITICAL


@dataclass
class DriftAlert:
    """Alert for detected concept drift."""

    model_name: str
    drift_type: str
    severity: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class DriftReport:
    """Comprehensive drift report for a model."""

    model_name: str
    concept_drift_detected: bool = False
    feature_drifts: Dict[str, float] = field(default_factory=dict)
    drift_severity: str = "none"
    affected_features: List[str] = field(default_factory=list)
    recommended_action: str = "none"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class RetrainJob:
    """Scheduled retraining job."""

    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_name: str = ""
    trigger: str = "drift"
    priority: int = 5
    estimated_duration: int = 300
    status: str = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d


@dataclass
class RetrainResult:
    """Result of a retraining operation."""

    job_id: str
    model_name: str
    success: bool
    old_metrics: Dict[str, float] = field(default_factory=dict)
    new_metrics: Dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class ModelUpdate:
    """Result of incremental or full retraining."""

    model_name: str
    version: str
    samples_used: int
    training_duration: float
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class ValidationResult:
    """Validation result comparing new vs old model."""

    model_name: str
    new_version: str
    old_version: str
    passes: bool
    metric_deltas: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class ModelVersion:
    """Version metadata for a saved model."""

    version_id: str
    model_name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = False
    file_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d


@dataclass
class VersionComparison:
    """Comparison between two model versions."""

    v1_id: str
    v2_id: str
    metric_deltas: Dict[str, float] = field(default_factory=dict)
    winner: str = ""
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnsembleHealth:
    """Health metrics for an ensemble of models."""

    ensemble_name: str
    overall_score: float = 0.0
    model_scores: Dict[str, float] = field(default_factory=dict)
    diversity_score: float = 0.0
    weak_models: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class ReplacementSuggestion:
    """Suggestion to replace a weak model in an ensemble."""

    ensemble_name: str
    old_model: str
    suggested_model: str
    reason: str
    expected_improvement: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Alert:
    """Alert notification for model issues."""

    alert_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_name: str = ""
    status: str = ""
    severity: str = "info"
    details: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d


@dataclass
class HealingAction:
    """A healing action taken by the self-healing manager."""

    model_name: str
    action: str
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class HealthReport:
    """Overall health report for all registered models."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_models: int = 0
    healthy_models: int = 0
    warning_models: int = 0
    critical_models: int = 0
    actions_taken: List[HealingAction] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        d["actions_taken"] = [a.to_dict() for a in self.actions_taken]
        return d


@dataclass
class HealingRecord:
    """Record of a healing action for history tracking."""

    model_name: str
    action: str
    reason: str
    timestamp: datetime
    success: bool
    duration_seconds: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


# ---------------------------------------------------------------------------
# ModelMonitor
# ---------------------------------------------------------------------------

class ModelMonitor:
    """Monitors model health and detects drift."""

    def __init__(self, *, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._accuracy_weight: float = float(cfg.get("accuracy_weight", 0.3))
        self._precision_weight: float = float(cfg.get("precision_weight", 0.15))
        self._recall_weight: float = float(cfg.get("recall_weight", 0.15))
        self._sharpe_weight: float = float(cfg.get("sharpe_weight", 0.2))
        self._drawdown_weight: float = float(cfg.get("drawdown_weight", 0.1))
        self._drift_weight: float = float(cfg.get("drift_weight", 0.1))
        self._prediction_drift_threshold: float = float(cfg.get("prediction_drift_threshold", 0.15))
        self._feature_drift_threshold: float = float(cfg.get("feature_drift_threshold", 0.1))
        self._ks_test_threshold: float = float(cfg.get("ks_test_threshold", 0.05))
        self._psi_threshold: float = float(cfg.get("psi_threshold", 0.2))

        logger.info("ModelMonitor initialised")

    def monitor_model(self, model: Any, test_data: Dict[str, Any]) -> ModelHealth:
        """Evaluate model health on test data.

        Parameters
        ----------
        model : object
            Model object with predict() method.
        test_data : dict
            Must contain 'X' (features), 'y' (labels), and optionally
            'predictions', 'actuals', 'returns' for sharpe/drawdown.

        Returns
        -------
        ModelHealth
        """
        model_name = getattr(model, "name", "unknown")
        try:
            X = np.array(test_data.get("X", []))
            y = np.array(test_data.get("y", []))

            if len(X) == 0 or len(y) == 0:
                return ModelHealth(model_name=model_name, health_score=0.0)

            predictions = test_data.get("predictions", model.predict(X))
            predictions = np.array(predictions)

            accuracy = float(np.mean(predictions == y)) if len(y) > 0 else 0.0
            precision, recall = self._compute_precision_recall(predictions, y)

            returns = test_data.get("returns", np.zeros(len(predictions)))
            sharpe = self._compute_sharpe(returns)
            max_dd = self._compute_max_drawdown(returns)

            pred_drift = test_data.get(
                "prediction_drift",
                self.detect_prediction_drift(predictions, test_data.get("baseline_predictions")),
            )

            feature_drift = test_data.get("feature_drift", {})

            health = ModelHealth(
                model_name=model_name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                sharpe=sharpe,
                max_drawdown=max_dd,
                prediction_drift=pred_drift,
                feature_drift=feature_drift,
                last_trained=getattr(model, "last_trained", datetime.now(timezone.utc)),
                training_samples=getattr(model, "training_samples", len(X)),
            )
            health.health_score = self.compute_health_score(health)
            return health

        except Exception:
            logger.exception("Error monitoring model '%s'", model_name)
            return ModelHealth(model_name=model_name, health_score=0.0)

    def compute_health_score(self, health: ModelHealth) -> float:
        """Compute composite health score (0-100)."""
        try:
            accuracy_score = health.accuracy * 100 * self._accuracy_weight
            precision_score = health.precision * 100 * self._precision_weight
            recall_score = health.recall * 100 * self._recall_weight

            sharpe_norm = min(max((health.sharpe + 2) / 4, 0), 1)
            sharpe_score = sharpe_norm * 100 * self._sharpe_weight

            drawdown_norm = min(max(1 - abs(health.max_drawdown) / 0.5, 0), 1)
            drawdown_score = drawdown_norm * 100 * self._drawdown_weight

            drift_penalty = min(health.prediction_drift / self._prediction_drift_threshold, 1)
            drift_score = (1 - drift_penalty) * 100 * self._drift_weight

            score = (
                accuracy_score + precision_score + recall_score
                + sharpe_score + drawdown_score + drift_score
            )
            return float(min(max(score, 0), 100))

        except Exception:
            logger.exception("Error computing health score")
            return 0.0

    def detect_concept_drift(self, model: Any, recent_data: Dict[str, Any]) -> DriftAlert:
        """Detect concept drift using recent data performance.

        Parameters
        ----------
        model : object
            Model with predict() method.
        recent_data : dict
            Must contain 'X', 'y', and optionally 'historical_accuracy'.

        Returns
        -------
        DriftAlert
        """
        model_name = getattr(model, "name", "unknown")
        try:
            X = np.array(recent_data.get("X", []))
            y = np.array(recent_data.get("y", []))

            if len(X) == 0 or len(y) == 0:
                return DriftAlert(
                    model_name=model_name,
                    drift_type="none",
                    severity=0.0,
                )

            predictions = model.predict(X)
            current_accuracy = float(np.mean(np.array(predictions) == y))
            historical_accuracy = float(recent_data.get("historical_accuracy", current_accuracy))

            accuracy_drop = historical_accuracy - current_accuracy
            severity = float(min(max(accuracy_drop / 0.2, 0), 1))

            drift_type = "none"
            if severity > 0.7:
                drift_type = "severe_concept_drift"
            elif severity > 0.4:
                drift_type = "moderate_concept_drift"
            elif severity > 0.2:
                drift_type = "mild_concept_drift"

            return DriftAlert(
                model_name=model_name,
                drift_type=drift_type,
                severity=severity,
                details={
                    "current_accuracy": current_accuracy,
                    "historical_accuracy": historical_accuracy,
                    "accuracy_drop": accuracy_drop,
                },
            )

        except Exception:
            logger.exception("Error detecting concept drift for '%s'", model_name)
            return DriftAlert(
                model_name=model_name,
                drift_type="error",
                severity=1.0,
            )

    def detect_prediction_drift(
        self,
        predictions: np.ndarray,
        baseline: Optional[np.ndarray] = None,
    ) -> float:
        """Detect drift in prediction distribution using KS test.

        Returns
        -------
        float
            Drift score (0-1), higher means more drift.
        """
        try:
            predictions = np.array(predictions).flatten()
            if baseline is None or len(baseline) == 0:
                return 0.0

            baseline = np.array(baseline).flatten()
            if len(predictions) == 0 or len(baseline) == 0:
                return 0.0

            stat, _ = self._ks_test(predictions, baseline)
            return float(min(stat, 1.0))

        except Exception:
            logger.exception("Error detecting prediction drift")
            return 0.0

    def detect_feature_drift(
        self,
        training_features: np.ndarray,
        current_features: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Detect per-feature drift using PSI (Population Stability Index).

        Returns
        -------
        dict
            feature_name -> drift_score
        """
        try:
            training_features = np.array(training_features)
            current_features = np.array(current_features)

            if training_features.ndim == 1:
                training_features = training_features.reshape(-1, 1)
                current_features = current_features.reshape(-1, 1)

            n_features = training_features.shape[1]
            names = feature_names or [f"feature_{i}" for i in range(n_features)]

            drifts = {}
            for i in range(n_features):
                name = names[i] if i < len(names) else f"feature_{i}"
                train_col = training_features[:, i]
                current_col = current_features[:, i]
                drifts[name] = self._compute_psi(train_col, current_col)

            return drifts

        except Exception:
            logger.exception("Error detecting feature drift")
            return {}

    def _compute_precision_recall(
        self, predictions: np.ndarray, actuals: np.ndarray
    ) -> Tuple[float, float]:
        """Compute precision and recall for binary classification."""
        try:
            tp = float(np.sum((predictions == 1) & (actuals == 1)))
            fp = float(np.sum((predictions == 1) & (actuals == 0)))
            fn = float(np.sum((predictions == 0) & (actuals == 1)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return precision, recall

        except Exception:
            return 0.0, 0.0

    def _compute_sharpe(self, returns: np.ndarray) -> float:
        """Compute annualized Sharpe ratio."""
        try:
            returns = np.array(returns).flatten()
            if len(returns) < 2:
                return 0.0
            mean_ret = float(np.mean(returns))
            std_ret = float(np.std(returns))
            if std_ret == 0:
                return 0.0
            return float(mean_ret / std_ret * np.sqrt(252))
        except Exception:
            return 0.0

    def _compute_max_drawdown(self, returns: np.ndarray) -> float:
        """Compute maximum drawdown from returns series."""
        try:
            returns = np.array(returns).flatten()
            if len(returns) < 2:
                return 0.0
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            return float(np.min(drawdowns))
        except Exception:
            return 0.0

    def _ks_test(self, sample1: np.ndarray, sample2: np.ndarray) -> Tuple[float, float]:
        """Kolmogorov-Smirnov two-sample test (pure NumPy implementation).

        Returns
        -------
        tuple
            (statistic, p_value_approx)
        """
        try:
            s1 = np.sort(sample1)
            s2 = np.sort(sample2)
            all_values = np.sort(np.concatenate([s1, s2]))

            n1 = len(s1)
            n2 = len(s2)

            cdf1 = np.searchsorted(s1, all_values, side="right") / n1
            cdf2 = np.searchsorted(s2, all_values, side="right") / n2

            statistic = float(np.max(np.abs(cdf1 - cdf2)))

            en = np.sqrt(n1 * n2 / (n1 + n2))
            p_value = float(max(0, 1 - 2 * np.exp(-2 * statistic**2 * en**2)))

            return statistic, p_value

        except Exception:
            return 0.0, 1.0

    def _compute_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Population Stability Index.

        PSI < 0.1: no significant change
        0.1 <= PSI < 0.2: moderate change
        PSI >= 0.2: significant change
        """
        try:
            baseline = np.array(baseline).flatten()
            current = np.array(current).flatten()

            if len(baseline) == 0 or len(current) == 0:
                return 0.0

            min_val = min(np.min(baseline), np.min(current))
            max_val = max(np.max(baseline), np.max(current))

            if min_val == max_val:
                return 0.0

            bins = np.linspace(min_val, max_val, n_bins + 1)

            base_counts = np.histogram(baseline, bins=bins)[0]
            curr_counts = np.histogram(current, bins=bins)[0]

            base_pct = (base_counts + 1e-6) / (len(baseline) + n_bins * 1e-6)
            curr_pct = (curr_counts + 1e-6) / (len(current) + n_bins * 1e-6)

            psi = float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))
            return max(psi, 0.0)

        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------

class DriftDetector:
    """Multi-algorithm drift detection for ML models.

    Implements:
    - ADWIN (Adaptive Windowing) for concept drift
    - Kolmogorov-Smirnov test for distribution drift
    - PSI (Population Stability Index) for feature drift
    - Page-Hinkley test for mean shift
    """

    def __init__(self, *, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._adwin_delta: float = float(cfg.get("adwin_delta", 0.002))
        self._adwin_min_window: int = int(cfg.get("adwin_min_window", 10))
        self._ks_threshold: float = float(cfg.get("ks_threshold", 0.05))
        self._psi_threshold: float = float(cfg.get("psi_threshold", 0.2))
        self._ph_threshold: float = float(cfg.get("ph_threshold", 5.0))
        self._ph_delta: float = float(cfg.get("ph_delta", 0.005))
        self._n_psi_bins: int = int(cfg.get("n_psi_bins", 10))

        logger.info("DriftDetector initialised")

    def detect_all_drifts(
        self,
        model: Any,
        data: Dict[str, Any],
    ) -> DriftReport:
        """Run all drift detection algorithms and produce a comprehensive report.

        Parameters
        ----------
        model : object
            Model with predict() method.
        data : dict
            Must contain:
            - 'X': current features
            - 'y': current labels
            - 'X_train': training features
            - 'y_train': training labels (optional)
            - 'predictions_history': historical predictions (for ADWIN/Page-Hinkley)
            - 'feature_names': list of feature names (optional)
            - 'errors_history': historical prediction errors (for ADWIN)

        Returns
        -------
        DriftReport
        """
        model_name = getattr(model, "name", "unknown")
        try:
            X = np.array(data.get("X", []))
            X_train = np.array(data.get("X_train", []))
            feature_names = data.get("feature_names")

            feature_drifts = {}
            if len(X) > 0 and len(X_train) > 0:
                feature_drifts = self._detect_feature_drift_psi(
                    X_train, X, feature_names
                )

            concept_drift = False
            errors_history = data.get("errors_history")
            if errors_history is not None:
                concept_drift = self._adwin_test(np.array(errors_history))

            ph_drift = False
            predictions_history = data.get("predictions_history")
            if predictions_history is not None:
                ph_drift = self._page_hinkley_test(np.array(predictions_history))

            all_drifts = list(feature_drifts.values())
            max_drift = max(all_drifts) if all_drifts else 0.0

            if concept_drift or ph_drift:
                max_drift = max(max_drift, 0.5)

            severity = self._classify_drift_severity(max_drift, concept_drift or ph_drift)
            affected = [
                name for name, score in feature_drifts.items()
                if score > self._psi_threshold
            ]

            action = "none"
            if severity == "severe":
                action = "immediate_retrain"
            elif severity == "moderate":
                action = "schedule_retrain"
            elif severity == "mild":
                action = "monitor_closely"

            return DriftReport(
                model_name=model_name,
                concept_drift_detected=concept_drift or ph_drift,
                feature_drifts=feature_drifts,
                drift_severity=severity,
                affected_features=affected,
                recommended_action=action,
            )

        except Exception:
            logger.exception("Error detecting drifts for '%s'", model_name)
            return DriftReport(
                model_name=model_name,
                concept_drift_detected=False,
                drift_severity="none",
                recommended_action="none",
            )

    def _adwin_test(self, errors: np.ndarray) -> bool:
        """ADWIN (Adaptive Windowing) algorithm for concept drift detection.

        Maintains a variable-size window and detects when the mean changes
        significantly, indicating concept drift.
        """
        try:
            errors = np.array(errors).flatten()
            if len(errors) < self._adwin_min_window * 2:
                return False

            window = list(errors[-(self._adwin_min_window * 10):])

            for cut_point in range(self._adwin_min_window, len(window) - self._adwin_min_window):
                w0 = window[:cut_point]
                w1 = window[cut_point:]

                n0 = len(w0)
                n1 = len(w1)
                mean0 = np.mean(w0)
                mean1 = np.mean(w1)

                if abs(mean0 - mean1) < 1e-10:
                    continue

                pooled_var = (np.var(w0) + np.var(w1)) / 2
                if pooled_var < 1e-10:
                    continue

                m = 1.0 / (1.0 / n0 + 1.0 / n1)
                epsilon = np.sqrt(2 * pooled_var * np.log(2.0 / self._adwin_delta) / m)

                if abs(mean0 - mean1) > epsilon:
                    return True

            return False

        except Exception:
            logger.exception("Error in ADWIN test")
            return False

    def _page_hinkley_test(self, values: np.ndarray) -> bool:
        """Page-Hinkley test for detecting mean shifts in a sequence.

        Sensitive to small but persistent changes in the mean.
        """
        try:
            values = np.array(values).flatten()
            if len(values) < 30:
                return False

            cumulative_sum = 0.0
            min_cumulative_sum = 0.0
            running_mean = 0.0

            for i, val in enumerate(values):
                running_mean = running_mean + (val - running_mean) / (i + 1)
                cumulative_sum += val - running_mean - self._ph_delta
                min_cumulative_sum = min(min_cumulative_sum, cumulative_sum)

                if (cumulative_sum - min_cumulative_sum) > self._ph_threshold:
                    return True

            return False

        except Exception:
            logger.exception("Error in Page-Hinkley test")
            return False

    def _detect_feature_drift_psi(
        self,
        X_train: np.ndarray,
        X_current: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Detect per-feature drift using PSI."""
        try:
            if X_train.ndim == 1:
                X_train = X_train.reshape(-1, 1)
                X_current = X_current.reshape(-1, 1)

            n_features = X_train.shape[1]
            names = feature_names or [f"feature_{i}" for i in range(n_features)]

            drifts = {}
            for i in range(n_features):
                name = names[i] if i < len(names) else f"feature_{i}"
                drifts[name] = self._compute_psi(
                    X_train[:, i], X_current[:, i]
                )

            return drifts

        except Exception:
            logger.exception("Error computing feature drift PSI")
            return {}

    def _compute_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Population Stability Index."""
        try:
            baseline = np.array(baseline).flatten()
            current = np.array(current).flatten()

            if len(baseline) == 0 or len(current) == 0:
                return 0.0

            min_val = min(np.min(baseline), np.min(current))
            max_val = max(np.max(baseline), np.max(current))

            if min_val == max_val:
                return 0.0

            bins = np.linspace(min_val, max_val, n_bins + 1)

            base_counts = np.histogram(baseline, bins=bins)[0]
            curr_counts = np.histogram(current, bins=bins)[0]

            base_pct = (base_counts + 1e-6) / (len(baseline) + n_bins * 1e-6)
            curr_pct = (curr_counts + 1e-6) / (len(current) + n_bins * 1e-6)

            psi = float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))
            return max(psi, 0.0)

        except Exception:
            return 0.0

    def _classify_drift_severity(self, max_drift: float, concept_drift: bool) -> str:
        """Classify overall drift severity."""
        if concept_drift and max_drift > self._psi_threshold:
            return "severe"
        if concept_drift or max_drift > self._psi_threshold:
            return "moderate"
        if max_drift > self._psi_threshold * 0.5:
            return "mild"
        return "none"


# ---------------------------------------------------------------------------
# AutoRetrainer
# ---------------------------------------------------------------------------

class AutoRetrainer:
    """Automatic model retraining with scheduling, incremental updates, and validation."""

    def __init__(self, *, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._max_concurrent_jobs: int = int(cfg.get("max_concurrent_jobs", 3))
        self._default_retrain_duration: int = int(cfg.get("default_retrain_duration", 300))
        self._min_improvement_threshold: float = float(cfg.get("min_improvement_threshold", 0.01))
        self._validation_split: float = float(cfg.get("validation_split", 0.2))
        self._job_history: List[RetrainJob] = []
        self._result_history: List[RetrainResult] = []

        logger.info("AutoRetrainer initialised")

    def schedule_retrain(
        self,
        model_name: str,
        trigger: str = "drift",
        priority: int = 5,
        estimated_duration: Optional[int] = None,
    ) -> RetrainJob:
        """Schedule a retraining job.

        Parameters
        ----------
        model_name : str
            Name of the model to retrain.
        trigger : str
            One of: "drift", "schedule", "performance", "manual".
        priority : int
            1 (highest) to 10 (lowest).
        estimated_duration : int, optional
            Estimated duration in seconds.

        Returns
        -------
        RetrainJob
        """
        valid_triggers = {"drift", "schedule", "performance", "manual"}
        if trigger not in valid_triggers:
            logger.warning("Unknown trigger '%s', defaulting to 'drift'", trigger)
            trigger = "drift"

        priority = max(1, min(10, priority))
        duration = estimated_duration or self._default_retrain_duration

        job = RetrainJob(
            model_name=model_name,
            trigger=trigger,
            priority=priority,
            estimated_duration=duration,
        )
        self._job_history.append(job)

        logger.info(
            "Scheduled retrain job %s for '%s' (trigger=%s, priority=%d)",
            job.job_id, model_name, trigger, priority,
        )
        return job

    def execute_retrain(self, job: RetrainJob) -> RetrainResult:
        """Execute a scheduled retraining job.

        Parameters
        ----------
        job : RetrainJob
            The job to execute.

        Returns
        -------
        RetrainResult
        """
        job.status = "running"
        start_time = datetime.now(timezone.utc)

        logger.info("Executing retrain job %s for '%s'", job.job_id, job.model_name)

        try:
            trainer_fn = getattr(self, f"_train_{job.model_name}", None)
            if trainer_fn is None:
                raise ValueError(f"No trainer registered for model '{job.model_name}'")

            old_metrics = trainer_fn.get("old_metrics", {})
            new_metrics = trainer_fn() if callable(trainer_fn) else {}

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()

            result = RetrainResult(
                job_id=job.job_id,
                model_name=job.model_name,
                success=True,
                old_metrics=old_metrics,
                new_metrics=new_metrics,
                duration_seconds=duration,
            )
            job.status = "completed"

        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            result = RetrainResult(
                job_id=job.job_id,
                model_name=job.model_name,
                success=False,
                error=str(e),
                duration_seconds=duration,
            )
            job.status = "failed"
            logger.exception("Retrain job %s failed", job.job_id)

        self._result_history.append(result)
        return result

    def incremental_retrain(
        self,
        model: Any,
        new_data: Dict[str, Any],
    ) -> ModelUpdate:
        """Perform incremental retraining with new data only.

        Parameters
        ----------
        model : object
            Model with partial_fit() or fit() method.
        new_data : dict
            Must contain 'X' and 'y'.

        Returns
        -------
        ModelUpdate
        """
        model_name = getattr(model, "name", "unknown")
        start_time = datetime.now(timezone.utc)

        try:
            X = np.array(new_data.get("X", []))
            y = np.array(new_data.get("y", []))

            if len(X) == 0:
                raise ValueError("No new data provided for incremental retrain")

            if hasattr(model, "partial_fit"):
                model.partial_fit(X, y)
            elif hasattr(model, "fit"):
                existing_X = new_data.get("existing_X", X)
                existing_y = new_data.get("existing_y", y)
                combined_X = np.vstack([np.array(existing_X), X])
                combined_y = np.concatenate([np.array(existing_y), y])
                model.fit(combined_X, combined_y)
            else:
                raise ValueError(f"Model '{model_name}' has no partial_fit or fit method")

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            version = f"incr_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

            metrics = {}
            if hasattr(model, "score"):
                metrics["accuracy"] = float(model.score(X, y))

            update = ModelUpdate(
                model_name=model_name,
                version=version,
                samples_used=len(X),
                training_duration=duration,
                metrics=metrics,
            )

            logger.info(
                "Incremental retrain for '%s' completed: %d samples, %.1fs",
                model_name, len(X), duration,
            )
            return update

        except Exception:
            logger.exception("Incremental retrain failed for '%s'", model_name)
            raise

    def full_retrain(
        self,
        model: Any,
        all_data: Dict[str, Any],
    ) -> ModelUpdate:
        """Perform full retraining with all available data.

        Parameters
        ----------
        model : object
            Model with fit() method.
        all_data : dict
            Must contain 'X' and 'y'.

        Returns
        -------
        ModelUpdate
        """
        model_name = getattr(model, "name", "unknown")
        start_time = datetime.now(timezone.utc)

        try:
            X = np.array(all_data.get("X", []))
            y = np.array(all_data.get("y", []))

            if len(X) == 0:
                raise ValueError("No data provided for full retrain")

            if not hasattr(model, "fit"):
                raise ValueError(f"Model '{model_name}' has no fit method")

            model.fit(X, y)

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            version = f"full_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

            metrics = {}
            if hasattr(model, "score"):
                metrics["accuracy"] = float(model.score(X, y))

            update = ModelUpdate(
                model_name=model_name,
                version=version,
                samples_used=len(X),
                training_duration=duration,
                metrics=metrics,
            )

            logger.info(
                "Full retrain for '%s' completed: %d samples, %.1fs",
                model_name, len(X), duration,
            )
            return update

        except Exception:
            logger.exception("Full retrain failed for '%s'", model_name)
            raise

    def validate_retrain(
        self,
        new_model: Any,
        validation_data: Dict[str, Any],
        old_metrics: Optional[Dict[str, float]] = None,
    ) -> ValidationResult:
        """Validate a retrained model against validation data.

        Parameters
        ----------
        new_model : object
            Newly trained model.
        validation_data : dict
            Must contain 'X' and 'y'.
        old_metrics : dict, optional
            Metrics from the old model for comparison.

        Returns
        -------
        ValidationResult
        """
        model_name = getattr(new_model, "name", "unknown")
        new_version = getattr(new_model, "version", "new")
        old_version = "previous"

        try:
            X = np.array(validation_data.get("X", []))
            y = np.array(validation_data.get("y", []))

            if len(X) == 0:
                return ValidationResult(
                    model_name=model_name,
                    new_version=new_version,
                    old_version=old_version,
                    passes=False,
                    warnings=["No validation data provided"],
                )

            new_metrics = {}
            if hasattr(new_model, "score"):
                new_metrics["accuracy"] = float(new_model.score(X, y))

            predictions = new_model.predict(X)
            new_metrics["precision"], new_metrics["recall"] = self._precision_recall(
                predictions, y
            )

            metric_deltas = {}
            warnings = []
            passes = True

            if old_metrics:
                for key in old_metrics:
                    if key in new_metrics:
                        delta = new_metrics[key] - old_metrics[key]
                        metric_deltas[key] = delta

                        if delta < -self._min_improvement_threshold:
                            warnings.append(
                                f"Metric '{key}' degraded by {abs(delta):.4f}"
                            )
                            passes = False

            result = ValidationResult(
                model_name=model_name,
                new_version=new_version,
                old_version=old_version,
                passes=passes,
                metric_deltas=metric_deltas,
                warnings=warnings,
            )

            logger.info(
                "Validation for '%s': %s (deltas=%s)",
                model_name, "PASSED" if passes else "FAILED", metric_deltas,
            )
            return result

        except Exception:
            logger.exception("Validation failed for '%s'", model_name)
            return ValidationResult(
                model_name=model_name,
                new_version=new_version,
                old_version=old_version,
                passes=False,
                warnings=["Validation error occurred"],
            )

    def _precision_recall(
        self, predictions: np.ndarray, actuals: np.ndarray
    ) -> Tuple[float, float]:
        """Compute precision and recall."""
        try:
            tp = float(np.sum((predictions == 1) & (actuals == 1)))
            fp = float(np.sum((predictions == 1) & (actuals == 0)))
            fn = float(np.sum((predictions == 0) & (actuals == 1)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return precision, recall
        except Exception:
            return 0.0, 0.0

    @property
    def job_history(self) -> List[RetrainJob]:
        return list(self._job_history)

    @property
    def result_history(self) -> List[RetrainResult]:
        return list(self._result_history)


# ---------------------------------------------------------------------------
# ModelVersionManager
# ---------------------------------------------------------------------------

class ModelVersionManager:
    """Manages model versions with save, load, compare, and rollback capabilities."""

    def __init__(self, *, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._max_versions: int = int(cfg.get("max_versions", 10))
        self._storage_path: str = str(cfg.get("storage_path", "./model_versions"))
        self._versions: Dict[str, List[ModelVersion]] = {}
        self._model_store: Dict[str, Any] = {}

        logger.info("ModelVersionManager initialised (max_versions=%d)", self._max_versions)

    def save_version(
        self,
        model: Any,
        metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """Save a model version.

        Parameters
        ----------
        model : object
            Model to save. Must have a 'name' attribute.
        metrics : dict, optional
            Performance metrics for this version.

        Returns
        -------
        str
            Version ID.
        """
        model_name = getattr(model, "name", "unknown")
        version_id = f"{model_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            metrics=metrics or {},
            is_active=True,
        )

        if model_name not in self._versions:
            self._versions[model_name] = []

        for v in self._versions[model_name]:
            v.is_active = False

        self._versions[model_name].append(version)
        self._model_store[version_id] = model

        if len(self._versions[model_name]) > self._max_versions:
            oldest = self._versions[model_name].pop(0)
            self._model_store.pop(oldest.version_id, None)

        logger.info(
            "Saved version %s for model '%s' (metrics=%s)",
            version_id, model_name, metrics,
        )
        return version_id

    def load_version(self, version_id: str) -> Optional[Any]:
        """Load a model by version ID.

        Parameters
        ----------
        version_id : str
            The version ID to load.

        Returns
        -------
        model or None
        """
        model = self._model_store.get(version_id)
        if model is None:
            logger.warning("Version '%s' not found", version_id)
        return model

    def compare_versions(
        self,
        v1_id: str,
        v2_id: str,
    ) -> VersionComparison:
        """Compare two model versions.

        Parameters
        ----------
        v1_id : str
            First version ID.
        v2_id : str
            Second version ID.

        Returns
        -------
        VersionComparison
        """
        v1 = self._find_version(v1_id)
        v2 = self._find_version(v2_id)

        if v1 is None or v2 is None:
            return VersionComparison(
                v1_id=v1_id,
                v2_id=v2_id,
                summary="One or both versions not found",
            )

        metric_deltas = {}
        all_keys = set(v1.metrics.keys()) | set(v2.metrics.keys())

        for key in all_keys:
            m1 = v1.metrics.get(key, 0.0)
            m2 = v2.metrics.get(key, 0.0)
            metric_deltas[key] = m2 - m1

        v1_score = sum(v1.metrics.values()) / max(len(v1.metrics), 1)
        v2_score = sum(v2.metrics.values()) / max(len(v2.metrics), 1)

        if v2_score > v1_score:
            winner = v2_id
            summary = f"Version {v2_id} outperforms {v1_id} (score {v2_score:.3f} vs {v1_score:.3f})"
        elif v1_score > v2_score:
            winner = v1_id
            summary = f"Version {v1_id} outperforms {v2_id} (score {v1_score:.3f} vs {v2_score:.3f})"
        else:
            winner = "tie"
            summary = f"Versions {v1_id} and {v2_id} are equivalent"

        return VersionComparison(
            v1_id=v1_id,
            v2_id=v2_id,
            metric_deltas=metric_deltas,
            winner=winner,
            summary=summary,
        )

    def rollback(self, version_id: str) -> bool:
        """Rollback to a specific version.

        Parameters
        ----------
        version_id : str
            Version to rollback to.

        Returns
        -------
        bool
            True if rollback succeeded.
        """
        version = self._find_version(version_id)
        if version is None:
            logger.warning("Cannot rollback: version '%s' not found", version_id)
            return False

        model_name = version.model_name
        for v in self._versions.get(model_name, []):
            v.is_active = (v.version_id == version_id)

        logger.info("Rolled back model '%s' to version %s", model_name, version_id)
        return True

    def list_versions(self, model_name: str) -> List[ModelVersion]:
        """List all versions for a model.

        Parameters
        ----------
        model_name : str
            Model name.

        Returns
        -------
        list[ModelVersion]
        """
        return list(self._versions.get(model_name, []))

    def auto_rollback_if_worse(
        self,
        new_version_id: str,
        threshold: float = 0.05,
    ) -> bool:
        """Automatically rollback if new version is worse than previous by threshold.

        Parameters
        ----------
        new_version_id : str
            The new version to evaluate.
        threshold : float
            Minimum degradation to trigger rollback (0-1).

        Returns
        -------
        bool
            True if rollback was performed.
        """
        new_version = self._find_version(new_version_id)
        if new_version is None:
            return False

        model_name = new_version.model_name
        versions = self._versions.get(model_name, [])

        if len(versions) < 2:
            return False

        previous_version = None
        for v in reversed(versions):
            if v.version_id != new_version_id:
                previous_version = v
                break

        if previous_version is None:
            return False

        new_score = sum(new_version.metrics.values()) / max(len(new_version.metrics), 1)
        prev_score = sum(previous_version.metrics.values()) / max(len(previous_version.metrics), 1)

        degradation = prev_score - new_score
        if degradation > threshold:
            logger.warning(
                "Auto-rollback triggered for '%s': new version degraded by %.4f",
                model_name, degradation,
            )
            return self.rollback(previous_version.version_id)

        return False

    def _find_version(self, version_id: str) -> Optional[ModelVersion]:
        """Find a version by ID across all models."""
        for versions in self._versions.values():
            for v in versions:
                if v.version_id == version_id:
                    return v
        return None


# ---------------------------------------------------------------------------
# EnsembleHealthManager
# ---------------------------------------------------------------------------

class EnsembleHealthManager:
    """Monitors and manages ensemble model health."""

    def __init__(self, *, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._weak_threshold: float = float(cfg.get("weak_threshold", 50.0))
        self._diversity_min: float = float(cfg.get("diversity_min", 0.1))
        self._monitor = ModelMonitor(config=cfg.get("monitor", {}))

        logger.info("EnsembleHealthManager initialised")

    def monitor_ensemble(self, ensemble: Dict[str, Any]) -> EnsembleHealth:
        """Monitor overall ensemble health.

        Parameters
        ----------
        ensemble : dict
            Must contain:
            - 'name': ensemble name
            - 'models': dict of model_name -> model_object
            - 'model_healths': dict of model_name -> ModelHealth (optional)

        Returns
        -------
        EnsembleHealth
        """
        ensemble_name = ensemble.get("name", "unknown")
        models = ensemble.get("models", {})

        model_scores = {}
        weak_models = []

        for name, model in models.items():
            health = ensemble.get("model_healths", {}).get(name)
            if health is None:
                health = self._monitor.monitor_model(model, ensemble.get("test_data", {}))
            model_scores[name] = health.health_score

            if health.health_score < self._weak_threshold:
                weak_models.append(name)

        overall = float(np.mean(list(model_scores.values()))) if model_scores else 0.0
        diversity = self._compute_diversity(models, ensemble.get("test_data", {}))

        return EnsembleHealth(
            ensemble_name=ensemble_name,
            overall_score=overall,
            model_scores=model_scores,
            diversity_score=diversity,
            weak_models=weak_models,
        )

    def identify_weak_models(self, ensemble: Dict[str, Any]) -> List[str]:
        """Identify weak models in an ensemble.

        Parameters
        ----------
        ensemble : dict
            Ensemble with 'models' and optionally 'model_healths'.

        Returns
        -------
        list[str]
            Names of weak models.
        """
        health = self.monitor_ensemble(ensemble)
        return health.weak_models

    def suggest_model_replacement(
        self,
        ensemble: Dict[str, Any],
    ) -> List[ReplacementSuggestion]:
        """Suggest replacements for weak models.

        Parameters
        ----------
        ensemble : dict
            Ensemble with 'models', 'model_healths', and 'candidate_models'.

        Returns
        -------
        list[ReplacementSuggestion]
        """
        health = self.monitor_ensemble(ensemble)
        candidates = ensemble.get("candidate_models", {})
        suggestions = []

        for weak_model in health.weak_models:
            best_candidate = None
            best_score = 0.0

            for cand_name, cand_model in candidates.items():
                cand_health = self._monitor.monitor_model(
                    cand_model, ensemble.get("test_data", {})
                )
                if cand_health.health_score > best_score:
                    best_score = cand_health.health_score
                    best_candidate = cand_name

            if best_candidate:
                improvement = best_score - health.model_scores.get(weak_model, 0)
                suggestions.append(
                    ReplacementSuggestion(
                        ensemble_name=ensemble.get("name", "unknown"),
                        old_model=weak_model,
                        suggested_model=best_candidate,
                        reason=f"Model '{weak_model}' score {health.model_scores.get(weak_model, 0):.1f} "
                               f"below threshold {self._weak_threshold:.1f}",
                        expected_improvement=improvement,
                    )
                )

        return suggestions

    def auto_replace_model(
        self,
        ensemble: Dict[str, Any],
        old_model_name: str,
        new_model: Any,
    ) -> bool:
        """Replace a model in the ensemble.

        Parameters
        ----------
        ensemble : dict
            Ensemble with 'models' dict.
        old_model_name : str
            Name of model to replace.
        new_model : object
            New model to insert.

        Returns
        -------
        bool
            True if replacement succeeded.
        """
        models = ensemble.get("models", {})
        if old_model_name not in models:
            logger.warning(
                "Cannot replace model '%s': not found in ensemble", old_model_name
            )
            return False

        models[old_model_name] = new_model
        logger.info(
            "Replaced model '%s' in ensemble '%s'",
            old_model_name, ensemble.get("name", "unknown"),
        )
        return True

    def _compute_diversity(
        self,
        models: Dict[str, Any],
        test_data: Dict[str, Any],
    ) -> float:
        """Compute ensemble diversity score (0-1).

        Higher diversity means models make different errors, which is good.
        """
        try:
            X = np.array(test_data.get("X", []))
            if len(X) == 0 or len(models) < 2:
                return 0.0

            predictions = []
            for name, model in models.items():
                try:
                    preds = model.predict(X)
                    predictions.append(np.array(preds).flatten())
                except Exception:
                    continue

            if len(predictions) < 2:
                return 0.0

            disagreements = []
            for i in range(len(predictions)):
                for j in range(i + 1, len(predictions)):
                    disagree = np.mean(predictions[i] != predictions[j])
                    disagreements.append(disagree)

            return float(np.mean(disagreements))

        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------

class AlertManager:
    """Manages alerts for model health issues with cooldown to prevent spam."""

    def __init__(self, *, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._cooldown_seconds: int = int(cfg.get("cooldown_seconds", 3600))
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        self.alert_cooldown: Dict[str, datetime] = {}
        self._alert_history: List[Alert] = []

        logger.info("AlertManager initialised (cooldown=%ds)", self._cooldown_seconds)

    def create_alert(
        self,
        model: Any,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """Create a new alert.

        Parameters
        ----------
        model : object
            Model with 'name' attribute.
        status : str
            Health status string.
        details : dict, optional
            Additional alert details.

        Returns
        -------
        Alert
        """
        model_name = getattr(model, "name", "unknown")
        severity = self.get_alert_severity(status)

        alert = Alert(
            model_name=model_name,
            status=status,
            severity=severity,
            details=details or {},
        )

        self._alert_history.append(alert)
        logger.info(
            "Alert created: [%s] model='%s' status='%s'",
            severity, model_name, status,
        )
        return alert

    def should_alert(
        self,
        current_status: str,
        previous_status: Optional[str] = None,
        model_name: str = "unknown",
    ) -> bool:
        """Check if an alert should be sent (respects cooldown).

        Parameters
        ----------
        current_status : str
            Current model status.
        previous_status : str, optional
            Previous model status.
        model_name : str
            Model name for cooldown tracking.

        Returns
        -------
        bool
            True if alert should be sent.
        """
        cooldown_key = f"{model_name}_{current_status}"
        now = datetime.now(timezone.utc)

        if cooldown_key in self.alert_cooldown:
            last_alert = self.alert_cooldown[cooldown_key]
            elapsed = (now - last_alert).total_seconds()
            if elapsed < self._cooldown_seconds:
                return False

        if previous_status and current_status == previous_status:
            return False

        status_order = {
            "healthy": 0,
            "warning": 1,
            "degraded": 2,
            "critical": 3,
            "failed": 4,
        }

        current_level = status_order.get(current_status, 0)
        previous_level = status_order.get(previous_status or "healthy", 0)

        if current_level <= previous_level and previous_status is not None:
            return False

        return True

    def get_alert_severity(self, status: str) -> str:
        """Map status to alert severity level.

        Parameters
        ----------
        status : str
            Health status.

        Returns
        -------
        str
            Severity level: "info", "warning", "error", "critical".
        """
        severity_map = {
            "healthy": "info",
            "warning": "warning",
            "degraded": "warning",
            "critical": "error",
            "failed": "critical",
        }
        return severity_map.get(status.lower(), "info")

    def send_alert(self, alert: Alert) -> None:
        """Send an alert through registered callbacks.

        Parameters
        ----------
        alert : Alert
            The alert to send.
        """
        cooldown_key = f"{alert.model_name}_{alert.status}"
        now = datetime.now(timezone.utc)
        self.alert_cooldown[cooldown_key] = now

        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception:
                logger.exception("Error sending alert for '%s'", alert.model_name)

        logger.info(
            "Alert sent: [%s] model='%s' status='%s'",
            alert.severity, alert.model_name, alert.status,
        )

    def register_callback(self, callback: Callable[[Alert], None]) -> None:
        """Register an alert callback.

        Parameters
        ----------
        callback : callable
            Function that takes an Alert object.
        """
        self._alert_callbacks.append(callback)
        logger.info("Alert callback registered")

    def clear_cooldown(self, model_name: Optional[str] = None) -> None:
        """Clear alert cooldowns.

        Parameters
        ----------
        model_name : str, optional
            If provided, clear only for this model. Otherwise clear all.
        """
        if model_name:
            keys_to_remove = [
                k for k in self.alert_cooldown
                if k.startswith(f"{model_name}_")
            ]
            for key in keys_to_remove:
                del self.alert_cooldown[key]
        else:
            self.alert_cooldown.clear()

    @property
    def alert_history(self) -> List[Alert]:
        return list(self._alert_history)


# ---------------------------------------------------------------------------
# SelfHealingManager (main class)
# ---------------------------------------------------------------------------

class SelfHealingManager:
    """Main self-healing manager that coordinates all model health components.

    Provides:
    - Model registration and monitoring
    - Automatic health assessment
    - Auto-retraining with validation
    - Auto-rollback on degradation
    - Ensemble health tracking
    - Alert management
    """

    def __init__(self, *, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.enable_auto_retrain: bool = bool(cfg.get("enable_auto_retrain", True))
        self.enable_auto_rollback: bool = bool(cfg.get("enable_auto_rollback", True))

        self._monitor = ModelMonitor(config=cfg.get("monitor", {}))
        self._drift_detector = DriftDetector(config=cfg.get("drift_detector", {}))
        self._retrainer = AutoRetrainer(config=cfg.get("retrainer", {}))
        self._version_manager = ModelVersionManager(config=cfg.get("version_manager", {}))
        self._ensemble_manager = EnsembleHealthManager(
            config=cfg.get("ensemble_manager", {})
        )
        self._alert_manager = AlertManager(config=cfg.get("alert_manager", {}))

        self._registered_models: Dict[str, Dict[str, Any]] = {}
        self._healing_history: List[HealingRecord] = []
        self._last_health_report: Optional[HealthReport] = None

        logger.info(
            "SelfHealingManager initialised (auto_retrain=%s, auto_rollback=%s)",
            self.enable_auto_retrain, self.enable_auto_rollback,
        )

    def register_model(
        self,
        name: str,
        model: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a model for monitoring and self-healing.

        Parameters
        ----------
        name : str
            Unique model name.
        model : object
            Model object.
        config : dict, optional
            Model-specific configuration.
        """
        self._registered_models[name] = {
            "model": model,
            "config": config or {},
            "registered_at": datetime.now(timezone.utc),
            "last_health_check": None,
            "last_health": None,
        }

        logger.info("Model '%s' registered for self-healing", name)

    def monitor_all(self) -> Dict[str, ModelHealth]:
        """Monitor all registered models.

        Returns
        -------
        dict
            model_name -> ModelHealth
        """
        results = {}

        for name, entry in self._registered_models.items():
            model = entry["model"]
            test_data = entry["config"].get("test_data", {})

            health = self._monitor.monitor_model(model, test_data)
            entry["last_health_check"] = datetime.now(timezone.utc)
            entry["last_health"] = health

            results[name] = health

            status = HealthStatus.from_score(health.health_score)
            if self._alert_manager.should_alert(
                status.value,
                entry.get("last_status"),
                name,
            ):
                alert = self._alert_manager.create_alert(
                    model, status.value,
                    {"health_score": health.health_score},
                )
                self._alert_manager.send_alert(alert)

            entry["last_status"] = status.value

        self._last_health_report = self._build_health_report(results, [])
        return results

    def auto_heal(self) -> List[HealingAction]:
        """Automatically heal models based on health assessment.

        Returns
        -------
        list[HealingAction]
            Actions taken.
        """
        actions = []

        for name, entry in self._registered_models.items():
            health = entry.get("last_health")
            if health is None:
                continue

            status = HealthStatus.from_score(health.health_score)
            model = entry["model"]

            if status in (HealthStatus.CRITICAL, HealthStatus.FAILED):
                action = self._handle_critical(name, model, health, entry)
                if action:
                    actions.append(action)

            elif status == HealthStatus.DEGRADED:
                action = self._handle_degraded(name, model, health, entry)
                if action:
                    actions.append(action)

            elif status == HealthStatus.WARNING:
                action = self._handle_warning(name, model, health, entry)
                if action:
                    actions.append(action)

        if actions:
            self._last_health_report = self._build_health_report(
                {n: e.get("last_health") for n, e in self._registered_models.items()},
                actions,
            )

        return actions

    def get_health_report(self) -> HealthReport:
        """Get the latest health report.

        Returns
        -------
        HealthReport
        """
        if self._last_health_report is None:
            health_map = self.monitor_all()
            return self._last_health_report or self._build_health_report(health_map, [])
        return self._last_health_report

    def get_healing_history(self) -> List[HealingRecord]:
        """Get the healing action history.

        Returns
        -------
        list[HealingRecord]
        """
        return list(self._healing_history)

    def _handle_critical(
        self,
        name: str,
        model: Any,
        health: ModelHealth,
        entry: Dict[str, Any],
    ) -> Optional[HealingAction]:
        """Handle critical model state."""
        if self.enable_auto_rollback:
            versions = self._version_manager.list_versions(name)
            if len(versions) >= 2:
                active = [v for v in versions if v.is_active]
                if active:
                    success = self._version_manager.rollback(active[0].version_id)
                    action = HealingAction(
                        model_name=name,
                        action="rollback",
                        reason=f"Critical health (score={health.health_score:.1f}), "
                               f"rolled back to previous version",
                        success=success,
                        details={"health_score": health.health_score},
                    )
                    self._record_healing(action)
                    return action

        if self.enable_auto_retrain:
            job = self._retrainer.schedule_retrain(
                name, trigger="performance", priority=1
            )
            action = HealingAction(
                model_name=name,
                action="retrain",
                reason=f"Critical health (score={health.health_score:.1f}), "
                       f"scheduled high-priority retrain",
                success=True,
                details={"job_id": job.job_id, "health_score": health.health_score},
            )
            self._record_healing(action)
            return action

        action = HealingAction(
            model_name=name,
            action="alert",
            reason=f"Critical health (score={health.health_score:.1f}), "
                   f"no auto-heal options available",
            success=False,
            details={"health_score": health.health_score},
        )
        alert = self._alert_manager.create_alert(
            model, "critical",
            {"health_score": health.health_score, "action": "manual_intervention_required"},
        )
        self._alert_manager.send_alert(alert)
        self._record_healing(action)
        return action

    def _handle_degraded(
        self,
        name: str,
        model: Any,
        health: ModelHealth,
        entry: Dict[str, Any],
    ) -> Optional[HealingAction]:
        """Handle degraded model state."""
        if self.enable_auto_retrain:
            job = self._retrainer.schedule_retrain(
                name, trigger="performance", priority=3
            )
            action = HealingAction(
                model_name=name,
                action="retrain",
                reason=f"Degraded health (score={health.health_score:.1f}), "
                       f"scheduled retrain",
                success=True,
                details={"job_id": job.job_id, "health_score": health.health_score},
            )
            self._record_healing(action)
            return action

        action = HealingAction(
            model_name=name,
            action="alert",
            reason=f"Degraded health (score={health.health_score:.1f})",
            success=False,
            details={"health_score": health.health_score},
        )
        self._record_healing(action)
        return action

    def _handle_warning(
        self,
        name: str,
        model: Any,
        health: ModelHealth,
        entry: Dict[str, Any],
    ) -> Optional[HealingAction]:
        """Handle warning model state."""
        drift_report = self._drift_detector.detect_all_drifts(
            model,
            entry["config"].get("drift_data", {}),
        )

        if drift_report.drift_severity in ("moderate", "severe"):
            if self.enable_auto_retrain:
                job = self._retrainer.schedule_retrain(
                    name, trigger="drift", priority=5
                )
                action = HealingAction(
                    model_name=name,
                    action="retrain",
                    reason=f"Warning health with {drift_report.drift_severity} drift detected",
                    success=True,
                    details={
                        "job_id": job.job_id,
                        "drift_severity": drift_report.drift_severity,
                        "health_score": health.health_score,
                    },
                )
                self._record_healing(action)
                return action

        action = HealingAction(
            model_name=name,
            action="alert",
            reason=f"Warning health (score={health.health_score:.1f})",
            success=False,
            details={"health_score": health.health_score},
        )
        self._record_healing(action)
        return action

    def _build_health_report(
        self,
        health_map: Dict[str, ModelHealth],
        actions: List[HealingAction],
    ) -> HealthReport:
        """Build a comprehensive health report."""
        total = len(health_map)
        healthy = 0
        warning = 0
        critical = 0
        recommendations = []

        for name, health in health_map.items():
            if health is None:
                critical += 1
                continue

            status = HealthStatus.from_score(health.health_score)
            if status == HealthStatus.HEALTHY:
                healthy += 1
            elif status == HealthStatus.WARNING:
                warning += 1
                recommendations.append(
                    f"Model '{name}' showing warning signs (score={health.health_score:.1f})"
                )
            else:
                critical += 1
                recommendations.append(
                    f"Model '{name}' needs immediate attention (score={health.health_score:.1f})"
                )

        if not actions and critical == 0 and warning == 0:
            recommendations.append("All models healthy, no action required")

        return HealthReport(
            total_models=total,
            healthy_models=healthy,
            warning_models=warning,
            critical_models=critical,
            actions_taken=actions,
            recommendations=recommendations,
        )

    def _record_healing(self, action: HealingAction) -> None:
        """Record a healing action in history."""
        record = HealingRecord(
            model_name=action.model_name,
            action=action.action,
            reason=action.reason,
            timestamp=action.timestamp,
            success=action.success,
            details=action.details,
        )
        self._healing_history.append(record)

    @property
    def monitor(self) -> ModelMonitor:
        return self._monitor

    @property
    def drift_detector(self) -> DriftDetector:
        return self._drift_detector

    @property
    def retrainer(self) -> AutoRetrainer:
        return self._retrainer

    @property
    def version_manager(self) -> ModelVersionManager:
        return self._version_manager

    @property
    def ensemble_manager(self) -> EnsembleHealthManager:
        return self._ensemble_manager

    @property
    def alert_manager(self) -> AlertManager:
        return self._alert_manager

    @property
    def registered_models(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._registered_models)
