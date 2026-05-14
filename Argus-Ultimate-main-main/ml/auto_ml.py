"""
AutoML Pipeline — automated model selection and hyperparameter tuning.

This module provides a lightweight, dependency-safe AutoML path for Argus ML
models. It intentionally uses deterministic random search rather than claiming
full Bayesian optimization; if optional libraries are unavailable, it falls back
to a small NumPy baseline so the trading system can still start and tests remain
portable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import cross_val_score

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    RandomForestClassifier = None
    RandomForestRegressor = None
    LogisticRegression = None
    Ridge = None
    cross_val_score = None
    _SKLEARN_AVAILABLE = False
    logger.debug("sklearn not available for AutoML")

try:
    from xgboost import XGBClassifier, XGBRegressor

    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    XGBClassifier = None
    XGBRegressor = None
    _XGB_AVAILABLE = False


class ModelType(Enum):
    """Supported model types."""

    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LOGISTIC_REGRESSION = "logistic_regression"
    RIDGE = "ridge"
    NUMPY_BASELINE = "numpy_baseline"


@dataclass
class TrialResult:
    """Result from a single hyperparameter trial."""

    trial_id: int
    model_type: str
    params: Dict[str, Any]
    train_score: float
    val_score: float
    cv_scores: List[float]
    training_time: float
    n_features: int
    cost_penalty: float = 0.0
    net_score: Optional[float] = None
    error: Optional[str] = None

    @property
    def mean_cv_score(self) -> float:
        return float(np.mean(self.cv_scores)) if self.cv_scores else 0.0

    @property
    def std_cv_score(self) -> float:
        return float(np.std(self.cv_scores)) if self.cv_scores else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "model_type": self.model_type,
            "params": self.params,
            "train_score": round(self.train_score, 4),
            "val_score": round(self.val_score, 4),
            "cost_penalty": round(self.cost_penalty, 6),
            "net_score": round(self.ranking_score, 6),
            "cv_mean": round(self.mean_cv_score, 4),
            "cv_std": round(self.std_cv_score, 4),
            "training_time": round(self.training_time, 2),
            "n_features": self.n_features,
            "error": self.error,
        }

    @property
    def ranking_score(self) -> float:
        return float(self.val_score - self.cost_penalty if self.net_score is None else self.net_score)


@dataclass
class AutoMLResult:
    """Result from AutoML pipeline."""

    best_model: Any
    best_params: Dict[str, Any]
    best_model_type: str
    best_score: float
    all_trials: List[TrialResult]
    feature_importance: Optional[Dict[str, float]]
    training_time: float
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_model_type": self.best_model_type,
            "best_params": self.best_params,
            "best_score": round(self.best_score, 4),
            "n_trials": len(self.all_trials),
            "training_time": round(self.training_time, 2),
            "top_trials": [
                t.to_dict()
                for t in sorted(self.all_trials, key=lambda t: t.ranking_score, reverse=True)[:5]
            ],
            "feature_importance": self.feature_importance,
            "timestamp": self.timestamp.isoformat(),
        }


class NumpyBaselineModel:
    """Small dependency-free baseline for classification or regression."""

    def __init__(self, task_type: str = "classification"):
        self.task_type = task_type
        self.constant_: Any = 0
        self.coef_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyBaselineModel":
        features = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if self.task_type == "classification":
            values, counts = np.unique(y, return_counts=True)
            self.constant_ = values[int(np.argmax(counts))]
            self.coef_ = np.zeros(features.shape[1], dtype=float)
            return self

        y_float = y.astype(float)
        self.constant_ = float(np.mean(y_float)) if len(y_float) else 0.0
        centered = features - np.mean(features, axis=0)
        denom = np.sum(centered * centered, axis=0)
        denom = np.where(denom == 0, 1.0, denom)
        self.coef_ = np.sum(centered * (y_float - self.constant_)[:, None], axis=0) / denom
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        features = np.asarray(X, dtype=float)
        if self.task_type == "classification":
            return np.full(features.shape[0], self.constant_)

        coef = self.coef_ if self.coef_ is not None else np.zeros(features.shape[1], dtype=float)
        return np.full(features.shape[0], self.constant_, dtype=float) + features @ coef


class AutoMLPipeline:
    """
    Lightweight automated model search for Argus ML components.

    Args:
        time_limit_minutes: Maximum search time.
        n_trials: Maximum number of trials.
        metric: Optimization metric: "accuracy", "f1", "sharpe", or "mse".
        cv_folds: Number of cross-validation folds when sklearn is available.
        early_stopping_rounds: Stop after this many non-improving trials.
        task_type: "auto", "classification", or "regression".
    """

    def __init__(
        self,
        time_limit_minutes: int = 30,
        n_trials: int = 100,
        metric: str = "accuracy",
        cv_folds: int = 5,
        early_stopping_rounds: int = 10,
        random_seed: int = 42,
        task_type: str = "auto",
    ):
        self.time_limit = timedelta(minutes=max(time_limit_minutes, 0))
        self.n_trials = max(int(n_trials), 0)
        self.metric = metric
        self.cv_folds = max(int(cv_folds), 2)
        self.early_stopping_rounds = max(int(early_stopping_rounds), 1)
        self.random_seed = random_seed
        self.task_type = task_type

        self._trials: List[TrialResult] = []
        self._rng = np.random.default_rng(random_seed)
        self._resolved_task_type = "classification"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        model_types: Optional[List[str]] = None,
        cost_model: Optional[Callable[[str, Dict[str, Any], float, float], float]] = None,
    ) -> AutoMLResult:
        """Run model selection and return the best fitted model."""
        start_time = time.time()
        X_train = self._validate_X(X_train, "X_train")
        y_train = self._validate_y(y_train, "y_train")
        X_val = self._validate_X(X_val, "X_val") if X_val is not None else None
        y_val = self._validate_y(y_val, "y_val") if y_val is not None else None

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must contain the same number of samples")
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be provided together")
        if X_val is not None and y_val is not None and X_val.shape[0] != y_val.shape[0]:
            raise ValueError("X_val and y_val must contain the same number of samples")

        self._resolved_task_type = self._infer_task_type(y_train)
        candidate_types = self._normalize_model_types(model_types)
        self._trials = []

        best_score = float("-inf")
        best_model = None
        best_params: Dict[str, Any] = {}
        best_model_type = ""
        no_improve_count = 0

        logger.info("Starting AutoML: %d trials, candidates=%s", self.n_trials, candidate_types)

        for trial_id in range(self.n_trials):
            if time.time() - start_time > self.time_limit.total_seconds():
                logger.info("Time limit reached at trial %d", trial_id)
                break

            model_type = candidate_types[trial_id % len(candidate_types)]
            params = self._sample_params(model_type)
            trial = self._run_trial(trial_id, model_type, params, X_train, y_train, X_val, y_val, cost_model)
            self._trials.append(trial)

            if trial.ranking_score > best_score:
                candidate_model = self._fit_model(model_type, params, X_train, y_train)
                if candidate_model is not None:
                    best_score = trial.ranking_score
                    best_params = dict(params)
                    best_model_type = model_type
                    best_model = candidate_model
                    no_improve_count = 0
                    logger.info("Trial %d: new best %s score=%.4f", trial_id, model_type, best_score)
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

            if no_improve_count >= self.early_stopping_rounds:
                logger.info("Early stopping at trial %d", trial_id)
                break

        if best_model is None:
            best_model_type = ModelType.NUMPY_BASELINE.value
            best_params = {"task_type": self._resolved_task_type}
            best_model = self._fit_model(best_model_type, best_params, X_train, y_train)
            fallback_trial = self._run_trial(
                len(self._trials), best_model_type, best_params, X_train, y_train, X_val, y_val, cost_model
            )
            self._trials.append(fallback_trial)
            best_score = fallback_trial.ranking_score

        feature_importance = self._extract_feature_importance(best_model, X_train.shape[1])
        return AutoMLResult(
            best_model=best_model,
            best_params=best_params,
            best_model_type=best_model_type,
            best_score=float(best_score),
            all_trials=self._trials,
            feature_importance=feature_importance,
            training_time=time.time() - start_time,
            timestamp=datetime.now(timezone.utc),
        )

    def _get_available_model_types(self) -> List[str]:
        """Get model types available in the current Python environment."""
        types: List[str] = []
        if _SKLEARN_AVAILABLE:
            types.append(ModelType.RANDOM_FOREST.value)
            if self._resolved_task_type == "classification":
                types.append(ModelType.LOGISTIC_REGRESSION.value)
            else:
                types.append(ModelType.RIDGE.value)
        if _XGB_AVAILABLE:
            types.append(ModelType.XGBOOST.value)
        types.append(ModelType.NUMPY_BASELINE.value)
        return types

    def _normalize_model_types(self, model_types: Optional[List[str]]) -> List[str]:
        available = set(self._get_available_model_types())
        requested = model_types or list(available)
        normalized: List[str] = []
        for model_type in requested:
            value = model_type.value if isinstance(model_type, ModelType) else str(model_type)
            if value in available and value not in normalized:
                normalized.append(value)
            elif value not in available:
                logger.warning("Skipping unavailable AutoML model type: %s", value)
        return normalized or [ModelType.NUMPY_BASELINE.value]

    def _sample_params(self, model_type: str) -> Dict[str, Any]:
        """Sample hyperparameters for a model type."""
        if model_type == ModelType.RANDOM_FOREST.value:
            max_depth_choice = self._rng.choice(np.array([3, 5, 7, 10, -1]))
            return {
                "n_estimators": int(self._rng.choice([25, 50, 100, 200])),
                "max_depth": None if int(max_depth_choice) == -1 else int(max_depth_choice),
                "min_samples_split": int(self._rng.choice([2, 5, 10])),
                "min_samples_leaf": int(self._rng.choice([1, 2, 4])),
            }
        if model_type == ModelType.XGBOOST.value:
            return {
                "n_estimators": int(self._rng.choice([25, 50, 100, 200])),
                "max_depth": int(self._rng.choice([2, 3, 5, 7])),
                "learning_rate": float(self._rng.choice([0.01, 0.05, 0.1, 0.2])),
                "subsample": float(self._rng.choice([0.7, 0.8, 0.9, 1.0])),
                "colsample_bytree": float(self._rng.choice([0.7, 0.8, 0.9, 1.0])),
            }
        if model_type == ModelType.LOGISTIC_REGRESSION.value:
            return {
                "C": float(self._rng.choice([0.01, 0.1, 1.0, 10.0])),
                "max_iter": int(self._rng.choice([200, 500, 1000])),
            }
        if model_type == ModelType.RIDGE.value:
            return {"alpha": float(self._rng.choice([0.01, 0.1, 1.0, 10.0]))}
        return {"task_type": self._resolved_task_type}

    def _create_model(self, model_type: str, params: Dict[str, Any]) -> Any:
        """Create an unfitted model with the given parameters."""
        if model_type == ModelType.RANDOM_FOREST.value and _SKLEARN_AVAILABLE:
            assert RandomForestClassifier is not None
            assert RandomForestRegressor is not None
            if self._resolved_task_type == "classification":
                return RandomForestClassifier(**params, random_state=self.random_seed, n_jobs=-1)
            return RandomForestRegressor(**params, random_state=self.random_seed, n_jobs=-1)
        if model_type == ModelType.XGBOOST.value and _XGB_AVAILABLE:
            assert XGBClassifier is not None
            assert XGBRegressor is not None
            if self._resolved_task_type == "classification":
                return XGBClassifier(**params, random_state=self.random_seed, eval_metric="logloss")
            return XGBRegressor(**params, random_state=self.random_seed)
        if model_type == ModelType.LOGISTIC_REGRESSION.value and _SKLEARN_AVAILABLE:
            assert LogisticRegression is not None
            return LogisticRegression(**params, random_state=self.random_seed)
        if model_type == ModelType.RIDGE.value and _SKLEARN_AVAILABLE:
            assert Ridge is not None
            return Ridge(**params)
        if model_type == ModelType.NUMPY_BASELINE.value:
            return NumpyBaselineModel(params.get("task_type", self._resolved_task_type))
        raise ValueError(f"Unsupported or unavailable model type: {model_type}")

    def _fit_model(self, model_type: str, params: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> Any:
        try:
            model = self._create_model(model_type, params)
            model.fit(X, y)
            return model
        except Exception as exc:
            logger.warning("Failed to fit %s: %s", model_type, exc)
            return None

    def _run_trial(
        self,
        trial_id: int,
        model_type: str,
        params: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        cost_model: Optional[Callable[[str, Dict[str, Any], float, float], float]] = None,
    ) -> TrialResult:
        """Run a single model trial."""
        trial_start = time.time()
        error: Optional[str] = None
        try:
            model = self._create_model(model_type, params)
            model.fit(X_train, y_train)
            train_score = self._compute_score(y_train, model.predict(X_train))
            if X_val is not None and y_val is not None:
                val_score = self._compute_score(y_val, model.predict(X_val))
            else:
                val_score = train_score
            cv_scores = self._compute_cv_score(model, X_train, y_train)
        except Exception as exc:
            error = str(exc)
            logger.warning("Trial %d failed: %s", trial_id, exc)
            train_score = 0.0
            val_score = 0.0
            cv_scores = [0.0]

        cost_penalty = self._compute_cost_penalty(cost_model, model_type, params, train_score, val_score)
        net_score = float(val_score) - cost_penalty

        return TrialResult(
            trial_id=trial_id,
            model_type=model_type,
            params=dict(params),
            train_score=float(train_score),
            val_score=float(val_score),
            cv_scores=[float(score) for score in cv_scores],
            training_time=time.time() - trial_start,
            n_features=X_train.shape[1],
            cost_penalty=cost_penalty,
            net_score=net_score,
            error=error,
        )

    def _compute_cost_penalty(
        self,
        cost_model: Optional[Callable[[str, Dict[str, Any], float, float], float]],
        model_type: str,
        params: Dict[str, Any],
        train_score: float,
        val_score: float,
    ) -> float:
        if cost_model is None:
            return 0.0
        try:
            penalty = float(cost_model(model_type, dict(params), float(train_score), float(val_score)))
        except Exception as exc:
            logger.warning("AutoML cost model failed for %s: %s", model_type, exc)
            return 0.0
        return float(np.clip(penalty, 0.0, 1.0))

    def _compute_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the configured optimization score. Higher is better."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if self.metric == "accuracy":
            return float(np.mean(y_true == y_pred))
        if self.metric == "f1":
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        if self.metric == "sharpe":
            direction_acc = np.mean(np.sign(y_true) == np.sign(y_pred))
            return float(direction_acc * 10 - 5)

        mse = np.mean((y_true.astype(float) - y_pred.astype(float)) ** 2)
        return float(-mse)

    def _compute_cv_score(self, model: Any, X: np.ndarray, y: np.ndarray) -> List[float]:
        """Compute cross-validation scores when sklearn supports the model."""
        if not _SKLEARN_AVAILABLE or cross_val_score is None or isinstance(model, NumpyBaselineModel):
            return []
        if len(X) < 4:
            return []
        scoring = "accuracy" if self._resolved_task_type == "classification" else "neg_mean_squared_error"
        try:
            folds = min(self.cv_folds, max(2, len(X) // 10))
            scores = cross_val_score(model, X, y, cv=folds, scoring=scoring)
            return [float(score) for score in scores]
        except Exception as exc:
            logger.debug("Cross-validation skipped: %s", exc)
            return []

    def _extract_feature_importance(self, model: Any, n_features: int) -> Optional[Dict[str, float]]:
        """Extract feature importance from fitted model when available."""
        if model is None:
            return None
        try:
            if hasattr(model, "feature_importances_"):
                importances = np.asarray(model.feature_importances_, dtype=float)
            elif hasattr(model, "coef_") and model.coef_ is not None:
                importances = np.abs(np.asarray(model.coef_, dtype=float)).reshape(-1)
            else:
                return None
            if len(importances) != n_features:
                return None
            total = float(np.sum(np.abs(importances)))
            if total > 0:
                importances = importances / total
            return {f"feature_{i}": float(value) for i, value in enumerate(importances)}
        except Exception as exc:
            logger.debug("Feature importance extraction failed: %s", exc)
            return None

    def _infer_task_type(self, y: np.ndarray) -> str:
        if self.task_type in {"classification", "regression"}:
            return self.task_type
        unique = np.unique(y)
        if y.dtype.kind in {"i", "b", "u", "O", "U", "S"} and len(unique) <= max(20, int(len(y) * 0.2)):
            return "classification"
        return "regression"

    @staticmethod
    def _validate_X(X: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(X, dtype=float)
        if array.ndim != 2:
            raise ValueError(f"{name} must be a 2D array")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError(f"{name} must not be empty")
        return array

    @staticmethod
    def _validate_y(y: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(y)
        if array.ndim != 1:
            raise ValueError(f"{name} must be a 1D array")
        if array.shape[0] == 0:
            raise ValueError(f"{name} must not be empty")
        return array
