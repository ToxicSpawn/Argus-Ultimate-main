"""
MetaLearner — LightGBM ensemble meta-learner.

Stacks outputs from base strategies (signals, confidence scores, regime
features) and learns a blended position sizing / direction prediction
using LightGBM gradient boosting.

Workflow
--------
1. Collect labelled training rows: each row = base strategy signals +
   market features; label = forward return sign (1 = up, 0 = down)
2. Train / retrain LightGBM classifier with walk-forward discipline
3. At inference time, pass current strategy outputs -> predicted
   probability of up-move -> scaled to [-1, 1] signal

Dependencies
------------
    pip install lightgbm scikit-learn numpy pandas

Graceful degradation: if LightGBM is not installed the class falls back
to an equal-weight average of the input signals.
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

logger = logging.getLogger(__name__)

DEFAULT_LGB_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 10,
    "verbose": -1,
    "n_jobs": -1,
}
DEFAULT_N_ESTIMATORS: int = 200
MIN_TRAIN_SAMPLES: int = 50


@dataclass
class TrainingSample:
    features: Dict[str, float]   # strategy signals + market features
    label: int                    # 1 = trade was profitable, 0 = not
    weight: float = 1.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class MetaPrediction:
    signal: float           # [-1, 1]  (2 * P(up) - 1)
    prob_up: float          # raw probability [0, 1]
    confidence: float       # |signal|  — how decisive the model is
    used_fallback: bool     # True if LGB unavailable or model not trained


class MetaLearner:
    """
    LightGBM-based meta-learner that stacks base strategy signals.

    Parameters
    ----------
    feature_names    : ordered list of feature keys expected in each sample
    lgb_params       : LightGBM training parameters (overrides defaults)
    n_estimators     : number of boosting rounds
    model_path       : optional path to persist/load trained model
    retrain_interval : retrain after this many new samples (0 = manual only)
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        lgb_params: Optional[dict] = None,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        model_path: Optional[str] = None,
        retrain_interval: int = 100,
    ) -> None:
        self._feature_names = feature_names or []
        self._params = {**DEFAULT_LGB_PARAMS, **(lgb_params or {})}
        self._n_estimators = n_estimators
        self._model_path = Path(model_path) if model_path else None
        self._retrain_interval = retrain_interval

        self._samples: List[TrainingSample] = []
        self._model: Optional[object] = None
        self._scaler: Optional[object] = None
        self._trained_at: Optional[float] = None
        self._samples_since_retrain: int = 0
        self._is_trained: bool = False

        if self._model_path and self._model_path.exists():
            self._load_model()

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def add_sample(self, sample: TrainingSample) -> None:
        """Add a labelled training sample and optionally trigger retraining."""
        if self._feature_names and sample.features:
            # Align feature order; fill missing with 0
            pass
        self._samples.append(sample)
        self._samples_since_retrain += 1

        if (
            self._retrain_interval > 0
            and self._samples_since_retrain >= self._retrain_interval
            and len(self._samples) >= MIN_TRAIN_SAMPLES
        ):
            self.train()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> bool:
        """
        Train (or retrain) the LightGBM model on accumulated samples.
        Returns True if training succeeded.
        """
        if not _LGB_AVAILABLE:
            logger.warning("LightGBM not available — meta-learner using fallback")
            return False

        if len(self._samples) < MIN_TRAIN_SAMPLES:
            logger.info(
                "MetaLearner: insufficient samples (%d < %d)",
                len(self._samples), MIN_TRAIN_SAMPLES,
            )
            return False

        X, y, w = self._build_matrices()
        if X.shape[0] == 0:
            return False

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X_scaled, y, w, test_size=0.2, shuffle=False
        )

        train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
        val_data   = lgb.Dataset(X_val,   label=y_val,   weight=w_val, reference=train_data)

        callbacks = [lgb.early_stopping(20, verbose=False), lgb.log_evaluation(period=-1)]

        self._model = lgb.train(
            self._params,
            train_data,
            num_boost_round=self._n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        self._trained_at = time.time()
        self._samples_since_retrain = 0
        self._is_trained = True
        logger.info(
            "MetaLearner: trained on %d samples, %d features",
            X.shape[0], X.shape[1],
        )

        if self._model_path:
            self._save_model()

        return True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, features: Dict[str, float]) -> MetaPrediction:
        """
        Predict directional signal from a feature dict.
        Falls back to equal-weight average if model not trained.
        """
        if not self._is_trained or not _LGB_AVAILABLE or self._model is None:
            return self._fallback_prediction(features)

        try:
            x = self._dict_to_vector(features)
            if self._scaler is not None:
                x = self._scaler.transform(x.reshape(1, -1))
            else:
                x = x.reshape(1, -1)
            prob_up = float(self._model.predict(x)[0])  # type: ignore[attr-defined]
            signal = max(-1.0, min(1.0, 2.0 * prob_up - 1.0))
            return MetaPrediction(
                signal=signal,
                prob_up=prob_up,
                confidence=abs(signal),
                used_fallback=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("MetaLearner predict error: %s", exc)
            return self._fallback_prediction(features)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> Dict[str, float]:
        """Return feature importances (gain) from the trained model."""
        if not self._is_trained or self._model is None:
            return {}
        try:
            names = self._feature_names or [
                f"f{i}" for i in range(len(self._model.feature_importance()))  # type: ignore
            ]
            importances = self._model.feature_importance(importance_type="gain")  # type: ignore
            return dict(zip(names, importances.tolist()))
        except Exception as exc:  # noqa: BLE001
            logger.warning("feature_importance error: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_matrices(
        self,
    ) -> tuple:  # (X: np.ndarray, y: np.ndarray, w: np.ndarray)
        names = self._feature_names or self._infer_feature_names()
        rows, labels, weights = [], [], []
        for s in self._samples:
            row = [s.features.get(n, 0.0) for n in names]
            rows.append(row)
            labels.append(s.label)
            weights.append(s.weight)
        X = np.array(rows, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        w = np.array(weights, dtype=np.float32)
        return X, y, w

    def _infer_feature_names(self) -> List[str]:
        if not self._samples:
            return []
        return sorted(self._samples[0].features.keys())

    def _dict_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        names = self._feature_names or self._infer_feature_names()
        return np.array([features.get(n, 0.0) for n in names], dtype=np.float32)

    def _fallback_prediction(self, features: Dict[str, float]) -> MetaPrediction:
        vals = list(features.values())
        avg = sum(vals) / len(vals) if vals else 0.0
        signal = max(-1.0, min(1.0, avg))
        return MetaPrediction(
            signal=signal,
            prob_up=(signal + 1.0) / 2.0,
            confidence=abs(signal),
            used_fallback=True,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_model(self) -> None:
        try:
            payload = {"model": self._model, "scaler": self._scaler,
                       "feature_names": self._feature_names}
            with open(self._model_path, "wb") as fh:  # type: ignore[arg-type]
                pickle.dump(payload, fh)
            logger.info("MetaLearner: model saved to %s", self._model_path)
        except OSError as exc:
            logger.error("MetaLearner: save failed: %s", exc)

    def _load_model(self) -> None:
        try:
            with open(self._model_path, "rb") as fh:  # type: ignore[arg-type]
                payload = pickle.load(fh)
            self._model = payload["model"]
            self._scaler = payload["scaler"]
            self._feature_names = payload.get("feature_names", self._feature_names)
            self._is_trained = True
            logger.info("MetaLearner: model loaded from %s", self._model_path)
        except (OSError, pickle.UnpicklingError, KeyError) as exc:
            logger.warning("MetaLearner: load failed: %s", exc)
