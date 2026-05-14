"""
Pre-trained model loader for ARGUS.

Loads sklearn models from pickle/joblib files with metadata validation,
version checking, and graceful fallback when models are missing or corrupted.

Usage::

    loader = TrainedModelLoader(models_dir="models/")
    regime_clf = loader.load_regime_classifier()
    if regime_clf is not None:
        prediction = regime_clf.predict(features)

    vol_model = loader.load_volatility_forecaster()
    if vol_model is not None:
        forecast = vol_model.predict(features)

    alpha = loader.load_alpha_model()
    if alpha is not None:
        direction, confidence = alpha.predict(features)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata stored alongside each trained model."""
    model_type: str
    training_date: str
    features: List[str]
    version: str
    extra: Dict[str, Any]


class PreTrainedRegimeClassifier:
    """Wrapper around a pre-trained regime classifier."""

    def __init__(self, model: Any, metadata: ModelMetadata) -> None:
        self._model = model
        self.metadata = metadata
        self._classes = metadata.extra.get("classes", [])

    def predict(self, features: np.ndarray) -> str:
        """Predict regime label from feature vector."""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        idx = int(self._model.predict(features)[0])
        if self._classes and idx < len(self._classes):
            return self._classes[idx]
        return str(idx)

    def predict_proba(self, features: np.ndarray) -> Dict[str, float]:
        """Return probability distribution over regime labels."""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        proba = self._model.predict_proba(features)[0]
        if self._classes:
            return {self._classes[i]: float(p) for i, p in enumerate(proba)}
        return {str(i): float(p) for i, p in enumerate(proba)}


class PreTrainedVolatilityForecaster:
    """Wrapper around a pre-trained volatility forecaster."""

    def __init__(self, model: Any, metadata: ModelMetadata) -> None:
        self._model = model
        self.metadata = metadata

    def predict(self, features: np.ndarray) -> float:
        """Predict next-5-day annualized volatility from feature vector."""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return float(self._model.predict(features)[0])


class PreTrainedAlphaModel:
    """Wrapper around a pre-trained alpha (direction) model."""

    def __init__(self, model: Any, metadata: ModelMetadata) -> None:
        self._model = model
        self.metadata = metadata

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict direction and confidence from feature vector.

        Returns
        -------
        direction : str
            "up" or "down"
        confidence : float
            Probability of the predicted direction (0.5 to 1.0)
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        proba = self._model.predict_proba(features)[0]
        if proba[1] >= 0.5:
            return "up", float(proba[1])
        else:
            return "down", float(proba[0])


def _load_model_file(path: Path) -> Optional[Dict[str, Any]]:
    """Load a joblib model file. Returns None on failure."""
    try:
        import joblib
        data = joblib.load(path)
        if not isinstance(data, dict) or "model" not in data or "metadata" not in data:
            logger.warning(
                "Model file %s has invalid structure (expected dict with 'model' and 'metadata' keys)",
                path,
            )
            return None
        return data
    except Exception as exc:
        logger.warning("Failed to load model from %s: %s", path, exc)
        return None


def _parse_metadata(raw: Dict[str, Any]) -> ModelMetadata:
    """Parse metadata dict into ModelMetadata dataclass."""
    return ModelMetadata(
        model_type=raw.get("model_type", "unknown"),
        training_date=raw.get("training_date", "unknown"),
        features=raw.get("features", []),
        version=raw.get("version", "0.0.0"),
        extra={k: v for k, v in raw.items() if k not in ("model_type", "training_date", "features", "version")},
    )


class TrainedModelLoader:
    """
    Loads pre-trained ML models from disk.

    All load methods return None if the model file is missing, corrupted,
    or fails validation. Callers should fall back to rule-based logic.
    """

    def __init__(self, models_dir: str = "models") -> None:
        self._dir = Path(models_dir)

    def load_regime_classifier(
        self, path: Optional[str] = None,
    ) -> Optional[PreTrainedRegimeClassifier]:
        """Load pre-trained regime classifier. Returns None if unavailable."""
        fpath = Path(path) if path else self._dir / "regime_classifier.pkl"
        if not fpath.exists():
            logger.info("Regime classifier model not found at %s — using rule-based fallback", fpath)
            return None

        data = _load_model_file(fpath)
        if data is None:
            return None

        metadata = _parse_metadata(data["metadata"])
        if metadata.model_type != "regime_classifier":
            logger.warning(
                "Model at %s has type '%s', expected 'regime_classifier'",
                fpath, metadata.model_type,
            )
            return None

        logger.info(
            "Loaded pre-trained regime classifier (version=%s, trained=%s, features=%d)",
            metadata.version, metadata.training_date, len(metadata.features),
        )
        return PreTrainedRegimeClassifier(data["model"], metadata)

    def load_volatility_forecaster(
        self, path: Optional[str] = None,
    ) -> Optional[PreTrainedVolatilityForecaster]:
        """Load pre-trained volatility forecaster. Returns None if unavailable."""
        fpath = Path(path) if path else self._dir / "volatility_forecaster.pkl"
        if not fpath.exists():
            logger.info("Volatility forecaster model not found at %s — using EWMA fallback", fpath)
            return None

        data = _load_model_file(fpath)
        if data is None:
            return None

        metadata = _parse_metadata(data["metadata"])
        if metadata.model_type != "volatility_forecaster":
            logger.warning(
                "Model at %s has type '%s', expected 'volatility_forecaster'",
                fpath, metadata.model_type,
            )
            return None

        logger.info(
            "Loaded pre-trained volatility forecaster (version=%s, trained=%s)",
            metadata.version, metadata.training_date,
        )
        return PreTrainedVolatilityForecaster(data["model"], metadata)

    def load_alpha_model(
        self, path: Optional[str] = None,
    ) -> Optional[PreTrainedAlphaModel]:
        """Load pre-trained alpha (direction) model. Returns None if unavailable."""
        fpath = Path(path) if path else self._dir / "alpha_model.pkl"
        if not fpath.exists():
            logger.info("Alpha model not found at %s — using factor-based fallback", fpath)
            return None

        data = _load_model_file(fpath)
        if data is None:
            return None

        metadata = _parse_metadata(data["metadata"])
        if metadata.model_type != "alpha_model":
            logger.warning(
                "Model at %s has type '%s', expected 'alpha_model'",
                fpath, metadata.model_type,
            )
            return None

        logger.info(
            "Loaded pre-trained alpha model (version=%s, trained=%s)",
            metadata.version, metadata.training_date,
        )
        return PreTrainedAlphaModel(data["model"], metadata)

    def check_model_versions(self) -> Dict[str, Optional[Dict[str, str]]]:
        """Check versions of all model files. Returns {model_name: {version, date} or None}."""
        result: Dict[str, Optional[Dict[str, str]]] = {}
        for name in ("regime_classifier", "volatility_forecaster", "alpha_model"):
            fpath = self._dir / f"{name}.pkl"
            if not fpath.exists():
                result[name] = None
                continue
            data = _load_model_file(fpath)
            if data is None:
                result[name] = None
                continue
            meta = data.get("metadata", {})
            result[name] = {
                "version": meta.get("version", "unknown"),
                "training_date": meta.get("training_date", "unknown"),
                "model_type": meta.get("model_type", "unknown"),
            }
        return result

    def models_available(self) -> bool:
        """Return True if all three model files exist."""
        for name in ("regime_classifier", "volatility_forecaster", "alpha_model"):
            if not (self._dir / f"{name}.pkl").exists():
                return False
        return True
