"""
Live Inference Service for ARGUS ML models.

Provides a unified interface for running predictions against any model
managed by ModelManager.  Features:

- In-memory model cache (avoids repeated disk I/O)
- Input validation (NaN / Inf rejection)
- Latency measurement per prediction
- Batch prediction
- Graceful fallback when a model is unavailable
- Hit-rate and latency statistics
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """Container for a single inference output."""

    prediction: Any
    confidence: float
    latency_ms: float
    model_version: int
    cache_hit: bool


# ---------------------------------------------------------------------------
# InferenceService
# ---------------------------------------------------------------------------

class InferenceService:
    """
    Thin serving layer that wraps :class:`ModelManager` with caching,
    input validation, latency tracking, and fallback behaviour.

    Parameters
    ----------
    model_manager
        A :class:`ml.model_manager.ModelManager` instance used to
        load models that are not yet in the local cache.
    """

    def __init__(self, model_manager: Any) -> None:
        self._model_manager = model_manager

        # model_name -> loaded model object
        self._cache: Dict[str, Any] = {}

        # Stats accumulators
        self._total_predictions: int = 0
        self._cache_hits: int = 0
        self._total_latency_ms: float = 0.0
        self._errors: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        model_name: str,
        features: Any,
        timeout: float = 5.0,
    ) -> InferenceResult:
        """
        Run a single prediction.

        Parameters
        ----------
        model_name : str
            Key in the ModelManager registry.
        features : array-like
            Input features (list, np.ndarray, etc.).
        timeout : float
            Not enforced as a hard deadline but recorded in stats.

        Returns
        -------
        InferenceResult
        """
        self._total_predictions += 1

        # Validate input
        self._validate_input(features)

        # Resolve model
        cache_hit = model_name in self._cache
        model = self._resolve_model(model_name)

        if cache_hit:
            self._cache_hits += 1

        # Get model version from registry
        version = 0
        if hasattr(self._model_manager, '_registry') and model_name in self._model_manager._registry:
            version = self._model_manager._registry[model_name].version

        # Run prediction
        t0 = time.perf_counter()
        try:
            raw = model.predict(features) if hasattr(model, 'predict') else model(features)
        except Exception as exc:
            self._errors += 1
            logger.warning("InferenceService: predict failed for '%s': %s", model_name, exc)
            fallback = self.get_fallback(model_name)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._total_latency_ms += latency_ms
            return InferenceResult(
                prediction=fallback,
                confidence=0.0,
                latency_ms=latency_ms,
                model_version=version,
                cache_hit=cache_hit,
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._total_latency_ms += latency_ms

        # Extract prediction and confidence
        prediction, confidence = self._extract_prediction(raw)

        return InferenceResult(
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency_ms,
            model_version=version,
            cache_hit=cache_hit,
        )

    def predict_batch(
        self,
        model_name: str,
        feature_list: List[Any],
    ) -> List[InferenceResult]:
        """
        Run predictions for a batch of feature vectors.

        Parameters
        ----------
        model_name : str
            Key in the ModelManager registry.
        feature_list : list
            List of feature vectors.

        Returns
        -------
        list[InferenceResult]
        """
        return [self.predict(model_name, f) for f in feature_list]

    def get_fallback(self, model_name: str) -> Any:
        """
        Return a neutral/default prediction when a model is unavailable.

        - Classification models (name contains 'classifier'): ``"HOLD"``
        - Everything else (regression): ``0.0``
        """
        if "classifier" in model_name.lower():
            return "HOLD"
        return 0.0

    def clear_cache(self, model_name: Optional[str] = None) -> None:
        """
        Remove one or all models from the local cache.

        Parameters
        ----------
        model_name : str or None
            If None, clear the entire cache.
        """
        if model_name is None:
            self._cache.clear()
            logger.info("InferenceService: entire cache cleared")
        else:
            self._cache.pop(model_name, None)
            logger.info("InferenceService: cache cleared for '%s'", model_name)

    def get_stats(self) -> dict:
        """
        Return aggregate performance statistics.

        Returns
        -------
        dict
            Keys: ``hit_rate``, ``avg_latency_ms``, ``total_predictions``,
            ``errors``, ``cached_models``.
        """
        hit_rate = (
            self._cache_hits / self._total_predictions
            if self._total_predictions > 0
            else 0.0
        )
        avg_latency = (
            self._total_latency_ms / self._total_predictions
            if self._total_predictions > 0
            else 0.0
        )
        return {
            "hit_rate": round(hit_rate, 4),
            "avg_latency_ms": round(avg_latency, 4),
            "total_predictions": self._total_predictions,
            "errors": self._errors,
            "cached_models": list(self._cache.keys()),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_input(self, features: Any) -> None:
        """Raise ValueError if features contain NaN or Inf."""
        arr = np.asarray(features, dtype=float)
        if np.any(np.isnan(arr)):
            raise ValueError("Input features contain NaN")
        if np.any(np.isinf(arr)):
            raise ValueError("Input features contain Inf")

    def _resolve_model(self, model_name: str) -> Any:
        """Return model from cache, or load via ModelManager and cache it."""
        if model_name in self._cache:
            return self._cache[model_name]

        # Try to get already-loaded object from ModelManager
        obj = None
        if hasattr(self._model_manager, 'get_object'):
            obj = self._model_manager.get_object(model_name)

        if obj is None:
            # Attempt to load from disk
            if hasattr(self._model_manager, 'load'):
                self._model_manager.load(model_name)
                obj = self._model_manager.get_object(model_name)

        if obj is None:
            raise RuntimeError(f"Model '{model_name}' could not be loaded")

        self._cache[model_name] = obj
        return obj

    @staticmethod
    def _extract_prediction(raw: Any) -> tuple:
        """
        Normalise a raw model output into (prediction, confidence).

        Handles:
        - tuple (prediction, confidence)
        - numpy array → first element
        - scalar
        """
        if isinstance(raw, tuple) and len(raw) == 2:
            return raw[0], float(raw[1])

        if isinstance(raw, np.ndarray):
            pred = raw.item() if raw.ndim == 0 else raw.flat[0]
            return pred, 1.0

        if isinstance(raw, (list, tuple)):
            return raw[0], 1.0

        return raw, 1.0
