"""Batch 1 — Regime-specific consensus weighting.

Adjusts ensemble signal weights based on the detected market regime
(trending, mean-reverting, volatile, ranging).  Weights are updated
via a rolling EWM performance tracker per regime label.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIME_LABELS = ["trending", "mean_reverting", "volatile", "ranging"]


class RegimeConsensusWeighter:
    """Tracks per-regime model performance and returns adaptive weights."""

    def __init__(
        self,
        model_names: List[str],
        ewm_alpha: float = 0.05,
        min_weight: float = 0.02,
        softmax_temp: float = 1.0,
    ) -> None:
        self._models = model_names
        self._alpha = ewm_alpha
        self._min_weight = min_weight
        self._temp = softmax_temp
        # scores[regime][model] → EWM score (higher = better)
        self._scores: Dict[str, Dict[str, float]] = {
            r: {m: 1.0 for m in model_names} for r in REGIME_LABELS
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_weights(self, regime: str) -> Dict[str, float]:
        """Return normalised weights for each model given the current regime."""
        regime = self._validate_regime(regime)
        raw = np.array([self._scores[regime][m] for m in self._models])
        # softmax with temperature
        exp = np.exp((raw - raw.max()) / self._temp)
        norm = exp / exp.sum()
        weights = {m: max(float(w), self._min_weight) for m, w in zip(self._models, norm)}
        # re-normalise after clamping
        total = sum(weights.values())
        return {m: w / total for m, w in weights.items()}

    def update(self, regime: str, model_pnl: Dict[str, float]) -> None:
        """Update EWM scores using realised PnL per model."""
        regime = self._validate_regime(regime)
        for model, pnl in model_pnl.items():
            if model not in self._scores[regime]:
                self._scores[regime][model] = 1.0
            old = self._scores[regime][model]
            self._scores[regime][model] = (1 - self._alpha) * old + self._alpha * pnl
        logger.debug("Regime %s weights updated: %s", regime, self._scores[regime])

    def weighted_signal(
        self,
        regime: str,
        model_signals: Dict[str, float],
    ) -> float:
        """Return a single consensus signal [-1, 1] weighted by regime weights."""
        weights = self.get_weights(regime)
        total_w = 0.0
        signal = 0.0
        for model, sig in model_signals.items():
            w = weights.get(model, 0.0)
            signal += w * sig
            total_w += w
        return signal / total_w if total_w > 0 else 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_regime(regime: str) -> str:
        if regime not in REGIME_LABELS:
            logger.warning("Unknown regime '%s', defaulting to 'ranging'", regime)
            return "ranging"
        return regime

    def to_dataframe(self) -> pd.DataFrame:
        """Inspect current scores across all regimes."""
        rows = []
        for regime in REGIME_LABELS:
            for model, score in self._scores[regime].items():
                rows.append({"regime": regime, "model": model, "score": score})
        return pd.DataFrame(rows)
