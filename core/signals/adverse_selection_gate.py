"""Push 66 — Adverse selection gate (LightGBM predictor stub).

Predicts whether the next fill will suffer adverse selection
based on LOB features. Uses LightGBM trained on:
  [obi, spread_norm, trade_flow_imbalance, bid_ask_depth_ratio,
   recent_vol, time_of_day, regime]

If predicted adverse_selection_score > threshold:
  -> skip fill or widen spread

Reference: SSRN 6344338 (2026) — LightGBM on 31M+ crypto LOB obs
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class AdverseSelectionResult:
    score: float           # [0, 1] — 1 = certain adverse selection
    threshold: float
    fill_recommended: bool
    features: List[float]


class AdverseSelectionGate:
    """LightGBM-based adverse selection predictor.

    Falls back to rule-based heuristic if LightGBM not available.
    """

    def __init__(
        self,
        model_path: str | None = None,
        threshold: float = 0.65,
    ):
        self.threshold = threshold
        self._model = None
        self._lgbm_available = False

        if model_path and Path(model_path).exists():
            try:
                import lightgbm as lgb
                self._model = lgb.Booster(model_file=model_path)
                self._lgbm_available = True
            except ImportError:
                pass

    def evaluate(
        self,
        obi: float,
        spread_norm: float,
        trade_flow_imbalance: float,
        depth_ratio: float,
        recent_vol: float,
        regime: float = 0.0,
    ) -> AdverseSelectionResult:
        features = [
            obi, spread_norm, trade_flow_imbalance,
            depth_ratio, recent_vol, regime,
        ]
        feature_arr = np.array(features).reshape(1, -1)

        if self._lgbm_available and self._model is not None:
            score = float(self._model.predict(feature_arr)[0])
        else:
            # Rule-based heuristic fallback
            # High spread + low OBI alignment = adverse selection risk
            alignment = obi * trade_flow_imbalance
            score = float(np.clip(
                0.3 + spread_norm * 2.0 - alignment * 0.5 + recent_vol * 1.5,
                0.0, 1.0
            ))

        return AdverseSelectionResult(
            score=score,
            threshold=self.threshold,
            fill_recommended=score < self.threshold,
            features=features,
        )

    def should_fill(self, *args, **kwargs) -> bool:
        """Convenience wrapper — True if fill is safe."""
        return self.evaluate(*args, **kwargs).fill_recommended
