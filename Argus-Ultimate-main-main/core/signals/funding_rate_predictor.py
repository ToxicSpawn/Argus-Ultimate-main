"""Push 96 — Funding rate predictor (v8.32.0).

Predicts next-period (8h) funding rate from:
  - Open interest trend
  - Perpetual premium / basis
  - Recent funding history
  - Short/long vol ratio
  - OBI trend

Exposes:
  FundingRatePredictor   — online LGB regressor
  FundingRateFeatures    — feature dataclass
  FundingRateAdvisor     — high-level: best_entry_side()
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

try:
    import lightgbm as lgb
    import numpy as np
    _LGB = True
except ImportError:
    _LGB = False


@dataclass
class FundingRateFeatures:
    """Features for 8h-ahead funding rate prediction."""
    current_funding:      float   # current funding rate (e.g. 0.0001)
    funding_ema_3:        float   # 3-period EMA of funding
    funding_ema_8:        float   # 8-period EMA of funding
    oi_delta_pct:         float   # OI change % over last period
    basis_bps:            float   # perp premium over spot in bps
    vol_ratio:            float   # short/long vol ratio
    obi_ema:              float   # EMA of OBI over last N ticks
    liq_imbalance:        float   # long liq / (long liq + short liq)

    def to_list(self) -> List[float]:
        return [
            self.current_funding, self.funding_ema_3, self.funding_ema_8,
            self.oi_delta_pct, self.basis_bps, self.vol_ratio,
            self.obi_ema, self.liq_imbalance,
        ]


@dataclass
class FundingOutcome:
    features:        FundingRateFeatures
    actual_funding:  float   # actual funding that was paid
    ts:              float = field(default_factory=time.time)


class FundingRatePredictor:
    """Online LightGBM funding rate predictor."""

    FEATURE_NAMES = [
        "current_funding", "funding_ema_3", "funding_ema_8",
        "oi_delta_pct", "basis_bps", "vol_ratio",
        "obi_ema", "liq_imbalance",
    ]

    def __init__(
        self,
        buffer_size:    int   = 500,
        retrain_every:  int   = 50,
        min_train_size: int   = 40,
    ) -> None:
        self._buffer        : Deque[FundingOutcome] = deque(maxlen=buffer_size)
        self._retrain_every  = retrain_every
        self._min_train_size = min_train_size
        self._model: Optional[object] = None
        self._since_retrain  = 0
        self._train_count    = 0

    def record_outcome(self, outcome: FundingOutcome) -> None:
        self._buffer.append(outcome)
        self._since_retrain += 1
        if (
            self._since_retrain >= self._retrain_every
            and len(self._buffer) >= self._min_train_size
        ):
            self._retrain()
            self._since_retrain = 0

    def predict(self, features: FundingRateFeatures) -> float:
        """Return predicted next-period funding rate."""
        if self._model is not None and _LGB:
            X = np.array([features.to_list()], dtype=np.float32)
            return float(self._model.predict(X)[0])
        return self._heuristic(features)

    @property
    def stats(self) -> dict:
        return {
            "buffer_size": len(self._buffer),
            "train_count": self._train_count,
            "model_ready": self._model is not None,
        }

    def _retrain(self) -> None:
        if not _LGB:
            return
        outcomes = list(self._buffer)
        X = np.array([o.features.to_list() for o in outcomes], dtype=np.float32)
        y = np.array([o.actual_funding for o in outcomes], dtype=np.float32)
        ds = lgb.Dataset(X, label=y, feature_name=self.FEATURE_NAMES, free_raw_data=False)
        params = {
            "objective":     "regression_l1",
            "metric":        "mae",
            "n_estimators":  60,
            "learning_rate": 0.05,
            "num_leaves":    15,
            "verbose":       -1,
        }
        callbacks = [lgb.log_evaluation(period=-1)]
        self._model = lgb.train(
            params, ds, callbacks=callbacks, init_model=self._model
        )
        self._train_count += 1

    @staticmethod
    def _heuristic(f: FundingRateFeatures) -> float:
        """Weighted EMA heuristic."""
        return 0.5 * f.funding_ema_3 + 0.3 * f.current_funding + 0.2 * f.funding_ema_8


class FundingRateAdvisor:
    """High-level funding rate advisor.

    Recommends entry side (long/short/neutral) based on predicted funding:
      - Predicted funding > pos_threshold  → prefer SHORT (collect funding)
      - Predicted funding < neg_threshold  → prefer LONG  (collect funding)
      - Otherwise                          → neutral
    """

    def __init__(
        self,
        pos_threshold: float = 0.0003,
        neg_threshold: float = -0.0003,
    ) -> None:
        self._predictor    = FundingRatePredictor()
        self._pos_threshold = pos_threshold
        self._neg_threshold = neg_threshold

    def predict_funding(self, features: FundingRateFeatures) -> float:
        return self._predictor.predict(features)

    def best_entry_side(self, features: FundingRateFeatures) -> str:
        """Returns 'SHORT', 'LONG', or 'NEUTRAL'."""
        predicted = self._predictor.predict(features)
        if predicted > self._pos_threshold:
            return "SHORT"
        if predicted < self._neg_threshold:
            return "LONG"
        return "NEUTRAL"

    def record(self, features: FundingRateFeatures, actual_funding: float) -> None:
        self._predictor.record_outcome(
            FundingOutcome(features=features, actual_funding=actual_funding)
        )

    @property
    def stats(self) -> dict:
        return self._predictor.stats
