"""Push 96 — Fill probability model (v8.32.0).

Predicts P(fill) for a limit order given:
  - Current order book depth and queue position
  - Price distance from mid
  - Volatility regime
  - Time-in-force budget

Used by DCAExecutor and IcebergExecutor to decide whether to push
an order to market or wait for fill.

Design:
  - FillProbabilityFeatures   dataclass
  - FillProbabilityModel      LGB regressor (online, warm-start)
  - FillProbabilityAdvisor    high-level: should_convert_to_market()

Fall-back: analytical Cox/Rubinstein approximation when LGB unavailable.
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
    try:
        import numpy as np
        _NP = True
    except ImportError:
        _NP = False


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------

@dataclass
class FillProbabilityFeatures:
    """8-dimensional feature vector for fill probability prediction."""
    price_distance_bps: float   # distance of limit from mid in bps (>0 = passive)
    queue_position:     float   # normalised queue position [0=front, 1=back]
    depth_at_level:     float   # total qty at limit level (normalised by ADV)
    vol_ratio:          float   # short/long vol (high = moves fast)
    spread_bps:         float   # current bid-ask spread in bps
    obi:                float   # order book imbalance [-1, 1]
    time_budget_secs:   float   # seconds remaining before cancel
    side_sign:          float   # +1 = buy limit, -1 = sell limit

    def to_list(self) -> List[float]:
        return [
            self.price_distance_bps, self.queue_position,
            self.depth_at_level, self.vol_ratio,
            self.spread_bps, self.obi,
            self.time_budget_secs, self.side_sign,
        ]


# ---------------------------------------------------------------------------
# Labelled outcome
# ---------------------------------------------------------------------------

@dataclass
class FillOutcome:
    features:   FillProbabilityFeatures
    filled:     bool       # True = order was filled within time budget
    fill_time:  float      # seconds to fill (0 if not filled)
    ts:         float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Online LightGBM regressor
# ---------------------------------------------------------------------------

class FillProbabilityModel:
    """Online fill-probability estimator.

    Trains a LightGBM binary classifier on fill outcomes.
    Provides predict(features) -> float in [0, 1].
    Falls back to analytical approximation if LGB unavailable.
    """

    FEATURE_NAMES = [
        "price_distance_bps", "queue_position",
        "depth_at_level", "vol_ratio",
        "spread_bps", "obi",
        "time_budget_secs", "side_sign",
    ]

    def __init__(
        self,
        buffer_size:    int   = 3000,
        retrain_every:  int   = 300,
        min_train_size: int   = 150,
        n_estimators:   int   = 80,
        learning_rate:  float = 0.05,
    ) -> None:
        self._buffer        : Deque[FillOutcome] = deque(maxlen=buffer_size)
        self._retrain_every  = retrain_every
        self._min_train_size = min_train_size
        self._n_estimators   = n_estimators
        self._learning_rate  = learning_rate
        self._model: Optional[object] = None
        self._since_retrain  = 0
        self._train_count    = 0

    def record_outcome(self, outcome: FillOutcome) -> None:
        self._buffer.append(outcome)
        self._since_retrain += 1
        if (
            self._since_retrain >= self._retrain_every
            and len(self._buffer) >= self._min_train_size
        ):
            self._retrain()
            self._since_retrain = 0

    def predict(self, features: FillProbabilityFeatures) -> float:
        """Return P(fill within time budget) in [0, 1]."""
        if self._model is not None and _LGB:
            X = np.array([features.to_list()], dtype=np.float32)
            return float(self._model.predict(X)[0])
        return self._analytical(features)

    @property
    def stats(self) -> dict:
        return {
            "buffer_size":  len(self._buffer),
            "train_count":  self._train_count,
            "model_ready":  self._model is not None,
            "lgb_available": _LGB,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _retrain(self) -> None:
        if not _LGB:
            return
        outcomes = list(self._buffer)
        X = np.array([o.features.to_list() for o in outcomes], dtype=np.float32)
        y = np.array([1 if o.filled else 0 for o in outcomes], dtype=np.int32)
        ds = lgb.Dataset(X, label=y, feature_name=self.FEATURE_NAMES, free_raw_data=False)
        params = {
            "objective":     "binary",
            "metric":        "auc",
            "n_estimators":  self._n_estimators,
            "learning_rate": self._learning_rate,
            "num_leaves":    31,
            "verbose":       -1,
        }
        callbacks = [lgb.early_stopping(15, verbose=False), lgb.log_evaluation(period=-1)]
        self._model = lgb.train(
            params, ds, valid_sets=[ds],
            callbacks=callbacks,
            init_model=self._model,
        )
        self._train_count += 1

    @staticmethod
    def _analytical(f: FillProbabilityFeatures) -> float:
        """Analytical approximation: exponential decay with distance + vol."""
        # Deeper passive placement + high vol + short time = lower P(fill)
        lam = max(0.01, f.vol_ratio * 0.5)
        p_dist  = math.exp(-lam * max(0, f.price_distance_bps) / 10.0)
        p_time  = 1.0 - math.exp(-f.time_budget_secs / 30.0)
        p_queue = 1.0 - f.queue_position * 0.5
        return min(0.99, max(0.01, p_dist * p_time * p_queue))


# ---------------------------------------------------------------------------
# High-level advisor
# ---------------------------------------------------------------------------

class FillProbabilityAdvisor:
    """Advises whether to convert a limit order to market.

    Usage:
        advisor = FillProbabilityAdvisor(convert_threshold=0.25)
        if advisor.should_convert_to_market(features):
            # resubmit as IOC market order
            ...
        # Record actual outcome
        advisor.record(features, filled=True, fill_time=1.2)
    """

    def __init__(
        self,
        convert_threshold: float = 0.25,
        buffer_size:       int   = 3000,
        retrain_every:     int   = 300,
    ) -> None:
        self._model      = FillProbabilityModel(
            buffer_size=buffer_size,
            retrain_every=retrain_every,
        )
        self._threshold  = convert_threshold
        self._converted  = 0
        self._waited     = 0

    def should_convert_to_market(self, features: FillProbabilityFeatures) -> bool:
        """True → convert limit to IOC market order."""
        p = self._model.predict(features)
        if p < self._threshold:
            self._converted += 1
            return True
        self._waited += 1
        return False

    def record(
        self,
        features:  FillProbabilityFeatures,
        filled:    bool,
        fill_time: float = 0.0,
    ) -> None:
        self._model.record_outcome(
            FillOutcome(features=features, filled=filled, fill_time=fill_time)
        )

    @property
    def stats(self) -> dict:
        return {
            "converted":  self._converted,
            "waited":     self._waited,
            "threshold":  self._threshold,
            **self._model.stats,
        }
