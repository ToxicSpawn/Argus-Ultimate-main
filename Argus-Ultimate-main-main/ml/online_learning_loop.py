"""Batch 3 — Online learning loop with meta-learner.

This module provides:
  * OnlineLearner — River-based incremental ML model (Hoeffding tree / PA).
  * MetaLearner — stacked ensemble that selects between base models based
    on rolling OOS performance using a softmax bandit.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Callable, Dict, Deque, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from river import linear_model, metrics, optim, preprocessing, tree  # noqa: F401
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    logger.warning("river not installed — OnlineLearner will use PA fallback")


class PassiveAggressiveFallback:
    """Minimal PA regressor implemented with numpy when river is absent."""

    def __init__(self, C: float = 1.0) -> None:
        self._C = C
        self._w: Optional[np.ndarray] = None

    def learn_one(self, x: Dict[str, float], y: float) -> None:
        feat = np.array(list(x.values()), dtype=float)
        if self._w is None:
            self._w = np.zeros(len(feat))
        pred = float(self._w @ feat)
        loss = max(0.0, abs(pred - y) - 0.0)  # epsilon-insensitive (ε=0)
        norm2 = float(feat @ feat)
        if norm2 == 0:
            return
        tau = min(self._C, loss / norm2)
        self._w = self._w + tau * np.sign(y - pred) * feat

    def predict_one(self, x: Dict[str, float]) -> float:
        if self._w is None:
            return 0.0
        feat = np.array(list(x.values()), dtype=float)
        return float(self._w @ feat)


class OnlineLearner:
    """Incremental regression model updated tick-by-tick."""

    def __init__(self, model_type: str = "pa", C: float = 1.0) -> None:
        if RIVER_AVAILABLE and model_type == "hoeffding":
            from river import tree as rtree
            self._model = rtree.HoeffdingTreeRegressor()
        elif RIVER_AVAILABLE and model_type == "pa":
            from river import linear_model as lm, optim as ro
            self._model = lm.PARegressor(C=C)
        else:
            self._model = PassiveAggressiveFallback(C=C)
        self._n = 0

    def update(self, features: Dict[str, float], target: float) -> float:
        """Learn from one sample; return prediction BEFORE learning."""
        pred = self._model.predict_one(features)
        self._model.learn_one(features, target)
        self._n += 1
        return pred if pred is not None else 0.0

    def predict(self, features: Dict[str, float]) -> float:
        result = self._model.predict_one(features)
        return result if result is not None else 0.0

    @property
    def n_samples(self) -> int:
        return self._n


class MetaLearner:
    """Softmax bandit that selects the best-performing base model per regime."""

    def __init__(
        self,
        model_names: List[str],
        window: int = 100,
        temperature: float = 0.5,
    ) -> None:
        self._names = model_names
        self._window = window
        self._temp = temperature
        self._reward_history: Dict[str, Deque[float]] = {
            n: deque(maxlen=window) for n in model_names
        }

    def record_reward(self, model_name: str, reward: float) -> None:
        """Record realised reward (e.g. PnL) for a model."""
        if model_name in self._reward_history:
            self._reward_history[model_name].append(reward)

    def select_model(self) -> str:
        """Return model name sampled from softmax distribution over mean rewards."""
        means = np.array(
            [
                np.mean(self._reward_history[n]) if self._reward_history[n] else 0.0
                for n in self._names
            ]
        )
        exp = np.exp((means - means.max()) / self._temp)
        probs = exp / exp.sum()
        return str(np.random.choice(self._names, p=probs))

    def get_weights(self) -> Dict[str, float]:
        """Return current probability weights for all models."""
        means = np.array(
            [
                np.mean(self._reward_history[n]) if self._reward_history[n] else 0.0
                for n in self._names
            ]
        )
        exp = np.exp((means - means.max()) / self._temp)
        probs = exp / exp.sum()
        return {n: float(p) for n, p in zip(self._names, probs)}
