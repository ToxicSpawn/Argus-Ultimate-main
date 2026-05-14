"""Volatility and trend regime detection with optional HMM support."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from collections.abc import Iterable
from typing import Any, Deque, cast

import numpy as np

logger = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _GaussianHMM = None

_HMM_AVAILABLE = _GaussianHMM is not None


class VolatilityRegime(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRISIS = "crisis"


class TrendRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


@dataclass(slots=True)
class MarketRegime:
    volatility: VolatilityRegime
    trend: TrendRegime
    probability: float
    duration: int
    realized_volatility: float
    trend_strength: float
    hmm_state: int | None = None
    probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.volatility.value


@dataclass(slots=True)
class RegimeDetectorConfig:
    lookback: int = 252
    volatility_window: int = 30
    trend_window: int = 20
    min_history: int = 40
    trend_threshold: float = 0.001
    hmm_states: int = 4
    hmm_refit_interval: int = 50
    low_vol_percentile: float = 0.35
    high_vol_percentile: float = 0.75
    crisis_vol_percentile: float = 0.93


class RegimeDetector:
    """Streaming regime detector with optional HMM-backed state inference."""

    def __init__(self, config: RegimeDetectorConfig | None = None) -> None:
        self.config: RegimeDetectorConfig = config or RegimeDetectorConfig()
        self._prices: Deque[float] = deque(maxlen=self.config.lookback + 1)
        self._returns: Deque[float] = deque(maxlen=self.config.lookback)
        self._vol_history: Deque[float] = deque(maxlen=self.config.lookback)
        self._current_label: VolatilityRegime = VolatilityRegime.MEDIUM
        self._current_duration: int = 0
        self._current_trend: TrendRegime = TrendRegime.SIDEWAYS
        self._current_hmm_state: int | None = None
        self._last_probabilities: dict[str, float] = self._uniform_probabilities()
        self._hmm_model: object | None = None
        self._updates_since_hmm_fit: int = 0

    def update(self, price: float, volume: float = 0.0) -> MarketRegime:
        if price <= 0:
            raise ValueError("price must be positive")
        self._prices.append(float(price))
        if len(self._prices) >= 2:
            previous = self._prices[-2]
            self._returns.append(float(np.log(price / max(previous, 1e-12))))

        realized_vol = self.realized_volatility()
        self._vol_history.append(realized_vol)
        trend = self._classify_trend()
        probabilities = self._volatility_probabilities(realized_vol)
        hmm_state, hmm_probabilities = self._infer_hmm_state(volume=volume)
        probabilities.update(hmm_probabilities)

        next_label = max(
            (VolatilityRegime.LOW, VolatilityRegime.MEDIUM, VolatilityRegime.HIGH, VolatilityRegime.CRISIS),
            key=lambda item: probabilities.get(item.value, 0.0),
        )
        if next_label == self._current_label:
            self._current_duration += 1
        else:
            self._current_label = next_label
            self._current_duration = 1

        self._current_trend = trend
        self._current_hmm_state = hmm_state
        self._last_probabilities = probabilities
        return MarketRegime(
            volatility=self._current_label,
            trend=self._current_trend,
            probability=float(probabilities.get(self._current_label.value, 0.25)),
            duration=self._current_duration,
            realized_volatility=realized_vol,
            trend_strength=self._trend_strength(),
            hmm_state=self._current_hmm_state,
            probabilities=dict(probabilities),
        )

    def batch_update(self, prices: Iterable[float]) -> list[MarketRegime]:
        return [self.update(float(price)) for price in prices]

    def current_regime(self) -> MarketRegime:
        return MarketRegime(
            volatility=self._current_label,
            trend=self._current_trend,
            probability=float(self._last_probabilities.get(self._current_label.value, 0.25)),
            duration=self._current_duration,
            realized_volatility=self.realized_volatility(),
            trend_strength=self._trend_strength(),
            hmm_state=self._current_hmm_state,
            probabilities=dict(self._last_probabilities),
        )

    def realized_volatility(self) -> float:
        if len(self._returns) < 2:
            return 0.0
        window = np.asarray(list(self._returns)[-self.config.volatility_window :], dtype=np.float64)
        return float(np.std(window))

    def _trend_strength(self) -> float:
        if not self._returns:
            return 0.0
        window = np.asarray(list(self._returns)[-self.config.trend_window :], dtype=np.float64)
        return float(np.mean(window))

    def _classify_trend(self) -> TrendRegime:
        strength = self._trend_strength()
        if len(self._returns) < self.config.min_history:
            return TrendRegime.SIDEWAYS
        if strength >= self.config.trend_threshold:
            return TrendRegime.BULL
        if strength <= -self.config.trend_threshold:
            return TrendRegime.BEAR
        return TrendRegime.SIDEWAYS

    def _volatility_probabilities(self, realized_vol: float) -> dict[str, float]:
        if len(self._vol_history) < self.config.min_history:
            return self._uniform_probabilities()
        history = np.asarray(self._vol_history, dtype=np.float64)
        low_cut = float(np.quantile(history, self.config.low_vol_percentile))
        high_cut = float(np.quantile(history, self.config.high_vol_percentile))
        crisis_cut = float(np.quantile(history, self.config.crisis_vol_percentile))
        scores = {
            VolatilityRegime.LOW.value: max(low_cut - realized_vol, 0.0) + 1e-6,
            VolatilityRegime.MEDIUM.value: max(1.0 - abs(realized_vol - (low_cut + high_cut) * 0.5) / max(high_cut, 1e-6), 0.0)
            + 1e-6,
            VolatilityRegime.HIGH.value: max(realized_vol - high_cut * 0.8, 0.0) + 1e-6,
            VolatilityRegime.CRISIS.value: max(realized_vol - crisis_cut, 0.0) * 2.0 + 1e-6,
        }
        total = float(sum(scores.values()))
        if total <= 0:
            return self._uniform_probabilities()
        return {name: float(score / total) for name, score in scores.items()}

    def _infer_hmm_state(self, volume: float) -> tuple[int | None, dict[str, float]]:
        if len(self._returns) < self.config.min_history:
            return None, {}
        if not _HMM_AVAILABLE:
            return self._fallback_hmm_state(volume=volume)
        try:
            self._fit_hmm_if_needed(volume=volume)
            if self._hmm_model is None:
                return None, {}
            hmm_model = cast(Any, self._hmm_model)
            features = self._feature_matrix(volume=volume)
            hidden_states = hmm_model.predict(features)
            posteriors = hmm_model.predict_proba(features)[-1]
            return int(hidden_states[-1]), {f"hmm_state_{idx}": float(prob) for idx, prob in enumerate(posteriors)}
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("HMM regime inference failed, using fallback state: %s", exc)
            return self._fallback_hmm_state(volume=volume)

    def _fit_hmm_if_needed(self, volume: float) -> None:
        self._updates_since_hmm_fit += 1
        if self._hmm_model is not None and self._updates_since_hmm_fit < self.config.hmm_refit_interval:
            return
        features = self._feature_matrix(volume=volume)
        if len(features) < self.config.min_history:
            return
        if _GaussianHMM is None:
            return
        model = _GaussianHMM(
            n_components=self.config.hmm_states,
            covariance_type="diag",
            n_iter=100,
            random_state=7,
        )
        model.fit(features)
        self._hmm_model = model
        self._updates_since_hmm_fit = 0

    def _feature_matrix(self, volume: float) -> np.ndarray:
        returns = np.asarray(self._returns, dtype=np.float64)
        vol = np.asarray(self._vol_history, dtype=np.float64)
        if len(vol) < len(returns):
            vol = np.pad(vol, (len(returns) - len(vol), 0), constant_values=0.0)
        volume_feature = np.full_like(returns, float(np.tanh(volume)))
        return np.column_stack([returns, vol[-len(returns) :], volume_feature])

    def _fallback_hmm_state(self, volume: float) -> tuple[int, dict[str, float]]:
        state_index = list(VolatilityRegime).index(self._current_label)
        base = np.full(self.config.hmm_states, 1.0 / max(self.config.hmm_states, 1), dtype=np.float64)
        if state_index < len(base):
            base *= 0.5
            base[state_index] += 0.5
        base /= base.sum()
        return state_index, {f"hmm_state_{idx}": float(prob) for idx, prob in enumerate(base)}

    @staticmethod
    def _uniform_probabilities() -> dict[str, float]:
        return {
            VolatilityRegime.LOW.value: 0.25,
            VolatilityRegime.MEDIUM.value: 0.25,
            VolatilityRegime.HIGH.value: 0.25,
            VolatilityRegime.CRISIS.value: 0.25,
        }
