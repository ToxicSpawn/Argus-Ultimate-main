"""
HMM Regime Detector with Online Estimation
Probabilistic regime detection using Gaussian HMM with online Baum-Welch updates.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class RegimeState(str, Enum):
    BULL_QUIET = "bull_quiet"
    BULL_VOLATILE = "bull_volatile"
    BEAR_QUIET = "bear_quiet"
    BEAR_VOLATILE = "bear_volatile"
    CRISIS = "crisis"


@dataclass
class RegimeResult:
    state: RegimeState
    probabilities: Dict[str, float]
    confidence: float
    change_point: bool
    duration_bars: int


class GaussianHMM:
    """Simple Gaussian HMM with online parameter updates (no hmmlearn dependency)."""

    def __init__(self, n_states: int = 5, n_features: int = 2):
        self.n_states = n_states
        self.n_features = n_features

        # Transition matrix (row-stochastic)
        self.A = np.ones((n_states, n_states)) / n_states
        for i in range(n_states):
            self.A[i, i] = 0.85
            others = (1.0 - 0.85) / max(n_states - 1, 1)
            for j in range(n_states):
                if j != i:
                    self.A[i, j] = others

        # Initial state distribution
        self.pi = np.ones(n_states) / n_states

        # Emission parameters: mean and variance per state per feature
        self.means = np.zeros((n_states, n_features))
        self.vars = np.ones((n_states, n_features))

        # Pre-set means for interpretable states: [return, volatility]
        self.means[0] = [0.002, 0.01]   # bull_quiet
        self.means[1] = [0.005, 0.04]   # bull_volatile
        self.means[2] = [-0.002, 0.01]  # bear_quiet
        self.means[3] = [-0.005, 0.04]  # bear_volatile
        self.means[4] = [-0.02, 0.08]   # crisis

        self.vars[0] = [0.001, 0.005]
        self.vars[1] = [0.003, 0.01]
        self.vars[2] = [0.001, 0.005]
        self.vars[3] = [0.003, 0.01]
        self.vars[4] = [0.01, 0.03]

        self._alpha = np.zeros(n_states)
        self._n_updates = 0
        self._learning_rate = 0.01

    def _emission_prob(self, obs: np.ndarray) -> np.ndarray:
        probs = np.ones(self.n_states)
        for s in range(self.n_states):
            p = 1.0
            for f in range(self.n_features):
                var = max(self.vars[s, f], 1e-10)
                diff = obs[f] - self.means[s, f]
                p *= np.exp(-0.5 * diff**2 / var) / np.sqrt(2 * np.pi * var)
            probs[s] = max(p, 1e-300)
        return probs

    def forward_step(self, obs: np.ndarray) -> np.ndarray:
        emission = self._emission_prob(obs)
        if self._n_updates == 0:
            self._alpha = self.pi * emission
        else:
            self._alpha = (self.A.T @ self._alpha) * emission
        total = self._alpha.sum()
        if total > 1e-300:
            self._alpha /= total
        else:
            self._alpha = np.ones(self.n_states) / self.n_states
        self._n_updates += 1
        return self._alpha.copy()

    def online_update(self, obs: np.ndarray, gamma: np.ndarray) -> None:
        lr = self._learning_rate / (1 + self._n_updates * 0.0001)
        for s in range(self.n_states):
            w = gamma[s]
            if w < 0.01:
                continue
            for f in range(self.n_features):
                diff = obs[f] - self.means[s, f]
                self.means[s, f] += lr * w * diff
                self.vars[s, f] += lr * w * (diff**2 - self.vars[s, f])
                self.vars[s, f] = max(self.vars[s, f], 1e-8)

    def most_likely_state(self) -> int:
        return int(np.argmax(self._alpha))

    def state_probabilities(self) -> np.ndarray:
        return self._alpha.copy()


class ChangePointDetector:
    """Online change-point detection using CUSUM."""

    def __init__(self, threshold: float = 3.0, drift: float = 0.5, lookback: int = 50):
        self.threshold = float(threshold)
        self.drift = float(drift)
        self._s_pos = 0.0
        self._s_neg = 0.0
        self._values: deque = deque(maxlen=lookback)

    def update(self, value: float) -> bool:
        self._values.append(float(value))
        if len(self._values) < 5:
            return False
        mean = float(np.mean(list(self._values)))
        std = float(np.std(list(self._values)))
        if std < 1e-12:
            return False
        z = (value - mean) / std
        self._s_pos = max(0, self._s_pos + z - self.drift)
        self._s_neg = max(0, self._s_neg - z - self.drift)
        detected = self._s_pos > self.threshold or self._s_neg > self.threshold
        if detected:
            self._s_pos = 0.0
            self._s_neg = 0.0
        return detected


class MultiScaleRegimeDetector:
    """Multi-timeframe regime detector combining fast and slow signals."""

    def __init__(self, scales: List[int] = None):
        if scales is None:
            scales = [10, 30, 100]
        self.scales = scales
        self._returns: deque = deque(maxlen=max(scales) * 2)

    def update(self, ret: float) -> Dict[str, str]:
        self._returns.append(float(ret))
        if len(self._returns) < min(self.scales):
            return {f"scale_{s}": "unknown" for s in self.scales}
        rets = np.array(list(self._returns))
        result = {}
        for scale in self.scales:
            if len(rets) < scale:
                result[f"scale_{scale}"] = "unknown"
                continue
            window = rets[-scale:]
            mu = float(np.mean(window))
            vol = float(np.std(window))
            if vol > np.percentile(np.abs(rets), 85):
                regime = "high_vol"
            elif mu > 0.001:
                regime = "trend_up"
            elif mu < -0.001:
                regime = "trend_down"
            else:
                regime = "range"
            result[f"scale_{scale}"] = regime
        return result


STATE_MAP = {
    0: RegimeState.BULL_QUIET,
    1: RegimeState.BULL_VOLATILE,
    2: RegimeState.BEAR_QUIET,
    3: RegimeState.BEAR_VOLATILE,
    4: RegimeState.CRISIS,
}


class HMMRegimeDetector:
    """Full regime detector combining HMM + change-point + multi-scale."""

    def __init__(self, n_states: int = 5, vol_lookback: int = 20):
        self.hmm = GaussianHMM(n_states=n_states, n_features=2)
        self.change_detector = ChangePointDetector()
        self.multi_scale = MultiScaleRegimeDetector()
        self.vol_lookback = int(vol_lookback)
        self._returns: deque = deque(maxlen=500)
        self._prev_state: int = -1
        self._state_duration: int = 0

    def update(self, ret: float) -> RegimeResult:
        self._returns.append(float(ret))
        rets = np.array(list(self._returns))
        n = len(rets)
        if n < 5:
            return RegimeResult(
                state=RegimeState.BULL_QUIET,
                probabilities={s.value: 1.0 / 5 for s in RegimeState},
                confidence=0.0, change_point=False, duration_bars=0,
            )

        vol = float(np.std(rets[-min(self.vol_lookback, n):]))
        obs = np.array([float(ret), vol])
        gamma = self.hmm.forward_step(obs)
        self.hmm.online_update(obs, gamma)
        change_point = self.change_detector.update(ret)
        self.multi_scale.update(ret)

        state_idx = self.hmm.most_likely_state()
        state = STATE_MAP.get(state_idx, RegimeState.BULL_QUIET)

        if state_idx == self._prev_state:
            self._state_duration += 1
        else:
            self._state_duration = 1
            self._prev_state = state_idx

        probs = self.hmm.state_probabilities()
        prob_dict = {STATE_MAP[i].value: float(probs[i]) for i in range(len(probs))}

        return RegimeResult(
            state=state, probabilities=prob_dict,
            confidence=float(probs[state_idx]),
            change_point=change_point, duration_bars=self._state_duration,
        )

    def get_regime_for_sizing(self) -> Tuple[str, float]:
        if self._prev_state < 0:
            return "unknown", 0.0
        probs = self.hmm.state_probabilities()
        state = STATE_MAP.get(self._prev_state, RegimeState.BULL_QUIET)
        sizing_map = {
            RegimeState.BULL_QUIET: "trend_up",
            RegimeState.BULL_VOLATILE: "high_vol",
            RegimeState.BEAR_QUIET: "trend_down",
            RegimeState.BEAR_VOLATILE: "high_vol",
            RegimeState.CRISIS: "crash",
        }
        return sizing_map.get(state, "unknown"), float(probs[self._prev_state])
