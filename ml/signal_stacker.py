"""
Signal Stacker — ensemble combiner for multiple alpha signals.

Combines signals from different strategies using:
  1. Static weighted average (baseline)
  2. Adaptive weights based on recent signal performance
  3. Kelly-optimal weights given signal correlations

Input: List of (signal_name, signal_value, confidence) tuples where signal_value ∈ [-1, 1].
Output: Combined signal ∈ [-1, 1] with overall confidence.

Useful when running multiple strategies simultaneously and wanting to combine
their views into a single position-sizing decision.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SignalSource:
    """Represents a single signal contributor."""

    name: str
    value: float  # [-1, 1]
    confidence: float  # [0, 1]
    weight: float = 1.0
    last_accuracy: float = 0.5  # prior: coin-flip


@dataclass
class StackedSignal:
    """Output of the signal stacking operation."""

    combined_value: float  # [-1, 1]
    confidence: float  # [0, 1]
    method: str  # "static" | "adaptive" | "kelly"
    component_weights: Dict[str, float] = field(default_factory=dict)
    n_signals: int = 0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHOD_STATIC = "static"
METHOD_ADAPTIVE = "adaptive"
METHOD_KELLY = "kelly"

_VALID_METHODS = {METHOD_STATIC, METHOD_ADAPTIVE, METHOD_KELLY}

# Exponential smoothing factor for weight updates
_ALPHA_EWM = 0.1

# Minimum weight floor (prevents division by zero and extreme concentration)
_MIN_WEIGHT = 1e-4

# ---------------------------------------------------------------------------
# Signal Stacker
# ---------------------------------------------------------------------------


class SignalStacker:
    """
    Ensemble combiner for multiple named alpha signals.

    Parameters
    ----------
    method : str
        Combination method: "static", "adaptive", or "kelly".
    lookback : int
        Number of past outcomes stored per signal for Kelly covariance estimation.
    min_signals : int
        Minimum number of registered signals required to produce a non-neutral result.
    """

    def __init__(
        self,
        method: str = METHOD_ADAPTIVE,
        lookback: int = 50,
        min_signals: int = 1,
    ) -> None:
        if method not in _VALID_METHODS:
            raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}.")
        self.method = method
        self.lookback = lookback
        self.min_signals = min_signals

        # Registry of active signal sources, keyed by name
        self._sources: Dict[str, SignalSource] = {}

        # Historical signal values per source (for Kelly covariance)
        self._history: Dict[str, deque] = {}

        # Outcome accuracy history per source (1.0 = correct, 0.0 = wrong)
        self._accuracy_history: Dict[str, deque] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update_signal(self, name: str, value: float, confidence: float) -> None:
        """
        Register or update a named signal source.

        Parameters
        ----------
        name : str
            Unique signal identifier (e.g. "momentum_1d", "hmm_regime").
        value : float
            Signal value in [-1, 1]. Clamped if out of range.
        confidence : float
            Signal confidence in [0, 1]. Clamped if out of range.
        """
        value = float(np.clip(value, -1.0, 1.0))
        confidence = float(np.clip(confidence, 0.0, 1.0))

        if name not in self._sources:
            self._sources[name] = SignalSource(
                name=name,
                value=value,
                confidence=confidence,
                weight=1.0,
                last_accuracy=0.5,
            )
            self._history[name] = deque(maxlen=self.lookback)
            self._accuracy_history[name] = deque(maxlen=self.lookback)
            logger.debug("SignalStacker: registered new signal %r.", name)
        else:
            src = self._sources[name]
            src.value = value
            src.confidence = confidence

        self._history[name].append(value)

    def record_outcome(self, name: str, actual_direction: int) -> None:
        """
        Record whether a signal correctly predicted direction.

        Parameters
        ----------
        name : str
            Signal name to update.
        actual_direction : int
            +1 if price went up, -1 if price went down, 0 if flat.
        """
        if name not in self._sources:
            logger.warning("SignalStacker.record_outcome: unknown signal %r — ignoring.", name)
            return

        src = self._sources[name]
        signal_dir = 1 if src.value > 0 else (-1 if src.value < 0 else 0)
        accuracy = 1.0 if (signal_dir == actual_direction and actual_direction != 0) else 0.0

        self._accuracy_history[name].append(accuracy)

        # Exponential smoothing weight update
        src.last_accuracy = (1.0 - _ALPHA_EWM) * src.last_accuracy + _ALPHA_EWM * accuracy
        src.weight = max(_MIN_WEIGHT, src.last_accuracy)

        logger.debug(
            "SignalStacker.record_outcome: %r accuracy=%.3f, new weight=%.3f.",
            name,
            accuracy,
            src.weight,
        )

    def stack(self) -> StackedSignal:
        """
        Combine all active signals using the configured method.

        Returns
        -------
        StackedSignal
            Combined signal. Returns neutral signal if fewer than min_signals available.
        """
        active = [s for s in self._sources.values()]
        if len(active) < self.min_signals:
            logger.debug(
                "SignalStacker.stack: %d active signals < min_signals %d — returning neutral.",
                len(active),
                self.min_signals,
            )
            return StackedSignal(
                combined_value=0.0,
                confidence=0.0,
                method=self.method,
                component_weights={},
                n_signals=len(active),
            )

        if self.method == METHOD_STATIC:
            return self._static_stack()
        elif self.method == METHOD_ADAPTIVE:
            return self._adaptive_stack()
        else:
            return self._kelly_stack()

    def get_signal_stats(self) -> Dict[str, dict]:
        """
        Return per-signal statistics.

        Returns
        -------
        dict
            Mapping from signal name → {accuracy, weight, value, n_outcomes}.
        """
        stats: Dict[str, dict] = {}
        for name, src in self._sources.items():
            acc_history = list(self._accuracy_history.get(name, []))
            stats[name] = {
                "accuracy": src.last_accuracy,
                "weight": src.weight,
                "value": src.value,
                "confidence": src.confidence,
                "n_outcomes": len(acc_history),
                "recent_accuracy": float(np.mean(acc_history)) if acc_history else 0.5,
            }
        return stats

    # ------------------------------------------------------------------
    # Stacking implementations
    # ------------------------------------------------------------------

    def _static_stack(self) -> StackedSignal:
        """Equal-weight combination of all signals."""
        sources = list(self._sources.values())
        n = len(sources)
        equal_weight = 1.0 / n
        weights = {s.name: equal_weight for s in sources}

        combined = sum(s.value * equal_weight for s in sources)
        combined = float(np.clip(combined, -1.0, 1.0))
        avg_confidence = float(np.mean([s.confidence for s in sources]))

        return StackedSignal(
            combined_value=combined,
            confidence=avg_confidence,
            method=METHOD_STATIC,
            component_weights=weights,
            n_signals=n,
        )

    def _adaptive_stack(self) -> StackedSignal:
        """
        Accuracy-weighted combination.
        Weights are proportional to each signal's exponentially-smoothed accuracy.
        """
        sources = list(self._sources.values())
        raw_weights = np.array([max(s.weight, _MIN_WEIGHT) for s in sources])
        norm_weights = raw_weights / raw_weights.sum()

        combined = float(np.dot([s.value for s in sources], norm_weights))
        combined = float(np.clip(combined, -1.0, 1.0))

        # Confidence: weighted average of individual confidences
        avg_confidence = float(np.dot([s.confidence for s in sources], norm_weights))

        weights_dict = {s.name: float(w) for s, w in zip(sources, norm_weights)}

        return StackedSignal(
            combined_value=combined,
            confidence=avg_confidence,
            method=METHOD_ADAPTIVE,
            component_weights=weights_dict,
            n_signals=len(sources),
        )

    def _kelly_stack(self) -> StackedSignal:
        """
        Kelly-optimal combination using inverse covariance of signal history.
        Falls back to adaptive if covariance matrix is not invertible.
        """
        sources = list(self._sources.values())
        n = len(sources)

        # Build signal history matrix (lookback × n_signals)
        histories: List[np.ndarray] = []
        min_len = self.lookback
        for s in sources:
            h = np.array(list(self._history[s.name]))
            histories.append(h)
            min_len = min(min_len, len(h))

        if min_len < 5:
            logger.debug("SignalStacker._kelly_stack: insufficient history — using adaptive.")
            result = self._adaptive_stack()
            result.method = METHOD_KELLY
            return result

        # Trim to common length
        mat = np.column_stack([h[-min_len:] for h in histories])  # (T, n)

        try:
            cov = np.cov(mat, rowvar=False)  # (n, n)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])
            # Regularise with small diagonal
            cov += np.eye(n) * 1e-6
            inv_cov = np.linalg.inv(cov)
            # Kelly weights: w ∝ Sigma^{-1} * mu (mean returns vector)
            mu = np.array([np.mean(list(self._history[s.name])) for s in sources])
            raw_weights = inv_cov @ mu
            # Allow only long/short — normalise by absolute sum
            abs_sum = np.sum(np.abs(raw_weights))
            if abs_sum < _MIN_WEIGHT:
                raise ValueError("Kelly weights near zero.")
            norm_weights = raw_weights / abs_sum
        except Exception as exc:
            logger.debug("Kelly weight computation failed (%s) — using adaptive.", exc)
            result = self._adaptive_stack()
            result.method = METHOD_KELLY
            return result

        combined = float(np.dot([s.value for s in sources], norm_weights))
        combined = float(np.clip(combined, -1.0, 1.0))
        avg_confidence = float(np.mean([s.confidence for s in sources]))

        weights_dict = {s.name: float(w) for s, w in zip(sources, norm_weights)}

        return StackedSignal(
            combined_value=combined,
            confidence=avg_confidence,
            method=METHOD_KELLY,
            component_weights=weights_dict,
            n_signals=n,
        )
