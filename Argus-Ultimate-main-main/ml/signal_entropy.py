"""
Signal Entropy Conviction Gate — Shannon entropy of the signal distribution.

When all signals agree (low entropy → high conviction), full position size is warranted.
When signals are conflicted (high entropy → low conviction), reduce size to protect capital.

This is a *coherence* gate: it doesn't evaluate whether any single signal is good,
it asks "are we sure enough collectively to commit capital?"

Physics analogy: entropy measures disorder. A crystal (ordered, low entropy) has
strong structure. A gas (disordered, high entropy) has no consistent direction.

Entropy calculation:
    1. Map each signal value to a probability-like weight: p_i = softmax(|signal_i|)
    2. Split signals into BULL (value > 0) and BEAR (value < 0) camps
    3. Compute "directional entropy": H = -sum(p_i * log(p_i)) over the direction distribution
    4. Normalise to [0, 1]: H_norm = H / log(n_signals)
    5. Conviction = 1 - H_norm

A conviction of 1.0 = all signals perfectly aligned.
A conviction of 0.0 = maximum disagreement.

Usage::

    gate = EntropyConvictionGate(min_signals=3)
    gate.update({"momentum": 0.6, "hmm": 0.5, "causal_graph": -0.4, "fear_greed": 0.3})
    conviction = gate.conviction()   # e.g. 0.72
    scale = gate.size_scale()        # e.g. 0.78 (applies a floor of 0.25)
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Minimum number of signals to compute meaningful entropy (below this → neutral)
_MIN_SIGNALS_FOR_ENTROPY = 2

# Below this absolute value a signal is considered "neutral" and excluded
_SIGNAL_THRESHOLD = 0.05

# Minimum size scale floor — even max-entropy still allows 25% of normal size
_MIN_SIZE_SCALE = 0.25


class EntropyConvictionGate:
    """
    Computes Shannon entropy of the active signal distribution to gate position size.

    High conviction (low entropy, signals agree) → scale approaches 1.0.
    Low conviction (high entropy, signals conflict) → scale approaches _MIN_SIZE_SCALE.

    Parameters
    ----------
    min_signals : int
        Minimum non-neutral signals for a valid entropy calculation.
    neutral_scale : float
        Scale to return when not enough signals are present.
    """

    def __init__(
        self,
        min_signals: int = _MIN_SIGNALS_FOR_ENTROPY,
        neutral_scale: float = 0.70,
    ) -> None:
        self.min_signals = min_signals
        self.neutral_scale = neutral_scale
        self._signals: Dict[str, float] = {}

    def update(self, signals: Dict[str, float]) -> None:
        """
        Replace the current signal snapshot.

        Parameters
        ----------
        signals : dict[str, float]
            Map from signal name → signal value in [-1, 1].
        """
        self._signals = {k: float(v) for k, v in signals.items()}

    def conviction(self) -> float:
        """
        Return conviction score in [0, 1].

        1.0 = all signals perfectly aligned.
        0.0 = maximum disagreement.
        0.5 = neutral (insufficient data or perfect balance).
        """
        active = {k: v for k, v in self._signals.items() if abs(v) >= _SIGNAL_THRESHOLD}
        n = len(active)
        if n < self.min_signals:
            return 0.5

        values = np.array(list(active.values()), dtype=float)

        # Split into directional camps
        bull_sum = float(np.sum(np.clip(values, 0, None)))
        bear_sum = float(np.sum(np.clip(-values, 0, None)))
        total = bull_sum + bear_sum
        if total < 1e-9:
            return 0.5

        # Direction distribution: [p_bull, p_bear]
        p_bull = bull_sum / total
        p_bear = bear_sum / total
        eps = 1e-10

        # Shannon entropy of direction distribution (max = log(2) for equal split)
        h = -(p_bull * math.log(p_bull + eps) + p_bear * math.log(p_bear + eps))
        h_max = math.log(2.0)

        # Normalised entropy → 0 = perfect alignment, 1 = perfect conflict
        h_norm = h / h_max

        # Additionally penalise weak signal magnitudes
        mean_abs = float(np.mean(np.abs(values)))
        magnitude_bonus = min(0.20, mean_abs * 0.40)

        conviction = float(np.clip(1.0 - h_norm + magnitude_bonus, 0.0, 1.0))
        return conviction

    def size_scale(self) -> float:
        """
        Return position size scale factor in [_MIN_SIZE_SCALE, 1.0].

        Scale = _MIN_SIZE_SCALE + (1 - _MIN_SIZE_SCALE) * conviction^0.5
        (square-root dampening so even moderate conviction allows decent sizing).
        """
        conv = self.conviction()
        if conv == 0.5 and len([v for v in self._signals.values() if abs(v) >= _SIGNAL_THRESHOLD]) < self.min_signals:
            return self.neutral_scale
        scale = _MIN_SIZE_SCALE + (1.0 - _MIN_SIZE_SCALE) * math.sqrt(conv)
        return float(np.clip(scale, _MIN_SIZE_SCALE, 1.0))

    def snapshot(self) -> dict:
        """Return diagnostic state."""
        conv = self.conviction()
        return {
            "conviction": round(conv, 4),
            "size_scale": round(self.size_scale(), 4),
            "n_signals": len(self._signals),
            "n_active": len([v for v in self._signals.values() if abs(v) >= _SIGNAL_THRESHOLD]),
            "bull_fraction": self._bull_fraction(),
        }

    def _bull_fraction(self) -> float:
        active = [v for v in self._signals.values() if abs(v) >= _SIGNAL_THRESHOLD]
        if not active:
            return 0.5
        bull = sum(1 for v in active if v > 0)
        return round(bull / len(active), 4)
