"""
Permutation Entropy — measures time series complexity via ordinal patterns.

The permutation entropy (PE) quantifies the diversity of ordinal patterns
in a time series. Key properties:

  - PE ≈ 0  → highly predictable (repetitive patterns)
  - PE ≈ 1  → highly complex/chaotic (random-like)
  - Sudden PE changes → regime transitions

This module computes PE using the Bandt-Pompe method on a rolling window,
then classifies market predictability and recommends trading behavior.

Example::

    pe = PermutationEntropy(window=100, order=3, delay=1)
    for price in live_prices:
        pe.update("BTC/USD", price)

    entropy = pe.get_entropy("BTC/USD")           # → 0.85 (complex)
    rec = pe.get_recommendation("BTC/USD")        # → "normal"
    predictability = pe.get_predictability("BTC/USD")  # → 0.15

Pure Python + numpy. No exchange or config dependencies.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Entropy thresholds
LOW_COMPLEXITY_THRESHOLD = 0.3    # PE below this → highly predictable
HIGH_COMPLEXITY_THRESHOLD = 0.8   # PE above this → chaotic/random
# Between thresholds → normal market behavior


@dataclass
class EntropySnapshot:
    """A single point-in-time entropy reading."""

    timestamp: float
    entropy: float
    predictability: float  # 1 - entropy (inverted for intuition)
    recommendation: str


class PermutationEntropy:
    """Rolling permutation entropy calculator for regime detection.

    Parameters
    ----------
    window : int
        Rolling window size for entropy computation (default 100).
    order : int
        Ordinal pattern order (embedding dimension) (default 3).
        Higher order captures more complex patterns but needs more data.
    delay : int
        Time delay for embedding (default 1).
    max_history : int
        Maximum number of price observations to retain per symbol (default 2000).
    low_threshold : float
        PE below this → highly predictable regime (default 0.3).
    high_threshold : float
        PE above this → chaotic regime (default 0.8).
    """

    def __init__(
        self,
        window: int = 100,
        order: int = 3,
        delay: int = 1,
        max_history: int = 2000,
        low_threshold: float = LOW_COMPLEXITY_THRESHOLD,
        high_threshold: float = HIGH_COMPLEXITY_THRESHOLD,
    ) -> None:
        if window < 20:
            raise ValueError(f"window must be >= 20, got {window}")
        if order < 2 or order > 7:
            raise ValueError(f"order must be 2-7, got {order}")
        if delay < 1:
            raise ValueError(f"delay must be >= 1, got {delay}")
        if low_threshold >= high_threshold:
            raise ValueError(
                f"low_threshold ({low_threshold}) must be < high_threshold ({high_threshold})"
            )

        self.window = window
        self.order = order
        self.delay = delay
        self.max_history = max_history
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        # Per-symbol price history
        self._prices: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        # Per-symbol entropy history
        self._entropy_history: Dict[str, Deque[EntropySnapshot]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )

        logger.info(
            "PermutationEntropy initialised: window=%d order=%d delay=%d "
            "low_thresh=%.2f high_thresh=%.2f",
            self.window, self.order, self.delay,
            self.low_threshold, self.high_threshold,
        )

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def update(self, symbol: str, price: float) -> Optional[float]:
        """Add a new price observation for *symbol*.

        If enough data is available, computes the permutation entropy and
        records an entropy snapshot.

        Parameters
        ----------
        symbol : str
            Trading pair identifier (e.g. "BTC/USD").
        price : float
            Latest price observation.

        Returns
        -------
        float or None
            The computed permutation entropy (0-1), or None if insufficient data.
        """
        if price <= 0:
            logger.warning("Non-positive price %.6f for %s — ignored", price, symbol)
            return None

        self._prices[symbol].append(price)

        # Need order * delay + 1 points for one pattern, plus window for statistics
        min_points = (self.order - 1) * self.delay + 1 + self.window
        if len(self._prices[symbol]) < min_points:
            return None

        # Compute on recent window
        recent_prices = list(self._prices[symbol])[-self.window:]
        pe = self._compute_entropy(recent_prices)
        predictability = 1.0 - pe
        rec = self._classify(pe)

        snapshot = EntropySnapshot(
            timestamp=time.time(),
            entropy=pe,
            predictability=predictability,
            recommendation=rec,
        )
        self._entropy_history[symbol].append(snapshot)

        logger.debug(
            "%s: PE=%.4f predict=%.4f → %s (prices=%d)",
            symbol, pe, predictability, rec, len(self._prices[symbol]),
        )
        return pe

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_entropy(
        self,
        symbol: str,
        window: Optional[int] = None,
    ) -> float:
        """Compute the current permutation entropy for *symbol*.

        Parameters
        ----------
        symbol : str
            Trading pair.
        window : int, optional
            Override the instance window size.

        Returns
        -------
        float
            Permutation entropy (0-1). Returns 0.5 if insufficient data.
        """
        prices = self._prices.get(symbol)
        if not prices:
            logger.warning("No price data for '%s'", symbol)
            return 0.5

        w = window if window is not None else self.window
        min_points = (self.order - 1) * self.delay + 1 + w
        if len(prices) < min_points:
            logger.debug(
                "%s: only %d prices, need %d — returning 0.5",
                symbol, len(prices), min_points,
            )
            return 0.5

        return self._compute_entropy(list(prices)[-w:])

    def get_predictability(self, symbol: str) -> float:
        """Get the predictability score (1 - entropy) for *symbol*.

        Higher values indicate more predictable patterns.
        """
        return 1.0 - self.get_entropy(symbol)

    def get_recommendation(self, symbol: str) -> str:
        """Get trading recommendation based on current entropy.

        Parameters
        ----------
        symbol : str
            Trading pair.

        Returns
        -------
        str
            One of ``"highly_predictable"``, ``"normal"``, or ``"chaotic"``.
        """
        pe = self.get_entropy(symbol)
        rec = self._classify(pe)
        logger.info("%s: PE=%.4f → recommendation=%s", symbol, pe, rec)
        return rec

    def get_entropy_history(
        self,
        symbol: str,
        lookback: int = 500,
    ) -> List[Tuple[float, float, float, str]]:
        """Return recent entropy history for *symbol*.

        Parameters
        ----------
        symbol : str
            Trading pair.
        lookback : int
            Maximum number of snapshots to return (most recent first).

        Returns
        -------
        list[tuple[float, float, float, str]]
            Each tuple is (timestamp, entropy, predictability, recommendation).
        """
        history = self._entropy_history.get(symbol)
        if not history:
            return []

        recent = list(history)[-lookback:]
        return [
            (s.timestamp, s.entropy, s.predictability, s.recommendation)
            for s in recent
        ]

    def get_all_symbols(self) -> List[str]:
        """Return all symbols with price data."""
        return sorted(self._prices.keys())

    def get_entropy_summary(self) -> Dict[str, Dict[str, float]]:
        """Return a summary of current entropy for all symbols.

        Returns
        -------
        dict[str, dict]
            Symbol → {"entropy": float, "predictability": float, "recommendation": str}.
        """
        summary = {}
        for symbol in self._prices:
            pe = self.get_entropy(symbol)
            summary[symbol] = {
                "entropy": pe,
                "predictability": 1.0 - pe,
                "recommendation": self._classify(pe),
            }
        return summary

    # ------------------------------------------------------------------
    # Permutation entropy computation: Bandt-Pompe method
    # ------------------------------------------------------------------

    def _compute_entropy(self, prices: List[float]) -> float:
        """Compute permutation entropy using the Bandt-Pompe method.

        Parameters
        ----------
        prices : list[float]
            Price series (minimum ~order*delay + window points).

        Returns
        -------
        float
            Normalized permutation entropy in [0, 1].
        """
        arr = np.asarray(prices, dtype=np.float64)
        n = len(arr)

        # Minimum points needed
        min_points = (self.order - 1) * self.delay + 1
        if n < min_points:
            return 0.5

        # Compute ordinal patterns
        patterns = self._get_ordinal_patterns(arr)
        if len(patterns) == 0:
            return 0.5

        # Count pattern frequencies
        unique, counts = np.unique(patterns, return_counts=True)
        probs = counts / len(patterns)

        # Compute Shannon entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-12))

        # Normalize by maximum possible entropy (log2 of number of permutations)
        max_entropy = math.log2(math.factorial(self.order))
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 0.0

        # Clamp to [0, 1]
        return float(max(0.0, min(1.0, normalized_entropy)))

    def _get_ordinal_patterns(self, series: np.ndarray) -> np.ndarray:
        """Extract ordinal patterns from time series using Bandt-Pompe method.

        Parameters
        ----------
        series : np.ndarray
            Time series array.

        Returns
        -------
        np.ndarray
            Array of ordinal pattern indices.
        """
        n = len(series)
        m = self.order
        tau = self.delay

        # Number of embedded vectors
        num_vectors = n - (m - 1) * tau
        if num_vectors <= 0:
            return np.array([], dtype=int)

        patterns = []
        for i in range(num_vectors):
            # Extract embedded vector
            vector = series[i:i + (m - 1) * tau + 1:tau]
            if len(vector) != m:
                continue

            # Get ordinal pattern (argsort gives the permutation)
            pattern = np.argsort(vector)
            # Convert permutation to index (Lehmer code / factorial number system)
            pattern_idx = self._permutation_to_index(pattern)
            patterns.append(pattern_idx)

        return np.array(patterns, dtype=int)

    def _permutation_to_index(self, perm: np.ndarray) -> int:
        """Convert a permutation to its lexicographic index.

        Parameters
        ----------
        perm : np.ndarray
            Permutation array (e.g., [2, 0, 1]).

        Returns
        -------
        int
            Lexicographic index of the permutation.
        """
        n = len(perm)
        index = 0
        factor = 1

        for i in range(n - 1, -1, -1):
            # Count elements smaller than perm[i] that appear after position i
            count = sum(1 for j in range(i + 1, n) if perm[j] < perm[i])
            index += count * factor
            factor *= (n - i)

        return index

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify(self, pe: float) -> str:
        """Classify market predictability from permutation entropy.

        Parameters
        ----------
        pe : float
            Permutation entropy (0-1).

        Returns
        -------
        str
            ``"highly_predictable"``, ``"normal"``, or ``"chaotic"``.
        """
        if pe < self.low_threshold:
            return "highly_predictable"
        if pe > self.high_threshold:
            return "chaotic"
        return "normal"

    def get_trading_scalar(self, symbol: str) -> float:
        """Get position size scalar based on predictability.

        - highly_predictable (PE < 0.3): 1.2 (increase size - patterns are clear)
        - normal (0.3 <= PE <= 0.8): 1.0 (standard size)
        - chaotic (PE > 0.8): 0.5 (reduce size - too random)

        Returns
        -------
        float
            Position size multiplier (0.5 to 1.2).
        """
        rec = self.get_recommendation(symbol)
        scalars = {
            "highly_predictable": 1.2,
            "normal": 1.0,
            "chaotic": 0.5,
        }
        return scalars.get(rec, 1.0)

    def is_regime_change(self, symbol: str, lookback: int = 20) -> bool:
        """Detect if a regime change occurred based on entropy shift.

        Parameters
        ----------
        symbol : str
            Trading pair.
        lookback : int
            Number of recent snapshots to compare.

        Returns
        -------
        bool
            True if recommendation changed in the last `lookback` snapshots.
        """
        history = self._entropy_history.get(symbol)
        if not history or len(history) < lookback:
            return False

        recent = list(history)[-lookback:]
        recommendations = [s.recommendation for s in recent]

        # Check if there are at least 2 different recommendations
        unique_recs = set(recommendations)
        return len(unique_recs) > 1


__all__ = [
    "PermutationEntropy",
    "EntropySnapshot",
    "LOW_COMPLEXITY_THRESHOLD",
    "HIGH_COMPLEXITY_THRESHOLD",
]
