#!/usr/bin/env python3
"""
Entropy-Based Signal Quality Analysis.

Provides information-theoretic tools for evaluating trading signal quality:

- **Shannon entropy** — measures signal randomness / information content.
- **Mutual information** — quantifies the predictive relationship between a
  signal and future returns.
- **Signal ranking** — orders multiple signals by their MI with returns.
- **Signal redundancy** — detects overlap between signal pairs (normalised MI).

Pure Python implementation with numpy acceleration when available.

Usage::

    ea = EntropyAnalyzer()
    mi = ea.compute_mutual_information(signal_values, forward_returns)
    ranking = ea.rank_signals({"momentum": mom, "rsi": rsi}, returns)
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False


class EntropyAnalyzer:
    """Information-theoretic signal quality analyser.

    All methods are stateless and can be called independently.
    """

    def __init__(self) -> None:
        logger.info("EntropyAnalyzer initialised")

    # ------------------------------------------------------------------
    # Shannon entropy
    # ------------------------------------------------------------------

    def compute_entropy(self, values: List[float], n_bins: int = 20) -> float:
        """Compute the Shannon entropy (in bits) of *values*.

        The continuous values are discretised into *n_bins* equal-width bins
        to estimate the probability distribution.

        Parameters
        ----------
        values:
            Sequence of scalar observations.
        n_bins:
            Number of histogram bins for discretisation.

        Returns
        -------
        float
            Shannon entropy in bits.  0 = perfectly deterministic,
            ``log2(n_bins)`` = uniform / maximum entropy.
        """
        if len(values) < 2:
            return 0.0

        counts = self._histogram(values, n_bins)
        total = sum(counts)
        if total == 0:
            return 0.0

        entropy = 0.0
        for c in counts:
            if c > 0:
                p = c / total
                entropy -= p * math.log2(p)

        return entropy

    # ------------------------------------------------------------------
    # Mutual information
    # ------------------------------------------------------------------

    def compute_mutual_information(
        self,
        signal: List[float],
        returns: List[float],
        n_bins: int = 20,
    ) -> float:
        """Estimate mutual information between *signal* and *returns*.

        MI(X;Y) = H(X) + H(Y) - H(X,Y)

        Higher MI means the signal carries more information about future returns.

        Parameters
        ----------
        signal:
            Signal values (must be same length as *returns*).
        returns:
            Corresponding forward returns.
        n_bins:
            Bins for each marginal axis (joint histogram is n_bins x n_bins).

        Returns
        -------
        float
            Mutual information in bits (non-negative).
        """
        n = min(len(signal), len(returns))
        if n < 10:
            logger.warning(
                "EntropyAnalyzer.compute_mutual_information: only %d samples "
                "(min 10 recommended)",
                n,
            )
            return 0.0

        sig = signal[:n]
        ret = returns[:n]

        h_signal = self.compute_entropy(sig, n_bins)
        h_returns = self.compute_entropy(ret, n_bins)
        h_joint = self._joint_entropy(sig, ret, n_bins)

        mi = h_signal + h_returns - h_joint
        # MI is theoretically non-negative; clamp to handle numerical noise
        mi = max(0.0, mi)
        return mi

    # ------------------------------------------------------------------
    # Signal ranking
    # ------------------------------------------------------------------

    def rank_signals(
        self,
        signals: Dict[str, List[float]],
        returns: List[float],
        n_bins: int = 20,
    ) -> List[Tuple[str, float]]:
        """Rank multiple signals by mutual information with *returns*.

        Parameters
        ----------
        signals:
            Mapping of ``signal_name → values``.
        returns:
            Forward returns to score against.
        n_bins:
            Histogram bins.

        Returns
        -------
        list[tuple[str, float]]
            ``(signal_name, mi_score)`` pairs sorted descending by MI.
        """
        scores: List[Tuple[str, float]] = []
        for name, sig_values in signals.items():
            mi = self.compute_mutual_information(sig_values, returns, n_bins)
            scores.append((name, round(mi, 6)))

        scores.sort(key=lambda x: x[1], reverse=True)
        logger.debug(
            "EntropyAnalyzer.rank_signals ranked %d signals: %s",
            len(scores),
            [(n, f"{s:.4f}") for n, s in scores[:5]],
        )
        return scores

    # ------------------------------------------------------------------
    # Signal redundancy
    # ------------------------------------------------------------------

    def get_signal_redundancy(
        self,
        signal_a: List[float],
        signal_b: List[float],
        n_bins: int = 20,
    ) -> float:
        """Compute normalised mutual information between two signals.

        Returns a value in [0, 1] where 1 means the signals are completely
        redundant (identical information content) and 0 means independent.

        ``NMI = 2 * MI(A, B) / (H(A) + H(B))``

        Parameters
        ----------
        signal_a:
            First signal values.
        signal_b:
            Second signal values (same length as *signal_a*).
        n_bins:
            Histogram bins.

        Returns
        -------
        float
            Normalised MI in [0, 1].
        """
        n = min(len(signal_a), len(signal_b))
        if n < 10:
            return 0.0

        a = signal_a[:n]
        b = signal_b[:n]

        h_a = self.compute_entropy(a, n_bins)
        h_b = self.compute_entropy(b, n_bins)

        if h_a + h_b < 1e-12:
            return 0.0

        mi = self.compute_mutual_information(a, b, n_bins)
        nmi = 2.0 * mi / (h_a + h_b)
        return max(0.0, min(1.0, nmi))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _histogram(self, values: List[float], n_bins: int) -> List[int]:
        """Compute a histogram of *values* into *n_bins* equal-width bins."""
        if not values:
            return [0] * n_bins

        if _HAS_NUMPY:
            arr = np.array(values, dtype=float)
            counts, _ = np.histogram(arr, bins=n_bins)
            return counts.tolist()

        # Pure Python fallback
        lo = min(values)
        hi = max(values)
        if hi == lo:
            counts = [0] * n_bins
            counts[0] = len(values)
            return counts

        width = (hi - lo) / n_bins
        counts = [0] * n_bins
        for v in values:
            idx = int((v - lo) / width)
            idx = min(idx, n_bins - 1)  # clamp upper edge
            counts[idx] += 1
        return counts

    def _joint_entropy(
        self, x: List[float], y: List[float], n_bins: int
    ) -> float:
        """Compute joint Shannon entropy H(X, Y) using a 2D histogram."""
        n = len(x)
        if n < 2:
            return 0.0

        if _HAS_NUMPY:
            arr_x = np.array(x, dtype=float)
            arr_y = np.array(y, dtype=float)
            hist_2d, _, _ = np.histogram2d(arr_x, arr_y, bins=n_bins)
            total = float(hist_2d.sum())
            if total == 0:
                return 0.0
            probs = hist_2d / total
            # Avoid log(0)
            mask = probs > 0
            entropy = -float(np.sum(probs[mask] * np.log2(probs[mask])))
            return entropy

        # Pure Python 2D histogram
        x_lo, x_hi = min(x), max(x)
        y_lo, y_hi = min(y), max(y)
        x_range = x_hi - x_lo if x_hi != x_lo else 1.0
        y_range = y_hi - y_lo if y_hi != y_lo else 1.0

        bins_2d: List[List[int]] = [[0] * n_bins for _ in range(n_bins)]
        for xi, yi in zip(x, y):
            bx = int((xi - x_lo) / x_range * n_bins)
            by = int((yi - y_lo) / y_range * n_bins)
            bx = min(bx, n_bins - 1)
            by = min(by, n_bins - 1)
            bins_2d[bx][by] += 1

        entropy = 0.0
        for row in bins_2d:
            for c in row:
                if c > 0:
                    p = c / n
                    entropy -= p * math.log2(p)
        return entropy
