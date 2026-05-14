"""
Rolling Feature Normalizer — EWM-based online Z-score standardisation.

Replaces static ``sklearn.StandardScaler`` with a stateful online scaler that
adapts to distributional shifts over time.  No sklearn dependency.

Uses exponentially-weighted mean and variance (EWM) so recent observations
have higher weight.  This is drift-aware: a feature that changes regime (e.g.
funding rate during a bull vs bear market) is normalised against its *recent*
distribution rather than an stale training-time mean/std.

Usage::

    scaler = RollingZScorer(halflife=200, min_obs=30)

    # During live trading — call each cycle:
    normed = scaler.transform(raw_features)   # np.ndarray shape (n_features,)

    # Batch transform (DataFrame or 2-D array):
    normed_df = scaler.transform_batch(df.values)

    # Inspect current stats:
    scaler.stats()  # → dict of {feature_idx: {"mean": ..., "std": ...}}
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class _FeatureStats:
    """Per-feature EWM statistics."""
    ewm_mean: float = 0.0
    ewm_var: float = 1.0     # initialised to 1 to avoid div-by-zero on first call
    n_obs: int = 0


class RollingZScorer:
    """
    Online EWM-based Z-score normaliser for a fixed-width feature vector.

    Each feature column is independently tracked with its own EWM mean and
    variance.  After ``min_obs`` observations the normalised value is returned;
    before that a pass-through (divide by sqrt(variance init)) is used.

    Parameters
    ----------
    n_features : int | None
        Number of features.  If None, inferred from first ``transform`` call.
    halflife : int
        EWM half-life in observations.  Smaller = faster adaptation.
        Typical range: 50–500 (50 ≈ ~35 cycles, 200 ≈ 140 cycles).
    min_obs : int
        Minimum observations before returning normalised values (not raw).
    clip_sigma : float
        Clip normalised output to ±clip_sigma.  Use 0 to disable clipping.
    """

    def __init__(
        self,
        n_features: Optional[int] = None,
        halflife: int = 200,
        min_obs: int = 30,
        clip_sigma: float = 5.0,
    ) -> None:
        self.halflife = halflife
        self.min_obs = min_obs
        self.clip_sigma = clip_sigma
        # EWM decay factor: alpha = 1 - exp(-ln2 / halflife)
        self._alpha = 1.0 - np.exp(-np.log(2.0) / max(1, halflife))
        self._stats: List[_FeatureStats] = []
        if n_features is not None:
            self._init_stats(n_features)

    # ── Public API ────────────────────────────────────────────────────────

    def transform(self, x: Sequence[float]) -> np.ndarray:
        """
        Update EWM statistics with ``x`` and return the normalised vector.

        Parameters
        ----------
        x : array-like, shape (n_features,)
            Raw feature values for one observation.

        Returns
        -------
        np.ndarray shape (n_features,)
            Z-score normalised values (updated online before returning).
        """
        arr = np.asarray(x, dtype=float)
        if not self._stats:
            self._init_stats(len(arr))
        if len(arr) != len(self._stats):
            raise ValueError(
                f"RollingZScorer: expected {len(self._stats)} features, got {len(arr)}"
            )

        out = np.empty(len(arr), dtype=float)
        for i, (val, st) in enumerate(zip(arr, self._stats)):
            if np.isnan(val):
                out[i] = 0.0
                continue
            # Update EWM mean / variance
            if st.n_obs == 0:
                st.ewm_mean = val
                st.ewm_var = 1.0
            else:
                delta = val - st.ewm_mean
                st.ewm_mean += self._alpha * delta
                # EWM variance update: Welford-style with decay
                st.ewm_var = (1.0 - self._alpha) * (st.ewm_var + self._alpha * delta ** 2)
            st.n_obs += 1

            if st.n_obs < self.min_obs:
                out[i] = 0.0  # not enough data yet — return neutral
            else:
                std = max(float(np.sqrt(max(st.ewm_var, 1e-12))), 1e-10)
                z = (val - st.ewm_mean) / std
                if self.clip_sigma > 0:
                    z = float(np.clip(z, -self.clip_sigma, self.clip_sigma))
                out[i] = z

        return out

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Transform a 2-D array of shape (n_obs, n_features) row-by-row.

        Each row updates the internal EWM state, so call order matters.
        Returns an array of the same shape with Z-scored values.
        """
        if X.ndim != 2:
            raise ValueError("transform_batch expects 2-D array (n_obs, n_features)")
        out = np.empty_like(X, dtype=float)
        for i in range(X.shape[0]):
            out[i] = self.transform(X[i])
        return out

    def reset(self) -> None:
        """Reset all statistics (e.g. after a detected regime shift)."""
        for st in self._stats:
            st.ewm_mean = 0.0
            st.ewm_var = 1.0
            st.n_obs = 0
        logger.info("RollingZScorer: statistics reset (%d features)", len(self._stats))

    def stats(self) -> Dict[int, dict]:
        """Return current per-feature statistics for diagnostics."""
        return {
            i: {
                "mean": round(st.ewm_mean, 6),
                "std": round(float(np.sqrt(max(st.ewm_var, 0))), 6),
                "n_obs": st.n_obs,
            }
            for i, st in enumerate(self._stats)
        }

    @property
    def n_features(self) -> int:
        return len(self._stats)

    @property
    def is_warm(self) -> bool:
        """True once every feature has at least min_obs observations."""
        return bool(self._stats) and all(st.n_obs >= self.min_obs for st in self._stats)

    # ── Internal ──────────────────────────────────────────────────────────

    def _init_stats(self, n: int) -> None:
        self._stats = [_FeatureStats() for _ in range(n)]
