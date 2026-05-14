"""
Hurst Exponent Regime Detector — rolling Hurst estimation for regime detection.

The Hurst exponent H characterises the persistence of a time series:

  - H < 0.5  →  mean-reverting (anti-persistent)
  - H ≈ 0.5  →  random walk (Brownian motion)
  - H > 0.5  →  trending (persistent)

This module computes H using the **rescaled range (R/S) method** on a
rolling window, then classifies the market regime and recommends the
appropriate strategy type.

Example::

    hrd = HurstRegimeDetector(window=100)
    for price in live_prices:
        hrd.update("BTC/USD", price)

    h = hrd.get_hurst("BTC/USD")           # → 0.62 (trending)
    rec = hrd.get_strategy_recommendation("BTC/USD")  # → "momentum"
    history = hrd.get_regime_history("BTC/USD", lookback=500)

Pure Python + numpy.  No exchange or config dependencies.
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

# Regime thresholds
MEAN_REVERSION_UPPER = 0.45   # H below this → mean-reverting
TRENDING_LOWER = 0.55          # H above this → trending
# Between 0.45 and 0.55 → random walk / "avoid"


@dataclass
class RegimeSnapshot:
    """A single point-in-time regime reading."""

    timestamp: float
    hurst: float
    recommendation: str


class HurstRegimeDetector:
    """Rolling Hurst exponent regime detector.

    Parameters
    ----------
    window : int
        Rolling window size for Hurst computation (default 100).
    max_history : int
        Maximum number of price observations to retain per symbol
        (default 2000).
    mean_reversion_threshold : float
        Hurst below this → mean-reverting regime (default 0.45).
    trending_threshold : float
        Hurst above this → trending regime (default 0.55).
    """

    def __init__(
        self,
        window: int = 100,
        max_history: int = 2000,
        mean_reversion_threshold: float = MEAN_REVERSION_UPPER,
        trending_threshold: float = TRENDING_LOWER,
    ) -> None:
        if window < 20:
            raise ValueError(f"window must be >= 20, got {window}")
        if mean_reversion_threshold >= trending_threshold:
            raise ValueError(
                f"mean_reversion_threshold ({mean_reversion_threshold}) must be "
                f"< trending_threshold ({trending_threshold})"
            )

        self.window = window
        self.max_history = max_history
        self.mean_reversion_threshold = mean_reversion_threshold
        self.trending_threshold = trending_threshold

        # Per-symbol price history
        self._prices: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        # Per-symbol regime history
        self._regime_history: Dict[str, Deque[RegimeSnapshot]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )

        logger.info(
            "HurstRegimeDetector initialised: window=%d mr_thresh=%.2f trend_thresh=%.2f",
            self.window, self.mean_reversion_threshold, self.trending_threshold,
        )

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def update(self, symbol: str, price: float) -> Optional[float]:
        """Add a new price observation for *symbol*.

        If enough data is available, computes the Hurst exponent and
        records a regime snapshot.

        Parameters
        ----------
        symbol : str
            Trading pair identifier (e.g. "BTC/USD").
        price : float
            Latest price observation.

        Returns
        -------
        float or None
            The computed Hurst exponent, or None if insufficient data.
        """
        if price <= 0:
            logger.warning("Non-positive price %.6f for %s — ignored", price, symbol)
            return None

        self._prices[symbol].append(price)

        if len(self._prices[symbol]) < self.window:
            return None

        h = self._compute_hurst(list(self._prices[symbol])[-self.window:])
        rec = self._classify(h)

        snapshot = RegimeSnapshot(
            timestamp=time.time(),
            hurst=h,
            recommendation=rec,
        )
        self._regime_history[symbol].append(snapshot)

        logger.debug(
            "%s: H=%.4f → %s (prices=%d)",
            symbol, h, rec, len(self._prices[symbol]),
        )
        return h

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_hurst(
        self,
        symbol: str,
        window: Optional[int] = None,
    ) -> float:
        """Compute the current Hurst exponent for *symbol*.

        Parameters
        ----------
        symbol : str
            Trading pair.
        window : int, optional
            Override the instance window size.

        Returns
        -------
        float
            Hurst exponent.  Returns 0.5 (random walk) if insufficient data.
        """
        prices = self._prices.get(symbol)
        if not prices:
            logger.warning("No price data for '%s'", symbol)
            return 0.5

        w = window if window is not None else self.window
        if len(prices) < w:
            logger.debug(
                "%s: only %d prices, need %d — returning 0.5",
                symbol, len(prices), w,
            )
            return 0.5

        return self._compute_hurst(list(prices)[-w:])

    def get_strategy_recommendation(self, symbol: str) -> str:
        """Recommend a strategy type based on the current Hurst exponent.

        Parameters
        ----------
        symbol : str
            Trading pair.

        Returns
        -------
        str
            One of ``"mean_reversion"``, ``"momentum"``, or ``"avoid"``.
        """
        h = self.get_hurst(symbol)
        rec = self._classify(h)
        logger.info("%s: H=%.4f → recommendation=%s", symbol, h, rec)
        return rec

    def get_regime_history(
        self,
        symbol: str,
        lookback: int = 500,
    ) -> List[Tuple[float, float, str]]:
        """Return recent regime history for *symbol*.

        Parameters
        ----------
        symbol : str
            Trading pair.
        lookback : int
            Maximum number of snapshots to return (most recent first).

        Returns
        -------
        list[tuple[float, float, str]]
            Each tuple is (timestamp, hurst, recommendation).
        """
        history = self._regime_history.get(symbol)
        if not history:
            return []

        recent = list(history)[-lookback:]
        return [(s.timestamp, s.hurst, s.recommendation) for s in recent]

    def get_all_symbols(self) -> List[str]:
        """Return all symbols with price data."""
        return sorted(self._prices.keys())

    def get_regime_summary(self) -> Dict[str, Dict[str, float]]:
        """Return a summary of current Hurst exponents for all symbols.

        Returns
        -------
        dict[str, dict]
            Symbol → {"hurst": float, "recommendation": str}.
        """
        summary = {}
        for symbol in self._prices:
            h = self.get_hurst(symbol)
            summary[symbol] = {
                "hurst": h,
                "recommendation": self._classify(h),
            }
        return summary

    # ------------------------------------------------------------------
    # Hurst computation: Rescaled Range (R/S) method
    # ------------------------------------------------------------------

    def _compute_hurst(self, prices: List[float]) -> float:
        """Compute the Hurst exponent using the R/S method.

        The rescaled range is computed for multiple sub-series lengths,
        then H is estimated from the log-log slope of
        ``E[R(n)/S(n)]`` vs ``n``.

        Parameters
        ----------
        prices : list[float]
            Price series (minimum ~20 points).

        Returns
        -------
        float
            Hurst exponent, typically in [0, 1].  Clamped to [0.01, 0.99].
        """
        arr = np.asarray(prices, dtype=np.float64)
        n = len(arr)
        if n < 20:
            return 0.5

        # Log returns
        log_returns = np.diff(np.log(np.maximum(arr, 1e-10)))

        # Test multiple sub-series sizes
        # Use powers of 2 from 8 to n//2 (at least 3 different sizes)
        min_size = 8
        max_size = len(log_returns) // 2
        if max_size < min_size:
            return 0.5

        sizes = []
        s = min_size
        while s <= max_size:
            sizes.append(s)
            s = int(s * 1.5)  # geometric progression
        if not sizes:
            return 0.5

        log_sizes = []
        log_rs = []

        for size in sizes:
            rs_values = self._rescaled_range_for_size(log_returns, size)
            if rs_values:
                mean_rs = np.mean(rs_values)
                if mean_rs > 0:
                    log_sizes.append(math.log(size))
                    log_rs.append(math.log(mean_rs))

        if len(log_sizes) < 2:
            return 0.5

        # Linear regression: log(R/S) = H * log(n) + c
        x = np.array(log_sizes)
        y = np.array(log_rs)
        n_pts = len(x)

        # Ordinary least squares
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if abs(denominator) < 1e-12:
            return 0.5

        h = float(numerator / denominator)

        # Clamp to valid range
        h = max(0.01, min(0.99, h))
        return h

    @staticmethod
    def _rescaled_range_for_size(
        returns: np.ndarray,
        size: int,
    ) -> List[float]:
        """Compute R/S for all non-overlapping sub-series of given size.

        Parameters
        ----------
        returns : np.ndarray
            Return series.
        size : int
            Sub-series length.

        Returns
        -------
        list[float]
            R/S values for each sub-series.
        """
        n = len(returns)
        num_segments = n // size
        if num_segments == 0:
            return []

        rs_values: List[float] = []
        for i in range(num_segments):
            segment = returns[i * size: (i + 1) * size]

            mean_seg = np.mean(segment)
            deviations = np.cumsum(segment - mean_seg)

            r = float(np.max(deviations) - np.min(deviations))
            s = float(np.std(segment, ddof=1))

            if s > 1e-12:
                rs_values.append(r / s)

        return rs_values

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify(self, h: float) -> str:
        """Classify regime from Hurst exponent.

        Parameters
        ----------
        h : float
            Hurst exponent.

        Returns
        -------
        str
            ``"mean_reversion"``, ``"momentum"``, or ``"avoid"``.
        """
        if h < self.mean_reversion_threshold:
            return "mean_reversion"
        if h > self.trending_threshold:
            return "momentum"
        return "avoid"
