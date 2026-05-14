"""
Cross-Market Lead-Lag Discovery — find predictive relationships between series.

Identifies which price series *leads* others by computing cross-correlation
at various lags and testing for statistical significance.  This is useful
for finding pairs where one market (e.g. BTC on Binance) consistently moves
before another (e.g. ETH on Kraken) and exploiting that delay.

Example::

    disc = LeadLagDiscoverer()
    disc.add_series("BTC_Binance", btc_prices, btc_timestamps)
    disc.add_series("ETH_Kraken", eth_prices, eth_timestamps)

    relations = disc.discover_leads(max_lag=10)
    # → [LeadLagRelation(leader="BTC_Binance", follower="ETH_Kraken",
    #        optimal_lag=3, correlation=0.72, p_value=0.001)]

    signal = disc.get_trading_signal("BTC_Binance", "ETH_Kraken")
    # → 0.65  (buy ETH based on BTC's recent move)

Pure Python + numpy.  No exchange or config dependencies.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LeadLagRelation:
    """Discovered lead-lag relationship between two series.

    Attributes
    ----------
    leader : str
        Name of the leading series.
    follower : str
        Name of the following series.
    optimal_lag : int
        Number of bars by which the leader precedes the follower.
        Positive means leader moves first.
    correlation : float
        Cross-correlation at the optimal lag (absolute value).
    p_value : float
        Approximate p-value for the correlation being non-zero.
    """

    leader: str
    follower: str
    optimal_lag: int
    correlation: float
    p_value: float


class LeadLagDiscoverer:
    """Cross-market lead-lag discovery engine.

    Parameters
    ----------
    min_correlation : float
        Minimum absolute cross-correlation to report a relationship
        (default 0.3).
    significance_level : float
        Maximum p-value to consider a relationship significant
        (default 0.05).
    """

    def __init__(
        self,
        min_correlation: float = 0.3,
        significance_level: float = 0.05,
    ) -> None:
        self.min_correlation = min_correlation
        self.significance_level = significance_level

        self._series: Dict[str, np.ndarray] = {}
        self._timestamps: Dict[str, List] = {}

        logger.info(
            "LeadLagDiscoverer initialised: min_corr=%.2f sig_level=%.3f",
            self.min_correlation, self.significance_level,
        )

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_series(
        self,
        name: str,
        values: Sequence[float],
        timestamps: Optional[Sequence] = None,
    ) -> None:
        """Add a price series for lead-lag analysis.

        Parameters
        ----------
        name : str
            Unique identifier for this series (e.g. "BTC_Binance").
        values : list[float]
            Price values (ordered chronologically).
        timestamps : list, optional
            Parallel timestamps (informational, for alignment checks).
        """
        arr = np.asarray(values, dtype=np.float64)
        if len(arr) < 3:
            raise ValueError(f"Series '{name}' too short ({len(arr)} < 3)")

        self._series[name] = arr
        self._timestamps[name] = list(timestamps) if timestamps else []
        logger.info("Added series '%s' (%d bars)", name, len(arr))

    def remove_series(self, name: str) -> None:
        """Remove a series by name."""
        self._series.pop(name, None)
        self._timestamps.pop(name, None)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover_leads(
        self,
        max_lag: int = 10,
        min_overlap: int = 50,
    ) -> List[LeadLagRelation]:
        """Discover lead-lag relationships among all added series.

        For each pair (A, B), computes the cross-correlation of log
        returns at lags -max_lag to +max_lag.  If the peak absolute
        correlation exceeds ``min_correlation`` and the p-value is
        below ``significance_level``, a :class:`LeadLagRelation` is
        created with the leading series as ``leader``.

        Parameters
        ----------
        max_lag : int
            Maximum number of bars to test in each direction.
        min_overlap : int
            Minimum overlapping bars required between two series.

        Returns
        -------
        list[LeadLagRelation]
            Discovered relationships, sorted by absolute correlation
            (strongest first).
        """
        if max_lag < 1:
            raise ValueError(f"max_lag must be >= 1, got {max_lag}")

        names = sorted(self._series.keys())
        results: List[LeadLagRelation] = []

        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                relation = self._test_pair(name_a, name_b, max_lag, min_overlap)
                if relation is not None:
                    results.append(relation)

        results.sort(key=lambda r: abs(r.correlation), reverse=True)

        logger.info(
            "Lead-lag discovery: %d pairs tested, %d significant relationships found",
            len(names) * (len(names) - 1) // 2, len(results),
        )
        return results

    def _test_pair(
        self,
        name_a: str,
        name_b: str,
        max_lag: int,
        min_overlap: int,
    ) -> Optional[LeadLagRelation]:
        """Test a single pair for lead-lag relationship."""
        a = self._series[name_a]
        b = self._series[name_b]

        # Align lengths
        n = min(len(a), len(b))
        if n < min_overlap:
            logger.debug(
                "Pair %s/%s: overlap %d < %d, skipping",
                name_a, name_b, n, min_overlap,
            )
            return None

        # Log returns
        ret_a = np.diff(np.log(np.maximum(a[:n], 1e-10)))
        ret_b = np.diff(np.log(np.maximum(b[:n], 1e-10)))

        # Standardise
        ret_a = self._standardise(ret_a)
        ret_b = self._standardise(ret_b)
        if ret_a is None or ret_b is None:
            return None

        n_ret = len(ret_a)

        best_corr = 0.0
        best_lag = 0

        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                # Contemporaneous — not a lead-lag
                continue

            if lag > 0:
                # A leads B by `lag` bars: correlate A[:-lag] with B[lag:]
                seg_a = ret_a[: n_ret - lag]
                seg_b = ret_b[lag:]
            else:
                # B leads A by `-lag` bars: correlate A[-lag:] with B[:n+lag]
                seg_a = ret_a[-lag:]
                seg_b = ret_b[: n_ret + lag]

            if len(seg_a) < 10:
                continue

            corr = float(np.corrcoef(seg_a, seg_b)[0, 1])
            if math.isnan(corr):
                continue

            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

        if abs(best_corr) < self.min_correlation:
            return None

        # Approximate p-value using Fisher's z-transform
        effective_n = n_ret - abs(best_lag)
        p_value = self._correlation_pvalue(best_corr, effective_n)

        if p_value > self.significance_level:
            return None

        # Determine leader/follower
        if best_lag > 0:
            leader, follower, lag = name_a, name_b, best_lag
        else:
            leader, follower, lag = name_b, name_a, -best_lag

        relation = LeadLagRelation(
            leader=leader,
            follower=follower,
            optimal_lag=lag,
            correlation=best_corr,
            p_value=p_value,
        )
        logger.info(
            "Lead-lag found: %s leads %s by %d bars (corr=%.4f, p=%.6f)",
            leader, follower, lag, best_corr, p_value,
        )
        return relation

    # ------------------------------------------------------------------
    # Trading signal
    # ------------------------------------------------------------------

    def get_trading_signal(
        self,
        leader: str,
        follower: str,
        lookback: int = 5,
    ) -> float:
        """Generate a trading signal for the follower based on the leader's
        recent move.

        Parameters
        ----------
        leader : str
            Name of the leading series.
        follower : str
            Name of the following series.
        lookback : int
            Number of recent bars to use from the leader.

        Returns
        -------
        float
            Signal in [-1, 1].  Positive = bullish on follower,
            negative = bearish.  Magnitude indicates strength.
        """
        if leader not in self._series:
            logger.warning("Leader '%s' not found", leader)
            return 0.0
        if follower not in self._series:
            logger.warning("Follower '%s' not found", follower)
            return 0.0

        leader_prices = self._series[leader]
        if len(leader_prices) < lookback + 1:
            return 0.0

        # Recent log return of leader
        recent = leader_prices[-lookback - 1:]
        log_returns = np.diff(np.log(np.maximum(recent, 1e-10)))

        # Weighted average return (more recent bars weighted higher)
        weights = np.linspace(0.5, 1.0, len(log_returns))
        weights /= weights.sum()
        weighted_return = float(np.dot(log_returns, weights))

        # Normalise to [-1, 1] using tanh scaling
        signal = float(np.tanh(weighted_return * 50))

        logger.debug(
            "Lead-lag signal: %s→%s = %.4f (weighted_return=%.6f)",
            leader, follower, signal, weighted_return,
        )
        return signal

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _standardise(arr: np.ndarray) -> Optional[np.ndarray]:
        """Zero-mean, unit-variance standardisation.  Returns None if constant."""
        std = np.std(arr)
        if std < 1e-12:
            return None
        return (arr - np.mean(arr)) / std

    @staticmethod
    def _correlation_pvalue(r: float, n: int) -> float:
        """Approximate two-tailed p-value for Pearson correlation.

        Uses the t-distribution approximation:
            t = r * sqrt((n-2) / (1 - r^2))

        For large n (>30), approximates via the normal distribution.

        Parameters
        ----------
        r : float
            Pearson correlation coefficient.
        n : int
            Number of observations.

        Returns
        -------
        float
            Two-tailed p-value.
        """
        if n < 3:
            return 1.0
        if abs(r) >= 1.0 - 1e-12:
            return 0.0

        t_stat = r * math.sqrt((n - 2) / (1.0 - r * r))
        # Approximate using normal CDF for large samples
        # (good enough for n > 30, conservative for smaller n)
        p = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))
        return max(p, 0.0)

    def get_all_series_names(self) -> List[str]:
        """Return names of all added series."""
        return sorted(self._series.keys())


def _normal_cdf(x: float) -> float:
    """Approximate the standard normal CDF using the error function.

    Accurate to ~1e-7 for all x.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
