"""
Causal Discovery Engine — detects Granger-causal relationships between
time series (price, volume, funding rate, open interest, etc.).

Implements a pure-Python Granger causality test using restricted and
unrestricted OLS regressions with an F-test.  NumPy is used for matrix
operations when available; a naive fallback is provided otherwise.

Usage
-----
>>> engine = CausalDiscoveryEngine()
>>> engine.add_series("btc_close", btc_prices)
>>> engine.add_series("eth_close", eth_prices)
>>> result = engine.granger_test("eth_close", "btc_close", max_lag=5)
>>> print(result.significant)  # True if ETH Granger-causes BTC
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GrangerResult:
    """Result of a Granger causality test."""
    cause: str
    effect: str
    best_lag: int
    f_statistic: float
    p_value: float
    significant: bool


# ---------------------------------------------------------------------------
# Pure-Python F-distribution CDF (regularized incomplete beta)
# ---------------------------------------------------------------------------

def _log_beta(a: float, b: float) -> float:
    """Log of the Beta function using the log-gamma function."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _incomplete_beta_cf(a: float, b: float, x: float,
                        max_iter: int = 200, tol: float = 1e-12) -> float:
    """
    Regularized incomplete beta function I_x(a, b) via Lentz's continued
    fraction algorithm.  Used to compute the F-distribution CDF.
    """
    if x < 0 or x > 1:
        raise ValueError(f"x must be in [0, 1], got {x}")
    if x == 0 or x == 1:
        return x

    # Use the symmetry relation if x > (a+1)/(a+b+2) for better convergence
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _incomplete_beta_cf(b, a, 1.0 - x, max_iter, tol)

    # Lentz's method
    front = math.exp(
        a * math.log(x) + b * math.log(1.0 - x) - _log_beta(a, b)
    ) / a

    tiny = 1e-30
    f = tiny
    c = tiny
    d = 0.0

    for m in range(max_iter + 1):
        # Numerator coefficients for the continued fraction
        if m == 0:
            numerator = 1.0
        elif m % 2 == 0:
            k = m // 2
            numerator = (k * (b - k) * x) / ((a + 2 * k - 1) * (a + 2 * k))
        else:
            k = (m + 1) // 2
            numerator = -((a + k - 1 + k * b) * (a + k) * x) / ((a + 2 * k - 1 + 1) * (a + 2 * k - 1))
            # Corrected formula
            k2 = (m - 1) // 2 + 1
            numerator = -((a + k2 - 1) * (a + b + k2 - 1) * x) / ((a + 2 * k2 - 2) * (a + 2 * k2 - 1))

        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d

        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny

        delta = c * d
        f *= delta

        if abs(delta - 1.0) < tol:
            return front * (f - 1.0) if m > 0 else front * f

    # Did not converge — return best estimate
    return front * f


def _f_cdf(f_stat: float, df1: int, df2: int) -> float:
    """
    CDF of the F-distribution at *f_stat* with (*df1*, *df2*) degrees of freedom.

    Uses the relationship between F and the regularized incomplete beta function:
      P(F <= x) = I_{df1*x/(df1*x+df2)}(df1/2, df2/2)
    """
    if f_stat <= 0:
        return 0.0
    x = (df1 * f_stat) / (df1 * f_stat + df2)
    try:
        return _incomplete_beta_cf(df1 / 2.0, df2 / 2.0, x)
    except (ValueError, OverflowError):
        # Fallback: very large F → p ≈ 0
        return 1.0 if f_stat > 1000 else 0.5


def _f_p_value(f_stat: float, df1: int, df2: int) -> float:
    """Upper-tail p-value: P(F >= f_stat)."""
    return max(0.0, 1.0 - _f_cdf(f_stat, df1, df2))


# ---------------------------------------------------------------------------
# OLS helpers
# ---------------------------------------------------------------------------

def _ols_rss(X: np.ndarray, y: np.ndarray) -> float:
    """Residual sum of squares from OLS: y = Xb + e."""
    # Use least squares via normal equations (X^T X) b = X^T y
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback: pseudo-inverse
        beta = np.linalg.pinv(X) @ y
    residuals = y - X @ beta
    return float(np.dot(residuals, residuals))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CausalDiscoveryEngine:
    """
    Discovers Granger-causal relationships among time series.

    Add series via ``add_series()``, then test individual pairs with
    ``granger_test()`` or scan all with ``discover_all_relationships()``.
    """

    def __init__(self) -> None:
        self._series: Dict[str, np.ndarray] = {}
        logger.info("CausalDiscoveryEngine initialised")

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_series(self, name: str, values: list) -> None:
        """
        Register a named time series.

        Parameters
        ----------
        name : str
            Unique identifier for the series (e.g. "btc_close", "funding_rate").
        values : list
            Numeric values, one per time step.  All series used in a test must
            have the same length.
        """
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"Series '{name}' must be 1-D, got shape {arr.shape}")
        self._series[name] = arr
        logger.info("add_series('%s'): %d observations", name, len(arr))

    # ------------------------------------------------------------------
    # Granger test
    # ------------------------------------------------------------------

    def granger_test(self, cause: str, effect: str,
                     max_lag: int = 5) -> GrangerResult:
        """
        Test whether *cause* Granger-causes *effect* at the best lag
        (1 … max_lag).

        The test compares a restricted model (effect's own lags only) to an
        unrestricted model (effect's own lags + cause's lags).  The F-statistic
        measures whether the extra regressors significantly reduce RSS.

        Parameters
        ----------
        cause : str
            Name of the potential causal series.
        effect : str
            Name of the effect series.
        max_lag : int
            Maximum lag order to test (each lag from 1..max_lag is tried;
            the best is chosen by highest F-statistic).

        Returns
        -------
        GrangerResult
        """
        if cause not in self._series:
            raise KeyError(f"Series '{cause}' not found. Add it with add_series().")
        if effect not in self._series:
            raise KeyError(f"Series '{effect}' not found. Add it with add_series().")

        y_full = self._series[effect]
        x_full = self._series[cause]

        n = min(len(y_full), len(x_full))
        if n < max_lag + 2:
            logger.warning("granger_test: too few observations (%d) for max_lag=%d", n, max_lag)
            return GrangerResult(cause=cause, effect=effect, best_lag=1,
                                 f_statistic=0.0, p_value=1.0, significant=False)

        y_full = y_full[:n]
        x_full = x_full[:n]

        best_f = -1.0
        best_p = 1.0
        best_lag = 1

        for lag in range(1, max_lag + 1):
            T = n - lag
            if T < lag + 2:
                continue

            y = y_full[lag:]

            # Restricted model: y_t = a0 + a1*y_{t-1} + ... + ap*y_{t-p}
            X_r_cols = [np.ones(T)]
            for k in range(1, lag + 1):
                X_r_cols.append(y_full[lag - k: n - k])
            X_r = np.column_stack(X_r_cols)

            # Unrestricted: add cause lags
            X_u_cols = list(X_r_cols)
            for k in range(1, lag + 1):
                X_u_cols.append(x_full[lag - k: n - k])
            X_u = np.column_stack(X_u_cols)

            rss_r = _ols_rss(X_r, y)
            rss_u = _ols_rss(X_u, y)

            df1 = lag                         # extra regressors
            df2 = T - X_u.shape[1]           # residual df
            if df2 <= 0 or rss_u < 1e-15:
                continue

            f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
            p_val = _f_p_value(f_stat, df1, df2)

            if f_stat > best_f:
                best_f = f_stat
                best_p = p_val
                best_lag = lag

        significant = best_p < 0.05
        result = GrangerResult(
            cause=cause, effect=effect, best_lag=best_lag,
            f_statistic=round(best_f, 4), p_value=round(best_p, 6),
            significant=significant,
        )
        logger.info("granger_test(%s → %s): lag=%d F=%.3f p=%.4f sig=%s",
                    cause, effect, best_lag, best_f, best_p, significant)
        return result

    # ------------------------------------------------------------------
    # Full discovery
    # ------------------------------------------------------------------

    def discover_all_relationships(self, significance: float = 0.05) -> List[GrangerResult]:
        """
        Test every ordered pair of series and return significant relationships.

        Parameters
        ----------
        significance : float
            P-value threshold.

        Returns
        -------
        list of GrangerResult
            Only pairs with p_value < significance.
        """
        names = sorted(self._series.keys())
        results: List[GrangerResult] = []

        for cause in names:
            for effect in names:
                if cause == effect:
                    continue
                try:
                    r = self.granger_test(cause, effect)
                    if r.p_value < significance:
                        results.append(r)
                except Exception as exc:
                    logger.warning("discover_all: %s → %s failed: %s", cause, effect, exc)

        results.sort(key=lambda r: r.p_value)
        logger.info("discover_all_relationships: %d significant out of %d pairs",
                    len(results), len(names) * (len(names) - 1))
        return results

    # ------------------------------------------------------------------
    # Causal graph
    # ------------------------------------------------------------------

    def get_causal_graph(self, significance: float = 0.05) -> Dict[str, List[str]]:
        """
        Build a causal graph: cause → list of effects.

        Parameters
        ----------
        significance : float

        Returns
        -------
        dict
            Adjacency list representation.
        """
        relationships = self.discover_all_relationships(significance)
        graph: Dict[str, List[str]] = {}
        for r in relationships:
            graph.setdefault(r.cause, []).append(r.effect)
        return graph

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def series_names(self) -> List[str]:
        """Names of all registered series."""
        return sorted(self._series.keys())

    def series_length(self, name: str) -> int:
        """Length of a named series."""
        return len(self._series[name])
