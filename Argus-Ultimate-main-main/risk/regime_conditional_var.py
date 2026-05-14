"""
Regime-Conditional Value at Risk — computes VaR and CVaR using the method
most appropriate for the current market regime.

Regime → Method mapping (default):
    normal    → Parametric (Gaussian)
    crisis    → Historical simulation
    volatile  → Monte Carlo (1000 simulations)
    trending  → Parametric with drift adjustment
    *other*   → Parametric fallback

All methods are implemented in pure Python with NumPy acceleration.
No external risk library dependencies.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class VaRResult:
    """Result of a VaR computation."""
    var_pct: float          # Value at Risk as a positive percentage loss
    cvar_pct: float         # Conditional VaR (Expected Shortfall)
    method_used: str        # "parametric", "historical", "monte_carlo"
    regime: str
    confidence: float       # e.g. 0.99
    n_samples: int
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _norm_ppf(p: float) -> float:
    """
    Inverse CDF (percent-point function) of the standard normal distribution.

    Uses the rational approximation from Abramowitz and Stegun (26.2.23)
    with absolute error < 4.5e-4.
    """
    if p <= 0:
        return -10.0
    if p >= 1:
        return 10.0
    if p == 0.5:
        return 0.0

    if p < 0.5:
        sign = -1.0
        p_inner = p
    else:
        sign = 1.0
        p_inner = 1.0 - p

    t = math.sqrt(-2.0 * math.log(p_inner))
    # Coefficients
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
    return sign * z


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RegimeConditionalVaR:
    """
    Computes Value at Risk and Conditional VaR using a method tailored to the
    current market regime.

    Parameters
    ----------
    mc_simulations : int
        Number of Monte Carlo simulations for the *volatile* regime.
    default_confidence : float
        Default confidence level (e.g. 0.99 for 99% VaR).
    regime_method_map : dict, optional
        Override the regime → method mapping.
    """

    # Default regime → method
    _DEFAULT_MAP: Dict[str, str] = {
        "normal": "parametric",
        "crisis": "historical",
        "volatile": "monte_carlo",
        "trending": "parametric",
    }

    def __init__(self, mc_simulations: int = 1000,
                 default_confidence: float = 0.99,
                 regime_method_map: Optional[Dict[str, str]] = None) -> None:
        self._mc_sims = mc_simulations
        self._default_confidence = default_confidence
        self._regime_map = dict(regime_method_map or self._DEFAULT_MAP)

        logger.info("RegimeConditionalVaR initialised (mc_sims=%d, confidence=%.2f)",
                     mc_simulations, default_confidence)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_var(self, returns: list, regime: str,
                    confidence: Optional[float] = None) -> VaRResult:
        """
        Compute VaR and CVaR for a return series under the given regime.

        Parameters
        ----------
        returns : list of float
            Period returns (e.g. daily log returns or simple returns).
        regime : str
            Current market regime label.
        confidence : float, optional
            Confidence level (default: ``default_confidence``).

        Returns
        -------
        VaRResult
        """
        conf = confidence or self._default_confidence
        arr = np.asarray(returns, dtype=np.float64)
        arr = arr[np.isfinite(arr)]

        if len(arr) < 2:
            logger.warning("compute_var: insufficient data (%d returns)", len(arr))
            return VaRResult(var_pct=0.0, cvar_pct=0.0, method_used="none",
                             regime=regime, confidence=conf, n_samples=len(arr))

        method = self._regime_map.get(regime, "parametric")

        if method == "historical":
            return self._historical_var(arr, regime, conf)
        elif method == "monte_carlo":
            return self._monte_carlo_var(arr, regime, conf)
        else:
            return self._parametric_var(arr, regime, conf)

    def get_portfolio_var(self, position_returns: Dict[str, list],
                          regime: str, confidence: Optional[float] = None) -> float:
        """
        Compute portfolio-level VaR from individual position return series.

        Uses equal-weighted aggregation: portfolio return per period is the
        mean of constituent returns.  Then regime-conditional VaR is computed
        on the aggregated series.

        Parameters
        ----------
        position_returns : dict
            symbol → list of returns.
        regime : str
        confidence : float, optional

        Returns
        -------
        float
            Portfolio VaR as a positive percentage.
        """
        if not position_returns:
            return 0.0

        # Align lengths
        min_len = min(len(v) for v in position_returns.values())
        if min_len < 2:
            return 0.0

        arrays = [np.asarray(v[:min_len], dtype=np.float64) for v in position_returns.values()]
        portfolio_returns = np.mean(arrays, axis=0)

        result = self.compute_var(portfolio_returns.tolist(), regime, confidence)
        logger.info("get_portfolio_var: %d positions, regime=%s, VaR=%.4f%%",
                    len(position_returns), regime, result.var_pct)
        return result.var_pct

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def _parametric_var(self, arr: np.ndarray, regime: str,
                        conf: float) -> VaRResult:
        """Gaussian parametric VaR."""
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1))
        if sigma < 1e-12:
            return VaRResult(var_pct=0.0, cvar_pct=0.0, method_used="parametric",
                             regime=regime, confidence=conf, n_samples=len(arr))

        z = _norm_ppf(1.0 - conf)  # negative quantile
        var_pct = -(mu + z * sigma) * 100.0   # positive = loss

        # CVaR (Expected Shortfall) for Gaussian
        # ES = mu + sigma * phi(z) / (1 - conf)
        phi_z = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        cvar_pct = -(mu - sigma * phi_z / (1.0 - conf)) * 100.0

        return VaRResult(
            var_pct=round(max(var_pct, 0.0), 6),
            cvar_pct=round(max(cvar_pct, 0.0), 6),
            method_used="parametric",
            regime=regime, confidence=conf, n_samples=len(arr),
        )

    def _historical_var(self, arr: np.ndarray, regime: str,
                        conf: float) -> VaRResult:
        """Historical simulation VaR (non-parametric)."""
        sorted_returns = np.sort(arr)
        idx = int(math.floor((1.0 - conf) * len(sorted_returns)))
        idx = max(0, min(idx, len(sorted_returns) - 1))

        var_val = -sorted_returns[idx]
        # CVaR = mean of losses beyond VaR
        tail = sorted_returns[:idx + 1]
        cvar_val = -float(np.mean(tail)) if len(tail) > 0 else var_val

        return VaRResult(
            var_pct=round(max(var_val * 100.0, 0.0), 6),
            cvar_pct=round(max(cvar_val * 100.0, 0.0), 6),
            method_used="historical",
            regime=regime, confidence=conf, n_samples=len(arr),
        )

    def _monte_carlo_var(self, arr: np.ndarray, regime: str,
                         conf: float) -> VaRResult:
        """Monte Carlo VaR with bootstrapped returns."""
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1))
        if sigma < 1e-12:
            return VaRResult(var_pct=0.0, cvar_pct=0.0, method_used="monte_carlo",
                             regime=regime, confidence=conf, n_samples=len(arr))

        rng = np.random.default_rng(42)
        simulated = rng.normal(mu, sigma, size=self._mc_sims)
        sorted_sims = np.sort(simulated)

        idx = int(math.floor((1.0 - conf) * self._mc_sims))
        idx = max(0, min(idx, self._mc_sims - 1))

        var_val = -sorted_sims[idx]
        tail = sorted_sims[:idx + 1]
        cvar_val = -float(np.mean(tail)) if len(tail) > 0 else var_val

        return VaRResult(
            var_pct=round(max(var_val * 100.0, 0.0), 6),
            cvar_pct=round(max(cvar_val * 100.0, 0.0), 6),
            method_used="monte_carlo",
            regime=regime, confidence=conf, n_samples=self._mc_sims,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_method_for_regime(self, regime: str) -> str:
        """Return the VaR method that would be used for *regime*."""
        return self._regime_map.get(regime, "parametric")

    def set_regime_method(self, regime: str, method: str) -> None:
        """Override the VaR method for a regime."""
        valid = {"parametric", "historical", "monte_carlo"}
        if method not in valid:
            raise ValueError(f"method must be one of {valid}, got '{method}'")
        self._regime_map[regime] = method
        logger.info("set_regime_method: %s → %s", regime, method)
