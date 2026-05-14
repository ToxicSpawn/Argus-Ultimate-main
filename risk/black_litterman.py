"""Black-Litterman portfolio weight estimator.

Drops into the execution engine's portfolio_weight_method='bl' path.
Rooted in the original He-Litterman (1999) formulation:

    mu_BL  = [(tau * Sigma)^-1 + P' * Omega^-1 * P]^-1
              * [(tau * Sigma)^-1 * Pi + P' * Omega^-1 * Q]

    w_BL   = (lambda * Sigma)^-1 * mu_BL   (normalised to sum = 1)

Numerical stability
-------------------
All matrix inversions use scipy.linalg.cho_solve(cho_factor(A), b) which is
~5x faster than np.linalg.inv and avoids ill-conditioned posterior precision
matrices on highly-correlated crypto pairs.

When no views are supplied the posterior collapses to the market-cap
equilibrium (Pi), giving HRP-like equal risk contribution as fallback.

Usage
-----
    from risk.black_litterman import BlackLittermanOptimizer

    bl = BlackLittermanOptimizer(risk_aversion=2.5, tau=0.05)
    weights = bl.weights(symbols, returns_df)          # dict {symbol: float}
    weights = bl.weights(symbols, returns_df, views)   # with analyst views

    # views format:
    # views = [
    #   {"assets": ["BTC/USD"], "coeffs": [1.0], "return": 0.05, "confidence": 0.8},
    #   {"assets": ["ETH/USD", "SOL/USD"], "coeffs": [1.0, -1.0], "return": 0.02, "confidence": 0.5},
    # ]
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    from scipy.linalg import cho_factor, cho_solve  # type: ignore[import]
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

logger = logging.getLogger(__name__)

_EPS = 1e-10


def _solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve A @ X = B.  Uses Cholesky when scipy is available (stable + fast).

    Falls back to np.linalg.solve for non-PD matrices or when scipy is absent.
    """
    if _SCIPY:
        try:
            c, low = cho_factor(A)
            if B.ndim == 1:
                return cho_solve((c, low), B)
            # B is a matrix — solve column-by-column
            return np.column_stack([cho_solve((c, low), B[:, j]) for j in range(B.shape[1])])
        except Exception:
            pass  # fall through to np.linalg.solve
    return np.linalg.solve(A, B)


def _inv(A: np.ndarray) -> np.ndarray:
    """Stable matrix inverse via _solve(A, I)."""
    return _solve(A, np.eye(A.shape[0]))


class BlackLittermanOptimizer:
    """He-Litterman Black-Litterman portfolio weight estimator."""

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        min_weight: float = 0.01,
        max_weight: float = 0.50,
        min_history: int = 20,
        prior_mode: str = "equal",
    ) -> None:
        """
        Parameters
        ----------
        risk_aversion : float
            Lambda — investor risk-aversion coefficient.
            Higher = more conservative allocation.
        tau : float
            Uncertainty in the prior (typically 0.02-0.10).
            Smaller = prior dominates; larger = views dominate.
        min_weight : float
            Hard floor for any single asset weight.
        max_weight : float
            Hard cap for any single asset weight.
        min_history : int
            Minimum number of return observations needed before BL is used;
            returns equal-weight below this threshold.
        prior_mode : str
            'equal'  — equal-weight equilibrium prior.
            'sharpe' — Sharpe-proportional equilibrium prior.
        """
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_history = min_history
        self.prior_mode = prior_mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def weights(
        self,
        symbols: Sequence[str],
        returns,
        views: Optional[List[dict]] = None,
    ) -> Dict[str, float]:
        """Compute BL posterior weights.

        Parameters
        ----------
        symbols : list[str]
            Asset names in the order they appear as columns in *returns*.
        returns : pd.DataFrame | np.ndarray
            T x N matrix of simple (or log) period returns.
        views : list[dict] | None
            Optional analyst views.  See module docstring for schema.

        Returns
        -------
        dict {symbol: weight}  — weights sum to 1.0, clipped to [min, max].
        """
        n = len(symbols)
        if n == 0:
            return {}

        arr = self._to_array(returns, n)
        if arr is None or arr.shape[0] < self.min_history:
            logger.debug(
                "BL: insufficient history (%s rows), falling back to equal-weight",
                arr.shape[0] if arr is not None else 0,
            )
            return self._equal_weight(symbols)

        try:
            sigma = self._cov(arr)             # N x N covariance
            pi = self._equilibrium_prior(arr, sigma, n)
            mu_bl = self._posterior_mu(sigma, pi, views, symbols)
            w_raw = self._mv_weights(mu_bl, sigma)
            return self._clip_and_normalise(symbols, w_raw)
        except Exception as exc:
            logger.warning("BL optimisation failed (%s), equal-weight fallback", exc)
            return self._equal_weight(symbols)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_array(returns, n: int):
        """Convert pd.DataFrame or np.ndarray -> float64 ndarray (T x N)."""
        if returns is None:
            return None
        if _PANDAS and isinstance(returns, pd.DataFrame):
            arr = returns.iloc[:, :n].values.astype(np.float64)
        else:
            arr = np.asarray(returns, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            arr = arr[:, :n]
        mask = ~np.isnan(arr).any(axis=1)
        return arr[mask]

    @staticmethod
    def _cov(arr: np.ndarray) -> np.ndarray:
        """Annualised sample covariance (assuming daily returns, 365 crypto days)."""
        sigma = np.cov(arr, rowvar=False) * 365.0
        min_eig = np.linalg.eigvalsh(sigma).min()
        if min_eig < _EPS:
            sigma += (_EPS - min_eig + 1e-8) * np.eye(sigma.shape[0])
        return sigma

    def _equilibrium_prior(self, arr: np.ndarray, sigma: np.ndarray, n: int) -> np.ndarray:
        """Implied equilibrium excess returns Pi = lambda * Sigma * w_eq."""
        if self.prior_mode == "sharpe":
            mean = arr.mean(axis=0) * 365.0
            std = np.sqrt(np.diag(sigma)) + _EPS
            sharpe = mean / std
            sharpe = np.maximum(sharpe, 0.0)
            w_eq = sharpe / (sharpe.sum() + _EPS)
        else:
            w_eq = np.full(n, 1.0 / n)
        return self.risk_aversion * sigma @ w_eq

    def _posterior_mu(self, sigma, pi, views, symbols) -> np.ndarray:
        """Black-Litterman posterior expected return vector."""
        n = len(pi)
        tau_sigma = self.tau * sigma
        tau_sigma_inv = _inv(tau_sigma)

        if not views:
            return pi.copy()

        sym_index = {s: i for i, s in enumerate(symbols)}
        k = len(views)
        P = np.zeros((k, n))
        Q = np.zeros(k)
        omega_diag = np.zeros(k)

        for j, v in enumerate(views):
            assets = v.get("assets", [])
            coeffs = v.get("coeffs", [1.0] * len(assets))
            ret = float(v.get("return", 0.0))
            conf = float(v.get("confidence", 0.5))
            conf = np.clip(conf, 1e-3, 1.0 - 1e-3)

            for asset, coeff in zip(assets, coeffs):
                idx = sym_index.get(asset)
                if idx is not None:
                    P[j, idx] = float(coeff)

            Q[j] = ret
            omega_diag[j] = (1.0 - conf) / conf

        omega_inv = np.diag(1.0 / (omega_diag + _EPS))

        # Posterior precision matrix A = tau_sigma_inv + P' * Omega_inv * P
        post_prec = tau_sigma_inv + P.T @ omega_inv @ P

        # Solve for posterior mean: post_prec @ mu_bl = rhs
        rhs = tau_sigma_inv @ pi + P.T @ omega_inv @ Q
        mu_bl = _solve(post_prec, rhs)
        return mu_bl

    def _mv_weights(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Mean-variance optimal weights: w = (lambda * Sigma)^-1 * mu."""
        w = _solve(self.risk_aversion * sigma, mu)
        return np.maximum(w, 0.0)

    def _clip_and_normalise(self, symbols: Sequence[str], w_raw: np.ndarray) -> Dict[str, float]:
        """Clip weights to [min, max] and normalise to sum = 1."""
        total = w_raw.sum()
        if total < _EPS:
            return self._equal_weight(symbols)
        w = w_raw / total
        for _ in range(20):
            w = np.clip(w, self.min_weight, self.max_weight)
            s = w.sum()
            if abs(s - 1.0) < 1e-9:
                break
            w = w / s
        return {sym: float(w[i]) for i, sym in enumerate(symbols)}

    def _equal_weight(self, symbols: Sequence[str]) -> Dict[str, float]:
        n = len(symbols)
        if n == 0:
            return {}
        return {s: 1.0 / n for s in symbols}


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_DEFAULT_OPTIMIZER: Optional[BlackLittermanOptimizer] = None


def get_bl_optimizer(cfg: Optional[dict] = None) -> BlackLittermanOptimizer:
    """Return (and cache) a module-level BL optimizer instance."""
    global _DEFAULT_OPTIMIZER
    if _DEFAULT_OPTIMIZER is None:
        bl_cfg = (cfg or {}).get("black_litterman", {})
        _DEFAULT_OPTIMIZER = BlackLittermanOptimizer(
            risk_aversion=bl_cfg.get("risk_aversion", 2.5),
            tau=bl_cfg.get("tau", 0.05),
            min_weight=bl_cfg.get("min_weight", 0.01),
            max_weight=bl_cfg.get("max_weight", 0.50),
            min_history=bl_cfg.get("min_history", 20),
            prior_mode=bl_cfg.get("prior_mode", "equal"),
        )
    return _DEFAULT_OPTIMIZER


def bl_weights(
    symbols: Sequence[str],
    returns,
    views: Optional[List[dict]] = None,
    cfg: Optional[dict] = None,
) -> Dict[str, float]:
    """Top-level function: compute BL weights, ready for execution engine."""
    return get_bl_optimizer(cfg).weights(symbols, returns, views)
