#!/usr/bin/env python3
"""
Risk Parity Allocation — equal risk contribution portfolio construction.

Each position is sized so it contributes equally to total portfolio variance.
Uses numpy for covariance estimation with a pure-Python fallback for small
portfolios (< 10 assets).

Standalone usage:
    allocator = RiskParityAllocator()
    weights = allocator.compute_weights(returns_dict, target_vol=0.15)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np  # type: ignore[import-untyped]

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False
    np = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pure_python_cov(returns: Dict[str, List[float]], symbols: List[str]) -> List[List[float]]:
    """Compute covariance matrix without numpy (for small universes)."""
    n = len(symbols)
    T = min(len(returns[s]) for s in symbols)
    if T < 2:
        # Return identity-ish
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    means = {s: sum(returns[s][-T:]) / T for s in symbols}
    cov = [[0.0] * n for _ in range(n)]
    for i in range(n):
        ri = returns[symbols[i]][-T:]
        mi = means[symbols[i]]
        for j in range(i, n):
            rj = returns[symbols[j]][-T:]
            mj = means[symbols[j]]
            val = sum((ri[t] - mi) * (rj[t] - mj) for t in range(T)) / (T - 1)
            cov[i][j] = val
            cov[j][i] = val
    return cov


def _pure_python_risk_parity(cov: List[List[float]], n: int, max_iter: int = 500, tol: float = 1e-8) -> List[float]:
    """
    Iterative risk-parity solver (cyclical coordinate descent) without numpy.

    Targets equal risk contribution: w_i * (Sigma @ w)_i = const for all i.
    """
    w = [1.0 / n] * n

    for _ in range(max_iter):
        # Compute sigma_w = Cov @ w
        sigma_w = [sum(cov[i][j] * w[j] for j in range(n)) for i in range(n)]
        total_risk = sum(w[i] * sigma_w[i] for i in range(n))
        if total_risk < 1e-15:
            return [1.0 / n] * n

        # Risk contribution per asset
        rc = [w[i] * sigma_w[i] / total_risk for i in range(n)]
        target_rc = 1.0 / n

        # Update weights proportionally to inverse marginal risk
        w_new = [0.0] * n
        for i in range(n):
            if sigma_w[i] > 1e-15:
                w_new[i] = target_rc / sigma_w[i]
            else:
                w_new[i] = 1.0 / n

        # Normalize
        s = sum(w_new)
        if s < 1e-15:
            return [1.0 / n] * n
        w_new = [x / s for x in w_new]

        # Convergence check
        delta = max(abs(w_new[i] - w[i]) for i in range(n))
        w = w_new
        if delta < tol:
            break

    return w


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RiskParityAllocator:
    """
    Constructs portfolios where each asset contributes equally to total risk.

    Parameters
    ----------
    min_history : int
        Minimum number of return observations required per asset (default 20).
    max_weight : float
        Maximum weight any single asset can receive (default 0.40 = 40%).
    """

    def __init__(self, min_history: int = 20, max_weight: float = 0.40,
                 use_quantum_eigenmode: bool = False):
        self.min_history = min_history
        self.max_weight = max_weight
        self.use_quantum_eigenmode = bool(use_quantum_eigenmode)
        self._last_weights: Dict[str, float] = {}

    def quantum_dominant_risk_mode(
        self,
        returns: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """
        Phase S/G2: extract the dominant risk eigenmode of the returns
        covariance matrix using VQE (or classical eigh fallback).

        Returns the top eigenvector and eigenvalue, used by the trading
        loop to identify the dominant systematic factor.
        """
        if not _HAS_NUMPY:
            return {"eigenvalue": 0.0, "eigenvector": [], "method": "no_numpy"}

        symbols = sorted(returns.keys())
        n = len(symbols)
        if n < 2:
            return {"eigenvalue": 0.0, "eigenvector": [], "method": "insufficient_assets"}

        T = min(len(returns[s]) for s in symbols)
        if T < self.min_history:
            return {"eigenvalue": 0.0, "eigenvector": [], "method": "insufficient_history"}

        mat = np.array([returns[s][-T:] for s in symbols])
        cov = np.cov(mat)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])

        if self.use_quantum_eigenmode and n <= 6:
            try:
                from quantum.finance.qpca import quantum_pca
                # Run QPCA on the returns directly
                result = quantum_pca(mat.T, n_components=1, use_vqe=False)
                top_eig = float(result["eigenvalues"][0])
                top_vec = list(result["eigenvectors"][0]) if result["eigenvectors"] else []
                return {
                    "eigenvalue": top_eig,
                    "eigenvector": top_vec,
                    "method": "quantum_pca",
                }
            except Exception as exc:
                logger.debug("quantum_pca failed, falling back: %s", exc)

        # Classical fallback
        eigvals, eigvecs = np.linalg.eigh(cov)
        return {
            "eigenvalue": float(eigvals[-1]),
            "eigenvector": eigvecs[:, -1].tolist(),
            "method": "classical_eigh",
        }
        self._last_compute_time_ms: float = 0.0
        logger.info(
            "RiskParityAllocator initialised (min_history=%d, max_weight=%.2f, numpy=%s)",
            min_history, max_weight, _HAS_NUMPY,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_weights(
        self,
        returns: Dict[str, List[float]],
        target_vol: float = 0.15,
    ) -> Dict[str, float]:
        """
        Compute risk-parity weights.

        Parameters
        ----------
        returns : dict
            Symbol -> list of periodic returns (e.g. daily % changes as decimals).
        target_vol : float
            Target annualised portfolio volatility (used for optional leverage scaling).

        Returns
        -------
        dict
            Symbol -> weight (sums to 1.0).
        """
        t0 = time.monotonic()
        symbols = sorted(returns.keys())
        n = len(symbols)

        if n == 0:
            logger.warning("compute_weights called with empty returns dict")
            return {}

        if n == 1:
            sym = symbols[0]
            self._last_weights = {sym: 1.0}
            self._last_compute_time_ms = (time.monotonic() - t0) * 1000
            return {sym: 1.0}

        # Filter assets with insufficient history
        valid = [s for s in symbols if len(returns[s]) >= self.min_history]
        if not valid:
            logger.warning(
                "No assets have >= %d return observations; falling back to equal weight",
                self.min_history,
            )
            w = {s: 1.0 / n for s in symbols}
            self._last_weights = w
            self._last_compute_time_ms = (time.monotonic() - t0) * 1000
            return w

        if len(valid) < n:
            logger.info(
                "Filtered %d/%d assets with insufficient history; using %d",
                n - len(valid), n, len(valid),
            )
            symbols = valid
            n = len(symbols)

        # Compute covariance matrix and solve
        if _HAS_NUMPY:
            weights = self._numpy_solve(returns, symbols, target_vol)
        else:
            weights = self._pure_python_solve(returns, symbols)

        # Apply max weight cap and re-normalise
        weights = self._cap_weights(weights, symbols)

        result = dict(zip(symbols, weights))
        self._last_weights = result
        self._last_compute_time_ms = (time.monotonic() - t0) * 1000
        logger.debug(
            "Risk-parity weights computed in %.1f ms for %d assets",
            self._last_compute_time_ms, n,
        )
        return result

    def get_risk_contributions(
        self,
        weights: Dict[str, float],
        returns: Dict[str, List[float]],
    ) -> Dict[str, float]:
        """
        Return each asset's percentage contribution to total portfolio variance.

        Parameters
        ----------
        weights : dict
            Symbol -> weight.
        returns : dict
            Symbol -> list of returns.

        Returns
        -------
        dict
            Symbol -> risk contribution as a fraction of 1.0.
        """
        symbols = sorted(weights.keys())
        n = len(symbols)
        if n == 0:
            return {}

        w_vec = [weights.get(s, 0.0) for s in symbols]

        if _HAS_NUMPY:
            T = min(len(returns[s]) for s in symbols)
            mat = np.array([returns[s][-T:] for s in symbols])
            cov = np.cov(mat)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])
            w_arr = np.array(w_vec)
            sigma_w = cov @ w_arr
            total = float(w_arr @ sigma_w)
            if abs(total) < 1e-15:
                return {s: 1.0 / n for s in symbols}
            rc = (w_arr * sigma_w) / total
            return {s: float(rc[i]) for i, s in enumerate(symbols)}
        else:
            cov = _pure_python_cov(returns, symbols)
            sigma_w = [sum(cov[i][j] * w_vec[j] for j in range(n)) for i in range(n)]
            total = sum(w_vec[i] * sigma_w[i] for i in range(n))
            if abs(total) < 1e-15:
                return {s: 1.0 / n for s in symbols}
            return {s: w_vec[i] * sigma_w[i] / total for i, s in enumerate(symbols)}

    def rebalance_needed(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        threshold_pct: float = 5.0,
    ) -> bool:
        """
        Check whether any weight deviates from target by more than *threshold_pct* (absolute).

        Parameters
        ----------
        current_weights : dict
            Current portfolio weights.
        target_weights : dict
            Target risk-parity weights.
        threshold_pct : float
            Maximum allowed deviation in percentage points (default 5.0).

        Returns
        -------
        bool
            True if rebalance is recommended.
        """
        all_syms = set(current_weights) | set(target_weights)
        for s in all_syms:
            cur = current_weights.get(s, 0.0)
            tgt = target_weights.get(s, 0.0)
            if abs(cur - tgt) * 100 > threshold_pct:
                logger.info(
                    "Rebalance triggered: %s current=%.2f%% target=%.2f%% (threshold=%.1f%%)",
                    s, cur * 100, tgt * 100, threshold_pct,
                )
                return True
        return False

    def get_marginal_risk(
        self,
        weights: Dict[str, float],
        returns: Dict[str, List[float]],
        symbol: str,
    ) -> float:
        """
        Marginal risk contribution of *symbol*: d(portfolio_vol) / d(w_symbol).

        Parameters
        ----------
        weights : dict
            Current weights.
        returns : dict
            Historical returns.
        symbol : str
            The asset to evaluate.

        Returns
        -------
        float
            Marginal risk contribution (annualised if daily returns provided).
        """
        symbols = sorted(weights.keys())
        if symbol not in symbols:
            logger.warning("get_marginal_risk: %s not in portfolio", symbol)
            return 0.0

        n = len(symbols)
        idx = symbols.index(symbol)
        w_vec = [weights.get(s, 0.0) for s in symbols]

        if _HAS_NUMPY:
            T = min(len(returns[s]) for s in symbols)
            mat = np.array([returns[s][-T:] for s in symbols])
            cov = np.cov(mat)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])
            w_arr = np.array(w_vec)
            port_var = float(w_arr @ cov @ w_arr)
            if port_var < 1e-15:
                return 0.0
            port_vol = port_var ** 0.5
            sigma_w = cov @ w_arr
            return float(sigma_w[idx]) / port_vol
        else:
            cov = _pure_python_cov(returns, symbols)
            sigma_w = [sum(cov[i][j] * w_vec[j] for j in range(n)) for i in range(n)]
            port_var = sum(w_vec[i] * sigma_w[i] for i in range(n))
            if port_var < 1e-15:
                return 0.0
            return sigma_w[idx] / (port_var ** 0.5)

    # ------------------------------------------------------------------
    # Internal solvers
    # ------------------------------------------------------------------

    def _numpy_solve(
        self,
        returns: Dict[str, List[float]],
        symbols: List[str],
        target_vol: float,
    ) -> List[float]:
        """Risk-parity via iterative reweighting with numpy."""
        n = len(symbols)
        T = min(len(returns[s]) for s in symbols)
        mat = np.array([returns[s][-T:] for s in symbols])
        cov = np.cov(mat)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])

        # Regularise if near-singular
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals.min() < 1e-10:
            cov += np.eye(n) * 1e-8
            logger.debug("Regularised covariance matrix (min eigenvalue was %.2e)", eigvals.min())

        w = np.ones(n) / n
        for _ in range(500):
            sigma_w = cov @ w
            total_risk = float(w @ sigma_w)
            if total_risk < 1e-15:
                return [1.0 / n] * n
            rc = w * sigma_w / total_risk
            target_rc = 1.0 / n
            w_new = np.where(sigma_w > 1e-15, target_rc / sigma_w, 1.0 / n)
            w_new /= w_new.sum()
            if np.max(np.abs(w_new - w)) < 1e-8:
                w = w_new
                break
            w = w_new

        return w.tolist()

    def _pure_python_solve(
        self,
        returns: Dict[str, List[float]],
        symbols: List[str],
    ) -> List[float]:
        """Risk-parity without numpy."""
        n = len(symbols)
        cov = _pure_python_cov(returns, symbols)
        return _pure_python_risk_parity(cov, n)

    def _cap_weights(self, weights: List[float], symbols: List[str]) -> List[float]:
        """Enforce max_weight cap and re-normalise (iterative to handle redistribution)."""
        n = len(weights)
        w = list(weights)
        for _ in range(20):  # iterate until stable
            s = sum(w)
            if s < 1e-15:
                return [1.0 / n] * n
            w = [x / s for x in w]
            violated = any(x > self.max_weight + 1e-9 for x in w)
            if not violated:
                break
            w = [min(x, self.max_weight) for x in w]
        # Final normalisation to ensure sum == 1.0
        s = sum(w)
        if s > 1e-15 and abs(s - 1.0) > 1e-9:
            w = [x / s for x in w]
        return w
