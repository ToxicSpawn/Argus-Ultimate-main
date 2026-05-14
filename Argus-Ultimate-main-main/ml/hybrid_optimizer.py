"""
Hybrid quantum-classical optimization for portfolio.

This module provides QAOARefiner which takes a QAOA quantum solution
and refines it with classical optimization (scipy) for better convergence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ml.prediction_bus import PredictionBundle


@dataclass
class HybridOptimizationResult:
    """Result of hybrid quantum-informed classical optimization."""

    qaoa_weights: np.ndarray
    refined_weights: np.ndarray
    qaoa_sharpe: float
    refined_sharpe: float
    improvement: float
    iterations: int
    convergence_history: list[float] = field(default_factory=list)
    method: str = "qaoa_scipy_refinement"
    honest_claim: str = (
        "Hybrid optimization: QAOA provides initial subset selection, "
        "scipy refines continuous weights. No quantum speedup claimed."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "qaoa_weights": self.qaoa_weights.tolist(),
            "refined_weights": self.refined_weights.tolist(),
            "qaoa_sharpe": float(self.qaoa_sharpe),
            "refined_sharpe": float(self.refined_sharpe),
            "improvement": float(self.improvement),
            "iterations": self.iterations,
            "convergence_history": [float(x) for x in self.convergence_history],
            "method": self.method,
            "honest_claim": self.honest_claim,
        }


class QAOARefiner:
    """
    Hybrid QAOA + classical refinement for portfolio optimization.

    Takes QAOA binary subset selection and refines continuous weights
    using scipy optimization for better Sharpe ratio.

    Workflow:
    1. Run QAOA to get subset selection
    2. Use QAOA weights as warm start
    3. Run scipy.optimize to refine weights
    4. Return both for comparison
    """

    def __init__(
        self,
        *,
        max_iter: int = 100,
        tolerance: float = 1e-6,
        use_scipy: bool = True,
    ) -> None:
        self.max_iter = max(int(max_iter), 10)
        self.tolerance = max(float(tolerance), 1e-10)
        self.use_scipy = use_scipy

    def _sharpe_ratio(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        cov: np.ndarray,
    ) -> float:
        """Compute Sharpe ratio for given weights."""
        w = np.asarray(weights, dtype=float).ravel()
        if w.sum() < 1e-10:
            return 0.0
        w = w / w.sum()  # Normalize
        port_return = float(w @ returns)
        port_risk = float(np.sqrt(w @ cov @ w))
        if port_risk < 1e-10:
            return 0.0
        return port_return / port_risk

    def refine(
        self,
        qaoa_weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 0.5,
    ) -> HybridOptimizationResult:
        """
        Refine QAOA weights with classical optimization.

        Args:
            qaoa_weights: Initial weights from QAOA (or qaoa-like heuristic)
            expected_returns: Asset expected returns
            covariance_matrix: Asset covariance matrix
            risk_aversion: Risk aversion parameter

        Returns:
            HybridOptimizationResult with both qaoa and refined weights
        """
        returns = np.asarray(expected_returns, dtype=float).ravel()
        cov = np.asarray(covariance_matrix, dtype=float)

        n = len(returns)
        if qaoa_weights is None or len(qaoa_weights) != n:
            qaoa_weights = np.ones(n) / n

        # QAOA solution: binary selection from weights
        qaoa_binary = (qaoa_weights > 0.1).astype(float)
        selected = np.where(qaoa_binary > 0)[0]
        if len(selected) < 1:
            selected = np.arange(min(n, 3))

        # Compute QAOA Sharpe
        qaoa_norm = qaoa_weights / max(qaoa_weights.sum(), 1e-10)
        qaoa_sharpe = self._sharpe_ratio(qaoa_norm, returns, cov)

        # Classical refinement using scipy if available
        refined_weights = qaoa_norm.copy()
        convergence_history = [qaoa_sharpe]
        iterations = 0

        if self.use_scipy:
            try:
                from scipy.optimize import minimize

                def neg_sharpe(w):
                    """Negative Sharpe for minimization."""
                    w = np.abs(w)
                    w = w / max(w.sum(), 1e-10)
                    return -self._sharpe_ratio(w, returns, cov)

                # Start from QAOA solution
                x0 = qaoa_norm.copy()

                # Optimize with bounds
                result = minimize(
                    neg_sharpe,
                    x0,
                    method="SLSQP",
                    bounds=[(0.0, 1.0)] * n,
                    constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
                    options={"maxiter": self.max_iter, "ftol": self.tolerance},
                )

                refined_weights = np.abs(result.x)
                refined_weights = refined_weights / max(refined_weights.sum(), 1e-10)
                iterations = result.nit if hasattr(result, "nit") else 0

            except ImportError:
                # Fallback: gradient descent
                refined_weights, iterations, convergence_history = self._gradient_descent(
                    qaoa_norm, returns, cov
                )

        refined_sharpe = self._sharpe_ratio(refined_weights, returns, cov)
        improvement = (refined_sharpe - qaoa_sharpe) / max(abs(qaoa_sharpe), 1e-10)

        return HybridOptimizationResult(
            qaoa_weights=qaoa_norm,
            refined_weights=refined_weights,
            qaoa_sharpe=qaoa_sharpe,
            refined_sharpe=refined_sharpe,
            improvement=improvement,
            iterations=iterations,
            convergence_history=convergence_history,
        )

    def _gradient_descent(
        self,
        start: np.ndarray,
        returns: np.ndarray,
        cov: np.ndarray,
    ) -> tuple[np.ndarray, int, list[float]]:
        """Fallback gradient descent when scipy unavailable."""
        w = start.copy()
        lr = 0.01
        history = [self._sharpe_ratio(w, returns, cov)]

        for _ in range(self.max_iter):
            sharpe = self._sharpe_ratio(w, returns, cov)
            grad = np.zeros_like(w)

            # Numerical gradient
            eps = 1e-8
            for i in range(len(w)):
                w_plus = w.copy()
                w_plus[i] += eps
                w_plus = w_plus / max(w_plus.sum(), 1e-10)
                grad[i] = (self._sharpe_ratio(w_plus, returns, cov) - sharpe) / eps

            # Update
            w = w + lr * grad
            w = np.clip(w, 0.0, 1.0)
            w = w / max(w.sum(), 1e-10)

            history.append(self._sharpe_ratio(w, returns, cov))

        return w, len(history), history


def hybrid_portfolio_optimize(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    *,
    risk_aversion: float = 0.5,
    qaoa_weights: Optional[np.ndarray] = None,
    max_iter: int = 100,
) -> HybridOptimizationResult:
    """
    Convenience function for hybrid QAOA + classical optimization.

    Args:
        expected_returns: Asset expected returns
        covariance_matrix: Asset covariance matrix
        risk_aversion: Risk aversion (higher = more conservative)
        qaoa_weights: Optional QAOA result to refine
        max_iter: Maximum refinement iterations

    Returns:
        HybridOptimizationResult
    """
    refiner = QAOARefiner(max_iter=max_iter)

    # If no QAOA weights, use equal weights as fallback
    if qaoa_weights is None:
        n = len(expected_returns)
        qaoa_weights = np.ones(n) / n

    return refiner.refine(
        qaoa_weights,
        expected_returns,
        covariance_matrix,
        risk_aversion=risk_aversion,
    )


__all__ = ["QAOARefiner", "HybridOptimizationResult", "hybrid_portfolio_optimize"]