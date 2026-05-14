from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any

np = importlib.import_module("numpy")

from .qubo_builder import PortfolioQUBOProblem

logger = logging.getLogger(__name__)

try:
    cp = importlib.import_module("cvxpy")
    _has_cvxpy = True
except Exception:
    cp = None
    _has_cvxpy = False


@dataclass(slots=True)
class ClassicalFallbackConfig:
    solver_name: str | None = None
    max_iterations: int = 250
    min_weight: float = 0.0
    max_weight: float = 1.0

    def __post_init__(self) -> None:
        self.max_iterations = max(50, int(self.max_iterations))
        self.min_weight = float(self.min_weight)
        self.max_weight = float(self.max_weight)
        if self.min_weight > self.max_weight:
            raise ValueError("min_weight must be less than or equal to max_weight")


@dataclass(slots=True)
class ClassicalOptimizationResult:
    weights: dict[str, float]
    objective_value: float
    expected_return: float
    portfolio_risk: float
    method: str
    solver_used: str
    success: bool


class ClassicalFallbackOptimizer:
    """Continuous CVXPY optimizer with a deterministic numpy fallback."""

    def __init__(self, config: ClassicalFallbackConfig | None = None) -> None:
        self.config = config or ClassicalFallbackConfig()

    def optimize(self, problem: PortfolioQUBOProblem) -> ClassicalOptimizationResult:
        if _has_cvxpy:
            try:
                return self._optimize_with_cvxpy(problem)
            except Exception as exc:
                logger.warning("CVXPY optimization failed, using numpy fallback: %s", exc)
        return self._optimize_with_numpy(problem)

    def _optimize_with_cvxpy(self, problem: PortfolioQUBOProblem) -> ClassicalOptimizationResult:
        if cp is None:
            raise RuntimeError("cvxpy is not available")
        n_assets = len(problem.symbols)
        weights = cp.Variable(n_assets)
        covariance = np.asarray(problem.covariance_matrix, dtype=float)
        expected_returns = np.asarray(problem.expected_returns, dtype=float)

        objective = cp.Minimize(
            problem.config.risk_aversion * cp.quad_form(weights, covariance) - expected_returns @ weights
        )
        constraints = [cp.sum(weights) == problem.config.budget]

        if problem.config.long_only:
            constraints.append(weights >= self.config.min_weight)
        if self.config.max_weight < 1.0 or problem.config.budget < float(n_assets):
            constraints.append(weights <= self.config.max_weight)

        optimization_problem = cp.Problem(objective, constraints)
        chosen_solver = self.config.solver_name or "SCS"
        optimization_problem.solve(solver=getattr(cp, chosen_solver, cp.SCS), max_iters=self.config.max_iterations)

        if weights.value is None:
            raise ValueError("CVXPY solver returned no solution")

        vector = np.asarray(weights.value, dtype=float).reshape(-1)
        vector = self._normalize(vector, float(problem.config.budget))
        return self._build_result(problem, vector, method="cvxpy", solver_used=chosen_solver, success=True)

    def _optimize_with_numpy(self, problem: PortfolioQUBOProblem) -> ClassicalOptimizationResult:
        covariance = np.asarray(problem.covariance_matrix, dtype=float)
        expected_returns = np.asarray(problem.expected_returns, dtype=float)
        ridge = covariance + np.eye(covariance.shape[0], dtype=float) * 1e-6
        raw = np.linalg.pinv(ridge) @ expected_returns
        if problem.config.long_only:
            raw = np.clip(raw, self.config.min_weight, None)
        raw = np.clip(raw, self.config.min_weight, self.config.max_weight)
        vector = self._normalize(raw, float(problem.config.budget))
        return self._build_result(problem, vector, method="numpy", solver_used="pinv", success=True)

    def _build_result(
        self,
        problem: PortfolioQUBOProblem,
        vector: Any,
        method: str,
        solver_used: str,
        success: bool,
    ) -> ClassicalOptimizationResult:
        covariance = np.asarray(problem.covariance_matrix, dtype=float)
        expected_returns = np.asarray(problem.expected_returns, dtype=float)
        portfolio_return = float(expected_returns @ vector)
        portfolio_risk = float(np.sqrt(max(vector @ covariance @ vector, 0.0)))
        objective_value = float(problem.config.risk_aversion * vector @ covariance @ vector - portfolio_return)
        weights = {symbol: float(weight) for symbol, weight in zip(problem.symbols, vector)}
        return ClassicalOptimizationResult(
            weights=weights,
            objective_value=objective_value,
            expected_return=portfolio_return,
            portfolio_risk=portfolio_risk,
            method=method,
            solver_used=solver_used,
            success=success,
        )

    @staticmethod
    def _normalize(vector: Any, budget: float) -> Any:
        total = float(np.sum(vector))
        if total <= 1e-10:
            return np.full(vector.shape[0], budget / max(vector.shape[0], 1), dtype=float)
        return vector * (budget / total)
