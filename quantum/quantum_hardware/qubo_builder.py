from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

np = importlib.import_module("numpy")

logger = logging.getLogger(__name__)

_EPS = 1e-10


@dataclass(slots=True)
class PortfolioQUBOConfig:
    risk_aversion: float = 1.0
    budget: float = 1.0
    budget_penalty: float = 6.0
    max_assets: int | None = None
    cardinality_penalty: float = 4.0
    long_only: bool = True

    def __post_init__(self) -> None:
        self.risk_aversion = float(self.risk_aversion)
        self.budget = float(self.budget)
        self.budget_penalty = float(self.budget_penalty)
        self.cardinality_penalty = float(self.cardinality_penalty)
        if self.risk_aversion < 0.0:
            raise ValueError("risk_aversion must be non-negative")
        if self.budget <= 0.0:
            raise ValueError("budget must be positive")
        if self.budget_penalty <= 0.0:
            raise ValueError("budget_penalty must be positive")
        if self.max_assets is not None and self.max_assets <= 0:
            raise ValueError("max_assets must be positive when provided")


@dataclass(slots=True)
class PortfolioQUBOProblem:
    symbols: Sequence[str]
    expected_returns: Sequence[float]
    covariance_matrix: Sequence[Sequence[float]]
    config: PortfolioQUBOConfig = field(default_factory=PortfolioQUBOConfig)

    def __post_init__(self) -> None:
        self.symbols = [str(symbol) for symbol in self.symbols]
        expected_returns = np.asarray(self.expected_returns, dtype=float).reshape(-1)
        covariance_matrix = np.asarray(self.covariance_matrix, dtype=float)
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix

        n_assets = len(self.symbols)
        if n_assets == 0:
            raise ValueError("symbols must not be empty")
        if expected_returns.shape[0] != n_assets:
            raise ValueError("expected_returns must match symbols length")
        if covariance_matrix.shape != (n_assets, n_assets):
            raise ValueError("covariance_matrix must be square with one row per symbol")
        if not np.all(np.isfinite(expected_returns)):
            raise ValueError("expected_returns contains non-finite values")
        if not np.all(np.isfinite(covariance_matrix)):
            raise ValueError("covariance_matrix contains non-finite values")
        if self.config.max_assets is not None and self.config.max_assets > n_assets:
            self.config.max_assets = n_assets


@dataclass(slots=True)
class PortfolioQUBOModel:
    symbols: list[str]
    qubo_matrix: Any
    linear_terms: Any
    metadata: dict[str, float]

    def energy(self, bitstring: Sequence[int]) -> float:
        x = np.asarray(bitstring, dtype=float).reshape(-1)
        return float(x @ self.qubo_matrix @ x + self.linear_terms @ x)

    def bitstring_to_weights(self, bitstring: Sequence[int]) -> dict[str, float]:
        x = np.asarray(bitstring, dtype=float).reshape(-1)
        x = np.clip(x, 0.0, 1.0)
        total = float(np.sum(x))
        if total <= _EPS:
            x = np.full(x.shape[0], 1.0 / x.shape[0], dtype=float)
            total = 1.0
        normalized = x / total
        return {symbol: float(weight) for symbol, weight in zip(self.symbols, normalized)}


class PortfolioQUBOBuilder:
    """Convert a long-only mean-variance portfolio problem into a binary QUBO."""

    def build(self, problem: PortfolioQUBOProblem) -> PortfolioQUBOModel:
        returns = np.asarray(problem.expected_returns, dtype=float)
        covariance = self._regularize_covariance(problem.covariance_matrix)
        n_assets = returns.shape[0]
        ones = np.ones(n_assets, dtype=float)

        return_scale = max(float(np.max(np.abs(returns))), 1.0)
        risk_scale = max(float(np.max(np.abs(covariance))), 1.0)
        scaled_returns = returns / return_scale
        scaled_covariance = covariance / risk_scale

        config = problem.config
        qubo = config.risk_aversion * scaled_covariance
        qubo = qubo + config.budget_penalty * np.outer(ones, ones)
        linear = -scaled_returns - (2.0 * config.budget_penalty * config.budget * ones)

        if config.max_assets is not None:
            target = min(config.max_assets, n_assets)
            qubo = qubo + config.cardinality_penalty * np.outer(ones, ones)
            linear = linear - (2.0 * config.cardinality_penalty * target * ones)
            linear = linear + (config.cardinality_penalty * ones)

        qubo = (qubo + qubo.T) / 2.0

        metadata = {
            "risk_aversion": float(config.risk_aversion),
            "budget": float(config.budget),
            "budget_penalty": float(config.budget_penalty),
            "return_scale": float(return_scale),
            "risk_scale": float(risk_scale),
            "n_assets": float(n_assets),
        }
        if config.max_assets is not None:
            metadata["max_assets"] = float(config.max_assets)
            metadata["cardinality_penalty"] = float(config.cardinality_penalty)

        logger.debug("Built portfolio QUBO for %d asset(s)", n_assets)
        return PortfolioQUBOModel(
            symbols=list(problem.symbols),
            qubo_matrix=qubo,
            linear_terms=linear,
            metadata=metadata,
        )

    def to_qubo_dict(self, model: PortfolioQUBOModel) -> dict[tuple[int, int], float]:
        matrix = np.asarray(model.qubo_matrix, dtype=float)
        linear = np.asarray(model.linear_terms, dtype=float)
        qubo: dict[tuple[int, int], float] = {}

        for i in range(matrix.shape[0]):
            coeff = float(matrix[i, i] + linear[i])
            if abs(coeff) > _EPS:
                qubo[(i, i)] = coeff
            for j in range(i + 1, matrix.shape[1]):
                value = float(matrix[i, j] + matrix[j, i])
                if abs(value) > _EPS:
                    qubo[(i, j)] = value
        return qubo

    @staticmethod
    def _regularize_covariance(covariance_matrix: Any) -> Any:
        covariance = np.asarray(covariance_matrix, dtype=float)
        covariance = (covariance + covariance.T) / 2.0
        min_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
        if min_eigenvalue < _EPS:
            covariance = covariance + np.eye(covariance.shape[0], dtype=float) * (_EPS - min_eigenvalue + 1e-8)
        return covariance
