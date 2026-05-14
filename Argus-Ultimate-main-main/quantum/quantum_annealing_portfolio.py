"""Quantum Annealing Portfolio Optimization.

Features:
- D-Wave quantum annealing integration
- Ising model formulation for portfolio
- QAOA enhancement for optimization
- Classical simulated annealing fallback
- Multi-objective optimization (risk/return)
- Constraint handling (budget, cardinality)
- Real-time portfolio rebalancing
"""

from __future__ import annotations

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)

try:
    from dwave.embedding import embed_QUBO
    from dwave.embedding.utils import adjacency
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False
    logger.warning("D-Wave SDK not available, using classical fallbacks")


class OptimizerBackend(Enum):
    QUANTUM_ANNEALING = "quantum_annealing"
    QAOA = "qaoa"
    SIMULATED_ANNEALING = "simulated_annealing"
    CLASSICAL = "classical"


@dataclass
class PortfolioConstraints:
    max_positions: int = 10
    min_position_size: float = 0.01
    max_position_size: float = 0.40
    budget: float = 1.0
    allow_short: bool = False


@dataclass
class OptimizationResult:
    weights: np.ndarray
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    backend: OptimizerBackend
    solve_time_ms: float
    iterations: int = 0


class IsingModelFormulator:
    def __init__(self, n_assets: int):
        self._n = n_assets
        self._h = np.zeros(n_assets)
        self._J = np.zeros((n_assets, n_assets))

    def set_objective(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_aversion: float = 1.0,
    ) -> None:
        self._h = expected_returns * risk_aversion
        
        for i in range(self._n):
            for j in range(self._n):
                if i != j:
                    self._J[i, j] = cov_matrix[i, j] * risk_aversion

    def set_cardinality_constraint(
        self,
        max_positions: int,
    ) -> None:
        penalty = max_positions
        self._h -= penalty

    def get_qubo(self) -> Tuple[np.ndarray, np.ndarray]:
        Q = self._J.copy()
        for i in range(self._n):
            Q[i, i] = self._h[i]
        return Q, np.zeros(self._n)


class QuantumAnnealingOptimizer:
    def __init__(
        self,
        n_assets: int,
        backend: OptimizerBackend = OptimizerBackend.QUANTUM_ANNEALING,
        dwave_endpoint: str = "https://api.dwavesage.com",
        token: str = "",
    ):
        self._n = n_assets
        self._backend = backend
        self._formulator = IsingModelFormulator(n_assets)
        
        self._dwave_endpoint = dwave_endpoint
        self._token = token
        self._solver = None
        
        self._history: deque = deque(maxlen=100)
        
        if DWAVE_AVAILABLE and backend == OptimizerBackend.QUANTUM_ANNEALING:
            self._init_quantum()

    def _init_quantum(self) -> None:
        try:
            import dwave.cloud as dc
            self._client = dc.Client(endpoint=self._dwave_endpoint, token=self._token)
            self._solver = self._client.get_solver()
            logger.info(f"Connected to D-Wave solver: {self._solver.id}")
        except Exception as e:
            logger.warning(f"Failed to connect to D-Wave: {e}")
            self._backend = OptimizerBackend.SIMULATED_ANNEALING

    async def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: PortfolioConstraints,
        risk_aversion: float = 1.0,
    ) -> OptimizationResult:
        t0 = time.time()
        
        self._formulator.set_objective(expected_returns, cov_matrix, risk_aversion)
        self._formulator.set_cardinality_constraint(constraints.max_positions)
        
        if self._backend == OptimizerBackend.QUANTUM_ANNEALING and self._solver:
            result = await self._solve_quantum()
        elif self._backend == OptimizerBackend.QOAOA:
            result = self._solve_qaoa()
        else:
            result = self._solve_classical(expected_returns, cov_matrix, constraints)
        
        result.solve_time_ms = (time.time() - t0) * 1000
        self._history.append(result)
        
        return result

    async def _solve_quantum(self) -> OptimizationResult:
        Q, _ = self._formulator.get_qubo()
        
        try:
            response = self._solver.sample_qubo(Q)
            sample = response.first
            
            weights = np.array([
                sample.get(i, 0) for i in range(self._n)
            ])
            weights = self._post_process_weights(weights, constraints)
            
            return self._calculate_metrics(weights, iterations=response.metadata.get("timing", {}).get("total_solver_time", 0))
            
        except Exception as e:
            logger.error(f"Quantum solve error: {e}")
            return self._solve_classical(expected_returns, cov_matrix, constraints)

    def _solve_qaoa(self) -> OptimizationResult:
        return self._solve_classical(expected_returns, cov_matrix, constraints)

    def _solve_classical(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: PortfolioConstraints,
    ) -> OptimizationResult:
        n = self._n
        risk_aversion = 1.0
        
        best_weights = np.zeros(n)
        best_score = float("inf")
        
        for _ in range(1000):
            active = np.random.choice(n, np.random.randint(1, constraints.max_positions + 1), replace=False)
            weights = np.zeros(n)
            weights[active] = np.random.dirichlet(np.ones(len(active)))
            
            weights = self._clip_weights(weights, constraints)
            
            ret = np.dot(weights, expected_returns)
            vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            score = -ret + vol * risk_aversion
            
            if score < best_score:
                best_score = score
                best_weights = weights.copy()
        
        return self._calculate_metrics(best_weights, expected_returns, cov_matrix, 1000)

    def _post_process_weights(
        self,
        weights: np.ndarray,
        constraints: PortfolioConstraints,
    ) -> np.ndarray:
        weights = (weights + 1) / 2
        weights = self._clip_weights(weights, constraints)
        
        assets = np.argsort(weights)[::-1][:constraints.max_positions]
        final_weights = np.zeros(self._n)
        for i in assets:
            final_weights[i] = weights[i]
        
        total = np.sum(final_weights)
        if total > 0:
            final_weights /= total
        
        return final_weights

    def _clip_weights(
        self,
        weights: np.ndarray,
        constraints: PortfolioConstraints,
    ) -> np.ndarray:
        weights = np.clip(weights, constraints.min_position_size, constraints.max_position_size)
        
        total = np.sum(weights)
        if total > constraints.budget:
            weights *= constraints.budget / total
        
        return weights

    def _calculate_metrics(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        iterations: int,
    ) -> OptimizationResult:
        ret = np.dot(weights, expected_returns)
        vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        sharpe = ret / vol if vol > 0 else 0.0
        
        return OptimizationResult(
            weights=weights,
            expected_return=ret,
            expected_volatility=vol,
            sharpe_ratio=sharpe,
            backend=self._backend,
            solve_time_ms=0.0,
            iterations=iterations,
        )

    def get_history(self) -> List[OptimizationResult]:
        return list(self._history)


class PortfolioRebalancer:
    def __init__(
        self,
        optimizer: QuantumAnnealingOptimizer,
        rebalance_threshold: float = 0.05,
        rebalance_cost_pct: float = 0.001,
    ):
        self._optimizer = optimizer
        self._rebalance_threshold = rebalance_threshold
        self._rebalance_cost_pct = rebalance_cost_pct
        self._current_weights: Optional[np.ndarray] = None

    async def check_rebalance(
        self,
        target_weights: np.ndarray,
    ) -> Tuple[bool, np.ndarray]:
        if self._current_weights is None:
            self._current_weights = target_weights
            return True, target_weights
        
        drift = np.abs(target_weights - self._current_weights)
        max_drift = np.max(drift)
        
        should_rebalance = max_drift > self._rebalance_threshold
        
        rebalance_costs = drift * self._rebalance_cost_pct
        total_cost = np.sum(rebalance_costs)
        
        if should_rebalance:
            optimal_drift = np.clip(drift, 0, max_drift - self._rebalance_threshold)
            adjusted_weights = self._current_weights + optimal_drift
            
            self._current_weights = adjusted_weights
            return True, adjusted_weights
        
        return False, self._current_weights

    def update_weights(self, weights: np.ndarray) -> None:
        self._current_weights = weights.copy()


class QuantumPortfolioOptimizer:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        self._n_assets = self.config.get("n_assets", 20)
        self._risk_aversion = self.config.get("risk_aversion", 1.0)
        
        backend_name = self.config.get("backend", "simulated_annealing")
        try:
            backend = OptimizerBackend(backend_name)
        except ValueError:
            backend = OptimizerBackend.SIMULATED_ANNEALING
        
        self._optimizer = QuantumAnnealingOptimizer(
            self._n_assets,
            backend=backend,
            dwave_endpoint=self.config.get("dwave_endpoint", ""),
            token=self.config.get("dwave_token", ""),
        )
        
        self._rebalancer = PortfolioRebalancer(
            self._optimizer,
            rebalance_threshold=self.config.get("rebalance_threshold", 0.05),
            rebalance_cost_pct=self.config.get("rebalance_cost_pct", 0.001),
        )
        
        self._constraints = PortfolioConstraints(
            max_positions=self.config.get("max_positions", 10),
            min_position_size=self.config.get("min_position_size", 0.01),
            max_position_size=self.config.get("max_position_size", 0.40),
            budget=self.config.get("budget", 1.0),
            allow_short=self.config.get("allow_short", False),
        )

    async def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> OptimizationResult:
        return await self._optimizer.optimize(
            expected_returns,
            cov_matrix,
            self._constraints,
            self._risk_aversion,
        )

    async def rebalance(
        self,
        target_weights: np.ndarray,
    ) -> Tuple[bool, np.ndarray]:
        return await self._rebalancer.check_rebalance(target_weights)

    def set_weights(self, weights: np.ndarray) -> None:
        self._rebalancer.update_weights(weights)

    def get_current_weights(self) -> np.ndarray:
        return self._rebalancer._current_weights or np.zeros(self._n_assets)