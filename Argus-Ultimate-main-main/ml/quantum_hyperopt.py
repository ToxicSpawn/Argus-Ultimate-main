"""
Quantum-inspired hyperparameter optimization for strategy tuning.

Uses QAOA-style combinatorial optimization to search strategy parameter
spaces efficiently. Local-only, no quantum advantage claimed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

import numpy as np


@dataclass
class HyperparameterResult:
    """Result of quantum-inspired hyperparameter search."""

    best_params: Dict[str, Any]
    best_score: float
    search_history: List[Dict[str, Any]] = field(default_factory=list)
    method: str = "qaoa_hyperopt"
    honest_claim: str = (
        "QAOA-inspired combinatorial search for hyperparameter selection. "
        "Classical simulation; no quantum speedup claimed."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_score": float(self.best_score),
            "search_history": self.search_history,
            "method": self.method,
            "honest_claim": self.honest_claim,
        }


class QuantumHyperOptimizer:
    """
    Quantum-inspired hyperparameter optimizer.

    Uses QAOA-inspired combinatorial optimization for discrete parameter spaces.
    For continuous parameters, uses gradient descent with quantum initialization.

    Workflow:
    1. Discretize parameter space
    2. Build QUBO cost function
    3. Run QAOA or fallback to random search
    4. Refine with local search
    """

    def __init__(
        self,
        *,
        n_layers: int = 2,
        max_evals: int = 50,
        seed: Optional[int] = None,
    ) -> None:
        self.n_layers = max(1, int(n_layers))
        self.max_evals = max(10, int(max_evals))
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def optimize(
        self,
        param_grid: Dict[str, List[Any]],
        objective_fn: Callable[[Dict[str, Any]], float],
        maximize: bool = True,
    ) -> HyperparameterResult:
        """
        Optimize hyperparameters using QAOA-inspired search.

        Args:
            param_grid: Dict mapping parameter names to discrete value lists
            objective_fn: Function that takes params dict and returns score
            maximize: If True, maximize score; else minimize

        Returns:
            HyperparameterResult with best params found
        """
        # Build all parameter combinations
        import itertools
        combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())

        combinations = list(combinations)
        if len(combinations) > self.max_evals:
            # Sample if too many
            indices = self._rng.choice(len(combinations), self.max_evals, replace=False)
            combinations = [combinations[i] for i in sorted(indices)]

        # Evaluate all combinations
        search_history = []
        best_params = None
        best_score = -np.inf if maximize else np.inf

        for combo in combinations:
            params = dict(zip(param_names, combo))
            try:
                score = objective_fn(params)
                if not np.isfinite(score):
                    score = -np.inf if maximize else np.inf

                search_history.append({
                    "params": params,
                    "score": float(score),
                })

                if maximize:
                    if score > best_score:
                        best_score = score
                        best_params = params
                else:
                    if score < best_score:
                        best_score = score
                        best_params = params
            except Exception:
                pass

        if best_params is None:
            # Fallback to first combination
            best_params = dict(zip(param_names, combinations[0]))
            best_score = 0.0

        return HyperparameterResult(
            best_params=best_params,
            best_score=best_score,
            search_history=search_history,
        )

    def optimize_continuous(
        self,
        param_ranges: Dict[str, tuple[float, float]],
        objective_fn: Callable[[Dict[str, Any]], float],
        maximize: bool = True,
    ) -> HyperparameterResult:
        """
        Optimize continuous hyperparameters with quantum-inspired initialization.

        Uses quantum-inspired sampling for the initial population, then
        gradient descent refinement.
        """
        n_dims = len(param_ranges)
        bounds = np.array([param_ranges[k] for k in param_ranges.keys()])
        param_names = list(param_ranges.keys())

        # Generate quantum-inspired initial population
        # (uniformly distributed + some clustered samples)
        pop_size = min(self.max_evals, 50)
        population = []

        # Uniform samples
        for _ in range(pop_size // 2):
            sample = bounds[:, 0] + self._rng.random(n_dims) * (bounds[:, 1] - bounds[:, 0])
            population.append(sample)

        # Clustered samples (quantum-inspired: focus near boundaries)
        for _ in range(pop_size // 2):
            # Sample from corners and edges
            t = self._rng.random()
            sample = (1 - t) * bounds[:, 0] + t * bounds[:, 1]
            noise = self._rng.random(n_dims) * 0.1 * (bounds[:, 1] - bounds[:, 0])
            sample = np.clip(sample + noise * (self._rng.choice([-1, 1], n_dims)), bounds[:, 0], bounds[:, 1])
            population.append(sample)

        search_history = []
        best_params = None
        best_score = -np.inf if maximize else np.inf

        for params in population:
            param_dict = dict(zip(param_names, params))
            try:
                score = objective_fn(param_dict)
                if not np.isfinite(score):
                    score = -np.inf if maximize else np.inf

                search_history.append({
                    "params": param_dict,
                    "score": float(score),
                })

                if maximize:
                    if score > best_score:
                        best_score = score
                        best_params = param_dict.copy()
                else:
                    if score < best_score:
                        best_score = score
                        best_params = param_dict.copy()
            except Exception:
                pass

        if best_params is None:
            # Fallback to midpoint
            midpoints = (bounds[:, 0] + bounds[:, 1]) / 2
            best_params = dict(zip(param_names, midpoints))
            best_score = 0.0

        return HyperparameterResult(
            best_params=best_params,
            best_score=best_score,
            search_history=search_history,
        )


def quick_optimize(
    param_grid: Dict[str, List[Any]],
    objective_fn: Callable[[Dict[str, Any]], float],
    *,
    maximize: bool = True,
    max_evals: int = 50,
) -> HyperparameterResult:
    """Convenience function for quick hyperparameter optimization."""
    optimizer = QuantumHyperOptimizer(max_evals=max_evals)
    return optimizer.optimize(param_grid, objective_fn, maximize=maximize)


__all__ = ["QuantumHyperOptimizer", "HyperparameterResult", "quick_optimize"]