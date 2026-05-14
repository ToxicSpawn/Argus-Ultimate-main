"""
Genetic Strategy Parameter Evolution for ARGUS.

Evolves strategy parameters over generations using selection, crossover,
and mutation. Maintains a population of parameter sets and produces
increasingly fit configurations based on measured trading performance.

Usage:
    evolver = GeneticEvolver(
        parameter_ranges={"rsi_period": (5, 30), "bb_width": (1.0, 3.0)},
        population_size=20,
    )
    population = evolver.get_population()
    # ... run each individual in backtest/paper, collect fitness ...
    evolver.evaluate_generation({"ind_0": 1.5, "ind_1": -0.3, ...})
    next_gen = evolver.evolve()
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GeneticEvolver
# ---------------------------------------------------------------------------


class GeneticEvolver:
    """
    Evolve strategy parameters via genetic algorithm.

    Parameters
    ----------
    parameter_ranges : dict[str, tuple[float, float]]
        {param_name: (min_value, max_value)} defining the search space.
    population_size : int
        Number of individuals per generation (default 20).
    elitism_rate : float
        Fraction of top performers that survive unchanged (default 0.2).
    crossover_rate : float
        Fraction of population created via crossover (default 0.6).
    mutation_rate : float
        Fraction of population created via mutation (default 0.2).
    mutation_strength : float
        Standard deviation of mutation as fraction of parameter range (default 0.1).
    """

    def __init__(
        self,
        parameter_ranges: Dict[str, Tuple[float, float]],
        population_size: int = 20,
        elitism_rate: float = 0.20,
        crossover_rate: float = 0.60,
        mutation_rate: float = 0.20,
        mutation_strength: float = 0.10,
    ) -> None:
        if not parameter_ranges:
            raise ValueError("parameter_ranges must not be empty")
        if population_size < 4:
            raise ValueError("population_size must be >= 4")

        self._param_ranges = dict(parameter_ranges)
        self._param_names = list(parameter_ranges.keys())
        self._pop_size = population_size
        self._elitism_rate = elitism_rate
        self._crossover_rate = crossover_rate
        self._mutation_rate = mutation_rate
        self._mutation_strength = mutation_strength

        # Current generation
        self._generation = 0
        self._population: List[Dict[str, float]] = []
        self._fitness: Dict[str, float] = {}  # individual_id -> fitness
        self._generation_history: List[Dict[str, Any]] = []

        # Best ever
        self._best_params: Optional[Dict[str, float]] = None
        self._best_fitness: float = float("-inf")

        # Initialize random population
        self._initialize_population()

    def _initialize_population(self) -> None:
        """Create initial random population within parameter ranges."""
        self._population = []
        for _ in range(self._pop_size):
            individual = {}
            for name, (lo, hi) in self._param_ranges.items():
                individual[name] = float(np.random.uniform(lo, hi))
            self._population.append(individual)

    # ── Access ────────────────────────────────────────────────────────────

    def get_population(self) -> List[Dict[str, float]]:
        """Return current population of parameter sets."""
        return [dict(ind) for ind in self._population]

    def get_individual_id(self, index: int) -> str:
        """Return a stable ID for an individual by index."""
        return f"gen{self._generation}_ind{index}"

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate_generation(self, fitness_scores: Dict[str, float]) -> None:
        """
        Score each individual in the current generation.

        Parameters
        ----------
        fitness_scores : dict[str, float]
            {individual_id: fitness_value} — higher is better.
            Can also be {str(index): fitness_value} for convenience.
        """
        self._fitness = {}
        for key, score in fitness_scores.items():
            self._fitness[str(key)] = float(score)

        # Track best-ever
        for key, score in self._fitness.items():
            if score > self._best_fitness:
                self._best_fitness = score
                # Resolve the individual from the key
                try:
                    idx = int(str(key).split("_ind")[-1]) if "_ind" in str(key) else int(key)
                    if 0 <= idx < len(self._population):
                        self._best_params = dict(self._population[idx])
                except (ValueError, IndexError):
                    pass

        # Record generation stats
        scores = list(self._fitness.values())
        if scores:
            self._generation_history.append({
                "generation": self._generation,
                "best_fitness": max(scores),
                "avg_fitness": float(np.mean(scores)),
                "worst_fitness": min(scores),
                "diversity": float(np.std(scores)),
                "timestamp": time.time(),
            })

    # ── Evolution ─────────────────────────────────────────────────────────

    def evolve(self) -> List[Dict[str, float]]:
        """
        Create next generation via:
        - Top 20% survive (elitism)
        - 60% crossover (blend two parents' parameters)
        - 20% mutation (random perturbation within bounds)

        Returns new parameter sets to test.
        """
        if not self._fitness:
            logger.warning("GeneticEvolver: no fitness scores — returning current population")
            return self.get_population()

        # Sort population by fitness
        scored = []
        for i, ind in enumerate(self._population):
            # Try various key formats
            fid = self.get_individual_id(i)
            fitness = self._fitness.get(fid, self._fitness.get(str(i), float("-inf")))
            scored.append((fitness, i, ind))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Determine counts
        n_elite = max(1, int(self._pop_size * self._elitism_rate))
        n_crossover = int(self._pop_size * self._crossover_rate)
        n_mutation = self._pop_size - n_elite - n_crossover

        new_population: List[Dict[str, float]] = []

        # 1. Elitism: top performers survive
        for fitness, idx, ind in scored[:n_elite]:
            new_population.append(dict(ind))

        # 2. Crossover: blend two parents
        parents = [ind for _, _, ind in scored[:max(2, n_elite * 2)]]
        for _ in range(n_crossover):
            p1, p2 = (
                parents[np.random.randint(len(parents))],
                parents[np.random.randint(len(parents))],
            )
            child = self._crossover(p1, p2)
            new_population.append(child)

        # 3. Mutation: perturb existing individuals
        for _ in range(n_mutation):
            base = scored[np.random.randint(len(scored))][2]
            mutant = self._mutate(dict(base))
            new_population.append(mutant)

        self._population = new_population[:self._pop_size]
        self._generation += 1
        self._fitness = {}

        logger.info(
            "GeneticEvolver: evolved to generation %d — %d elite, %d crossover, %d mutation",
            self._generation, n_elite, n_crossover, n_mutation,
        )

        return self.get_population()

    def _crossover(self, p1: Dict[str, float], p2: Dict[str, float]) -> Dict[str, float]:
        """Blend crossover: child = alpha * p1 + (1-alpha) * p2."""
        alpha = np.random.uniform(0.2, 0.8)
        child = {}
        for name, (lo, hi) in self._param_ranges.items():
            v1 = p1.get(name, (lo + hi) / 2)
            v2 = p2.get(name, (lo + hi) / 2)
            val = alpha * v1 + (1.0 - alpha) * v2
            child[name] = float(np.clip(val, lo, hi))
        return child

    def _mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """Gaussian mutation within bounds."""
        mutant = {}
        for name, (lo, hi) in self._param_ranges.items():
            val = individual.get(name, (lo + hi) / 2)
            spread = (hi - lo) * self._mutation_strength
            val += float(np.random.normal(0, spread))
            mutant[name] = float(np.clip(val, lo, hi))
        return mutant

    # ── Queries ───────────────────────────────────────────────────────────

    def get_best(self) -> Dict[str, float]:
        """Return best parameters found across all generations."""
        if self._best_params is not None:
            return dict(self._best_params)
        # Fallback: return first individual
        if self._population:
            return dict(self._population[0])
        return {}

    def get_generation_stats(self) -> dict:
        """Return {generation, best_fitness, avg_fitness, diversity}."""
        if self._generation_history:
            return dict(self._generation_history[-1])
        return {
            "generation": self._generation,
            "best_fitness": 0.0,
            "avg_fitness": 0.0,
            "worst_fitness": 0.0,
            "diversity": 0.0,
        }

    def snapshot(self) -> Dict[str, Any]:
        """Return current state for dashboard/logging."""
        return {
            "generation": self._generation,
            "population_size": len(self._population),
            "best_fitness": self._best_fitness if self._best_fitness > float("-inf") else None,
            "best_params": self._best_params,
            "stats": self.get_generation_stats(),
        }
