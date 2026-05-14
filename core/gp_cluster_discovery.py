"""Genetic programming-based cluster discovery for correlated parameters.

Many ARGUS parameters are not independent — stop-loss, take-profit, and
position-size multipliers frequently move together inside a regime-specific
"cluster" of settings.  Finding those clusters by hand is tedious and easily
misses non-obvious co-dependencies.  This module uses a lightweight genetic
algorithm to search for parameter clusters whose joint outcomes correlate,
surfacing combinations that are good candidates for joint optimisation.

Usage::

    discoverer = GPClusterDiscoverer(seed=42)
    discoverer.set_observations([
        {"params": {"stop_loss": 0.02, "take_profit": 0.05, ...}, "outcome_pnl": 12.3},
        ...
    ])
    discoverer.evolve(n_generations=30)
    top = discoverer.get_top_clusters(5)

Observations are dicts with ``params`` (a mapping of parameter name to float)
and ``outcome_pnl`` (a float).  The fitness of a cluster is the absolute value
of the mean pairwise correlation between the observations' parameter values
and the outcome P&L, so either positive or negative co-movement qualifies.

Everything is pure numpy — no external GP library required.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Genome
# ---------------------------------------------------------------------------


@dataclass
class ClusterGenome:
    """One candidate cluster: a set of parameter names with a fitness score."""

    member_params: List[str] = field(default_factory=list)
    fitness: float = -math.inf
    coverage: int = 0

    def as_sorted(self) -> List[str]:
        return sorted(self.member_params)

    def signature(self) -> str:
        return "|".join(self.as_sorted())

    def snapshot(self) -> Dict[str, Any]:
        return {
            "members": self.as_sorted(),
            "n_members": len(self.member_params),
            "fitness": None if not math.isfinite(self.fitness) else float(self.fitness),
            "coverage": int(self.coverage),
        }


# ---------------------------------------------------------------------------
# Discoverer
# ---------------------------------------------------------------------------


class GPClusterDiscoverer:
    """Genetic search over subsets of parameter names.

    Fitness of a subset is the mean absolute correlation between each member's
    values (across all observations) and the observation P&L, multiplied by a
    "coherence bonus" that rewards clusters whose members are also correlated
    with each other.  This tends to find groups of parameters that jointly
    influence the outcome.
    """

    def __init__(
        self,
        population_size: int = 40,
        min_cluster_size: int = 2,
        max_cluster_size: int = 6,
        elitism: int = 4,
        mutation_rate: float = 0.25,
        seed: Optional[int] = None,
    ) -> None:
        self.population_size = int(max(4, population_size))
        self.min_cluster_size = int(max(2, min_cluster_size))
        self.max_cluster_size = int(max(min_cluster_size, max_cluster_size))
        self.elitism = int(max(1, elitism))
        self.mutation_rate = float(mutation_rate)
        self._rng = np.random.default_rng(seed)

        self._observations: List[Dict[str, Any]] = []
        self._param_names: List[str] = []
        self._param_matrix: np.ndarray = np.zeros((0, 0))
        self._pnl_vec: np.ndarray = np.zeros((0,))

        self._population: List[ClusterGenome] = []
        self._history: List[float] = []
        self._generation: int = 0

    # -- data intake ---------------------------------------------------------

    def set_observations(self, observations: Sequence[Dict[str, Any]]) -> None:
        """Install a batch of observations and rebuild the parameter matrix."""
        self._observations = list(observations)
        names: List[str] = []
        seen = set()
        for obs in self._observations:
            params = obs.get("params", {}) or {}
            for key in params.keys():
                if key not in seen:
                    seen.add(key)
                    names.append(key)

        self._param_names = names
        n = len(self._observations)
        m = len(names)
        mat = np.full((n, m), np.nan, dtype=np.float64)
        pnl = np.zeros(n, dtype=np.float64)

        for i, obs in enumerate(self._observations):
            params = obs.get("params", {}) or {}
            for j, key in enumerate(names):
                if key in params:
                    try:
                        mat[i, j] = float(params[key])
                    except (TypeError, ValueError):
                        mat[i, j] = np.nan
            try:
                pnl[i] = float(obs.get("outcome_pnl", 0.0))
            except (TypeError, ValueError):
                pnl[i] = 0.0

        self._param_matrix = mat
        self._pnl_vec = pnl
        self._population = []
        self._history = []
        self._generation = 0

    # -- fitness -------------------------------------------------------------

    @staticmethod
    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        mask = np.isfinite(a) & np.isfinite(b)
        if int(np.sum(mask)) < 3:
            return 0.0
        aa = a[mask]
        bb = b[mask]
        if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
            return 0.0
        return float(np.corrcoef(aa, bb)[0, 1])

    def _cluster_fitness(self, members: Sequence[str]) -> Tuple[float, int]:
        if not members:
            return -math.inf, 0
        if len(members) < self.min_cluster_size:
            return -math.inf, 0
        if self._param_matrix.size == 0:
            return -math.inf, 0

        idx = [self._param_names.index(m) for m in members if m in self._param_names]
        if len(idx) < self.min_cluster_size:
            return -math.inf, 0

        sub = self._param_matrix[:, idx]  # (n_obs, k)
        coverage_mask = np.all(np.isfinite(sub), axis=1)
        coverage = int(np.sum(coverage_mask))
        if coverage < 3:
            return -math.inf, coverage

        # Per-member correlation with P&L (how strongly each member moves the outcome).
        pnl_corrs = []
        for col in range(sub.shape[1]):
            pnl_corrs.append(abs(self._safe_corr(sub[:, col], self._pnl_vec)))
        mean_pnl_corr = float(np.mean(pnl_corrs)) if pnl_corrs else 0.0

        # Pairwise coherence: do members co-vary with each other?
        k = sub.shape[1]
        if k >= 2:
            pairwise = []
            for i in range(k):
                for j in range(i + 1, k):
                    pairwise.append(abs(self._safe_corr(sub[:, i], sub[:, j])))
            coherence = float(np.mean(pairwise)) if pairwise else 0.0
        else:
            coherence = 0.0

        # Penalty for very large clusters (diminishing returns).
        size_penalty = 1.0 / (1.0 + 0.1 * max(0, k - self.min_cluster_size))

        fitness = (0.7 * mean_pnl_corr + 0.3 * coherence) * size_penalty
        return float(fitness), coverage

    # -- genetic operators ---------------------------------------------------

    def _random_cluster(self) -> ClusterGenome:
        if not self._param_names:
            return ClusterGenome()
        size = int(
            self._rng.integers(self.min_cluster_size, self.max_cluster_size + 1)
        )
        size = min(size, len(self._param_names))
        members = list(
            self._rng.choice(self._param_names, size=size, replace=False)
        )
        return ClusterGenome(member_params=members)

    def _mutate(self, genome: ClusterGenome) -> ClusterGenome:
        if not self._param_names:
            return genome
        members = list(genome.member_params)
        roll = self._rng.random()
        if roll < 0.4 and len(members) < self.max_cluster_size:
            # ADD a new param (if any remain).
            candidates = [p for p in self._param_names if p not in members]
            if candidates:
                members.append(str(self._rng.choice(candidates)))
        elif roll < 0.7 and len(members) > self.min_cluster_size:
            # REMOVE a param.
            drop_idx = int(self._rng.integers(0, len(members)))
            members.pop(drop_idx)
        else:
            # SWAP a param.
            if members:
                swap_idx = int(self._rng.integers(0, len(members)))
                candidates = [p for p in self._param_names if p not in members]
                if candidates:
                    members[swap_idx] = str(self._rng.choice(candidates))
        return ClusterGenome(member_params=members)

    def _crossover(self, a: ClusterGenome, b: ClusterGenome) -> ClusterGenome:
        pool = list(set(a.member_params) | set(b.member_params))
        if not pool:
            return self._random_cluster()
        target_size = int(
            np.clip(
                (len(a.member_params) + len(b.member_params)) // 2,
                self.min_cluster_size,
                min(self.max_cluster_size, len(pool)),
            )
        )
        members = list(self._rng.choice(pool, size=target_size, replace=False))
        return ClusterGenome(member_params=members)

    def _tournament(self) -> ClusterGenome:
        if not self._population:
            return self._random_cluster()
        k = min(3, len(self._population))
        contenders = list(self._rng.choice(self._population, size=k, replace=False))
        contenders.sort(key=lambda g: g.fitness, reverse=True)
        return contenders[0]

    # -- evolution loop ------------------------------------------------------

    def _evaluate_population(self) -> None:
        seen: Dict[str, ClusterGenome] = {}
        for genome in self._population:
            sig = genome.signature()
            if sig in seen:
                continue
            fit, cov = self._cluster_fitness(genome.member_params)
            genome.fitness = fit
            genome.coverage = cov
            seen[sig] = genome
        self._population = list(seen.values())
        self._population.sort(key=lambda g: g.fitness, reverse=True)

    def evolve(self, n_generations: int = 20) -> ClusterGenome:
        if not self._param_names:
            logger.warning("evolve() called with no observations registered")
            return ClusterGenome()

        # Seed population if empty.
        if not self._population:
            self._population = [self._random_cluster() for _ in range(self.population_size)]
            self._evaluate_population()

        for _ in range(max(0, int(n_generations))):
            self._generation += 1
            elites = self._population[: self.elitism]
            children: List[ClusterGenome] = list(elites)

            while len(children) < self.population_size:
                parent_a = self._tournament()
                parent_b = self._tournament()
                child = self._crossover(parent_a, parent_b)
                if self._rng.random() < self.mutation_rate:
                    child = self._mutate(child)
                children.append(child)

            self._population = children
            self._evaluate_population()

            best = self._population[0] if self._population else ClusterGenome()
            self._history.append(float(best.fitness if math.isfinite(best.fitness) else 0.0))

        return self._population[0] if self._population else ClusterGenome()

    # -- inspection ----------------------------------------------------------

    def get_top_clusters(self, n: int = 5) -> List[ClusterGenome]:
        return list(self._population[: max(1, int(n))])

    def snapshot(self) -> Dict[str, Any]:
        return {
            "n_observations": len(self._observations),
            "n_params_seen": len(self._param_names),
            "generation": self._generation,
            "population_size": len(self._population),
            "best_history": list(self._history[-20:]),
            "top": [g.snapshot() for g in self.get_top_clusters(5)],
            "params_seen": list(self._param_names),
        }


__all__ = ["ClusterGenome", "GPClusterDiscoverer"]
