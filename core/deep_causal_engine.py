"""Deep causal inference engine using Pearl's do-calculus.

This module provides a compact implementation of causal reasoning
suitable for ARGUS. It supports:

* Causal DAG construction via :class:`CausalDAG` with cycle detection.
* Interventions computed with the truncated-product formula through
  :class:`InterventionEngine`.
* A facade :class:`DeepCausalEngine` that glues it all together,
  exposing observation, intervention, inference, and basic structure
  learning primitives (a simplified PC algorithm).

Because we forbid scipy/sklearn, conditional independence tests use
partial-correlation based statistics computed directly from numpy.
The resulting graph is an approximation but captures the core idea
of constraint-based structure learning.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Causal DAG
# ---------------------------------------------------------------------------


@dataclass
class CausalDAG:
    """Directed acyclic graph of causal relationships."""

    nodes: Set[str] = field(default_factory=set)
    edges: Set[Tuple[str, str]] = field(default_factory=set)
    conditional_probs: Dict[str, Dict[FrozenSet[Tuple[str, float]], float]] = field(
        default_factory=dict
    )

    def add_node(self, name: str) -> None:
        self.nodes.add(name)
        self.conditional_probs.setdefault(name, {})

    def add_edge(self, cause: str, effect: str) -> bool:
        """Add a directed edge; rejects additions that would create a cycle."""
        if cause == effect:
            return False
        self.add_node(cause)
        self.add_node(effect)
        self.edges.add((cause, effect))
        if self._has_cycle():
            self.edges.discard((cause, effect))
            return False
        return True

    def remove_edge(self, cause: str, effect: str) -> None:
        self.edges.discard((cause, effect))

    def parents(self, node: str) -> List[str]:
        return [c for (c, e) in self.edges if e == node]

    def children(self, node: str) -> List[str]:
        return [e for (c, e) in self.edges if c == node]

    def descendants(self, node: str) -> Set[str]:
        desc: Set[str] = set()
        stack = list(self.children(node))
        while stack:
            n = stack.pop()
            if n in desc:
                continue
            desc.add(n)
            stack.extend(self.children(n))
        return desc

    def ancestors(self, node: str) -> Set[str]:
        anc: Set[str] = set()
        stack = list(self.parents(node))
        while stack:
            n = stack.pop()
            if n in anc:
                continue
            anc.add(n)
            stack.extend(self.parents(n))
        return anc

    def _has_cycle(self) -> bool:
        color: Dict[str, int] = {n: 0 for n in self.nodes}

        def visit(u: str) -> bool:
            color[u] = 1
            for v in self.children(u):
                if color.get(v, 0) == 1:
                    return True
                if color.get(v, 0) == 0 and visit(v):
                    return True
            color[u] = 2
            return False

        for n in list(self.nodes):
            if color.get(n, 0) == 0 and visit(n):
                return True
        return False

    def topological_order(self) -> List[str]:
        in_deg = {n: 0 for n in self.nodes}
        for _, e in self.edges:
            in_deg[e] = in_deg.get(e, 0) + 1
        queue = [n for n, d in in_deg.items() if d == 0]
        order: List[str] = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for c in self.children(n):
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)
        return order


# ---------------------------------------------------------------------------
# Intervention engine (do-calculus)
# ---------------------------------------------------------------------------


class InterventionEngine:
    """Computes post-intervention distributions using do-calculus.

    The truncated product formula states that for an intervention
    ``do(X = x)`` we delete the factor ``P(X | Pa(X))`` from the joint
    distribution and clamp ``X``. This class assumes discrete or
    Gaussian-linear conditional distributions — for simplicity we
    treat each variable as Gaussian with parameters learned from
    observed data (or supplied directly).
    """

    def __init__(self, dag: CausalDAG) -> None:
        self.dag = dag
        # Each node has (mu, sigma, weights) where weights map parent -> coef.
        self._params: Dict[str, Dict[str, Any]] = {}
        for n in dag.nodes:
            self._params[n] = {"mu": 0.0, "sigma": 1.0, "weights": {}}

    def set_params(
        self,
        node: str,
        mu: float,
        sigma: float,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self._params.setdefault(node, {"mu": 0.0, "sigma": 1.0, "weights": {}})
        self._params[node]["mu"] = float(mu)
        self._params[node]["sigma"] = max(float(sigma), 1e-9)
        if weights is not None:
            self._params[node]["weights"] = {k: float(v) for k, v in weights.items()}

    def _expected(
        self,
        node: str,
        assignments: Dict[str, float],
        interventions: Dict[str, float],
    ) -> float:
        if node in interventions:
            return float(interventions[node])
        if node in assignments:
            return float(assignments[node])
        params = self._params.get(node, {"mu": 0.0, "sigma": 1.0, "weights": {}})
        mu = params.get("mu", 0.0)
        weights = params.get("weights", {})
        for parent, coef in weights.items():
            mu += coef * self._expected(parent, assignments, interventions)
        return float(mu)

    def truncated_product(
        self,
        target: str,
        interventions: Dict[str, float],
        observations: Dict[str, float],
        n_samples: int = 512,
    ) -> Dict[str, float]:
        """Estimate ``P(target | do(interventions), obs)`` via sampling."""
        order = self.dag.topological_order()
        samples = np.zeros(n_samples, dtype=float)
        for s in range(n_samples):
            values: Dict[str, float] = dict(observations)
            values.update(interventions)
            for node in order:
                if node in interventions:
                    continue
                if node in observations:
                    continue
                params = self._params.get(node)
                if params is None:
                    values[node] = 0.0
                    continue
                mean = params.get("mu", 0.0)
                weights = params.get("weights", {})
                for parent, coef in weights.items():
                    mean += coef * values.get(parent, 0.0)
                sigma = params.get("sigma", 1.0)
                values[node] = float(np.random.normal(mean, sigma))
            samples[s] = values.get(target, 0.0)
        return {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
            "p05": float(np.percentile(samples, 5)),
            "p95": float(np.percentile(samples, 95)),
        }


# ---------------------------------------------------------------------------
# Deep causal engine facade
# ---------------------------------------------------------------------------


class DeepCausalEngine:
    """High-level causal reasoning facade with structure learning.

    Typical workflow::

        engine = DeepCausalEngine()
        engine.add_variable('rate')
        engine.add_variable('price')
        engine.add_edge('rate', 'price')
        engine.observe('rate', 0.05)
        post = engine.infer('price')
        cf = engine.intervene('rate', 0.10)
    """

    def __init__(self) -> None:
        self.dag = CausalDAG()
        self._engine = InterventionEngine(self.dag)
        self._observations: Dict[str, float] = {}
        self._data: Dict[str, List[float]] = {}
        self._last_structure_size = 0

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_variable(self, name: str, mu: float = 0.0, sigma: float = 1.0) -> None:
        self.dag.add_node(name)
        self._engine.set_params(name, mu=mu, sigma=sigma)
        self._data.setdefault(name, [])

    def add_edge(self, cause: str, effect: str, coef: float = 1.0) -> bool:
        ok = self.dag.add_edge(cause, effect)
        if ok:
            params = self._engine._params.setdefault(
                effect, {"mu": 0.0, "sigma": 1.0, "weights": {}}
            )
            params["weights"][cause] = float(coef)
        return ok

    # ------------------------------------------------------------------
    # Evidence and intervention
    # ------------------------------------------------------------------

    def observe(self, variable: str, value: float) -> None:
        if variable not in self.dag.nodes:
            self.add_variable(variable)
        self._observations[variable] = float(value)
        self._data.setdefault(variable, []).append(float(value))

    def clear_observations(self) -> None:
        self._observations.clear()

    def intervene(
        self,
        variable: str,
        value: float,
        target: Optional[str] = None,
        n_samples: int = 512,
    ) -> Dict[str, Any]:
        """Compute ``p(target | do(variable = value), obs)``.

        If ``target`` is omitted, returns the expected value of every
        descendant of the intervened variable.
        """
        interventions = {variable: float(value)}
        if target is not None:
            return {
                target: self._engine.truncated_product(
                    target, interventions, self._observations, n_samples
                )
            }
        descendants = self.dag.descendants(variable)
        results: Dict[str, Any] = {}
        for d in descendants:
            results[d] = self._engine.truncated_product(
                d, interventions, self._observations, n_samples
            )
        return results

    def infer(self, target_var: str, n_samples: int = 512) -> Dict[str, float]:
        """Compute ``p(target_var | observations)`` with no interventions."""
        if target_var in self._observations:
            val = self._observations[target_var]
            return {"mean": val, "std": 0.0, "p05": val, "p95": val}
        return self._engine.truncated_product(target_var, {}, self._observations, n_samples)

    # ------------------------------------------------------------------
    # Adjustment helpers
    # ------------------------------------------------------------------

    def backdoor_adjustment(self, cause: str, effect: str) -> Set[str]:
        """Return a minimal backdoor adjustment set via parent pruning."""
        if cause not in self.dag.nodes or effect not in self.dag.nodes:
            return set()
        # Parents of cause typically form a valid backdoor set, excluding
        # descendants of cause to satisfy the backdoor criterion.
        parents = set(self.dag.parents(cause))
        desc = self.dag.descendants(cause)
        return parents - desc - {effect}

    def frontdoor_adjustment(self, cause: str, effect: str) -> Set[str]:
        """Return a candidate front-door adjustment set (mediator)."""
        if cause not in self.dag.nodes or effect not in self.dag.nodes:
            return set()
        mediators = set(self.dag.children(cause)) & set(self.dag.parents(effect))
        return mediators

    # ------------------------------------------------------------------
    # Structure learning (simplified PC algorithm)
    # ------------------------------------------------------------------

    @staticmethod
    def _partial_correlation(
        x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> float:
        if z.shape[0] == 0 or z.shape[1] == 0:
            if len(x) < 3 or len(y) < 3:
                return 0.0
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                return 0.0
            return float(np.clip(np.corrcoef(x, y)[0, 1], -1.0, 1.0))
        # Residualise x and y against z via least squares.
        z_aug = np.hstack([np.ones((z.shape[0], 1)), z])
        try:
            bx, *_ = np.linalg.lstsq(z_aug, x, rcond=None)
            by, *_ = np.linalg.lstsq(z_aug, y, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0
        rx = x - z_aug @ bx
        ry = y - z_aug @ by
        if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
            return 0.0
        return float(np.clip(np.corrcoef(rx, ry)[0, 1], -1.0, 1.0))

    def discover_structure(
        self,
        data: Dict[str, np.ndarray],
        alpha: float = 0.25,
    ) -> List[Tuple[str, str]]:
        """Simplified PC-style discovery on ``{variable: samples}``."""
        variables = list(data.keys())
        n = min(len(v) for v in data.values()) if variables else 0
        if n < 5 or len(variables) < 2:
            return []
        arrays = {k: np.asarray(v, dtype=float)[:n] for k, v in data.items()}
        edges: List[Tuple[str, str]] = []
        # Use partial correlation with the empty conditioning set, then with
        # each other variable, as a coarse independence test.
        for i, a in enumerate(variables):
            for j, b in enumerate(variables):
                if i == j:
                    continue
                x = arrays[a]
                y = arrays[b]
                empty_cond = np.zeros((n, 0))
                corr0 = self._partial_correlation(x, y, empty_cond)
                if abs(corr0) < alpha:
                    continue
                # Orient via time-lag heuristic: x causes y if x-lag correlates stronger.
                shifted = np.concatenate([[0.0], x[:-1]])
                lag_corr = float(np.corrcoef(shifted, y)[0, 1]) if np.std(shifted) > 0 else 0.0
                if abs(lag_corr) > abs(corr0) * 0.9:
                    edges.append((a, b))
        # Apply to DAG.
        self.dag.edges.clear()
        for n_ in variables:
            self.add_variable(n_)
        for a, b in edges:
            self.add_edge(a, b)
        self._last_structure_size = len(self.dag.edges)
        return list(self.dag.edges)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        return {
            "n_nodes": len(self.dag.nodes),
            "n_edges": len(self.dag.edges),
            "nodes": sorted(self.dag.nodes),
            "edges": [{"cause": c, "effect": e} for (c, e) in sorted(self.dag.edges)],
            "topological_order": self.dag.topological_order(),
            "n_observations": len(self._observations),
            "observations": dict(self._observations),
            "last_structure_size": self._last_structure_size,
        }


__all__ = ["CausalDAG", "InterventionEngine", "DeepCausalEngine"]
