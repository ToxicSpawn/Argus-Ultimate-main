"""Simulated quantum annealing for portfolio optimisation.

This module treats portfolio optimisation as a QUBO (Quadratic
Unconstrained Binary Optimisation) problem and solves it via a
classical simulated annealer with a Metropolis-Hastings acceptance
rule and a geometric cooling schedule. While we do not have access
to a real quantum annealer, the formulation matches what a D-Wave
style device would expect, and soft constraints are handled with
Lagrange multipliers.

The main entry point is :class:`QuantumAnnealer` which accepts a
list of candidate assets, expected returns, and a covariance matrix,
and returns a binary selection vector indicating which assets to
include in the portfolio.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QUBO representation
# ---------------------------------------------------------------------------


@dataclass
class QUBO:
    """Quadratic Unconstrained Binary Optimisation problem.

    Energy is defined as::

        E(x) = x^T Q x + constant

    where ``x`` is a binary vector and ``Q`` is a symmetric (or upper
    triangular) matrix. Linear terms are encoded on the diagonal of
    ``Q``.
    """

    size: int
    Q: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    constant: float = 0.0

    def __post_init__(self) -> None:
        if self.Q.size == 0:
            self.Q = np.zeros((self.size, self.size), dtype=float)

    def set_linear(self, i: int, value: float) -> None:
        self.Q[i, i] = float(value)

    def add_linear(self, i: int, value: float) -> None:
        self.Q[i, i] += float(value)

    def set_quadratic(self, i: int, j: int, value: float) -> None:
        self.Q[i, j] = float(value)
        if i != j:
            self.Q[j, i] = float(value)

    def add_quadratic(self, i: int, j: int, value: float) -> None:
        self.Q[i, j] += float(value)
        if i != j:
            self.Q[j, i] += float(value)

    def energy(self, x: np.ndarray) -> float:
        x = x.astype(float)
        return float(x @ self.Q @ x + self.constant)

    def delta_energy(self, x: np.ndarray, bit: int) -> float:
        """Energy change from flipping a single bit."""
        # E(x') - E(x) = (1 - 2*x[bit]) * (Q[bit,bit] + 2 * sum_j Q[bit,j] * x[j] for j != bit)
        flip_sign = 1.0 - 2.0 * float(x[bit])
        q_diag = float(self.Q[bit, bit])
        q_row = self.Q[bit].copy()
        q_row[bit] = 0.0
        interaction = 2.0 * float(q_row @ x)
        return flip_sign * (q_diag + interaction)


# ---------------------------------------------------------------------------
# Simulated annealing solver
# ---------------------------------------------------------------------------


class SimulatedAnnealingSolver:
    """Generic simulated annealing solver for QUBO problems.

    Parameters:
        n_iterations: Total number of Metropolis steps.
        start_temp: Initial temperature.
        end_temp: Final temperature.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_iterations: int = 2000,
        start_temp: float = 1.0,
        end_temp: float = 1e-3,
        seed: Optional[int] = None,
    ) -> None:
        self.n_iterations = int(n_iterations)
        self.start_temp = float(start_temp)
        self.end_temp = float(max(end_temp, 1e-9))
        self._rng = random.Random(seed)
        self._history: List[float] = []
        self._best_energy: float = math.inf
        self._best_state: Optional[np.ndarray] = None

    def _temperature(self, step: int) -> float:
        if self.n_iterations <= 1:
            return self.end_temp
        frac = step / (self.n_iterations - 1)
        ratio = self.end_temp / self.start_temp
        return self.start_temp * (ratio ** frac)

    def solve(self, qubo: QUBO, init_state: Optional[np.ndarray] = None) -> np.ndarray:
        """Run simulated annealing and return the best state found."""
        n = qubo.size
        if n == 0:
            return np.zeros(0, dtype=int)
        if init_state is None:
            state = np.array(
                [1 if self._rng.random() < 0.5 else 0 for _ in range(n)],
                dtype=int,
            )
        else:
            state = init_state.astype(int).copy()

        current_energy = qubo.energy(state)
        self._best_energy = current_energy
        self._best_state = state.copy()
        self._history = [current_energy]

        for step in range(self.n_iterations):
            temp = self._temperature(step)
            bit = self._rng.randrange(n)
            de = qubo.delta_energy(state, bit)
            accept = False
            if de <= 0.0:
                accept = True
            else:
                prob = math.exp(-de / max(temp, 1e-12))
                if self._rng.random() < prob:
                    accept = True
            if accept:
                state[bit] = 1 - state[bit]
                current_energy += de
                if current_energy < self._best_energy:
                    self._best_energy = current_energy
                    self._best_state = state.copy()
            self._history.append(current_energy)

        assert self._best_state is not None
        return self._best_state

    def snapshot(self) -> Dict[str, Any]:
        return {
            "n_iterations": self.n_iterations,
            "start_temp": self.start_temp,
            "end_temp": self.end_temp,
            "best_energy": float(self._best_energy) if self._best_state is not None else None,
            "history_len": len(self._history),
        }


# ---------------------------------------------------------------------------
# Portfolio annealer
# ---------------------------------------------------------------------------


class QuantumAnnealer:
    """Simulated quantum annealer for portfolio selection.

    The objective combines expected return maximisation with a
    quadratic risk penalty and an optional Lagrangian constraint on
    the number of assets selected::

        E(x) = -lambda_ret * mu^T x
             +  lambda_risk * x^T Sigma x
             +  lambda_card * (sum(x) - k_target)^2
    """

    def __init__(
        self,
        lambda_ret: float = 1.0,
        lambda_risk: float = 1.0,
        lambda_card: float = 0.5,
        target_count: Optional[int] = None,
        n_iterations: int = 2000,
        start_temp: float = 1.0,
        end_temp: float = 1e-3,
        seed: Optional[int] = None,
    ) -> None:
        self.lambda_ret = float(lambda_ret)
        self.lambda_risk = float(lambda_risk)
        self.lambda_card = float(lambda_card)
        self.target_count = target_count
        self._assets: List[str] = []
        self._returns: Optional[np.ndarray] = None
        self._cov: Optional[np.ndarray] = None
        self._qubo: Optional[QUBO] = None
        self._solver = SimulatedAnnealingSolver(
            n_iterations=n_iterations,
            start_temp=start_temp,
            end_temp=end_temp,
            seed=seed,
        )
        self._best_state: Optional[np.ndarray] = None

    def set_portfolio_problem(
        self,
        assets: List[str],
        returns: np.ndarray,
        covariance: np.ndarray,
    ) -> None:
        """Construct the QUBO from returns/covariance data."""
        assets = list(assets)
        n = len(assets)
        returns = np.asarray(returns, dtype=float).flatten()
        cov = np.asarray(covariance, dtype=float)
        if returns.shape[0] != n:
            raise ValueError("returns length mismatch vs assets")
        if cov.shape != (n, n):
            raise ValueError("covariance must be (n, n)")

        self._assets = assets
        self._returns = returns
        self._cov = cov

        qubo = QUBO(size=n)
        for i in range(n):
            qubo.add_linear(i, -self.lambda_ret * float(returns[i]))
            for j in range(n):
                qubo.add_quadratic(i, j, self.lambda_risk * float(cov[i, j]) * 0.5)

        # Cardinality constraint via Lagrange multiplier.
        if self.target_count is not None and self.lambda_card > 0.0:
            k = float(self.target_count)
            # (sum(x) - k)^2 = sum_i x_i + 2 sum_{i<j} x_i x_j - 2k sum_i x_i + k^2
            for i in range(n):
                qubo.add_linear(i, self.lambda_card * (1.0 - 2.0 * k))
                for j in range(i + 1, n):
                    qubo.add_quadratic(i, j, self.lambda_card * 2.0 * 0.5)
            qubo.constant += self.lambda_card * k * k

        self._qubo = qubo

    def solve(self, n_iterations: Optional[int] = None) -> np.ndarray:
        """Run the annealer and return the best binary selection."""
        if self._qubo is None:
            raise RuntimeError("call set_portfolio_problem before solve()")
        if n_iterations is not None:
            self._solver.n_iterations = int(n_iterations)
        self._best_state = self._solver.solve(self._qubo)
        return self._best_state

    def get_best_allocation(self) -> Dict[str, Any]:
        """Return the currently stored best allocation."""
        if self._best_state is None:
            return {"assets": [], "weights": {}, "energy": None}
        selected = [a for a, b in zip(self._assets, self._best_state) if b == 1]
        k = max(len(selected), 1)
        weights = {a: 1.0 / k for a in selected}
        energy = self._qubo.energy(self._best_state) if self._qubo is not None else None
        expected_return = None
        if self._returns is not None and selected:
            mask = np.array(self._best_state, dtype=float)
            expected_return = float(np.sum(self._returns * mask) / k)
        return {
            "assets": selected,
            "weights": weights,
            "energy": float(energy) if energy is not None else None,
            "expected_return": expected_return,
            "n_selected": len(selected),
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "n_assets": len(self._assets),
            "assets": list(self._assets),
            "has_problem": self._qubo is not None,
            "lambda_ret": self.lambda_ret,
            "lambda_risk": self.lambda_risk,
            "lambda_card": self.lambda_card,
            "target_count": self.target_count,
            "solver": self._solver.snapshot(),
            "best_allocation": self.get_best_allocation() if self._best_state is not None else None,
        }


__all__ = ["QUBO", "SimulatedAnnealingSolver", "QuantumAnnealer"]
