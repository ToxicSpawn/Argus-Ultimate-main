"""
Quantum Natural Gradient (QNG) optimizer.

QNG performs gradient descent in the parameter space of variational quantum
circuits using the Fubini-Study metric tensor (the quantum analogue of the
Fisher information matrix). It typically converges in 5-10x fewer iterations
than vanilla gradient descent.

Reference
---------
Stokes, Izaac, Killoran, Carleo, "Quantum Natural Gradient,"
Quantum 4, 269 (2020). arXiv:1909.02108

Update rule
-----------
    θ_{k+1} = θ_k - η · g(θ_k)^{-1} · ∇L(θ_k)

where g_ij = Re⟨∂_iψ|∂_jψ⟩ - ⟨∂_iψ|ψ⟩⟨ψ|∂_jψ⟩ is the Fubini-Study metric.

For block-diagonal QNG (the practical variant), we approximate g as block-
diagonal with one block per layer of parameters. This is what Stokes et al.
recommend and is much cheaper than the full metric.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from quantum_simulator import (
    QuantumCircuit,
    expval,
    gradient,
    pauli_z_observable,
    simulate,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════


def quantum_natural_gradient(
    circuit_builder: Callable[[np.ndarray], QuantumCircuit],
    initial_params: np.ndarray,
    observable: Callable[[Dict[str, int]], float],
    *,
    learning_rate: float = 0.1,
    n_steps: int = 50,
    shots: int = 2048,
    seed: Optional[int] = 42,
    metric_method: str = "block_diagonal",
    regularization: float = 1e-3,
) -> Dict[str, Any]:
    """
    Quantum Natural Gradient optimization loop.

    Parameters
    ----------
    circuit_builder : Callable[[np.ndarray], QuantumCircuit]
        Builds a parameterized circuit.
    initial_params : np.ndarray
        Starting parameter vector.
    observable : Callable[[Dict[str, int]], float]
        Energy / cost observable.
    learning_rate : float
        Step size η.
    n_steps : int
        Number of QNG iterations.
    shots : int
        Shots per circuit evaluation.
    metric_method : str
        ``"block_diagonal"`` (cheap, recommended) or ``"diagonal"`` (cheapest).
    regularization : float
        Tikhonov regularization added to the metric tensor before inversion.

    Returns
    -------
    Dict[str, Any]
        ``{"final_params", "params_history", "energies", "metric_history",
          "method", "elapsed_ms"}``
    """
    t0 = time.perf_counter()
    params = np.asarray(initial_params, dtype=float).copy()
    n_params = params.size

    energies: List[float] = []
    params_history: List[np.ndarray] = [params.copy()]
    metric_history: List[float] = []  # store determinants

    for step in range(n_steps):
        # Compute gradient via parameter-shift
        grad = gradient(
            circuit_builder,
            params,
            observable,
            shots=shots,
            seed=seed,
            method="parameter_shift",
        )

        # Compute Fubini-Study metric tensor
        if metric_method == "diagonal":
            g = _diagonal_metric(circuit_builder, params, shots=shots, seed=seed)
        else:
            g = _block_diagonal_metric(
                circuit_builder, params, shots=shots, seed=seed,
            )

        # Regularize and invert
        g_reg = g + regularization * np.eye(n_params)
        try:
            g_inv = np.linalg.inv(g_reg)
        except np.linalg.LinAlgError:
            g_inv = np.linalg.pinv(g_reg)

        # Natural gradient step
        natural_grad = g_inv @ grad
        params = params - learning_rate * natural_grad

        # Track energy
        qc = circuit_builder(params)
        e = expval(qc, observable, shots=shots, seed=seed)
        energies.append(float(e))
        params_history.append(params.copy())
        metric_history.append(float(np.linalg.det(g_reg)))

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "final_params": params,
        "params_history": params_history,
        "energies": energies,
        "metric_history": metric_history,
        "method": "quantum_natural_gradient",
        "metric_method": metric_method,
        "elapsed_ms": elapsed_ms,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Metric tensor estimators
# ═════════════════════════════════════════════════════════════════════════════


def _diagonal_metric(
    circuit_builder: Callable[[np.ndarray], QuantumCircuit],
    params: np.ndarray,
    *,
    shots: int,
    seed: Optional[int],
) -> np.ndarray:
    """
    Diagonal approximation of the Fubini-Study metric.

    g_ii = (1 - ⟨ψ(θ)|ψ(θ + π e_i)⟩²) / 4

    Each diagonal entry requires one extra circuit run.
    """
    n = params.size
    g = np.zeros((n, n), dtype=float)
    base_qc = circuit_builder(params)
    base_state = _statevector_from_circuit(base_qc)

    for i in range(n):
        params_shift = params.copy()
        params_shift[i] += np.pi
        shifted_qc = circuit_builder(params_shift)
        shifted_state = _statevector_from_circuit(shifted_qc)
        overlap = abs(np.vdot(base_state, shifted_state)) ** 2
        g[i, i] = (1.0 - overlap) / 4.0

    return g


def _block_diagonal_metric(
    circuit_builder: Callable[[np.ndarray], QuantumCircuit],
    params: np.ndarray,
    *,
    shots: int,
    seed: Optional[int],
) -> np.ndarray:
    """
    Block-diagonal approximation: per-pair off-diagonal entries computed.

    Slower than diagonal but more accurate. For each pair (i, j) we compute
    g_ij from circuit overlaps.
    """
    n = params.size
    g = np.zeros((n, n), dtype=float)
    base_qc = circuit_builder(params)
    base_state = _statevector_from_circuit(base_qc)

    # Diagonal entries
    for i in range(n):
        ps = params.copy()
        ps[i] += np.pi
        ss = _statevector_from_circuit(circuit_builder(ps))
        overlap = abs(np.vdot(base_state, ss)) ** 2
        g[i, i] = (1.0 - overlap) / 4.0

    # Off-diagonal entries (cheap finite-difference approximation)
    eps = 1e-3
    for i in range(n):
        for j in range(i + 1, n):
            ps_pp = params.copy()
            ps_pp[i] += eps
            ps_pp[j] += eps
            s_pp = _statevector_from_circuit(circuit_builder(ps_pp))
            ps_pm = params.copy()
            ps_pm[i] += eps
            ps_pm[j] -= eps
            s_pm = _statevector_from_circuit(circuit_builder(ps_pm))
            ps_mp = params.copy()
            ps_mp[i] -= eps
            ps_mp[j] += eps
            s_mp = _statevector_from_circuit(circuit_builder(ps_mp))
            ps_mm = params.copy()
            ps_mm[i] -= eps
            ps_mm[j] -= eps
            s_mm = _statevector_from_circuit(circuit_builder(ps_mm))
            # Numerical second-order derivative
            d2 = (
                abs(np.vdot(s_pp, s_mm)) ** 2
                - abs(np.vdot(s_pm, s_mp)) ** 2
            ) / (4 * eps * eps)
            g[i, j] = float(d2)
            g[j, i] = float(d2)

    return g


def _statevector_from_circuit(qc: QuantumCircuit) -> np.ndarray:
    """Extract the statevector from a circuit (uses simulator's internal helper)."""
    from quantum_simulator import _simulate_statevector
    return _simulate_statevector(qc)
