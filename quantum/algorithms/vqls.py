"""
Variational Quantum Linear Solver (VQLS).

VQLS is an NISQ-friendly alternative to HHL for solving Ax = b. It
variationally prepares |x⟩ = argmin ||A|x⟩ - |b⟩||² via a parameterized
circuit + classical outer optimizer.

Advantages over HHL:
- Shorter circuits (no QPE, no controlled-U)
- More NISQ-friendly (works with depolarizing noise)
- Can handle ill-conditioned systems more gracefully

Disadvantages:
- Convergence is not guaranteed
- Classical outer loop is needed

Reference
---------
Bravo-Prieto, LaRose, Cerezo, Subasi, Cincio, Coles,
"Variational Quantum Linear Solver," Quantum 7, 1188 (2023)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from quantum_simulator import QuantumCircuit, _simulate_statevector

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Variational Quantum Linear Solver
# ═════════════════════════════════════════════════════════════════════════════


class VQLSSolver:
    """
    Variational Quantum Linear Solver.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the state register (log2 of matrix size).
    n_layers : int
        Ansatz depth.
    """

    def __init__(self, n_qubits: int, n_layers: int = 3) -> None:
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.n_params = self.n_qubits * self.n_layers

    def solve(
        self,
        A: np.ndarray,
        b: np.ndarray,
        *,
        max_iter: int = 100,
        seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """
        Solve Ax = b variationally.

        Parameters
        ----------
        A : np.ndarray
            (d x d) Hermitian matrix (d = 2^n_qubits).
        b : np.ndarray
            (d,) right-hand side vector (will be normalized).

        Returns
        -------
        Dict[str, Any]
            ``{"x", "fidelity", "final_cost", "convergence", "method"}``
        """
        from scipy.optimize import minimize as sp_minimize

        t0 = time.perf_counter()
        d = 1 << self.n_qubits
        A = np.asarray(A, dtype=np.complex128)
        b = np.asarray(b, dtype=np.complex128).ravel()

        # Normalize b
        b_norm = float(np.linalg.norm(b))
        if b_norm < 1e-12:
            return {
                "x": np.zeros(d),
                "fidelity": 0.0,
                "final_cost": 0.0,
                "method": "vqls_trivial",
                "elapsed_ms": 0.0,
            }
        b_normalized = b / b_norm

        rng = np.random.default_rng(seed)
        convergence: List[float] = []

        def cost_fn(params):
            qc = self._build_ansatz(params)
            psi = _simulate_statevector(qc)
            # Local cost function: || A|x⟩ - c|b⟩||² where c = ⟨b|A|x⟩
            A_psi = A @ psi
            c = np.vdot(b_normalized, A_psi)
            residual = A_psi - c * b_normalized
            cost = float(np.real(np.vdot(residual, residual)))
            convergence.append(cost)
            return cost

        # Multiple random restarts
        best_cost = float("inf")
        best_params: Optional[np.ndarray] = None
        for restart in range(3):
            x0 = rng.uniform(-np.pi, np.pi, self.n_params)
            opt = sp_minimize(
                cost_fn,
                x0,
                method="COBYLA",
                options={"maxiter": max_iter, "rhobeg": 0.3},
            )
            if opt.fun < best_cost:
                best_cost = float(opt.fun)
                best_params = np.asarray(opt.x, dtype=float)

        if best_params is None:
            best_params = np.zeros(self.n_params)

        # Build final solution
        qc = self._build_ansatz(best_params)
        psi_final = _simulate_statevector(qc)

        # Rescale to match classical solution
        try:
            x_classical = np.linalg.solve(A, b)
            norm_class = float(np.linalg.norm(x_classical))
            if norm_class > 1e-12:
                x_vqls = psi_final * norm_class
                fid = float(abs(np.vdot(psi_final, x_classical / norm_class)) ** 2)
            else:
                x_vqls = psi_final
                fid = 0.0
        except np.linalg.LinAlgError:
            x_vqls = psi_final
            fid = 0.0

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "x": x_vqls,
            "fidelity": fid,
            "final_cost": best_cost,
            "convergence": convergence[-50:],
            "method": "vqls_in_repo",
            "elapsed_ms": elapsed_ms,
            "n_params": self.n_params,
        }

    def _build_ansatz(self, params: np.ndarray) -> QuantumCircuit:
        """Hardware-efficient ansatz: alternating RY + CNOT-ring layers."""
        qc = QuantumCircuit(self.n_qubits)
        idx = 0
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                qc.ry(float(params[idx]), q)
                idx += 1
            # CNOT ring (only if n_qubits >= 2)
            if self.n_qubits >= 2:
                for q in range(self.n_qubits):
                    qc.cnot(q, (q + 1) % self.n_qubits)
        return qc
