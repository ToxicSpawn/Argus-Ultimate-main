"""
GRAPE: Gradient Ascent Pulse Engineering.

Optimizes the time-dependent control fields needed to implement a target
unitary on a quantum system. The control field is discretized into N time
slices, and we use gradient ascent on the fidelity to find the optimal
amplitudes.

Reference
---------
Khaneja, Reiss, Kehlet, Schulte-Herbrüggen, Glaser, "Optimal control of
coupled spin dynamics: design of NMR pulse sequences by gradient ascent
algorithms," J. Magn. Reson. 172, 296 (2005)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from scipy.linalg import expm

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# GRAPE optimizer
# ═════════════════════════════════════════════════════════════════════════════


class GRAPEOptimizer:
    """
    Optimize a piecewise-constant control field to implement a target unitary.

    Parameters
    ----------
    target_unitary : np.ndarray
        Target unitary U_target (d x d).
    n_time_slices : int
        Number of discrete time slices.
    H0 : np.ndarray, optional
        Drift Hamiltonian (always on). Defaults to zero.
    H_controls : List[np.ndarray], optional
        Control Hamiltonians. Defaults to [σx, σy, σz] for single qubit.
    duration : float
        Total pulse duration.
    """

    def __init__(
        self,
        target_unitary: np.ndarray,
        n_time_slices: int = 50,
        H0: Optional[np.ndarray] = None,
        H_controls: Optional[List[np.ndarray]] = None,
        duration: float = 1.0,
    ) -> None:
        self.U_target = np.asarray(target_unitary, dtype=np.complex128)
        d = self.U_target.shape[0]
        self.dim = d
        self.n_slices = int(n_time_slices)
        self.dt = float(duration) / self.n_slices

        if H0 is None:
            H0 = np.zeros((d, d), dtype=np.complex128)
        self.H0 = H0

        if H_controls is None:
            # Default Pauli controls for single qubit
            X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
            Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
            Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
            H_controls = [X, Y, Z][:d]
        self.H_controls = H_controls
        self.n_controls = len(H_controls)

        # Initialize random control amplitudes
        rng = np.random.default_rng(42)
        self.controls = rng.normal(0, 0.1, (self.n_controls, self.n_slices))

    def fidelity(self, controls: np.ndarray) -> float:
        """Compute the fidelity F = |Tr(U_target† U)|² / d²."""
        U = self._propagate(controls)
        overlap = np.trace(self.U_target.conj().T @ U)
        return float(np.abs(overlap) ** 2 / (self.dim ** 2))

    def _propagate(self, controls: np.ndarray) -> np.ndarray:
        """Propagate the system through all time slices."""
        U_total = np.eye(self.dim, dtype=np.complex128)
        for k in range(self.n_slices):
            H_k = self.H0.copy()
            for c in range(self.n_controls):
                H_k = H_k + controls[c, k] * self.H_controls[c]
            U_k = expm(-1j * H_k * self.dt)
            U_total = U_k @ U_total
        return U_total

    def optimize(
        self,
        *,
        n_iter: int = 100,
        learning_rate: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Run gradient ascent on the control fields.

        Uses finite-difference gradients for simplicity (the full GRAPE
        algorithm uses analytical gradients via the propagator).
        """
        t0 = time.perf_counter()
        controls = self.controls.copy()
        fidelities = []
        eps = 1e-3

        for it in range(n_iter):
            grad = np.zeros_like(controls)
            for c in range(self.n_controls):
                for k in range(self.n_slices):
                    controls[c, k] += eps
                    f_plus = self.fidelity(controls)
                    controls[c, k] -= 2 * eps
                    f_minus = self.fidelity(controls)
                    controls[c, k] += eps
                    grad[c, k] = (f_plus - f_minus) / (2 * eps)
            controls = controls + learning_rate * grad
            fidelities.append(self.fidelity(controls))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.controls = controls

        return {
            "final_fidelity": float(fidelities[-1] if fidelities else 0.0),
            "fidelity_history": fidelities,
            "controls": controls.tolist(),
            "n_iter": n_iter,
            "method": "grape_finite_diff",
            "elapsed_ms": elapsed_ms,
        }
