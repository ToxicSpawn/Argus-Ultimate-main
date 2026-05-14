"""
Variational Quantum Eigensolver (VQE) for Correlation Matrix Eigendecomposition.

Implements a parameterized quantum circuit ansatz and classical optimizer loop
to find the minimum eigenvalue and eigenvector of a Hamiltonian derived from
a correlation matrix.  The minimum eigenvector corresponds to the minimum-
variance portfolio direction.

This is a classical simulation of VQE using statevector evolution.  No quantum
hardware is used.  The value is in exploring the VQE algorithm structure for
portfolio optimization -- when fault-tolerant quantum hardware arrives, the
same ansatz could be executed on real qubits.

Typical usage::

    from quantum.hybrid.variational import VariationalQuantumEigensolver

    vqe = VariationalQuantumEigensolver(n_qubits=4)
    result = vqe.solve(correlation_matrix)
    weights = vqe.portfolio_weights(correlation_matrix)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize as sp_minimize

logger = logging.getLogger(__name__)


class VariationalQuantumEigensolver:
    """
    VQE for finding the minimum eigenvalue of a Hamiltonian derived from
    a correlation matrix.

    The ansatz is a hardware-efficient circuit with RY rotations and
    CNOT entangling layers.  The cost function is the expectation value
    <psi(theta)|H|psi(theta)> computed via statevector simulation.

    Attributes:
        n_qubits: Number of qubits in the ansatz.
        n_layers: Number of RY+CNOT layers.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        seed: Optional[int] = None,
    ) -> None:
        if n_qubits < 1 or n_qubits > 14:
            raise ValueError(f"n_qubits must be in [1, 14], got {n_qubits}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dim = 2 ** n_qubits
        self._rng = np.random.RandomState(seed)

        # Number of variational parameters: n_layers * n_qubits (one RY angle per qubit per layer)
        self.n_params = n_layers * n_qubits

        self._last_result: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Hamiltonian construction
    # ------------------------------------------------------------------

    def build_hamiltonian(self, correlation_matrix: np.ndarray) -> np.ndarray:
        """Convert a correlation matrix to a qubit Hamiltonian.

        Strategy: embed the correlation matrix into the 2^n Hilbert space.
        For an n x n correlation matrix with n <= 2^n_qubits, we map each
        asset to a computational basis state and construct:

            H = sum_{i,j} C_{i,j} |i><j|

        padded with zeros for unused basis states.  This preserves the
        eigenvalues and eigenvectors of C within the relevant subspace.

        Args:
            correlation_matrix: Symmetric positive semi-definite matrix.

        Returns:
            Hermitian matrix of size (2^n_qubits, 2^n_qubits).
        """
        C = np.asarray(correlation_matrix, dtype=np.float64)
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValueError(f"Correlation matrix must be square, got {C.shape}")

        n = C.shape[0]
        if n > self.dim:
            raise ValueError(
                f"Correlation matrix size {n} exceeds Hilbert space dim {self.dim}. "
                f"Need n_qubits >= {int(np.ceil(np.log2(max(n, 2))))}"
            )

        # Make symmetric
        C = (C + C.T) / 2.0

        # Embed into 2^n_qubits space
        H = np.zeros((self.dim, self.dim), dtype=np.complex128)
        H[:n, :n] = C.astype(np.complex128)

        return H

    # ------------------------------------------------------------------
    # Ansatz circuit (statevector simulation)
    # ------------------------------------------------------------------

    def ansatz(self, params: np.ndarray) -> np.ndarray:
        """Build the parameterized quantum state |psi(theta)>.

        Hardware-efficient ansatz:
        - Layer structure: RY(theta_i) on each qubit, then CNOT cascade
        - Repeated n_layers times
        - Starts from |0...0>

        Args:
            params: 1D array of length n_params.

        Returns:
            Complex statevector of length 2^n_qubits.
        """
        params = np.asarray(params, dtype=np.float64).ravel()
        if len(params) != self.n_params:
            raise ValueError(
                f"Expected {self.n_params} parameters, got {len(params)}"
            )

        # Start with |0...0>
        state = np.zeros(self.dim, dtype=np.complex128)
        state[0] = 1.0

        idx = 0
        for layer in range(self.n_layers):
            # RY rotations
            for q in range(self.n_qubits):
                state = self._apply_ry(state, q, params[idx])
                idx += 1

            # CNOT cascade (nearest-neighbour ring)
            for q in range(self.n_qubits - 1):
                state = self._apply_cnot(state, q, q + 1)

            # Close the ring for >= 3 qubits
            if self.n_qubits >= 3:
                state = self._apply_cnot(state, self.n_qubits - 1, 0)

        return state

    def _apply_ry(self, state: np.ndarray, qubit: int, angle: float) -> np.ndarray:
        """Apply RY(angle) gate to a qubit in the statevector."""
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        new_state = state.copy()
        mask = 1 << (self.n_qubits - 1 - qubit)

        for i in range(self.dim):
            if i & mask:
                continue
            j = i | mask
            a0, a1 = state[i], state[j]
            new_state[i] = c * a0 - s * a1
            new_state[j] = s * a0 + c * a1

        return new_state

    def _apply_cnot(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
        """Apply CNOT gate."""
        c_mask = 1 << (self.n_qubits - 1 - control)
        t_mask = 1 << (self.n_qubits - 1 - target)
        new_state = state.copy()

        for i in range(self.dim):
            if i & c_mask:
                j = i ^ t_mask
                new_state[i] = state[j]
                new_state[j] = state[i]

        return new_state

    # ------------------------------------------------------------------
    # Cost function
    # ------------------------------------------------------------------

    def cost_function(self, params: np.ndarray, hamiltonian: np.ndarray) -> float:
        """Compute <psi(theta)|H|psi(theta)>.

        This is the energy expectation value that VQE minimizes.

        Args:
            params: Variational parameters.
            hamiltonian: The Hamiltonian matrix.

        Returns:
            Real-valued expectation (the energy).
        """
        state = self.ansatz(params)
        energy = np.real(np.conj(state) @ hamiltonian @ state)
        return float(energy)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        correlation_matrix: np.ndarray,
        n_iter: int = 100,
        method: str = "COBYLA",
        n_restarts: int = 3,
    ) -> Dict[str, Any]:
        """Find the minimum eigenvalue/eigenvector of the correlation matrix via VQE.

        Uses multiple random restarts to mitigate local minima.

        Args:
            correlation_matrix: Symmetric matrix to decompose.
            n_iter: Maximum iterations per optimization run.
            method: Scipy optimizer method.
            n_restarts: Number of random restarts.

        Returns:
            Dict with keys: eigenvalue, eigenvector, params, iterations,
            converged, classical_eigenvalue, time_s.
        """
        t0 = time.monotonic()
        H = self.build_hamiltonian(correlation_matrix)

        # Classical reference (for comparison)
        classical_eigenvalues = np.linalg.eigvalsh(correlation_matrix)
        min_classical = float(classical_eigenvalues[0])

        best_energy = float("inf")
        best_params = None
        best_result = None
        total_iters = 0

        for restart in range(max(1, n_restarts)):
            # Random initial parameters
            init_params = self._rng.uniform(0, 2 * np.pi, self.n_params)

            try:
                opt = sp_minimize(
                    self.cost_function,
                    init_params,
                    args=(H,),
                    method=method,
                    options={"maxiter": n_iter, "rhobeg": 0.5},
                )

                total_iters += opt.nfev
                if opt.fun < best_energy:
                    best_energy = opt.fun
                    best_params = opt.x
                    best_result = opt
            except Exception as e:
                logger.debug("VQE restart %d failed: %s", restart, e)
                continue

        if best_params is None:
            # All restarts failed, return classical result
            eigvals, eigvecs = np.linalg.eigh(correlation_matrix)
            elapsed = time.monotonic() - t0
            result = {
                "eigenvalue": float(eigvals[0]),
                "eigenvector": eigvecs[:, 0].tolist(),
                "params": np.zeros(self.n_params).tolist(),
                "iterations": 0,
                "converged": False,
                "classical_eigenvalue": min_classical,
                "error": abs(float(eigvals[0]) - min_classical),
                "time_s": round(elapsed, 4),
                "method": "classical_fallback",
            }
            self._last_result = result
            return result

        # Extract eigenvector from optimized state
        opt_state = self.ansatz(best_params)
        n_assets = correlation_matrix.shape[0]
        # The eigenvector in the computational basis subspace
        eigvec = np.real(opt_state[:n_assets])
        norm = np.linalg.norm(eigvec)
        if norm > 1e-12:
            eigvec = eigvec / norm
        else:
            eigvec = np.ones(n_assets) / np.sqrt(n_assets)

        converged = abs(best_energy - min_classical) < 0.1 * abs(min_classical) + 1e-6
        elapsed = time.monotonic() - t0

        result = {
            "eigenvalue": round(float(best_energy), 8),
            "eigenvector": eigvec.tolist(),
            "params": best_params.tolist(),
            "iterations": total_iters,
            "converged": bool(converged),
            "classical_eigenvalue": round(min_classical, 8),
            "error": round(abs(float(best_energy) - min_classical), 8),
            "time_s": round(elapsed, 4),
            "method": "vqe_ry_cnot_ansatz",
        }

        self._last_result = result
        logger.info(
            "VQE solve: eigenvalue=%.6f (classical=%.6f), converged=%s, "
            "iters=%d, time=%.3fs",
            best_energy, min_classical, converged, total_iters, elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Portfolio weights
    # ------------------------------------------------------------------

    def portfolio_weights(self, correlation_matrix: np.ndarray) -> np.ndarray:
        """Compute minimum-variance portfolio weights via VQE.

        The minimum eigenvector of the correlation (covariance) matrix
        defines the direction of minimum variance.  We normalize it to
        sum to 1 with non-negative components (long-only constraint).

        Args:
            correlation_matrix: Covariance or correlation matrix.

        Returns:
            Array of portfolio weights summing to 1.
        """
        result = self.solve(correlation_matrix)
        eigvec = np.array(result["eigenvector"])

        # Make non-negative (long-only)
        weights = np.abs(eigvec)
        total = weights.sum()
        if total > 1e-12:
            weights = weights / total
        else:
            n = len(eigvec)
            weights = np.ones(n) / n

        return weights

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the VQE configuration and last result."""
        info = {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_params": self.n_params,
            "hilbert_dim": self.dim,
            "method": "classical_vqe_simulation",
        }
        if self._last_result is not None:
            info["last_eigenvalue"] = self._last_result["eigenvalue"]
            info["last_converged"] = self._last_result["converged"]
            info["last_error"] = self._last_result["error"]
        return info
