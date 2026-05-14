"""
Quantum Boltzmann Machine with Transverse-Field Ising Hamiltonian.

A QBM is a generative model where the energy function is a quantum
Hamiltonian and the generated samples come from the thermal (Gibbs)
state of that Hamiltonian.

Reference
---------
Amin, Andriyash, Rolfe, Kulchytskyy, Melko,
"Quantum Boltzmann Machine," PRX 8, 021050 (2018)

This is a new implementation using the TFI Hamiltonian — complementary
to the existing ``quantum_boltzmann.py`` (which uses a different design).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class QuantumBoltzmannMachineTFI:
    """
    Transverse-field Ising QBM.

    Energy: H = Σ_i h_i Z_i + Σ_{i<j} J_{ij} Z_i Z_j + Γ Σ_i X_i

    Parameters h, J, Γ are learned from data via gradient ascent on the
    log-likelihood.
    """

    def __init__(
        self,
        n_qubits: int,
        *,
        beta: float = 1.0,
        gamma: float = 0.5,
        seed: Optional[int] = 42,
    ) -> None:
        if n_qubits < 1 or n_qubits > 8:
            raise ValueError(f"n_qubits must be 1..8, got {n_qubits}")
        self.n_qubits = int(n_qubits)
        self.beta = float(beta)
        self.gamma = float(gamma)
        rng = np.random.default_rng(seed)
        self.h = rng.uniform(-0.3, 0.3, n_qubits)
        self.J = np.zeros((n_qubits, n_qubits))
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                self.J[i, j] = rng.uniform(-0.3, 0.3)

    def hamiltonian(self) -> np.ndarray:
        """Build the Hamiltonian as a dense matrix."""
        d = 1 << self.n_qubits
        H = np.zeros((d, d), dtype=np.complex128)
        I = np.eye(2, dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        def kron_at(op: np.ndarray, qubit: int) -> np.ndarray:
            result = np.array([[1]], dtype=np.complex128)
            for q in range(self.n_qubits):
                result = np.kron(result, op if q == qubit else I)
            return result

        def kron_at2(op1: np.ndarray, q1: int, op2: np.ndarray, q2: int) -> np.ndarray:
            result = np.array([[1]], dtype=np.complex128)
            for q in range(self.n_qubits):
                if q == q1:
                    result = np.kron(result, op1)
                elif q == q2:
                    result = np.kron(result, op2)
                else:
                    result = np.kron(result, I)
            return result

        for i in range(self.n_qubits):
            H += float(self.h[i]) * kron_at(Z, i)
            H += self.gamma * kron_at(X, i)

        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                if abs(self.J[i, j]) > 1e-12:
                    H += float(self.J[i, j]) * kron_at2(Z, i, Z, j)

        return H

    def gibbs_state(self) -> np.ndarray:
        """Compute ρ = e^(-βH) / Tr(e^(-βH))."""
        H = self.hamiltonian()
        eigvals, eigvecs = np.linalg.eigh(H)
        boltzmann = np.exp(-self.beta * eigvals)
        Z = float(np.sum(boltzmann))
        return eigvecs @ np.diag(boltzmann / Z) @ eigvecs.conj().T

    def sample(self, n_samples: int = 100) -> np.ndarray:
        """Sample bit strings from the Gibbs distribution."""
        rho = self.gibbs_state()
        probs = np.real(np.diag(rho))
        probs = np.clip(probs, 0.0, None)
        probs = probs / max(float(probs.sum()), 1e-12)
        rng = np.random.default_rng()
        samples = rng.choice(len(probs), size=n_samples, p=probs)
        bits = np.zeros((n_samples, self.n_qubits), dtype=int)
        for i, s in enumerate(samples):
            for q in range(self.n_qubits):
                bits[i, q] = (s >> q) & 1
        return bits

    def fit(
        self,
        data: np.ndarray,
        *,
        n_epochs: int = 20,
        learning_rate: float = 0.05,
    ) -> Dict[str, Any]:
        """Train via gradient ascent (classical estimator for small systems)."""
        t0 = time.perf_counter()
        data = np.asarray(data, dtype=int)
        n_samples = data.shape[0]
        history: List[float] = []

        for epoch in range(n_epochs):
            # Data-side expectations
            spins_data = 1.0 - 2.0 * data  # 0 → +1, 1 → -1
            data_Z = spins_data.mean(axis=0)
            data_ZZ = np.zeros((self.n_qubits, self.n_qubits))
            for sample in spins_data:
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        data_ZZ[i, j] += sample[i] * sample[j]
            data_ZZ /= n_samples

            # Model-side expectations
            rho = self.gibbs_state()
            model_Z, model_ZZ = self._model_expectations(rho)

            # Update
            self.h -= learning_rate * (data_Z - model_Z)
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    self.J[i, j] -= learning_rate * (data_ZZ[i, j] - model_ZZ[i, j])

            history.append(float(np.sum(np.abs(data_Z - model_Z))))

        return {
            "history": history,
            "final_h": self.h.tolist(),
            "final_J": self.J.tolist(),
            "method": "qbm_tfi_gradient_ascent",
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }

    def _model_expectations(self, rho: np.ndarray) -> tuple:
        """Compute ⟨Z_i⟩ and ⟨Z_i Z_j⟩ from the Gibbs state."""
        d = 1 << self.n_qubits
        Z_exp = np.zeros(self.n_qubits)
        ZZ_exp = np.zeros((self.n_qubits, self.n_qubits))
        diag = np.real(np.diag(rho))
        for idx in range(d):
            prob = float(diag[idx])
            spins = np.array([1.0 - 2.0 * ((idx >> q) & 1) for q in range(self.n_qubits)])
            Z_exp += prob * spins
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    ZZ_exp[i, j] += prob * spins[i] * spins[j]
        return Z_exp, ZZ_exp
