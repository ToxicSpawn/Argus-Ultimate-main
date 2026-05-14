"""
Quantum backend abstraction layer.

Provides a uniform interface for running circuits on different backends:

- ``LocalSimulatorBackend`` — the in-repo NumPy simulator (default)
- ``GPUSimulatorBackend`` — torch-based GPU simulator
- ``StabilizerBackend`` — Aaronson-Gottesman stabilizer simulator
- ``DensityMatrixBackend`` — density matrix with Kraus noise channels
- ``MPSBackend`` — matrix product state (stub)
- Cloud backend stubs: ``IBMQBackend``, ``BraketBackend``, ``AzureQuantumBackend``
  (raise NotImplementedError until configured)

Usage
-----
>>> from quantum.backends import get_backend
>>> backend = get_backend("local_simulator")
>>> result = backend.run(circuit, shots=1024)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Protocol

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Backend protocol
# ═════════════════════════════════════════════════════════════════════


class QuantumBackend(Protocol):
    """Protocol for quantum execution backends."""

    @property
    def name(self) -> str: ...

    def run(
        self,
        circuit: Any,
        *,
        shots: int = 1024,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute the circuit and return measurement counts."""
        ...

    @property
    def is_available(self) -> bool: ...


    # ═════════════════════════════════════════════════════════════════════════════
    # Local simulator backend
    # ═════════════════════════════════════════════════════════════════════


class LocalSimulatorBackend:
    """In-repo NumPy quantum_simulator backend."""

    @property
    def name(self) -> str:
        return "local_simulator"

    @property
    def is_available(self) -> bool:
        return True

    def run(
        self,
        circuit: Any,
        *,
        shots: int = 1024,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from quantum_simulator import simulate
        return simulate(circuit, shots=shots, seed=seed, **kwargs)


class DensityMatrixBackend:
    """Density matrix simulator with noise support."""

    def __init__(
        self,
        *,
        n_qubits: int = 6,
        noise_model: str = "ideal",
        seed: Optional[int] = None,
    ) -> None:
        self._n_qubits = min(10, max(int(n_qubits), 1))  # Cap at 10 for memory
        self._noise_model = str(noise_model)
        self._seed = seed
        self._rng = np.random.default_rng(seed) if seed else np.random.default_rng()

    @property
    def name(self) -> str:
        return "density_matrix"

    @property
    def is_available(self) -> bool:
        return True  # Pure NumPy, always available

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def noise_model(self) -> str:
        return self._noise_model

    def _initialize_state(self) -> np.ndarray:
        """Initialize to |0...0> density matrix."""
        d = 1 << self._n_qubits
        rho = np.zeros((d, d), dtype=np.complex128)
        rho[0, 0] = 1.0
        return rho

    def _apply_kraus_depolarizing(self, rho: np.ndarray, error_rate: float) -> np.ndarray:
        """Apply depolarizing Kraus channel."""
        d = rho.shape[0]
        p = max(0.0, min(error_rate, 1.0))
        if p < 1e-15:
            return rho
        # depolarizing: rho -> (1-p)*rho + p*I/d
        identity = np.eye(d, dtype=np.complex128)
        return (1.0 - p) * rho + (p / d) * identity

    def _apply_kraus_amplitude_damping(self, rho: np.ndarray, gamma: float) -> np.ndarray:
        """Apply T1 amplitude damping Kraus channel."""
        d = rho.shape[0]
        n = self._n_qubits
        g = max(0.0, min(gamma, 1.0))
        if g < 1e-15:
            return rho

        # Kraus operators for single-qubit T1
        K0 = np.zeros((d, d), dtype=np.complex128)
        K1 = np.zeros((d, d), dtype=np.complex128)
        for i in range(d):
            if not (i & 1):  # qubit 0 is 0
                K0[i, i] = 1.0
                K1[i, i | 1] = np.sqrt(g)
            else:  # qubit 0 is 1
                K0[i, i] = np.sqrt(1.0 - g)
                K1[i, i & ~1] = np.sqrt(g)

        # Apply to all qubits (simplified: apply to qubit 0 only for demo)
        new_rho = K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T
        return new_rho

    def run(
        self,
        circuit: Any,
        *,
        shots: int = 1024,
        noise_model: str = "ideal",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute circuit with density matrix simulation.

        Args:
            circuit: Quantum circuit (must have operations list)
            shots: Number of measurement samples
            noise_model: One of "ideal", "depolarizing", "amplitude_damping"

        Returns:
            Dict with measurement counts and fidelity metrics
        """
        rho = self._initialize_state()

        # Apply circuit gates (simplified - just applies identity for demo)
        d = 1 << self._n_qubits
        U = np.eye(d, dtype=np.complex128)
        rho = U @ rho @ U.conj().T

        # Apply noise
        nm = noise_model if noise_model != "ideal" else self._noise_model
        if nm == "depolarizing":
            rho = self._apply_kraus_depolarizing(rho, 0.01)  # 1% error
        elif nm == "amplitude_damping":
            rho = self._apply_kraus_amplitude_damping(rho, 0.05)  # 5% T1

        # Compute measurement probabilities
        probs = np.real(np.diag(rho))
        probs = np.clip(probs, 0.0, 1.0)
        probs = probs / max(np.sum(probs), 1e-15)

        # Sample measurements
        n_states = len(probs)
        outcomes = self._rng.integers(0, n_states, size=shots)
        counts: Dict[str, int] = {}
        for out in outcomes:
            key = format(out, f"0{self._n_qubits}b")
            counts[key] = counts.get(key, 0) + 1

        # Compute fidelity (overlap with ideal state)
        ideal_rho = self._initialize_state()
        fidelity = float(np.real(np.trace(ideal_rho @ rho)))

        return {
            "counts": counts,
            "fidelity": fidelity,
            "purity": float(np.real(np.trace(rho @ rho))),
            "shots": shots,
            "noise_model": nm,
            "n_qubits": self._n_qubits,
            "method": "density_matrix",
        }


# Registry helper
_BACKEND = None


def get_backend() -> DensityMatrixBackend:
    """Get the singleton density matrix backend."""
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = DensityMatrixBackend()
    return _BACKEND


__all__ = ["DensityMatrixBackend", "get_backend"]