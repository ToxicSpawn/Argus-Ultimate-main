"""MPS (Matrix Product State) tensor-network simulator backend.

Enables larger qubit counts (20-30 on typical machines) by using a
tensor-network representation instead of dense statevectors.

Limitations:
- Best for low-entanglement circuits (nearest-neighbor gates, shallow depth)
- Bond dimension limits entanglement entropy that can be represented
- Slower per-gate than dense statevector for small qubit counts
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class MPSState:
    """Matrix Product State representation.

    Each tensor A[i] has shape (bond_left, 2, bond_right).
    For open boundary conditions: A[0] has bond_left=1, A[-1] has bond_right=1.
    """

    tensors: List[np.ndarray]
    n_qubits: int
    bond_dim: int = 16
    max_bond_dim: int = 64

    @classmethod
    def zero_state(cls, n_qubits: int, bond_dim: int = 16, max_bond_dim: int = 64) -> "MPSState":
        """Create |0...0⟩ MPS state."""
        tensors = []
        for i in range(n_qubits):
            if i == 0:
                t = np.zeros((1, 2, 1), dtype=np.complex128)
                t[0, 0, 0] = 1.0  # |0⟩
            else:
                t = np.zeros((1, 2, 1), dtype=np.complex128)
                t[0, 0, 0] = 1.0
            tensors.append(t)
        return cls(tensors=tensors, n_qubits=n_qubits, bond_dim=bond_dim, max_bond_dim=max_bond_dim)

    def apply_single_gate(self, qubit: int, gate: np.ndarray) -> None:
        """Apply a 2x2 gate to a single qubit."""
        self.tensors[qubit] = np.einsum("ijk,li->ljk", self.tensors[qubit], gate)

    def apply_two_qubit_gate(self, q0: int, q1: int, gate: np.ndarray) -> None:
        """Apply a 4x4 gate to two adjacent qubits.

        Assumes q0 and q1 are adjacent (|q0 - q1| == 1).
        """
        if abs(q0 - q1) != 1:
            raise ValueError("MPS backend requires adjacent qubits for two-qubit gates")

        left, right = min(q0, q1), max(q0, q1)

        # Contract the two tensors
        A_left = self.tensors[left]
        A_right = self.tensors[right]

        # Combined tensor: (bond_left, 2, bond_mid) x (bond_mid, 2, bond_right)
        # -> (bond_left, 2, 2, bond_right)
        combined = np.einsum("ijk,klm->ijlm", A_left, A_right)

        # Apply the 4x4 gate to the combined tensor
        bl, d1, d2, br = combined.shape
        # Reshape to (bl, 4, br), apply 4x4 gate, reshape back
        combined_flat = combined.reshape(bl, 4, br)
        gated = np.einsum("ij,kjl->kil", gate.reshape(4, 4), combined_flat)
        gated = gated.reshape(bl, 2, 2, br)

        # SVD to split back into two tensors
        mat = gated.reshape(bl * 2, 2 * br)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        # Truncate to max bond dimension
        chi = min(len(S), self.max_bond_dim)
        U = U[:, :chi]
        S = S[:chi]
        Vh = Vh[:chi, :]

        # Reshape back
        self.tensors[left] = (U * S).reshape(bl, 2, chi)
        self.tensors[right] = Vh.reshape(chi, 2, br)

    def probabilities(self) -> Dict[str, float]:
        """Compute probability distribution from MPS."""
        # Contract all tensors
        result = self.tensors[0]
        for i in range(1, self.n_qubits):
            result = np.einsum("ijk,klm->ijlm", result, self.tensors[i])
            shape = result.shape
            result = result.reshape(shape[0], shape[1] * shape[2], shape[3])

        # Flatten to statevector
        statevector = result.ravel()
        probs = np.abs(statevector) ** 2
        total = probs.sum()
        if total <= 0:
            return {format(0, f"0{self.n_qubits}b"): 1.0}
        probs = probs / total
        return {format(i, f"0{self.n_qubits}b"): float(p) for i, p in enumerate(probs) if p > 1e-15}

    def sample(self, shots: int, seed: int | None = None) -> Dict[str, int]:
        """Sample measurement outcomes from the MPS."""
        probs_dict = self.probabilities()
        if not probs_dict:
            return {format(0, f"0{self.n_qubits}b"): shots}

        states = list(probs_dict.keys())
        probs = np.array([probs_dict[s] for s in states])
        probs = probs / probs.sum()

        rng = np.random.default_rng(seed)
        samples = rng.choice(len(states), size=shots, p=probs)
        counts = {}
        for idx in samples:
            s = states[idx]
            counts[s] = counts.get(s, 0) + 1
        return counts

    def entanglement_entropy(self, bond: int) -> float:
        """Von Neumann entropy of the bipartition at a given bond.

        bond=0 means between qubit 0 and qubit 1.
        """
        A_left = self.tensors[bond]
        mat = A_left.reshape(-1, A_left.shape[-1])
        _, S, _ = np.linalg.svd(mat, full_matrices=False)
        S = S[S > 1e-15]
        return float(-np.sum(S**2 * np.log2(S**2)))


@dataclass(frozen=True)
class MPSResult:
    """Result from MPS simulation."""

    circuit_name: str
    counts: dict[str, int]
    probabilities: dict[str, float]
    n_qubits: int
    shots: int
    max_bond_dim: int
    execution_mode: str
    entanglement_entropy: dict[int, float]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "circuit_name": self.circuit_name,
            "counts": dict(self.counts),
            "probabilities": dict(self.probabilities),
            "n_qubits": self.n_qubits,
            "shots": self.shots,
            "max_bond_dim": self.max_bond_dim,
            "execution_mode": self.execution_mode,
            "entanglement_entropy": {str(k): v for k, v in self.entanglement_entropy.items()},
            "warnings": list(self.warnings),
        }


def mps_simulate_ghz(
    n_qubits: int = 20,
    shots: int = 1024,
    max_bond_dim: int = 64,
    seed: int | None = None,
) -> dict[str, Any]:
    """Simulate a GHZ state using MPS backend.

    This demonstrates MPS for a maximally entangled state — the bond dimension
    will be fully saturated.
    """
    mps = MPSState.zero_state(n_qubits, max_bond_dim=max_bond_dim)

    # H on qubit 0
    _H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    mps.apply_single_gate(0, _H)

    # CNOT chain
    _CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128).reshape(2, 2, 2, 2)
    for i in range(n_qubits - 1):
        mps.apply_two_qubit_gate(i, i + 1, _CNOT.reshape(4, 4))

    # Measure
    counts = mps.sample(shots, seed=seed)
    probabilities = {k: v / shots for k, v in sorted(counts.items())}

    # Entanglement entropy at each bond
    entropy = {}
    for bond in range(min(n_qubits - 1, 5)):
        entropy[bond] = mps.entanglement_entropy(bond)

    return MPSResult(
        circuit_name="mps_ghz",
        counts=counts,
        probabilities=probabilities,
        n_qubits=n_qubits,
        shots=shots,
        max_bond_dim=max_bond_dim,
        execution_mode="mps_tensor_network",
        entanglement_entropy=entropy,
    ).to_dict()


def mps_simulate_product(
    n_qubits: int = 20,
    shots: int = 1024,
    seed: int | None = None,
) -> dict[str, Any]:
    """Simulate a product state (all qubits in |0⟩) using MPS.

    This is the trivial case — bond dimension stays 1.
    """
    mps = MPSState.zero_state(n_qubits, max_bond_dim=1)

    counts = mps.sample(shots, seed=seed)
    probabilities = {k: v / shots for k, v in sorted(counts.items())}

    entropy = {}
    for bond in range(min(n_qubits - 1, 5)):
        entropy[bond] = mps.entanglement_entropy(bond)

    return MPSResult(
        circuit_name="mps_product",
        counts=counts,
        probabilities=probabilities,
        n_qubits=n_qubits,
        shots=shots,
        max_bond_dim=1,
        execution_mode="mps_tensor_network",
        entanglement_entropy=entropy,
    ).to_dict()


__all__ = [
    "MPSState",
    "MPSResult",
    "mps_simulate_ghz",
    "mps_simulate_product",
]
