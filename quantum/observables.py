"""Statevector inspection, observables, and entanglement measures.

Provides:
- Full statevector probability and amplitude access
- Pauli observables (Z, X, Y, ZZ, custom)
- Von Neumann entropy
- Purity
- Concurrence (two-qubit entanglement)
- Schmidt decomposition
- Fidelity between two statevectors
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Statevector inspection
# ---------------------------------------------------------------------------


def probabilities_from_statevector(state: np.ndarray, n_qubits: int) -> Dict[str, float]:
    """Convert a statevector to a probability distribution."""
    probs = np.abs(state) ** 2
    total = probs.sum()
    if total <= 0:
        return {format(0, f"0{n_qubits}b"): 1.0}
    probs = probs / total
    return {format(i, f"0{n_qubits}b"): float(p) for i, p in enumerate(probs) if p > 1e-15}


def amplitudes_from_statevector(state: np.ndarray, n_qubits: int) -> Dict[str, complex]:
    """Convert a statevector to a complex amplitude dict."""
    return {format(i, f"0{n_qubits}b"): complex(state[i]) for i in range(len(state))}


def density_matrix(state: np.ndarray) -> np.ndarray:
    """Compute pure-state density matrix ρ = |ψ⟩⟨ψ|."""
    return np.outer(state, np.conj(state))


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------


def pauli_z(n_qubits: int, qubit: int) -> np.ndarray:
    """Pauli Z operator on a specific qubit in n-qubit system."""
    diag = np.ones(2**n_qubits, dtype=np.complex128)
    for i in range(2**n_qubits):
        if (i >> (n_qubits - 1 - qubit)) & 1:
            diag[i] = -1
    return np.diag(diag)


def pauli_x(n_qubits: int, qubit: int) -> np.ndarray:
    """Pauli X operator on a specific qubit in n-qubit system."""
    dim = 2**n_qubits
    mat = np.zeros((dim, dim), dtype=np.complex128)
    s = n_qubits - 1 - qubit
    mask = 1 << s
    for i in range(dim):
        j = i ^ mask
        mat[i, j] = 1
    return mat


def pauli_y(n_qubits: int, qubit: int) -> np.ndarray:
    """Pauli Y operator on a specific qubit in n-qubit system."""
    dim = 2**n_qubits
    mat = np.zeros((dim, dim), dtype=np.complex128)
    s = n_qubits - 1 - qubit
    mask = 1 << s
    for i in range(dim):
        j = i ^ mask
        bit_j = (j >> s) & 1
        mat[i, j] = 1j if bit_j == 0 else -1j
    return mat


def pauli_zz(n_qubits: int, q0: int, q1: int) -> np.ndarray:
    """ZZ operator: Z_q0 ⊗ Z_q1."""
    diag = np.ones(2**n_qubits, dtype=np.complex128)
    s0 = n_qubits - 1 - q0
    s1 = n_qubits - 1 - q1
    for i in range(2**n_qubits):
        b0 = (i >> s0) & 1
        b1 = (i >> s1) & 1
        if b0 != b1:
            diag[i] = -1
    return np.diag(diag)


def expectation_value(state: np.ndarray, observable: np.ndarray) -> float:
    """Compute ⟨ψ|O|ψ⟩."""
    return float(np.real(np.vdot(state, observable @ state)))


def variance(state: np.ndarray, observable: np.ndarray) -> float:
    """Compute Var(O) = ⟨ψ|O²|ψ⟩ - ⟨ψ|O|ψ⟩²."""
    exp = expectation_value(state, observable)
    obs2 = observable @ observable
    exp2 = expectation_value(state, obs2)
    return float(exp2 - exp**2)


# ---------------------------------------------------------------------------
# Entanglement measures
# ---------------------------------------------------------------------------


def von_neumann_entropy(state: np.ndarray) -> float:
    """Von Neumann entropy S(ρ) = -Tr(ρ log ρ) for a pure state.

    For a pure state this is 0; for a mixed state (after partial trace) it
    measures entanglement across the bipartition.
    """
    rho = density_matrix(state)
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-15]
    return float(-np.sum(eigvals * np.log2(eigvals)))


def purity(state: np.ndarray) -> float:
    """Purity Tr(ρ²). 1.0 for pure states."""
    rho = density_matrix(state)
    return float(np.real(np.trace(rho @ rho)))


def concurrence(state: np.ndarray, n_qubits: int, q0: int, q1: int) -> float:
    """Two-qubit concurrence measure.

    Extracts the 2-qubit reduced density matrix for qubits q0, q1 and
    computes the Wootters concurrence.
    """
    # Partial trace to get 2-qubit reduced density matrix
    rho = density_matrix(state)
    target = sorted([q0, q1])
    q0_idx, q1_idx = target

    # Indices for the 2-qubit subsystem
    dim_2q = 4
    rho_2q = np.zeros((dim_2q, dim_2q), dtype=np.complex128)

    for a in range(2**n_qubits):
        for b in range(2**n_qubits):
            # Extract bits for target qubits
            a0 = (a >> (n_qubits - 1 - q0_idx)) & 1
            a1 = (a >> (n_qubits - 1 - q1_idx)) & 1
            b0 = (b >> (n_qubits - 1 - q0_idx)) & 1
            b1 = (b >> (n_qubits - 1 - q1_idx)) & 1

            # Check if all OTHER qubits are equal
            match = True
            for q in range(n_qubits):
                if q == q0_idx or q == q1_idx:
                    continue
                if ((a >> (n_qubits - 1 - q)) & 1) != ((b >> (n_qubits - 1 - q)) & 1):
                    match = False
                    break
            if match:
                rho_2q[a0 * 2 + a1, b0 * 2 + b1] += rho[a, b]

    # Wootters concurrence
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    flip = np.kron(sigma_y, sigma_y)
    rho_tilde = flip @ np.conj(rho_2q) @ flip
    R = rho_2q @ rho_tilde
    eigvals = np.sqrt(np.maximum(np.real(np.linalg.eigvals(R)), 0.0))
    eigvals = np.sort(eigvals)[::-1]
    c = max(0.0, eigvals[0] - eigvals[1] - eigvals[2] - eigvals[3])
    return float(min(1.0, c))


def fidelity(state_a: np.ndarray, state_b: np.ndarray) -> float:
    """Fidelity |⟨ψ_a|ψ_b⟩|² between two pure states."""
    return float(np.abs(np.vdot(state_a, state_b)) ** 2)


def schmidt_coefficients(state: np.ndarray, n_qubits: int, bipartition: int) -> np.ndarray:
    """Schmidt coefficients for a bipartition of the state.

    bipartition is the number of qubits in subsystem A.
    """
    dim_a = 2**bipartition
    dim_b = 2**(n_qubits - bipartition)
    matrix = state.reshape(dim_a, dim_b)
    return np.linalg.svd(matrix, compute_uv=False)


__all__ = [
    "probabilities_from_statevector",
    "amplitudes_from_statevector",
    "density_matrix",
    "pauli_z",
    "pauli_x",
    "pauli_y",
    "pauli_zz",
    "expectation_value",
    "variance",
    "von_neumann_entropy",
    "purity",
    "concurrence",
    "fidelity",
    "schmidt_coefficients",
]
