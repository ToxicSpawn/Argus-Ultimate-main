"""
Quantum entropy and information measures.

All functions take either a statevector (1D ndarray) or density matrix (2D
ndarray) as input. They use NumPy throughout for accuracy and speed on small
systems (n <= 10 qubits).
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import numpy as np


def _to_density_matrix(state: np.ndarray) -> np.ndarray:
    """Convert a statevector to a density matrix; pass through if already 2D."""
    if state.ndim == 1:
        return np.outer(state, state.conj())
    return state


def von_neumann_entropy(state: np.ndarray) -> float:
    """
    Von Neumann entropy S(ρ) = -Tr(ρ log₂ ρ) = -Σ λ_i log₂ λ_i.

    Parameters
    ----------
    state : np.ndarray
        Statevector (1D) or density matrix (2D).

    Returns
    -------
    float
        Entropy in bits. Pure states give 0; maximally mixed states give log₂(d).
    """
    rho = _to_density_matrix(state)
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]  # numerical safety
    if len(eigvals) == 0:
        return 0.0
    return float(-np.sum(eigvals * np.log2(eigvals)))


def entanglement_entropy(
    state: np.ndarray,
    partition: List[int],
    n_qubits: Optional[int] = None,
) -> float:
    """
    Entanglement entropy of a pure state across a bipartition.

    Computes the von Neumann entropy of the reduced density matrix on the
    qubits in ``partition``.

    Parameters
    ----------
    state : np.ndarray
        Statevector (1D, length 2^n).
    partition : List[int]
        Qubit indices in subsystem A (the rest are subsystem B).
    n_qubits : int, optional
        Total number of qubits. Inferred from state size if not given.

    Returns
    -------
    float
        Entanglement entropy in bits.
    """
    if state.ndim != 1:
        raise ValueError("entanglement_entropy requires a pure-state vector")
    if n_qubits is None:
        n_qubits = int(np.log2(state.shape[0]))
    if 1 << n_qubits != state.shape[0]:
        raise ValueError(f"state size {state.shape[0]} is not 2^{n_qubits}")

    n = n_qubits
    # Partition qubit indices into A and B (qubit 0 = LSB)
    A = sorted(set(int(q) for q in partition))
    B = [q for q in range(n) if q not in A]
    if not A or not B:
        return 0.0  # trivial bipartition

    # Reshape state to (2,)*n tensor; axis (n - 1 - q) corresponds to qubit q
    tensor = state.reshape((2,) * n)
    # Move A axes to the front, B axes to the back
    a_axes = [n - 1 - q for q in A]
    b_axes = [n - 1 - q for q in B]
    perm = a_axes + b_axes
    tensor = np.transpose(tensor, axes=perm)
    # Reshape to (2^|A|, 2^|B|) and compute reduced density matrix on A
    M = tensor.reshape(1 << len(A), 1 << len(B))
    rho_A = M @ M.conj().T
    return von_neumann_entropy(rho_A)


def mutual_information(
    state: np.ndarray,
    A: List[int],
    B: List[int],
    n_qubits: Optional[int] = None,
) -> float:
    """
    Quantum mutual information I(A:B) = S(A) + S(B) - S(AB).

    Parameters
    ----------
    state : np.ndarray
        Statevector or density matrix.
    A, B : List[int]
        Disjoint subsets of qubit indices.
    """
    if set(A) & set(B):
        raise ValueError("A and B must be disjoint")
    AB = sorted(set(A) | set(B))
    s_a = entanglement_entropy(state, A, n_qubits)
    s_b = entanglement_entropy(state, B, n_qubits)
    s_ab = entanglement_entropy(state, AB, n_qubits)
    return float(s_a + s_b - s_ab)


def purity(state: np.ndarray) -> float:
    """
    Purity Tr(ρ²). Equals 1 for pure states, 1/d for maximally mixed.
    """
    rho = _to_density_matrix(state)
    return float(np.real(np.trace(rho @ rho)))


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Quantum fidelity F(ρ, σ) = (Tr √(√ρ σ √ρ))².

    For pure states ρ = |ψ⟩⟨ψ| and σ = |φ⟩⟨φ|, this reduces to |⟨ψ|φ⟩|².
    """
    rho = _to_density_matrix(rho)
    sigma = _to_density_matrix(sigma)
    # For two pure states
    if np.linalg.matrix_rank(rho) == 1 and np.linalg.matrix_rank(sigma) == 1:
        # Find eigenvectors with eigenvalue 1
        ev_r, vec_r = np.linalg.eigh(rho)
        ev_s, vec_s = np.linalg.eigh(sigma)
        psi = vec_r[:, -1]
        phi = vec_s[:, -1]
        return float(abs(np.vdot(psi, phi)) ** 2)
    # General mixed-state fidelity (Uhlmann)
    sqrt_rho = _matrix_sqrt(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    inner_sqrt = _matrix_sqrt(inner)
    tr = np.trace(inner_sqrt)
    return float(np.real(tr) ** 2)


def _matrix_sqrt(M: np.ndarray) -> np.ndarray:
    """Hermitian matrix square root via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 0.0)  # numerical safety
    sqrt_eigvals = np.sqrt(eigvals)
    return eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.conj().T


def quantum_fisher_information_matrix(
    circuit_builder: Callable[[np.ndarray], Any],
    params: np.ndarray,
    *,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Quantum Fisher Information Matrix (QFIM).

    QFIM_ij = 4 · Re[⟨∂_i ψ | ∂_j ψ⟩ - ⟨∂_i ψ | ψ⟩⟨ψ | ∂_j ψ⟩]

    Used by Quantum Natural Gradient (Phase F3) to precondition the gradient
    update direction.

    Parameters
    ----------
    circuit_builder : Callable[[np.ndarray], QuantumCircuit]
        Function that takes a parameter vector and returns a QuantumCircuit.
    params : np.ndarray
        Current parameter point.
    eps : float
        Finite-difference step size.

    Returns
    -------
    np.ndarray
        (n_params, n_params) QFIM matrix.
    """
    from quantum_simulator import _simulate_statevector

    n = params.size
    qfim = np.zeros((n, n), dtype=float)

    base_state = _simulate_statevector(circuit_builder(params))

    # Estimate ∂_i |ψ⟩ via finite difference
    derivs = []
    for i in range(n):
        ps = params.copy()
        ms = params.copy()
        ps[i] += eps
        ms[i] -= eps
        d = (
            _simulate_statevector(circuit_builder(ps))
            - _simulate_statevector(circuit_builder(ms))
        ) / (2 * eps)
        derivs.append(d)

    # QFIM = 4 Re[⟨∂_i|∂_j⟩ - ⟨∂_i|ψ⟩⟨ψ|∂_j⟩]
    for i in range(n):
        for j in range(n):
            inner_ij = np.vdot(derivs[i], derivs[j])
            inner_i_psi = np.vdot(derivs[i], base_state)
            inner_psi_j = np.vdot(base_state, derivs[j])
            qfim[i, j] = 4.0 * float(np.real(inner_ij - inner_i_psi * inner_psi_j))

    return qfim
