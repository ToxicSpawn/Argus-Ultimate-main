"""
Quantum state and process tomography.

State tomography reconstructs the full density matrix of an unknown state
by measuring it in all Pauli bases. Process tomography reconstructs the
Choi matrix of an unknown quantum channel.

For n qubits, state tomography requires 4^n measurement settings (3^n basis
combinations). This is feasible for n <= 4 in practice.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np

from quantum_simulator import QuantumCircuit, simulate


# ═════════════════════════════════════════════════════════════════════════════
# State tomography
# ═════════════════════════════════════════════════════════════════════════════


def state_tomography(
    circuit: QuantumCircuit,
    *,
    shots_per_basis: int = 1024,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Reconstruct the density matrix of a state prepared by ``circuit``.

    Uses linear inversion: for each Pauli measurement basis P, computes the
    expectation ⟨P⟩ from measurements, then reconstructs ρ via
        ρ = (1/2^n) Σ_P ⟨P⟩ · P

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit that prepares the unknown state. Must NOT contain measure_all.
    shots_per_basis : int
        Shots per Pauli basis measurement.

    Returns
    -------
    np.ndarray
        (2^n, 2^n) reconstructed density matrix.
    """
    n = circuit.num_qubits
    if n > 5:
        raise ValueError(f"State tomography limited to n<=5 qubits, got {n}")

    rho = np.zeros((1 << n, 1 << n), dtype=complex)

    # Iterate over all Pauli strings on n qubits
    pauli_letters = ["I", "X", "Y", "Z"]
    for pauli_string in product(pauli_letters, repeat=n):
        # Identity contributes Tr(ρ) = 1 directly to the I⊗...⊗I term
        if all(p == "I" for p in pauli_string):
            rho += np.eye(1 << n, dtype=complex) / (1 << n)
            continue

        # Build the rotated circuit (basis change)
        meas_circ = QuantumCircuit(n)
        for op in circuit.operations:
            meas_circ._ops.append(op)
        for q in range(n):
            p = pauli_string[q]
            if p == "X":
                meas_circ.h(q)
            elif p == "Y":
                meas_circ.sdg(q)
                meas_circ.h(q)
        meas_circ.measure_all()

        result = simulate(meas_circ, shots=shots_per_basis, seed=seed)
        counts = result["counts"]
        total = sum(counts.values())
        if total == 0:
            continue

        # Compute expectation value of this Pauli string
        exp = 0.0
        for bitstring, c in counts.items():
            sign = 1.0
            for q in range(n):
                p = pauli_string[q]
                if p == "I":
                    continue
                bit = bitstring[len(bitstring) - 1 - q] if q < len(bitstring) else "0"
                sign *= 1.0 if bit == "0" else -1.0
            exp += c * sign
        exp /= total

        # Build the Pauli matrix
        P = _pauli_string_matrix(pauli_string)
        rho += exp * P / (1 << n)

    # Make Hermitian and trace-normalize
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho)
    if abs(tr) > 1e-12:
        rho = rho / tr * 1.0
    return rho


def _pauli_string_matrix(pauli_string: tuple) -> np.ndarray:
    """Build the matrix of a Pauli string by tensor product."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    paulis = {"I": I, "X": X, "Y": Y, "Z": Z}

    # Order: pauli_string[0] is qubit 0 (LSB) — needs to be the rightmost in
    # the tensor product (so it ends up at the lowest index of the basis).
    M = paulis[pauli_string[-1]]
    for p in reversed(pauli_string[:-1]):
        M = np.kron(paulis[p], M)
    return M


# ═════════════════════════════════════════════════════════════════════════════
# Process tomography
# ═════════════════════════════════════════════════════════════════════════════


def process_tomography(
    process_circuit_builder: Any,  # Callable[[QuantumCircuit], None]
    n_qubits: int,
    *,
    shots_per_basis: int = 1024,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Reconstruct the Choi matrix of a quantum process.

    For each pair (input state, output Pauli basis), measure the resulting
    counts and use linear inversion to recover the channel.

    Parameters
    ----------
    process_circuit_builder : Callable[[QuantumCircuit], None]
        Function that takes a QuantumCircuit and applies the process to it
        in place.
    n_qubits : int
        Number of qubits the process acts on.

    Returns
    -------
    np.ndarray
        (2^(2n), 2^(2n)) Choi matrix of the channel.
    """
    if n_qubits > 3:
        raise ValueError(f"Process tomography limited to n<=3 qubits, got {n_qubits}")

    # Use the standard "rho_in basis" {|0⟩, |1⟩, |+⟩, |+i⟩}^n input states
    input_states = ["0", "1", "+", "+i"]
    input_state_combos = list(product(input_states, repeat=n_qubits))

    n_inputs = len(input_state_combos)
    dim = 1 << n_qubits

    # Choi matrix: stack vectorized output density matrices
    choi = np.zeros((dim * dim, dim * dim), dtype=complex)

    for input_combo in input_state_combos:
        # Build input preparation circuit
        prep = QuantumCircuit(n_qubits)
        for q, s in enumerate(input_combo):
            if s == "1":
                prep.x(q)
            elif s == "+":
                prep.h(q)
            elif s == "+i":
                prep.h(q)
                prep.s(q)

        # Apply the process
        process_circuit_builder(prep)

        # State tomography on the output
        rho_out = state_tomography(prep, shots_per_basis=shots_per_basis, seed=seed)

        # Compute the input density matrix classically
        rho_in = _input_state_density(input_combo)

        # Add to Choi matrix
        choi += np.kron(rho_in.T, rho_out)

    choi /= n_inputs
    return choi


def _input_state_density(input_combo: tuple) -> np.ndarray:
    """Build the density matrix of a tensor product of single-qubit states."""
    states = {
        "0": np.array([1, 0], dtype=complex),
        "1": np.array([0, 1], dtype=complex),
        "+": np.array([1, 1], dtype=complex) / np.sqrt(2),
        "+i": np.array([1, 1j], dtype=complex) / np.sqrt(2),
    }
    psi = states[input_combo[-1]]
    for s in reversed(input_combo[:-1]):
        psi = np.kron(states[s], psi)
    return np.outer(psi, psi.conj())
