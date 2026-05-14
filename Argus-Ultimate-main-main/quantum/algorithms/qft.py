"""
Quantum Fourier Transform (QFT) on the ARGUS in-repo simulator.

The QFT maps a computational-basis state |x⟩ to

    |x⟩ → (1 / √(2^n)) Σ_y exp(2πi x y / 2^n) |y⟩

and is the Fourier analogue used as a subroutine in QPE, Shor, and many other
algorithms.

Implementation: standard Hadamard + controlled-phase ladder, followed by a
reversal of the output qubits (implemented via SWAP gates). Uses only gates
defined in ``quantum_simulator`` so it runs on the in-repo simulator.

Usage
-----
>>> from quantum.algorithms.qft import quantum_fourier_transform
>>> qc = quantum_fourier_transform(4)
>>> # qc is a QuantumCircuit ready to have measure_all() appended
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from quantum_simulator import QuantumCircuit


def quantum_fourier_transform(num_qubits: int, inverse: bool = False) -> QuantumCircuit:
    """
    Build a QFT (or inverse QFT) circuit on ``num_qubits`` qubits.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the register.
    inverse : bool
        If True, return the inverse QFT (QFT†).

    Returns
    -------
    QuantumCircuit
        Circuit implementing QFT or inverse QFT. Does NOT include measurement.
    """
    n = int(num_qubits)
    if n <= 0:
        raise ValueError("num_qubits must be >= 1")
    qc = QuantumCircuit(n)
    apply_qft_inplace(qc, list(range(n)), inverse=inverse)
    return qc


def apply_qft_inplace(
    qc: QuantumCircuit,
    qubits: List[int],
    inverse: bool = False,
) -> None:
    """
    Apply QFT (or inverse QFT) to a subset of qubits in an existing circuit.

    Parameters
    ----------
    qc : QuantumCircuit
        Target circuit; gates are appended in place.
    qubits : List[int]
        Ordered list of qubit indices. ``qubits[0]`` is treated as the
        **least-significant** qubit (matching ``quantum_simulator``'s LSB-first
        computational basis), and ``qubits[-1]`` is the most significant.
    inverse : bool
        If True, apply QFT† instead of QFT.
    """
    n = len(qubits)
    if n == 0:
        return

    if not inverse:
        # Forward QFT. Process from MSB (qubits[n-1]) down to LSB (qubits[0]).
        for i in reversed(range(n)):
            qi = qubits[i]
            qc.h(qi)
            # Controlled-phase from each less-significant qubit
            for j in reversed(range(i)):
                qj = qubits[j]
                angle = 2.0 * np.pi / float(1 << (i - j + 1))
                qc.cphase(angle, qj, qi)
        # Swap qubits to reverse the bit order (QFT output is bit-reversed).
        for k in range(n // 2):
            qc.swap(qubits[k], qubits[n - 1 - k])
    else:
        # Inverse QFT: undo swap first, then run ladder in reverse with negated angles.
        for k in range(n // 2):
            qc.swap(qubits[k], qubits[n - 1 - k])
        for i in range(n):
            qi = qubits[i]
            for j in range(i):
                qj = qubits[j]
                angle = -2.0 * np.pi / float(1 << (i - j + 1))
                qc.cphase(angle, qj, qi)
            qc.h(qi)


def qft_matrix(num_qubits: int, inverse: bool = False) -> np.ndarray:
    """
    Return the analytical QFT unitary matrix (for tests / sanity checks).

    Notes
    -----
    The matrix uses the convention matching ``quantum_simulator``'s basis
    ordering, where qubit 0 is the least-significant bit in the integer
    representation of the computational basis index.
    """
    n = int(num_qubits)
    dim = 1 << n
    W = np.exp(2j * np.pi / dim)
    if inverse:
        W = np.conj(W)
    M = np.zeros((dim, dim), dtype=np.complex128)
    for j in range(dim):
        for k in range(dim):
            M[j, k] = W ** (j * k)
    return M / np.sqrt(dim)
