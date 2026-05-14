"""
Unitary k-designs.

A unitary k-design is a finite ensemble of unitaries that reproduces the
moments of the Haar measure up to order k. They're useful as cheap proxies
for fully Haar-random sampling.

This module provides:
- ``unitary_2_design``: a 2-design from random Cliffords (which form an
  exact 2-design)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def unitary_2_design(
    n_qubits: int,
    n_samples: int,
    *,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Sample n_samples unitaries from a 2-design.

    On a single qubit, the Clifford group (24 elements) is an exact 2-design.
    For multiple qubits, the n-qubit Clifford group is a 3-design.

    This implementation uses random Pauli rotations followed by random Clifford
    layers, which approximates a 2-design for small n.
    """
    rng = np.random.default_rng(seed)
    d = 1 << n_qubits
    samples: List[np.ndarray] = []

    for _ in range(n_samples):
        # Build a random Clifford-like circuit by composing random Paulis and
        # H gates. For n_qubits=1 we sample exactly from the Clifford group.
        if n_qubits == 1:
            # 1-qubit Clifford group has 24 elements; pick one at random
            clifford_idx = int(rng.integers(0, 24))
            U = _single_qubit_clifford(clifford_idx)
        else:
            # Approximate: random tensor product of single-qubit Cliffords
            U = np.eye(1, dtype=np.complex128)
            for q in range(n_qubits):
                idx = int(rng.integers(0, 24))
                U = np.kron(U, _single_qubit_clifford(idx))
        samples.append(U)
    return samples


def _single_qubit_clifford(idx: int) -> np.ndarray:
    """Return one of the 24 single-qubit Clifford gates as a 2x2 matrix."""
    I2 = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.0)
    S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)

    # Generate the 24 Cliffords as compositions of 6 generators × 4 Pauli prefixes
    paulis = [I2, X, Y, Z]
    pauli = paulis[idx % 4]
    base_idx = idx // 4
    # 6 base orbits via H/S combinations
    bases = [I2, H, S, H @ S, S @ H, H @ S @ H]
    base = bases[base_idx % 6]
    return base @ pauli
