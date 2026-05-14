"""
Haar-random unitaries and states.

A Haar-random unitary is sampled uniformly from the unitary group U(d).
Equivalently, a Haar-random state is uniformly distributed on the
complex projective space CP^(d-1).

This module:
- Samples a Haar-random U(d) matrix via the Mezzadri (2007) algorithm
- Samples a Haar-random state vector

Reference
---------
Mezzadri, "How to generate random matrices from the classical compact
groups," Notices of the AMS 54, 592 (2007)
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def haar_random_unitary(
    n_qubits: int,
    *,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample a Haar-random unitary acting on n_qubits qubits.

    Algorithm:
    1. Generate a random complex matrix Z with i.i.d. complex Gaussians.
    2. QR-decompose: Z = QR.
    3. Multiply Q by the diagonal phase matrix Λ = diag(R[i,i] / |R[i,i]|).
       This ensures Q is uniformly distributed in U(d).
    """
    d = 1 << int(n_qubits)
    rng = np.random.default_rng(seed)
    Z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    diag_R = np.diag(R)
    Lam = diag_R / np.abs(diag_R)
    return Q * Lam[np.newaxis, :]


def haar_random_state(
    n_qubits: int,
    *,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample a Haar-random pure state.

    Equivalent to sampling a Haar-random unitary and applying its first
    column to |0...0⟩.
    """
    U = haar_random_unitary(n_qubits, seed=seed)
    return U[:, 0]
