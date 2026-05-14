"""
Random quantum circuit sampling.

Random circuit sampling is the canonical "quantum supremacy" benchmark
(Google 2019). It samples from random circuits whose output distributions
are exponentially hard to compute classically but easy to sample on a
quantum computer.

This module:
- Generates random universal circuits with brick-wall structure
- Samples outcomes via the in-repo simulator
- Computes the cross-entropy benchmark (XEB) score for quality estimation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from quantum_simulator import QuantumCircuit


# ═════════════════════════════════════════════════════════════════════════════
# Random circuit
# ═════════════════════════════════════════════════════════════════════════════


def random_circuit(
    n_qubits: int,
    depth: int,
    *,
    seed: Optional[int] = None,
) -> QuantumCircuit:
    """
    Generate a random universal quantum circuit.

    Each layer alternates 1q gates and 2q gates. The 1q gates are random
    elements from {RX, RY, RZ}; the 2q gates are CNOTs between random pairs.
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)
    for d in range(depth):
        # 1q gates on every qubit
        for q in range(n_qubits):
            angle = float(rng.uniform(0, 2 * np.pi))
            gate_choice = int(rng.integers(0, 3))
            if gate_choice == 0:
                qc.rx(angle, q)
            elif gate_choice == 1:
                qc.ry(angle, q)
            else:
                qc.rz(angle, q)
        # 2q gates: pair up qubits
        pairs = list(range(n_qubits))
        rng.shuffle(pairs)
        for i in range(0, n_qubits - 1, 2):
            qc.cnot(pairs[i], pairs[i + 1])
    return qc


def brick_wall_random_circuit(
    n_qubits: int,
    depth: int,
    *,
    seed: Optional[int] = None,
) -> QuantumCircuit:
    """
    Brick-wall random circuit (Google supremacy style).

    Alternates two patterns of nearest-neighbor 2q gates:
        Pattern A: (0,1), (2,3), (4,5), ...
        Pattern B: (1,2), (3,4), (5,6), ...
    Each layer also applies random 1q gates between the entangling layers.
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)
    for d in range(depth):
        # 1q layer: random RX/RY/RZ
        for q in range(n_qubits):
            angle = float(rng.uniform(0, 2 * np.pi))
            choice = int(rng.integers(0, 3))
            if choice == 0:
                qc.rx(angle, q)
            elif choice == 1:
                qc.ry(angle, q)
            else:
                qc.rz(angle, q)
        # 2q layer: brick-wall pattern
        if d % 2 == 0:
            for q in range(0, n_qubits - 1, 2):
                qc.cnot(q, q + 1)
        else:
            for q in range(1, n_qubits - 1, 2):
                qc.cnot(q, q + 1)
    return qc


def cross_entropy_benchmark(
    counts: Dict[str, int],
    ideal_probs: np.ndarray,
) -> float:
    """
    Linear cross-entropy benchmark score (XEB):

        F_XEB = (2^n / N) Σ_x p_ideal(x) - 1

    where the sum is over the N measured samples and p_ideal is the noise-free
    probability. F_XEB = 1 for perfect sampling, 0 for uniform random.
    """
    n_qubits = int(np.log2(len(ideal_probs)))
    N = sum(counts.values())
    if N == 0:
        return 0.0
    total = 0.0
    for bitstring, c in counts.items():
        idx = int(bitstring, 2)
        if idx < len(ideal_probs):
            total += c * float(ideal_probs[idx])
    return float((1 << n_qubits) * total / N - 1.0)
