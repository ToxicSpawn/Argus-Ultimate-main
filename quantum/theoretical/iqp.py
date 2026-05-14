"""
Instantaneous Quantum Polynomial (IQP) circuits.

IQP circuits are a restricted class with structure:
    H ⊗ ... ⊗ H · D · H ⊗ ... ⊗ H
where D is a diagonal matrix of phase gates. Despite their simplicity,
sampling from IQP output distributions is classically hard (Bremner,
Jozsa, Shepherd 2010).

Used as a quantum supremacy candidate and as a feature map for QML.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from quantum_simulator import QuantumCircuit, simulate


def iqp_circuit(
    n_qubits: int,
    *,
    n_diagonal_terms: int = 5,
    seed: Optional[int] = None,
) -> QuantumCircuit:
    """
    Build a random IQP circuit.

    Structure:
        H on every qubit
        Random diagonal D = ∏ exp(i θ_i Z_S_i) for random subsets S_i
        H on every qubit
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)

    # Hadamard layer
    for q in range(n_qubits):
        qc.h(q)

    # Random diagonal gates: each is a Z, ZZ, or ZZZ Pauli with random angle
    for _ in range(n_diagonal_terms):
        # Pick 1, 2, or 3 qubits at random
        size = int(rng.integers(1, min(4, n_qubits + 1)))
        qubits = sorted(rng.choice(n_qubits, size=size, replace=False).tolist())
        angle = float(rng.uniform(0, 2 * np.pi))
        if size == 1:
            qc.rz(angle, qubits[0])
        elif size == 2:
            qc.rzz(angle, qubits[0], qubits[1])
        else:
            # 3-qubit Z⊗Z⊗Z via CNOT staircase + RZ
            qc.cnot(qubits[0], qubits[1])
            qc.cnot(qubits[1], qubits[2])
            qc.rz(angle, qubits[2])
            qc.cnot(qubits[1], qubits[2])
            qc.cnot(qubits[0], qubits[1])

    # Final Hadamard layer
    for q in range(n_qubits):
        qc.h(q)

    return qc


def iqp_sample(
    n_qubits: int,
    *,
    n_diagonal_terms: int = 5,
    shots: int = 1024,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build and sample from a random IQP circuit.
    """
    qc = iqp_circuit(n_qubits, n_diagonal_terms=n_diagonal_terms, seed=seed)
    qc.measure_all()
    result = simulate(qc, shots=shots, seed=seed)
    return {
        "counts": result["counts"],
        "shots": shots,
        "n_qubits": n_qubits,
        "method": "iqp_sampling",
    }
