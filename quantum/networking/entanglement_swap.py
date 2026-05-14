"""
Entanglement swapping and GHZ-state distribution.

Entanglement swapping: given Bell pairs (A,B) and (C,D), perform a Bell
measurement on (B,C) — afterward, A and D are entangled even though they
never interacted directly.

GHZ-state distribution: extends entanglement to N parties via Bell pairs
plus stabilizer measurements.
"""

from __future__ import annotations

from typing import List

from quantum_simulator import QuantumCircuit


def entanglement_swapping() -> QuantumCircuit:
    """
    4-qubit circuit demonstrating entanglement swapping.

    Qubits 0, 1: Bell pair (A, B)
    Qubits 2, 3: Bell pair (C, D)
    Bell measurement on (1, 2) entangles (0, 3).
    """
    qc = QuantumCircuit(4)
    # Two Bell pairs
    qc.h(0)
    qc.cnot(0, 1)
    qc.h(2)
    qc.cnot(2, 3)
    # Bell measurement on (1, 2): CNOT then H
    qc.cnot(1, 2)
    qc.h(1)
    return qc


def build_ghz(n: int) -> QuantumCircuit:
    """
    Build an n-qubit GHZ state |0...0⟩ + |1...1⟩.

    Used for multi-party entanglement protocols.
    """
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cnot(i, i + 1)
    return qc
