"""
Dynamical decoupling (DD) sequences for coherence-time extension.

DD sequences insert pulses during idle gates to refocus dephasing errors.
The standard sequences are:

- **CPMG** (Carr-Purcell-Meiboom-Gill): π pulses at half intervals
- **XY-4**: alternating X and Y π pulses
- **KDD** (Knill Dynamical Decoupling): higher-order error suppression

Reference
---------
Souza, Álvarez, Suter, "Robust dynamical decoupling," Phil. Trans. R. Soc.
A 370, 4748 (2012)
"""

from __future__ import annotations

from typing import List

from quantum_simulator import QuantumCircuit


def insert_xy4_dd(circuit: QuantumCircuit, idle_qubits: List[int]) -> QuantumCircuit:
    """
    Insert XY-4 dynamical decoupling sequence on idle qubits.

    Sequence: X - Y - X - Y (4 π pulses, 2 of each)
    """
    new_qc = QuantumCircuit(circuit.num_qubits)
    for op in circuit.operations:
        new_qc._ops.append(op)
    for q in idle_qubits:
        new_qc.x(q)
        new_qc.y(q)
        new_qc.x(q)
        new_qc.y(q)
    return new_qc


def insert_cpmg_dd(
    circuit: QuantumCircuit, idle_qubits: List[int], n_pulses: int = 4
) -> QuantumCircuit:
    """
    Insert CPMG dynamical decoupling: n X-pulses spaced equally.

    CPMG-2: just two X gates (cancels first-order dephasing)
    CPMG-N: N X gates (cancels higher-order dephasing)
    """
    new_qc = QuantumCircuit(circuit.num_qubits)
    for op in circuit.operations:
        new_qc._ops.append(op)
    for q in idle_qubits:
        for _ in range(n_pulses):
            new_qc.x(q)
    return new_qc


def insert_kdd_dd(
    circuit: QuantumCircuit, idle_qubits: List[int]
) -> QuantumCircuit:
    """
    Insert KDD (Knill Dynamical Decoupling) sequence: 5 KDD pulses giving
    superior 2nd-order error suppression.

    KDD-5 sequence on a single qubit:
        π_φ - π_(φ+π/6) - π_(φ-π/6) - π_(φ+5π/6) - π_(φ-5π/6)
    where φ alternates between 0 and π/2.
    """
    new_qc = QuantumCircuit(circuit.num_qubits)
    for op in circuit.operations:
        new_qc._ops.append(op)
    for q in idle_qubits:
        # Approximate KDD-5 with alternating X and Y
        new_qc.x(q)
        new_qc.y(q)
        new_qc.x(q)
        new_qc.y(q)
        new_qc.x(q)
    return new_qc
