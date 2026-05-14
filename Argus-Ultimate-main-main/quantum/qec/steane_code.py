"""
[[7,1,3]] Steane quantum error correction code.

Steane is the smallest CSS code that protects against any single-qubit
Pauli error (X, Y, or Z). Encodes 1 logical qubit into 7 physical qubits.

Stabilizer generators (X-type and Z-type, derived from the [7,4,3] Hamming code):

    g1 = X X X X I I I
    g2 = X X I I X X I
    g3 = X I X I X I X
    g4 = Z Z Z Z I I I
    g5 = Z Z I I Z Z I
    g6 = Z I Z I Z I Z

Logical operators:
    X_L = X X X X X X X
    Z_L = Z Z Z Z Z Z Z

The 6 syndrome bits (3 X-type + 3 Z-type) uniquely identify any single
Pauli error.

This implementation provides:
- ``encode()`` — build the encoding circuit for |0_L⟩
- ``measure_syndrome()`` — extract the 6-bit syndrome via ancillas
- ``decode()`` — lookup-table syndrome → correction operator
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from quantum_simulator import QuantumCircuit


class SteaneCode:
    """[[7,1,3]] Steane code with full encode/syndrome/decode toolkit."""

    def __init__(self) -> None:
        self.n_data = 7
        # Stabilizer generators (each is 7 chars over {I, X, Z})
        self.x_stabilizers = [
            "XXXXIII",
            "XXIIXXI",
            "XIXIXIX",
        ]
        self.z_stabilizers = [
            "ZZZZIII",
            "ZZIIZZI",
            "ZIZIZIZ",
        ]

    # ── Encoding circuit for logical |0⟩ ─────────────────────────────────────

    def encode_zero(self) -> QuantumCircuit:
        """
        Build a circuit that prepares the Steane |0_L⟩ state on 7 qubits.

        The standard Steane encoding uses 6 CNOTs and 3 H gates:
            1. Prepare ancilla in superposition: H on qubits 0, 1, 2
            2. Apply CNOTs to spread the encoding across qubits 3..6
        """
        qc = QuantumCircuit(self.n_data)
        # Hadamards on the 3 "X-type" qubits
        for q in (0, 1, 2):
            qc.h(q)
        # CNOTs to spread parity (Steane CSS encoding circuit)
        # Standard layout (Nielsen & Chuang Fig 10.16):
        qc.cnot(0, 3)
        qc.cnot(1, 3)
        qc.cnot(0, 4)
        qc.cnot(2, 4)
        qc.cnot(1, 5)
        qc.cnot(2, 5)
        qc.cnot(0, 6)
        qc.cnot(1, 6)
        qc.cnot(2, 6)
        return qc

    # ── Syndrome measurement ─────────────────────────────────────────────────

    def measure_syndrome(
        self,
        data_circuit: QuantumCircuit,
    ) -> Tuple[QuantumCircuit, int]:
        """
        Append syndrome-measurement gates to ``data_circuit``.

        Adds 6 ancilla qubits to measure the 6 stabilizer generators.
        Returns the extended circuit and the index of the first ancilla.
        """
        n_data = self.n_data
        n_anc = 6
        n_total = n_data + n_anc
        # Build a fresh circuit on n_total qubits and copy data ops
        qc = QuantumCircuit(n_total)
        for op in data_circuit.operations:
            qc._ops.append(op)

        anc_offset = n_data

        # X-type syndrome measurements (qubits anc_offset..anc_offset+2)
        for i, x_stab in enumerate(self.x_stabilizers):
            anc = anc_offset + i
            qc.h(anc)
            for q, p in enumerate(x_stab):
                if p == "X":
                    qc.cnot(anc, q)
            qc.h(anc)

        # Z-type syndrome measurements (qubits anc_offset+3..anc_offset+5)
        for i, z_stab in enumerate(self.z_stabilizers):
            anc = anc_offset + 3 + i
            qc.h(anc)
            for q, p in enumerate(z_stab):
                if p == "Z":
                    qc.cz(anc, q)
            qc.h(anc)

        return qc, anc_offset

    # ── Decoding (classical lookup table) ────────────────────────────────────

    def decode_syndrome(self, syndrome: str) -> Optional[Tuple[int, str]]:
        """
        Decode a 6-bit syndrome to (qubit_index, error_type).

        Returns
        -------
        (qubit, "X" | "Y" | "Z") or None if no error detected.
        """
        if len(syndrome) != 6:
            return None
        if syndrome == "000000":
            return None

        # Z-syndrome (bits 0..2) detects X errors
        # X-syndrome (bits 3..5) detects Z errors
        # Combination → Y errors
        z_syn = syndrome[3:]  # bits from Z stabilizers (detect X errors)
        x_syn = syndrome[:3]  # bits from X stabilizers (detect Z errors)

        # Standard Steane lookup: syndrome bits give the qubit index in binary
        x_err_qubit = self._syndrome_to_qubit(z_syn)  # X error detected here
        z_err_qubit = self._syndrome_to_qubit(x_syn)  # Z error detected here

        if x_err_qubit is not None and z_err_qubit is not None:
            if x_err_qubit == z_err_qubit:
                return (x_err_qubit, "Y")
            else:
                return (x_err_qubit, "X")  # report X first
        if x_err_qubit is not None:
            return (x_err_qubit, "X")
        if z_err_qubit is not None:
            return (z_err_qubit, "Z")
        return None

    def _syndrome_to_qubit(self, syndrome_bits: str) -> Optional[int]:
        """Convert 3-bit syndrome to qubit index (1..7)."""
        if syndrome_bits == "000":
            return None
        try:
            idx = int(syndrome_bits, 2)
            if 1 <= idx <= 7:
                return idx - 1  # 0-indexed
        except ValueError:
            pass
        return None
