"""
3-qubit bit-flip repetition code.

Encodes 1 logical qubit into 3 physical qubits:
    |0_L⟩ = |000⟩
    |1_L⟩ = |111⟩

Detects and corrects 1 bit-flip (X) error. Cannot correct phase-flip (Z) errors.

Encoding circuit:
    |ψ⟩|0⟩|0⟩ → |ψ⟩|ψ⟩|ψ⟩ via two CNOTs

Syndrome measurement uses ancillas to detect parity violations:
    syndrome bit 0 = (q_0 XOR q_1)
    syndrome bit 1 = (q_1 XOR q_2)

Lookup decoder:
    syndrome 00 → no error
    syndrome 01 → error on qubit 2 (apply X_2 to correct)
    syndrome 10 → error on qubit 0 (apply X_0 to correct)
    syndrome 11 → error on qubit 1 (apply X_1 to correct)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from quantum_simulator import QuantumCircuit, simulate


class RepetitionCode:
    """3-qubit bit-flip repetition code."""

    def __init__(self) -> None:
        self.n_data = 3
        self.n_ancilla = 2
        self.n_total = self.n_data + self.n_ancilla

    # ── Encoding ─────────────────────────────────────────────────────────────

    def encode(self, alpha: complex = 1.0, beta: complex = 0.0) -> QuantumCircuit:
        """
        Encode |ψ⟩ = α|0⟩ + β|1⟩ into |ψ_L⟩ = α|000⟩ + β|111⟩.

        For testing we use a state preparation that produces this exact state.
        Returns a 3-qubit data + 2-qubit ancilla circuit.
        """
        qc = QuantumCircuit(self.n_total)
        # Prepare |ψ⟩ on qubit 0
        if abs(beta) > 1e-12:
            theta = 2.0 * np.arctan2(abs(beta), abs(alpha))
            qc.ry(theta, 0)
        # Encode: CNOT(0, 1), CNOT(0, 2)
        qc.cnot(0, 1)
        qc.cnot(0, 2)
        return qc

    # ── Syndrome measurement ─────────────────────────────────────────────────

    def measure_syndrome(self, qc: QuantumCircuit) -> QuantumCircuit:
        """
        Append syndrome measurement gates to ``qc``.

        Syndrome bit 0 = parity(q_0, q_1) = q_0 ⊕ q_1
        Syndrome bit 1 = parity(q_1, q_2) = q_1 ⊕ q_2
        Stored on ancillas q_3 and q_4.
        """
        # Sx0: ancilla q_3 ← q_0 ⊕ q_1
        qc.cnot(0, 3)
        qc.cnot(1, 3)
        # Sx1: ancilla q_4 ← q_1 ⊕ q_2
        qc.cnot(1, 4)
        qc.cnot(2, 4)
        return qc

    # ── Decoding (classical lookup) ──────────────────────────────────────────

    def decode(self, syndrome: str) -> Optional[int]:
        """
        Decode the 2-bit syndrome and return the qubit (0, 1, 2) to correct,
        or None if no error.

        ``syndrome`` is a 2-character string ``"s1 s0"`` where s0 = parity(q0,q1),
        s1 = parity(q1,q2).

        Truth table:
            00 → no error
            01 → only s0 = 1, only q0,q1 parity broken → error on q0
            10 → only s1 = 1, only q1,q2 parity broken → error on q2
            11 → both parities broken → error on q1 (the one shared)
        """
        s = syndrome.strip()
        if s == "00":
            return None
        if s == "01":
            return 0  # only s0 active → q_0 flipped
        if s == "10":
            return 2  # only s1 active → q_2 flipped
        if s == "11":
            return 1  # both → q_1 flipped
        return None

    # ── Full encode-error-decode demo ────────────────────────────────────────

    def demo_correct_single_bit_flip(
        self,
        error_qubit: int = 0,
        *,
        shots: int = 1024,
        seed: Optional[int] = 42,
    ) -> Dict[str, any]:
        """
        Demonstrate the code correcting a single bit-flip error on the
        specified qubit. Returns the syndrome distribution and decoded
        correction.
        """
        # Encode |0_L⟩
        qc = self.encode(alpha=1.0, beta=0.0)
        # Inject an X error
        qc.x(error_qubit)
        # Measure syndrome
        self.measure_syndrome(qc)
        qc.measure_all()

        result = simulate(qc, shots=shots, seed=seed)
        counts = result["counts"]

        # Extract syndrome bits (qubits 3 and 4 = positions n-1-3 and n-1-4 in
        # MSB-first bitstring)
        syndrome_counts: Dict[str, int] = {}
        for bitstring, c in counts.items():
            # Bitstring is MSB-first; qubit 0 = rightmost
            n = len(bitstring)
            s0 = bitstring[n - 1 - 3]  # ancilla q_3
            s1 = bitstring[n - 1 - 4]  # ancilla q_4
            syndrome = f"{s1}{s0}"
            syndrome_counts[syndrome] = syndrome_counts.get(syndrome, 0) + c

        # The most-likely syndrome should match the injected error
        top_syndrome = max(syndrome_counts.items(), key=lambda kv: kv[1])[0]
        decoded = self.decode(top_syndrome)

        return {
            "injected_error_qubit": error_qubit,
            "syndrome_counts": syndrome_counts,
            "top_syndrome": top_syndrome,
            "decoded_correction": decoded,
            "correction_correct": decoded == error_qubit,
        }
