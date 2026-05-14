"""
Superdense coding.

Alice can send 2 classical bits to Bob using only 1 qubit, provided they
share a Bell pair beforehand:

1. Alice and Bob share Bell pair |Φ⁺⟩ = (|00⟩ + |11⟩) / √2
2. Alice encodes 2 bits (b1, b0) by applying:
   - 00 → I
   - 01 → X
   - 10 → Z
   - 11 → ZX = iY
3. Alice sends her qubit to Bob
4. Bob applies CNOT then H, then measures both qubits to read the 2 bits
"""

from __future__ import annotations

from typing import Tuple

from quantum_simulator import QuantumCircuit, simulate


def superdense_encode(message: int) -> QuantumCircuit:
    """
    Encode 2 classical bits via superdense coding.

    Parameters
    ----------
    message : int
        2-bit message in {0, 1, 2, 3}.

    Returns
    -------
    QuantumCircuit
        Full encode + decode circuit. After measurement, the result
        bitstring should equal ``format(message, '02b')``.
    """
    if message < 0 or message > 3:
        raise ValueError(f"message must be in [0, 3], got {message}")

    qc = QuantumCircuit(2)
    # Step 1: prepare Bell pair on (q1, q0) — q1 is the high bit, q0 the low
    qc.h(1)
    qc.cnot(1, 0)

    # Step 2: Alice encodes 2-bit message (bit1, bit0) on qubit 1
    # The standard mapping after the (CNOT, H) decode is:
    #   00 → I, 01 → X, 10 → Z, 11 → ZX
    # We extract bit0 and bit1 from `message`.
    bit0 = message & 1
    bit1 = (message >> 1) & 1
    if bit0 == 1:
        qc.x(1)
    if bit1 == 1:
        qc.z(1)

    # Step 3: Bob decodes via CNOT + H
    qc.cnot(1, 0)
    qc.h(1)
    qc.measure_all()
    return qc


def superdense_decode(message: int, *, shots: int = 100, seed: int = 42) -> int:
    """
    Run superdense coding for a given message and recover the decoded result.
    """
    qc = superdense_encode(message)
    res = simulate(qc, shots=shots, seed=seed)
    counts = res["counts"]
    top_bitstring = max(counts.items(), key=lambda kv: kv[1])[0]
    # Bitstring is MSB-first; qubit 0 is the rightmost char.
    # The 2-bit message is stored as (q1 high, q0 low) → ordinary int(bs, 2).
    return int(top_bitstring, 2)
