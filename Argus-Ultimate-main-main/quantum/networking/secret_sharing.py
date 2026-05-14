"""
Quantum secret sharing.

Hillery-Bužek-Berthiaume (1999) protocol: Alice shares a secret with two
parties (Bob and Charlie) such that neither can recover the secret alone,
but they can together by combining their measurements.

The shared secret is encoded into a 3-qubit GHZ state, with each party
holding one qubit. Reconstructing the secret requires both Bob's and
Charlie's measurement results.

Reference
---------
Hillery, Bužek, Berthiaume, "Quantum secret sharing," PRA 59, 1829 (1999)
"""

from __future__ import annotations

from typing import Any, Dict, List

from quantum_simulator import QuantumCircuit


def quantum_secret_share(secret_bit: int = 0) -> Dict[str, Any]:
    """
    Build the HBB99 quantum secret sharing circuit.

    Encodes ``secret_bit`` (0 or 1) into a 3-qubit GHZ state. Returns the
    full protocol circuit.

    Parameters
    ----------
    secret_bit : int
        The secret bit to share (0 or 1).

    Returns
    -------
    Dict[str, Any]
        ``{"circuit", "secret_bit", "n_parties", "method"}``
    """
    qc = QuantumCircuit(3)

    # Encode the secret bit on Alice's qubit (q0)
    if secret_bit == 1:
        qc.x(0)

    # Build GHZ state across all three parties
    qc.h(0)
    qc.cnot(0, 1)
    qc.cnot(1, 2)

    # Each party measures in the X basis (apply H before measurement)
    qc.h(0)
    qc.h(1)
    qc.h(2)

    return {
        "circuit": qc,
        "secret_bit": int(secret_bit),
        "n_parties": 3,
        "method": "hillery_buzek_berthiaume_99",
    }
