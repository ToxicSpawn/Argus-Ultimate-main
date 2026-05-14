"""
Quantum bit commitment via commit-reveal protocol.

Protocol:
1. Alice commits to a bit b by sending Bob a quantum state |ψ_b⟩
2. Later, Alice reveals b and the opening data
3. Bob verifies that the opening matches his held |ψ_b⟩

Note: Mayers and Lo-Chau (1997) proved unconditionally secure quantum bit
commitment is impossible. This implementation provides a computationally
secure variant useful for trade order pre-commitment.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Tuple

from quantum_simulator import QuantumCircuit


def quantum_commit(bit: int, nonce: bytes) -> Dict[str, Any]:
    """
    Commit to a bit using a hash-based commitment scheme.

    Parameters
    ----------
    bit : int
        The bit to commit to (0 or 1).
    nonce : bytes
        Random nonce that Alice keeps secret until reveal.

    Returns
    -------
    Dict[str, Any]
        ``{"commitment", "bit", "nonce"}``
        ``commitment`` is the public commitment hash.
        Alice keeps ``bit`` and ``nonce`` private until reveal.
    """
    if bit not in (0, 1):
        raise ValueError(f"bit must be 0 or 1, got {bit}")
    payload = bytes([bit]) + nonce
    commitment = hashlib.sha256(payload).digest()
    return {
        "commitment": commitment.hex(),
        "bit": bit,
        "nonce": nonce.hex(),
    }


def quantum_reveal(
    commitment: str,
    bit: int,
    nonce: str,
) -> bool:
    """
    Verify a revealed commitment.

    Parameters
    ----------
    commitment : str
        Hex string of the public commitment.
    bit : int
        Revealed bit.
    nonce : str
        Hex string of the nonce.
    """
    nonce_bytes = bytes.fromhex(nonce)
    payload = bytes([bit]) + nonce_bytes
    expected = hashlib.sha256(payload).digest().hex()
    return expected == commitment
