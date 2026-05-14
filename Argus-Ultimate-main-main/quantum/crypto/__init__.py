"""
Quantum cryptography toolkit.

- ``bb84``: Bennett-Brassard 1984 quantum key distribution
- ``e91``: Ekert 1991 entanglement-based QKD
- ``lamport``: Lamport one-time signature (post-quantum-safe)
- ``commit_reveal``: quantum bit commitment

All protocols use only the in-repo simulator.
"""

from .bb84 import bb84_qkd, bb84_with_eavesdropping
from .e91 import e91_qkd, chsh_test
from .lamport_signature import LamportKeyPair, lamport_sign, lamport_verify
from .commit_reveal import quantum_commit, quantum_reveal

__all__ = [
    "bb84_qkd",
    "bb84_with_eavesdropping",
    "e91_qkd",
    "chsh_test",
    "LamportKeyPair",
    "lamport_sign",
    "lamport_verify",
    "quantum_commit",
    "quantum_reveal",
]
