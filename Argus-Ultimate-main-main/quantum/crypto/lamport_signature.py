"""
Lamport one-time signatures.

Lamport signatures are post-quantum-secure: their security relies only on
the existence of a one-way function (typically a cryptographic hash), which
is unaffected by quantum computers.

Each signature key can sign exactly ONE message. For multi-message signing,
use a Merkle tree on top.

Trading use
-----------
ARGUS uses Lamport-style signatures (or their tree extensions) for signing
trade orders in a post-quantum-safe manner. This module provides the
single-message primitive.
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Dict, List, Tuple


# ═════════════════════════════════════════════════════════════════════════════
# Lamport key pair
# ═════════════════════════════════════════════════════════════════════════════


class LamportKeyPair:
    """
    A Lamport one-time signature key pair.

    The private key is 256 pairs of random 256-bit values (one pair per bit
    of a SHA-256 message digest). The public key is the SHA-256 hash of each
    private key value.

    Parameters
    ----------
    seed : bytes, optional
        Optional seed for deterministic key generation (testing only).
    """

    def __init__(self, seed: bytes = None) -> None:
        self.n_bits = 256  # SHA-256 digest length

        if seed is not None:
            # Deterministic key generation for testing
            rng = self._seeded_rng(seed)
        else:
            rng = secrets.token_bytes

        self.private_key: List[Tuple[bytes, bytes]] = []
        for i in range(self.n_bits):
            # Each bit needs 2 random 256-bit values (one for 0, one for 1)
            sk0 = rng(32)
            sk1 = rng(32)
            self.private_key.append((sk0, sk1))

        # Public key: hash of each private value
        self.public_key: List[Tuple[bytes, bytes]] = []
        for sk0, sk1 in self.private_key:
            pk0 = hashlib.sha256(sk0).digest()
            pk1 = hashlib.sha256(sk1).digest()
            self.public_key.append((pk0, pk1))

        self._used = False

    @staticmethod
    def _seeded_rng(seed: bytes):
        """Build a deterministic byte generator from a seed."""
        counter = [0]

        def _gen(n: int) -> bytes:
            counter[0] += 1
            h = hashlib.sha256(seed + counter[0].to_bytes(8, "big")).digest()
            while len(h) < n:
                h += hashlib.sha256(h).digest()
            return h[:n]

        return _gen


def lamport_sign(keypair: LamportKeyPair, message: bytes) -> List[bytes]:
    """
    Sign a message with a Lamport key pair.

    Each bit of the SHA-256(message) selects either sk0 or sk1 from the
    private key pair at that position.

    The keypair MUST NOT be reused (Lamport is a one-time signature).
    """
    if keypair._used:
        raise RuntimeError("Lamport keypair already used (one-time only)")

    digest = hashlib.sha256(message).digest()
    signature: List[bytes] = []
    for i in range(keypair.n_bits):
        bit = (digest[i // 8] >> (7 - (i % 8))) & 1
        sk0, sk1 = keypair.private_key[i]
        signature.append(sk1 if bit == 1 else sk0)

    keypair._used = True
    return signature


def lamport_verify(
    public_key: List[Tuple[bytes, bytes]],
    message: bytes,
    signature: List[bytes],
) -> bool:
    """
    Verify a Lamport signature against a public key.

    For each bit of SHA-256(message), check that hash(signature[i]) matches
    the corresponding public key element.
    """
    if len(signature) != len(public_key):
        return False

    digest = hashlib.sha256(message).digest()
    for i in range(len(public_key)):
        bit = (digest[i // 8] >> (7 - (i % 8))) & 1
        pk0, pk1 = public_key[i]
        sig_hash = hashlib.sha256(signature[i]).digest()
        expected = pk1 if bit == 1 else pk0
        if sig_hash != expected:
            return False
    return True
