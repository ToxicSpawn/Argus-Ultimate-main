"""
Secrets at Rest — basic encryption for SQLite DB files and sensitive data.

Provides XOR-based stream cipher encryption using PBKDF2-derived keys.
This is basic protection for data at rest, NOT military-grade encryption.
Uses only Python standard library (hashlib, os).

For production deployments requiring stronger guarantees, replace with
AES-256-GCM via the ``cryptography`` package.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

# PBKDF2 parameters
_PBKDF2_ITERATIONS = 100_000
_PBKDF2_HASH = "sha256"
_KEY_LENGTH = 32  # 256 bits
_SALT_LENGTH = 16  # 128 bits

# Magic header to identify encrypted files
_MAGIC = b"ARGUS_ENC_V1"


def derive_key(password: str, salt: bytes) -> bytes:
    """Derive an encryption key from a password using PBKDF2.

    Parameters
    ----------
    password : str
        User-supplied password or passphrase.
    salt : bytes
        Random salt (should be stored alongside the ciphertext).

    Returns
    -------
    bytes
        Derived key of ``_KEY_LENGTH`` bytes.
    """
    return hashlib.pbkdf2_hmac(
        _PBKDF2_HASH,
        password.encode("utf-8"),
        salt,
        _PBKDF2_ITERATIONS,
        dklen=_KEY_LENGTH,
    )


def _xor_stream(data: bytes, key: bytes) -> bytes:
    """XOR data against a repeating key stream.

    The key is extended by hashing successive blocks to avoid simple
    repeating-XOR weaknesses.  Each 32-byte block of the key stream
    is derived as SHA-256(key || block_index).
    """
    result = bytearray(len(data))
    block_size = _KEY_LENGTH
    for i in range(0, len(data), block_size):
        block_idx = i // block_size
        stream_block = hashlib.sha256(
            key + block_idx.to_bytes(8, "big")
        ).digest()
        chunk = data[i : i + block_size]
        for j, byte in enumerate(chunk):
            result[i + j] = byte ^ stream_block[j]
    return bytes(result)


def encrypt_file(path: Union[str, Path], key: str) -> None:
    """Encrypt a file at *path* in place.

    The encrypted file format:
        [12-byte magic] [16-byte salt] [encrypted data]

    Parameters
    ----------
    path : str or Path
        File to encrypt (overwritten in place).
    key : str
        Encryption password.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {path}")

    plaintext = filepath.read_bytes()

    # Don't double-encrypt
    if plaintext[:len(_MAGIC)] == _MAGIC:
        logger.warning("File %s is already encrypted, skipping", path)
        return

    salt = os.urandom(_SALT_LENGTH)
    derived = derive_key(key, salt)
    ciphertext = _xor_stream(plaintext, derived)

    filepath.write_bytes(_MAGIC + salt + ciphertext)
    logger.info("Encrypted file: %s (%d bytes)", path, len(plaintext))


def decrypt_file(path: Union[str, Path], key: str) -> bytes:
    """Decrypt a file encrypted with ``encrypt_file``.

    Parameters
    ----------
    path : str or Path
        Path to the encrypted file.
    key : str
        Encryption password (must match the one used for encryption).

    Returns
    -------
    bytes
        The decrypted file contents.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {path}")

    raw = filepath.read_bytes()
    magic_len = len(_MAGIC)

    if raw[:magic_len] != _MAGIC:
        raise ValueError(f"File {path} does not appear to be encrypted (bad magic)")

    salt = raw[magic_len : magic_len + _SALT_LENGTH]
    ciphertext = raw[magic_len + _SALT_LENGTH :]

    derived = derive_key(key, salt)
    plaintext = _xor_stream(ciphertext, derived)

    logger.info("Decrypted file: %s (%d bytes)", path, len(plaintext))
    return plaintext
