"""
BB84 quantum key distribution protocol.

Bennett-Brassard 1984: Alice and Bob establish a shared secret key by
sending random qubits in random bases. The security comes from the
no-cloning theorem — Eve cannot intercept and resend without disturbing
the qubits, and any disturbance is detected via the bit error rate.

Reference
---------
Bennett, Brassard, "Quantum cryptography: Public key distribution and coin
tossing," Theoretical Computer Science 560, 7 (1984/2014)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from quantum_simulator import QuantumCircuit, simulate


# ═════════════════════════════════════════════════════════════════════════════
# BB84 protocol
# ═════════════════════════════════════════════════════════════════════════════


def bb84_qkd(
    n_bits: int = 100,
    *,
    eavesdropper: bool = False,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Run a BB84 key distribution session.

    Parameters
    ----------
    n_bits : int
        Number of raw bits to transmit (sifted key will be ~n_bits/2).
    eavesdropper : bool
        If True, simulate Eve intercept-resend attack.

    Returns
    -------
    Dict[str, Any]
        ``{"alice_key", "bob_key", "matching_bases_count", "key_match_rate",
          "qber", "secure", "method"}``
    """
    rng = np.random.default_rng(seed)

    # Step 1: Alice generates random bits and bases
    alice_bits = rng.integers(0, 2, n_bits).astype(int)
    alice_bases = rng.integers(0, 2, n_bits).astype(int)  # 0 = Z basis, 1 = X basis

    # Step 2: Bob picks random measurement bases
    bob_bases = rng.integers(0, 2, n_bits).astype(int)

    # Step 3: For each bit, Alice prepares the qubit, optionally Eve intercepts,
    # and Bob measures.
    bob_results = []
    for i in range(n_bits):
        qc = QuantumCircuit(1)
        # Alice prepares the qubit
        if alice_bits[i] == 1:
            qc.x(0)
        if alice_bases[i] == 1:
            qc.h(0)

        # Eve intercepts (if enabled)
        if eavesdropper:
            eve_basis = int(rng.integers(0, 2))
            if eve_basis == 1:
                qc.h(0)
            qc.measure_all()
            res_eve = simulate(qc, shots=1, seed=int(rng.integers(0, 2**31 - 1)))
            eve_bit = int(next(iter(res_eve["counts"].keys())))
            # Re-prepare the qubit Eve measured
            qc = QuantumCircuit(1)
            if eve_bit == 1:
                qc.x(0)
            if eve_basis == 1:
                qc.h(0)

        # Bob measures in his basis
        if bob_bases[i] == 1:
            qc.h(0)
        qc.measure_all()
        res = simulate(qc, shots=1, seed=int(rng.integers(0, 2**31 - 1)))
        bob_bit = int(next(iter(res["counts"].keys())))
        bob_results.append(bob_bit)

    bob_results = np.array(bob_results, dtype=int)

    # Step 4: Sifting — keep only bits where bases matched
    matching = alice_bases == bob_bases
    sifted_alice = alice_bits[matching]
    sifted_bob = bob_results[matching]

    # Step 5: Compute the QBER (quantum bit error rate)
    n_sifted = len(sifted_alice)
    if n_sifted > 0:
        n_errors = int(np.sum(sifted_alice != sifted_bob))
        qber = n_errors / n_sifted
    else:
        qber = 0.0
        n_errors = 0

    # BB84 is secure if QBER < ~11% (theoretical bound)
    secure = qber < 0.11

    return {
        "alice_key": sifted_alice.tolist(),
        "bob_key": sifted_bob.tolist(),
        "n_raw_bits": n_bits,
        "n_sifted_bits": n_sifted,
        "matching_bases_count": int(np.sum(matching)),
        "n_errors": n_errors,
        "qber": float(qber),
        "key_match_rate": 1.0 - qber,
        "secure": secure,
        "eavesdropper_present": eavesdropper,
        "method": "bb84",
    }


def bb84_with_eavesdropping(n_bits: int = 100) -> Dict[str, Any]:
    """Run BB84 with an active eavesdropper to demonstrate detection."""
    return bb84_qkd(n_bits=n_bits, eavesdropper=True)
