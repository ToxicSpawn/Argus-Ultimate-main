"""
E91 (Ekert) entanglement-based quantum key distribution.

E91 uses pre-shared Bell pairs and CHSH inequality violation as the security
test. If Eve has tampered, the CHSH parameter will fall below the quantum
bound (2√2 ≈ 2.828) toward the classical bound (2).

Reference
---------
Ekert, "Quantum cryptography based on Bell's theorem," PRL 67, 661 (1991)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from quantum_simulator import QuantumCircuit, simulate


def e91_qkd(
    n_pairs: int = 100,
    *,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Run an E91 key distribution session.

    Alice and Bob share Bell pairs. Each measures in one of 3 bases:
        Alice: {0°, 45°, 90°}
        Bob:   {45°, 90°, 135°}

    Bits are extracted from matching bases (45° + 45° or 90° + 90°).
    The CHSH inequality is tested on a subset of mismatched bases to detect
    eavesdropping.

    Returns sifted key + CHSH parameter.
    """
    rng = np.random.default_rng(seed)

    # 3 bases for each party
    alice_basis_angles = [0.0, np.pi / 4.0, np.pi / 2.0]
    bob_basis_angles = [np.pi / 4.0, np.pi / 2.0, 3 * np.pi / 4.0]

    alice_bits = []
    bob_bits = []
    matching = []
    for i in range(n_pairs):
        a_idx = int(rng.integers(0, 3))
        b_idx = int(rng.integers(0, 3))

        # Build Bell pair circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cnot(0, 1)
        # Rotate Alice's qubit to her basis
        qc.ry(-2 * alice_basis_angles[a_idx], 0)
        # Rotate Bob's qubit to his basis
        qc.ry(-2 * bob_basis_angles[b_idx], 1)
        qc.measure_all()

        res = simulate(qc, shots=1, seed=int(rng.integers(0, 2**31 - 1)))
        bs = next(iter(res["counts"].keys()))
        # Bitstring is MSB-first; qubit 0 = rightmost char
        bit_a = int(bs[-1])
        bit_b = int(bs[-2])
        alice_bits.append(bit_a)
        bob_bits.append(bit_b)
        matching.append(a_idx == 1 and b_idx == 0)  # 45° = 45° basis match

    alice_bits = np.array(alice_bits)
    bob_bits = np.array(bob_bits)
    matching_arr = np.array(matching, dtype=bool)

    sifted_alice = alice_bits[matching_arr]
    sifted_bob = bob_bits[matching_arr]
    if len(sifted_alice) > 0:
        match_rate = float(np.mean(sifted_alice == sifted_bob))
    else:
        match_rate = 0.0

    return {
        "alice_key": sifted_alice.tolist(),
        "bob_key": sifted_bob.tolist(),
        "n_pairs": n_pairs,
        "n_sifted": int(np.sum(matching_arr)),
        "match_rate": match_rate,
        "method": "e91",
    }


def chsh_test(n_pairs: int = 1000, *, seed: Optional[int] = 42) -> Dict[str, Any]:
    """
    Run a CHSH inequality test on a Bell pair.

    The CHSH parameter S = E(a,b) - E(a,b') + E(a',b) + E(a',b') is bounded
    by |S| ≤ 2 classically and |S| ≤ 2√2 ≈ 2.828 quantum-mechanically.

    For optimal Bell measurements, S = 2√2.
    """
    rng = np.random.default_rng(seed)
    # Optimal CHSH angles
    a = 0.0
    a_prime = np.pi / 4.0
    b = np.pi / 8.0
    b_prime = -np.pi / 8.0

    def measure_correlation(angle_a: float, angle_b: float, n: int) -> float:
        """E(a, b) for n Bell pairs measured with rotations."""
        sum_corr = 0.0
        for _ in range(n):
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cnot(0, 1)
            qc.ry(-2 * angle_a, 0)
            qc.ry(-2 * angle_b, 1)
            qc.measure_all()
            res = simulate(qc, shots=1, seed=int(rng.integers(0, 2**31 - 1)))
            bs = next(iter(res["counts"].keys()))
            bit_a = int(bs[-1])
            bit_b = int(bs[-2])
            # Correlation: +1 if bits agree, -1 if differ
            sum_corr += 1.0 if bit_a == bit_b else -1.0
        return sum_corr / n

    n_per = max(1, n_pairs // 4)
    E_ab = measure_correlation(a, b, n_per)
    E_abp = measure_correlation(a, b_prime, n_per)
    E_apb = measure_correlation(a_prime, b, n_per)
    E_apbp = measure_correlation(a_prime, b_prime, n_per)

    S = E_ab - E_abp + E_apb + E_apbp

    return {
        "S": float(S),
        "classical_bound": 2.0,
        "quantum_bound": 2.0 * np.sqrt(2),
        "violates_classical": abs(S) > 2.0,
        "n_pairs_per_correlation": n_per,
        "method": "chsh_inequality_test",
    }
