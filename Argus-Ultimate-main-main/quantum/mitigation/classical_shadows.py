"""
Classical shadow tomography (Huang, Kueng, Preskill 2020).

Classical shadows enable predicting many properties of an unknown quantum
state from a small number of measurements. The state is "shadow-encoded" by
applying random Clifford rotations followed by computational-basis measurements.

Reference
---------
Huang, Kueng, Preskill, "Predicting Many Properties of a Quantum System
from Very Few Measurements," Nature Physics 16, 1050 (2020)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from quantum_simulator import QuantumCircuit, simulate


# ═════════════════════════════════════════════════════════════════════════════
# Classical shadow tomography
# ═════════════════════════════════════════════════════════════════════════════


def classical_shadow_tomography(
    state_prep: Callable[[QuantumCircuit], None],
    n_qubits: int,
    *,
    n_shadows: int = 200,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build a classical shadow representation of an unknown quantum state.

    Each shadow is a randomly chosen Pauli-basis measurement followed by an
    inverse channel. The collection of shadows can be used to predict any
    Pauli observable expectation in O(M) time without state reconstruction.

    Parameters
    ----------
    state_prep : Callable[[QuantumCircuit], None]
        Function that prepares the unknown state on a fresh circuit.
    n_qubits : int
        Number of qubits.
    n_shadows : int
        Number of shadow snapshots to take.

    Returns
    -------
    Dict[str, Any]
        ``{"shadows", "n_qubits", "n_shadows", "method"}``
        where ``shadows`` is a list of (random_basis, measurement_outcome) pairs.
    """
    rng = np.random.default_rng(seed)
    shadows = []

    for s in range(n_shadows):
        # Pick a random Pauli basis for each qubit (X, Y, or Z)
        bases = rng.choice(["X", "Y", "Z"], size=n_qubits)

        qc = QuantumCircuit(n_qubits)
        state_prep(qc)
        # Rotate each qubit into its random basis
        for q in range(n_qubits):
            if bases[q] == "X":
                qc.h(q)
            elif bases[q] == "Y":
                qc.sdg(q)
                qc.h(q)
        qc.measure_all()

        result = simulate(qc, shots=1, seed=int(rng.integers(0, 2**31 - 1)))
        bitstring = next(iter(result["counts"].keys()))
        # Extract per-qubit outcome (qubit 0 = rightmost char)
        outcomes = []
        for q in range(n_qubits):
            bit = bitstring[len(bitstring) - 1 - q] if q < len(bitstring) else "0"
            outcomes.append(int(bit))
        shadows.append({"bases": list(bases), "outcomes": outcomes})

    return {
        "shadows": shadows,
        "n_qubits": n_qubits,
        "n_shadows": n_shadows,
        "method": "classical_shadow_random_pauli",
    }


def shadow_estimator(
    shadows_dict: Dict[str, Any],
    pauli_observable: str,
) -> float:
    """
    Estimate ⟨P⟩ for a Pauli observable using a precomputed classical shadow.

    For each shadow, the contribution to ⟨P⟩ is:
        contribution = ∏_q {3 × ⟨σ_q⟩_shadow if basis matches P_q else 0}

    where σ_q is the measured Pauli on qubit q.
    """
    shadows = shadows_dict["shadows"]
    n_qubits = shadows_dict["n_qubits"]
    if len(pauli_observable) != n_qubits:
        raise ValueError(
            f"Pauli observable length {len(pauli_observable)} != n_qubits {n_qubits}"
        )

    contributions = []
    for shadow in shadows:
        bases = shadow["bases"]
        outcomes = shadow["outcomes"]
        contrib = 1.0
        for q in range(n_qubits):
            p = pauli_observable[q]
            if p == "I":
                # Identity: factor of 1 (always trace)
                continue
            if bases[q] != p:
                # Mismatched basis → this shadow doesn't contribute
                contrib = 0.0
                break
            # Matched basis: factor of 3 × measurement sign
            sign = 1.0 if outcomes[q] == 0 else -1.0
            contrib *= 3.0 * sign
        contributions.append(contrib)

    return float(np.mean(contributions))
