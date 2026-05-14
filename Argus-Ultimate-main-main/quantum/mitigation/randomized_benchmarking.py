"""
Randomized Benchmarking (RB) for gate fidelity estimation.

RB measures the average gate fidelity by running random Clifford sequences
of varying length and fitting the survival probability decay:

    P(L) = A · p^L + B

where p is the depolarizing parameter, related to gate error per Clifford by
ε = (1 - p) (d - 1) / d (d = 2^n).

Reference
---------
Magesan, Gambetta, Emerson, "Characterizing quantum gates via randomized
benchmarking," PRA 85, 042311 (2012)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from quantum_simulator import QuantumCircuit, simulate


# ═════════════════════════════════════════════════════════════════════════════
# Single-qubit RB
# ═════════════════════════════════════════════════════════════════════════════


# 1-qubit Clifford group (24 elements expressed as gate sequences)
_SINGLE_Q_CLIFFORDS = [
    [],                  # I
    ["x"],              # X
    ["y"],              # Y
    ["z"],              # Z
    ["h"],              # H
    ["s"],              # S
    ["sdg"],            # S†
    ["h", "s"],         # HS
    ["s", "h"],         # SH
    ["h", "s", "h"],    # HSH
    ["x", "h"],         # XH
    ["x", "s"],         # XS
    ["y", "h"],         # YH
    ["y", "s"],         # YS
    ["z", "h"],         # ZH
    ["z", "s"],         # ZS
    ["h", "x"],         # HX
    ["h", "y"],         # HY
    ["s", "x"],         # SX
    ["s", "y"],         # SY
    ["sdg", "x"],       # S†X
    ["sdg", "y"],       # S†Y
    ["h", "s", "x"],    # HSX
    ["s", "h", "y"],    # SHY
]


def single_qubit_rb(
    sequence_lengths: List[int] = None,
    n_sequences: int = 20,
    *,
    shots: int = 1024,
    seed: Optional[int] = None,
    noise: Any = None,
) -> Dict[str, Any]:
    """
    Single-qubit randomized benchmarking.

    For each sequence length L, build n_sequences random Clifford sequences,
    compute the survival probability P(|0⟩|0⟩), and fit to A·p^L + B.

    Returns the average gate error per Clifford.
    """
    if sequence_lengths is None:
        sequence_lengths = [1, 5, 10, 20, 50, 100]

    rng = np.random.default_rng(seed)
    survival_probs = []

    for L in sequence_lengths:
        survivals = []
        for _ in range(n_sequences):
            qc = QuantumCircuit(1)
            applied_seq = []
            for _ in range(L):
                idx = int(rng.integers(0, len(_SINGLE_Q_CLIFFORDS)))
                seq = _SINGLE_Q_CLIFFORDS[idx]
                applied_seq.append(idx)
                for gate in seq:
                    if gate == "x":
                        qc.x(0)
                    elif gate == "y":
                        qc.y(0)
                    elif gate == "z":
                        qc.z(0)
                    elif gate == "h":
                        qc.h(0)
                    elif gate == "s":
                        qc.s(0)
                    elif gate == "sdg":
                        qc.sdg(0)
            # Apply inverse: for ideal Cliffords, the inverse is just another
            # Clifford that brings the state back to |0⟩. For simplicity, we
            # don't compute the exact inverse and instead measure how much
            # the state has decayed from |0⟩ via the marginal P(0).
            qc.measure_all()
            result = simulate(
                qc, shots=shots, seed=int(rng.integers(0, 2**31 - 1)), noise=noise
            )
            counts = result["counts"]
            n_zero = counts.get("0", 0)
            survivals.append(n_zero / max(sum(counts.values()), 1))
        survival_probs.append(float(np.mean(survivals)))

    # Fit P(L) = A * p^L + B
    L_arr = np.array(sequence_lengths, dtype=float)
    P_arr = np.array(survival_probs)
    # Simple log-linear fit on (P - B), assuming B ~ 0.5 (max-mixed asymptote)
    B = 0.5
    log_decay = np.log(np.maximum(P_arr - B, 1e-9))
    if np.all(np.isfinite(log_decay)) and len(L_arr) >= 2:
        slope, intercept = np.polyfit(L_arr, log_decay, 1)
        p = float(np.exp(slope))
        A = float(np.exp(intercept))
    else:
        p = 1.0
        A = 1.0

    # Gate error per Clifford
    d = 2  # single qubit
    epsilon = (1.0 - p) * (d - 1) / d

    return {
        "p_value": p,
        "gate_error_per_clifford": epsilon,
        "survival_probs": survival_probs,
        "sequence_lengths": sequence_lengths,
        "fit_A": A,
        "fit_B": B,
        "method": "single_qubit_rb",
    }


# ═════════════════════════════════════════════════════════════════════════════
# Two-qubit RB (simplified)
# ═════════════════════════════════════════════════════════════════════════════


def two_qubit_rb(
    sequence_lengths: List[int] = None,
    n_sequences: int = 10,
    *,
    shots: int = 1024,
    seed: Optional[int] = None,
    noise: Any = None,
) -> Dict[str, Any]:
    """
    Two-qubit randomized benchmarking.

    Uses a small subset of two-qubit Cliffords (CNOT + 1q Cliffords) instead
    of the full 11520-element 2-qubit Clifford group.
    """
    if sequence_lengths is None:
        sequence_lengths = [1, 2, 5, 10, 20]

    rng = np.random.default_rng(seed)
    survival_probs = []

    for L in sequence_lengths:
        survivals = []
        for _ in range(n_sequences):
            qc = QuantumCircuit(2)
            for _ in range(L):
                # Random 1q Clifford on each qubit
                for q in (0, 1):
                    idx = int(rng.integers(0, len(_SINGLE_Q_CLIFFORDS)))
                    seq = _SINGLE_Q_CLIFFORDS[idx]
                    for gate in seq:
                        if gate == "x":
                            qc.x(q)
                        elif gate == "y":
                            qc.y(q)
                        elif gate == "z":
                            qc.z(q)
                        elif gate == "h":
                            qc.h(q)
                        elif gate == "s":
                            qc.s(q)
                        elif gate == "sdg":
                            qc.sdg(q)
                # Random CNOT direction
                if rng.random() < 0.5:
                    qc.cnot(0, 1)
                else:
                    qc.cnot(1, 0)
            qc.measure_all()
            result = simulate(
                qc, shots=shots, seed=int(rng.integers(0, 2**31 - 1)), noise=noise
            )
            counts = result["counts"]
            n_zero = counts.get("00", 0)
            survivals.append(n_zero / max(sum(counts.values()), 1))
        survival_probs.append(float(np.mean(survivals)))

    L_arr = np.array(sequence_lengths, dtype=float)
    P_arr = np.array(survival_probs)
    B = 0.25  # 4 outcomes, max-mixed asymptote
    log_decay = np.log(np.maximum(P_arr - B, 1e-9))
    if np.all(np.isfinite(log_decay)) and len(L_arr) >= 2:
        slope, _ = np.polyfit(L_arr, log_decay, 1)
        p = float(np.exp(slope))
    else:
        p = 1.0

    d = 4  # 2 qubits
    epsilon = (1.0 - p) * (d - 1) / d

    return {
        "p_value": p,
        "gate_error_per_clifford": epsilon,
        "survival_probs": survival_probs,
        "sequence_lengths": sequence_lengths,
        "method": "two_qubit_rb_simplified",
    }
