"""
Quantum Phase Estimation (QPE) on the ARGUS in-repo simulator.

Given a unitary U with eigenstate |ψ⟩ such that U|ψ⟩ = e^(2πi φ) |ψ⟩, QPE
estimates the phase φ ∈ [0, 1).

Standard textbook algorithm:

    1. Prepare ``n_ancilla`` ancillas in the |+⟩ state via Hadamards.
    2. Prepare the target register in |ψ⟩.
    3. Apply controlled-U^(2^k) from ancilla k to target, for k = 0..n_ancilla-1.
    4. Apply inverse QFT to the ancilla register.
    5. Measure the ancilla register; phase ≈ measured_int / 2^n_ancilla.

This implementation uses only gates defined in ``quantum_simulator`` and runs
on the in-repo simulator. Because we simulate classically, any speedup is
academic — the value is architectural correctness and readiness for real
hardware backends.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from quantum_simulator import QuantumCircuit, simulate
from quantum.algorithms.qft import apply_qft_inplace


UnitaryBuilder = Callable[[QuantumCircuit, int, int], None]
"""
Signature: ``unitary_builder(qc, control_ancilla, power)`` should append
``C-U^(2^power)`` to circuit ``qc``, where ``control_ancilla`` is the ancilla
qubit acting as control and ``power`` specifies U^(2^power).

Example::

    def unitary_builder(qc, control, power):
        # U = Z (so U^(2^k) = Z^(2^k) = I if 2^k even, else Z)
        if (1 << power) % 2 == 1:
            qc.cz(control, target_qubit)
"""


def quantum_phase_estimation(
    unitary_builder: UnitaryBuilder,
    *,
    n_ancilla: int,
    n_target: int,
    target_prep: Optional[Callable[[QuantumCircuit, int], None]] = None,
    shots: int = 8192,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Estimate the phase of an eigenvalue of U.

    Parameters
    ----------
    unitary_builder : Callable[[QuantumCircuit, int, int], None]
        Function that appends ``C-U^(2^power)`` to a circuit given a
        control-ancilla index and power.
    n_ancilla : int
        Number of ancilla qubits (precision = 2^-n_ancilla).
    n_target : int
        Number of target-register qubits on which U acts.
    target_prep : Callable[[QuantumCircuit, int], None], optional
        Function that prepares the target register in an eigenstate. Takes
        ``(qc, target_offset)`` where the target register starts at qubit
        index ``target_offset``. If None, the target is left in |0...0⟩.
    shots : int
        Number of measurement shots.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        {
            "phase_estimate": float,           # most likely phase
            "phase_probability": float,        # probability of top outcome
            "counts": Dict[str, int],          # raw ancilla measurement counts
            "all_phases": List[Tuple[float, int]],  # (phase, count) sorted desc
        }
    """
    if n_ancilla <= 0:
        raise ValueError("n_ancilla must be >= 1")
    if n_target < 0:
        raise ValueError("n_target must be >= 0")

    total_qubits = int(n_ancilla) + int(n_target)
    qc = QuantumCircuit(total_qubits)

    # Ancilla register: qubits [0, n_ancilla). Target: [n_ancilla, total).
    ancilla_range = list(range(int(n_ancilla)))
    target_offset = int(n_ancilla)

    # Optional target preparation
    if target_prep is not None:
        target_prep(qc, target_offset)

    # Hadamards on all ancillas
    for a in ancilla_range:
        qc.h(a)

    # Controlled U^(2^k) from ancilla k
    for k in range(int(n_ancilla)):
        unitary_builder(qc, k, k)

    # Inverse QFT on ancilla register
    apply_qft_inplace(qc, ancilla_range, inverse=True)

    # Measure
    qc.measure_all()
    res = simulate(qc, shots=shots, seed=seed)
    counts = res["counts"]

    # Extract ancilla bits only. We sum over target register outcomes.
    # Bitstrings are MSB-first; qubit 0 is the rightmost char; the ancilla
    # register is qubits [0, n_ancilla), so the ancilla substring is the last
    # n_ancilla characters of each bitstring.
    ancilla_counts: Dict[str, int] = {}
    for bitstring, c in counts.items():
        anc = bitstring[-int(n_ancilla):]
        ancilla_counts[anc] = ancilla_counts.get(anc, 0) + c

    # Each ancilla bitstring → measured integer; phase = measured / 2^n_ancilla.
    # Bit order: within the substring, rightmost char is qubit 0 (LSB).
    phase_hist = []
    total_shots = sum(ancilla_counts.values())
    for anc_str, c in ancilla_counts.items():
        # Convert LSB-first (as in our bitstring ordering)
        value = 0
        nchars = len(anc_str)
        for i, ch in enumerate(anc_str):
            bit_qubit = nchars - 1 - i  # qubit index in ancilla register
            if ch == "1":
                value += 1 << bit_qubit
        phase = float(value) / float(1 << int(n_ancilla))
        phase_hist.append((phase, c))
    phase_hist.sort(key=lambda x: -x[1])

    top_phase = phase_hist[0][0] if phase_hist else 0.0
    top_prob = (phase_hist[0][1] / total_shots) if total_shots > 0 else 0.0

    return {
        "phase_estimate": top_phase,
        "phase_probability": top_prob,
        "counts": counts,
        "ancilla_counts": ancilla_counts,
        "all_phases": phase_hist,
        "n_ancilla": int(n_ancilla),
        "n_target": int(n_target),
    }


def build_phase_rotation_unitary(
    phi: float, target_qubit_offset: int = 0
) -> UnitaryBuilder:
    """
    Construct a unitary builder for U = PHASE(2π φ), which has eigenvalue
    e^(2πi φ) on |1⟩ and eigenvalue 1 on |0⟩.

    Useful for QPE test vectors: prepare the target in |1⟩, run QPE, and
    verify the measured phase matches φ.

    ``C-U^(2^k)`` is implemented as ``CPHASE(2π · φ · 2^k)`` from the control
    ancilla to the target qubit.
    """
    def builder(qc: QuantumCircuit, control_ancilla: int, power: int) -> None:
        angle = 2.0 * np.pi * float(phi) * float(1 << power)
        # Normalize angle to avoid huge accumulations
        angle = angle % (2.0 * np.pi)
        qc.cphase(angle, control_ancilla, target_qubit_offset + int(power == power) * 0 + 0)  # placeholder
    return builder


def _phase_rotation_builder(phi: float, target_qubit: int) -> UnitaryBuilder:
    """
    Actual phase-rotation unitary: C-PHASE(2π·φ·2^k) from ancilla to a fixed
    target qubit. The ``target_qubit`` should be passed absolutely (i.e.,
    include any ancilla offset).
    """
    def builder(qc: QuantumCircuit, control_ancilla: int, power: int) -> None:
        angle = (2.0 * np.pi * float(phi) * float(1 << power)) % (2.0 * np.pi)
        qc.cphase(angle, control_ancilla, target_qubit)
    return builder


def estimate_phase(
    phi_true: float,
    n_ancilla: int = 6,
    shots: int = 4096,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Convenience helper: estimate the phase φ of the PHASE(2πφ) gate via QPE.

    Used in tests and benchmarks. Returns the QPE result dict plus the true
    phase for comparison.
    """
    target = n_ancilla  # target qubit is the one after the ancillas
    builder = _phase_rotation_builder(phi_true, target_qubit=target)

    def prep(qc: QuantumCircuit, offset: int) -> None:
        # Target in |1⟩ (eigenstate of PHASE with eigenvalue e^(2πi φ))
        qc.x(offset)

    result = quantum_phase_estimation(
        builder,
        n_ancilla=n_ancilla,
        n_target=1,
        target_prep=prep,
        shots=shots,
        seed=seed,
    )
    result["phase_true"] = float(phi_true)
    result["phase_error"] = abs(result["phase_estimate"] - (float(phi_true) % 1.0))
    return result
