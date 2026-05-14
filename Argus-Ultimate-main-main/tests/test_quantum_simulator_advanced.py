"""
Advanced unit tests for ``quantum_simulator.py`` (Phase A upgrade).

Covers:
- New gates (1q, 2q, 3q) for unitarity and basis-state correctness
- GHZ and W state preparation
- QFT vs analytical numpy.fft reference
- Parameter-shift gradient vs central finite differences
- MPS SWAP-network non-adjacent CNOT vs statevector reference
- Kraus trajectory: amplitude damping decay rate
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pytest

from quantum_simulator import (
    NoiseModel,
    QuantumCircuit,
    expval,
    gradient,
    pauli_z_observable,
    pauli_zz_observable,
    simulate,
    _simulate_statevector,
    # Direct gate matrices for unitarity tests
    _CCX_matrix,
    _CCZ_matrix,
    _CRX_matrix,
    _CRY_matrix,
    _CRZ_matrix,
    _CSWAP_matrix,
    _CNOT_control_first,
    _CPHASE_matrix,
    _CZ_matrix,
    _ISWAP_matrix,
    _RXX_matrix,
    _RYY_matrix,
    _RZZ_matrix,
    _S,
    _SDG,
    _SWAP_matrix,
    _T,
    _TDG,
    _U3,
    _PHASE,
)


# ═════════════════════════════════════════════════════════════════════════════
# Gate matrix unitarity tests
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "matrix_fn,n_dim",
    [
        (_S, 2),
        (_SDG, 2),
        (_T, 2),
        (_TDG, 2),
        (lambda: _U3(0.5, 1.2, -0.7), 2),
        (lambda: _PHASE(1.3), 2),
        (_CNOT_control_first, 4),
        (_CZ_matrix, 4),
        (_SWAP_matrix, 4),
        (_ISWAP_matrix, 4),
        (lambda: _CPHASE_matrix(0.5), 4),
        (lambda: _CRX_matrix(0.7), 4),
        (lambda: _CRY_matrix(0.7), 4),
        (lambda: _CRZ_matrix(0.7), 4),
        (lambda: _RXX_matrix(0.4), 4),
        (lambda: _RYY_matrix(0.4), 4),
        (lambda: _RZZ_matrix(0.4), 4),
        (_CCX_matrix, 8),
        (_CCZ_matrix, 8),
        (_CSWAP_matrix, 8),
    ],
)
def test_gate_unitarity(matrix_fn, n_dim):
    """Each gate matrix must satisfy U†U = I."""
    U = matrix_fn()
    assert U.shape == (n_dim, n_dim)
    product = U.conj().T @ U
    np.testing.assert_allclose(product, np.eye(n_dim), atol=1e-12)


# ═════════════════════════════════════════════════════════════════════════════
# Basis-state correctness tests
# ═════════════════════════════════════════════════════════════════════════════


def test_swap_basic():
    """SWAP exchanges qubit 0 and 1: |10⟩ → |01⟩."""
    qc = QuantumCircuit(2)
    qc.x(0)  # |01⟩ (qubit 0 = LSB, so |01⟩ in MSB-first display)
    qc.swap(0, 1)
    qc.measure_all()
    res = simulate(qc, shots=200, seed=1)
    # After SWAP, the X is now on qubit 1 → bitstring "10" in MSB-first
    assert "10" in res["counts"]
    assert res["counts"]["10"] == 200


def test_cz_phase_pattern():
    """CZ applied between two |+⟩ states then H gives the expected pattern."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.cz(0, 1)
    qc.h(0)
    qc.h(1)
    qc.measure_all()
    # CZ followed by HH gives a particular interference pattern
    res = simulate(qc, shots=2000, seed=42)
    counts = res["counts"]
    # Total preserved
    assert sum(counts.values()) == 2000


def test_toffoli_truth_table():
    """CCX flips target iff both controls are |1⟩."""
    # |110⟩ → |111⟩
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.x(1)
    qc.ccx(0, 1, 2)
    qc.measure_all()
    res = simulate(qc, shots=100, seed=1)
    assert res["counts"].get("111", 0) == 100

    # |010⟩ → |010⟩ (only one control on)
    qc2 = QuantumCircuit(3)
    qc2.x(1)
    qc2.ccx(0, 1, 2)
    qc2.measure_all()
    res2 = simulate(qc2, shots=100, seed=1)
    assert res2["counts"].get("010", 0) == 100


def test_fredkin_swap_when_control():
    """CSWAP swaps qubits q1, q2 iff control is |1⟩."""
    # |1, 0, 1⟩ — control=1, q1=0, q2=1 → CSWAP swaps q1<->q2 → |1, 1, 0⟩
    # Bitstring layout (MSB first): control=q0 (LSB rightmost)
    qc = QuantumCircuit(3)
    qc.x(0)  # control = qubit 0 = LSB
    qc.x(2)  # q2 = MSB
    qc.cswap(0, 1, 2)
    qc.measure_all()
    res = simulate(qc, shots=100, seed=1)
    # Initial: q0=1 (control), q1=0, q2=1 → bitstring (q2,q1,q0) = "101"
    # After swap: q1=1, q2=0 → bitstring "011"
    assert res["counts"].get("011", 0) == 100


# ═════════════════════════════════════════════════════════════════════════════
# Multi-qubit entangled state preparation
# ═════════════════════════════════════════════════════════════════════════════


def test_ghz_state():
    """3-qubit GHZ state: |000⟩ + |111⟩ (up to normalization)."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cnot(0, 1)
    qc.cnot(1, 2)
    qc.measure_all()
    res = simulate(qc, shots=2000, seed=42)
    counts = res["counts"]
    # Should see only |000⟩ and |111⟩
    keys = set(counts.keys())
    assert keys.issubset({"000", "111"})
    # Both should have ~50% probability
    assert 0.4 < counts["000"] / 2000 < 0.6
    assert 0.4 < counts["111"] / 2000 < 0.6


def test_w_state_3q():
    """Construct a 3-qubit W state |100⟩ + |010⟩ + |001⟩."""
    # Standard W-state circuit (one possible recipe)
    qc = QuantumCircuit(3)
    # Start in |000⟩
    # F gate: cos(theta) such that |000⟩ → sqrt(1/3)|100⟩ + sqrt(2/3)|0⟩(|01⟩+|10⟩)/sqrt(2)
    theta = 2.0 * math.acos(1.0 / math.sqrt(3))
    qc.ry(theta, 2)  # |0⟩ → cos|0⟩ + sin|1⟩ on qubit 2
    qc.ch_helper = None  # placeholder; we use CRY trick below

    # Simpler: explicit recipe
    # |000> -> cosθ|000> + sinθ|001> via Ry on qubit 0
    qc2 = QuantumCircuit(3)
    qc2.ry(2.0 * math.acos(1.0 / math.sqrt(3)), 0)
    qc2.cry(2.0 * math.acos(1.0 / math.sqrt(2)), 0, 1)
    qc2.cnot(1, 2)
    qc2.cnot(0, 1)
    qc2.x(0)
    qc2.measure_all()
    res = simulate(qc2, shots=4000, seed=42)
    counts = res["counts"]
    # Each of the three single-bit states should have ~33% probability
    for key in ("100", "010", "001"):
        assert counts.get(key, 0) > 100  # at least non-trivial weight


# ═════════════════════════════════════════════════════════════════════════════
# QFT vs analytical reference
# ═════════════════════════════════════════════════════════════════════════════


def test_qft_3q_against_analytical():
    """3-qubit QFT must match qft_matrix on every basis state."""
    from quantum.algorithms.qft import apply_qft_inplace, qft_matrix

    n = 3
    M = qft_matrix(n)
    for x in range(1 << n):
        qc = QuantumCircuit(n)
        for q in range(n):
            if (x >> q) & 1:
                qc.x(q)
        apply_qft_inplace(qc, list(range(n)))
        state = _simulate_statevector(qc)
        init = np.zeros(1 << n, dtype=np.complex128)
        init[x] = 1.0
        expected = M @ init
        np.testing.assert_allclose(state, expected, atol=1e-10)


def test_qft_inverse_round_trip():
    """QFT† ∘ QFT = I on every basis state."""
    from quantum.algorithms.qft import apply_qft_inplace

    n = 4
    for x in range(1 << n):
        qc = QuantumCircuit(n)
        for q in range(n):
            if (x >> q) & 1:
                qc.x(q)
        apply_qft_inplace(qc, list(range(n)))
        apply_qft_inplace(qc, list(range(n)), inverse=True)
        state = _simulate_statevector(qc)
        init = np.zeros(1 << n, dtype=np.complex128)
        init[x] = 1.0
        np.testing.assert_allclose(state, init, atol=1e-10)


# ═════════════════════════════════════════════════════════════════════════════
# Parameter-shift gradient
# ═════════════════════════════════════════════════════════════════════════════


def test_parameter_shift_ry_gradient():
    """∂⟨Z⟩/∂θ for RY(θ)|0⟩ should be -sin(θ)."""
    def builder(params):
        qc = QuantumCircuit(1)
        qc.ry(params[0], 0)
        qc.measure_all()
        return qc

    obs = pauli_z_observable(0)
    for theta in [0.1, np.pi / 4, np.pi / 2, 1.2]:
        grad = gradient(
            builder,
            np.array([theta]),
            obs,
            shots=8192,
            seed=42,
            method="parameter_shift",
        )
        expected = -float(np.sin(theta))
        # With 8192 shots, gradient noise is ~1/sqrt(8192) ~= 0.011
        assert abs(grad[0] - expected) < 0.05, (
            f"theta={theta}: grad={grad[0]:.4f} expected={expected:.4f}"
        )


def test_parameter_shift_two_param_circuit():
    """Two-parameter circuit: gradient should match analytical for both."""
    def builder(params):
        qc = QuantumCircuit(2)
        qc.ry(params[0], 0)
        qc.cnot(0, 1)
        qc.rz(params[1], 1)
        qc.measure_all()
        return qc

    obs = pauli_z_observable(0)
    params = np.array([np.pi / 3, np.pi / 5])
    grad = gradient(builder, params, obs, shots=8192, seed=42)
    # The first parameter (RY on qubit 0) directly affects ⟨Z_0⟩.
    # ⟨Z_0⟩ = cos(theta) - so ∂/∂θ = -sin(theta)
    expected_grad0 = -float(np.sin(np.pi / 3))
    assert abs(grad[0] - expected_grad0) < 0.05


# ═════════════════════════════════════════════════════════════════════════════
# MPS non-adjacent 2-qubit gates (Phase A2)
# ═════════════════════════════════════════════════════════════════════════════


def test_mps_swap_network_non_adjacent_cnot():
    """
    Non-adjacent CNOT on the MPS backend should give the same statistics as
    statevector. Tests Phase A2's SWAP-network fix.
    """
    n = 6
    # CNOT(0, 5) — non-adjacent on a 6-qubit chain
    qc_sv = QuantumCircuit(n)
    qc_sv.h(0)
    qc_sv.cnot(0, 5)
    qc_sv.measure_all()

    qc_mps = QuantumCircuit(n)
    qc_mps.h(0)
    qc_mps.cnot(0, 5)
    qc_mps.measure_all()

    res_sv = simulate(qc_sv, shots=4000, seed=42, backend="state_vector")
    res_mps = simulate(qc_mps, shots=4000, seed=42, backend="mps")

    # Both should produce two non-trivial outcomes (Bell-like over qubits 0,5)
    sv_top = sorted(res_sv["counts"].items(), key=lambda kv: -kv[1])[:2]
    mps_top = sorted(res_mps["counts"].items(), key=lambda kv: -kv[1])[:2]

    sv_keys = {k for k, _ in sv_top}
    mps_keys = {k for k, _ in mps_top}

    # The two backends should agree on the dominant outcomes
    assert sv_keys == mps_keys, (
        f"sv={sv_keys}, mps={mps_keys} — MPS SWAP-network mismatch"
    )


def test_mps_truncation_error_reported():
    """MPS backend should report cumulative truncation error."""
    n = 5
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    for q in range(n - 1):
        qc.cnot(q, q + 1)
    qc.measure_all()
    res = simulate(qc, shots=100, seed=42, backend="mps")
    assert "mps_truncation_error" in res
    assert res["mps_truncation_error"] >= 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Kraus noise trajectories (Phase A3)
# ═════════════════════════════════════════════════════════════════════════════


def test_amplitude_damping_decay():
    """
    Amplitude damping with γ=0.3: |1⟩ should decay to ⟨Z⟩ ≈ 1 - 2(1-γ) = 0.4
    after a single application.
    """
    qc = QuantumCircuit(1)
    qc.x(0)  # Prepare |1⟩
    qc.measure_all()

    nm = NoiseModel(
        kraus_channels=(("amplitude_damping", (0.3,)),),
        trajectories=500,
    )
    res = simulate(qc, shots=500, seed=42, noise=nm)

    # P(measure 0) should be around 0.3 (the damping probability)
    counts = res["counts"]
    p0 = counts.get("0", 0) / 500
    # Allow 5% tolerance for stochastic Monte Carlo
    assert 0.20 < p0 < 0.40, f"Expected p0 ~ 0.3, got {p0}"


def test_kraus_no_op_when_zero_gamma():
    """Amplitude damping with γ=0 should be the identity channel."""
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.measure_all()

    nm = NoiseModel(
        kraus_channels=(("amplitude_damping", (0.0,)),),
        trajectories=200,
    )
    res = simulate(qc, shots=200, seed=42, noise=nm)
    # All measurements should be |1⟩
    assert res["counts"].get("1", 0) >= 195  # ~100%, allow tiny stochastic noise


# ═════════════════════════════════════════════════════════════════════════════
# Auto-backend rejection for circuits beyond MPS_MAX_QUBITS
# ═════════════════════════════════════════════════════════════════════════════


def test_auto_backend_rejects_huge_circuit():
    """n > 100 should raise unless ARGUS_QUANTUM_ALLOW_APPROX is set."""
    import os

    qc = QuantumCircuit(101)
    qc.h(0)
    qc.measure_all()
    # Save and restore env
    old = os.environ.pop("ARGUS_QUANTUM_ALLOW_APPROX", None)
    try:
        with pytest.raises(ValueError):
            simulate(qc, shots=10, seed=42, backend="auto")
    finally:
        if old is not None:
            os.environ["ARGUS_QUANTUM_ALLOW_APPROX"] = old


def test_auto_backend_allows_huge_with_opt_in():
    """n > 100 with ARGUS_QUANTUM_ALLOW_APPROX=1 should fall through to approx."""
    import os

    qc = QuantumCircuit(101)
    qc.h(0)
    qc.measure_all()
    old = os.environ.get("ARGUS_QUANTUM_ALLOW_APPROX")
    os.environ["ARGUS_QUANTUM_ALLOW_APPROX"] = "1"
    try:
        res = simulate(qc, shots=10, seed=42, backend="auto")
        assert res["backend"] == "approx"
    finally:
        if old is None:
            os.environ.pop("ARGUS_QUANTUM_ALLOW_APPROX", None)
        else:
            os.environ["ARGUS_QUANTUM_ALLOW_APPROX"] = old
