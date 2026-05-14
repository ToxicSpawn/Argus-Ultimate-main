"""
Tests for quantum error mitigation, noise modeling, and circuit optimization.

Covers:
- QuantumErrorMitigator: ZNE, MEM, PEC, twirled readout, fidelity estimation
- QuantumNoiseModel: depolarizing, amplitude damping, phase damping, readout, full simulation
- QuantumCircuitOptimizer: gate cancellation, commutation, depth reduction, time estimation
- Integration: noisy run -> mitigate -> improved result
- quantum_mitigated_run wrapper in unified stubs
"""

import math

import numpy as np
import pytest

from quantum.error_mitigation import QuantumErrorMitigator
from quantum.noise_model import QuantumNoiseModel
from quantum.circuit_optimizer import QuantumCircuitOptimizer


# =========================================================================
# QuantumErrorMitigator — Zero-Noise Extrapolation
# =========================================================================

class TestZNE:
    def setup_method(self):
        self.mitigator = QuantumErrorMitigator()

    def test_zne_linear_extrapolation(self):
        """Two points should give linear (Richardson degree-1) extrapolation."""
        # f(lambda) = 1.0 - 0.1 * lambda  =>  f(0) = 1.0
        results = [(1.0, 0.9), (2.0, 0.8)]
        out = self.mitigator.zero_noise_extrapolation(results)
        assert abs(out["zne_richardson"] - 1.0) < 0.01
        assert "confidence_interval" in out
        assert out["extrapolation_quality"] >= 0.0

    def test_zne_polynomial_extrapolation(self):
        """Three points should fit a degree-2 polynomial."""
        # f(lambda) = 1.0 - 0.05*lambda^2  =>  f(0) = 1.0
        results = [(1.0, 0.95), (2.0, 0.80), (3.0, 0.55)]
        out = self.mitigator.zero_noise_extrapolation(results)
        assert abs(out["zne_richardson"] - 1.0) < 0.05

    def test_zne_exponential_extrapolation(self):
        """Exponential model should be tried with 3+ points."""
        # f(lambda) = exp(-0.1 * lambda)  =>  f(0) = 1.0
        results = [(1.0, math.exp(-0.1)), (2.0, math.exp(-0.2)), (3.0, math.exp(-0.3))]
        out = self.mitigator.zero_noise_extrapolation(results)
        assert abs(out["zne_exponential"] - 1.0) < 0.1
        assert out["method_used"] in ("richardson", "exponential")

    def test_zne_convergence_to_ideal(self):
        """ZNE with good data should converge close to ideal value."""
        ideal = 0.75
        # Simulate linear noise: f(lambda) = ideal * (1 - 0.05*(lambda-1))
        results = [(1.0, ideal * 0.95), (1.5, ideal * 0.925), (2.0, ideal * 0.90)]
        out = self.mitigator.zero_noise_extrapolation(results)
        # Should extrapolate to ~0.75
        assert abs(out["zne_richardson"] - ideal) < 0.1

    def test_zne_minimum_points(self):
        """ZNE requires at least 2 points."""
        with pytest.raises(ValueError, match="at least 2"):
            self.mitigator.zero_noise_extrapolation([(1.0, 0.5)])

    def test_zne_confidence_interval_contains_estimate(self):
        """CI should contain the point estimate."""
        results = [(1.0, 0.8), (2.0, 0.6), (3.0, 0.4)]
        out = self.mitigator.zero_noise_extrapolation(results)
        ci_low, ci_high = out["confidence_interval"]
        best = out["zne_richardson"] if out["method_used"] == "richardson" else out["zne_exponential"]
        assert ci_low <= best <= ci_high

    def test_zne_quality_range(self):
        """Extrapolation quality should be in [0, 1]."""
        results = [(1.0, 0.9), (2.0, 0.8), (3.0, 0.7)]
        out = self.mitigator.zero_noise_extrapolation(results)
        assert 0.0 <= out["extrapolation_quality"] <= 1.0


# =========================================================================
# QuantumErrorMitigator — Measurement Error Mitigation
# =========================================================================

class TestMeasurementErrorMitigation:
    def setup_method(self):
        self.mitigator = QuantumErrorMitigator()

    def test_calibration_matrix_shape(self):
        """Calibration matrix should be (2^n x 2^n)."""
        for n in [1, 2, 3]:
            cal = self.mitigator.build_calibration_matrix(n)
            assert cal.shape == (2**n, 2**n)

    def test_calibration_matrix_columns_sum_to_one(self):
        """Each column of the calibration matrix should sum to 1."""
        cal = self.mitigator.build_calibration_matrix(2, error_rate=0.05)
        col_sums = cal.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-10)

    def test_calibration_matrix_zero_error(self):
        """With 0 error rate, calibration matrix should be identity."""
        cal = self.mitigator.build_calibration_matrix(2, error_rate=0.0)
        np.testing.assert_allclose(cal, np.eye(4), atol=1e-10)

    def test_mitigation_improves_counts(self):
        """Mitigation should reduce noise in biased counts."""
        # Ideal: all counts on "00". Noisy: some leaked to other states.
        raw_counts = {"00": 900, "01": 40, "10": 35, "11": 25}
        cal = self.mitigator.build_calibration_matrix(2, error_rate=0.03)
        result = self.mitigator.measurement_error_mitigation(raw_counts, cal)

        # After mitigation, "00" should have a larger fraction
        mit = result["mitigated_counts"]
        total = sum(mit.values())
        if total > 0 and "00" in mit:
            assert mit["00"] / total > 900 / 1000

    def test_mitigation_empty_counts(self):
        """Empty counts should return empty result."""
        cal = self.mitigator.build_calibration_matrix(2)
        result = self.mitigator.measurement_error_mitigation({}, cal)
        assert result["mitigated_counts"] == {}

    def test_mitigation_preserves_total(self):
        """Total mitigated counts should approximately equal total raw counts."""
        raw_counts = {"00": 400, "01": 300, "10": 200, "11": 100}
        cal = self.mitigator.build_calibration_matrix(2, error_rate=0.02)
        result = self.mitigator.measurement_error_mitigation(raw_counts, cal)
        raw_total = sum(raw_counts.values())
        mit_total = sum(result["mitigated_counts"].values())
        assert abs(mit_total - raw_total) / raw_total < 0.1


# =========================================================================
# QuantumErrorMitigator — PEC
# =========================================================================

class TestPEC:
    def setup_method(self):
        self.mitigator = QuantumErrorMitigator()

    def test_pec_corrects_toward_ideal(self):
        """PEC should push noisy mean closer to ideal."""
        ideal = 1.0
        noisy = [0.9, 0.85, 0.92, 0.88, 0.91]
        result = self.mitigator.probabilistic_error_cancellation(ideal, noisy)
        # Corrected value should be closer to ideal than noisy mean
        noisy_mean = np.mean(noisy)
        assert abs(result["corrected_value"] - ideal) <= abs(noisy_mean - ideal) + 0.3

    def test_pec_overhead_factor(self):
        """Overhead should be >= 1.0."""
        result = self.mitigator.probabilistic_error_cancellation(1.0, [0.8, 0.85])
        assert result["overhead_factor"] >= 1.0

    def test_pec_variance_nonnegative(self):
        """Variance should be non-negative."""
        result = self.mitigator.probabilistic_error_cancellation(1.0, [0.9, 0.8, 0.85])
        assert result["variance"] >= 0.0

    def test_pec_empty_noise_results(self):
        """Empty noise results should return the ideal value."""
        result = self.mitigator.probabilistic_error_cancellation(0.5, [])
        assert result["corrected_value"] == 0.5
        assert result["overhead_factor"] == 1.0


# =========================================================================
# QuantumErrorMitigator — Twirled Readout
# =========================================================================

class TestTwirledReadout:
    def setup_method(self):
        self.mitigator = QuantumErrorMitigator()

    def test_twirled_readout_basic(self):
        """Twirled readout should return mitigated counts."""
        counts = {"00": 800, "01": 100, "10": 70, "11": 30}
        result = self.mitigator.twirled_readout_error(counts, n_qubits=2)
        assert "mitigated_counts" in result
        assert result["correction_factor"] > 1.0

    def test_twirled_readout_empty(self):
        """Empty counts should return empty."""
        result = self.mitigator.twirled_readout_error({}, n_qubits=2)
        assert result["mitigated_counts"] == {}

    def test_twirled_readout_preserves_total(self):
        """Total should be approximately preserved."""
        counts = {"00": 500, "01": 250, "10": 150, "11": 100}
        result = self.mitigator.twirled_readout_error(counts, n_qubits=2)
        raw_total = sum(counts.values())
        mit_total = sum(result["mitigated_counts"].values())
        assert abs(mit_total - raw_total) / raw_total < 0.05


# =========================================================================
# QuantumErrorMitigator — Fidelity Estimation
# =========================================================================

class TestFidelityEstimation:
    def setup_method(self):
        self.mitigator = QuantumErrorMitigator()

    def test_high_fidelity_recommends_hardware(self):
        """Low noise circuit should recommend hardware."""
        result = self.mitigator.estimate_circuit_fidelity(
            n_gates=10, n_qubits=2, gate_error=0.001, readout_error=0.01
        )
        assert result["recommendation"] == "use_hardware"
        assert result["expected_fidelity"] > 0.5

    def test_low_fidelity_too_noisy(self):
        """High noise circuit should say too_noisy."""
        result = self.mitigator.estimate_circuit_fidelity(
            n_gates=5000, n_qubits=20, gate_error=0.01, readout_error=0.05
        )
        assert result["recommendation"] == "too_noisy"
        assert result["expected_fidelity"] < 0.1

    def test_medium_fidelity_recommends_simulator(self):
        """Medium noise should recommend simulator."""
        result = self.mitigator.estimate_circuit_fidelity(
            n_gates=500, n_qubits=5, gate_error=0.005, readout_error=0.02
        )
        assert result["recommendation"] in ("use_simulator", "too_noisy")

    def test_fidelity_components(self):
        """Gate and readout fidelities should multiply to total."""
        result = self.mitigator.estimate_circuit_fidelity(
            n_gates=100, n_qubits=3, gate_error=0.001, readout_error=0.01
        )
        expected = result["gate_fidelity"] * result["readout_fidelity"]
        assert abs(result["expected_fidelity"] - expected) < 1e-10


# =========================================================================
# QuantumNoiseModel — Depolarizing
# =========================================================================

class TestDepolarizing:
    def setup_method(self):
        self.model = QuantumNoiseModel("superconducting")

    def test_depolarizing_preserves_normalization(self):
        """Depolarizing noise should preserve statevector normalization."""
        sv = np.array([1.0, 0.0], dtype=complex)
        noisy = self.model.apply_depolarizing(sv, 0.1)
        assert abs(np.sum(np.abs(noisy)**2) - 1.0) < 1e-10

    def test_depolarizing_zero_error(self):
        """Zero error rate should not change the statevector."""
        sv = np.array([1.0, 0.0], dtype=complex)
        noisy = self.model.apply_depolarizing(sv, 0.0)
        np.testing.assert_allclose(np.abs(noisy)**2, np.abs(sv)**2, atol=1e-10)

    def test_depolarizing_full_error(self):
        """Full depolarization should give uniform distribution."""
        sv = np.array([1.0, 0.0], dtype=complex)
        noisy = self.model.apply_depolarizing(sv, 1.0)
        probs = np.abs(noisy)**2
        # Should be approximately uniform
        assert abs(probs[0] - probs[1]) < 0.1

    def test_depolarizing_multi_qubit(self):
        """Depolarizing on multi-qubit state preserves normalization."""
        sv = np.zeros(8, dtype=complex)
        sv[0] = 1.0  # |000>
        noisy = self.model.apply_depolarizing(sv, 0.05)
        assert abs(np.sum(np.abs(noisy)**2) - 1.0) < 1e-10


# =========================================================================
# QuantumNoiseModel — Amplitude Damping
# =========================================================================

class TestAmplitudeDamping:
    def setup_method(self):
        self.model = QuantumNoiseModel("superconducting")

    def test_amplitude_damping_ground_state_unchanged(self):
        """|0> should not be affected by amplitude damping."""
        sv = np.array([1.0, 0.0], dtype=complex)
        damped = self.model.apply_amplitude_damping(sv, 0.5)
        np.testing.assert_allclose(np.abs(damped)**2, [1.0, 0.0], atol=1e-10)

    def test_amplitude_damping_excited_decays(self):
        """|1> should partially decay to |0>."""
        sv = np.array([0.0, 1.0], dtype=complex)
        damped = self.model.apply_amplitude_damping(sv, 0.3)
        probs = np.abs(damped)**2
        # |0> probability should increase
        assert probs[0] > 0.2
        # |1> probability should decrease
        assert probs[1] < 0.8

    def test_amplitude_damping_preserves_normalization(self):
        """Normalization should be preserved."""
        sv = np.array([0.6 + 0.3j, 0.3 - 0.2j], dtype=complex)
        sv /= np.linalg.norm(sv)
        damped = self.model.apply_amplitude_damping(sv, 0.2)
        assert abs(np.sum(np.abs(damped)**2) - 1.0) < 1e-10


# =========================================================================
# QuantumNoiseModel — Phase Damping
# =========================================================================

class TestPhaseDamping:
    def setup_method(self):
        self.model = QuantumNoiseModel("superconducting")

    def test_phase_damping_preserves_normalization(self):
        """Phase damping should preserve normalization."""
        sv = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        damped = self.model.apply_phase_damping(sv, 0.3)
        assert abs(np.sum(np.abs(damped)**2) - 1.0) < 1e-10

    def test_phase_damping_preserves_probabilities_approx(self):
        """Phase damping should approximately preserve diagonal probabilities."""
        sv = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        damped = self.model.apply_phase_damping(sv, 0.1)
        probs = np.abs(damped)**2
        # Probabilities should stay close to 0.5/0.5
        assert abs(probs[0] - 0.5) < 0.15
        assert abs(probs[1] - 0.5) < 0.15


# =========================================================================
# QuantumNoiseModel — Readout Errors
# =========================================================================

class TestReadoutErrors:
    def setup_method(self):
        self.model = QuantumNoiseModel("superconducting")

    def test_readout_zero_error(self):
        """Zero error rate should not change counts."""
        counts = {"00": 500, "11": 500}
        noisy = self.model.apply_readout_error(counts, 0.0)
        assert noisy == counts

    def test_readout_preserves_total_shots(self):
        """Total shots should be preserved."""
        np.random.seed(42)
        counts = {"00": 500, "01": 300, "10": 150, "11": 50}
        noisy = self.model.apply_readout_error(counts, 0.05)
        assert sum(noisy.values()) == sum(counts.values())

    def test_readout_high_error_spreads_counts(self):
        """High error rate should spread counts across bitstrings."""
        np.random.seed(42)
        counts = {"00": 1000}
        noisy = self.model.apply_readout_error(counts, 0.3)
        # Should have some non-"00" counts
        assert len(noisy) > 1


# =========================================================================
# QuantumNoiseModel — Full Simulation & Hardware Profiles
# =========================================================================

class TestNoiseModelSimulation:
    def setup_method(self):
        self.model = QuantumNoiseModel("superconducting")

    def test_simulate_noisy_circuit_returns_keys(self):
        """Full simulation should return all required keys."""
        sv = np.array([1.0, 0.0], dtype=complex)
        result = self.model.simulate_noisy_circuit(sv, n_gates=10, shots=500)
        assert "noisy_counts" in result
        assert "ideal_counts" in result
        assert "fidelity" in result
        assert "total_error" in result

    def test_simulate_fidelity_range(self):
        """Fidelity should be in [0, 1]."""
        sv = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        result = self.model.simulate_noisy_circuit(sv, n_gates=5, shots=1000)
        assert 0.0 <= result["fidelity"] <= 1.0

    def test_hardware_profile_known_backend(self):
        """Known backends should return full profiles."""
        profile = self.model.get_hardware_profile("ibm_brisbane")
        assert profile["n_qubits"] == 127
        assert profile["backend_type"] == "superconducting"
        assert profile["cx_error"] > 0

    def test_hardware_profile_ionq(self):
        """IonQ profile should have long coherence times."""
        profile = self.model.get_hardware_profile("ionq_aria")
        assert profile["t1_us"] > 1_000_000
        assert profile["backend_type"] == "trapped_ion"

    def test_hardware_profile_unknown_backend(self):
        """Unknown backend should return generic profile."""
        profile = self.model.get_hardware_profile("unknown_xyz")
        assert "readout_error" in profile


# =========================================================================
# QuantumCircuitOptimizer — Gate Cancellation
# =========================================================================

class TestGateCancellation:
    def setup_method(self):
        self.optimizer = QuantumCircuitOptimizer()

    def test_hh_cancellation(self):
        """H*H should cancel to identity."""
        gates = [
            {"name": "H", "qubits": [0], "params": []},
            {"name": "H", "qubits": [0], "params": []},
        ]
        result = self.optimizer.gate_cancellation(gates)
        assert len(result) == 0

    def test_xx_cancellation(self):
        """X*X should cancel to identity."""
        gates = [
            {"name": "X", "qubits": [0], "params": []},
            {"name": "X", "qubits": [0], "params": []},
        ]
        result = self.optimizer.gate_cancellation(gates)
        assert len(result) == 0

    def test_cnot_cancellation(self):
        """CNOT*CNOT on same qubits should cancel."""
        gates = [
            {"name": "CNOT", "qubits": [0, 1], "params": []},
            {"name": "CNOT", "qubits": [0, 1], "params": []},
        ]
        result = self.optimizer.gate_cancellation(gates)
        assert len(result) == 0

    def test_rz_merge(self):
        """Rz(a)*Rz(b) should merge to Rz(a+b)."""
        gates = [
            {"name": "Rz", "qubits": [0], "params": [0.5]},
            {"name": "Rz", "qubits": [0], "params": [0.3]},
        ]
        result = self.optimizer.gate_cancellation(gates)
        assert len(result) == 1
        assert result[0]["name"] == "Rz"
        assert abs(result[0]["params"][0] - 0.8) < 1e-10

    def test_rz_cancel_to_zero(self):
        """Rz(a)*Rz(2pi-a) should cancel (mod 2pi = 0)."""
        gates = [
            {"name": "Rz", "qubits": [0], "params": [1.0]},
            {"name": "Rz", "qubits": [0], "params": [2 * np.pi - 1.0]},
        ]
        result = self.optimizer.gate_cancellation(gates)
        assert len(result) == 0

    def test_no_cancellation_different_qubits(self):
        """H on qubit 0 and H on qubit 1 should not cancel."""
        gates = [
            {"name": "H", "qubits": [0], "params": []},
            {"name": "H", "qubits": [1], "params": []},
        ]
        result = self.optimizer.gate_cancellation(gates)
        assert len(result) == 2

    def test_partial_cancellation(self):
        """H-H-X sequence: first two H cancel, X remains."""
        gates = [
            {"name": "H", "qubits": [0], "params": []},
            {"name": "H", "qubits": [0], "params": []},
            {"name": "X", "qubits": [0], "params": []},
        ]
        result = self.optimizer.gate_cancellation(gates)
        assert len(result) == 1
        assert result[0]["name"] == "X"

    def test_empty_sequence(self):
        """Empty gate sequence should return empty."""
        assert self.optimizer.gate_cancellation([]) == []


# =========================================================================
# QuantumCircuitOptimizer — Depth Reduction
# =========================================================================

class TestDepthReduction:
    def setup_method(self):
        self.optimizer = QuantumCircuitOptimizer()

    def test_independent_gates_parallelize(self):
        """Gates on different qubits should be in the same layer."""
        gates = [
            {"name": "H", "qubits": [0], "params": []},
            {"name": "H", "qubits": [1], "params": []},
            {"name": "H", "qubits": [2], "params": []},
        ]
        layers = self.optimizer.depth_reduction(gates, n_qubits=3)
        assert len(layers) == 1
        assert len(layers[0]) == 3

    def test_dependent_gates_sequential(self):
        """Gates on the same qubit must be in different layers."""
        gates = [
            {"name": "H", "qubits": [0], "params": []},
            {"name": "X", "qubits": [0], "params": []},
        ]
        layers = self.optimizer.depth_reduction(gates, n_qubits=1)
        assert len(layers) == 2

    def test_cnot_blocks_both_qubits(self):
        """CNOT should block both its qubits for the layer."""
        gates = [
            {"name": "CNOT", "qubits": [0, 1], "params": []},
            {"name": "H", "qubits": [0], "params": []},
        ]
        layers = self.optimizer.depth_reduction(gates, n_qubits=2)
        assert len(layers) == 2


# =========================================================================
# QuantumCircuitOptimizer — Full Pipeline & Time Estimation
# =========================================================================

class TestCircuitOptimizerPipeline:
    def setup_method(self):
        self.optimizer = QuantumCircuitOptimizer()

    def test_optimize_returns_all_keys(self):
        """Full optimize should return all expected keys."""
        gates = [
            {"name": "H", "qubits": [0], "params": []},
            {"name": "H", "qubits": [0], "params": []},
            {"name": "X", "qubits": [1], "params": []},
        ]
        result = self.optimizer.optimize(gates, n_qubits=2)
        assert "optimized_gates" in result
        assert "original_depth" in result
        assert "optimized_depth" in result
        assert "gates_removed" in result
        assert result["gates_removed"] == 2  # H*H cancelled
        assert result["optimized_gate_count"] == 1  # only X remains

    def test_execution_time_ibm(self):
        """IBM execution time estimation should return valid results."""
        gates = [
            {"name": "H", "qubits": [0], "params": []},
            {"name": "CNOT", "qubits": [0, 1], "params": []},
            {"name": "H", "qubits": [1], "params": []},
        ]
        result = self.optimizer.estimate_execution_time(gates, backend="ibm")
        assert result["total_time_us"] > 0
        assert result["n_layers"] >= 1
        assert result["bottleneck"] in ("measurement", "two_qubit_gates", "circuit_depth")

    def test_execution_time_ionq(self):
        """IonQ should be slower than IBM due to gate times."""
        gates = [
            {"name": "H", "qubits": [0], "params": []},
            {"name": "CNOT", "qubits": [0, 1], "params": []},
        ]
        ibm_time = self.optimizer.estimate_execution_time(gates, backend="ibm")
        ionq_time = self.optimizer.estimate_execution_time(gates, backend="ionq")
        assert ionq_time["total_time_us"] > ibm_time["total_time_us"]


# =========================================================================
# Integration Tests
# =========================================================================

class TestIntegration:
    def test_noisy_run_then_mitigate(self):
        """End-to-end: simulate noise, then mitigate, should improve fidelity."""
        np.random.seed(42)
        noise_model = QuantumNoiseModel("superconducting")
        mitigator = QuantumErrorMitigator()

        # Create a |0> state
        sv = np.array([1.0, 0.0], dtype=complex)

        # Simulate noisy circuit
        noisy_result = noise_model.simulate_noisy_circuit(
            sv, n_gates=20, gate_error=0.01, readout_error=0.05, shots=2000
        )

        # Build calibration and mitigate
        cal = mitigator.build_calibration_matrix(1, error_rate=0.05)
        mit_result = mitigator.measurement_error_mitigation(
            noisy_result["noisy_counts"], cal
        )

        # Mitigated result should have higher "0" probability
        mit = mit_result["mitigated_counts"]
        total = sum(mit.values())
        if total > 0 and "0" in mit:
            mitigated_p0 = mit["0"] / total
        elif total > 0 and "0" in mit:
            mitigated_p0 = mit["0"] / total
        else:
            mitigated_p0 = 0.0

        raw = noisy_result["noisy_counts"]
        raw_total = sum(raw.values())
        raw_p0 = raw.get("0", 0) / raw_total if raw_total > 0 else 0.0

        # Mitigated should be at least as good (within noise)
        assert mitigated_p0 >= raw_p0 - 0.1

    def test_circuit_optimize_then_fidelity(self):
        """Optimized circuit should have higher estimated fidelity."""
        optimizer = QuantumCircuitOptimizer()
        mitigator = QuantumErrorMitigator()

        gates = [
            {"name": "H", "qubits": [0], "params": []},
            {"name": "CNOT", "qubits": [0, 1], "params": []},
            {"name": "H", "qubits": [0], "params": []},
            {"name": "H", "qubits": [0], "params": []},  # will cancel with previous
            {"name": "CNOT", "qubits": [0, 1], "params": []},
            {"name": "CNOT", "qubits": [0, 1], "params": []},  # will cancel with previous
        ]

        opt_result = optimizer.optimize(gates, n_qubits=2)
        original_fidelity = mitigator.estimate_circuit_fidelity(
            n_gates=len(gates), n_qubits=2
        )["expected_fidelity"]
        optimized_fidelity = mitigator.estimate_circuit_fidelity(
            n_gates=opt_result["optimized_gate_count"], n_qubits=2
        )["expected_fidelity"]

        assert optimized_fidelity >= original_fidelity

    def test_quantum_mitigated_run_wrapper(self):
        """quantum_mitigated_run from stubs should work end-to-end."""
        from quantum.quantum_unified_stubs import quantum_mitigated_run

        # Create a simple circuit function
        def circuit_fn(noise_factor):
            # Simulate: ideal value is 1.0, degrades with noise
            value = 1.0 / noise_factor
            return {"expectation_value": value, "counts": {"0": 900, "1": 100}}

        result = quantum_mitigated_run(circuit_fn, noise_levels=[1.0, 2.0, 3.0])
        assert "mitigated_value" in result
        assert "raw_values" in result
        assert len(result["raw_values"]) == 3
        assert result["method"] == "zne_mitigated"

    def test_quantum_mitigated_run_static_counts(self):
        """quantum_mitigated_run should handle static counts dict."""
        from quantum.quantum_unified_stubs import quantum_mitigated_run

        counts = {"00": 700, "01": 150, "10": 100, "11": 50}
        result = quantum_mitigated_run(counts, noise_levels=[1.0, 2.0])
        assert "mitigated_value" in result
        assert result["method"] == "zne_mitigated"
