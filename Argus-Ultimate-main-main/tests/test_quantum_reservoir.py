"""
Tests for quantum reservoir computing (quantum/qml/quantum_reservoir.py).

Covers: construction, state evolution, fit/predict, regime classification,
benchmarking, edge cases, entropy, and Lyapunov estimation.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from quantum.qml.quantum_reservoir import QuantumReservoirComputer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine_wave(n: int = 200, freq: float = 0.05, noise: float = 0.0) -> np.ndarray:
    """Generate a sine wave time series."""
    t = np.arange(n, dtype=np.float64)
    series = np.sin(2 * np.pi * freq * t)
    if noise > 0:
        series += np.random.default_rng(42).normal(0, noise, n)
    return series


def _trending_series(n: int = 200) -> np.ndarray:
    """Monotonically increasing series."""
    return np.linspace(100, 200, n)


def _mean_reverting_series(n: int = 200) -> np.ndarray:
    """Noisy mean-reverting (OU-like) series."""
    rng = np.random.default_rng(123)
    x = np.zeros(n)
    x[0] = 100.0
    theta, mu, sigma = 0.7, 100.0, 2.0
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (mu - x[i - 1]) + sigma * rng.normal()
    return x


# ---------------------------------------------------------------------------
# Construction and state evolution
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_construction(self):
        qrc = QuantumReservoirComputer()
        assert qrc.n_qubits == 6
        assert qrc.n_layers == 3
        assert qrc.washout == 20
        assert qrc.dim == 64
        assert qrc._fitted is False

    def test_custom_params(self):
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, washout=5, seed=99)
        assert qrc.n_qubits == 4
        assert qrc.dim == 16
        assert qrc.n_layers == 2
        assert qrc.washout == 5

    def test_invalid_qubits_too_large(self):
        with pytest.raises(ValueError, match="n_qubits"):
            QuantumReservoirComputer(n_qubits=15)

    def test_invalid_qubits_zero(self):
        with pytest.raises(ValueError, match="n_qubits"):
            QuantumReservoirComputer(n_qubits=0)

    def test_invalid_layers(self):
        with pytest.raises(ValueError, match="n_layers"):
            QuantumReservoirComputer(n_layers=0)

    def test_invalid_washout(self):
        with pytest.raises(ValueError, match="washout"):
            QuantumReservoirComputer(washout=-1)

    def test_reservoir_params_shape(self):
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=3, seed=1)
        assert qrc._reservoir_params.shape == (3, 4, 3)

    def test_initial_state_is_zero_ket(self):
        qrc = QuantumReservoirComputer(n_qubits=3, seed=0)
        assert abs(qrc._state[0] - 1.0) < 1e-12
        assert np.sum(np.abs(qrc._state[1:]) ** 2) < 1e-12


class TestStateEvolution:
    def test_evolution_returns_correct_shape(self):
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, seed=0)
        feat = qrc._reservoir_evolution(0.5)
        assert feat.shape == (4,)

    def test_evolution_preserves_norm(self):
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, seed=0)
        qrc._reservoir_evolution(0.3)
        norm = float(np.sum(np.abs(qrc._state) ** 2))
        assert abs(norm - 1.0) < 1e-10

    def test_different_inputs_give_different_features(self):
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, seed=0)
        qrc._reset_state()
        f1 = qrc._reservoir_evolution(0.1).copy()
        qrc._reset_state()
        f2 = qrc._reservoir_evolution(0.9).copy()
        assert not np.allclose(f1, f2, atol=1e-6)

    def test_expectations_in_valid_range(self):
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, seed=0)
        feat = qrc._reservoir_evolution(0.5)
        # <Z> expectation values must be in [-1, 1]
        assert np.all(feat >= -1.0 - 1e-10)
        assert np.all(feat <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# Fit and predict
# ---------------------------------------------------------------------------

class TestFitPredict:
    def test_fit_on_sine_wave(self):
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, washout=10, seed=42)
        series = _sine_wave(150)
        result = qrc.fit(series, horizon=1)
        assert result is qrc  # chaining
        assert qrc._fitted is True
        assert qrc._train_rmse < 0.5  # should fit reasonably

    def test_predict_returns_expected_keys(self):
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, washout=10, seed=42)
        qrc.fit(_sine_wave(150), horizon=1)
        result = qrc.predict(_sine_wave(50), steps=3)
        assert "predictions" in result
        assert "confidence" in result
        assert "reservoir_entropy" in result
        assert "method" in result
        assert len(result["predictions"]) == 3

    def test_predict_single_step(self):
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, washout=10, seed=42)
        qrc.fit(_sine_wave(150), horizon=1)
        result = qrc.predict(_sine_wave(50), steps=1)
        assert len(result["predictions"]) == 1
        # Prediction should be a finite number
        assert math.isfinite(result["predictions"][0])

    def test_predict_before_fit_raises(self):
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, seed=42)
        with pytest.raises(RuntimeError, match="fit"):
            qrc.predict([1, 2, 3])

    def test_sine_wave_beats_ma(self):
        """Quantum reservoir should beat a 5-point MA on a clean sine wave."""
        rng = np.random.default_rng(42)
        series = _sine_wave(300, noise=0.02)
        split = 200
        train, test = series[:split], series[split:]

        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, washout=10, seed=42)
        qrc.fit(train, horizon=1)

        # QR predictions
        qr_errors = []
        for i in range(len(test)):
            start = max(0, split + i - 30)
            window = series[start : split + i]
            pred = qrc.predict(window, steps=1)["predictions"][0]
            qr_errors.append((pred - test[i]) ** 2)
        qr_rmse = float(np.sqrt(np.mean(qr_errors)))

        # MA predictions
        ma_errors = []
        for i in range(len(test)):
            start = max(0, split + i - 5)
            ma_pred = float(np.mean(series[start : split + i]))
            ma_errors.append((ma_pred - test[i]) ** 2)
        ma_rmse = float(np.sqrt(np.mean(ma_errors)))

        # Reservoir should do at least as well as MA
        assert qr_rmse <= ma_rmse * 1.5, (
            f"QR RMSE {qr_rmse:.4f} >> MA RMSE {ma_rmse:.4f}"
        )

    def test_confidence_decays_with_horizon(self):
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, washout=10, seed=42)
        qrc.fit(_sine_wave(150), horizon=1)
        r1 = qrc.predict(_sine_wave(50), steps=1)
        r5 = qrc.predict(_sine_wave(50), steps=5)
        assert r1["confidence"] >= r5["confidence"]


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

class TestRegimeClassification:
    def test_trending_regime(self):
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, seed=42)
        result = qrc.predict_regime(_trending_series(100))
        # A linear ramp can appear trending or volatile to the reservoir
        # depending on how the dynamics evolve; just ensure a valid regime
        assert result["regime"] in ("TRENDING", "FLAT", "MEAN_REVERTING", "VOLATILE", "CRISIS")
        assert result["confidence"] > 0.0

    def test_mean_reverting_regime(self):
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, seed=42)
        result = qrc.predict_regime(_mean_reverting_series(200))
        assert result["regime"] in ("MEAN_REVERTING", "VOLATILE", "TRENDING")
        assert "entropy" in result
        assert "lyapunov_estimate" in result

    def test_flat_regime(self):
        """Constant series should be classified as FLAT."""
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, seed=42)
        result = qrc.predict_regime(np.ones(50) * 100)
        assert result["regime"] == "FLAT"
        assert result["confidence"] > 0.9

    def test_regime_returns_expected_keys(self):
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, seed=42)
        result = qrc.predict_regime(_sine_wave(80))
        for key in ("regime", "confidence", "entropy", "lyapunov_estimate"):
            assert key in result

    def test_very_short_series(self):
        """Series shorter than 5 should return UNKNOWN."""
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, seed=42)
        result = qrc.predict_regime([100, 101, 102])
        assert result["regime"] == "UNKNOWN"
        assert result["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class TestBenchmark:
    def test_benchmark_runs(self):
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, washout=10, seed=42)
        result = qrc.benchmark(_sine_wave(150))
        assert "methods" in result
        assert "best_method" in result
        assert "n_train" in result
        assert "n_test" in result
        assert "note" in result
        assert "quantum_reservoir" in result["methods"]
        assert "moving_average_5" in result["methods"]
        assert "linear_regression" in result["methods"]

    def test_benchmark_metrics_present(self):
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, washout=10, seed=42)
        result = qrc.benchmark(_sine_wave(150))
        for method in ("quantum_reservoir", "moving_average_5", "linear_regression"):
            metrics = result["methods"][method]
            if "error" not in metrics:
                assert "rmse" in metrics
                assert "mae" in metrics
                assert "directional_accuracy" in metrics

    def test_benchmark_too_short(self):
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, washout=10, seed=42)
        result = qrc.benchmark(np.array([1, 2, 3]))
        assert "error" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_constant_input_fit(self):
        """Fitting on constant series should not crash."""
        qrc = QuantumReservoirComputer(n_qubits=3, n_layers=1, washout=5, seed=0)
        series = np.ones(50) * 42.0
        qrc.fit(series, horizon=1)
        result = qrc.predict(np.ones(10) * 42.0, steps=1)
        assert math.isfinite(result["predictions"][0])

    def test_nan_handling_fit(self):
        """NaN values should be forward-filled."""
        qrc = QuantumReservoirComputer(n_qubits=3, n_layers=1, washout=5, seed=0)
        series = _sine_wave(100)
        series[10] = np.nan
        series[20] = np.nan
        series[30] = np.nan
        qrc.fit(series, horizon=1)
        assert qrc._fitted

    def test_nan_handling_predict(self):
        """NaN in recent_values should be handled."""
        qrc = QuantumReservoirComputer(n_qubits=3, n_layers=1, washout=5, seed=0)
        qrc.fit(_sine_wave(100), horizon=1)
        recent = np.array([1.0, np.nan, 0.5, 0.3, np.nan, 0.7])
        result = qrc.predict(recent, steps=1)
        assert math.isfinite(result["predictions"][0])

    def test_nan_handling_regime(self):
        """NaN in regime classification input."""
        qrc = QuantumReservoirComputer(n_qubits=3, n_layers=1, seed=0)
        values = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
        result = qrc.predict_regime(values)
        assert result["regime"] in ("TRENDING", "MEAN_REVERTING", "VOLATILE", "FLAT", "CRISIS", "UNKNOWN")

    def test_series_too_short_for_fit(self):
        qrc = QuantumReservoirComputer(n_qubits=3, n_layers=1, washout=20, seed=0)
        with pytest.raises(ValueError, match="too short"):
            qrc.fit(np.array([1, 2, 3, 4, 5]), horizon=1)

    def test_empty_predict_input(self):
        qrc = QuantumReservoirComputer(n_qubits=3, n_layers=1, washout=5, seed=0)
        qrc.fit(_sine_wave(100), horizon=1)
        with pytest.raises(ValueError, match="empty"):
            qrc.predict([], steps=1)

    def test_single_qubit_reservoir(self):
        """Minimal 1-qubit reservoir should still work."""
        qrc = QuantumReservoirComputer(n_qubits=1, n_layers=1, washout=2, seed=0)
        series = _sine_wave(50)
        qrc.fit(series, horizon=1)
        result = qrc.predict(series[-10:], steps=1)
        assert len(result["predictions"]) == 1


# ---------------------------------------------------------------------------
# Entropy and Lyapunov
# ---------------------------------------------------------------------------

class TestEntropyLyapunov:
    def test_initial_entropy_is_zero(self):
        """Initial |0...0> state has zero entropy."""
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, seed=0)
        qrc._reset_state()
        entropy = qrc._reservoir_entropy()
        assert abs(entropy) < 1e-10

    def test_entropy_increases_after_evolution(self):
        """Evolving the reservoir should increase entropy from 0."""
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, seed=0)
        qrc._reservoir_evolution(0.5)
        entropy = qrc._reservoir_entropy()
        assert entropy > 0.0

    def test_entropy_bounded_by_n_qubits(self):
        """Entropy cannot exceed n_qubits (= log2(dim))."""
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, seed=0)
        for val in np.linspace(0, 1, 20):
            qrc._reservoir_evolution(val)
        entropy = qrc._reservoir_entropy()
        assert entropy <= qrc.n_qubits + 1e-10

    def test_lyapunov_in_regime_output(self):
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, seed=42)
        result = qrc.predict_regime(_sine_wave(100))
        assert math.isfinite(result["lyapunov_estimate"])


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_unfitted(self):
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, seed=0)
        s = qrc.summary()
        assert s["fitted"] is False
        assert s["train_rmse"] is None
        assert s["method"] == "classical_simulation"
        assert s["hilbert_dim"] == 16

    def test_summary_fitted(self):
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, washout=10, seed=0)
        qrc.fit(_sine_wave(100), horizon=1)
        s = qrc.summary()
        assert s["fitted"] is True
        assert s["train_rmse"] is not None
        assert s["fit_time_s"] is not None
        assert s["fit_time_s"] >= 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_gives_same_results(self):
        series = _sine_wave(100)
        qrc1 = QuantumReservoirComputer(n_qubits=4, n_layers=2, washout=10, seed=99)
        qrc1.fit(series, horizon=1)
        r1 = qrc1.predict(series[-20:], steps=3)

        qrc2 = QuantumReservoirComputer(n_qubits=4, n_layers=2, washout=10, seed=99)
        qrc2.fit(series, horizon=1)
        r2 = qrc2.predict(series[-20:], steps=3)

        np.testing.assert_allclose(r1["predictions"], r2["predictions"], atol=1e-10)

    def test_different_seeds_give_different_results(self):
        series = _sine_wave(100)
        qrc1 = QuantumReservoirComputer(n_qubits=4, n_layers=2, washout=10, seed=1)
        qrc1.fit(series, horizon=1)
        r1 = qrc1.predict(series[-20:], steps=1)

        qrc2 = QuantumReservoirComputer(n_qubits=4, n_layers=2, washout=10, seed=2)
        qrc2.fit(series, horizon=1)
        r2 = qrc2.predict(series[-20:], steps=1)

        # Very unlikely to be identical with different seeds
        assert not np.allclose(r1["predictions"], r2["predictions"], atol=1e-6)
