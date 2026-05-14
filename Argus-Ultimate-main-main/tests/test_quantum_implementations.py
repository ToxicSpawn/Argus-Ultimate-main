"""
Tests for quantum implementations: VQE, QMC backtesting, MPS tensor networks,
Boltzmann machine, reservoir computing, signal classifier, and benchmarks.

Covers 7 modules with 35+ tests.
"""

import numpy as np
import pytest


# ======================================================================
# 1. Quantum Boltzmann Machine (already real -- verify API works)
# ======================================================================

class TestQuantumBoltzmannMachine:
    """Tests for quantum.qml.quantum_boltzmann.QuantumBoltzmannMachine."""

    def test_import(self):
        from quantum.qml.quantum_boltzmann import QuantumBoltzmannMachine
        qbm = QuantumBoltzmannMachine(n_visible=5, n_hidden=3, seed=42)
        assert qbm.n_visible == 5
        assert qbm.n_hidden == 3

    def test_fit_and_generate(self):
        from quantum.qml.quantum_boltzmann import QuantumBoltzmannMachine
        qbm = QuantumBoltzmannMachine(n_visible=4, n_hidden=2, seed=42)
        data = np.random.RandomState(42).randn(50, 4)
        qbm.fit(data, epochs=5, batch_size=16)
        assert qbm._fitted is True
        samples = qbm.generate_samples(10)
        assert samples.shape == (10, 4)

    def test_free_energy(self):
        from quantum.qml.quantum_boltzmann import QuantumBoltzmannMachine
        qbm = QuantumBoltzmannMachine(n_visible=4, n_hidden=2, seed=42)
        data = np.random.RandomState(42).randn(30, 4)
        qbm.fit(data, epochs=3)
        fe = qbm._free_energy(np.array([[0.5, 0.5, 0.5, 0.5]]))
        assert isinstance(fe, np.ndarray)
        assert fe.shape == (1,)

    def test_anomaly_score(self):
        from quantum.qml.quantum_boltzmann import QuantumBoltzmannMachine
        qbm = QuantumBoltzmannMachine(n_visible=4, n_hidden=2, seed=42)
        data = np.random.RandomState(42).randn(30, 4)
        qbm.fit(data, epochs=3)
        score = qbm.anomaly_score(np.array([0.1, 0.2, 0.3, 0.4]))
        assert 0.0 <= score <= 1.0

    def test_reconstruct_via_generate(self):
        """Test that generate_samples works (reconstruction path)."""
        from quantum.qml.quantum_boltzmann import QuantumBoltzmannMachine
        qbm = QuantumBoltzmannMachine(n_visible=3, n_hidden=2, seed=42)
        data = np.random.RandomState(42).randn(20, 3)
        qbm.fit(data, epochs=3)
        samples = qbm.generate_samples(5, burn_in=5)
        assert samples.shape[0] == 5
        assert samples.shape[1] == 3


# ======================================================================
# 2. Quantum Reservoir Computer (already real -- verify API works)
# ======================================================================

class TestQuantumReservoirComputer:
    """Tests for quantum.qml.quantum_reservoir.QuantumReservoirComputer."""

    def test_import_and_init(self):
        from quantum.qml.quantum_reservoir import QuantumReservoirComputer
        qrc = QuantumReservoirComputer(n_qubits=4, n_layers=2, washout=5, seed=42)
        assert qrc.n_qubits == 4
        assert qrc.dim == 16

    def test_fit_and_predict(self):
        from quantum.qml.quantum_reservoir import QuantumReservoirComputer
        qrc = QuantumReservoirComputer(n_qubits=3, n_layers=1, washout=5, seed=42)
        ts = np.sin(np.linspace(0, 4 * np.pi, 100))
        qrc.fit(ts, horizon=1)
        assert qrc._fitted is True
        result = qrc.predict(ts[-20:], steps=2)
        assert "predictions" in result
        assert len(result["predictions"]) == 2
        assert "confidence" in result

    def test_encode_evolve_measure(self):
        from quantum.qml.quantum_reservoir import QuantumReservoirComputer
        qrc = QuantumReservoirComputer(n_qubits=3, n_layers=1, seed=42)
        qrc._reset_state()
        feat = qrc._reservoir_evolution(0.5)
        assert feat.shape == (3,)
        # Expectations should be in [-1, 1]
        assert np.all(np.abs(feat) <= 1.0 + 1e-10)

    def test_predict_regime(self):
        from quantum.qml.quantum_reservoir import QuantumReservoirComputer
        qrc = QuantumReservoirComputer(n_qubits=3, n_layers=1, seed=42)
        ts = np.sin(np.linspace(0, 4 * np.pi, 50))
        result = qrc.predict_regime(ts)
        assert "regime" in result
        assert result["regime"] in ("TRENDING", "MEAN_REVERTING", "VOLATILE", "CRISIS", "FLAT", "UNKNOWN")


# ======================================================================
# 3. Variational Quantum Eigensolver (NEW)
# ======================================================================

class TestVariationalQuantumEigensolver:
    """Tests for quantum.hybrid.variational.VariationalQuantumEigensolver."""

    def test_import_and_init(self):
        from quantum.hybrid.variational import VariationalQuantumEigensolver
        vqe = VariationalQuantumEigensolver(n_qubits=3, n_layers=2, seed=42)
        assert vqe.n_qubits == 3
        assert vqe.n_params == 6  # 2 layers * 3 qubits
        assert vqe.dim == 8

    def test_build_hamiltonian(self):
        from quantum.hybrid.variational import VariationalQuantumEigensolver
        vqe = VariationalQuantumEigensolver(n_qubits=2, seed=42)
        C = np.array([[1.0, 0.5], [0.5, 1.0]])
        H = vqe.build_hamiltonian(C)
        assert H.shape == (4, 4)
        # Check Hermitian
        assert np.allclose(H, H.conj().T)
        # Top-left block should be C
        assert np.allclose(np.real(H[:2, :2]), C)

    def test_ansatz_produces_normalized_state(self):
        from quantum.hybrid.variational import VariationalQuantumEigensolver
        vqe = VariationalQuantumEigensolver(n_qubits=3, n_layers=2, seed=42)
        params = np.random.RandomState(42).uniform(0, 2 * np.pi, vqe.n_params)
        state = vqe.ansatz(params)
        assert state.shape == (8,)
        norm = np.linalg.norm(state)
        assert abs(norm - 1.0) < 1e-10

    def test_cost_function(self):
        from quantum.hybrid.variational import VariationalQuantumEigensolver
        vqe = VariationalQuantumEigensolver(n_qubits=2, seed=42)
        C = np.array([[2.0, 0.5], [0.5, 1.0]])
        H = vqe.build_hamiltonian(C)
        params = np.zeros(vqe.n_params)
        energy = vqe.cost_function(params, H)
        assert isinstance(energy, float)

    def test_solve_finds_minimum_eigenvalue(self):
        from quantum.hybrid.variational import VariationalQuantumEigensolver
        C = np.array([[2.0, 0.5], [0.5, 1.0]])
        classical_min = float(np.linalg.eigvalsh(C)[0])

        vqe = VariationalQuantumEigensolver(n_qubits=2, n_layers=2, seed=42)
        result = vqe.solve(C, n_iter=200, n_restarts=3)

        assert "eigenvalue" in result
        assert "eigenvector" in result
        assert "classical_eigenvalue" in result
        assert result["classical_eigenvalue"] == pytest.approx(classical_min, abs=1e-6)
        # VQE should get reasonably close
        assert result["eigenvalue"] < classical_min + 0.5

    def test_portfolio_weights(self):
        from quantum.hybrid.variational import VariationalQuantumEigensolver
        C = np.array([[1.0, 0.3, 0.1],
                       [0.3, 1.0, 0.2],
                       [0.1, 0.2, 1.0]])
        vqe = VariationalQuantumEigensolver(n_qubits=2, n_layers=2, seed=42)
        weights = vqe.portfolio_weights(C)
        assert len(weights) == 3
        assert abs(weights.sum() - 1.0) < 1e-10
        assert np.all(weights >= 0)

    def test_solve_result_keys(self):
        from quantum.hybrid.variational import VariationalQuantumEigensolver
        C = np.eye(3)
        vqe = VariationalQuantumEigensolver(n_qubits=2, n_layers=1, seed=42)
        result = vqe.solve(C, n_iter=20, n_restarts=1)
        expected_keys = {"eigenvalue", "eigenvector", "params", "iterations",
                         "converged", "classical_eigenvalue", "error", "time_s", "method"}
        assert expected_keys.issubset(set(result.keys()))

    def test_summary(self):
        from quantum.hybrid.variational import VariationalQuantumEigensolver
        vqe = VariationalQuantumEigensolver(n_qubits=2, seed=42)
        s = vqe.summary()
        assert s["n_qubits"] == 2
        assert s["method"] == "classical_vqe_simulation"


# ======================================================================
# 4. Quantum Signal Classifier (already real -- verify API works)
# ======================================================================

class TestQuantumSignalClassifier:
    """Tests for quantum.qml.quantum_signal_classifier.QuantumSignalClassifier."""

    def test_import_and_init(self):
        from quantum.qml.quantum_signal_classifier import QuantumSignalClassifier
        qsc = QuantumSignalClassifier(n_features=4, n_qubits=4)
        assert qsc.n_features == 4

    def test_encode_features(self):
        from quantum.qml.quantum_signal_classifier import QuantumSignalClassifier
        qsc = QuantumSignalClassifier(n_features=3, n_qubits=3, n_layers=1)
        state = qsc._build_feature_map(np.array([0.1, 0.2, 0.3]))
        assert state.shape == (8,)
        norm = np.linalg.norm(state)
        assert abs(norm - 1.0) < 1e-8

    def test_fit_and_predict(self):
        from quantum.qml.quantum_signal_classifier import QuantumSignalClassifier
        rng = np.random.RandomState(42)
        X = rng.randn(30, 4)
        y = (X[:, 0] > 0).astype(int)  # Binary classification
        qsc = QuantumSignalClassifier(n_features=4, n_qubits=4, n_layers=1)
        qsc.fit(X, y)
        preds = qsc.predict(X[:5])
        assert len(preds) == 5

    def test_classify_returns_dict(self):
        from quantum.qml.quantum_signal_classifier import QuantumSignalClassifier
        rng = np.random.RandomState(42)
        X = rng.randn(30, 4)
        y = np.array([0, 1, 2] * 10)
        qsc = QuantumSignalClassifier(n_features=4, n_qubits=4, n_layers=1)
        qsc.fit(X, y)
        result = qsc.predict_signal_quality(X[0])
        assert "quality" in result
        assert "confidence" in result


# ======================================================================
# 5. Matrix Product State for Time Series (NEW)
# ======================================================================

class TestMatrixProductState:
    """Tests for quantum.tensor_networks.MatrixProductState."""

    def test_import_and_init(self):
        from quantum.tensor_networks import MatrixProductState
        mps = MatrixProductState(n_sites=8, bond_dim=4)
        assert mps.n_sites == 8
        assert mps.bond_dim == 4
        assert mps._fitted is False

    def test_compress(self):
        from quantum.tensor_networks import MatrixProductState
        mps = MatrixProductState(n_sites=10, bond_dim=4)
        data = np.sin(np.linspace(0, 2 * np.pi, 50))
        mps.compress(data)
        assert mps._fitted is True
        assert mps._tensors is not None
        assert len(mps._tensors) == 10

    def test_inner_product_self(self):
        from quantum.tensor_networks import MatrixProductState
        mps = MatrixProductState(n_sites=8, bond_dim=4)
        data = np.linspace(0, 1, 20)
        mps.compress(data)
        ip = mps.inner_product(mps)
        assert ip > 0  # Self-overlap should be positive

    def test_inner_product_different(self):
        from quantum.tensor_networks import MatrixProductState
        mps1 = MatrixProductState(n_sites=8, bond_dim=4)
        mps2 = MatrixProductState(n_sites=8, bond_dim=4)
        data1 = np.sin(np.linspace(0, 2 * np.pi, 20))
        data2 = np.cos(np.linspace(0, 2 * np.pi, 20))
        mps1.compress(data1)
        mps2.compress(data2)
        ip = mps1.inner_product(mps2)
        assert isinstance(ip, float)

    def test_predict_next(self):
        from quantum.tensor_networks import MatrixProductState
        mps = MatrixProductState(n_sites=8, bond_dim=4, physical_dim=4)
        data = np.linspace(10, 20, 30)
        mps.compress(data)
        probs = mps.predict_next()
        assert probs.shape == (4,)
        assert abs(probs.sum() - 1.0) < 1e-10
        assert np.all(probs >= 0)

    def test_predict_next_value(self):
        from quantum.tensor_networks import MatrixProductState
        mps = MatrixProductState(n_sites=8, bond_dim=4, physical_dim=4)
        data = np.linspace(10, 20, 30)
        mps.compress(data)
        val = mps.predict_next_value()
        assert isinstance(val, float)
        # Should be in reasonable range of original data
        assert 5 <= val <= 25

    def test_anomaly_score(self):
        from quantum.tensor_networks import MatrixProductState
        mps = MatrixProductState(n_sites=8, bond_dim=4)
        data = np.sin(np.linspace(0, 2 * np.pi, 50))
        mps.compress(data)

        # Normal data should have lower anomaly score
        normal = np.sin(np.linspace(2 * np.pi, 4 * np.pi, 20))
        score_normal = mps.anomaly_score(normal)
        assert 0.0 <= score_normal <= 1.0

    def test_anomaly_score_unfitted(self):
        from quantum.tensor_networks import MatrixProductState
        mps = MatrixProductState(n_sites=8, bond_dim=4)
        score = mps.anomaly_score(np.array([1, 2, 3]))
        assert score == 0.5

    def test_summary(self):
        from quantum.tensor_networks import MatrixProductState
        mps = MatrixProductState(n_sites=8, bond_dim=4)
        s = mps.summary()
        assert s["n_sites"] == 8
        assert s["method"] == "matrix_product_state"
        assert s["fitted"] is False

    def test_short_data_padded(self):
        from quantum.tensor_networks import MatrixProductState
        mps = MatrixProductState(n_sites=10, bond_dim=2)
        data = np.array([1.0, 2.0, 3.0])  # Shorter than n_sites
        mps.compress(data)
        assert mps._fitted is True


# ======================================================================
# 6. Quantum Backtest Accelerator (NEW)
# ======================================================================

class TestQuantumBacktestAccelerator:
    """Tests for quantum.backtesting.quantum_backtest.QuantumBacktestAccelerator."""

    def test_import_and_init(self):
        from quantum.backtesting.quantum_backtest import QuantumBacktestAccelerator
        qba = QuantumBacktestAccelerator(n_scenarios=100, seed=42)
        assert qba.n_scenarios == 100

    def test_run_qmc_scenarios(self):
        from quantum.backtesting.quantum_backtest import QuantumBacktestAccelerator
        qba = QuantumBacktestAccelerator(n_scenarios=100, seed=42)
        returns = np.random.RandomState(42).normal(0.001, 0.02, 200)
        result = qba.run_qmc_scenarios(returns)
        assert "mean_return" in result
        assert "std_return" in result
        assert "sharpe" in result
        assert "var_95" in result
        assert "cvar_95" in result
        assert "max_drawdown" in result
        assert result["n_scenarios"] == 100
        assert result["method"] == "quasi_monte_carlo_sobol"

    def test_run_qmc_with_strategy(self):
        from quantum.backtesting.quantum_backtest import QuantumBacktestAccelerator
        qba = QuantumBacktestAccelerator(n_scenarios=50, seed=42)
        returns = np.random.RandomState(42).normal(0.001, 0.02, 100)

        def simple_strategy(r):
            # Only take positive returns (momentum)
            signals = np.where(r > 0, r, 0)
            return float(np.sum(signals))

        result = qba.run_qmc_scenarios(returns, strategy_func=simple_strategy)
        assert result["mean_return"] >= 0  # Momentum strategy should be non-negative on average

    def test_compare_classical(self):
        from quantum.backtesting.quantum_backtest import QuantumBacktestAccelerator
        qba = QuantumBacktestAccelerator(n_scenarios=200, seed=42)
        returns = np.random.RandomState(42).normal(0.001, 0.02, 100)
        result = qba.compare_classical(returns, n_scenarios=200)
        assert "qmc" in result
        assert "classical" in result
        assert "comparison" in result
        assert "convergence" in result["comparison"]

    def test_insufficient_data(self):
        from quantum.backtesting.quantum_backtest import QuantumBacktestAccelerator
        qba = QuantumBacktestAccelerator(n_scenarios=100, seed=42)
        returns = np.array([0.01, 0.02])  # Too short
        result = qba.run_qmc_scenarios(returns)
        assert "error" in result or result["n_scenarios"] == 100

    def test_summary(self):
        from quantum.backtesting.quantum_backtest import QuantumBacktestAccelerator
        qba = QuantumBacktestAccelerator(n_scenarios=500, seed=42)
        s = qba.summary()
        assert s["n_scenarios"] == 500
        assert s["method"] == "quasi_monte_carlo"


# ======================================================================
# 7. Quantum Benchmarks (EXTENDED)
# ======================================================================

class TestQuantumBenchmarks:
    """Tests for quantum.benchmarks.QuantumBenchmarkSuite / QuantumBenchmark."""

    def test_import_alias(self):
        from quantum.benchmarks import QuantumBenchmark, QuantumBenchmarkSuite
        assert QuantumBenchmark is QuantumBenchmarkSuite

    def test_benchmark_var(self):
        from quantum.benchmarks import QuantumBenchmark
        bench = QuantumBenchmark()
        returns = np.random.RandomState(42).normal(-0.001, 0.02, 200)
        result = bench.benchmark_var(returns=returns, n_trials=3)
        assert "true_var_95" in result
        assert "qmc_mean_error" in result
        assert "mc_mean_error" in result
        assert "n_trials" in result

    def test_benchmark_portfolio(self):
        from quantum.benchmarks import QuantumBenchmark
        bench = QuantumBenchmark()
        C = np.array([[1.0, 0.3], [0.3, 1.0]])
        result = bench.benchmark_portfolio(correlation_matrix=C, n_trials=3)
        assert "vqe_mean_variance" in result
        assert "classical_mean_variance" in result
        assert "n_assets" in result

    def test_benchmark_pairs(self):
        from quantum.benchmarks import QuantumBenchmark
        bench = QuantumBenchmark()
        result = bench.benchmark_pairs(n_trials=2)
        assert "quantum_pairs_mean" in result
        assert "classical_pairs_mean" in result
        assert "verdict" in result

    def test_run_all(self):
        from quantum.benchmarks import QuantumBenchmark
        bench = QuantumBenchmark()
        result = bench.run_all()
        assert "var" in result
        assert "portfolio" in result
        assert "pairs" in result
        assert "overall" in result
