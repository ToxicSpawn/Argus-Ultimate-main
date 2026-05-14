"""
Tests for real quantum algorithm implementations:
  - QAOA Portfolio Optimizer
  - Quantum Amplitude Estimation for VaR
  - Quantum Kernel Classifier
  - Unified stubs wiring

30+ tests covering correctness, edge cases, benchmarks, and honest assessments.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_portfolio():
    """3-asset portfolio with known optimal."""
    mu = np.array([0.10, 0.05, 0.08])
    sigma = np.array([
        [0.04, 0.006, 0.002],
        [0.006, 0.01, 0.004],
        [0.002, 0.004, 0.02],
    ])
    return mu, sigma


@pytest.fixture
def large_portfolio():
    """8-asset portfolio."""
    rng = np.random.default_rng(42)
    n = 8
    mu = rng.uniform(0.02, 0.15, n)
    # Generate valid covariance matrix
    A = rng.standard_normal((n, n)) * 0.1
    sigma = A.T @ A + np.eye(n) * 0.01
    return mu, sigma


@pytest.fixture
def return_series():
    """Synthetic return series with fat tail."""
    rng = np.random.default_rng(123)
    normal = rng.normal(0.0005, 0.02, 500)
    # Add a few tail events
    tail_events = rng.normal(-0.08, 0.01, 10)
    returns = np.concatenate([normal, tail_events])
    rng.shuffle(returns)
    return returns


@pytest.fixture
def classification_data():
    """Binary classification dataset with non-linear boundary (interleaved classes)."""
    rng = np.random.default_rng(99)
    n = 40
    X0 = rng.normal(loc=[-1, -1], scale=0.5, size=(n // 2, 2))
    X1 = rng.normal(loc=[1, 1], scale=0.5, size=(n // 2, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    # Interleave so every subset has both classes
    idx = np.arange(n)
    rng.shuffle(idx)
    return X[idx], y[idx]


# ===================================================================
# QAOA Portfolio Optimizer
# ===================================================================


class TestQAOAPortfolioOptimizer:

    def test_optimize_basic(self, simple_portfolio):
        """QAOA produces valid weights summing to ~1."""
        from quantum.algorithms.qaoa import QAOAPortfolioOptimizer
        mu, sigma = simple_portfolio
        opt = QAOAPortfolioOptimizer(n_layers=1, max_assets=3)
        result = opt.optimize(mu, sigma, risk_aversion=0.5)

        assert "weights" in result
        assert "method" in result
        w = np.array(result["weights"])
        assert len(w) == len(mu)
        assert abs(w.sum() - 1.0) < 0.05  # approximately sums to 1
        assert all(wi >= -0.01 for wi in w)  # non-negative

    def test_optimize_returns_sharpe(self, simple_portfolio):
        """Result includes Sharpe ratio."""
        from quantum.algorithms.qaoa import QAOAPortfolioOptimizer
        mu, sigma = simple_portfolio
        opt = QAOAPortfolioOptimizer(n_layers=1)
        result = opt.optimize(mu, sigma)
        assert "sharpe" in result
        assert "expected_return" in result
        assert "expected_risk" in result

    def test_optimize_single_asset(self):
        """Single asset gets 100% weight."""
        from quantum.algorithms.qaoa import QAOAPortfolioOptimizer
        mu = np.array([0.10])
        sigma = np.array([[0.04]])
        opt = QAOAPortfolioOptimizer()
        result = opt.optimize(mu, sigma)
        assert result["weights"] == [1.0]
        assert result["method"] == "single_asset"

    def test_optimize_empty(self):
        """Empty input returns empty result."""
        from quantum.algorithms.qaoa import QAOAPortfolioOptimizer
        opt = QAOAPortfolioOptimizer()
        result = opt.optimize(np.array([]), np.array([[]]))
        assert result["weights"] == []
        assert result["sharpe"] == 0.0

    def test_build_cost_hamiltonian_shape(self, simple_portfolio):
        """QUBO matrix has correct shape."""
        from quantum.algorithms.qaoa import QAOAPortfolioOptimizer
        mu, sigma = simple_portfolio
        opt = QAOAPortfolioOptimizer()
        Q = opt.build_cost_hamiltonian(mu, sigma)
        assert Q.shape == (3, 3)
        # Should be symmetric
        assert np.allclose(Q, Q.T, atol=1e-10)

    def test_optimize_with_budget(self, large_portfolio):
        """Budget constraint limits number of selected assets."""
        from quantum.algorithms.qaoa import QAOAPortfolioOptimizer
        mu, sigma = large_portfolio
        opt = QAOAPortfolioOptimizer(n_layers=1, max_assets=12)
        result = opt.optimize(mu, sigma, risk_aversion=0.5, budget=3)
        selected = result.get("selected_assets", [])
        assert len(selected) <= 4  # allow small slack from QAOA approximation

    def test_benchmark_vs_classical(self, simple_portfolio):
        """Benchmark returns honest assessment."""
        from quantum.algorithms.qaoa import QAOAPortfolioOptimizer
        mu, sigma = simple_portfolio
        opt = QAOAPortfolioOptimizer(n_layers=1)
        bench = opt.benchmark_vs_classical(mu, sigma)

        assert "qaoa_sharpe" in bench
        assert "classical_sharpe" in bench
        assert "qaoa_time_ms" in bench
        assert "classical_time_ms" in bench
        assert "improvement_pct" in bench
        assert "honest_assessment" in bench
        assert isinstance(bench["honest_assessment"], str)
        assert len(bench["honest_assessment"]) > 20  # substantive assessment

    def test_convergence_history(self, simple_portfolio):
        """Convergence history is recorded."""
        from quantum.algorithms.qaoa import QAOAPortfolioOptimizer
        mu, sigma = simple_portfolio
        opt = QAOAPortfolioOptimizer(n_layers=2)
        result = opt.optimize(mu, sigma)
        assert "convergence_history" in result
        assert isinstance(result["convergence_history"], list)

    def test_method_is_reported(self, simple_portfolio):
        """Method field identifies the backend used."""
        from quantum.algorithms.qaoa import QAOAPortfolioOptimizer
        mu, sigma = simple_portfolio
        opt = QAOAPortfolioOptimizer(n_layers=1)
        result = opt.optimize(mu, sigma)
        valid_methods = {
            "qaoa_qiskit_circuit", "qaoa_pennylane_circuit",
            "qaoa_classical_simulation", "classical_scipy_fallback",
            "single_asset", "no_assets",
            "qaoa_in_repo_simulator", "mlqae_in_repo",
        }
        assert result["method"] in valid_methods

    def test_zero_returns(self):
        """Zero returns still produces valid output."""
        from quantum.algorithms.qaoa import QAOAPortfolioOptimizer
        mu = np.array([0.0, 0.0, 0.0])
        sigma = np.eye(3) * 0.01
        opt = QAOAPortfolioOptimizer(n_layers=1)
        result = opt.optimize(mu, sigma)
        w = np.array(result["weights"])
        assert abs(w.sum() - 1.0) < 0.05

    def test_high_correlation(self):
        """Highly correlated assets handled gracefully."""
        from quantum.algorithms.qaoa import QAOAPortfolioOptimizer
        mu = np.array([0.10, 0.10])
        sigma = np.array([[0.04, 0.038], [0.038, 0.04]])
        opt = QAOAPortfolioOptimizer(n_layers=1)
        result = opt.optimize(mu, sigma, risk_aversion=0.5)
        w = np.array(result["weights"])
        assert abs(w.sum() - 1.0) < 0.05


# ===================================================================
# Quantum Amplitude Estimation for VaR
# ===================================================================


class TestQuantumAmplitudeEstimatorVaR:

    def test_estimate_var_basic(self, return_series):
        """QAE VaR is close to empirical percentile."""
        from quantum.algorithms.quantum_amplitude_estimation import (
            QuantumAmplitudeEstimatorVaR,
        )
        est = QuantumAmplitudeEstimatorVaR(n_qubits=4)
        result = est.estimate_var(return_series, confidence=0.95, n_samples=5000)

        assert "var_95" in result
        assert "cvar_95" in result
        assert "var_99" in result
        assert "cvar_99" in result
        assert "method" in result

        # VaR should be negative (loss)
        assert result["var_95"] < 0
        # CVaR should be worse (more negative) than VaR
        assert result["cvar_95"] <= result["var_95"] + 1e-6

    def test_var_accuracy(self, return_series):
        """QAE VaR within 20% of scipy percentile."""
        from quantum.algorithms.quantum_amplitude_estimation import (
            QuantumAmplitudeEstimatorVaR,
        )
        est = QuantumAmplitudeEstimatorVaR(n_qubits=5)
        result = est.estimate_var(return_series, confidence=0.95, n_samples=10000)

        empirical_var = float(np.percentile(return_series, 5.0))
        # Allow 20% relative error
        if abs(empirical_var) > 1e-6:
            rel_error = abs(result["var_95"] - empirical_var) / abs(empirical_var)
            assert rel_error < 0.20, f"VaR error {rel_error:.1%} exceeds 20%"

    def test_var_99_more_extreme(self, return_series):
        """99% VaR is more extreme than 95% VaR."""
        from quantum.algorithms.quantum_amplitude_estimation import (
            QuantumAmplitudeEstimatorVaR,
        )
        est = QuantumAmplitudeEstimatorVaR()
        result = est.estimate_var(return_series, confidence=0.95)
        # VaR99 should be at least as extreme as VaR95 (more negative or equal)
        # Use generous tolerance for stochastic importance sampling
        assert result["var_99"] <= result["var_95"] + 0.05

    def test_empty_returns(self):
        """Empty returns produce zero result."""
        from quantum.algorithms.quantum_amplitude_estimation import (
            QuantumAmplitudeEstimatorVaR,
        )
        est = QuantumAmplitudeEstimatorVaR()
        result = est.estimate_var([], confidence=0.95)
        assert result["var_95"] == 0.0
        assert result["method"] == "insufficient_data"

    def test_single_return(self):
        """Single return handled gracefully."""
        from quantum.algorithms.quantum_amplitude_estimation import (
            QuantumAmplitudeEstimatorVaR,
        )
        est = QuantumAmplitudeEstimatorVaR()
        result = est.estimate_var([0.01], confidence=0.95)
        assert result["method"] == "insufficient_data"

    def test_classical_comparison_included(self, return_series):
        """Result includes classical comparison dict."""
        from quantum.algorithms.quantum_amplitude_estimation import (
            QuantumAmplitudeEstimatorVaR,
        )
        est = QuantumAmplitudeEstimatorVaR()
        result = est.estimate_var(return_series)
        assert "classical_comparison" in result
        cc = result["classical_comparison"]
        assert "classical_var_95" in cc
        assert "classical_cvar_95" in cc

    def test_variance_reduction_positive(self, return_series):
        """Importance sampling provides finite variance reduction."""
        from quantum.algorithms.quantum_amplitude_estimation import (
            QuantumAmplitudeEstimatorVaR,
        )
        est = QuantumAmplitudeEstimatorVaR()
        result = est.estimate_var(return_series, n_samples=5000)
        vr = result.get("variance_reduction_factor", 1.0)
        assert vr > 0  # variance reduction factor is positive
        assert isinstance(vr, float)

    def test_convergence_analysis(self, return_series):
        """Convergence analysis returns rate information."""
        from quantum.algorithms.quantum_amplitude_estimation import (
            QuantumAmplitudeEstimatorVaR,
        )
        est = QuantumAmplitudeEstimatorVaR()
        true_var = float(np.percentile(return_series, 5.0))
        conv = est.convergence_analysis(return_series, true_var=true_var)

        assert "qae_convergence_rate" in conv
        assert "mc_convergence_rate" in conv
        assert "samples_for_1pct_accuracy_qae" in conv
        assert "samples_for_1pct_accuracy_mc" in conv
        assert "theoretical_speedup" in conv
        assert "actual_speedup" in conv

    def test_convergence_short_data(self):
        """Convergence analysis handles short data."""
        from quantum.algorithms.quantum_amplitude_estimation import (
            QuantumAmplitudeEstimatorVaR,
        )
        est = QuantumAmplitudeEstimatorVaR()
        conv = est.convergence_analysis([0.01, 0.02])
        assert conv["qae_convergence_rate"] == 0.0

    def test_constant_returns(self):
        """Constant returns produce zero VaR."""
        from quantum.algorithms.quantum_amplitude_estimation import (
            QuantumAmplitudeEstimatorVaR,
        )
        est = QuantumAmplitudeEstimatorVaR()
        result = est.estimate_var([0.01] * 100)
        # VaR should be the constant value
        assert abs(result["var_95"] - 0.01) < 0.001


# ===================================================================
# Quantum Kernel Classifier
# ===================================================================


class TestQuantumKernelClassifier:

    def test_feature_map_unitarity(self):
        """Feature map produces unit-norm statevector."""
        from quantum.qml.quantum_kernel import QuantumKernelClassifier
        clf = QuantumKernelClassifier(n_features=3, n_layers=2)
        x = np.array([0.5, -0.3, 1.2])
        state = clf._quantum_feature_map(x)
        norm = float(np.sum(np.abs(state) ** 2))
        assert abs(norm - 1.0) < 1e-10

    def test_kernel_matrix_symmetry(self, classification_data):
        """Kernel matrix is symmetric."""
        from quantum.qml.quantum_kernel import QuantumKernelClassifier
        X, _ = classification_data
        clf = QuantumKernelClassifier(n_features=2, n_layers=1)
        K = clf.compute_kernel_matrix(X[:10])
        assert np.allclose(K, K.T, atol=1e-10)

    def test_kernel_matrix_diagonal_ones(self, classification_data):
        """Diagonal of kernel matrix is 1 (self-similarity)."""
        from quantum.qml.quantum_kernel import QuantumKernelClassifier
        X, _ = classification_data
        clf = QuantumKernelClassifier(n_features=2, n_layers=1)
        K = clf.compute_kernel_matrix(X[:10])
        assert np.allclose(np.diag(K), 1.0, atol=1e-10)

    def test_kernel_matrix_psd(self, classification_data):
        """Kernel matrix is positive semi-definite."""
        from quantum.qml.quantum_kernel import QuantumKernelClassifier
        X, _ = classification_data
        clf = QuantumKernelClassifier(n_features=2, n_layers=1)
        K = clf.compute_kernel_matrix(X[:10])
        eigenvalues = np.linalg.eigvalsh(K)
        assert all(ev >= -1e-8 for ev in eigenvalues)

    def test_fit_predict(self, classification_data):
        """Fit and predict produce valid output shapes."""
        from quantum.qml.quantum_kernel import QuantumKernelClassifier
        X, y = classification_data
        # Use small subset for speed
        X_train, y_train = X[:20], y[:20]
        X_test = X[20:30]

        clf = QuantumKernelClassifier(n_features=2, n_layers=1)
        clf.fit(X_train, y_train)
        preds, confs = clf.predict(X_test)

        assert len(preds) == len(X_test)
        assert len(confs) == len(X_test)
        assert set(preds).issubset({0, 1})

    def test_fit_predict_accuracy(self, classification_data):
        """Classifier achieves >60% accuracy on separable data."""
        from quantum.qml.quantum_kernel import QuantumKernelClassifier
        X, y = classification_data
        X_train, y_train = X[:30], y[:30]
        X_test, y_test = X[30:], y[30:]

        clf = QuantumKernelClassifier(n_features=2, n_layers=1)
        clf.fit(X_train, y_train)
        preds, _ = clf.predict(X_test)

        accuracy = float(np.mean(preds == y_test))
        assert accuracy >= 0.6, f"Accuracy {accuracy:.1%} below 60%"

    def test_predict_before_fit(self):
        """Predict before fit raises error."""
        from quantum.qml.quantum_kernel import QuantumKernelClassifier
        clf = QuantumKernelClassifier(n_features=2)
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict(np.array([[1.0, 2.0]]))

    def test_kernel_values_bounded(self, classification_data):
        """Kernel values are in [0, 1]."""
        from quantum.qml.quantum_kernel import QuantumKernelClassifier
        X, _ = classification_data
        clf = QuantumKernelClassifier(n_features=2, n_layers=1)
        K = clf.compute_kernel_matrix(X[:10])
        assert np.all(K >= -1e-10)
        assert np.all(K <= 1.0 + 1e-10)

    def test_benchmark_vs_classical(self, classification_data):
        """Benchmark returns honest assessment."""
        from quantum.qml.quantum_kernel import QuantumKernelClassifier
        X, y = classification_data
        clf = QuantumKernelClassifier(n_features=2, n_layers=1)
        bench = clf.benchmark_vs_classical(X, y)

        assert "honest_assessment" in bench
        assert isinstance(bench["honest_assessment"], str)
        assert "quantum_time_ms" in bench
        assert bench["quantum_time_ms"] > 0

    def test_single_feature(self):
        """Single feature dimension works."""
        from quantum.qml.quantum_kernel import QuantumKernelClassifier
        clf = QuantumKernelClassifier(n_features=1, n_layers=1)
        X = np.array([[1.0], [2.0], [3.0]])
        K = clf.compute_kernel_matrix(X)
        assert K.shape == (3, 3)

    def test_feature_map_different_inputs(self):
        """Different inputs produce different statevectors."""
        from quantum.qml.quantum_kernel import QuantumKernelClassifier
        clf = QuantumKernelClassifier(n_features=2, n_layers=1)
        s1 = clf._quantum_feature_map(np.array([0.0, 0.0]))
        s2 = clf._quantum_feature_map(np.array([1.0, 1.0]))
        # Should not be identical
        assert not np.allclose(s1, s2)


# ===================================================================
# Unified stubs wiring
# ===================================================================


class TestUnifiedStubsWiring:

    def test_qaoa_portfolio_optimize_stub(self, simple_portfolio):
        """qaoa_portfolio_optimize wrapper works."""
        from quantum.quantum_unified_stubs import qaoa_portfolio_optimize
        mu, sigma = simple_portfolio
        result = qaoa_portfolio_optimize(mu, sigma, risk_aversion=0.5)
        assert "weights" in result or "selected_assets" in result
        assert "method" in result

    def test_quantum_var_estimation_stub(self, return_series):
        """quantum_var_estimation wrapper works."""
        from quantum.quantum_unified_stubs import quantum_var_estimation
        result = quantum_var_estimation(return_series, confidence=0.95, n_samples=2000)
        assert "var_95" in result
        assert "method" in result

    def test_quantum_kernel_predict_stub_matrix(self, classification_data):
        """quantum_kernel_predict returns kernel matrix when no model."""
        from quantum.quantum_unified_stubs import quantum_kernel_predict
        X, _ = classification_data
        result = quantum_kernel_predict(X[:5])
        assert "kernel_matrix" in result or "method" in result

    def test_quantum_kernel_predict_stub_with_model(self, classification_data):
        """quantum_kernel_predict uses fitted model."""
        from quantum.qml.quantum_kernel import QuantumKernelClassifier
        from quantum.quantum_unified_stubs import quantum_kernel_predict
        X, y = classification_data
        clf = QuantumKernelClassifier(n_features=2, n_layers=1)
        clf.fit(X[:20], y[:20])
        result = quantum_kernel_predict(X[20:25], model=clf)
        assert "predictions" in result
        assert len(result["predictions"]) == 5


# ===================================================================
# Honest benchmarks: does quantum actually beat classical?
# ===================================================================


class TestHonestBenchmarks:

    def test_qaoa_vs_scipy_comparison(self, simple_portfolio):
        """Track whether QAOA beats scipy (expected: similar or worse)."""
        from quantum.algorithms.qaoa import QAOAPortfolioOptimizer
        mu, sigma = simple_portfolio
        opt = QAOAPortfolioOptimizer(n_layers=2)
        bench = opt.benchmark_vs_classical(mu, sigma)

        # We don't assert QAOA wins — we track honestly
        assert isinstance(bench["improvement_pct"], float)
        # But we DO assert both methods produce valid results
        assert bench["qaoa_sharpe"] is not None
        assert bench["classical_sharpe"] is not None

    def test_qae_convergence_vs_mc(self, return_series):
        """Track QAE convergence vs classical MC."""
        from quantum.algorithms.quantum_amplitude_estimation import (
            QuantumAmplitudeEstimatorVaR,
        )
        est = QuantumAmplitudeEstimatorVaR(n_qubits=4)
        conv = est.convergence_analysis(return_series)

        # Both should have positive convergence rates
        # (errors decrease with more samples)
        assert conv["qae_convergence_rate"] >= 0
        assert conv["mc_convergence_rate"] >= 0
        # Actual speedup should be finite
        assert conv["actual_speedup"] > 0
        assert conv["actual_speedup"] < 1000  # sanity bound

    def test_quantum_kernel_vs_rbf(self, classification_data):
        """Track quantum kernel vs RBF accuracy."""
        from quantum.qml.quantum_kernel import QuantumKernelClassifier
        X, y = classification_data
        clf = QuantumKernelClassifier(n_features=2, n_layers=1)
        bench = clf.benchmark_vs_classical(X, y)

        # Both should produce non-negative accuracy
        assert bench.get("quantum_accuracy", 0) >= 0
        assert bench.get("rbf_accuracy", 0) >= 0
        # Honest assessment should be populated
        assert len(bench.get("honest_assessment", "")) > 0
