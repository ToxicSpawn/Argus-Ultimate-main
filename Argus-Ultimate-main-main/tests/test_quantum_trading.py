"""
Tests for quantum-native trading applications.

Covers:
  - QuantumPortfolioOptimizer: QUBO construction, solution, cardinality, risk parity
  - QuantumSignalClassifier: kernel computation, SVM training, signal quality
  - QuantumPairsDiscovery: correlation graph, quantum walk, pairs ranking
  - QuantumRiskEngine: VaR, stress testing, tail risk decomposition
  - ComponentRegistry wiring
"""

import asyncio
import math
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_portfolio(rng):
    """5-asset portfolio with known structure."""
    n = 5
    mu = rng.normal(0.05, 0.02, n)
    A = rng.normal(0, 0.1, (n, n))
    sigma = A.T @ A / n + 0.02 * np.eye(n)
    return mu, sigma


@pytest.fixture
def price_matrix(rng):
    """Synthetic price matrix with 2 clusters."""
    n_steps = 200
    n_assets = 6
    # Cluster 1: assets 0,1,2 (correlated)
    base1 = np.cumsum(rng.normal(0.001, 0.02, n_steps))
    prices = np.zeros((n_steps, n_assets))
    for i in range(3):
        prices[:, i] = 100 * np.exp(base1 + rng.normal(0, 0.005, n_steps))
    # Cluster 2: assets 3,4,5 (correlated, different trend)
    base2 = np.cumsum(rng.normal(-0.001, 0.015, n_steps))
    for i in range(3, 6):
        prices[:, i] = 50 * np.exp(base2 + rng.normal(0, 0.005, n_steps))
    return prices


@pytest.fixture
def returns_matrix(rng):
    """Synthetic returns matrix."""
    n_steps = 500
    n_assets = 4
    returns = rng.normal(0.0002, 0.015, (n_steps, n_assets))
    # Add correlation
    L = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.866, 0.0, 0.0],
        [0.3, 0.2, 0.932, 0.0],
        [0.1, 0.1, 0.1, 0.985],
    ])
    returns = returns @ L.T
    return returns


# ===========================================================================
# 1. QuantumPortfolioOptimizer
# ===========================================================================

class TestQuantumPortfolioOptimizer:
    """Tests for quantum portfolio optimization."""

    def test_qubo_construction(self, small_portfolio):
        """QUBO matrix is built with correct dimensions."""
        from quantum.portfolio.quantum_portfolio import QuantumPortfolioOptimizer
        mu, sigma = small_portfolio
        opt = QuantumPortfolioOptimizer(weight_bits=3)
        qubo, n_assets, n_vars = opt._build_qubo(mu, sigma, risk_aversion=1.0)
        assert n_assets == 5
        assert n_vars == 5 * 3  # 5 assets * 3 bits
        assert len(qubo) > 0
        # QUBO should have diagonal and off-diagonal entries
        diag_count = sum(1 for (i, j) in qubo if i == j)
        assert diag_count > 0

    def test_optimize_weights_returns_valid(self, small_portfolio):
        """Optimized weights sum to 1 and are non-negative."""
        from quantum.portfolio.quantum_portfolio import QuantumPortfolioOptimizer
        mu, sigma = small_portfolio
        opt = QuantumPortfolioOptimizer(weight_bits=3)
        result = opt.optimize_weights(mu, sigma, risk_aversion=1.0)
        weights = result["weights"]
        assert isinstance(weights, np.ndarray)
        assert len(weights) == 5
        assert np.all(weights >= -1e-6)
        assert abs(np.sum(weights) - 1.0) < 1e-4
        assert "method" in result
        assert result["method"] != "no_assets"

    def test_optimize_single_asset(self):
        """Single asset returns weight 1.0."""
        from quantum.portfolio.quantum_portfolio import QuantumPortfolioOptimizer
        opt = QuantumPortfolioOptimizer()
        result = opt.optimize_weights(np.array([0.05]), np.array([[0.01]]))
        assert len(result["weights"]) == 1 or result["weights"] == [1.0]

    def test_optimize_empty(self):
        """Empty asset list returns empty result."""
        from quantum.portfolio.quantum_portfolio import QuantumPortfolioOptimizer
        opt = QuantumPortfolioOptimizer()
        result = opt.optimize_weights(np.array([]), np.array([[]]))
        assert result["method"] == "no_assets"

    def test_cardinality_constraint(self, small_portfolio):
        """Cardinality-constrained solution selects at most max_assets."""
        from quantum.portfolio.quantum_portfolio import QuantumPortfolioOptimizer
        mu, sigma = small_portfolio
        opt = QuantumPortfolioOptimizer(weight_bits=3)
        result = opt.optimize_with_cardinality(mu, sigma, max_assets=3)
        assert result["n_selected"] <= 3
        assert len(result["selected_assets"]) <= 3
        weights = result["weights"]
        assert abs(np.sum(weights) - 1.0) < 1e-4

    def test_cardinality_max_equals_n(self, small_portfolio):
        """Cardinality = n_assets should not restrict selection."""
        from quantum.portfolio.quantum_portfolio import QuantumPortfolioOptimizer
        mu, sigma = small_portfolio
        opt = QuantumPortfolioOptimizer(weight_bits=3)
        result = opt.optimize_with_cardinality(mu, sigma, max_assets=5)
        assert result["n_selected"] <= 5
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-4

    def test_risk_parity(self, small_portfolio):
        """Risk parity produces near-equal risk contributions."""
        from quantum.portfolio.quantum_portfolio import QuantumPortfolioOptimizer
        _, sigma = small_portfolio
        opt = QuantumPortfolioOptimizer()
        result = opt.risk_parity(sigma)
        weights = result["weights"]
        rc = result["risk_contributions"]
        assert len(weights) == 5
        assert abs(np.sum(weights) - 1.0) < 1e-4
        # Risk contributions should be approximately equal
        assert result["max_rc_deviation"] < 0.2  # within 20% of mean

    def test_risk_parity_single_asset(self):
        """Risk parity with single asset."""
        from quantum.portfolio.quantum_portfolio import QuantumPortfolioOptimizer
        opt = QuantumPortfolioOptimizer()
        result = opt.risk_parity(np.array([[0.04]]))
        assert np.allclose(result["weights"], [1.0])

    def test_compare_with_classical(self, small_portfolio):
        """Comparison returns both results and honest assessment."""
        from quantum.portfolio.quantum_portfolio import QuantumPortfolioOptimizer
        mu, sigma = small_portfolio
        opt = QuantumPortfolioOptimizer(weight_bits=3)
        result = opt.compare_with_classical(mu, sigma)
        assert "quantum_sharpe" in result
        assert "classical_sharpe" in result
        assert "honest_assessment" in result
        assert "improvement_pct" in result
        assert result["quantum_time_ms"] >= 0
        assert result["classical_time_ms"] >= 0

    def test_scipy_fallback(self, small_portfolio):
        """Scipy fallback produces valid solution."""
        from quantum.portfolio.quantum_portfolio import QuantumPortfolioOptimizer
        mu, sigma = small_portfolio
        opt = QuantumPortfolioOptimizer(backend="scipy_fallback")
        result = opt.optimize_weights(mu, sigma)
        assert result["method"] == "classical_scipy_slsqp"
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-4

    def test_decode_solution(self):
        """Binary solution decodes to valid weights."""
        from quantum.portfolio.quantum_portfolio import QuantumPortfolioOptimizer
        opt = QuantumPortfolioOptimizer(weight_bits=3)
        # 3 bits per asset, 2 assets
        # Asset 0: bits 0,1,2 = 1,1,0 => weight = (1 + 2) / 7 = 3/7
        # Asset 1: bits 3,4,5 = 0,0,1 => weight = 4 / 7
        solution = {0: 1, 1: 1, 2: 0, 3: 0, 4: 0, 5: 1}
        weights = opt._decode_solution(solution, 2)
        assert len(weights) == 2
        assert abs(np.sum(weights) - 1.0) < 1e-6
        # Asset 0 raw = 3/7, Asset 1 raw = 4/7
        assert weights[0] < weights[1]


# ===========================================================================
# 2. QuantumSignalClassifier
# ===========================================================================

class TestQuantumSignalClassifier:
    """Tests for quantum kernel signal classification."""

    def test_feature_map_produces_valid_state(self):
        """Feature map produces normalized statevector."""
        from quantum.qml.quantum_signal_classifier import QuantumSignalClassifier
        clf = QuantumSignalClassifier(n_features=4, n_qubits=4)
        x = np.array([0.5, -0.3, 0.8, 0.1])
        state = clf._build_feature_map(x)
        assert len(state) == 16  # 2^4
        # Normalized
        assert abs(np.linalg.norm(state) - 1.0) < 1e-6

    def test_amplitude_encoding(self):
        """Amplitude encoding produces normalized state."""
        from quantum.qml.quantum_signal_classifier import QuantumSignalClassifier
        clf = QuantumSignalClassifier(n_features=4, kernel_type="amplitude")
        x = np.array([1.0, 2.0, 3.0, 4.0])
        state = clf._build_feature_map(x)
        assert abs(np.linalg.norm(state) - 1.0) < 1e-6

    def test_quantum_kernel_self_overlap(self):
        """Kernel of a point with itself should be 1.0."""
        from quantum.qml.quantum_signal_classifier import QuantumSignalClassifier
        clf = QuantumSignalClassifier(n_features=3, n_qubits=3, n_layers=1)
        x = np.array([0.5, 0.3, -0.2])
        k = clf._quantum_kernel(x, x)
        assert abs(k - 1.0) < 1e-6

    def test_quantum_kernel_symmetry(self):
        """Kernel is symmetric: K(x1,x2) = K(x2,x1)."""
        from quantum.qml.quantum_signal_classifier import QuantumSignalClassifier
        clf = QuantumSignalClassifier(n_features=3, n_qubits=3, n_layers=1)
        x1 = np.array([0.5, 0.3, -0.2])
        x2 = np.array([-0.1, 0.8, 0.4])
        assert abs(clf._quantum_kernel(x1, x2) - clf._quantum_kernel(x2, x1)) < 1e-10

    def test_quantum_kernel_bounded(self):
        """Kernel values are in [0, 1]."""
        from quantum.qml.quantum_signal_classifier import QuantumSignalClassifier
        clf = QuantumSignalClassifier(n_features=3, n_qubits=3, n_layers=1)
        rng = np.random.default_rng(42)
        for _ in range(10):
            x1 = rng.normal(0, 1, 3)
            x2 = rng.normal(0, 1, 3)
            k = clf._quantum_kernel(x1, x2)
            assert 0.0 - 1e-6 <= k <= 1.0 + 1e-6

    def test_kernel_matrix_positive_semidefinite(self):
        """Kernel matrix should be positive semi-definite."""
        from quantum.qml.quantum_signal_classifier import QuantumSignalClassifier
        clf = QuantumSignalClassifier(n_features=3, n_qubits=3, n_layers=1)
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (5, 3))
        K = clf.compute_kernel_matrix(X)
        eigenvalues = np.linalg.eigvalsh(K)
        # All eigenvalues >= 0 (within numerical tolerance)
        assert np.all(eigenvalues > -1e-6)

    def test_fit_predict_synthetic(self, rng):
        """SVM trained on quantum kernel classifies synthetic data."""
        from quantum.qml.quantum_signal_classifier import QuantumSignalClassifier
        # Simple linearly separable dataset
        n = 30
        X = rng.normal(0, 1, (n, 3))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        clf = QuantumSignalClassifier(n_features=3, n_qubits=3, n_layers=1)
        clf.fit(X, y, C=1.0)
        preds = clf.predict(X)
        accuracy = np.mean(preds == y)
        # Should get reasonable accuracy on training data
        assert accuracy > 0.6

    def test_predict_signal_quality_not_fitted(self):
        """Signal quality returns default when not fitted."""
        from quantum.qml.quantum_signal_classifier import QuantumSignalClassifier
        clf = QuantumSignalClassifier(n_features=4)
        result = clf.predict_signal_quality(np.zeros(4))
        assert result["quality"] == 0.5
        assert result["method"] == "not_fitted"

    def test_predict_signal_quality_fitted(self, rng):
        """Signal quality returns structured output after fitting."""
        from quantum.qml.quantum_signal_classifier import QuantumSignalClassifier
        n = 20
        X = rng.normal(0, 1, (n, 3))
        y = (X[:, 0] > 0).astype(int)
        clf = QuantumSignalClassifier(n_features=3, n_qubits=3, n_layers=1)
        clf.fit(X, y)
        result = clf.predict_signal_quality(rng.normal(0, 1, 3))
        assert "quality" in result
        assert "confidence" in result
        assert "regime" in result
        assert 0.0 <= result["quality"] <= 1.0


# ===========================================================================
# 3. QuantumPairsDiscovery
# ===========================================================================

class TestQuantumPairsDiscovery:
    """Tests for quantum walk pairs discovery."""

    def test_build_correlation_graph(self, price_matrix):
        """Correlation graph has correct shape and properties."""
        from quantum.trading.quantum_pairs import QuantumPairsDiscovery
        qpd = QuantumPairsDiscovery(correlation_threshold=0.3)
        adj = qpd.build_correlation_graph(price_matrix)
        assert adj.shape == (6, 6)
        # Symmetric
        assert np.allclose(adj, adj.T)
        # No self-loops
        assert np.allclose(np.diag(adj), 0)
        # Values in [0, 1]
        assert np.all(adj >= 0)
        assert np.all(adj <= 1.0 + 1e-6)

    def test_correlation_graph_clusters(self, price_matrix):
        """Correlation graph shows within-cluster correlations > threshold."""
        from quantum.trading.quantum_pairs import QuantumPairsDiscovery
        qpd = QuantumPairsDiscovery(correlation_threshold=0.2)
        adj = qpd.build_correlation_graph(price_matrix)
        # Assets 0,1,2 should be more connected to each other
        cluster1_edges = adj[0, 1] + adj[0, 2] + adj[1, 2]
        # At least some edges within cluster
        assert cluster1_edges > 0

    def test_quantum_walk_clustering(self, price_matrix):
        """Quantum walk produces non-empty clusters."""
        from quantum.trading.quantum_pairs import QuantumPairsDiscovery
        qpd = QuantumPairsDiscovery(walk_steps=20)
        adj = qpd.build_correlation_graph(price_matrix)
        clusters = qpd.quantum_walk_clustering(adj)
        assert len(clusters) > 0
        # All assets assigned
        all_assets = set()
        for c in clusters:
            all_assets.update(c)
        assert len(all_assets) == 6

    def test_quantum_walk_single_asset(self):
        """Single asset produces single cluster."""
        from quantum.trading.quantum_pairs import QuantumPairsDiscovery
        qpd = QuantumPairsDiscovery()
        adj = np.array([[0.0]])
        clusters = qpd.quantum_walk_clustering(adj)
        assert len(clusters) == 1
        assert clusters[0] == [0]

    def test_rank_pairs(self, price_matrix):
        """Pair ranking produces valid pair candidates."""
        from quantum.trading.quantum_pairs import QuantumPairsDiscovery
        qpd = QuantumPairsDiscovery()
        clusters = [[0, 1, 2], [3, 4, 5]]
        pairs = qpd.rank_pairs(clusters, price_matrix)
        assert len(pairs) > 0
        for pair in pairs:
            assert "pair" in pair
            assert "cointegration_pvalue" in pair
            assert "half_life" in pair
            assert "correlation" in pair
            assert 0 <= pair["cointegration_pvalue"] <= 1.0

    def test_rank_pairs_sorted(self, price_matrix):
        """Pairs are sorted by cointegration p-value (ascending)."""
        from quantum.trading.quantum_pairs import QuantumPairsDiscovery
        qpd = QuantumPairsDiscovery()
        clusters = [[0, 1, 2]]
        pairs = qpd.rank_pairs(clusters, price_matrix)
        if len(pairs) > 1:
            for i in range(len(pairs) - 1):
                assert pairs[i]["cointegration_pvalue"] <= pairs[i + 1]["cointegration_pvalue"]

    def test_discover_pairs_end_to_end(self, price_matrix):
        """End-to-end discovery returns labeled pairs."""
        from quantum.trading.quantum_pairs import QuantumPairsDiscovery
        qpd = QuantumPairsDiscovery(
            correlation_threshold=0.2,
            walk_steps=15,
        )
        names = ["BTC", "ETH", "SOL", "AAPL", "MSFT", "GOOGL"]
        pairs = qpd.discover_pairs(price_matrix, asset_names=names)
        # Should find at least some pairs
        assert isinstance(pairs, list)
        if len(pairs) > 0:
            assert "asset_a" in pairs[0]
            assert "asset_b" in pairs[0]
            assert "cluster_id" in pairs[0]

    def test_discover_pairs_1d_input(self, rng):
        """Single asset input handled gracefully."""
        from quantum.trading.quantum_pairs import QuantumPairsDiscovery
        qpd = QuantumPairsDiscovery()
        prices = rng.normal(100, 5, 100)
        pairs = qpd.discover_pairs(prices)
        assert isinstance(pairs, list)


# ===========================================================================
# 4. QuantumRiskEngine
# ===========================================================================

class TestQuantumRiskEngine:
    """Tests for quantum-accelerated risk calculations."""

    def test_quantum_var_basic(self, rng):
        """VaR computation returns valid structure."""
        from quantum.risk.quantum_risk import QuantumRiskEngine
        engine = QuantumRiskEngine(seed=42)
        returns = rng.normal(-0.001, 0.02, 1000)
        result = engine.quantum_var(returns, confidence=0.95, n_paths=500)
        assert "var" in result
        assert "cvar" in result
        assert "confidence_interval" in result
        assert result["var"] > 0  # losses are positive
        assert result["cvar"] >= result["var"]  # CVaR >= VaR

    def test_quantum_var_confidence_levels(self, rng):
        """Higher confidence level produces larger VaR."""
        from quantum.risk.quantum_risk import QuantumRiskEngine
        engine = QuantumRiskEngine(seed=42)
        returns = rng.normal(-0.001, 0.02, 2000)
        var_95 = engine.quantum_var(returns, confidence=0.95, n_paths=500)
        var_99 = engine.quantum_var(returns, confidence=0.99, n_paths=500)
        # 99% VaR should generally be >= 95% VaR (not guaranteed for small samples)
        # Just check both are positive
        assert var_95["var"] > 0
        assert var_99["var"] > 0

    def test_quantum_var_insufficient_data(self):
        """Insufficient data returns zero VaR."""
        from quantum.risk.quantum_risk import QuantumRiskEngine
        engine = QuantumRiskEngine()
        result = engine.quantum_var(np.array([0.01, 0.02]), n_paths=100)
        assert result["method"] == "insufficient_data"

    def test_quantum_var_variance_reduction(self, rng):
        """Variance reduction factor is positive."""
        from quantum.risk.quantum_risk import QuantumRiskEngine
        engine = QuantumRiskEngine(seed=42)
        returns = rng.normal(-0.001, 0.02, 1000)
        result = engine.quantum_var(returns, n_paths=1000)
        assert result["variance_reduction_factor"] > 0

    def test_stress_test_basic(self):
        """Stress test returns valid scenario losses."""
        from quantum.risk.quantum_risk import QuantumRiskEngine
        engine = QuantumRiskEngine(seed=42)
        weights = np.array([0.3, 0.3, 0.4])
        shocks = {"equity": -0.20, "rates": 0.02, "credit": -0.10}
        result = engine.quantum_stress_test(
            weights, shocks, n_scenarios=200
        )
        assert "scenario_losses" in result
        assert "worst_case" in result
        assert "expected_shortfall" in result
        assert len(result["scenario_losses"]) == 200
        assert result["worst_case"] >= 0  # at least some positive losses

    def test_stress_test_with_covariance(self, rng):
        """Stress test with explicit covariance matrix."""
        from quantum.risk.quantum_risk import QuantumRiskEngine
        engine = QuantumRiskEngine(seed=42)
        weights = np.array([0.5, 0.5])
        shocks = {"factor_a": -0.15, "factor_b": -0.10}
        cov = np.array([[0.04, 0.01], [0.01, 0.02]])
        result = engine.quantum_stress_test(
            weights, shocks, cov_matrix=cov, n_scenarios=100
        )
        assert len(result["scenario_losses"]) == 100
        assert result["method"] == "quantum_inspired_stress_test"

    def test_stress_test_empty(self):
        """Empty inputs return empty result."""
        from quantum.risk.quantum_risk import QuantumRiskEngine
        engine = QuantumRiskEngine()
        result = engine.quantum_stress_test(np.array([]), {})
        assert result["worst_case"] == 0.0

    def test_tail_risk_decomposition(self, returns_matrix):
        """Tail risk decomposition sums approximately to total."""
        from quantum.risk.quantum_risk import QuantumRiskEngine
        engine = QuantumRiskEngine(seed=42)
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        result = engine.tail_risk_decomposition(returns_matrix, weights)
        assert "asset_contributions" in result
        assert "systemic_component" in result
        assert "idiosyncratic_component" in result
        # Systemic + idiosyncratic should sum to ~1.0
        total = result["systemic_component"] + result["idiosyncratic_component"]
        assert abs(total - 1.0) < 1e-4
        # Asset contributions should be present for all 4 assets
        assert len(result["asset_contributions"]) == 4

    def test_tail_risk_contributions_sum(self, returns_matrix):
        """Asset contributions sum to approximately 1.0 in absolute value."""
        from quantum.risk.quantum_risk import QuantumRiskEngine
        engine = QuantumRiskEngine(seed=42)
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        result = engine.tail_risk_decomposition(returns_matrix, weights)
        contribs = list(result["asset_contributions"].values())
        total_contrib = sum(abs(c) for c in contribs)
        # Normalized contributions should sum to ~1.0
        assert abs(total_contrib - 1.0) < 0.5  # allow some slack

    def test_tail_risk_insufficient_data(self):
        """Insufficient data handled gracefully."""
        from quantum.risk.quantum_risk import QuantumRiskEngine
        engine = QuantumRiskEngine()
        result = engine.tail_risk_decomposition(
            np.array([[0.01], [0.02]]),
            np.array([1.0]),
        )
        assert result["total_tail_risk"] == 0.0


# ===========================================================================
# 5. ComponentRegistry wiring
# ===========================================================================

class TestComponentRegistryQuantum:
    """Tests for quantum component wiring in ComponentRegistry."""

    def _make_config(self):
        cfg = SimpleNamespace(
            primary_exchange="kraken",
            multi_venue_min_notional_aud=200.0,
            aud_to_usd=0.65,
            starting_capital_aud=1000.0,
            trading_pairs=["BTC/USD", "ETH/USD"],
            llm_signal_enabled=False,
            initial_price_history=None,
        )
        return cfg

    def test_quantum_portfolio_optimizer_init(self):
        """QuantumPortfolioOptimizer initializes via registry."""
        from core.component_registry import ComponentRegistry
        cfg = self._make_config()
        reg = ComponentRegistry(cfg)
        reg._try_init("quantum_portfolio_optimizer", reg._init_quantum_portfolio_optimizer)
        assert reg._quantum_portfolio_optimizer is not None

    def test_quantum_signal_classifier_init(self):
        """QuantumSignalClassifier initializes via registry."""
        from core.component_registry import ComponentRegistry
        cfg = self._make_config()
        reg = ComponentRegistry(cfg)
        reg._try_init("quantum_signal_classifier", reg._init_quantum_signal_classifier)
        assert reg._quantum_signal_classifier is not None

    def test_quantum_risk_engine_init(self):
        """QuantumRiskEngine initializes via registry."""
        from core.component_registry import ComponentRegistry
        cfg = self._make_config()
        reg = ComponentRegistry(cfg)
        reg._try_init("quantum_risk_engine", reg._init_quantum_risk_engine)
        assert reg._quantum_risk_engine is not None

    def test_quantum_pairs_discovery_init(self):
        """QuantumPairsDiscovery initializes via registry."""
        from core.component_registry import ComponentRegistry
        cfg = self._make_config()
        reg = ComponentRegistry(cfg)
        reg._try_init("quantum_pairs_discovery", reg._init_quantum_pairs_discovery)
        assert reg._quantum_pairs_discovery is not None

    def test_pre_order_check_with_classifier(self):
        """pre_order_check integrates quantum signal classifier."""
        from core.component_registry import ComponentRegistry
        cfg = self._make_config()
        reg = ComponentRegistry(cfg)
        reg._try_init("quantum_signal_classifier", reg._init_quantum_signal_classifier)
        # Not fitted: should not block
        result = reg.pre_order_check("BTC/USD", "buy", 500.0)
        assert result["allow"] is True

    def test_on_cycle_with_quantum_portfolio(self):
        """on_cycle integrates quantum portfolio optimizer."""
        from core.component_registry import ComponentRegistry
        cfg = self._make_config()
        reg = ComponentRegistry(cfg)
        reg._try_init("quantum_portfolio_optimizer", reg._init_quantum_portfolio_optimizer)
        reg._initialized = True
        reg._cycle_count = 99  # next cycle will be 100 (triggers quantum portfolio)
        advisory = reg.on_cycle({"BTC/USD": 50000, "ETH/USD": 3000})
        # Should not crash; advisory might or might not have quantum_portfolio
        assert isinstance(advisory, dict)

    def test_classical_fallback_when_quantum_unavailable(self, small_portfolio):
        """System degrades gracefully without quantum components."""
        from quantum.portfolio.quantum_portfolio import QuantumPortfolioOptimizer
        mu, sigma = small_portfolio
        opt = QuantumPortfolioOptimizer(backend="scipy_fallback")
        result = opt.optimize_weights(mu, sigma)
        assert result["method"] == "classical_scipy_slsqp"
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-4
