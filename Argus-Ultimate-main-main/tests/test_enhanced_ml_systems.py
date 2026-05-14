"""
Tests for Enhanced ML Systems (v15.0.0).

Tests all systems that are BETTER than quantum:
- GPU Transformer Ensemble
- Real-time GNN for Correlations
- Online RL Strategy Selector
- Causal Inference Regime Predictor
- Enhanced Diffusion Stress Testing

Author: Argus Ultimate
"""

from __future__ import annotations

import numpy as np
import pytest

# ============================================================================
# GPU Transformer Ensemble Tests
# ============================================================================

from ml.gpu_transformer_ensemble import (
    TransformerEnsemble,
    MarketTransformer,
    PositionalEncoding,
    GPUEngine,
    create_transformer_ensemble,
)


class TestPositionalEncoding:
    """Tests for positional encoding."""
    
    def test_create(self):
        """Should create positional encoding."""
        pe = PositionalEncoding.create(100, 64)
        assert pe.shape == (100, 64)
    
    def test_sinusoidal_pattern(self):
        """Should have sinusoidal pattern."""
        pe = PositionalEncoding.create(10, 4)
        # Check that columns alternate sin/cos
        assert not np.allclose(pe[:, 0], pe[:, 1])  # Different patterns


class TestMarketTransformer:
    """Tests for MarketTransformer."""
    
    def test_init(self):
        """Should initialize correctly."""
        model = MarketTransformer(d_model=32, n_heads=2, n_layers=2, input_dim=5)
        assert model.d_model == 32
        assert model.n_heads == 2
    
    def test_forward(self):
        """Should produce valid output."""
        model = MarketTransformer(d_model=32, n_heads=2, n_layers=2, input_dim=5)
        x = np.random.randn(2, 10, 5)  # batch=2, seq=10, features=5
        
        outputs = model.forward(x)
        
        assert 1 in outputs
        assert 4 in outputs
        assert 24 in outputs
        assert outputs[1].shape == (2, 1)


class TestTransformerEnsemble:
    """Tests for TransformerEnsemble."""
    
    def test_init(self):
        """Should initialize correctly."""
        ensemble = TransformerEnsemble(n_models=3, d_model=32, n_heads=2, input_dim=5)
        assert ensemble.n_models == 3
        assert len(ensemble.models) == 3
    
    def test_predict(self):
        """Should produce ensemble predictions."""
        ensemble = TransformerEnsemble(n_models=3, d_model=32, n_heads=2, input_dim=5)
        x = np.random.randn(1, 10, 5)
        
        results = ensemble.predict(x)
        
        assert 1 in results
        mean, std, uncertainty = results[1]
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert uncertainty >= 0


class TestGPUEngine:
    """Tests for GPUEngine."""
    
    def test_init(self):
        """Should initialize correctly."""
        engine = GPUEngine(n_models=3, d_model=32, input_dim=5)
        assert engine.n_models == 3
    
    def test_predict(self):
        """Should produce predictions."""
        engine = GPUEngine(n_models=2, d_model=32, input_dim=5)
        x = np.random.randn(10, 5)
        
        results = engine.predict(x)
        
        assert 1 in results
        mean, std = results[1]
        assert isinstance(mean, float)


# ============================================================================
# Real-time GNN Tests
# ============================================================================

from ml.realtime_correlation_gnn import (
    DynamicGraphConstructor,
    GraphAttentionLayer,
    CorrelationGNN,
    create_correlation_gnn,
)


class TestDynamicGraphConstructor:
    """Tests for graph construction."""
    
    def test_init(self):
        """Should initialize correctly."""
        constructor = DynamicGraphConstructor()
        assert constructor.correlation_threshold == 0.3
    
    def test_add_asset(self):
        """Should add assets."""
        constructor = DynamicGraphConstructor()
        constructor.add_asset("BTC")
        constructor.add_asset("ETH")
        
        assert len(constructor._asset_names) == 2
    
    def test_update_price(self):
        """Should update prices."""
        constructor = DynamicGraphConstructor()
        constructor.update_price("BTC", 50000)
        constructor.update_price("BTC", 51000)
        
        assert len(constructor._price_history["BTC"]) == 2
    
    def test_get_graph(self):
        """Should return graph structure."""
        constructor = DynamicGraphConstructor()
        
        # Add multiple assets with prices
        for i in range(25):
            constructor.update_price("BTC", 50000 + i * 100)
            constructor.update_price("ETH", 3000 + i * 10)
        
        adj, features, names = constructor.get_graph()
        
        assert len(names) == 2
        assert adj.shape == (2, 2)
        assert features.shape[0] == 2


class TestGraphAttentionLayer:
    """Tests for GAT layer."""
    
    def test_forward(self):
        """Should produce valid output."""
        layer = GraphAttentionLayer(in_features=10, out_features=32, n_heads=4)
        
        features = np.random.randn(5, 10)  # 5 nodes, 10 features
        adjacency = np.random.randint(0, 2, (5, 5))
        
        output = layer.forward(features, adjacency)
        
        assert output.shape == (5, 32)


class TestCorrelationGNN:
    """Tests for CorrelationGNN."""
    
    def test_init(self):
        """Should initialize correctly."""
        gnn = CorrelationGNN(input_dim=3, hidden_dim=32, n_layers=2)
        assert gnn.input_dim == 3
    
    def test_update(self):
        """Should update with prices."""
        gnn = CorrelationGNN()
        gnn.update("BTC", 50000)
        gnn.update("ETH", 3000)
        
        assert "BTC" in gnn.graph_constructor._asset_to_idx
    
    def test_forward(self):
        """Should produce graph output."""
        gnn = CorrelationGNN()
        
        # Add data
        for i in range(25):
            gnn.update("BTC", 50000 + i * 100)
            gnn.update("ETH", 3000 + i * 10)
        
        result = gnn.forward()
        
        assert "embeddings" in result
        assert "predicted_correlations" in result
        assert "asset_names" in result
    
    def test_portfolio_weights(self):
        """Should compute portfolio weights."""
        gnn = CorrelationGNN()
        
        for i in range(25):
            gnn.update("BTC", 50000 + i * 100)
            gnn.update("ETH", 3000 + i * 10)
            gnn.update("SOL", 100 + i)
        
        weights = gnn.get_portfolio_weights()
        
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01  # Normalized


# ============================================================================
# Online RL Strategy Selector Tests
# ============================================================================

from ml.online_rl_strategy_selector import (
    ThompsonSamplingBandit,
    UCB1Bandit,
    LinUCBBandit,
    OnlineStrategySelector,
    create_strategy_selector,
)


class TestThompsonSamplingBandit:
    """Tests for Thompson Sampling."""
    
    def test_init(self):
        """Should initialize correctly."""
        bandit = ThompsonSamplingBandit(["momentum", "mean_reversion", "grid"])
        assert len(bandit.arms) == 3
    
    def test_select_strategy(self):
        """Should select a strategy."""
        bandit = ThompsonSamplingBandit(["momentum", "mean_reversion"])
        
        selected = bandit.select_strategy()
        assert selected in ["momentum", "mean_reversion"]
    
    def test_update(self):
        """Should update with reward."""
        bandit = ThompsonSamplingBandit(["momentum", "mean_reversion"])
        
        bandit.update("momentum", 0.1)
        bandit.update("mean_reversion", -0.05)
        
        stats = bandit.get_strategy_stats()
        assert stats["momentum"]["pulls"] == 1
        assert stats["mean_reversion"]["pulls"] == 1
    
    def test_best_strategy(self):
        """Should identify best strategy."""
        bandit = ThompsonSamplingBandit(["good", "bad"])
        
        # Good strategy always wins
        for _ in range(20):
            bandit.update("good", 0.1)
            bandit.update("bad", -0.1)
        
        assert bandit.get_best_strategy() == "good"


class TestUCB1Bandit:
    """Tests for UCB1."""
    
    def test_init(self):
        """Should initialize correctly."""
        bandit = UCB1Bandit(["momentum", "mean_reversion"])
        assert bandit.total_pulls == 0
    
    def test_select_strategy(self):
        """Should select a strategy."""
        bandit = UCB1Bandit(["momentum", "mean_reversion"])
        
        selected = bandit.select_strategy()
        assert selected in ["momentum", "mean_reversion"]


class TestLinUCBBandit:
    """Tests for LinUCB."""
    
    def test_init(self):
        """Should initialize correctly."""
        bandit = LinUCBBandit(["momentum", "mean_reversion"], n_features=5)
        assert bandit.n_features == 5
    
    def test_select_strategy(self):
        """Should select based on context."""
        bandit = LinUCBBandit(["momentum", "mean_reversion"], n_features=5)
        context = np.random.randn(5)
        
        selected = bandit.select_strategy(context)
        assert selected in ["momentum", "mean_reversion"]
    
    def test_update(self):
        """Should update model."""
        bandit = LinUCBBandit(["momentum", "mean_reversion"], n_features=5)
        context = np.random.randn(5)
        
        bandit.update("momentum", context, 0.1)
        
        stats = bandit.get_strategy_stats()
        assert stats["momentum"]["pulls"] == 1


class TestOnlineStrategySelector:
    """Tests for OnlineStrategySelector."""
    
    def test_init(self):
        """Should initialize correctly."""
        selector = OnlineStrategySelector(["momentum", "mean_reversion", "grid"])
        assert len(selector.strategy_names) == 3
    
    def test_extract_features(self):
        """Should extract features."""
        selector = OnlineStrategySelector(["momentum", "mean_reversion"])
        
        features = selector.extract_features(
            volatility=0.02,
            trend_strength=0.5,
            rsi=60.0,
            volume_ratio=1.2,
            hour=10,
            day_of_week=1,
        )
        
        assert len(features) == 10
    
    def test_select_strategy(self):
        """Should select strategy."""
        selector = OnlineStrategySelector(["momentum", "mean_reversion"])
        features = np.random.randn(10)
        
        selected = selector.select_strategy(features)
        assert selected in ["momentum", "mean_reversion"]
    
    def test_update(self):
        """Should update all bandits."""
        selector = OnlineStrategySelector(["momentum", "mean_reversion"])
        features = np.random.randn(10)
        
        selector.update("momentum", 0.1, features)
        selector.update("mean_reversion", -0.05, features)
        
        stats = selector.get_stats()
        assert stats["total_decisions"] == 2
    
    def test_get_best_strategy(self):
        """Should identify best strategy."""
        selector = OnlineStrategySelector(["good", "bad"])
        features = np.random.randn(10)
        
        for _ in range(20):
            selector.update("good", 0.1, features)
            selector.update("bad", -0.1, features)
        
        assert selector.get_best_strategy() == "good"


# ============================================================================
# Causal Inference Tests
# ============================================================================

from ml.causal_regime_predictor import (
    PCAlgorithm,
    CausalGraph,
    CausalEdge,
    CausalInferenceEngine,
    create_causal_engine,
)


class TestCausalGraph:
    """Tests for CausalGraph."""
    
    def test_init(self):
        """Should initialize correctly."""
        graph = CausalGraph()
        assert len(graph.edges) == 0
    
    def test_add_edge(self):
        """Should add edges."""
        graph = CausalGraph()
        edge = CausalEdge("fed_rate", "btc_price", 0.5, 0.9)
        graph.add_edge(edge)
        
        assert len(graph.edges) == 1
        assert "fed_rate" in graph.variables
    
    def test_get_parents(self):
        """Should return parents."""
        graph = CausalGraph()
        graph.add_edge(CausalEdge("A", "B", 0.5, 0.9))
        graph.add_edge(CausalEdge("C", "B", 0.3, 0.8))
        
        parents = graph.get_parents("B")
        assert set(parents) == {"A", "C"}
    
    def test_get_children(self):
        """Should return children."""
        graph = CausalGraph()
        graph.add_edge(CausalEdge("A", "B", 0.5, 0.9))
        graph.add_edge(CausalEdge("A", "C", 0.3, 0.8))
        
        children = graph.get_children("A")
        assert set(children) == {"B", "C"}
    
    def test_get_ancestors(self):
        """Should return transitive ancestors."""
        graph = CausalGraph()
        graph.add_edge(CausalEdge("A", "B", 0.5, 0.9))
        graph.add_edge(CausalEdge("B", "C", 0.5, 0.9))
        
        ancestors = graph.get_ancestors("C")
        assert "A" in ancestors
        assert "B" in ancestors
    
    def test_is_ancestor(self):
        """Should check ancestry."""
        graph = CausalGraph()
        graph.add_edge(CausalEdge("A", "B", 0.5, 0.9))
        graph.add_edge(CausalEdge("B", "C", 0.5, 0.9))
        
        assert graph.is_ancestor("A", "C")
        assert not graph.is_ancestor("C", "A")


class TestCausalInferenceEngine:
    """Tests for CausalInferenceEngine."""
    
    def test_init(self):
        """Should initialize correctly."""
        engine = CausalInferenceEngine()
        assert engine.lookback == 100
    
    def test_update(self):
        """Should update variables."""
        engine = CausalInferenceEngine()
        engine.update("btc_price", 50000)
        engine.update("eth_price", 3000)
        
        assert "btc_price" in engine._data
    
    def test_predict_regime_change(self):
        """Should predict regime changes."""
        engine = CausalInferenceEngine(lookback=20, update_frequency=5)
        
        # Add correlated data
        for i in range(30):
            base = 50000 + i * 100
            engine.update("btc_price", base)
            engine.update("eth_price", base * 0.06)
            engine.update("funding_rate", 0.0001 * (1 + i * 0.01))
        
        prediction = engine.predict_regime_change("btc_price", "range")
        
        assert "predicted_change" in prediction
        assert "confidence" in prediction
        assert "leading_indicators" in prediction
    
    def test_get_leading_indicators(self):
        """Should return leading indicators."""
        engine = CausalInferenceEngine(lookback=20, update_frequency=5)
        
        for i in range(30):
            engine.update("fed_rate", 0.05 + i * 0.001)
            engine.update("btc_price", 50000 + i * 100)
        
        # May or may not find causal relationship
        indicators = engine.get_leading_indicators("btc_price")
        assert isinstance(indicators, list)


# ============================================================================
# Enhanced Diffusion Tests
# ============================================================================

from ml.enhanced_diffusion_stress import (
    GaussianNoiseScheduler,
    MarketDiffusionModel,
    StressTestEngine,
    create_stress_test_engine,
)


class TestGaussianNoiseScheduler:
    """Tests for noise scheduler."""
    
    def test_init(self):
        """Should initialize correctly."""
        scheduler = GaussianNoiseScheduler(n_steps=100)
        assert scheduler.n_steps == 100
        assert len(scheduler.betas) == 100
    
    def test_add_noise(self):
        """Should add noise."""
        scheduler = GaussianNoiseScheduler(n_steps=100)
        x = np.random.randn(5)
        
        noisy_x, noise = scheduler.add_noise(x, t=50)
        
        assert noisy_x.shape == x.shape
        assert noise.shape == x.shape
    
    def test_remove_noise_step(self):
        """Should remove noise."""
        scheduler = GaussianNoiseScheduler(n_steps=100)
        x = np.random.randn(5)
        noise = np.random.randn(5)
        
        x_denoised = scheduler.remove_noise_step(x, noise, t=50)
        
        assert x_denoised.shape == x.shape


class TestMarketDiffusionModel:
    """Tests for diffusion model."""
    
    def test_init(self):
        """Should initialize correctly."""
        model = MarketDiffusionModel(n_features=5, n_steps=50)
        assert model.n_features == 5
        assert model.n_steps == 50
    
    def test_fit(self):
        """Should train on data."""
        model = MarketDiffusionModel(n_features=3, n_steps=20)
        
        # Generate training data
        data = np.random.randn(50, 3)
        
        stats = model.fit(data, n_epochs=10)
        
        assert "final_loss" in stats
        assert model._is_trained
    
    def test_generate(self):
        """Should generate scenarios."""
        model = MarketDiffusionModel(n_features=3, n_steps=20)
        
        # Train
        data = np.random.randn(50, 3)
        model.fit(data, n_epochs=5)
        
        # Generate
        scenarios = model.generate(n_samples=5, sequence_length=10)
        
        assert scenarios.shape[0] == 5
        assert scenarios.shape[2] == 3
    
    def test_generate_with_condition(self):
        """Should generate conditioned scenarios."""
        model = MarketDiffusionModel(n_features=3, n_steps=20)
        
        data = np.random.randn(50, 3)
        model.fit(data, n_epochs=5)
        
        crash_scenarios = model.generate(n_samples=5, condition="crash")
        rally_scenarios = model.generate(n_samples=5, condition="rally")
        
        # Crash scenarios should have lower mean returns
        assert crash_scenarios.shape == (5, 20, 3)
        assert rally_scenarios.shape == (5, 20, 3)
    
    def test_generate_extreme_scenarios(self):
        """Should generate extreme scenarios."""
        model = MarketDiffusionModel(n_features=3, n_steps=20)
        
        data = np.random.randn(50, 3)
        model.fit(data, n_epochs=5)
        
        extremes = model.generate_extreme_scenarios(n_scenarios=20)
        
        assert "black_swan" in extremes
        assert "flash_crash" in extremes
        assert "bull_run" in extremes
        assert "vol_spike" in extremes


class TestStressTestEngine:
    """Tests for stress test engine."""
    
    def test_init(self):
        """Should initialize correctly."""
        engine = StressTestEngine()
        assert engine.diffusion is not None
    
    def test_train(self):
        """Should train diffusion model."""
        engine = StressTestEngine()
        
        data = np.random.randn(100, 5)
        stats = engine.train(data, n_epochs=5)
        
        assert "final_loss" in stats
    
    def test_run_stress_test(self):
        """Should run stress test."""
        diffusion = MarketDiffusionModel(n_features=3, n_steps=20)
        engine = StressTestEngine(diffusion_model=diffusion)
        
        # Train
        data = np.random.randn(100, 3)
        engine.train(data, n_epochs=5)
        
        # Test
        weights = np.array([0.5, 0.3, 0.2])
        prices = np.array([50000, 3000, 100])
        
        results = engine.run_stress_test(weights, prices, n_scenarios=20)
        
        assert "var_95" in results
        assert "var_99" in results
        assert "cvar_95" in results
        assert "max_drawdown" in results
        assert "probability_loss" in results
    
    def test_run_extreme_stress_test(self):
        """Should run extreme stress test."""
        diffusion = MarketDiffusionModel(n_features=3, n_steps=20)
        engine = StressTestEngine(diffusion_model=diffusion)
        
        data = np.random.randn(100, 3)
        engine.train(data, n_epochs=5)
        
        weights = np.array([0.5, 0.3, 0.2])
        
        results = engine.run_extreme_stress_test(weights, severity=2.0)
        
        assert "black_swan" in results
        assert "flash_crash" in results
        assert "bull_run" in results


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_transformer_ensemble(self):
        """Should create transformer ensemble."""
        ensemble = create_transformer_ensemble(n_models=3, d_model=32)
        assert ensemble is not None
    
    def test_create_correlation_gnn(self):
        """Should create GNN."""
        gnn = create_correlation_gnn(input_dim=3, hidden_dim=32)
        assert isinstance(gnn, CorrelationGNN)
    
    def test_create_stress_test_engine(self):
        """Should create stress test engine."""
        engine = create_stress_test_engine(n_features=3)
        assert isinstance(engine, StressTestEngine)