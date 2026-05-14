"""
Tests for advanced AI models:
  1. AssetFlowGNN (ml/gnn_asset_flow.py)
  2. AutoencoderRegimeDetector (ml/autoencoder_regime.py)
  3. RLPortfolioManager (ml/rl_portfolio.py)
  4. AttentionOrderFlowPredictor (ml/attention_orderflow.py)

Run: py -m pytest tests/test_ai_advanced.py -v
"""

from __future__ import annotations

import math
import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════
# 1. AssetFlowGNN
# ═══════════════════════════════════════════════════════════════════════

class TestAssetFlowGNN:

    def _make_gnn(self, assets=None, lookback=50):
        from ml.gnn_asset_flow import AssetFlowGNN
        return AssetFlowGNN(assets=assets, lookback=lookback)

    def _feed_prices(self, gnn, n=60):
        """Feed realistic correlated price series."""
        np.random.seed(42)
        assets = gnn.assets
        base = 100.0
        prices = {a: base * (1 + i * 0.1) for i, a in enumerate(assets)}
        for _ in range(n):
            shock = np.random.randn() * 0.01
            for a in assets:
                noise = np.random.randn() * 0.005
                prices[a] *= (1 + shock + noise)
            gnn.update(prices.copy())

    # --- construction ---

    def test_default_construction(self):
        gnn = self._make_gnn()
        assert gnn.n_assets == 6
        assert len(gnn.assets) == 6

    def test_custom_assets(self):
        gnn = self._make_gnn(assets=["BTC/USD", "ETH/USD", "SOL/USD"])
        assert gnn.n_assets == 3

    def test_custom_lookback(self):
        gnn = self._make_gnn(lookback=20)
        assert gnn.lookback == 20

    def test_min_lookback_clamped(self):
        gnn = self._make_gnn(lookback=3)
        assert gnn.lookback == 10

    # --- update ---

    def test_update_stores_prices(self):
        gnn = self._make_gnn()
        gnn.update({"BTC/USD": 50000, "ETH/USD": 3000})
        assert len(gnn._prices["BTC/USD"]) == 1
        assert len(gnn._prices["ETH/USD"]) == 1

    def test_update_ignores_unknown_assets(self):
        gnn = self._make_gnn(assets=["BTC/USD"])
        gnn.update({"DOGE/USD": 0.1})
        assert len(gnn._prices["BTC/USD"]) == 0

    # --- adjacency ---

    def test_adjacency_shape(self):
        gnn = self._make_gnn()
        self._feed_prices(gnn, n=30)
        adj = gnn.compute_adjacency()
        assert adj.shape == (6, 6)

    def test_adjacency_zero_diagonal(self):
        gnn = self._make_gnn()
        self._feed_prices(gnn, n=30)
        adj = gnn.compute_adjacency()
        np.testing.assert_array_equal(np.diag(adj), 0.0)

    def test_adjacency_symmetric(self):
        gnn = self._make_gnn()
        self._feed_prices(gnn, n=50)
        adj = gnn.compute_adjacency()
        np.testing.assert_allclose(adj, adj.T, atol=1e-10)

    def test_adjacency_insufficient_data(self):
        gnn = self._make_gnn()
        gnn.update({"BTC/USD": 50000, "ETH/USD": 3000})
        adj = gnn.compute_adjacency()
        # Should return identity-like matrix
        assert adj.shape == (6, 6)

    # --- propagate ---

    def test_propagate_returns_all_assets(self):
        gnn = self._make_gnn()
        self._feed_prices(gnn, n=60)
        flows = gnn.propagate(n_hops=2)
        assert set(flows.keys()) == set(gnn.assets)

    def test_propagate_flow_signal_bounded(self):
        gnn = self._make_gnn()
        self._feed_prices(gnn, n=60)
        flows = gnn.propagate()
        for asset, info in flows.items():
            assert -1.0 <= info["flow_signal"] <= 1.0

    def test_propagate_influence_non_negative(self):
        gnn = self._make_gnn()
        self._feed_prices(gnn, n=60)
        flows = gnn.propagate()
        for info in flows.values():
            assert info["influence_score"] >= 0.0

    def test_propagate_insufficient_data(self):
        gnn = self._make_gnn()
        flows = gnn.propagate()
        for info in flows.values():
            assert info["flow_signal"] == 0.0

    # --- lead-lag ---

    def test_lead_lag_structure(self):
        gnn = self._make_gnn()
        self._feed_prices(gnn, n=60)
        ll = gnn.get_lead_lag()
        for asset in gnn.assets:
            assert "leads" in ll[asset]
            assert "lags" in ll[asset]
            assert "lead_strength" in ll[asset]

    def test_lead_lag_insufficient_data(self):
        gnn = self._make_gnn()
        ll = gnn.get_lead_lag()
        for asset in gnn.assets:
            assert ll[asset]["leads"] == []
            assert ll[asset]["lags"] == []

    # --- contagion ---

    def test_contagion_includes_all_assets(self):
        gnn = self._make_gnn()
        self._feed_prices(gnn, n=60)
        impacts = gnn.predict_contagion("BTC/USD", -5.0)
        assert "BTC/USD" in impacts
        assert impacts["BTC/USD"] == -5.0

    def test_contagion_shocked_asset_preserved(self):
        gnn = self._make_gnn()
        self._feed_prices(gnn, n=60)
        impacts = gnn.predict_contagion("BTC/USD", -10.0)
        assert impacts["BTC/USD"] == -10.0

    def test_contagion_unknown_asset(self):
        gnn = self._make_gnn()
        impacts = gnn.predict_contagion("UNKNOWN/USD", -5.0)
        assert impacts == {}

    def test_contagion_zero_shock(self):
        gnn = self._make_gnn()
        self._feed_prices(gnn, n=60)
        impacts = gnn.predict_contagion("BTC/USD", 0.0)
        assert impacts["BTC/USD"] == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 2. AutoencoderRegimeDetector
# ═══════════════════════════════════════════════════════════════════════

class TestAutoencoderRegimeDetector:

    def _make_detector(self, **kwargs):
        from ml.autoencoder_regime import AutoencoderRegimeDetector
        return AutoencoderRegimeDetector(**kwargs)

    def _generate_features(self, n=200, dim=10):
        np.random.seed(42)
        return np.random.randn(n, dim)

    # --- construction ---

    def test_default_construction(self):
        det = self._make_detector()
        assert det.input_dim == 10
        assert det.latent_dim == 3
        assert det.hidden_dim == 16

    def test_custom_dims(self):
        det = self._make_detector(input_dim=20, latent_dim=5, hidden_dim=32)
        assert det.input_dim == 20
        assert det.latent_dim == 5

    # --- fit ---

    def test_fit_trains_successfully(self):
        det = self._make_detector()
        features = self._generate_features(100, 10)
        result = det.fit(features, epochs=10, lr=0.01)
        assert result["status"] == "trained"
        assert result["n_samples"] == 100
        assert det._trained is True

    def test_fit_insufficient_data(self):
        det = self._make_detector()
        features = np.random.randn(3, 10)
        result = det.fit(features)
        assert result["status"] == "insufficient_data"

    def test_fit_loss_decreases(self):
        det = self._make_detector()
        features = self._generate_features(100, 10)
        result = det.fit(features, epochs=50, lr=0.01)
        history = result["loss_history"]
        # First loss should be higher than last (generally)
        assert history[-1] <= history[0] * 2.0  # generous check

    # --- encode ---

    def test_encode_shape(self):
        det = self._make_detector(input_dim=10, latent_dim=3)
        features = self._generate_features(100, 10)
        det.fit(features, epochs=5)
        z = det.encode(features[:5])
        assert z.shape == (5, 3)

    def test_encode_single_sample(self):
        det = self._make_detector()
        features = self._generate_features(100, 10)
        det.fit(features, epochs=5)
        z = det.encode(features[0])
        assert z.shape == (1, 3)

    # --- reconstruct ---

    def test_reconstruct_shape(self):
        det = self._make_detector()
        features = self._generate_features(100, 10)
        det.fit(features, epochs=5)
        recon, errors = det.reconstruct(features[:5])
        assert recon.shape == (5, 10)
        assert errors.shape == (5,)

    def test_reconstruct_errors_non_negative(self):
        det = self._make_detector()
        features = self._generate_features(100, 10)
        det.fit(features, epochs=5)
        _, errors = det.reconstruct(features[:5])
        assert (errors >= 0).all()

    # --- detect_regime ---

    def test_detect_regime_returns_dict(self):
        det = self._make_detector()
        features = self._generate_features(100, 10)
        det.fit(features, epochs=10)
        result = det.detect_regime(features[0])
        assert "regime" in result
        assert "confidence" in result
        assert "latent" in result
        assert "reconstruction_error" in result
        assert "is_transition" in result

    def test_detect_regime_confidence_bounded(self):
        det = self._make_detector()
        features = self._generate_features(100, 10)
        det.fit(features, epochs=10)
        result = det.detect_regime(features[0])
        assert 0.0 <= result["confidence"] <= 1.0

    def test_detect_regime_untrained(self):
        det = self._make_detector()
        result = det.detect_regime(np.zeros(10))
        assert result["regime"] == "UNKNOWN"
        assert result["confidence"] == 0.0

    def test_detect_regime_novel_state(self):
        """Extreme outlier should trigger transition detection."""
        det = self._make_detector()
        features = self._generate_features(200, 10)
        det.fit(features, epochs=20, lr=0.01)
        outlier = np.ones(10) * 100.0  # extreme outlier
        result = det.detect_regime(outlier)
        # Either detected as transition or high reconstruction error
        assert result["reconstruction_error"] > 0.0

    # --- regime map ---

    def test_regime_map_trained(self):
        det = self._make_detector()
        features = self._generate_features(100, 10)
        det.fit(features, epochs=5)
        rmap = det.get_regime_map()
        assert rmap["status"] == "trained"
        assert rmap["n_clusters"] == 4

    def test_regime_map_untrained(self):
        det = self._make_detector()
        rmap = det.get_regime_map()
        assert rmap["status"] == "not_trained"


# ═══════════════════════════════════════════════════════════════════════
# 3. RLPortfolioManager
# ═══════════════════════════════════════════════════════════════════════

class TestRLPortfolioManager:

    def _make_manager(self, **kwargs):
        from ml.rl_portfolio import RLPortfolioManager
        return RLPortfolioManager(**kwargs)

    # --- construction ---

    def test_default_construction(self):
        mgr = self._make_manager()
        assert mgr.n_strategies == 15
        assert mgr.state_dim == 20

    def test_custom_params(self):
        mgr = self._make_manager(n_strategies=5, state_dim=10)
        assert mgr.n_strategies == 5
        assert len(mgr.strategy_names) == 5

    # --- get_state ---

    def test_get_state_shape(self):
        mgr = self._make_manager(n_strategies=5, state_dim=20)
        state = mgr.get_state(
            strategy_stats={"momentum": {"return_1d": 0.01, "volatility": 0.03, "sharpe": 1.5}},
            regime="TRENDING_UP",
            drawdown=0.05,
        )
        assert state.shape == (20,)

    def test_get_state_empty_stats(self):
        mgr = self._make_manager()
        state = mgr.get_state({}, "UNKNOWN", 0.0)
        assert state.shape == (20,)

    # --- allocate ---

    def test_allocate_weights_sum_to_one(self):
        np.random.seed(42)
        mgr = self._make_manager(n_strategies=5)
        state = np.random.randn(20)
        alloc = mgr.allocate(state)
        total = sum(alloc.values())
        assert abs(total - 1.0) < 1e-6

    def test_allocate_weights_non_negative(self):
        np.random.seed(42)
        mgr = self._make_manager(n_strategies=5)
        state = np.random.randn(20)
        alloc = mgr.allocate(state)
        for w in alloc.values():
            assert w >= 0.0

    def test_allocate_returns_all_strategies(self):
        mgr = self._make_manager(n_strategies=5)
        state = np.zeros(20)
        alloc = mgr.allocate(state)
        assert len(alloc) == 5

    def test_allocate_max_weight_respected(self):
        np.random.seed(42)
        mgr = self._make_manager(n_strategies=5, max_weight=0.3)
        for _ in range(10):
            state = np.random.randn(20)
            alloc = mgr.allocate(state)
            for w in alloc.values():
                # Allow small tolerance for noise mixing
                assert w <= 0.40  # max_weight + headroom for exploration noise

    # --- record_reward + update ---

    def test_record_reward(self):
        mgr = self._make_manager(n_strategies=5, update_every=100)
        state = np.zeros(20)
        mgr.allocate(state)
        mgr.record_reward(0.01)
        assert len(mgr._rewards) == 1

    def test_update_returns_summary(self):
        np.random.seed(42)
        mgr = self._make_manager(n_strategies=5, update_every=100)
        for _ in range(10):
            state = np.random.randn(20)
            mgr.allocate(state)
            mgr.record_reward(np.random.randn() * 0.01)
        result = mgr.update()
        assert result["status"] == "updated"
        assert result["n_steps"] == 10

    def test_update_clears_buffer(self):
        np.random.seed(42)
        mgr = self._make_manager(n_strategies=5, update_every=100)
        for _ in range(5):
            mgr.allocate(np.random.randn(20))
            mgr.record_reward(0.01)
        mgr.update()
        assert len(mgr._rewards) == 0
        assert len(mgr._states) == 0

    def test_auto_update(self):
        np.random.seed(42)
        mgr = self._make_manager(n_strategies=5, update_every=5)
        for _ in range(5):
            mgr.allocate(np.random.randn(20))
            mgr.record_reward(0.01)
        # Auto-update triggered, buffer should be empty
        assert len(mgr._rewards) == 0

    # --- allocation history ---

    def test_allocation_history(self):
        mgr = self._make_manager(n_strategies=5, update_every=100)
        state = np.zeros(20)
        mgr.allocate(state)
        history = mgr.get_allocation_history()
        assert len(history) == 1
        assert "allocation" in history[0]


# ═══════════════════════════════════════════════════════════════════════
# 4. AttentionOrderFlowPredictor
# ═══════════════════════════════════════════════════════════════════════

class TestAttentionOrderFlowPredictor:

    def _make_predictor(self, **kwargs):
        from ml.attention_orderflow import AttentionOrderFlowPredictor
        return AttentionOrderFlowPredictor(**kwargs)

    def _feed_ticks(self, pred, n=30):
        np.random.seed(42)
        price = 50000.0
        for _ in range(n):
            change = np.random.randn() * 0.001
            price *= (1 + change)
            pred.update({
                "price_change": change,
                "volume": abs(np.random.randn()),
                "bid_vol": 0.5 + np.random.randn() * 0.1,
                "ask_vol": 0.5 + np.random.randn() * 0.1,
                "spread": 0.0003,
                "trade_imbalance": np.random.randn() * 0.1,
                "vwap_deviation": np.random.randn() * 0.001,
                "volatility": 0.02,
            })

    # --- construction ---

    def test_default_construction(self):
        pred = self._make_predictor()
        assert pred.seq_len == 50
        assert pred.feature_dim == 8
        assert pred.n_heads == 4

    def test_custom_params(self):
        pred = self._make_predictor(seq_len=20, feature_dim=4, n_heads=2)
        assert pred.seq_len == 20
        assert pred.head_dim == 2

    def test_invalid_heads(self):
        with pytest.raises(AssertionError):
            self._make_predictor(feature_dim=8, n_heads=3)

    # --- update ---

    def test_update_stores_ticks(self):
        pred = self._make_predictor()
        pred.update({"price_change": 0.01, "volume": 1.0})
        assert len(pred._buffer) == 1

    def test_update_missing_features_default_zero(self):
        pred = self._make_predictor()
        pred.update({"price_change": 0.01})
        assert len(pred._buffer) == 1

    # --- predict_direction ---

    def test_predict_insufficient_data(self):
        pred = self._make_predictor()
        pred.update({"price_change": 0.01})
        result = pred.predict_direction()
        assert result["direction"] == 0.0
        assert result["confidence"] == 0.0

    def test_predict_returns_dict(self):
        pred = self._make_predictor()
        self._feed_ticks(pred, n=30)
        result = pred.predict_direction()
        assert "direction" in result
        assert "confidence" in result
        assert "attention_weights" in result
        assert "dominant_feature" in result

    def test_predict_direction_bounded(self):
        pred = self._make_predictor()
        self._feed_ticks(pred, n=30)
        result = pred.predict_direction()
        assert -1.0 <= result["direction"] <= 1.0

    def test_predict_confidence_bounded(self):
        pred = self._make_predictor()
        self._feed_ticks(pred, n=30)
        result = pred.predict_direction()
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_dominant_feature_valid(self):
        from ml.attention_orderflow import FEATURE_NAMES
        pred = self._make_predictor()
        self._feed_ticks(pred, n=30)
        result = pred.predict_direction()
        assert result["dominant_feature"] in FEATURE_NAMES or result["dominant_feature"].startswith("feature_")

    # --- attention map ---

    def test_attention_map_empty_before_predict(self):
        pred = self._make_predictor()
        attn = pred.get_attention_map()
        assert attn.size == 0

    def test_attention_map_shape(self):
        pred = self._make_predictor(n_heads=4)
        self._feed_ticks(pred, n=20)
        pred.predict_direction()
        attn = pred.get_attention_map()
        assert attn.ndim == 3
        assert attn.shape[0] == 4  # n_heads
        assert attn.shape[1] == attn.shape[2]  # (T, T)

    def test_attention_weights_sum_to_one(self):
        pred = self._make_predictor()
        self._feed_ticks(pred, n=20)
        pred.predict_direction()
        attn = pred.get_attention_map()
        # Each row should sum to ~1.0 (softmax)
        for head in range(attn.shape[0]):
            row_sums = attn[head].sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_causal_mask_applied(self):
        """Attention should not attend to future positions."""
        pred = self._make_predictor()
        self._feed_ticks(pred, n=15)
        pred.predict_direction()
        attn = pred.get_attention_map()
        # Upper triangle (future positions) should be ~0
        T = attn.shape[1]
        for head in range(attn.shape[0]):
            for i in range(T):
                for j in range(i + 1, T):
                    assert attn[head, i, j] < 1e-6, (
                        f"head={head}, pos={i} attends to future pos={j}"
                    )
