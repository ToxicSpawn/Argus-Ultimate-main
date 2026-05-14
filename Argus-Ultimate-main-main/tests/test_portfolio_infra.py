#!/usr/bin/env python3
"""
Tests for portfolio infrastructure modules (Batch — March 2026).

Covers:
    - risk/risk_parity_allocator.py   (RiskParityAllocator)
    - risk/factor_model.py            (FactorModel, FactorDecomposition)
    - risk/hierarchical_risk_parity.py (HierarchicalRiskParity)
    - core/feature_store.py           (FeatureStore)
    - core/event_bus.py               (EventBus)
    - ml/onnx_model_server.py         (ONNXModelServer)

60+ tests total.
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import tempfile
import threading
import time
import unittest

# ---------------------------------------------------------------------------
# Synthetic return generator
# ---------------------------------------------------------------------------

def _make_returns(n: int = 60, seed: int = 42, mu: float = 0.001, sigma: float = 0.02) -> list:
    """Generate a list of synthetic periodic returns."""
    rng = random.Random(seed)
    return [rng.gauss(mu, sigma) for _ in range(n)]


def _make_multi_returns(symbols: list, n: int = 60, base_seed: int = 42) -> dict:
    """Generate correlated-ish returns for multiple symbols."""
    market = _make_returns(n, seed=base_seed, mu=0.001, sigma=0.015)
    out = {}
    for i, sym in enumerate(symbols):
        rng = random.Random(base_seed + i + 1)
        beta = 0.5 + rng.random() * 1.0  # 0.5–1.5
        idio = [rng.gauss(0, 0.01) for _ in range(n)]
        out[sym] = [market[t] * beta + idio[t] for t in range(n)]
    return out


# ===================================================================
# RiskParityAllocator tests
# ===================================================================

class TestRiskParityAllocator(unittest.TestCase):
    """Tests for risk.risk_parity_allocator.RiskParityAllocator."""

    def setUp(self):
        from risk.risk_parity_allocator import RiskParityAllocator
        self.allocator = RiskParityAllocator(min_history=10)
        self.symbols = ["BTC/AUD", "ETH/AUD", "SOL/AUD", "ADA/AUD"]
        self.returns = _make_multi_returns(self.symbols, n=60)

    def test_empty_returns(self):
        w = self.allocator.compute_weights({})
        self.assertEqual(w, {})

    def test_single_asset(self):
        w = self.allocator.compute_weights({"BTC/AUD": _make_returns(30)})
        self.assertAlmostEqual(w["BTC/AUD"], 1.0)

    def test_weights_sum_to_one(self):
        w = self.allocator.compute_weights(self.returns)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)

    def test_all_weights_positive(self):
        w = self.allocator.compute_weights(self.returns)
        for sym, wt in w.items():
            self.assertGreaterEqual(wt, 0.0, f"{sym} weight is negative")

    def test_max_weight_cap(self):
        alloc = __import__("risk.risk_parity_allocator", fromlist=["RiskParityAllocator"]).RiskParityAllocator(
            min_history=10, max_weight=0.30
        )
        w = alloc.compute_weights(self.returns)
        for sym, wt in w.items():
            self.assertLessEqual(wt, 0.30 + 1e-6, f"{sym} exceeds max_weight")

    def test_insufficient_history_fallback(self):
        short = {s: _make_returns(5) for s in self.symbols}
        alloc = __import__("risk.risk_parity_allocator", fromlist=["RiskParityAllocator"]).RiskParityAllocator(
            min_history=20
        )
        w = alloc.compute_weights(short)
        # Should fall back to equal weight
        for wt in w.values():
            self.assertAlmostEqual(wt, 0.25, places=4)

    def test_risk_contributions(self):
        w = self.allocator.compute_weights(self.returns)
        rc = self.allocator.get_risk_contributions(w, self.returns)
        self.assertEqual(set(rc.keys()), set(self.symbols))
        self.assertAlmostEqual(sum(rc.values()), 1.0, places=4)

    def test_rebalance_needed_true(self):
        current = {"BTC/AUD": 0.5, "ETH/AUD": 0.5}
        target = {"BTC/AUD": 0.3, "ETH/AUD": 0.7}
        self.assertTrue(self.allocator.rebalance_needed(current, target, threshold_pct=5.0))

    def test_rebalance_needed_false(self):
        current = {"BTC/AUD": 0.50, "ETH/AUD": 0.50}
        target = {"BTC/AUD": 0.51, "ETH/AUD": 0.49}
        self.assertFalse(self.allocator.rebalance_needed(current, target, threshold_pct=5.0))

    def test_marginal_risk(self):
        w = self.allocator.compute_weights(self.returns)
        mr = self.allocator.get_marginal_risk(w, self.returns, "BTC/AUD")
        self.assertIsInstance(mr, float)
        self.assertGreater(mr, 0.0)

    def test_marginal_risk_unknown_symbol(self):
        w = self.allocator.compute_weights(self.returns)
        mr = self.allocator.get_marginal_risk(w, self.returns, "DOGE/AUD")
        self.assertEqual(mr, 0.0)

    def test_two_assets(self):
        ret2 = _make_multi_returns(["A", "B"], n=60)
        w = self.allocator.compute_weights(ret2)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)
        self.assertIn("A", w)
        self.assertIn("B", w)


# ===================================================================
# FactorModel tests
# ===================================================================

class TestFactorModel(unittest.TestCase):
    """Tests for risk.factor_model.FactorModel."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_factor.db")
        from risk.factor_model import FactorModel
        self.fm = FactorModel(db_path=self.db_path, momentum_window=3, vol_window=3)
        self.rng = random.Random(99)

    def _seed_data(self, symbol="BTC/AUD", n=50):
        market = [self.rng.gauss(0.001, 0.015) for _ in range(n)]
        for i in range(n):
            ret = market[i] * 1.2 + self.rng.gauss(0.0005, 0.005)
            self.fm.update_returns(symbol, ret, market[i], timestamp=1000000 + i * 86400)

    def test_update_returns(self):
        self.fm.update_returns("BTC/AUD", 0.02, 0.015)
        self.assertEqual(len(self.fm._returns["BTC/AUD"]), 1)

    def test_decompose_insufficient_data(self):
        self.fm.update_returns("BTC/AUD", 0.01, 0.01)
        result = self.fm.decompose("BTC/AUD", lookback_days=30)
        self.assertIsNone(result)

    def test_decompose_success(self):
        self._seed_data("BTC/AUD", 50)
        decomp = self.fm.decompose("BTC/AUD", lookback_days=40)
        self.assertIsNotNone(decomp)
        self.assertEqual(decomp.symbol, "BTC/AUD")
        self.assertIsInstance(decomp.alpha, float)
        self.assertIsInstance(decomp.market_beta, float)
        self.assertGreaterEqual(decomp.r_squared, 0.0)
        self.assertLessEqual(decomp.r_squared, 1.0)

    def test_decompose_beta_positive_for_correlated(self):
        """Asset with positive beta to market should show positive market_beta."""
        self._seed_data("BTC/AUD", 60)
        decomp = self.fm.decompose("BTC/AUD", lookback_days=50)
        self.assertIsNotNone(decomp)
        # We seeded with beta=1.2, so expect positive
        self.assertGreater(decomp.market_beta, 0.0)

    def test_portfolio_factor_exposure(self):
        self._seed_data("BTC/AUD", 50)
        self._seed_data("ETH/AUD", 50)
        self.fm.decompose("BTC/AUD", lookback_days=40)
        self.fm.decompose("ETH/AUD", lookback_days=40)
        exposure = self.fm.get_portfolio_factor_exposure({"BTC/AUD": 0.6, "ETH/AUD": 0.4})
        self.assertIn("alpha", exposure)
        self.assertIn("market_beta", exposure)
        self.assertIn("momentum", exposure)
        self.assertIn("volatility", exposure)

    def test_alpha_ranked(self):
        self._seed_data("BTC/AUD", 50)
        self._seed_data("ETH/AUD", 50)
        self.fm.decompose("BTC/AUD", lookback_days=40)
        self.fm.decompose("ETH/AUD", lookback_days=40)
        ranked = self.fm.get_alpha_ranked()
        self.assertEqual(len(ranked), 2)
        # Should be descending
        self.assertGreaterEqual(ranked[0][1], ranked[1][1])

    def test_sqlite_persistence(self):
        self._seed_data("BTC/AUD", 50)
        self.fm.decompose("BTC/AUD", lookback_days=40)
        # Check DB has rows
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM factor_returns").fetchone()[0]
            self.assertGreater(count, 0)
            decomp_count = conn.execute("SELECT COUNT(*) FROM factor_decompositions").fetchone()[0]
            self.assertGreater(decomp_count, 0)

    def test_factor_decomposition_dataclass(self):
        from risk.factor_model import FactorDecomposition
        fd = FactorDecomposition(
            symbol="TEST", alpha=0.05, market_beta=1.1, momentum_loading=0.3,
            volatility_loading=-0.2, residual_pct=0.15, r_squared=0.85,
        )
        self.assertEqual(fd.symbol, "TEST")
        self.assertIsInstance(fd.timestamp, float)


# ===================================================================
# HierarchicalRiskParity tests
# ===================================================================

class TestHierarchicalRiskParity(unittest.TestCase):
    """Tests for risk.hierarchical_risk_parity.HierarchicalRiskParity."""

    def setUp(self):
        from risk.hierarchical_risk_parity import HierarchicalRiskParity
        self.hrp = HierarchicalRiskParity(min_history=10)
        self.symbols = ["BTC/AUD", "ETH/AUD", "SOL/AUD", "ADA/AUD", "DOT/AUD"]
        self.returns = _make_multi_returns(self.symbols, n=60)

    def test_empty_returns(self):
        w = self.hrp.compute_weights({})
        self.assertEqual(w, {})

    def test_single_asset(self):
        w = self.hrp.compute_weights({"BTC/AUD": _make_returns(30)})
        self.assertAlmostEqual(w["BTC/AUD"], 1.0)

    def test_weights_sum_to_one(self):
        w = self.hrp.compute_weights(self.returns)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=5)

    def test_all_weights_positive(self):
        w = self.hrp.compute_weights(self.returns)
        for sym, wt in w.items():
            self.assertGreater(wt, 0.0, f"{sym} weight should be positive")

    def test_weights_less_than_one(self):
        w = self.hrp.compute_weights(self.returns)
        for sym, wt in w.items():
            self.assertLess(wt, 1.0, f"{sym} weight should be < 1")

    def test_two_assets(self):
        ret2 = _make_multi_returns(["A", "B"], n=60)
        w = self.hrp.compute_weights(ret2)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)

    def test_get_clusters(self):
        clusters = self.hrp.get_clusters(self.returns, n_clusters=2)
        self.assertIsInstance(clusters, list)
        self.assertGreaterEqual(len(clusters), 1)
        all_syms = [s for c in clusters for s in c]
        self.assertEqual(sorted(all_syms), sorted(self.symbols))

    def test_get_clusters_single(self):
        clusters = self.hrp.get_clusters(self.returns, n_clusters=5)
        self.assertGreaterEqual(len(clusters), 1)

    def test_get_dendrogram_data(self):
        data = self.hrp.get_dendrogram_data(self.returns)
        self.assertIn("symbols", data)
        self.assertIn("linkage", data)
        self.assertIn("sort_order", data)
        self.assertEqual(len(data["symbols"]), len(self.symbols))
        self.assertEqual(len(data["linkage"]), len(self.symbols) - 1)

    def test_insufficient_history_fallback(self):
        short = {s: _make_returns(3) for s in self.symbols}
        w = self.hrp.compute_weights(short)
        # Should be equal weight fallback
        for wt in w.values():
            self.assertAlmostEqual(wt, 1.0 / len(self.symbols), places=4)


# ===================================================================
# FeatureStore tests
# ===================================================================

class TestFeatureStore(unittest.TestCase):
    """Tests for core.feature_store.FeatureStore."""

    def setUp(self):
        from core.feature_store import FeatureStore
        self.store = FeatureStore(default_ttl_s=10.0, background=False)

    def test_set_and_get(self):
        self.store.set("BTC/AUD", "rsi_14", 62.5)
        self.assertEqual(self.store.get("BTC/AUD", "rsi_14"), 62.5)

    def test_get_missing_symbol(self):
        self.assertIsNone(self.store.get("NOPE", "rsi_14"))

    def test_get_missing_feature(self):
        self.store.set("BTC/AUD", "rsi_14", 50.0)
        self.assertIsNone(self.store.get("BTC/AUD", "macd"))

    def test_ttl_expiry(self):
        self.store.set("BTC/AUD", "fast", 1.0, ttl_s=0.01)
        time.sleep(0.02)
        self.assertIsNone(self.store.get("BTC/AUD", "fast"))

    def test_no_expiry(self):
        self.store.set("BTC/AUD", "perm", 42, ttl_s=0)
        # Should not expire (ttl=0 means infinite)
        self.assertEqual(self.store.get("BTC/AUD", "perm"), 42)

    def test_get_all(self):
        self.store.set("BTC/AUD", "rsi", 60.0)
        self.store.set("BTC/AUD", "macd", 0.5)
        all_feats = self.store.get_all("BTC/AUD")
        self.assertEqual(all_feats, {"rsi": 60.0, "macd": 0.5})

    def test_get_all_empty(self):
        self.assertEqual(self.store.get_all("NOPE"), {})

    def test_get_feature_vector(self):
        self.store.set("BTC/AUD", "rsi", 60.0)
        self.store.set("BTC/AUD", "vol", 0.03)
        vec = self.store.get_feature_vector("BTC/AUD", ["rsi", "vol", "missing"])
        self.assertEqual(vec, [60.0, 0.03, None])

    def test_set_batch(self):
        self.store.set_batch("ETH/AUD", {"rsi": 55.0, "spread": 0.1, "depth": 100})
        self.assertEqual(self.store.get("ETH/AUD", "rsi"), 55.0)
        self.assertEqual(self.store.get("ETH/AUD", "spread"), 0.1)

    def test_get_stats(self):
        self.store.set("BTC/AUD", "rsi", 60.0)
        self.store.set("ETH/AUD", "rsi", 55.0)
        stats = self.store.get_stats()
        self.assertEqual(stats["total_features"], 2)
        self.assertEqual(stats["symbols_count"], 2)
        self.assertGreaterEqual(stats["total_sets"], 2)

    def test_cleanup_removes_expired(self):
        self.store.set("BTC/AUD", "old", 1.0, ttl_s=0.01)
        self.store.set("BTC/AUD", "new", 2.0, ttl_s=999)
        time.sleep(0.02)
        removed = self.store.cleanup()
        self.assertEqual(removed, 1)
        self.assertIsNone(self.store.get("BTC/AUD", "old"))
        self.assertEqual(self.store.get("BTC/AUD", "new"), 2.0)

    def test_thread_safety(self):
        """Concurrent writes should not corrupt the store."""
        errors = []

        def writer(sym, n):
            try:
                for i in range(n):
                    self.store.set(sym, f"feat_{i}", i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"SYM{t}", 50)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        # Each of 5 threads wrote 50 features
        self.assertGreaterEqual(self.store.get_stats()["total_sets"], 250)

    def test_overwrite(self):
        self.store.set("BTC/AUD", "rsi", 50.0)
        self.store.set("BTC/AUD", "rsi", 70.0)
        self.assertEqual(self.store.get("BTC/AUD", "rsi"), 70.0)

    def test_background_cleaner_init(self):
        from core.feature_store import FeatureStore
        store = FeatureStore(background=True, cleanup_interval_s=0.1)
        try:
            self.assertIsNotNone(store._cleaner_thread)
            self.assertTrue(store._cleaner_thread.is_alive())
        finally:
            store.stop()


# ===================================================================
# EventBus tests
# ===================================================================

class TestEventBus(unittest.TestCase):
    """Tests for core.event_bus.EventBus."""

    def setUp(self):
        from core.event_bus import EventBus
        self.bus = EventBus(history_size=100)

    def test_subscribe_and_publish(self):
        received = []
        def handler(event_type, data):
            received.append((event_type, data))

        self.bus.subscribe("price_update", handler)
        self.bus.publish("price_update", {"price": 95000})
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0][0], "price_update")
        self.assertEqual(received[0][1]["price"], 95000)

    def test_priority_ordering(self):
        order = []
        def low(et, d):
            order.append("low")
        def high(et, d):
            order.append("high")

        self.bus.subscribe("test", low, priority=1)
        self.bus.subscribe("test", high, priority=10)
        self.bus.publish("test")
        self.assertEqual(order, ["high", "low"])

    def test_unsubscribe(self):
        received = []
        def handler(et, d):
            received.append(1)

        self.bus.subscribe("test", handler)
        self.bus.publish("test")
        self.assertEqual(len(received), 1)

        result = self.bus.unsubscribe("test", handler)
        self.assertTrue(result)
        self.bus.publish("test")
        self.assertEqual(len(received), 1)  # no new call

    def test_unsubscribe_not_found(self):
        def handler(et, d):
            pass
        self.assertFalse(self.bus.unsubscribe("test", handler))

    def test_publish_returns_handler_count(self):
        def h1(et, d): pass
        def h2(et, d): pass
        self.bus.subscribe("test", h1)
        self.bus.subscribe("test", h2)
        count = self.bus.publish("test")
        self.assertEqual(count, 2)

    def test_publish_no_subscribers(self):
        count = self.bus.publish("nobody_listening", {"data": 1})
        self.assertEqual(count, 0)

    def test_handler_exception_doesnt_break_others(self):
        received = []
        def bad(et, d):
            raise ValueError("boom")
        def good(et, d):
            received.append(1)

        self.bus.subscribe("test", bad, priority=10)
        self.bus.subscribe("test", good, priority=1)
        count = self.bus.publish("test")
        self.assertEqual(len(received), 1)
        # bad raised but good still ran; bad counts as not invoked
        self.assertEqual(count, 1)

    def test_get_stats(self):
        def h(et, d): pass
        self.bus.subscribe("a", h)
        self.bus.subscribe("b", h)
        self.bus.publish("a")
        self.bus.publish("a")
        stats = self.bus.get_stats()
        self.assertEqual(stats["events_published"], 2)
        self.assertEqual(stats["handlers_registered"], 2)
        self.assertIn("avg_dispatch_time_ms", stats)

    def test_event_history(self):
        self.bus.publish("price_update", {"p": 1})
        self.bus.publish("order_filled", {"id": 2})
        hist = self.bus.get_history()
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[0]["event_type"], "price_update")

    def test_event_history_filtered(self):
        self.bus.publish("a", {})
        self.bus.publish("b", {})
        self.bus.publish("a", {})
        hist = self.bus.get_history(event_type="a")
        self.assertEqual(len(hist), 2)

    def test_publish_async(self):
        received = []
        async def async_handler(et, d):
            received.append(d)

        self.bus.subscribe("test", async_handler)

        async def run():
            count = await self.bus.publish_async("test", {"val": 42})
            return count

        count = asyncio.run(run())
        self.assertEqual(count, 1)
        self.assertEqual(received[0]["val"], 42)

    def test_mixed_sync_async(self):
        order = []
        def sync_h(et, d):
            order.append("sync")
        async def async_h(et, d):
            order.append("async")

        self.bus.subscribe("test", sync_h, priority=1)
        self.bus.subscribe("test", async_h, priority=10)

        async def run():
            return await self.bus.publish_async("test")

        asyncio.run(run())
        self.assertEqual(order, ["async", "sync"])

    def test_history_ring_buffer_limit(self):
        from core.event_bus import EventBus
        bus = EventBus(history_size=5)
        for i in range(10):
            bus.publish("test", {"i": i})
        hist = bus.get_history()
        self.assertEqual(len(hist), 5)
        self.assertEqual(hist[0]["data"]["i"], 5)


# ===================================================================
# ONNXModelServer tests
# ===================================================================

class TestONNXModelServer(unittest.TestCase):
    """Tests for ml.onnx_model_server.ONNXModelServer."""

    def setUp(self):
        from ml.onnx_model_server import ONNXModelServer
        self.server = ONNXModelServer(models_dir=tempfile.mkdtemp())

    def test_init(self):
        self.assertIsInstance(self.server._models, dict)

    def test_list_models_empty(self):
        self.assertEqual(self.server.list_models(), [])

    def test_load_model_missing_file(self):
        result = self.server.load_model("missing", "/nonexistent/model.onnx")
        self.assertFalse(result)

    def test_predict_no_model(self):
        result = self.server.predict("nope", [[1.0, 2.0]])
        self.assertIsNone(result)

    def test_latency_stats_no_model(self):
        stats = self.server.get_latency_stats("nope")
        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["avg_ms"], 0.0)

    def test_unload_model_not_loaded(self):
        self.assertFalse(self.server.unload_model("nope"))

    def test_export_sklearn_no_skl2onnx(self):
        from ml.onnx_model_server import _HAS_SKL2ONNX
        if _HAS_SKL2ONNX:
            self.skipTest("skl2onnx is installed; cannot test fallback")
        result = self.server.export_sklearn_to_onnx(None, "test", (1, 10))
        self.assertIsNone(result)

    def test_load_and_predict_if_ort_available(self):
        """Integration test — only runs if onnxruntime + sklearn are installed."""
        from ml.onnx_model_server import _HAS_ORT, _HAS_SKL2ONNX
        if not _HAS_ORT or not _HAS_SKL2ONNX:
            self.skipTest("onnxruntime or skl2onnx not installed")

        try:
            from sklearn.linear_model import LinearRegression
            import numpy as np
        except ImportError:
            self.skipTest("sklearn or numpy not installed")

        # Train a trivial model
        X = np.random.randn(20, 3).astype(np.float32)
        y = X @ np.array([1.0, 2.0, 3.0]) + 0.5
        model = LinearRegression().fit(X, y)

        # Export
        path = self.server.export_sklearn_to_onnx(model, "lr_test", (1, 3))
        self.assertIsNotNone(path)

        # Load
        self.assertTrue(self.server.load_model("lr_test", path))
        models = self.server.list_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["name"], "lr_test")

        # Predict
        preds = self.server.predict("lr_test", [[1.0, 2.0, 3.0]])
        self.assertIsNotNone(preds)
        self.assertIsInstance(preds, list)

        # Latency
        stats = self.server.get_latency_stats("lr_test")
        self.assertEqual(stats["count"], 1)
        self.assertGreater(stats["avg_ms"], 0.0)

        # Unload
        self.assertTrue(self.server.unload_model("lr_test"))
        self.assertEqual(self.server.list_models(), [])


# ===================================================================
# Cross-module integration tests
# ===================================================================

class TestCrossModuleIntegration(unittest.TestCase):
    """Integration tests combining multiple modules."""

    def test_risk_parity_vs_hrp_weights(self):
        """Both allocators should produce valid weights for the same universe."""
        from risk.risk_parity_allocator import RiskParityAllocator
        from risk.hierarchical_risk_parity import HierarchicalRiskParity

        symbols = ["BTC/AUD", "ETH/AUD", "SOL/AUD"]
        returns = _make_multi_returns(symbols, n=60)

        rp = RiskParityAllocator(min_history=10)
        hrp = HierarchicalRiskParity(min_history=10)

        w_rp = rp.compute_weights(returns)
        w_hrp = hrp.compute_weights(returns)

        self.assertAlmostEqual(sum(w_rp.values()), 1.0, places=5)
        self.assertAlmostEqual(sum(w_hrp.values()), 1.0, places=5)
        self.assertEqual(set(w_rp.keys()), set(symbols))
        self.assertEqual(set(w_hrp.keys()), set(symbols))

    def test_feature_store_with_event_bus(self):
        """EventBus can trigger feature store updates."""
        from core.feature_store import FeatureStore
        from core.event_bus import EventBus

        store = FeatureStore(background=False)
        bus = EventBus()

        def on_price(event_type, data):
            store.set(data["symbol"], "last_price", data["price"])

        bus.subscribe("price_update", on_price)
        bus.publish("price_update", {"symbol": "BTC/AUD", "price": 95000})

        self.assertEqual(store.get("BTC/AUD", "last_price"), 95000)

    def test_factor_model_feeds_risk_parity(self):
        """Factor model decompositions can inform risk parity inputs."""
        from risk.factor_model import FactorModel
        from risk.risk_parity_allocator import RiskParityAllocator

        tmpdir = tempfile.mkdtemp()
        fm = FactorModel(db_path=os.path.join(tmpdir, "fm.db"), momentum_window=3, vol_window=3)
        rp = RiskParityAllocator(min_history=10)

        symbols = ["BTC/AUD", "ETH/AUD"]
        rng = random.Random(123)
        market = [rng.gauss(0.001, 0.015) for _ in range(50)]
        returns = {}
        for sym in symbols:
            sym_rng = random.Random(hash(sym))
            beta = 0.8 + sym_rng.random() * 0.4
            rets = [market[t] * beta + sym_rng.gauss(0, 0.005) for t in range(50)]
            returns[sym] = rets
            for i, r in enumerate(rets):
                fm.update_returns(sym, r, market[i])

        # Both should work independently
        weights = rp.compute_weights(returns)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)

        for sym in symbols:
            decomp = fm.decompose(sym, lookback_days=40)
            self.assertIsNotNone(decomp)


if __name__ == "__main__":
    unittest.main()
