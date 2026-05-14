"""
Tests for the ARGUS polyglot multi-language engine.

All tests run in fallback mode (no Rust/Go/Node/C binaries compiled).
They verify that the Python fallback implementations produce correct results
and that the bridge classes handle missing binaries gracefully.

Run:
    py -m pytest tests/test_polyglot.py -v
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# RustEngine tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRustEngine:
    """Tests for multilang.workers.rust_engine.bridge.RustEngine."""

    def _make_engine(self):
        from multilang.workers.rust_engine.bridge import RustEngine
        return RustEngine()

    def test_fallback_mode(self):
        engine = self._make_engine()
        assert engine.backend == "fallback"
        assert not engine.available

    def test_correlation_matrix_identity(self):
        """Perfect correlation with self."""
        engine = self._make_engine()
        series = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        result = engine.compute("correlation_matrix", {"series": series})
        assert isinstance(result, list)
        assert len(result) == 1
        assert abs(result[0][0] - 1.0) < 1e-10

    def test_correlation_matrix_two_series(self):
        """Two identical series should have correlation 1.0."""
        engine = self._make_engine()
        s = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = engine.compute("correlation_matrix", {"series": [s, s]})
        assert len(result) == 2
        assert abs(result[0][1] - 1.0) < 1e-10
        assert abs(result[1][0] - 1.0) < 1e-10

    def test_correlation_matrix_negatively_correlated(self):
        """Perfectly negatively correlated series."""
        engine = self._make_engine()
        s1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        s2 = [5.0, 4.0, 3.0, 2.0, 1.0]
        result = engine.compute("correlation_matrix", {"series": [s1, s2]})
        assert abs(result[0][1] - (-1.0)) < 1e-10

    def test_correlation_matrix_empty(self):
        engine = self._make_engine()
        result = engine.compute("correlation_matrix", {"series": []})
        assert result == []

    def test_portfolio_var_basic(self):
        engine = self._make_engine()
        np.random.seed(42)
        returns = np.random.randn(200).tolist()  # 100 periods, 2 assets
        weights = [0.6, 0.4]
        result = engine.compute("portfolio_var", {
            "returns": returns,
            "weights": weights,
            "confidence": 0.95,
        })
        assert "var" in result
        assert result["var"] > 0  # VaR should be positive (loss)
        assert result["n_periods"] == 100
        assert result["confidence"] == 0.95

    def test_portfolio_var_single_asset(self):
        engine = self._make_engine()
        returns = [-0.05, 0.02, -0.03, 0.01, -0.04, 0.03, -0.02, 0.01, -0.06, 0.02]
        result = engine.compute("portfolio_var", {
            "returns": returns,
            "weights": [1.0],
            "confidence": 0.95,
        })
        assert result["var"] > 0
        assert result["n_periods"] == 10

    def test_kelly_fraction_even_odds(self):
        """50% win rate, 2:1 payoff -> kelly = 0.5 - 0.5/2 = 0.25."""
        engine = self._make_engine()
        result = engine.compute("kelly_fraction", {
            "win_rate": 0.5,
            "avg_win": 2.0,
            "avg_loss": 1.0,
        })
        assert abs(result["kelly"] - 0.25) < 1e-10
        assert abs(result["payoff_ratio"] - 2.0) < 1e-10

    def test_kelly_fraction_negative_edge(self):
        """Bad edge should give kelly = 0 (clamped)."""
        engine = self._make_engine()
        result = engine.compute("kelly_fraction", {
            "win_rate": 0.3,
            "avg_win": 1.0,
            "avg_loss": 1.0,
        })
        assert result["kelly"] == 0.0
        assert result["kelly_raw"] < 0.0

    def test_kelly_fraction_zero_loss(self):
        engine = self._make_engine()
        result = engine.compute("kelly_fraction", {
            "win_rate": 0.5,
            "avg_win": 1.0,
            "avg_loss": 0.0,
        })
        assert result["kelly"] == 0.0

    def test_signal_zscore_basic(self):
        engine = self._make_engine()
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = engine.compute("signal_zscore", {"values": values})
        zscores = result["zscores"]
        assert len(zscores) == 5
        # Mean of z-scores should be ~0
        assert abs(sum(zscores) / len(zscores)) < 1e-10
        # Std of z-scores should be ~1
        z_std = np.std(zscores, ddof=1)
        assert abs(z_std - 1.0) < 1e-10

    def test_signal_zscore_empty(self):
        engine = self._make_engine()
        result = engine.compute("signal_zscore", {"values": []})
        assert result["zscores"] == []

    def test_signal_zscore_constant(self):
        """All same values -> zscores all zero."""
        engine = self._make_engine()
        result = engine.compute("signal_zscore", {"values": [5.0, 5.0, 5.0]})
        assert all(z == 0.0 for z in result["zscores"])

    def test_unknown_command_raises(self):
        engine = self._make_engine()
        with pytest.raises(ValueError, match="Unknown command"):
            engine.compute("nonexistent_cmd", {})

    def test_latency_tracking(self):
        engine = self._make_engine()
        engine.compute("kelly_fraction", {"win_rate": 0.5, "avg_win": 1.0, "avg_loss": 1.0})
        assert engine.avg_latency_ms > 0


# ═══════════════════════════════════════════════════════════════════════════════
# FastMath tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFastMath:
    """Tests for multilang.c_extensions.fast_math.FastMath."""

    def _make(self):
        from multilang.c_extensions.fast_math import FastMath
        return FastMath()

    def test_fallback_mode(self):
        fm = self._make()
        assert fm.backend == "fallback"
        assert not fm.available

    def test_ema_basic(self):
        fm = self._make()
        prices = np.array([10.0, 11.0, 12.0, 11.5, 13.0, 12.5])
        result = fm.ema(prices, period=3)
        assert len(result) == len(prices)
        # First value should equal first price
        assert result[0] == prices[0]
        # EMA should be smoothed
        assert result[-1] != prices[-1]

    def test_ema_period_1(self):
        """Period 1 EMA (alpha=1) should equal raw prices."""
        fm = self._make()
        prices = np.array([10.0, 20.0, 30.0])
        result = fm.ema(prices, period=1)
        np.testing.assert_allclose(result, prices)

    def test_ema_constant_series(self):
        """EMA of constant series should be that constant."""
        fm = self._make()
        prices = np.full(20, 42.0)
        result = fm.ema(prices, period=10)
        np.testing.assert_allclose(result, 42.0)

    def test_ema_accuracy(self):
        """Check EMA against manual calculation."""
        fm = self._make()
        prices = np.array([1.0, 2.0, 3.0])
        alpha = 2.0 / (3.0 + 1.0)  # period=3
        expected = [1.0, alpha * 2.0 + (1 - alpha) * 1.0, 0.0]
        expected[2] = alpha * 3.0 + (1 - alpha) * expected[1]
        result = fm.ema(prices, period=3)
        np.testing.assert_allclose(result, expected)

    def test_rolling_zscore_basic(self):
        fm = self._make()
        values = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
        result = fm.rolling_zscore(values, window=5)
        assert len(result) == len(values)
        # First 4 values should be 0 (insufficient window)
        for i in range(4):
            assert result[i] == 0.0
        # After that, values should be non-zero for a trend
        assert result[-1] != 0.0

    def test_rolling_zscore_constant(self):
        """Constant series -> all z-scores are 0."""
        fm = self._make()
        values = np.full(20, 5.0)
        result = fm.rolling_zscore(values, window=5)
        np.testing.assert_allclose(result, 0.0)

    def test_rolling_zscore_accuracy(self):
        """Check z-score against manual computation at a specific index."""
        fm = self._make()
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = fm.rolling_zscore(values, window=3)
        # At index 2, window is [1,2,3], mean=2, std=1, zscore=(3-2)/1=1
        assert abs(result[2] - 1.0) < 1e-10

    def test_weighted_mid_basic(self):
        fm = self._make()
        bids = np.array([100.0, 200.0])
        asks = np.array([101.0, 201.0])
        bid_sizes = np.array([10.0, 20.0])
        ask_sizes = np.array([10.0, 20.0])
        result = fm.weighted_mid(bids, asks, bid_sizes, ask_sizes)
        # Equal sizes -> simple mid
        np.testing.assert_allclose(result, [100.5, 200.5])

    def test_weighted_mid_skewed(self):
        """More ask size -> mid pulled toward bid."""
        fm = self._make()
        bids = np.array([100.0])
        asks = np.array([102.0])
        bid_sizes = np.array([1.0])
        ask_sizes = np.array([3.0])
        result = fm.weighted_mid(bids, asks, bid_sizes, ask_sizes)
        # wmid = (100*3 + 102*1) / (1+3) = 402/4 = 100.5
        assert abs(result[0] - 100.5) < 1e-10

    def test_weighted_mid_zero_sizes(self):
        """Zero sizes should fall back to simple mid."""
        fm = self._make()
        bids = np.array([100.0])
        asks = np.array([102.0])
        bid_sizes = np.array([0.0])
        ask_sizes = np.array([0.0])
        result = fm.weighted_mid(bids, asks, bid_sizes, ask_sizes)
        assert abs(result[0] - 101.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# NodeWSMultiplexer tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestNodeWSMultiplexer:
    """Tests for multilang.workers.node_ws.bridge.NodeWSMultiplexer."""

    def _make(self):
        from multilang.workers.node_ws.bridge import NodeWSMultiplexer
        return NodeWSMultiplexer()

    def test_fallback_mode(self):
        mux = self._make()
        assert mux.backend == "fallback"
        # Node.js + node_modules not present -> not available
        assert not mux.available

    def test_parse_valid_message(self):
        mux = self._make()
        raw = json.dumps({
            "exchange": "kraken",
            "channel": "ticker",
            "data": {"price": 50000},
            "timestamp": 1234567890,
        })
        msg = mux.parse_message(raw)
        assert msg is not None
        assert msg["exchange"] == "kraken"
        assert msg["data"]["price"] == 50000

    def test_parse_invalid_json(self):
        mux = self._make()
        assert mux.parse_message("not json!!!") is None

    def test_parse_non_dict(self):
        mux = self._make()
        assert mux.parse_message("[1,2,3]") is None

    def test_add_feed(self):
        mux = self._make()
        mux.add_feed("kraken", "wss://ws.kraken.com", {"subscribe": "ticker"})
        assert len(mux._feeds) == 1
        assert mux._feeds[0]["exchange"] == "kraken"

    def test_on_message_callback(self):
        mux = self._make()
        received = []
        mux.on_message(lambda msg: received.append(msg))
        assert len(mux._callbacks) == 1

    def test_message_count(self):
        mux = self._make()
        assert mux.message_count == 0
        mux.parse_message(json.dumps({"exchange": "x", "data": {}}))
        assert mux.message_count == 1


# ═══════════════════════════════════════════════════════════════════════════════
# GoOrderRouter tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGoOrderRouter:
    """Tests for multilang.workers.go_router.bridge.GoOrderRouter."""

    def _make(self):
        from multilang.workers.go_router.bridge import GoOrderRouter
        return GoOrderRouter()

    def test_fallback_mode(self):
        router = self._make()
        assert router.backend == "fallback"
        assert not router.available

    def test_route_returns_venue(self):
        router = self._make()
        result = router.route({"symbol": "BTC/AUD", "side": "buy", "quantity": 0.1, "type": "limit"})
        assert "venue" in result
        assert result["venue"] in ("kraken", "coinbase")
        assert "score" in result
        assert "reason" in result

    def test_route_with_preferred_venues(self):
        router = self._make()
        result = router.route(
            {"symbol": "BTC/AUD", "side": "buy", "quantity": 0.1, "type": "limit"},
            venues=["coinbase"],
        )
        assert result["venue"] == "coinbase"

    def test_submit_returns_order_id(self):
        router = self._make()
        result = router.submit(
            {"symbol": "BTC/AUD", "side": "buy", "quantity": 0.1, "type": "limit"},
            venue="kraken",
        )
        assert "order_id" in result
        assert result["status"] == "submitted"
        assert result["venue"] == "kraken"

    def test_submit_auto_routes(self):
        """Submit without venue should auto-route."""
        router = self._make()
        result = router.submit(
            {"symbol": "BTC/AUD", "side": "buy", "quantity": 0.1, "type": "limit"},
        )
        assert result["venue"] in ("kraken", "coinbase")
        assert result["status"] == "submitted"

    def test_status_returns_venues(self):
        router = self._make()
        stats = router.status()
        assert isinstance(stats, list)
        assert len(stats) >= 2
        names = {s["name"] for s in stats}
        assert "kraken" in names

    def test_latency_tracking(self):
        router = self._make()
        router.route({"symbol": "X", "side": "buy", "quantity": 1, "type": "market"})
        assert router.avg_latency_ms >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# PolyglotEngine tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPolyglotEngine:
    """Tests for core.polyglot_engine.PolyglotEngine."""

    def _make(self):
        from core.polyglot_engine import PolyglotEngine
        engine = PolyglotEngine()
        engine.initialize()
        return engine

    def test_all_components_initialize(self):
        engine = self._make()
        assert engine.rust is not None
        assert engine.c_math is not None
        assert engine.node_ws is not None
        assert engine.go_router is not None

    def test_status_has_all_languages(self):
        engine = self._make()
        status = engine.get_status()
        assert "rust" in status
        assert "c" in status
        assert "javascript" in status
        assert "go" in status

    def test_status_reports_fallback(self):
        engine = self._make()
        status = engine.get_status()
        # Without compiled binaries, all should be fallback
        assert status["rust"]["backend"] == "fallback"
        assert status["c"]["backend"] == "fallback"
        assert status["javascript"]["backend"] == "fallback"
        assert status["go"]["backend"] == "fallback"

    def test_status_structure(self):
        engine = self._make()
        status = engine.get_status()
        for lang, info in status.items():
            assert "available" in info
            assert "backend" in info
            assert "latency_ms" in info

    def test_benchmark_runs(self):
        engine = self._make()
        results = engine.benchmark(n=100)
        assert isinstance(results, dict)
        # Should have at least rust_correlation and c_ema
        assert "rust_correlation" in results
        assert "c_ema" in results
        assert "go_route" in results

    def test_benchmark_structure(self):
        engine = self._make()
        results = engine.benchmark(n=50)
        for name, bench in results.items():
            assert "native_ms" in bench
            assert "fallback_ms" in bench
            assert "speedup_factor" in bench
            assert bench["fallback_ms"] >= 0
            assert bench["speedup_factor"] > 0

    def test_initialize_returns_count(self):
        from core.polyglot_engine import PolyglotEngine
        engine = PolyglotEngine()
        count = engine.initialize()
        assert count == 10  # All 10 language bridges load (in fallback mode)

    def test_end_to_end_rust_via_polyglot(self):
        """Use PolyglotEngine to compute Kelly fraction."""
        engine = self._make()
        result = engine.rust.compute("kelly_fraction", {
            "win_rate": 0.6,
            "avg_win": 1.5,
            "avg_loss": 1.0,
        })
        assert result["kelly"] > 0
        assert result["kelly"] <= 1.0

    def test_end_to_end_c_via_polyglot(self):
        """Use PolyglotEngine to compute EMA."""
        engine = self._make()
        prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        result = engine.c_math.ema(prices, period=3)
        assert len(result) == 5
        assert result[0] == 10.0
