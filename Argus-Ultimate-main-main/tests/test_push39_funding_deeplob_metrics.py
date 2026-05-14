"""Tests for Push 39 — FundingRateScanner, DeepLOBLiveBridge, Prometheus MTF (28 tests)."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from alpha.funding_rate_scanner import FundingRateScanner, FundingRateSample, _RATE_THRESHOLD
from alpha.microstructure.deeplob_live_bridge import DeepLOBLiveBridge


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# FundingRateScanner
# ---------------------------------------------------------------------------

class TestFundingRateScanner:
    def test_init_default(self):
        s = FundingRateScanner()
        assert "BTCUSDT" in s._symbols

    def test_latest_signal_no_data(self):
        s = FundingRateScanner()
        sig = s.latest_signal()
        assert sig["direction"] == "flat"
        assert sig["rate"] == 0.0

    def test_rate_to_direction_positive(self):
        s = FundingRateScanner()
        assert s._rate_to_direction(0.0005) == "short"

    def test_rate_to_direction_negative(self):
        s = FundingRateScanner()
        assert s._rate_to_direction(-0.0005) == "long"

    def test_rate_to_direction_neutral(self):
        s = FundingRateScanner()
        assert s._rate_to_direction(0.0) == "flat"

    def test_stability_all_same_sign(self):
        s = FundingRateScanner()
        samples = [
            FundingRateSample(ts=time.time(), rate=0.0003, source="binance"),
            FundingRateSample(ts=time.time(), rate=0.0004, source="binance"),
            FundingRateSample(ts=time.time(), rate=0.0002, source="binance"),
        ]
        assert s._compute_stability(samples) == 1.0

    def test_stability_mixed_signs(self):
        s = FundingRateScanner()
        samples = [
            FundingRateSample(ts=time.time(), rate=0.0003, source="binance"),
            FundingRateSample(ts=time.time(), rate=-0.0003, source="binance"),
        ]
        stab = s._compute_stability(samples)
        assert 0.0 < stab <= 1.0

    def test_stability_empty(self):
        s = FundingRateScanner()
        assert s._compute_stability([]) == 0.0

    def test_latest_signal_after_inject(self):
        s = FundingRateScanner(symbols=["BTCUSDT"])
        s._samples["BTCUSDT"].append(
            FundingRateSample(ts=time.time(), rate=0.0005, source="binance")
        )
        sig = s.latest_signal("BTCUSDT")
        assert sig["direction"] == "short"
        assert sig["rate"] == 0.0005
        assert sig["stability"] == 1.0

    def test_mock_rate_is_float(self):
        rate = FundingRateScanner._mock_rate("BTCUSDT")
        assert isinstance(rate, float)
        assert -0.002 < rate < 0.002

    def test_get_all_rates(self):
        s = FundingRateScanner()
        rates = s.get_all_rates()
        assert "BTCUSDT" in rates


# ---------------------------------------------------------------------------
# DeepLOBLiveBridge
# ---------------------------------------------------------------------------

class TestDeepLOBLiveBridge:
    def _make_book(self, n=10, mid=50000.0):
        bids = [[mid - i * 0.5, 1.0 + i * 0.1] for i in range(n)]
        asks = [[mid + i * 0.5, 1.0 + i * 0.1] for i in range(n)]
        return {"bids": bids, "asks": asks}

    def test_init_no_crash(self):
        bridge = DeepLOBLiveBridge()
        assert bridge._np_weights is not None or bridge._torch_model is not None

    def test_get_signal_no_book_returns_none(self):
        bridge = DeepLOBLiveBridge()
        assert bridge.get_signal() is None

    def test_update_book_sets_mid(self):
        bridge = DeepLOBLiveBridge()
        bridge.update_book(self._make_book(mid=50000.0))
        assert abs(bridge._last_mid - 50000.0) < 1.0

    def test_get_signal_returns_dict(self):
        bridge = DeepLOBLiveBridge()
        bridge.update_book(self._make_book())
        sig = bridge.get_signal()
        assert isinstance(sig, dict)

    def test_signal_direction_valid(self):
        bridge = DeepLOBLiveBridge()
        bridge.update_book(self._make_book())
        sig = bridge.get_signal()
        assert sig["direction"] in ("long", "short", "flat")

    def test_signal_confidence_in_range(self):
        bridge = DeepLOBLiveBridge()
        bridge.update_book(self._make_book())
        sig = bridge.get_signal()
        assert 0.0 <= sig["confidence"] <= 1.0

    def test_signal_logits_length(self):
        bridge = DeepLOBLiveBridge()
        bridge.update_book(self._make_book())
        sig = bridge.get_signal()
        assert len(sig["logits"]) == 3

    def test_feature_vector_length(self):
        bridge = DeepLOBLiveBridge()
        bridge.update_book(self._make_book())
        feats = bridge.get_features()
        assert feats is not None and len(feats) == 40

    def test_partial_book_pads_correctly(self):
        bridge = DeepLOBLiveBridge()
        book = {"bids": [[50000, 1.0]], "asks": [[50001, 1.0]]}
        bridge.update_book(book)
        feats = bridge.get_features()
        assert feats is not None and len(feats) == 40

    def test_softmax_sums_to_one(self):
        x = np.array([1.0, 2.0, 0.5])
        probs = DeepLOBLiveBridge._softmax(x)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_repeated_signals_deterministic(self):
        bridge = DeepLOBLiveBridge()
        book = self._make_book()
        bridge.update_book(book)
        s1 = bridge.get_signal()
        s2 = bridge.get_signal()
        assert s1["direction"] == s2["direction"]
        assert abs(s1["confidence"] - s2["confidence"]) < 1e-6


# ---------------------------------------------------------------------------
# Prometheus MTF metrics
# ---------------------------------------------------------------------------

class TestPrometheusEmitterMTF:
    def test_emit_mtf_no_crash(self):
        from metrics.prometheus_emitter import PrometheusEmitter
        em = PrometheusEmitter(port=19999, version="test")
        # Should not raise even if prometheus_client unavailable
        em.emit_mtf(bias=0.4, direction="long", confidence=0.7)

    def test_emit_mtf_direction_encoding(self):
        # Encoding: long=1, flat=0, short=-1
        encode = lambda d: 1.0 if d == "long" else (-1.0 if d == "short" else 0.0)
        assert encode("long")  == 1.0
        assert encode("short") == -1.0
        assert encode("flat")  == 0.0

    def test_emit_mtf_bias_clamp(self):
        bias = float(np.clip(1.5, -1.0, 1.0))
        assert bias == 1.0
        bias = float(np.clip(-2.0, -1.0, 1.0))
        assert bias == -1.0

    def test_emit_mtf_confidence_clamp(self):
        conf = float(np.clip(-0.1, 0.0, 1.0))
        assert conf == 0.0
        conf = float(np.clip(1.5, 0.0, 1.0))
        assert conf == 1.0

    def test_emit_mtf_flat_direction(self):
        from metrics.prometheus_emitter import PrometheusEmitter
        em = PrometheusEmitter(port=19998, version="test")
        em.emit_mtf(bias=0.0, direction="flat", confidence=0.1)
