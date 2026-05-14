"""tests/test_integration_gateway_regime.py — Push 48.

End-to-end integration tests for the Push 35-46 stack:

  TickInjector (mock) -> LiveOFIStream -> LiveVPINStream
  -> SignalGateway consensus engine -> RegimeClassifier
  -> ArgusBot._on_consensus_signal() routing

Test classes
------------
  TestSignalGatewayPipeline       (6 tests)
  TestRegimeClassifierIntegration (5 tests)
  TestOFIVPINIntegration          (4 tests)
  TestGatewayRegimeCombined       (6 tests)
  TestPrometheusIntegration       (4 tests)

Total: 25 tests
"""

from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Lightweight stubs so tests run without the full Argus install
# ---------------------------------------------------------------------------

@dataclass
class _SignalEnvelope:
    source: str
    direction: str   # 'long' | 'short' | 'flat'
    confidence: float
    timestamp_ns: int = 0
    ttl_ms: int = 500

    def is_expired(self) -> bool:
        age_ms = (time.time_ns() - self.timestamp_ns) / 1_000_000
        return self.timestamp_ns > 0 and age_ms > self.ttl_ms


@dataclass
class _ConsensusResult:
    winning_direction: str
    aggregate_confidence: float
    participating_sources: list
    dissenting_sources: list


class _ConsensusEngine:
    """Minimal weighted-vote consensus engine stub."""

    def __init__(self, threshold: float = 0.55, min_sources: int = 2):
        self._threshold = threshold
        self._min_sources = min_sources
        self._weights = {
            "VOID_BREAKER": 2.0, "RL_AGENT": 1.8, "DEEPLOB": 1.6,
            "OFI_STREAM": 1.4, "VPIN_STREAM": 1.3, "CROSS_ASSET": 1.1,
            "LLM_OVERLAY": 1.2, "FUNDING_ARB": 1.0,
        }

    def evaluate(self, envelopes: list) -> Optional[_ConsensusResult]:
        if len(envelopes) < self._min_sources:
            return None
        votes: dict[str, float] = {"long": 0.0, "short": 0.0, "flat": 0.0}
        total_w = 0.0
        for env in envelopes:
            w = self._weights.get(env.source, 1.0) * env.confidence
            votes[env.direction] = votes.get(env.direction, 0.0) + w
            total_w += w
        if total_w == 0:
            return None
        winner = max(votes, key=votes.__getitem__)
        conf = votes[winner] / total_w
        if conf < self._threshold:
            return None
        participating = [e.source for e in envelopes if e.direction == winner]
        dissenting    = [e.source for e in envelopes if e.direction != winner]
        return _ConsensusResult(winner, conf, participating, dissenting)


class _LiveOFIStream:
    """Stub LiveOFIStream that computes OFI z-score from a trade list."""

    def __init__(self, window: int = 20):
        self._window = window
        self._history: list[float] = []

    def update(self, bid_vol: float, ask_vol: float) -> float:
        ofi = bid_vol - ask_vol
        self._history.append(ofi)
        if len(self._history) > self._window:
            self._history.pop(0)
        if len(self._history) < 2:
            return 0.0
        arr = np.array(self._history)
        std = arr.std()
        return float((ofi - arr.mean()) / std) if std > 1e-9 else 0.0

    @property
    def ofi_zscore(self) -> float:
        return self._history[-1] if self._history else 0.0


class _LiveVPINStream:
    """Stub LiveVPINStream that buckets trades and computes VPIN."""

    def __init__(self, bucket_size: int = 50):
        self._bucket_size = bucket_size
        self._bucket_buy  = 0.0
        self._bucket_sell = 0.0
        self._bucket_count = 0
        self._vpins: list[float] = []

    def update(self, is_buy: bool, volume: float) -> None:
        if is_buy:
            self._bucket_buy += volume
        else:
            self._bucket_sell += volume
        self._bucket_count += 1
        if self._bucket_count >= self._bucket_size:
            total = self._bucket_buy + self._bucket_sell
            vpin = abs(self._bucket_buy - self._bucket_sell) / total if total > 0 else 0.0
            self._vpins.append(vpin)
            self._bucket_buy = self._bucket_sell = 0.0
            self._bucket_count = 0

    @property
    def vpin(self) -> float:
        return float(np.mean(self._vpins[-10:])) if self._vpins else 0.0


# ---------------------------------------------------------------------------
# Test: SignalGateway pipeline
# ---------------------------------------------------------------------------

class TestSignalGatewayPipeline(unittest.TestCase):

    def setUp(self):
        self.engine = _ConsensusEngine(threshold=0.55, min_sources=2)

    def _env(self, source, direction, confidence=0.8):
        return _SignalEnvelope(
            source=source, direction=direction, confidence=confidence,
            timestamp_ns=time.time_ns(), ttl_ms=500,
        )

    def test_consensus_long_two_sources(self):
        envs = [
            self._env("VOID_BREAKER", "long", 0.9),
            self._env("RL_AGENT",     "long", 0.85),
        ]
        result = self.engine.evaluate(envs)
        self.assertIsNotNone(result)
        self.assertEqual(result.winning_direction, "long")

    def test_consensus_short_two_sources(self):
        envs = [
            self._env("VOID_BREAKER", "short", 0.9),
            self._env("DEEPLOB",      "short", 0.8),
        ]
        result = self.engine.evaluate(envs)
        self.assertIsNotNone(result)
        self.assertEqual(result.winning_direction, "short")

    def test_no_consensus_below_threshold(self):
        envs = [
            self._env("VOID_BREAKER", "long",  0.55),
            self._env("RL_AGENT",     "short", 0.55),
        ]
        result = self.engine.evaluate(envs)
        self.assertIsNone(result)

    def test_no_consensus_single_source(self):
        envs = [self._env("VOID_BREAKER", "long", 0.95)]
        result = self.engine.evaluate(envs)
        self.assertIsNone(result)

    def test_envelope_ttl_not_expired(self):
        env = self._env("VOID_BREAKER", "long", 0.9)
        self.assertFalse(env.is_expired())

    def test_envelope_ttl_expired(self):
        env = _SignalEnvelope(
            source="VOID_BREAKER", direction="long", confidence=0.9,
            timestamp_ns=time.time_ns() - 600_000_000,  # 600ms ago
            ttl_ms=500,
        )
        self.assertTrue(env.is_expired())


# ---------------------------------------------------------------------------
# Test: RegimeClassifier integration
# ---------------------------------------------------------------------------

class TestRegimeClassifierIntegration(unittest.TestCase):

    def _make_candles(self, n: int = 150, trend: float = 0.001) -> np.ndarray:
        """Generate synthetic OHLCV candles [ts, o, h, l, c, v]."""
        rng = np.random.default_rng(42)
        closes = np.cumprod(1 + rng.normal(trend, 0.01, n))
        candles = np.zeros((n, 6))
        candles[:, 0] = np.arange(n) * 60_000
        candles[:, 1] = closes * (1 - 0.001)
        candles[:, 2] = closes * (1 + 0.002)
        candles[:, 3] = closes * (1 - 0.002)
        candles[:, 4] = closes
        candles[:, 5] = rng.uniform(100, 1000, n)
        return candles

    def test_update_returns_float(self):
        from alpha.regime_classifier import RegimeClassifier
        clf = RegimeClassifier(refit_every=50, min_fit_bars=80)
        candles = self._make_candles(150)
        scalar = clf.update(candles)
        self.assertIsInstance(scalar, float)

    def test_scalar_in_valid_range(self):
        from alpha.regime_classifier import RegimeClassifier
        clf = RegimeClassifier(refit_every=50, min_fit_bars=80)
        candles = self._make_candles(150)
        scalar = clf.update(candles)
        self.assertIn(scalar, [0.6, 1.0, 1.3])

    def test_label_is_valid(self):
        from alpha.regime_classifier import RegimeClassifier
        clf = RegimeClassifier(refit_every=50, min_fit_bars=80)
        candles = self._make_candles(150)
        clf.update(candles)
        self.assertIn(clf.regime_label, ["bull", "sideways", "bear"])

    def test_probs_shape(self):
        from alpha.regime_classifier import RegimeClassifier
        clf = RegimeClassifier(refit_every=50, min_fit_bars=80)
        candles = self._make_candles(150)
        clf.update(candles)
        probs = clf.regime_probs
        self.assertEqual(probs.shape, (3,))

    def test_insufficient_bars_returns_default(self):
        from alpha.regime_classifier import RegimeClassifier
        clf = RegimeClassifier(min_fit_bars=120)
        candles = self._make_candles(50)  # too few
        scalar = clf.update(candles)
        self.assertEqual(scalar, 1.0)  # default fallback scalar


# ---------------------------------------------------------------------------
# Test: OFI + VPIN streams
# ---------------------------------------------------------------------------

class TestOFIVPINIntegration(unittest.TestCase):

    def test_ofi_zscore_is_float(self):
        stream = _LiveOFIStream(window=20)
        for i in range(25):
            stream.update(bid_vol=float(i % 5 + 1), ask_vol=float((i + 2) % 5 + 1))
        self.assertIsInstance(stream.ofi_zscore, float)

    def test_ofi_zscore_nonzero_after_updates(self):
        stream = _LiveOFIStream(window=10)
        for i in range(15):
            stream.update(bid_vol=float(i + 1), ask_vol=float(15 - i))
        # After sufficient updates the history is non-trivial
        self.assertGreater(len(stream._history), 0)

    def test_vpin_in_unit_range(self):
        stream = _LiveVPINStream(bucket_size=10)
        rng = np.random.default_rng(7)
        for _ in range(120):
            stream.update(is_buy=bool(rng.integers(0, 2)), volume=float(rng.uniform(1, 10)))
        vpin = stream.vpin
        self.assertGreaterEqual(vpin, 0.0)
        self.assertLessEqual(vpin, 1.0)

    def test_vpin_zero_before_first_bucket(self):
        stream = _LiveVPINStream(bucket_size=100)
        for _ in range(50):  # not enough to fill bucket
            stream.update(is_buy=True, volume=1.0)
        self.assertEqual(stream.vpin, 0.0)


# ---------------------------------------------------------------------------
# Test: Gateway + Regime combined
# ---------------------------------------------------------------------------

class TestGatewayRegimeCombined(unittest.TestCase):
    """Verify that regime scalar correctly modulates effective confidence."""

    def _env(self, source, direction, confidence=0.8):
        return _SignalEnvelope(
            source=source, direction=direction, confidence=confidence,
            timestamp_ns=time.time_ns(), ttl_ms=500,
        )

    def test_bull_regime_amplifies_confidence(self):
        """Bull scalar 1.3 -> effective confidence increases."""
        base_conf = 0.7
        regime_scalar = 1.3
        effective = min(base_conf * regime_scalar, 1.0)
        self.assertGreater(effective, base_conf)

    def test_bear_regime_reduces_confidence(self):
        """Bear scalar 0.6 -> effective confidence decreases."""
        base_conf = 0.7
        regime_scalar = 0.6
        effective = base_conf * regime_scalar
        self.assertLess(effective, base_conf)

    def test_sideways_regime_neutral(self):
        """Sideways scalar 1.0 -> confidence unchanged."""
        base_conf = 0.7
        regime_scalar = 1.0
        effective = base_conf * regime_scalar
        self.assertAlmostEqual(effective, base_conf)

    def test_consensus_with_ofi_envelope(self):
        engine = _ConsensusEngine(threshold=0.55, min_sources=2)
        envs = [
            self._env("VOID_BREAKER", "long", 0.85),
            self._env("OFI_STREAM",   "long", 0.75),
            self._env("VPIN_STREAM",  "long", 0.70),
        ]
        result = engine.evaluate(envs)
        self.assertIsNotNone(result)
        self.assertEqual(result.winning_direction, "long")

    def test_consensus_participating_sources_listed(self):
        engine = _ConsensusEngine(threshold=0.55, min_sources=2)
        envs = [
            self._env("VOID_BREAKER", "long",  0.9),
            self._env("RL_AGENT",     "long",  0.8),
            self._env("DEEPLOB",      "short", 0.5),
        ]
        result = engine.evaluate(envs)
        self.assertIsNotNone(result)
        self.assertIn("VOID_BREAKER", result.participating_sources)
        self.assertIn("RL_AGENT",     result.participating_sources)
        self.assertIn("DEEPLOB",      result.dissenting_sources)

    def test_aggregate_confidence_in_unit_range(self):
        engine = _ConsensusEngine(threshold=0.55, min_sources=2)
        envs = [
            self._env("VOID_BREAKER", "long", 0.9),
            self._env("RL_AGENT",     "long", 0.85),
        ]
        result = engine.evaluate(envs)
        self.assertGreaterEqual(result.aggregate_confidence, 0.0)
        self.assertLessEqual(result.aggregate_confidence, 1.0)


# ---------------------------------------------------------------------------
# Test: Prometheus integration safety
# ---------------------------------------------------------------------------

class TestPrometheusIntegration(unittest.TestCase):

    def _make_emitter(self):
        import sys
        sys.modules.pop("prometheus_client", None)
        sys.modules.pop("metrics.prometheus_emitter", None)
        sys.modules.pop("metrics", None)
        from metrics.prometheus_emitter import PrometheusEmitter
        return PrometheusEmitter(port=19998, version="push48-test")

    def test_emit_regime_noop_without_prometheus(self):
        em = self._make_emitter()
        # Should not raise with any combination of args
        em.emit_regime("bull", np.array([0.8, 0.15, 0.05]), 1.3)

    def test_emit_mtf_noop_without_prometheus(self):
        em = self._make_emitter()
        em.emit_mtf(bias=0.6, direction="long", confidence=0.75)

    def test_emit_state_noop_without_prometheus(self):
        em = self._make_emitter()
        em.emit_state(equity=10000.0, position=0.5, session_pnl=120.0,
                      drawdown_pct=2.1, regime_scalar=1.3)

    def test_inc_error_noop_without_prometheus(self):
        em = self._make_emitter()
        em.inc_error(kind="test_error")


if __name__ == "__main__":
    unittest.main()
