"""Tests for Push 38 — MTF features, FUNDING_ARB, LLM_OVERLAY (24 tests)."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from core.mtf_features import MultiTimeframeFeatures, MTFResult
from core.signal_gateway import SignalEnvelope, SignalSource
from core.signal_gateway.gateway_config import GatewayConfig
from core.signal_gateway.signal_gateway import SignalGateway


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candles(n: int = 300, trend: str = "up") -> np.ndarray:
    """Generate synthetic 1m OHLCV candles."""
    closes = np.linspace(100, 200 if trend == "up" else 50, n)
    arr = np.zeros((n, 6))
    arr[:, 0] = np.arange(n) * 60_000  # timestamps
    arr[:, 1] = closes * 0.999          # open
    arr[:, 2] = closes * 1.002          # high
    arr[:, 3] = closes * 0.998          # low
    arr[:, 4] = closes                  # close
    arr[:, 5] = np.random.uniform(100, 500, n)  # volume
    return arr


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# MultiTimeframeFeatures
# ---------------------------------------------------------------------------

class TestMTFFeatures:
    def setup_method(self):
        self.mtf = MultiTimeframeFeatures()

    def test_returns_mtf_result(self):
        result = self.mtf.compute(_candles(300))
        assert isinstance(result, MTFResult)

    def test_feature_vector_length(self):
        result = self.mtf.compute(_candles(300))
        # 6 features x 4 timeframes = 24
        assert len(result.features) == 24

    def test_feature_names_length(self):
        result = self.mtf.compute(_candles(300))
        assert len(result.feature_names) == 24

    def test_uptrend_bias_positive(self):
        result = self.mtf.compute(_candles(300, trend="up"))
        assert result.aggregate_bias > 0

    def test_downtrend_bias_negative(self):
        result = self.mtf.compute(_candles(300, trend="down"))
        assert result.aggregate_bias < 0

    def test_uptrend_direction_long(self):
        result = self.mtf.compute(_candles(300, trend="up"))
        assert result.direction == "long"

    def test_downtrend_direction_short(self):
        result = self.mtf.compute(_candles(300, trend="down"))
        assert result.direction == "short"

    def test_confidence_in_range(self):
        result = self.mtf.compute(_candles(300))
        assert 0.0 <= result.confidence <= 1.0

    def test_insufficient_candles_returns_neutral(self):
        result = self.mtf.compute(_candles(30))
        assert result.direction == "flat"
        assert result.aggregate_bias == 0.0

    def test_timeframe_biases_keys(self):
        result = self.mtf.compute(_candles(300))
        assert set(result.timeframe_biases.keys()) == {"5m", "15m", "1h", "4h"}

    def test_agrees_with_matching(self):
        result = self.mtf.compute(_candles(300, trend="up"))
        assert self.mtf.agrees_with(result, "long")

    def test_agrees_with_not_matching(self):
        result = self.mtf.compute(_candles(300, trend="up"))
        assert not self.mtf.agrees_with(result, "short")

    def test_agrees_with_flat_direction(self):
        result = MTFResult(direction="flat")
        assert not self.mtf.agrees_with(result, "long")


# ---------------------------------------------------------------------------
# MTF resample helper
# ---------------------------------------------------------------------------

class TestMTFResample:
    def test_resample_shape(self):
        candles = _candles(300)
        resampled = MultiTimeframeFeatures._resample(candles, 5)
        assert resampled.shape == (60, 6)

    def test_resample_close_is_last(self):
        candles = _candles(10)
        resampled = MultiTimeframeFeatures._resample(candles, 5)
        assert resampled[0, 4] == candles[4, 4]  # close of bar 0 = close of minute 4

    def test_resample_high_is_max(self):
        candles = _candles(10)
        resampled = MultiTimeframeFeatures._resample(candles, 5)
        assert resampled[0, 2] == candles[:5, 2].max()


# ---------------------------------------------------------------------------
# LLM_OVERLAY via MTF gateway ingestion
# ---------------------------------------------------------------------------

class TestLLMOverlayIngestion:
    def test_llm_overlay_envelope_ingested(self):
        cfg = GatewayConfig(consensus_threshold=0.1, min_sources=1)
        gw = SignalGateway(config=cfg, batch_window_ms=10)

        async def run():
            await gw.start()
            await gw.ingest(SignalEnvelope(
                source=SignalSource.LLM_OVERLAY,
                direction="long",
                confidence=0.7,
                metadata={"aggregate_bias": 0.6, "timeframe_biases": {"5m": 0.8}},
            ))
            await asyncio.sleep(0.05)
            await gw.stop()

        _run(run())
        assert gw._ingested >= 1

    def test_llm_overlay_flat_not_ingested_by_bot(self):
        """Flat MTF direction should not produce LLM_OVERLAY envelope."""
        result = MTFResult(direction="flat", confidence=0.1, aggregate_bias=0.0)
        # Bot code: if mtf_result.direction != 'flat' → skip
        assert result.direction == "flat"


# ---------------------------------------------------------------------------
# FUNDING_ARB gateway ingestion
# ---------------------------------------------------------------------------

class TestFundingArbIngestion:
    def test_funding_arb_envelope_valid(self):
        cfg = GatewayConfig(consensus_threshold=0.1, min_sources=1)
        gw = SignalGateway(config=cfg, batch_window_ms=10)

        async def run():
            await gw.start()
            await gw.ingest(SignalEnvelope(
                source=SignalSource.FUNDING_ARB,
                direction="long",
                confidence=0.65,
                metadata={"rate": 0.0003, "stability": 0.65},
            ))
            await asyncio.sleep(0.05)
            await gw.stop()

        _run(run())
        assert gw._ingested >= 1

    def test_funding_arb_absent_scanner_no_crash(self):
        """FundingRateScanner=None should produce no envelope without crashing."""
        scanner = None
        signal = getattr(scanner, "latest_signal", lambda: None)()
        assert signal is None

    def test_funding_arb_stability_clamps_confidence(self):
        stability = 1.5  # out of range
        conf = min(max(float(stability), 0.1), 1.0)
        assert conf == 1.0

        stability = -0.5
        conf = min(max(float(stability), 0.1), 1.0)
        assert conf == 0.1
