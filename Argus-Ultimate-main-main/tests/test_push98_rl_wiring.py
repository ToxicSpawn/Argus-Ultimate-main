"""Push 98 — Tests for RLWiring (v8.34.0)."""
from __future__ import annotations

import math
import pytest
from unittest.mock import MagicMock

from core.rl.rl_live_inference import (
    ConvictionCalibrator,
    RegimeRewardShaper,
    RLCheckpointLoader,
    RLInferencePipeline,
    RLLiveConfig,
    RLLiveStrategy,
)


# ---------------------------------------------------------------------------
# RLCheckpointLoader
# ---------------------------------------------------------------------------

class TestRLCheckpointLoader:
    def test_load_missing_file_returns_false(self):
        loader = RLCheckpointLoader("/nonexistent/path.zip", algo="ppo")
        result = loader.load()
        assert not result
        assert not loader.loaded

    def test_stub_predict_returns_array(self):
        loader = RLCheckpointLoader("", algo="ppo")
        action, state = loader.predict([0.0] * 7)
        assert action is not None

    def test_stats_structure(self):
        loader = RLCheckpointLoader("/tmp/fake.zip", algo="td3")
        s = loader.stats
        assert "checkpoint" in s
        assert "loaded" in s
        assert "algo" in s


# ---------------------------------------------------------------------------
# RegimeRewardShaper
# ---------------------------------------------------------------------------

class TestRegimeRewardShaper:
    def test_trending_boosts_momentum(self):
        shaper = RegimeRewardShaper()
        m = shaper.multiplier("TRENDING_BULL", "momentum")
        assert m > 1.0

    def test_ranging_reduces_momentum(self):
        shaper = RegimeRewardShaper()
        m = shaper.multiplier("RANGING", "momentum")
        assert m < 1.0

    def test_shape_applies_multiplier(self):
        shaper = RegimeRewardShaper()
        r = shaper.shape(1.0, "TRENDING_BULL", "momentum")
        assert r == pytest.approx(1.3)

    def test_unknown_regime_returns_1(self):
        shaper = RegimeRewardShaper()
        m = shaper.multiplier("UNKNOWN", "rl")
        assert m == 1.0

    def test_set_custom_multiplier(self):
        shaper = RegimeRewardShaper()
        shaper.set_multiplier("HIGH_VOL", "rl", 0.3)
        assert shaper.multiplier("HIGH_VOL", "rl") == pytest.approx(0.3)

    def test_negative_multiplier_clamped_to_zero(self):
        shaper = RegimeRewardShaper()
        shaper.set_multiplier("RANGING", "rl", -1.0)
        assert shaper.multiplier("RANGING", "rl") == 0.0

    def test_stats_structure(self):
        shaper = RegimeRewardShaper()
        s = shaper.stats
        assert "shape_calls" in s
        assert "regimes" in s


# ---------------------------------------------------------------------------
# ConvictionCalibrator
# ---------------------------------------------------------------------------

class TestConvictionCalibrator:
    def test_calibrate_zero_score_near_half(self):
        cal = ConvictionCalibrator(init_A=1.0, init_B=0.0)
        p = cal.calibrate(0.0)
        assert abs(p - 0.5) < 1e-6

    def test_calibrate_range(self):
        cal = ConvictionCalibrator()
        for score in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            p = cal.calibrate(score)
            assert 0.0 < p < 1.0

    def test_update_reduces_loss(self):
        cal = ConvictionCalibrator(lr=0.1)
        # Positive label, positive score: should converge
        for _ in range(50):
            cal.update(2.0, 1.0)
        assert cal.calibrate(2.0) > 0.7

    def test_stats_structure(self):
        cal = ConvictionCalibrator()
        s = cal.stats
        assert "A" in s and "B" in s and "n_updates" in s


# ---------------------------------------------------------------------------
# RLLiveStrategy
# ---------------------------------------------------------------------------

class TestRLLiveStrategy:
    def _make_strategy(self, gate: float = 0.0) -> RLLiveStrategy:
        cfg = RLLiveConfig(
            checkpoint_path="",
            algo="ppo",
            symbol="BTCUSDT",
            obs_window=10,
            confidence_gate=gate,
        )
        return RLLiveStrategy(cfg)

    def test_no_signal_before_window_filled(self):
        s = self._make_strategy()
        for _ in range(5):
            result = s.tick(50000.0)
        assert result is None

    def test_signal_after_window_filled(self):
        s = self._make_strategy(gate=0.0)
        result = None
        for i in range(20):
            result = s.tick(50000.0 + i)
        assert result is not None

    def test_signal_has_required_fields(self):
        s = self._make_strategy(gate=0.0)
        result = None
        for i in range(20):
            result = s.tick(50000.0 + i)
        if result:
            assert "symbol" in result
            assert "side" in result
            assert "strength" in result

    def test_high_gate_blocks_signal(self):
        s = self._make_strategy(gate=0.9999)
        result = None
        for i in range(30):
            result = s.tick(50000.0 + i)
        assert result is None

    def test_stats_structure(self):
        s = self._make_strategy()
        st = s.stats
        assert "signal_count" in st
        assert "blocked_count" in st


# ---------------------------------------------------------------------------
# RLInferencePipeline
# ---------------------------------------------------------------------------

class TestRLInferencePipeline:
    def test_add_strategy(self):
        ctx = MagicMock()
        ctx.regime_detector = None
        ctx.signal_bus = None
        pipeline = RLInferencePipeline(ctx)
        cfg = RLLiveConfig(checkpoint_path="", symbol="BTCUSDT", obs_window=10)
        pipeline.add_strategy(cfg)
        assert pipeline.stats["strategies"] == 1

    def test_tick_before_window(self):
        ctx = MagicMock()
        ctx.regime_detector = None
        ctx.signal_bus = None
        pipeline = RLInferencePipeline(ctx)
        cfg = RLLiveConfig(checkpoint_path="", symbol="BTCUSDT", obs_window=20)
        pipeline.add_strategy(cfg)
        for _ in range(5):
            sigs = pipeline.tick(50000.0)
        assert sigs == []

    def test_tick_count_increments(self):
        ctx = MagicMock()
        ctx.regime_detector = None
        ctx.signal_bus = None
        pipeline = RLInferencePipeline(ctx)
        pipeline.tick(50000.0)
        pipeline.tick(50001.0)
        assert pipeline.stats["tick_count"] == 2

    def test_disabled_strategy_skipped(self):
        ctx = MagicMock()
        ctx.regime_detector = None
        ctx.signal_bus = None
        pipeline = RLInferencePipeline(ctx)
        cfg = RLLiveConfig(
            checkpoint_path="", symbol="BTCUSDT",
            obs_window=5, confidence_gate=0.0, enabled=False
        )
        pipeline.add_strategy(cfg)
        for i in range(20):
            sigs = pipeline.tick(50000.0 + i)
        assert sigs == []
