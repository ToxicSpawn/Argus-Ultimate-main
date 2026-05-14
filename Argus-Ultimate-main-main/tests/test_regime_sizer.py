"""
Push 92 — unit tests for RegimeAwareSizer
"""
import pytest
from unittest.mock import MagicMock
from core.risk.regime_sizer import (
    RegimeAwareSizer,
    RegimeSizerConfig,
    DEFAULT_REGIME_SCALARS,
)
from core.risk.position_sizer import SizerConfig, SizingMethod


@pytest.fixture
def sizer():
    return RegimeAwareSizer()


class TestDefaultScalars:
    def test_trending_bull_larger_than_ranging(self, sizer):
        base   = sizer.size(10_000, 50_000, 0.5, regime="RANGING")
        bull   = sizer.size(10_000, 50_000, 0.5, regime="TRENDING_BULL")
        assert bull > base

    def test_high_vol_smallest(self, sizer):
        ranging  = sizer.size(10_000, 50_000, 0.5, regime="RANGING")
        high_vol = sizer.size(10_000, 50_000, 0.5, regime="HIGH_VOL")
        assert high_vol < ranging

    def test_trending_bear_smaller_than_ranging(self, sizer):
        ranging = sizer.size(10_000, 50_000, 0.5, regime="RANGING")
        bear    = sizer.size(10_000, 50_000, 0.5, regime="TRENDING_BEAR")
        assert bear < ranging

    def test_unknown_conservative(self, sizer):
        ranging = sizer.size(10_000, 50_000, 0.5, regime="RANGING")
        unknown = sizer.size(10_000, 50_000, 0.5, regime="UNKNOWN")
        assert unknown < ranging


class TestScalarClamping:
    def test_custom_scalar_clamped_to_max(self):
        cfg = RegimeSizerConfig()
        cfg.regime_scalars["RANGING"] = 99.0
        s = RegimeAwareSizer(config=cfg)
        scalar = s.scalar_for_regime("RANGING")
        assert scalar <= cfg.max_scalar

    def test_custom_scalar_clamped_to_min(self):
        cfg = RegimeSizerConfig()
        cfg.regime_scalars["RANGING"] = 0.0
        s = RegimeAwareSizer(config=cfg)
        scalar = s.scalar_for_regime("RANGING")
        assert scalar >= cfg.min_scalar


class TestDetectorIntegration:
    def test_reads_regime_from_detector(self):
        mock_snap = MagicMock()
        mock_snap.regime = MagicMock()
        mock_snap.regime.value = "HIGH_VOL"
        mock_det = MagicMock()
        mock_det.snapshot.return_value = mock_snap

        sizer = RegimeAwareSizer(detector=mock_det)
        qty_auto   = sizer.size(10_000, 50_000, 0.5)
        qty_manual = sizer.size(10_000, 50_000, 0.5, regime="HIGH_VOL")
        assert qty_auto == qty_manual

    def test_explicit_regime_overrides_detector(self):
        mock_snap = MagicMock()
        mock_snap.regime = MagicMock()
        mock_snap.regime.value = "HIGH_VOL"
        mock_det = MagicMock()
        mock_det.snapshot.return_value = mock_snap

        sizer = RegimeAwareSizer(detector=mock_det)
        qty_explicit = sizer.size(10_000, 50_000, 0.5, regime="TRENDING_BULL")
        qty_detector = sizer.size(10_000, 50_000, 0.5)
        assert qty_explicit != qty_detector

    def test_fallback_when_detector_raises(self):
        mock_det = MagicMock()
        mock_det.snapshot.side_effect = RuntimeError("no data")
        sizer = RegimeAwareSizer(detector=mock_det)
        qty = sizer.size(10_000, 50_000, 0.5)
        assert qty >= 0


class TestRuntimeOverride:
    def test_set_regime_scalar(self, sizer):
        sizer.set_regime_scalar("HIGH_VOL", 0.20)
        assert sizer.config.regime_scalars["HIGH_VOL"] == 0.20

    def test_set_regime_scalar_clamped(self, sizer):
        sizer.set_regime_scalar("HIGH_VOL", -5.0)
        assert sizer.config.regime_scalars["HIGH_VOL"] >= sizer.config.min_scalar


class TestSummary:
    def test_summary_keys(self, sizer):
        s = sizer.summary()
        assert "current_regime"  in s
        assert "current_scalar"  in s
        assert "regime_scalars"  in s
        assert "detector_wired"  in s
        assert "base_method"     in s

    def test_summary_no_detector(self, sizer):
        assert sizer.summary()["detector_wired"] is False

    def test_summary_with_detector(self):
        sizer = RegimeAwareSizer(detector=MagicMock())
        assert sizer.summary()["detector_wired"] is True


class TestZeroPrice:
    def test_zero_price_returns_zero(self, sizer):
        assert sizer.size(10_000, 0.0, 0.5, regime="RANGING") == 0.0

    def test_zero_equity_returns_zero(self, sizer):
        assert sizer.size(0.0, 50_000, 0.5, regime="RANGING") == 0.0


class TestMethodPassthrough:
    def test_vol_adjusted_method(self, sizer):
        qty = sizer.size(
            10_000, 50_000, 0.5,
            realised_vol_pct=30.0,
            method=SizingMethod.VOL_ADJUSTED,
            regime="RANGING",
        )
        assert qty > 0

    def test_fixed_frac_method(self, sizer):
        qty = sizer.size(
            10_000, 50_000, 0.5,
            atr=500.0,
            method=SizingMethod.FIXED_FRAC,
            regime="RANGING",
        )
        assert qty > 0
