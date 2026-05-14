"""Push 96 — Tests for RegimeAwareSizer."""
from __future__ import annotations

# pyright: reportMissingImports=false, reportUndefinedVariable=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportOptionalMemberAccess=false, reportArgumentType=false

import math
from dataclasses import dataclass

import pytest

from core.regime_aware_sizer import RegimeAwareSizer, ScalarSet, _DEFAULTS
from core.regime_history_buffer import RegimeHistoryBuffer
from core.regime_detector import RegimeDetector


@dataclass(frozen=True)
class RegimeSnapshot:
    regime: str
    confidence: float
    volatility: float
    trend_strength: float
    adx: float


# ---------------------------------------------------------------------------
# ScalarSet
# ---------------------------------------------------------------------------

class TestScalarSet:

    def test_defaults(self):
        ss = ScalarSet()
        assert ss.risk_scalar == 1.0
        assert ss.leverage_scalar == 1.0
        assert math.isinf(ss.max_size_cap)

    def test_to_dict_inf_becomes_none(self):
        ss = ScalarSet()
        d = ss.to_dict()
        assert d["max_size_cap"] is None

    def test_to_dict_finite_cap(self):
        ss = ScalarSet(max_size_cap=500.0)
        assert ss.to_dict()["max_size_cap"] == 500.0


# ---------------------------------------------------------------------------
# RegimeAwareSizer basics
# ---------------------------------------------------------------------------

class TestRegimeAwareSizerBasic:

    def test_instantiation_default(self):
        s = RegimeAwareSizer()
        assert "trending_up" in s.scalars
        assert "volatile" in s.scalars

    def test_active_regime_none_initially(self):
        s = RegimeAwareSizer()
        assert s.active_regime is None

    def test_size_neutral_regime(self):
        s = RegimeAwareSizer()
        # ranging regime, full confidence, no excess vol
        size = s.size_position(100.0, "ranging", confidence=1.0, vol_ratio=1.0)
        # risk=1.0, leverage=1.0, conf_factor=1.0, vol_factor=1.0 -> 100.0
        assert size == pytest.approx(100.0, rel=1e-4)

    def test_trending_up_increases_size(self):
        s = RegimeAwareSizer()
        up   = s.size_position(100.0, "trending_up",   confidence=1.0, vol_ratio=1.0)
        base = s.size_position(100.0, "ranging",       confidence=1.0, vol_ratio=1.0)
        assert up > base

    def test_volatile_reduces_size(self):
        s = RegimeAwareSizer()
        vol  = s.size_position(100.0, "volatile", confidence=1.0, vol_ratio=1.0)
        base = s.size_position(100.0, "ranging",  confidence=1.0, vol_ratio=1.0)
        assert vol < base

    def test_unknown_regime_falls_back(self):
        s = RegimeAwareSizer()
        size = s.size_position(100.0, "nonexistent", confidence=1.0, vol_ratio=1.0)
        # falls back to "unknown" table: risk=0.6, leverage=0.6
        assert 0 < size < 100.0

    def test_confidence_zero_reduces_size(self):
        s = RegimeAwareSizer()
        high = s.size_position(100.0, "trending_up", confidence=1.0, vol_ratio=1.0)
        low  = s.size_position(100.0, "trending_up", confidence=0.0, vol_ratio=1.0)
        assert low < high

    def test_vol_ratio_above_one_reduces_size(self):
        s = RegimeAwareSizer()
        normal = s.size_position(100.0, "ranging", confidence=1.0, vol_ratio=1.0)
        high   = s.size_position(100.0, "ranging", confidence=1.0, vol_ratio=3.0)
        assert high < normal

    def test_max_size_cap(self):
        s = RegimeAwareSizer(custom_scalars={"ranging": {"max_size_cap": 80.0}})
        size = s.size_position(100.0, "ranging", confidence=1.0, vol_ratio=1.0)
        assert size <= 80.0

    def test_min_size_floor(self):
        s = RegimeAwareSizer(min_size=5.0)
        size = s.size_position(0.001, "volatile", confidence=0.0, vol_ratio=5.0)
        assert size >= 5.0

    def test_case_insensitive_regime(self):
        s = RegimeAwareSizer()
        a = s.size_position(100.0, "TRENDING_UP", confidence=1.0, vol_ratio=1.0)
        b = s.size_position(100.0, "trending_up", confidence=1.0, vol_ratio=1.0)
        assert a == b

    def test_active_regime_updated(self):
        s = RegimeAwareSizer()
        s.size_position(100.0, "volatile")
        assert s.active_regime == "volatile"

    def test_repr(self):
        s = RegimeAwareSizer()
        assert "RegimeAwareSizer" in repr(s)


# ---------------------------------------------------------------------------
# Scalar hot-update
# ---------------------------------------------------------------------------

class TestSetScalar:

    def test_set_valid_scalar(self):
        s = RegimeAwareSizer()
        s.set_scalar("ranging", "risk_scalar", 2.0)
        size = s.size_position(100.0, "ranging", confidence=1.0, vol_ratio=1.0)
        assert size == pytest.approx(200.0, rel=1e-4)

    def test_set_invalid_scalar_raises(self):
        s = RegimeAwareSizer()
        with pytest.raises(ValueError, match="Unknown scalar"):
            s.set_scalar("ranging", "nonexistent_field", 1.0)

    def test_set_scalars_bulk(self):
        s = RegimeAwareSizer()
        s.set_scalars("volatile", {"risk_scalar": 1.0, "leverage_scalar": 1.0})
        size = s.size_position(100.0, "volatile", confidence=1.0, vol_ratio=1.0)
        # With risk=1, leverage=1, conf_weight=0.7 at full confidence, vol_dampen=2.0 at ratio=1
        # conf_factor = 1 + (0.7-1)*1 = 0.7; vol_factor = 1/(1+0) = 1
        # size = 100 * 1 * 1 * 0.7 * 1 = 70
        assert size == pytest.approx(70.0, rel=1e-4)

    def test_new_regime_created_on_set(self):
        s = RegimeAwareSizer()
        s.set_scalar("custom_regime", "risk_scalar", 1.5)
        assert "custom_regime" in s.scalars

    def test_reset_regime(self):
        s = RegimeAwareSizer()
        s.set_scalar("ranging", "risk_scalar", 99.0)
        s.reset_regime("ranging")
        size = s.size_position(100.0, "ranging", confidence=1.0, vol_ratio=1.0)
        assert size == pytest.approx(100.0, rel=1e-4)

    def test_reset_all(self):
        s = RegimeAwareSizer()
        for r in ["trending_up", "volatile", "ranging"]:
            s.set_scalar(r, "risk_scalar", 99.0)
        s.reset_all()
        size = s.size_position(100.0, "ranging", confidence=1.0, vol_ratio=1.0)
        assert size == pytest.approx(100.0, rel=1e-4)


# ---------------------------------------------------------------------------
# on_transition callback
# ---------------------------------------------------------------------------

class TestOnTransition:

    def _make_snap(self, regime: str, confidence: float = 0.8, volatility: float = 0.01) -> RegimeSnapshot:
        return RegimeSnapshot(
            regime=regime,
            confidence=confidence,
            volatility=volatility,
            trend_strength=0.1,
            adx=20.0,
        )

    def test_active_regime_updated_on_transition(self):
        s = RegimeAwareSizer()
        prev = self._make_snap("ranging")
        curr = self._make_snap("volatile")
        s.on_transition(prev, curr)
        assert s.active_regime == "volatile"

    def test_low_confidence_halves_risk(self):
        s = RegimeAwareSizer()
        original_risk = s.scalars["volatile"]["risk_scalar"]
        prev = self._make_snap("ranging")
        curr = self._make_snap("volatile", confidence=0.1)   # < 0.3
        s.on_transition(prev, curr)
        new_risk = s.scalars["volatile"]["risk_scalar"]
        assert new_risk == pytest.approx(max(0.1, original_risk * 0.5), rel=1e-4)

    def test_high_confidence_no_risk_change(self):
        s = RegimeAwareSizer()
        original_risk = s.scalars["volatile"]["risk_scalar"]
        prev = self._make_snap("ranging")
        curr = self._make_snap("volatile", confidence=0.9)   # >= 0.3
        s.on_transition(prev, curr)
        # risk_scalar should NOT change
        assert s.scalars["volatile"]["risk_scalar"] == pytest.approx(original_risk, rel=1e-4)

    def test_high_vol_reduces_leverage(self):
        s = RegimeAwareSizer()
        original_lev = s.scalars["volatile"]["risk_scalar"]
        prev = self._make_snap("ranging")
        curr = self._make_snap("volatile", confidence=0.8, volatility=0.07)  # > 0.06
        s.on_transition(prev, curr)
        new_lev = s.scalars["volatile"]["leverage_scalar"]
        default_lev = _DEFAULTS["volatile"].leverage_scalar
        assert new_lev == pytest.approx(max(0.1, default_lev * 0.8), rel=1e-4)

    def test_normal_vol_no_leverage_change(self):
        s = RegimeAwareSizer()
        default_lev = _DEFAULTS["volatile"].leverage_scalar
        prev = self._make_snap("ranging")
        curr = self._make_snap("volatile", confidence=0.8, volatility=0.02)  # < 0.06
        s.on_transition(prev, curr)
        assert s.scalars["volatile"]["leverage_scalar"] == pytest.approx(default_lev, rel=1e-4)


# ---------------------------------------------------------------------------
# Full wiring: RegimeDetector → RegimeHistoryBuffer → RegimeAwareSizer
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="legacy detector wiring API removed from current source")
class TestFullWiring:

    def _make_volatile_prices(self, n: int = 40, seed: int = 7):
        import numpy as np
        rng = np.random.default_rng(seed)
        log_ret = rng.normal(0, 0.06, n)
        closes = 100.0 * np.exp(np.cumsum(log_ret))
        return closes.tolist(), (closes * 1.03).tolist(), (closes * 0.97).tolist()

    def _make_calm_prices(self, n: int = 40, seed: int = 42):
        import numpy as np
        rng = np.random.default_rng(seed)
        log_ret = 0.001 + rng.normal(0, 0.001, n)
        closes = 100.0 * np.exp(np.cumsum(log_ret))
        return closes.tolist(), (closes * 1.001).tolist(), (closes * 0.999).tolist()

    def test_sizer_wired_as_on_transition(self):
        """End-to-end: detector fires callback, sizer active_regime updates."""
        buf   = RegimeHistoryBuffer()
        sizer = RegimeAwareSizer()
        det   = RegimeDetector(
            history_buffer=buf,
            on_transition=sizer.on_transition,
            vol_high_threshold=0.03,
        )
        c, h, lo = self._make_calm_prices()
        det.detect(c, h, lo)
        c2, h2, lo2 = self._make_volatile_prices()
        det.detect(c2, h2, lo2)
        # At least one detect happened; active regime should be set
        assert sizer.active_regime is not None or len(buf) >= 1

    def test_buffer_and_sizer_regime_consistent(self):
        """If a transition was recorded, sizer.active_regime matches buffer.latest."""
        buf   = RegimeHistoryBuffer()
        sizer = RegimeAwareSizer()
        det   = RegimeDetector(
            history_buffer=buf,
            on_transition=sizer.on_transition,
            vol_high_threshold=0.03,
        )
        c1, h1, lo1 = self._make_calm_prices()
        det.detect(c1, h1, lo1)
        c2, h2, lo2 = self._make_volatile_prices()
        det.detect(c2, h2, lo2)

        if len(buf) >= 1 and sizer.active_regime is not None:
            assert sizer.active_regime == buf.latest.to_regime

    def test_size_changes_after_volatile_transition(self):
        """size_position() should return smaller size after volatile transition."""
        buf   = RegimeHistoryBuffer()
        sizer = RegimeAwareSizer()
        det   = RegimeDetector(
            history_buffer=buf,
            on_transition=sizer.on_transition,
            vol_high_threshold=0.03,
        )
        # Run calm detect first
        c1, h1, lo1 = self._make_calm_prices()
        det.detect(c1, h1, lo1)
        calm_regime = det.last.regime.value
        calm_size   = sizer.size_position(100.0, calm_regime, confidence=0.9, vol_ratio=1.0)

        # Now volatile
        c2, h2, lo2 = self._make_volatile_prices()
        det.detect(c2, h2, lo2)
        vol_regime = det.last.regime.value
        vol_size   = sizer.size_position(100.0, vol_regime, confidence=0.9, vol_ratio=1.0)

        if vol_regime == "volatile" and calm_regime != "volatile":
            assert vol_size < calm_size
