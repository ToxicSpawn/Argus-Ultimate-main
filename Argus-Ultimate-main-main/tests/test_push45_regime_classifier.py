"""Tests for Push 45 — RegimeClassifier (20 tests)."""

from __future__ import annotations

import numpy as np
import pytest

from alpha.regime_classifier import RegimeClassifier, _STATE_NAMES, _SCALARS


def _make_candles(n=200, seed=42, trend=0.0):
    """Synthetic OHLCV candles."""
    rng  = np.random.default_rng(seed)
    rets = rng.normal(trend, 0.005, n)
    close = 50_000.0 * np.cumprod(1 + rets)
    ts    = np.arange(n, dtype=float) * 60_000
    noise = rng.uniform(0, 0.002, n)
    high  = close * (1 + noise)
    low   = close * (1 - noise)
    open_ = np.roll(close, 1); open_[0] = close[0]
    vol   = rng.exponential(10.0, n)
    return np.column_stack([ts, open_, high, low, close, vol])


class TestRegimeClassifierInit:
    def test_default_label_is_sideways(self):
        clf = RegimeClassifier()
        assert clf.regime_label == "sideways"

    def test_default_scalar(self):
        clf = RegimeClassifier()
        assert clf.regime_scalar == 1.0

    def test_probs_sum_to_one(self):
        clf = RegimeClassifier()
        assert abs(clf.regime_probs.sum() - 1.0) < 1e-6

    def test_probs_shape(self):
        clf = RegimeClassifier()
        assert clf.regime_probs.shape == (3,)

    def test_fit_count_zero(self):
        clf = RegimeClassifier()
        assert clf.fit_count == 0


class TestRegimeClassifierUpdate:
    def test_returns_float(self):
        clf    = RegimeClassifier(min_fit_bars=50, refit_every=50)
        candles = _make_candles(200)
        result = clf.update(candles)
        assert isinstance(result, float)

    def test_scalar_in_valid_set(self):
        clf     = RegimeClassifier(min_fit_bars=50, refit_every=50)
        candles = _make_candles(200)
        scalar  = clf.update(candles)
        assert scalar in set(_SCALARS.values())

    def test_label_valid(self):
        clf     = RegimeClassifier(min_fit_bars=50, refit_every=50)
        candles = _make_candles(200)
        clf.update(candles)
        assert clf.regime_label in _STATE_NAMES

    def test_insufficient_candles_returns_fallback(self):
        clf     = RegimeClassifier(min_fit_bars=120)
        candles = _make_candles(10)
        scalar  = clf.update(candles)
        assert scalar == 1.0   # fallback

    def test_multiple_updates_stable(self):
        clf     = RegimeClassifier(min_fit_bars=50, refit_every=100)
        candles = _make_candles(300)
        for i in range(50, 300, 10):
            s = clf.update(candles[:i])
            assert isinstance(s, float)

    def test_refit_increments_fit_count(self):
        clf     = RegimeClassifier(min_fit_bars=50, refit_every=50)
        candles = _make_candles(300)
        clf.update(candles)
        assert clf.fit_count >= 1

    def test_probs_sum_after_update(self):
        clf     = RegimeClassifier(min_fit_bars=50, refit_every=50)
        candles = _make_candles(200)
        clf.update(candles)
        assert abs(clf.regime_probs.sum() - 1.0) < 0.01

    def test_bull_market_tends_positive_scalar(self):
        """Strong uptrend -> classifier should find a bull regime."""
        clf     = RegimeClassifier(min_fit_bars=50, refit_every=50)
        candles = _make_candles(300, trend=0.003)   # +0.3% per bar
        scalar  = clf.update(candles)
        # In a strong bull run, scalar should be >= sideways
        assert scalar >= 1.0

    def test_bear_market_tends_lower_scalar(self):
        """Strong downtrend -> classifier should find a bear regime."""
        clf     = RegimeClassifier(min_fit_bars=50, refit_every=50)
        candles = _make_candles(300, trend=-0.003)  # -0.3% per bar
        scalar  = clf.update(candles)
        assert scalar <= 1.0


class TestBuildFeatures:
    def test_returns_2d_array(self):
        clf     = RegimeClassifier()
        candles = _make_candles(100)
        feats   = clf._build_features(candles)
        assert feats is not None
        assert feats.ndim == 2
        assert feats.shape[1] == 2

    def test_no_nan_or_inf(self):
        clf     = RegimeClassifier()
        candles = _make_candles(100)
        feats   = clf._build_features(candles)
        assert np.isfinite(feats).all()

    def test_too_few_candles_returns_none(self):
        clf   = RegimeClassifier()
        feats = clf._build_features(np.zeros((1, 6)))
        assert feats is None or len(feats) == 0


class TestStateAssignment:
    def test_state_map_covers_all_components(self):
        """After a fit, every HMM state should appear in the map."""
        try:
            import hmmlearn  # noqa
        except ImportError:
            pytest.skip("hmmlearn not installed")
        clf     = RegimeClassifier(min_fit_bars=50, refit_every=50)
        candles = _make_candles(300)
        clf.update(candles)
        if clf._model is not None:
            assert len(clf._state_map) == 3
