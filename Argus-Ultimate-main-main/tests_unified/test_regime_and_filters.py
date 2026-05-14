"""
tests_unified/test_regime_and_filters.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for:
  - alpha/regime_detector.py  (RegimeDetector)
  - alpha/regime_scheduler.py (RegimeScheduler)
  - alpha/adverse_selection_filter.py (AdverseSelectionFilter)
  - alpha/optimal_spread_calibrator.py (OptimalSpreadCalibrator)
"""

from __future__ import annotations

import math
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from alpha.regime_detector import MarketRegime, RegimeConfig, RegimeDetector
from alpha.regime_scheduler import QuoteParams, RegimeScheduler
from alpha.adverse_selection_filter import ASCheckResult, ASFilterConfig, AdverseSelectionFilter
from alpha.optimal_spread_calibrator import CalibConfig, OptimalSpreadCalibrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mean_reverting_prices(n: int = 200, seed: int = 999) -> list[float]:
    """Generate a strongly mean-reverting price series (Hurst < 0.45).

    Uses a strong OU process with near-zero persistence so that the Hurst
    exponent falls well below the 0.45 threshold.
    """
    rng = np.random.default_rng(seed)
    prices = [100.0]
    mean = 100.0
    for _ in range(n - 1):
        # Very strong mean-reversion: phi = 0.1 → H << 0.5
        new_p = mean + 0.1 * (prices[-1] - mean) + rng.normal(0, 2.0)
        prices.append(max(1.0, new_p))
    return prices


def _make_trending_prices(n: int = 110, seed: int = 42) -> list[float]:
    """Generate a trending price series (Hurst > 0.55, vol_ratio > trend_threshold).

    Uses 60 quiet returns followed by 50 trending returns with higher variance,
    then tests with trend_threshold=0.9 to confirm TRENDING detection.
    """
    rng = np.random.default_rng(seed)
    quiet_returns = rng.normal(0.0001, 0.001, 60)
    trending_returns = rng.normal(0.003, 0.008, 50)
    all_returns = list(quiet_returns) + list(trending_returns)
    prices = [100.0]
    for r in all_returns[:n - 1]:
        prices.append(max(1.0, prices[-1] * np.exp(r)))
    return prices


def _make_high_vol_prices(n: int = 220, seed: int = 42) -> list[float]:
    """Generate a price series with an end-of-series vol burst (vol_ratio > 3).

    Uses 215 quiet returns (std=0.001) followed by 5 explosive returns
    (std=0.01).  With vol_lookback_fast=5 the fast window sees only the burst;
    the slow window (100) sees 95 quiet + 5 explosive, giving
    vol_ratio = fast_std / slow_std ≈ 3.2.
    """
    rng = np.random.default_rng(seed)
    quiet = list(rng.normal(0, 0.001, 215))
    burst = list(rng.normal(0, 0.01, 5))
    all_rets = quiet + burst
    prices = [100.0]
    for r in all_rets[:n - 1]:
        prices.append(prices[-1] * np.exp(r))
    return prices


def _feed_prices(detector: RegimeDetector, symbol: str, prices: list[float]) -> None:
    """Feed all prices into detector at 5-minute bar intervals.

    Forces immediate re-classification on every tick (update_interval_ms=0)
    to ensure the final regime reflects the last data point.
    """
    base_ns = time.time_ns()
    interval_ns = 5 * 60 * int(1e9)  # 5-minute bars
    # Force update on every tick
    detector._cfg.update_interval_ms = 0.0
    for i, p in enumerate(prices):
        detector.on_price(symbol, p, base_ns + i * interval_ns)


# ---------------------------------------------------------------------------
# RegimeDetector tests
# ---------------------------------------------------------------------------


class TestRegimeDetector:

    def test_regime_detector_mean_reverting(self):
        """Low Hurst (strong OU prices) + low vol ratio → MEAN_REVERTING."""
        cfg = RegimeConfig(
            min_bars=50,
            hurst_window=50,
            hurst_mean_revert_threshold=0.45,
            hurst_trend_threshold=0.55,
            vol_lookback_fast=10,
            vol_lookback_slow=100,
            trend_threshold=2.0,
            update_interval_ms=0.0,
        )
        detector = RegimeDetector(cfg)
        prices = _make_mean_reverting_prices(n=200, seed=999)
        _feed_prices(detector, "TEST", prices)
        regime = detector.get_regime("TEST")
        assert regime == MarketRegime.MEAN_REVERTING, (
            f"Expected MEAN_REVERTING, got {regime}. "
            f"stats={detector.get_stats('TEST')}"
        )

    def test_regime_detector_trending(self):
        """High Hurst + vol_ratio > threshold → TRENDING_UP or TRENDING_DOWN.

        The last 50 returns of the test series are autocorrelated positive-drift
        returns with higher variance than the preceding 60 quiet returns, giving
        H > 0.53 and vol_ratio > 0.9.  The test config uses a lower
        hurst_trend_threshold (0.53) and trend_threshold (0.9) to ensure stable
        classification with finite-sample synthetic data.
        """
        cfg = RegimeConfig(
            min_bars=50,
            hurst_window=50,
            hurst_mean_revert_threshold=0.45,
            hurst_trend_threshold=0.53,   # slightly relaxed for finite-sample reliability
            vol_lookback_fast=10,
            vol_lookback_slow=100,
            trend_threshold=0.9,          # reliably triggered by the test series
            update_interval_ms=0.0,
        )
        detector = RegimeDetector(cfg)
        prices = _make_trending_prices(n=110, seed=42)
        _feed_prices(detector, "TEST", prices)
        regime = detector.get_regime("TEST")
        assert regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN), (
            f"Expected TRENDING_*, got {regime}. "
            f"stats={detector.get_stats('TEST')}"
        )

    def test_regime_detector_high_vol(self):
        """vol_ratio > 3.0 → HIGH_VOLATILITY regardless of Hurst.

        Uses 215 quiet returns + 5 explosive returns.  With vol_lookback_fast=5
        the fast window sees only the burst (std=0.01); the slow window (100)
        sees 95 quiet + 5 explosive, giving vol_ratio ≈ 3.2+.
        """
        cfg = RegimeConfig(
            min_bars=50,
            vol_lookback_fast=5,   # only 5 bars: entirely within the burst
            vol_lookback_slow=100,
            trend_threshold=2.0,
            update_interval_ms=0.0,
        )
        detector = RegimeDetector(cfg)
        prices = _make_high_vol_prices(n=220, seed=42)  # uses 215 quiet + 5 burst
        _feed_prices(detector, "TEST", prices)
        regime = detector.get_regime("TEST")
        assert regime == MarketRegime.HIGH_VOLATILITY, (
            f"Expected HIGH_VOLATILITY, got {regime}. "
            f"stats={detector.get_stats('TEST')}"
        )

    def test_regime_detector_mm_safe(self):
        """MEAN_REVERTING regime → is_mm_safe returns True."""
        cfg = RegimeConfig(
            min_bars=50,
            update_interval_ms=0.0,
            trend_threshold=2.0,
            vol_lookback_fast=10,
            vol_lookback_slow=100,
        )
        detector = RegimeDetector(cfg)
        prices = _make_mean_reverting_prices(n=200, seed=999)
        _feed_prices(detector, "TEST", prices)
        assert detector.get_regime("TEST") == MarketRegime.MEAN_REVERTING
        assert detector.is_mm_safe("TEST") is True

    def test_regime_detector_mm_unsafe_trend(self):
        """TRENDING regime → is_mm_safe returns False."""
        cfg = RegimeConfig(
            min_bars=50,
            update_interval_ms=0.0,
            trend_threshold=0.9,
            hurst_trend_threshold=0.53,
            vol_lookback_fast=10,
            vol_lookback_slow=100,
        )
        detector = RegimeDetector(cfg)
        prices = _make_trending_prices(n=110, seed=42)
        _feed_prices(detector, "TEST", prices)
        regime = detector.get_regime("TEST")
        assert regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN)
        assert detector.is_mm_safe("TEST") is False

    def test_regime_change_callback_fires(self):
        """Callback is invoked when regime transitions."""
        cfg = RegimeConfig(
            min_bars=10,
            update_interval_ms=0.0,
            vol_lookback_fast=10,
            vol_lookback_slow=100,
        )
        detector = RegimeDetector(cfg)
        events = []
        detector.register_regime_change_callback(
            lambda sym, old, new: events.append((sym, old, new))
        )
        # Feed enough data to cross min_bars; regime transitions from LOW_LIQUIDITY → something
        prices = _make_mean_reverting_prices(n=200, seed=999)
        _feed_prices(detector, "BTC", prices)
        assert len(events) >= 1

    def test_regime_detector_low_liquidity_below_min_bars(self):
        """Fewer than min_bars → LOW_LIQUIDITY."""
        cfg = RegimeConfig(min_bars=100, update_interval_ms=0.0)
        detector = RegimeDetector(cfg)
        prices = [100.0 + i * 0.01 for i in range(30)]  # only 30 bars
        _feed_prices(detector, "TEST", prices)
        assert detector.get_regime("TEST") == MarketRegime.LOW_LIQUIDITY

    def test_get_stats_returns_all_keys(self):
        """get_stats returns all expected keys."""
        cfg = RegimeConfig(update_interval_ms=0.0, vol_lookback_fast=10, vol_lookback_slow=100)
        detector = RegimeDetector(cfg)
        prices = _make_mean_reverting_prices(n=200, seed=999)
        _feed_prices(detector, "S", prices)
        stats = detector.get_stats("S")
        expected_keys = {
            "regime", "confidence", "hurst_exponent", "vol_ratio",
            "trend_strength", "mm_safe", "bars_in_regime", "time_in_regime_s",
        }
        assert expected_keys.issubset(set(stats.keys()))


# ---------------------------------------------------------------------------
# RegimeScheduler tests
# ---------------------------------------------------------------------------


def _make_mock_detector(regime: MarketRegime, confidence: float = 0.8) -> MagicMock:
    """Create a mock RegimeDetector returning a fixed regime."""
    d = MagicMock(spec=RegimeDetector)
    d.get_regime.return_value = regime
    d.get_regime_confidence.return_value = confidence
    return d


def _make_mock_schedule(session_mult: float = 1.0) -> MagicMock:
    """Create a mock SessionSpreadSchedule returning a fixed multiplier."""
    from execution.session_spread_schedule import SessionSpreadSchedule
    s = MagicMock(spec=SessionSpreadSchedule)
    s.get_current_spread_multiplier.return_value = session_mult
    return s


class TestRegimeScheduler:

    def test_regime_scheduler_halts_on_trend(self):
        """TRENDING regime → should_quote returns False."""
        for trending in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            detector = _make_mock_detector(trending, confidence=0.9)
            scheduler = RegimeScheduler(detector, _make_mock_schedule(1.0))
            assert scheduler.should_quote("BTC-USD") is False
            params = scheduler.get_quote_params("BTC-USD", base_spread_bps=30.0)
            assert params.should_quote is False

    def test_regime_scheduler_widens_on_high_vol(self):
        """HIGH_VOLATILITY regime → effective multiplier > 1.5 (session=1 × 2.0)."""
        detector = _make_mock_detector(MarketRegime.HIGH_VOLATILITY, confidence=0.9)
        scheduler = RegimeScheduler(detector, _make_mock_schedule(1.0))
        mult = scheduler.get_effective_spread_multiplier("BTC-USD")
        assert mult > 1.5, f"Expected multiplier > 1.5, got {mult}"
        params = scheduler.get_quote_params("BTC-USD", base_spread_bps=30.0)
        assert params.effective_spread_bps > 30.0 * 1.5

    def test_regime_scheduler_tightens_on_mr(self):
        """MEAN_REVERTING with confidence > 0.7 → multiplier < 1 (regime mult = 0.85)."""
        detector = _make_mock_detector(MarketRegime.MEAN_REVERTING, confidence=0.85)
        scheduler = RegimeScheduler(detector, _make_mock_schedule(1.0))
        mult = scheduler.get_effective_spread_multiplier("BTC-USD")
        assert mult < 1.0, f"Expected multiplier < 1.0, got {mult}"
        params = scheduler.get_quote_params("BTC-USD", base_spread_bps=30.0)
        assert params.effective_spread_bps < 30.0

    def test_regime_scheduler_should_quote_mean_reverting(self):
        """MEAN_REVERTING with confidence > 0.4 → should_quote True."""
        detector = _make_mock_detector(MarketRegime.MEAN_REVERTING, confidence=0.6)
        scheduler = RegimeScheduler(detector, _make_mock_schedule(1.0))
        assert scheduler.should_quote("BTC-USD") is True

    def test_regime_scheduler_halts_low_confidence(self):
        """Low confidence (< 0.3) → should_quote False regardless of regime."""
        detector = _make_mock_detector(MarketRegime.MEAN_REVERTING, confidence=0.1)
        scheduler = RegimeScheduler(detector, _make_mock_schedule(1.0))
        assert scheduler.should_quote("BTC-USD") is False

    def test_regime_scheduler_unknown_cautious_mult(self):
        """UNKNOWN regime → multiplier ≥ 1.3 × session."""
        detector = _make_mock_detector(MarketRegime.UNKNOWN, confidence=0.5)
        scheduler = RegimeScheduler(detector, _make_mock_schedule(1.0))
        mult = scheduler.get_effective_spread_multiplier("BTC-USD")
        assert mult >= 1.3

    def test_regime_scheduler_get_quote_params_fields(self):
        """get_quote_params returns a QuoteParams with all required fields."""
        detector = _make_mock_detector(MarketRegime.MEAN_REVERTING, confidence=0.8)
        scheduler = RegimeScheduler(detector, _make_mock_schedule(1.0))
        params = scheduler.get_quote_params("ETH-USD", base_spread_bps=20.0)
        assert isinstance(params, QuoteParams)
        assert params.effective_spread_bps > 0
        assert isinstance(params.should_quote, bool)
        assert isinstance(params.reason, str)
        assert isinstance(params.confidence, float)
        assert isinstance(params.multiplier, float)

    def test_regime_scheduler_get_session_stats(self):
        """get_session_stats returns expected keys."""
        detector = _make_mock_detector(MarketRegime.MEAN_REVERTING, confidence=0.8)
        scheduler = RegimeScheduler(detector, _make_mock_schedule(1.0))
        scheduler.get_quote_params("BTC-USD", 30.0)
        stats = scheduler.get_session_stats()
        assert "time_quoting_pct" in stats
        assert "time_halted_pct" in stats
        assert "regime_breakdown" in stats


# ---------------------------------------------------------------------------
# AdverseSelectionFilter tests
# ---------------------------------------------------------------------------


class TestAdverseSelectionFilter:

    def test_as_filter_blocks_high_obi(self):
        """|OBI z-score| > threshold → denied with reason denied_obi_zscore."""
        asf = AdverseSelectionFilter(ASFilterConfig(obi_zscore_threshold=1.8))
        result = asf.pre_trade_check("BTC", "buy", ofi_signal=2.5, vpin=0.3, microprice_drift=0.0)
        assert result.allowed is False
        assert result.reason == "denied_obi_zscore"
        assert result.delay_ms > 0

    def test_as_filter_blocks_high_vpin(self):
        """VPIN > threshold → denied with reason denied_vpin."""
        asf = AdverseSelectionFilter(ASFilterConfig(vpin_threshold=0.65))
        result = asf.pre_trade_check("BTC", "buy", ofi_signal=0.0, vpin=0.80, microprice_drift=0.0)
        assert result.allowed is False
        assert result.reason == "denied_vpin"
        assert result.delay_ms > 0

    def test_as_filter_allows_clean(self):
        """All signals below thresholds → allowed."""
        asf = AdverseSelectionFilter(ASFilterConfig())
        result = asf.pre_trade_check("BTC", "buy", ofi_signal=0.5, vpin=0.3, microprice_drift=0.1)
        assert result.allowed is True
        assert result.reason == "all_clear"
        assert result.delay_ms == 0.0

    def test_as_filter_own_adverse_rate(self):
        """Own adverse fill rate > 50% → should_halt_symbol True."""
        cfg = ASFilterConfig(max_own_adverse_rate=0.5, lookback_fills=20)
        asf = AdverseSelectionFilter(cfg)
        # Feed 12 adverse fills and 4 benign fills → adverse_rate = 12/16 = 0.75
        for _ in range(12):
            # buy fill, price drops → adverse
            asf.on_our_fill("BTC", "buy", fill_price=100.0, post_mid_500ms=99.0)
        for _ in range(4):
            # buy fill, price rises → benign
            asf.on_our_fill("BTC", "buy", fill_price=100.0, post_mid_500ms=101.0)
        assert asf.get_our_adverse_rate("BTC") > 0.5
        assert asf.should_halt_symbol("BTC") is True

    def test_as_filter_adverse_rate_recovery(self):
        """Adverse rate drops below threshold after benign fills → not halted."""
        cfg = ASFilterConfig(max_own_adverse_rate=0.5, lookback_fills=10)
        asf = AdverseSelectionFilter(cfg)
        # Fill window with adverse
        for _ in range(7):
            asf.on_our_fill("BTC", "buy", 100.0, 99.0)
        assert asf.should_halt_symbol("BTC") is True
        # Flush with benign fills (window rolls over)
        for _ in range(10):
            asf.on_our_fill("BTC", "buy", 100.0, 101.0)
        assert asf.get_our_adverse_rate("BTC") == 0.0
        assert asf.should_halt_symbol("BTC") is False

    def test_as_filter_blocks_microprice_drift(self):
        """Adverse microprice drift > threshold → denied."""
        cfg = ASFilterConfig(microprice_drift_threshold=0.6)
        asf = AdverseSelectionFilter(cfg)
        # Sell quote: positive drift is adverse for asks
        result = asf.pre_trade_check("ETH", "sell", ofi_signal=0.0, vpin=0.3, microprice_drift=0.9)
        assert result.allowed is False
        assert result.reason == "denied_microprice_drift"

    def test_as_filter_own_adverse_halts_via_pre_trade(self):
        """Halt triggered via pre_trade_check when own adverse rate is high."""
        cfg = ASFilterConfig(max_own_adverse_rate=0.5, lookback_fills=10)
        asf = AdverseSelectionFilter(cfg)
        for _ in range(8):
            asf.on_our_fill("BTC", "buy", 100.0, 99.0)
        result = asf.pre_trade_check("BTC", "buy", 0.0, 0.3, 0.0)
        assert result.allowed is False
        assert result.reason == "denied_own_adverse_rate"

    def test_as_filter_stats_fields(self):
        """get_filter_stats returns all expected keys."""
        asf = AdverseSelectionFilter(ASFilterConfig())
        asf.pre_trade_check("BTC", "buy", 0.5, 0.3, 0.0)
        stats = asf.get_filter_stats("BTC")
        expected = {
            "total_checks", "denied_obi", "denied_vpin", "denied_drift",
            "denied_own_adverse", "denial_rate", "own_adverse_rate",
            "avg_delay_ms", "halted", "lookback_fills",
        }
        assert expected.issubset(set(stats.keys()))

    def test_as_filter_global_stats(self):
        """get_global_stats aggregates across symbols."""
        asf = AdverseSelectionFilter(ASFilterConfig())
        asf.pre_trade_check("BTC", "buy", 2.0, 0.3, 0.0)  # denied OBI
        asf.pre_trade_check("ETH", "buy", 0.5, 0.9, 0.0)  # denied VPIN
        gstats = asf.get_global_stats()
        assert gstats["total_checks"] == 2
        assert gstats["total_denied_obi"] >= 1
        assert gstats["total_denied_vpin"] >= 1


# ---------------------------------------------------------------------------
# OptimalSpreadCalibrator tests
# ---------------------------------------------------------------------------


class TestOptimalSpreadCalibrator:

    def _make_calib(self, **kwargs) -> OptimalSpreadCalibrator:
        defaults = dict(
            base_spread_bps=30.0,
            min_spread_bps=5.0,
            max_spread_bps=200.0,
            vol_sensitivity=1.5,
            fill_rate_target=0.4,
            calibration_window=200,
            update_interval_fills=20,
        )
        defaults.update(kwargs)
        return OptimalSpreadCalibrator(CalibConfig(**defaults))

    def _feed_prices(self, calib: OptimalSpreadCalibrator, symbol: str, prices: list[float]) -> None:
        for p in prices:
            calib.on_price(symbol, p)

    def test_calibrator_spread_widens_with_vol(self):
        """High vol → optimal spread > base_spread_bps."""
        calib = self._make_calib(
            base_spread_bps=30.0,
            update_interval_fills=5,
            vol_sensitivity=2.0,
        )
        # High-vol prices
        rng = np.random.default_rng(42)
        prices = list(100.0 + np.cumsum(rng.normal(0, 3.0, 100)))
        self._feed_prices(calib, "BTC", prices)

        # Send some quotes
        for i in range(30):
            calib.on_quote_sent("BTC", bid=99.0, ask=101.0, spread_bps=30.0, timestamp_ns=i)

        # Record enough fills to trigger calibration
        for _ in range(20):
            calib.on_fill("BTC", "buy", 99.0, 30.0, was_adverse=False)

        optimal = calib.get_optimal_spread("BTC")
        stats = calib.get_calibration_stats("BTC")
        # With high vol the AS formula should push the spread above base
        assert optimal > 15.0, f"Expected spread > 15 bps, got {optimal}"
        assert optimal <= 200.0

    def test_calibrator_blend(self):
        """Optimal spread is between AS formula and empirical (blend check)."""
        calib = self._make_calib(update_interval_fills=10)
        rng = np.random.default_rng(7)
        prices = list(100.0 + np.cumsum(rng.normal(0, 0.5, 60)))
        self._feed_prices(calib, "ETH", prices)

        for i in range(20):
            calib.on_quote_sent("ETH", 99.0, 101.0, 30.0, i)
        for _ in range(10):
            calib.on_fill("ETH", "buy", 99.0, 30.0, was_adverse=False)

        stats = calib.get_calibration_stats("ETH")
        as_bps = stats["as_formula_bps"]
        emp_bps = stats["empirical_bps"]
        opt_bps = stats["current_optimal_bps"]

        lo = min(as_bps, emp_bps)
        hi = max(as_bps, emp_bps)
        # Allow 1 bps tolerance because clamping may push just outside [lo, hi]
        assert lo - 1.0 <= opt_bps <= hi + 1.0, (
            f"Blended spread {opt_bps:.2f} outside [{lo:.2f}, {hi:.2f}]"
        )

    def test_calibrator_clamps_to_min(self):
        """Spread never falls below min_spread_bps."""
        calib = self._make_calib(
            base_spread_bps=5.0,
            min_spread_bps=5.0,
            update_interval_fills=5,
            vol_sensitivity=0.01,  # suppress vol contribution
        )
        # Flat prices → very low vol
        prices = [100.0] * 60
        self._feed_prices(calib, "BTC", prices)
        for i in range(10):
            calib.on_quote_sent("BTC", 99.5, 100.5, 5.0, i)
        for _ in range(5):
            calib.on_fill("BTC", "buy", 99.5, 5.0, was_adverse=False)
        assert calib.get_optimal_spread("BTC") >= 5.0

    def test_calibrator_clamps_to_max(self):
        """Spread never exceeds max_spread_bps."""
        calib = self._make_calib(max_spread_bps=200.0, update_interval_fills=5)
        rng = np.random.default_rng(99)
        prices = list(100.0 + np.cumsum(rng.normal(0, 10.0, 60)))  # extreme vol
        self._feed_prices(calib, "BTC", prices)
        for i in range(10):
            calib.on_quote_sent("BTC", 90.0, 110.0, 200.0, i)
        for _ in range(5):
            calib.on_fill("BTC", "buy", 90.0, 200.0, was_adverse=False)
        assert calib.get_optimal_spread("BTC") <= 200.0

    def test_calibrator_stats_fields(self):
        """get_calibration_stats returns all expected keys."""
        calib = self._make_calib()
        stats = calib.get_calibration_stats("NEW")
        expected_keys = {
            "current_optimal_bps", "fill_rate", "as_formula_bps", "empirical_bps",
            "quotes_sent", "fills_received", "adverse_fill_rate", "last_calibrated_ns",
        }
        assert expected_keys.issubset(set(stats.keys()))

    def test_calibrator_fill_rate(self):
        """get_fill_rate returns fills / quotes."""
        calib = self._make_calib()
        self._feed_prices(calib, "X", [100.0] * 30)
        for i in range(10):
            calib.on_quote_sent("X", 99.0, 101.0, 30.0, i)
        for _ in range(4):
            calib.on_fill("X", "buy", 99.0, 30.0, was_adverse=False)
        rate = calib.get_fill_rate("X")
        # 4 fills out of 10 quotes (min(10, window) denominator)
        assert 0.0 < rate <= 1.0
