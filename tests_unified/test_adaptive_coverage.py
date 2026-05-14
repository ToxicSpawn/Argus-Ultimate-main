"""
M22 — Adaptive module test coverage.

Covers:
  - adaptive/regime.py          — RegimeDetector.detect() returns valid MarketRegime enum
  - adaptive/feature_engineering.py — generate_candidate_features() and AutomatedFeatureEngine
  - adaptive/online_tuner.py    — OnlineStrategyTuner init and record_trade/confidence_multiplier
  - adaptive/auto_risk_adjuster.py — AutoRiskAdjuster.assess_risk_level() returns float multiplier
"""
from __future__ import annotations

import math
import pytest
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Regime detection tests
# ---------------------------------------------------------------------------

class TestRegimeDetector:
    """Tests for adaptive/regime.py — RegimeDetector."""

    def _make_df(self, n: int = 120, trend: float = 0.0, noise: float = 1.0) -> pd.DataFrame:
        """Create a synthetic OHLCV DataFrame."""
        rng = np.random.default_rng(42)
        close = 65_000.0 + np.cumsum(rng.normal(trend, noise, n))
        close = np.maximum(close, 100.0)  # keep positive
        return pd.DataFrame({
            "close": close,
            "open": close * 0.999,
            "high": close * 1.001,
            "low":  close * 0.998,
            "volume": rng.uniform(100, 500, n),
        })

    def test_detect_returns_regime_snapshot(self):
        """detect() returns a non-None RegimeSnapshot for valid data."""
        from adaptive.regime import RegimeDetector, RegimeSnapshot

        detector = RegimeDetector()
        df = self._make_df(n=120)
        snapshot = detector.detect(df)

        assert snapshot is not None
        assert isinstance(snapshot, RegimeSnapshot)

    def test_detect_regime_is_valid_enum(self):
        """The regime field is a valid MarketRegime enum member."""
        from adaptive.regime import RegimeDetector, MarketRegime

        detector = RegimeDetector()
        df = self._make_df(n=120)
        snapshot = detector.detect(df)

        assert snapshot.regime in MarketRegime

    def test_detect_returns_none_on_empty_df(self):
        """detect() returns None when given an empty DataFrame."""
        from adaptive.regime import RegimeDetector

        detector = RegimeDetector()
        empty = pd.DataFrame({"close": pd.Series(dtype=float)})
        assert detector.detect(empty) is None

    def test_detect_returns_none_on_insufficient_data(self):
        """detect() returns None when data is too short for the slow EMA."""
        from adaptive.regime import RegimeDetector

        detector = RegimeDetector(slow_ema=48)
        short_df = self._make_df(n=10)
        assert detector.detect(short_df) is None

    def test_detect_trend_up_regime(self):
        """Strong uptrend data produces TREND_UP or HIGH_VOL regime."""
        from adaptive.regime import RegimeDetector, MarketRegime

        detector = RegimeDetector(
            fast_ema=5, slow_ema=20, vol_window=20, trend_threshold=0.0001, high_vol_threshold=999.0
        )
        # Strongly trending up: each candle +100
        close = np.arange(65_000.0, 65_000.0 + 200 * 100, 100)  # 200 candles
        df = pd.DataFrame({"close": close})
        snap = detector.detect(df)
        assert snap is not None
        assert snap.regime in (MarketRegime.TREND_UP, MarketRegime.HIGH_VOL)

    def test_detect_trend_score_is_finite(self):
        """trend_score is a finite float (not nan/inf)."""
        from adaptive.regime import RegimeDetector

        detector = RegimeDetector()
        df = self._make_df(n=120)
        snap = detector.detect(df)
        assert snap is not None
        assert math.isfinite(snap.trend_score)
        assert math.isfinite(snap.vol_annualized)

    def test_detect_vol_annualized_positive(self):
        """vol_annualized is always positive."""
        from adaptive.regime import RegimeDetector

        detector = RegimeDetector()
        df = self._make_df(n=120, noise=5.0)
        snap = detector.detect(df)
        assert snap is not None
        assert snap.vol_annualized > 0.0

    def test_detect_no_close_column(self):
        """detect() handles missing 'close' column gracefully."""
        from adaptive.regime import RegimeDetector

        detector = RegimeDetector()
        df = pd.DataFrame({"price": [1, 2, 3]})
        assert detector.detect(df) is None

    def test_online_threshold_update(self):
        """update_thresholds() updates internal adaptive thresholds without crashing."""
        from adaptive.regime import RegimeDetector

        detector = RegimeDetector()
        for i in range(30):
            detector.update_thresholds(trend_score=0.001 * i, vol_annualized=0.5 + 0.01 * i)
        # Thresholds should still be within valid bounds
        assert 0.0008 <= detector.trend_threshold <= 0.01
        assert 0.5 <= detector.high_vol_threshold <= 2.5


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    """Tests for adaptive/feature_engineering.py."""

    def _make_close(self, n: int = 100) -> pd.Series:
        rng = np.random.default_rng(7)
        return pd.Series(65_000.0 + np.cumsum(rng.normal(0, 10, n)))

    def test_generate_candidate_features_returns_list(self):
        """generate_candidate_features returns a non-empty list of (name, series) tuples."""
        from adaptive.feature_engineering import generate_candidate_features

        close = self._make_close(n=100)
        features = generate_candidate_features(close)

        assert isinstance(features, list)
        assert len(features) > 0

    def test_generated_feature_tuples_structure(self):
        """Each item in generate_candidate_features output is (str, pd.Series)."""
        from adaptive.feature_engineering import generate_candidate_features

        close = self._make_close(n=100)
        features = generate_candidate_features(close)

        for name, series in features:
            assert isinstance(name, str), f"Feature name should be str, got {type(name)}"
            assert isinstance(series, pd.Series), f"Feature value should be pd.Series, got {type(series)}"

    def test_generated_feature_names_meaningful(self):
        """Feature names include expected patterns like rsi_, ret_, vol_."""
        from adaptive.feature_engineering import generate_candidate_features

        close = self._make_close(n=100)
        names = [name for name, _ in generate_candidate_features(close)]

        has_expected = any(
            name.startswith("rsi_") or name.startswith("ret_") or name.startswith("vol_")
            for name in names
        )
        assert has_expected, f"No expected feature names found: {names}"

    def test_generate_candidate_features_limit(self):
        """generate_candidate_features respects the `limit` parameter."""
        from adaptive.feature_engineering import generate_candidate_features

        close = self._make_close(n=100)
        limited = generate_candidate_features(close, limit=3)
        assert len(limited) <= 3

    def test_generate_features_correct_index_length(self):
        """Feature series have same index as input close series."""
        from adaptive.feature_engineering import generate_candidate_features

        n = 80
        close = self._make_close(n=n)
        features = generate_candidate_features(close)

        for name, series in features:
            assert len(series) == n, f"{name} series length mismatch: {len(series)} vs {n}"

    def test_automated_feature_engine_init(self):
        """AutomatedFeatureEngine initialises without error."""
        from adaptive.feature_engineering import AutomatedFeatureEngine

        engine = AutomatedFeatureEngine(top_k=5)
        assert engine.top_k == 5

    def test_automated_feature_engine_fit_and_select_returns_list(self):
        """AutomatedFeatureEngine.fit_and_select() returns a list of feature name strings."""
        from adaptive.feature_engineering import AutomatedFeatureEngine

        rng = np.random.default_rng(99)
        n = 100
        df = pd.DataFrame({
            "close": 65_000.0 + np.cumsum(rng.normal(0, 10, n)),
            "open":  65_000.0 + np.cumsum(rng.normal(0, 10, n)),
            "high":  65_100.0 + np.cumsum(rng.normal(0, 10, n)),
            "low":   64_900.0 + np.cumsum(rng.normal(0, 10, n)),
            "volume": rng.uniform(100, 500, n),
        })

        engine = AutomatedFeatureEngine(top_k=5)
        selected = engine.fit_and_select(df)

        assert isinstance(selected, list)
        for name in selected:
            assert isinstance(name, str)

    def test_automated_feature_engine_empty_df(self):
        """AutomatedFeatureEngine.fit_and_select() returns [] for empty DataFrame."""
        from adaptive.feature_engineering import AutomatedFeatureEngine

        engine = AutomatedFeatureEngine(top_k=5)
        assert engine.fit_and_select(pd.DataFrame()) == []

    def test_score_feature_returns_float(self):
        """score_feature returns a float in [-1, 1] range."""
        from adaptive.feature_engineering import score_feature

        rng = np.random.default_rng(12)
        n = 60
        feat = pd.Series(rng.standard_normal(n))
        fwd  = pd.Series(rng.standard_normal(n))
        score = score_feature(feat, fwd)

        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Online tuner tests
# ---------------------------------------------------------------------------

class TestOnlineStrategyTuner:
    """Tests for adaptive/online_tuner.py — OnlineStrategyTuner."""

    def _make_tuner(self):
        from adaptive.online_tuner import OnlineStrategyTuner
        return OnlineStrategyTuner(alpha=0.15, min_trades_before_bias=5)

    def test_tuner_init_no_crash(self):
        """OnlineStrategyTuner initialises without error."""
        tuner = self._make_tuner()
        assert tuner.alpha == 0.15

    def test_tuner_record_trade_no_crash(self):
        """record_trade() runs without raising."""
        from adaptive.regime import MarketRegime

        tuner = self._make_tuner()
        tuner.record_trade(
            symbol="BTC/USDT",
            regime=MarketRegime.TREND_UP,
            mode="momentum",
            pnl_pct=1.5,
        )

    def test_tuner_confidence_multiplier_default_before_min_trades(self):
        """confidence_multiplier returns base value before min_trades_before_bias."""
        from adaptive.regime import MarketRegime

        tuner = self._make_tuner()
        # No trades recorded yet
        result = tuner.confidence_multiplier(
            symbol="BTC/USDT",
            regime=MarketRegime.TREND_UP,
            mode="momentum",
            base=1.0,
        )
        assert result == 1.0

    def test_tuner_confidence_multiplier_after_trades(self):
        """confidence_multiplier returns a finite float in [0.7, 1.3] after min trades."""
        from adaptive.regime import MarketRegime

        tuner = self._make_tuner()
        # Record enough trades to exceed min_trades_before_bias=5
        for _ in range(10):
            tuner.record_trade(
                symbol="ETH/USDT",
                regime=MarketRegime.RANGE,
                mode="scalping",
                pnl_pct=0.5,
            )

        result = tuner.confidence_multiplier(
            symbol="ETH/USDT",
            regime=MarketRegime.RANGE,
            mode="scalping",
            base=1.0,
        )
        assert isinstance(result, float)
        assert math.isfinite(result)
        assert 0.7 <= result <= 1.3

    def test_tuner_threshold_adjustments_empty_before_trades(self):
        """threshold_adjustments() returns {} before enough trades."""
        from adaptive.regime import MarketRegime

        tuner = self._make_tuner()
        adj = tuner.threshold_adjustments(
            symbol="BTC/USDT",
            regime=MarketRegime.TREND_DOWN,
            mode="mean_reversion",
        )
        assert adj == {}

    def test_tuner_threshold_adjustments_after_trades(self):
        """threshold_adjustments returns a dict with 'selectivity' key after enough trades."""
        from adaptive.regime import MarketRegime

        tuner = self._make_tuner()
        for _ in range(10):
            tuner.record_trade(
                symbol="BTC/USDT",
                regime=MarketRegime.TREND_DOWN,
                mode="mean_reversion",
                pnl_pct=-0.3,
            )
        adj = tuner.threshold_adjustments(
            symbol="BTC/USDT",
            regime=MarketRegime.TREND_DOWN,
            mode="mean_reversion",
        )
        assert isinstance(adj, dict)
        assert "selectivity" in adj
        assert isinstance(adj["selectivity"], float)

    def test_tuner_status_returns_dict(self):
        """status() returns a dict with recorded symbol data."""
        from adaptive.regime import MarketRegime

        tuner = self._make_tuner()
        tuner.record_trade(
            symbol="SOL/USDT",
            regime=MarketRegime.HIGH_VOL,
            mode="breakout",
            pnl_pct=2.0,
        )
        status = tuner.status()
        assert isinstance(status, dict)
        assert "SOL/USDT" in status

    def test_tuner_multiple_symbols_no_interference(self):
        """Records for different symbols are isolated."""
        from adaptive.regime import MarketRegime

        tuner = self._make_tuner()
        for _ in range(10):
            tuner.record_trade(symbol="BTC/USDT", regime=MarketRegime.TREND_UP, mode="momentum", pnl_pct=1.0)
        for _ in range(10):
            tuner.record_trade(symbol="ETH/USDT", regime=MarketRegime.TREND_DOWN, mode="momentum", pnl_pct=-2.0)

        btc_mult = tuner.confidence_multiplier(symbol="BTC/USDT", regime=MarketRegime.TREND_UP, mode="momentum")
        eth_mult = tuner.confidence_multiplier(symbol="ETH/USDT", regime=MarketRegime.TREND_DOWN, mode="momentum")

        # BTC (positive pnl) should have higher multiplier than ETH (negative pnl)
        assert btc_mult > eth_mult, "Positive-PnL symbol should have higher confidence than negative-PnL"


# ---------------------------------------------------------------------------
# AutoRiskAdjuster tests
# ---------------------------------------------------------------------------

class TestAutoRiskAdjuster:
    """Tests for adaptive/auto_risk_adjuster.py — AutoRiskAdjuster."""

    def _make_adjuster(self, **overrides):
        from adaptive.auto_risk_adjuster import AutoRiskAdjuster
        return AutoRiskAdjuster(config=overrides)

    def test_adjuster_init_no_crash(self):
        """AutoRiskAdjuster initialises without error."""
        adjuster = self._make_adjuster()
        assert adjuster is not None

    def test_assess_risk_level_returns_assessment(self):
        """assess_risk_level() returns a RiskAssessment object."""
        from adaptive.auto_risk_adjuster import AutoRiskAdjuster, RiskAssessment

        adjuster = AutoRiskAdjuster()
        result = adjuster.assess_risk_level({})
        assert isinstance(result, RiskAssessment)

    def test_assess_risk_level_multiplier_is_float(self):
        """position_multiplier is a finite float."""
        adjuster = self._make_adjuster()
        result = adjuster.assess_risk_level({
            "drawdown_pct": 0.0,
            "volatility": 0.3,
            "win_streak": 0,
            "loss_streak": 0,
        })
        assert isinstance(result.position_multiplier, float)
        assert math.isfinite(result.position_multiplier)

    def test_assess_risk_multiplier_clamped(self):
        """position_multiplier stays in [0.5, 1.5] regardless of inputs."""
        adjuster = self._make_adjuster()

        # Extreme drawdown — should pull multiplier toward floor
        result_dd = adjuster.assess_risk_level({
            "drawdown_pct": 50.0,
            "volatility": 2.0,
            "loss_streak": 20,
            "utc_hour": 2,
            "day_of_week": 6,
        })
        assert 0.5 <= result_dd.position_multiplier <= 1.5

        # Great conditions — should stay below ceiling
        result_great = adjuster.assess_risk_level({
            "drawdown_pct": 0.0,
            "volatility": 0.1,
            "win_streak": 10,
            "utc_hour": 14,
            "day_of_week": 2,
        })
        assert 0.5 <= result_great.position_multiplier <= 1.5

    def test_assess_risk_level_no_drawdown(self):
        """No drawdown leads to multiplier near 1.0 (no reduction)."""
        adjuster = self._make_adjuster()
        result = adjuster.assess_risk_level({
            "drawdown_pct": 0.0,
            "volatility": 0.3,
            "utc_hour": 14,
            "day_of_week": 1,
        })
        assert result.position_multiplier >= 0.9

    def test_assess_risk_level_high_drawdown_reduces_multiplier(self):
        """Large drawdown reduces position multiplier below no-drawdown case."""
        adjuster = self._make_adjuster()
        no_dd = adjuster.assess_risk_level({"drawdown_pct": 0.0, "utc_hour": 14, "day_of_week": 1})
        big_dd = adjuster.assess_risk_level({"drawdown_pct": 30.0, "utc_hour": 14, "day_of_week": 1})
        assert big_dd.position_multiplier < no_dd.position_multiplier

    def test_assess_risk_level_returns_valid_level_string(self):
        """Level is one of 'conservative', 'normal', 'aggressive'."""
        adjuster = self._make_adjuster()
        result = adjuster.assess_risk_level({})
        assert result.level in ("conservative", "normal", "aggressive")

    def test_assess_risk_level_disabled_returns_normal(self):
        """When enabled=False, returns default normal assessment."""
        from adaptive.auto_risk_adjuster import AutoRiskAdjuster

        adjuster = AutoRiskAdjuster(config={"enabled": False})
        result = adjuster.assess_risk_level({"drawdown_pct": 50.0, "volatility": 5.0})
        assert result.level == "normal"
        assert result.position_multiplier == 1.0

    def test_assess_macro_event_reduces_risk(self):
        """Upcoming macro event reduces position multiplier."""
        adjuster = self._make_adjuster()
        no_event = adjuster.assess_risk_level({"utc_hour": 14, "day_of_week": 1})
        with_event = adjuster.assess_risk_level({
            "utc_hour": 14,
            "day_of_week": 1,
            "upcoming_events": [{"name": "FOMC", "hours_until": 1.0}],
        })
        assert with_event.position_multiplier < no_event.position_multiplier

    def test_apply_risk_level_no_system(self):
        """apply_risk_level returns False when no system is provided."""
        adjuster = self._make_adjuster()
        result = adjuster.assess_risk_level({})
        applied = adjuster.apply_risk_level(result, system=None)
        assert applied is False

    def test_adjuster_history_accumulates(self):
        """history grows after repeated assess_risk_level calls."""
        adjuster = self._make_adjuster()
        for _ in range(5):
            adjuster.assess_risk_level({})
        assert len(adjuster.history) == 5
