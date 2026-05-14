"""
Tests for Deep Intelligence modules — 60+ tests covering:

1. TransformerPricePredictor  (ml/transformer_price_predictor.py)
2. MultiHorizonFusion         (ml/multi_horizon_fusion.py)
3. CausalDiscoveryEngine      (ml/causal_discovery.py)
4. DynamicDrawdownController   (risk/dynamic_drawdown_controller.py)
5. RegimeConditionalVaR        (risk/regime_conditional_var.py)
6. OpportunityCostTracker      (monitoring/opportunity_cost_tracker.py)
"""

from __future__ import annotations

import math
import os
import tempfile
import time

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 1. TransformerPricePredictor
# ---------------------------------------------------------------------------
from ml.transformer_price_predictor import TransformerPricePredictor, PricePrediction


class TestTransformerPricePredictor:
    """Tests for TransformerPricePredictor (momentum fallback path)."""

    def _make_ohlcv(self, n: int = 100, trend: float = 0.001) -> list:
        """Generate synthetic OHLCV bars with upward trend."""
        rng = np.random.default_rng(42)
        bars = []
        price = 100.0
        for _ in range(n):
            ret = trend + rng.normal(0, 0.01)
            o = price
            c = price * (1 + ret)
            h = max(o, c) * (1 + abs(rng.normal(0, 0.003)))
            l = min(o, c) * (1 - abs(rng.normal(0, 0.003)))
            v = rng.uniform(1000, 5000)
            bars.append([o, h, l, c, v])
            price = c
        return bars

    def test_init(self):
        pred = TransformerPricePredictor()
        assert not pred.is_trained
        assert pred.prediction_count == 0

    def test_fit_fallback(self):
        pred = TransformerPricePredictor()
        bars = self._make_ohlcv(100)
        result = pred.fit(bars, seq_len=60, epochs=2)
        assert result["method"] == "momentum_fallback" or result["method"] == "transformer"
        assert pred.is_trained

    def test_fit_too_few_bars(self):
        pred = TransformerPricePredictor()
        result = pred.fit([[100, 101, 99, 100, 1000]], seq_len=60)
        assert pred.is_trained
        assert result["n_samples"] == 1

    def test_predict_next_bar_returns_prediction(self):
        pred = TransformerPricePredictor()
        bars = self._make_ohlcv(100)
        pred.fit(bars, seq_len=20)
        result = pred.predict_next_bar(bars[-30:])
        assert isinstance(result, PricePrediction)
        assert result.direction in ("up", "down")
        assert result.magnitude_pct >= 0.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.timestamp > 0

    def test_predict_with_two_bars(self):
        pred = TransformerPricePredictor()
        bars = [[100, 101, 99, 100, 1000], [100, 102, 99, 101, 1200]]
        pred.fit(bars, seq_len=1)
        result = pred.predict_next_bar(bars)
        assert result.direction in ("up", "down")

    def test_predict_single_bar(self):
        pred = TransformerPricePredictor()
        result = pred.predict_next_bar([[100, 101, 99, 100, 1000]])
        assert result.confidence == 0.0

    def test_accuracy_no_data(self):
        pred = TransformerPricePredictor()
        assert pred.get_accuracy() == 0.0

    def test_accuracy_tracking(self):
        pred = TransformerPricePredictor()
        bars = self._make_ohlcv(50, trend=0.005)
        pred.fit(bars, seq_len=10)
        # Make multiple predictions with price updates between them
        for i in range(10, 48):
            pred.predict_next_bar(bars[:i + 1])
        acc = pred.get_accuracy(lookback=100)
        assert 0.0 <= acc <= 1.0

    def test_prediction_count_increments(self):
        pred = TransformerPricePredictor()
        bars = self._make_ohlcv(30)
        pred.fit(bars, seq_len=5)
        pred.predict_next_bar(bars[:10])
        pred.predict_next_bar(bars[:11])
        assert pred.prediction_count == 2

    def test_uptrend_detects_up(self):
        """Strong uptrend should predict 'up' most of the time."""
        pred = TransformerPricePredictor(fallback_lookback=10)
        bars = self._make_ohlcv(40, trend=0.02)
        pred.fit(bars, seq_len=5)
        result = pred.predict_next_bar(bars[-15:])
        assert result.direction == "up"

    def test_downtrend_detects_down(self):
        pred = TransformerPricePredictor(fallback_lookback=10)
        bars = self._make_ohlcv(40, trend=-0.02)
        pred.fit(bars, seq_len=5)
        result = pred.predict_next_bar(bars[-15:])
        assert result.direction == "down"


# ---------------------------------------------------------------------------
# 2. MultiHorizonFusion
# ---------------------------------------------------------------------------
from ml.multi_horizon_fusion import MultiHorizonFusion, FusedSignal, VALID_TIMEFRAMES


class TestMultiHorizonFusion:

    def test_init(self):
        mhf = MultiHorizonFusion()
        assert mhf.active_signals_count == 0

    def test_add_signal(self):
        mhf = MultiHorizonFusion()
        mhf.add_signal("1h", "BTC/AUD", "long", 0.8)
        assert mhf.active_signals_count == 1

    def test_add_invalid_timeframe_ignored(self):
        mhf = MultiHorizonFusion()
        mhf.add_signal("2h", "BTC/AUD", "long", 0.8)
        assert mhf.active_signals_count == 0

    def test_add_invalid_direction_ignored(self):
        mhf = MultiHorizonFusion()
        mhf.add_signal("1h", "BTC/AUD", "sideways", 0.8)
        assert mhf.active_signals_count == 0

    def test_fuse_no_signals(self):
        mhf = MultiHorizonFusion()
        result = mhf.fuse("BTC/AUD")
        assert result.direction == "neutral"
        assert result.confidence == 0.0

    def test_fuse_single_long(self):
        mhf = MultiHorizonFusion()
        mhf.add_signal("1h", "BTC/AUD", "long", 0.9)
        result = mhf.fuse("BTC/AUD")
        assert result.direction == "long"
        assert result.confidence > 0

    def test_fuse_single_short(self):
        mhf = MultiHorizonFusion()
        mhf.add_signal("4h", "BTC/AUD", "short", 0.85)
        result = mhf.fuse("BTC/AUD")
        assert result.direction == "short"

    def test_fuse_unanimous_long(self):
        mhf = MultiHorizonFusion()
        for tf in VALID_TIMEFRAMES:
            mhf.add_signal(tf, "ETH/AUD", "long", 0.7)
        result = mhf.fuse("ETH/AUD")
        assert result.direction == "long"
        assert result.agreement_pct == 1.0
        assert len(result.contributing_timeframes) == len(VALID_TIMEFRAMES)

    def test_fuse_mixed_majority_long(self):
        mhf = MultiHorizonFusion()
        mhf.add_signal("1m", "BTC/AUD", "long", 0.8)
        mhf.add_signal("5m", "BTC/AUD", "long", 0.7)
        mhf.add_signal("15m", "BTC/AUD", "long", 0.6)
        mhf.add_signal("1h", "BTC/AUD", "short", 0.5)
        result = mhf.fuse("BTC/AUD")
        assert result.direction == "long"
        assert result.agreement_pct > 0.5

    def test_fuse_stale_signals_excluded(self):
        mhf = MultiHorizonFusion(signal_ttl_s=0.01)
        mhf.add_signal("1h", "BTC/AUD", "long", 0.9)
        time.sleep(0.02)
        result = mhf.fuse("BTC/AUD")
        assert result.direction == "neutral"

    def test_dominant_timeframe(self):
        mhf = MultiHorizonFusion()
        mhf.add_signal("1m", "BTC/AUD", "long", 0.3)
        mhf.add_signal("1d", "BTC/AUD", "long", 0.99)
        result = mhf.fuse("BTC/AUD")
        assert result.dominant_timeframe in ("1m", "1d")

    def test_update_regime_weights(self):
        mhf = MultiHorizonFusion()
        mhf.update_regime_weights("crisis", {"1d": 5.0, "4h": 3.0, "1h": 1.0})
        weights = mhf.get_regime_weights("crisis")
        assert weights["1d"] > weights["1h"]

    def test_set_regime(self):
        mhf = MultiHorizonFusion()
        mhf.update_regime_weights("trending", {"1h": 0.8, "4h": 0.2})
        mhf.set_regime("trending")
        weights = mhf.get_regime_weights()
        assert "1h" in weights

    def test_record_outcome_and_accuracy(self):
        mhf = MultiHorizonFusion()
        mhf.record_outcome("1h", "long", True)
        mhf.record_outcome("1h", "long", True)
        mhf.record_outcome("1h", "short", False)
        acc = mhf.get_timeframe_accuracy("1h", lookback_days=1)
        assert abs(acc - 2.0 / 3.0) < 0.01

    def test_accuracy_no_data(self):
        mhf = MultiHorizonFusion()
        assert mhf.get_timeframe_accuracy("1h") == 0.0


# ---------------------------------------------------------------------------
# 3. CausalDiscoveryEngine
# ---------------------------------------------------------------------------
from ml.causal_discovery import CausalDiscoveryEngine, GrangerResult


class TestCausalDiscoveryEngine:

    def _make_causal_pair(self, n: int = 200, lag: int = 2):
        """X causes Y with a lag."""
        rng = np.random.default_rng(123)
        x = rng.normal(0, 1, n)
        noise = rng.normal(0, 0.3, n)
        y = np.zeros(n)
        for t in range(lag, n):
            y[t] = 0.7 * x[t - lag] + noise[t]
        return x.tolist(), y.tolist()

    def test_init(self):
        engine = CausalDiscoveryEngine()
        assert engine.series_names == []

    def test_add_series(self):
        engine = CausalDiscoveryEngine()
        engine.add_series("price", [1, 2, 3, 4, 5])
        assert "price" in engine.series_names
        assert engine.series_length("price") == 5

    def test_add_series_2d_raises(self):
        engine = CausalDiscoveryEngine()
        with pytest.raises(ValueError):
            engine.add_series("bad", [[1, 2], [3, 4]])

    def test_granger_missing_series(self):
        engine = CausalDiscoveryEngine()
        engine.add_series("x", [1, 2, 3])
        with pytest.raises(KeyError):
            engine.granger_test("x", "y")

    def test_granger_too_few_observations(self):
        engine = CausalDiscoveryEngine()
        engine.add_series("x", [1, 2, 3])
        engine.add_series("y", [4, 5, 6])
        result = engine.granger_test("x", "y", max_lag=5)
        assert result.p_value == 1.0
        assert not result.significant

    def test_granger_causal_relationship(self):
        engine = CausalDiscoveryEngine()
        x, y = self._make_causal_pair(200, lag=2)
        engine.add_series("x", x)
        engine.add_series("y", y)
        result = engine.granger_test("x", "y", max_lag=5)
        assert isinstance(result, GrangerResult)
        assert result.cause == "x"
        assert result.effect == "y"
        assert result.f_statistic > 0
        # With a clear causal link, this should be significant
        assert result.significant

    def test_granger_no_relationship(self):
        engine = CausalDiscoveryEngine()
        rng = np.random.default_rng(99)
        x = rng.normal(0, 1, 200).tolist()
        y = rng.normal(0, 1, 200).tolist()
        engine.add_series("x", x)
        engine.add_series("y", y)
        result = engine.granger_test("x", "y", max_lag=3)
        # Independent series should typically not be significant
        # (not guaranteed, but p > 0.05 most of the time with seed=99)
        assert result.p_value > 0.01

    def test_discover_all_relationships(self):
        engine = CausalDiscoveryEngine()
        x, y = self._make_causal_pair(200, lag=2)
        rng = np.random.default_rng(77)
        z = rng.normal(0, 1, 200).tolist()
        engine.add_series("x", x)
        engine.add_series("y", y)
        engine.add_series("z", z)
        results = engine.discover_all_relationships(significance=0.05)
        # At minimum x→y should appear
        causes = {(r.cause, r.effect) for r in results}
        assert ("x", "y") in causes

    def test_get_causal_graph(self):
        engine = CausalDiscoveryEngine()
        x, y = self._make_causal_pair(200, lag=2)
        engine.add_series("x", x)
        engine.add_series("y", y)
        graph = engine.get_causal_graph()
        assert isinstance(graph, dict)
        assert "x" in graph
        assert "y" in graph["x"]

    def test_granger_result_fields(self):
        engine = CausalDiscoveryEngine()
        x, y = self._make_causal_pair(200, lag=2)
        engine.add_series("x", x)
        engine.add_series("y", y)
        r = engine.granger_test("x", "y")
        assert r.best_lag >= 1
        assert r.f_statistic >= 0
        assert 0.0 <= r.p_value <= 1.0


# ---------------------------------------------------------------------------
# 4. DynamicDrawdownController
# ---------------------------------------------------------------------------
from risk.dynamic_drawdown_controller import DynamicDrawdownController, DrawdownState


class TestDynamicDrawdownController:

    @pytest.fixture
    def db_path(self, tmp_path):
        return str(tmp_path / "dd_test.db")

    def test_init(self, db_path):
        ctrl = DynamicDrawdownController(db_path=db_path, initial_equity=1000)
        assert ctrl.get_drawdown_pct() == 0.0
        assert ctrl.get_position_multiplier() == 1.0
        assert not ctrl.is_halted()
        ctrl.close()

    def test_update_equity(self, db_path):
        ctrl = DynamicDrawdownController(db_path=db_path, initial_equity=1000)
        state = ctrl.update(950)
        assert state.drawdown_pct == pytest.approx(5.0, abs=0.01)
        ctrl.close()

    def test_multiplier_at_5pct(self, db_path):
        ctrl = DynamicDrawdownController(db_path=db_path, initial_equity=1000)
        ctrl.update(950)
        mult = ctrl.get_position_multiplier()
        assert mult == pytest.approx(0.8, abs=0.01)
        ctrl.close()

    def test_multiplier_at_10pct(self, db_path):
        ctrl = DynamicDrawdownController(db_path=db_path, initial_equity=1000)
        ctrl.update(900)
        mult = ctrl.get_position_multiplier()
        assert mult == pytest.approx(0.5, abs=0.01)
        ctrl.close()

    def test_multiplier_at_15pct(self, db_path):
        ctrl = DynamicDrawdownController(db_path=db_path, initial_equity=1000)
        ctrl.update(850)
        mult = ctrl.get_position_multiplier()
        assert mult == pytest.approx(0.2, abs=0.01)
        ctrl.close()

    def test_halt_at_20pct(self, db_path):
        ctrl = DynamicDrawdownController(db_path=db_path, initial_equity=1000)
        ctrl.update(800)
        assert ctrl.is_halted()
        assert ctrl.get_position_multiplier() == 0.0
        ctrl.close()

    def test_halt_beyond_20pct(self, db_path):
        ctrl = DynamicDrawdownController(db_path=db_path, initial_equity=1000)
        ctrl.update(700)
        assert ctrl.is_halted()
        ctrl.close()

    def test_interpolation_between_points(self, db_path):
        """7.5% drawdown should give multiplier between 0.8 and 0.5."""
        ctrl = DynamicDrawdownController(db_path=db_path, initial_equity=1000)
        ctrl.update(925)  # 7.5% DD
        mult = ctrl.get_position_multiplier()
        assert 0.5 < mult < 0.8
        ctrl.close()

    def test_recovery_estimate(self, db_path):
        ctrl = DynamicDrawdownController(db_path=db_path, initial_equity=1000)
        ctrl.update(900)  # 10% DD
        days = ctrl.get_recovery_estimate(avg_daily_return_pct=0.5)
        assert days > 0
        ctrl.close()

    def test_recovery_no_drawdown(self, db_path):
        ctrl = DynamicDrawdownController(db_path=db_path, initial_equity=1000)
        assert ctrl.get_recovery_estimate(0.5) == 0
        ctrl.close()

    def test_recovery_impossible(self, db_path):
        ctrl = DynamicDrawdownController(db_path=db_path, initial_equity=1000)
        ctrl.update(900)
        assert ctrl.get_recovery_estimate(-0.5) == -1
        ctrl.close()

    def test_persistence(self, db_path):
        ctrl = DynamicDrawdownController(db_path=db_path, initial_equity=1000)
        ctrl.update(920)
        ctrl.close()
        # Re-open
        ctrl2 = DynamicDrawdownController(db_path=db_path)
        assert ctrl2.get_drawdown_pct() == pytest.approx(8.0, abs=0.01)
        ctrl2.close()

    def test_peak_tracking(self, db_path):
        ctrl = DynamicDrawdownController(db_path=db_path, initial_equity=1000)
        ctrl.update(1100)  # new peak
        ctrl.update(1050)  # slight dip from 1100 peak
        dd = ctrl.get_drawdown_pct()
        expected = (1100 - 1050) / 1100 * 100
        assert dd == pytest.approx(expected, abs=0.1)
        ctrl.close()

    def test_get_state(self, db_path):
        ctrl = DynamicDrawdownController(db_path=db_path, initial_equity=1000)
        ctrl.update(950)
        state = ctrl.get_state()
        assert isinstance(state, DrawdownState)
        assert state.current_equity == 950
        assert state.peak_equity == 1000
        assert state.timestamp > 0
        ctrl.close()


# ---------------------------------------------------------------------------
# 5. RegimeConditionalVaR
# ---------------------------------------------------------------------------
from risk.regime_conditional_var import RegimeConditionalVaR, VaRResult


class TestRegimeConditionalVaR:

    def _make_returns(self, n: int = 500, mu: float = 0.0, sigma: float = 0.01):
        rng = np.random.default_rng(42)
        return rng.normal(mu, sigma, n).tolist()

    def test_init(self):
        var = RegimeConditionalVaR()
        assert var.get_method_for_regime("normal") == "parametric"
        assert var.get_method_for_regime("crisis") == "historical"
        assert var.get_method_for_regime("volatile") == "monte_carlo"

    def test_parametric_normal(self):
        var = RegimeConditionalVaR()
        returns = self._make_returns(500, mu=0.0, sigma=0.02)
        result = var.compute_var(returns, "normal", confidence=0.99)
        assert isinstance(result, VaRResult)
        assert result.var_pct > 0
        assert result.cvar_pct >= result.var_pct
        assert result.method_used == "parametric"
        assert result.regime == "normal"

    def test_historical_crisis(self):
        var = RegimeConditionalVaR()
        returns = self._make_returns(500, mu=-0.005, sigma=0.04)
        result = var.compute_var(returns, "crisis", confidence=0.99)
        assert result.var_pct > 0
        assert result.method_used == "historical"

    def test_monte_carlo_volatile(self):
        var = RegimeConditionalVaR()
        returns = self._make_returns(500, mu=0.0, sigma=0.03)
        result = var.compute_var(returns, "volatile", confidence=0.99)
        assert result.var_pct > 0
        assert result.method_used == "monte_carlo"
        assert result.n_samples == 1000

    def test_unknown_regime_uses_parametric(self):
        var = RegimeConditionalVaR()
        returns = self._make_returns(100)
        result = var.compute_var(returns, "alien_regime")
        assert result.method_used == "parametric"

    def test_insufficient_data(self):
        var = RegimeConditionalVaR()
        result = var.compute_var([0.01], "normal")
        assert result.var_pct == 0.0

    def test_empty_returns(self):
        var = RegimeConditionalVaR()
        result = var.compute_var([], "normal")
        assert result.var_pct == 0.0

    def test_portfolio_var(self):
        var = RegimeConditionalVaR()
        rets = {
            "BTC": self._make_returns(200, sigma=0.02),
            "ETH": self._make_returns(200, sigma=0.03),
        }
        pvar = var.get_portfolio_var(rets, "normal")
        assert pvar > 0

    def test_portfolio_var_empty(self):
        var = RegimeConditionalVaR()
        assert var.get_portfolio_var({}, "normal") == 0.0

    def test_higher_confidence_higher_var(self):
        var = RegimeConditionalVaR()
        returns = self._make_returns(500, sigma=0.02)
        var95 = var.compute_var(returns, "normal", confidence=0.95)
        var99 = var.compute_var(returns, "normal", confidence=0.99)
        assert var99.var_pct >= var95.var_pct

    def test_set_regime_method(self):
        var = RegimeConditionalVaR()
        var.set_regime_method("normal", "monte_carlo")
        assert var.get_method_for_regime("normal") == "monte_carlo"

    def test_set_invalid_method_raises(self):
        var = RegimeConditionalVaR()
        with pytest.raises(ValueError):
            var.set_regime_method("normal", "magic")

    def test_var_result_fields(self):
        var = RegimeConditionalVaR()
        returns = self._make_returns(100)
        r = var.compute_var(returns, "normal", confidence=0.95)
        assert r.confidence == 0.95
        assert r.n_samples > 0
        assert r.timestamp > 0


# ---------------------------------------------------------------------------
# 6. OpportunityCostTracker
# ---------------------------------------------------------------------------
from monitoring.opportunity_cost_tracker import OpportunityCostTracker, MissedPnL


class TestOpportunityCostTracker:

    @pytest.fixture
    def db_path(self, tmp_path):
        return str(tmp_path / "opp_cost_test.db")

    def test_init(self, db_path):
        tracker = OpportunityCostTracker(db_path=db_path)
        assert tracker.total_skipped == 0
        tracker.close()

    def test_record_skipped_signal(self, db_path):
        tracker = OpportunityCostTracker(db_path=db_path)
        row_id = tracker.record_skipped_signal(
            "BTC/AUD", "long", 0.72, "confidence_too_low", 98500.0)
        assert row_id > 0
        assert tracker.total_skipped == 1
        tracker.close()

    def test_update_prices_long(self, db_path):
        tracker = OpportunityCostTracker(db_path=db_path)
        tracker.record_skipped_signal("BTC/AUD", "long", 0.72, "risk", 100.0)
        count = tracker.update_prices("BTC/AUD", 110.0)
        assert count == 1
        tracker.close()

    def test_update_prices_short(self, db_path):
        tracker = OpportunityCostTracker(db_path=db_path)
        tracker.record_skipped_signal("BTC/AUD", "short", 0.65, "risk", 100.0)
        tracker.update_prices("BTC/AUD", 90.0)
        missed = tracker.get_missed_pnl(lookback_days=1)
        assert missed.total_missed_usd > 0
        tracker.close()

    def test_missed_pnl_no_data(self, db_path):
        tracker = OpportunityCostTracker(db_path=db_path)
        missed = tracker.get_missed_pnl()
        assert missed.count == 0
        assert missed.total_missed_usd == 0.0
        tracker.close()

    def test_missed_pnl_with_data(self, db_path):
        tracker = OpportunityCostTracker(db_path=db_path, position_size_usd=1000)
        tracker.record_skipped_signal("BTC/AUD", "long", 0.7, "risk", 100.0)
        tracker.record_skipped_signal("ETH/AUD", "long", 0.6, "conf", 50.0)
        tracker.update_prices("BTC/AUD", 110.0)    # +10%
        tracker.update_prices("ETH/AUD", 45.0)     # -10%
        missed = tracker.get_missed_pnl(lookback_days=1)
        assert missed.count == 2
        assert missed.best_missed > 0
        assert missed.worst_missed < 0
        tracker.close()

    def test_skip_reason_analysis(self, db_path):
        tracker = OpportunityCostTracker(db_path=db_path, position_size_usd=1000)
        tracker.record_skipped_signal("BTC/AUD", "long", 0.7, "risk_limit", 100.0)
        tracker.record_skipped_signal("ETH/AUD", "long", 0.5, "confidence", 100.0)
        tracker.update_prices("BTC/AUD", 105.0)
        tracker.update_prices("ETH/AUD", 103.0)
        analysis = tracker.get_skip_reason_analysis()
        assert "risk_limit" in analysis
        assert "confidence" in analysis
        tracker.close()

    def test_get_recent_skips(self, db_path):
        tracker = OpportunityCostTracker(db_path=db_path)
        tracker.record_skipped_signal("BTC/AUD", "long", 0.7, "test", 100.0)
        tracker.record_skipped_signal("ETH/AUD", "short", 0.6, "test", 50.0)
        recent = tracker.get_recent_skips(limit=5)
        assert len(recent) == 2
        assert recent[0].symbol == "ETH/AUD"  # most recent first
        tracker.close()

    def test_multiple_symbols(self, db_path):
        tracker = OpportunityCostTracker(db_path=db_path)
        tracker.record_skipped_signal("BTC/AUD", "long", 0.7, "risk", 100.0)
        tracker.record_skipped_signal("BTC/AUD", "short", 0.6, "risk", 100.0)
        tracker.record_skipped_signal("SOL/AUD", "long", 0.8, "risk", 30.0)
        assert tracker.total_skipped == 3
        # Update only BTC
        count = tracker.update_prices("BTC/AUD", 105.0)
        assert count == 2
        tracker.close()

    def test_avg_missed(self, db_path):
        tracker = OpportunityCostTracker(db_path=db_path, position_size_usd=100)
        for i in range(5):
            tracker.record_skipped_signal("BTC/AUD", "long", 0.7, "test", 100.0)
        tracker.update_prices("BTC/AUD", 110.0)
        missed = tracker.get_missed_pnl(lookback_days=1)
        assert missed.avg_missed > 0
        assert missed.count == 5
        tracker.close()
