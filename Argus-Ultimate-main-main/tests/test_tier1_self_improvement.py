"""
tests/test_tier1_self_improvement.py — Tests for Tier 1 Self-Improvement Modules

Covers:
  - OptunaTuner (Bayesian hyperparameter optimization)
  - RegimeStrategyRouter (regime-aware strategy switching)
  - ExecutionQualityTracker (execution quality feedback loop)
  - DynamicCycleTimer (dynamic cycle timing)
  - KellyPositionSizer (Kelly criterion position sizing)

50+ tests total, 10+ per module.
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import pytest


# ============================================================================
# OptunaTuner
# ============================================================================

class TestOptunaTuner:
    """Tests for ml.optuna_tuner.OptunaTuner."""

    def _make_tuner(self, tmp_path):
        from ml.optuna_tuner import OptunaTuner
        db = os.path.join(str(tmp_path), "test_optuna.db")
        return OptunaTuner(db_path=db)

    def test_init(self, tmp_path):
        tuner = self._make_tuner(tmp_path)
        assert tuner is not None

    def test_create_study(self, tmp_path):
        from ml.optuna_tuner import _HAS_OPTUNA
        tuner = self._make_tuner(tmp_path)
        study = tuner.create_study("test_strat")
        if _HAS_OPTUNA:
            assert study is not None
            assert study.study_name == "test_strat"
        else:
            assert study is None

    def test_create_study_idempotent(self, tmp_path):
        from ml.optuna_tuner import _HAS_OPTUNA
        if not _HAS_OPTUNA:
            pytest.skip("optuna not installed")
        tuner = self._make_tuner(tmp_path)
        s1 = tuner.create_study("strat_a")
        s2 = tuner.create_study("strat_a")
        assert s1 is s2

    def test_optimize_simple_objective(self, tmp_path):
        from ml.optuna_tuner import _HAS_OPTUNA
        if not _HAS_OPTUNA:
            pytest.skip("optuna not installed")
        tuner = self._make_tuner(tmp_path)

        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return -(x - 3.0) ** 2  # max at x=3

        best = tuner.optimize("simple", objective, n_trials=30, timeout_s=30)
        assert "x" in best
        assert abs(best["x"] - 3.0) < 2.0  # should be near 3

    def test_get_best_params_after_optimize(self, tmp_path):
        from ml.optuna_tuner import _HAS_OPTUNA
        if not _HAS_OPTUNA:
            pytest.skip("optuna not installed")
        tuner = self._make_tuner(tmp_path)

        def objective(trial):
            x = trial.suggest_float("x", 0, 1)
            return -x  # minimize x for maximize → wants x near 0... wait, direction=maximize
            # maximize -x → x should be small

        tuner.optimize("bp_test", objective, n_trials=10, timeout_s=10)
        best = tuner.get_best_params("bp_test")
        assert "x" in best

    def test_get_best_params_no_study(self, tmp_path):
        tuner = self._make_tuner(tmp_path)
        result = tuner.get_best_params("nonexistent")
        assert result == {}

    def test_suggest_params_float(self, tmp_path):
        from ml.optuna_tuner import OptunaTuner, _HAS_OPTUNA
        if not _HAS_OPTUNA:
            pytest.skip("optuna not installed")
        import optuna
        study = optuna.create_study()
        trial = study.ask()
        spec = {"lr": {"type": "float", "low": 0.001, "high": 0.1, "log": True}}
        params = OptunaTuner.suggest_params(trial, spec)
        assert 0.001 <= params["lr"] <= 0.1

    def test_suggest_params_int(self, tmp_path):
        from ml.optuna_tuner import OptunaTuner, _HAS_OPTUNA
        if not _HAS_OPTUNA:
            pytest.skip("optuna not installed")
        import optuna
        study = optuna.create_study()
        trial = study.ask()
        spec = {"depth": {"type": "int", "low": 1, "high": 10}}
        params = OptunaTuner.suggest_params(trial, spec)
        assert 1 <= params["depth"] <= 10
        assert isinstance(params["depth"], int)

    def test_suggest_params_categorical(self, tmp_path):
        from ml.optuna_tuner import OptunaTuner, _HAS_OPTUNA
        if not _HAS_OPTUNA:
            pytest.skip("optuna not installed")
        import optuna
        study = optuna.create_study()
        trial = study.ask()
        spec = {"algo": {"type": "categorical", "choices": ["a", "b", "c"]}}
        params = OptunaTuner.suggest_params(trial, spec)
        assert params["algo"] in ("a", "b", "c")

    def test_save_and_load_results(self, tmp_path):
        from ml.optuna_tuner import _HAS_OPTUNA
        if not _HAS_OPTUNA:
            pytest.skip("optuna not installed")
        tuner = self._make_tuner(tmp_path)

        def objective(trial):
            x = trial.suggest_float("x", -5, 5)
            return -(x ** 2)

        tuner.optimize("save_test", objective, n_trials=5, timeout_s=10)
        path = os.path.join(str(tmp_path), "results.json")
        tuner.save_results(path)
        assert os.path.exists(path)

        tuner2 = self._make_tuner(tmp_path)
        loaded = tuner2.load_results(path)
        assert "save_test" in loaded

    def test_load_results_missing_file(self, tmp_path):
        tuner = self._make_tuner(tmp_path)
        result = tuner.load_results("/nonexistent/path.json")
        assert result == {}

    def test_no_optuna_fallback(self, tmp_path, monkeypatch):
        """When _HAS_OPTUNA is False, methods return empty/defaults."""
        import ml.optuna_tuner as mod
        monkeypatch.setattr(mod, "_HAS_OPTUNA", False)
        monkeypatch.setattr(mod, "_OPTUNA_WARN_ONCE", False)
        tuner = mod.OptunaTuner(db_path=os.path.join(str(tmp_path), "x.db"))
        assert tuner.create_study("x") is None
        assert tuner.get_best_params("x") == {}
        assert tuner.optimize("x", lambda t: 0.0) == {}
        assert mod.OptunaTuner.suggest_params(None, {"a": {"type": "float", "low": 0, "high": 1}}) == {}


# ============================================================================
# RegimeStrategyRouter
# ============================================================================

class TestRegimeStrategyRouter:
    """Tests for adaptive.regime_strategy_router.RegimeStrategyRouter."""

    def _make_router(self):
        from adaptive.regime_strategy_router import RegimeStrategyRouter
        return RegimeStrategyRouter()

    def _base_weights(self):
        return {
            "momentum": 1.0,
            "mean_reversion": 1.0,
            "breakout": 1.0,
            "scalping": 1.0,
            "market_maker": 1.0,
            "tail_hedge": 1.0,
            "trend_following": 1.0,
        }

    def test_init(self):
        router = self._make_router()
        assert len(router.supported_regimes) >= 6

    def test_trending_boosts_momentum(self):
        router = self._make_router()
        w = router.get_weights("trending", self._base_weights())
        assert w["momentum"] > 1.0
        assert w["mean_reversion"] < 1.0

    def test_bull_boosts_momentum(self):
        router = self._make_router()
        w = router.get_weights("bull", self._base_weights())
        assert w["momentum"] > 1.0

    def test_mean_revert_boosts_mr(self):
        router = self._make_router()
        w = router.get_weights("mean_revert", self._base_weights())
        assert w["mean_reversion"] > 1.0
        assert w["momentum"] < 1.0

    def test_range_boosts_scalping(self):
        router = self._make_router()
        w = router.get_weights("range", self._base_weights())
        assert w["scalping"] > 1.0

    def test_crisis_boosts_tail_hedge(self):
        router = self._make_router()
        w = router.get_weights("crisis", self._base_weights())
        # tail_hedge is boosted relative to penalised strategies (not necessarily > 1.0
        # because crisis size_scale=0.4 applies globally)
        assert w["tail_hedge"] > w["momentum"]

    def test_crisis_skips_trend_following(self):
        router = self._make_router()
        assert router.should_skip_strategy("trend_following", "crisis")

    def test_crisis_reduces_size(self):
        router = self._make_router()
        w = router.get_weights("crisis", self._base_weights())
        # momentum is penalised AND size_scale is 0.4
        assert w["momentum"] < 0.5

    def test_volatile_halves_sizes(self):
        router = self._make_router()
        w = router.get_weights("volatile", self._base_weights())
        # Non-boosted, non-penalised strategies get size_scale=0.5
        assert w["tail_hedge"] == pytest.approx(0.5, abs=0.01)

    def test_high_vol_same_as_volatile(self):
        router = self._make_router()
        w1 = router.get_weights("volatile", self._base_weights())
        w2 = router.get_weights("high_vol", self._base_weights())
        assert w1 == w2

    def test_unknown_regime_returns_base(self):
        router = self._make_router()
        base = self._base_weights()
        w = router.get_weights("unknown_regime_xyz", base)
        assert w == base

    def test_should_skip_false_for_normal(self):
        router = self._make_router()
        assert not router.should_skip_strategy("momentum", "trending")

    def test_weights_non_negative(self):
        router = self._make_router()
        for regime in router.supported_regimes:
            w = router.get_weights(regime, self._base_weights())
            for v in w.values():
                assert v >= 0.0

    def test_regime_profile_exists(self):
        router = self._make_router()
        profile = router.get_regime_profile("crisis")
        assert profile is not None
        assert "boost" in profile


# ============================================================================
# ExecutionQualityTracker
# ============================================================================

class TestExecutionQualityTracker:
    """Tests for execution.execution_quality_tracker.ExecutionQualityTracker."""

    def _make_tracker(self, tmp_path):
        from execution.execution_quality_tracker import ExecutionQualityTracker
        db = os.path.join(str(tmp_path), "eq_test.db")
        return ExecutionQualityTracker(db_path=db)

    def test_init(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        assert tracker is not None

    def test_record_fill_returns_slippage(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        bps = tracker.record_fill("BTC/USD", "momentum", 14, 65000.0, 65013.0, 500.0)
        # (13/65000)*10000 = 2.0 bps
        assert bps == pytest.approx(2.0, abs=0.1)

    def test_record_fill_negative_slip(self, tmp_path):
        """Fill below expected also records positive slippage (absolute)."""
        tracker = self._make_tracker(tmp_path)
        bps = tracker.record_fill("BTC/USD", "momentum", 14, 65000.0, 64987.0, 500.0)
        assert bps > 0

    def test_avg_slippage(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.record_fill("ETH/USD", "scalp", 10, 3000.0, 3003.0, 200.0)
        tracker.record_fill("ETH/USD", "scalp", 10, 3000.0, 3006.0, 200.0)
        avg = tracker.get_avg_slippage_bps("ETH/USD", "scalp", 10)
        # fills: 10 bps and 20 bps → avg = 15 bps
        assert avg == pytest.approx(15.0, abs=0.5)

    def test_avg_slippage_no_data(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        assert tracker.get_avg_slippage_bps("NOPE/USD") == 0.0

    def test_size_adjustment_good_slippage(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        # Record low-slippage fills
        for _ in range(5):
            tracker.record_fill("BTC/USD", "mom", 12, 65000.0, 65001.0, 500.0)
        adj = tracker.get_size_adjustment("BTC/USD", "mom", 12)
        assert adj == pytest.approx(1.0, abs=0.05)

    def test_size_adjustment_bad_slippage(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        # Record high-slippage fills (~40 bps)
        for _ in range(5):
            tracker.record_fill("BTC/USD", "mom", 12, 65000.0, 65260.0, 500.0)
        adj = tracker.get_size_adjustment("BTC/USD", "mom", 12)
        assert adj == pytest.approx(0.25, abs=0.05)

    def test_size_adjustment_medium_slippage(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        # ~15 bps slippage → between good and bad thresholds
        for _ in range(5):
            tracker.record_fill("BTC/USD", "mom", 12, 65000.0, 65097.5, 500.0)
        adj = tracker.get_size_adjustment("BTC/USD", "mom", 12)
        assert 0.25 < adj < 1.0

    def test_worst_hours(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        # Hour 3: low slip
        tracker.record_fill("BTC/USD", "x", 3, 100.0, 100.01, 10.0)
        # Hour 14: high slip
        tracker.record_fill("BTC/USD", "x", 14, 100.0, 100.10, 10.0)
        # Hour 22: medium slip
        tracker.record_fill("BTC/USD", "x", 22, 100.0, 100.05, 10.0)

        worst = tracker.get_worst_hours("BTC/USD", top_n=2)
        assert len(worst) == 2
        assert worst[0][0] == 14  # hour 14 is worst

    def test_record_fill_zero_price_guard(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        bps = tracker.record_fill("BTC/USD", "x", 0, 0.0, 100.0, 10.0)
        assert bps == 0.0

    def test_filter_by_strategy(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.record_fill("BTC/USD", "alpha", 10, 100.0, 100.05, 10.0)  # 50 bps
        tracker.record_fill("BTC/USD", "beta", 10, 100.0, 100.01, 10.0)   # 10 bps
        avg_alpha = tracker.get_avg_slippage_bps("BTC/USD", strategy="alpha")
        avg_beta = tracker.get_avg_slippage_bps("BTC/USD", strategy="beta")
        assert avg_alpha > avg_beta

    def test_close(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.close()  # should not raise


# ============================================================================
# DynamicCycleTimer
# ============================================================================

class TestDynamicCycleTimer:
    """Tests for adaptive.dynamic_cycle_timer.DynamicCycleTimer."""

    def _make_timer(self, baseline=0.01):
        from adaptive.dynamic_cycle_timer import DynamicCycleTimer
        return DynamicCycleTimer(baseline_vol=baseline)

    def test_init(self):
        timer = self._make_timer()
        assert timer.baseline_vol == pytest.approx(0.01)

    def test_normal_vol_60s(self):
        timer = self._make_timer(baseline=0.01)
        interval = timer.get_cycle_interval(0.01, "normal")
        assert interval == 60

    def test_high_vol_30s(self):
        timer = self._make_timer(baseline=0.01)
        interval = timer.get_cycle_interval(0.025, "trending")  # > 2x baseline
        assert interval == 30

    def test_low_vol_300s(self):
        timer = self._make_timer(baseline=0.01)
        interval = timer.get_cycle_interval(0.003, "normal")  # < 0.5x baseline
        assert interval == 300

    def test_crisis_15s(self):
        timer = self._make_timer(baseline=0.01)
        interval = timer.get_cycle_interval(0.01, "crisis")
        assert interval == 15

    def test_bear_15s(self):
        timer = self._make_timer(baseline=0.01)
        interval = timer.get_cycle_interval(0.01, "bear")
        assert interval == 15

    def test_force_cycle_large_move(self):
        timer = self._make_timer()
        assert timer.should_force_cycle(2.5) is True

    def test_no_force_cycle_small_move(self):
        timer = self._make_timer()
        assert timer.should_force_cycle(0.5) is False

    def test_force_cycle_negative_move(self):
        timer = self._make_timer()
        assert timer.should_force_cycle(-3.0) is True

    def test_force_cycle_exact_threshold(self):
        timer = self._make_timer()
        assert timer.should_force_cycle(2.0) is True

    def test_update_baseline_vol(self):
        timer = self._make_timer(baseline=0.01)
        # With alpha=0.05: new = 0.05*0.05 + 0.95*0.01 = 0.0025 + 0.0095 = 0.012
        new_bl = timer.update_baseline_vol(0.05)
        assert new_bl == pytest.approx(0.012, abs=0.001)

    def test_update_baseline_negative_ignored(self):
        timer = self._make_timer(baseline=0.01)
        old = timer.baseline_vol
        timer.update_baseline_vol(-0.05)
        assert timer.baseline_vol == old

    def test_last_interval_property(self):
        timer = self._make_timer()
        assert timer.last_interval is None
        timer.get_cycle_interval(0.01, "normal")
        assert timer.last_interval == 60


# ============================================================================
# KellyPositionSizer
# ============================================================================

class TestKellyPositionSizer:
    """Tests for risk.kelly_position_sizer.KellyPositionSizer."""

    def _make_sizer(self, tmp_path, min_trades=20):
        from risk.kelly_position_sizer import KellyPositionSizer
        db = os.path.join(str(tmp_path), "kelly_test.db")
        return KellyPositionSizer(db_path=db, min_trades=min_trades)

    def test_init(self, tmp_path):
        sizer = self._make_sizer(tmp_path)
        assert sizer is not None

    def test_conservative_default_few_trades(self, tmp_path):
        sizer = self._make_sizer(tmp_path)
        for i in range(5):
            sizer.update_outcome("test", won=True, return_pct=1.0)
        frac = sizer.get_kelly_fraction("test")
        assert frac == pytest.approx(0.01)  # conservative default

    def test_kelly_kicks_in_after_min_trades(self, tmp_path):
        sizer = self._make_sizer(tmp_path, min_trades=5)
        # 4 wins, 2 losses
        for _ in range(4):
            sizer.update_outcome("strat", won=True, return_pct=2.0)
        for _ in range(2):
            sizer.update_outcome("strat", won=False, return_pct=-1.0)
        frac = sizer.get_kelly_fraction("strat")
        # Should be > conservative default since we have positive edge
        assert frac > 0.01

    def test_half_kelly_is_half(self, tmp_path):
        sizer = self._make_sizer(tmp_path, min_trades=5)
        for _ in range(4):
            sizer.update_outcome("hk", won=True, return_pct=2.0)
        for _ in range(2):
            sizer.update_outcome("hk", won=False, return_pct=-1.0)
        full = sizer.get_kelly_fraction("hk")
        half = sizer.get_half_kelly("hk")
        assert half == pytest.approx(full / 2.0)

    def test_negative_edge_returns_zero(self, tmp_path):
        sizer = self._make_sizer(tmp_path, min_trades=5)
        # Many losses, few wins with small returns
        for _ in range(5):
            sizer.update_outcome("loser", won=False, return_pct=-5.0)
        for _ in range(1):
            sizer.update_outcome("loser", won=True, return_pct=0.1)
        frac = sizer.get_kelly_fraction("loser")
        assert frac == 0.0

    def test_position_size_usd(self, tmp_path):
        sizer = self._make_sizer(tmp_path)
        size = sizer.get_position_size_usd("new_strat", capital_usd=10000.0)
        # With conservative default (0.01/2 = 0.005), size = $50
        assert size == pytest.approx(50.0, abs=1.0)

    def test_position_size_capped_at_max_pct(self, tmp_path):
        sizer = self._make_sizer(tmp_path, min_trades=3)
        # Create very high edge strategy
        for _ in range(5):
            sizer.update_outcome("killer", won=True, return_pct=10.0)
        for _ in range(1):
            sizer.update_outcome("killer", won=False, return_pct=-0.5)
        size = sizer.get_position_size_usd("killer", capital_usd=10000.0, max_pct=0.02)
        # Half-kelly might be large, but capped at 2% of 10k = $200
        assert size <= 200.0

    def test_trade_count(self, tmp_path):
        sizer = self._make_sizer(tmp_path)
        sizer.update_outcome("x", won=True, return_pct=1.0)
        sizer.update_outcome("x", won=False, return_pct=-0.5)
        sizer.update_outcome("y", won=True, return_pct=2.0)
        assert sizer.get_trade_count("x") == 2
        assert sizer.get_trade_count("y") == 1

    def test_get_all_strategies(self, tmp_path):
        sizer = self._make_sizer(tmp_path)
        sizer.update_outcome("alpha", won=True, return_pct=1.0)
        sizer.update_outcome("beta", won=True, return_pct=2.0)
        strats = sizer.get_all_strategies()
        assert "alpha" in strats
        assert "beta" in strats

    def test_kelly_capped_at_max(self, tmp_path):
        sizer = self._make_sizer(tmp_path, min_trades=3)
        # Extreme win rate and ratio
        for _ in range(10):
            sizer.update_outcome("extreme", won=True, return_pct=50.0)
        for _ in range(1):
            sizer.update_outcome("extreme", won=False, return_pct=-0.1)
        frac = sizer.get_kelly_fraction("extreme")
        assert frac <= 0.25  # max_kelly cap

    def test_close(self, tmp_path):
        sizer = self._make_sizer(tmp_path)
        sizer.close()  # should not raise

    def test_no_data_returns_conservative(self, tmp_path):
        sizer = self._make_sizer(tmp_path)
        frac = sizer.get_kelly_fraction("empty")
        assert frac == pytest.approx(0.01)
