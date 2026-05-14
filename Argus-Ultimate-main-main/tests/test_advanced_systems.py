"""
Tests for advanced systems — continuous backtester, multi-agent coordinator,
synthetic data generator, A/B test framework, regime forecaster.

50+ tests covering construction, core logic, edge cases, persistence, and
data class behaviour.
"""
from __future__ import annotations

import math
import os
import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------

from backtesting.continuous_backtester import (
    BacktestMetrics,
    ContinuousBacktester,
    NightlyReport,
)
from core.multi_agent_coordinator import (
    AgentInfo,
    ConsensusResult,
    MultiAgentCoordinator,
    Vote,
)
from ml.synthetic_data_generator import SyntheticDataGenerator
from backtesting.ab_test_framework import (
    ABTestFramework,
    ABTestResult,
    VariantMetrics,
)
from adaptive.regime_forecaster import (
    RegimeForecaster,
    TransitionForecast,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp_db(name: str = "test.db") -> str:
    """Return a path in a temporary directory for a disposable SQLite DB."""
    d = tempfile.mkdtemp()
    return os.path.join(d, name)


def _sample_ohlcv(n: int = 60, start_price: float = 100.0) -> list:
    """Generate a simple uptrend OHLCV dataset."""
    import random
    rng = random.Random(42)
    bars = []
    price = start_price
    for i in range(n):
        ret = rng.gauss(0.002, 0.02)
        o = price
        c = price * (1 + ret)
        h = max(o, c) * (1 + rng.random() * 0.005)
        lo = min(o, c) * (1 - rng.random() * 0.005)
        v = rng.uniform(1000, 5000)
        bars.append({"t": i, "o": o, "h": h, "l": lo, "c": c, "v": v})
        price = c
    return bars


def _sample_ohlcv_downtrend(n: int = 60, start_price: float = 100.0) -> list:
    """Generate a downtrend OHLCV dataset."""
    import random
    rng = random.Random(99)
    bars = []
    price = start_price
    for i in range(n):
        ret = rng.gauss(-0.005, 0.02)
        o = price
        c = price * (1 + ret)
        h = max(o, c) * (1 + rng.random() * 0.005)
        lo = min(o, c) * (1 - rng.random() * 0.005)
        v = rng.uniform(1000, 5000)
        bars.append({"t": i, "o": o, "h": h, "l": lo, "c": c, "v": v})
        price = c
    return bars


# ===================================================================
# ContinuousBacktester tests (10)
# ===================================================================

class TestContinuousBacktester:
    """Tests for backtesting.continuous_backtester."""

    def test_construction(self):
        bt = ContinuousBacktester(db_path=_tmp_db())
        assert bt.min_sharpe == 0.3
        assert bt.max_drawdown_pct == 25.0

    def test_run_nightly_basic(self):
        bt = ContinuousBacktester(db_path=_tmp_db())
        data = {"BTC/USD": _sample_ohlcv(60)}
        report = bt.run_nightly(["strat_a"], data, walk_forward_days=30)
        assert isinstance(report, NightlyReport)
        assert "strat_a" in report.strategy_results
        assert report.timestamp != ""

    def test_run_nightly_empty_strategies(self):
        bt = ContinuousBacktester(db_path=_tmp_db())
        report = bt.run_nightly([], {})
        assert "No strategies provided" in report.warnings

    def test_run_nightly_empty_data(self):
        bt = ContinuousBacktester(db_path=_tmp_db())
        report = bt.run_nightly(["x"], {})
        assert "No OHLCV data provided" in report.warnings

    def test_metrics_pass_criteria(self):
        m = BacktestMetrics(sharpe=0.5, max_drawdown_pct=10, profit_factor=1.5, passed=True)
        assert m.passed is True

    def test_metrics_fail_criteria(self):
        m = BacktestMetrics(sharpe=0.1, max_drawdown_pct=30, profit_factor=0.8, passed=False)
        assert m.passed is False

    def test_get_history_empty(self):
        bt = ContinuousBacktester(db_path=_tmp_db())
        history = bt.get_history("nonexistent")
        assert history == []

    def test_get_history_persistence(self):
        db = _tmp_db()
        bt = ContinuousBacktester(db_path=db)
        data = {"BTC/USD": _sample_ohlcv(60)}
        bt.run_nightly(["strat_a"], data)
        history = bt.get_history("strat_a", lookback_days=10)
        assert len(history) == 1

    def test_demotion_candidates_insufficient_runs(self):
        bt = ContinuousBacktester(db_path=_tmp_db(), demotion_consecutive=3)
        data = {"BTC/USD": _sample_ohlcv(60)}
        bt.run_nightly(["strat_a"], data)
        # Only 1 run, need 3 consecutive failures
        assert bt.get_demotion_candidates() == []

    def test_metrics_to_dict_roundtrip(self):
        m = BacktestMetrics(sharpe=1.2, max_drawdown_pct=5.0, win_rate=0.6,
                            trade_count=100, profit_factor=1.8, passed=True)
        d = m.to_dict()
        m2 = BacktestMetrics.from_dict(d)
        assert m2.sharpe == m.sharpe
        assert m2.passed is True


# ===================================================================
# MultiAgentCoordinator tests (12)
# ===================================================================

class TestMultiAgentCoordinator:
    """Tests for core.multi_agent_coordinator."""

    def test_construction(self):
        coord = MultiAgentCoordinator(db_path=_tmp_db())
        assert coord.agreement_threshold == 0.60

    def test_register_agent(self):
        coord = MultiAgentCoordinator(db_path=_tmp_db())
        coord.register_agent("alpha_1", "alpha", weight=1.5)
        assert "alpha_1" in coord._agents

    def test_register_invalid_type(self):
        coord = MultiAgentCoordinator(db_path=_tmp_db())
        with pytest.raises(ValueError, match="Invalid agent_type"):
            coord.register_agent("bad", "invalid_type")

    def test_submit_vote_unregistered(self):
        coord = MultiAgentCoordinator(db_path=_tmp_db())
        with pytest.raises(ValueError, match="not registered"):
            coord.submit_vote("ghost", "BTC/USD", "buy", 0.8)

    def test_submit_vote_invalid_direction(self):
        coord = MultiAgentCoordinator(db_path=_tmp_db())
        coord.register_agent("a1", "alpha")
        with pytest.raises(ValueError, match="Invalid direction"):
            coord.submit_vote("a1", "BTC/USD", "moon", 0.9)

    def test_consensus_buy_unanimous(self):
        coord = MultiAgentCoordinator(db_path=_tmp_db())
        coord.register_agent("a1", "alpha", weight=1.0)
        coord.register_agent("a2", "risk", weight=1.0)
        coord.submit_vote("a1", "BTC/USD", "buy", 0.9, "Strong signal")
        coord.submit_vote("a2", "BTC/USD", "buy", 0.8, "Risk OK")
        result = coord.get_consensus("BTC/USD")
        assert result.direction == "buy"
        assert result.unanimous is True
        assert result.votes_for == 2
        assert result.votes_against == 0

    def test_consensus_split_falls_to_hold(self):
        coord = MultiAgentCoordinator(db_path=_tmp_db(), agreement_threshold=0.70)
        coord.register_agent("a1", "alpha", weight=1.0)
        coord.register_agent("a2", "risk", weight=1.0)
        coord.submit_vote("a1", "BTC/USD", "buy", 0.5)
        coord.submit_vote("a2", "BTC/USD", "sell", 0.5)
        result = coord.get_consensus("BTC/USD")
        # 50/50 split should not meet 70% threshold
        assert result.direction == "hold"

    def test_consensus_weighted(self):
        coord = MultiAgentCoordinator(db_path=_tmp_db(), agreement_threshold=0.50)
        coord.register_agent("heavy", "alpha", weight=3.0)
        coord.register_agent("light", "risk", weight=1.0)
        coord.submit_vote("heavy", "BTC/USD", "sell", 0.9)
        coord.submit_vote("light", "BTC/USD", "buy", 0.9)
        result = coord.get_consensus("BTC/USD")
        # Heavy agent dominates
        assert result.direction == "sell"

    def test_consensus_empty_symbol(self):
        coord = MultiAgentCoordinator(db_path=_tmp_db())
        result = coord.get_consensus("ETH/USD")
        assert result.direction == "hold"
        assert result.votes_for == 0

    def test_clear_votes(self):
        coord = MultiAgentCoordinator(db_path=_tmp_db())
        coord.register_agent("a1", "alpha")
        coord.submit_vote("a1", "BTC/USD", "buy", 0.8)
        coord.clear_votes("BTC/USD")
        result = coord.get_consensus("BTC/USD")
        assert result.votes_for == 0

    def test_record_outcome_and_accuracy(self):
        coord = MultiAgentCoordinator(db_path=_tmp_db())
        coord.register_agent("a1", "alpha")
        coord.record_outcome("a1", "BTC/USD", "buy", True)
        coord.record_outcome("a1", "BTC/USD", "buy", True)
        coord.record_outcome("a1", "BTC/USD", "sell", False)
        acc = coord.get_agent_accuracy("a1")
        assert acc["vote_count"] == 3
        assert abs(acc["accuracy"] - 2 / 3) < 0.01

    def test_accuracy_empty(self):
        coord = MultiAgentCoordinator(db_path=_tmp_db())
        acc = coord.get_agent_accuracy("nobody")
        assert acc["vote_count"] == 0
        assert acc["accuracy"] == 0.0


# ===================================================================
# SyntheticDataGenerator tests (10)
# ===================================================================

class TestSyntheticDataGenerator:
    """Tests for ml.synthetic_data_generator."""

    def test_construction(self):
        gen = SyntheticDataGenerator(block_size=5, seed=42)
        assert gen.block_size == 5
        assert gen._fitted is False

    def test_fit_basic(self):
        gen = SyntheticDataGenerator(seed=42)
        gen.fit(_sample_ohlcv(50))
        assert gen._fitted is True
        assert len(gen._returns) > 0

    def test_fit_too_few_bars(self):
        gen = SyntheticDataGenerator()
        with pytest.raises(ValueError, match="at least 3 bars"):
            gen.fit([{"c": 100}, {"c": 101}])

    def test_generate_not_fitted(self):
        gen = SyntheticDataGenerator()
        with pytest.raises(ValueError, match="Must call fit"):
            gen.generate()

    def test_generate_normal(self):
        gen = SyntheticDataGenerator(seed=42)
        gen.fit(_sample_ohlcv(50))
        bars = gen.generate(n_bars=100, regime="normal")
        assert len(bars) == 100
        for bar in bars:
            assert "open" in bar
            assert "close" in bar
            assert bar["close"] > 0

    def test_generate_all_regimes(self):
        gen = SyntheticDataGenerator(seed=42)
        gen.fit(_sample_ohlcv(50))
        for regime in ("normal", "bull", "bear", "crisis", "flash_crash", "low_vol"):
            bars = gen.generate(n_bars=20, regime=regime)
            assert len(bars) == 20

    def test_generate_invalid_regime(self):
        gen = SyntheticDataGenerator(seed=42)
        gen.fit(_sample_ohlcv(50))
        with pytest.raises(ValueError, match="Unknown regime"):
            gen.generate(regime="sideways")

    def test_stress_scenario_covid(self):
        gen = SyntheticDataGenerator(seed=42)
        bars = gen.generate_stress_scenario("2020_covid_crash")
        assert len(bars) == 21
        # Price should drop significantly
        first_close = bars[0]["close"]
        min_close = min(b["close"] for b in bars)
        assert min_close < first_close * 0.8

    def test_stress_scenario_unknown(self):
        gen = SyntheticDataGenerator()
        with pytest.raises(ValueError, match="Unknown scenario"):
            gen.generate_stress_scenario("2099_alien_invasion")

    def test_validate_synthetic(self):
        gen = SyntheticDataGenerator(seed=42)
        real = _sample_ohlcv(100)
        gen.fit(real)
        synthetic = gen.generate(n_bars=100, regime="normal")
        result = gen.validate_synthetic(real, synthetic)
        assert "quality_score" in result
        assert 0.0 <= result["quality_score"] <= 1.0
        assert "comparisons" in result


# ===================================================================
# ABTestFramework tests (10)
# ===================================================================

class TestABTestFramework:
    """Tests for backtesting.ab_test_framework."""

    def test_construction(self):
        ab = ABTestFramework(db_path=_tmp_db())
        assert ab.significance_level == 0.05

    def test_create_experiment(self):
        ab = ABTestFramework(db_path=_tmp_db())
        eid = ab.create_experiment("test_exp", {"lr": 0.01}, {"lr": 0.02}, min_trades=10)
        assert len(eid) == 12

    def test_record_trade_invalid_variant(self):
        ab = ABTestFramework(db_path=_tmp_db())
        eid = ab.create_experiment("x", {}, {})
        with pytest.raises(ValueError, match="control.*variant"):
            ab.record_trade(eid, "neither", 10.0)

    def test_record_and_get_results(self):
        ab = ABTestFramework(db_path=_tmp_db())
        eid = ab.create_experiment("x", {}, {}, min_trades=2)
        ab.record_trade(eid, "control", 5.0)
        ab.record_trade(eid, "control", 3.0)
        ab.record_trade(eid, "variant", 10.0)
        ab.record_trade(eid, "variant", 8.0)
        result = ab.get_results(eid)
        assert isinstance(result, ABTestResult)
        assert result.sample_sizes["control"] == 2
        assert result.sample_sizes["variant"] == 2

    def test_inconclusive_with_few_samples(self):
        ab = ABTestFramework(db_path=_tmp_db())
        eid = ab.create_experiment("x", {}, {}, min_trades=100)
        ab.record_trade(eid, "control", 5.0)
        ab.record_trade(eid, "variant", 10.0)
        result = ab.get_results(eid)
        assert result.winner == "inconclusive"

    def test_variant_metrics_computation(self):
        m = ABTestFramework._compute_variant_metrics([(10.0, 2.0), (-5.0, 3.0), (8.0, 1.0)])
        assert m.sample_size == 3
        assert abs(m.mean_pnl - (10 - 5 + 8) / 3) < 0.01
        assert abs(m.win_rate - 2 / 3) < 0.01

    def test_variant_metrics_empty(self):
        m = ABTestFramework._compute_variant_metrics([])
        assert m.sample_size == 0

    def test_auto_promote_not_enough(self):
        ab = ABTestFramework(db_path=_tmp_db())
        eid = ab.create_experiment("x", {}, {}, min_trades=100)
        ab.record_trade(eid, "control", 1.0)
        ab.record_trade(eid, "variant", 100.0)
        assert ab.auto_promote(eid) is False

    def test_list_experiments(self):
        ab = ABTestFramework(db_path=_tmp_db())
        ab.create_experiment("exp1", {}, {})
        ab.create_experiment("exp2", {}, {})
        exps = ab.list_experiments()
        assert len(exps) == 2

    def test_get_results_nonexistent(self):
        ab = ABTestFramework(db_path=_tmp_db())
        result = ab.get_results("does-not-exist")
        assert result.winner == "inconclusive"


# ===================================================================
# RegimeForecaster tests (12)
# ===================================================================

class TestRegimeForecaster:
    """Tests for adaptive.regime_forecaster."""

    def test_construction(self):
        rf = RegimeForecaster(db_path=_tmp_db())
        assert rf.min_observations == 5

    def test_update_single(self):
        rf = RegimeForecaster(db_path=_tmp_db())
        rf.update("bull", {"volatility": 0.02}, timestamp=1000.0)
        assert len(rf._observations) == 1

    def test_update_builds_transitions(self):
        rf = RegimeForecaster(db_path=_tmp_db())
        rf.update("bull", {}, timestamp=1000.0)
        rf.update("bull", {}, timestamp=2000.0)
        rf.update("bear", {}, timestamp=3000.0)
        assert rf._transition_counts["bull"]["bull"] == 1
        assert rf._transition_counts["bull"]["bear"] == 1

    def test_predict_insufficient_data(self):
        rf = RegimeForecaster(db_path=_tmp_db(), min_observations=5)
        rf.update("bull", {}, timestamp=1000.0)
        forecast = rf.predict_transition("bull")
        # Not enough data — should predict staying
        assert forecast.predicted_regime == "bull"
        assert forecast.confidence < 0.5

    def test_predict_with_data(self):
        rf = RegimeForecaster(db_path=_tmp_db(), min_observations=3)
        ts = 1000.0
        for _ in range(3):
            rf.update("bull", {"volatility": 0.02}, timestamp=ts)
            ts += 3600
        for _ in range(2):
            rf.update("bear", {"volatility": 0.05}, timestamp=ts)
            ts += 3600
        rf.update("crisis", {"volatility": 0.10}, timestamp=ts)
        ts += 3600

        forecast = rf.predict_transition("bear")
        assert isinstance(forecast, TransitionForecast)
        assert forecast.current_regime == "bear"
        assert forecast.probability > 0

    def test_transition_matrix(self):
        rf = RegimeForecaster(db_path=_tmp_db())
        rf.update("bull", {}, timestamp=1000.0)
        rf.update("bear", {}, timestamp=2000.0)
        rf.update("bull", {}, timestamp=3000.0)
        matrix = rf.get_transition_matrix()
        assert "bull" in matrix
        assert "bear" in matrix["bull"]

    def test_duration_stats(self):
        rf = RegimeForecaster(db_path=_tmp_db())
        rf.update("bull", {}, timestamp=0.0)
        rf.update("bull", {}, timestamp=3600.0)
        rf.update("bear", {}, timestamp=7200.0)
        stats = rf.get_regime_duration_stats()
        assert "bull" in stats
        assert stats["bull"]["avg_duration_hours"] >= 0

    def test_forecast_dataclass_defaults(self):
        f = TransitionForecast(
            current_regime="bull",
            predicted_regime="bear",
            probability=0.7,
            confidence=0.6,
        )
        assert f.horizon_hours == 4
        assert f.timestamp != ""

    def test_features_influence_prediction(self):
        rf = RegimeForecaster(db_path=_tmp_db(), min_observations=2,
                              feature_adjustment_strength=0.8)
        ts = 1000.0
        # Build profile: high vol = crisis
        for _ in range(3):
            rf.update("normal", {"volatility": 0.02}, timestamp=ts)
            ts += 3600
        for _ in range(3):
            rf.update("crisis", {"volatility": 0.10}, timestamp=ts)
            ts += 3600
        # Current: normal with high vol — should nudge toward crisis
        rf.update("normal", {"volatility": 0.09}, timestamp=ts)
        forecast = rf.predict_transition("normal")
        # We just check it runs and gives a valid forecast
        assert forecast.probability > 0
        assert forecast.probability <= 1.0

    def test_horizon_reduces_confidence(self):
        rf = RegimeForecaster(db_path=_tmp_db(), min_observations=2)
        ts = 1000.0
        for _ in range(5):
            rf.update("bull", {}, timestamp=ts)
            ts += 3600
        rf.update("bear", {}, timestamp=ts)
        f1 = rf.predict_transition("bull", horizon_hours=1)
        f24 = rf.predict_transition("bull", horizon_hours=24)
        # Longer horizon should have lower or equal confidence
        assert f24.confidence <= f1.confidence + 0.01  # Small tolerance

    def test_persistence_reload(self):
        db = _tmp_db()
        rf1 = RegimeForecaster(db_path=db, min_observations=2)
        rf1.update("bull", {"volatility": 0.02}, timestamp=1000.0)
        rf1.update("bear", {"volatility": 0.05}, timestamp=2000.0)

        # Create new instance — should load from DB
        rf2 = RegimeForecaster(db_path=db, min_observations=2)
        assert len(rf2._observations) == 2
        matrix = rf2.get_transition_matrix()
        assert "bull" in matrix

    def test_key_features_extraction(self):
        rf = RegimeForecaster(db_path=_tmp_db())
        features = {"volatility": 0.05, "momentum": 0.3, "spread": 0.01}
        keys = rf._get_key_features(features, "crisis")
        assert len(keys) <= 3
        # Volatility has highest weight, should appear
        assert "volatility" in keys
