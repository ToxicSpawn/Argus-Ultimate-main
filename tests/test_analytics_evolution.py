"""
Tests for analytics and autonomous evolution modules (Batch — Analytics & Evolution).

Modules covered:
    - monitoring.attribution_engine (AttributionEngine)
    - monitoring.strategy_similarity_detector (StrategySimilarityDetector)
    - monitoring.drawdown_dna (DrawdownDNA)
    - monitoring.streak_analyzer (StreakAnalyzer)
    - evolution.neuroevolution (Neuroevolution)
    - evolution.strategy_breeder (StrategyBreeder)
    - adaptive.self_debugger (SelfDebugger)

Target: 70+ tests.
"""

from __future__ import annotations

import math
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures — each test gets its own temp DB to avoid cross-test interference
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """Return a function that creates unique temp DB paths."""
    counter = [0]
    def _make(name="test"):
        counter[0] += 1
        return str(tmp_path / f"{name}_{counter[0]}.db")
    return _make


# ===========================================================================
# AttributionEngine
# ===========================================================================

class TestAttributionEngine:

    def _make(self, tmp_db):
        from monitoring.attribution_engine import AttributionEngine
        return AttributionEngine(db_path=tmp_db("attr"))

    def test_record_and_decompose_empty(self, tmp_db):
        engine = self._make(tmp_db)
        attr = engine.decompose(lookback_days=30)
        assert attr.total_pnl == 0.0
        assert attr.alpha_pnl == 0.0

    def test_record_single_trade(self, tmp_db):
        engine = self._make(tmp_db)
        engine.record_trade("BTC/AUD", "momentum", 60000, 61200, 0.1,
                            market_return_pct=1.5, slippage_bps=3.0, fees_usd=1.2)
        attr = engine.decompose(lookback_days=30)
        assert attr.total_pnl == pytest.approx(120.0, abs=0.01)
        assert attr.fee_cost == pytest.approx(1.2, abs=0.01)
        assert attr.slippage_cost > 0

    def test_multiple_trades_decompose(self, tmp_db):
        engine = self._make(tmp_db)
        for i in range(5):
            engine.record_trade("BTC/AUD", "momentum", 60000 + i * 100,
                                60100 + i * 100, 0.1,
                                market_return_pct=0.5, slippage_bps=2.0, fees_usd=0.5)
        attr = engine.decompose(lookback_days=30)
        assert attr.total_pnl > 0
        assert attr.execution_cost == attr.slippage_cost + attr.fee_cost

    def test_strategy_attribution(self, tmp_db):
        engine = self._make(tmp_db)
        engine.record_trade("BTC/AUD", "alpha_strat", 60000, 61000, 0.1,
                            market_return_pct=0.5, slippage_bps=1.0, fees_usd=0.5)
        engine.record_trade("ETH/AUD", "beta_strat", 3000, 2900, 1.0,
                            market_return_pct=-2.0, slippage_bps=5.0, fees_usd=1.0)

        attr_a = engine.get_strategy_attribution("alpha_strat")
        attr_b = engine.get_strategy_attribution("beta_strat")
        assert attr_a.total_pnl > 0
        assert attr_b.total_pnl < 0

    def test_get_strategy_attribution_empty(self, tmp_db):
        engine = self._make(tmp_db)
        attr = engine.get_strategy_attribution("nonexistent")
        assert attr.total_pnl == 0.0

    def test_best_alpha_source(self, tmp_db):
        engine = self._make(tmp_db)
        engine.record_trade("BTC/AUD", "good_alpha", 60000, 62000, 0.1,
                            market_return_pct=0.1, slippage_bps=1.0, fees_usd=0.5)
        engine.record_trade("BTC/AUD", "bad_alpha", 60000, 59000, 0.1,
                            market_return_pct=0.1, slippage_bps=1.0, fees_usd=0.5)
        assert engine.get_best_alpha_source() == "good_alpha"

    def test_best_alpha_source_empty(self, tmp_db):
        engine = self._make(tmp_db)
        assert engine.get_best_alpha_source() == ""

    def test_lookback_window(self, tmp_db):
        engine = self._make(tmp_db)
        engine.record_trade("BTC/AUD", "strat", 60000, 61000, 0.1)
        # lookback=0 should return nothing (cutoff is now)
        attr = engine.decompose(lookback_days=0)
        assert attr.total_pnl == 0.0

    def test_attribution_timestamp(self, tmp_db):
        engine = self._make(tmp_db)
        attr = engine.decompose()
        assert attr.timestamp is not None


# ===========================================================================
# StrategySimilarityDetector
# ===========================================================================

class TestStrategySimilarityDetector:

    def _make(self, tmp_db):
        from monitoring.strategy_similarity_detector import StrategySimilarityDetector
        return StrategySimilarityDetector(db_path=tmp_db("sim"))

    def test_identical_signals(self, tmp_db):
        det = self._make(tmp_db)
        now = datetime.now(timezone.utc)
        for sym in ["BTC/AUD", "ETH/AUD"]:
            det.record_signal("strat_a", sym, "long", now)
            det.record_signal("strat_b", sym, "long", now)
        sim = det.compute_similarity("strat_a", "strat_b", lookback_days=7)
        assert sim == pytest.approx(1.0)

    def test_disjoint_signals(self, tmp_db):
        det = self._make(tmp_db)
        now = datetime.now(timezone.utc)
        det.record_signal("strat_a", "BTC/AUD", "long", now)
        det.record_signal("strat_b", "ETH/AUD", "short", now)
        sim = det.compute_similarity("strat_a", "strat_b", lookback_days=7)
        assert sim == pytest.approx(0.0)

    def test_partial_overlap(self, tmp_db):
        det = self._make(tmp_db)
        now = datetime.now(timezone.utc)
        det.record_signal("a", "BTC/AUD", "long", now)
        det.record_signal("a", "ETH/AUD", "long", now)
        det.record_signal("b", "BTC/AUD", "long", now)
        det.record_signal("b", "SOL/AUD", "long", now)
        sim = det.compute_similarity("a", "b", lookback_days=7)
        # Jaccard: 1 intersection / 3 union = 0.333...
        assert 0.3 < sim < 0.4

    def test_empty_similarity(self, tmp_db):
        det = self._make(tmp_db)
        sim = det.compute_similarity("x", "y", lookback_days=7)
        assert sim == 0.0

    def test_similarity_matrix(self, tmp_db):
        det = self._make(tmp_db)
        now = datetime.now(timezone.utc)
        det.record_signal("a", "BTC/AUD", "long", now)
        det.record_signal("b", "BTC/AUD", "long", now)
        det.record_signal("c", "ETH/AUD", "short", now)
        matrix = det.get_similarity_matrix(["a", "b", "c"])
        assert matrix["a"]["a"] == 1.0
        assert matrix["a"]["b"] == matrix["b"]["a"]  # symmetric
        assert matrix["a"]["c"] == 0.0

    def test_find_redundant_pairs(self, tmp_db):
        det = self._make(tmp_db)
        now = datetime.now(timezone.utc)
        # Make a and b identical
        det.record_signal("a", "BTC/AUD", "long", now)
        det.record_signal("b", "BTC/AUD", "long", now)
        det.record_signal("c", "ETH/AUD", "short", now)
        pairs = det.find_redundant_pairs(threshold=0.8)
        assert len(pairs) >= 1
        assert pairs[0][2] >= 0.8

    def test_find_redundant_pairs_none(self, tmp_db):
        det = self._make(tmp_db)
        now = datetime.now(timezone.utc)
        det.record_signal("a", "BTC/AUD", "long", now)
        det.record_signal("b", "ETH/AUD", "short", now)
        pairs = det.find_redundant_pairs(threshold=0.8)
        assert len(pairs) == 0

    def test_diversification_score_identical(self, tmp_db):
        det = self._make(tmp_db)
        now = datetime.now(timezone.utc)
        det.record_signal("a", "BTC/AUD", "long", now)
        det.record_signal("b", "BTC/AUD", "long", now)
        score = det.get_diversification_score(["a", "b"])
        assert score == pytest.approx(0.0)  # identical → no diversification

    def test_diversification_score_single(self, tmp_db):
        det = self._make(tmp_db)
        score = det.get_diversification_score(["a"])
        assert score == 1.0  # single strategy → perfect diversification


# ===========================================================================
# DrawdownDNA
# ===========================================================================

class TestDrawdownDNA:

    def _make(self, tmp_db):
        from monitoring.drawdown_dna import DrawdownDNA
        return DrawdownDNA(db_path=tmp_db("dd"))

    def test_record_and_classify_normal(self, tmp_db):
        dna = self._make(tmp_db)
        now = datetime.now(timezone.utc)
        dd_id = dna.record_drawdown(
            start_equity=10000, trough_equity=9800, end_equity=9900,
            start_time=now - timedelta(hours=2), end_time=now,
            trades_during=[{"symbol": "BTC/AUD", "pnl": -100, "slippage_bps": 2}],
        )
        assert dd_id > 0
        cl = dna.classify(dd_id)
        assert cl.drawdown_id == dd_id
        assert cl.cause in ("normal_variance", "regime_change", "model_drift",
                            "execution_failure", "black_swan", "strategy_decay")

    def test_classify_black_swan(self, tmp_db):
        dna = self._make(tmp_db)
        now = datetime.now(timezone.utc)
        dd_id = dna.record_drawdown(
            start_equity=10000, trough_equity=8500, end_equity=8600,
            start_time=now - timedelta(hours=1), end_time=now,
        )
        cl = dna.classify(dd_id)
        assert cl.cause == "black_swan"

    def test_classify_execution_failure(self, tmp_db):
        dna = self._make(tmp_db)
        now = datetime.now(timezone.utc)
        trades = [{"slippage_bps": 50}, {"slippage_bps": 40}, {"slippage_bps": 30}]
        dd_id = dna.record_drawdown(
            start_equity=10000, trough_equity=9500, end_equity=9600,
            start_time=now - timedelta(hours=12), end_time=now,
            trades_during=trades,
        )
        cl = dna.classify(dd_id)
        assert cl.cause == "execution_failure"

    def test_classify_strategy_decay(self, tmp_db):
        dna = self._make(tmp_db)
        now = datetime.now(timezone.utc)
        dd_id = dna.record_drawdown(
            start_equity=10000, trough_equity=9600, end_equity=9650,
            start_time=now - timedelta(days=10), end_time=now,
            trades_during=[{"pnl": -10, "slippage_bps": 1}] * 5,
        )
        cl = dna.classify(dd_id)
        assert cl.cause == "strategy_decay"

    def test_classify_model_drift(self, tmp_db):
        dna = self._make(tmp_db)
        now = datetime.now(timezone.utc)
        dd_id = dna.record_drawdown(
            start_equity=10000, trough_equity=9200, end_equity=9300,
            start_time=now - timedelta(hours=48), end_time=now,
            trades_during=[{"pnl": -50, "slippage_bps": 2}] * 15,
        )
        cl = dna.classify(dd_id)
        assert cl.cause == "model_drift"

    def test_classify_regime_change(self, tmp_db):
        dna = self._make(tmp_db)
        now = datetime.now(timezone.utc)
        dd_id = dna.record_drawdown(
            start_equity=10000, trough_equity=9500, end_equity=9600,
            start_time=now - timedelta(hours=6), end_time=now,
            trades_during=[{"pnl": -200}],
        )
        cl = dna.classify(dd_id)
        assert cl.cause == "regime_change"

    def test_classify_nonexistent(self, tmp_db):
        dna = self._make(tmp_db)
        cl = dna.classify(999)
        assert cl.cause == "unknown"

    def test_get_drawdown_history(self, tmp_db):
        dna = self._make(tmp_db)
        now = datetime.now(timezone.utc)
        for i in range(3):
            dna.record_drawdown(
                start_equity=10000, trough_equity=9800 - i * 100,
                end_equity=9900,
                start_time=now - timedelta(days=i + 1), end_time=now - timedelta(days=i),
            )
        history = dna.get_drawdown_history(lookback_days=90)
        assert len(history) == 3

    def test_get_common_causes(self, tmp_db):
        dna = self._make(tmp_db)
        now = datetime.now(timezone.utc)
        # Create and classify some drawdowns
        for i in range(3):
            dd_id = dna.record_drawdown(
                start_equity=10000, trough_equity=9800, end_equity=9900,
                start_time=now - timedelta(hours=2), end_time=now,
            )
            dna.classify(dd_id)
        causes = dna.get_common_causes()
        assert isinstance(causes, dict)
        total = sum(causes.values())
        assert total == 3


# ===========================================================================
# StreakAnalyzer
# ===========================================================================

class TestStreakAnalyzer:

    def _make(self, tmp_db):
        from monitoring.streak_analyzer import StreakAnalyzer
        return StreakAnalyzer(db_path=tmp_db("streak"))

    def test_empty_streak(self, tmp_db):
        sa = self._make(tmp_db)
        streak = sa.get_current_streak("test")
        assert streak.length == 0

    def test_winning_streak(self, tmp_db):
        sa = self._make(tmp_db)
        for _ in range(5):
            sa.record_outcome("test", won=True, pnl=10.0)
        streak = sa.get_current_streak("test")
        assert streak.type == "win"
        assert streak.length == 5
        assert streak.total_pnl == pytest.approx(50.0)

    def test_losing_streak(self, tmp_db):
        sa = self._make(tmp_db)
        for _ in range(3):
            sa.record_outcome("test", won=False, pnl=-5.0)
        streak = sa.get_current_streak("test")
        assert streak.type == "loss"
        assert streak.length == 3

    def test_streak_breaks(self, tmp_db):
        sa = self._make(tmp_db)
        sa.record_outcome("test", won=True, pnl=10.0)
        sa.record_outcome("test", won=True, pnl=10.0)
        sa.record_outcome("test", won=False, pnl=-5.0)
        streak = sa.get_current_streak("test")
        assert streak.type == "loss"
        assert streak.length == 1

    def test_is_on_tilt_true(self, tmp_db):
        sa = self._make(tmp_db)
        for _ in range(6):
            sa.record_outcome("test", won=False, pnl=-10.0)
        assert sa.is_on_tilt("test", loss_streak_threshold=5) is True

    def test_is_on_tilt_false(self, tmp_db):
        sa = self._make(tmp_db)
        for _ in range(3):
            sa.record_outcome("test", won=False, pnl=-10.0)
        assert sa.is_on_tilt("test", loss_streak_threshold=5) is False

    def test_runs_test_insufficient_data(self, tmp_db):
        sa = self._make(tmp_db)
        for _ in range(5):
            sa.record_outcome("test", won=True, pnl=10.0)
        result = sa.runs_test("test")
        assert result.random is True
        assert "Insufficient" in result.interpretation

    def test_runs_test_random_sequence(self, tmp_db):
        sa = self._make(tmp_db)
        import random
        rng = random.Random(42)
        for _ in range(100):
            won = rng.random() > 0.5
            sa.record_outcome("test", won=won, pnl=10.0 if won else -10.0)
        result = sa.runs_test("test")
        # Should generally be random
        assert isinstance(result.z_score, float)
        assert isinstance(result.p_value, float)

    def test_runs_test_all_same(self, tmp_db):
        sa = self._make(tmp_db)
        for _ in range(20):
            sa.record_outcome("test", won=True, pnl=10.0)
        result = sa.runs_test("test")
        assert result.random is False

    def test_streak_stats(self, tmp_db):
        sa = self._make(tmp_db)
        # Win 3, lose 2, win 1
        for _ in range(3):
            sa.record_outcome("test", won=True, pnl=10.0)
        for _ in range(2):
            sa.record_outcome("test", won=False, pnl=-5.0)
        sa.record_outcome("test", won=True, pnl=10.0)

        stats = sa.get_streak_stats("test")
        assert stats["longest_win"] == 3
        assert stats["longest_loss"] == 2

    def test_streak_stats_empty(self, tmp_db):
        sa = self._make(tmp_db)
        stats = sa.get_streak_stats("empty")
        assert stats["longest_win"] == 0
        assert stats["longest_loss"] == 0


# ===========================================================================
# Neuroevolution
# ===========================================================================

class TestNeuroevolution:

    def test_create_population(self):
        from evolution.neuroevolution import Neuroevolution
        ne = Neuroevolution(seed=42)
        pop = ne.create_population(pop_size=10, input_dim=5, output_dim=1)
        assert len(pop) == 10
        for g in pop:
            assert len(g.layers) >= 1
            assert len(g.activations) == len(g.layers)
            assert len(g.dropout_rates) == len(g.layers)
            assert g.input_dim == 5
            assert g.output_dim == 1

    def test_evaluate_python(self):
        from evolution.neuroevolution import Neuroevolution
        ne = Neuroevolution(seed=42)
        pop = ne.create_population(pop_size=3, input_dim=4, output_dim=1)
        # Simple dummy data
        train_x = [[1, 2, 3, 4]] * 10
        train_y = [1.0] * 10
        val_x = [[1, 2, 3, 4]] * 5
        val_y = [1.0] * 5
        for g in pop:
            fitness = ne.evaluate(g, (train_x, train_y), (val_x, val_y))
            assert isinstance(fitness, float)
            assert not math.isnan(fitness)

    def test_evolve(self):
        from evolution.neuroevolution import Neuroevolution
        ne = Neuroevolution(seed=42)
        pop = ne.create_population(pop_size=10, input_dim=4, output_dim=1)
        # Assign dummy fitness
        for i, g in enumerate(pop):
            g.fitness = float(i)
        new_pop = ne.evolve(pop, top_k=3)
        assert len(new_pop) == 10
        assert all(g.generation == 1 for g in new_pop)

    def test_get_best_genome(self):
        from evolution.neuroevolution import Neuroevolution
        ne = Neuroevolution(seed=42)
        pop = ne.create_population(pop_size=5, input_dim=4, output_dim=1)
        for i, g in enumerate(pop):
            g.fitness = float(i * 10)
        ne.evolve(pop)
        best = ne.get_best_genome()
        assert best is not None
        assert best.fitness == 40.0

    def test_get_best_genome_before_evolve(self):
        from evolution.neuroevolution import Neuroevolution
        ne = Neuroevolution(seed=42)
        assert ne.get_best_genome() is None

    def test_evolve_empty(self):
        from evolution.neuroevolution import Neuroevolution
        ne = Neuroevolution(seed=42)
        assert ne.evolve([]) == []

    def test_genome_repr(self):
        from evolution.neuroevolution import NetworkGenome
        g = NetworkGenome(layers=[64, 32], activations=["relu", "tanh"],
                          dropout_rates=[0.1, 0.2], fitness=0.5, generation=1)
        assert "64->32" in repr(g)

    def test_activation_functions(self):
        from evolution.neuroevolution import Neuroevolution
        ne = Neuroevolution(seed=42)
        # Test all activation functions
        for act in ["relu", "tanh", "sigmoid", "leaky_relu", "elu", "gelu"]:
            result = ne._apply_activation(1.0, act)
            assert isinstance(result, float)
            assert not math.isnan(result)

    def test_activation_edge_cases(self):
        from evolution.neuroevolution import Neuroevolution
        ne = Neuroevolution(seed=42)
        assert ne._apply_activation(-1.0, "relu") == 0.0
        assert ne._apply_activation(-1.0, "leaky_relu") == pytest.approx(-0.01)
        assert 0 < ne._apply_activation(100.0, "sigmoid") <= 1.0


# ===========================================================================
# StrategyBreeder
# ===========================================================================

class TestStrategyBreeder:

    def _make(self, tmp_db):
        from evolution.strategy_breeder import StrategyBreeder
        return StrategyBreeder(db_path=tmp_db("breed"), seed=42)

    def test_register_and_crossover(self, tmp_db):
        breeder = self._make(tmp_db)
        breeder.register_strategy("a", {"lookback": 20, "threshold": 0.6}, fitness=1.0)
        breeder.register_strategy("b", {"lookback": 30, "threshold": 0.4}, fitness=2.0)
        child = breeder.crossover("a", "b")
        assert "lookback" in child
        assert "threshold" in child
        assert child["lookback"] in (20, 30)
        assert child["threshold"] in (0.6, 0.4)

    def test_crossover_missing_parent(self, tmp_db):
        breeder = self._make(tmp_db)
        breeder.register_strategy("a", {"x": 1}, fitness=1.0)
        with pytest.raises(ValueError):
            breeder.crossover("a", "nonexistent")

    def test_mutate_numeric(self, tmp_db):
        breeder = self._make(tmp_db)
        params = {"lookback": 20, "threshold": 0.5, "name": "test"}
        mutated = breeder.mutate(params, mutation_rate=1.0)
        # Name should be unchanged (string), numerics may have changed
        assert mutated["name"] == "test"
        assert isinstance(mutated["lookback"], int)
        assert isinstance(mutated["threshold"], float)

    def test_mutate_boolean(self, tmp_db):
        breeder = self._make(tmp_db)
        params = {"enabled": True}
        # Run many times to ensure at least one flip
        flipped = False
        for _ in range(50):
            m = breeder.mutate(params, mutation_rate=1.0)
            if m["enabled"] is False:
                flipped = True
                break
        assert flipped

    def test_mutate_preserves_original(self, tmp_db):
        breeder = self._make(tmp_db)
        params = {"x": 100}
        breeder.mutate(params, mutation_rate=1.0)
        assert params["x"] == 100  # Original unchanged

    def test_breed_generation(self, tmp_db):
        breeder = self._make(tmp_db)
        for i in range(5):
            breeder.register_strategy(f"strat_{i}",
                                      {"lookback": 10 + i, "mult": 1.0 + i * 0.1},
                                      fitness=float(i))
        children = breeder.breed_generation(top_k=3, offspring=5)
        assert len(children) == 5
        for child in children:
            assert "lookback" in child
            assert "mult" in child

    def test_breed_generation_insufficient(self, tmp_db):
        breeder = self._make(tmp_db)
        breeder.register_strategy("only_one", {"x": 1}, fitness=1.0)
        children = breeder.breed_generation(top_k=3, offspring=5)
        assert len(children) == 0

    def test_lineage(self, tmp_db):
        breeder = self._make(tmp_db)
        breeder.register_strategy("grandparent_a", {"x": 1}, fitness=1.0)
        breeder.register_strategy("grandparent_b", {"x": 2}, fitness=2.0)
        breeder.register_strategy("parent_a", {"x": 3}, fitness=3.0,
                                  parent_a="grandparent_a", parent_b="grandparent_b")
        breeder.register_strategy("child", {"x": 4}, fitness=4.0,
                                  parent_a="parent_a", parent_b="grandparent_b")

        lineage = breeder.get_lineage("child")
        assert "parent_a" in lineage
        assert "grandparent_b" in lineage
        assert "grandparent_a" in lineage

    def test_lineage_no_parents(self, tmp_db):
        breeder = self._make(tmp_db)
        breeder.register_strategy("root", {"x": 1}, fitness=1.0)
        lineage = breeder.get_lineage("root")
        assert lineage == []

    def test_register_overwrites(self, tmp_db):
        breeder = self._make(tmp_db)
        breeder.register_strategy("strat", {"x": 1}, fitness=1.0)
        breeder.register_strategy("strat", {"x": 2}, fitness=3.0)
        child = breeder.crossover.__func__  # just verify no error on re-register
        # Verify latest params are stored
        params = breeder._get_params("strat")
        assert params["x"] == 2


# ===========================================================================
# SelfDebugger
# ===========================================================================

class TestSelfDebugger:

    def test_healthy_component(self):
        import time
        from adaptive.self_debugger import SelfDebugger
        dbg = SelfDebugger()
        for i in range(30):
            dbg.record_output("comp", float(i))
            time.sleep(0.001)  # ~1ms between outputs — realistic cadence
        alert = dbg.detect_anomaly("comp")
        assert alert.anomalous is False

    def test_nan_detection(self):
        from adaptive.self_debugger import SelfDebugger
        dbg = SelfDebugger()
        dbg.record_output("comp", float("nan"))
        alert = dbg.detect_anomaly("comp")
        assert alert.anomalous is True
        assert "NaN" in alert.reason

    def test_none_detection(self):
        from adaptive.self_debugger import SelfDebugger
        dbg = SelfDebugger()
        dbg.record_output("comp", None)
        alert = dbg.detect_anomaly("comp")
        assert alert.anomalous is True
        assert "NaN" in alert.reason

    def test_sigma_outlier(self):
        from adaptive.self_debugger import SelfDebugger
        dbg = SelfDebugger(sigma_threshold=3.0)
        # Record many normal values
        for _ in range(50):
            dbg.record_output("comp", 1.0)
        # Record a huge outlier
        dbg.record_output("comp", 1000.0)
        alert = dbg.detect_anomaly("comp")
        assert alert.anomalous is True
        assert "sigma" in alert.reason

    def test_stuck_detection(self):
        from adaptive.self_debugger import SelfDebugger
        dbg = SelfDebugger(stuck_threshold=10)
        for _ in range(25):
            dbg.record_output("comp", 42.0)
        alert = dbg.detect_anomaly("comp")
        assert alert.anomalous is True
        assert "stuck" in alert.reason.lower()

    def test_auto_disable(self):
        from adaptive.self_debugger import SelfDebugger
        dbg = SelfDebugger(auto_disable_threshold=3)
        # Record NaN to trigger anomalies
        for _ in range(5):
            dbg.record_output("comp", None)
            dbg.detect_anomaly("comp")
        disabled = dbg.auto_disable("comp")
        assert disabled is True
        health = dbg.get_component_health()
        assert health["comp"] == "disabled"

    def test_auto_disable_not_enough(self):
        from adaptive.self_debugger import SelfDebugger
        dbg = SelfDebugger(auto_disable_threshold=10)
        for _ in range(3):
            dbg.record_output("comp", None)
            dbg.detect_anomaly("comp")
        assert dbg.auto_disable("comp") is False

    def test_re_enable(self):
        from adaptive.self_debugger import SelfDebugger
        dbg = SelfDebugger(auto_disable_threshold=2)
        dbg.record_output("comp", None)
        dbg.detect_anomaly("comp")
        dbg.record_output("comp", None)
        dbg.detect_anomaly("comp")
        dbg.auto_disable("comp")
        assert dbg.get_component_health()["comp"] == "disabled"
        assert dbg.re_enable("comp") is True
        assert dbg.get_component_health()["comp"] == "healthy"

    def test_re_enable_not_disabled(self):
        from adaptive.self_debugger import SelfDebugger
        dbg = SelfDebugger()
        assert dbg.re_enable("nonexistent") is False

    def test_component_health_multiple(self):
        from adaptive.self_debugger import SelfDebugger
        dbg = SelfDebugger()
        dbg.record_output("good", 1.0)
        dbg.detect_anomaly("good")
        dbg.record_output("bad", None)
        dbg.detect_anomaly("bad")
        health = dbg.get_component_health()
        assert health["good"] == "healthy"
        assert health["bad"] == "warning"

    def test_no_data_component(self):
        from adaptive.self_debugger import SelfDebugger
        dbg = SelfDebugger()
        alert = dbg.detect_anomaly("unknown")
        assert alert.anomalous is False
        assert "No data" in alert.reason

    def test_disabled_component_alert(self):
        from adaptive.self_debugger import SelfDebugger
        dbg = SelfDebugger(auto_disable_threshold=1)
        dbg.record_output("comp", None)
        dbg.detect_anomaly("comp")
        dbg.auto_disable("comp")
        alert = dbg.detect_anomaly("comp")
        assert alert.anomalous is True
        assert "auto-disabled" in alert.reason.lower()

    def test_non_numeric_recorded_as_nan(self):
        from adaptive.self_debugger import SelfDebugger
        dbg = SelfDebugger()
        dbg.record_output("comp", "not_a_number")
        alert = dbg.detect_anomaly("comp")
        assert alert.anomalous is True

    def test_inf_recorded_as_nan(self):
        from adaptive.self_debugger import SelfDebugger
        dbg = SelfDebugger()
        dbg.record_output("comp", float("inf"))
        alert = dbg.detect_anomaly("comp")
        assert alert.anomalous is True
