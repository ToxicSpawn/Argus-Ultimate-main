"""
tests/test_tier2_self_improvement.py — Tests for Tier 2 Self-Improvement Modules

Covers:
    1. TradeOutcomeLabeler   — good/bad/neutral logic, persistence, accuracy, dataset
    2. FeatureDriftDetector  — snapshot recording, drift detection, declining features, retrain
    3. StrategyDecayDetector — trade recording, decay detection, allocation multiplier, health
    4. CorrelationRegimeLearner — return recording, correlation matrix, diversification, hedges
"""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------

from ml.trade_outcome_labeler import TradeOutcomeLabeler, TradeLabel
from ml.feature_drift_detector import FeatureDriftDetector, DriftReport
from adaptive.strategy_decay_detector import StrategyDecayDetector, DecayReport
from risk.correlation_regime_learner import CorrelationRegimeLearner


# ===========================================================================
# 1. TradeOutcomeLabeler
# ===========================================================================

class TestTradeOutcomeLabeler(unittest.TestCase):
    """Tests for automatic trade outcome labeling."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self._tmp, "labels.db")
        self.labeler = TradeOutcomeLabeler(db_path=self.db_path)

    # ----- labelling logic -----

    def test_good_label_buy(self) -> None:
        """Buy trade with >0.5% profit → 'good'."""
        label = self.labeler.label_trade(
            symbol="BTC/USD", side="buy",
            entry_price=100.0, exit_price=101.0,
        )
        self.assertEqual(label.label, "good")
        self.assertAlmostEqual(label.post_trade_return_pct, 1.0, places=2)

    def test_good_label_sell(self) -> None:
        """Sell trade with >0.5% profit → 'good'."""
        label = self.labeler.label_trade(
            symbol="ETH/USD", side="sell",
            entry_price=100.0, exit_price=99.0,
        )
        self.assertEqual(label.label, "good")
        self.assertAlmostEqual(label.post_trade_return_pct, 1.0, places=2)

    def test_bad_label_adverse_excursion(self) -> None:
        """Trade with adverse excursion > 1% → 'bad'."""
        label = self.labeler.label_trade(
            symbol="BTC/USD", side="buy",
            entry_price=100.0, exit_price=100.2,
            observed_prices=[98.5, 99.0, 100.2],
        )
        self.assertEqual(label.label, "bad")
        self.assertGreater(label.max_adverse_pct, 1.0)

    def test_neutral_label(self) -> None:
        """Flat trade → 'neutral'."""
        label = self.labeler.label_trade(
            symbol="BTC/USD", side="buy",
            entry_price=100.0, exit_price=100.1,
        )
        self.assertEqual(label.label, "neutral")

    def test_mfe_mae_from_observed_prices(self) -> None:
        """MFE and MAE computed correctly from observed prices."""
        label = self.labeler.label_trade(
            symbol="BTC/USD", side="buy",
            entry_price=100.0, exit_price=101.0,
            observed_prices=[99.0, 102.0, 101.0],
        )
        self.assertAlmostEqual(label.max_favorable_pct, 2.0, places=2)
        self.assertAlmostEqual(label.max_adverse_pct, 1.0, places=2)

    def test_invalid_side_raises(self) -> None:
        """Invalid side raises ValueError."""
        with self.assertRaises(ValueError):
            self.labeler.label_trade("BTC/USD", "hold", 100.0)

    def test_invalid_price_raises(self) -> None:
        """Negative entry price raises ValueError."""
        with self.assertRaises(ValueError):
            self.labeler.label_trade("BTC/USD", "buy", -1.0)

    # ----- persistence -----

    def test_persistence_round_trip(self) -> None:
        """Labels are persisted and can be retrieved."""
        self.labeler.label_trade("BTC/USD", "buy", 100.0, exit_price=101.0, strategy="mom")
        self.labeler.label_trade("ETH/USD", "sell", 200.0, exit_price=199.0, strategy="mom")
        labels = self.labeler.get_all_labels()
        self.assertEqual(len(labels), 2)

    # ----- accuracy -----

    def test_strategy_accuracy(self) -> None:
        """get_strategy_accuracy returns correct counts."""
        for _ in range(3):
            self.labeler.label_trade("BTC/USD", "buy", 100.0, exit_price=101.0, strategy="alpha")
        for _ in range(2):
            self.labeler.label_trade("BTC/USD", "buy", 100.0, exit_price=100.0, strategy="alpha")

        stats = self.labeler.get_strategy_accuracy("alpha")
        self.assertEqual(stats["good_count"], 3)
        self.assertEqual(stats["neutral_count"], 2)
        self.assertEqual(stats["total"], 5)
        self.assertAlmostEqual(stats["win_rate"], 0.6, places=2)

    # ----- dataset -----

    def test_build_classifier_dataset(self) -> None:
        """build_classifier_dataset returns enriched dicts."""
        self.labeler.label_trade("BTC/USD", "buy", 100.0, exit_price=101.0, strategy="test")
        ds = self.labeler.build_classifier_dataset()
        self.assertGreaterEqual(len(ds), 1)
        row = ds[0]
        self.assertIn("side_numeric", row)
        self.assertIn("label_numeric", row)
        self.assertEqual(row["side_numeric"], 1)
        self.assertEqual(row["label_numeric"], 1)

    def test_no_exit_defaults_neutral(self) -> None:
        """Trade with no exit price → neutral (0% return)."""
        label = self.labeler.label_trade("BTC/USD", "buy", 100.0)
        self.assertEqual(label.label, "neutral")
        self.assertAlmostEqual(label.post_trade_return_pct, 0.0)


# ===========================================================================
# 2. FeatureDriftDetector
# ===========================================================================

class TestFeatureDriftDetector(unittest.TestCase):
    """Tests for feature importance drift detection."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self._tmp, "drift.db")
        self.detector = FeatureDriftDetector(db_path=self.db_path, default_threshold=0.3)

    def test_no_drift_with_stable_features(self) -> None:
        """Identical snapshots produce no drift."""
        imp = {"feat_a": 0.5, "feat_b": 0.3, "feat_c": 0.2}
        self.detector.record_feature_importance("model_x", imp)
        self.detector.record_feature_importance("model_x", imp)
        report = self.detector.detect_drift("model_x")
        self.assertFalse(report.alert)
        self.assertEqual(len(report.drifted_features), 0)

    def test_drift_detected_with_shifted_features(self) -> None:
        """Large shift in feature importance triggers drift alert."""
        self.detector.record_feature_importance("model_x", {"a": 0.9, "b": 0.1})
        self.detector.record_feature_importance("model_x", {"a": 0.1, "b": 0.9})
        report = self.detector.detect_drift("model_x")
        self.assertTrue(report.alert)
        self.assertGreater(len(report.drifted_features), 0)

    def test_drift_scores_contain_all_features(self) -> None:
        """Drift scores dict has entries for all features."""
        self.detector.record_feature_importance("m", {"x": 0.5, "y": 0.5})
        self.detector.record_feature_importance("m", {"x": 0.5, "y": 0.3, "z": 0.2})
        report = self.detector.detect_drift("m")
        self.assertIn("x", report.drift_scores)
        self.assertIn("y", report.drift_scores)
        self.assertIn("z", report.drift_scores)

    def test_insufficient_snapshots(self) -> None:
        """With only 1 snapshot, no drift is reported."""
        self.detector.record_feature_importance("lonely", {"a": 1.0})
        report = self.detector.detect_drift("lonely")
        self.assertFalse(report.alert)

    def test_declining_features(self) -> None:
        """get_declining_features returns features whose importance dropped."""
        self.detector.record_feature_importance("m", {"a": 0.8, "b": 0.2})
        self.detector.record_feature_importance("m", {"a": 0.3, "b": 0.7})
        declining = self.detector.get_declining_features("m")
        feature_names = [d["feature"] for d in declining]
        self.assertIn("a", feature_names)

    def test_should_retrain_true(self) -> None:
        """should_retrain returns True when >3 features drifted."""
        # Use low threshold to guarantee drift
        det = FeatureDriftDetector(db_path=self.db_path + ".2", default_threshold=0.01)
        det.record_feature_importance("big", {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1})
        det.record_feature_importance("big", {"a": 0.1, "b": 0.1, "c": 0.4, "d": 0.4})
        self.assertTrue(det.should_retrain("big"))

    def test_should_retrain_false(self) -> None:
        """should_retrain returns False when features are stable."""
        self.detector.record_feature_importance("stable", {"a": 0.5, "b": 0.5})
        self.detector.record_feature_importance("stable", {"a": 0.5, "b": 0.5})
        self.assertFalse(self.detector.should_retrain("stable"))

    def test_normalisation(self) -> None:
        """Normalisation makes values sum to 1."""
        norm = FeatureDriftDetector._normalise({"a": 10, "b": 20, "c": 70})
        total = sum(norm.values())
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_zero_importances(self) -> None:
        """All-zero importances normalise without error."""
        norm = FeatureDriftDetector._normalise({"a": 0, "b": 0})
        self.assertEqual(norm["a"], 0.0)
        self.assertEqual(norm["b"], 0.0)

    def test_drift_report_dataclass(self) -> None:
        """DriftReport is immutable and has correct fields."""
        report = DriftReport(
            model_name="test", drifted_features=["a"],
            drift_scores={"a": 0.5}, alert=True,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        self.assertEqual(report.model_name, "test")
        self.assertTrue(report.alert)


# ===========================================================================
# 3. StrategyDecayDetector
# ===========================================================================

class TestStrategyDecayDetector(unittest.TestCase):
    """Tests for strategy alpha decay detection."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self._tmp, "decay.db")
        self.detector = StrategyDecayDetector(db_path=self.db_path)

    def test_healthy_strategy(self) -> None:
        """Consistently profitable strategy shows no decay."""
        for _ in range(50):
            self.detector.record_trade("winner", 10.0)
        report = self.detector.detect_decay("winner")
        self.assertFalse(report.decaying)
        self.assertEqual(report.recommendation, "maintain")

    def test_decaying_strategy(self) -> None:
        """Strategy with declining returns is flagged as decaying."""
        # Start profitable, then go negative
        for i in range(25):
            self.detector.record_trade("loser", 20.0 - i * 2)
        for _ in range(25):
            self.detector.record_trade("loser", -10.0)
        report = self.detector.detect_decay("loser")
        self.assertTrue(report.decaying)
        self.assertIn(report.recommendation, ("reduce", "disable"))

    def test_insufficient_trades(self) -> None:
        """With fewer than min_trades, returns healthy."""
        for _ in range(5):
            self.detector.record_trade("new_strat", 1.0)
        report = self.detector.detect_decay("new_strat", min_trades=20)
        self.assertFalse(report.decaying)
        self.assertEqual(report.recommendation, "maintain")

    def test_allocation_multiplier_healthy(self) -> None:
        """Healthy strategy gets 1.0 multiplier."""
        for _ in range(50):
            self.detector.record_trade("strong", 5.0)
        mult = self.detector.get_allocation_multiplier("strong")
        self.assertEqual(mult, 1.0)

    def test_allocation_multiplier_decaying(self) -> None:
        """Decaying strategy gets 0.5 multiplier."""
        for i in range(25):
            self.detector.record_trade("fading", 20.0 - i * 2)
        for _ in range(25):
            self.detector.record_trade("fading", -10.0)
        mult = self.detector.get_allocation_multiplier("fading")
        self.assertLessEqual(mult, 0.5)

    def test_get_all_strategies_health(self) -> None:
        """get_all_strategies_health returns reports for all strategies."""
        for _ in range(25):
            self.detector.record_trade("strat_a", 5.0)
            self.detector.record_trade("strat_b", -2.0)
        health = self.detector.get_all_strategies_health()
        self.assertIn("strat_a", health)
        self.assertIn("strat_b", health)
        for v in health.values():
            self.assertIsInstance(v, DecayReport)

    def test_sharpe_computation(self) -> None:
        """_sharpe returns positive for positive-mean returns."""
        s = StrategyDecayDetector._sharpe([1.0, 2.0, 1.5, 3.0, 2.5])
        self.assertGreater(s, 0.0)

    def test_linear_slope_positive(self) -> None:
        """_linear_slope returns positive for upward series."""
        slope = StrategyDecayDetector._linear_slope([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertGreater(slope, 0.0)

    def test_linear_slope_negative(self) -> None:
        """_linear_slope returns negative for downward series."""
        slope = StrategyDecayDetector._linear_slope([5.0, 4.0, 3.0, 2.0, 1.0])
        self.assertLess(slope, 0.0)

    def test_decay_report_fields(self) -> None:
        """DecayReport has all expected fields."""
        for _ in range(50):
            self.detector.record_trade("check", 1.0)
        report = self.detector.detect_decay("check")
        self.assertIsInstance(report.strategy, str)
        self.assertIsInstance(report.decaying, bool)
        self.assertIsInstance(report.sharpe_trend, float)
        self.assertIsInstance(report.recommendation, str)


# ===========================================================================
# 4. CorrelationRegimeLearner
# ===========================================================================

class TestCorrelationRegimeLearner(unittest.TestCase):
    """Tests for correlation regime learning."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self._tmp, "corr.db")
        self.learner = CorrelationRegimeLearner(db_path=self.db_path)

    def test_update_returns_stores_data(self) -> None:
        """update_returns populates in-memory buffer."""
        self.learner.update_returns("BTC/USD", 1.5)
        self.learner.update_returns("BTC/USD", -0.3)
        self.assertEqual(len(self.learner._returns["BTC/USD"]), 2)

    def test_correlation_matrix_identity_single(self) -> None:
        """Single symbol produces 1.0 self-correlation."""
        for _ in range(10):
            self.learner.update_returns("BTC/USD", 1.0)
        matrix = self.learner.compute_correlation_matrix()
        self.assertAlmostEqual(matrix["BTC/USD"]["BTC/USD"], 1.0)

    def test_correlation_matrix_two_symbols(self) -> None:
        """Two perfectly correlated symbols → correlation near 1.0."""
        import random
        random.seed(42)
        for _ in range(50):
            r = random.gauss(0, 1)
            self.learner.update_returns("A", r)
            self.learner.update_returns("B", r)
        matrix = self.learner.compute_correlation_matrix()
        self.assertGreater(matrix["A"]["B"], 0.95)

    def test_negative_correlation(self) -> None:
        """Anti-correlated symbols → negative correlation."""
        import random
        random.seed(42)
        for _ in range(50):
            r = random.gauss(0, 1)
            self.learner.update_returns("X", r)
            self.learner.update_returns("Y", -r)
        matrix = self.learner.compute_correlation_matrix()
        self.assertLess(matrix["X"]["Y"], -0.95)

    def test_diversification_score_correlated(self) -> None:
        """Perfectly correlated portfolio → low diversification."""
        import random
        random.seed(42)
        for _ in range(50):
            r = random.gauss(0, 1)
            self.learner.update_returns("A", r)
            self.learner.update_returns("B", r)
        self.learner.compute_correlation_matrix()
        score = self.learner.get_diversification_score(["A", "B"])
        self.assertLess(score, 0.1)

    def test_diversification_score_uncorrelated(self) -> None:
        """Uncorrelated portfolio → high diversification."""
        import random
        random.seed(123)
        for _ in range(200):
            self.learner.update_returns("P", random.gauss(0, 1))
            self.learner.update_returns("Q", random.gauss(0, 1))
        self.learner.compute_correlation_matrix()
        score = self.learner.get_diversification_score(["P", "Q"])
        self.assertGreater(score, 0.5)

    def test_stress_hedges_returns_negatively_correlated(self) -> None:
        """get_stress_hedges returns negatively correlated symbols first."""
        import random
        random.seed(42)
        for _ in range(50):
            r = random.gauss(0, 1)
            self.learner.update_returns("BTC", r)
            self.learner.update_returns("HEDGE", -r)
            self.learner.update_returns("SAME", r)
        self.learner.compute_correlation_matrix()
        hedges = self.learner.get_stress_hedges("BTC")
        self.assertEqual(hedges[0], "HEDGE")

    def test_decorrelation_detection(self) -> None:
        """detect_decorrelation_events detects regime change."""
        import random
        random.seed(42)
        # Phase 1: correlated
        for _ in range(50):
            r = random.gauss(0, 1)
            self.learner.update_returns("M", r)
            self.learner.update_returns("N", r * 0.9)
        self.learner.compute_correlation_matrix()

        # Phase 2: anti-correlated (replace buffer)
        self.learner._returns["M"] = []
        self.learner._returns["N"] = []
        for _ in range(50):
            r = random.gauss(0, 1)
            self.learner.update_returns("M", r)
            self.learner.update_returns("N", -r)
        self.learner.compute_correlation_matrix()

        events = self.learner.detect_decorrelation_events(threshold=0.3)
        self.assertGreater(len(events), 0)

    def test_flush_to_db(self) -> None:
        """flush() writes to SQLite without error."""
        for _ in range(10):
            self.learner.update_returns("BTC", 1.0)
        self.learner.flush()
        # Verify DB has data
        import sqlite3
        con = sqlite3.connect(self.db_path)
        count = con.execute("SELECT COUNT(*) FROM return_observations").fetchone()[0]
        con.close()
        self.assertEqual(count, 10)

    def test_single_symbol_diversification(self) -> None:
        """Single symbol → diversification score 1.0."""
        score = self.learner.get_diversification_score(["BTC"])
        self.assertEqual(score, 1.0)

    def test_memory_trimming(self) -> None:
        """In-memory buffer is trimmed at max_memory_points."""
        learner = CorrelationRegimeLearner(db_path=self.db_path + ".trim", max_memory_points=50)
        for i in range(100):
            learner.update_returns("BTC", float(i))
        self.assertEqual(len(learner._returns["BTC"]), 50)


# ===========================================================================

if __name__ == "__main__":
    unittest.main()
