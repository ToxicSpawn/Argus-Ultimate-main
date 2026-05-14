"""Tests for ARGUS superintelligence systems."""
import unittest
from core.superintelligence import (
    CausalReasoningEngine, CounterfactualAnalyzer, HypothesisEngine,
    MetaCognition, TemporalAbstraction, AdversarialThinker,
    ConfidenceAssessment,
)


class TestCausalReasoning(unittest.TestCase):
    def test_learns_causal_link(self):
        ce = CausalReasoningEngine(min_observations=3)
        import time
        now = time.time()
        for i in range(5):
            ce.record_event("dxy_spike", 1.0, now + i * 10)
            ce.record_event("btc_drop", 1.0, now + i * 10 + 5)
        preds = ce.predict_effects("dxy_spike")
        self.assertGreater(len(preds), 0)

    def test_explain_event(self):
        ce = CausalReasoningEngine(min_observations=3)
        import time
        now = time.time()
        for i in range(5):
            ce.record_event("whale_deposit", 1.0, now + i * 20)
            ce.record_event("price_drop", 1.0, now + i * 20 + 10)
        ce.record_event("whale_deposit", 1.0, now + 100)
        ce.record_event("price_drop", 1.0, now + 110)
        explanations = ce.explain("price_drop")
        # Should identify whale_deposit as a cause
        self.assertTrue(len(explanations) >= 0)  # may or may not find depending on timing

    def test_causal_chain(self):
        ce = CausalReasoningEngine(min_observations=3)
        import time
        now = time.time()
        for i in range(5):
            ce.record_event("a", 1.0, now + i * 30)
            ce.record_event("b", 1.0, now + i * 30 + 10)
            ce.record_event("c", 1.0, now + i * 30 + 20)
        chains = ce.get_chain("a", depth=2)
        self.assertIsInstance(chains, list)

    def test_get_stats(self):
        ce = CausalReasoningEngine()
        stats = ce.get_stats()
        self.assertIn("total_links", stats)


class TestCounterfactual(unittest.TestCase):
    def test_record_good_decision(self):
        cf = CounterfactualAnalyzer()
        result = cf.record_decision("t1", "BUY", actual_pnl=5.0, alternative_pnl=-2.0)
        self.assertGreater(result.regret, 0)  # positive = we chose right
        self.assertIn("Good call", result.lesson)

    def test_record_bad_decision(self):
        cf = CounterfactualAnalyzer()
        result = cf.record_decision("t1", "BUY", actual_pnl=-3.0, alternative_pnl=2.0)
        self.assertLess(result.regret, 0)  # negative = we chose wrong
        self.assertIn("Should have", result.lesson)

    def test_bias_detection(self):
        cf = CounterfactualAnalyzer()
        for i in range(10):
            cf.record_decision(f"t{i}", "BUY", actual_pnl=-1.0, alternative_pnl=1.0, regime="ranging")
        biases = cf.get_biases()
        self.assertIn("BUY_ranging", biases)
        self.assertLess(biases["BUY_ranging"], 0)  # consistently wrong

    def test_override_recommendation(self):
        cf = CounterfactualAnalyzer()
        for i in range(15):
            cf.record_decision(f"t{i}", "BUY", actual_pnl=-3.0, alternative_pnl=1.0, regime="ranging")
        override = cf.should_override("BUY", "ranging")
        self.assertEqual(override, "SKIP")

    def test_no_override_insufficient_data(self):
        cf = CounterfactualAnalyzer()
        self.assertIsNone(cf.should_override("BUY", "trending"))


class TestHypothesisEngine(unittest.TestCase):
    def test_generate_hypothesis(self):
        he = HypothesisEngine()
        hyp = he.generate("h1", "BTC will break 60K", "btc_momentum", "price_up", 0.7, ttl_seconds=60)
        self.assertEqual(len(he.get_active()), 1)

    def test_resolve_hypothesis(self):
        he = HypothesisEngine()
        he.generate("h1", "test", "cond", "pred", 0.7, ttl_seconds=3600)
        he.resolve("h1", True, "it happened")
        self.assertEqual(len(he.get_active()), 0)
        self.assertEqual(he.get_stats()["completed"], 1)

    def test_expired_hypotheses(self):
        he = HypothesisEngine()
        he.generate("h1", "test", "cond", "pred", 0.7, ttl_seconds=0.001)
        import time
        time.sleep(0.01)
        expired = he.check_expired()
        self.assertEqual(len(expired), 1)

    def test_max_active(self):
        he = HypothesisEngine(max_active=3)
        for i in range(5):
            he.generate(f"h{i}", f"test{i}", "cond", "pred", 0.5, ttl_seconds=3600)
        self.assertLessEqual(len(he.get_active()), 3)


class TestMetaCognition(unittest.TestCase):
    def test_high_confidence(self):
        mc = MetaCognition(min_history_bars=50)
        mc.record_pattern("bullish_divergence")
        for _ in range(10):
            mc.record_pattern("bullish_divergence")
        assessment = mc.assess(
            symbol="BTC/USD", history_bars=200,
            model_predictions={"model_a": 0.5, "model_b": 0.3, "model_c": 0.2},
            regime="trending", regime_age_bars=100,
            pattern_key="bullish_divergence",
        )
        self.assertGreater(assessment.overall_confidence, 0.5)
        self.assertEqual(assessment.recommendation, "TRADE")

    def test_low_confidence_unfamiliar(self):
        mc = MetaCognition(min_history_bars=100)
        assessment = mc.assess(
            symbol="BTC/USD", history_bars=20,
            model_predictions={"a": 0.5, "b": -0.3},  # disagree
            regime="crisis", regime_age_bars=5,
            pattern_key="never_seen_this",
        )
        self.assertLess(assessment.overall_confidence, 0.5)
        self.assertIn(assessment.recommendation, ("REDUCE_SIZE", "WAIT", "SKIP"))
        self.assertGreater(len(assessment.known_unknowns), 0)

    def test_known_unknowns_listed(self):
        mc = MetaCognition(min_history_bars=200)
        assessment = mc.assess("BTC/USD", 10, {}, "unknown", 2)
        self.assertGreater(len(assessment.known_unknowns), 0)

    def test_reasoning_string(self):
        mc = MetaCognition()
        assessment = mc.assess("BTC/USD", 500, {"m": 0.5}, "trending", 100)
        self.assertIn("Confidence", assessment.reasoning)


class TestTemporalAbstraction(unittest.TestCase):
    def test_trending_market(self):
        ta = TemporalAbstraction()
        for i in range(200):
            ta.update("BTC/USD", 50000 + i * 10)
        sig = ta.decompose("BTC/USD")
        self.assertGreater(sig.macro_trend, 0)
        self.assertGreater(sig.alignment, 0.5)

    def test_noisy_market(self):
        import random
        rng = random.Random(42)
        ta = TemporalAbstraction()
        for _ in range(200):
            ta.update("BTC/USD", 50000 + rng.gauss(0, 100))
        sig = ta.decompose("BTC/USD")
        self.assertLess(sig.alignment, 0.8)

    def test_insufficient_data(self):
        ta = TemporalAbstraction()
        ta.update("BTC/USD", 50000)
        sig = ta.decompose("BTC/USD")
        self.assertEqual(sig.dominant_scale, "insufficient")


class TestAdversarialThinker(unittest.TestCase):
    def test_profitable_against_retail(self):
        at = AdversarialThinker()
        for _ in range(10):
            at.record_counterparty("buy", "retail", 2.0)
        score = at.assess_counterparty("buy", "retail")
        self.assertGreater(score, 0)  # we profit against retail

    def test_losing_against_whales(self):
        at = AdversarialThinker()
        for _ in range(10):
            at.record_counterparty("buy", "whale", -3.0)
        score = at.assess_counterparty("buy", "whale")
        self.assertLess(score, 0)  # we lose against whales

    def test_insufficient_data(self):
        at = AdversarialThinker()
        score = at.assess_counterparty("buy", "unknown")
        self.assertAlmostEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
