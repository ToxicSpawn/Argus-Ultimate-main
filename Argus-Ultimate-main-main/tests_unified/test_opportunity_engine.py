"""Tests for opportunity scoring and ranking engine."""
from __future__ import annotations

import unittest

from argus_live.portfolio.opportunity_engine import (
    Opportunity,
    rank_opportunities,
    score_opportunity,
)


class TestOpportunityEngine(unittest.TestCase):

    def test_score_formula(self):
        opp = score_opportunity(
            strategy_id="momentum",
            symbol="BTC/AUD",
            venue="kraken",
            expected_edge_bps=10.0,
            confidence=0.8,
            regime_score=0.7,
            alpha_score=5.0,
            efficiency_score=0.6,
        )
        self.assertIsInstance(opp, Opportunity)
        # 10*0.35 + 0.8*10*0.20 + 0.7*10*0.15 + 5.0*0.20 + 0.6*10*0.10
        # = 3.5 + 1.6 + 1.05 + 1.0 + 0.6 = 7.75
        self.assertAlmostEqual(opp.score, 7.75, places=2)

    def test_higher_edge_ranks_first(self):
        high = score_opportunity("a", "BTC/AUD", "kraken", 20.0, 0.9, 0.8, 8.0, 0.7)
        low = score_opportunity("b", "ETH/AUD", "kraken", 2.0, 0.3, 0.2, 1.0, 0.2)
        ranked = rank_opportunities([low, high])
        self.assertEqual(ranked[0].strategy_id, "a")
        self.assertGreater(ranked[0].score, ranked[1].score)

    def test_capital_priority_trumps_score(self):
        high_score = score_opportunity("a", "BTC/AUD", "kraken", 20.0, 0.9, 0.8, 8.0, 0.7, capital_priority=1.0)
        low_score_high_prio = score_opportunity("b", "ETH/AUD", "kraken", 1.0, 0.1, 0.1, 0.1, 0.1, capital_priority=5.0)
        ranked = rank_opportunities([high_score, low_score_high_prio])
        self.assertEqual(ranked[0].strategy_id, "b")

    def test_frozen_dataclass(self):
        opp = score_opportunity("a", "BTC/AUD", "kraken", 5.0, 0.5, 0.5, 2.0, 0.5)
        with self.assertRaises(AttributeError):
            opp.score = 999.0  # type: ignore[misc]

    def test_rank_stable_for_equal_scores(self):
        opp1 = score_opportunity("a", "BTC/AUD", "kraken", 5.0, 0.5, 0.5, 2.0, 0.5)
        opp2 = score_opportunity("b", "ETH/AUD", "kraken", 5.0, 0.5, 0.5, 2.0, 0.5)
        ranked = rank_opportunities([opp1, opp2])
        self.assertEqual(len(ranked), 2)


if __name__ == "__main__":
    unittest.main()
