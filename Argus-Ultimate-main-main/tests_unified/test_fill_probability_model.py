from __future__ import annotations

import unittest

from argus_live.execution.fill_probability_model import estimate_fill_probability


class TestFillProbabilityModel(unittest.TestCase):
    def test_higher_flow_higher_probability(self):
        low_flow = estimate_fill_probability(1000.0, 10.0)
        high_flow = estimate_fill_probability(1000.0, 100.0)
        self.assertGreater(high_flow.maker_fill_probability, low_flow.maker_fill_probability)

    def test_zero_flow(self):
        est = estimate_fill_probability(1000.0, 0.0)
        self.assertEqual(est.maker_fill_probability, 0.0)
        self.assertEqual(est.reason, "no_trade_flow")

    def test_cancellation_relief_increases_prob(self):
        base = estimate_fill_probability(1000.0, 50.0, cancellation_relief_ratio=0.0)
        relief = estimate_fill_probability(1000.0, 50.0, cancellation_relief_ratio=0.5)
        self.assertGreater(relief.maker_fill_probability, base.maker_fill_probability)

    def test_small_queue_high_prob(self):
        est = estimate_fill_probability(1.0, 100.0)
        self.assertGreater(est.maker_fill_probability, 0.9)

    def test_large_queue_low_prob(self):
        est = estimate_fill_probability(100000.0, 1.0)
        self.assertLess(est.maker_fill_probability, 0.01)


if __name__ == "__main__":
    unittest.main()
