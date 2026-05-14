from __future__ import annotations

import unittest

from argus_live.execution.adaptive_slice_planner import build_adaptive_slice_plan
from argus_live.execution.execution_alpha_engine import Aggression, ExecutionAlphaDecision


class TestAdaptiveSlicePlanner(unittest.TestCase):
    def test_no_slice_single(self):
        alpha = ExecutionAlphaDecision(
            aggression=Aggression.MEDIUM,
            maker_preferred=True,
            should_slice=False,
            wait_preferred=False,
            reason="test",
        )
        plan = build_adaptive_slice_plan(10.0, alpha, volatility_bps=50.0)
        self.assertEqual(plan.slice_count, 1)
        self.assertEqual(plan.slice_quantity, 10.0)

    def test_should_slice_high_vol(self):
        alpha = ExecutionAlphaDecision(
            aggression=Aggression.LOW,
            maker_preferred=False,
            should_slice=True,
            wait_preferred=True,
            reason="test",
        )
        plan = build_adaptive_slice_plan(10.0, alpha, volatility_bps=150.0)
        self.assertGreater(plan.slice_count, 1)
        self.assertEqual(plan.slice_count, 4)
        self.assertAlmostEqual(plan.slice_quantity, 2.5, places=4)

    def test_should_slice_low_vol(self):
        alpha = ExecutionAlphaDecision(
            aggression=Aggression.MEDIUM,
            maker_preferred=False,
            should_slice=True,
            wait_preferred=False,
            reason="test",
        )
        plan = build_adaptive_slice_plan(10.0, alpha, volatility_bps=50.0)
        self.assertEqual(plan.slice_count, 2)

    def test_spacing_by_aggression(self):
        for aggr, expected_spacing in [
            (Aggression.HIGH, 5.0),
            (Aggression.MEDIUM, 10.0),
            (Aggression.LOW, 20.0),
        ]:
            alpha = ExecutionAlphaDecision(
                aggression=aggr,
                maker_preferred=False,
                should_slice=True,
                wait_preferred=False,
                reason="test",
            )
            plan = build_adaptive_slice_plan(10.0, alpha, volatility_bps=50.0)
            self.assertEqual(plan.spacing_seconds, expected_spacing, msg=f"Failed for {aggr}")


if __name__ == "__main__":
    unittest.main()
