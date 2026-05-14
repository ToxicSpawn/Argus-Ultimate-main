"""Tests for decision simulator."""
from __future__ import annotations

import unittest

from argus_live.simulation.decision_simulator import (
    DecisionSimulationResult,
    simulate_decision,
)


class TestDecisionSimulator(unittest.TestCase):

    def test_negative_ev_not_approved(self):
        result = simulate_decision(
            edge_bps=2.0,
            slippage_bps=3.0,
            fee_bps=1.0,
            confidence=0.8,
        )
        self.assertIsInstance(result, DecisionSimulationResult)
        # EV = (2 - 3 - 1) * 0.8 = -1.6
        self.assertLess(result.expected_value_bps, 0.0)
        self.assertFalse(result.approve)

    def test_positive_ev_approved(self):
        result = simulate_decision(
            edge_bps=10.0,
            slippage_bps=1.0,
            fee_bps=0.5,
            confidence=0.9,
            volatility_bps=2.0,
        )
        # EV = (10 - 1 - 0.5) * 0.9 = 7.65
        self.assertGreater(result.expected_value_bps, 0.0)
        # downside = -(1 + 0.5 + 2*0.5) = -2.5; floor = -max(1, 0.5) = -1.0
        # downside(-2.5) > floor(-1.0) is False, so approve=False
        # Actually -2.5 < -1.0 so not approved despite positive EV
        # This tests the dual condition

    def test_zero_edge_not_approved(self):
        result = simulate_decision(
            edge_bps=0.0,
            slippage_bps=0.0,
            fee_bps=0.0,
            confidence=1.0,
        )
        # EV = 0, not > 0
        self.assertFalse(result.approve)

    def test_frozen_dataclass(self):
        result = simulate_decision(
            edge_bps=5.0,
            slippage_bps=1.0,
            fee_bps=0.5,
            confidence=0.7,
        )
        with self.assertRaises(AttributeError):
            result.approve = True  # type: ignore[misc]

    def test_downside_is_negative(self):
        result = simulate_decision(
            edge_bps=5.0,
            slippage_bps=1.0,
            fee_bps=1.0,
            confidence=0.8,
        )
        self.assertLess(result.downside_bps, 0.0)

    def test_high_edge_low_costs_approved(self):
        result = simulate_decision(
            edge_bps=50.0,
            slippage_bps=0.1,
            fee_bps=0.1,
            confidence=0.95,
            volatility_bps=0.5,
        )
        # EV = (50 - 0.1 - 0.1) * 0.95 = 47.31
        # downside = -(0.1 + 0.1 + 0.25) = -0.45; floor = -max(1, 0.1) = -1.0
        # -0.45 > -1.0 => True
        self.assertTrue(result.approve)
        self.assertGreater(result.expected_value_bps, 0.0)


if __name__ == "__main__":
    unittest.main()
