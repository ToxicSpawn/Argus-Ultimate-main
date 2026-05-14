"""Tests for regime transition engine."""
from __future__ import annotations

import unittest

from argus_live.regime.regime_transition_engine import (
    RegimeTransitionSignal,
    predict_regime_transition,
)


class TestRegimeTransitionEngine(unittest.TestCase):

    def test_high_stress_returns_stressed(self):
        sig = predict_regime_transition(
            volatility_slope=2.0,
            spread_widening_bps_per_cycle=1.5,
            imbalance_acceleration=0.5,
            correlation_tightening=1.0,
        )
        self.assertIsInstance(sig, RegimeTransitionSignal)
        self.assertEqual(sig.next_regime, "STRESSED")
        self.assertGreater(sig.probability, 0.0)
        self.assertLessEqual(sig.probability, 1.0)

    def test_low_stress_returns_trending(self):
        sig = predict_regime_transition(
            volatility_slope=0.0,
            spread_widening_bps_per_cycle=0.0,
            imbalance_acceleration=1.0,
            correlation_tightening=0.0,
        )
        # With only imbalance, trend_score should dominate
        self.assertEqual(sig.next_regime, "TRENDING")

    def test_probability_clamped(self):
        sig = predict_regime_transition(
            volatility_slope=10.0,
            spread_widening_bps_per_cycle=10.0,
            imbalance_acceleration=10.0,
            correlation_tightening=10.0,
        )
        self.assertLessEqual(sig.probability, 1.0)
        self.assertGreaterEqual(sig.probability, 0.0)

    def test_frozen_dataclass(self):
        sig = predict_regime_transition(
            volatility_slope=1.0,
            spread_widening_bps_per_cycle=0.5,
            imbalance_acceleration=0.2,
            correlation_tightening=0.3,
        )
        with self.assertRaises(AttributeError):
            sig.next_regime = "OTHER"  # type: ignore[misc]

    def test_horizon_positive(self):
        sig = predict_regime_transition(
            volatility_slope=2.0,
            spread_widening_bps_per_cycle=1.5,
            imbalance_acceleration=0.5,
            correlation_tightening=1.0,
        )
        self.assertGreaterEqual(sig.horizon_cycles, 1)


if __name__ == "__main__":
    unittest.main()
