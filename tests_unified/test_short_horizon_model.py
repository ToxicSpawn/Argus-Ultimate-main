"""Tests for short horizon model."""
from __future__ import annotations

import unittest

from argus_live.prediction.short_horizon_model import (
    ShortHorizonForecast,
    forecast_short_horizon,
)


class TestShortHorizonModel(unittest.TestCase):

    def test_positive_drift_from_imbalance_and_momentum(self):
        fc = forecast_short_horizon(
            imbalance=0.5,
            volatility_bps=10.0,
            spread_bps=5.0,
            momentum_bps=4.0,
        )
        self.assertIsInstance(fc, ShortHorizonForecast)
        # drift = 0.5*5 + 4.0*0.5 = 2.5 + 2.0 = 4.5
        self.assertGreater(fc.expected_drift_bps, 0.0)
        self.assertAlmostEqual(fc.expected_drift_bps, 4.5, places=5)

    def test_negative_drift(self):
        fc = forecast_short_horizon(
            imbalance=-0.8,
            volatility_bps=10.0,
            spread_bps=5.0,
            momentum_bps=-6.0,
        )
        # drift = -0.8*5 + -6.0*0.5 = -4.0 + -3.0 = -7.0
        self.assertLess(fc.expected_drift_bps, 0.0)

    def test_spread_change_clamped(self):
        fc = forecast_short_horizon(
            imbalance=100.0,
            volatility_bps=10.0,
            spread_bps=5.0,
            momentum_bps=0.0,
        )
        # spread_change should be clamped to spread_bps
        self.assertLessEqual(fc.expected_spread_change_bps, 5.0)
        self.assertGreaterEqual(fc.expected_spread_change_bps, -5.0)

    def test_volatility_includes_imbalance(self):
        fc = forecast_short_horizon(
            imbalance=0.5,
            volatility_bps=10.0,
            spread_bps=5.0,
            momentum_bps=0.0,
        )
        # expected_volatility = 10 + |0.5|*2 = 11.0
        self.assertAlmostEqual(fc.expected_volatility_bps, 11.0, places=5)

    def test_confidence_bounded(self):
        fc = forecast_short_horizon(
            imbalance=0.0,
            volatility_bps=0.0,
            spread_bps=1.0,
            momentum_bps=0.0,
        )
        self.assertGreaterEqual(fc.confidence, 0.1)
        self.assertLessEqual(fc.confidence, 1.0)

    def test_frozen_dataclass(self):
        fc = forecast_short_horizon(
            imbalance=0.1,
            volatility_bps=5.0,
            spread_bps=2.0,
            momentum_bps=1.0,
        )
        with self.assertRaises(AttributeError):
            fc.expected_drift_bps = 0.0  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
