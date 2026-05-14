from __future__ import annotations

import unittest

from argus_live.execution.execution_alpha_engine import (
    Aggression,
    build_execution_alpha_decision,
)
from argus_live.execution.fill_probability_model import FillProbabilityEstimate
from argus_live.execution.liquidity_fade_detector import LiquidityFadeSignal
from argus_live.execution.microstructure_imbalance import ImbalanceSignal, Pressure


class TestExecutionAlphaEngine(unittest.TestCase):
    def _make_imbalance(self, pressure: Pressure) -> ImbalanceSignal:
        return ImbalanceSignal(imbalance=0.5, spread_bps=10.0, pressure=pressure, reason="test")

    def _make_fill_prob(self, prob: float) -> FillProbabilityEstimate:
        return FillProbabilityEstimate(maker_fill_probability=prob, expected_wait_seconds=5.0, reason="test")

    def _make_fade(self, suspicious: bool) -> LiquidityFadeSignal:
        return LiquidityFadeSignal(fade_risk=0.8 if suspicious else 0.2, suspicious=suspicious, reason="test")

    def test_buy_pressure_low_fill_high_aggression(self):
        dec = build_execution_alpha_decision(
            side="BUY",
            imbalance=self._make_imbalance(Pressure.BUY_PRESSURE),
            fill_prob=self._make_fill_prob(0.3),
            fade=self._make_fade(False),
            volatility_bps=30.0,
        )
        self.assertEqual(dec.aggression, Aggression.HIGH)

    def test_fade_suspicious_overrides(self):
        dec = build_execution_alpha_decision(
            side="BUY",
            imbalance=self._make_imbalance(Pressure.BUY_PRESSURE),
            fill_prob=self._make_fill_prob(0.3),
            fade=self._make_fade(True),
            volatility_bps=30.0,
        )
        self.assertEqual(dec.aggression, Aggression.LOW)
        self.assertTrue(dec.should_slice)
        self.assertTrue(dec.wait_preferred)

    def test_buy_sell_pressure_low_aggression(self):
        dec = build_execution_alpha_decision(
            side="BUY",
            imbalance=self._make_imbalance(Pressure.SELL_PRESSURE),
            fill_prob=self._make_fill_prob(0.8),
            fade=self._make_fade(False),
            volatility_bps=30.0,
        )
        self.assertEqual(dec.aggression, Aggression.LOW)
        self.assertTrue(dec.maker_preferred)
        self.assertTrue(dec.wait_preferred)

    def test_sell_pressure_low_fill_high_aggression(self):
        dec = build_execution_alpha_decision(
            side="SELL",
            imbalance=self._make_imbalance(Pressure.SELL_PRESSURE),
            fill_prob=self._make_fill_prob(0.2),
            fade=self._make_fade(False),
            volatility_bps=30.0,
        )
        self.assertEqual(dec.aggression, Aggression.HIGH)

    def test_default_medium(self):
        dec = build_execution_alpha_decision(
            side="BUY",
            imbalance=self._make_imbalance(Pressure.BALANCED),
            fill_prob=self._make_fill_prob(0.8),
            fade=self._make_fade(False),
            volatility_bps=30.0,
        )
        self.assertEqual(dec.aggression, Aggression.MEDIUM)


if __name__ == "__main__":
    unittest.main()
