from __future__ import annotations

import unittest

from argus_live.execution.liquidity_fade_detector import detect_liquidity_fade


class TestLiquidityFadeDetector(unittest.TestCase):
    def test_low_exec_high_cancel_suspicious(self):
        sig = detect_liquidity_fade(
            displayed_notional=1000.0,
            executed_notional=100.0,
            cancel_rate=0.5,
        )
        self.assertTrue(sig.suspicious)
        self.assertGreater(sig.fade_risk, 0.7)

    def test_good_execution_not_suspicious(self):
        sig = detect_liquidity_fade(
            displayed_notional=1000.0,
            executed_notional=900.0,
            cancel_rate=0.05,
        )
        self.assertFalse(sig.suspicious)
        self.assertLess(sig.fade_risk, 0.7)

    def test_zero_displayed(self):
        sig = detect_liquidity_fade(
            displayed_notional=0.0,
            executed_notional=0.0,
            cancel_rate=0.0,
        )
        self.assertFalse(sig.suspicious)
        self.assertEqual(sig.reason, "no_displayed_liquidity")

    def test_full_execution_zero_cancel(self):
        sig = detect_liquidity_fade(
            displayed_notional=1000.0,
            executed_notional=1000.0,
            cancel_rate=0.0,
        )
        self.assertAlmostEqual(sig.fade_risk, 0.0, places=4)
        self.assertFalse(sig.suspicious)


if __name__ == "__main__":
    unittest.main()
