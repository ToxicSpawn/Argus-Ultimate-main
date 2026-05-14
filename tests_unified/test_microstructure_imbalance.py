from __future__ import annotations

import unittest

from argus_live.execution.microstructure_imbalance import (
    OrderBookSnapshot,
    Pressure,
    compute_imbalance,
)


class TestComputeImbalance(unittest.TestCase):
    def test_buy_pressure(self):
        snap = OrderBookSnapshot(bid_size=200, ask_size=50, bid_price=100.0, ask_price=100.1)
        sig = compute_imbalance(snap)
        self.assertEqual(sig.pressure, Pressure.BUY_PRESSURE)
        self.assertGreater(sig.imbalance, 0.2)

    def test_sell_pressure(self):
        snap = OrderBookSnapshot(bid_size=50, ask_size=200, bid_price=100.0, ask_price=100.1)
        sig = compute_imbalance(snap)
        self.assertEqual(sig.pressure, Pressure.SELL_PRESSURE)
        self.assertLess(sig.imbalance, -0.2)

    def test_balanced(self):
        snap = OrderBookSnapshot(bid_size=100, ask_size=100, bid_price=100.0, ask_price=100.1)
        sig = compute_imbalance(snap)
        self.assertEqual(sig.pressure, Pressure.BALANCED)
        self.assertAlmostEqual(sig.imbalance, 0.0, places=4)

    def test_spread_bps(self):
        snap = OrderBookSnapshot(bid_size=100, ask_size=100, bid_price=100.0, ask_price=101.0)
        sig = compute_imbalance(snap)
        # spread = 1.0, mid = 100.5, bps = 1/100.5 * 10000 ~ 99.5
        self.assertGreater(sig.spread_bps, 90)

    def test_zero_total(self):
        snap = OrderBookSnapshot(bid_size=0, ask_size=0, bid_price=100.0, ask_price=100.1)
        sig = compute_imbalance(snap)
        self.assertEqual(sig.pressure, Pressure.BALANCED)
        self.assertEqual(sig.reason, "no_liquidity")


if __name__ == "__main__":
    unittest.main()
