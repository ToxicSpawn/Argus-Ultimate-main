from __future__ import annotations

import unittest

from argus_live.portfolio.regime_allocator import MarketRegime, build_regime_policy


class TestRegimeAllocator(unittest.TestCase):
    def test_high_vol_regime_reduces_exposure(self) -> None:
        policy = build_regime_policy(MarketRegime.HIGH_VOL)
        self.assertLess(policy.gross_exposure_multiplier, 1.0)
        self.assertEqual(policy.regime, MarketRegime.HIGH_VOL)


if __name__ == "__main__":
    unittest.main()
