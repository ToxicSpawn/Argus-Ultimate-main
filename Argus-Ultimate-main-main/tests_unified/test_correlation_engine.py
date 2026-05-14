from __future__ import annotations

import unittest

from argus_live.portfolio.correlation_engine import build_correlation_report


class TestCorrelationEngine(unittest.TestCase):
    def test_correlation_report_builds_pairs(self) -> None:
        series_a = [0.01, 0.02, -0.01, 0.03, 0.00]
        series_b = [0.01, 0.02, -0.01, 0.03, 0.00]  # identical

        report = build_correlation_report(
            strategy_return_series={"strat_x": series_a, "strat_y": series_b},
            symbol_return_series={"BTC": series_a, "ETH": series_b},
        )

        # Two strategies -> 1 pair
        self.assertEqual(len(report.strategy_pairs), 1)
        self.assertAlmostEqual(report.strategy_pairs[0].correlation, 1.0, places=6)

        # Two symbols -> 1 pair
        self.assertEqual(len(report.symbol_pairs), 1)
        self.assertAlmostEqual(report.symbol_pairs[0].correlation, 1.0, places=6)

        self.assertAlmostEqual(report.average_strategy_correlation, 1.0, places=6)
        self.assertAlmostEqual(report.average_symbol_correlation, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
