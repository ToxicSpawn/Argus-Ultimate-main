from __future__ import annotations

import unittest

from argus_live.evidence.portfolio_edge import compute_portfolio_edge


class TestPortfolioEdge(unittest.TestCase):
    def test_portfolio_edge_net_is_reduced_by_costs(self) -> None:
        report = compute_portfolio_edge(
            gross_edge_bps=20.0,
            total_fee_bps=2.0,
            total_slippage_bps=3.0,
            turnover_penalty_bps=4.0,
        )
        self.assertAlmostEqual(report.net_portfolio_edge_bps, 11.0, places=6)


if __name__ == "__main__":
    unittest.main()
