from __future__ import annotations

import unittest

from argus_live.optimization.policy_adaptation import build_policy_adaptation


class TestPolicyAdaptation(unittest.TestCase):
    def test_policy_adaptation_raises_thresholds_when_needed(self) -> None:
        pack = build_policy_adaptation(
            net_edge_bps=-5.0,       # negative edge
            slippage_bps=8.0,        # exceeds target
            slippage_target_bps=5.0,
            turnover=0.15,           # exceeds target
            turnover_target=0.10,
        )
        self.assertGreaterEqual(len(pack.suggestions), 2)
        params = {s.parameter for s in pack.suggestions}
        self.assertIn("rebalance_threshold", params)
        # At least one of slippage_limit or turnover_limit
        self.assertTrue(params & {"slippage_limit", "turnover_limit"})


if __name__ == "__main__":
    unittest.main()
