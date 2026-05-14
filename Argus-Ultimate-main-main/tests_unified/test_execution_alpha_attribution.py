from __future__ import annotations

import unittest

from argus_live.evidence.execution_alpha_attribution import attribute_execution_alpha


class TestExecutionAlphaAttribution(unittest.TestCase):
    def test_positive_alpha(self):
        attr = attribute_execution_alpha(expected_slippage_bps=5.0, realized_slippage_bps=3.0)
        self.assertAlmostEqual(attr.execution_alpha_bps, 2.0, places=4)
        self.assertEqual(attr.reason, "positive_alpha")

    def test_negative_alpha(self):
        attr = attribute_execution_alpha(expected_slippage_bps=3.0, realized_slippage_bps=5.0)
        self.assertAlmostEqual(attr.execution_alpha_bps, -2.0, places=4)
        self.assertEqual(attr.reason, "negative_alpha")

    def test_neutral_alpha(self):
        attr = attribute_execution_alpha(expected_slippage_bps=5.0, realized_slippage_bps=5.0)
        self.assertAlmostEqual(attr.execution_alpha_bps, 0.0, places=4)
        self.assertEqual(attr.reason, "neutral")

    def test_fields_preserved(self):
        attr = attribute_execution_alpha(expected_slippage_bps=10.0, realized_slippage_bps=7.5)
        self.assertAlmostEqual(attr.expected_slippage_bps, 10.0)
        self.assertAlmostEqual(attr.realized_slippage_bps, 7.5)
        self.assertAlmostEqual(attr.execution_alpha_bps, 2.5)


if __name__ == "__main__":
    unittest.main()
