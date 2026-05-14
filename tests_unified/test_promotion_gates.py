from __future__ import annotations

import unittest

from adaptive.promotion_gates import PromotionGate


class TestPromotionGates(unittest.TestCase):
    def test_requires_improvement_all_timeframes(self) -> None:
        gate = PromotionGate(min_delta_score=0.1, max_drawdown_pct=10.0, min_trades=3, require_all_timeframes=True)
        baseline = {
            "1h": {"sharpe": 0.0, "sortino": 0.0, "max_drawdown_pct": 1.0, "trades": 5},
            "15m": {"sharpe": 0.0, "sortino": 0.0, "max_drawdown_pct": 1.0, "trades": 5},
        }
        # improve only one timeframe
        cand = {
            "1h": {"sharpe": 1.0, "sortino": 0.0, "max_drawdown_pct": 1.0, "trades": 5},
            "15m": {"sharpe": 0.0, "sortino": 0.0, "max_drawdown_pct": 1.0, "trades": 5},
        }
        d = gate.evaluate(baseline_by_tf=baseline, candidate_by_tf=cand, timeframes=["1h", "15m"])
        self.assertFalse(d.ok)

    def test_passes_when_both_timeframes_improve(self) -> None:
        gate = PromotionGate(min_delta_score=0.1, max_drawdown_pct=10.0, min_trades=3, require_all_timeframes=True)
        baseline = {
            "1h": {"sharpe": 0.0, "sortino": 0.0, "max_drawdown_pct": 1.0, "trades": 5},
            "15m": {"sharpe": 0.0, "sortino": 0.0, "max_drawdown_pct": 1.0, "trades": 5},
        }
        cand = {
            "1h": {"sharpe": 1.0, "sortino": 0.0, "max_drawdown_pct": 1.0, "trades": 5},
            "15m": {"sharpe": 1.0, "sortino": 0.0, "max_drawdown_pct": 1.0, "trades": 5},
        }
        d = gate.evaluate(baseline_by_tf=baseline, candidate_by_tf=cand, timeframes=["1h", "15m"])
        self.assertTrue(d.ok)

