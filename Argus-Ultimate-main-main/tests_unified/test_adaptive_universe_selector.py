from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from adaptive.universe_selector import AdaptiveUniverseSelector


class TestAdaptiveUniverseSelector(unittest.TestCase):
    def test_selects_and_holds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "u.json"
            us = AdaptiveUniverseSelector(persist_path=str(p), max_active=2, min_hold_cycles=5)
            cands = ["BTC/USD", "ETH/USD", "SOL/USD"]
            active0 = us.select_active(candidate_symbols=cands, cycle_id=0)
            self.assertEqual(len(active0), 2)
            # add performance for SOL to win, but hold should prevent immediate swap
            us.observe_trade_close(symbol="SOL/USD", pnl_pct=5.0)
            active1 = us.select_active(candidate_symbols=cands, cycle_id=1)
            self.assertEqual(active1, active0)
            # after hold expires, SOL can be promoted
            active6 = us.select_active(candidate_symbols=cands, cycle_id=10)
            self.assertIn("SOL/USD", active6)

