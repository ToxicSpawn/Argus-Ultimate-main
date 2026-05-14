from __future__ import annotations

import unittest
from types import SimpleNamespace

from adaptive.adaptive_risk_controller import AdaptiveRiskController
from unified_execution_engine import ExecutionRiskManager


class _Sig(SimpleNamespace):
    pass


class TestAdaptiveRiskController(unittest.TestCase):
    def test_hard_caps_respected(self) -> None:
        cfg = SimpleNamespace(
            max_total_exposure_pct=0.40,
            max_daily_loss_pct=0.02,
            max_drawdown_pct=0.10,
            min_signal_confidence=0.75,
            max_concurrent_signals=2,
            edge_cost_gate_buffer_mult=1.25,
            edge_cost_gate_min_edge_pct=0.0,
            stop_loss_pct=0.01,
            take_profit_pct=0.03,
        )
        arc = AdaptiveRiskController(config=cfg)
        prof = arc.update_profile(drawdown_pct=9.0, daily_return_pct=-2.0, last_regime_by_symbol={"BTC/USD": "high_vol"})
        self.assertLessEqual(prof.max_total_exposure_pct, 0.40)
        self.assertGreaterEqual(prof.max_total_exposure_pct, 0.05)
        self.assertGreaterEqual(prof.min_signal_confidence, 0.45)
        self.assertLessEqual(prof.min_signal_confidence, 0.95)

    def test_strategy_cooldown_blocks_in_live_mode(self) -> None:
        """Strategy cooldown only enforced in live mode with >= 5 cooldown cycles."""
        cfg = SimpleNamespace(
            run_mode="live",
            min_signal_confidence=0.0,
            max_position_size_aud=999999.0,
            aud_to_usd=0.65,
            kraken_taker_fee=0.0005,
            coinbase_taker_fee=0.0005,
            slippage_pct=0.0001,
            edge_cost_gate_enabled=False,
        )
        rm = ExecutionRiskManager(cfg)
        setattr(cfg, "_adaptive_risk_profile", {"strategy_cooldown_cycles": {"momentum": 5}})
        s = _Sig(symbol="BTC/USD", action="BUY", confidence=1.0, quantity=0.001, entry_price=50000.0, source_strategy="momentum")
        self.assertFalse(rm.approve_signal(s))
        self.assertEqual((rm.last_rejection or {}).get("reason"), "strategy_cooldown")

    def test_strategy_cooldown_skipped_in_paper_mode(self) -> None:
        """Strategy cooldown is skipped in paper mode to avoid blocking from pessimistic fill sim."""
        cfg = SimpleNamespace(
            run_mode="paper",
            min_signal_confidence=0.0,
            max_position_size_aud=999999.0,
            aud_to_usd=0.65,
            kraken_taker_fee=0.0005,
            coinbase_taker_fee=0.0005,
            slippage_pct=0.0001,
            edge_cost_gate_enabled=False,
        )
        rm = ExecutionRiskManager(cfg)
        setattr(cfg, "_adaptive_risk_profile", {"strategy_cooldown_cycles": {"momentum": 10}})
        s = _Sig(symbol="BTC/USD", action="BUY", confidence=1.0, quantity=0.001, entry_price=50000.0, source_strategy="momentum")
        # In paper mode, cooldown should NOT block the signal
        self.assertTrue(rm.approve_signal(s))

