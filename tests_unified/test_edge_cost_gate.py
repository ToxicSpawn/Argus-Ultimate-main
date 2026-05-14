from __future__ import annotations

import unittest
from types import SimpleNamespace

from unified_execution_engine import ExecutionRiskManager


class _Sig(SimpleNamespace):
    pass


class TestEdgeCostGate(unittest.TestCase):
    def test_blocks_when_edge_below_cost(self) -> None:
        cfg = SimpleNamespace(
            run_mode="paper",
            min_signal_confidence=0.0,
            max_position_size_aud=999999.0,
            aud_to_usd=0.65,
            kraken_taker_fee=0.0026,
            coinbase_taker_fee=0.005,
            slippage_pct=0.002,
            take_profit_pct=0.01,
            edge_cost_gate_enabled=True,
            edge_cost_gate_modes=["paper", "backtest"],
            edge_cost_gate_buffer_mult=1.25,
            edge_cost_gate_min_edge_pct=0.0,
            edge_cost_gate_fee_mult=2.0,
            edge_cost_gate_slippage_mult=2.0,
        )
        rm = ExecutionRiskManager(cfg)
        s = _Sig(symbol="BTC/USD", action="BUY", confidence=1.0, quantity=0.001, entry_price=50000.0, take_profit=None)
        self.assertFalse(rm.approve_signal(s))

    def test_allows_when_edge_above_cost(self) -> None:
        cfg = SimpleNamespace(
            run_mode="paper",
            min_signal_confidence=0.0,
            max_position_size_aud=999999.0,
            aud_to_usd=0.65,
            kraken_taker_fee=0.0005,
            coinbase_taker_fee=0.0005,
            slippage_pct=0.0005,
            take_profit_pct=0.03,
            edge_cost_gate_enabled=True,
            edge_cost_gate_modes=["paper", "backtest"],
            edge_cost_gate_buffer_mult=1.1,
            edge_cost_gate_min_edge_pct=0.0,
            edge_cost_gate_fee_mult=2.0,
            edge_cost_gate_slippage_mult=2.0,
        )
        rm = ExecutionRiskManager(cfg)
        s = _Sig(symbol="BTC/USD", action="BUY", confidence=1.0, quantity=0.001, entry_price=50000.0, take_profit=None)
        self.assertTrue(rm.approve_signal(s))

