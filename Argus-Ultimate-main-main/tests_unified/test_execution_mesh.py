from __future__ import annotations

import unittest
from types import SimpleNamespace

from execution.execution_mesh import ExecutionMeshCoordinator, ExecutionMeshError


class TestExecutionMesh(unittest.IsolatedAsyncioTestCase):
    async def test_isolated_lanes_by_symbol(self) -> None:
        mesh = ExecutionMeshCoordinator(
            max_lanes=8,
            max_queue_per_lane=16,
            batch_size=2,
            parallel_lanes=False,
            halt_on_lane_error=True,
        )
        calls: list[tuple[str, int, str]] = []

        async def _execute(symbol, lane_signals, corr_id):
            calls.append((symbol, len(lane_signals), corr_id))
            return [{"status": "filled", "symbol": symbol, "n": len(lane_signals)}]

        signals = [
            SimpleNamespace(symbol="BTC/USD", action="BUY", quantity=0.01, entry_price=100.0),
            SimpleNamespace(symbol="ETH/USD", action="BUY", quantity=0.02, entry_price=200.0),
            SimpleNamespace(symbol="BTC/USD", action="SELL", quantity=0.03, entry_price=110.0),
        ]
        results, summary = await mesh.execute_cycle(signals, execute_fn=_execute, correlation_id="corr_a")

        self.assertEqual(summary["accepted_signals"], 3)
        self.assertEqual(summary["lanes_active"], 2)
        self.assertEqual(len(results), 2)
        self.assertTrue(any(c[0] == "BTC/USD" for c in calls))
        self.assertTrue(any(c[0] == "ETH/USD" for c in calls))

    async def test_queue_drop_when_lane_full(self) -> None:
        mesh = ExecutionMeshCoordinator(
            max_lanes=4,
            max_queue_per_lane=1,
            batch_size=1,
            parallel_lanes=False,
            halt_on_lane_error=True,
        )

        async def _execute(symbol, lane_signals, corr_id):
            _ = corr_id
            return [{"status": "filled", "symbol": symbol, "n": len(lane_signals)}]

        signals = [
            {"symbol": "BTC/USD", "action": "BUY", "quantity": 0.01, "entry_price": 100.0},
            {"symbol": "BTC/USD", "action": "BUY", "quantity": 0.02, "entry_price": 101.0},
            {"symbol": "BTC/USD", "action": "SELL", "quantity": 0.03, "entry_price": 99.0},
        ]
        results, summary = await mesh.execute_cycle(signals, execute_fn=_execute, correlation_id="corr_b")

        self.assertEqual(summary["accepted_signals"], 1)
        self.assertEqual(summary["dropped_signals"], 2)
        self.assertEqual(len(results), 1)

    async def test_fail_closed_on_lane_error(self) -> None:
        mesh = ExecutionMeshCoordinator(
            max_lanes=8,
            max_queue_per_lane=8,
            batch_size=4,
            parallel_lanes=True,
            halt_on_lane_error=True,
        )

        async def _execute(symbol, lane_signals, corr_id):
            _ = (lane_signals, corr_id)
            if symbol == "ETH/USD":
                raise RuntimeError("simulated lane failure")
            return [{"status": "filled", "symbol": symbol}]

        signals = [
            {"symbol": "BTC/USD", "action": "BUY", "quantity": 0.01, "entry_price": 100.0},
            {"symbol": "ETH/USD", "action": "BUY", "quantity": 0.01, "entry_price": 200.0},
        ]
        with self.assertRaises(ExecutionMeshError):
            await mesh.execute_cycle(signals, execute_fn=_execute, correlation_id="corr_c")


if __name__ == "__main__":
    unittest.main()
