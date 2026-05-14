from __future__ import annotations

# pyright: reportMissingImports=false, reportUndefinedVariable=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportPossiblyUnboundVariable=false, reportUninitializedInstanceVariable=false, reportArgumentType=false, reportOperatorIssue=false, reportIndexIssue=false, reportMissingTypeArgument=false, reportOptionalSubscript=false

import asyncio
import sys
import types
import unittest
from datetime import datetime

_smart_order_router_stub = types.ModuleType("execution.smart_order_router")


class _OrderRequest:  # pragma: no cover - test import shim
    pass


class _SmartOrderRouter:  # pragma: no cover - test import shim
    pass


_smart_order_router_stub.OrderRequest = _OrderRequest
_smart_order_router_stub.SmartOrderRouter = _SmartOrderRouter
sys.modules["execution.smart_order_router"] = _smart_order_router_stub

from execution.pov_executor import POVConfig, POVExecutor
from execution.execution_models import ExecutionResult

sys.modules.pop("execution.smart_order_router", None)


class _StaticMarketDataFeed:
    async def get_market_snapshot(self, symbol: str):
        return {
            "symbol": symbol,
            "volume": 1_500.0,
            "bid": 99.95,
            "ask": 100.05,
            "last": 100.0,
            "spread_bps": 10.0,
        }


class _StaticVolatilityProvider:
    async def get_volatility(self, symbol: str, snapshot: dict):
        return 0.015


class _StubRouter:
    def __init__(self):
        self.venues = []
        self.market_data = type("MarketData", (), {"market_data": {}})()

    async def submit_order(self, order_request):
        return type("Plan", (), {"order_request": order_request, "venue_allocations": [{"venue": "TEST", "quantity": order_request.quantity}]})()

    async def execute_plan(self, plan):
        qty = plan.order_request.quantity
        return [
            ExecutionResult(
                execution_id="child-1",
                order_id=plan.order_request.order_id,
                venue="TEST",
                executed_quantity=qty,
                executed_price=100.0,
                execution_time=datetime.utcnow(),
                fees=0.01,
                slippage=0.001,
                market_impact=0.0005,
                status="filled",
            )
        ]


class TestPOVConfig(unittest.TestCase):
    def test_defaults_are_valid(self):
        config = POVConfig()
        self.assertEqual(config.participation_rate, 0.10)
        self.assertEqual(config.urgency, "medium")

    def test_invalid_participation_rate_raises(self):
        with self.assertRaises(ValueError):
            POVConfig(participation_rate=0.04)


class TestPOVExecutor(unittest.TestCase):
    def setUp(self):
        self.executor = POVExecutor(
            config=POVConfig(max_duration_minutes=3, volume_threshold=10.0),
            smart_order_router=_StubRouter(),
            market_data_feed=_StaticMarketDataFeed(),
            volatility_provider=_StaticVolatilityProvider(),
        )

    def test_adaptive_participation_clamps_to_config_range(self):
        rate = self.executor.adaptive_participation(current_volatility=0.05, spread_bps=15.0)
        self.assertGreaterEqual(rate, self.executor.config.min_participation_rate)
        self.assertLessEqual(rate, self.executor.config.max_participation_rate)

    def test_should_accelerate_near_deadline(self):
        self.executor._execution_state["initial_shares"] = 100.0
        self.assertTrue(self.executor.should_accelerate(remaining_time=2, remaining_shares=40.0))
        self.assertFalse(self.executor.should_accelerate(remaining_time=10, remaining_shares=5.0))

    def test_calculate_order_size_respects_volume_threshold(self):
        self.assertEqual(self.executor.calculate_order_size(100, 1, 0), 0.0)
        self.assertGreater(self.executor.calculate_order_size(100, 1000, 0), 0.0)

    def test_execute_pov_returns_fill_results_and_summary(self):
        fills = asyncio.run(self.executor.execute_pov(total_shares=120, symbol="BTC/AUD", side="buy"))
        self.assertTrue(fills)
        self.assertGreater(sum(fill.quantity for fill in fills), 0.0)

        summary = self.executor.get_execution_summary()
        self.assertEqual(summary["symbol"], "BTC/AUD")
        self.assertIn(summary["status"], {"completed", "expired"})
        self.assertGreater(summary["filled_shares"], 0.0)


if __name__ == "__main__":
    unittest.main()
