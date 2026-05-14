"""Tests for London VPS execution relay."""
import time
import asyncio
import unittest
from execution.london_relay import LondonRelay, ExecutionIntent, RelayFill


class TestLondonRelay(unittest.TestCase):
    def _make_intent(self, **kwargs):
        defaults = {
            "intent_id": "test_1",
            "symbol": "BTC/USD",
            "side": "buy",
            "order_type": "limit",
            "quantity": 0.01,
            "price": 50000.0,
            "stop_price": None,
            "strategy": "momentum",
            "urgency": "medium",
            "max_slippage_bps": 5.0,
            "signal_timestamp": time.time() - 0.3,
            "sent_timestamp": time.time(),
        }
        defaults.update(kwargs)
        return ExecutionIntent(**defaults)

    def test_execute_intent(self):
        relay = LondonRelay()
        intent = self._make_intent()
        fill = asyncio.run(relay.execute_intent(intent))
        self.assertIsNotNone(fill)
        self.assertIsInstance(fill, RelayFill)
        self.assertEqual(fill.symbol, "BTC/USD")
        self.assertGreater(fill.filled_qty, 0)

    def test_position_tracking(self):
        relay = LondonRelay()
        intent = self._make_intent(quantity=0.05)
        asyncio.run(relay.execute_intent(intent))
        pos = relay.get_position("BTC/USD")
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos.quantity, 0.05)

    def test_sell_reduces_position(self):
        relay = LondonRelay()
        asyncio.run(relay.execute_intent(self._make_intent(side="buy", quantity=0.1)))
        asyncio.run(relay.execute_intent(self._make_intent(intent_id="t2", side="sell", quantity=0.04)))
        pos = relay.get_position("BTC/USD")
        self.assertAlmostEqual(pos.quantity, 0.06)

    def test_max_pending_rejects(self):
        relay = LondonRelay(max_pending_orders=0)
        fill = asyncio.run(relay.execute_intent(self._make_intent()))
        self.assertIsNone(fill)

    def test_latency_tracking(self):
        relay = LondonRelay()
        asyncio.run(relay.execute_intent(self._make_intent()))
        stats = relay.get_stats()
        self.assertEqual(stats["total_orders"], 1)
        self.assertEqual(stats["total_fills"], 1)
        self.assertGreaterEqual(stats["avg_relay_latency_ms"], 0)

    def test_update_prices(self):
        relay = LondonRelay()
        asyncio.run(relay.execute_intent(self._make_intent(price=50000, quantity=0.1)))
        relay.update_prices({"BTC/USD": 51000})
        pos = relay.get_position("BTC/USD")
        self.assertGreater(pos.unrealized_pnl, 0)

    def test_signal_latency(self):
        intent = self._make_intent(signal_timestamp=time.time() - 0.5)
        self.assertGreater(intent.latency_ms(), 400)

    def test_get_all_positions(self):
        relay = LondonRelay()
        asyncio.run(relay.execute_intent(self._make_intent(symbol="BTC/USD")))
        asyncio.run(relay.execute_intent(self._make_intent(intent_id="t2", symbol="ETH/USD")))
        positions = relay.get_all_positions()
        self.assertIn("BTC/USD", positions)
        self.assertIn("ETH/USD", positions)


if __name__ == "__main__":
    unittest.main()
