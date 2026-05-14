"""
tests_unified/test_tier2_order_mgmt.py
========================================
Tier 2 Order Management — unit test suite.

Covers:
  - OrderStateMachine FSM transitions and fill accounting
  - OrderBook net position and session PnL
  - MultiLegExecutor simultaneous submission and partial-failure rollback
  - QueuePositionTracker registration and cancel-and-rejoin heuristic
  - IcebergExecutor slice sizing and fill accumulation
"""

from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, call
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from execution.order_state_machine import (
    OrderBook,
    OrderEvent,
    OrderState,
    OrderStateMachine,
    InvalidTransition,
)
from execution.multi_leg_executor import LegSpec, MultiLegExecutor
from execution.queue_position_tracker import QueuePositionTracker
from execution.iceberg_executor import IcebergExecutor, IcebergState, IcebergStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_osm(
    order_id: str = "ORD-001",
    symbol: str = "BTC-USD",
    side: str = "buy",
    price: float = 50_000.0,
    size: float = 1.0,
    exchange: str = "binance",
) -> OrderStateMachine:
    return OrderStateMachine(order_id, symbol, side, price, size, exchange)


def make_exchange_client(
    order_id: str = "EX-001",
    fill_price: float = 50_000.0,
    fill_size: float = 1.0,
    should_fail: bool = False,
) -> MagicMock:
    """Return a mock exchange client with async submit_order / cancel_order."""
    client = MagicMock()
    if should_fail:
        client.submit_order = AsyncMock(side_effect=RuntimeError("Exchange error"))
    else:
        client.submit_order = AsyncMock(
            return_value={
                "order_id":   order_id,
                "fill_price": fill_price,
                "filled_size": fill_size,
                "status":     "open",
            }
        )
    client.cancel_order = AsyncMock(return_value=True)
    client.get_order    = AsyncMock(
        return_value={
            "order_id":   order_id,
            "filled_size": fill_size,
            "fill_price":  fill_price,
            "status":      "filled",
        }
    )
    client.get_ticker = AsyncMock(
        return_value={"bid": fill_price - 1, "ask": fill_price + 1}
    )
    return client


def make_order_book_processor(
    level_size: float = 100_000.0,
    l3_queue: list = None,
) -> MagicMock:
    obp = MagicMock()
    obp.get_level_size = MagicMock(return_value=level_size)
    obp.get_l3_queue   = MagicMock(return_value=l3_queue)
    return obp


def run_async(coro):
    """Helper to run a coroutine in a new event loop (for tests)."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ===========================================================================
# OrderStateMachine tests
# ===========================================================================

class TestOrderStateMachineHappyPath(unittest.TestCase):
    """PENDING → OPEN → PARTIALLY_FILLED → FILLED happy path."""

    def test_happy_path(self):
        osm = make_osm(size=10.0)

        # PENDING → OPEN on SUBMIT
        state = osm.transition(OrderEvent.SUBMIT)
        self.assertEqual(state, OrderState.OPEN)
        self.assertEqual(osm.state, OrderState.OPEN)

        # OPEN → PARTIALLY_FILLED on PARTIAL_FILL
        state = osm.transition(OrderEvent.PARTIAL_FILL, fill_size=4.0, fill_price=50_100.0)
        self.assertEqual(state, OrderState.PARTIALLY_FILLED)
        self.assertAlmostEqual(osm.filled_size, 4.0)
        self.assertAlmostEqual(osm.remaining_size, 6.0)

        # PARTIALLY_FILLED → FILLED on FULL_FILL
        state = osm.transition(OrderEvent.FULL_FILL, fill_size=6.0, fill_price=50_200.0)
        self.assertEqual(state, OrderState.FILLED)
        self.assertAlmostEqual(osm.filled_size, 10.0)
        self.assertAlmostEqual(osm.remaining_size, 0.0)

    def test_initial_state_is_pending(self):
        osm = make_osm()
        self.assertEqual(osm.state, OrderState.PENDING)

    def test_age_ms_increases(self):
        osm = make_osm()
        t0 = osm.age_ms()
        time.sleep(0.005)
        t1 = osm.age_ms()
        self.assertGreater(t1, t0)

    def test_to_dict_keys(self):
        osm = make_osm()
        d = osm.to_dict()
        for key in ("order_id", "symbol", "side", "state", "filled_size", "remaining_size"):
            self.assertIn(key, d)

    def test_open_to_cancelled(self):
        osm = make_osm()
        osm.transition(OrderEvent.SUBMIT)
        osm.transition(OrderEvent.CANCEL_REQ)
        self.assertTrue(osm.cancel_pending)
        state = osm.transition(OrderEvent.CANCEL_ACK)
        self.assertEqual(state, OrderState.CANCELLED)

    def test_open_to_rejected(self):
        osm = make_osm()
        osm.transition(OrderEvent.SUBMIT)
        state = osm.transition(OrderEvent.REJECT)
        self.assertEqual(state, OrderState.REJECTED)

    def test_open_to_expired(self):
        osm = make_osm()
        osm.transition(OrderEvent.SUBMIT)
        state = osm.transition(OrderEvent.EXPIRE)
        self.assertEqual(state, OrderState.EXPIRED)


class TestOrderStateMachineInvalidTransition(unittest.TestCase):
    """Terminal states must reject further events."""

    def test_filled_rejects_cancel_req(self):
        osm = make_osm(size=1.0)
        osm.transition(OrderEvent.SUBMIT)
        osm.transition(OrderEvent.FULL_FILL, fill_size=1.0, fill_price=50_000.0)
        self.assertEqual(osm.state, OrderState.FILLED)

        with self.assertRaises(InvalidTransition) as ctx:
            osm.transition(OrderEvent.CANCEL_REQ)

        exc = ctx.exception
        self.assertEqual(exc.order_id, "ORD-001")
        self.assertEqual(exc.current_state, OrderState.FILLED)
        self.assertEqual(exc.event, OrderEvent.CANCEL_REQ)

    def test_pending_rejects_partial_fill(self):
        osm = make_osm()
        with self.assertRaises(InvalidTransition):
            osm.transition(OrderEvent.PARTIAL_FILL, fill_size=0.5, fill_price=50_000.0)

    def test_cancelled_rejects_all_events(self):
        osm = make_osm()
        osm.transition(OrderEvent.SUBMIT)
        osm.transition(OrderEvent.CANCEL_REQ)
        osm.transition(OrderEvent.CANCEL_ACK)

        for event in OrderEvent:
            with self.assertRaises(InvalidTransition):
                osm.transition(event)

    def test_invalid_transition_str(self):
        osm = make_osm()
        try:
            osm.transition(OrderEvent.CANCEL_ACK)  # PENDING + CANCEL_ACK is illegal
        except InvalidTransition as exc:
            self.assertIn("ORD-001", str(exc))
            self.assertIn("PENDING", str(exc))
            self.assertIn("CANCEL_ACK", str(exc))


class TestOrderStateMachineAvgFillPrice(unittest.TestCase):
    """Two partial fills → correct VWAP."""

    def test_vwap_two_fills(self):
        osm = make_osm(size=10.0)
        osm.transition(OrderEvent.SUBMIT)

        # Fill 1: 4 units @ 100
        osm.transition(OrderEvent.PARTIAL_FILL, fill_size=4.0, fill_price=100.0)
        # Fill 2: 6 units @ 200
        osm.transition(OrderEvent.FULL_FILL,    fill_size=6.0, fill_price=200.0)

        # VWAP = (4*100 + 6*200) / 10 = (400 + 1200) / 10 = 160
        self.assertAlmostEqual(osm.avg_fill_price, 160.0, places=6)

    def test_single_fill_vwap(self):
        osm = make_osm(size=5.0)
        osm.transition(OrderEvent.SUBMIT)
        osm.transition(OrderEvent.FULL_FILL, fill_size=5.0, fill_price=75.0)
        self.assertAlmostEqual(osm.avg_fill_price, 75.0, places=6)

    def test_pnl_buy_side(self):
        osm = make_osm(size=10.0, side="buy")
        osm.transition(OrderEvent.SUBMIT)
        # Fill at 50 with cost basis of 45 → profit of (50-45)*10 = 50
        osm.transition(OrderEvent.FULL_FILL, fill_size=10.0, fill_price=50.0, cost_basis=45.0)
        self.assertAlmostEqual(osm.pnl_realised, 50.0, places=6)

    def test_pnl_sell_side(self):
        osm = make_osm(size=10.0, side="sell")
        osm.transition(OrderEvent.SUBMIT)
        # Sell at 60 with cost basis of 55 → profit (60-55)*10 = 50
        osm.transition(OrderEvent.FULL_FILL, fill_size=10.0, fill_price=60.0, cost_basis=55.0)
        self.assertAlmostEqual(osm.pnl_realised, 50.0, places=6)


# ===========================================================================
# OrderBook tests
# ===========================================================================

class TestOrderBookNetPosition(unittest.TestCase):
    """Buy 1 + sell 0.5 → net 0.5."""

    def _fill(self, osm: OrderStateMachine) -> None:
        osm.transition(OrderEvent.SUBMIT)
        osm.transition(OrderEvent.FULL_FILL, fill_size=osm.size, fill_price=osm.price)

    def test_net_position_buy_sell(self):
        book = OrderBook()

        buy_osm = make_osm("BUY-1", size=1.0, side="buy")
        sell_osm = make_osm("SELL-1", size=0.5, side="sell")

        book.register(buy_osm)
        book.register(sell_osm)

        self._fill(buy_osm)
        self._fill(sell_osm)

        net = book.net_position("BTC-USD")
        self.assertAlmostEqual(net, 0.5, places=6)

    def test_net_position_no_fills(self):
        book = OrderBook()
        osm = make_osm("ORD-X", size=1.0)
        book.register(osm)
        self.assertAlmostEqual(book.net_position("BTC-USD"), 0.0)

    def test_session_pnl(self):
        book = OrderBook()
        osm = make_osm("ORD-P", size=5.0, side="buy")
        book.register(osm)
        osm.transition(OrderEvent.SUBMIT)
        osm.transition(OrderEvent.FULL_FILL, fill_size=5.0, fill_price=100.0, cost_basis=90.0)
        # pnl = (100-90)*5 = 50
        self.assertAlmostEqual(book.session_pnl(), 50.0, places=6)

    def test_open_and_closed_orders(self):
        book = OrderBook()
        open_osm = make_osm("O1")
        closed_osm = make_osm("O2", size=1.0)

        book.register(open_osm)
        book.register(closed_osm)

        open_osm.transition(OrderEvent.SUBMIT)
        closed_osm.transition(OrderEvent.SUBMIT)
        closed_osm.transition(OrderEvent.FULL_FILL, fill_size=1.0, fill_price=50_000.0)

        self.assertEqual(len(book.open_orders()), 1)
        self.assertEqual(len(book.closed_orders()), 1)

    def test_register_duplicate_raises(self):
        book = OrderBook()
        osm = make_osm()
        book.register(osm)
        with self.assertRaises(ValueError):
            book.register(osm)


# ===========================================================================
# MultiLegExecutor tests
# ===========================================================================

class TestMultiLegExecutorSimultaneous(unittest.TestCase):
    """Verify asyncio.gather is used for simultaneous submission."""

    def test_gather_called_for_all_legs(self):
        ex_a = make_exchange_client("A-001")
        ex_b = make_exchange_client("B-001")
        executor = MultiLegExecutor({"A": ex_a, "B": ex_b})

        legs = [
            LegSpec("A", "ETH-USD", "buy",  1.0, 2000.0, "limit"),
            LegSpec("B", "ETH-USD", "sell", 1.0, 2001.0, "limit"),
        ]

        with patch("asyncio.gather", wraps=asyncio.gather) as mock_gather:
            results = run_async(executor.submit_legs(legs))
            self.assertTrue(mock_gather.called)

        self.assertEqual(len(results), 2)

        # Both legs should succeed
        self.assertTrue(results[0]["success"])
        self.assertTrue(results[1]["success"])

    def test_arb_pair_returns_two_results(self):
        ex_a = make_exchange_client("A-ARB")
        ex_b = make_exchange_client("B-ARB")
        executor = MultiLegExecutor({"A": ex_a, "B": ex_b})

        buy_result, sell_result = run_async(
            executor.submit_arb_pair("A", "B", "BTC-USD", 0.1, 50_000.0, 50_001.0)
        )

        self.assertIn("order", buy_result)
        self.assertIn("order", sell_result)
        self.assertEqual(buy_result["side"], "buy")
        self.assertEqual(sell_result["side"], "sell")

    def test_stats_updated_after_submission(self):
        ex = make_exchange_client("STAT-001")
        executor = MultiLegExecutor({"EX": ex})
        legs = [LegSpec("EX", "BTC-USD", "buy", 1.0, 50_000.0)]
        run_async(executor.submit_legs(legs))

        stats = executor.get_stats()
        self.assertEqual(stats["legs_submitted"], 1)
        self.assertEqual(stats["legs_filled"], 1)
        self.assertEqual(stats["submit_legs_calls"], 1)


class TestMultiLegExecutorPartialFailure(unittest.TestCase):
    """One leg fails → cancel attempt on successful legs."""

    def test_cancel_attempted_on_success_when_other_fails(self):
        ex_ok   = make_exchange_client("OK-001")
        ex_fail = make_exchange_client(should_fail=True)

        executor = MultiLegExecutor({"OK": ex_ok, "FAIL": ex_fail})

        legs = [
            LegSpec("OK",   "BTC-USD", "buy",  1.0, 50_000.0, "limit"),
            LegSpec("FAIL", "BTC-USD", "sell", 1.0, 50_001.0, "limit"),
        ]

        results = run_async(executor.submit_legs(legs))

        # First leg succeeded
        self.assertTrue(results[0]["success"])
        # Second leg failed
        self.assertFalse(results[1]["success"])

        # Cancel must have been called on the successful leg's exchange
        ex_ok.cancel_order.assert_awaited_once()

    def test_all_fail_no_cancel_needed(self):
        ex_a = make_exchange_client(should_fail=True)
        ex_b = make_exchange_client(should_fail=True)

        executor = MultiLegExecutor({"A": ex_a, "B": ex_b})

        legs = [
            LegSpec("A", "BTC-USD", "buy",  1.0, 50_000.0),
            LegSpec("B", "BTC-USD", "sell", 1.0, 50_001.0),
        ]

        results = run_async(executor.submit_legs(legs))

        self.assertFalse(results[0]["success"])
        self.assertFalse(results[1]["success"])
        # No cancel should have been attempted
        ex_a.cancel_order.assert_not_awaited()
        ex_b.cancel_order.assert_not_awaited()

    def test_empty_legs_returns_empty(self):
        executor = MultiLegExecutor({"X": make_exchange_client()})
        results = run_async(executor.submit_legs([]))
        self.assertEqual(results, [])


# ===========================================================================
# QueuePositionTracker tests
# ===========================================================================

class TestQueuePositionTrackerRegister(unittest.TestCase):
    """Register an order and verify it is tracked."""

    def test_register_and_retrieve(self):
        obp     = make_order_book_processor(level_size=5000.0, l3_queue=None)
        tracker = QueuePositionTracker(obp)

        tracker.register_our_order(
            order_id     = "Q-001",
            symbol       = "BTC-USD",
            side         = "buy",
            price        = 50_000.0,
            size         = 0.5,
            timestamp_ns = time.perf_counter_ns(),
        )

        to = tracker.get_order("Q-001")
        self.assertIsNotNone(to)
        self.assertEqual(to.order_id, "Q-001")
        self.assertEqual(to.symbol, "BTC-USD")
        self.assertEqual(to.side, "buy")
        self.assertTrue(to.is_active)

    def test_registered_order_in_active_list(self):
        obp     = make_order_book_processor()
        tracker = QueuePositionTracker(obp)

        tracker.register_our_order("Q-002", "ETH-USD", "sell", 2000.0, 1.0, time.perf_counter_ns())

        active = tracker.all_active_orders()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].order_id, "Q-002")

    def test_queue_stats_structure(self):
        obp     = make_order_book_processor(level_size=1000.0)
        tracker = QueuePositionTracker(obp)
        tracker.register_our_order("Q-003", "BTC-USD", "buy", 50_000.0, 1.0, time.perf_counter_ns())

        stats = tracker.get_queue_stats("BTC-USD")
        self.assertIn("symbol", stats)
        self.assertIn("our_active_orders", stats)
        self.assertIn("avg_queue_position_bid", stats)


class TestQueuePositionTrackerShouldCancel(unittest.TestCase):
    """Old order with large queue depth → should_cancel_and_rejoin returns True."""

    def test_should_cancel_when_queue_deep_and_old(self):
        obp = make_order_book_processor(level_size=5_000_000.0)  # huge level
        tracker = QueuePositionTracker(
            obp,
            cancel_queue_threshold  = 100,       # very low threshold for test
            cancel_age_threshold_ms = 0.0,       # no age requirement
        )

        ts = time.perf_counter_ns() - 10_000_000_000  # 10 seconds ago

        tracker.register_our_order(
            order_id     = "OLD-001",
            symbol       = "BTC-USD",
            side         = "buy",
            price        = 50_000.0,
            size         = 1.0,
            timestamp_ns = ts,
        )

        # Force queue_position above threshold
        to = tracker.get_order("OLD-001")
        to.queue_position = 50_000  # deep in queue

        result = tracker.should_cancel_and_rejoin("OLD-001", max_wait_ms=0.0)
        self.assertTrue(result)

    def test_should_not_cancel_when_queue_shallow(self):
        obp = make_order_book_processor(level_size=10.0)
        tracker = QueuePositionTracker(
            obp,
            cancel_queue_threshold  = 50_000,
            cancel_age_threshold_ms = 0.0,
        )
        ts = time.perf_counter_ns() - 1_000_000_000  # 1 second ago

        tracker.register_our_order("SHALLOW-001", "BTC-USD", "buy", 50_000.0, 1.0, ts)

        to = tracker.get_order("SHALLOW-001")
        to.queue_position = 5  # very shallow queue

        result = tracker.should_cancel_and_rejoin("SHALLOW-001", max_wait_ms=0.0)
        self.assertFalse(result)

    def test_should_cancel_unknown_order_returns_false(self):
        obp     = make_order_book_processor()
        tracker = QueuePositionTracker(obp)
        self.assertFalse(tracker.should_cancel_and_rejoin("UNKNOWN-999"))

    def test_update_queue_reduces_position(self):
        obp = make_order_book_processor(level_size=1000.0, l3_queue=None)
        tracker = QueuePositionTracker(obp, cancel_queue_threshold=0, cancel_age_threshold_ms=0.0)

        tracker.register_our_order("UQ-001", "BTC-USD", "buy", 50_000.0, 1.0, time.perf_counter_ns())
        to = tracker.get_order("UQ-001")
        initial_ahead = to.visible_size_ahead

        # Process a fill at this level
        tracker.update_queue("BTC-USD", "buy", 50_000.0, cancelled_size=0.0, filled_size=100.0)

        self.assertLessEqual(to.visible_size_ahead, initial_ahead)


# ===========================================================================
# IcebergExecutor tests
# ===========================================================================

class TestIcebergExecutorSliceSize(unittest.TestCase):
    """Verify random slice is within [min_visible_pct, max_visible_pct] of total_size."""

    def test_slice_within_bounds(self):
        ex       = make_exchange_client("ICE-001", fill_size=0.0)
        executor = IcebergExecutor(
            exchange          = ex,
            min_visible_pct   = 0.10,
            max_visible_pct   = 0.20,
            poll_interval_ms  = 99999,  # no polls in test
        )

        for _ in range(100):
            total = 100.0
            size  = executor._random_slice_size(total)
            self.assertGreaterEqual(size, 0.10 * total - 1e-9)
            self.assertLessEqual(size, 0.20 * total + 1e-9)

    def test_slice_less_than_total(self):
        ex       = make_exchange_client()
        executor = IcebergExecutor(ex, min_visible_pct=0.05, max_visible_pct=0.15)
        size     = executor._random_slice_size(1000.0)
        self.assertLess(size, 1000.0)

    def test_invalid_pct_raises(self):
        ex = make_exchange_client()
        with self.assertRaises(ValueError):
            IcebergExecutor(ex, min_visible_pct=0.5, max_visible_pct=0.3)


class TestIcebergStateTracking(unittest.TestCase):
    """Verify filled_size accumulates correctly across fills."""

    def test_apply_fill_accumulates(self):
        state = IcebergState(
            iceberg_id    = "ICE-TEST-1",
            symbol        = "BTC-USD",
            side          = "buy",
            total_size    = 10.0,
            price         = 50_000.0,
            exchange_name = "binance",
        )

        state.apply_fill(fill_size=3.0, fill_price=50_100.0)
        self.assertAlmostEqual(state.filled_size, 3.0)
        self.assertAlmostEqual(state.remaining_size, 7.0)

        state.apply_fill(fill_size=4.0, fill_price=50_200.0)
        self.assertAlmostEqual(state.filled_size, 7.0)
        self.assertAlmostEqual(state.remaining_size, 3.0)

    def test_vwap_calculation(self):
        state = IcebergState(
            iceberg_id    = "ICE-TEST-2",
            symbol        = "ETH-USD",
            side          = "sell",
            total_size    = 10.0,
            price         = 2000.0,
            exchange_name = "coinbase",
        )
        state.apply_fill(fill_size=5.0, fill_price=2000.0)
        state.apply_fill(fill_size=5.0, fill_price=2100.0)
        # VWAP = (5*2000 + 5*2100) / 10 = 2050
        self.assertAlmostEqual(state.avg_fill_price, 2050.0, places=4)

    def test_status_is_active_initially(self):
        state = IcebergState(
            iceberg_id="ICE-TEST-3", symbol="BTC-USD", side="buy",
            total_size=5.0, price=50_000.0, exchange_name="kraken",
        )
        self.assertEqual(state.status, IcebergStatus.ACTIVE)

    def test_to_dict_contains_required_fields(self):
        state = IcebergState(
            iceberg_id="ICE-TEST-4", symbol="BTC-USD", side="buy",
            total_size=5.0, price=50_000.0, exchange_name="ftx",
        )
        d = state.to_dict()
        for key in ("iceberg_id", "symbol", "side", "total_size", "filled_size",
                    "remaining_size", "active_slice_order_id", "active_slice_size",
                    "slice_count", "avg_fill_price", "status"):
            self.assertIn(key, d, msg=f"Missing key: {key}")

    def test_execute_iceberg_submits_initial_slice(self):
        """End-to-end: execute_iceberg submits an initial slice and returns IcebergState."""
        ex = make_exchange_client("ICE-EXEC-001", fill_size=10.0, fill_price=50_000.0)
        executor = IcebergExecutor(
            exchange         = ex,
            min_visible_pct  = 0.10,
            max_visible_pct  = 0.20,
            poll_interval_ms = 99999,  # prevent background polling
        )

        async def _run():
            state = await executor.execute_iceberg(
                symbol        = "BTC-USD",
                side          = "buy",
                total_size    = 10.0,
                price         = 50_000.0,
                exchange_name = "binance",
            )
            return state

        state = asyncio.get_event_loop().run_until_complete(_run())

        # Initial slice was submitted
        self.assertIsNotNone(state.active_slice_order_id)
        self.assertEqual(state.slice_count, 1)
        self.assertGreater(state.active_slice_size, 0)
        self.assertLess(state.active_slice_size, 10.0)
        self.assertEqual(state.status, IcebergStatus.ACTIVE)

        # Cancel background task cleanly
        asyncio.get_event_loop().run_until_complete(
            executor.cancel_iceberg(state.iceberg_id)
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
