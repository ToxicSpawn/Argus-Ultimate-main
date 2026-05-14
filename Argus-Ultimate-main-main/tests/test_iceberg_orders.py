"""
Tests for execution/iceberg_orders.py — IcebergOrderManager
============================================================

20+ tests covering creation, execution, randomization, timeout,
front-running detection, cancellation, and report generation.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from execution.iceberg_orders import (
    IcebergOrder,
    IcebergOrderManager,
    IcebergResult,
    IcebergSlice,
    IcebergStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeExchange:
    """Mock exchange with configurable fill behavior."""

    def __init__(self, fill_immediately: bool = True, fill_price: float | None = None):
        self.fill_immediately = fill_immediately
        self.fill_price = fill_price
        self.placed_orders: list[dict] = []
        self.cancelled_orders: list[str] = []
        self._order_counter = 0
        self._check_count: dict[str, int] = {}

    async def place_limit_order(self, symbol, side, qty, price):
        self._order_counter += 1
        oid = f"EX-{self._order_counter}"
        self.placed_orders.append({
            "order_id": oid,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
        })
        self._check_count[oid] = 0
        return oid

    async def check_order(self, order_id):
        self._check_count[order_id] = self._check_count.get(order_id, 0) + 1
        placed = [o for o in self.placed_orders if o["order_id"] == order_id]
        if not placed:
            return {"filled_qty": 0, "avg_price": 0}
        o = placed[0]
        if self.fill_immediately or self._check_count[order_id] >= 2:
            fp = self.fill_price if self.fill_price is not None else o["price"]
            return {"filled_qty": o["qty"], "avg_price": fp}
        return {"filled_qty": 0, "avg_price": 0}

    async def cancel_order(self, order_id):
        self.cancelled_orders.append(order_id)


# ---------------------------------------------------------------------------
# Creation tests
# ---------------------------------------------------------------------------

class TestIcebergCreation:
    def test_create_basic(self):
        mgr = IcebergOrderManager(visible_pct=0.10)
        order = mgr.create_iceberg("BTC/USD", "buy", 1.0, 50000.0)
        assert order.symbol == "BTC/USD"
        assert order.side == "buy"
        assert order.total_quantity == 1.0
        assert order.price == 50000.0
        assert order.visible_qty == pytest.approx(0.1, abs=0.01)
        assert order.remaining_qty == 1.0
        assert order.status == IcebergStatus.PENDING

    def test_create_sell_order(self):
        mgr = IcebergOrderManager()
        order = mgr.create_iceberg("ETH/USD", "sell", 10.0, 3000.0)
        assert order.side == "sell"
        assert order.total_quantity == 10.0

    def test_create_with_override_pct(self):
        mgr = IcebergOrderManager(visible_pct=0.10)
        order = mgr.create_iceberg("BTC/USD", "buy", 1.0, 50000.0, visible_pct=0.25)
        assert order.visible_pct == 0.25
        assert order.visible_qty == pytest.approx(0.25, abs=0.01)

    def test_create_min_visible_qty_enforced(self):
        mgr = IcebergOrderManager(visible_pct=0.01, min_visible_qty=0.05)
        order = mgr.create_iceberg("BTC/USD", "buy", 1.0, 50000.0)
        assert order.visible_qty >= 0.05

    def test_create_visible_capped_at_total(self):
        mgr = IcebergOrderManager(visible_pct=0.50, min_visible_qty=0.001)
        order = mgr.create_iceberg("BTC/USD", "buy", 0.001, 50000.0)
        assert order.visible_qty <= order.total_quantity

    def test_create_invalid_symbol_raises(self):
        mgr = IcebergOrderManager()
        with pytest.raises(ValueError, match="symbol"):
            mgr.create_iceberg("", "buy", 1.0, 50000.0)

    def test_create_invalid_side_raises(self):
        mgr = IcebergOrderManager()
        with pytest.raises(ValueError, match="side"):
            mgr.create_iceberg("BTC/USD", "short", 1.0, 50000.0)

    def test_create_invalid_quantity_raises(self):
        mgr = IcebergOrderManager()
        with pytest.raises(ValueError, match="total_quantity"):
            mgr.create_iceberg("BTC/USD", "buy", -1.0, 50000.0)

    def test_create_invalid_price_raises(self):
        mgr = IcebergOrderManager()
        with pytest.raises(ValueError, match="price"):
            mgr.create_iceberg("BTC/USD", "buy", 1.0, 0.0)

    def test_constructor_invalid_visible_pct(self):
        with pytest.raises(ValueError, match="visible_pct"):
            IcebergOrderManager(visible_pct=0.0)
        with pytest.raises(ValueError, match="visible_pct"):
            IcebergOrderManager(visible_pct=1.5)

    def test_constructor_invalid_min_visible(self):
        with pytest.raises(ValueError, match="min_visible_qty"):
            IcebergOrderManager(min_visible_qty=-1)

    def test_constructor_invalid_max_duration(self):
        with pytest.raises(ValueError, match="max_duration"):
            IcebergOrderManager(max_duration=0)


# ---------------------------------------------------------------------------
# Execution tests
# ---------------------------------------------------------------------------

class TestIcebergExecution:
    @pytest.mark.asyncio
    async def test_full_execution_immediate_fills(self):
        mgr = IcebergOrderManager(visible_pct=0.25, randomize_visible=False)
        order = mgr.create_iceberg("BTC/USD", "buy", 1.0, 50000.0)
        exchange = FakeExchange(fill_immediately=True)
        result = await mgr.execute_iceberg(order, exchange, check_interval=0.01)

        assert result.status == IcebergStatus.COMPLETED
        assert result.total_filled == pytest.approx(1.0, abs=0.01)
        assert result.n_slices >= 3  # 1.0 / 0.25 = 4 slices expected
        assert result.avg_price == pytest.approx(50000.0, abs=1.0)
        assert result.order_id == order.order_id

    @pytest.mark.asyncio
    async def test_execution_delayed_fills(self):
        mgr = IcebergOrderManager(visible_pct=0.50, randomize_visible=False)
        order = mgr.create_iceberg("BTC/USD", "buy", 1.0, 50000.0)
        exchange = FakeExchange(fill_immediately=False)
        result = await mgr.execute_iceberg(order, exchange, check_interval=0.01)

        assert result.status == IcebergStatus.COMPLETED
        assert result.total_filled == pytest.approx(1.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_execution_timeout(self):
        mgr = IcebergOrderManager(visible_pct=0.10, randomize_visible=False, max_duration=0.05)
        order = mgr.create_iceberg("BTC/USD", "buy", 10.0, 50000.0)

        class NeverFillExchange(FakeExchange):
            async def check_order(self, order_id):
                return {"filled_qty": 0, "avg_price": 0}

        exchange = NeverFillExchange()
        result = await mgr.execute_iceberg(order, exchange, check_interval=0.01)

        assert result.status == IcebergStatus.TIMED_OUT
        assert result.total_filled == 0.0

    @pytest.mark.asyncio
    async def test_execution_place_order_failure(self):
        mgr = IcebergOrderManager(visible_pct=0.50, randomize_visible=False)
        order = mgr.create_iceberg("BTC/USD", "buy", 1.0, 50000.0)

        exchange = FakeExchange()
        exchange.place_limit_order = AsyncMock(side_effect=RuntimeError("API error"))

        result = await mgr.execute_iceberg(order, exchange, check_interval=0.01)
        assert result.status == IcebergStatus.ERROR

    @pytest.mark.asyncio
    async def test_execution_tracks_slippage(self):
        mgr = IcebergOrderManager(visible_pct=0.50, randomize_visible=False)
        order = mgr.create_iceberg("BTC/USD", "buy", 1.0, 50000.0)
        exchange = FakeExchange(fill_immediately=True, fill_price=50010.0)
        result = await mgr.execute_iceberg(order, exchange, check_interval=0.01)

        assert result.slippage_bps > 0
        expected_bps = abs(50010.0 - 50000.0) / 50000.0 * 10000
        assert result.slippage_bps == pytest.approx(expected_bps, abs=0.5)

    @pytest.mark.asyncio
    async def test_front_running_detection(self):
        mgr = IcebergOrderManager(
            visible_pct=0.50, randomize_visible=False, front_run_threshold_bps=5.0,
        )
        order = mgr.create_iceberg("BTC/USD", "buy", 1.0, 50000.0)

        call_count = 0

        class DriftingExchange(FakeExchange):
            async def check_order(self, order_id):
                nonlocal call_count
                call_count += 1
                placed = [o for o in self.placed_orders if o["order_id"] == order_id]
                if not placed:
                    return {"filled_qty": 0, "avg_price": 0}
                o = placed[0]
                # Large price drift between slices
                drift = len(self.placed_orders) * 50  # $50 per slice
                return {"filled_qty": o["qty"], "avg_price": o["price"] + drift}

        exchange = DriftingExchange()
        result = await mgr.execute_iceberg(order, exchange, check_interval=0.01)

        assert result.detected_front_running is True

    @pytest.mark.asyncio
    async def test_randomized_visible_sizes(self):
        mgr = IcebergOrderManager(visible_pct=0.10, randomize_visible=True)
        order = mgr.create_iceberg("BTC/USD", "buy", 10.0, 50000.0)
        exchange = FakeExchange(fill_immediately=True)
        result = await mgr.execute_iceberg(order, exchange, check_interval=0.01)

        # Slices should have varying visible_qty due to randomization
        visible_sizes = [s.visible_qty for s in result.slices if s.status == "filled"]
        if len(visible_sizes) >= 3:
            # Not all the same size (randomization should cause variance)
            assert len(set(round(v, 6) for v in visible_sizes)) > 1

    @pytest.mark.asyncio
    async def test_cannot_execute_completed_order(self):
        mgr = IcebergOrderManager(visible_pct=0.50, randomize_visible=False)
        order = mgr.create_iceberg("BTC/USD", "buy", 1.0, 50000.0)
        order.status = IcebergStatus.COMPLETED
        exchange = FakeExchange()
        with pytest.raises(ValueError, match="Cannot execute"):
            await mgr.execute_iceberg(order, exchange, check_interval=0.01)


# ---------------------------------------------------------------------------
# Report and cancellation tests
# ---------------------------------------------------------------------------

class TestIcebergReportAndCancel:
    @pytest.mark.asyncio
    async def test_get_execution_report_after_completion(self):
        mgr = IcebergOrderManager(visible_pct=0.50, randomize_visible=False)
        order = mgr.create_iceberg("BTC/USD", "buy", 1.0, 50000.0)
        exchange = FakeExchange(fill_immediately=True)
        await mgr.execute_iceberg(order, exchange, check_interval=0.01)

        report = mgr.get_execution_report(order.order_id)
        assert report["order_id"] == order.order_id
        assert report["total_filled"] == pytest.approx(1.0, abs=0.01)
        assert report["status"] == "completed"
        assert report["n_slices"] >= 1
        assert "slices" in report

    def test_get_execution_report_pending_order(self):
        mgr = IcebergOrderManager()
        order = mgr.create_iceberg("BTC/USD", "buy", 1.0, 50000.0)
        report = mgr.get_execution_report(order.order_id)
        assert report["status"] == "pending"
        assert report["total_filled"] == 0.0

    def test_get_report_unknown_id(self):
        mgr = IcebergOrderManager()
        report = mgr.get_execution_report("nonexistent")
        assert report == {}

    def test_cancel_pending_order(self):
        mgr = IcebergOrderManager()
        order = mgr.create_iceberg("BTC/USD", "buy", 1.0, 50000.0)
        assert mgr.cancel_order(order.order_id) is True
        assert order.status == IcebergStatus.CANCELLED

    def test_cancel_already_completed(self):
        mgr = IcebergOrderManager()
        order = mgr.create_iceberg("BTC/USD", "buy", 1.0, 50000.0)
        order.status = IcebergStatus.COMPLETED
        assert mgr.cancel_order(order.order_id) is False

    def test_cancel_unknown_id(self):
        mgr = IcebergOrderManager()
        assert mgr.cancel_order("nonexistent") is False

    def test_active_orders_property(self):
        mgr = IcebergOrderManager()
        o1 = mgr.create_iceberg("BTC/USD", "buy", 1.0, 50000.0)
        o2 = mgr.create_iceberg("ETH/USD", "sell", 5.0, 3000.0)
        o2.status = IcebergStatus.COMPLETED

        active = mgr.active_orders
        assert len(active) == 1
        assert active[0].order_id == o1.order_id


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_iceberg_slice_defaults(self):
        s = IcebergSlice()
        assert s.visible_qty == 0.0
        assert s.status == "pending"
        assert len(s.slice_id) == 12

    def test_iceberg_order_defaults(self):
        o = IcebergOrder()
        assert o.status == IcebergStatus.PENDING
        assert len(o.order_id) == 16
        assert o.slices == []

    def test_iceberg_result_defaults(self):
        r = IcebergResult()
        assert r.total_filled == 0.0
        assert r.status == IcebergStatus.COMPLETED
        assert r.detected_front_running is False
