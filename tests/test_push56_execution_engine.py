"""Push 56 — Order execution engine + fill simulator: 28 tests."""
from __future__ import annotations

# pyright: reportMissingImports=false, reportUndefinedVariable=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportOptionalMemberAccess=false, reportArgumentType=false

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from prometheus_client import REGISTRY

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.skip(
    "legacy execution engine tests target removed APIs",
    allow_module_level=True,
)

for metric_name, collector in list(getattr(REGISTRY, "_names_to_collectors", {}).items()):
    if metric_name == "argus_orders_total":
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Order model tests (7)
# ---------------------------------------------------------------------------
from core.execution.order_models import Order, Fill, OrderStatus, OrderType, OrderSide


class TestOrderModels:
    def _order(self, side=OrderSide.BUY, qty=1.0, limit=None):
        return Order(
            symbol="BTCUSDT", side=side,
            order_type=OrderType.LIMIT if limit else OrderType.MARKET,
            qty=qty, limit_price=limit,
        )

    def test_initial_status_pending(self):
        o = self._order()
        assert o.status == OrderStatus.PENDING

    def test_filled_qty_zero_initially(self):
        o = self._order()
        assert o.filled_qty == pytest.approx(0.0)

    def test_add_fill_updates_status(self):
        o = self._order(qty=1.0)
        f = Fill(order_id=o.order_id, symbol="BTCUSDT", side=OrderSide.BUY,
                 price=65000.0, qty=1.0, fee=1.3)
        o.add_fill(f)
        assert o.status == OrderStatus.FILLED
        assert o.filled_qty == pytest.approx(1.0)

    def test_partial_fill_status(self):
        o = self._order(qty=2.0)
        f = Fill(order_id=o.order_id, symbol="BTCUSDT", side=OrderSide.BUY,
                 price=65000.0, qty=1.0, fee=0.5)
        o.add_fill(f)
        assert o.status == OrderStatus.PARTIAL
        assert o.remaining_qty == pytest.approx(1.0)

    def test_avg_fill_price(self):
        o = self._order(qty=2.0)
        for price in [100.0, 200.0]:
            o.add_fill(Fill(order_id=o.order_id, symbol="X",
                            side=OrderSide.BUY, price=price, qty=1.0))
        assert o.avg_fill_price == pytest.approx(150.0)

    def test_is_complete_after_fill(self):
        o = self._order(qty=1.0)
        o.add_fill(Fill(order_id=o.order_id, symbol="X",
                        side=OrderSide.BUY, price=100.0, qty=1.0))
        assert o.is_complete is True

    def test_to_dict_keys(self):
        o = self._order()
        d = o.to_dict()
        assert "order_id" in d and "status" in d and "avg_fill_price" in d


# ---------------------------------------------------------------------------
# OrderBook tests (5)
# ---------------------------------------------------------------------------
from core.execution.order_book import OrderBook


class TestOrderBook:
    def _open_order(self):
        o = Order("BTCUSDT", OrderSide.BUY, OrderType.MARKET, qty=1.0)
        return o

    def test_submit_sets_open(self):
        book = OrderBook()
        o = self._open_order()
        book.submit(o)
        assert o.status == OrderStatus.OPEN
        assert book.get(o.order_id) is o

    def test_cancel_sets_cancelled(self):
        book = OrderBook()
        o = self._open_order()
        book.submit(o)
        result = book.cancel(o.order_id)
        assert result is True
        assert o.status == OrderStatus.CANCELLED

    def test_cancel_nonexistent_returns_false(self):
        book = OrderBook()
        assert book.cancel("nonexistent") is False

    def test_open_orders_excludes_cancelled(self):
        book = OrderBook()
        o = self._open_order()
        book.submit(o)
        book.cancel(o.order_id)
        assert o not in book.open_orders

    def test_orders_by_symbol(self):
        book = OrderBook()
        o1 = Order("BTCUSDT", OrderSide.BUY, OrderType.MARKET, 1.0)
        o2 = Order("ETHUSDT", OrderSide.BUY, OrderType.MARKET, 1.0)
        book.submit(o1)
        book.submit(o2)
        btc = book.orders_by_symbol("BTCUSDT")
        assert len(btc) == 1 and btc[0].symbol == "BTCUSDT"


# ---------------------------------------------------------------------------
# FillSimulator tests (8)
# ---------------------------------------------------------------------------
from core.execution.fill_simulator import FillSimulator


class TestFillSimulator:
    def _sim(self):
        return FillSimulator(spread_bps=1.0, slippage_bps=0.5, fill_probability=1.0, seed=42)

    def _order(self, otype=OrderType.MARKET, side=OrderSide.BUY,
                qty=1.0, limit=None, stop=None):
        o = Order("BTCUSDT", side, otype, qty,
                  limit_price=limit, stop_price=stop)
        o.status = OrderStatus.OPEN
        return o

    def test_market_buy_fills_immediately(self):
        sim = self._sim()
        o = self._order(otype=OrderType.MARKET)
        fills = sim.process_tick(o, mid_price=65000.0)
        assert len(fills) == 1
        assert o.status == OrderStatus.FILLED

    def test_market_buy_price_above_mid(self):
        sim = self._sim()
        o = self._order(otype=OrderType.MARKET)
        fills = sim.process_tick(o, mid_price=65000.0)
        assert fills[0].price > 65000.0

    def test_market_sell_price_below_mid(self):
        sim = self._sim()
        o = self._order(otype=OrderType.MARKET, side=OrderSide.SELL)
        fills = sim.process_tick(o, mid_price=65000.0)
        assert fills[0].price < 65000.0

    def test_limit_buy_fills_when_price_low(self):
        sim = self._sim()
        o = self._order(otype=OrderType.LIMIT, side=OrderSide.BUY, limit=65100.0)
        fills = sim.process_tick(o, mid_price=65000.0)  # mid < limit -> fill
        assert len(fills) == 1

    def test_limit_buy_no_fill_when_price_high(self):
        sim = self._sim()
        o = self._order(otype=OrderType.LIMIT, side=OrderSide.BUY, limit=64000.0)
        fills = sim.process_tick(o, mid_price=65000.0)  # mid > limit -> no fill
        assert len(fills) == 0

    def test_stop_sell_triggers_below_stop(self):
        sim = self._sim()
        o = self._order(otype=OrderType.STOP, side=OrderSide.SELL, stop=64000.0)
        fills = sim.process_tick(o, mid_price=63000.0)  # mid <= stop -> trigger
        assert len(fills) == 1

    def test_stop_sell_no_trigger_above_stop(self):
        sim = self._sim()
        o = self._order(otype=OrderType.STOP, side=OrderSide.SELL, stop=64000.0)
        fills = sim.process_tick(o, mid_price=65000.0)
        assert len(fills) == 0

    def test_fill_fee_positive(self):
        sim = FillSimulator(fee_bps=5.0, seed=42)
        o = self._order(otype=OrderType.MARKET)
        fills = sim.process_tick(o, mid_price=65000.0)
        assert fills[0].fee > 0


# ---------------------------------------------------------------------------
# ExecutionEngine tests (8)
# ---------------------------------------------------------------------------
from core.execution.execution_engine import ExecutionEngine


class TestExecutionEngine:
    def _engine(self):
        return ExecutionEngine(paper_trading=True)

    def _order(self, side=OrderSide.BUY, qty=1.0):
        return Order("BTCUSDT", side, OrderType.MARKET, qty)

    def test_submit_market_order_fills(self):
        engine = self._engine()
        order = self._order()
        asyncio.get_event_loop().run_until_complete(
            engine.submit_order(order, mid_price=65000.0)
        )
        assert order.status == OrderStatus.FILLED
        assert engine.filled == 1

    def test_submit_increments_counter(self):
        engine = self._engine()
        asyncio.get_event_loop().run_until_complete(
            engine.submit_order(self._order(), mid_price=65000.0)
        )
        assert engine.submitted == 1

    def test_cancel_open_limit_order(self):
        engine = self._engine()
        o = Order("BTCUSDT", OrderSide.BUY, OrderType.LIMIT, 1.0, limit_price=60000.0)
        asyncio.get_event_loop().run_until_complete(
            engine.submit_order(o, mid_price=65000.0)
        )
        result = engine.cancel_order(o.order_id)
        assert result is True

    def test_fill_callback_called(self):
        engine = self._engine()
        called = []
        async def cb(order, fill):
            called.append(fill)
        engine.add_fill_callback(cb)
        asyncio.get_event_loop().run_until_complete(
            engine.submit_order(self._order(), mid_price=65000.0)
        )
        assert len(called) == 1

    def test_on_market_tick_fills_limit(self):
        engine = self._engine()
        o = Order("BTCUSDT", OrderSide.BUY, OrderType.LIMIT, 1.0, limit_price=65500.0)
        asyncio.get_event_loop().run_until_complete(
            engine.submit_order(o, mid_price=66000.0)  # no fill yet
        )
        fills = asyncio.get_event_loop().run_until_complete(
            engine.on_market_tick("BTCUSDT", mid_price=65000.0)
        )
        assert len(fills) == 1

    def test_status_dict(self):
        engine = self._engine()
        s = engine.status()
        assert "submitted" in s and "paper_trading" in s

    def test_risk_rejection(self):
        from core.risk.risk_manager import RiskManager
        from core.risk.risk_config import RiskConfig
        rm = RiskManager(RiskConfig(min_confidence=0.99), equity=10_000)
        engine = ExecutionEngine(risk_manager=rm, paper_trading=True)
        o = self._order()
        asyncio.get_event_loop().run_until_complete(
            engine.submit_order(o, mid_price=65000.0, confidence=0.5)
        )
        assert o.status == OrderStatus.REJECTED
        assert engine.rejected == 1

    def test_sell_closes_pnl_position(self):
        from core.pnl.pnl_tracker import PnLTracker
        pnl = PnLTracker(fee_bps=0)
        engine = ExecutionEngine(pnl_tracker=pnl, paper_trading=True)
        # Open long
        asyncio.get_event_loop().run_until_complete(
            engine.submit_order(
                Order("BTCUSDT", OrderSide.BUY, OrderType.MARKET, 1.0),
                mid_price=65000.0,
            )
        )
        # Close
        asyncio.get_event_loop().run_until_complete(
            engine.submit_order(
                Order("BTCUSDT", OrderSide.SELL, OrderType.MARKET, 1.0),
                mid_price=65500.0,
            )
        )
        assert len(pnl.closed_trades) == 1
