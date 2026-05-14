"""Push 77 — Tests: Order, Fill, Position, OrderManager,
PaperAdapter, ExecutionEngine. 30 tests.
"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_order(symbol="BTCUSDT", side="BUY", qty=0.1, price=50000.0):
    from core.execution.order import Order, OrderSide, OrderType
    return Order(
        symbol=symbol,
        side=OrderSide(side),
        order_type=OrderType.MARKET,
        qty=qty,
        price=price,
    )


def _make_fill(order, price=50000.0, qty=None):
    from core.execution.order import Fill
    return Fill(
        order_id=order.order_id,
        symbol=order.symbol,
        side=order.side,
        qty=qty or order.qty,
        price=price,
        fee=qty * price * 0.001 if qty else order.qty * price * 0.001,
    )


# ---------------------------------------------------------------------------
# Order (5)
# ---------------------------------------------------------------------------

class TestOrder:
    def test_order_created(self):
        o = _make_order()
        assert o.symbol == "BTCUSDT"
        assert o.is_active

    def test_apply_fill_full(self):
        from core.execution.order import OrderStatus
        o = _make_order(qty=0.5)
        f = _make_fill(o, qty=0.5)
        o.apply_fill(f)
        assert o.status == OrderStatus.FILLED
        assert o.filled_qty == pytest.approx(0.5)

    def test_apply_fill_partial(self):
        from core.execution.order import OrderStatus
        o = _make_order(qty=1.0)
        f = _make_fill(o, qty=0.4)
        o.apply_fill(f)
        assert o.status == OrderStatus.PARTIALLY_FILLED
        assert o.remaining_qty == pytest.approx(0.6)

    def test_avg_fill_price(self):
        o = _make_order(qty=1.0)
        f1 = _make_fill(o, price=50000, qty=0.5)
        f2 = _make_fill(o, price=51000, qty=0.5)
        o.apply_fill(f1)
        o.apply_fill(f2)
        assert o.avg_fill_price == pytest.approx(50500.0)

    def test_to_dict_keys(self):
        o = _make_order()
        d = o.to_dict()
        assert "order_id" in d and "status" in d


# ---------------------------------------------------------------------------
# Fill (3)
# ---------------------------------------------------------------------------

class TestFill:
    def test_notional(self):
        from core.execution.order import Fill, OrderSide
        f = Fill(order_id="x", symbol="BTCUSDT", side=OrderSide.BUY,
                 qty=0.1, price=50000.0, fee=5.0)
        assert f.notional == pytest.approx(5000.0)

    def test_net_proceeds_buy(self):
        from core.execution.order import Fill, OrderSide
        f = Fill(order_id="x", symbol="BTCUSDT", side=OrderSide.BUY,
                 qty=0.1, price=50000.0, fee=5.0)
        assert f.net_proceeds < 0  # costs money

    def test_net_proceeds_sell(self):
        from core.execution.order import Fill, OrderSide
        f = Fill(order_id="x", symbol="BTCUSDT", side=OrderSide.SELL,
                 qty=0.1, price=50000.0, fee=5.0)
        assert f.net_proceeds > 0


# ---------------------------------------------------------------------------
# OrderManager (8)
# ---------------------------------------------------------------------------

class TestOrderManager:
    def _om(self):
        from core.execution.order_manager import OrderManager
        return OrderManager(max_open_orders=5, max_position_usd=1_000_000)

    def test_submit_order(self):
        from core.execution.order import OrderStatus
        om = self._om()
        o = _make_order()
        result = run(om.submit_order(o))
        assert result.status == OrderStatus.SUBMITTED

    def test_max_orders_rejected(self):
        from core.execution.order import OrderStatus
        from core.execution.order_manager import OrderManager
        om = OrderManager(max_open_orders=2)
        for _ in range(2):
            run(om.submit_order(_make_order()))
        extra = _make_order()
        result = run(om.submit_order(extra))
        assert result.status == OrderStatus.REJECTED

    def test_cancel_order(self):
        from core.execution.order import OrderStatus
        om = self._om()
        o = _make_order()
        run(om.submit_order(o))
        cancelled = run(om.cancel_order(o.order_id))
        assert cancelled.status == OrderStatus.CANCELLED

    def test_on_fill_updates_position(self):
        om = self._om()
        o = _make_order(qty=0.5)
        run(om.submit_order(o))
        f = _make_fill(o, qty=0.5)
        run(om.on_fill(f))
        pos = om.get_position("BTCUSDT")
        assert pos is not None
        assert pos.qty == pytest.approx(0.5)

    def test_position_long(self):
        from core.execution.order import PositionSide
        om = self._om()
        o = _make_order(qty=1.0)
        run(om.submit_order(o))
        run(om.on_fill(_make_fill(o, qty=1.0)))
        pos = om.get_position("BTCUSDT")
        assert pos.side == PositionSide.LONG

    def test_position_close_on_sell(self):
        from core.execution.order import PositionSide
        om = self._om()
        buy = _make_order(side="BUY", qty=1.0)
        run(om.submit_order(buy))
        run(om.on_fill(_make_fill(buy, qty=1.0)))
        sell = _make_order(side="SELL", qty=1.0)
        run(om.submit_order(sell))
        run(om.on_fill(_make_fill(sell, qty=1.0)))
        pos = om.get_position("BTCUSDT")
        assert pos.is_flat

    def test_realised_pnl_on_close(self):
        om = self._om()
        buy = _make_order(side="BUY", qty=1.0, price=50000)
        run(om.submit_order(buy))
        run(om.on_fill(_make_fill(buy, price=50000, qty=1.0)))
        sell = _make_order(side="SELL", qty=1.0, price=51000)
        run(om.submit_order(sell))
        run(om.on_fill(_make_fill(sell, price=51000, qty=1.0)))
        pos = om.get_position("BTCUSDT")
        assert pos.realised_pnl == pytest.approx(1000.0, abs=1.0)

    def test_stats_dict(self):
        om = self._om()
        s = om.stats
        assert "open_orders" in s and "total_fees" in s


# ---------------------------------------------------------------------------
# PaperAdapter (6)
# ---------------------------------------------------------------------------

class TestPaperAdapter:
    def _adapter(self):
        from core.execution.exchange_adapter import PaperAdapter
        return PaperAdapter(initial_balance=100_000)

    def test_initial_balance(self):
        a = self._adapter()
        bal = run(a.get_balance("USDT"))
        assert bal == pytest.approx(100_000)

    def test_place_order_fills(self):
        from core.execution.order import OrderStatus
        a = self._adapter()
        a.set_price("BTCUSDT", 50000.0)
        o = _make_order(qty=0.1)
        eid = run(a.place_order(o))
        assert eid.startswith("PAPER_")
        assert o.status == OrderStatus.FILLED

    def test_buy_reduces_usdt(self):
        a = self._adapter()
        a.set_price("BTCUSDT", 50000.0)
        o = _make_order(qty=0.1)
        run(a.place_order(o))
        bal = run(a.get_balance("USDT"))
        assert bal < 100_000

    def test_sell_increases_usdt(self):
        a = self._adapter()
        a.set_price("BTCUSDT", 50000.0)
        sell = _make_order(side="SELL", qty=0.1)
        run(a.place_order(sell))
        bal = run(a.get_balance("USDT"))
        assert bal > 100_000

    def test_subscribe_price_callback(self):
        a = self._adapter()
        ticks = []
        run(a.subscribe_price("BTCUSDT", lambda s, p: ticks.append(p)))
        a.set_price("BTCUSDT", 55000.0)
        assert 55000.0 in ticks

    def test_cancel_always_true(self):
        a = self._adapter()
        result = run(a.cancel_order("PAPER_001", "BTCUSDT"))
        assert result is True


# ---------------------------------------------------------------------------
# ExecutionEngine (8)
# ---------------------------------------------------------------------------

class TestExecutionEngine:
    def _engine(self):
        from core.execution.order_manager import OrderManager
        from core.execution.exchange_adapter import PaperAdapter
        from core.execution.execution_engine import ExecutionEngine
        from core.strategy.signal_bus import AsyncSignalBus
        om  = OrderManager()
        adp = PaperAdapter(initial_balance=100_000)
        bus = AsyncSignalBus()
        eng = ExecutionEngine(om, adp, bus, signal_cooldown_secs=0)
        return eng, om, adp, bus

    def test_instantiates(self):
        eng, *_ = self._engine()
        assert eng is not None

    def test_start_stop(self):
        eng, *_ = self._engine()
        run(eng.start())
        assert eng.is_running
        run(eng.stop())
        assert not eng.is_running

    def test_signal_to_order_long(self):
        from core.strategy.signal import Signal, SignalSide
        from core.execution.order import OrderSide
        eng, *_ = self._engine()
        sig = Signal(symbol="BTCUSDT", side=SignalSide.LONG,
                     strength=0.8, price=50000.0)
        order = eng.signal_to_order(sig, 50000.0)
        assert order is not None
        assert order.side == OrderSide.BUY

    def test_signal_to_order_short(self):
        from core.strategy.signal import Signal, SignalSide
        from core.execution.order import OrderSide
        eng, *_ = self._engine()
        sig = Signal(symbol="BTCUSDT", side=SignalSide.SHORT,
                     strength=0.6, price=50000.0)
        order = eng.signal_to_order(sig, 50000.0)
        assert order.side == OrderSide.SELL

    def test_signal_to_order_flat_no_position(self):
        from core.strategy.signal import Signal, SignalSide
        eng, *_ = self._engine()
        sig = Signal(symbol="BTCUSDT", side=SignalSide.FLAT, strength=1.0)
        order = eng.signal_to_order(sig, 50000.0)
        assert order is None

    def test_publish_signal_submits_order(self):
        from core.strategy.signal import Signal, SignalSide
        eng, om, adp, bus = self._engine()
        adp.set_price("BTCUSDT", 50000.0)
        run(eng.start())
        sig = Signal(symbol="BTCUSDT", side=SignalSide.LONG,
                     strength=0.8, price=50000.0)
        run(bus.publish(sig))
        assert eng.stats["signals_received"] >= 1
        run(eng.stop())

    def test_stats_dict(self):
        eng, *_ = self._engine()
        s = eng.stats
        assert "signals_received" in s and "avg_latency_us" in s

    def test_dedup_same_side_position(self):
        from core.strategy.signal import Signal, SignalSide
        from core.execution.order import PositionSide
        eng, om, adp, bus = self._engine()
        adp.set_price("BTCUSDT", 50000.0)
        # Manually set position to LONG
        from core.execution.order import Position
        om._positions["BTCUSDT"] = Position(
            symbol="BTCUSDT", side=PositionSide.LONG, qty=0.1, avg_entry=50000
        )
        sig = Signal(symbol="BTCUSDT", side=SignalSide.LONG, strength=0.8, price=50000.0)
        order = eng.signal_to_order(sig, 50000.0)
        assert order is None  # deduplicated
