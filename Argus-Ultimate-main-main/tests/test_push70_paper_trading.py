"""Push 70 — Tests: PaperTrader, ReconnectPolicy, RealTimePnLTracker,
AsyncWebSocketFeed, PaperTradingSession. 26 tests.
"""
from __future__ import annotations
import asyncio
import sys
import time
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# PaperTrader (8)
# ---------------------------------------------------------------------------

class TestPaperTrader:
    def _trader(self):
        from core.paper_trading.paper_trader import PaperTrader
        return PaperTrader(initial_cash=10_000, commission_bps=10, slippage_bps=5)

    def test_market_buy_fills_immediately(self):
        from core.paper_trading.paper_trader import OrderSide
        t = self._trader()
        order = t.place_market_order("BTCUSDT", OrderSide.BUY, 0.1, 50_000)
        assert order.status.value == "FILLED"

    def test_market_buy_reduces_cash(self):
        from core.paper_trading.paper_trader import OrderSide
        t = self._trader()
        t.place_market_order("BTCUSDT", OrderSide.BUY, 0.1, 50_000)
        assert t.cash < 10_000

    def test_position_created_after_buy(self):
        from core.paper_trading.paper_trader import OrderSide
        t = self._trader()
        t.place_market_order("BTCUSDT", OrderSide.BUY, 0.1, 50_000)
        assert "BTCUSDT" in t.positions
        assert t.positions["BTCUSDT"].side == "long"

    def test_limit_order_not_filled_immediately(self):
        from core.paper_trading.paper_trader import OrderSide
        t = self._trader()
        order = t.place_limit_order("BTCUSDT", OrderSide.BUY, 0.1, 45_000)
        assert order.status.value == "OPEN"
        assert len(t.open_orders) == 1

    def test_limit_order_fills_on_tick(self):
        from core.paper_trading.paper_trader import OrderSide
        t = self._trader()
        t.place_limit_order("BTCUSDT", OrderSide.BUY, 0.1, 45_000)
        fills = t.on_price_tick("BTCUSDT", 44_900)
        assert len(fills) == 1

    def test_cancel_order(self):
        from core.paper_trading.paper_trader import OrderSide
        t = self._trader()
        order = t.place_limit_order("BTCUSDT", OrderSide.BUY, 0.1, 45_000)
        result = t.cancel_order(order.order_id)
        assert result is True
        assert len(t.open_orders) == 0

    def test_equity_includes_unrealised(self):
        from core.paper_trading.paper_trader import OrderSide
        t = self._trader()
        t.place_market_order("BTCUSDT", OrderSide.BUY, 0.1, 50_000)
        eq = t.equity({"BTCUSDT": 55_000})
        assert eq > t.cash   # unrealised gain added

    def test_reset_clears_state(self):
        from core.paper_trading.paper_trader import OrderSide
        t = self._trader()
        t.place_market_order("BTCUSDT", OrderSide.BUY, 0.1, 50_000)
        t.reset()
        assert t.cash == 10_000
        assert len(t.positions) == 0
        assert t.n_fills == 0


# ---------------------------------------------------------------------------
# ReconnectPolicy (5)
# ---------------------------------------------------------------------------

class TestReconnectPolicy:
    def test_exponential_increases(self):
        from core.paper_trading.reconnect import ReconnectPolicy
        p = ReconnectPolicy(strategy="exponential", base_delay_secs=1.0,
                             jitter_fraction=0, max_retries=5)
        delays = [p.next_delay() for _ in range(3)]
        assert delays[0] < delays[1] < delays[2]

    def test_max_retries_returns_none(self):
        from core.paper_trading.reconnect import ReconnectPolicy
        p = ReconnectPolicy(max_retries=2, jitter_fraction=0)
        p.next_delay()
        p.next_delay()
        assert p.next_delay() is None

    def test_reset_clears_attempt(self):
        from core.paper_trading.reconnect import ReconnectPolicy
        p = ReconnectPolicy(max_retries=3, jitter_fraction=0)
        p.next_delay()
        p.reset()
        assert p.attempt == 0

    def test_circuit_breaker_trips(self):
        from core.paper_trading.reconnect import ReconnectPolicy
        p = ReconnectPolicy(max_consecutive_failures=3,
                             max_retries=100, jitter_fraction=0)
        for _ in range(3):
            p.next_delay()
        assert p.is_tripped
        assert p.next_delay() is None

    def test_fibonacci_strategy(self):
        from core.paper_trading.reconnect import ReconnectPolicy
        p = ReconnectPolicy(strategy="fibonacci", base_delay_secs=1.0,
                             jitter_fraction=0, max_retries=5)
        d1 = p.next_delay()
        d2 = p.next_delay()
        d3 = p.next_delay()
        assert d1 <= d2 <= d3


# ---------------------------------------------------------------------------
# RealTimePnLTracker (6)
# ---------------------------------------------------------------------------

class TestRealTimePnLTracker:
    def test_initial_state(self):
        from core.paper_trading.pnl_tracker import RealTimePnLTracker
        t = RealTimePnLTracker(initial_equity=10_000)
        assert t.realised_pnl == 0.0
        assert t.n_trades == 0

    def test_record_fill_increases_realised(self):
        from core.paper_trading.pnl_tracker import RealTimePnLTracker
        t = RealTimePnLTracker()
        t.record_fill("BTCUSDT", "long", 0.1, 50_000, pnl=100.0, commission=5.0)
        assert t.realised_pnl == 100.0
        assert t.n_trades == 1

    def test_win_rate_calculation(self):
        from core.paper_trading.pnl_tracker import RealTimePnLTracker
        t = RealTimePnLTracker()
        t.record_fill("X", "long", 1, 100, pnl=50, commission=1)
        t.record_fill("X", "long", 1, 100, pnl=-20, commission=1)
        t.record_fill("X", "long", 1, 100, pnl=30, commission=1)
        assert abs(t.win_rate - 2/3) < 1e-9

    def test_drawdown_updates(self):
        from core.paper_trading.pnl_tracker import RealTimePnLTracker
        t = RealTimePnLTracker(initial_equity=10_000)
        t.update_equity(9_000)
        assert t.current_drawdown_pct > 0
        assert t.max_drawdown_pct > 0

    def test_daily_pnl_window(self):
        from core.paper_trading.pnl_tracker import RealTimePnLTracker
        t = RealTimePnLTracker()
        t.record_fill("X", "long", 1, 100, pnl=100, commission=1)
        t.record_fill("X", "long", 1, 100, pnl=50, commission=1)
        assert t.daily_pnl == 150.0

    def test_symbol_pnl(self):
        from core.paper_trading.pnl_tracker import RealTimePnLTracker
        t = RealTimePnLTracker()
        t.record_fill("BTCUSDT", "long", 0.1, 50_000, pnl=200, commission=5)
        t.record_fill("ETHUSDT", "long", 1.0, 3_000, pnl=-30, commission=2)
        assert t.symbol_pnl("BTCUSDT")["realised"] == 200.0
        assert t.symbol_pnl("ETHUSDT")["realised"] == -30.0


# ---------------------------------------------------------------------------
# AsyncWebSocketFeed (3)
# ---------------------------------------------------------------------------

class TestAsyncWebSocketFeed:
    def test_instantiates(self):
        from core.paper_trading.ws_feed import AsyncWebSocketFeed
        feed = AsyncWebSocketFeed()
        assert not feed.is_connected

    def test_state_is_disconnected_initially(self):
        from core.paper_trading.ws_feed import AsyncWebSocketFeed, FeedState
        feed = AsyncWebSocketFeed()
        assert feed.state == FeedState.DISCONNECTED

    def test_start_stop_no_error(self):
        from core.paper_trading.ws_feed import AsyncWebSocketFeed
        feed = AsyncWebSocketFeed()
        async def run():
            await feed.start()
            await asyncio.sleep(0.1)
            await feed.stop()
        asyncio.get_event_loop().run_until_complete(run())
        assert not feed.is_connected


# ---------------------------------------------------------------------------
# PaperTradingSession (4)
# ---------------------------------------------------------------------------

class TestPaperTradingSession:
    def _session(self):
        from core.paper_trading.session_manager import PaperTradingSession, SessionConfig
        cfg = SessionConfig(symbol="BTCUSDT", initial_equity=10_000)
        return PaperTradingSession(config=cfg)

    def test_instantiates(self):
        s = self._session()
        assert s is not None

    def test_inject_price_updates_equity(self):
        s = self._session()
        s.inject_price("BTCUSDT", 50_000)
        assert s.last_price == 50_000

    def test_snapshot_returns_dict(self):
        s = self._session()
        s.inject_price("BTCUSDT", 50_000)
        snap = s.snapshot()
        assert "equity" in snap
        assert "total_pnl" in snap
        assert "feed_state" in snap

    def test_start_stop_lifecycle(self):
        from core.paper_trading.session_manager import PaperTradingSession
        s = PaperTradingSession()
        async def run():
            await s.start()
            await asyncio.sleep(0.05)
            await s.stop()
        asyncio.get_event_loop().run_until_complete(run())
        assert not s.is_running
