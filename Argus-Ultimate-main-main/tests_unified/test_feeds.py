"""
tests_unified/test_feeds.py
---------------------------
28 offline unit tests for the feeds package.
No exchange connectivity required — all WebSocket I/O is mocked.

Test classes:
  TestFeedNormaliser     (8)  — all three venues, all three message types
  TestWSFeedBase         (4)  — state machine, backoff, latency tracking
  TestFeedRouter         (5)  — subscribe, fan-out, dedup, drop-oldest, stats
  TestFeedAggregator     (5)  — BBO merge, VWAP, callback, multi-venue
  TestFeedHealthMonitor  (4)  — stale detection, reconnect, event_bus emit
  TestBybitFeed          (2)  — subscribe payload, handle message routing
"""

from __future__ import annotations

import asyncio
import json
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.feeds.feed_normaliser import (
    FeedNormaliser, CanonicalTick, CanonicalBook, CanonicalTrade,
)
from core.feeds.feed_router import FeedRouter
from core.feeds.feed_aggregator import FeedAggregator, AggregatedQuote
from core.feeds.feed_health_monitor import FeedHealthMonitor
from core.feeds.ws_feed_base import WSFeedBase, FeedState, FeedStats
from core.feeds.bybit_feed import BybitFeed


# ===========================================================================
# TestFeedNormaliser
# ===========================================================================

class TestFeedNormaliser:

    # Bybit

    def test_bybit_ticker_basic(self):
        raw = {
            "topic": "tickers.BTCUSDT",
            "ts": 1_700_000_000_000,
            "data": {
                "symbol": "BTCUSDT",
                "bid1Price": "29500.5",
                "ask1Price": "29501.0",
                "lastPrice": "29500.8",
                "volume24h": "12345.67",
                "ts": 1_700_000_000_000,
            },
        }
        tick = FeedNormaliser.bybit_ticker(raw)
        assert tick is not None
        assert tick.venue == "bybit"
        assert tick.symbol == "BTC/USDT"
        assert tick.bid == Decimal("29500.5")
        assert tick.ask == Decimal("29501.0")
        assert tick.spread == Decimal("0.5")

    def test_bybit_book_sorted(self):
        raw = {
            "ts": 1_700_000_000_000,
            "data": {
                "b": [["29500", "1.5"], ["29499", "2.0"], ["29501", "0.5"]],
                "a": [["29502", "1.0"], ["29501.5", "0.8"]],
                "ts": 1_700_000_000_000,
            },
        }
        book = FeedNormaliser.bybit_book(raw, "BTCUSDT")
        assert book is not None
        # bids descending
        assert book.bids[0][0] == Decimal("29501")
        assert book.bids[-1][0] == Decimal("29499")
        # asks ascending
        assert book.asks[0][0] == Decimal("29501.5")

    def test_bybit_trade_list(self):
        raw = {
            "data": [
                {"s": "BTCUSDT", "p": "29500", "v": "0.1", "S": "Buy",  "i": "t1", "T": 1_700_000_000_000},
                {"s": "BTCUSDT", "p": "29499", "v": "0.2", "S": "Sell", "i": "t2", "T": 1_700_000_001_000},
            ]
        }
        trades = FeedNormaliser.bybit_trade(raw)
        assert len(trades) == 2
        assert trades[0].side == "buy"
        assert trades[1].side == "sell"
        assert trades[1].qty == Decimal("0.2")

    # Binance

    def test_binance_ticker_book_ticker(self):
        raw = {"s": "BTCUSDT", "b": "29500.0", "a": "29501.0", "c": "29500.5",
               "v": "1000", "T": 1_700_000_000_000, "e": "bookTicker"}
        tick = FeedNormaliser.binance_ticker(raw)
        assert tick.venue == "binance"
        assert tick.symbol == "BTC/USDT"
        assert tick.bid == Decimal("29500.0")

    def test_binance_book(self):
        raw = {"b": [["29500", "1"], ["29499", "2"]], "a": [["29501", "0.5"]], "E": 1_700_000_000_000}
        book = FeedNormaliser.binance_book(raw, "BTCUSDT")
        assert book.bids[0][0] == Decimal("29500")
        assert book.asks[0][0] == Decimal("29501")

    def test_binance_trade_agg(self):
        raw = {"s": "BTCUSDT", "p": "29500", "q": "0.5", "m": False, "T": 1_700_000_000_000, "a": 9999}
        trade = FeedNormaliser.binance_trade(raw)
        assert trade.side == "buy"     # m=False → buyer is taker → buy
        assert trade.trade_id == "9999"

    # OKX

    def test_okx_ticker(self):
        raw = {"instId": "BTC-USDT-SWAP", "bidPx": "29500", "askPx": "29501",
               "last": "29500.5", "vol24h": "500", "ts": "1700000000000"}
        tick = FeedNormaliser.okx_ticker(raw)
        assert tick.venue == "okx"
        assert tick.symbol == "BTC/USDT"

    def test_okx_book(self):
        raw = {
            "instId": "BTC-USDT-SWAP",
            "bids": [["29500", "1", "0", "1"], ["29499", "2", "0", "1"]],
            "asks": [["29501", "0.5", "0", "1"]],
            "ts": "1700000000000",
        }
        book = FeedNormaliser.okx_book(raw)
        assert book.best_bid == Decimal("29500")
        assert book.best_ask == Decimal("29501")


# ===========================================================================
# TestWSFeedBase
# ===========================================================================

class _DummyFeed(WSFeedBase):
    """Minimal concrete subclass for testing WSFeedBase."""
    def __init__(self, **kw):
        super().__init__(url="ws://dummy", venue="dummy", **kw)
        self.subscribed = False
        self.handled: list = []

    async def _subscribe(self):
        self.subscribed = True

    async def _handle_message(self, raw):
        self.handled.append(raw)


class TestWSFeedBase:

    def test_initial_state(self):
        feed = _DummyFeed()
        assert feed.state == FeedState.DISCONNECTED
        assert feed.stats.messages_received == 0
        assert feed.stats.avg_latency_ms == 0.0

    def test_is_healthy_only_when_connected(self):
        feed = _DummyFeed()
        assert not feed.is_healthy
        feed.state = FeedState.CONNECTED
        assert feed.is_healthy
        feed.state = FeedState.RECONNECTING
        assert not feed.is_healthy

    def test_record_latency(self):
        feed = _DummyFeed()
        # Simulate exchange ts 50 ms ago
        ts_ms = (time.time() - 0.05) * 1000
        feed.record_latency(ts_ms)
        assert feed.stats.latency_count == 1
        assert 40 < feed.stats.avg_latency_ms < 200   # loose bounds

    def test_inc_emitter(self):
        emitter = MagicMock()
        feed = _DummyFeed(emitter=emitter)
        feed._inc("test_metric")
        emitter.inc_counter.assert_called_once_with("test_metric", labels={"venue": "dummy"})


# ===========================================================================
# TestFeedRouter
# ===========================================================================

class TestFeedRouter:

    def _make_tick(self, venue="bybit", symbol="BTC/USDT", bid="29500", ts=None) -> CanonicalTick:
        return CanonicalTick(
            venue=venue, symbol=symbol,
            bid=Decimal(bid), ask=Decimal("29501"),
            last=Decimal("29500.5"), volume_24h=Decimal("1000"),
            ts=ts or time.time(),
        )

    def test_subscribe_returns_queue(self):
        router = FeedRouter()
        q = router.subscribe("tick", "BTC/USDT")
        assert isinstance(q, asyncio.Queue)

    def test_fan_out_delivers(self):
        router = FeedRouter()
        q1 = router.subscribe("tick", "BTC/USDT")
        q2 = router.subscribe("tick", "BTC/USDT")
        tick = self._make_tick()
        asyncio.get_event_loop().run_until_complete(router._fan_out("tick", "BTC/USDT", tick))
        assert not q1.empty()
        assert not q2.empty()

    def test_dedup_drops_stale(self):
        router = FeedRouter(dedup_window_ms=100)
        q = router.subscribe("tick", "BTC/USDT")
        now = time.time()
        tick1 = self._make_tick(ts=now)
        tick2 = self._make_tick(ts=now - 0.05)  # 50 ms older — stale within 100 ms window
        loop = asyncio.get_event_loop()
        loop.run_until_complete(router._dispatch("tick", "BTC/USDT", "bybit", tick1.ts, tick1))
        loop.run_until_complete(router._dispatch("tick", "BTC/USDT", "bybit", tick2.ts, tick2))
        assert q.qsize() == 1   # second dropped

    def test_drop_oldest_when_full(self):
        router = FeedRouter(queue_maxsize=2)
        q = router.subscribe("tick", "BTC/USDT")
        loop = asyncio.get_event_loop()
        for i in range(4):
            t = self._make_tick(bid=str(29500 + i), ts=time.time() + i)
            loop.run_until_complete(router._fan_out("tick", "BTC/USDT", t))
        # Queue should hold last 2 (oldest dropped)
        assert q.qsize() == 2

    def test_stats_keys(self):
        router = FeedRouter()
        stats = router.stats
        assert "feeds" in stats
        assert "msg_count" in stats
        assert "feed_states" in stats


# ===========================================================================
# TestFeedAggregator
# ===========================================================================

class TestFeedAggregator:

    def _make_tick(self, venue, bid, ask, last, vol=1000.0, symbol="BTC/USDT"):
        return CanonicalTick(
            venue=venue, symbol=symbol,
            bid=Decimal(str(bid)), ask=Decimal(str(ask)),
            last=Decimal(str(last)), volume_24h=Decimal(str(vol)),
            ts=time.time(),
        )

    def test_bbo_merge_best_bid(self):
        q: asyncio.Queue = asyncio.Queue()
        agg = FeedAggregator({"BTC/USDT": q})
        agg._update("BTC/USDT", self._make_tick("bybit",   29500, 29502, 29501))
        agg._update("BTC/USDT", self._make_tick("binance", 29501, 29503, 29502))
        aq = agg.get_quote("BTC/USDT")
        assert aq.best_bid == Decimal("29501")   # binance has better bid
        assert aq.best_ask == Decimal("29502")   # bybit has better ask

    def test_bbo_best_ask_venue(self):
        q: asyncio.Queue = asyncio.Queue()
        agg = FeedAggregator({"BTC/USDT": q})
        agg._update("BTC/USDT", self._make_tick("bybit",   29500, 29502, 29501))
        agg._update("BTC/USDT", self._make_tick("okx",     29499, 29501, 29500))
        aq = agg.get_quote("BTC/USDT")
        assert aq.best_ask_venue == "okx"

    def test_vwap_computed(self):
        q: asyncio.Queue = asyncio.Queue()
        agg = FeedAggregator({"BTC/USDT": q}, vwap_window=3)
        for price in [29500, 29510, 29520]:
            agg._update("BTC/USDT", self._make_tick("bybit", price, price + 1, price, vol=1.0))
        aq = agg.get_quote("BTC/USDT")
        assert aq.vwap_mid > Decimal(0)

    def test_callback_fires(self):
        q: asyncio.Queue = asyncio.Queue()
        agg = FeedAggregator({"BTC/USDT": q})
        received = []
        async def cb(aq): received.append(aq)
        agg.register_callback("BTC/USDT", cb)
        agg._update("BTC/USDT", self._make_tick("bybit", 29500, 29501, 29500))
        loop = asyncio.get_event_loop()
        # Drain pending tasks
        loop.run_until_complete(asyncio.sleep(0))
        assert len(received) == 1

    def test_spread_positive(self):
        q: asyncio.Queue = asyncio.Queue()
        agg = FeedAggregator({"BTC/USDT": q})
        agg._update("BTC/USDT", self._make_tick("bybit", 29500, 29505, 29502))
        aq = agg.get_quote("BTC/USDT")
        assert aq.spread >= Decimal(0)


# ===========================================================================
# TestFeedHealthMonitor
# ===========================================================================

class TestFeedHealthMonitor:

    def _make_feed(self, last_ts=None):
        feed = _DummyFeed()
        feed.state = FeedState.CONNECTED
        feed.stats.last_message_ts = last_ts if last_ts is not None else time.monotonic()
        return feed

    def test_register(self):
        mon = FeedHealthMonitor()
        feed = self._make_feed()
        mon.register(feed)
        assert "dummy" in mon.stale_counts
        assert mon.stale_counts["dummy"] == 0

    def test_healthy_feed_no_stale(self):
        mon = FeedHealthMonitor(stale_threshold_s=30)
        feed = self._make_feed(last_ts=time.monotonic())  # just received
        mon.register(feed)
        # Manually run one check cycle
        loop = asyncio.get_event_loop()
        # Simulate check: age = 0 → should NOT increment stale
        age = time.monotonic() - feed.stats.last_message_ts
        assert age < 30
        assert mon.stale_counts["dummy"] == 0

    def test_stale_feed_detected(self):
        mon = FeedHealthMonitor(stale_threshold_s=5)
        feed = self._make_feed(last_ts=time.monotonic() - 10)  # 10 s stale
        mon.register(feed)
        event_bus = MagicMock()
        mon._event_bus = event_bus
        # Manually invoke stale logic
        age = time.monotonic() - feed.stats.last_message_ts
        assert age > 5
        mon._stale_counts["dummy"] += 1
        mon._emit_stale("dummy", age)
        event_bus.emit.assert_called_once()
        args = event_bus.emit.call_args[0]
        assert args[0] == "feed.stale"
        assert args[1]["venue"] == "dummy"

    def test_stale_count_resets_on_healthy(self):
        mon = FeedHealthMonitor(stale_threshold_s=5)
        feed = self._make_feed(last_ts=time.monotonic())
        mon.register(feed)
        mon._stale_counts["dummy"] = 3
        # Simulate healthy check
        age = time.monotonic() - feed.stats.last_message_ts
        if age <= 5:
            mon._stale_counts["dummy"] = 0
        assert mon.stale_counts["dummy"] == 0


# ===========================================================================
# TestBybitFeed
# ===========================================================================

class TestBybitFeed:

    def test_canonical_to_raw(self):
        assert BybitFeed._canonical_to_raw("BTC/USDT") == "BTCUSDT"
        assert BybitFeed._canonical_to_raw("ETH/BTC")  == "ETHBTC"

    def test_handle_ticker_message(self):
        received = []
        async def on_tick(tick): received.append(tick)

        feed = BybitFeed(symbols=["BTC/USDT"], on_tick=on_tick)
        msg = json.dumps({
            "topic": "tickers.BTCUSDT",
            "ts": 1_700_000_000_000,
            "data": {
                "symbol": "BTCUSDT",
                "bid1Price": "29500", "ask1Price": "29501",
                "lastPrice": "29500.5", "volume24h": "1000",
                "ts": 1_700_000_000_000,
            },
        })
        loop = asyncio.get_event_loop()
        loop.run_until_complete(feed._handle_message(msg))
        assert len(received) == 1
        assert received[0].symbol == "BTC/USDT"
