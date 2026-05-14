"""tests/test_binance_connector.py

Unit tests for Push 82 — Binance connector components.
All tests use mocks only — no live network calls.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────
# OrderBook tests
# ─────────────────────────────────────────────────────────────

class TestBinanceOrderBook:
    def _make_book(self):
        from connectors.binance.order_book import BinanceOrderBook
        book = BinanceOrderBook("BTCUSDT", depth=20)
        book.apply_snapshot(
            bids=[["65000.0", "1.5"], ["64990.0", "2.0"], ["64980.0", "0.5"]],
            asks=[["65010.0", "1.0"], ["65020.0", "3.0"], ["65030.0", "0.8"]],
            last_update_id=1000,
        )
        return book

    def test_snapshot_initialises_book(self):
        book = self._make_book()
        assert book.is_ready
        assert book.best_bid == 65000.0
        assert book.best_ask == 65010.0

    def test_mid_price(self):
        book = self._make_book()
        assert book.mid_price == pytest.approx(65005.0)

    def test_spread(self):
        book = self._make_book()
        assert book.spread == pytest.approx(10.0)

    def test_spread_bps(self):
        book = self._make_book()
        bps = book.spread_bps
        assert bps is not None
        assert bps == pytest.approx(10.0 / 65005.0 * 10000, rel=1e-3)

    def test_imbalance_range(self):
        book = self._make_book()
        imb = book.imbalance()
        assert imb is not None
        assert -1.0 <= imb <= 1.0

    def test_apply_update_modifies_book(self):
        book = self._make_book()
        ok = book.apply_update(
            bids=[["65000.0", "0.0"]],  # remove best bid
            asks=[["65010.0", "5.0"]],  # update ask qty
            first_id=1001,
            final_id=1001,
        )
        assert ok
        assert book.best_bid == 64990.0

    def test_stale_update_ignored(self):
        book = self._make_book()
        ok = book.apply_update(
            bids=[], asks=[], first_id=500, final_id=999
        )
        assert ok  # stale but not an error

    def test_gap_triggers_resync(self):
        book = self._make_book()
        ok = book.apply_update(
            bids=[], asks=[], first_id=1100, final_id=1200
        )
        assert not ok  # gap detected
        assert not book.is_ready
        assert book._resync_count == 1

    def test_vwap(self):
        book = self._make_book()
        vwap = book.vwap(side="bid", levels=3)
        assert vwap is not None
        assert 64980 <= vwap <= 65000

    def test_liquidity_within(self):
        book = self._make_book()
        bid_liq, ask_liq = book.liquidity_within(pct=0.1)
        assert bid_liq >= 0
        assert ask_liq >= 0

    def test_bid_ask_levels(self):
        book = self._make_book()
        bids = book.bid_levels(3)
        asks = book.ask_levels(3)
        assert len(bids) == 3
        assert len(asks) == 3
        assert bids[0].price > bids[1].price  # sorted desc
        assert asks[0].price < asks[1].price  # sorted asc

    def test_get_stats(self):
        book = self._make_book()
        stats = book.get_stats()
        assert stats["is_ready"]
        assert stats["best_bid"] == 65000.0
        assert stats["best_ask"] == 65010.0
        assert stats["resync_count"] == 0


# ─────────────────────────────────────────────────────────────
# KlineBuffer tests
# ─────────────────────────────────────────────────────────────

class TestKlineBuffer:
    def _make_buffer(self, n: int = 50):
        from connectors.binance.kline_buffer import KlineBuffer, Candle
        buf = KlineBuffer("BTCUSDT", "1m", maxlen=500)
        base = 65000.0
        for i in range(n):
            candle = Candle(
                symbol="BTCUSDT",
                interval="1m",
                open=base + i,
                high=base + i + 10,
                low=base + i - 5,
                close=base + i + 2,
                volume=10.0 + i * 0.1,
                timestamp_ms=1_700_000_000_000 + i * 60_000,
                trades=100,
                taker_buy_volume=5.0,
            )
            buf.push(candle)
        return buf

    def test_is_ready_after_26_candles(self):
        buf = self._make_buffer(26)
        assert buf.is_ready

    def test_not_ready_before_26(self):
        buf = self._make_buffer(10)
        assert not buf.is_ready

    def test_sma(self):
        buf = self._make_buffer(50)
        sma = buf.sma(20)
        assert sma is not None
        assert sma > 0

    def test_ema(self):
        buf = self._make_buffer(50)
        ema = buf.ema(20)
        assert ema is not None

    def test_rsi_range(self):
        buf = self._make_buffer(50)
        rsi = buf.rsi(14)
        assert rsi is not None
        assert 0 <= rsi <= 100

    def test_bollinger_bands(self):
        buf = self._make_buffer(50)
        upper, mid, lower = buf.bollinger_bands(20)
        assert upper is not None and mid is not None and lower is not None
        assert upper > mid > lower

    def test_atr(self):
        buf = self._make_buffer(50)
        atr = buf.atr(14)
        assert atr is not None
        assert atr > 0

    def test_maxlen_enforced(self):
        from connectors.binance.kline_buffer import KlineBuffer, Candle
        buf = KlineBuffer("BTCUSDT", "1m", maxlen=10)
        for i in range(20):
            buf.push(Candle("BTCUSDT", "1m", 1.0, 1.0, 1.0, 1.0, 1.0, i, 1, 0.5))
        assert buf.size == 10

    def test_to_list(self):
        buf = self._make_buffer(5)
        lst = buf.to_list()
        assert len(lst) == 5
        assert "c" in lst[0]


# ─────────────────────────────────────────────────────────────
# BinanceLiveAdapter tests (all mocked)
# ─────────────────────────────────────────────────────────────

class TestBinanceLiveAdapter:
    @pytest.mark.asyncio
    async def test_paper_guard_blocks_orders(self):
        from connectors.binance.adapter import BinanceLiveAdapter
        with patch("connectors.binance.adapter.BinanceWSFeed") as MockFeed, \
             patch("connectors.binance.adapter.BinanceRESTClient") as MockREST:
            MockFeed.return_value.start = AsyncMock()
            MockFeed.return_value.stop = AsyncMock()
            MockFeed.return_value.get_stats = MagicMock(return_value={})
            MockREST.return_value.get_depth = AsyncMock(return_value={
                "bids": [["65000", "1.0"]], "asks": [["65010", "1.0"]], "lastUpdateId": 1
            })
            MockREST.return_value.close = AsyncMock()

            adapter = BinanceLiveAdapter(
                symbols=["BTCUSDT"],
                paper_guard=True,
            )
            await adapter.start()
            result = await adapter.place_order("BTCUSDT", "BUY", "MARKET", 0.001)
            assert result["status"] == "BLOCKED"
            await adapter.stop()

    @pytest.mark.asyncio
    async def test_trade_callback_fires(self):
        from connectors.binance.adapter import BinanceLiveAdapter
        from connectors.binance.ws_feed import TradeEvent

        received = []

        async def on_tick(sym, price, event):
            received.append((sym, price))

        with patch("connectors.binance.adapter.BinanceWSFeed") as MockFeed, \
             patch("connectors.binance.adapter.BinanceRESTClient") as MockREST:
            MockFeed.return_value.start = AsyncMock()
            MockFeed.return_value.stop = AsyncMock()
            MockFeed.return_value.get_stats = MagicMock(return_value={})
            MockREST.return_value.get_depth = AsyncMock(return_value={
                "bids": [["65000", "1.0"]], "asks": [["65010", "1.0"]], "lastUpdateId": 1
            })
            MockREST.return_value.close = AsyncMock()

            adapter = BinanceLiveAdapter(
                symbols=["BTCUSDT"],
                paper_guard=True,
                on_tick=on_tick,
            )
            await adapter.start()

            event = TradeEvent(
                symbol="BTCUSDT", price=65500.0, qty=0.1,
                timestamp_ms=int(time.time() * 1000),
                is_buyer_maker=False, trade_id=123456,
            )
            await adapter._handle_trade(event)
            assert len(received) == 1
            assert received[0] == ("BTCUSDT", 65500.0)
            await adapter.stop()

    @pytest.mark.asyncio
    async def test_depth_update_applied_to_book(self):
        from connectors.binance.adapter import BinanceLiveAdapter
        from connectors.binance.ws_feed import DepthEvent

        with patch("connectors.binance.adapter.BinanceWSFeed") as MockFeed, \
             patch("connectors.binance.adapter.BinanceRESTClient") as MockREST:
            MockFeed.return_value.start = AsyncMock()
            MockFeed.return_value.stop = AsyncMock()
            MockFeed.return_value.get_stats = MagicMock(return_value={})
            MockREST.return_value.get_depth = AsyncMock(return_value={
                "bids": [["65000", "1.0"]], "asks": [["65010", "1.0"]], "lastUpdateId": 100
            })
            MockREST.return_value.close = AsyncMock()

            adapter = BinanceLiveAdapter(symbols=["BTCUSDT"], paper_guard=True)
            await adapter.start()

            depth_event = DepthEvent(
                symbol="BTCUSDT",
                bids=[(65005.0, 2.0)],
                asks=[(65015.0, 1.5)],
                last_update_id=101,
            )
            await adapter._handle_depth(depth_event)
            book = adapter.order_book("BTCUSDT")
            assert book is not None
            await adapter.stop()

    def test_get_stats_structure(self):
        from connectors.binance.adapter import BinanceLiveAdapter
        with patch("connectors.binance.adapter.BinanceWSFeed") as MockFeed, \
             patch("connectors.binance.adapter.BinanceRESTClient"):
            MockFeed.return_value.get_stats = MagicMock(return_value={"messages_received": 0})
            adapter = BinanceLiveAdapter(symbols=["BTCUSDT"])
            stats = adapter.get_stats()
            assert "feed" in stats
            assert "order_books" in stats
            assert "paper_guard" in stats
