"""Tests for data/orderbook/l2_feed.py — L2OrderbookFeed, PriceLevel, OrderbookSnapshot."""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data.orderbook.l2_feed import (
    L2OrderbookFeed,
    OrderbookSnapshot,
    PriceLevel,
    _from_kraken_symbol,
    _to_kraken_symbol,
)


# ─── PriceLevel tests ────────────────────────────────────────────────────────

class TestPriceLevel:
    def test_creation(self):
        pl = PriceLevel(price=100.5, size=1.2)
        assert pl.price == 100.5
        assert pl.size == 1.2

    def test_frozen(self):
        pl = PriceLevel(price=50.0, size=0.5)
        with pytest.raises(AttributeError):
            pl.price = 999  # type: ignore[misc]

    def test_equality(self):
        a = PriceLevel(price=10.0, size=1.0)
        b = PriceLevel(price=10.0, size=1.0)
        assert a == b


# ─── OrderbookSnapshot tests ─────────────────────────────────────────────────

class TestOrderbookSnapshot:
    def test_empty_snapshot_mid_price(self):
        snap = OrderbookSnapshot(symbol="BTC/USD")
        assert snap.mid_price is None

    def test_empty_snapshot_spread_bps(self):
        snap = OrderbookSnapshot(symbol="BTC/USD")
        assert snap.spread_bps is None

    def test_mid_price(self):
        snap = OrderbookSnapshot(
            symbol="BTC/USD",
            bids=[PriceLevel(price=100.0, size=1.0)],
            asks=[PriceLevel(price=102.0, size=1.0)],
        )
        assert snap.mid_price == pytest.approx(101.0)

    def test_spread_bps(self):
        snap = OrderbookSnapshot(
            symbol="BTC/USD",
            bids=[PriceLevel(price=100.0, size=1.0)],
            asks=[PriceLevel(price=102.0, size=1.0)],
        )
        # spread = 2.0, mid = 101.0, bps = (2/101) * 10000 ≈ 198.02
        assert snap.spread_bps is not None
        assert snap.spread_bps == pytest.approx(198.0198, rel=1e-3)

    def test_spread_bps_zero_mid(self):
        snap = OrderbookSnapshot(
            symbol="BTC/USD",
            bids=[PriceLevel(price=0.0, size=1.0)],
            asks=[PriceLevel(price=0.0, size=1.0)],
        )
        assert snap.spread_bps is None

    def test_symbol_preserved(self):
        snap = OrderbookSnapshot(symbol="ETH/USD")
        assert snap.symbol == "ETH/USD"

    def test_timestamp_auto_set(self):
        snap = OrderbookSnapshot(symbol="BTC/USD")
        assert isinstance(snap.timestamp, datetime)


# ─── Symbol mapping tests ────────────────────────────────────────────────────

class TestSymbolMapping:
    def test_btc_to_kraken(self):
        # Kraken WS v2 uses BTC/USD directly
        assert _to_kraken_symbol("BTC/USD") == "BTC/USD"

    def test_eth_passthrough(self):
        assert _to_kraken_symbol("ETH/USD") == "ETH/USD"

    def test_from_kraken_xbt(self):
        # With empty map, XBT passes through
        assert _from_kraken_symbol("BTC/USD") == "BTC/USD"

    def test_from_kraken_passthrough(self):
        assert _from_kraken_symbol("ETH/USD") == "ETH/USD"


# ─── L2OrderbookFeed unit tests ──────────────────────────────────────────────

class TestL2OrderbookFeed:
    def test_get_book_not_connected(self):
        feed = L2OrderbookFeed(exchange="kraken", depth=10)
        assert feed.get_book("BTC/USD") is None

    def test_is_connected_default(self):
        feed = L2OrderbookFeed()
        assert feed.is_connected is False

    def test_depth_stored(self):
        feed = L2OrderbookFeed(depth=50)
        assert feed.depth == 50

    def test_apply_snapshot(self):
        feed = L2OrderbookFeed()
        feed._books["BTC/USD"] = {"bids": {}, "asks": {}}
        feed._apply_snapshot(
            "BTC/USD",
            [{"price": "100.0", "qty": "1.0"}, {"price": "99.0", "qty": "2.0"}],
            [{"price": "101.0", "qty": "0.5"}, {"price": "102.0", "qty": "1.5"}],
        )
        book = feed.get_book("BTC/USD")
        assert book is not None
        assert len(book.bids) == 2
        assert len(book.asks) == 2
        assert book.bids[0].price == 100.0  # highest bid first
        assert book.asks[0].price == 101.0  # lowest ask first

    def test_apply_delta_add(self):
        feed = L2OrderbookFeed()
        feed._books["BTC/USD"] = {"bids": {100.0: 1.0}, "asks": {101.0: 0.5}}
        feed._apply_delta(
            "BTC/USD",
            [{"price": "99.0", "qty": "3.0"}],
            [{"price": "102.0", "qty": "2.0"}],
        )
        book = feed.get_book("BTC/USD")
        assert book is not None
        assert len(book.bids) == 2
        assert len(book.asks) == 2

    def test_apply_delta_remove(self):
        feed = L2OrderbookFeed()
        feed._books["BTC/USD"] = {"bids": {100.0: 1.0, 99.0: 2.0}, "asks": {101.0: 0.5}}
        # qty=0 means remove that level
        feed._apply_delta(
            "BTC/USD",
            [{"price": "99.0", "qty": "0"}],
            [],
        )
        book = feed.get_book("BTC/USD")
        assert book is not None
        assert len(book.bids) == 1
        assert book.bids[0].price == 100.0

    def test_mid_price_after_snapshot(self):
        feed = L2OrderbookFeed()
        feed._books["ETH/USD"] = {"bids": {}, "asks": {}}
        feed._apply_snapshot(
            "ETH/USD",
            [{"price": "2000.0", "qty": "5.0"}],
            [{"price": "2010.0", "qty": "3.0"}],
        )
        book = feed.get_book("ETH/USD")
        assert book is not None
        assert book.mid_price == pytest.approx(2005.0)

    def test_spread_bps_after_snapshot(self):
        feed = L2OrderbookFeed()
        feed._books["ETH/USD"] = {"bids": {}, "asks": {}}
        feed._apply_snapshot(
            "ETH/USD",
            [{"price": "2000.0", "qty": "5.0"}],
            [{"price": "2010.0", "qty": "3.0"}],
        )
        book = feed.get_book("ETH/USD")
        assert book is not None
        # spread = 10, mid = 2005, bps = (10/2005)*10000 ≈ 49.88
        assert book.spread_bps == pytest.approx(49.875, rel=1e-2)

    def test_depth_truncation(self):
        feed = L2OrderbookFeed(depth=2)
        feed._books["BTC/USD"] = {
            "bids": {100.0: 1.0, 99.0: 2.0, 98.0: 3.0},
            "asks": {101.0: 0.5, 102.0: 1.0, 103.0: 1.5},
        }
        book = feed.get_book("BTC/USD")
        assert book is not None
        assert len(book.bids) == 2
        assert len(book.asks) == 2

    @pytest.mark.asyncio
    async def test_subscribe_no_websockets(self):
        feed = L2OrderbookFeed()
        with patch("data.orderbook.l2_feed._HAS_WEBSOCKETS", False):
            result = await feed.subscribe(["BTC/USD"])
            assert result is False

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        feed = L2OrderbookFeed()
        # Should not raise
        await feed.disconnect()
        assert feed.is_connected is False

    @pytest.mark.asyncio
    async def test_dispatch_snapshot(self):
        feed = L2OrderbookFeed()
        feed._books["BTC/USD"] = {"bids": {}, "asks": {}}
        msg = json.dumps({
            "channel": "book",
            "type": "snapshot",
            "data": [{
                "symbol": "BTC/USD",
                "bids": [{"price": "50000.0", "qty": "1.0"}],
                "asks": [{"price": "50100.0", "qty": "0.5"}],
            }],
        })
        await feed._dispatch(msg)
        book = feed.get_book("BTC/USD")
        assert book is not None
        assert book.bids[0].price == 50000.0

    @pytest.mark.asyncio
    async def test_dispatch_delta(self):
        feed = L2OrderbookFeed()
        feed._books["BTC/USD"] = {"bids": {50000.0: 1.0}, "asks": {50100.0: 0.5}}
        msg = json.dumps({
            "channel": "book",
            "type": "update",
            "data": [{
                "symbol": "BTC/USD",
                "bids": [{"price": "49900.0", "qty": "2.0"}],
                "asks": [],
            }],
        })
        await feed._dispatch(msg)
        book = feed.get_book("BTC/USD")
        assert book is not None
        assert len(book.bids) == 2

    @pytest.mark.asyncio
    async def test_dispatch_invalid_json(self):
        feed = L2OrderbookFeed()
        # Should not raise
        await feed._dispatch("not json at all")

    @pytest.mark.asyncio
    async def test_dispatch_heartbeat(self):
        feed = L2OrderbookFeed()
        msg = json.dumps({"channel": "heartbeat"})
        # Should not raise
        await feed._dispatch(msg)

    @pytest.mark.asyncio
    async def test_context_manager(self):
        feed = L2OrderbookFeed()
        async with feed as f:
            assert f is feed
        assert feed.is_connected is False
