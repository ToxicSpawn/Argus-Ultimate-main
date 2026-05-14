"""Push 52 — Bybit WebSocket connector: 24 tests."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# BybitModels tests (8)
# ---------------------------------------------------------------------------
from connectors.bybit.bybit_models import BybitTick, BybitOrderBook, BybitTicker, BybitKline


class TestBybitModels:
    def test_tick_from_ws_data(self):
        d = {"s": "BTCUSDT", "p": "65000.5", "v": "0.01", "S": "Buy", "T": 1700000000000, "i": "abc"}
        tick = BybitTick.from_ws_data(d)
        assert tick.symbol == "BTCUSDT"
        assert tick.price == pytest.approx(65000.5)
        assert tick.side == "Buy"

    def test_tick_sell_side(self):
        d = {"s": "ETHUSDT", "p": "3000", "v": "1.5", "S": "Sell", "T": 0, "i": ""}
        tick = BybitTick.from_ws_data(d)
        assert tick.side == "Sell"

    def test_orderbook_best_bid_ask(self):
        book = BybitOrderBook(
            symbol="BTCUSDT",
            bids=[[65000.0, 1.0], [64999.0, 2.0]],
            asks=[[65001.0, 0.5]],
        )
        assert book.best_bid == pytest.approx(65000.0)
        assert book.best_ask == pytest.approx(65001.0)

    def test_orderbook_mid_price(self):
        book = BybitOrderBook(
            symbol="BTCUSDT",
            bids=[[65000.0, 1.0]],
            asks=[[65002.0, 0.5]],
        )
        assert book.mid_price == pytest.approx(65001.0)

    def test_orderbook_spread_bps(self):
        book = BybitOrderBook(
            symbol="BTCUSDT",
            bids=[[65000.0, 1.0]],
            asks=[[65065.0, 0.5]],
        )
        assert book.spread_bps is not None
        assert book.spread_bps == pytest.approx(10.0, rel=1e-3)

    def test_orderbook_empty_returns_none(self):
        book = BybitOrderBook(symbol="BTCUSDT")
        assert book.best_bid is None
        assert book.mid_price is None

    def test_ticker_from_ws_data(self):
        d = {
            "symbol": "BTCUSDT", "lastPrice": "65000",
            "bid1Price": "64999", "ask1Price": "65001",
            "volume24h": "10000", "turnover24h": "650000000",
            "price24hPcnt": "0.02", "ts": 1700000000000,
        }
        ticker = BybitTicker.from_ws_data(d)
        assert ticker.last_price == pytest.approx(65000.0)
        assert ticker.price_change_pct_24h == pytest.approx(0.02)

    def test_kline_fields(self):
        k = BybitKline("BTCUSDT", "1", 1700000000000, 65000, 65100, 64900, 65050, 100.0, 6500000.0)
        assert k.close == pytest.approx(65050.0)


# ---------------------------------------------------------------------------
# BybitWsClient tests (8)
# ---------------------------------------------------------------------------
from connectors.bybit.bybit_ws_client import BybitWsClient


class TestBybitWsClient:
    def test_init_default_url_mainnet(self):
        client = BybitWsClient(symbols=["BTCUSDT"])
        assert "bybit.com" in client._url
        assert "testnet" not in client._url

    def test_init_testnet(self):
        client = BybitWsClient(symbols=["BTCUSDT"], testnet=True)
        assert "testnet" in client._url

    def test_subscription_topics_count(self):
        client = BybitWsClient(symbols=["BTCUSDT", "ETHUSDT"])
        topics = client._subscription_args()
        # 3 topics per symbol
        assert len(topics) == 6

    def test_subscription_msg_json(self):
        client = BybitWsClient(symbols=["BTCUSDT"])
        msg = json.loads(client._subscribe_msg())
        assert msg["op"] == "subscribe"
        assert any("BTCUSDT" in t for t in msg["args"])

    def test_is_running_false_on_init(self):
        client = BybitWsClient(symbols=["BTCUSDT"])
        assert client.is_running is False

    def test_stop_sets_running_false(self):
        client = BybitWsClient(symbols=["BTCUSDT"])
        client._running = True
        client.stop()
        assert client.is_running is False

    def test_apply_delta_add(self):
        side = [[65000.0, 1.0]]
        BybitWsClient._apply_delta(side, 65001.0, 0.5)
        assert any(e[0] == 65001.0 for e in side)

    def test_apply_delta_remove(self):
        side = [[65000.0, 1.0], [65001.0, 0.5]]
        BybitWsClient._apply_delta(side, 65000.0, 0.0)
        assert not any(e[0] == 65000.0 for e in side)

    def test_dispatch_tick_callback(self):
        received = []

        async def on_tick(tick):
            received.append(tick)

        client = BybitWsClient(symbols=["BTCUSDT"], on_tick=on_tick)
        msg = json.dumps({
            "topic": "publicTrade.BTCUSDT",
            "data": [{"s": "BTCUSDT", "p": "65000", "v": "0.01", "S": "Buy", "T": 0, "i": ""}],
        })
        asyncio.get_event_loop().run_until_complete(client._dispatch(msg))
        assert len(received) == 1
        assert received[0].symbol == "BTCUSDT"


# ---------------------------------------------------------------------------
# BybitRestClient tests (4)
# ---------------------------------------------------------------------------
from connectors.bybit.bybit_rest_client import BybitRestClient


class TestBybitRestClient:
    def test_init_mainnet(self):
        client = BybitRestClient()
        assert "bybit.com" in client._base

    def test_init_testnet(self):
        client = BybitRestClient(testnet=True)
        assert "testnet" in client._base

    def test_sign_returns_hex_string(self):
        client = BybitRestClient(api_key="key", api_secret="secret")
        sig = client._sign("params", "1700000000000")
        assert isinstance(sig, str)
        assert len(sig) == 64

    def test_category_default(self):
        client = BybitRestClient()
        assert client.category == "linear"


# ---------------------------------------------------------------------------
# BybitFeedAdapter tests (4)
# ---------------------------------------------------------------------------
from connectors.bybit.bybit_feed_adapter import BybitFeedAdapter


class TestBybitFeedAdapter:
    def test_init_symbols_uppercase(self):
        adapter = BybitFeedAdapter(symbols=["btcusdt", "ethusdt"])
        assert "BTCUSDT" in adapter.symbols
        assert "ETHUSDT" in adapter.symbols

    def test_ws_client_created(self):
        adapter = BybitFeedAdapter(symbols=["BTCUSDT"])
        assert adapter.ws_client is not None

    def test_on_book_updates_last_mid(self):
        adapter = BybitFeedAdapter(symbols=["BTCUSDT"])
        book = BybitOrderBook(
            symbol="BTCUSDT",
            bids=[[65000.0, 1.0]],
            asks=[[65002.0, 0.5]],
            timestamp_ms=1700000000000,
        )
        asyncio.get_event_loop().run_until_complete(adapter._on_book(book))
        assert adapter._last_mid.get("BTCUSDT") == pytest.approx(65001.0)

    def test_flush_bar_no_crash_on_empty(self):
        adapter = BybitFeedAdapter(symbols=["BTCUSDT"])
        asyncio.get_event_loop().run_until_complete(adapter._flush_bar("BTCUSDT"))
        # Should not raise even with empty tick buffer
