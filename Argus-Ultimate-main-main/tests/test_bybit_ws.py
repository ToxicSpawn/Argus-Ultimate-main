"""Tests for core/connectors/bybit_ws_connector.py — BybitWSConnector."""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.connectors.bybit_ws_connector import (
    BybitWSConnector,
    to_bybit_symbol,
    from_bybit_symbol,
    INITIAL_BACKOFF_S,
    MAX_BACKOFF_S,
)


# ─── Symbol mapping ──────────────────────────────────────────────────────────

class TestBybitSymbolMapping:
    def test_btc_usd_to_bybit(self):
        assert to_bybit_symbol("BTC/USD") == "BTCUSDT"

    def test_btc_usdt_to_bybit(self):
        assert to_bybit_symbol("BTC/USDT") == "BTCUSDT"

    def test_eth_usd_to_bybit(self):
        assert to_bybit_symbol("ETH/USD") == "ETHUSDT"

    def test_sol_usdt_to_bybit(self):
        assert to_bybit_symbol("SOL/USDT") == "SOLUSDT"

    def test_unknown_symbol_best_effort(self):
        # Unknown pair — best effort: strip slash, append USDT if needed
        result = to_bybit_symbol("MATIC/USD")
        assert result == "MATICUSDT"

    def test_from_bybit_btcusdt(self):
        assert from_bybit_symbol("BTCUSDT") == "BTC/USDT"

    def test_from_bybit_ethusdt(self):
        assert from_bybit_symbol("ETHUSDT") == "ETH/USDT"

    def test_from_bybit_unknown(self):
        result = from_bybit_symbol("FOOUSDT")
        assert result == "FOO/USDT"

    def test_from_bybit_no_usdt(self):
        result = from_bybit_symbol("FOOBAR")
        assert result == "FOOBAR"


# ─── Message parsing ─────────────────────────────────────────────────────────

class TestBybitMessageParsing:
    def setup_method(self):
        self.connector = BybitWSConnector(symbols=["BTC/USDT"])

    def test_parse_ticker(self):
        data = {
            "topic": "tickers.BTCUSDT",
            "type": "snapshot",
            "data": {
                "symbol": "BTCUSDT",
                "bid1Price": "50000.0",
                "ask1Price": "50100.0",
                "lastPrice": "50050.0",
                "volume24h": "1234.5",
            },
        }
        result = self.connector._parse_ticker(data)
        assert result is not None
        assert result["symbol"] == "BTC/USDT"
        assert result["bid"] == 50000.0
        assert result["ask"] == 50100.0
        assert result["last"] == 50050.0
        assert result["volume_24h"] == 1234.5
        assert isinstance(result["timestamp"], datetime)

    def test_parse_ticker_missing_fields(self):
        data = {
            "topic": "tickers.BTCUSDT",
            "data": {"symbol": "BTCUSDT"},
        }
        result = self.connector._parse_ticker(data)
        assert result is not None
        assert result["bid"] == 0
        assert result["ask"] == 0

    def test_parse_book_snapshot(self):
        data = {
            "topic": "orderbook.25.BTCUSDT",
            "type": "snapshot",
            "data": {
                "s": "BTCUSDT",
                "b": [["50000.0", "1.0"], ["49900.0", "2.0"]],
                "a": [["50100.0", "0.5"], ["50200.0", "1.5"]],
            },
        }
        result = self.connector._parse_book(data)
        assert result is not None
        assert result["symbol"] == "BTC/USDT"
        assert result["type"] == "snapshot"
        assert len(result["bids"]) == 2
        assert len(result["asks"]) == 2
        assert result["bids"][0]["price"] == 50000.0
        assert result["bids"][0]["qty"] == 1.0

    def test_parse_book_delta(self):
        data = {
            "topic": "orderbook.25.BTCUSDT",
            "type": "delta",
            "data": {
                "s": "BTCUSDT",
                "b": [["49800.0", "3.0"]],
                "a": [],
            },
        }
        result = self.connector._parse_book(data)
        assert result is not None
        assert result["type"] == "update"
        assert len(result["bids"]) == 1
        assert result["asks"] == []

    def test_parse_book_empty_data(self):
        data = {
            "topic": "orderbook.25.BTCUSDT",
            "type": "snapshot",
            "data": {"s": "BTCUSDT", "b": [], "a": []},
        }
        result = self.connector._parse_book(data)
        assert result is not None
        assert result["bids"] == []
        assert result["asks"] == []


# ─── Dispatch ─────────────────────────────────────────────────────────────────

class TestBybitDispatch:
    def setup_method(self):
        self.connector = BybitWSConnector(symbols=["BTC/USDT"])

    @pytest.mark.asyncio
    async def test_dispatch_ticker_calls_callback(self):
        received = []
        self.connector.on_ticker(lambda d: received.append(d))

        msg = json.dumps({
            "topic": "tickers.BTCUSDT",
            "type": "snapshot",
            "data": {
                "symbol": "BTCUSDT",
                "bid1Price": "50000.0",
                "ask1Price": "50100.0",
                "lastPrice": "50050.0",
                "volume24h": "100.0",
            },
        })
        await self.connector._dispatch(msg)
        assert len(received) == 1
        assert received[0]["symbol"] == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_dispatch_book_calls_callback(self):
        received = []
        self.connector.on_book_update(lambda d: received.append(d))

        msg = json.dumps({
            "topic": "orderbook.25.BTCUSDT",
            "type": "snapshot",
            "data": {
                "s": "BTCUSDT",
                "b": [["50000.0", "1.0"]],
                "a": [["50100.0", "0.5"]],
            },
        })
        await self.connector._dispatch(msg)
        assert len(received) == 1
        assert received[0]["type"] == "snapshot"

    @pytest.mark.asyncio
    async def test_dispatch_ping_handled(self):
        msg = json.dumps({"op": "ping"})
        # In dispatch, ping messages are handled (no crash)
        await self.connector._dispatch(msg)

    @pytest.mark.asyncio
    async def test_dispatch_subscription_confirmed(self):
        msg = json.dumps({"op": "subscribe", "success": True, "ret_msg": "ok"})
        await self.connector._dispatch(msg)

    @pytest.mark.asyncio
    async def test_dispatch_subscription_failed(self):
        msg = json.dumps({"op": "subscribe", "success": False, "ret_msg": "invalid topic"})
        await self.connector._dispatch(msg)

    @pytest.mark.asyncio
    async def test_dispatch_invalid_json(self):
        await self.connector._dispatch("not json")

    @pytest.mark.asyncio
    async def test_dispatch_async_callback(self):
        received = []

        async def async_cb(data):
            received.append(data)

        self.connector.on_ticker(async_cb)
        msg = json.dumps({
            "topic": "tickers.BTCUSDT",
            "type": "snapshot",
            "data": {
                "symbol": "BTCUSDT",
                "bid1Price": "1",
                "ask1Price": "2",
                "lastPrice": "1.5",
                "volume24h": "10",
            },
        })
        await self.connector._dispatch(msg)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_dispatch_callback_error_handled(self):
        def bad_cb(data):
            raise ValueError("boom")

        self.connector.on_ticker(bad_cb)
        msg = json.dumps({
            "topic": "tickers.BTCUSDT",
            "type": "snapshot",
            "data": {
                "symbol": "BTCUSDT",
                "bid1Price": "1",
                "ask1Price": "2",
                "lastPrice": "1.5",
                "volume24h": "10",
            },
        })
        # Should not raise
        await self.connector._dispatch(msg)


# ─── Reconnect logic ─────────────────────────────────────────────────────────

class TestBybitReconnect:
    def test_initial_backoff(self):
        c = BybitWSConnector()
        assert c._backoff_s == INITIAL_BACKOFF_S

    @pytest.mark.asyncio
    async def test_reconnect_increments_count(self):
        c = BybitWSConnector()
        c._ws = None
        with patch("core.connectors.bybit_ws_connector._HAS_WEBSOCKETS", False):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await c._reconnect()
        assert c._reconnect_count == 1

    @pytest.mark.asyncio
    async def test_reconnect_backoff_doubles(self):
        c = BybitWSConnector()
        c._ws = None
        with patch("core.connectors.bybit_ws_connector._HAS_WEBSOCKETS", False):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await c._reconnect()
                first_backoff = c._backoff_s
                await c._reconnect()
                assert c._backoff_s == min(first_backoff * 2, MAX_BACKOFF_S)

    @pytest.mark.asyncio
    async def test_backoff_caps_at_max(self):
        c = BybitWSConnector()
        c._ws = None
        c._backoff_s = MAX_BACKOFF_S
        with patch("core.connectors.bybit_ws_connector._HAS_WEBSOCKETS", False):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await c._reconnect()
        assert c._backoff_s == MAX_BACKOFF_S


# ─── Ping/pong handling ──────────────────────────────────────────────────────

class TestBybitPingPong:
    @pytest.mark.asyncio
    async def test_receive_loop_handles_ping(self):
        """Verify that a ping message triggers a pong response."""
        c = BybitWSConnector()
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=[
            json.dumps({"op": "ping"}),
            asyncio.CancelledError(),
        ])
        mock_ws.send = AsyncMock()
        c._ws = mock_ws
        c.connected = True
        c._running = True
        c._last_message_time = time.monotonic()

        try:
            await c._receive_loop()
        except asyncio.CancelledError:
            pass

        # Verify pong was sent
        assert mock_ws.send.called
        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["op"] == "pong"


# ─── Connection / lifecycle ───────────────────────────────────────────────────

class TestBybitLifecycle:
    def test_defaults(self):
        c = BybitWSConnector()
        assert c.connected is False
        assert c.symbols == ["BTC/USDT", "ETH/USDT"]
        assert "tickers" in c.topics
        assert "orderbook.25" in c.topics

    @pytest.mark.asyncio
    async def test_connect_no_websockets(self):
        c = BybitWSConnector()
        with patch("core.connectors.bybit_ws_connector._HAS_WEBSOCKETS", False):
            result = await c._connect()
            assert result is False
            assert c.connected is False

    @pytest.mark.asyncio
    async def test_stop_when_not_connected(self):
        c = BybitWSConnector()
        await c.stop()
        assert c.connected is False

    def test_callback_registration(self):
        c = BybitWSConnector()
        cb = lambda d: None
        c.on_ticker(cb)
        c.on_book_update(cb)
        assert len(c._ticker_callbacks) == 1
        assert len(c._book_callbacks) == 1
