"""Tests for core/connectors/kraken_ws_connector.py — KrakenWSConnector."""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.connectors.kraken_ws_connector import (
    KrakenWSConnector,
    to_kraken_symbol,
    from_kraken_symbol,
    INITIAL_BACKOFF_S,
    MAX_BACKOFF_S,
)


# ─── Symbol mapping ──────────────────────────────────────────────────────────

class TestKrakenSymbolMapping:
    def test_btc_usd_to_kraken(self):
        # Kraken WS v2 uses BTC/USD directly (not XBT/USD)
        assert to_kraken_symbol("BTC/USD") == "BTC/USD"

    def test_btc_aud_to_kraken(self):
        assert to_kraken_symbol("BTC/AUD") == "BTC/AUD"

    def test_eth_passthrough(self):
        assert to_kraken_symbol("ETH/USD") == "ETH/USD"

    def test_sol_passthrough(self):
        assert to_kraken_symbol("SOL/USD") == "SOL/USD"

    def test_from_kraken_xbt_usd(self):
        # With empty map, XBT/USD passes through (WS v2 uses BTC/USD directly)
        assert from_kraken_symbol("XBT/USD") == "XBT/USD"

    def test_from_kraken_xbt_aud(self):
        assert from_kraken_symbol("XBT/AUD") == "XBT/AUD"

    def test_from_kraken_passthrough(self):
        assert from_kraken_symbol("ETH/USD") == "ETH/USD"


# ─── Message parsing ─────────────────────────────────────────────────────────

class TestKrakenMessageParsing:
    def setup_method(self):
        self.connector = KrakenWSConnector(symbols=["BTC/USD", "ETH/USD"])

    def test_parse_ticker(self):
        data = {
            "channel": "ticker",
            "type": "update",
            "data": [{
                "symbol": "BTC/USD",
                "bid": 50000.0,
                "ask": 50100.0,
                "last": 50050.0,
                "volume": 1234.5,
            }],
        }
        results = self.connector._parse_ticker(data)
        assert len(results) == 1
        r = results[0]
        assert r["symbol"] == "BTC/USD"
        assert r["bid"] == 50000.0
        assert r["ask"] == 50100.0
        assert r["last"] == 50050.0
        assert r["volume_24h"] == 1234.5
        assert isinstance(r["timestamp"], datetime)

    def test_parse_ticker_empty_data(self):
        data = {"channel": "ticker", "data": []}
        results = self.connector._parse_ticker(data)
        assert results == []

    def test_parse_book_snapshot(self):
        data = {
            "channel": "book",
            "type": "snapshot",
            "data": [{
                "symbol": "BTC/USD",
                "bids": [{"price": "50000.0", "qty": "1.0"}],
                "asks": [{"price": "50100.0", "qty": "0.5"}],
            }],
        }
        results = self.connector._parse_book(data)
        assert len(results) == 1
        r = results[0]
        assert r["symbol"] == "BTC/USD"
        assert r["type"] == "snapshot"
        assert len(r["bids"]) == 1
        assert len(r["asks"]) == 1

    def test_parse_book_delta(self):
        data = {
            "channel": "book",
            "type": "update",
            "data": [{
                "symbol": "ETH/USD",
                "bids": [{"price": "2000.0", "qty": "3.0"}],
                "asks": [],
            }],
        }
        results = self.connector._parse_book(data)
        assert len(results) == 1
        assert results[0]["type"] == "update"
        assert results[0]["symbol"] == "ETH/USD"

    def test_parse_ticker_malformed(self):
        data = {"channel": "ticker", "data": [{"symbol": "BTC/USD"}]}
        results = self.connector._parse_ticker(data)
        assert len(results) == 1
        assert results[0]["bid"] == 0  # defaults to 0


# ─── Dispatch ─────────────────────────────────────────────────────────────────

class TestKrakenDispatch:
    def setup_method(self):
        self.connector = KrakenWSConnector(symbols=["BTC/USD"])

    @pytest.mark.asyncio
    async def test_dispatch_ticker_calls_callback(self):
        received = []
        self.connector.on_ticker(lambda d: received.append(d))

        msg = json.dumps({
            "channel": "ticker",
            "type": "update",
            "data": [{
                "symbol": "BTC/USD",
                "bid": 50000.0,
                "ask": 50100.0,
                "last": 50050.0,
                "volume": 100.0,
            }],
        })
        await self.connector._dispatch(msg)
        assert len(received) == 1
        assert received[0]["symbol"] == "BTC/USD"

    @pytest.mark.asyncio
    async def test_dispatch_book_calls_callback(self):
        received = []
        self.connector.on_book_update(lambda d: received.append(d))

        msg = json.dumps({
            "channel": "book",
            "type": "snapshot",
            "data": [{
                "symbol": "BTC/USD",
                "bids": [{"price": "50000.0", "qty": "1.0"}],
                "asks": [{"price": "50100.0", "qty": "0.5"}],
            }],
        })
        await self.connector._dispatch(msg)
        assert len(received) == 1
        assert received[0]["type"] == "snapshot"

    @pytest.mark.asyncio
    async def test_dispatch_heartbeat_no_error(self):
        msg = json.dumps({"channel": "heartbeat"})
        await self.connector._dispatch(msg)

    @pytest.mark.asyncio
    async def test_dispatch_subscription_confirmed(self):
        msg = json.dumps({"method": "subscribe", "success": True})
        await self.connector._dispatch(msg)

    @pytest.mark.asyncio
    async def test_dispatch_subscription_failed(self):
        msg = json.dumps({"method": "subscribe", "success": False, "error": "bad request"})
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
            "channel": "ticker",
            "type": "update",
            "data": [{"symbol": "BTC/USD", "bid": 1.0, "ask": 2.0, "last": 1.5, "volume": 10.0}],
        })
        await self.connector._dispatch(msg)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_dispatch_callback_error_handled(self):
        def bad_cb(data):
            raise ValueError("boom")

        self.connector.on_ticker(bad_cb)
        msg = json.dumps({
            "channel": "ticker",
            "type": "update",
            "data": [{"symbol": "BTC/USD", "bid": 1.0, "ask": 2.0, "last": 1.5, "volume": 10.0}],
        })
        # Should not raise
        await self.connector._dispatch(msg)


# ─── Reconnect logic ─────────────────────────────────────────────────────────

class TestKrakenReconnect:
    def test_initial_backoff(self):
        c = KrakenWSConnector()
        assert c._backoff_s == INITIAL_BACKOFF_S

    @pytest.mark.asyncio
    async def test_reconnect_increments_count(self):
        c = KrakenWSConnector()
        c._ws = None
        with patch("core.connectors.kraken_ws_connector._HAS_WEBSOCKETS", False):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await c._reconnect()
        assert c._reconnect_count == 1

    @pytest.mark.asyncio
    async def test_reconnect_backoff_doubles(self):
        c = KrakenWSConnector()
        c._ws = None
        with patch("core.connectors.kraken_ws_connector._HAS_WEBSOCKETS", False):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await c._reconnect()
                first_backoff = c._backoff_s
                await c._reconnect()
                assert c._backoff_s == min(first_backoff * 2, MAX_BACKOFF_S)

    @pytest.mark.asyncio
    async def test_backoff_caps_at_max(self):
        c = KrakenWSConnector()
        c._ws = None
        c._backoff_s = MAX_BACKOFF_S
        with patch("core.connectors.kraken_ws_connector._HAS_WEBSOCKETS", False):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await c._reconnect()
        assert c._backoff_s == MAX_BACKOFF_S


# ─── Connection / lifecycle ───────────────────────────────────────────────────

class TestKrakenLifecycle:
    def test_defaults(self):
        c = KrakenWSConnector()
        assert c.connected is False
        assert c.symbols == ["BTC/AUD", "ETH/AUD"]
        assert "ticker" in c.channels
        assert "book" in c.channels

    @pytest.mark.asyncio
    async def test_connect_no_websockets(self):
        c = KrakenWSConnector()
        with patch("core.connectors.kraken_ws_connector._HAS_WEBSOCKETS", False):
            result = await c._connect()
            assert result is False
            assert c.connected is False

    @pytest.mark.asyncio
    async def test_stop_when_not_connected(self):
        c = KrakenWSConnector()
        await c.stop()
        assert c.connected is False

    def test_callback_registration(self):
        c = KrakenWSConnector()
        cb = lambda d: None
        c.on_ticker(cb)
        c.on_book_update(cb)
        assert len(c._ticker_callbacks) == 1
        assert len(c._book_callbacks) == 1
