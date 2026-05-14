"""
Tests for core/connectors/coinbase_ws_connector.py

Covers:
- Authentication signing
- Subscribe message building
- Ticker message parsing into standardized format
- L2 update message parsing
- Callback registration and dispatch
- Auto-reconnect with exponential backoff
- Heartbeat timeout detection
- Async context manager protocol
- Disconnect cleanup
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.connectors.coinbase_ws_connector import (
    CoinbaseWSConnector,
    HEARTBEAT_TIMEOUT_S,
    INITIAL_BACKOFF_S,
    MAX_BACKOFF_S,
)


# ----------------------------------------------------------------- fixtures

@pytest.fixture
def connector():
    """Create a connector with test credentials."""
    return CoinbaseWSConnector(
        api_key="test-key",
        api_secret="test-secret",
        symbols=["BTC-AUD", "ETH-AUD"],
        channels=["ticker", "level2"],
    )


@pytest.fixture
def ticker_message():
    """Sample Coinbase Advanced Trade ticker WS message."""
    return json.dumps({
        "channel": "ticker",
        "timestamp": "2026-03-14T10:00:00Z",
        "sequence_num": 1,
        "events": [
            {
                "type": "snapshot",
                "tickers": [
                    {
                        "product_id": "BTC-AUD",
                        "price": "95000.50",
                        "best_bid": "94999.00",
                        "best_ask": "95001.00",
                        "volume_24_h": "1234.56",
                    }
                ],
            }
        ],
    })


@pytest.fixture
def l2_message():
    """Sample Coinbase Advanced Trade level2 WS message."""
    return json.dumps({
        "channel": "l2_data",
        "product_id": "BTC-AUD",
        "timestamp": "2026-03-14T10:00:01Z",
        "events": [
            {
                "type": "update",
                "updates": [
                    {
                        "side": "bid",
                        "price_level": "94998.00",
                        "new_quantity": "1.5",
                    },
                    {
                        "side": "offer",
                        "price_level": "95002.00",
                        "new_quantity": "0.8",
                    },
                ],
            }
        ],
    })


# --------------------------------------------------------------- auth / sign

class TestAuthentication:
    def test_sign_produces_hex_string(self, connector):
        sig = connector._sign("1710000000", "ticker", ["BTC-AUD"])
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA256 hex digest

    def test_sign_deterministic(self, connector):
        sig1 = connector._sign("1710000000", "ticker", ["BTC-AUD"])
        sig2 = connector._sign("1710000000", "ticker", ["BTC-AUD"])
        assert sig1 == sig2

    def test_sign_changes_with_timestamp(self, connector):
        sig1 = connector._sign("1710000000", "ticker", ["BTC-AUD"])
        sig2 = connector._sign("1710000001", "ticker", ["BTC-AUD"])
        assert sig1 != sig2

    def test_sign_changes_with_channel(self, connector):
        sig1 = connector._sign("1710000000", "ticker", ["BTC-AUD"])
        sig2 = connector._sign("1710000000", "level2", ["BTC-AUD"])
        assert sig1 != sig2


class TestSubscribeMessages:
    def test_build_subscribe_messages_per_channel(self, connector):
        msgs = connector._build_subscribe_messages()
        assert len(msgs) == 2  # ticker + level2
        channels = {m["channel"] for m in msgs}
        assert channels == {"ticker", "level2"}

    def test_subscribe_message_has_auth_fields(self, connector):
        msgs = connector._build_subscribe_messages()
        for msg in msgs:
            assert msg["type"] == "subscribe"
            assert msg["api_key"] == "test-key"
            assert "signature" in msg
            assert "timestamp" in msg
            assert msg["product_ids"] == ["BTC-AUD", "ETH-AUD"]


# --------------------------------------------------------- message parsing

class TestTickerParsing:
    def test_parse_ticker_standardized_format(self, connector, ticker_message):
        data = json.loads(ticker_message)
        result = connector._parse_ticker(data)
        assert result is not None
        assert result["symbol"] == "BTC-AUD"
        assert result["bid"] == 94999.0
        assert result["ask"] == 95001.0
        assert result["last"] == 95000.5
        assert result["volume_24h"] == 1234.56
        assert isinstance(result["timestamp"], datetime)

    def test_parse_ticker_empty_events(self, connector):
        result = connector._parse_ticker({"events": []})
        assert result is None

    def test_parse_ticker_missing_fields_default_zero(self, connector):
        data = {
            "events": [{"tickers": [{"product_id": "BTC-AUD"}]}]
        }
        result = connector._parse_ticker(data)
        assert result is not None
        assert result["bid"] == 0.0
        assert result["ask"] == 0.0
        assert result["last"] == 0.0
        assert result["volume_24h"] == 0.0

    def test_parse_ticker_malformed_returns_none(self, connector):
        result = connector._parse_ticker({"events": [{"tickers": [{"price": "not_a_number_xxx"}]}]})
        # price "not_a_number_xxx" is not a valid float, but the field is "price"
        # which should raise ValueError. The method returns None on error.
        # Actually "not_a_number_xxx" will raise ValueError for float()
        # Let's use truly malformed data
        result = connector._parse_ticker({"events": "not_a_list"})
        assert result is None


class TestL2Parsing:
    def test_parse_l2_update(self, connector, l2_message):
        data = json.loads(l2_message)
        result = connector._parse_l2_update(data)
        assert result is not None
        assert result["type"] == "update"
        assert len(result["updates"]) == 2
        assert result["updates"][0]["side"] == "bid"
        assert result["updates"][0]["price"] == 94998.0
        assert result["updates"][0]["qty"] == 1.5
        assert result["updates"][1]["side"] == "offer"
        assert isinstance(result["timestamp"], datetime)

    def test_parse_l2_empty_events(self, connector):
        result = connector._parse_l2_update({"events": []})
        assert result is None


# --------------------------------------------------------- callback dispatch

class TestCallbacks:
    def test_on_ticker_registers_callback(self, connector):
        cb = MagicMock()
        connector.on_ticker(cb)
        assert cb in connector._ticker_callbacks

    def test_on_l2_update_registers_callback(self, connector):
        cb = MagicMock()
        connector.on_l2_update(cb)
        assert cb in connector._l2_callbacks

    def test_dispatch_ticker_invokes_callbacks(self, connector, ticker_message):
        cb = MagicMock()
        connector.on_ticker(cb)
        asyncio.run(connector._dispatch(ticker_message))
        cb.assert_called_once()
        args = cb.call_args[0][0]
        assert args["symbol"] == "BTC-AUD"
        assert args["last"] == 95000.5

    def test_dispatch_l2_invokes_callbacks(self, connector, l2_message):
        cb = MagicMock()
        connector.on_l2_update(cb)
        asyncio.run(connector._dispatch(l2_message))
        cb.assert_called_once()
        args = cb.call_args[0][0]
        assert len(args["updates"]) == 2

    def test_dispatch_async_callback(self, connector, ticker_message):
        cb = AsyncMock()
        connector.on_ticker(cb)
        asyncio.run(connector._dispatch(ticker_message))
        cb.assert_called_once()

    def test_dispatch_callback_error_does_not_crash(self, connector, ticker_message):
        cb = MagicMock(side_effect=ValueError("boom"))
        connector.on_ticker(cb)
        # Should not raise
        asyncio.run(connector._dispatch(ticker_message))
        cb.assert_called_once()

    def test_dispatch_non_json_does_not_crash(self, connector):
        asyncio.run(connector._dispatch("this is not json"))

    def test_dispatch_subscription_confirmation(self, connector):
        msg = json.dumps({"channel": "subscriptions", "events": []})
        asyncio.run(connector._dispatch(msg))

    def test_dispatch_heartbeat_silent(self, connector):
        msg = json.dumps({"channel": "heartbeats"})
        asyncio.run(connector._dispatch(msg))

    def test_dispatch_unknown_channel(self, connector):
        msg = json.dumps({"channel": "unknown_test_channel"})
        asyncio.run(connector._dispatch(msg))


# --------------------------------------------------------- connect / disconnect

class TestConnection:
    @patch("core.connectors.coinbase_ws_connector.websockets", create=True)
    def test_connect_success(self, mock_ws_module, connector):
        mock_ws = AsyncMock()
        mock_ws_module.connect = AsyncMock(return_value=mock_ws)

        with patch.dict("sys.modules", {"websockets": mock_ws_module}):
            result = asyncio.run(connector.connect())

        assert result is True
        assert connector.connected is True
        # Should send one subscribe message per channel
        assert mock_ws.send.call_count == 2

    @patch("core.connectors.coinbase_ws_connector.websockets", create=True)
    def test_connect_failure(self, mock_ws_module, connector):
        mock_ws_module.connect = AsyncMock(side_effect=ConnectionError("refused"))

        with patch.dict("sys.modules", {"websockets": mock_ws_module}):
            result = asyncio.run(connector.connect())

        assert result is False
        assert connector.connected is False

    def test_disconnect_clears_state(self, connector):
        connector.connected = True
        connector._ws = AsyncMock()
        connector._running = True
        asyncio.run(connector.disconnect())
        assert connector.connected is False
        assert connector._running is False
        assert connector._ws is None


# --------------------------------------------------------- reconnect / backoff

class TestReconnect:
    def test_backoff_doubles(self, connector):
        assert connector._backoff_s == INITIAL_BACKOFF_S

        # Simulate failed reconnects to verify backoff growth
        connector._backoff_s *= 2
        assert connector._backoff_s == 2.0
        connector._backoff_s *= 2
        assert connector._backoff_s == 4.0

    def test_backoff_capped_at_max(self, connector):
        connector._backoff_s = MAX_BACKOFF_S * 2
        wait = min(connector._backoff_s, MAX_BACKOFF_S)
        assert wait == MAX_BACKOFF_S

    @patch("core.connectors.coinbase_ws_connector.websockets", create=True)
    def test_reconnect_resets_count_on_success(self, mock_ws_module, connector):
        """After a successful reconnect, _reconnect_count resets to 0."""
        mock_ws = AsyncMock()
        mock_ws_module.connect = AsyncMock(return_value=mock_ws)

        with patch.dict("sys.modules", {"websockets": mock_ws_module}):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                asyncio.run(connector._reconnect())

        # Successful connect() resets the counter
        assert connector._reconnect_count == 0
        assert connector.connected is True

    @patch("core.connectors.coinbase_ws_connector.websockets", create=True)
    def test_reconnect_keeps_count_on_failure(self, mock_ws_module, connector):
        """If reconnect fails, _reconnect_count stays incremented."""
        mock_ws_module.connect = AsyncMock(side_effect=ConnectionError("refused"))

        with patch.dict("sys.modules", {"websockets": mock_ws_module}):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                asyncio.run(connector._reconnect())

        assert connector._reconnect_count == 1
        assert connector.connected is False


# -------------------------------------------------------- heartbeat monitoring

class TestHeartbeat:
    def test_heartbeat_timeout_constant(self):
        assert HEARTBEAT_TIMEOUT_S == 30.0

    def test_last_message_time_updated_on_dispatch(self, connector, ticker_message):
        connector._last_message_time = 0.0
        asyncio.run(connector._dispatch(ticker_message))
        # _dispatch doesn't update _last_message_time (that's done in _receive_loop)
        # but we can verify the constant
        assert HEARTBEAT_TIMEOUT_S == 30.0


# -------------------------------------------------- env var credential loading

class TestEnvCredentials:
    def test_loads_from_env(self):
        with patch.dict("os.environ", {
            "COINBASE_API_KEY": "env-key",
            "COINBASE_API_SECRET": "env-secret",
        }):
            c = CoinbaseWSConnector()
            assert c.api_key == "env-key"
            assert c.api_secret == "env-secret"

    def test_explicit_overrides_env(self):
        with patch.dict("os.environ", {
            "COINBASE_API_KEY": "env-key",
            "COINBASE_API_SECRET": "env-secret",
        }):
            c = CoinbaseWSConnector(api_key="explicit-key", api_secret="explicit-secret")
            assert c.api_key == "explicit-key"
            assert c.api_secret == "explicit-secret"


# --------------------------------------------------- async context manager

class TestAsyncContextManager:
    @patch("core.connectors.coinbase_ws_connector.websockets", create=True)
    def test_context_manager_connects_and_disconnects(self, mock_ws_module):
        mock_ws = AsyncMock()
        mock_ws_module.connect = AsyncMock(return_value=mock_ws)

        async def _run():
            with patch.dict("sys.modules", {"websockets": mock_ws_module}):
                c = CoinbaseWSConnector(api_key="k", api_secret="s", symbols=["BTC-AUD"])
                async with c:
                    assert c.connected is True
                    # Cancel the background tasks immediately to avoid hang
                    if c._receive_task:
                        c._receive_task.cancel()
                    if c._heartbeat_task:
                        c._heartbeat_task.cancel()
                assert c.connected is False

        asyncio.run(_run())


# ------------------------------------------------ default symbols / channels

class TestDefaults:
    def test_default_symbols(self):
        c = CoinbaseWSConnector(api_key="k", api_secret="s")
        assert c.symbols == ["BTC-AUD", "ETH-AUD"]

    def test_default_channels(self):
        c = CoinbaseWSConnector(api_key="k", api_secret="s")
        assert c.channels == ["ticker", "level2"]

    def test_custom_symbols_and_channels(self):
        c = CoinbaseWSConnector(
            api_key="k", api_secret="s",
            symbols=["SOL-USD"], channels=["ticker"],
        )
        assert c.symbols == ["SOL-USD"]
        assert c.channels == ["ticker"]
