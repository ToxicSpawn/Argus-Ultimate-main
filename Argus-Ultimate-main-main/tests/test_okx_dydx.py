"""
Tests for OKX and dYdX v4 connectors, exchange manager wiring,
and cross-venue funding rate arbitrage.

Covers 45+ test cases:
- OKX: auth signing, symbol mapping, ticker, orderbook, funding rate,
  mark price, balances, order placement, cancel, order status, positions,
  leverage, rate limiting, health check, connection
- dYdX: symbol mapping, ticker, orderbook, funding rates, markets,
  balances, positions, fills, orders, funding opportunity, health check
- Funding rate aggregator: cross-venue comparison, arb detection, edge cases
- Exchange manager: OKX/dYdX wiring, perp exchange routing, failover
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import connectors
# ---------------------------------------------------------------------------
from core.connectors.okx_connector import (
    OKXConnector,
    OKXAPIError,
    OKXRateLimiter,
)
from core.connectors.dydx_v4_connector import (
    DYDXv4Connector,
    DYDXAPIError,
    DYDXRateLimiter,
)
from core.exchange_manager import ExchangeManager, ExchangeConfig
from strategies.funding_rate_harvester import FundingRateHarvester, SPOT_TO_PERP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _mock_okx_response(data: Any, code: str = "0", msg: str = "") -> Dict[str, Any]:
    """Build a mock OKX V5 JSON response."""
    return {"code": code, "msg": msg, "data": data}


def _mock_dydx_markets_response(ticker: str = "BTC-USD") -> Dict[str, Any]:
    return {
        "markets": {
            ticker: {
                "ticker": ticker,
                "oraclePrice": "65000.5",
                "volume24H": "1234567890",
                "openInterest": "50000",
                "nextFundingRate": "0.000125",
                "priceChange24H": "0.025",
            }
        }
    }


def _mock_dydx_orderbook() -> Dict[str, Any]:
    return {
        "bids": [
            {"price": "65000.0", "size": "1.5"},
            {"price": "64999.0", "size": "2.0"},
        ],
        "asks": [
            {"price": "65001.0", "size": "1.0"},
            {"price": "65002.0", "size": "3.0"},
        ],
    }


# ===========================================================================
# OKX Connector Tests
# ===========================================================================


class TestOKXAuth:
    """Test OKX HMAC-SHA256 signing and auth headers."""

    def test_sign_produces_base64(self):
        conn = OKXConnector(api_key="key", api_secret="secret", passphrase="pass")
        ts = "2026-03-18T12:00:00.000Z"
        sig = conn._sign(ts, "GET", "/api/v5/account/balance", "")
        # Verify it's valid base64
        decoded = base64.b64decode(sig)
        assert len(decoded) == 32  # SHA-256 produces 32 bytes

    def test_sign_deterministic(self):
        conn = OKXConnector(api_key="k", api_secret="s", passphrase="p")
        ts = "2026-01-01T00:00:00.000Z"
        sig1 = conn._sign(ts, "GET", "/api/v5/test", "")
        sig2 = conn._sign(ts, "GET", "/api/v5/test", "")
        assert sig1 == sig2

    def test_sign_varies_with_body(self):
        conn = OKXConnector(api_key="k", api_secret="s", passphrase="p")
        ts = "2026-01-01T00:00:00.000Z"
        sig1 = conn._sign(ts, "POST", "/api/v5/trade/order", '{"instId":"BTC"}')
        sig2 = conn._sign(ts, "POST", "/api/v5/trade/order", '{"instId":"ETH"}')
        assert sig1 != sig2

    def test_sign_matches_manual_hmac(self):
        secret = "mysecret"
        conn = OKXConnector(api_key="k", api_secret=secret, passphrase="p")
        ts = "2026-03-18T12:00:00.000Z"
        method = "GET"
        path = "/api/v5/account/balance"
        body = ""
        message = ts + method + path + body
        expected = base64.b64encode(
            hmac.new(secret.encode(), message.encode(), hashlib.sha256).digest()
        ).decode()
        assert conn._sign(ts, method, path, body) == expected

    def test_auth_headers_structure(self):
        conn = OKXConnector(api_key="testkey", api_secret="testsecret", passphrase="testpass")
        headers = conn._auth_headers("GET", "/api/v5/test")
        assert "OK-ACCESS-KEY" in headers
        assert headers["OK-ACCESS-KEY"] == "testkey"
        assert "OK-ACCESS-SIGN" in headers
        assert "OK-ACCESS-TIMESTAMP" in headers
        assert "OK-ACCESS-PASSPHRASE" in headers
        assert headers["OK-ACCESS-PASSPHRASE"] == "testpass"
        assert headers["Content-Type"] == "application/json"

    def test_auth_headers_testnet_flag(self):
        conn = OKXConnector(api_key="k", api_secret="s", passphrase="p", testnet=True)
        headers = conn._auth_headers("GET", "/api/v5/test")
        assert headers.get("x-simulated-trading") == "1"

    def test_iso_timestamp_format(self):
        ts = OKXConnector._iso_timestamp()
        # Should end with 'Z' and contain 'T'
        assert ts.endswith("Z")
        assert "T" in ts
        # Should have milliseconds (3 digits before Z)
        assert len(ts.split(".")[-1]) == 4  # "xxxZ"


class TestOKXSymbolMapping:
    """Test OKX symbol conversion."""

    def test_standard_to_swap(self):
        assert OKXConnector.to_okx_symbol("BTC/USDT") == "BTC-USDT-SWAP"

    def test_ccxt_perp_format(self):
        assert OKXConnector.to_okx_symbol("BTC/USDT:USDT") == "BTC-USDT-SWAP"

    def test_already_okx_format(self):
        assert OKXConnector.to_okx_symbol("BTC-USDT-SWAP") == "BTC-USDT-SWAP"

    def test_spot_symbol(self):
        assert OKXConnector.to_okx_symbol("BTC-USDT", inst_type="SPOT") == "BTC-USDT"

    def test_from_okx_swap(self):
        assert OKXConnector.from_okx_symbol("BTC-USDT-SWAP") == "BTC/USDT"

    def test_from_okx_spot(self):
        assert OKXConnector.from_okx_symbol("ETH-USDT") == "ETH/USDT"

    def test_eth_mapping(self):
        assert OKXConnector.to_okx_symbol("ETH/USDT") == "ETH-USDT-SWAP"

    def test_sol_mapping(self):
        assert OKXConnector.to_okx_symbol("SOL/USDT") == "SOL-USDT-SWAP"


class TestOKXMarketData:
    """Test OKX market data endpoints with mocked HTTP."""

    def test_get_ticker(self):
        conn = OKXConnector(api_key="k", api_secret="s", passphrase="p")
        mock_resp = _mock_okx_response([{
            "instId": "BTC-USDT-SWAP",
            "last": "65000.5",
            "bidPx": "65000.0",
            "askPx": "65001.0",
            "bidSz": "10",
            "askSz": "5",
            "vol24h": "123456",
            "volCcy24h": "8000000000",
            "ts": "1710720000000",
        }])

        async def mock_request(*args, **kwargs):
            return mock_resp["data"]

        conn._public_get = AsyncMock(return_value=mock_resp["data"])
        result = run(conn.get_ticker("BTC/USDT"))

        assert result is not None
        assert result["last"] == 65000.5
        assert result["bid"] == 65000.0
        assert result["ask"] == 65001.0
        assert result["volume_24h"] == 123456.0
        assert result["exchange"] == "okx"

    def test_get_orderbook(self):
        conn = OKXConnector()
        mock_data = [{
            "bids": [["65000.0", "1.5", "0", "3"], ["64999.0", "2.0", "0", "5"]],
            "asks": [["65001.0", "1.0", "0", "2"], ["65002.0", "3.0", "0", "4"]],
            "ts": "1710720000000",
        }]
        conn._public_get = AsyncMock(return_value=mock_data)
        result = run(conn.get_orderbook("BTC/USDT", depth=5))

        assert result is not None
        assert len(result["bids"]) == 2
        assert len(result["asks"]) == 2
        assert result["bids"][0] == (65000.0, 1.5)
        assert result["asks"][0] == (65001.0, 1.0)
        assert result["exchange"] == "okx"

    def test_get_funding_rate(self):
        conn = OKXConnector()
        mock_data = [{
            "instId": "BTC-USDT-SWAP",
            "fundingRate": "0.0003",
            "nextFundingRate": "0.00025",
            "nextFundingTime": "1710748800000",
        }]
        conn._public_get = AsyncMock(return_value=mock_data)
        result = run(conn.get_funding_rate("BTC/USDT"))

        assert result["funding_rate"] == 0.0003
        assert result["current_rate"] == 0.0003
        assert result["predicted_rate"] == 0.00025
        assert result["next_settlement"] == 1710748800000
        assert result["exchange"] == "okx"

    def test_get_mark_price(self):
        conn = OKXConnector()
        mock_data = [{"instId": "BTC-USDT-SWAP", "markPx": "65050.75"}]
        conn._public_get = AsyncMock(return_value=mock_data)
        result = run(conn.get_mark_price("BTC/USDT"))
        assert result == 65050.75

    def test_get_ticker_failure_returns_none(self):
        conn = OKXConnector()
        conn._public_get = AsyncMock(side_effect=Exception("Network error"))
        result = run(conn.get_ticker("BTC/USDT"))
        assert result is None

    def test_fetch_funding_rates_multiple(self):
        conn = OKXConnector()
        call_count = 0
        async def mock_get_fr(sym):
            nonlocal call_count
            call_count += 1
            rates = {"BTC/USDT": 0.0003, "ETH/USDT": 0.0001}
            return {"symbol": sym, "funding_rate": rates.get(sym, 0.0), "exchange": "okx"}
        conn.get_funding_rate = mock_get_fr
        result = run(conn.fetch_funding_rates(["BTC/USDT", "ETH/USDT"]))
        assert result["BTC/USDT"] == 0.0003
        assert result["ETH/USDT"] == 0.0001
        assert call_count == 2


class TestOKXPrivateEndpoints:
    """Test OKX private endpoints with mocked HTTP."""

    def test_get_balances(self):
        conn = OKXConnector(api_key="k", api_secret="s", passphrase="p")
        mock_data = [{
            "details": [
                {
                    "ccy": "USDT",
                    "availBal": "5000.0",
                    "cashBal": "6000.0",
                    "eq": "6500.0",
                    "upl": "500.0",
                },
                {
                    "ccy": "BTC",
                    "availBal": "0.1",
                    "cashBal": "0.15",
                    "eq": "0.15",
                    "upl": "0.01",
                },
            ]
        }]
        conn._private_get = AsyncMock(return_value=mock_data)
        result = run(conn.get_balances())

        assert "USDT" in result
        assert result["USDT"]["available"] == 5000.0
        assert result["USDT"]["total"] == 6000.0
        assert result["USDT"]["equity"] == 6500.0
        assert "BTC" in result

    def test_place_order_market(self):
        conn = OKXConnector(api_key="k", api_secret="s", passphrase="p")
        mock_data = [{"ordId": "12345", "clOrdId": "", "sCode": "0", "sMsg": ""}]
        conn._private_post = AsyncMock(return_value=mock_data)
        result = run(conn.place_order("BTC/USDT", "buy", "market", 0.01))

        assert result["order_id"] == "12345"
        assert result["side"] == "buy"
        assert result["exchange"] == "okx"

    def test_place_order_limit(self):
        conn = OKXConnector(api_key="k", api_secret="s", passphrase="p")
        mock_data = [{"ordId": "67890", "clOrdId": "", "sCode": "0", "sMsg": ""}]
        conn._private_post = AsyncMock(return_value=mock_data)
        result = run(conn.place_order("ETH/USDT", "sell", "limit", 1.0, price=3000.0))

        assert result["order_id"] == "67890"
        assert result["side"] == "sell"
        assert result["price"] == 3000.0

    def test_place_order_api_error(self):
        conn = OKXConnector(api_key="k", api_secret="s", passphrase="p")
        mock_data = [{"ordId": "", "clOrdId": "", "sCode": "51000", "sMsg": "Insufficient balance"}]
        conn._private_post = AsyncMock(return_value=mock_data)
        with pytest.raises(OKXAPIError) as exc_info:
            run(conn.place_order("BTC/USDT", "buy", "market", 100.0))
        assert "51000" in str(exc_info.value)

    def test_cancel_order(self):
        conn = OKXConnector(api_key="k", api_secret="s", passphrase="p")
        mock_data = [{"ordId": "12345", "sCode": "0", "sMsg": ""}]
        conn._private_post = AsyncMock(return_value=mock_data)
        result = run(conn.cancel_order("BTC/USDT", "12345"))
        assert result is True

    def test_get_order(self):
        conn = OKXConnector(api_key="k", api_secret="s", passphrase="p")
        mock_data = [{
            "ordId": "12345",
            "clOrdId": "cl1",
            "side": "buy",
            "ordType": "market",
            "sz": "0.01",
            "accFillSz": "0.01",
            "px": "0",
            "avgPx": "65000",
            "state": "filled",
            "fee": "-0.5",
            "pnl": "0",
            "cTime": "1710720000000",
        }]
        conn._private_get = AsyncMock(return_value=mock_data)
        result = run(conn.get_order("BTC/USDT", "12345"))

        assert result["order_id"] == "12345"
        assert result["status"] == "filled"
        assert result["avg_price"] == 65000.0
        assert result["filled_size"] == 0.01

    def test_get_positions(self):
        conn = OKXConnector(api_key="k", api_secret="s", passphrase="p")
        mock_data = [
            {
                "instId": "BTC-USDT-SWAP",
                "pos": "0.1",
                "avgPx": "64000",
                "markPx": "65000",
                "upl": "100",
                "lever": "10",
                "liqPx": "58000",
                "margin": "640",
            },
            {
                "instId": "ETH-USDT-SWAP",
                "pos": "-2.0",
                "avgPx": "3200",
                "markPx": "3150",
                "upl": "100",
                "lever": "5",
                "liqPx": "3500",
                "margin": "1280",
            },
        ]
        conn._private_get = AsyncMock(return_value=mock_data)
        result = run(conn.get_positions())

        assert len(result) == 2
        assert result[0]["side"] == "long"
        assert result[0]["size"] == 0.1
        assert result[1]["side"] == "short"
        assert result[1]["size"] == 2.0

    def test_get_position_specific(self):
        conn = OKXConnector(api_key="k", api_secret="s", passphrase="p")
        mock_data = [{
            "instId": "BTC-USDT-SWAP",
            "pos": "0.5",
            "avgPx": "64000",
            "markPx": "65000",
            "upl": "500",
            "lever": "10",
            "liqPx": "58000",
        }]
        conn._private_get = AsyncMock(return_value=mock_data)
        result = run(conn.get_position("BTC/USDT"))

        assert result is not None
        assert result["size"] == 0.5
        assert result["side"] == "long"
        assert result["entry_price"] == 64000.0

    def test_set_leverage(self):
        conn = OKXConnector(api_key="k", api_secret="s", passphrase="p")
        conn._private_post = AsyncMock(return_value=[{}])
        result = run(conn.set_leverage("BTC/USDT", 10))
        assert result is True


class TestOKXRateLimiter:
    """Test rate limiter behavior."""

    def test_rate_limiter_init(self):
        rl = OKXRateLimiter(20)
        assert rl._max == 20
        assert rl._tokens == 20.0

    def test_rate_limiter_acquire(self):
        rl = OKXRateLimiter(100)
        # Should not block with plenty of tokens
        run(rl.acquire())
        assert rl._tokens < 100.0


class TestOKXConnection:
    """Test connection management."""

    def test_connect_success(self):
        conn = OKXConnector()
        conn._public_get = AsyncMock(return_value=[{"instId": "BTC-USDT-SWAP", "last": "65000"}])
        result = run(conn.connect())
        assert result is True
        assert conn.connected is True

    def test_connect_failure(self):
        conn = OKXConnector()
        conn._get_session = AsyncMock(side_effect=Exception("Connection refused"))
        result = run(conn.connect())
        assert result is False
        assert conn.connected is False

    def test_disconnect(self):
        conn = OKXConnector()
        conn.connected = True
        conn._session = MagicMock()
        conn._session.closed = False
        conn._session.close = AsyncMock()
        run(conn.disconnect())
        assert conn.connected is False

    def test_health_check(self):
        conn = OKXConnector()
        conn._public_get = AsyncMock(return_value=[{"instId": "BTC-USDT-SWAP"}])
        result = run(conn.health_check())
        assert result["healthy"] is True
        assert result["exchange"] == "okx"
        assert "latency_ms" in result

    def test_env_var_defaults(self):
        """Verify constructor reads from env vars."""
        with patch.dict("os.environ", {
            "OKX_API_KEY": "envkey",
            "OKX_API_SECRET": "envsecret",
            "OKX_PASSPHRASE": "envpass",
        }):
            conn = OKXConnector()
            assert conn.api_key == "envkey"
            assert conn.api_secret == "envsecret"
            assert conn.passphrase == "envpass"


# ===========================================================================
# dYdX v4 Connector Tests
# ===========================================================================


class TestDYDXSymbolMapping:
    """Test dYdX symbol conversion."""

    def test_standard_to_dydx(self):
        assert DYDXv4Connector.to_dydx_symbol("BTC/USD") == "BTC-USD"

    def test_usdt_to_usd(self):
        assert DYDXv4Connector.to_dydx_symbol("ETH/USDT") == "ETH-USD"

    def test_already_dydx_format(self):
        assert DYDXv4Connector.to_dydx_symbol("SOL-USD") == "SOL-USD"

    def test_from_dydx(self):
        assert DYDXv4Connector.from_dydx_symbol("BTC-USD") == "BTC/USD"

    def test_ccxt_perp_strip(self):
        assert DYDXv4Connector.to_dydx_symbol("BTC/USDT:USDT") == "BTC-USD"

    def test_usdt_dash_to_usd(self):
        assert DYDXv4Connector.to_dydx_symbol("BTC-USDT") == "BTC-USD"


class TestDYDXMarketData:
    """Test dYdX public endpoints with mocked HTTP."""

    def test_get_markets(self):
        conn = DYDXv4Connector()
        mock_data = {
            "markets": {
                "BTC-USD": {"ticker": "BTC-USD", "oraclePrice": "65000"},
                "ETH-USD": {"ticker": "ETH-USD", "oraclePrice": "3200"},
            }
        }
        conn._public_get = AsyncMock(return_value=mock_data)
        result = run(conn.get_markets())
        assert "BTC-USD" in result
        assert "ETH-USD" in result

    def test_get_ticker(self):
        conn = DYDXv4Connector()
        mock_markets = _mock_dydx_markets_response("BTC-USD")
        mock_ob = {
            "symbol": "BTC/USD",
            "bids": [(65000.0, 1.5)],
            "asks": [(65001.0, 1.0)],
            "exchange": "dydx",
        }
        conn._public_get = AsyncMock(return_value=mock_markets)
        conn.get_orderbook = AsyncMock(return_value=mock_ob)
        result = run(conn.get_ticker("BTC/USD"))

        assert result is not None
        assert result["last"] == 65000.5
        assert result["bid"] == 65000.0
        assert result["ask"] == 65001.0
        assert result["funding_rate"] == 0.000125
        assert result["exchange"] == "dydx"

    def test_get_orderbook(self):
        conn = DYDXv4Connector()
        mock_data = _mock_dydx_orderbook()
        conn._public_get = AsyncMock(return_value=mock_data)
        result = run(conn.get_orderbook("BTC/USD", depth=5))

        assert result is not None
        assert len(result["bids"]) == 2
        assert len(result["asks"]) == 2
        assert result["bids"][0] == (65000.0, 1.5)
        assert result["asks"][0] == (65001.0, 1.0)
        assert result["exchange"] == "dydx"

    def test_get_funding_rates_historical(self):
        conn = DYDXv4Connector()
        mock_data = {
            "historicalFunding": [
                {"rate": "0.000125", "effectiveAt": "2026-03-18T00:00:00Z"},
                {"rate": "0.000100", "effectiveAt": "2026-03-17T16:00:00Z"},
                {"rate": "0.000080", "effectiveAt": "2026-03-17T08:00:00Z"},
            ]
        }
        conn._public_get = AsyncMock(return_value=mock_data)
        result = run(conn.get_funding_rates("BTC/USD", limit=10))

        assert len(result) == 3
        assert result[0]["rate"] == 0.000125
        assert result[0]["exchange"] == "dydx"

    def test_get_funding_rate_current(self):
        conn = DYDXv4Connector()
        mock_data = _mock_dydx_markets_response("BTC-USD")
        conn._public_get = AsyncMock(return_value=mock_data)
        result = run(conn.get_funding_rate("BTC/USD"))

        assert result["funding_rate"] == 0.000125
        assert result["exchange"] == "dydx"

    def test_fetch_funding_rates_multi(self):
        conn = DYDXv4Connector()
        async def mock_fr(sym):
            rates = {"BTC/USD": 0.00012, "ETH/USD": 0.00008}
            return {"symbol": sym, "funding_rate": rates.get(sym, 0.0), "exchange": "dydx"}
        conn.get_funding_rate = mock_fr
        result = run(conn.fetch_funding_rates(["BTC/USD", "ETH/USD"]))
        assert result["BTC/USD"] == 0.00012
        assert result["ETH/USD"] == 0.00008

    def test_get_ticker_failure(self):
        conn = DYDXv4Connector()
        conn._public_get = AsyncMock(side_effect=Exception("Timeout"))
        result = run(conn.get_ticker("BTC/USD"))
        assert result is None


class TestDYDXPrivateEndpoints:
    """Test dYdX private endpoints with mocked HTTP."""

    def test_get_balances(self):
        conn = DYDXv4Connector(api_key="k", api_secret="s", passphrase="p")
        mock_data = {
            "subaccount": {
                "equity": "10000.0",
                "freeCollateral": "8000.0",
                "marginEnabled": "2000.0",
                "openPerpetualPositionsCount": 2,
            }
        }
        conn._private_get = AsyncMock(return_value=mock_data)
        result = run(conn.get_balances())

        assert result["equity"] == 10000.0
        assert result["free_collateral"] == 8000.0
        assert result["exchange"] == "dydx"

    def test_get_positions(self):
        conn = DYDXv4Connector(api_key="k", api_secret="s", passphrase="p")
        mock_data = {
            "subaccount": {
                "openPerpetualPositions": {
                    "BTC-USD": {
                        "size": "0.5",
                        "entryPrice": "64000",
                        "unrealizedPnl": "500",
                        "realizedPnl": "100",
                        "sumOpen": "0.5",
                        "sumClose": "0",
                    },
                    "ETH-USD": {
                        "size": "-10.0",
                        "entryPrice": "3200",
                        "unrealizedPnl": "-200",
                        "realizedPnl": "50",
                        "sumOpen": "10.0",
                        "sumClose": "0",
                    },
                }
            }
        }
        conn._private_get = AsyncMock(return_value=mock_data)
        result = run(conn.get_positions())

        assert len(result) == 2
        btc_pos = [p for p in result if "BTC" in p["symbol"]][0]
        assert btc_pos["side"] == "long"
        assert btc_pos["size"] == 0.5
        eth_pos = [p for p in result if "ETH" in p["symbol"]][0]
        assert eth_pos["side"] == "short"
        assert eth_pos["size"] == 10.0

    def test_get_fills(self):
        conn = DYDXv4Connector(api_key="k", api_secret="s", passphrase="p")
        mock_data = {
            "fills": [
                {
                    "id": "fill1",
                    "market": "BTC-USD",
                    "side": "BUY",
                    "size": "0.1",
                    "price": "65000",
                    "fee": "0.5",
                    "type": "LIMIT",
                    "createdAt": "2026-03-18T10:00:00Z",
                },
            ]
        }
        conn._private_get = AsyncMock(return_value=mock_data)
        result = run(conn.get_fills("BTC/USD"))

        assert len(result) == 1
        assert result[0]["fill_id"] == "fill1"
        assert result[0]["side"] == "buy"
        assert result[0]["size"] == 0.1
        assert result[0]["price"] == 65000.0

    def test_get_orders(self):
        conn = DYDXv4Connector(api_key="k", api_secret="s", passphrase="p")
        mock_data = {
            "orders": [
                {
                    "id": "ord1",
                    "clientId": "cl1",
                    "ticker": "BTC-USD",
                    "side": "BUY",
                    "size": "0.5",
                    "price": "64000",
                    "status": "OPEN",
                    "type": "LIMIT",
                    "createdAtHeight": "12345",
                },
            ]
        }
        conn._private_get = AsyncMock(return_value=mock_data)
        result = run(conn.get_orders("BTC/USD"))

        assert len(result) == 1
        assert result[0]["order_id"] == "ord1"
        assert result[0]["status"] == "OPEN"

    def test_no_auth_returns_empty(self):
        conn = DYDXv4Connector()  # No API keys
        result = run(conn.get_positions())
        assert result == []
        result2 = run(conn.get_fills("BTC/USD"))
        assert result2 == []
        result3 = run(conn.get_orders("BTC/USD"))
        assert result3 == []


class TestDYDXConnection:
    """Test dYdX connection management."""

    def test_connect_success(self):
        conn = DYDXv4Connector()
        conn._public_get = AsyncMock(return_value=_mock_dydx_markets_response())
        result = run(conn.connect())
        assert result is True
        assert conn.connected is True

    def test_connect_failure(self):
        conn = DYDXv4Connector()
        conn._get_session = AsyncMock(side_effect=Exception("DNS failure"))
        result = run(conn.connect())
        assert result is False
        assert conn.connected is False

    def test_disconnect(self):
        conn = DYDXv4Connector()
        conn.connected = True
        conn._session = MagicMock()
        conn._session.closed = False
        conn._session.close = AsyncMock()
        run(conn.disconnect())
        assert conn.connected is False

    def test_health_check(self):
        conn = DYDXv4Connector()
        conn._public_get = AsyncMock(return_value=_mock_dydx_markets_response())
        result = run(conn.health_check())
        assert result["healthy"] is True
        assert result["exchange"] == "dydx"

    def test_has_auth_true(self):
        conn = DYDXv4Connector(api_key="k", api_secret="s")
        assert conn.has_auth is True

    def test_has_auth_false(self):
        conn = DYDXv4Connector()
        assert conn.has_auth is False


class TestDYDXFundingOpportunity:
    """Test dYdX cross-venue funding rate arb detection."""

    def test_funding_opportunity_found(self):
        conn = DYDXv4Connector()
        # Mock dYdX rates
        async def mock_fr(sym):
            return {"symbol": sym, "funding_rate": -0.0002, "exchange": "dydx"}
        conn.get_funding_rate = mock_fr

        bybit_rates = {"BTC/USDT": 0.0005}
        result = run(conn.get_funding_opportunity(bybit_rates=bybit_rates))

        assert result is not None
        assert result["long_venue"] == "dydx"
        assert result["short_venue"] == "bybit"
        assert result["spread_bps"] > 0

    def test_funding_opportunity_three_venues(self):
        conn = DYDXv4Connector()
        async def mock_fr(sym):
            return {"symbol": sym, "funding_rate": 0.0001, "exchange": "dydx"}
        conn.get_funding_rate = mock_fr

        bybit_rates = {"BTC/USDT": 0.0008}
        okx_rates = {"BTC-USDT-SWAP": -0.0003}

        result = run(conn.get_funding_opportunity(
            bybit_rates=bybit_rates, okx_rates=okx_rates
        ))

        assert result is not None
        assert result["short_venue"] == "bybit"  # Highest rate
        assert result["long_venue"] == "okx"     # Lowest rate

    def test_funding_opportunity_none_when_no_spread(self):
        conn = DYDXv4Connector()
        async def mock_fr(sym):
            return {"symbol": sym, "funding_rate": 0.0001, "exchange": "dydx"}
        conn.get_funding_rate = mock_fr

        # Only dYdX rates, no other venues
        result = run(conn.get_funding_opportunity())
        assert result is None


# ===========================================================================
# Funding Rate Aggregator Tests
# ===========================================================================


class TestFundingRateAggregator:
    """Test cross-venue funding rate arbitrage in FundingRateHarvester."""

    def test_find_best_opportunity_basic(self):
        rates = {
            "BTC/USD": {"bybit": 0.0008, "okx": 0.0003, "dydx": -0.0002},
        }
        result = FundingRateHarvester.find_best_funding_opportunity(rates)

        assert result is not None
        assert result["symbol"] == "BTC/USD"
        assert result["long_venue"] == "dydx"    # Lowest rate
        assert result["short_venue"] == "bybit"   # Highest rate
        assert result["spread_bps"] == 10.0        # (0.0008 - (-0.0002)) * 10000
        assert result["annualized_apr"] > 0

    def test_find_best_opportunity_multi_symbol(self):
        rates = {
            "BTC/USD": {"bybit": 0.0005, "okx": 0.0003},
            "ETH/USD": {"bybit": 0.0010, "okx": -0.0001},
        }
        result = FundingRateHarvester.find_best_funding_opportunity(rates)

        assert result is not None
        assert result["symbol"] == "ETH/USD"  # Bigger spread
        assert result["spread_bps"] == 11.0

    def test_find_best_opportunity_below_threshold(self):
        rates = {
            "BTC/USD": {"bybit": 0.0001, "okx": 0.00009},
        }
        result = FundingRateHarvester.find_best_funding_opportunity(rates, min_spread_bps=5.0)
        # Spread = 0.1 bps, below 5 bps threshold
        assert result is None

    def test_find_best_opportunity_single_venue(self):
        rates = {"BTC/USD": {"bybit": 0.0005}}
        result = FundingRateHarvester.find_best_funding_opportunity(rates)
        assert result is None  # Need at least 2 venues

    def test_find_best_opportunity_recommended_size(self):
        rates = {
            "BTC/USD": {"bybit": 0.0050, "dydx": -0.0010},  # 60 bps spread
        }
        result = FundingRateHarvester.find_best_funding_opportunity(
            rates, capital=10000.0, max_position_pct=0.25
        )
        assert result is not None
        assert result["recommended_size"] > 0
        assert result["recommended_size"] <= 2500.0  # 25% of 10000

    def test_spot_to_perp_includes_dydx(self):
        """Verify dYdX was added to SPOT_TO_PERP mapping."""
        assert "dydx" in SPOT_TO_PERP["BTC/USD"]
        assert SPOT_TO_PERP["BTC/USD"]["dydx"] == "BTC-USD"
        assert "dydx" in SPOT_TO_PERP["ETH/USD"]
        assert "dydx" in SPOT_TO_PERP["SOL/USD"]

    def test_find_best_opportunity_empty_rates(self):
        result = FundingRateHarvester.find_best_funding_opportunity({})
        assert result is None

    def test_find_best_opportunity_annualized_apr(self):
        rates = {
            "BTC/USD": {"bybit": 0.001, "dydx": 0.0},  # 10 bps per 8h
        }
        result = FundingRateHarvester.find_best_funding_opportunity(rates)
        assert result is not None
        # APR = 0.001 * 3 * 365 * 100 = 109.5%
        assert abs(result["annualized_apr"] - 109.5) < 0.1


# ===========================================================================
# Exchange Manager Wiring Tests
# ===========================================================================


class TestExchangeManagerWiring:
    """Test that OKX and dYdX are wired correctly into ExchangeManager."""

    def test_create_okx_connector(self):
        em = ExchangeManager()
        connector = em._create_connector("okx", "key", "secret")
        assert isinstance(connector, OKXConnector)
        assert connector.api_key == "key"
        assert connector.api_secret == "secret"

    def test_create_okex_alias(self):
        em = ExchangeManager()
        connector = em._create_connector("okex", "key", "secret")
        assert isinstance(connector, OKXConnector)

    def test_create_dydx_connector(self):
        em = ExchangeManager()
        connector = em._create_connector("dydx", "key", "secret")
        assert isinstance(connector, DYDXv4Connector)
        assert connector.api_key == "key"

    def test_create_dydx_v4_alias(self):
        em = ExchangeManager()
        connector = em._create_connector("dydx_v4", "key", "secret")
        assert isinstance(connector, DYDXv4Connector)

    def test_add_okx_exchange(self):
        em = ExchangeManager()
        config = ExchangeConfig(name="okx", api_key="k", api_secret="s", priority=2)
        em.add_exchange(config)
        assert "okx" in em.exchanges
        assert isinstance(em.exchanges["okx"], OKXConnector)

    def test_add_dydx_exchange(self):
        em = ExchangeManager()
        config = ExchangeConfig(name="dydx", api_key="k", api_secret="s", priority=1)
        em.add_exchange(config)
        assert "dydx" in em.exchanges
        assert isinstance(em.exchanges["dydx"], DYDXv4Connector)

    def test_perp_exchange_routing_okx(self):
        em = ExchangeManager()
        config = ExchangeConfig(name="okx", api_key="k", api_secret="s", priority=2)
        em.add_exchange(config)
        perp_ex = em._get_perp_exchange()
        assert perp_ex == "okx"

    def test_perp_exchange_routing_dydx(self):
        em = ExchangeManager()
        config = ExchangeConfig(name="dydx", api_key="k", api_secret="s", priority=1)
        em.add_exchange(config)
        perp_ex = em._get_perp_exchange()
        assert perp_ex == "dydx"

    def test_failover_chain(self):
        """Test that failover works across multiple perp venues."""
        em = ExchangeManager()
        em.add_exchange(ExchangeConfig(name="bybit", api_key="k", api_secret="s", priority=3))
        em.add_exchange(ExchangeConfig(name="okx", api_key="k", api_secret="s", priority=2))
        em.add_exchange(ExchangeConfig(name="dydx", api_key="k", api_secret="s", priority=1))

        # All three should be active
        assert len(em.active_exchanges) == 3
        # Primary should be highest priority
        assert em.primary_exchange == "bybit"

        # Simulate bybit failure
        em.exchanges["bybit"].connected = True
        em.exchanges["okx"].connected = True
        em.exchanges["dydx"].connected = True

        # Mark bybit as failed and failover
        em.exchanges["bybit"].connected = False
        result = run(em.failover_to_backup("bybit"))
        assert result is True
        assert em.primary_exchange == "okx"

    def test_ticker_with_okx(self):
        em = ExchangeManager()
        em.add_exchange(ExchangeConfig(name="okx", api_key="k", api_secret="s", priority=1))
        mock_ticker = {
            "symbol": "BTC/USDT",
            "last": 65000.0,
            "bid": 64999.0,
            "ask": 65001.0,
            "exchange": "okx",
        }
        em.exchanges["okx"].get_ticker = AsyncMock(return_value=mock_ticker)
        result = run(em.get_ticker("BTC/USDT", "okx"))
        assert result is not None
        assert result["exchange"] == "okx"
        assert result["last"] == 65000.0


# ===========================================================================
# OKX API Error Tests
# ===========================================================================


class TestOKXAPIError:
    """Test OKXAPIError exception class."""

    def test_error_attributes(self):
        err = OKXAPIError("51000", "Insufficient balance", "/api/v5/trade/order")
        assert err.code == "51000"
        assert err.msg == "Insufficient balance"
        assert err.endpoint == "/api/v5/trade/order"

    def test_error_str(self):
        err = OKXAPIError("51000", "Insufficient balance")
        assert "51000" in str(err)
        assert "Insufficient balance" in str(err)


class TestDYDXAPIError:
    """Test DYDXAPIError exception class."""

    def test_error_attributes(self):
        err = DYDXAPIError("Not found", 404, "/perpetualMarkets")
        assert err.status_code == 404
        assert err.endpoint == "/perpetualMarkets"

    def test_error_str(self):
        err = DYDXAPIError("Rate limited", 429)
        assert "429" in str(err)
        assert "Rate limited" in str(err)
