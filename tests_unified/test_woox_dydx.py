"""
Tests for WOO X and dYdX Exchange Clients
==========================================

Covers:
  - WOO X fee constants (all four zero)
  - WOO X symbol normalisation (spot, perp, reverse)
  - WOO X HMAC-SHA256 signing (deterministic)
  - WOO X post_only flag on create_order → POST_ONLY order type
  - WOO X fetch_ticker (mocked aiohttp response)
  - dYdX fee constants (maker=0.0001, taker=0.0005)
  - dYdX is_custodial() → False
  - dYdX DYDX_REQUIRES_KYC → False
  - dYdX fetch_ticker (mocked indexer response)
  - dYdX create_order without key → NotImplementedError
  - exchange_registry has woox and dydx entries
  - exchange_registry woox maker fee == 0
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Imports under test — WOO X
# ---------------------------------------------------------------------------
from exchanges.woox_client import (
    WOOXAPIError,
    WOOXClient,
    WOOX_SPOT_MAKER_FEE,
    WOOX_SPOT_TAKER_FEE,
    WOOX_FUTURES_MAKER_FEE,
    WOOX_FUTURES_TAKER_FEE,
    WOOX_BASE_URL,
    WOOX_WS_PUBLIC,
    WOOX_WS_PRIVATE,
    to_woox_symbol,
    to_woox_perp,
    from_woox_symbol,
    get_exchange_info as woox_get_exchange_info,
)

# ---------------------------------------------------------------------------
# Imports under test — dYdX
# ---------------------------------------------------------------------------
from exchanges.dydx_client import (
    DYDXClient,
    DYDX_MAKER_FEE,
    DYDX_TAKER_FEE,
    DYDX_INDEXER_URL,
    DYDX_WS_URL,
    DYDX_IS_DECENTRALISED,
    DYDX_REQUIRES_KYC,
    to_dydx_symbol,
    from_dydx_symbol,
    get_exchange_info as dydx_get_exchange_info,
)

# ---------------------------------------------------------------------------
# Imports under test — registry
# ---------------------------------------------------------------------------
from exchanges.exchange_registry import (
    EXCHANGE_REGISTRY,
    min_spread_to_profit,
)


# ===========================================================================
# 1. WOO X — Constants
# ===========================================================================

class TestWOOXConstants(unittest.TestCase):
    """All four WOO X fee constants must be exactly zero."""

    def test_spot_maker_fee_zero(self):
        assert WOOX_SPOT_MAKER_FEE == 0.0, "WOO X spot maker fee must be 0.0%"

    def test_spot_taker_fee_zero(self):
        assert WOOX_SPOT_TAKER_FEE == 0.0, "WOO X spot taker fee must be 0.0%"

    def test_futures_maker_fee_zero(self):
        assert WOOX_FUTURES_MAKER_FEE == 0.0, "WOO X futures maker fee must be 0.0%"

    def test_futures_taker_fee_zero(self):
        assert WOOX_FUTURES_TAKER_FEE == 0.0, "WOO X futures taker fee must be 0.0%"

    def test_all_four_fees_zero(self):
        """Convenience: confirm all four in one assertion for CI clarity."""
        fees = [
            WOOX_SPOT_MAKER_FEE,
            WOOX_SPOT_TAKER_FEE,
            WOOX_FUTURES_MAKER_FEE,
            WOOX_FUTURES_TAKER_FEE,
        ]
        assert all(f == 0.0 for f in fees), f"Not all fees zero: {fees}"

    def test_base_url(self):
        assert WOOX_BASE_URL == "https://api.woo.org"

    def test_ws_public_url(self):
        assert "wss://" in WOOX_WS_PUBLIC

    def test_ws_private_url(self):
        assert "wss://" in WOOX_WS_PRIVATE

    def test_get_exchange_info_structure(self):
        info = woox_get_exchange_info()
        assert info["exchange"] == "woox"
        assert info["fee_rates"]["spot_maker"] == 0.0
        assert info["fee_rates"]["futures_maker"] == 0.0


# ===========================================================================
# 2. WOO X — Symbol normalisation: spot
# ===========================================================================

class TestWOOXSymbolToSpot(unittest.TestCase):
    """to_woox_symbol converts slash-format to SPOT_ prefix."""

    def test_btc_usdt(self):
        assert to_woox_symbol("BTC/USDT") == "SPOT_BTC_USDT"

    def test_eth_usdt(self):
        assert to_woox_symbol("ETH/USDT") == "SPOT_ETH_USDT"

    def test_already_spot_prefix(self):
        """Already-converted symbol should pass through unchanged."""
        assert to_woox_symbol("SPOT_BTC_USDT") == "SPOT_BTC_USDT"

    def test_lowercase_input(self):
        assert to_woox_symbol("btc/usdt") == "SPOT_BTC_USDT"

    def test_sol_usdt(self):
        assert to_woox_symbol("SOL/USDT") == "SPOT_SOL_USDT"

    def test_instance_method_matches_module_function(self):
        client = WOOXClient(api_key="test_key", api_secret="test_secret")
        assert client.to_woox_symbol("BTC/USDT") == to_woox_symbol("BTC/USDT")


# ===========================================================================
# 3. WOO X — Symbol normalisation: perp
# ===========================================================================

class TestWOOXSymbolToPerp(unittest.TestCase):
    """to_woox_perp converts slash-format to PERP_ prefix."""

    def test_btc_usdt(self):
        assert to_woox_perp("BTC/USDT") == "PERP_BTC_USDT"

    def test_eth_usdt(self):
        assert to_woox_perp("ETH/USDT") == "PERP_ETH_USDT"

    def test_already_perp_prefix(self):
        assert to_woox_perp("PERP_BTC_USDT") == "PERP_BTC_USDT"

    def test_spot_prefix_converted_to_perp(self):
        assert to_woox_perp("SPOT_BTC_USDT") == "PERP_BTC_USDT"

    def test_lowercase_input(self):
        assert to_woox_perp("btc/usdt") == "PERP_BTC_USDT"

    def test_instance_method_matches_module_function(self):
        client = WOOXClient(api_key="test_key", api_secret="test_secret")
        assert client.to_woox_perp("BTC/USDT") == to_woox_perp("BTC/USDT")


# ===========================================================================
# 4. WOO X — Symbol normalisation: reverse
# ===========================================================================

class TestWOOXFromSymbol(unittest.TestCase):
    """from_woox_symbol converts WOO X format back to slash-separated."""

    def test_spot_btc_usdt(self):
        assert from_woox_symbol("SPOT_BTC_USDT") == "BTC/USDT"

    def test_perp_btc_usdt(self):
        assert from_woox_symbol("PERP_BTC_USDT") == "BTC/USDT"

    def test_spot_eth_usdt(self):
        assert from_woox_symbol("SPOT_ETH_USDT") == "ETH/USDT"

    def test_perp_eth_usdt(self):
        assert from_woox_symbol("PERP_ETH_USDT") == "ETH/USDT"

    def test_roundtrip_spot(self):
        """Roundtrip: to_woox_symbol → from_woox_symbol should be identity."""
        original = "BTC/USDT"
        assert from_woox_symbol(to_woox_symbol(original)) == original

    def test_roundtrip_perp(self):
        original = "ETH/USDT"
        assert from_woox_symbol(to_woox_perp(original)) == original

    def test_instance_method_matches_module_function(self):
        client = WOOXClient(api_key="test_key", api_secret="test_secret")
        assert client.from_woox_symbol("SPOT_BTC_USDT") == from_woox_symbol("SPOT_BTC_USDT")


# ===========================================================================
# 5. WOO X — HMAC signing (deterministic)
# ===========================================================================

class TestWOOXSignDeterministic(unittest.TestCase):
    """
    WOO X signing must be deterministic: same inputs always produce
    the same HMAC-SHA256 hex signature.
    """

    def _expected_sig(
        self, secret: str, timestamp: str, method: str, path: str, body: str = ""
    ) -> str:
        """Compute expected signature independently of WOOXClient._sign."""
        message = f"{timestamp}{method}{path}{body}"
        return hmac.new(
            secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def test_sign_is_deterministic(self):
        """Same inputs → same signature, always."""
        client = WOOXClient(api_key="apikey123", api_secret="secretabc")
        ts = "1700000000000"
        method = "GET"
        path = "/v3/orders"
        body = ""

        sig1 = client._sign(ts, method, path, body)
        sig2 = client._sign(ts, method, path, body)
        assert sig1 == sig2, "Signature not deterministic"

    def test_sign_matches_reference_hmac(self):
        """Client signature matches independent HMAC computation."""
        secret = "my_super_secret_key"
        client = WOOXClient(api_key="any_key", api_secret=secret)
        ts = "1700000000000"
        method = "POST"
        path = "/v3/order"
        body = "symbol=SPOT_BTC_USDT&order_type=LIMIT&side=BUY"

        client_sig = client._sign(ts, method, path, body)
        expected_sig = self._expected_sig(secret, ts, method, path, body)

        assert client_sig == expected_sig, (
            f"Signature mismatch:\n  client:   {client_sig}\n  expected: {expected_sig}"
        )

    def test_sign_different_timestamps_different_sig(self):
        """Different timestamps must produce different signatures."""
        client = WOOXClient(api_key="key", api_secret="secret")
        path = "/v3/order"
        method = "GET"
        sig_a = client._sign("1000000000000", method, path)
        sig_b = client._sign("9999999999999", method, path)
        assert sig_a != sig_b, "Different timestamps must give different signatures"

    def test_sign_returns_hex_string(self):
        """Signature must be a 64-char hex string (SHA-256 output)."""
        client = WOOXClient(api_key="k", api_secret="s")
        sig = client._sign("1234567890123", "GET", "/v1/public/info")
        assert isinstance(sig, str)
        assert len(sig) == 64
        int(sig, 16)  # must be valid hex (raises ValueError if not)


# ===========================================================================
# 6. WOO X — post_only flag in create_order
# ===========================================================================

class TestWOOXPostOnlyFlag(unittest.TestCase):
    """
    When post_only=True, create_order must send order_type=POST_ONLY,
    guaranteeing maker execution and the zero maker fee.
    """

    def setUp(self):
        self.client = WOOXClient(api_key="key", api_secret="secret")

    def test_post_only_sets_order_type_post_only(self):
        """post_only=True should result in POST_ONLY order type in request."""
        captured: dict = {}

        async def fake_request(method, path, *, data=None, params=None,
                               signed=False, api_version="v1"):
            captured["method"] = method
            captured["data"] = dict(data or {})
            # Return a minimal success response
            return {
                "success": True,
                "order_id": 99999,
                "status": "NEW",
            }

        self.client._request = fake_request

        result = asyncio.run(self.client.create_order(
            symbol="BTC/USDT",
            side="BUY",
            order_type="LIMIT",
            quantity=0.001,
            price=30000.0,
            post_only=True,
        ))

        assert captured["data"]["order_type"] == "POST_ONLY", (
            f"Expected POST_ONLY order type, got: {captured['data'].get('order_type')}"
        )
        assert result["post_only"] is True

    def test_post_only_false_keeps_limit_type(self):
        """post_only=False should keep the order_type as LIMIT."""
        captured: dict = {}

        async def fake_request(method, path, *, data=None, params=None,
                               signed=False, api_version="v1"):
            captured["data"] = dict(data or {})
            return {"success": True, "order_id": 12345, "status": "NEW"}

        self.client._request = fake_request

        asyncio.run(self.client.create_order(
            symbol="BTC/USDT",
            side="BUY",
            order_type="LIMIT",
            quantity=0.001,
            price=30000.0,
            post_only=False,
        ))

        assert captured["data"]["order_type"] == "LIMIT"


# ===========================================================================
# 7. WOO X — fetch_ticker (mocked)
# ===========================================================================

class TestWOOXFetchTickerMock(unittest.TestCase):
    """fetch_ticker normalises the WOO X response correctly."""

    def setUp(self):
        self.client = WOOXClient(api_key="key", api_secret="secret")

    def test_fetch_ticker_normalised(self):
        """Mocked ticker response should be normalised with correct field names."""
        call_count = [0]

        async def fake_request(method, path, *, params=None, data=None,
                               signed=False, api_version="v1"):
            call_count[0] += 1
            if "market_trades" in path:
                return {
                    "success": True,
                    "rows": [
                        {"executed_price": 65000.5, "executed_quantity": 0.001}
                    ]
                }
            elif "orderbook" in path:
                return {
                    "success": True,
                    "timestamp": 1700000000000,
                    "bids": [[65000.0, 1.5, 2], [64999.0, 0.5, 1]],
                    "asks": [[65001.0, 0.8, 1], [65002.0, 2.0, 3]],
                }
            return {}

        self.client._request = fake_request

        ticker = asyncio.run(self.client.fetch_ticker("BTC/USDT"))

        assert ticker["exchange"] == "woox"
        assert ticker["symbol"] == "BTC/USDT"
        # bid should be first bid level price
        assert ticker["bid"] == pytest.approx(65000.0)
        # ask should be first ask level price
        assert ticker["ask"] == pytest.approx(65001.0)
        # last from trade
        assert ticker["last"] == pytest.approx(65000.5)
        # Both requests were made
        assert call_count[0] >= 2


# ===========================================================================
# 8. dYdX — Constants
# ===========================================================================

class TestDYDXConstants(unittest.TestCase):
    """dYdX fee constants: maker=0.0001, taker=0.0005."""

    def test_maker_fee(self):
        assert DYDX_MAKER_FEE == 0.0001, f"Expected 0.0001, got {DYDX_MAKER_FEE}"

    def test_taker_fee(self):
        assert DYDX_TAKER_FEE == 0.0005, f"Expected 0.0005, got {DYDX_TAKER_FEE}"

    def test_maker_fee_bps(self):
        """Maker fee in bps should be 1 bps (0.01%)."""
        assert DYDX_MAKER_FEE * 10_000 == pytest.approx(1.0)

    def test_taker_fee_bps(self):
        """Taker fee in bps should be 5 bps (0.05%)."""
        assert DYDX_TAKER_FEE * 10_000 == pytest.approx(5.0)

    def test_maker_less_than_taker(self):
        assert DYDX_MAKER_FEE < DYDX_TAKER_FEE

    def test_indexer_url(self):
        assert DYDX_INDEXER_URL == "https://indexer.dydx.trade/v4"

    def test_ws_url(self):
        assert DYDX_WS_URL == "wss://indexer.dydx.trade/v4/ws"

    def test_is_decentralised_flag(self):
        assert DYDX_IS_DECENTRALISED is True

    def test_get_exchange_info_structure(self):
        info = dydx_get_exchange_info()
        assert info["exchange"] == "dydx"
        assert info["fee_rates"]["maker"] == 0.0001
        assert info["fee_rates"]["taker"] == 0.0005
        assert info["properties"]["is_decentralised"] is True
        assert info["properties"]["requires_kyc"] is False


# ===========================================================================
# 9. dYdX — is_custodial() always False
# ===========================================================================

class TestDYDXIsNotCustodial(unittest.TestCase):
    """DYDXClient.is_custodial() must always return False."""

    def test_is_custodial_false_without_key(self):
        client = DYDXClient(wallet_address="dydx1abc")
        assert client.is_custodial() is False

    def test_is_custodial_false_with_mnemonic(self):
        client = DYDXClient(
            wallet_address="dydx1abc",
            mnemonic="word " * 12,
        )
        assert client.is_custodial() is False

    def test_is_custodial_false_with_private_key(self):
        client = DYDXClient(
            wallet_address="dydx1abc",
            private_key_hex="a" * 64,
        )
        assert client.is_custodial() is False

    def test_is_custodial_on_testnet(self):
        client = DYDXClient(wallet_address="dydx1abc", testnet=True)
        assert client.is_custodial() is False


# ===========================================================================
# 10. dYdX — DYDX_REQUIRES_KYC must be False
# ===========================================================================

class TestDYDXNoKYC(unittest.TestCase):
    """dYdX requires no KYC — self-custodial, permissionless."""

    def test_requires_kyc_constant_false(self):
        assert DYDX_REQUIRES_KYC is False

    def test_requires_kyc_in_exchange_info(self):
        info = dydx_get_exchange_info()
        assert info["properties"]["requires_kyc"] is False

    def test_get_estimated_fees_structure(self):
        client = DYDXClient(wallet_address="dydx1abc")
        fees = client.get_estimated_fees(size_usd=10_000.0)
        assert fees["maker_fee_usd"] == pytest.approx(10_000.0 * DYDX_MAKER_FEE)
        assert fees["taker_fee_usd"] == pytest.approx(10_000.0 * DYDX_TAKER_FEE)
        assert fees["maker_fee_usd"] < fees["taker_fee_usd"]


# ===========================================================================
# 11. dYdX — fetch_ticker mock
# ===========================================================================

class TestDYDXFetchTickerMock(unittest.TestCase):
    """fetch_ticker normalises the dYdX indexer response correctly."""

    def setUp(self):
        self.client = DYDXClient(wallet_address="dydx1testaddress")

    def test_fetch_ticker_normalised(self):
        """Mocked indexer response should be normalised with correct field names."""
        mock_response = {
            "markets": {
                "BTC-USD": {
                    "ticker": "BTC-USD",
                    "lastTradedPrice": "65123.45",
                    "oraclePrice": "65100.00",
                    "bidPrice": "65120.00",
                    "askPrice": "65125.00",
                    "volume24H": "500000000",
                    "trades24H": "15234",
                    "openInterest": "1250.5",
                    "nextFundingRate": "0.00005",
                    "nextFundingAt": "2024-01-01T08:00:00Z",
                    "status": "ACTIVE",
                }
            }
        }

        async def fake_request(method, path, *, params=None, json_body=None):
            return mock_response

        self.client._request = fake_request

        ticker = asyncio.run(self.client.fetch_ticker("BTC/USD"))

        assert ticker["exchange"] == "dydx"
        assert ticker["symbol"] == "BTC/USD"
        assert ticker["bid"] == pytest.approx(65120.00)
        assert ticker["ask"] == pytest.approx(65125.00)
        assert ticker["last"] == pytest.approx(65123.45)
        assert ticker["oracle_price"] == pytest.approx(65100.00)
        assert ticker["funding_rate"] == pytest.approx(0.00005)
        assert ticker["status"] == "ACTIVE"

    def test_fetch_ticker_symbol_normalisation(self):
        """BTC/USDT should be normalised to BTC-USD for dYdX."""
        called_with_symbol: list = []

        async def fake_request(method, path, *, params=None, json_body=None):
            called_with_symbol.append(params.get("ticker", "") if params else "")
            return {"markets": {"BTC-USD": {"oraclePrice": "65000"}}}

        self.client._request = fake_request
        asyncio.run(self.client.fetch_ticker("BTC/USDT"))

        # Should have been called with BTC-USD (USDT → USD normalised)
        assert "BTC-USD" in called_with_symbol


# ===========================================================================
# 12. dYdX — create_order without key → NotImplementedError
# ===========================================================================

class TestDYDXOrderWithoutKey(unittest.TestCase):
    """
    DYDXClient without a private key must raise NotImplementedError
    when order placement methods are called.
    """

    def setUp(self):
        # Initialise without any signing key
        self.client = DYDXClient(wallet_address="dydx1testaddress")

    def test_create_order_raises_not_implemented(self):
        with pytest.raises(NotImplementedError) as exc_info:
            asyncio.run(self.client.create_order(
                symbol="BTC-USD",
                side="BUY",
                order_type="LIMIT",
                size=0.001,
                price=65000.0,
            ))
        # Message should be informative
        msg = str(exc_info.value)
        assert "private key" in msg.lower() or "mnemonic" in msg.lower() or "key" in msg.lower()

    def test_cancel_order_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            asyncio.run(self.client.cancel_order(
                symbol="BTC-USD",
                order_id="some-order-id-123",
            ))

    def test_no_key_client_still_has_is_custodial(self):
        """Client without key can still call is_custodial()."""
        assert self.client.is_custodial() is False

    def test_fetch_positions_without_key_returns_empty(self):
        """fetch_positions with no wallet_address returns empty list."""
        client = DYDXClient(wallet_address="")
        positions = asyncio.run(client.fetch_positions())
        assert positions == []


# ===========================================================================
# 13. Registry — has woox and dydx
# ===========================================================================

class TestRegistryHasWooxDydx(unittest.TestCase):
    """EXCHANGE_REGISTRY must contain both woox and dydx entries."""

    def test_woox_in_registry(self):
        assert "woox" in EXCHANGE_REGISTRY, (
            f"'woox' not found in EXCHANGE_REGISTRY. Keys: {list(EXCHANGE_REGISTRY.keys())}"
        )

    def test_dydx_in_registry(self):
        assert "dydx" in EXCHANGE_REGISTRY, (
            f"'dydx' not found in EXCHANGE_REGISTRY. Keys: {list(EXCHANGE_REGISTRY.keys())}"
        )

    def test_woox_profile_structure(self):
        profile = EXCHANGE_REGISTRY["woox"]
        assert profile.name == "woox"
        assert hasattr(profile, "spot_maker_fee")
        assert hasattr(profile, "preferred_for_mm")
        assert hasattr(profile, "preferred_for_funding_arb")

    def test_dydx_profile_structure(self):
        profile = EXCHANGE_REGISTRY["dydx"]
        assert profile.name == "dydx"
        assert hasattr(profile, "futures_maker_fee")
        assert hasattr(profile, "preferred_for_funding_arb")

    def test_woox_preferred_for_mm(self):
        assert EXCHANGE_REGISTRY["woox"].preferred_for_mm is True

    def test_dydx_preferred_for_funding_arb(self):
        assert EXCHANGE_REGISTRY["dydx"].preferred_for_funding_arb is True

    def test_min_spread_woox(self):
        spread = min_spread_to_profit("woox")
        assert spread == pytest.approx(0.0)

    def test_min_spread_dydx(self):
        spread = min_spread_to_profit("dydx")
        assert spread == pytest.approx(2.0)


# ===========================================================================
# 14. Registry — woox maker fee zero
# ===========================================================================

class TestRegistryWooxZeroFee(unittest.TestCase):
    """WOO X spot_maker_fee in the registry must be exactly zero."""

    def test_woox_spot_maker_fee_zero(self):
        profile = EXCHANGE_REGISTRY["woox"]
        assert profile.spot_maker_fee == 0.0, (
            f"woox spot_maker_fee must be 0.0, got {profile.spot_maker_fee}"
        )

    def test_woox_spot_taker_fee_zero(self):
        profile = EXCHANGE_REGISTRY["woox"]
        assert profile.spot_taker_fee == 0.0

    def test_woox_futures_maker_fee_zero(self):
        profile = EXCHANGE_REGISTRY["woox"]
        assert profile.futures_maker_fee == 0.0

    def test_woox_futures_taker_fee_zero(self):
        profile = EXCHANGE_REGISTRY["woox"]
        assert profile.futures_taker_fee == 0.0

    def test_dydx_maker_fee_matches_constant(self):
        """Registry dydx maker fee should match DYDX_MAKER_FEE constant."""
        profile = EXCHANGE_REGISTRY["dydx"]
        assert profile.futures_maker_fee == pytest.approx(DYDX_MAKER_FEE)

    def test_dydx_not_aus_regulated(self):
        profile = EXCHANGE_REGISTRY["dydx"]
        assert profile.is_aus_regulated is False

    def test_woox_not_aus_regulated(self):
        profile = EXCHANGE_REGISTRY["woox"]
        assert profile.is_aus_regulated is False

    def test_dydx_client_class_name(self):
        profile = EXCHANGE_REGISTRY["dydx"]
        assert profile.client_class == "DYDXClient"

    def test_woox_client_class_name(self):
        profile = EXCHANGE_REGISTRY["woox"]
        assert profile.client_class == "WOOXClient"
