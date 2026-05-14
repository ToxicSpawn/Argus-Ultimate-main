"""
OKX Perpetual Futures Connector (V5 REST API).

Full trading connector supporting:
- Order placement (market/limit) and cancellation
- Position management (get/set leverage)
- Funding rate queries (current + predicted)
- Balance queries (equity, available, margin)
- Orderbook depth
- Mark price
- HMAC-SHA256 authentication (base64 encoded)
- Token-bucket rate limiting (20 req/s private, 40 req/s public)

OKX supports Australian users and offers:
- USDT-margined perpetual swaps
- Among the highest funding rates during bull markets (good for harvesting)
- Deep liquidity across 50+ pairs
- No geo-restriction for Australian residents

API keys: Set OKX_API_KEY / OKX_API_SECRET / OKX_PASSPHRASE in environment.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# OKX V5 API base URLs
OKX_V5_MAINNET = "https://www.okx.com"
OKX_V5_TESTNET = "https://www.okx.com"  # OKX uses same domain, demo flag in header

# Rate limits per OKX docs
_PRIVATE_RATE_LIMIT = 20   # 20 req/s for private endpoints
_PUBLIC_RATE_LIMIT = 40    # 40 req/s for public endpoints


class OKXRateLimiter:
    """Token-bucket rate limiter for OKX API calls."""

    def __init__(self, max_per_second: int = 20):
        self._max = max_per_second
        self._tokens: float = float(max_per_second)
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._max, self._tokens + elapsed * self._max)
            self._last_refill = now
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._max
                await asyncio.sleep(wait)
                self._tokens = 0.0
                self._last_refill = time.monotonic()
            else:
                self._tokens -= 1.0


class OKXAPIError(Exception):
    """Raised when OKX V5 API returns a non-zero error code."""

    def __init__(self, code: str, msg: str, endpoint: str = ""):
        self.code = code
        self.msg = msg
        self.endpoint = endpoint
        super().__init__(f"OKX API error {code}: {msg} (endpoint={endpoint})")


class OKXConnector:
    """
    OKX USDT perpetual swaps connector using V5 REST API.

    Provides async methods for market data and authenticated trading
    via aiohttp + HMAC-SHA256 authentication (base64 encoded).
    """

    health_check_symbol = "BTC-USDT-SWAP"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        testnet: bool = False,
    ):
        self.api_key = api_key or os.environ.get("OKX_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("OKX_API_SECRET", "")
        self.passphrase = passphrase or os.environ.get("OKX_PASSPHRASE", "")
        self.testnet = testnet
        self.base_url = OKX_V5_MAINNET
        self._session: Any = None  # aiohttp.ClientSession
        self.connected: bool = False

        # Rate limiters
        self._private_limiter = OKXRateLimiter(_PRIVATE_RATE_LIMIT)
        self._public_limiter = OKXRateLimiter(_PUBLIC_RATE_LIMIT)

    # ------------------------------------------------------------------
    # Authentication — HMAC-SHA256 signing per OKX V5 spec
    # ------------------------------------------------------------------

    @staticmethod
    def _iso_timestamp() -> str:
        """Generate ISO 8601 timestamp for OKX API (UTC)."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """
        Generate HMAC-SHA256 signature for OKX V5 API.

        Sign string = timestamp + method + path + body
        Result is base64-encoded.
        """
        message = timestamp + method.upper() + path + body
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(signature).decode("utf-8")

    def _auth_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Build authenticated headers for OKX V5 API request."""
        timestamp = self._iso_timestamp()
        signature = self._sign(timestamp, method, path, body)
        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }
        if self.testnet:
            headers["x-simulated-trading"] = "1"
        return headers

    # ------------------------------------------------------------------
    # Symbol mapping
    # ------------------------------------------------------------------

    @staticmethod
    def to_okx_symbol(symbol: str, inst_type: str = "SWAP") -> str:
        """
        Convert standard symbol to OKX instrument ID.

        "BTC/USDT" perp -> "BTC-USDT-SWAP"
        "BTC/USDT" spot -> "BTC-USDT"
        "BTC-USDT-SWAP"  -> "BTC-USDT-SWAP" (no-op)
        """
        # Already in OKX format
        if "-SWAP" in symbol or symbol.count("-") >= 1 and "/" not in symbol:
            return symbol

        # Strip CCXT perp suffix
        s = symbol.replace(":USDT", "").replace(":USD", "")
        # "BTC/USDT" -> "BTC-USDT"
        s = s.replace("/", "-")

        if inst_type == "SWAP":
            if not s.endswith("-SWAP"):
                s = s + "-SWAP"
        return s

    @staticmethod
    def from_okx_symbol(inst_id: str) -> str:
        """
        Convert OKX instrument ID to standard symbol.

        "BTC-USDT-SWAP" -> "BTC/USDT"
        "BTC-USDT" -> "BTC/USDT"
        """
        parts = inst_id.split("-")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return inst_id

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _get_session(self) -> Any:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession()
            except ImportError:
                logger.warning("aiohttp not available, OKX direct API disabled")
                return None
        return self._session

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        auth: bool = False,
    ) -> Dict[str, Any]:
        """
        Make a request to OKX V5 API.

        Args:
            method: HTTP method (GET or POST)
            endpoint: API path, e.g. "/api/v5/market/ticker"
            params: Query parameters for GET
            body: JSON body for POST
            auth: Whether to sign the request

        Returns:
            Parsed 'data' field from the response

        Raises:
            OKXAPIError: If the response code is not "0"
        """
        session = await self._get_session()
        if session is None:
            raise RuntimeError("aiohttp session not available")

        url = f"{self.base_url}{endpoint}"

        # Build query string for GET
        query_string = ""
        if params:
            query_string = "?" + "&".join(f"{k}={v}" for k, v in params.items())

        body_str = ""
        if body is not None:
            body_str = json.dumps(body)

        # Select rate limiter
        limiter = self._private_limiter if auth else self._public_limiter
        await limiter.acquire()

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if auth:
            sign_path = endpoint + query_string
            headers = self._auth_headers(method.upper(), sign_path, body_str)

        if method.upper() == "GET":
            async with session.get(
                url + query_string, headers=headers, timeout=10
            ) as resp:
                data = await resp.json()
        else:
            async with session.post(
                url + query_string, data=body_str, headers=headers, timeout=10
            ) as resp:
                data = await resp.json()

        code = data.get("code", "-1")
        if code != "0":
            raise OKXAPIError(code, data.get("msg", "unknown"), endpoint)

        return data.get("data", [])

    async def _public_get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Make a public (unauthenticated) GET request."""
        return await self._request("GET", endpoint, params=params, auth=False)

    async def _private_get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Make an authenticated GET request."""
        return await self._request("GET", endpoint, params=params, auth=True)

    async def _private_post(
        self, endpoint: str, body: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Make an authenticated POST request."""
        return await self._request("POST", endpoint, body=body or {}, auth=True)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect and verify API access."""
        try:
            session = await self._get_session()
            if session is not None:
                result = await self._public_get(
                    "/api/v5/market/ticker",
                    {"instId": "BTC-USDT-SWAP"},
                )
                if result and len(result) > 0:
                    self.connected = True
                    logger.info("OKX V5 API connected (direct)")
                    return True
        except Exception as exc:
            logger.warning("OKX connect failed: %s", exc)

        self.connected = False
        return False

    async def disconnect(self) -> None:
        """Disconnect and clean up resources."""
        self.connected = False
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "OKXConnector":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Public endpoints — Market data
    # ------------------------------------------------------------------

    async def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current ticker data for a symbol.

        Args:
            symbol: Trading pair, e.g. "BTC/USDT" or "BTC-USDT-SWAP"

        Returns:
            {bid, ask, last, volume_24h, funding_rate, next_funding_time, symbol, exchange}
        """
        inst_id = self.to_okx_symbol(symbol)
        try:
            result = await self._public_get(
                "/api/v5/market/ticker",
                {"instId": inst_id},
            )
            if result and len(result) > 0:
                t = result[0]
                return {
                    "symbol": symbol,
                    "inst_id": inst_id,
                    "last": float(t.get("last", 0) or 0),
                    "bid": float(t.get("bidPx", 0) or 0),
                    "ask": float(t.get("askPx", 0) or 0),
                    "bid_size": float(t.get("bidSz", 0) or 0),
                    "ask_size": float(t.get("askSz", 0) or 0),
                    "volume_24h": float(t.get("vol24h", 0) or 0),
                    "volume_ccy_24h": float(t.get("volCcy24h", 0) or 0),
                    "timestamp": int(t.get("ts", 0) or 0),
                    "exchange": "okx",
                }
            return None
        except Exception as exc:
            logger.warning("OKX get_ticker %s failed: %s", symbol, exc)
            return None

    async def get_orderbook(
        self, symbol: str, depth: int = 20
    ) -> Optional[Dict[str, Any]]:
        """
        Get orderbook for a symbol.

        Args:
            symbol: Trading pair
            depth: Number of levels (max 400)

        Returns:
            {bids: [(price, qty), ...], asks: [(price, qty), ...], symbol, timestamp}
        """
        inst_id = self.to_okx_symbol(symbol)
        sz = min(depth, 400)
        try:
            result = await self._public_get(
                "/api/v5/market/books",
                {"instId": inst_id, "sz": str(sz)},
            )
            if result and len(result) > 0:
                book = result[0]
                bids = [
                    (float(level[0]), float(level[1]))
                    for level in book.get("bids", [])
                ]
                asks = [
                    (float(level[0]), float(level[1]))
                    for level in book.get("asks", [])
                ]
                return {
                    "symbol": symbol,
                    "inst_id": inst_id,
                    "bids": bids,
                    "asks": asks,
                    "timestamp": int(book.get("ts", 0) or 0),
                    "exchange": "okx",
                }
            return None
        except Exception as exc:
            logger.warning("OKX get_orderbook %s failed: %s", symbol, exc)
            return None

    async def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Get current and predicted funding rate for a symbol.

        Args:
            symbol: Trading pair, e.g. "BTC/USDT" or "BTC-USDT-SWAP"

        Returns:
            {current_rate, predicted_rate, next_settlement, symbol, exchange}
        """
        inst_id = self.to_okx_symbol(symbol)
        try:
            result = await self._public_get(
                "/api/v5/public/funding-rate",
                {"instId": inst_id},
            )
            if result and len(result) > 0:
                fr = result[0]
                return {
                    "symbol": symbol,
                    "inst_id": inst_id,
                    "funding_rate": float(fr.get("fundingRate", 0) or 0),
                    "current_rate": float(fr.get("fundingRate", 0) or 0),
                    "predicted_rate": float(fr.get("nextFundingRate", 0) or 0),
                    "next_settlement": int(fr.get("nextFundingTime", 0) or 0),
                    "exchange": "okx",
                }
            return {"symbol": symbol, "funding_rate": 0.0, "exchange": "okx"}
        except Exception as exc:
            logger.warning("OKX get_funding_rate %s failed: %s", symbol, exc)
            return {"symbol": symbol, "funding_rate": 0.0, "exchange": "okx", "error": str(exc)}

    # Alias for ExchangeManager compatibility
    async def fetch_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """Alias for get_funding_rate() — backward compatibility."""
        return await self.get_funding_rate(symbol)

    async def fetch_funding_rates(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch funding rates for multiple symbols. Returns {symbol: rate}."""
        rates: Dict[str, float] = {}
        for sym in symbols:
            info = await self.get_funding_rate(sym)
            rates[sym] = info.get("funding_rate", 0.0)
        return rates

    async def get_mark_price(self, symbol: str) -> float:
        """
        Get mark price for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            Mark price as float, or 0.0 on error
        """
        inst_id = self.to_okx_symbol(symbol)
        try:
            result = await self._public_get(
                "/api/v5/public/mark-price",
                {"instType": "SWAP", "instId": inst_id},
            )
            if result and len(result) > 0:
                return float(result[0].get("markPx", 0) or 0)
            return 0.0
        except Exception as exc:
            logger.warning("OKX get_mark_price %s failed: %s", symbol, exc)
            return 0.0

    # ------------------------------------------------------------------
    # Private endpoints — Account
    # ------------------------------------------------------------------

    async def get_balances(self) -> Dict[str, Dict[str, float]]:
        """
        Get account balances for all currencies.

        Returns:
            {currency: {available, total, equity}}
        """
        try:
            result = await self._private_get("/api/v5/account/balance")
            balances: Dict[str, Dict[str, float]] = {}
            if result and len(result) > 0:
                for detail in result[0].get("details", []):
                    ccy = detail.get("ccy", "")
                    balances[ccy] = {
                        "available": float(detail.get("availBal", 0) or 0),
                        "total": float(detail.get("cashBal", 0) or 0),
                        "equity": float(detail.get("eq", 0) or 0),
                        "unrealized_pnl": float(detail.get("upl", 0) or 0),
                    }
            return balances
        except Exception as exc:
            logger.warning("OKX get_balances failed: %s", exc)
            return {}

    async def get_balance(self) -> Dict[str, Any]:
        """Alias for get_balances() — ExchangeManager compatibility."""
        return await self.get_balances()

    # ------------------------------------------------------------------
    # Private endpoints — Trading
    # ------------------------------------------------------------------

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str = "market",
        size: float = 0.0,
        price: Optional[float] = None,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Place an order on OKX perpetual swaps.

        Args:
            symbol: Trading pair, e.g. "BTC/USDT" or "BTC-USDT-SWAP"
            side: "buy" or "sell" (case-insensitive)
            order_type: "market" or "limit" (case-insensitive)
            size: Order size in contracts
            price: Required for limit orders
            reduce_only: If True, only reduce existing position

        Returns:
            Order response dict with ordId, sCode, sMsg, etc.

        Raises:
            OKXAPIError: On API errors
        """
        inst_id = self.to_okx_symbol(symbol)
        side_lower = side.lower()
        type_lower = order_type.lower()

        body: Dict[str, Any] = {
            "instId": inst_id,
            "tdMode": "cross",       # cross margin
            "side": side_lower,
            "ordType": type_lower,
            "sz": str(size),
        }

        if reduce_only:
            body["reduceOnly"] = True

        if price is not None and type_lower == "limit":
            body["px"] = str(price)

        try:
            result = await self._private_post("/api/v5/trade/order", body)
            if result and len(result) > 0:
                order_data = result[0]
                s_code = order_data.get("sCode", "")
                if s_code != "0":
                    raise OKXAPIError(
                        s_code, order_data.get("sMsg", ""), "/api/v5/trade/order"
                    )
                logger.info(
                    "OKX order placed: %s %s %.6f %s (ordId=%s)",
                    side_lower, inst_id, size, type_lower,
                    order_data.get("ordId", "?"),
                )
                return {
                    "order_id": order_data.get("ordId", ""),
                    "client_order_id": order_data.get("clOrdId", ""),
                    "symbol": symbol,
                    "inst_id": inst_id,
                    "side": side_lower,
                    "order_type": type_lower,
                    "size": size,
                    "price": price,
                    "exchange": "okx",
                }
            return {}
        except OKXAPIError:
            raise
        except Exception as exc:
            logger.error("OKX place_order failed: %s", exc)
            raise

    async def create_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = "market",
        price: Optional[float] = None,
        reduce_only: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """CCXT-compatible order interface."""
        try:
            return await self.place_order(
                symbol, side, order_type, amount, price, reduce_only
            )
        except Exception as exc:
            logger.error("OKX create_order %s %s %s: %s", side, amount, symbol, exc)
            return None

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            symbol: Trading pair
            order_id: OKX order ID

        Returns:
            True if cancellation was accepted
        """
        inst_id = self.to_okx_symbol(symbol)
        try:
            result = await self._private_post(
                "/api/v5/trade/cancel-order",
                {"instId": inst_id, "ordId": order_id},
            )
            if result and len(result) > 0:
                s_code = result[0].get("sCode", "")
                if s_code == "0":
                    logger.info("OKX order cancelled: %s %s", order_id, inst_id)
                    return True
                raise OKXAPIError(
                    s_code, result[0].get("sMsg", ""), "/api/v5/trade/cancel-order"
                )
            return False
        except OKXAPIError:
            raise
        except Exception as exc:
            logger.error("OKX cancel_order %s failed: %s", order_id, exc)
            return False

    async def get_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Get details for a specific order.

        Args:
            symbol: Trading pair
            order_id: OKX order ID

        Returns:
            Order status dict
        """
        inst_id = self.to_okx_symbol(symbol)
        try:
            result = await self._private_get(
                "/api/v5/trade/order",
                {"instId": inst_id, "ordId": order_id},
            )
            if result and len(result) > 0:
                o = result[0]
                return {
                    "order_id": o.get("ordId", ""),
                    "client_order_id": o.get("clOrdId", ""),
                    "symbol": symbol,
                    "inst_id": inst_id,
                    "side": o.get("side", ""),
                    "order_type": o.get("ordType", ""),
                    "size": float(o.get("sz", 0) or 0),
                    "filled_size": float(o.get("accFillSz", 0) or 0),
                    "price": float(o.get("px", 0) or 0),
                    "avg_price": float(o.get("avgPx", 0) or 0),
                    "status": o.get("state", ""),
                    "fee": float(o.get("fee", 0) or 0),
                    "pnl": float(o.get("pnl", 0) or 0),
                    "timestamp": int(o.get("cTime", 0) or 0),
                    "exchange": "okx",
                }
            return {}
        except Exception as exc:
            logger.warning("OKX get_order %s failed: %s", order_id, exc)
            return {}

    # ------------------------------------------------------------------
    # Private endpoints — Positions
    # ------------------------------------------------------------------

    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open perpetual positions.

        Returns:
            List of position dicts
        """
        try:
            result = await self._private_get(
                "/api/v5/account/positions",
                {"instType": "SWAP"},
            )
            positions: List[Dict[str, Any]] = []
            for pos in (result or []):
                size = float(pos.get("pos", 0) or 0)
                if size != 0:
                    positions.append({
                        "symbol": self.from_okx_symbol(pos.get("instId", "")),
                        "inst_id": pos.get("instId", ""),
                        "size": abs(size),
                        "side": "long" if size > 0 else "short",
                        "entry_price": float(pos.get("avgPx", 0) or 0),
                        "mark_price": float(pos.get("markPx", 0) or 0),
                        "unrealized_pnl": float(pos.get("upl", 0) or 0),
                        "leverage": float(pos.get("lever", 1) or 1),
                        "liq_price": float(pos.get("liqPx", 0) or 0),
                        "margin": float(pos.get("margin", 0) or 0),
                        "exchange": "okx",
                    })
            return positions
        except Exception as exc:
            logger.warning("OKX get_positions failed: %s", exc)
            return []

    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current position for a specific symbol.

        Args:
            symbol: Trading pair

        Returns:
            Position dict or None
        """
        inst_id = self.to_okx_symbol(symbol)
        try:
            result = await self._private_get(
                "/api/v5/account/positions",
                {"instType": "SWAP", "instId": inst_id},
            )
            for pos in (result or []):
                size = float(pos.get("pos", 0) or 0)
                if size != 0:
                    return {
                        "symbol": symbol,
                        "inst_id": inst_id,
                        "size": abs(size),
                        "side": "long" if size > 0 else "short",
                        "entry_price": float(pos.get("avgPx", 0) or 0),
                        "mark_price": float(pos.get("markPx", 0) or 0),
                        "unrealized_pnl": float(pos.get("upl", 0) or 0),
                        "leverage": float(pos.get("lever", 1) or 1),
                        "liq_price": float(pos.get("liqPx", 0) or 0),
                        "exchange": "okx",
                    }
            return None
        except Exception as exc:
            logger.warning("OKX get_position %s failed: %s", symbol, exc)
            return None

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a symbol.

        Args:
            symbol: Trading pair
            leverage: Leverage value (1-125)

        Returns:
            True if leverage was set successfully
        """
        inst_id = self.to_okx_symbol(symbol)
        try:
            await self._private_post(
                "/api/v5/account/set-leverage",
                {
                    "instId": inst_id,
                    "lever": str(leverage),
                    "mgnMode": "cross",
                },
            )
            logger.info("OKX leverage set: %s -> %dx", inst_id, leverage)
            return True
        except OKXAPIError as exc:
            # Code 59000 = leverage not changed
            if exc.code == "59000":
                logger.debug("Leverage already set to %dx for %s", leverage, inst_id)
                return True
            logger.error("OKX set_leverage %s failed: %s", symbol, exc)
            return False
        except Exception as exc:
            logger.error("OKX set_leverage %s failed: %s", symbol, exc)
            return False

    # ------------------------------------------------------------------
    # Position close
    # ------------------------------------------------------------------

    async def close_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Close an open position at market."""
        try:
            pos = await self.get_position(symbol)
            if not pos or pos.get("size", 0) == 0:
                logger.info("No open position to close for %s", symbol)
                return None
            side = "sell" if pos.get("side") == "long" else "buy"
            amount = abs(pos.get("size", 0))
            return await self.place_order(symbol, side, "market", amount, reduce_only=True)
        except Exception as exc:
            logger.error("OKX close_position %s failed: %s", symbol, exc)
            return None

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """Ping OKX to verify connectivity."""
        t0 = time.perf_counter()
        try:
            result = await self._public_get(
                "/api/v5/market/ticker",
                {"instId": "BTC-USDT-SWAP"},
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            healthy = result is not None and len(result) > 0
            return {
                "healthy": healthy,
                "latency_ms": round(latency_ms, 2),
                "exchange": "okx",
            }
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "healthy": False,
                "latency_ms": round(latency_ms, 2),
                "exchange": "okx",
                "error": str(exc),
            }
