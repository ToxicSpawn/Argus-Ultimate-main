"""
Independent Reserve (Australian Exchange) REST Connector.

Production-grade connector for Independent Reserve — Australia's longest-running
digital currency exchange. Supports:
- Market data (ticker, orderbook, market summary)
- Authenticated trading (limit/market orders, cancellation)
- Account queries (balances, open orders, trade history)
- HMAC-SHA256 authentication with nonce
- Token-bucket rate limiting (max 1 req / 100ms)
- Automatic retry on 429 with exponential backoff

API docs: https://www.independentreserve.com/api
API base: https://api.independentreserve.com

API keys: Set INDEPENDENT_RESERVE_API_KEY / INDEPENDENT_RESERVE_API_SECRET
in environment variables.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IR_API_BASE = "https://api.independentreserve.com"

# Symbol mapping: standard → IR naming
_PRIMARY_CURRENCY_MAP: Dict[str, str] = {
    "BTC": "Xbt",
    "ETH": "Eth",
    "LTC": "Ltc",
    "XRP": "Xrp",
    "SOL": "Sol",
    "USDT": "Usdt",
    "USDC": "Usdc",
    "DOGE": "Doge",
    "ADA": "Ada",
    "DOT": "Dot",
    "LINK": "Link",
    "UNI": "Uni",
    "AVAX": "Avax",
    "MATIC": "Matic",
}

_SECONDARY_CURRENCY_MAP: Dict[str, str] = {
    "AUD": "Aud",
    "USD": "Usd",
    "NZD": "Nzd",
    "SGD": "Sgd",
}

# Reverse maps for parsing responses
_PRIMARY_REVERSE: Dict[str, str] = {v.lower(): k for k, v in _PRIMARY_CURRENCY_MAP.items()}
_SECONDARY_REVERSE: Dict[str, str] = {v.lower(): k for k, v in _SECONDARY_CURRENCY_MAP.items()}

# Rate limit: max 10 requests/second (100ms between requests)
_MAX_REQUESTS_PER_SECOND = 10

# Retry settings for 429
_MAX_RETRIES = 3
_RETRY_BASE_DELAY_S = 1.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class IndependentReserveError(Exception):
    """Base exception for Independent Reserve API errors."""

    def __init__(self, message: str, status_code: int = 0, endpoint: str = ""):
        self.status_code = status_code
        self.endpoint = endpoint
        super().__init__(f"IR API error (HTTP {status_code}): {message} [endpoint={endpoint}]")


class IndependentReserveAuthError(IndependentReserveError):
    """Raised on authentication failures (invalid key, expired nonce, etc.)."""
    pass


class IndependentReserveRateLimitError(IndependentReserveError):
    """Raised when rate limited (HTTP 429)."""
    pass


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class IRRateLimiter:
    """Token-bucket rate limiter — max 10 requests per second."""

    def __init__(self, max_per_second: int = _MAX_REQUESTS_PER_SECOND):
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


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class IndependentReserveConnector:
    """
    Independent Reserve REST API connector.

    Provides async methods for market data and authenticated trading.
    Uses aiohttp for HTTP transport and HMAC-SHA256 for request signing.
    """

    health_check_symbol = "BTC/AUD"

    # Supported trading pairs
    SUPPORTED_PAIRS: List[str] = [
        "BTC/AUD", "ETH/AUD", "LTC/AUD", "XRP/AUD", "SOL/AUD", "USDT/AUD",
    ]

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = IR_API_BASE,
    ):
        self.api_key = api_key or os.environ.get("INDEPENDENT_RESERVE_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("INDEPENDENT_RESERVE_API_SECRET", "")
        self.base_url = base_url.rstrip("/")
        self._session: Any = None  # aiohttp.ClientSession
        self._rate_limiter = IRRateLimiter()
        self._nonce: int = int(time.time() * 1000)
        self._nonce_lock = asyncio.Lock()
        self.connected: bool = False

    # ------------------------------------------------------------------
    # Symbol mapping
    # ------------------------------------------------------------------

    @staticmethod
    def parse_symbol(symbol: str) -> Tuple[str, str]:
        """
        Convert standard symbol to IR primary/secondary currency codes.

        "BTC/AUD" -> ("Xbt", "Aud")
        "ETH/AUD" -> ("Eth", "Aud")

        Raises ValueError if the symbol is not supported.
        """
        parts = symbol.upper().split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid symbol format '{symbol}', expected 'BASE/QUOTE'")

        base, quote = parts
        primary = _PRIMARY_CURRENCY_MAP.get(base)
        secondary = _SECONDARY_CURRENCY_MAP.get(quote)

        if primary is None:
            raise ValueError(f"Unsupported primary currency: {base}")
        if secondary is None:
            raise ValueError(f"Unsupported secondary currency: {quote}")

        return primary, secondary

    @staticmethod
    def to_standard_symbol(primary: str, secondary: str) -> str:
        """
        Convert IR currency codes back to standard symbol.

        ("Xbt", "Aud") -> "BTC/AUD"
        """
        base = _PRIMARY_REVERSE.get(primary.lower(), primary.upper())
        quote = _SECONDARY_REVERSE.get(secondary.lower(), secondary.upper())
        return f"{base}/{quote}"

    # ------------------------------------------------------------------
    # Nonce management
    # ------------------------------------------------------------------

    async def _next_nonce(self) -> int:
        """Get next incrementing nonce (thread-safe)."""
        async with self._nonce_lock:
            now_ms = int(time.time() * 1000)
            if now_ms > self._nonce:
                self._nonce = now_ms
            else:
                self._nonce += 1
            return self._nonce

    # ------------------------------------------------------------------
    # HMAC-SHA256 Authentication
    # ------------------------------------------------------------------

    def _sign(self, url: str, params: List[Tuple[str, str]]) -> str:
        """
        Create HMAC-SHA256 signature for a private API request.

        The message is constructed as:
            url,param1=value1,param2=value2,...
        where params are sorted alphabetically by key name.

        The signature is the HMAC-SHA256 hex digest using the API secret.
        """
        # Build message: URL followed by comma-separated key=value pairs
        # Parameters must include apiKey and nonce, sorted alphabetically
        parts = [url]
        for key, value in sorted(params, key=lambda x: x[0]):
            parts.append(f"{key}={value}")
        message = ",".join(parts)

        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return signature

    # ------------------------------------------------------------------
    # HTTP transport
    # ------------------------------------------------------------------

    async def _get_session(self) -> Any:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession()
            except ImportError:
                logger.error("aiohttp not available — cannot use IndependentReserveConnector")
                raise RuntimeError("aiohttp is required for IndependentReserveConnector")
        return self._session

    async def _public_get(
        self, endpoint: str, params: Optional[Dict[str, str]] = None
    ) -> Any:
        """
        Make a GET request to a public (unauthenticated) IR endpoint.

        Args:
            endpoint: API path, e.g. "/Public/GetMarketSummary"
            params: Query parameters

        Returns:
            Parsed JSON response
        """
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"

        await self._rate_limiter.acquire()

        for attempt in range(1, _MAX_RETRIES + 1):
            async with session.get(url, params=params, timeout=15) as resp:
                if resp.status == 429:
                    if attempt < _MAX_RETRIES:
                        delay = _RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
                        logger.warning(
                            "IR rate limited on %s, retrying in %.1fs (attempt %d/%d)",
                            endpoint, delay, attempt, _MAX_RETRIES,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise IndependentReserveRateLimitError(
                        "Rate limited after max retries", 429, endpoint
                    )

                if resp.status == 403:
                    text = await resp.text()
                    raise IndependentReserveAuthError(text, 403, endpoint)

                if resp.status >= 400:
                    text = await resp.text()
                    raise IndependentReserveError(text, resp.status, endpoint)

                return await resp.json()

        raise IndependentReserveError("Max retries exceeded", 0, endpoint)

    async def _private_post(
        self, endpoint: str, extra_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Make a POST request to a private (authenticated) IR endpoint.

        Authentication parameters (apiKey, nonce, signature) are added
        automatically.

        Args:
            endpoint: API path, e.g. "/Private/GetAccounts"
            extra_params: Additional request parameters

        Returns:
            Parsed JSON response
        """
        if not self.api_key or not self.api_secret:
            raise IndependentReserveAuthError(
                "API key and secret are required for private endpoints", 401, endpoint
            )

        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"
        nonce = await self._next_nonce()

        # Build parameter list for signing (alphabetical)
        sign_params: List[Tuple[str, str]] = [
            ("apiKey", self.api_key),
            ("nonce", str(nonce)),
        ]
        if extra_params:
            for k, v in extra_params.items():
                sign_params.append((k, str(v)))

        signature = self._sign(url, sign_params)

        # POST body includes all params + signature
        body: Dict[str, Any] = {
            "apiKey": self.api_key,
            "nonce": nonce,
            "signature": signature,
        }
        if extra_params:
            body.update(extra_params)

        await self._rate_limiter.acquire()

        for attempt in range(1, _MAX_RETRIES + 1):
            async with session.post(
                url, json=body, timeout=15,
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status == 429:
                    if attempt < _MAX_RETRIES:
                        delay = _RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
                        logger.warning(
                            "IR rate limited on %s, retrying in %.1fs (attempt %d/%d)",
                            endpoint, delay, attempt, _MAX_RETRIES,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise IndependentReserveRateLimitError(
                        "Rate limited after max retries", 429, endpoint
                    )

                if resp.status in (401, 403):
                    text = await resp.text()
                    raise IndependentReserveAuthError(text, resp.status, endpoint)

                if resp.status >= 400:
                    text = await resp.text()
                    raise IndependentReserveError(text, resp.status, endpoint)

                return await resp.json()

        raise IndependentReserveError("Max retries exceeded", 0, endpoint)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect and verify API access by pinging a public endpoint."""
        try:
            result = await self._public_get(
                "/Public/GetMarketSummary",
                {"primaryCurrencyCode": "xbt", "secondaryCurrencyCode": "aud"},
            )
            if result and "LastPrice" in result:
                self.connected = True
                logger.info("Independent Reserve connected (last BTC/AUD: %s)", result.get("LastPrice"))
                return True
            # Some IR responses might have different casing
            if result:
                self.connected = True
                logger.info("Independent Reserve connected")
                return True
            self.connected = False
            return False
        except Exception as exc:
            logger.error("Independent Reserve connect failed: %s", exc)
            self.connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect and clean up resources."""
        self.connected = False
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
        logger.info("Independent Reserve disconnected")

    async def __aenter__(self) -> "IndependentReserveConnector":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    async def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current ticker data for a trading pair.

        Args:
            symbol: Standard pair, e.g. "BTC/AUD"

        Returns:
            Standardized ticker dict: {bid, ask, last, volume_24h, timestamp, symbol}
        """
        try:
            primary, secondary = self.parse_symbol(symbol)
        except ValueError as exc:
            logger.warning("get_ticker invalid symbol %s: %s", symbol, exc)
            return None

        try:
            data = await self._public_get(
                "/Public/GetMarketSummary",
                {
                    "primaryCurrencyCode": primary.lower(),
                    "secondaryCurrencyCode": secondary.lower(),
                },
            )
            return {
                "symbol": symbol,
                "bid": float(data.get("CurrentHighestBidPrice", 0) or 0),
                "ask": float(data.get("CurrentLowestOfferPrice", 0) or 0),
                "last": float(data.get("LastPrice", 0) or 0),
                "volume_24h": float(data.get("DayVolumeXbt", 0) or 0),
                "high_24h": float(data.get("DayHighestPrice", 0) or 0),
                "low_24h": float(data.get("DayLowestPrice", 0) or 0),
                "timestamp": data.get("CreatedTimestampUtc"),
                "exchange": "independent_reserve",
            }
        except Exception as exc:
            logger.warning("IR get_ticker %s failed: %s", symbol, exc)
            return None

    async def get_orderbook(
        self, symbol: str, depth: int = 50
    ) -> Optional[Dict[str, Any]]:
        """
        Get order book for a trading pair.

        Args:
            symbol: Standard pair, e.g. "BTC/AUD"
            depth: Max number of levels per side (IR returns all; we truncate)

        Returns:
            {bids: [(price, qty), ...], asks: [(price, qty), ...], symbol, timestamp}
        """
        try:
            primary, secondary = self.parse_symbol(symbol)
        except ValueError as exc:
            logger.warning("get_orderbook invalid symbol %s: %s", symbol, exc)
            return None

        try:
            data = await self._public_get(
                "/Public/GetOrderBook",
                {
                    "primaryCurrencyCode": primary.lower(),
                    "secondaryCurrencyCode": secondary.lower(),
                },
            )

            bids = [
                (float(o["Price"]), float(o["Volume"]))
                for o in (data.get("BuyOrders") or [])[:depth]
            ]
            asks = [
                (float(o["Price"]), float(o["Volume"]))
                for o in (data.get("SellOrders") or [])[:depth]
            ]

            return {
                "symbol": symbol,
                "bids": bids,
                "asks": asks,
                "timestamp": data.get("CreatedTimestampUtc"),
                "exchange": "independent_reserve",
            }
        except Exception as exc:
            logger.warning("IR get_orderbook %s failed: %s", symbol, exc)
            return None

    async def get_market_summary(self) -> List[Dict[str, Any]]:
        """
        Get 24h market summary for all supported trading pairs.

        Returns:
            List of ticker dicts for each supported pair.
        """
        summaries: List[Dict[str, Any]] = []
        for pair in self.SUPPORTED_PAIRS:
            ticker = await self.get_ticker(pair)
            if ticker is not None:
                summaries.append(ticker)
        return summaries

    # ------------------------------------------------------------------
    # Private endpoints — Account
    # ------------------------------------------------------------------

    async def get_balances(self) -> Dict[str, Dict[str, float]]:
        """
        Get account balances for all currencies.

        Returns:
            {currency: {available: float, total: float}}
            e.g. {"BTC": {"available": 0.5, "total": 1.0}, "AUD": {...}}
        """
        data = await self._private_post("/Private/GetAccounts")

        balances: Dict[str, Dict[str, float]] = {}
        for acct in data if isinstance(data, list) else []:
            code = acct.get("CurrencyCode", "")
            # Map IR code back to standard
            standard = _PRIMARY_REVERSE.get(code.lower(), code.upper())
            if standard == code.upper():
                standard = _SECONDARY_REVERSE.get(code.lower(), code.upper())

            available = float(acct.get("AvailableBalance", 0) or 0)
            total = float(acct.get("TotalBalance", 0) or 0)
            balances[standard] = {"available": available, "total": total}

        return balances

    # Alias for ExchangeManager compatibility
    async def get_balance(self) -> Dict[str, Dict[str, float]]:
        """Alias for get_balances() — ExchangeManager compatibility."""
        return await self.get_balances()

    # ------------------------------------------------------------------
    # Private endpoints — Trading
    # ------------------------------------------------------------------

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str = "limit",
        quantity: float = 0.0,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Place an order on Independent Reserve.

        Args:
            symbol: Standard pair, e.g. "BTC/AUD"
            side: "buy" or "sell" (case-insensitive)
            order_type: "limit" or "market" (case-insensitive)
            quantity: Amount of primary currency
            price: Required for limit orders

        Returns:
            Order dict with OrderGuid, Status, etc.

        Raises:
            ValueError: If parameters are invalid
            IndependentReserveError: On API errors
        """
        primary, secondary = self.parse_symbol(symbol)
        side_lower = side.lower()
        type_lower = order_type.lower()

        if side_lower not in ("buy", "sell"):
            raise ValueError(f"Invalid side: {side}, expected 'buy' or 'sell'")
        if type_lower not in ("limit", "market"):
            raise ValueError(f"Invalid order_type: {order_type}, expected 'limit' or 'market'")
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")

        if type_lower == "limit":
            if price is None or price <= 0:
                raise ValueError("Price is required and must be positive for limit orders")

            # IR order type: "LimitBid" or "LimitOffer"
            ir_order_type = "LimitBid" if side_lower == "buy" else "LimitOffer"
            endpoint = "/Private/PlaceLimitOrder"
            params: Dict[str, Any] = {
                "primaryCurrencyCode": primary.lower(),
                "secondaryCurrencyCode": secondary.lower(),
                "orderType": ir_order_type,
                "price": price,
                "volume": quantity,
            }
        else:
            # Market order
            ir_order_type = "MarketBid" if side_lower == "buy" else "MarketOffer"
            endpoint = "/Private/PlaceMarketOrder"
            params = {
                "primaryCurrencyCode": primary.lower(),
                "secondaryCurrencyCode": secondary.lower(),
                "orderType": ir_order_type,
                "volume": quantity,
            }

        result = await self._private_post(endpoint, params)

        logger.info(
            "IR order placed: %s %s %.8f %s @ %s (OrderGuid=%s)",
            side_lower, symbol, quantity, type_lower,
            price if price else "market",
            result.get("OrderGuid", "?"),
        )

        return {
            "order_id": result.get("OrderGuid", ""),
            "symbol": symbol,
            "side": side_lower,
            "order_type": type_lower,
            "quantity": quantity,
            "price": price,
            "status": result.get("Status", ""),
            "timestamp": result.get("CreatedTimestampUtc", ""),
            "exchange": "independent_reserve",
            "raw": result,
        }

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: The OrderGuid to cancel

        Returns:
            True if cancellation was accepted
        """
        try:
            await self._private_post(
                "/Private/CancelOrder",
                {"orderGuid": order_id},
            )
            logger.info("IR order cancelled: %s", order_id)
            return True
        except IndependentReserveError as exc:
            logger.error("IR cancel_order %s failed: %s", order_id, exc)
            raise

    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get details for a specific order.

        Args:
            order_id: The OrderGuid

        Returns:
            Order status dict
        """
        data = await self._private_post(
            "/Private/GetOrderDetails",
            {"orderGuid": order_id},
        )

        return {
            "order_id": data.get("OrderGuid", ""),
            "status": data.get("Status", ""),
            "order_type": data.get("OrderType", ""),
            "volume": float(data.get("Volume", 0) or 0),
            "outstanding": float(data.get("Outstanding", 0) or 0),
            "price": float(data.get("Price", 0) or 0),
            "avg_price": float(data.get("AvgPrice", 0) or 0),
            "timestamp": data.get("CreatedTimestampUtc", ""),
            "exchange": "independent_reserve",
            "raw": data,
        }

    async def get_open_orders(
        self,
        symbol: Optional[str] = None,
        page_index: int = 1,
        page_size: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get all open orders, optionally filtered by symbol.

        Args:
            symbol: Optional filter by trading pair
            page_index: Page number (1-based)
            page_size: Results per page

        Returns:
            List of open order dicts
        """
        params: Dict[str, Any] = {
            "pageIndex": page_index,
            "pageSize": page_size,
        }
        if symbol:
            primary, secondary = self.parse_symbol(symbol)
            params["primaryCurrencyCode"] = primary.lower()
            params["secondaryCurrencyCode"] = secondary.lower()

        data = await self._private_post("/Private/GetOpenOrders", params)

        orders: List[Dict[str, Any]] = []
        for item in data.get("Data", []) if isinstance(data, dict) else []:
            orders.append({
                "order_id": item.get("OrderGuid", ""),
                "status": item.get("Status", ""),
                "order_type": item.get("OrderType", ""),
                "volume": float(item.get("Volume", 0) or 0),
                "outstanding": float(item.get("Outstanding", 0) or 0),
                "price": float(item.get("Price", 0) or 0),
                "timestamp": item.get("CreatedTimestampUtc", ""),
                "exchange": "independent_reserve",
            })

        return orders

    async def get_trade_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 50,
        page_index: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Get trade (fill) history.

        Args:
            symbol: Optional filter by trading pair
            limit: Max trades to return
            page_index: Page number (1-based)

        Returns:
            List of trade/fill dicts
        """
        params: Dict[str, Any] = {
            "pageIndex": page_index,
            "pageSize": limit,
        }
        if symbol:
            primary, secondary = self.parse_symbol(symbol)
            params["primaryCurrencyCode"] = primary.lower()
            params["secondaryCurrencyCode"] = secondary.lower()

        data = await self._private_post("/Private/GetTrades", params)

        trades: List[Dict[str, Any]] = []
        for item in data.get("Data", []) if isinstance(data, dict) else []:
            trades.append({
                "trade_id": item.get("TradeGuid", ""),
                "order_id": item.get("OrderGuid", ""),
                "side": "buy" if item.get("Taker", "") == "Bid" else "sell",
                "quantity": float(item.get("PrimaryCurrencyAmount", 0) or 0),
                "price": float(item.get("SecondaryCurrencyTradePrice", 0) or 0),
                "fee": float(item.get("BrokeFee", 0) or 0),
                "timestamp": item.get("TradeTimestampUtc", ""),
                "exchange": "independent_reserve",
            })

        return trades

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """
        Ping a public endpoint to verify connectivity.

        Returns:
            {"healthy": bool, "latency_ms": float, "exchange": str}
        """
        t0 = time.perf_counter()
        try:
            data = await self._public_get(
                "/Public/GetMarketSummary",
                {"primaryCurrencyCode": "xbt", "secondaryCurrencyCode": "aud"},
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            healthy = data is not None and "LastPrice" in (data or {})
            return {
                "healthy": healthy,
                "latency_ms": round(latency_ms, 2),
                "exchange": "independent_reserve",
                "last_price": float(data.get("LastPrice", 0) or 0) if data else 0.0,
            }
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "healthy": False,
                "latency_ms": round(latency_ms, 2),
                "exchange": "independent_reserve",
                "error": str(exc),
            }
