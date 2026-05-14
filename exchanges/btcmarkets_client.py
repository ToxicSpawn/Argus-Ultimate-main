"""
BTC Markets Exchange Client
============================

Production-grade REST + WebSocket client for BTC Markets — Australia's
AUSTRAC-registered digital currency exchange.

Key characteristics:
- AUD-quoted markets (BTC-AUD, ETH-AUD, SOL-AUD, …)
- Maker fee: -0.05%  (NEGATIVE — exchange pays a rebate per maker fill)
- Taker fee:  0.20%
- REST base:  https://api.btcmarkets.net/v3
- WebSocket:  wss://socket.btcmarkets.net/v2
- Auth:       BTC-APIKEY + BTC-TIMESTAMP + BTC-SIGNATURE headers
  Signature = base64(HMAC-SHA256(base64decode(api_secret), method+path+ts+body))
- Rate limit: 500 req / 10 min  →  conservative bucket at 45 req/min

Usage::

    client = BTCMarketsClient(api_key="...", api_secret="...")
    ticker = await client.fetch_ticker("BTC/AUD")
    print(ticker)   # {"bid": ..., "ask": ..., "last": ..., "volume": ..., "timestamp": ...}

Author: Argus Trading System
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
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

BTCM_MAKER_FEE: float = -0.0005   # Negative: exchange pays maker rebate
BTCM_TAKER_FEE: float = 0.002     # 0.20% taker fee

BTCM_BASE_URL: str = "https://api.btcmarkets.net/v3"
BTCM_WS_URL:   str = "wss://socket.btcmarkets.net/v2"

BTCM_MARKETS: List[str] = [
    "BTC-AUD",
    "ETH-AUD",
    "SOL-AUD",
    "XRP-AUD",
    "LTC-AUD",
    "USDT-AUD",
    "DOGE-AUD",
    "LINK-AUD",
]

# Conservative rate limit: 45 req/min < 500 req/10 min
_RATE_LIMIT_PER_MIN: int = 45

# HTTP retry settings
_MAX_RETRIES: int = 3
_RETRY_BASE_DELAY_S: float = 1.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BTCMarketsAPIError(Exception):
    """Raised when the BTC Markets API returns an error."""

    def __init__(self, code: str, message: str, status_code: int = 0) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(f"BTCMarkets API error [{code}] HTTP {status_code}: {message}")


class BTCMarketsAuthError(BTCMarketsAPIError):
    """Authentication failure (invalid key / bad signature)."""
    pass


class BTCMarketsRateLimitError(BTCMarketsAPIError):
    """Rate limit exceeded (HTTP 429)."""
    pass


# ---------------------------------------------------------------------------
# Token-bucket rate limiter (45 req/min)
# ---------------------------------------------------------------------------

class _BTCMRateLimiter:
    """Token-bucket rate limiter.

    Allows up to *max_per_min* requests per minute, enforced as a sliding
    token bucket.  Callers ``await acquire()`` before every HTTP request.
    """

    def __init__(self, max_per_min: int = _RATE_LIMIT_PER_MIN) -> None:
        self._max: int = max_per_min
        self._tokens: float = float(max_per_min)
        self._last_refill: float = time.monotonic()
        self._lock: asyncio.Lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            # Refill tokens at rate of max_per_min per 60 seconds
            self._tokens = min(
                float(self._max),
                self._tokens + elapsed * (self._max / 60.0),
            )
            self._last_refill = now
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / (self._max / 60.0)
                logger.debug("BTCMarkets rate limiter: waiting %.3fs", wait)
                await asyncio.sleep(wait)
                self._tokens = 0.0
                self._last_refill = time.monotonic()
            else:
                self._tokens -= 1.0


# ---------------------------------------------------------------------------
# Symbol normalisation helpers
# ---------------------------------------------------------------------------

def to_btcm_symbol(symbol: str) -> str:
    """Convert a symbol to BTC Markets dash-separated format.

    Examples:
        "BTC/AUD"  →  "BTC-AUD"
        "BTCAUD"   →  "BTC-AUD"
        "BTC-AUD"  →  "BTC-AUD"  (no-op)
    """
    symbol = symbol.upper().strip()
    # Already in correct format
    if "-" in symbol:
        return symbol
    # Slash-separated  "BTC/AUD"
    if "/" in symbol:
        return symbol.replace("/", "-")
    # Concatenated — try to identify known quote suffixes
    for quote in ("USDT", "AUD", "USD", "BTC", "ETH"):
        if symbol.endswith(quote) and len(symbol) > len(quote):
            base = symbol[: len(symbol) - len(quote)]
            return f"{base}-{quote}"
    # Fallback: assume 3-letter base + 3-letter quote
    if len(symbol) == 6:
        return f"{symbol[:3]}-{symbol[3:]}"
    return symbol


def from_btcm_symbol(symbol: str) -> str:
    """Convert a BTC Markets dash-separated symbol to slash-separated.

    Example:  "BTC-AUD"  →  "BTC/AUD"
    """
    return symbol.replace("-", "/")


# ---------------------------------------------------------------------------
# Exchange info helper
# ---------------------------------------------------------------------------

def get_exchange_info() -> Dict[str, Any]:
    """Return static BTC Markets exchange information."""
    return {
        "name": "btcmarkets",
        "display_name": "BTC Markets",
        "country": "AU",
        "maker_fee": BTCM_MAKER_FEE,
        "taker_fee": BTCM_TAKER_FEE,
        "maker_rebate_note": (
            "Maker fee is NEGATIVE: exchange pays you "
            f"{abs(BTCM_MAKER_FEE) * 100:.3f}% per maker fill"
        ),
        "markets": BTCM_MARKETS,
        "rest_url": BTCM_BASE_URL,
        "ws_url": BTCM_WS_URL,
        "rate_limit_req_per_10min": 500,
    }


# ---------------------------------------------------------------------------
# Main client class
# ---------------------------------------------------------------------------

class BTCMarketsClient:
    """BTC Markets REST + WebSocket client.

    Provides:
    - Full REST API for market data and authenticated trading
    - WebSocket subscriptions for live orderbook, trades, and user data
    - Maker rebate tracking across the session
    - AUD-to-USD normalisation for USD-denominated PnL metrics

    Args:
        api_key:      BTC Markets API key (or set BTCM_API_KEY env var)
        api_secret:   BTC Markets API secret (base-64 encoded, or BTCM_API_SECRET)
        aud_usd_rate: Approximate AUD/USD exchange rate for USD normalisation
        base_url:     Override REST base URL (e.g. for testing)
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        aud_usd_rate: float = 0.62,
        base_url: str = BTCM_BASE_URL,
    ) -> None:
        self.api_key: str = api_key or os.environ.get("BTCM_API_KEY", "")
        self.api_secret: str = api_secret or os.environ.get("BTCM_API_SECRET", "")
        self.aud_usd_rate: float = aud_usd_rate
        self.base_url: str = base_url.rstrip("/")

        self._session: Any = None  # aiohttp.ClientSession (lazy init)
        self._rate_limiter: _BTCMRateLimiter = _BTCMRateLimiter(_RATE_LIMIT_PER_MIN)

        # WebSocket state
        self._ws: Any = None
        self._ws_running: bool = False
        self._ws_tasks: List[asyncio.Task] = []  # type: ignore[type-arg]

        # Rebate tracking
        self._session_maker_volume_aud: float = 0.0  # AUD notional traded as maker

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _sign(self, method: str, path: str, timestamp: str, body: str = "") -> str:
        """Compute the HMAC-SHA256 BTC Markets request signature.

        The message is:  method + path + timestamp + body
        The key is:      base64decode(api_secret)
        The signature is: base64encode(HMAC-SHA256(key, message.encode()))

        Args:
            method:    HTTP method in uppercase ("GET", "POST", …)
            path:      API path including query string (e.g. "/v3/markets/BTC-AUD/ticker")
            timestamp: Unix epoch milliseconds as a string
            body:      JSON request body string (empty string for GET)

        Returns:
            Base-64 encoded signature string.
        """
        message = method + path + timestamp + body
        secret_bytes = base64.b64decode(self.api_secret)
        signature = hmac.new(
            secret_bytes,
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(signature).decode("utf-8")

    def _auth_headers(
        self, method: str, path: str, body: str = ""
    ) -> Dict[str, str]:
        """Build authentication headers for a private request."""
        timestamp = str(int(time.time() * 1000))
        signature = self._sign(method, path, timestamp, body)
        return {
            "BTC-APIKEY": self.api_key,
            "BTC-TIMESTAMP": timestamp,
            "BTC-SIGNATURE": signature,
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # HTTP transport
    # ------------------------------------------------------------------

    async def _get_session(self) -> Any:
        """Lazily create and return the aiohttp ClientSession."""
        if self._session is None or self._session.closed:
            try:
                import aiohttp  # type: ignore[import]
                self._session = aiohttp.ClientSession(
                    headers={"Accept": "application/json"},
                    connector=aiohttp.TCPConnector(limit=50, ssl=True),
                )
            except ImportError as exc:
                raise RuntimeError("aiohttp is required: pip install aiohttp") from exc
        return self._session

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        authenticated: bool = False,
    ) -> Any:
        """Make an HTTP request to the BTC Markets REST API.

        Args:
            method:        HTTP method ("GET", "POST", "DELETE")
            path:          API path, e.g. "/v3/markets/BTC-AUD/ticker"
            params:        Query-string parameters
            body:          JSON body (for POST/PUT)
            authenticated: If True, include auth headers

        Returns:
            Parsed JSON response (dict or list)

        Raises:
            BTCMarketsAuthError:       On 401/403
            BTCMarketsRateLimitError:  On 429
            BTCMarketsAPIError:        On other HTTP errors
        """
        session = await self._get_session()
        url = self.base_url + path
        body_str = json.dumps(body) if body else ""

        headers: Dict[str, str] = {"Accept": "application/json"}
        if authenticated:
            if not self.api_key or not self.api_secret:
                raise BTCMarketsAuthError(
                    "AUTH_REQUIRED",
                    "API key and secret required for private endpoints",
                    401,
                )
            headers.update(self._auth_headers(method, path, body_str))

        await self._rate_limiter.acquire()

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                request_kwargs: Dict[str, Any] = {
                    "headers": headers,
                    "params": params or {},
                }
                if body_str:
                    request_kwargs["data"] = body_str
                    headers["Content-Type"] = "application/json"

                async with getattr(session, method.lower())(
                    url, **request_kwargs
                ) as resp:
                    status = resp.status

                    if status == 429:
                        if attempt < _MAX_RETRIES:
                            delay = _RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
                            logger.warning(
                                "BTCMarkets rate-limited on %s %s, retry in %.1fs (%d/%d)",
                                method, path, delay, attempt, _MAX_RETRIES,
                            )
                            await asyncio.sleep(delay)
                            continue
                        raise BTCMarketsRateLimitError(
                            "RATE_LIMIT", "Rate limited after max retries", 429
                        )

                    if status in (401, 403):
                        text = await resp.text()
                        raise BTCMarketsAuthError("AUTH_FAILED", text, status)

                    if status >= 400:
                        try:
                            err_data = await resp.json()
                            code = err_data.get("code", str(status))
                            msg = err_data.get("message", str(err_data))
                        except Exception:
                            msg = await resp.text()
                            code = str(status)
                        raise BTCMarketsAPIError(code, msg, status)

                    if status == 204:
                        return {}

                    return await resp.json()

            except BTCMarketsAPIError:
                raise
            except Exception as exc:
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_BASE_DELAY_S * attempt
                    logger.warning(
                        "BTCMarkets request error on attempt %d/%d: %s",
                        attempt, _MAX_RETRIES, exc,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise BTCMarketsAPIError("NETWORK_ERROR", str(exc)) from exc

        raise BTCMarketsAPIError("NETWORK_ERROR", "Max retries exceeded")

    # ------------------------------------------------------------------
    # Public REST methods
    # ------------------------------------------------------------------

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch current best bid/ask, last price, and 24h volume.

        Args:
            symbol: Trading pair in any format ("BTC/AUD", "BTCAUD", "BTC-AUD")

        Returns:
            Normalised ticker dict::

                {
                    "symbol":    "BTC/AUD",
                    "bid":       float,
                    "ask":       float,
                    "last":      float,
                    "volume":    float,   # 24h base volume
                    "timestamp": str,     # ISO 8601
                    "exchange":  "btcmarkets",
                }
        """
        btcm_sym = to_btcm_symbol(symbol)
        data = await self._request("GET", f"/v3/markets/{btcm_sym}/ticker")
        return {
            "symbol":    from_btcm_symbol(btcm_sym),
            "bid":       float(data.get("bestBid", 0) or 0),
            "ask":       float(data.get("bestAsk", 0) or 0),
            "last":      float(data.get("lastPrice", 0) or 0),
            "volume":    float(data.get("volume24h", 0) or 0),
            "timestamp": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "exchange":  "btcmarkets",
        }

    async def fetch_order_book(
        self, symbol: str, limit: int = 20
    ) -> Dict[str, Any]:
        """Fetch current order book snapshot.

        Args:
            symbol: Trading pair
            limit:  Number of price levels per side (default 20)

        Returns:
            Dict with bids (sorted descending) and asks (sorted ascending)::

                {
                    "symbol":    "BTC/AUD",
                    "bids":      [[price, qty], ...],   # highest first
                    "asks":      [[price, qty], ...],   # lowest first
                    "timestamp": str,
                    "exchange":  "btcmarkets",
                }
        """
        btcm_sym = to_btcm_symbol(symbol)
        # BTC Markets supports bestOnly or depth param
        params: Dict[str, Any] = {}
        if limit:
            params["depth"] = limit
        data = await self._request(
            "GET", f"/v3/markets/{btcm_sym}/orderbook", params=params
        )

        raw_bids = data.get("bids", [])
        raw_asks = data.get("asks", [])

        # Normalise to [[float, float], ...] and sort
        bids = sorted(
            [[float(b[0]), float(b[1])] for b in raw_bids],
            key=lambda x: -x[0],
        )[:limit]
        asks = sorted(
            [[float(a[0]), float(a[1])] for a in raw_asks],
            key=lambda x: x[0],
        )[:limit]

        return {
            "symbol":    from_btcm_symbol(btcm_sym),
            "bids":      bids,
            "asks":      asks,
            "timestamp": data.get("snapshotId", datetime.now(timezone.utc).isoformat()),
            "exchange":  "btcmarkets",
        }

    async def fetch_trades(
        self, symbol: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Fetch recent public trades.

        Args:
            symbol: Trading pair
            limit:  Max number of trades to return

        Returns:
            List of trade dicts, newest first::

                [{
                    "trade_id": str,
                    "price":    float,
                    "quantity": float,
                    "side":     "buy" | "sell",
                    "timestamp": str,
                }, ...]
        """
        btcm_sym = to_btcm_symbol(symbol)
        data = await self._request(
            "GET",
            f"/v3/markets/{btcm_sym}/trades",
            params={"limit": limit},
        )

        trades: List[Dict[str, Any]] = []
        for t in data if isinstance(data, list) else []:
            trades.append({
                "trade_id":  str(t.get("id", "")),
                "price":     float(t.get("price", 0) or 0),
                "quantity":  float(t.get("amount", 0) or 0),
                "side":      "buy" if t.get("side", "").lower() == "bid" else "sell",
                "timestamp": t.get("timestamp", ""),
                "exchange":  "btcmarkets",
            })
        return trades

    async def fetch_markets(self) -> List[Dict[str, Any]]:
        """Fetch all available trading pairs with min order sizes.

        Returns:
            List of market info dicts::

                [{
                    "symbol":         "BTC/AUD",
                    "btcm_symbol":    "BTC-AUD",
                    "base":           "BTC",
                    "quote":          "AUD",
                    "min_order_size": float,
                    "price_decimals": int,
                    "qty_decimals":   int,
                    "status":         str,
                }, ...]
        """
        data = await self._request("GET", "/v3/markets")
        markets: List[Dict[str, Any]] = []
        for m in data if isinstance(data, list) else []:
            btcm_sym = m.get("marketId", "")
            parts = btcm_sym.split("-")
            base  = parts[0] if len(parts) >= 1 else ""
            quote = parts[1] if len(parts) >= 2 else ""
            markets.append({
                "symbol":         from_btcm_symbol(btcm_sym),
                "btcm_symbol":    btcm_sym,
                "base":           base,
                "quote":          quote,
                "min_order_size": float(m.get("minOrderSize", 0) or 0),
                "price_decimals": int(m.get("priceDecimals", 2)),
                "qty_decimals":   int(m.get("amountDecimals", 8)),
                "status":         m.get("status", ""),
                "exchange":       "btcmarkets",
            })
        return markets

    # ------------------------------------------------------------------
    # Authenticated REST methods
    # ------------------------------------------------------------------

    async def fetch_balance(self) -> Dict[str, Any]:
        """Fetch account balances for all currencies.

        Returns:
            Dict mapping currency code to balance info::

                {
                    "BTC": {"available": float, "total": float},
                    "AUD": {"available": float, "total": float},
                    ...
                }
        """
        data = await self._request("GET", "/v3/accounts/me/balances", authenticated=True)
        balances: Dict[str, Any] = {}
        for item in data if isinstance(data, list) else []:
            currency = item.get("assetName", "")
            if currency:
                balances[currency] = {
                    "available": float(item.get("available", 0) or 0),
                    "total":     float(item.get("balance", 0) or 0),
                    "locked":    float(item.get("locked", 0) or 0),
                }
        return balances

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        post_only: bool = True,
    ) -> Dict[str, Any]:
        """Place an order on BTC Markets.

        Args:
            symbol:     Trading pair ("BTC/AUD", "BTC-AUD", …)
            side:       "buy" or "sell" (case-insensitive)
            order_type: "Limit" or "Market" (case-insensitive)
            quantity:   Order size in base currency
            price:      Limit price (required for Limit orders)
            post_only:  If True, set timeInForce="GTC" and postOnly=True
                        to guarantee maker execution and earn the rebate

        Returns:
            Order placement result dict::

                {
                    "order_id":  str,
                    "symbol":    "BTC/AUD",
                    "side":      "buy" | "sell",
                    "type":      "Limit" | "Market",
                    "quantity":  float,
                    "price":     float | None,
                    "status":    str,
                    "post_only": bool,
                    "timestamp": str,
                    "exchange":  "btcmarkets",
                }
        """
        btcm_sym  = to_btcm_symbol(symbol)
        side_str  = side.lower()
        type_str  = order_type.capitalize()  # "Limit" or "Market"

        if type_str not in ("Limit", "Market"):
            raise ValueError(f"Invalid order_type: {order_type}. Use 'Limit' or 'Market'.")
        if side_str not in ("buy", "sell"):
            raise ValueError(f"Invalid side: {side}. Use 'buy' or 'sell'.")
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")
        if type_str == "Limit" and (price is None or price <= 0):
            raise ValueError("Price is required and must be positive for Limit orders")

        body: Dict[str, Any] = {
            "marketId": btcm_sym,
            "side":     "Bid" if side_str == "buy" else "Ask",
            "type":     type_str,
            "amount":   str(quantity),
        }

        if type_str == "Limit":
            body["price"] = str(price)
            if post_only:
                body["timeInForce"] = "GTC"
                body["postOnly"]    = True

        result = await self._request(
            "POST", "/v3/orders", body=body, authenticated=True
        )

        logger.info(
            "BTCMarkets order placed: %s %s %s qty=%.8f price=%s id=%s",
            type_str, side_str, btcm_sym, quantity,
            price if price else "market",
            result.get("orderId", "?"),
        )

        return {
            "order_id":  str(result.get("orderId", "")),
            "symbol":    from_btcm_symbol(btcm_sym),
            "side":      side_str,
            "type":      type_str,
            "quantity":  quantity,
            "price":     price,
            "status":    result.get("status", ""),
            "post_only": post_only and type_str == "Limit",
            "timestamp": result.get("creationTime", ""),
            "exchange":  "btcmarkets",
            "raw":       result,
        }

    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancel an open order by ID.

        Args:
            symbol:   Trading pair (not used by the BTC Markets API but
                      included for interface consistency)
            order_id: BTC Markets order ID

        Returns:
            Dict with cancellation confirmation or error details
        """
        try:
            result = await self._request(
                "DELETE", f"/v3/orders/{order_id}", authenticated=True
            )
            logger.info("BTCMarkets order cancelled: %s", order_id)
            return {
                "order_id": order_id,
                "symbol":   symbol,
                "status":   result.get("status", "Cancelled"),
                "exchange": "btcmarkets",
                "raw":      result,
            }
        except BTCMarketsAPIError as exc:
            logger.error("BTCMarkets cancel_order %s failed: %s", order_id, exc)
            raise

    async def fetch_open_orders(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch all open orders, optionally filtered by symbol.

        Args:
            symbol: Trading pair filter (optional)

        Returns:
            List of open order dicts
        """
        params: Dict[str, Any] = {"status": "open"}
        if symbol:
            params["marketId"] = to_btcm_symbol(symbol)

        data = await self._request(
            "GET", "/v3/orders", params=params, authenticated=True
        )

        orders: List[Dict[str, Any]] = []
        for item in data if isinstance(data, list) else []:
            btcm_sym = item.get("marketId", "")
            orders.append({
                "order_id":   str(item.get("orderId", "")),
                "symbol":     from_btcm_symbol(btcm_sym),
                "side":       "buy" if item.get("side", "") == "Bid" else "sell",
                "type":       item.get("type", ""),
                "quantity":   float(item.get("amount", 0) or 0),
                "filled":     float(item.get("openAmount", 0) or 0),
                "price":      float(item.get("price", 0) or 0),
                "status":     item.get("status", ""),
                "post_only":  item.get("postOnly", False),
                "timestamp":  item.get("creationTime", ""),
                "exchange":   "btcmarkets",
            })
        return orders

    async def fetch_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Fetch details of a specific order.

        Args:
            symbol:   Trading pair (for interface consistency)
            order_id: BTC Markets order ID

        Returns:
            Order detail dict
        """
        data = await self._request(
            "GET", f"/v3/orders/{order_id}", authenticated=True
        )
        btcm_sym = data.get("marketId", "")
        return {
            "order_id":   str(data.get("orderId", "")),
            "symbol":     from_btcm_symbol(btcm_sym) or symbol,
            "side":       "buy" if data.get("side", "") == "Bid" else "sell",
            "type":       data.get("type", ""),
            "quantity":   float(data.get("amount", 0) or 0),
            "filled":     float(data.get("openAmount", 0) or 0),
            "price":      float(data.get("price", 0) or 0),
            "avg_price":  float(data.get("avgPrice", 0) or 0),
            "status":     data.get("status", ""),
            "post_only":  data.get("postOnly", False),
            "timestamp":  data.get("creationTime", ""),
            "exchange":   "btcmarkets",
            "raw":        data,
        }

    # ------------------------------------------------------------------
    # Rebate tracking
    # ------------------------------------------------------------------

    def record_maker_fill(self, fill_quantity: float, fill_price_aud: float) -> float:
        """Record a confirmed maker fill and accumulate session rebate.

        Args:
            fill_quantity:  Quantity filled (base currency units)
            fill_price_aud: Fill price in AUD

        Returns:
            Rebate earned on this fill (AUD), which is positive because
            BTCM_MAKER_FEE is negative (exchange pays us).
        """
        notional_aud = fill_quantity * fill_price_aud
        # Rebate is abs(BTCM_MAKER_FEE) * notional
        rebate_aud = abs(BTCM_MAKER_FEE) * notional_aud
        self._session_maker_volume_aud += notional_aud
        logger.debug(
            "BTCMarkets maker fill recorded: qty=%.8f @ %.2f AUD, rebate=%.4f AUD",
            fill_quantity, fill_price_aud, rebate_aud,
        )
        return rebate_aud

    def get_rebate_earned_session(self) -> float:
        """Return cumulative maker rebate earned this session (USD equivalent).

        Uses the AUD/USD rate supplied at construction to convert the
        AUD-denominated rebate to USD.

        Returns:
            Cumulative rebate in USD (positive number — this is income)
        """
        rebate_aud = abs(BTCM_MAKER_FEE) * self._session_maker_volume_aud
        return rebate_aud * self.aud_usd_rate

    # ------------------------------------------------------------------
    # WebSocket subscriptions
    # ------------------------------------------------------------------

    async def subscribe_order_book(
        self, symbol: str, callback: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """Subscribe to live orderbook updates for a symbol.

        Connects to BTC Markets WebSocket v2 and calls *callback* on each
        orderbook message.  Runs until the task is cancelled.

        Args:
            symbol:   Trading pair ("BTC/AUD", "BTC-AUD", …)
            callback: Async or sync callable receiving normalised orderbook dict
        """
        btcm_sym = to_btcm_symbol(symbol)
        await self._ws_subscribe(
            market_ids=[btcm_sym],
            channels=["orderbook"],
            callback=callback,
            msg_type="orderbook",
        )

    async def subscribe_trades(
        self, symbol: str, callback: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """Subscribe to live trade feed for a symbol.

        Args:
            symbol:   Trading pair
            callback: Callable receiving normalised trade dict
        """
        btcm_sym = to_btcm_symbol(symbol)
        await self._ws_subscribe(
            market_ids=[btcm_sym],
            channels=["trade"],
            callback=callback,
            msg_type="trade",
        )

    async def subscribe_user_data(
        self, callback: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """Subscribe to private user data channel (order fills, account updates).

        Requires valid API credentials.

        Args:
            callback: Callable receiving normalised user-data event dict
        """
        if not self.api_key or not self.api_secret:
            raise BTCMarketsAuthError(
                "AUTH_REQUIRED",
                "API key and secret required for user data subscription",
                401,
            )
        await self._ws_subscribe(
            market_ids=[],
            channels=["fundChange", "orderChange"],
            callback=callback,
            msg_type="userdata",
            authenticated=True,
        )

    async def _ws_subscribe(
        self,
        market_ids: List[str],
        channels: List[str],
        callback: Callable[[Dict[str, Any]], Any],
        msg_type: str,
        authenticated: bool = False,
    ) -> None:
        """Internal WebSocket subscription handler with auto-reconnect.

        Subscribes to the given channels, dispatches messages to *callback*,
        and reconnects automatically on disconnect.
        """
        try:
            import websockets  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "websockets is required for WS subscriptions: pip install websockets"
            ) from exc

        backoff = 1.0
        max_backoff = 30.0

        while True:
            try:
                logger.info(
                    "BTCMarkets WS connecting for channels=%s markets=%s",
                    channels, market_ids,
                )
                async with websockets.connect(BTCM_WS_URL, ping_interval=20) as ws:
                    backoff = 1.0  # reset on successful connect

                    # Build subscription message
                    sub_msg: Dict[str, Any] = {
                        "marketIds":   market_ids,
                        "channels":    channels,
                        "messageType": "subscribe",
                    }
                    if authenticated:
                        ts = str(int(time.time() * 1000))
                        sig = self._sign("GET", "/users/self/subscribe", ts, "")
                        sub_msg["key"]       = self.api_key
                        sub_msg["timestamp"] = ts
                        sub_msg["signature"] = sig

                    await ws.send(json.dumps(sub_msg))
                    logger.info(
                        "BTCMarkets WS subscribed to %s for %s", channels, market_ids
                    )

                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            logger.debug("BTCMarkets WS non-JSON: %.120s", raw)
                            continue

                        # Filter to the expected message type
                        mt = msg.get("messageType", "")
                        if mt not in (msg_type, "subscribeConfirmation",
                                      "error", "orderbook", "trade",
                                      "fundChange", "orderChange",
                                      "heartbeat"):
                            continue

                        if mt == "error":
                            logger.error("BTCMarkets WS error: %s", msg)
                            continue
                        if mt in ("subscribeConfirmation", "heartbeat"):
                            continue

                        # Dispatch normalised message
                        try:
                            result = callback(msg)
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception as cb_exc:
                            logger.error(
                                "BTCMarkets WS callback error: %s", cb_exc
                            )

            except asyncio.CancelledError:
                logger.info("BTCMarkets WS subscription cancelled")
                return
            except Exception as exc:
                logger.warning(
                    "BTCMarkets WS disconnected (%s), reconnecting in %.1fs",
                    exc, backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the HTTP session and any open WebSocket connections."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
        logger.info("BTCMarketsClient session closed")

    async def __aenter__(self) -> "BTCMarketsClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def ping(self) -> Dict[str, Any]:
        """Quick connectivity check against a public endpoint.

        Returns:
            {"healthy": bool, "latency_ms": float, "exchange": str}
        """
        t0 = time.perf_counter()
        try:
            await self._request("GET", "/v3/markets/BTC-AUD/ticker")
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "healthy": True,
                "latency_ms": round(latency_ms, 2),
                "exchange": "btcmarkets",
            }
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "healthy": False,
                "latency_ms": round(latency_ms, 2),
                "exchange": "btcmarkets",
                "error": str(exc),
            }
