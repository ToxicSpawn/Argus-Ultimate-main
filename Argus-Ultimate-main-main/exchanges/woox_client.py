"""
WOO X Exchange Client
=====================

Production-quality REST + WebSocket client for the WOO X exchange,
compatible with the Argus exchange interface pattern.

WOO X fee schedule (zero maker/taker on eligible pairs):
  Spot   maker: 0.00%   taker: 0.00%  (zero on 90+ pairs)
  Futures maker: 0.00%  taker: 0.00%  (zero on 71+ pairs)

REST base URLs:
  Public:  https://api.woo.org/v1
  Private: https://api.woo.org/v3

WebSocket URLs:
  Public:  wss://wss.woo.org/v2/ws/public
  Private: wss://wss.woo.org/v2/ws/private

Authentication:
  Headers: WOO-KEY, WOO-TIMESTAMP, WOO-SIGNATURE
  Signature: HMAC-SHA256 of "{timestamp}{METHOD}{path}{body}"

Rate limit: 10 req/s per endpoint. Token bucket budgeted at 9/s.

Symbol format:
  Spot:    SPOT_BTC_USDT
  Perps:   PERP_BTC_USDT
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import aiohttp
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger("argus.woox_client")

# ---------------------------------------------------------------------------
# Module-level fee & URL constants
# ---------------------------------------------------------------------------

WOOX_SPOT_MAKER_FEE: float = 0.0
WOOX_SPOT_TAKER_FEE: float = 0.0
WOOX_FUTURES_MAKER_FEE: float = 0.0
WOOX_FUTURES_TAKER_FEE: float = 0.0

WOOX_BASE_URL: str = "https://api.woo.org"
WOOX_TESTNET_URL: str = "https://api.staging.woo.org"

WOOX_WS_PUBLIC: str = "wss://wss.woo.org/v2/ws/public"
WOOX_WS_PRIVATE: str = "wss://wss.woo.org/v2/ws/private"

WOOX_WS_PUBLIC_TESTNET: str = "wss://wss.staging.woo.org/v2/ws/public"
WOOX_WS_PRIVATE_TESTNET: str = "wss://wss.staging.woo.org/v2/ws/private"

# Rate limit: 9 req/s (actual limit is 10/s per endpoint)
_RATE_LIMIT_PER_SECOND: float = 9.0


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def get_exchange_info() -> Dict[str, Any]:
    """Return a dict with WOO X fee constants and endpoint info."""
    return {
        "exchange": "woox",
        "display_name": "WOO X",
        "fee_rates": {
            "spot_maker": WOOX_SPOT_MAKER_FEE,
            "spot_taker": WOOX_SPOT_TAKER_FEE,
            "futures_maker": WOOX_FUTURES_MAKER_FEE,
            "futures_taker": WOOX_FUTURES_TAKER_FEE,
        },
        "urls": {
            "rest": WOOX_BASE_URL,
            "ws_public": WOOX_WS_PUBLIC,
            "ws_private": WOOX_WS_PRIVATE,
        },
        "rate_limit": {
            "requests_per_second": _RATE_LIMIT_PER_SECOND,
        },
        "notes": "Zero maker/taker fee on 90+ spot pairs and 71+ futures pairs.",
    }


# ---------------------------------------------------------------------------
# Symbol normalisation helpers (module-level)
# ---------------------------------------------------------------------------

def to_woox_symbol(symbol: str) -> str:
    """
    Convert a slash-separated symbol to WOO X spot format.

    Examples
    --------
    "BTC/USDT"  →  "SPOT_BTC_USDT"
    "ETH/USDT"  →  "SPOT_ETH_USDT"
    "SPOT_BTC_USDT"  →  "SPOT_BTC_USDT"  (no-op)
    """
    symbol = symbol.strip().upper()
    if symbol.startswith("SPOT_") or symbol.startswith("PERP_"):
        return symbol
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
        return f"SPOT_{base}_{quote}"
    # Already underscored but no prefix — assume it's correct
    return f"SPOT_{symbol}"


def to_woox_perp(symbol: str) -> str:
    """
    Convert a slash-separated symbol to WOO X perpetual format.

    Examples
    --------
    "BTC/USDT"  →  "PERP_BTC_USDT"
    "ETH/USDT"  →  "PERP_ETH_USDT"
    "PERP_BTC_USDT"  →  "PERP_BTC_USDT"  (no-op)
    """
    symbol = symbol.strip().upper()
    if symbol.startswith("PERP_"):
        return symbol
    if symbol.startswith("SPOT_"):
        return symbol.replace("SPOT_", "PERP_", 1)
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
        return f"PERP_{base}_{quote}"
    return f"PERP_{symbol}"


def from_woox_symbol(symbol: str) -> str:
    """
    Convert a WOO X symbol to slash-separated format.

    Examples
    --------
    "SPOT_BTC_USDT"  →  "BTC/USDT"
    "PERP_BTC_USDT"  →  "BTC/USDT"
    """
    symbol = symbol.strip().upper()
    for prefix in ("SPOT_", "PERP_"):
        if symbol.startswith(prefix):
            remainder = symbol[len(prefix):]
            # remainder is e.g. "BTC_USDT" — split on last underscore
            parts = remainder.rsplit("_", 1)
            if len(parts) == 2:
                return f"{parts[0]}/{parts[1]}"
            return remainder
    return symbol


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class WOOXAPIError(Exception):
    """Raised when the WOO X API returns an error payload."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"WOO X API error {code}: {message}")
        self.code: int = code
        self.message: str = message

    def __repr__(self) -> str:  # pragma: no cover
        return f"WOOXAPIError(code={self.code}, message={self.message!r})"


# ---------------------------------------------------------------------------
# Internal: token-bucket rate limiter (9 req/s)
# ---------------------------------------------------------------------------

class _TokenBucket:
    """
    Simple async token-bucket rate limiter.

    Allows *capacity* tokens per second, refilled continuously.
    """

    def __init__(self, capacity: float = _RATE_LIMIT_PER_SECOND) -> None:
        self._capacity = capacity
        self._tokens: float = capacity
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until a token is available, then consume one."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._capacity, self._tokens + elapsed * self._capacity)
            self._last_refill = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return

            wait = (1.0 - self._tokens) / self._capacity
        await asyncio.sleep(wait)
        await self.acquire()


# ---------------------------------------------------------------------------
# Retry predicate
# ---------------------------------------------------------------------------

def _is_retryable(exc: BaseException) -> bool:
    """Return True for rate-limit / server errors and network failures."""
    if isinstance(exc, WOOXAPIError):
        return exc.code in (429, 500, 502, 503, 504)
    if isinstance(exc, (aiohttp.ClientConnectionError,
                        aiohttp.ServerTimeoutError,
                        asyncio.TimeoutError)):
        return True
    return False


# ---------------------------------------------------------------------------
# WOOXClient
# ---------------------------------------------------------------------------

class WOOXClient:
    """
    Async REST + WebSocket client for WOO X (spot and perpetuals).

    Parameters
    ----------
    api_key:
        WOO X API key.
    api_secret:
        WOO X API secret (used for HMAC-SHA256 signing).
    testnet:
        If True, connect to the WOO X staging environment.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet

        if testnet:
            self._base_url = WOOX_TESTNET_URL
            self._ws_public_url = WOOX_WS_PUBLIC_TESTNET
            self._ws_private_url = WOOX_WS_PRIVATE_TESTNET
            log.info("WOOXClient: using testnet (staging) endpoints")
        else:
            self._base_url = WOOX_BASE_URL
            self._ws_public_url = WOOX_WS_PUBLIC
            self._ws_private_url = WOOX_WS_PRIVATE

        self._rate_limiter = _TokenBucket()
        self._session: Optional[aiohttp.ClientSession] = None

        # Active WebSocket tasks
        self._ws_tasks: Dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=10.0)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session and cancel WS tasks."""
        for task in self._ws_tasks.values():
            task.cancel()
        self._ws_tasks.clear()
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self) -> "WOOXClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """
        Produce an HMAC-SHA256 signature for WOO X.

        Message format: "{timestamp}{METHOD}{path}{body}"

        Parameters
        ----------
        timestamp:
            Unix timestamp in milliseconds as a string.
        method:
            HTTP method in uppercase ("GET", "POST", …).
        path:
            API path without base URL, e.g. "/v1/public/market_trades".
        body:
            URL-encoded or JSON body string (empty for GET).

        Returns
        -------
        str
            Hex-encoded HMAC-SHA256 signature.
        """
        message = f"{timestamp}{method}{path}{body}"
        sig = hmac.new(
            self._api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return sig

    def _auth_headers(
        self, method: str, path: str, body: str = ""
    ) -> Dict[str, str]:
        """Build WOO X authentication headers for a private request."""
        ts = str(int(time.time() * 1000))
        sig = self._sign(ts, method, path, body)
        return {
            "WOO-KEY": self._api_key,
            "WOO-TIMESTAMP": ts,
            "WOO-SIGNATURE": sig,
            "Content-Type": "application/x-www-form-urlencoded",
        }

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
        reraise=True,
    )
    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        api_version: str = "v1",
    ) -> Any:
        """
        Perform a rate-limited, optionally-signed HTTP request.

        Parameters
        ----------
        method:
            HTTP method ("GET", "POST", "DELETE", "PUT").
        path:
            API path fragment, e.g. "/public/market_trades".
        params:
            Query-string parameters (used for GET).
        data:
            Form-encoded body parameters (used for POST/DELETE).
        signed:
            If True, include WOO X authentication headers.
        api_version:
            API version prefix ("v1", "v3").

        Raises
        ------
        WOOXAPIError
            If the exchange returns a non-success payload.
        """
        await self._rate_limiter.acquire()

        session = await self._get_session()
        full_path = f"/{api_version}{path}"
        url = f"{self._base_url}{full_path}"

        # Build body string for signing
        body_str = ""
        headers: Dict[str, str] = {"Content-Type": "application/x-www-form-urlencoded"}

        if signed:
            if data and method != "GET":
                import urllib.parse
                body_str = urllib.parse.urlencode(data)
            headers.update(self._auth_headers(method, full_path, body_str))

        async with session.request(
            method,
            url,
            params=params if method == "GET" else None,
            data=body_str if (method != "GET" and data) else None,
            headers=headers,
        ) as resp:
            raw = await resp.text()
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                resp.raise_for_status()
                return raw

            # WOO X error format: {"success": false, "code": -1, "message": "..."}
            if isinstance(result, dict):
                success = result.get("success", True)
                if success is False:
                    code = result.get("code", -1)
                    msg = result.get("message", result.get("msg", "Unknown WOO X error"))
                    raise WOOXAPIError(int(code), str(msg))

            return result

    # ------------------------------------------------------------------
    # Symbol helpers (instance methods mirror module-level functions)
    # ------------------------------------------------------------------

    @staticmethod
    def to_woox_symbol(symbol: str) -> str:
        """Convert "BTC/USDT" → "SPOT_BTC_USDT"."""
        return to_woox_symbol(symbol)

    @staticmethod
    def to_woox_perp(symbol: str) -> str:
        """Convert "BTC/USDT" → "PERP_BTC_USDT"."""
        return to_woox_perp(symbol)

    @staticmethod
    def from_woox_symbol(symbol: str) -> str:
        """Convert "SPOT_BTC_USDT" → "BTC/USDT"."""
        return from_woox_symbol(symbol)

    # ------------------------------------------------------------------
    # Spot REST — market data
    # ------------------------------------------------------------------

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch best bid/ask, last price, and 24h volume for *symbol*.

        Parameters
        ----------
        symbol:
            Any format: "BTC/USDT", "SPOT_BTC_USDT".

        Returns
        -------
        dict with keys: symbol, bid, ask, last, volume, timestamp, exchange
        """
        woox_sym = to_woox_symbol(symbol)
        raw = await self._request(
            "GET",
            f"/public/market_trades",
            params={"symbol": woox_sym, "limit": 1},
            api_version="v1",
        )
        # Fetch orderbook for bid/ask
        ob_raw = await self._request(
            "GET",
            f"/public/orderbook",
            params={"symbol": woox_sym},
            api_version="v1",
        )

        bids_raw = ob_raw.get("bids", {})
        asks_raw = ob_raw.get("asks", {})

        # WOO X orderbook: {"bids": {"0": [price, qty], ...}, "asks": {...}}
        # or as lists in some versions
        best_bid = 0.0
        best_ask = 0.0

        if isinstance(bids_raw, dict):
            bid_levels = list(bids_raw.values())
            if bid_levels:
                best_bid = float(bid_levels[0][0]) if bid_levels[0] else 0.0
        elif isinstance(bids_raw, list) and bids_raw:
            best_bid = float(bids_raw[0][0])

        if isinstance(asks_raw, dict):
            ask_levels = list(asks_raw.values())
            if ask_levels:
                best_ask = float(ask_levels[0][0]) if ask_levels[0] else 0.0
        elif isinstance(asks_raw, list) and asks_raw:
            best_ask = float(asks_raw[0][0])

        # Last trade price
        trades = raw.get("rows", [])
        last = float(trades[0].get("executed_price", 0)) if trades else 0.0

        return {
            "symbol": from_woox_symbol(woox_sym),
            "bid": best_bid,
            "ask": best_ask,
            "last": last,
            "volume": float(ob_raw.get("timestamp", 0)),
            "timestamp": int(time.time() * 1000),
            "exchange": "woox",
            "raw_ob": ob_raw,
        }

    async def fetch_order_book(
        self, symbol: str, limit: int = 20
    ) -> Dict[str, Any]:
        """
        Fetch order book depth for *symbol*.

        Parameters
        ----------
        symbol:
            Any format: "BTC/USDT", "SPOT_BTC_USDT".
        limit:
            Number of price levels per side (max 100).

        Returns
        -------
        dict with keys: symbol, bids [[price, qty]...], asks [[price, qty]...], timestamp
        """
        woox_sym = to_woox_symbol(symbol)
        raw = await self._request(
            "GET",
            "/public/orderbook",
            params={"symbol": woox_sym, "max_level": min(limit, 100)},
            api_version="v1",
        )

        # WOO X returns: {"bids": [[price, qty, count], ...], "asks": [...]}
        raw_bids = raw.get("bids", [])
        raw_asks = raw.get("asks", [])

        # Handle both list-of-lists and dict-of-lists formats
        if isinstance(raw_bids, dict):
            raw_bids = list(raw_bids.values())
        if isinstance(raw_asks, dict):
            raw_asks = list(raw_asks.values())

        bids = sorted(
            [[float(row[0]), float(row[1])] for row in raw_bids if len(row) >= 2],
            key=lambda x: x[0],
            reverse=True,
        )[:limit]
        asks = sorted(
            [[float(row[0]), float(row[1])] for row in raw_asks if len(row) >= 2],
            key=lambda x: x[0],
        )[:limit]

        return {
            "symbol": from_woox_symbol(woox_sym),
            "bids": bids,
            "asks": asks,
            "timestamp": int(raw.get("timestamp", time.time() * 1000)),
            "exchange": "woox",
        }

    # ------------------------------------------------------------------
    # Spot REST — account
    # ------------------------------------------------------------------

    async def fetch_balance(self) -> Dict[str, Dict[str, float]]:
        """
        Fetch spot account balances.

        Returns
        -------
        dict mapping asset → {holding, frozen, staked, total}
        """
        raw = await self._request(
            "GET",
            "/client/holding",
            signed=True,
            api_version="v3",
        )
        result: Dict[str, Dict[str, float]] = {}
        for entry in raw.get("holding", []):
            token = entry.get("token", "")
            holding = float(entry.get("holding", 0))
            frozen = float(entry.get("frozen", 0))
            staked = float(entry.get("staked", 0))
            if holding > 0 or frozen > 0:
                result[token] = {
                    "free": holding - frozen,
                    "locked": frozen,
                    "staked": staked,
                    "total": holding,
                }
        return result

    # ------------------------------------------------------------------
    # Spot REST — order management
    # ------------------------------------------------------------------

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        post_only: bool = True,
    ) -> Dict[str, Any]:
        """
        Place a spot order on WOO X.

        Parameters
        ----------
        symbol:
            E.g. "BTC/USDT" or "SPOT_BTC_USDT".
        side:
            "BUY" or "SELL".
        order_type:
            "LIMIT", "MARKET", "IOC", "FOK", "POST_ONLY".
        quantity:
            Order size in base asset.
        price:
            Limit price (required for LIMIT orders).
        post_only:
            If True (default), sends order_tag="post_only" to WOO X.
            This maps to the POST_ONLY order type which guarantees maker execution
            and the zero maker fee.

        Returns
        -------
        dict with order details as returned by WOO X.
        """
        woox_sym = to_woox_symbol(symbol)

        order_data: Dict[str, Any] = {
            "symbol": woox_sym,
            "order_type": order_type.upper(),
            "side": side.upper(),
            "order_quantity": str(quantity),
        }

        if price is not None:
            order_data["order_price"] = str(price)

        if post_only and order_type.upper() == "LIMIT":
            # WOO X post_only flag: use order_type POST_ONLY or add order_tag
            order_data["order_type"] = "POST_ONLY"

        raw = await self._request(
            "POST",
            "/order",
            data=order_data,
            signed=True,
            api_version="v3",
        )

        return {
            "order_id": str(raw.get("order_id", "")),
            "symbol": from_woox_symbol(woox_sym),
            "side": side.upper(),
            "type": order_data["order_type"],
            "quantity": quantity,
            "price": price,
            "status": raw.get("status", ""),
            "post_only": post_only,
            "exchange": "woox",
            "raw": raw,
        }

    async def cancel_order(self, symbol: str, order_id: Any) -> Dict[str, Any]:
        """Cancel an open order by ID."""
        woox_sym = to_woox_symbol(symbol)
        raw = await self._request(
            "DELETE",
            f"/order/{order_id}",
            data={"symbol": woox_sym},
            signed=True,
            api_version="v3",
        )
        return {
            "order_id": str(order_id),
            "symbol": from_woox_symbol(woox_sym),
            "status": raw.get("status", "CANCELLED"),
            "exchange": "woox",
            "raw": raw,
        }

    async def fetch_open_orders(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch all open orders, optionally filtered by *symbol*."""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = to_woox_symbol(symbol)

        raw = await self._request(
            "GET",
            "/orders",
            params={**params, "status": "INCOMPLETE"},
            signed=True,
            api_version="v3",
        )

        orders = []
        for item in raw.get("rows", []):
            woox_sym = item.get("symbol", "")
            orders.append({
                "order_id": str(item.get("order_id", "")),
                "symbol": from_woox_symbol(woox_sym),
                "side": item.get("side", ""),
                "type": item.get("type", ""),
                "quantity": float(item.get("quantity", 0)),
                "executed": float(item.get("executed", 0)),
                "price": float(item.get("price", 0)),
                "status": item.get("status", ""),
                "timestamp": item.get("created_time", ""),
                "exchange": "woox",
            })
        return orders

    async def fetch_order(self, symbol: str, order_id: Any) -> Dict[str, Any]:
        """Fetch details of a specific order."""
        raw = await self._request(
            "GET",
            f"/order/{order_id}",
            signed=True,
            api_version="v3",
        )
        woox_sym = raw.get("symbol", to_woox_symbol(symbol))
        return {
            "order_id": str(raw.get("order_id", "")),
            "symbol": from_woox_symbol(woox_sym),
            "side": raw.get("side", ""),
            "type": raw.get("type", ""),
            "quantity": float(raw.get("quantity", 0)),
            "executed": float(raw.get("executed", 0)),
            "price": float(raw.get("price", 0)),
            "avg_price": float(raw.get("average_executed_price", 0)),
            "status": raw.get("status", ""),
            "timestamp": raw.get("created_time", ""),
            "exchange": "woox",
            "raw": raw,
        }

    # ------------------------------------------------------------------
    # Futures REST — market data
    # ------------------------------------------------------------------

    async def fetch_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch the current funding rate for a perpetual contract.

        Parameters
        ----------
        symbol:
            Any format: "BTC/USDT", "PERP_BTC_USDT".

        Returns
        -------
        dict with keys: symbol, rate, next_time, predicted, exchange
        """
        woox_sym = to_woox_perp(symbol)
        raw = await self._request(
            "GET",
            f"/public/funding_rate/{woox_sym}",
            api_version="v1",
        )
        return {
            "symbol": woox_sym,
            "rate": float(raw.get("funding_rate", 0)),
            "next_time": raw.get("next_funding_time"),
            "predicted": float(raw.get("predicted_funding_rate",
                                        raw.get("funding_rate", 0))),
            "last_funding_time": raw.get("last_funding_time"),
            "exchange": "woox",
            "raw": raw,
        }

    async def fetch_all_funding_rates(self) -> List[Dict[str, Any]]:
        """
        Fetch funding rates for all perpetual contracts.

        Returns
        -------
        list of dicts (one per symbol), each with rate, next_time, predicted.
        """
        raw = await self._request(
            "GET",
            "/public/funding_rates",
            api_version="v1",
        )
        results = []
        for entry in raw.get("rows", []):
            results.append({
                "symbol": entry.get("symbol", ""),
                "rate": float(entry.get("funding_rate", 0)),
                "next_time": entry.get("next_funding_time"),
                "predicted": float(entry.get("predicted_funding_rate",
                                              entry.get("funding_rate", 0))),
                "exchange": "woox",
                "raw": entry,
            })
        return results

    # ------------------------------------------------------------------
    # Futures REST — account / positions
    # ------------------------------------------------------------------

    async def fetch_positions(self) -> List[Dict[str, Any]]:
        """
        Fetch all open perpetual positions.

        Returns
        -------
        list of position dicts.
        """
        raw = await self._request(
            "GET",
            "/positions",
            signed=True,
            api_version="v3",
        )
        positions = []
        for entry in raw.get("positions", {}).get("rows", []):
            positions.append({
                "symbol": entry.get("symbol", ""),
                "holding": float(entry.get("holding", 0)),
                "pending_long_qty": float(entry.get("pending_long_qty", 0)),
                "pending_short_qty": float(entry.get("pending_short_qty", 0)),
                "average_open_price": float(entry.get("average_open_price", 0)),
                "mark_price": float(entry.get("mark_price", 0)),
                "unrealized_pnl": float(entry.get("unrealized_pnl", 0)),
                "settle_price": float(entry.get("settle_price", 0)),
                "side": "LONG" if float(entry.get("holding", 0)) > 0 else "SHORT",
                "exchange": "woox",
                "raw": entry,
            })
        return positions

    async def create_futures_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Place a perpetual futures order on WOO X.

        Parameters
        ----------
        symbol:
            Any format: "BTC/USDT", "PERP_BTC_USDT".
        side:
            "BUY" or "SELL".
        order_type:
            "LIMIT", "MARKET", "POST_ONLY".
        quantity:
            Contract quantity.
        price:
            Limit price (required for LIMIT/POST_ONLY orders).

        Returns
        -------
        dict with order details.
        """
        woox_sym = to_woox_perp(symbol)

        order_data: Dict[str, Any] = {
            "symbol": woox_sym,
            "order_type": order_type.upper(),
            "side": side.upper(),
            "order_quantity": str(quantity),
        }

        if price is not None:
            order_data["order_price"] = str(price)

        raw = await self._request(
            "POST",
            "/order",
            data=order_data,
            signed=True,
            api_version="v3",
        )

        return {
            "order_id": str(raw.get("order_id", "")),
            "symbol": woox_sym,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity,
            "price": price,
            "status": raw.get("status", ""),
            "exchange": "woox",
            "raw": raw,
        }

    # ------------------------------------------------------------------
    # WebSocket subscriptions
    # ------------------------------------------------------------------

    async def subscribe_order_book(
        self,
        symbol: str,
        callback: Callable,
    ) -> None:
        """
        Subscribe to the WOO X orderbook WebSocket channel.

        WOO X subscription topic: "{SPOT_BTC_USDT}@orderbook"

        *callback* is called with (symbol, bids, asks, timestamp_ns).
        The task runs until cancelled.

        Parameters
        ----------
        symbol:
            Any format: "BTC/USDT", "SPOT_BTC_USDT".
        callback:
            Coroutine or callable accepting (symbol, bids, asks, timestamp_ns).
        """
        woox_sym = to_woox_symbol(symbol)
        topic = f"{woox_sym}@orderbook"
        task_key = f"ob_{woox_sym}"

        task = asyncio.create_task(
            self._ws_public_stream(topic, woox_sym, self._handle_order_book, callback),
            name=f"woox_ob_{woox_sym}",
        )
        self._ws_tasks[task_key] = task

    async def subscribe_trades(
        self,
        symbol: str,
        callback: Callable,
    ) -> None:
        """
        Subscribe to the WOO X trade channel.

        WOO X topic: "{SPOT_BTC_USDT}@trade"

        *callback* is called with (symbol, side, size, price, timestamp_ns).

        Parameters
        ----------
        symbol:
            Any format: "BTC/USDT", "SPOT_BTC_USDT".
        callback:
            Coroutine or callable accepting (symbol, side, size, price, timestamp_ns).
        """
        woox_sym = to_woox_symbol(symbol)
        topic = f"{woox_sym}@trade"
        task_key = f"trades_{woox_sym}"

        task = asyncio.create_task(
            self._ws_public_stream(topic, woox_sym, self._handle_trades, callback),
            name=f"woox_trades_{woox_sym}",
        )
        self._ws_tasks[task_key] = task

    async def subscribe_user_data(self, callback: Callable) -> None:
        """
        Subscribe to the authenticated WOO X user data stream.

        Covers: order fills, order updates, account balance changes.
        *callback* is called with the raw event dict.
        """
        task = asyncio.create_task(
            self._ws_private_stream(callback),
            name="woox_user_stream",
        )
        self._ws_tasks["user_stream"] = task

    # ------------------------------------------------------------------
    # WebSocket internals
    # ------------------------------------------------------------------

    async def _ws_public_stream(
        self,
        topic: str,
        symbol: str,
        handler: Callable,
        callback: Callable,
    ) -> None:
        """Internal WS connection loop for public streams with auto-reconnect."""
        backoff = 1.0
        while True:
            try:
                session = await self._get_session()
                async with session.ws_connect(
                    self._ws_public_url,
                    heartbeat=30.0,
                    receive_timeout=60.0,
                ) as ws:
                    sub_msg = json.dumps({
                        "event": "subscribe",
                        "topic": topic,
                    })
                    await ws.send_str(sub_msg)
                    log.info("WOOXClient: subscribed to topic=%s", topic)
                    backoff = 1.0  # reset on successful connection

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await handler(msg.data, symbol, callback)
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            log.warning("WOOXClient WS closed/error for %s", topic)
                            break

            except asyncio.CancelledError:
                log.info("WOOXClient: WS stream %s cancelled", topic)
                return
            except Exception as exc:
                log.warning(
                    "WOOXClient WS error on %s: %s — reconnect in %.1fs",
                    topic, exc, backoff,
                )

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)

    async def _ws_private_stream(self, callback: Callable) -> None:
        """Internal WS connection loop for authenticated private stream."""
        backoff = 1.0
        while True:
            try:
                session = await self._get_session()
                async with session.ws_connect(
                    self._ws_private_url,
                    heartbeat=30.0,
                    receive_timeout=120.0,
                ) as ws:
                    # Authenticate the WS connection
                    ts = str(int(time.time() * 1000))
                    sig = self._sign(ts, "GET", "/v2/ws/private/stream", "")
                    auth_msg = json.dumps({
                        "event": "auth",
                        "params": {
                            "apikey": self._api_key,
                            "sign": sig,
                            "timestamp": ts,
                        },
                    })
                    await ws.send_str(auth_msg)
                    log.info("WOOXClient: authenticated private WS")
                    backoff = 1.0

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                                # Skip auth confirmation messages
                                event = data.get("event", "")
                                if event in ("auth", "ping", "pong"):
                                    continue
                                if callable(callback):
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(data)
                                    else:
                                        callback(data)
                            except Exception as exc:
                                log.debug("WOOXClient private WS parse error: %s", exc)
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            log.warning("WOOXClient private WS closed/error")
                            break

            except asyncio.CancelledError:
                log.info("WOOXClient: private WS stream cancelled")
                return
            except Exception as exc:
                log.warning(
                    "WOOXClient private WS error: %s — reconnect in %.1fs", exc, backoff
                )

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)

    @staticmethod
    async def _handle_order_book(
        raw: str,
        symbol: str,
        callback: Callable,
    ) -> None:
        """
        Parse WOO X orderbook update and invoke callback.

        Expected WOO X orderbook format:
        {
            "topic": "SPOT_BTC_USDT@orderbook",
            "ts": 1234567890123,
            "data": {
                "bids": [[price, qty], ...],
                "asks": [[price, qty], ...],
                "ts": 1234567890123
            }
        }
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        if not isinstance(data, dict):
            return

        # Skip subscription confirmations and pings
        event = data.get("event", "")
        if event in ("subscribe", "unsubscribe", "ping", "pong", "error"):
            return

        book_data = data.get("data", {})
        if not book_data:
            return

        ts_ms = data.get("ts", book_data.get("ts", int(time.time() * 1000)))
        ts_ns = int(ts_ms) * 1_000_000

        raw_bids = book_data.get("bids", [])
        raw_asks = book_data.get("asks", [])

        bids = [[float(row[0]), float(row[1])] for row in raw_bids if len(row) >= 2]
        asks = [[float(row[0]), float(row[1])] for row in raw_asks if len(row) >= 2]

        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(symbol, bids, asks, ts_ns)
            else:
                callback(symbol, bids, asks, ts_ns)
        except Exception as exc:
            log.debug("WOOXClient order_book callback error: %s", exc)

    @staticmethod
    async def _handle_trades(
        raw: str,
        symbol: str,
        callback: Callable,
    ) -> None:
        """
        Parse WOO X trade update and invoke callback.

        Expected WOO X trade format:
        {
            "topic": "SPOT_BTC_USDT@trade",
            "ts": 1234567890123,
            "data": {
                "symbol": "SPOT_BTC_USDT",
                "price": 29000.00,
                "size": 0.001,
                "side": "BUY",
                "source": 0,
                "ts": 1234567890123
            }
        }
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        if not isinstance(data, dict):
            return

        event = data.get("event", "")
        if event in ("subscribe", "unsubscribe", "ping", "pong", "error"):
            return

        trade_data = data.get("data", {})
        if not trade_data:
            return

        ts_ms = trade_data.get("ts", data.get("ts", int(time.time() * 1000)))
        ts_ns = int(ts_ms) * 1_000_000

        side = str(trade_data.get("side", "BUY")).upper()
        price = float(trade_data.get("price", 0))
        size = float(trade_data.get("size", 0))

        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(symbol, side, size, price, ts_ns)
            else:
                callback(symbol, side, size, price, ts_ns)
        except Exception as exc:
            log.debug("WOOXClient trades callback error: %s", exc)
