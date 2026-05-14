"""
MEXC Exchange Client
====================

Production-quality REST + WebSocket client for the MEXC exchange,
compatible with the Argus exchange interface pattern.

MEXC fee schedule (zero maker fees):
  Spot   maker: 0.00%   taker: 0.05%
  Futures maker: 0.00%  taker: 0.02%

REST base URLs:
  Spot:    https://api.mexc.com
  Futures: https://contract.mexc.com

WebSocket URLs:
  Spot:    wss://wbs.mexc.com/ws
  Futures: wss://contract.mexc.com/edge

Authentication:
  Header: MEXC-APIKEY
  Signature: HMAC-SHA256 of sorted query string, appended as &signature=...

Rate limit: 500 requests / 10 s per IP.  We budget 490 to leave headroom.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
import urllib.parse
from typing import Any, Callable, Dict, List, Optional

import aiohttp
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger("argus.mexc_client")

# ---------------------------------------------------------------------------
# Module-level fee & URL constants
# ---------------------------------------------------------------------------

MEXC_SPOT_MAKER_FEE: float = 0.0
MEXC_SPOT_TAKER_FEE: float = 0.0005
MEXC_FUTURES_MAKER_FEE: float = 0.0
MEXC_FUTURES_TAKER_FEE: float = 0.0002

MEXC_SPOT_BASE_URL: str = "https://api.mexc.com"
MEXC_FUTURES_BASE_URL: str = "https://contract.mexc.com"
MEXC_SPOT_WS_URL: str = "wss://wbs.mexc.com/ws"
MEXC_FUTURES_WS_URL: str = "wss://contract.mexc.com/edge"

# Rate limit budget: 490 req / 10 s  (actual limit is 500)
_RATE_LIMIT_CAPACITY: int = 490
_RATE_LIMIT_WINDOW_S: float = 10.0


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def get_exchange_info() -> Dict[str, Any]:
    """Return a dict with MEXC fee constants and endpoint info."""
    return {
        "exchange": "mexc",
        "fee_rates": {
            "spot_maker": MEXC_SPOT_MAKER_FEE,
            "spot_taker": MEXC_SPOT_TAKER_FEE,
            "futures_maker": MEXC_FUTURES_MAKER_FEE,
            "futures_taker": MEXC_FUTURES_TAKER_FEE,
        },
        "urls": {
            "spot_rest": MEXC_SPOT_BASE_URL,
            "futures_rest": MEXC_FUTURES_BASE_URL,
            "spot_ws": MEXC_SPOT_WS_URL,
            "futures_ws": MEXC_FUTURES_WS_URL,
        },
        "rate_limit": {
            "requests": _RATE_LIMIT_CAPACITY,
            "window_seconds": _RATE_LIMIT_WINDOW_S,
        },
    }


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class MEXCAPIError(Exception):
    """Raised when the MEXC API returns an error payload."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"MEXC API error {code}: {message}")
        self.code: int = code
        self.message: str = message

    def __repr__(self) -> str:  # pragma: no cover
        return f"MEXCAPIError(code={self.code}, message={self.message!r})"


# ---------------------------------------------------------------------------
# Internal: token-bucket rate limiter
# ---------------------------------------------------------------------------

class _TokenBucket:
    """
    Simple async token-bucket rate limiter.

    Allows ``capacity`` tokens over each ``window`` second window.
    Tokens refill continuously (not in discrete windows).
    """

    def __init__(self, capacity: int = _RATE_LIMIT_CAPACITY,
                 window: float = _RATE_LIMIT_WINDOW_S) -> None:
        self._capacity = float(capacity)
        self._window = window
        self._tokens: float = float(capacity)
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until a token is available, then consume one."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            # Refill: capacity tokens per window seconds
            refill = elapsed * (self._capacity / self._window)
            self._tokens = min(self._capacity, self._tokens + refill)
            self._last_refill = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return

            # Must wait for one token to accrue
            wait = (1.0 - self._tokens) * (self._window / self._capacity)
        # Release lock before sleeping
        await asyncio.sleep(wait)
        # Recurse to re-check (handles rare case where another coroutine
        # consumed the token while we were sleeping)
        await self.acquire()


# ---------------------------------------------------------------------------
# Retry predicate helpers
# ---------------------------------------------------------------------------

def _is_retryable(exc: BaseException) -> bool:
    """Return True for 429 / 5xx MEXCAPIErrors and network errors."""
    if isinstance(exc, MEXCAPIError):
        return exc.code in (429, 500, 502, 503, 504)
    if isinstance(exc, (aiohttp.ClientConnectionError,
                        aiohttp.ServerTimeoutError,
                        asyncio.TimeoutError)):
        return True
    return False


# ---------------------------------------------------------------------------
# MEXCClient
# ---------------------------------------------------------------------------

class MEXCClient:
    """
    Async REST + WebSocket client for MEXC (spot and futures).

    Parameters
    ----------
    api_key:
        MEXC API key.
    api_secret:
        MEXC API secret (used for HMAC-SHA256 signing).
    testnet:
        Not officially supported by MEXC; if True a warning is logged and
        production endpoints are still used (MEXC has no public testnet).
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        if testnet:
            log.warning(
                "MEXCClient: testnet=True requested but MEXC has no public "
                "testnet; using production endpoints."
            )

        self._spot_base = MEXC_SPOT_BASE_URL
        self._futures_base = MEXC_FUTURES_BASE_URL

        self._rate_limiter = _TokenBucket()
        self._session: Optional[aiohttp.ClientSession] = None

        # Active WebSocket subscription tasks (symbol → task)
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

    async def __aenter__(self) -> "MEXCClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def _sign(self, params: Dict[str, Any]) -> str:
        """
        Produce an HMAC-SHA256 signature of the sorted query string.

        Parameters are sorted alphabetically by key, URL-encoded, then
        signed with the API secret.

        Returns
        -------
        str
            Hex-encoded signature string.
        """
        sorted_params = dict(sorted(params.items()))
        query_string = urllib.parse.urlencode(sorted_params)
        sig = hmac.new(
            self._api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return sig

    def _auth_params(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Inject timestamp + signature into *params* (in-place copy)."""
        p = dict(params or {})
        p["timestamp"] = int(time.time() * 1000)
        p["signature"] = self._sign(p)
        return p

    def _spot_headers(self) -> Dict[str, str]:
        return {
            "MEXC-APIKEY": self._api_key,
            "Content-Type": "application/json",
        }

    def _futures_headers(self) -> Dict[str, str]:
        return {
            "ApiKey": self._api_key,
            "Content-Type": "application/json",
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
        base_url: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        futures: bool = False,
    ) -> Any:
        """
        Perform a rate-limited, optionally-signed HTTP request.

        Raises
        ------
        MEXCAPIError
            If the exchange returns a non-zero error code.
        aiohttp.ClientResponseError
            For unexpected HTTP error responses.
        """
        await self._rate_limiter.acquire()

        session = await self._get_session()
        headers = self._futures_headers() if futures else self._spot_headers()
        req_params = dict(params or {})

        if signed:
            req_params = self._auth_params(req_params)

        url = f"{base_url}{path}"

        async with session.request(
            method,
            url,
            params=req_params if method == "GET" else None,
            json=json_body if method != "GET" else None,
            headers=headers,
        ) as resp:
            raw = await resp.text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                resp.raise_for_status()
                return raw

            # MEXC spot errors: {"code": <non-zero>, "msg": "..."}
            # MEXC futures errors: {"success": false, "code": ..., "message": "..."}
            if isinstance(data, dict):
                code = data.get("code")
                success = data.get("success", True)

                # Spot: code 0 or 200 means OK; non-zero is error
                if code is not None and code not in (0, 200) and not success:
                    msg = data.get("msg") or data.get("message") or "Unknown error"
                    raise MEXCAPIError(int(code), str(msg))
                # Futures: success=False
                if success is False:
                    err_code = data.get("code", -1)
                    msg = data.get("message", "Unknown futures error")
                    raise MEXCAPIError(int(err_code), str(msg))

            return data

    # ------------------------------------------------------------------
    # Spot REST — market data
    # ------------------------------------------------------------------

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch best bid/ask, last price, and 24h volume for *symbol*.

        Returns
        -------
        dict with keys: bid, ask, last, volume, timestamp
        """
        raw = await self._request(
            "GET",
            self._spot_base,
            "/api/v3/ticker/bookTicker",
            params={"symbol": symbol},
        )
        # bookTicker: {"symbol": ..., "bidPrice": ..., "bidQty": ..., "askPrice": ..., "askQty": ...}
        # Also fetch 24h stats for last/volume
        stats = await self._request(
            "GET",
            self._spot_base,
            "/api/v3/ticker/24hr",
            params={"symbol": symbol},
        )
        return {
            "symbol": symbol,
            "bid": float(raw.get("bidPrice", 0)),
            "ask": float(raw.get("askPrice", 0)),
            "last": float(stats.get("lastPrice", 0)),
            "volume": float(stats.get("volume", 0)),
            "timestamp": int(time.time() * 1000),
        }

    async def fetch_order_book(
        self, symbol: str, limit: int = 20
    ) -> Dict[str, Any]:
        """
        Fetch order book depth.

        Returns
        -------
        dict with keys: bids (list of [price, qty] sorted desc),
                        asks (list of [price, qty] sorted asc),
                        timestamp
        """
        raw = await self._request(
            "GET",
            self._spot_base,
            "/api/v3/depth",
            params={"symbol": symbol, "limit": limit},
        )
        bids = sorted(
            [[float(p), float(q)] for p, q in raw.get("bids", [])],
            key=lambda x: x[0],
            reverse=True,
        )
        asks = sorted(
            [[float(p), float(q)] for p, q in raw.get("asks", [])],
            key=lambda x: x[0],
        )
        return {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "timestamp": raw.get("lastUpdateId", int(time.time() * 1000)),
        }

    # ------------------------------------------------------------------
    # Spot REST — account
    # ------------------------------------------------------------------

    async def fetch_balance(self) -> Dict[str, Dict[str, float]]:
        """
        Fetch spot account balances.

        Returns
        -------
        dict mapping asset → {free, locked, total}
        """
        raw = await self._request(
            "GET",
            self._spot_base,
            "/api/v3/account",
            signed=True,
        )
        result: Dict[str, Dict[str, float]] = {}
        for entry in raw.get("balances", []):
            asset = entry["asset"]
            free = float(entry.get("free", 0))
            locked = float(entry.get("locked", 0))
            if free > 0 or locked > 0:
                result[asset] = {
                    "free": free,
                    "locked": locked,
                    "total": free + locked,
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
        Place a spot order on MEXC.

        Parameters
        ----------
        symbol:
            E.g. ``"BTCUSDT"``.
        side:
            ``"BUY"`` or ``"SELL"``.
        order_type:
            ``"LIMIT"``, ``"MARKET"``, etc.
        quantity:
            Order size in base asset.
        price:
            Limit price (required for LIMIT orders).
        post_only:
            If True (default) the order is placed as a maker-only order
            (``timeInForce="PO"``).  This guarantees the zero maker fee.
            If the order would immediately match, MEXC rejects it (no fill
            as taker), so we cancel proactively — the method raises
            ``MEXCAPIError`` with code 30005 in that case and the caller
            should reprice.

        Returns
        -------
        dict with order details as returned by MEXC.
        """
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity),
        }
        if price is not None:
            params["price"] = str(price)
        if post_only and order_type.upper() == "LIMIT":
            # MEXC uses timeInForce=PO for post-only maker orders
            params["timeInForce"] = "PO"

        raw = await self._request(
            "POST",
            self._spot_base,
            "/api/v3/order",
            params=params,
            signed=True,
        )
        return raw

    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancel an open spot order."""
        params: Dict[str, Any] = {
            "symbol": symbol,
            "orderId": order_id,
        }
        raw = await self._request(
            "DELETE",
            self._spot_base,
            "/api/v3/order",
            params=params,
            signed=True,
        )
        return raw

    async def fetch_open_orders(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch all open spot orders, optionally filtered by *symbol*."""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        raw = await self._request(
            "GET",
            self._spot_base,
            "/api/v3/openOrders",
            params=params,
            signed=True,
        )
        if isinstance(raw, list):
            return raw
        return raw.get("orders", [])

    async def fetch_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Fetch details for a specific spot order."""
        params: Dict[str, Any] = {
            "symbol": symbol,
            "orderId": order_id,
        }
        raw = await self._request(
            "GET",
            self._spot_base,
            "/api/v3/order",
            params=params,
            signed=True,
        )
        return raw

    async def amend_order(
        self,
        symbol: str,
        order_id: str,
        new_price: float,
        new_quantity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Amend (edit) an existing spot limit order.

        MEXC supports in-place order amendment via the ``/api/v3/order``
        PUT endpoint.  The amended order retains its original order ID.
        """
        params: Dict[str, Any] = {
            "symbol": symbol,
            "orderId": order_id,
            "price": str(new_price),
        }
        if new_quantity is not None:
            params["quantity"] = str(new_quantity)
        raw = await self._request(
            "PUT",
            self._spot_base,
            "/api/v3/order",
            params=params,
            signed=True,
        )
        return raw

    # ------------------------------------------------------------------
    # Futures REST — market data
    # ------------------------------------------------------------------

    async def fetch_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch the current funding rate for a futures contract.

        Returns
        -------
        dict with keys: symbol, rate, next_time, predicted
        """
        raw = await self._request(
            "GET",
            self._futures_base,
            "/api/v1/contract/funding_rate",
            params={"symbol": symbol},
            futures=True,
        )
        data = raw.get("data", {})
        return {
            "symbol": symbol,
            "rate": float(data.get("fundingRate", 0)),
            "next_time": data.get("nextSettleTime"),
            "predicted": float(data.get("nextFundingRate", data.get("fundingRate", 0))),
            "raw": data,
        }

    async def fetch_all_funding_rates(self) -> List[Dict[str, Any]]:
        """
        Fetch funding rates for all futures contracts.

        Returns
        -------
        list of dicts (one per symbol), each with rate, next_time, predicted.
        """
        raw = await self._request(
            "GET",
            self._futures_base,
            "/api/v1/contract/funding_rate",
            futures=True,
        )
        results = []
        for entry in raw.get("data", []):
            results.append({
                "symbol": entry.get("symbol"),
                "rate": float(entry.get("fundingRate", 0)),
                "next_time": entry.get("nextSettleTime"),
                "predicted": float(entry.get("nextFundingRate",
                                              entry.get("fundingRate", 0))),
                "raw": entry,
            })
        return results

    # ------------------------------------------------------------------
    # Futures REST — account / positions
    # ------------------------------------------------------------------

    async def fetch_positions(self) -> List[Dict[str, Any]]:
        """
        Fetch all open futures positions.

        Returns
        -------
        list of position dicts.
        """
        raw = await self._request(
            "GET",
            self._futures_base,
            "/api/v1/private/position/open_positions",
            signed=True,
            futures=True,
        )
        return raw.get("data", [])

    async def create_futures_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Place a futures order.

        Parameters
        ----------
        symbol:
            Futures contract symbol, e.g. ``"BTC_USDT"``.
        side:
            ``1`` (open long), ``2`` (close short), ``3`` (open short),
            ``4`` (close long) — or pass string ``"BUY"`` / ``"SELL"``
            which maps to open long / open short respectively.
        order_type:
            ``"LIMIT"`` or ``"MARKET"``.
        quantity:
            Contract quantity (number of contracts).
        price:
            Limit price (required for LIMIT orders).
        reduce_only:
            If True, set ``reduceOnly=true`` on the order.

        Returns
        -------
        dict with order details.
        """
        # MEXC futures side mapping: 1=open_long, 2=close_short, 3=open_short, 4=close_long
        _side_map = {"BUY": 1, "SELL": 3}
        side_val = _side_map.get(str(side).upper(), side)

        _type_map = {"LIMIT": 1, "MARKET": 5, "STOP": 3}
        type_val = _type_map.get(str(order_type).upper(), 1)

        body: Dict[str, Any] = {
            "symbol": symbol,
            "side": side_val,
            "openType": 2,  # 2 = cross margin (default); use 1 for isolated
            "type": type_val,
            "vol": quantity,
        }
        if price is not None:
            body["price"] = price
        if reduce_only:
            body["reduceOnly"] = True

        # Add auth params
        ts = int(time.time() * 1000)
        body["timestamp"] = ts
        body["signature"] = self._sign(body)

        raw = await self._request(
            "POST",
            self._futures_base,
            "/api/v1/private/order/submit",
            json_body=body,
            signed=False,  # Already signed in body
            futures=True,
        )
        return raw.get("data", raw)

    # ------------------------------------------------------------------
    # WebSocket subscriptions
    # ------------------------------------------------------------------

    async def subscribe_order_book(
        self,
        symbol: str,
        callback: Callable,
        depth: int = 20,
    ) -> None:
        """
        Subscribe to the spot L2 order book WebSocket stream.

        *callback* is called with ``(symbol, bids, asks, timestamp_ns)``
        on each update.  The task runs until cancelled.

        Parameters
        ----------
        symbol:
            MEXC spot symbol, e.g. ``"BTCUSDT"``.
        callback:
            Coroutine or regular callable accepting
            ``(symbol, bids, asks, timestamp_ns)``.
        depth:
            Book depth (5, 10, or 20).
        """
        channel = f"spot@public.limit.depth.v3.api@{symbol}@{depth}"
        task = asyncio.create_task(
            self._ws_stream(
                MEXC_SPOT_WS_URL,
                channel,
                symbol,
                self._handle_order_book,
                callback,
            ),
            name=f"mexc_ob_{symbol}",
        )
        self._ws_tasks[f"ob_{symbol}"] = task

    async def subscribe_trades(
        self,
        symbol: str,
        callback: Callable,
    ) -> None:
        """
        Subscribe to the spot trade stream.

        *callback* is called with ``(symbol, side, size, price, timestamp_ns)``.
        """
        channel = f"spot@public.deals.v3.api@{symbol}"
        task = asyncio.create_task(
            self._ws_stream(
                MEXC_SPOT_WS_URL,
                channel,
                symbol,
                self._handle_trades,
                callback,
            ),
            name=f"mexc_trades_{symbol}",
        )
        self._ws_tasks[f"trades_{symbol}"] = task

    async def subscribe_user_data(self, callback: Callable) -> None:
        """
        Subscribe to the authenticated user-data stream (fills, order updates).

        *callback* is called with the raw event dict.
        """
        # MEXC requires a listen key for private WS
        listen_key = await self._get_listen_key()
        url = f"{MEXC_SPOT_WS_URL}?listenKey={listen_key}"

        task = asyncio.create_task(
            self._ws_user_stream(url, callback),
            name="mexc_user_stream",
        )
        self._ws_tasks["user_stream"] = task

    # ------------------------------------------------------------------
    # WebSocket internals
    # ------------------------------------------------------------------

    async def _get_listen_key(self) -> str:
        """Create or retrieve a MEXC spot user data stream listen key."""
        raw = await self._request(
            "POST",
            self._spot_base,
            "/api/v3/userDataStream",
            signed=False,
        )
        return raw.get("listenKey", "")

    async def _ws_stream(
        self,
        url: str,
        channel: str,
        symbol: str,
        handler: Callable,
        callback: Callable,
    ) -> None:
        """
        Internal WS connection loop for public streams with auto-reconnect.
        """
        backoff = 1.0
        while True:
            try:
                session = await self._get_session()
                async with session.ws_connect(
                    url,
                    heartbeat=20.0,
                    receive_timeout=60.0,
                ) as ws:
                    sub_msg = json.dumps({
                        "method": "SUBSCRIPTION",
                        "params": [channel],
                    })
                    await ws.send_str(sub_msg)
                    log.info("MEXCClient: subscribed to %s", channel)
                    backoff = 1.0  # reset on successful connection

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await handler(msg.data, symbol, callback)
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            log.warning("MEXCClient WS closed/error for %s", channel)
                            break

            except asyncio.CancelledError:
                log.info("MEXCClient: WS stream %s cancelled", channel)
                return
            except Exception as exc:
                log.warning(
                    "MEXCClient WS error on %s: %s — reconnect in %.1fs",
                    channel, exc, backoff,
                )

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)

    async def _ws_user_stream(self, url: str, callback: Callable) -> None:
        """Internal WS connection loop for authenticated user stream."""
        backoff = 1.0
        while True:
            try:
                session = await self._get_session()
                async with session.ws_connect(
                    url,
                    heartbeat=20.0,
                    receive_timeout=120.0,
                ) as ws:
                    log.info("MEXCClient: connected to user stream")
                    backoff = 1.0

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                                if callable(callback):
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(data)
                                    else:
                                        callback(data)
                            except Exception as exc:
                                log.debug("MEXCClient user stream parse error: %s", exc)
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            log.warning("MEXCClient user stream closed/error")
                            break

            except asyncio.CancelledError:
                log.info("MEXCClient: user stream cancelled")
                return
            except Exception as exc:
                log.warning("MEXCClient user stream error: %s — reconnect in %.1fs", exc, backoff)

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)
            # Refresh listen key on reconnect
            try:
                listen_key = await self._get_listen_key()
                url = f"{MEXC_SPOT_WS_URL}?listenKey={listen_key}"
            except Exception as exc:
                log.warning("MEXCClient: failed to refresh listen key: %s", exc)

    @staticmethod
    async def _handle_order_book(
        raw: str,
        symbol: str,
        callback: Callable,
    ) -> None:
        """
        Parse MEXC depth update and invoke callback.

        Expected MEXC depth format:
        {
            "c": "spot@public.limit.depth.v3.api@BTCUSDT@20",
            "d": {
                "bids": [["price", "qty"], ...],
                "asks": [["price", "qty"], ...],
                "e": 1000
            },
            "t": 1234567890123
        }
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        if not isinstance(data, dict):
            return

        # Subscription confirmation messages
        if "id" in data or "msg" in data:
            return

        d = data.get("d", {})
        if not d:
            return

        ts_ms = data.get("t", int(time.time() * 1000))
        ts_ns = ts_ms * 1_000_000

        raw_bids = d.get("bids", [])
        raw_asks = d.get("asks", [])

        bids = [[float(p), float(q)] for p, q in raw_bids]
        asks = [[float(p), float(q)] for p, q in raw_asks]

        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(symbol, bids, asks, ts_ns)
            else:
                callback(symbol, bids, asks, ts_ns)
        except Exception as exc:
            log.debug("MEXCClient order_book callback error: %s", exc)

    @staticmethod
    async def _handle_trades(
        raw: str,
        symbol: str,
        callback: Callable,
    ) -> None:
        """
        Parse MEXC trade update and invoke callback.

        Expected trade stream format:
        {
            "c": "spot@public.deals.v3.api@BTCUSDT",
            "d": {
                "deals": [
                    {"S": 1, "p": "29000.00", "v": "0.001", "t": 1234567890123}
                ]
            },
            "t": 1234567890123
        }
        S=1 → buy, S=2 → sell
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        if not isinstance(data, dict):
            return
        if "id" in data or "msg" in data:
            return

        d = data.get("d", {})
        deals = d.get("deals", [])

        for deal in deals:
            side = "BUY" if deal.get("S") == 1 else "SELL"
            price = float(deal.get("p", 0))
            size = float(deal.get("v", 0))
            ts_ns = int(deal.get("t", time.time() * 1000)) * 1_000_000
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(symbol, side, size, price, ts_ns)
                else:
                    callback(symbol, side, size, price, ts_ns)
            except Exception as exc:
                log.debug("MEXCClient trades callback error: %s", exc)
