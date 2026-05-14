"""
WebSocket Order Placer — Kraken WS API v2.

Places orders via Kraken's authenticated WebSocket API instead of REST,
reducing round-trip latency from ~150ms REST to ~30ms WebSocket.
Critical for scalping and market making where execution speed matters.

Auth flow:
  1. GET https://api.kraken.com/0/private/GetWebSocketsToken  (REST, signed)
  2. Connect to wss://ws-auth.kraken.com/v2
  3. All order operations sent as JSON messages with the WS token

Falls back to REST if WebSocket is unavailable or disconnected.

Docs: https://docs.kraken.com/websockets-v2/
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
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

WS_URL   = "wss://ws-auth.kraken.com/v2"
REST_URL = "https://api.kraken.com/0/private"
REST_PUBLIC_URL = "https://api.kraken.com/0/public"


class KrakenWebSocketOrderPlacer:
    """
    Authenticated Kraken WebSocket order placement client.

    Maintains a persistent WS connection and sends order messages
    without the overhead of individual REST calls.
    """

    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key or os.environ.get("KRAKEN_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("KRAKEN_API_SECRET", "")
        self._ws: Any = None
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0  # monotonic timestamp
        self._connected: bool = False
        self._listener_task: Optional[asyncio.Task] = None
        self._pending: Dict[str, asyncio.Future] = {}
        self._req_id: int = 0

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _kraken_sign(self, url_path: str, data: Dict[str, str]) -> str:
        """Generate Kraken API v0 HMAC-SHA512 signature."""
        nonce = data["nonce"]
        post_data = urllib.parse.urlencode(data)
        encoded = (nonce + post_data).encode("utf-8")
        message = url_path.encode("utf-8") + hashlib.sha256(encoded).digest()
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message,
            hashlib.sha512,
        )
        return base64.b64encode(signature.digest()).decode()

    def _rest_post(self, path: str, data: Dict[str, str]) -> Dict[str, Any]:
        """Synchronous REST POST to Kraken private API."""
        data["nonce"] = str(int(time.time() * 1000))
        signature = self._kraken_sign(f"/0/private/{path}", data)
        headers = {
            "API-Key": self.api_key,
            "API-Sign": signature,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        body = urllib.parse.urlencode(data).encode("utf-8")
        req = urllib.request.Request(f"{REST_URL}/{path}", data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    async def _get_ws_token(self) -> Optional[str]:
        """Obtain a WebSocket authentication token from Kraken REST API."""
        if not self.api_key or not self.api_secret:
            logger.warning("KrakenWS: no API credentials — will use public data only")
            return None
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: self._rest_post("GetWebSocketsToken", {}))
            if result.get("error"):
                logger.error("KrakenWS token error: %s", result["error"])
                return None
            token = result.get("result", {}).get("token")
            if token:
                self._token_expiry = time.monotonic() + 25 * 60  # 25 minutes
            logger.info("KrakenWS: obtained auth token")
            return token
        except Exception as exc:
            logger.error("KrakenWS: failed to get WS token: %s", exc)
            return None

    async def _ensure_token_fresh(self) -> None:
        """Re-fetch WS token if within 5 minutes of expiry."""
        if self._token is None:
            return
        time_until_expiry = self._token_expiry - time.monotonic()
        if time_until_expiry <= 5 * 60:  # 5 minutes
            logger.info("KrakenWS: token expiring in %.0fs — refreshing", max(0, time_until_expiry))
            new_token = await self._get_ws_token()
            if new_token:
                self._token = new_token
                logger.info("KrakenWS: token refreshed successfully")
            else:
                logger.warning("KrakenWS: token refresh failed — using existing token")

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Establish authenticated WebSocket connection."""
        try:
            import websockets  # type: ignore[import]
        except ImportError:
            logger.warning("KrakenWS: websockets library not installed — REST fallback only")
            return False

        self._token = await self._get_ws_token()
        if not self._token:
            logger.warning("KrakenWS: no token — REST fallback only")
            return False

        try:
            self._ws = await websockets.connect(
                WS_URL,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            )
            self._connected = True
            self._listener_task = asyncio.create_task(self._listen())
            logger.info("KrakenWS: connected to %s", WS_URL)
            return True
        except Exception as exc:
            logger.error("KrakenWS connect failed: %s", exc)
            self._connected = False
            return False

    async def disconnect(self) -> None:
        self._connected = False
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception as _e:
                logger.debug("websocket_order_placer error: %s", _e)
        logger.info("KrakenWS: disconnected")

    async def _listen(self) -> None:
        """Background task: read and dispatch incoming WS messages."""
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                    self._dispatch(msg)
                except Exception as exc:
                    logger.debug("KrakenWS: malformed message: %s", exc)
        except Exception as exc:
            logger.warning("KrakenWS listener stopped: %s", exc)
            self._connected = False

    def _dispatch(self, msg: Dict[str, Any]) -> None:
        """Route incoming WS message to a pending Future."""
        req_id = str(msg.get("req_id", ""))
        if req_id and req_id in self._pending:
            fut = self._pending.pop(req_id)
            if not fut.done():
                if msg.get("success") is False:
                    fut.set_exception(RuntimeError(str(msg.get("error", "WS order failed"))))
                else:
                    fut.set_result(msg)

    # ------------------------------------------------------------------
    # Order operations
    # ------------------------------------------------------------------

    async def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = "market",
        price: Optional[float] = None,
        timeout: float = 5.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Place an order. Uses WebSocket if connected, falls back to REST.

        Args:
            symbol:     "BTC/USD" or Kraken format "XBT/USD"
            side:       "buy" or "sell"
            amount:     quantity in base currency
            order_type: "market" or "limit"
            price:      limit price (required for limit orders)
            timeout:    seconds to wait for WS confirmation

        Returns:
            Order dict or None on failure.
        """
        # Ensure token is fresh before placing order
        await self._ensure_token_fresh()

        # Normalise symbol for Kraken (BTC -> XBT)
        kraken_symbol = symbol.replace("BTC", "XBT")

        if self._connected and self._ws:
            return await self._ws_place_order(kraken_symbol, side, amount, order_type, price, timeout)

        logger.debug("KrakenWS: not connected — using REST fallback for %s", symbol)
        return await self._rest_place_order(kraken_symbol, side, amount, order_type, price)

    async def _ws_place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str,
        price: Optional[float],
        timeout: float,
    ) -> Optional[Dict[str, Any]]:
        self._req_id += 1
        req_id = str(self._req_id)

        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "order_qty": amount,
            "token": self._token,
        }
        if order_type == "limit" and price is not None:
            params["limit_price"] = price

        msg = {"method": "add_order", "params": params, "req_id": req_id}

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut

        try:
            await self._ws.send(json.dumps(msg))
            result = await asyncio.wait_for(fut, timeout=timeout)
            logger.info("KrakenWS order confirmed: %s %s %s %.6f", side, amount, symbol, price or 0)
            return result
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            logger.warning("KrakenWS order timed out after %.1fs — falling back to REST", timeout)
            return await self._rest_place_order(symbol, side, amount, order_type, price)
        except Exception as exc:
            self._pending.pop(req_id, None)
            logger.error("KrakenWS order error: %s", exc)
            return None

    async def _rest_place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str,
        price: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        """REST fallback order placement."""
        if not self.api_key or not self.api_secret:
            logger.error("KrakenWS REST fallback: no API credentials")
            return None
        try:
            data: Dict[str, str] = {
                "pair": symbol.replace("/", ""),
                "type": side,
                "ordertype": order_type,
                "volume": str(amount),
            }
            if order_type == "limit" and price is not None:
                data["price"] = str(price)

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, lambda: self._rest_post("AddOrder", data)
            )
            if result.get("error"):
                logger.error("Kraken REST order error: %s", result["error"])
                return None
            logger.info("Kraken REST order placed: %s %s %s %.6f", side, amount, symbol, price or 0)
            return result.get("result")
        except Exception as exc:
            logger.error("Kraken REST order failed: %s", exc)
            return None

    async def cancel_order(self, order_id: str, timeout: float = 5.0) -> bool:
        """Cancel an order by ID."""
        if self._connected and self._ws:
            self._req_id += 1
            req_id = str(self._req_id)
            msg = {
                "method": "cancel_order",
                "params": {"order_id": [order_id], "token": self._token},
                "req_id": req_id,
            }
            loop = asyncio.get_running_loop()
            fut: asyncio.Future = loop.create_future()
            self._pending[req_id] = fut
            try:
                await self._ws.send(json.dumps(msg))
                await asyncio.wait_for(fut, timeout=timeout)
                return True
            except Exception as exc:
                self._pending.pop(req_id, None)
                logger.warning("KrakenWS cancel_order failed: %s", exc)
                return False

        # REST fallback cancel
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, lambda: self._rest_post("CancelOrder", {"txid": order_id})
            )
            return not bool(result.get("error"))
        except Exception as exc:
            logger.error("Kraken REST cancel_order failed: %s", exc)
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected
