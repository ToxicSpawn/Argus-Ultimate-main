"""
Coinbase Advanced Trade WebSocket Connector.

Provides real-time market data via Coinbase Advanced Trade WebSocket API:
- Ticker updates (best bid/ask, last trade, 24h volume)
- Level 2 order book updates
- Auto-reconnect with exponential backoff
- Heartbeat monitoring (reconnect if no message for 30s)

API keys: Set COINBASE_API_KEY / COINBASE_API_SECRET in environment.
Authentication uses HMAC-SHA256 signing per Coinbase Advanced Trade spec.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

WS_URL = "wss://advanced-trade-ws.coinbase.com"
HEARTBEAT_TIMEOUT_S = 30.0
MAX_BACKOFF_S = 30.0
INITIAL_BACKOFF_S = 1.0


class CoinbaseWSConnector:
    """Coinbase Advanced Trade WebSocket connector for real-time market data."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        symbols: Optional[List[str]] = None,
        channels: Optional[List[str]] = None,
        ws_url: str = WS_URL,
    ):
        self.api_key = api_key or os.environ.get("COINBASE_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("COINBASE_API_SECRET", "")
        self.symbols = symbols or ["BTC-AUD", "ETH-AUD"]
        self.channels = channels or ["ticker", "level2"]
        self.ws_url = ws_url

        self._ws: Any = None
        self._running = False
        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._last_message_time: float = 0.0
        self._backoff_s: float = INITIAL_BACKOFF_S
        self._reconnect_count: int = 0

        # Callbacks
        self._ticker_callbacks: List[Callable[[Dict[str, Any]], Any]] = []
        self._l2_callbacks: List[Callable[[Dict[str, Any]], Any]] = []

        self.connected: bool = False

    # ------------------------------------------------------------------ auth
    def _sign(self, timestamp: str, channel: str, product_ids: List[str]) -> str:
        """Create HMAC-SHA256 signature for Coinbase Advanced Trade WS auth."""
        message = f"{timestamp}{channel}{','.join(product_ids)}"
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def _build_subscribe_msg(self) -> Dict[str, Any]:
        """Build the subscription message with authentication."""
        timestamp = str(int(time.time()))
        # Sign each channel separately; Coinbase expects one subscribe per batch
        channel = ",".join(self.channels)
        sig = self._sign(timestamp, channel, self.symbols)

        return {
            "type": "subscribe",
            "product_ids": self.symbols,
            "channel": self.channels[0] if len(self.channels) == 1 else self.channels[0],
            "api_key": self.api_key,
            "timestamp": timestamp,
            "signature": sig,
        }

    def _build_subscribe_messages(self) -> List[Dict[str, Any]]:
        """Build one subscribe message per channel (Coinbase requires separate subs)."""
        messages = []
        timestamp = str(int(time.time()))
        for channel in self.channels:
            sig = self._sign(timestamp, channel, self.symbols)
            messages.append({
                "type": "subscribe",
                "product_ids": self.symbols,
                "channel": channel,
                "api_key": self.api_key,
                "timestamp": timestamp,
                "signature": sig,
            })
        return messages

    # -------------------------------------------------------------- callbacks
    def on_ticker(self, callback: Callable[[Dict[str, Any]], Any]) -> None:
        """Register a callback for ticker updates.

        Callback receives: {"symbol": str, "bid": float, "ask": float,
                            "last": float, "volume_24h": float,
                            "timestamp": datetime}
        """
        self._ticker_callbacks.append(callback)
        logger.debug("Registered ticker callback: %s", callback)

    def on_l2_update(self, callback: Callable[[Dict[str, Any]], Any]) -> None:
        """Register a callback for level2 order book updates.

        Callback receives the parsed L2 update dict.
        """
        self._l2_callbacks.append(callback)
        logger.debug("Registered L2 update callback: %s", callback)

    # --------------------------------------------------------- message parsing
    def _parse_ticker(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a ticker event into standardized format."""
        try:
            events = data.get("events", [])
            for event in events:
                tickers = event.get("tickers", [])
                for t in tickers:
                    return {
                        "symbol": t.get("product_id", ""),
                        "bid": float(t.get("best_bid", 0) or 0),
                        "ask": float(t.get("best_ask", 0) or 0),
                        "last": float(t.get("price", 0) or 0),
                        "volume_24h": float(t.get("volume_24_h", 0) or 0),
                        "timestamp": datetime.now(timezone.utc),
                    }
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            logger.warning("Failed to parse ticker message: %s", exc)
        return None

    def _parse_l2_update(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a level2 event into standardized format."""
        try:
            events = data.get("events", [])
            for event in events:
                updates = event.get("updates", [])
                return {
                    "symbol": data.get("product_id", ""),
                    "type": event.get("type", "update"),
                    "updates": [
                        {
                            "side": u.get("side", ""),
                            "price": float(u.get("price_level", 0) or 0),
                            "qty": float(u.get("new_quantity", 0) or 0),
                        }
                        for u in updates
                    ],
                    "timestamp": datetime.now(timezone.utc),
                }
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning("Failed to parse L2 message: %s", exc)
        return None

    async def _dispatch(self, raw: str) -> None:
        """Parse and dispatch a raw WebSocket message."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Non-JSON WebSocket message: %.120s", raw)
            return

        channel = data.get("channel", "")

        if channel == "ticker":
            parsed = self._parse_ticker(data)
            if parsed:
                for cb in self._ticker_callbacks:
                    try:
                        result = cb(parsed)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as exc:
                        logger.error("Ticker callback error: %s", exc)

        elif channel == "l2_data":
            parsed = self._parse_l2_update(data)
            if parsed:
                for cb in self._l2_callbacks:
                    try:
                        result = cb(parsed)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as exc:
                        logger.error("L2 callback error: %s", exc)

        elif channel == "subscriptions":
            logger.info("Subscription confirmed: %s", data)
        elif channel == "heartbeats":
            pass  # heartbeat from server
        else:
            logger.debug("Unhandled channel '%s': %.200s", channel, raw)

    # ---------------------------------------------------------- connection
    async def connect(self) -> bool:
        """Connect to Coinbase WS and subscribe to channels."""
        try:
            import websockets  # type: ignore[import]

            logger.info("Connecting to Coinbase WS: %s", self.ws_url)
            self._ws = await websockets.connect(self.ws_url, ping_interval=20)

            # Subscribe to all channels
            for msg in self._build_subscribe_messages():
                await self._ws.send(json.dumps(msg))
                logger.info("Subscribed to channel=%s symbols=%s", msg["channel"], msg["product_ids"])

            self.connected = True
            self._last_message_time = time.monotonic()
            self._backoff_s = INITIAL_BACKOFF_S
            self._reconnect_count = 0
            logger.info("Coinbase WS connected successfully")
            return True

        except Exception as exc:
            logger.error("Coinbase WS connect failed: %s", exc)
            self.connected = False
            return False

    async def disconnect(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._running = False
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.debug("Callback error in %s: %s", self._ws.close.__name__ if hasattr(self._ws.close, '__name__') else 'ws_close', e)
            self._ws = None
        self.connected = False
        logger.info("Coinbase WS disconnected")

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff."""
        self.connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.debug("Callback error in %s: %s", self._ws.close.__name__ if hasattr(self._ws.close, '__name__') else 'ws_close', e)
            self._ws = None

        self._reconnect_count += 1
        wait = min(self._backoff_s, MAX_BACKOFF_S)
        logger.warning(
            "Coinbase WS reconnecting in %.1fs (attempt %d)",
            wait, self._reconnect_count,
        )
        await asyncio.sleep(wait)
        self._backoff_s = min(self._backoff_s * 2, MAX_BACKOFF_S)

        await self.connect()

    # ---------------------------------------------------------- run loop
    async def _receive_loop(self) -> None:
        """Main receive loop — reads messages and dispatches."""
        while self._running:
            if not self._ws or not self.connected:
                await self._reconnect()
                if not self.connected:
                    continue

            try:
                raw = await self._ws.recv()
                self._last_message_time = time.monotonic()
                await self._dispatch(raw)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Coinbase WS receive error: %s", exc)
                if self._running:
                    await self._reconnect()

    async def _heartbeat_loop(self) -> None:
        """Monitor for stale connections — reconnect if no message for 30s."""
        while self._running:
            try:
                await asyncio.sleep(5.0)
                if not self.connected:
                    continue
                elapsed = time.monotonic() - self._last_message_time
                if elapsed > HEARTBEAT_TIMEOUT_S:
                    logger.warning(
                        "No Coinbase WS message for %.0fs, reconnecting", elapsed
                    )
                    await self._reconnect()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Heartbeat loop error: %s", exc)

    async def run(self) -> None:
        """Start the connector — connect and run receive + heartbeat loops."""
        self._running = True
        if not self.connected:
            await self.connect()

        self._receive_task = asyncio.create_task(self._receive_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        try:
            await asyncio.gather(self._receive_task, self._heartbeat_task)
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------ async context manager
    async def __aenter__(self) -> "CoinbaseWSConnector":
        self._running = True
        if not self.connected:
            await self.connect()
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.disconnect()
