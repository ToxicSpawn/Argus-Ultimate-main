"""
Kraken WebSocket v2 Connector.

Provides real-time market data via Kraken WS v2 API:
- Ticker updates (best bid/ask, last trade, volume)
- Book updates (snapshots and deltas)
- Auto-reconnect with exponential backoff
- Heartbeat monitoring (reconnect if no message for 30s)

No API keys required for public channels.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import websockets  # type: ignore[import]
    _HAS_WEBSOCKETS = True
except ImportError:
    websockets = None  # type: ignore[assignment]
    _HAS_WEBSOCKETS = False

WS_URL = "wss://ws.kraken.com/v2"
HEARTBEAT_TIMEOUT_S = 30.0
MAX_BACKOFF_S = 30.0
INITIAL_BACKOFF_S = 1.0

# Kraken WS v2 uses standard symbols (BTC/USD, not XBT/USD)
_KRAKEN_SYMBOL_MAP: Dict[str, str] = {}

_KRAKEN_SYMBOL_REVERSE: Dict[str, str] = {v: k for k, v in _KRAKEN_SYMBOL_MAP.items()}


def to_kraken_symbol(symbol: str) -> str:
    """Convert standard symbol to Kraken WS v2 format."""
    return _KRAKEN_SYMBOL_MAP.get(symbol, symbol)


def from_kraken_symbol(symbol: str) -> str:
    """Convert Kraken WS v2 symbol back to standard format."""
    return _KRAKEN_SYMBOL_REVERSE.get(symbol, symbol)


class KrakenWSConnector:
    """Kraken WS v2 connector for real-time market data (public channels).

    Usage::

        connector = KrakenWSConnector(symbols=["BTC/USD", "ETH/USD"])
        connector.on_ticker(my_ticker_handler)
        connector.on_book_update(my_book_handler)
        await connector.start()
        # ... later ...
        await connector.stop()
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        channels: Optional[List[str]] = None,
        ws_url: str = WS_URL,
        book_depth: int = 25,
    ):
        self.symbols = symbols or ["BTC/AUD", "ETH/AUD"]
        self.channels = channels or ["ticker", "book"]
        self.ws_url = ws_url
        self.book_depth = book_depth

        self._ws: Any = None
        self._running = False
        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._last_message_time: float = 0.0
        self._backoff_s: float = INITIAL_BACKOFF_S
        self._reconnect_count: int = 0
        self._lock = threading.Lock()

        # Callbacks
        self._ticker_callbacks: List[Callable[[Dict[str, Any]], Any]] = []
        self._book_callbacks: List[Callable[[Dict[str, Any]], Any]] = []

        self.connected: bool = False

    # -------------------------------------------------------------- callbacks
    def on_ticker(self, callback: Callable[[Dict[str, Any]], Any]) -> None:
        """Register a callback for ticker updates.

        Callback receives: {"symbol": str, "bid": float, "ask": float,
                            "last": float, "volume_24h": float,
                            "timestamp": datetime}
        """
        self._ticker_callbacks.append(callback)
        logger.debug("Registered ticker callback: %s", callback)

    def on_book_update(self, callback: Callable[[Dict[str, Any]], Any]) -> None:
        """Register a callback for book updates (snapshots and deltas).

        Callback receives: {"symbol": str, "type": "snapshot"|"update",
                            "bids": list, "asks": list, "timestamp": datetime}
        """
        self._book_callbacks.append(callback)
        logger.debug("Registered book update callback: %s", callback)

    # --------------------------------------------------------- message parsing
    def _parse_ticker(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse a ticker channel message into standardized dicts."""
        results = []
        try:
            entries = data.get("data", [])
            for t in entries:
                symbol = from_kraken_symbol(t.get("symbol", ""))
                results.append({
                    "symbol": symbol,
                    "bid": float(t.get("bid", 0) or 0),
                    "ask": float(t.get("ask", 0) or 0),
                    "last": float(t.get("last", 0) or 0),
                    "volume_24h": float(t.get("volume", 0) or 0),
                    "timestamp": datetime.now(timezone.utc),
                })
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            logger.warning("Failed to parse Kraken ticker: %s", exc)
        return results

    def _parse_book(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse a book channel message (snapshot or delta)."""
        results = []
        try:
            msg_type = data.get("type", "update")
            entries = data.get("data", [])
            for entry in entries:
                symbol = from_kraken_symbol(entry.get("symbol", ""))
                results.append({
                    "symbol": symbol,
                    "type": "snapshot" if msg_type == "snapshot" else "update",
                    "bids": entry.get("bids", []),
                    "asks": entry.get("asks", []),
                    "timestamp": datetime.now(timezone.utc),
                })
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            logger.warning("Failed to parse Kraken book: %s", exc)
        return results

    async def _dispatch(self, raw: str) -> None:
        """Parse and dispatch a raw WebSocket message."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Non-JSON WS message: %.120s", raw)
            return

        channel = data.get("channel", "")

        if channel == "ticker":
            for parsed in self._parse_ticker(data):
                for cb in self._ticker_callbacks:
                    try:
                        result = cb(parsed)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as exc:
                        logger.error("Ticker callback error: %s", exc)

        elif channel == "book":
            for parsed in self._parse_book(data):
                for cb in self._book_callbacks:
                    try:
                        result = cb(parsed)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as exc:
                        logger.error("Book callback error: %s", exc)

        elif channel == "heartbeat":
            pass  # Kraken heartbeat, just update last_message_time
        elif channel == "status":
            logger.info("Kraken WS status: %s", data.get("data", ""))
        elif data.get("method") == "subscribe":
            if data.get("success"):
                logger.info("Kraken subscription confirmed: %s", data)
            else:
                logger.error("Kraken subscription failed: %s", data.get("error", "unknown"))
        else:
            logger.debug("Unhandled Kraken WS message: %.200s", raw)

    # ---------------------------------------------------------- connection
    async def _connect(self) -> bool:
        """Connect to Kraken WS v2 and subscribe to channels."""
        if not _HAS_WEBSOCKETS:
            logger.warning("websockets library not installed — Kraken WS unavailable")
            return False

        try:
            logger.info("Connecting to Kraken WS v2: %s", self.ws_url)
            self._ws = await websockets.connect(self.ws_url, ping_interval=20)  # type: ignore[union-attr]

            kraken_symbols = [to_kraken_symbol(s) for s in self.symbols]

            # Subscribe to each channel
            for channel in self.channels:
                params: Dict[str, Any] = {
                    "channel": channel,
                    "symbol": kraken_symbols,
                }
                if channel == "book":
                    params["depth"] = self.book_depth

                msg = {"method": "subscribe", "params": params}
                await self._ws.send(json.dumps(msg))
                logger.info("Subscribed to Kraken %s for %s", channel, kraken_symbols)

            self.connected = True
            self._last_message_time = time.monotonic()
            self._backoff_s = INITIAL_BACKOFF_S
            self._reconnect_count = 0
            logger.info("Kraken WS connected successfully")
            return True

        except Exception as exc:
            logger.error("Kraken WS connect failed: %s", exc)
            self.connected = False
            return False

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff."""
        self.connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception as exc:
                logger.debug("WS close error during reconnect: %s", exc)
            self._ws = None

        self._reconnect_count += 1
        wait = min(self._backoff_s, MAX_BACKOFF_S)
        logger.warning(
            "Kraken WS reconnecting in %.1fs (attempt %d)", wait, self._reconnect_count
        )
        await asyncio.sleep(wait)
        self._backoff_s = min(self._backoff_s * 2, MAX_BACKOFF_S)
        await self._connect()

    # ---------------------------------------------------------- lifecycle
    async def start(self, symbols: Optional[List[str]] = None) -> bool:
        """Start the connector — connect and run receive + heartbeat loops.

        Args:
            symbols: Override symbols list (optional, uses constructor list if not given).

        Returns:
            True if initial connection succeeded.
        """
        if symbols:
            self.symbols = list(symbols)

        self._running = True
        success = await self._connect()

        self._receive_task = asyncio.create_task(self._receive_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        return success

    async def stop(self) -> None:
        """Gracefully stop the connector."""
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
            except Exception as exc:
                logger.debug("WS close error: %s", exc)
            self._ws = None
        self.connected = False
        logger.info("Kraken WS connector stopped")

    # ---------------------------------------------------------- run loops
    async def _receive_loop(self) -> None:
        """Main receive loop."""
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
                logger.warning("Kraken WS receive error: %s", exc)
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
                        "No Kraken WS message for %.0fs, reconnecting", elapsed
                    )
                    await self._reconnect()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Kraken heartbeat loop error: %s", exc)

    # ------------------------------------------------ async context manager
    async def __aenter__(self) -> "KrakenWSConnector":
        self._running = True
        await self._connect()
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()
