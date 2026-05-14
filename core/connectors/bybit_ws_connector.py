"""
Bybit WebSocket v5 Connector (Public Linear/USDT Perpetual).

Provides real-time market data via Bybit WS v5 public API:
- Ticker updates (best bid/ask, last trade, 24h volume)
- Orderbook updates (snapshots and deltas, depth 25)
- Auto-reconnect with exponential backoff
- Ping/pong handling (Bybit sends ping, we respond with pong)

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

WS_URL = "wss://stream.bybit.com/v5/public/linear"
HEARTBEAT_TIMEOUT_S = 30.0
MAX_BACKOFF_S = 30.0
INITIAL_BACKOFF_S = 1.0
PING_INTERVAL_S = 20.0

# Bybit uses BTCUSDT format, not BTC/USD
_BYBIT_SYMBOL_MAP: Dict[str, str] = {
    "BTC/USD": "BTCUSDT",
    "BTC/USDT": "BTCUSDT",
    "ETH/USD": "ETHUSDT",
    "ETH/USDT": "ETHUSDT",
    "SOL/USD": "SOLUSDT",
    "SOL/USDT": "SOLUSDT",
    "XRP/USD": "XRPUSDT",
    "XRP/USDT": "XRPUSDT",
    "ADA/USD": "ADAUSDT",
    "ADA/USDT": "ADAUSDT",
    "DOGE/USD": "DOGEUSDT",
    "DOGE/USDT": "DOGEUSDT",
    "AVAX/USD": "AVAXUSDT",
    "AVAX/USDT": "AVAXUSDT",
    "DOT/USD": "DOTUSDT",
    "DOT/USDT": "DOTUSDT",
    "LINK/USD": "LINKUSDT",
    "LINK/USDT": "LINKUSDT",
}

_BYBIT_SYMBOL_REVERSE: Dict[str, str] = {}
# Build reverse map — prefer /USDT form
for _std, _bybit in _BYBIT_SYMBOL_MAP.items():
    if _bybit not in _BYBIT_SYMBOL_REVERSE or _std.endswith("/USDT"):
        _BYBIT_SYMBOL_REVERSE[_bybit] = _std


def to_bybit_symbol(symbol: str) -> str:
    """Convert standard symbol (BTC/USD, BTC/USDT) to Bybit format (BTCUSDT).

    If not in the map, strip '/' and append 'USDT' if needed.
    """
    if symbol in _BYBIT_SYMBOL_MAP:
        return _BYBIT_SYMBOL_MAP[symbol]
    # Best-effort: remove slash
    cleaned = symbol.replace("/", "")
    if not cleaned.endswith("USDT") and not cleaned.endswith("USD"):
        cleaned += "USDT"
    elif cleaned.endswith("USD") and not cleaned.endswith("USDT"):
        cleaned = cleaned[:-3] + "USDT"
    return cleaned


def from_bybit_symbol(symbol: str) -> str:
    """Convert Bybit format (BTCUSDT) back to standard (BTC/USDT)."""
    if symbol in _BYBIT_SYMBOL_REVERSE:
        return _BYBIT_SYMBOL_REVERSE[symbol]
    # Best-effort: insert slash before USDT
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        return f"{base}/USDT"
    return symbol


class BybitWSConnector:
    """Bybit WS v5 connector for real-time market data (public linear channels).

    Usage::

        connector = BybitWSConnector(symbols=["BTC/USDT", "ETH/USDT"])
        connector.on_ticker(my_ticker_handler)
        connector.on_book_update(my_book_handler)
        await connector.start()
        # ... later ...
        await connector.stop()
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        ws_url: str = WS_URL,
        book_depth: int = 25,
    ):
        self.symbols = symbols or ["BTC/USDT", "ETH/USDT"]
        self.topics = topics or ["tickers", "orderbook.25"]
        self.ws_url = ws_url
        self.book_depth = book_depth

        self._ws: Any = None
        self._running = False
        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
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
        logger.debug("Registered Bybit ticker callback: %s", callback)

    def on_book_update(self, callback: Callable[[Dict[str, Any]], Any]) -> None:
        """Register a callback for orderbook updates.

        Callback receives: {"symbol": str, "type": "snapshot"|"delta",
                            "bids": list, "asks": list, "timestamp": datetime}
        """
        self._book_callbacks.append(callback)
        logger.debug("Registered Bybit book update callback: %s", callback)

    # --------------------------------------------------------- message parsing
    def _parse_ticker(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a tickers topic message into standardized format."""
        try:
            d = data.get("data", {})
            symbol = from_bybit_symbol(d.get("symbol", ""))
            return {
                "symbol": symbol,
                "bid": float(d.get("bid1Price", 0) or 0),
                "ask": float(d.get("ask1Price", 0) or 0),
                "last": float(d.get("lastPrice", 0) or 0),
                "volume_24h": float(d.get("volume24h", 0) or 0),
                "timestamp": datetime.now(timezone.utc),
            }
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            logger.warning("Failed to parse Bybit ticker: %s", exc)
            return None

    def _parse_book(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse an orderbook topic message (snapshot or delta)."""
        try:
            d = data.get("data", {})
            symbol = from_bybit_symbol(d.get("s", ""))
            msg_type = data.get("type", "delta")

            # Bybit sends bids/asks as [[price, qty], ...]
            bids = [{"price": float(b[0]), "qty": float(b[1])} for b in d.get("b", [])]
            asks = [{"price": float(a[0]), "qty": float(a[1])} for a in d.get("a", [])]

            return {
                "symbol": symbol,
                "type": "snapshot" if msg_type == "snapshot" else "update",
                "bids": bids,
                "asks": asks,
                "timestamp": datetime.now(timezone.utc),
            }
        except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
            logger.warning("Failed to parse Bybit book: %s", exc)
            return None

    async def _dispatch(self, raw: str) -> None:
        """Parse and dispatch a raw WebSocket message."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Non-JSON Bybit WS message: %.120s", raw)
            return

        # Bybit ping: respond with pong
        if data.get("op") == "ping" or data.get("ret_msg") == "pong":
            return
        if "ret_msg" in data and data.get("op") == "pong":
            return

        # Subscription confirmations
        if data.get("op") == "subscribe":
            if data.get("success"):
                logger.info("Bybit subscription confirmed: %s", data.get("ret_msg", ""))
            else:
                logger.error("Bybit subscription failed: %s", data.get("ret_msg", "unknown"))
            return

        topic = data.get("topic", "")

        if topic.startswith("tickers."):
            parsed = self._parse_ticker(data)
            if parsed:
                for cb in self._ticker_callbacks:
                    try:
                        result = cb(parsed)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as exc:
                        logger.error("Bybit ticker callback error: %s", exc)

        elif topic.startswith("orderbook."):
            parsed = self._parse_book(data)
            if parsed:
                for cb in self._book_callbacks:
                    try:
                        result = cb(parsed)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as exc:
                        logger.error("Bybit book callback error: %s", exc)
        else:
            logger.debug("Unhandled Bybit WS topic '%s': %.200s", topic, raw)

    # ---------------------------------------------------------- connection
    async def _connect(self) -> bool:
        """Connect to Bybit WS and subscribe to topics."""
        if not _HAS_WEBSOCKETS:
            logger.warning("websockets library not installed — Bybit WS unavailable")
            return False

        try:
            logger.info("Connecting to Bybit WS: %s", self.ws_url)
            self._ws = await websockets.connect(self.ws_url, ping_interval=None)  # type: ignore[union-attr]

            # Build subscription args: "tickers.BTCUSDT", "orderbook.25.BTCUSDT"
            bybit_symbols = [to_bybit_symbol(s) for s in self.symbols]
            args: List[str] = []
            for topic_base in self.topics:
                for sym in bybit_symbols:
                    args.append(f"{topic_base}.{sym}")

            subscribe_msg = {
                "op": "subscribe",
                "args": args,
            }
            await self._ws.send(json.dumps(subscribe_msg))
            logger.info("Subscribed to Bybit topics: %s", args)

            self.connected = True
            self._last_message_time = time.monotonic()
            self._backoff_s = INITIAL_BACKOFF_S
            self._reconnect_count = 0
            logger.info("Bybit WS connected successfully")
            return True

        except Exception as exc:
            logger.error("Bybit WS connect failed: %s", exc)
            self.connected = False
            return False

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff."""
        self.connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception as exc:
                logger.debug("Bybit WS close error during reconnect: %s", exc)
            self._ws = None

        self._reconnect_count += 1
        wait = min(self._backoff_s, MAX_BACKOFF_S)
        logger.warning(
            "Bybit WS reconnecting in %.1fs (attempt %d)", wait, self._reconnect_count
        )
        await asyncio.sleep(wait)
        self._backoff_s = min(self._backoff_s * 2, MAX_BACKOFF_S)
        await self._connect()

    # ---------------------------------------------------------- lifecycle
    async def start(self, symbols: Optional[List[str]] = None) -> bool:
        """Start the connector.

        Args:
            symbols: Override symbols list (optional).

        Returns:
            True if initial connection succeeded.
        """
        if symbols:
            self.symbols = list(symbols)

        self._running = True
        success = await self._connect()

        self._receive_task = asyncio.create_task(self._receive_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._ping_task = asyncio.create_task(self._ping_loop())

        return success

    async def stop(self) -> None:
        """Gracefully stop the connector."""
        self._running = False
        for task in (self._receive_task, self._heartbeat_task, self._ping_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception as exc:
                logger.debug("Bybit WS close error: %s", exc)
            self._ws = None
        self.connected = False
        logger.info("Bybit WS connector stopped")

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

                # Handle Bybit ping frame (text: '{"op":"ping", ...}' or raw ping)
                try:
                    maybe_ping = json.loads(raw)
                    if maybe_ping.get("op") == "ping":
                        pong = json.dumps({"op": "pong", "args": [str(int(time.time() * 1000))]})
                        await self._ws.send(pong)
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass

                await self._dispatch(raw)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Bybit WS receive error: %s", exc)
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
                        "No Bybit WS message for %.0fs, reconnecting", elapsed
                    )
                    await self._reconnect()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Bybit heartbeat loop error: %s", exc)

    async def _ping_loop(self) -> None:
        """Send periodic pings to keep connection alive.

        Bybit requires the client to send ping messages to prevent disconnection.
        """
        while self._running:
            try:
                await asyncio.sleep(PING_INTERVAL_S)
                if self._ws and self.connected:
                    ping_msg = json.dumps({"op": "ping"})
                    await self._ws.send(ping_msg)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Bybit ping error: %s", exc)

    # ------------------------------------------------ async context manager
    async def __aenter__(self) -> "BybitWSConnector":
        self._running = True
        await self._connect()
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._ping_task = asyncio.create_task(self._ping_loop())
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()
