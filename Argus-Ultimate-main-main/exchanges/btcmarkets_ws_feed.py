"""
BTC Markets WebSocket Feed
===========================

Real-time market data feed for BTC Markets via their WebSocket v2 API.

Features:
- Live orderbook snapshots (full depth)
- Live trade feed
- Private user-data channel (order fills, fund changes)  — requires API keys
- Local best-bid/ask cache per symbol
- Spread-in-basis-points helper
- Auto-reconnect with exponential backoff
- Heartbeat monitoring

BTC Markets WS v2:
  URL:  wss://socket.btcmarkets.net/v2
  Subscribe message::

    {
      "marketIds": ["BTC-AUD", "ETH-AUD"],
      "channels":  ["orderbook", "trade"],
      "messageType": "subscribe"
    }

  Orderbook message::

    {
      "messageType": "orderbook",
      "marketId":    "BTC-AUD",
      "timestamp":   "2024-01-15T10:00:00.000000Z",
      "snapshotId":  123456789,
      "bids":        [["95000.00", "0.5"], ...],   # price, qty strings
      "asks":        [["95001.00", "0.3"], ...]
    }

  Trade message::

    {
      "messageType": "trade",
      "marketId":    "BTC-AUD",
      "timestamp":   "2024-01-15T10:00:01.000000Z",
      "trades": [
        {"tradeId": "abc123", "price": "95000.50", "amount": "0.1", "side": "Ask"}
      ]
    }

  Private auth (include in subscribe message for private channels)::

    {
      "key":       "<api_key>",
      "timestamp": "<epoch_ms>",
      "signature": "<base64_hmac_sha256>",
      ...
    }

Author: Argus Trading System
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import websockets  # type: ignore[import]
    _HAS_WEBSOCKETS = True
except ImportError:
    websockets = None  # type: ignore[assignment]
    _HAS_WEBSOCKETS = False

# WebSocket endpoint
BTCM_WS_URL: str = "wss://socket.btcmarkets.net/v2"

# Timing constants
HEARTBEAT_TIMEOUT_S: float = 30.0
INITIAL_BACKOFF_S:   float = 1.0
MAX_BACKOFF_S:       float = 30.0
PING_INTERVAL_S:     float = 20.0


# ---------------------------------------------------------------------------
# Local orderbook structure
# ---------------------------------------------------------------------------

class _LocalBook:
    """Thread-safe local orderbook cache for a single symbol."""

    __slots__ = ("symbol", "_bids", "_asks", "_last_update", "_lock")

    def __init__(self, symbol: str) -> None:
        self.symbol   = symbol
        self._bids:   Dict[float, float] = {}   # price -> qty
        self._asks:   Dict[float, float] = {}
        self._last_update: float = 0.0
        self._lock    = asyncio.Lock()

    async def apply_snapshot(
        self,
        bids: List[List[Any]],
        asks: List[List[Any]],
    ) -> None:
        """Replace local book with a fresh snapshot."""
        async with self._lock:
            self._bids = {float(b[0]): float(b[1]) for b in bids}
            self._asks = {float(a[0]): float(a[1]) for a in asks}
            # Purge zero-quantity levels
            self._bids = {p: q for p, q in self._bids.items() if q > 0}
            self._asks = {p: q for p, q in self._asks.items() if q > 0}
            self._last_update = time.monotonic()

    @property
    def best_bid(self) -> Optional[float]:
        return max(self._bids.keys(), default=None)

    @property
    def best_ask(self) -> Optional[float]:
        return min(self._asks.keys(), default=None)

    @property
    def spread_bps(self) -> float:
        """Return the bid-ask spread in basis points."""
        bid = self.best_bid
        ask = self.best_ask
        if bid is None or ask is None or bid <= 0:
            return float("inf")
        mid = (bid + ask) / 2.0
        return ((ask - bid) / mid) * 10_000.0

    def sorted_bids(self, limit: int = 20) -> List[Tuple[float, float]]:
        """Return bids sorted descending by price."""
        return sorted(self._bids.items(), key=lambda x: -x[0])[:limit]

    def sorted_asks(self, limit: int = 20) -> List[Tuple[float, float]]:
        """Return asks sorted ascending by price."""
        return sorted(self._asks.items(), key=lambda x: x[0])[:limit]


# ---------------------------------------------------------------------------
# Main feed class
# ---------------------------------------------------------------------------

class BTCMarketsWSFeed:
    """BTC Markets WebSocket v2 feed.

    Maintains a live local orderbook for each subscribed symbol and fires
    user-supplied callbacks on every update.

    Usage::

        async def on_book(msg):
            feed = BTCMarketsWSFeed(["BTC-AUD"])  # already started
            print(feed.get_best_bid("BTC-AUD"))

        feed = BTCMarketsWSFeed(
            symbols=["BTC-AUD", "ETH-AUD"],
            on_book_update=on_book,
            on_trade=lambda t: print(t),
        )
        await feed.start()

    Args:
        symbols:        List of BTC Markets symbols ("BTC-AUD", …)
        on_book_update: Callback fired on every orderbook message
        on_trade:       Callback fired on every trade message (optional)
        api_key:        BTC Markets API key (for private channels)
        api_secret:     BTC Markets API secret (base-64 encoded)
        ws_url:         Override WebSocket URL (for testing)
    """

    def __init__(
        self,
        symbols: List[str],
        on_book_update: Callable[[Dict[str, Any]], Any],
        on_trade: Optional[Callable[[Dict[str, Any]], Any]] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        ws_url: str = BTCM_WS_URL,
    ) -> None:
        self.symbols:        List[str]   = [s.upper().replace("/", "-") for s in symbols]
        self.on_book_update: Callable[[Dict[str, Any]], Any] = on_book_update
        self.on_trade:       Optional[Callable[[Dict[str, Any]], Any]] = on_trade
        self.api_key:        Optional[str] = api_key
        self.api_secret:     Optional[str] = api_secret
        self.ws_url:         str           = ws_url

        # One local book per symbol
        self._books: Dict[str, _LocalBook] = {
            sym: _LocalBook(sym) for sym in self.symbols
        }

        # WebSocket state
        self._ws:            Any                    = None
        self._running:       bool                   = False
        self._connected:     bool                   = False
        self._last_msg_time: float                  = 0.0
        self._backoff_s:     float                  = INITIAL_BACKOFF_S
        self._reconnect_cnt: int                    = 0
        self._recv_task:     Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._hb_task:       Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._ping_task:     Optional[asyncio.Task] = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_best_bid(self, symbol: str) -> float:
        """Return best (highest) bid price for *symbol*, or 0.0 if unknown."""
        sym = symbol.upper().replace("/", "-")
        book = self._books.get(sym)
        if book is None:
            return 0.0
        bid = book.best_bid
        return bid if bid is not None else 0.0

    def get_best_ask(self, symbol: str) -> float:
        """Return best (lowest) ask price for *symbol*, or 0.0 if unknown."""
        sym = symbol.upper().replace("/", "-")
        book = self._books.get(sym)
        if book is None:
            return 0.0
        ask = book.best_ask
        return ask if ask is not None else 0.0

    def get_spread_bps(self, symbol: str) -> float:
        """Return bid-ask spread in basis points for *symbol*.

        Returns float('inf') if the book is empty.
        """
        sym = symbol.upper().replace("/", "-")
        book = self._books.get(sym)
        if book is None:
            return float("inf")
        return book.spread_bps

    def get_order_book_snapshot(
        self, symbol: str, depth: int = 20
    ) -> Dict[str, Any]:
        """Return a snapshot of the local order book.

        Returns:
            {"symbol": str, "bids": [[price, qty], ...], "asks": [[price, qty], ...]}
        """
        sym  = symbol.upper().replace("/", "-")
        book = self._books.get(sym)
        if book is None:
            return {"symbol": sym, "bids": [], "asks": []}
        return {
            "symbol": sym,
            "bids":   [[p, q] for p, q in book.sorted_bids(depth)],
            "asks":   [[p, q] for p, q in book.sorted_asks(depth)],
        }

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _sign_ws(self) -> Dict[str, str]:
        """Build auth payload for private WS channel subscription."""
        if not self.api_key or not self.api_secret:
            return {}
        ts = str(int(time.time() * 1000))
        message = "GET/users/self/subscribe" + ts
        secret_bytes = base64.b64decode(self.api_secret)
        sig = hmac.new(secret_bytes, message.encode("utf-8"), hashlib.sha256).digest()
        signature = base64.b64encode(sig).decode("utf-8")
        return {
            "key":       self.api_key,
            "timestamp": ts,
            "signature": signature,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect to BTC Markets WebSocket and start processing messages.

        Spawns background tasks for receiving messages, heartbeat
        monitoring, and keep-alive pings.  Returns after the initial
        connection attempt (successful or not — reconnect runs in background).
        """
        if not _HAS_WEBSOCKETS:
            raise RuntimeError(
                "websockets library is required: pip install websockets"
            )
        self._running = True
        await self._connect()
        self._recv_task = asyncio.create_task(self._receive_loop())
        self._hb_task   = asyncio.create_task(self._heartbeat_loop())
        self._ping_task = asyncio.create_task(self._ping_loop())

    async def stop(self) -> None:
        """Gracefully stop the feed and clean up resources."""
        self._running = False
        for task in (self._recv_task, self._hb_task, self._ping_task):
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
                logger.debug("BTCMarketsWSFeed close error: %s", exc)
            self._ws = None
        self._connected = False
        logger.info("BTCMarketsWSFeed stopped")

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def _connect(self) -> bool:
        """Open WebSocket connection and send subscribe message."""
        try:
            logger.info("BTCMarketsWSFeed connecting to %s", self.ws_url)
            self._ws = await websockets.connect(  # type: ignore[union-attr]
                self.ws_url,
                ping_interval=None,  # we manage our own pings
                close_timeout=5,
            )

            # Build subscription message
            channels = ["orderbook"]
            if self.on_trade is not None:
                channels.append("trade")

            sub_msg: Dict[str, Any] = {
                "marketIds":   self.symbols,
                "channels":    channels,
                "messageType": "subscribe",
            }

            # Add auth if private channels are requested and credentials present
            if self.api_key and self.api_secret:
                sub_msg.update(self._sign_ws())

            await self._ws.send(json.dumps(sub_msg))
            logger.info(
                "BTCMarketsWSFeed subscribed to %s for symbols %s",
                channels, self.symbols,
            )

            self._connected      = True
            self._last_msg_time  = time.monotonic()
            self._backoff_s      = INITIAL_BACKOFF_S
            self._reconnect_cnt  = 0
            return True

        except Exception as exc:
            logger.error("BTCMarketsWSFeed connect failed: %s", exc)
            self._connected = False
            return False

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff."""
        self._connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        self._reconnect_cnt += 1
        wait = min(self._backoff_s, MAX_BACKOFF_S)
        logger.warning(
            "BTCMarketsWSFeed reconnecting in %.1fs (attempt %d)",
            wait, self._reconnect_cnt,
        )
        await asyncio.sleep(wait)
        self._backoff_s = min(self._backoff_s * 2, MAX_BACKOFF_S)
        await self._connect()

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, raw: str) -> None:
        """Parse and dispatch a raw WebSocket message."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("BTCMarketsWSFeed non-JSON: %.120s", raw)
            return

        self._last_msg_time = time.monotonic()
        msg_type = msg.get("messageType", "")

        if msg_type == "heartbeat":
            return
        if msg_type == "error":
            logger.error("BTCMarketsWSFeed server error: %s", msg)
            return
        if msg_type == "subscribeConfirmation":
            logger.info(
                "BTCMarketsWSFeed subscription confirmed for %s",
                msg.get("marketIds", []),
            )
            return

        if msg_type == "orderbook":
            await self._handle_orderbook(msg)
        elif msg_type == "trade":
            await self._handle_trade(msg)
        elif msg_type in ("fundChange", "orderChange"):
            await self._handle_user_data(msg)
        else:
            logger.debug(
                "BTCMarketsWSFeed unhandled messageType '%s'", msg_type
            )

    async def _handle_orderbook(self, msg: Dict[str, Any]) -> None:
        """Process an orderbook snapshot message."""
        market_id = msg.get("marketId", "")
        bids = msg.get("bids", [])
        asks = msg.get("asks", [])

        # Ensure the book exists (handles dynamic symbols)
        if market_id not in self._books:
            self._books[market_id] = _LocalBook(market_id)

        await self._books[market_id].apply_snapshot(bids, asks)
        book = self._books[market_id]

        # Build normalised book dict for callback
        normalised: Dict[str, Any] = {
            "messageType": "orderbook",
            "symbol":      market_id,
            "bids":        [[float(b[0]), float(b[1])] for b in bids],
            "asks":        [[float(a[0]), float(a[1])] for a in asks],
            "best_bid":    book.best_bid or 0.0,
            "best_ask":    book.best_ask or 0.0,
            "spread_bps":  book.spread_bps,
            "snapshot_id": msg.get("snapshotId"),
            "timestamp":   msg.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "exchange":    "btcmarkets",
        }

        try:
            result = self.on_book_update(normalised)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            logger.error("BTCMarketsWSFeed on_book_update callback error: %s", exc)

    async def _handle_trade(self, msg: Dict[str, Any]) -> None:
        """Process a trade message."""
        if self.on_trade is None:
            return

        market_id = msg.get("marketId", "")
        ts        = msg.get("timestamp", datetime.now(timezone.utc).isoformat())

        for trade in msg.get("trades", []):
            side_raw = trade.get("side", "")
            # BTC Markets: "Ask" = seller-initiated = buy order filled
            #              "Bid" = buyer-initiated  = sell order filled
            side = "sell" if side_raw.lower() == "ask" else "buy"

            normalised: Dict[str, Any] = {
                "messageType": "trade",
                "symbol":      market_id,
                "trade_id":    str(trade.get("tradeId", trade.get("id", ""))),
                "price":       float(trade.get("price", 0) or 0),
                "quantity":    float(trade.get("amount", trade.get("volume", 0)) or 0),
                "side":        side,
                "timestamp":   ts,
                "exchange":    "btcmarkets",
            }

            try:
                result = self.on_trade(normalised)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.error("BTCMarketsWSFeed on_trade callback error: %s", exc)

    async def _handle_user_data(self, msg: Dict[str, Any]) -> None:
        """Process a private user-data message (fund change or order change)."""
        # User data callbacks are routed to on_book_update if no separate handler
        # is provided, but callers should register via subscribe_user_data() on
        # the BTCMarketsClient class.  Here we simply log.
        logger.debug("BTCMarketsWSFeed user-data: %s", msg)

    # ------------------------------------------------------------------
    # Background loops
    # ------------------------------------------------------------------

    async def _receive_loop(self) -> None:
        """Main receive loop — dispatches all incoming WS messages."""
        while self._running:
            if not self._ws or not self._connected:
                await self._reconnect()
                if not self._connected:
                    continue

            try:
                raw = await self._ws.recv()
                await self._dispatch(raw)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("BTCMarketsWSFeed receive error: %s", exc)
                if self._running:
                    await self._reconnect()

    async def _heartbeat_loop(self) -> None:
        """Monitor for stale connections — reconnect if silent for 30 s."""
        while self._running:
            try:
                await asyncio.sleep(5.0)
                if not self._connected:
                    continue
                elapsed = time.monotonic() - self._last_msg_time
                if elapsed > HEARTBEAT_TIMEOUT_S:
                    logger.warning(
                        "BTCMarketsWSFeed no message for %.0fs, reconnecting",
                        elapsed,
                    )
                    await self._reconnect()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("BTCMarketsWSFeed heartbeat error: %s", exc)

    async def _ping_loop(self) -> None:
        """Send periodic pings to keep the connection alive."""
        while self._running:
            try:
                await asyncio.sleep(PING_INTERVAL_S)
                if self._ws and self._connected:
                    try:
                        await self._ws.ping()
                    except Exception as exc:
                        logger.debug("BTCMarketsWSFeed ping error: %s", exc)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("BTCMarketsWSFeed ping loop error: %s", exc)

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "BTCMarketsWSFeed":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()
