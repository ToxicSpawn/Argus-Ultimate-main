#!/usr/bin/env python3
"""
L2 Orderbook Feed — async WebSocket-based Level-2 orderbook subscription.

Supports Kraken and Coinbase Advanced. Falls back gracefully when websockets
is not installed or the connection fails. All errors are swallowed so the main
trading loop is never blocked by market-data connectivity issues.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PriceLevel:
    """Price level in an orderbook.
    
    Attributes
    ----------
    price : float
        Price level
    size : float
        Quantity at this price level
    """
    price: float = 0.0
    size: float = 0.0


def _from_kraken_symbol(kraken_symbol: str) -> str:
    """Convert Kraken symbol format to standard format.
    
    Example: "XBTUSD" -> "BTC/USD", "ETHUSD" -> "ETH/USD"
    """
    symbol = kraken_symbol.upper()
    
    # Handle XBT -> BTC
    if symbol.startswith("XBT"):
        base = "BTC"
        quote = symbol[3:]
    else:
        # Find where base ends and quote begins
        # Common quotes: USD, USDT, EUR, GBP, etc.
        for quote in ["USDT", "USD", "EUR", "GBP", "AUD", "CAD", "JPY"]:
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                break
        else:
            # Default: split in half
            mid = len(symbol) // 2
            base = symbol[:mid]
            quote = symbol[mid:]
    
    return f"{base}/{quote}"


def _to_kraken_symbol(standard_symbol: str) -> str:
    """Convert standard symbol format to Kraken format.
    
    Example: "BTC/USD" -> "XBTUSD", "ETH/USD" -> "ETHUSD"
    """
    parts = standard_symbol.upper().split("/")
    if len(parts) != 2:
        return standard_symbol.replace("/", "").replace("-", "")
    
    base, quote = parts
    
    # Handle BTC -> XBT for Kraken
    if base == "BTC":
        base = "XBT"
    
    return f"{base}{quote}"


@dataclass
class OrderbookSnapshot:
    """Point-in-time snapshot of an orderbook.
    
    Attributes
    ----------
    symbol : str
        Trading pair symbol (e.g., "BTC/USDT")
    timestamp : float
        Unix timestamp of the snapshot
    bids : list[tuple[float, float]]
        List of (price, quantity) tuples for bids, sorted descending by price
    asks : list[tuple[float, float]]
        List of (price, quantity) tuples for asks, sorted ascending by price
    sequence : int
        Exchange sequence number for ordering
    """
    symbol: str = ""
    timestamp: float = field(default_factory=time.time)
    bids: List[Tuple[float, float]] = field(default_factory=list)
    asks: List[Tuple[float, float]] = field(default_factory=list)
    sequence: int = 0
    
    @property
    def best_bid(self) -> Optional[float]:
        """Return the best (highest) bid price, or None if empty."""
        return self.bids[0][0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        """Return the best (lowest) ask price, or None if empty."""
        return self.asks[0][0] if self.asks else None
    
    @property
    def spread(self) -> Optional[float]:
        """Return the bid-ask spread, or None if either side is empty."""
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid
    
    @property
    def mid_price(self) -> Optional[float]:
        """Return the mid price between best bid and ask."""
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

_WS_URLS: Dict[str, str] = {
    "kraken": "wss://ws.kraken.com",
    "coinbase_advanced": "wss://advanced-trade-ws.coinbase.com",
    "coinbase": "wss://advanced-trade-ws.coinbase.com",
}


class L2OrderbookFeed:
    """
    Maintains an in-memory best-bid/ask + shallow book per subscribed symbol.
    Push-style: call `subscribe()` to start. Optionally register a callback
    via `on_update` which fires on every book update.
    """

    def __init__(self, exchange: str = "kraken", *, reconnect_delay: float = 5.0):
        self.exchange = str(exchange or "kraken").lower()
        self.reconnect_delay = max(1.0, float(reconnect_delay))
        self._pairs: List[str] = []
        # symbol -> {bids: [(price, qty), ...], asks: [(price, qty), ...], ts: float}
        self._books: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"bids": [], "asks": [], "ts": 0.0})
        self._callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None

    # ---------------------------------------------------------------- public API

    def on_update(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Register a callback(symbol, book) fired on every book update."""
        self._callbacks.append(callback)

    def best_bid(self, symbol: str) -> Optional[float]:
        bids = self._books[symbol].get("bids", [])
        return float(bids[0][0]) if bids else None

    def best_ask(self, symbol: str) -> Optional[float]:
        asks = self._books[symbol].get("asks", [])
        return float(asks[0][0]) if asks else None

    def spread_bps(self, symbol: str) -> Optional[float]:
        bid = self.best_bid(symbol)
        ask = self.best_ask(symbol)
        if bid and ask and ask > 0:
            return ((ask - bid) / ask) * 10_000.0
        return None

    def get_book(self, symbol: str) -> Dict[str, Any]:
        return dict(self._books[symbol])

    async def subscribe(self, pairs: List[str]) -> None:
        """Start the WebSocket feed for the given pairs (fire-and-forget)."""
        self._pairs = list(pairs or [])
        if not self._pairs:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_forever())

    async def close(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    # ---------------------------------------------------------------- internals

    async def _run_forever(self) -> None:
        while self._running:
            try:
                await self._connect_and_consume()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("L2Feed %s disconnected (%s), reconnecting in %.1fs", self.exchange, e, self.reconnect_delay)
            if self._running:
                await asyncio.sleep(self.reconnect_delay)

    async def _connect_and_consume(self) -> None:
        try:
            import websockets  # type: ignore[import]
        except ImportError:
            logger.debug("L2Feed: websockets not installed, running in stub mode")
            await asyncio.sleep(60)
            return

        url = _WS_URLS.get(self.exchange, _WS_URLS["kraken"])
        sub_msg = self._build_subscribe_message()

        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            await ws.send(json.dumps(sub_msg))
            logger.info("L2Feed: subscribed to %s on %s", self._pairs, self.exchange)
            async for raw in ws:
                if not self._running:
                    break
                try:
                    msg = json.loads(raw)
                    self._handle_message(msg)
                except Exception:
                    pass

    def _build_subscribe_message(self) -> Dict[str, Any]:
        if self.exchange == "kraken":
            return {
                "event": "subscribe",
                "pair": [p.replace("/", "") for p in self._pairs],
                "subscription": {"name": "book", "depth": 10},
            }
        # Coinbase Advanced
        return {
            "type": "subscribe",
            "product_ids": [p.replace("/", "-") for p in self._pairs],
            "channel": "level2",
        }

    def _handle_message(self, msg: Any) -> None:
        if not isinstance(msg, (dict, list)):
            return
        if self.exchange == "kraken":
            self._handle_kraken(msg)
        else:
            self._handle_coinbase(msg)

    def _handle_kraken(self, msg: Any) -> None:
        if not isinstance(msg, list) or len(msg) < 4:
            return
        pair_raw = str(msg[-1] or "")
        symbol = pair_raw.replace("XBT", "BTC").replace("/", "/")
        data = msg[1] if len(msg) > 1 else {}
        book = self._books[symbol]
        book["ts"] = time.time()
        if isinstance(data, dict):
            if "bs" in data:
                book["bids"] = sorted([(float(p), float(q)) for p, q, *_ in data["bs"]], reverse=True)[:10]
            if "as" in data:
                book["asks"] = sorted([(float(p), float(q)) for p, q, *_ in data["as"]])[:10]
            if "b" in data:
                self._apply_delta(book["bids"], data["b"], descending=True)
            if "a" in data:
                self._apply_delta(book["asks"], data["a"], descending=False)
        self._fire(symbol, book)

    def _handle_coinbase(self, msg: Any) -> None:
        if not isinstance(msg, dict):
            return
        events = msg.get("events") or []
        for event in events:
            updates = event.get("updates") or []
            for upd in updates:
                product_id = str(upd.get("product_id") or "").replace("-", "/")
                if not product_id:
                    continue
                book = self._books[product_id]
                book["ts"] = time.time()
                side = str(upd.get("side") or "").lower()
                price = float(upd.get("new_price") or upd.get("price") or 0)
                qty = float(upd.get("new_quantity") or upd.get("quantity") or 0)
                if side == "bid":
                    self._apply_single(book["bids"], price, qty, descending=True)
                elif side == "offer" or side == "ask":
                    self._apply_single(book["asks"], price, qty, descending=False)
                self._fire(product_id, book)

    @staticmethod
    def _apply_delta(levels: List[Tuple[float, float]], deltas: List[Any], *, descending: bool) -> None:
        level_map: Dict[float, float] = {p: q for p, q in levels}
        for entry in deltas:
            price, qty = float(entry[0]), float(entry[1])
            if qty == 0.0:
                level_map.pop(price, None)
            else:
                level_map[price] = qty
        levels.clear()
        levels.extend(sorted(level_map.items(), reverse=descending)[:10])

    @staticmethod
    def _apply_single(levels: List[Tuple[float, float]], price: float, qty: float, *, descending: bool) -> None:
        level_map: Dict[float, float] = {p: q for p, q in levels}
        if qty == 0.0:
            level_map.pop(price, None)
        else:
            level_map[price] = qty
        levels.clear()
        levels.extend(sorted(level_map.items(), reverse=descending)[:10])

    def _fire(self, symbol: str, book: Dict[str, Any]) -> None:
        for cb in self._callbacks:
            try:
                cb(symbol, book)
            except Exception:
                pass
