"""
Hyperliquid WebSocket L2 Feed
==============================
Drops into the same interface as data/orderbook/l2_feed.py.
Provides real-time L2 orderbook snapshots + incremental updates
and trade stream for Hyperliquid perpetuals.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from decimal import Decimal
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

OrderbookLevel = Tuple[Decimal, Decimal]  # (price, size)


class HyperliquidL2Feed:
    """
    Real-time L2 orderbook + trade feed for Hyperliquid.

    Usage::

        feed = HyperliquidL2Feed(symbols=["BTC", "ETH"])
        feed.on_book_update("BTC", my_callback)
        await feed.start()
    """

    WS_URL = "wss://api.hyperliquid.xyz/ws"
    WS_URL_TESTNET = "wss://api.hyperliquid-testnet.xyz/ws"

    def __init__(
        self,
        symbols: List[str],
        testnet: bool = False,
        depth: int = 20,
    ) -> None:
        self.symbols = symbols
        self.testnet = testnet
        self.depth = depth

        self._bids: Dict[str, Dict[Decimal, Decimal]] = defaultdict(dict)
        self._asks: Dict[str, Dict[Decimal, Decimal]] = defaultdict(dict)
        self._book_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._trade_callbacks: Dict[str, List[Callable]] = defaultdict(list)

        self._ws_task: Optional[asyncio.Task] = None
        self._running = False

        # Metrics
        self.messages_received: int = 0
        self.last_update_ts: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def on_book_update(self, symbol: str, callback: Callable[[dict], None]) -> None:
        """Register callback for L2 book updates on `symbol`."""
        self._book_callbacks[symbol].append(callback)

    def on_trade(self, symbol: str, callback: Callable[[dict], None]) -> None:
        """Register callback for trade events on `symbol`."""
        self._trade_callbacks[symbol].append(callback)

    async def start(self) -> None:
        """Start WS listener task."""
        self._running = True
        self._ws_task = asyncio.create_task(self._run())
        logger.info("HyperliquidL2Feed started | symbols=%s", self.symbols)

    async def stop(self) -> None:
        """Stop WS listener."""
        self._running = False
        if self._ws_task:
            self._ws_task.cancel()
        logger.info("HyperliquidL2Feed stopped")

    def get_book(self, symbol: str) -> dict:
        """Return current in-memory book snapshot."""
        bids = sorted(self._bids[symbol].items(), reverse=True)[: self.depth]
        asks = sorted(self._asks[symbol].items())[: self.depth]
        return {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "timestamp": self.last_update_ts.get(symbol, 0.0),
        }

    def get_best_bid(self, symbol: str) -> Optional[Decimal]:
        if not self._bids[symbol]:
            return None
        return max(self._bids[symbol].keys())

    def get_best_ask(self, symbol: str) -> Optional[Decimal]:
        if not self._asks[symbol]:
            return None
        return min(self._asks[symbol].keys())

    def get_mid_price(self, symbol: str) -> Optional[Decimal]:
        bid = self.get_best_bid(symbol)
        ask = self.get_best_ask(symbol)
        if bid and ask:
            return (bid + ask) / 2
        return None

    def get_spread_bps(self, symbol: str) -> Optional[Decimal]:
        """Return bid-ask spread in basis points."""
        bid = self.get_best_bid(symbol)
        ask = self.get_best_ask(symbol)
        if bid and ask and bid > 0:
            return ((ask - bid) / bid) * 10000
        return None

    # ------------------------------------------------------------------
    # Internal WS loop
    # ------------------------------------------------------------------
    async def _run(self) -> None:
        import websockets
        url = self.WS_URL_TESTNET if self.testnet else self.WS_URL
        backoff = 1

        while self._running:
            try:
                async with websockets.connect(
                    url, ping_interval=20, ping_timeout=10
                ) as ws:
                    backoff = 1
                    # Subscribe to all symbols
                    for sym in self.symbols:
                        await ws.send(json.dumps({
                            "method": "subscribe",
                            "subscription": {"type": "l2Book", "coin": sym},
                        }))
                        await ws.send(json.dumps({
                            "method": "subscribe",
                            "subscription": {"type": "trades", "coin": sym},
                        }))
                    logger.info("WS subscribed to %d symbols", len(self.symbols))

                    async for raw in ws:
                        if not self._running:
                            break
                        self.messages_received += 1
                        try:
                            msg = json.loads(raw)
                            await self._handle_message(msg)
                        except Exception as e:
                            logger.debug("WS parse error: %s", e)

            except Exception as exc:
                if not self._running:
                    break
                logger.warning(
                    "WS error: %s | reconnecting in %ds", exc, backoff
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def _handle_message(self, msg: dict) -> None:
        channel = msg.get("channel")
        data = msg.get("data", {})

        if channel == "l2Book":
            await self._process_book_update(data)
        elif channel == "trades":
            await self._process_trades(data)

    async def _process_book_update(self, data: dict) -> None:
        symbol = data.get("coin", "")
        if not symbol:
            return

        levels = data.get("levels", [[], []])
        ts = time.time()

        # Full snapshot on first message, incremental after
        # Hyperliquid sends full book snapshots periodically
        new_bids: Dict[Decimal, Decimal] = {}
        new_asks: Dict[Decimal, Decimal] = {}

        for price_str, size_str in levels[0]:  # bids
            price, size = Decimal(price_str), Decimal(size_str)
            if size > 0:
                new_bids[price] = size

        for price_str, size_str in levels[1]:  # asks
            price, size = Decimal(price_str), Decimal(size_str)
            if size > 0:
                new_asks[price] = size

        self._bids[symbol] = new_bids
        self._asks[symbol] = new_asks
        self.last_update_ts[symbol] = ts

        # Notify callbacks
        book_snapshot = self.get_book(symbol)
        for cb in self._book_callbacks.get(symbol, []):
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(book_snapshot)
                else:
                    cb(book_snapshot)
            except Exception as e:
                logger.error("book callback error: %s", e)

    async def _process_trades(self, data) -> None:
        if not isinstance(data, list):
            return
        for trade in data:
            symbol = trade.get("coin", "")
            if not symbol:
                continue
            trade_event = {
                "symbol": symbol,
                "price": Decimal(str(trade.get("px", "0"))),
                "size": Decimal(str(trade.get("sz", "0"))),
                "side": trade.get("side", ""),
                "timestamp": trade.get("time", time.time() * 1000) / 1000,
            }
            for cb in self._trade_callbacks.get(symbol, []):
                try:
                    if asyncio.iscoroutinefunction(cb):
                        await cb(trade_event)
                    else:
                        cb(trade_event)
                except Exception as e:
                    logger.error("trade callback error: %s", e)
