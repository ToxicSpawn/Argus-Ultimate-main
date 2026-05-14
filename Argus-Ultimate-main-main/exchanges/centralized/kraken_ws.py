"""Kraken WebSocket v2 feed client.

Opens a persistent connection to wss://ws.kraken.com/v2, subscribes to the
``trade`` and ``book`` (depth-10) channels for each symbol, and routes every
incoming message directly into WSFeedAdapter's Kraken-specific handlers::

    ws_adapter.on_kraken_trade(payload)
    ws_adapter.on_kraken_book(payload)

This makes OFI, VPIN, MicropriceDrift, and LatencyTelemetry update on every
tick rather than only on the 5-second OHLCV poll cycle.

Usage
-----
    ws = KrakenWSClient(
        symbols=["BTC/AUD", "ETH/AUD"],
        kraken_adapter=bot.kraken_adapter,
        ws_adapter=bot.ws_adapter,
    )
    task = asyncio.create_task(ws.run())
    # ... later ...
    task.cancel()
    await ws.close()
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

_KRAKEN_WS_URL = "wss://ws.kraken.com/v2"
_RECONNECT_DELAY_S = 3.0
_PING_INTERVAL_S = 20.0


def _ccxt_to_kraken_ws_symbol(symbol: str) -> str:
    """Convert CCXT symbol ("BTC/AUD") to Kraken WS pair ("BTC/AUD")."""
    return symbol


class KrakenWSClient:
    """Long-running asyncio WS client for Kraken public feeds.

    Parameters
    ----------
    symbols : list[str]
        CCXT-format trading pairs (e.g. ``["BTC/AUD", "ETH/AUD"]``).
    kraken_adapter : CcxtKrakenAdapter
        Adapter whose ``submit_limit_order`` handles live order placement.
    ws_adapter : WSFeedAdapter
        Feed adapter whose ``on_kraken_trade`` / ``on_kraken_book`` methods
        receive normalised Kraken WS payloads and route them into LiveSignalBus.
    reconnect : bool
        Automatically reconnect on connection drops (default True).
    """

    def __init__(
        self,
        symbols: List[str],
        kraken_adapter: Any,
        ws_adapter: Any,
        reconnect: bool = True,
    ) -> None:
        self.symbols = [_ccxt_to_kraken_ws_symbol(s) for s in symbols]
        self.kraken_adapter = kraken_adapter
        self.ws_adapter = ws_adapter
        self.reconnect = reconnect
        self._running = False
        self._ws: Optional[Any] = None
        self._ping_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Connect and consume messages. Reconnects on failure if enabled."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_consume()
            except asyncio.CancelledError:
                logger.info("KrakenWSClient cancelled")
                self._running = False
                break
            except Exception as exc:
                logger.warning("KrakenWSClient disconnected: %s", exc)
                if not self.reconnect or not self._running:
                    break
                logger.info("Reconnecting in %.1fs ...", _RECONNECT_DELAY_S)
                await asyncio.sleep(_RECONNECT_DELAY_S)

    async def close(self) -> None:
        """Stop the feed gracefully."""
        self._running = False
        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _connect_and_consume(self) -> None:
        try:
            import websockets  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "websockets package not installed — run: pip install websockets"
            ) from exc

        logger.info("KrakenWSClient connecting to %s", _KRAKEN_WS_URL)
        async with websockets.connect(
            _KRAKEN_WS_URL,
            ping_interval=None,   # we manage pings manually
            max_size=2 ** 22,     # 4 MB — large enough for full book snapshots
        ) as ws:
            self._ws = ws
            logger.info("KrakenWSClient connected")

            await self._subscribe(ws)

            self._ping_task = asyncio.create_task(self._heartbeat(ws))

            async for raw in ws:
                if not self._running:
                    break
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await self._dispatch(msg)

    async def _subscribe(self, ws: Any) -> None:
        """Send subscribe messages for trade and book channels."""
        trade_sub = {
            "method": "subscribe",
            "params": {
                "channel": "trade",
                "symbol": self.symbols,
            },
        }
        book_sub = {
            "method": "subscribe",
            "params": {
                "channel": "book",
                "symbol": self.symbols,
                "depth": 10,
            },
        }
        await ws.send(json.dumps(trade_sub))
        logger.debug("KrakenWSClient: subscribed to trade channel")
        await ws.send(json.dumps(book_sub))
        logger.debug("KrakenWSClient: subscribed to book channel")

    async def _dispatch(self, msg: dict) -> None:
        """Route a parsed WS message to WSFeedAdapter's Kraken handlers.

        WSFeedAdapter.on_kraken_trade() expects:
            {"channel": "trade", "data": [
                {"symbol": str, "side": str, "qty": float,
                 "price": float, "timestamp": str}
            ]}

        WSFeedAdapter.on_kraken_book() expects:
            {"channel": "book", "data": [
                {"symbol": str,
                 "bids": [{"price": float, "qty": float}, ...],
                 "asks": [{"price": float, "qty": float}, ...]}
            ]}
        """
        channel = msg.get("channel", "")
        msg_type = msg.get("type", "")

        # Skip control messages
        if channel in ("heartbeat", "status") or msg_type in (
            "subscribed", "unsubscribed", "error"
        ):
            return

        if channel == "trade":
            # Kraken v2 trade data arrives as a list under "data".
            # Pass the raw message directly to on_kraken_trade() — it already
            # knows how to parse this exact shape.
            data = msg.get("data")
            if not data:
                return
            try:
                self.ws_adapter.on_kraken_trade(msg)
            except Exception as exc:
                logger.debug("on_kraken_trade dispatch error: %s", exc)

        elif channel == "book":
            # Kraken v2 book: bids/asks come as [{"price": p, "qty": q}, ...]
            # on_kraken_book() already handles this shape — just pass through.
            # However the raw Kraken v2 WS format uses float arrays [[p,q]]
            # in some firmware versions, so we normalise to dict-of-dicts here
            # to be safe.
            data = msg.get("data")
            if not data:
                return
            try:
                normalised_data = []
                for entry in data:
                    raw_bids = entry.get("bids") or []
                    raw_asks = entry.get("asks") or []

                    # Normalise both list-of-lists [[p,q]] and
                    # list-of-dicts [{"price":p,"qty":q}] to the dict form.
                    def _to_dict(level: Any) -> dict:
                        if isinstance(level, dict):
                            return {
                                "price": float(level.get("price", level.get("p", 0))),
                                "qty":   float(level.get("qty",   level.get("q", 0))),
                            }
                        # list / tuple form [price, qty]
                        return {"price": float(level[0]), "qty": float(level[1])}

                    normalised_data.append({
                        "symbol": entry.get("symbol", ""),
                        "bids": [_to_dict(b) for b in raw_bids],
                        "asks": [_to_dict(a) for a in raw_asks],
                        "timestamp": entry.get("timestamp", time.time()),
                    })

                self.ws_adapter.on_kraken_book({"channel": "book", "data": normalised_data})
            except Exception as exc:
                logger.debug("on_kraken_book dispatch error: %s", exc)

    async def _heartbeat(self, ws: Any) -> None:
        """Send a ping every _PING_INTERVAL_S seconds."""
        try:
            while self._running:
                await asyncio.sleep(_PING_INTERVAL_S)
                if not self._running:
                    break
                await ws.send(json.dumps({"method": "ping"}))
                logger.debug("KrakenWSClient: ping sent")
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("KrakenWSClient heartbeat error: %s", exc)
