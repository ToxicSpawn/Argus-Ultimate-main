"""
Live Market Data Manager — real-time WebSocket price feeds with REST fallback.

Aggregates WebSocket tick data from exchange connectors (Coinbase, Kraken)
into a unified interface for the trading loop. Replaces 30-second REST
polling with sub-second price updates for strategies, risk, and execution.

Features:
  - Latest price, bid, ask, spread, volume per symbol
  - Orderbook snapshots from L2 feed
  - Staleness detection (is_stale)
  - Tick callbacks (on_tick)
  - Automatic REST fallback when WebSocket fails
  - Graceful lifecycle (subscribe / disconnect)

Usage:
    mgr = LiveMarketDataManager(exchanges={"kraken": ccxt_ex})
    await mgr.subscribe(["BTC/USD", "ETH/USD"], exchange="kraken")
    latest = mgr.get_latest("BTC/USD")
    if not mgr.is_stale("BTC/USD"):
        logger.info(latest["price"], latest["spread"])
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class LiveMarketDataManager:
    """
    Unified live market data aggregator.

    Manages WebSocket connections to exchanges for real-time tick data and
    falls back to REST polling when WebSocket is unavailable.

    Args:
        exchanges:          Dict mapping exchange name to ccxt async exchange
                            instance (used for REST fallback).
        market_data_service: Optional MarketDataService for REST fallback.
        rest_fallback_interval_s: Seconds between REST polls when WS is down.
    """

    def __init__(
        self,
        exchanges: Optional[Dict[str, Any]] = None,
        market_data_service: Optional[Any] = None,
        rest_fallback_interval_s: float = 5.0,
    ):
        self._exchanges = exchanges or {}
        self._market_data_service = market_data_service
        self._rest_fallback_interval_s = rest_fallback_interval_s

        # Latest tick data per symbol
        # {symbol: {price, bid, ask, spread, volume, timestamp, age_ms}}
        self._latest: Dict[str, Dict[str, Any]] = {}

        # Orderbook data per symbol (from L2 feed or WS connector)
        self._orderbooks: Dict[str, Dict[str, Any]] = {}

        # Tick callbacks
        self._tick_callbacks: List[Callable[[Dict[str, Any]], Any]] = []

        # WebSocket connectors (keyed by exchange name)
        self._ws_connectors: Dict[str, Any] = {}
        self._ws_tasks: Dict[str, asyncio.Task] = {}

        # L2 feed (optional, from data/orderbook/l2_feed.py)
        self._l2_feed: Optional[Any] = None

        # REST fallback tasks
        self._rest_fallback_tasks: Dict[str, asyncio.Task] = {}
        self._rest_fallback_active: Dict[str, bool] = {}

        # Subscriptions tracking
        self._subscriptions: Dict[str, str] = {}  # symbol -> exchange

        # Tick store integration (set via set_tick_store)
        self._tick_store: Optional[Any] = None

        self._running = False

    # ------------------------------------------------------------------ public

    async def subscribe(
        self,
        symbols: List[str],
        exchange: str = "kraken",
    ) -> bool:
        """
        Subscribe to live market data for the given symbols on the given exchange.

        Attempts WebSocket first; if it fails, starts REST fallback polling.

        Args:
            symbols:  List of symbols in "BASE/QUOTE" format (e.g. "BTC/USD").
            exchange: Exchange identifier — "kraken" or "coinbase".

        Returns:
            True if at least one data source (WS or REST) is active.
        """
        self._running = True
        exchange = exchange.lower()

        for sym in symbols:
            self._subscriptions[sym] = exchange

        ws_ok = await self._start_ws(symbols, exchange)

        if ws_ok:
            logger.info(
                "LiveMarketData: WebSocket connected for %s on %s",
                symbols, exchange,
            )
            return True

        # WebSocket failed — start REST fallback
        logger.warning(
            "LiveMarketData: WebSocket unavailable for %s, starting REST fallback",
            exchange,
        )
        self._start_rest_fallback(symbols, exchange)
        return True

    def get_latest(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest tick data for a symbol.

        Returns:
            Dict with keys: price, bid, ask, spread, volume, timestamp, age_ms.
            None if no data has been received yet.
        """
        data = self._latest.get(symbol)
        if data is None:
            return None
        # Recompute age_ms on every call
        ts = data.get("timestamp")
        if ts and isinstance(ts, datetime):
            age_ms = (datetime.now(timezone.utc) - ts).total_seconds() * 1000.0
        else:
            age_ms = (time.time() - float(data.get("_mono", time.time()))) * 1000.0
        result = dict(data)
        result["age_ms"] = age_ms
        return result

    def get_orderbook(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest orderbook snapshot for a symbol.

        Returns:
            Dict with keys: bids, asks, mid_price, spread_bps.
            None if no orderbook data is available.
        """
        # Try L2 feed first
        if self._l2_feed is not None:
            book = self._l2_feed.get_book(symbol)
            if book is not None:
                bids = [[lv.price, lv.size] for lv in book.bids[:10]]
                asks = [[lv.price, lv.size] for lv in book.asks[:10]]
                return {
                    "bids": bids,
                    "asks": asks,
                    "mid_price": book.mid_price,
                    "spread_bps": book.spread_bps,
                }

        # Fall back to cached orderbook
        return self._orderbooks.get(symbol)

    def is_stale(self, symbol: str, max_age_ms: float = 5000.0) -> bool:
        """
        Check if the latest data for a symbol is stale.

        Args:
            symbol:     Symbol to check.
            max_age_ms: Maximum acceptable age in milliseconds.

        Returns:
            True if the data is older than max_age_ms or if no data exists.
        """
        data = self._latest.get(symbol)
        if data is None:
            return True
        mono = data.get("_mono", 0.0)
        age_ms = (time.monotonic() - mono) * 1000.0
        return age_ms > max_age_ms

    def on_tick(self, callback: Callable[[Dict[str, Any]], Any]) -> None:
        """
        Register a callback that fires on every tick update.

        Callback receives: {symbol, price, bid, ask, spread, volume, timestamp}
        """
        self._tick_callbacks.append(callback)

    def set_tick_store(self, tick_store: Any) -> None:
        """
        Attach a TickStore instance for persisting every tick to disk.

        When set, every price update (from WS or REST) is automatically
        forwarded to ``tick_store.record_tick()``.
        """
        self._tick_store = tick_store
        logger.info("LiveMarketData: TickStore attached for tick persistence")

    def set_l2_feed(self, l2_feed: Any) -> None:
        """Attach an L2OrderbookFeed instance for orderbook data."""
        self._l2_feed = l2_feed
        logger.info("LiveMarketData: L2 feed attached")

    @property
    def connected_exchanges(self) -> List[str]:
        """List of exchanges with active WebSocket connections."""
        return [
            ex for ex, conn in self._ws_connectors.items()
            if getattr(conn, "connected", False)
        ]

    @property
    def subscribed_symbols(self) -> List[str]:
        """List of all subscribed symbols."""
        return list(self._subscriptions.keys())

    # ------------------------------------------------------------------ lifecycle

    async def disconnect(self) -> None:
        """Disconnect all WebSocket feeds and stop REST fallback tasks."""
        self._running = False

        # Stop REST fallback
        for sym, task in list(self._rest_fallback_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._rest_fallback_tasks.clear()
        self._rest_fallback_active.clear()

        # Stop WS connectors
        for name, conn in list(self._ws_connectors.items()):
            try:
                if hasattr(conn, "disconnect"):
                    await conn.disconnect()
            except Exception as exc:
                logger.debug("LiveMarketData: error disconnecting %s WS: %s", name, exc)

        # Stop WS tasks
        for name, task in list(self._ws_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._ws_tasks.clear()
        self._ws_connectors.clear()

        # Disconnect L2 feed
        if self._l2_feed is not None and hasattr(self._l2_feed, "disconnect"):
            try:
                await self._l2_feed.disconnect()
            except Exception as exc:
                logger.debug("LiveMarketData: L2 feed disconnect error: %s", exc)

        logger.info("LiveMarketData: all feeds disconnected")

    # ------------------------------------------------------------------ WS setup

    async def _start_ws(self, symbols: List[str], exchange: str) -> bool:
        """
        Attempt to start a WebSocket connection for the given exchange.

        Returns True if the WebSocket connected successfully.
        """
        if exchange in self._ws_connectors:
            # Already connected
            return getattr(self._ws_connectors[exchange], "connected", False)

        try:
            if exchange == "coinbase":
                return await self._start_coinbase_ws(symbols)
            elif exchange == "kraken":
                return await self._start_kraken_ws(symbols)
            else:
                logger.warning("LiveMarketData: no WS connector for exchange %s", exchange)
                return False
        except Exception as exc:
            logger.warning("LiveMarketData: WS start failed for %s: %s", exchange, exc)
            return False

    async def _start_coinbase_ws(self, symbols: List[str]) -> bool:
        """Start Coinbase WebSocket connector."""
        try:
            from core.connectors.coinbase_ws_connector import CoinbaseWSConnector

            # Convert BTC/USD -> BTC-AUD style for Coinbase
            cb_symbols = [s.replace("/", "-") for s in symbols]
            conn = CoinbaseWSConnector(symbols=cb_symbols, channels=["ticker"])
            conn.on_ticker(self._handle_coinbase_tick)

            connected = await conn.connect()
            if not connected:
                return False

            self._ws_connectors["coinbase"] = conn
            # Start the receive loop in background
            task = asyncio.create_task(conn.run())
            self._ws_tasks["coinbase"] = task
            return True
        except ImportError:
            logger.debug("LiveMarketData: CoinbaseWSConnector not available")
            return False
        except Exception as exc:
            logger.warning("LiveMarketData: Coinbase WS start error: %s", exc)
            return False

    async def _start_kraken_ws(self, symbols: List[str]) -> bool:
        """
        Start Kraken public WebSocket for ticker data.

        Uses the L2OrderbookFeed which already handles Kraken WS v2.
        Also subscribes the L2 feed for orderbook data.
        """
        try:
            from data.orderbook.l2_feed import L2OrderbookFeed

            feed = L2OrderbookFeed(exchange="kraken", depth=10)
            ok = await feed.subscribe(symbols)
            if not ok:
                return False

            self._l2_feed = feed
            self._ws_connectors["kraken"] = feed

            # Start a background task that polls the L2 feed for tick-like data
            task = asyncio.create_task(self._poll_l2_for_ticks(symbols))
            self._ws_tasks["kraken"] = task
            return True
        except ImportError:
            logger.debug("LiveMarketData: L2OrderbookFeed not available")
            return False
        except Exception as exc:
            logger.warning("LiveMarketData: Kraken WS start error: %s", exc)
            return False

    # ------------------------------------------------------------------ tick handling

    def _handle_coinbase_tick(self, tick: Dict[str, Any]) -> None:
        """
        Handle a ticker update from the Coinbase WebSocket connector.

        Expected format from CoinbaseWSConnector._parse_ticker:
          {symbol, bid, ask, last, volume_24h, timestamp}
        """
        symbol = tick.get("symbol", "")
        # Coinbase uses "BTC-AUD" — normalize to "BTC/AUD"
        symbol = symbol.replace("-", "/")

        bid = float(tick.get("bid", 0) or 0)
        ask = float(tick.get("ask", 0) or 0)
        last = float(tick.get("last", 0) or 0)
        price = last or ((bid + ask) / 2.0 if bid and ask else 0.0)
        spread = ask - bid if bid and ask else 0.0

        data = {
            "symbol": symbol,
            "price": price,
            "bid": bid,
            "ask": ask,
            "spread": spread,
            "volume": float(tick.get("volume_24h", 0) or 0),
            "timestamp": tick.get("timestamp", datetime.now(timezone.utc)),
            "_mono": time.monotonic(),
        }
        self._latest[symbol] = data
        self._fire_tick_callbacks(data)

    async def _poll_l2_for_ticks(self, symbols: List[str]) -> None:
        """
        Poll L2 feed orderbooks every 100ms and update _latest with mid-price.

        This bridges the L2 orderbook feed into the tick-data interface.
        """
        try:
            while self._running:
                for sym in symbols:
                    if self._l2_feed is None:
                        break
                    book = self._l2_feed.get_book(sym)
                    if book is None:
                        continue

                    bid = book.best_bid or 0.0
                    ask = book.best_ask or 0.0
                    mid = book.mid_price or 0.0
                    spread = book.spread or 0.0

                    data = {
                        "symbol": sym,
                        "price": mid,
                        "bid": bid,
                        "ask": ask,
                        "spread": spread,
                        "volume": 0.0,  # L2 feed does not carry volume
                        "timestamp": datetime.now(timezone.utc),
                        "_mono": time.monotonic(),
                    }
                    self._latest[sym] = data
                    self._fire_tick_callbacks(data)

                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("LiveMarketData: L2 tick poll error: %s", exc)

    def _fire_tick_callbacks(self, data: Dict[str, Any]) -> None:
        """Fire all registered tick callbacks and record to TickStore."""
        # Persist to TickStore if attached
        if self._tick_store is not None:
            try:
                self._tick_store.record_tick(
                    symbol=data.get("symbol", ""),
                    timestamp=data.get("timestamp", time.time()),
                    bid=data.get("bid", 0.0),
                    ask=data.get("ask", 0.0),
                    last=data.get("price", 0.0),
                    volume=data.get("volume", 0.0),
                    exchange=data.get("_source", "unknown"),
                )
            except Exception as exc:
                logger.debug("LiveMarketData: TickStore record error: %s", exc)

        for cb in self._tick_callbacks:
            try:
                cb(data)
            except Exception as exc:
                logger.debug("LiveMarketData: tick callback error: %s", exc)

    # ------------------------------------------------------------------ REST fallback

    def _start_rest_fallback(self, symbols: List[str], exchange: str) -> None:
        """Start a background REST polling task for the given symbols."""
        key = exchange
        if key in self._rest_fallback_tasks and not self._rest_fallback_tasks[key].done():
            return  # Already running
        self._rest_fallback_active[key] = True
        task = asyncio.create_task(self._rest_poll_loop(symbols, exchange))
        self._rest_fallback_tasks[key] = task

    async def _rest_poll_loop(self, symbols: List[str], exchange: str) -> None:
        """Background loop that polls REST for ticker data."""
        try:
            while self._running and self._rest_fallback_active.get(exchange, False):
                for sym in symbols:
                    try:
                        ticker = await self._fetch_rest_ticker(sym, exchange)
                        if ticker:
                            bid = float(ticker.get("bid", 0) or 0)
                            ask = float(ticker.get("ask", 0) or 0)
                            last = float(ticker.get("last", 0) or 0)
                            price = last or ((bid + ask) / 2.0 if bid and ask else 0.0)

                            data = {
                                "symbol": sym,
                                "price": price,
                                "bid": bid,
                                "ask": ask,
                                "spread": (ask - bid) if bid and ask else 0.0,
                                "volume": float(ticker.get("baseVolume", 0) or ticker.get("volume", 0) or 0),
                                "timestamp": datetime.now(timezone.utc),
                                "_mono": time.monotonic(),
                                "_source": "rest",
                            }
                            self._latest[sym] = data
                            self._fire_tick_callbacks(data)
                    except Exception as exc:
                        logger.debug("LiveMarketData: REST poll error for %s: %s", sym, exc)

                await asyncio.sleep(self._rest_fallback_interval_s)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("LiveMarketData: REST poll loop error: %s", exc)

    async def _fetch_rest_ticker(self, symbol: str, exchange: str) -> Optional[Dict[str, Any]]:
        """Fetch ticker via REST (ccxt exchange or market_data_service)."""
        # Try market_data_service first
        if self._market_data_service is not None:
            try:
                return await self._market_data_service.fetch_ticker(symbol)
            except Exception as _e:
                logger.debug("live_market_data error: %s", _e)

        # Try ccxt exchange directly
        ex = self._exchanges.get(exchange)
        if ex is not None and hasattr(ex, "fetch_ticker"):
            try:
                return await ex.fetch_ticker(symbol)
            except Exception as _e:
                logger.debug("live_market_data error: %s", _e)

        return None
