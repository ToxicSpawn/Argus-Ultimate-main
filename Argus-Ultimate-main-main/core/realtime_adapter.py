#!/usr/bin/env python3
"""
Realtime Adapter — thin bridge between LiveMarketDataManager tick streams
and the main trading loop's signal pipeline.

Subscribes to ticker updates and writes them into a thread-safe ring buffer
that the async trading loop polls each cycle.
"""

from __future__ import annotations

import asyncio
import logging
import time
import threading
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


class RealtimeAdapter:
    """
    Bridges a LiveMarketDataManager into the trading loop.

    Call `subscribe(symbols)` to start receiving ticks.
    Call `get_latest(symbol)` from the trading loop to fetch the most recent tick.
    Register callbacks via `on_tick(fn)` for push-style consumption.
    """

    def __init__(self, live_market_data: Any = None, *, buffer_size: int = 200):
        self.live_market_data = live_market_data
        self.buffer_size = max(10, int(buffer_size))
        self._buffers: Dict[str, Deque[Dict[str, Any]]] = {}
        self._callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        self._lock = threading.Lock()
        self._subscribed: bool = False
        self._tick_count: int = 0

    def on_tick(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Register a callback(symbol, tick) fired on every incoming tick."""
        self._callbacks.append(callback)

    async def subscribe(self, symbols: List[str]) -> None:
        """Wire tick callbacks from LiveMarketDataManager for the given symbols."""
        lmd = self.live_market_data
        if lmd is None:
            logger.warning("RealtimeAdapter: no LiveMarketDataManager attached")
            return
        for sym in symbols:
            with self._lock:
                if sym not in self._buffers:
                    self._buffers[sym] = deque(maxlen=self.buffer_size)
        # Attempt to register with the live data manager
        if hasattr(lmd, "register_tick_callback"):
            lmd.register_tick_callback(self._on_tick_internal)
            self._subscribed = True
            logger.info("RealtimeAdapter: subscribed to %d symbols via LiveMarketDataManager", len(symbols))
        elif hasattr(lmd, "subscribe"):
            try:
                await lmd.subscribe(symbols, callback=self._on_tick_internal)
                self._subscribed = True
                logger.info("RealtimeAdapter: async-subscribed to %s", symbols)
            except Exception as e:
                logger.warning("RealtimeAdapter subscribe: %s", e)
        else:
            logger.warning("RealtimeAdapter: LiveMarketDataManager has no subscription interface")

    def _on_tick_internal(self, symbol: str, tick: Dict[str, Any]) -> None:
        """Internal callback — writes to buffer and fires registered callbacks."""
        sym = str(symbol or "")
        entry = dict(tick)
        entry.setdefault("_ts", time.time())
        with self._lock:
            if sym not in self._buffers:
                self._buffers[sym] = deque(maxlen=self.buffer_size)
            self._buffers[sym].append(entry)
        self._tick_count += 1
        for cb in self._callbacks:
            try:
                cb(sym, entry)
            except Exception:
                pass

    def get_latest(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return the most recent tick for a symbol, or None."""
        with self._lock:
            buf = self._buffers.get(str(symbol or ""))
            if buf:
                return dict(buf[-1])
        return None

    def get_recent(self, symbol: str, n: int = 10) -> List[Dict[str, Any]]:
        """Return up to n most recent ticks for a symbol."""
        with self._lock:
            buf = self._buffers.get(str(symbol or ""))
            if buf:
                return [dict(t) for t in list(buf)[-n:]]
        return []

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "subscribed": self._subscribed,
                "symbols": len(self._buffers),
                "tick_count": self._tick_count,
                "buffer_sizes": {sym: len(buf) for sym, buf in self._buffers.items()},
            }
