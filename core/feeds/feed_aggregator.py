"""
feed_aggregator.py
------------------
Cross-venue quote aggregator.

Consumes CanonicalTick from a FeedRouter subscriber queue and publishes:
  - AggregatedQuote: best bid/ask across all venues (BBO merge)
  - VWAP mid: volume-weighted average of last prices
  - Per-symbol spread tracker

Designed to run as a background coroutine.  Interested components
subscribe via register_callback(symbol, async_fn).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

from .feed_normaliser import CanonicalTick

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AggregatedQuote:
    symbol: str
    best_bid: Decimal           # highest bid across venues
    best_ask: Decimal           # lowest ask across venues
    best_bid_venue: str
    best_ask_venue: str
    vwap_mid: Decimal           # volume-weighted mid
    spread: Decimal
    ts: float                   # UTC epoch seconds

    @property
    def mid(self) -> Decimal:
        return (self.best_bid + self.best_ask) / 2


class FeedAggregator:
    """
    Aggregates ticks from all venues into a single BBO + VWAP view.

    Parameters
    ----------
    tick_queue : asyncio.Queue[CanonicalTick]
        Queue produced by FeedRouter.subscribe('tick', symbol).
    symbols : list[str]
        Symbols to track.
    vwap_window : int
        Number of recent ticks per venue to include in VWAP (default 20).
    """

    def __init__(
        self,
        tick_queues: Dict[str, asyncio.Queue],   # symbol → queue
        vwap_window: int = 20,
    ) -> None:
        self._queues = tick_queues
        self._vwap_window = vwap_window

        # (symbol, venue) → latest CanonicalTick
        self._latest: Dict[Tuple[str, str], CanonicalTick] = {}

        # (symbol, venue) → deque of (last_price, volume) for VWAP
        self._vwap_buf: Dict[Tuple[str, str], List[Tuple[Decimal, Decimal]]] = defaultdict(list)

        # symbol → latest AggregatedQuote
        self._agg: Dict[str, AggregatedQuote] = {}

        # symbol → list of async callbacks
        self._callbacks: Dict[str, List[Callable[[AggregatedQuote], Coroutine]]] = defaultdict(list)

        self._task: Optional[asyncio.Task] = None

    def register_callback(
        self,
        symbol: str,
        fn: Callable[[AggregatedQuote], Coroutine],
    ) -> None:
        self._callbacks[symbol].append(fn)

    async def start(self) -> None:
        tasks = [
            asyncio.ensure_future(self._drain(symbol, q))
            for symbol, q in self._queues.items()
        ]
        self._task = asyncio.ensure_future(self._noop())  # placeholder handle
        logger.info("FeedAggregator started for %d symbols", len(self._queues))

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()

    def get_quote(self, symbol: str) -> Optional[AggregatedQuote]:
        return self._agg.get(symbol)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _drain(self, symbol: str, q: asyncio.Queue) -> None:
        while True:
            tick: CanonicalTick = await q.get()
            self._update(symbol, tick)

    def _update(self, symbol: str, tick: CanonicalTick) -> None:
        key = (symbol, tick.venue)
        self._latest[key] = tick

        # Update VWAP buffer
        buf = self._vwap_buf[key]
        buf.append((tick.last, tick.volume_24h))
        if len(buf) > self._vwap_window:
            buf.pop(0)

        # Recompute BBO across all venues for this symbol
        venue_ticks = [
            v for k, v in self._latest.items() if k[0] == symbol
        ]
        if not venue_ticks:
            return

        best_bid_tick = max(venue_ticks, key=lambda t: t.bid)
        best_ask_tick = min(venue_ticks, key=lambda t: t.ask)

        # VWAP mid across all venues
        all_bufs = [
            self._vwap_buf[(symbol, t.venue)]
            for t in venue_ticks
        ]
        total_vol = Decimal(0)
        total_pv  = Decimal(0)
        for buf in all_bufs:
            for price, vol in buf:
                total_pv  += price * vol
                total_vol += vol
        vwap = total_pv / total_vol if total_vol else best_bid_tick.mid

        aq = AggregatedQuote(
            symbol=symbol,
            best_bid=best_bid_tick.bid,
            best_ask=best_ask_tick.ask,
            best_bid_venue=best_bid_tick.venue,
            best_ask_venue=best_ask_tick.venue,
            vwap_mid=vwap,
            spread=best_ask_tick.ask - best_bid_tick.bid,
            ts=tick.ts,
        )
        self._agg[symbol] = aq

        # Fire callbacks
        cbs = self._callbacks.get(symbol, [])
        for cb in cbs:
            try:
                asyncio.ensure_future(cb(aq))
            except Exception as exc:
                logger.debug("FeedAggregator callback error: %s", exc)

    async def _noop(self) -> None:
        """Placeholder coroutine to hold a task handle."""
        await asyncio.sleep(float("inf"))
