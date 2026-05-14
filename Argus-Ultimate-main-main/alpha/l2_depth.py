"""
L2 Order Book Depth Imbalance Signal

Fetches top-N bid/ask levels from Kraken and computes:
  - bid_volume / (bid_volume + ask_volume) → imbalance [0, 1]
  - Toxicity flag: imbalance > 0.7 = buy pressure, < 0.3 = sell pressure
  - Direction signal: +1 (bullish), -1 (bearish), 0 (neutral)

Exposes L2Signal for LiveSignalBus and direct use in run_ultimate.py.

Usage:
    l2 = L2DepthSignal(exchange)
    sig = await l2.compute("BTC/AUD", depth=10)
    if sig.direction == 1 and sig.imbalance > 0.65:
        # additional bullish confirmation
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Thresholds for directional signal
_BUY_PRESSURE_THRESHOLD = 0.65
_SELL_PRESSURE_THRESHOLD = 0.35
_CACHE_TTL_S = 2.0  # L2 data stales quickly; 2-second cache


@dataclass
class L2Signal:
    symbol: str
    imbalance: float          # bid_vol / total_vol  [0, 1]
    bid_volume: float
    ask_volume: float
    direction: int            # +1 / -1 / 0
    toxic: bool               # True if extreme imbalance
    spread_bps: float
    fetched_at: float

    @property
    def confidence(self) -> float:
        """Confidence score derived from imbalance extremity."""
        return round(abs(self.imbalance - 0.5) * 2.0, 4)  # 0 at 50/50, 1 at 100/0


class L2DepthSignal:
    """Async L2 order book depth imbalance computer with TTL cache."""

    def __init__(self, exchange: Any, cache_ttl: float = _CACHE_TTL_S) -> None:
        self._exchange = exchange
        self._cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, L2Signal]] = {}

    async def compute(
        self,
        symbol: str,
        depth: int = 10,
        force: bool = False,
    ) -> Optional[L2Signal]:
        """
        Compute L2 imbalance signal for `symbol`.

        Parameters
        ----------
        depth : int
            Number of bid/ask levels to aggregate (5, 10, or 20).
        force : bool
            Bypass TTL cache.

        Returns
        -------
        L2Signal or None on fetch failure.
        """
        now = time.monotonic()
        cached_ts, cached_sig = self._cache.get(symbol, (0.0, None))
        if not force and cached_sig and (now - cached_ts) < self._cache_ttl:
            return cached_sig

        try:
            ob = await self._exchange.fetch_order_book(symbol, limit=depth)
        except Exception as exc:
            logger.warning("L2DepthSignal: fetch_order_book failed %s: %s", symbol, exc)
            return None

        bids = ob.get("bids", [])[:depth]
        asks = ob.get("asks", [])[:depth]

        if not bids or not asks:
            return None

        bid_vol = sum(float(b[1]) for b in bids)
        ask_vol = sum(float(a[1]) for a in asks)
        total_vol = bid_vol + ask_vol

        if total_vol < 1e-10:
            return None

        imbalance = bid_vol / total_vol

        if imbalance >= _BUY_PRESSURE_THRESHOLD:
            direction = 1
            toxic = imbalance > 0.80
        elif imbalance <= _SELL_PRESSURE_THRESHOLD:
            direction = -1
            toxic = imbalance < 0.20
        else:
            direction = 0
            toxic = False

        # Spread in bps
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = (best_bid + best_ask) / 2
        spread_bps = ((best_ask - best_bid) / mid * 10_000) if mid > 0 else 0.0

        sig = L2Signal(
            symbol=symbol,
            imbalance=round(imbalance, 4),
            bid_volume=round(bid_vol, 6),
            ask_volume=round(ask_vol, 6),
            direction=direction,
            toxic=toxic,
            spread_bps=round(spread_bps, 2),
            fetched_at=time.time(),
        )
        self._cache[symbol] = (now, sig)
        return sig

    async def batch_compute(
        self,
        symbols: list[str],
        depth: int = 10,
    ) -> dict[str, Optional[L2Signal]]:
        """Concurrently fetch L2 signals for multiple symbols."""
        tasks = [self.compute(sym, depth=depth) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: dict[str, Optional[L2Signal]] = {}
        for sym, res in zip(symbols, results):
            if isinstance(res, Exception):
                logger.warning("L2DepthSignal batch error %s: %s", sym, res)
                out[sym] = None
            else:
                out[sym] = res
        return out
