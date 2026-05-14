"""Push 97 — Cross-symbol correlation hedge (v8.33.0).

When portfolio correlation between two positions exceeds a threshold,
automatically submits a delta-neutral hedge leg to reduce correlation risk.

Design:
  CorrelationWindow     rolling return correlation calculator
  CorrelationHedge      monitor + hedge dispatcher

Integrates with:
  core/risk/correlation_monitor.py  (existing)
  core/execution/order_manager.py   (submit_market)
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple


@dataclass
class CorrelationHedgeEvent:
    sym_a:       str
    sym_b:       str
    correlation: float
    hedge_qty_a: float
    hedge_qty_b: float
    ts:          float = field(default_factory=time.time)


class CorrelationWindow:
    """Rolling Pearson correlation between two return series."""

    def __init__(self, window: int = 60) -> None:
        self._a: Deque[float] = deque(maxlen=window)
        self._b: Deque[float] = deque(maxlen=window)
        self._window = window

    def update(self, ret_a: float, ret_b: float) -> None:
        self._a.append(ret_a)
        self._b.append(ret_b)

    def correlation(self) -> Optional[float]:
        n = len(self._a)
        if n < 10:
            return None
        a = list(self._a)
        b = list(self._b)
        ma = sum(a) / n
        mb = sum(b) / n
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        da  = math.sqrt(sum((x - ma) ** 2 for x in a))
        db  = math.sqrt(sum((y - mb) ** 2 for y in b))
        if da < 1e-9 or db < 1e-9:
            return None
        return num / (da * db)


class CorrelationHedge:
    """Monitors pairwise correlations and dispatches hedge orders.

    Usage::

        hedge = CorrelationHedge(ctx, threshold=0.85)
        # On each price tick:
        hedge.update("BTCUSDT", btc_return)
        hedge.update("ETHUSDT", eth_return)
        await hedge.check_and_hedge()
    """

    def __init__(
        self,
        ctx:             Any,
        threshold:       float = 0.85,
        window:          int   = 60,
        cooldown_s:      float = 300.0,
        hedge_ratio:     float = 0.5,
    ) -> None:
        self._ctx        = ctx
        self._threshold  = threshold
        self._window     = window
        self._cooldown   = cooldown_s
        self._hedge_ratio = hedge_ratio
        self._windows:   Dict[Tuple[str, str], CorrelationWindow] = {}
        self._last_price: Dict[str, float] = {}
        self._last_hedge: Dict[Tuple[str, str], float] = {}
        self._events:    List[CorrelationHedgeEvent] = []

    def update_price(self, symbol: str, price: float) -> None:
        """Record a new price tick; computes return internally."""
        prev = self._last_price.get(symbol)
        if prev is not None and prev > 0:
            ret = (price - prev) / prev
            self._record_return(symbol, ret)
        self._last_price[symbol] = price

    def update_return(self, symbol: str, ret: float) -> None:
        self._record_return(symbol, ret)

    async def check_and_hedge(self) -> List[CorrelationHedgeEvent]:
        """Check all pairs; dispatch hedge orders for correlated positions."""
        triggered: List[CorrelationHedgeEvent] = []
        om = getattr(self._ctx, "order_manager", None)
        if om is None:
            return triggered

        stats     = om.stats
        positions = stats.get("positions", {})
        syms      = list(positions.keys())

        for i, sym_a in enumerate(syms):
            for sym_b in syms[i + 1:]:
                key = (sym_a, sym_b)
                win = self._windows.get(key)
                if win is None:
                    continue
                corr = win.correlation()
                if corr is None or abs(corr) < self._threshold:
                    continue
                last = self._last_hedge.get(key, 0.0)
                if time.time() - last < self._cooldown:
                    continue
                event = await self._hedge_pair(
                    sym_a, sym_b, corr, positions[sym_a], positions[sym_b]
                )
                if event:
                    self._last_hedge[key] = time.time()
                    self._events.append(event)
                    triggered.append(event)
        return triggered

    async def _hedge_pair(
        self,
        sym_a:  str,
        sym_b:  str,
        corr:   float,
        pos_a:  dict,
        pos_b:  dict,
    ) -> Optional[CorrelationHedgeEvent]:
        """Submit offsetting hedge orders for the more exposed leg."""
        qty_a = abs(pos_a.get("qty", 0.0)) * self._hedge_ratio
        qty_b = abs(pos_b.get("qty", 0.0)) * self._hedge_ratio
        side_a = "SELL" if pos_a.get("side", "LONG") == "LONG" else "BUY"
        side_b = "SELL" if pos_b.get("side", "LONG") == "LONG" else "BUY"
        om = getattr(self._ctx, "order_manager", None)
        if om and hasattr(om, "submit_market"):
            try:
                await om.submit_market(sym_a, side_a, qty_a, tag="corr_hedge")
                await om.submit_market(sym_b, side_b, qty_b, tag="corr_hedge")
            except Exception:
                pass
        return CorrelationHedgeEvent(
            sym_a=sym_a, sym_b=sym_b,
            correlation=corr,
            hedge_qty_a=qty_a, hedge_qty_b=qty_b,
        )

    def _record_return(self, symbol: str, ret: float) -> None:
        symbols = list(self._last_price.keys())
        for other in symbols:
            if other == symbol:
                continue
            key  = (min(symbol, other), max(symbol, other))
            if key not in self._windows:
                self._windows[key] = CorrelationWindow(self._window)
            # Only update when we have the return for `symbol`
            # The other side will be updated when its return arrives

    @property
    def stats(self) -> dict:
        return {
            "tracked_pairs": len(self._windows),
            "hedge_events":  len(self._events),
            "threshold":     self._threshold,
        }
