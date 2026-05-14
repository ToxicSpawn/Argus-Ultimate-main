"""Push 77 — ExecutionEngine: bridges SignalBus → OrderManager → ExchangeAdapter.

Flow:
  1. strategy.tick() → Signal
  2. AsyncSignalBus.publish(signal)
  3. ExecutionEngine._on_signal(signal)
  4. signal_to_order(signal) → Order
  5. OrderManager.submit_order(order)
  6. ExchangeAdapter.place_order(order)
  7. ExchangeAdapter fill event → OrderManager.on_fill(fill)
  8. Strategy.on_fill(fill_event) [optional]

Features:
  - Per-cycle latency tracking (microseconds)
  - Graceful start/stop via asyncio
  - Signal deduplication (cooldown per symbol)
  - Position-aware: skip LONG signal if already long
  - Paper and live mode via adapter injection
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from core.execution.order import Order, OrderSide, OrderType, PositionSide
from core.execution.order_manager import OrderManager
from core.execution.exchange_adapter import AbstractExchangeAdapter
from core.strategy.signal import Signal, SignalSide
from core.strategy.signal_bus import AsyncSignalBus


@dataclass
class EngineStats:
    signals_received:   int   = 0
    orders_submitted:   int   = 0
    orders_rejected:    int   = 0
    avg_latency_us:     float = 0.0
    min_latency_us:     float = float("inf")
    max_latency_us:     float = 0.0
    uptime_secs:        float = 0.0
    _latencies:         List[float] = field(default_factory=list)

    def record_latency(self, us: float) -> None:
        self._latencies.append(us)
        if len(self._latencies) > 1000:
            self._latencies.pop(0)
        self.avg_latency_us = sum(self._latencies) / len(self._latencies)
        self.min_latency_us = min(self.min_latency_us, us)
        self.max_latency_us = max(self.max_latency_us, us)

    def to_dict(self) -> dict:
        return {
            "signals_received": self.signals_received,
            "orders_submitted": self.orders_submitted,
            "orders_rejected":  self.orders_rejected,
            "avg_latency_us":   round(self.avg_latency_us, 1),
            "min_latency_us":   round(self.min_latency_us, 1) if self.min_latency_us != float("inf") else 0,
            "max_latency_us":   round(self.max_latency_us, 1),
            "uptime_secs":      round(self.uptime_secs, 1),
        }


class ExecutionEngine:
    """Async execution engine connecting signals to exchange orders.

    Args:
        order_manager:  OrderManager instance
        adapter:        ExchangeAdapter (PaperAdapter or BinanceAdapter)
        signal_bus:     AsyncSignalBus to subscribe to
        signal_cooldown_secs: Min seconds between orders per symbol
        initial_equity: Used for Kelly sizing
    """

    def __init__(
        self,
        order_manager:        OrderManager,
        adapter:              AbstractExchangeAdapter,
        signal_bus:           Optional[AsyncSignalBus] = None,
        signal_cooldown_secs: float = 5.0,
        initial_equity:       float = 10_000.0,
    ):
        self._om             = order_manager
        self._adapter        = adapter
        self._bus            = signal_bus or AsyncSignalBus()
        self._cooldown       = signal_cooldown_secs
        self._equity         = initial_equity
        self._running        = False
        self._start_time:    Optional[float] = None
        self._last_signal_ts: Dict[str, float] = {}
        self._stats          = EngineStats()
        self._sub_id:        Optional[str] = None
        self._tasks:         List[asyncio.Task] = []

    # ------------------------------------------------------------------
    # Signal → Order conversion
    # ------------------------------------------------------------------

    def signal_to_order(
        self,
        signal: Signal,
        current_price: float,
        equity: Optional[float] = None,
    ) -> Optional[Order]:
        """Convert a Signal to an Order. Returns None if no action needed."""
        eq = equity or self._equity
        if current_price <= 0 or eq <= 0:
            return None

        # Position-aware deduplication
        pos = self._om.get_position(signal.symbol)
        if pos and not pos.is_flat:
            if signal.side == SignalSide.LONG  and pos.side == PositionSide.LONG:
                return None
            if signal.side == SignalSide.SHORT and pos.side == PositionSide.SHORT:
                return None

        if signal.side == SignalSide.FLAT:
            # Close existing position
            if pos is None or pos.is_flat:
                return None
            side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY
            qty  = pos.qty
        elif signal.side == SignalSide.LONG:
            side = OrderSide.BUY
            qty  = (eq * 0.25 * signal.strength) / current_price
        else:  # SHORT
            side = OrderSide.SELL
            qty  = (eq * 0.25 * signal.strength) / current_price

        if qty <= 1e-9:
            return None

        return Order(
            symbol=signal.symbol,
            side=side,
            order_type=OrderType.MARKET,
            qty=round(qty, 6),
            strategy_id=signal.strategy_id,
            signal_strength=signal.strength,
        )

    # ------------------------------------------------------------------
    # Signal handler (subscribed to bus)
    # ------------------------------------------------------------------

    async def _on_signal(self, signal: Signal) -> None:
        t0 = time.perf_counter()
        self._stats.signals_received += 1

        # Cooldown check
        last = self._last_signal_ts.get(signal.symbol, 0.0)
        if time.time() - last < self._cooldown:
            return
        self._last_signal_ts[signal.symbol] = time.time()

        # Get current price from adapter (paper: injected, live: last tick)
        price = getattr(self._adapter, "_prices", {}).get(signal.symbol, 0.0)
        if signal.price and signal.price > 0:
            price = signal.price
        if price <= 0:
            return

        order = self.signal_to_order(signal, price)
        if order is None:
            return

        submitted = await self._om.submit_order(order)
        if submitted.status.value in ("REJECTED",):
            self._stats.orders_rejected += 1
            return

        self._stats.orders_submitted += 1
        try:
            exchange_id = await self._adapter.place_order(submitted)
            submitted.exchange_id = exchange_id
        except Exception:
            submitted.status = __import__("core.execution.order", fromlist=["OrderStatus"]).OrderStatus.REJECTED

        latency_us = (time.perf_counter() - t0) * 1_000_000
        self._stats.record_latency(latency_us)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the engine: subscribe to signal bus."""
        self._running   = True
        self._start_time = time.time()
        self._sub_id    = self._bus.subscribe(self._on_signal)

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        if self._sub_id:
            self._bus.unsubscribe(self._sub_id)
        for task in self._tasks:
            task.cancel()
        await self._adapter.close()
        if self._start_time:
            self._stats.uptime_secs = time.time() - self._start_time

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        if self._start_time:
            self._stats.uptime_secs = time.time() - self._start_time
        return self._stats.to_dict()

    @property
    def is_running(self) -> bool:
        return self._running
