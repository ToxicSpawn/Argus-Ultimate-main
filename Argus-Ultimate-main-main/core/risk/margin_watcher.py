"""Push 78 — MarginWatcher: async margin ratio monitor.

Margin ratio = used_margin / equity

Thresholds (configurable):
  soft_threshold: warn + emit MARGIN_SOFT event
  hard_threshold: auto-reduce largest position + emit MARGIN_HARD
  critical_threshold: activate kill switch

Polling:
  Runs as asyncio task at poll_interval_secs.
  Calls adapter.get_balance() for equity.
  Calls order_manager.stats for open notionals.

Auto-reduce:
  Cancels all open orders for the largest-notional symbol.
  Optionally emits a POSITION_REDUCED risk event.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

from core.risk.risk_event import RiskEvent, RiskEventBus, RiskEventType


@dataclass
class MarginConfig:
    soft_threshold:     float = 0.70   # 70% margin utilisation
    hard_threshold:     float = 0.85   # 85% → auto-reduce
    critical_threshold: float = 0.95   # 95% → kill switch
    poll_interval_secs: float = 5.0
    initial_equity:     float = 10_000.0


class MarginWatcher:
    """Async margin ratio monitor.

    Args:
        config:        MarginConfig
        order_manager: OrderManager (to query notionals + cancel orders)
        adapter:       ExchangeAdapter (to query balance)
        risk_manager:  RiskManager (to activate kill switch)
        event_bus:     RiskEventBus
    """

    def __init__(
        self,
        config:        Optional[MarginConfig] = None,
        order_manager  = None,
        adapter        = None,
        risk_manager   = None,
        event_bus:     Optional[RiskEventBus] = None,
    ):
        self.config        = config or MarginConfig()
        self._om           = order_manager
        self._adapter      = adapter
        self._rm           = risk_manager
        self.event_bus     = event_bus or RiskEventBus()
        self._running      = False
        self._task:        Optional[asyncio.Task] = None
        self._last_ratio:  float = 0.0
        self._poll_count:  int   = 0
        self._breach_count: int  = 0

    async def start(self) -> None:
        self._running = True
        self._task    = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self._check_margin()
            except Exception:
                pass
            await asyncio.sleep(self.config.poll_interval_secs)

    async def _get_equity(self) -> float:
        if self._adapter:
            try:
                return await self._adapter.get_balance("USDT")
            except Exception:
                pass
        return self.config.initial_equity

    def _get_used_margin(self) -> float:
        """Sum of open notionals from order manager."""
        if self._om is None:
            return 0.0
        stats = self._om.stats
        positions = stats.get("positions", {})
        return sum(p.get("notional", 0.0) for p in positions.values())

    async def _check_margin(self) -> None:
        equity      = await self._get_equity()
        used_margin = self._get_used_margin()
        ratio       = used_margin / equity if equity > 0 else 0.0
        self._last_ratio  = ratio
        self._poll_count += 1

        if ratio >= self.config.critical_threshold:
            self._breach_count += 1
            msg = f"CRITICAL margin ratio {ratio:.2%} >= {self.config.critical_threshold:.0%}"
            self.event_bus.emit(RiskEvent(
                RiskEventType.MARGIN_HARD, msg,
                value=ratio, threshold=self.config.critical_threshold
            ))
            if self._rm:
                self._rm.activate_kill_switch(msg)
            await self._auto_reduce()

        elif ratio >= self.config.hard_threshold:
            self._breach_count += 1
            msg = f"Hard margin breach {ratio:.2%} >= {self.config.hard_threshold:.0%}"
            self.event_bus.emit(RiskEvent(
                RiskEventType.MARGIN_HARD, msg,
                value=ratio, threshold=self.config.hard_threshold
            ))
            await self._auto_reduce()

        elif ratio >= self.config.soft_threshold:
            msg = f"Soft margin warning {ratio:.2%} >= {self.config.soft_threshold:.0%}"
            self.event_bus.emit(RiskEvent(
                RiskEventType.MARGIN_SOFT, msg,
                value=ratio, threshold=self.config.soft_threshold
            ))

    async def _auto_reduce(self) -> None:
        """Cancel all open orders for largest-notional symbol."""
        if self._om is None:
            return
        stats     = self._om.stats
        positions = stats.get("positions", {})
        if not positions:
            return
        # Find largest by notional
        largest_sym = max(positions, key=lambda s: positions[s].get("notional", 0))
        open_orders = self._om.get_open_orders(symbol=largest_sym)
        for order in open_orders:
            await self._om.cancel_order(order.order_id)
        self.event_bus.emit(RiskEvent(
            RiskEventType.POSITION_REDUCED,
            f"Auto-reduced orders for {largest_sym} (margin breach)",
            symbol=largest_sym,
            value=self._last_ratio,
        ))

    # ------------------------------------------------------------------
    # Manual check (sync convenience)
    # ------------------------------------------------------------------

    def check_margin_sync(
        self,
        equity: float,
        used_margin: float,
    ) -> tuple[float, str]:
        """Synchronous margin check. Returns (ratio, level)."""
        ratio = used_margin / equity if equity > 0 else 0.0
        if ratio >= self.config.critical_threshold:
            return ratio, "CRITICAL"
        if ratio >= self.config.hard_threshold:
            return ratio, "HARD"
        if ratio >= self.config.soft_threshold:
            return ratio, "SOFT"
        return ratio, "OK"

    @property
    def stats(self) -> dict:
        return {
            "last_ratio":   round(self._last_ratio, 4),
            "poll_count":   self._poll_count,
            "breach_count": self._breach_count,
            "running":      self._running,
        }
