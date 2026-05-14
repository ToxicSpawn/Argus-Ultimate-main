"""DashboardFeed — wires execution/PnL/risk into WsHub — Push 57.

Subscribes to:
  - ExecutionEngine fill callbacks  -> FILL + PNL_SNAPSHOT + RISK_STATUS
  - Market tick events              -> TICK broadcast
  - Periodic heartbeat              -> HEARTBEAT every N seconds

Usage::

    feed = DashboardFeed(hub=hub, engine=engine, pnl=pnl, risk=rm)
    feed.attach()                    # registers fill callback
    asyncio.create_task(feed.start_heartbeat(interval=10))
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from core.broadcast.ws_hub import WsHub
from core.broadcast.ws_message import MessageType, WsMessage

logger = logging.getLogger(__name__)


class DashboardFeed:
    """Bridges trading events into the WsHub broadcast stream.

    Parameters
    ----------
    hub : WsHub
    engine : ExecutionEngine, optional
    pnl : PnLTracker, optional
    risk : RiskManager, optional
    """

    def __init__(
        self,
        hub: WsHub,
        engine: Optional[Any] = None,
        pnl: Optional[Any] = None,
        risk: Optional[Any] = None,
    ) -> None:
        self._hub = hub
        self._engine = engine
        self._pnl = pnl
        self._risk = risk
        self._tick_count = 0
        self._fill_count = 0

    # ------------------------------------------------------------------
    # Attach callbacks
    # ------------------------------------------------------------------

    def attach(self) -> None:
        """Register fill callback on the execution engine."""
        if self._engine is not None:
            self._engine.add_fill_callback(self._on_fill)
            logger.info("DashboardFeed: attached to ExecutionEngine")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_fill(self, order: Any, fill: Any) -> None:
        """Called by ExecutionEngine on every fill."""
        self._fill_count += 1

        # FILL message
        fill_data = {
            "order_id": fill.order_id,
            "symbol": fill.symbol,
            "side": fill.side.value if hasattr(fill.side, "value") else fill.side,
            "price": fill.price,
            "qty": fill.qty,
            "fee": fill.fee,
            "venue": fill.venue,
        }
        await self._hub.broadcast(WsMessage(type=MessageType.FILL, data=fill_data))

        # PNL_SNAPSHOT
        await self._hub.broadcast(WsMessage(
            type=MessageType.PNL_SNAPSHOT,
            data=self._pnl_snapshot(),
        ))

        # RISK_STATUS
        await self._hub.broadcast(WsMessage(
            type=MessageType.RISK_STATUS,
            data=self._risk_snapshot(),
        ))

    async def on_tick(self, symbol: str, price: float,
                      bid: float = 0.0, ask: float = 0.0) -> None:
        """Broadcast a market tick."""
        self._tick_count += 1
        await self._hub.broadcast(WsMessage.tick(symbol, price, bid, ask))

    async def broadcast_session_stats(self) -> None:
        """Broadcast a full session stats snapshot."""
        if self._pnl is None:
            return
        stats = self._pnl.session_stats()
        await self._hub.broadcast(WsMessage(
            type=MessageType.SESSION_STATS,
            data=stats.to_dict(),
        ))

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def start_heartbeat(self, interval: float = 10.0) -> None:
        """Coroutine: broadcasts HEARTBEAT every `interval` seconds."""
        logger.info("DashboardFeed: heartbeat started (interval=%.1fs)", interval)
        while True:
            await asyncio.sleep(interval)
            await self._hub.broadcast(WsMessage.heartbeat())

    # ------------------------------------------------------------------
    # Snapshot builders
    # ------------------------------------------------------------------

    def _pnl_snapshot(self) -> dict:
        if self._pnl is None:
            return {}
        try:
            stats = self._pnl.session_stats()
            return {
                "equity": self._pnl.equity,
                "net_pnl": stats.net_pnl,
                "n_trades": stats.n_trades,
                "win_rate": stats.win_rate,
                "max_drawdown": stats.max_drawdown,
                "open_symbols": self._pnl.open_symbols,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("DashboardFeed: pnl snapshot error: %s", exc)
            return {}

    def _risk_snapshot(self) -> dict:
        if self._risk is None:
            return {}
        try:
            return self._risk.status()
        except Exception as exc:  # noqa: BLE001
            logger.warning("DashboardFeed: risk snapshot error: %s", exc)
            return {}

    def full_snapshot(self) -> dict:
        """Build a complete dashboard snapshot for new client connections."""
        return {
            "pnl": self._pnl_snapshot(),
            "risk": self._risk_snapshot(),
            "engine": self._engine.status() if self._engine else {},
            "hub": self._hub.status(),
            "ts": time.time(),
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def fill_count(self) -> int:
        return self._fill_count
