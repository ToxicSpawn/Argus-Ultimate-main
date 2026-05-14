"""Push 67 — StrategyController: Hummingbot V2-style base controller.

Architecture:
  Controller proposes actions -> Executors decide execution

Public API:
  on_tick()                  — 1-second heartbeat (override)
  create_actions_proposal()  — propose open actions (override)
  stop_actions_proposal()    — propose close actions (override)
  on_bar(bar)                — called on each OHLCV bar
  start() / stop()           — lifecycle
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from core.strategy.v2.controller_config import ControllerConfig


class ActionType(str, Enum):
    OPEN_LONG  = "OPEN_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT= "CLOSE_SHORT"
    HOLD       = "HOLD"


@dataclass
class ExecutorAction:
    action_type: ActionType
    symbol: str
    size_usd: float = 0.0
    price: Optional[float] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    proposed_at: float = field(default_factory=time.time)


class StrategyController:
    """Base V2 controller. Subclass and override on_tick /
    create_actions_proposal / stop_actions_proposal.
    """

    name: str = "BaseController"

    def __init__(self, config: ControllerConfig | None = None):
        self.config = config or ControllerConfig()
        self._running = False
        self._tick_count: int = 0
        self._tick_task: Optional[asyncio.Task] = None
        self._last_actions: List[ExecutorAction] = []
        self._executors: Dict[str, Any] = {}
        self._start_time: float = 0.0
        self._bars_processed: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        self._start_time = time.time()
        self._tick_task = asyncio.create_task(self._tick_loop())
        await self.on_start()

    async def stop(self) -> None:
        self._running = False
        if self._tick_task and not self._tick_task.done():
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass
        await self.on_stop()

    # ------------------------------------------------------------------
    # Tick loop (1-second heartbeat)
    # ------------------------------------------------------------------

    async def _tick_loop(self) -> None:
        while self._running:
            await asyncio.sleep(1.0)
            if not self._running:
                break
            self._tick_count += 1
            try:
                await self.on_tick()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Override hooks
    # ------------------------------------------------------------------

    async def on_start(self) -> None:
        """Called once on controller start."""

    async def on_stop(self) -> None:
        """Called once on controller stop."""

    async def on_tick(self) -> None:
        """1-second heartbeat. Override to implement tick-driven logic."""

    async def on_bar(self, bar: Any) -> List[ExecutorAction]:
        """Called on each OHLCV bar. Returns proposed actions."""
        self._bars_processed += 1
        actions = await self.create_actions_proposal(bar)
        self._last_actions = actions
        return actions

    async def create_actions_proposal(
        self, bar: Any
    ) -> List[ExecutorAction]:
        """Override: propose open actions based on bar data."""
        return []

    async def stop_actions_proposal(
        self, bar: Any
    ) -> List[ExecutorAction]:
        """Override: propose close actions based on bar data."""
        return []

    # ------------------------------------------------------------------
    # Executor registry
    # ------------------------------------------------------------------

    def register_executor(self, name: str, executor: Any) -> None:
        self._executors[name] = executor

    def get_executor(self, name: str) -> Any:
        return self._executors.get(name)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def bars_processed(self) -> int:
        return self._bars_processed

    @property
    def uptime_s(self) -> float:
        return time.time() - self._start_time if self._start_time else 0.0
