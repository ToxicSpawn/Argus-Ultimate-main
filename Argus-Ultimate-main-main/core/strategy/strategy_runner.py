"""StrategyRunner — lifecycle manager + tick/fill dispatcher — Push 58.

Manages instantiated AbstractStrategy objects:
  - start / pause / resume / stop lifecycle
  - dispatch_tick() fans out to all RUNNING strategies
  - dispatch_fill() routes fill to originating strategy
  - Per-strategy error isolation: exceptions caught,
    strategy moved to ERROR state, ALERT broadcast via WsHub
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from core.strategy.base_strategy import AbstractStrategy, StrategyState
from core.strategy.strategy_registry import StrategyRegistry

logger = logging.getLogger(__name__)


class StrategyRunner:
    """Manages and dispatches events to strategy instances.

    Parameters
    ----------
    registry : StrategyRegistry
    hub : WsHub, optional
        If provided, broadcasts ALERT on strategy errors.
    """

    def __init__(
        self,
        registry: StrategyRegistry,
        hub: Optional[Any] = None,
    ) -> None:
        self._registry = registry
        self._hub = hub
        self._instances: Dict[str, AbstractStrategy] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, name: str, **kwargs) -> AbstractStrategy:
        """Instantiate (if needed) and start a strategy."""
        if name not in self._instances:
            self._instances[name] = self._registry.instantiate(name, **kwargs)
        strategy = self._instances[name]
        if strategy.state not in {StrategyState.IDLE, StrategyState.PAUSED}:
            logger.warning("StrategyRunner: %s already in state %s", name, strategy.state)
            return strategy
        strategy._state = StrategyState.RUNNING
        try:
            await strategy.on_start()
        except Exception as exc:
            await self._handle_error(name, strategy, exc)
        logger.info("StrategyRunner: started '%s'", name)
        return strategy

    async def pause(self, name: str) -> None:
        strategy = self._get(name)
        if strategy.state != StrategyState.RUNNING:
            return
        strategy._state = StrategyState.PAUSED
        await strategy.on_pause()
        logger.info("StrategyRunner: paused '%s'", name)

    async def resume(self, name: str) -> None:
        strategy = self._get(name)
        if strategy.state != StrategyState.PAUSED:
            return
        strategy._state = StrategyState.RUNNING
        await strategy.on_resume()
        logger.info("StrategyRunner: resumed '%s'", name)

    async def stop(self, name: str) -> None:
        strategy = self._get(name)
        if strategy.state == StrategyState.STOPPED:
            return
        strategy._state = StrategyState.STOPPED
        try:
            await strategy.on_stop()
        except Exception as exc:
            logger.error("StrategyRunner: error stopping '%s': %s", name, exc)
        logger.info("StrategyRunner: stopped '%s'", name)

    async def stop_all(self) -> None:
        for name in list(self._instances):
            await self.stop(name)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def dispatch_tick(
        self,
        symbol: str,
        price: float,
        bid: float = 0.0,
        ask: float = 0.0,
        **kwargs,
    ) -> None:
        """Fan out a market tick to all RUNNING strategies."""
        for name, strategy in list(self._instances.items()):
            if strategy.state != StrategyState.RUNNING:
                continue
            if strategy.metadata.symbols and symbol not in strategy.metadata.symbols:
                continue
            try:
                strategy._tick_count += 1
                await strategy.on_tick(symbol, price, bid=bid, ask=ask, **kwargs)
            except Exception as exc:
                await self._handle_error(name, strategy, exc)

    async def dispatch_fill(self, order: Any, fill: Any) -> None:
        """Route a fill to all RUNNING strategies."""
        for name, strategy in list(self._instances.items()):
            if strategy.state != StrategyState.RUNNING:
                continue
            try:
                strategy._fill_count += 1
                await strategy.on_fill(order, fill)
            except Exception as exc:
                await self._handle_error(name, strategy, exc)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_instance(self, name: str) -> Optional[AbstractStrategy]:
        return self._instances.get(name)

    @property
    def running_strategies(self) -> List[str]:
        return [
            n for n, s in self._instances.items()
            if s.state == StrategyState.RUNNING
        ]

    @property
    def all_strategies(self) -> List[str]:
        return list(self._instances.keys())

    def status(self) -> List[dict]:
        return [s.to_dict() for s in self._instances.values()]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get(self, name: str) -> AbstractStrategy:
        s = self._instances.get(name)
        if s is None:
            raise KeyError(f"Strategy '{name}' not instantiated")
        return s

    async def _handle_error(
        self, name: str, strategy: AbstractStrategy, exc: Exception
    ) -> None:
        strategy._state = StrategyState.ERROR
        strategy._error = str(exc)
        logger.error("StrategyRunner: strategy '%s' error: %s", name, exc)
        if self._hub is not None:
            try:
                from core.broadcast.ws_message import WsMessage
                await self._hub.broadcast(
                    WsMessage.alert("error", f"Strategy '{name}' error: {exc}")
                )
            except Exception:  # noqa: BLE001
                pass
