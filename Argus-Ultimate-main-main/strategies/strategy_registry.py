"""
StrategyRegistry
================
Central registry for all strategy instances.  Acts as the single
source of truth for enabling / disabling strategies at runtime
without restarting the bot.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("argus.strategy_registry")


class StrategyRegistry:
    """
    Thread-safe (GIL-sufficient for asyncio) strategy catalogue.

    Usage::

        registry = StrategyRegistry()
        registry.register(momentum_strategy)
        registry.disable("mean_reversion")
        active = registry.active_strategies()
    """

    def __init__(self) -> None:
        self._strategies: Dict[str, Any] = {}   # name -> strategy object
        self._enabled: Dict[str, bool] = {}
        self._tags: Dict[str, List[str]] = {}   # name -> [tag, ...]

    # ------------------------------------------------------------------ #
    # Registration                                                         #
    # ------------------------------------------------------------------ #

    def register(
        self,
        strategy: Any,
        *,
        tags: Optional[List[str]] = None,
        enabled: bool = True,
    ) -> None:
        name = getattr(strategy, "name", strategy.__class__.__name__)
        if name in self._strategies:
            logger.debug("Registry: overwriting %s", name)
        self._strategies[name] = strategy
        self._enabled[name] = enabled
        self._tags[name] = tags or []
        logger.debug("Registry: registered %s (enabled=%s)", name, enabled)

    def register_all(
        self,
        strategies: List[Any],
        *,
        tags: Optional[List[str]] = None,
        enabled: bool = True,
    ) -> None:
        for s in strategies:
            self.register(s, tags=tags, enabled=enabled)

    # ------------------------------------------------------------------ #
    # Enable / disable                                                     #
    # ------------------------------------------------------------------ #

    def enable(self, name: str) -> None:
        if name in self._enabled:
            self._enabled[name] = True
            logger.info("Registry: enabled %s", name)
        else:
            logger.warning("Registry: unknown strategy %s", name)

    def disable(self, name: str) -> None:
        if name in self._enabled:
            self._enabled[name] = False
            logger.info("Registry: disabled %s", name)
        else:
            logger.warning("Registry: unknown strategy %s", name)

    def set_enabled_subset(self, names: List[str]) -> None:
        """Enable exactly the strategies in *names*, disable all others."""
        for n in self._enabled:
            self._enabled[n] = n in names

    # ------------------------------------------------------------------ #
    # Queries                                                              #
    # ------------------------------------------------------------------ #

    def get(self, name: str) -> Optional[Any]:
        return self._strategies.get(name)

    def active_strategies(self) -> List[Any]:
        return [s for n, s in self._strategies.items() if self._enabled.get(n, False)]

    def all_strategies(self) -> List[Any]:
        return list(self._strategies.values())

    def names(self, *, active_only: bool = False) -> List[str]:
        if active_only:
            return [n for n, en in self._enabled.items() if en]
        return list(self._strategies.keys())

    def by_tag(self, tag: str) -> List[Any]:
        return [
            self._strategies[n]
            for n, tags in self._tags.items()
            if tag in tags and self._enabled.get(n, False)
        ]

    def is_enabled(self, name: str) -> bool:
        return self._enabled.get(name, False)

    def summary(self) -> List[dict]:
        return [
            {
                "name": n,
                "enabled": self._enabled.get(n, False),
                "tags": self._tags.get(n, []),
                "class": type(s).__name__,
            }
            for n, s in self._strategies.items()
        ]
