"""Push 75 — StrategyRegistry: dynamic strategy registration and loading.

Supports:
  - register(name, class_or_instance)
  - get(name) -> class
  - instantiate(name, config) -> BaseStrategy
  - load_from_config(list[dict]) -> list[BaseStrategy]
  - Singleton global registry via get_registry()
"""
from __future__ import annotations

from typing import Dict, List, Optional, Type, Union

from core.strategy.base_strategy import BaseStrategy, StrategyConfig


class StrategyRegistry:
    """Registry for strategy classes.

    Usage:
        registry = StrategyRegistry()
        registry.register("momentum", MomentumStrategy)
        strategy = registry.instantiate("momentum", config)
    """

    def __init__(self):
        self._registry: Dict[str, Type[BaseStrategy]] = {}

    def register(
        self,
        name: str,
        strategy_class: Type[BaseStrategy],
        overwrite: bool = False,
    ) -> None:
        """Register a strategy class under a name."""
        name = name.lower().strip()
        if name in self._registry and not overwrite:
            raise ValueError(
                f"Strategy '{name}' already registered. "
                f"Use overwrite=True to replace."
            )
        if not (isinstance(strategy_class, type) and
                issubclass(strategy_class, BaseStrategy)):
            raise TypeError(
                f"strategy_class must be a subclass of BaseStrategy, "
                f"got {strategy_class!r}"
            )
        self._registry[name] = strategy_class

    def get(self, name: str) -> Type[BaseStrategy]:
        """Return the strategy class for name. Raises KeyError if not found."""
        name = name.lower().strip()
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"Strategy '{name}' not registered. "
                f"Available: {available}"
            )
        return self._registry[name]

    def instantiate(
        self,
        name: str,
        config: StrategyConfig,
    ) -> BaseStrategy:
        """Instantiate a registered strategy with config."""
        cls = self.get(name)
        return cls(config)

    def load_from_config(self, configs: List[dict]) -> List[BaseStrategy]:
        """Bulk-instantiate strategies from a list of config dicts.

        Each dict must have:
          - 'name': registered strategy name
          - 'strategy_id': unique identifier
          - 'symbol': trading pair
          - optional StrategyConfig fields
        """
        strategies = []
        for cfg_dict in configs:
            name = cfg_dict.pop("name")
            config = StrategyConfig(**cfg_dict)
            strategies.append(self.instantiate(name, config))
        return strategies

    def list_names(self) -> List[str]:
        return sorted(self._registry.keys())

    def unregister(self, name: str) -> None:
        self._registry.pop(name.lower().strip(), None)

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, name: str) -> bool:
        return name.lower().strip() in self._registry


# Global singleton registry
_GLOBAL_REGISTRY: Optional[StrategyRegistry] = None


def get_registry() -> StrategyRegistry:
    """Return the global StrategyRegistry, auto-populated with built-ins."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = StrategyRegistry()
        _register_builtins(_GLOBAL_REGISTRY)
    return _GLOBAL_REGISTRY


def _register_builtins(registry: StrategyRegistry) -> None:
    from core.strategy.momentum_strategy import MomentumStrategy
    from core.strategy.mean_reversion_strategy import MeanReversionStrategy
    from core.strategy.ml_strategy import MLStrategy
    registry.register("momentum",       MomentumStrategy)
    registry.register("mean_reversion",  MeanReversionStrategy)
    registry.register("ml",              MLStrategy)
