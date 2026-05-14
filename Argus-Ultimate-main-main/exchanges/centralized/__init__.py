"""
Centralized exchange connectors.

This repo contains many legacy stub connector modules. Some are intentionally incomplete
and/or may not import cleanly. To prevent import-time failures, we do **not** eagerly
import every exchange client here.

Preferred usage:
- `from exchanges.centralized.kraken import KrakenClient`
- `from exchanges.centralized.coinbase_advanced import CoinbaseAdvancedClient`

This module supports `from exchanges.centralized import KrakenClient` via lazy loading.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict

# Keep this list intentionally small and focused on the exchanges actively used by
# the unified system. Add more mappings as real connectors are implemented.
_LAZY_CLASS_TO_MODULE: Dict[str, str] = {
    "KrakenClient": "exchanges.centralized.kraken",
    "CoinbaseAdvancedClient": "exchanges.centralized.coinbase_advanced",
    # Backwards compatibility: legacy name kept as a thin wrapper/deprecation shim.
    "CoinbaseProClient": "exchanges.centralized.coinbase_pro",
}

__all__ = sorted(_LAZY_CLASS_TO_MODULE.keys())


def __getattr__(name: str) -> Any:
    mod_path = _LAZY_CLASS_TO_MODULE.get(name)
    if not mod_path:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(mod_path)
    try:
        return getattr(module, name)
    except AttributeError as e:
        raise AttributeError(f"module {mod_path!r} has no attribute {name!r}") from e


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
