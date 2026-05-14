"""Dynamic Kelly integration layer.

Drops `DynamicKellySizer` into the existing `_calculate_quantity` flow
via a module-level singleton.  Import `get_kelly_sizer()` anywhere.

The sizer is fed realised PnL after each fill via `record_trade_pnl()`.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_KELLY_SIZER: Optional[Any] = None


def get_kelly_sizer(config: Optional[Any] = None) -> Any:
    """Return (or lazily create) the singleton DynamicKellySizer."""
    global _KELLY_SIZER  # noqa: PLW0603
    if _KELLY_SIZER is not None:
        return _KELLY_SIZER

    from risk.dynamic_kelly import DynamicKellySizer

    _KELLY_SIZER = DynamicKellySizer(
        window=int(getattr(config, "kelly_window", 50) or 50),
        max_fraction=float(getattr(config, "kelly_max_fraction", 0.25) or 0.25),
        min_fraction=float(getattr(config, "kelly_min_fraction", 0.01) or 0.01),
        full_kelly_cap=float(getattr(config, "kelly_full_cap", 1.0) or 1.0),
    )
    logger.info("DynamicKellySizer initialised (window=%d max_fraction=%.2f)",
                _KELLY_SIZER._window, _KELLY_SIZER._max_f)
    return _KELLY_SIZER


def record_trade_pnl(pnl_pct: float, config: Optional[Any] = None) -> None:
    """Record a realised trade return (fraction of capital) into the Kelly sizer."""
    get_kelly_sizer(config).record_trade(pnl_pct)


def kelly_qty(
    capital: float,
    price: float,
    regime: str = "ranging",
    config: Optional[Any] = None,
) -> float:
    """Return position quantity using Dynamic Kelly sizing."""
    sizer = get_kelly_sizer(config)
    return sizer.size_from_capital(capital, price, regime=regime)
