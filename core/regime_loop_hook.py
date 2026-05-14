"""Regime loop hook — call `tick(prices_dict)` each trading cycle to keep
the RegimeManager fresh.  Designed to be dropped into the main loop with
minimal coupling:

    from core.regime_loop_hook import tick as regime_tick

    # inside your per-cycle coroutine:
    await regime_tick(prices_dict, primary="BTC/USDT")

prices_dict format:  {"BTC/USDT": pd.Series, "ETH/USDT": pd.Series, ...}
Each Series should be a close-price series (index = datetime, values = float).
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


async def tick(
    prices: Dict[str, pd.Series],
    primary: str = "BTC/USDT",
    force: bool = False,
) -> Optional[str]:
    """Update the shared RegimeManager if stale (or forced).

    Returns the current regime label, or None if the manager is not yet
    bootstrapped.
    """
    try:
        from core.regime_bootstrap import get_regime_manager
        mgr = get_regime_manager()
        if mgr is None:
            return None
        if force or mgr.is_stale():
            regime = mgr.update_regime(prices, primary=primary)
            logger.debug("Regime updated via loop hook: %s", regime)
            return regime
        return mgr.get()
    except Exception as e:
        logger.debug("regime_loop_hook.tick: %s", e)
        return None


def sync_tick(
    prices: Dict[str, pd.Series],
    primary: str = "BTC/USDT",
    force: bool = False,
) -> Optional[str]:
    """Synchronous variant for non-async callers."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Schedule as a task; caller must await separately if needed
            loop.create_task(tick(prices, primary=primary, force=force))
            from core.regime_bootstrap import get_regime_manager
            mgr = get_regime_manager()
            return mgr.get() if mgr else None
        return loop.run_until_complete(tick(prices, primary=primary, force=force))
    except Exception as e:
        logger.debug("regime_loop_hook.sync_tick: %s", e)
        return None
