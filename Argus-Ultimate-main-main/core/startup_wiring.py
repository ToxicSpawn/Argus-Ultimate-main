#!/usr/bin/env python3
"""
Startup Wiring Helpers
======================
Called by main.py _initialize_execution_engine and run_cycle to apply the
three post-init wiring steps that were missing:

  1. attach_state_store  — wire OmegaSQLiteStore into OmegaExecutionEngine
  2. paper_trading_peak_mode advisory log — surfaces flag state at every boot
  3. bl_views pass-through — forward AI brain views into compute_budgets()

All functions are safe to call even when optional dependencies are absent.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. attach_state_store
# ---------------------------------------------------------------------------

def attach_state_store(execution_engine: Any, db_path: str = "data/omega_state.db") -> Optional[Any]:
    """
    Wire an OmegaSQLiteStore into *execution_engine* via attach_state_store().

    Returns the store instance on success, None if unavailable.
    Call immediately after _initialize_execution_engine completes.

    Example usage in main.py::

        from core.startup_wiring import attach_state_store
        self._state_store = attach_state_store(self.omega_execution)
    """
    if execution_engine is None:
        logger.debug("attach_state_store: execution_engine is None — skipping")
        return None

    try:
        from core.omega_sqlite_store import OmegaSQLiteStore
        store = OmegaSQLiteStore(db_path=db_path)
        if hasattr(execution_engine, "attach_state_store"):
            execution_engine.attach_state_store(store)
            logger.info(
                "OmegaSQLiteStore: wired into execution engine (db=%s)", db_path
            )
        else:
            logger.warning(
                "OmegaSQLiteStore: execution engine has no attach_state_store() — "
                "store created but not wired. Will be available via self._state_store."
            )
        return store
    except ImportError:
        logger.warning(
            "OmegaSQLiteStore not available — state persistence disabled. "
            "Install core/omega_sqlite_store.py or check its dependencies."
        )
        return None
    except Exception as exc:
        logger.error("attach_state_store failed: %s", exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# 2. paper_trading_peak_mode advisory log
# ---------------------------------------------------------------------------

def log_peak_mode_state(config: Any = None) -> None:
    """
    Emit an INFO advisory showing the current paper_trading_peak_mode state.

    Call once during Argus.__init__ after the mode/capital header block so the
    flag is always visible in argus.log at startup regardless of verbosity.

    Example usage in main.py::

        from core.startup_wiring import log_peak_mode_state
        log_peak_mode_state(self._config)   # or pass None to use defaults
    """
    peak_active: bool = True  # default: peak mode is ON

    if config is not None:
        # Support both attribute-style (dataclass/object) and dict-style configs
        if hasattr(config, "paper_trading_peak_mode"):
            peak_active = bool(config.paper_trading_peak_mode)
        elif hasattr(config, "get"):
            peak_active = bool(config.get("paper_trading_peak_mode", True))

    logger.info(
        "paper_trading_peak_mode: %s  "
        "(peak-P&L position sizing + aggressive exits %s)",
        "ENABLED" if peak_active else "DISABLED",
        "active" if peak_active else "inactive",
    )


# ---------------------------------------------------------------------------
# 3. bl_views pass-through
# ---------------------------------------------------------------------------

def get_bl_views(omega_ml: Any = None, advanced_intelligence: Any = None) -> Dict[str, Any]:
    """
    Collect Black-Litterman views from the AI brain layer.

    Tries omega_ml.get_bl_views() first, then falls back to
    advanced_intelligence.get_views(). Always returns a dict (never None)
    so it is safe to pass directly to capital_optimizer.compute_budgets().

    Example usage in run_cycle::

        from core.startup_wiring import get_bl_views
        bl_views = get_bl_views(self.omega_ml, self.advanced_intelligence)
        budgets = self.capital_optimizer.compute_budgets(
            portfolio_value=portfolio_value,
            regime=state.regime.value,
            bl_views=bl_views,
        )
    """
    # Primary: omega_ml
    if omega_ml is not None and hasattr(omega_ml, "get_bl_views"):
        try:
            views = omega_ml.get_bl_views()
            if isinstance(views, dict):
                return views
        except Exception as exc:
            logger.debug("get_bl_views: omega_ml.get_bl_views() failed: %s", exc)

    # Fallback: advanced_intelligence
    if advanced_intelligence is not None and hasattr(advanced_intelligence, "get_views"):
        try:
            views = advanced_intelligence.get_views()
            if isinstance(views, dict):
                return views
        except Exception as exc:
            logger.debug("get_bl_views: advanced_intelligence.get_views() failed: %s", exc)

    return {}
