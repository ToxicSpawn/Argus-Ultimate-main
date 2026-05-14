"""
core/patch_scp.py

Import this module ONCE at application startup (e.g. top of main.py / bot.py)
to activate the SmallCapitalPipeline sizing path across the whole system.

    import core.patch_scp  # noqa: F401  — side-effect import, must be first

What this does
--------------
1. Monkey-patches UnifiedTradingSystem._compute_position_size with the SCP
   version from core._scp_position_size.
2. Attaches _scp_after_fill and _scp_after_close lifecycle hooks to UTS.
3. Wraps KrakenDCAExecutionEngine.execute_signals so that:
   - After a confirmed BUY fill  -> self._scp_after_fill() is called.
   - After a confirmed SELL fill -> self._scp_after_close(strategy, symbol, pnl)
     is called (pnl pulled from execution result).

All patches are idempotent (guarded by _SCP_PATCHED flag).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SCP_PATCHED = False


def apply() -> None:
    global _SCP_PATCHED
    if _SCP_PATCHED:
        return

    # ------------------------------------------------------------------ #
    # 1. Patch UnifiedTradingSystem                                        #
    # ------------------------------------------------------------------ #
    try:
        from core._scp_position_size import (
            _compute_position_size,
            _after_fill_hook,
            _after_close_hook,
        )
        from core.unified_trading_system import UnifiedTradingSystem

        UnifiedTradingSystem._compute_position_size = _compute_position_size
        UnifiedTradingSystem._scp_after_fill        = _after_fill_hook
        UnifiedTradingSystem._scp_after_close       = _after_close_hook
        logger.info("[patch_scp] UnifiedTradingSystem patched with SmallCapitalPipeline sizing")
    except Exception as exc:
        logger.warning("[patch_scp] Could not patch UnifiedTradingSystem: %s", exc)

    # ------------------------------------------------------------------ #
    # 2. Wrap KrakenDCAExecutionEngine.execute_signals                     #
    # ------------------------------------------------------------------ #
    try:
        from unified_execution_engine import KrakenDCAExecutionEngine

        _orig_execute_signals = KrakenDCAExecutionEngine.execute_signals

        async def _patched_execute_signals(self, signals, correlation_id=None):  # type: ignore[override]
            results = await _orig_execute_signals(self, signals, correlation_id=correlation_id)

            # Fire lifecycle hooks on the UTS instance if available
            _uts = getattr(self, "_uts", None) or getattr(self, "trading_system", None)
            if _uts is None:
                return results

            _after_fill  = getattr(_uts, "_scp_after_fill",  None)
            _after_close = getattr(_uts, "_scp_after_close", None)

            for rd in (results or []):
                if not isinstance(rd, dict):
                    continue
                status = str(rd.get("status") or "").lower()
                if status not in {"filled", "closed"}:
                    continue
                side = str(rd.get("side") or "").upper()
                if side == "BUY":
                    if callable(_after_fill):
                        try:
                            _after_fill(_uts)
                        except Exception as _e:
                            logger.debug("[patch_scp] _after_fill hook failed: %s", _e)
                elif side == "SELL":
                    if callable(_after_close):
                        try:
                            _strategy = str(rd.get("source_strategy") or "unknown")
                            _symbol   = str(rd.get("symbol") or "")
                            _pnl      = float(rd.get("pnl") or 0.0)
                            _after_close(_uts, _strategy, _symbol, _pnl)
                        except Exception as _e:
                            logger.debug("[patch_scp] _after_close hook failed: %s", _e)

            return results

        KrakenDCAExecutionEngine.execute_signals = _patched_execute_signals
        logger.info("[patch_scp] KrakenDCAExecutionEngine.execute_signals wrapped with SCP lifecycle hooks")
    except Exception as exc:
        logger.warning("[patch_scp] Could not wrap KrakenDCAExecutionEngine: %s", exc)

    _SCP_PATCHED = True


# Auto-apply on import
apply()
