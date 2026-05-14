"""Live patch: injects Dynamic Kelly + RegimeManager into KrakenDCAExecutionEngine.

Import this module ONCE at startup (before the engine runs any cycle).
It monkey-patches three things without touching the 200KB source file:

  1. KrakenDCAExecutionEngine._calculate_quantity
     → uses DynamicKellySizer (rolling empirical Kelly) scaled by regime

  2. KrakenDCAExecutionEngine.initialize  (post-hook)
     → calls bootstrap_regime_manager(config) after the existing init

  3. KrakenDCAExecutionEngine._record_fill_pnl  (new helper)
     → called from execute_signals after each fill to feed the Kelly sizer

Usage (in main.py or unified_trading_system.py, near the top):

    import core.execution_engine_live_patch  # noqa: F401
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _patched_calculate_quantity(self: Any, signal: Any) -> float:
    """Dynamic Kelly quantity: rolling empirical Kelly scaled by regime + confidence."""
    # ---- equity / FX ----
    equity_aud = float(getattr(self.config, "current_equity_aud", None) or 0.0)
    if equity_aud <= 0:
        equity_aud = float(getattr(self.config, "starting_capital_aud", 1000.0) or 1000.0)
    aud_to_usd = float(getattr(self.config, "aud_to_usd", 0.65) or 0.65)
    capital_usd = equity_aud * aud_to_usd

    entry_price = float(getattr(signal, "entry_price", 0.0) or 0.0)
    if entry_price <= 0:
        return 0.0

    use_dynamic_kelly = bool(getattr(self.config, "dynamic_kelly_sizing", True))

    if use_dynamic_kelly:
        try:
            from risk.kelly_integration import kelly_qty, get_kelly_sizer
            from core.regime_bootstrap import get_regime_manager

            regime_mgr = get_regime_manager()
            regime = regime_mgr.get() if regime_mgr else "ranging"

            quantity = kelly_qty(
                capital=capital_usd,
                price=entry_price,
                regime=regime,
                config=self.config,
            )
            logger.debug(
                "DynamicKelly qty=%.8f regime=%s capital_usd=%.2f price=%.2f",
                quantity, regime, capital_usd, entry_price,
            )
        except Exception as e:
            logger.debug("DynamicKelly fallback to static: %s", e)
            quantity = _static_kelly_quantity(self, signal, equity_aud, aud_to_usd)
    else:
        quantity = _static_kelly_quantity(self, signal, equity_aud, aud_to_usd)

    # Portfolio weight scaling (BL / HRP / MPT) — preserved from original
    weight_method = str(getattr(self.config, "portfolio_weight_method", "") or "").strip().lower()
    if weight_method in ("mpt", "bl", "hrp") and quantity > 0:
        try:
            try:
                from portfolio.weight_provider import get_weights, get_weights_correlation_aware
            except ImportError:
                get_weights = get_weights_correlation_aware = None
            if get_weights is not None:
                symbols_cfg = (
                    getattr(self.config, "symbols", None)
                    or getattr(self.config, "trading_pairs", None)
                    or [getattr(signal, "symbol", "BTC/USD")]
                )
                symbols_list = symbols_cfg if isinstance(symbols_cfg, (list, tuple)) else [getattr(signal, "symbol", "BTC/USD")]
                base_weights = get_weights(symbols_list, method=weight_method)
                if (
                    getattr(self.config, "use_correlation_aware_sizing", False)
                    and get_weights_correlation_aware is not None
                    and len(symbols_list) > 1
                ):
                    corr_matrix = getattr(self.config, "correlation_matrix", None) or {}
                    if corr_matrix:
                        base_weights = get_weights_correlation_aware(
                            symbols_list,
                            base_weights,
                            correlation_matrix=corr_matrix,
                            max_correlated_exposure=float(
                                getattr(self.config, "max_correlated_exposure", 0.6) or 0.6
                            ),
                        )
                sym = getattr(signal, "symbol", None) or (symbols_list[0] if symbols_list else "BTC/USD")
                quantity *= float(base_weights.get(sym, 1.0))
        except Exception:
            pass

    return max(0.0, quantity)


def _static_kelly_quantity(self: Any, signal: Any, equity_aud: float, aud_to_usd: float) -> float:
    """Original confidence-proxy Kelly (fallback when DynamicKellySizer unavailable)."""
    position_value_aud = min(
        float(getattr(self.config, "max_position_size_aud", equity_aud * 0.25) or equity_aud * 0.25),
        equity_aud * float(getattr(self.config, "max_position_pct", 0.25) or 0.25),
    )
    position_value_usd = position_value_aud * aud_to_usd
    entry_price = float(getattr(signal, "entry_price", 0.0) or 0.0)
    quantity = position_value_usd / entry_price if entry_price > 0 else 0.0

    use_kelly = bool(getattr(self.config, "dynamic_kelly_sizing", True))
    if use_kelly and quantity > 0:
        conf = float(getattr(signal, "confidence", 0.5) or 0.5)
        strength = float(getattr(signal, "strength", 0.5) or 0.5)
        edge_proxy = (conf + strength) / 2.0
        vol_proxy = float(getattr(self.config, "kelly_vol_proxy", 0.02) or 0.02)
        max_kelly = float(getattr(self.config, "kelly_max_fraction", 0.5) or 0.5)
        kelly_frac = min(max_kelly, max(0.2, edge_proxy / max(vol_proxy, 1e-6)))
        quantity *= kelly_frac
    return max(0.0, quantity)


def _record_fill_pnl(self: Any, rd: dict) -> None:
    """Feed realised PnL back into the DynamicKellySizer after each fill."""
    try:
        pnl = float(rd.get("pnl", 0.0) or 0.0)
        equity_aud = float(
            getattr(self.config, "current_equity_aud", None)
            or getattr(self.config, "starting_capital_aud", 1000.0)
            or 1000.0
        )
        aud_to_usd = float(getattr(self.config, "aud_to_usd", 0.65) or 0.65)
        capital_usd = max(equity_aud * aud_to_usd, 1.0)
        pnl_pct = pnl / capital_usd

        from risk.kelly_integration import record_trade_pnl
        record_trade_pnl(pnl_pct, config=self.config)
        logger.debug("Kelly PnL recorded: pnl_pct=%.6f", pnl_pct)
    except Exception as e:
        logger.debug("record_fill_pnl: %s", e)


async def _patched_initialize(self: Any) -> None:
    """Wraps original initialize() and bootstraps RegimeManager afterward."""
    await _original_initialize(self)
    try:
        from core.regime_bootstrap import bootstrap_regime_manager
        bootstrap_regime_manager(self.config)
        logger.info("RegimeManager bootstrapped via execution engine init patch.")
    except Exception as e:
        logger.warning("RegimeManager bootstrap failed (non-fatal): %s", e)


def _apply_patch() -> None:
    """Apply all monkey-patches to KrakenDCAExecutionEngine."""
    try:
        from unified_execution_engine import KrakenDCAExecutionEngine
    except ImportError as e:
        logger.warning("execution_engine_live_patch: could not import engine — %s", e)
        return

    global _original_initialize  # noqa: PLW0603

    # Guard: don't double-patch
    if getattr(KrakenDCAExecutionEngine, "_dynamic_kelly_patched", False):
        logger.debug("execution_engine_live_patch: already applied, skipping.")
        return

    # Stash original initialize for wrapping
    _original_initialize = KrakenDCAExecutionEngine.initialize

    # Apply patches
    KrakenDCAExecutionEngine._calculate_quantity = _patched_calculate_quantity
    KrakenDCAExecutionEngine.initialize = _patched_initialize
    KrakenDCAExecutionEngine._record_fill_pnl = _record_fill_pnl
    KrakenDCAExecutionEngine._dynamic_kelly_patched = True

    logger.info(
        "execution_engine_live_patch applied: "
        "_calculate_quantity → DynamicKelly, "
        "initialize → RegimeManager bootstrap, "
        "_record_fill_pnl registered."
    )


# Sentinel for original initialize (set in _apply_patch)
_original_initialize: Any = None

# Auto-apply on import
_apply_patch()
