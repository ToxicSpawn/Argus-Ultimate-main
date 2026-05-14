"""
core/signal_routing.py
======================
Pre-execution context builder and signal field extractor.
Exact logic preserved from execute_signals_helpers._pre_execute_context
and _extract_signal_fields — now importable as standalone functions.
"""
from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


def build_execution_context(system: Any) -> dict:
    """
    Build the per-cycle execution context dict.
    Reads macro calendar, regime, session multiplier, and portfolio-level
    risk gates from *system* (UnifiedTradingSystem instance).

    Returns dict with keys:
        macro_event_imminent, macro_event_name, macro_event_hours,
        regime, regime_pos_mult, regime_stop_mult, regime_tp_mult,
        session_mult, mode, is_live, aud_to_usd, portfolio_value,
        daily_loss_exceeded, var_breach, _cycle_advisory
    """
    mode = str(getattr(system.config, "run_mode", "paper") or "paper").lower()
    is_live = mode == "live"
    aud_to_usd = float(getattr(system.config, "aud_to_usd", 0.65) or 0.65)
    portfolio_value = float(system.portfolio_value_aud)

    # Macro calendar
    macro_event_imminent = False
    macro_event_name = ""
    macro_event_hours: Optional[float] = None
    try:
        _fred_cal = None
        if system.component_registry is not None:
            _fred_cal = getattr(system.component_registry, "fred_calendar", None)
        if _fred_cal is None:
            from data.macro.fred_calendar import FREDCalendar
            _fred_cal = FREDCalendar()
        snap = _fred_cal.get_upcoming(days=1)
        if snap.hours_to_next_high is not None and snap.hours_to_next_high <= 2.0:
            macro_event_imminent = True
            macro_event_name = snap.next_high_impact.name if snap.next_high_impact else "unknown"
            macro_event_hours = snap.hours_to_next_high
            logger.warning(
                "build_execution_context: MACRO EVENT — '%s' in %.1fh, blocking new BUY entries",
                macro_event_name, macro_event_hours,
            )
    except Exception as exc:
        logger.debug("build_execution_context: macro calendar check failed: %s", exc)

    # Regime
    regime = str(getattr(system, "_latest_regime_label", "") or "").upper().strip()
    if not regime:
        fn = getattr(system, "_compute_fallback_regime", None)
        regime = fn() if callable(fn) else "NORMAL"
        try:
            system._latest_regime_label = regime
        except AttributeError:
            pass
    regime_pos_mult = system.REGIME_POSITION_SCALE.get(regime, 1.0)
    regime_stop_mult = system.REGIME_STOP_SCALE.get(regime, 1.0)
    regime_tp_mult = system.REGIME_TP_SCALE.get(regime, 1.0)

    # Session multiplier
    hour_utc = datetime.now(tz=timezone.utc).hour
    if 13 <= hour_utc <= 17:
        session_mult = 1.1
    elif 8 <= hour_utc <= 10:
        session_mult = 1.05
    elif 1 <= hour_utc <= 5:
        session_mult = 0.8
    else:
        session_mult = 1.0

    logger.info(
        "build_execution_context: regime=%s (pos*%.2f stop*%.2f tp*%.2f) session=%.2f macro=%s",
        regime, regime_pos_mult, regime_stop_mult, regime_tp_mult,
        session_mult, macro_event_imminent,
    )

    # Daily loss gate
    daily_loss_exceeded = (
        system.unified_risk_manager is not None
        and system.unified_risk_manager.is_daily_loss_limit_exceeded()
    )
    if daily_loss_exceeded:
        logger.warning("build_execution_context: daily loss limit exceeded")

    # VaR / CVaR gate
    var_limit_pct = float(getattr(system.config, "portfolio_var_limit_pct", 0.0) or 0.0)
    cvar_limit_pct = float(getattr(system.config, "portfolio_cvar_limit_pct", 0.0) or 0.0)
    var_breach = False
    if system.unified_risk_manager is not None and (var_limit_pct > 0 or cvar_limit_pct > 0):
        try:
            metrics = system.unified_risk_manager.get_risk_metrics()
            capital = max(metrics.current_capital, 1e-9)
            if var_limit_pct > 0 and abs(metrics.var_95) / capital >= var_limit_pct:
                var_breach = True
                logger.warning("build_execution_context: VaR breach")
            if cvar_limit_pct > 0 and abs(metrics.var_99) / capital >= cvar_limit_pct:
                var_breach = True
                logger.warning("build_execution_context: CVaR breach")
        except Exception as exc:
            logger.debug("build_execution_context: VaR check failed: %s", exc)

    _cycle_advisory = getattr(system, "_last_cycle_advisory", None) or {}

    return {
        "macro_event_imminent": macro_event_imminent,
        "macro_event_name": macro_event_name,
        "macro_event_hours": macro_event_hours,
        "regime": regime,
        "regime_pos_mult": regime_pos_mult,
        "regime_stop_mult": regime_stop_mult,
        "regime_tp_mult": regime_tp_mult,
        "session_mult": session_mult,
        "mode": mode,
        "is_live": is_live,
        "aud_to_usd": aud_to_usd,
        "portfolio_value": portfolio_value,
        "daily_loss_exceeded": daily_loss_exceeded,
        "var_breach": var_breach,
        "_cycle_advisory": _cycle_advisory,
    }


def extract_signal_fields(sig: Any) -> dict | None:
    """
    Extract and validate fields from a single TradingSignal.

    Returns:
        dict of extracted fields on success.
        {"_blocked": True, "result": {...}} if signal is too stale.
        None if action is not BUY/SELL.
    """
    symbol = str(getattr(sig, "symbol", "") or "")
    action = str(getattr(sig, "action", "") or "").upper()
    confidence = float(getattr(sig, "confidence", 0.0) or 0.0)
    strength = float(getattr(sig, "strength", 0.0) or 0.0)
    entry_price = float(getattr(sig, "entry_price", 0.0) or 0.0)
    stop_loss = getattr(sig, "stop_loss", None)
    take_profit = getattr(sig, "take_profit", None)
    reasoning = str(getattr(sig, "reasoning", "") or "")
    source_strategy = str(
        getattr(sig, "strategy", "")
        or getattr(sig, "source_strategy", "")
        or (sig.get("strategy") if isinstance(sig, dict) else "")
        or (sig.get("source_strategy") if isinstance(sig, dict) else "")
        or "unknown"
    )

    # Staleness check
    _sig_ts = getattr(sig, "timestamp", None)
    if _sig_ts is None:
        _sig_age = 0.0
    elif isinstance(_sig_ts, (int, float)):
        _sig_age = time.time() - float(_sig_ts)
    elif hasattr(_sig_ts, "timestamp"):
        _sig_age = time.time() - _sig_ts.timestamp()
    else:
        _sig_age = 0.0

    if _sig_age > 120.0:
        logger.warning(
            "extract_signal_fields: signal too stale age=%.1fs > 120s, rejecting %s %s",
            _sig_age, action, symbol,
        )
        return {
            "_blocked": True,
            "result": {
                "symbol": symbol,
                "side": action,
                "status": "blocked",
                "reason": f"signal_too_stale:age={_sig_age:.1f}s",
            },
        }

    if _sig_age > 0.1:
        confidence *= math.exp(-_sig_age / 30.0)

    if _sig_age < 5.0:
        _age_urgency = 0.2
    elif _sig_age < 30.0:
        _age_urgency = 0.5
    elif _sig_age < 60.0:
        _age_urgency = 0.8
    else:
        _age_urgency = 1.0

    if action not in ("BUY", "SELL"):
        logger.debug("extract_signal_fields: skipping action=%s for %s", action, symbol)
        return None

    return {
        "symbol": symbol,
        "action": action,
        "confidence": confidence,
        "strength": strength,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "reasoning": reasoning,
        "source_strategy": source_strategy,
        "_sig_age": _sig_age,
        "_age_urgency": _age_urgency,
        "_num_confirmations": int(getattr(sig, "num_confirmations", 0) or 0),
        "_sig_obj": sig,
    }
