"""
core/risk_gates.py
==================
All blocking risk gates that were previously inline `continue` statements
in execute_signals_helpers._apply_risk_gates.

Each gate is a pure function: (system, sig_fields, ctx) -> (approved, reason).
They are composed in apply_all_risk_gates() which preserves the original
ordering exactly.
"""
from __future__ import annotations

import logging
from typing import Any, Tuple

logger = logging.getLogger(__name__)

Approval = Tuple[bool, str]


# ---------------------------------------------------------------------------
# Individual gate functions
# ---------------------------------------------------------------------------

def gate_strategy_cooldown(system: Any, sig_fields: dict, ctx: dict) -> Approval:
    source_strategy = sig_fields["source_strategy"]
    action = sig_fields["action"]
    _sss = getattr(system, "_strategy_state_store", None)
    if _sss is not None and _sss.check_cooldown(source_strategy):
        remaining = _sss.cooldown_remaining_seconds(source_strategy)
        logger.info(
            "gate_strategy_cooldown: DROPPED %s %s — cooldown %.0fs remaining",
            action, sig_fields["symbol"], remaining,
        )
        return False, f"strategy_cooldown:{source_strategy}"
    return True, ""


def gate_invalid_signal(system: Any, sig_fields: dict, ctx: dict) -> Approval:
    if not sig_fields["symbol"] or sig_fields["entry_price"] <= 0:
        logger.warning(
            "gate_invalid_signal: symbol=%s entry_price=%s",
            sig_fields["symbol"], sig_fields["entry_price"],
        )
        return False, "invalid_signal"
    return True, ""


def gate_daily_loss_limit(system: Any, sig_fields: dict, ctx: dict) -> Approval:
    if ctx["daily_loss_exceeded"] and sig_fields["action"] == "BUY":
        logger.warning(
            "gate_daily_loss_limit: REJECTED %s %s — daily loss limit exceeded",
            sig_fields["action"], sig_fields["symbol"],
        )
        return False, "daily_loss_limit_exceeded"
    return True, ""


def gate_macro_event(system: Any, sig_fields: dict, ctx: dict) -> Approval:
    if ctx["macro_event_imminent"] and sig_fields["action"] == "BUY":
        logger.warning(
            "gate_macro_event: REJECTED %s %s — macro event '%s' in %.1fh",
            sig_fields["action"], sig_fields["symbol"],
            ctx["macro_event_name"], ctx["macro_event_hours"] or 0.0,
        )
        return False, f"macro_event_imminent:{ctx['macro_event_name']}"
    return True, ""


def gate_var_breach(system: Any, sig_fields: dict, ctx: dict) -> Approval:
    if ctx["var_breach"] and sig_fields["action"] == "BUY":
        logger.warning(
            "gate_var_breach: REJECTED %s %s — VaR/CVaR limit breached",
            sig_fields["action"], sig_fields["symbol"],
        )
        return False, "var_limit_breached"
    return True, ""


def gate_unified_risk_manager(system: Any, sig_fields: dict, ctx: dict) -> Approval:
    """Pre-trade risk check with smart position size estimate."""
    action = sig_fields["action"]
    symbol = sig_fields["symbol"]
    if system.unified_risk_manager is None or action != "BUY":
        return True, ""

    confidence = sig_fields["confidence"]
    strength = sig_fields["strength"]
    portfolio_value = ctx["portfolio_value"]
    aud_to_usd = ctx["aud_to_usd"]

    size_usd = confidence * strength * portfolio_value * aud_to_usd * 0.1

    # Volatility adjustment
    try:
        vol = system._get_current_vol(symbol) or 0.005
    except Exception:
        vol = 0.005
    if vol > 0.01:
        size_usd *= 0.70
    elif vol < 0.004:
        size_usd *= 1.20

    # Conviction weighting
    n_conf = int(sig_fields.get("_num_confirmations", 0) or 0)
    size_usd *= {5: 1.0, 3: 0.70, 1: 0.40}.get(
        max(k for k in [5, 3, 1, 0] if k <= n_conf), 0.30
    )

    # Portfolio heat
    open_count = sum(
        1 for p in (system.positions or {}).values()
        if float((p or {}).get("quantity", 0) or 0) > 0
    )
    if open_count >= 2:
        size_usd *= 0.60

    # Drawdown penalty
    if system.peak_equity_aud > 0:
        dd = (system.peak_equity_aud - system.portfolio_value_aud) / system.peak_equity_aud
        if dd > 0.05:
            size_usd *= 0.50

    # Same-symbol penalty
    existing_qty = float(
        (system.positions or {}).get(symbol, {}).get("quantity", 0) or 0
    )
    if existing_qty > 0:
        size_usd *= 0.50

    approved, reason = system.unified_risk_manager.pre_trade_risk_check(
        symbol=symbol, position_size_usd=size_usd
    )
    if not approved:
        logger.warning(
            "gate_unified_risk_manager: REJECTED %s %s — %s", action, symbol, reason
        )
        return False, reason
    return True, ""


def gate_max_concurrent_positions(system: Any, sig_fields: dict, ctx: dict) -> Approval:
    action = sig_fields["action"]
    if action != "BUY":
        return True, ""
    max_pos = int(getattr(system.config, "max_concurrent_positions", 5) or 0)
    if max_pos <= 0:
        return True, ""
    current = sum(
        1 for p in (system.positions or {}).values()
        if float((p or {}).get("quantity", 0) or 0) > 0
    )
    if current >= max_pos:
        logger.warning(
            "gate_max_concurrent_positions: REJECTED %s %s — %d/%d",
            action, sig_fields["symbol"], current, max_pos,
        )
        return False, f"max_concurrent_positions ({current}/{max_pos})"
    return True, ""


def gate_component_registry_pre_order(
    system: Any, sig_fields: dict, ctx: dict
) -> Approval:
    if system.component_registry is None:
        return True, ""
    size_usd = ctx["portfolio_value"] * ctx["aud_to_usd"] * 0.05
    exchange = str(getattr(system.config, "primary_exchange", "kraken") or "kraken")
    check = system.component_registry.pre_order_check(
        sig_fields["symbol"], sig_fields["action"], size_usd, exchange
    )
    if not check.get("allow", True):
        logger.info(
            "gate_component_registry_pre_order: blocked %s %s: %s",
            sig_fields["action"], sig_fields["symbol"], check.get("reasons", []),
        )
        return False, str(check.get("reasons", ["risk_gate"]))
    return True, ""


# ---------------------------------------------------------------------------
# Composed entry point
# ---------------------------------------------------------------------------

_GATES = [
    gate_strategy_cooldown,
    gate_invalid_signal,
    gate_daily_loss_limit,
    gate_macro_event,
    gate_var_breach,
    gate_unified_risk_manager,
    gate_max_concurrent_positions,
    gate_component_registry_pre_order,
]


def apply_all_risk_gates(
    system: Any, sig_fields: dict, ctx: dict
) -> Approval:
    """
    Run all risk gates in order.  Short-circuits on first rejection.
    Returns (approved: bool, reason: str).
    """
    for gate in _GATES:
        approved, reason = gate(system, sig_fields, ctx)
        if not approved:
            return False, reason
    return True, ""
