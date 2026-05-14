"""
Risk, compliance, and audit trail integration for the execution path.

Provides: audit trail (order/fill/risk), limit hierarchy check/update, pre-trade compliance,
latency attribution. Used by unified_execution_engine.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, cast

logger = logging.getLogger(__name__)

# Lazy singletons
_audit_trail: Any = None
_limit_hierarchy: Any = None
_compliance_engine: Any = None
_latency_tracker: Any = None


def get_audit_trail():
    global _audit_trail
    if _audit_trail is None:
        try:
            from monitoring.audit_trail import get_audit_trail as _get
            _audit_trail = _get()
        except Exception as e:
            logger.debug("audit_trail not available: %s", e)
    return _audit_trail


def get_limit_hierarchy():
    global _limit_hierarchy
    if _limit_hierarchy is None:
        try:
            from risk.limit_hierarchy import LimitHierarchy
            _limit_hierarchy = LimitHierarchy()
        except Exception as e:
            logger.debug("limit_hierarchy not available: %s", e)
    return _limit_hierarchy


def get_compliance_engine():
    global _compliance_engine
    if _compliance_engine is None:
        try:
            from compliance.rules_engine import get_compliance_engine as _get
            _compliance_engine = _get()
        except Exception as e:
            logger.debug("compliance_engine not available: %s", e)
    return _compliance_engine


def get_latency_tracker():
    global _latency_tracker
    if _latency_tracker is None:
        try:
            from execution.latency_attribution import get_latency_tracker as _get
            _latency_tracker = _get()
        except Exception as e:
            logger.debug("latency_tracker not available: %s", e)
    return _latency_tracker


def append_order(
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    order_id: str,
    exchange: str,
    correlation_id: Optional[str] = None,
) -> None:
    trail = get_audit_trail()
    if trail is not None:
        try:
            payload = {"symbol": symbol, "side": side, "quantity": quantity, "price": price, "order_id": order_id, "exchange": exchange}
            if correlation_id is not None:
                payload["correlation_id"] = correlation_id
            trail.append("order", payload)
        except Exception as e:
            logger.debug("audit append order: %s", e)


def append_fill(
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    order_id: str,
    exchange: str,
    status: str,
    correlation_id: Optional[str] = None,
) -> None:
    trail = get_audit_trail()
    if trail is not None:
        try:
            payload = {"symbol": symbol, "side": side, "quantity": quantity, "price": price, "order_id": order_id, "exchange": exchange, "status": status}
            if correlation_id is not None:
                payload["correlation_id"] = correlation_id
            trail.append("fill", payload)
        except Exception as e:
            logger.debug("audit append fill: %s", e)


def append_implementation_shortfall(
    symbol: str,
    side: str,
    quantity: float,
    decision_price: float,
    execution_avg_price: float,
    implementation_shortfall: float,
    implementation_shortfall_bps: float,
    strategy: Optional[str] = None,
) -> None:
    """Append implementation shortfall (TCA) to audit trail; record in IS tracker for by-strategy/symbol gate."""
    trail = get_audit_trail()
    if trail is not None:
        try:
            payload = {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "decision_price": decision_price,
                "execution_avg_price": execution_avg_price,
                "implementation_shortfall": implementation_shortfall,
                "implementation_shortfall_bps": implementation_shortfall_bps,
            }
            if strategy is not None:
                payload["strategy"] = strategy
            trail.append("implementation_shortfall", payload)
        except Exception as e:
            logger.debug("audit append implementation_shortfall: %s", e)
    try:
        from execution.is_tracker import get_is_tracker
        get_is_tracker().record(
            implementation_shortfall_bps,
            strategy=strategy if strategy else None,
            symbol=symbol,
        )
    except Exception as _e:
        logger.debug("risk_compliance_audit error: %s", _e)


def append_risk_breach(limit_name: str, current: float, hard_limit: float) -> None:
    trail = get_audit_trail()
    if trail is not None:
        try:
            trail.append("risk_breach", {"limit": limit_name, "current": current, "hard_limit": hard_limit})
        except Exception as e:
            logger.debug("audit append risk_breach: %s", e)


def check_limits(positions: Dict[str, float], prices: Dict[str, float], daily_pnl_pct: float = 0.0, max_dd_pct: float = 0.0) -> List[str]:
    """Update limit hierarchy from current state and return list of breached limit names."""
    hierarchy = get_limit_hierarchy()
    if hierarchy is None:
        return []
    try:
        total_notional = sum(abs(positions.get(s, 0) * prices.get(s, 0)) for s in positions)
        hierarchy.update("book_notional", total_notional)
        hierarchy.update("daily_loss_pct", abs(min(0, daily_pnl_pct)))
        hierarchy.update("max_drawdown_pct", max_dd_pct)
        breaches = hierarchy.check()
        for b in breaches:
            lvl = hierarchy._levels.get(b)
            if lvl is not None:
                append_risk_breach(b, lvl.current, lvl.hard_limit)
        return cast(List[str], list(breaches))
    except Exception as e:
        logger.debug("check_limits: %s", e)
        return []


def pre_trade_exposure_position_gate(
    current_exposure_aud: float,
    new_order_notional_aud: float,
    equity_aud: float,
    max_total_exposure_pct: float,
    max_position_aud: float,
    max_position_pct: float,
) -> Tuple[bool, str]:
    """
    FPGA-style pre-trade gate: approve only if (exposure + new order) <= max exposure
    and new order <= max position. Returns (passed, reason).
    """
    if equity_aud <= 0:
        return False, "equity_zero"
    max_exposure_aud = equity_aud * max_total_exposure_pct
    if current_exposure_aud + new_order_notional_aud > max_exposure_aud:
        return False, "max_exposure_exceeded"
    max_single_aud = min(max_position_aud, equity_aud * max_position_pct)
    if new_order_notional_aud > max_single_aud:
        return False, "max_position_exceeded"
    return True, "ok"


def pre_trade_compliance(
    symbol: str,
    side: str,
    size: float,
    price: float,
    positions: Dict[str, float],
    prices: Dict[str, float],
    daily_pnl_pct: float = 0.0,
) -> bool:
    """True if order is allowed; False if compliance says block."""
    engine = get_compliance_engine()
    if engine is None:
        return True
    try:
        results = engine.pre_trade(symbol, side, size, price, positions, prices, daily_pnl_pct)
        if engine.should_block_order(results):
            return False
        return True
    except Exception as e:
        logger.debug("pre_trade_compliance: %s", e)
        return True


def pre_trade_risk_block(
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    positions: Dict[str, float],
    prices: Dict[str, float],
    *,
    daily_pnl_pct: float = 0.0,
    max_drawdown_pct: float = 0.12,
    equity_aud: float = 1000.0,
    max_total_exposure_pct: float = 0.5,
    max_position_aud: float = 100.0,
    max_position_pct: float = 0.1,
    aud_to_usd: float = 0.65,
) -> Tuple[bool, str]:
    """
    Single pre-trade contract (FPGA-style): run limits, exposure/position gate, and compliance.
    Returns (approved, reason). Use one call to get trade_valid -> trade_approved.
    """
    # 1) Limit hierarchy (drawdown, daily loss)
    breaches = check_limits(positions, prices, daily_pnl_pct, max_drawdown_pct)
    if breaches:
        return False, "limit_breach:" + ",".join(breaches)
    # 2) Exposure and position size gate
    quote = symbol.split("/")[-1].upper() if "/" in symbol else "USD"
    notional_quote = quantity * price
    notional_aud = notional_quote / aud_to_usd if quote in ("USD", "USDT") else notional_quote
    current_exposure_aud = sum(
        abs(positions.get(s, 0) * prices.get(s, 0)) / aud_to_usd
        for s in (positions or {})
    )
    passed, reason = pre_trade_exposure_position_gate(
        current_exposure_aud, notional_aud, equity_aud,
        max_total_exposure_pct, max_position_aud, max_position_pct,
    )
    if not passed:
        return False, reason
    # 3) Compliance engine
    if not pre_trade_compliance(symbol, side, quantity, price, positions, prices, daily_pnl_pct):
        return False, "compliance_block"
    return True, "ok"


def latency_start(component: str) -> None:
    t = get_latency_tracker()
    if t is not None:
        t.start(component)


def latency_end(component: str, extra: Optional[Dict[str, Any]] = None) -> None:
    t = get_latency_tracker()
    if t is not None:
        t.end(component, extra)
