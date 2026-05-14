"""
Implementation shortfall: cost of execution vs decision/arrival price.

IS = (execution_avg_price - decision_price) * quantity for buys;
     (decision_price - execution_avg_price) * quantity for sells.
Used for TCA and execution quality; reward signal for RL execution agents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ImplementationShortfallResult:
    """Result of implementation shortfall computation."""
    symbol: str
    side: str
    quantity: float
    decision_price: float
    execution_avg_price: float
    implementation_shortfall: float  # in quote currency (positive = cost)
    implementation_shortfall_bps: float  # basis points vs decision


def compute_implementation_shortfall(
    symbol: str,
    side: str,
    quantity: float,
    decision_price: float,
    execution_avg_price: float,
) -> ImplementationShortfallResult:
    """
    Compute implementation shortfall: cost vs decision price.
    For buys: IS = (exec_avg - decision) * qty (positive = paid more).
    For sells: IS = (decision - exec_avg) * qty (positive = received less).
    """
    q = float(quantity)
    if q <= 0:
        is_quote = 0.0
        is_bps = 0.0
    else:
        if str(side).lower() in ("buy", "b"):
            is_quote = (float(execution_avg_price) - float(decision_price)) * q
        else:
            is_quote = (float(decision_price) - float(execution_avg_price)) * q
        decision_notional = decision_price * q
        is_bps = (is_quote / decision_notional * 1e4) if decision_notional else 0.0
    return ImplementationShortfallResult(
        symbol=symbol,
        side=str(side),
        quantity=q,
        decision_price=float(decision_price),
        execution_avg_price=float(execution_avg_price),
        implementation_shortfall=is_quote,
        implementation_shortfall_bps=is_bps,
    )


def implementation_shortfall_to_dict(r: ImplementationShortfallResult) -> Dict[str, Any]:
    """Serialize for audit/ledger."""
    return {
        "symbol": r.symbol,
        "side": r.side,
        "quantity": r.quantity,
        "decision_price": r.decision_price,
        "execution_avg_price": r.execution_avg_price,
        "implementation_shortfall": r.implementation_shortfall,
        "implementation_shortfall_bps": r.implementation_shortfall_bps,
    }
