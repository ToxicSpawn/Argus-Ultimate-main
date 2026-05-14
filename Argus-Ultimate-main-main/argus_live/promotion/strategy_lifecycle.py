from __future__ import annotations

import enum
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class LifecycleState(enum.Enum):
    SHADOW = "SHADOW"
    PAPER = "PAPER"
    LIVE_SMALL = "LIVE_SMALL"
    LIVE_FULL = "LIVE_FULL"
    SUSPENDED = "SUSPENDED"
    RETIRED = "RETIRED"


@dataclass(frozen=True)
class StrategyLifecycleDecision:
    """Immutable lifecycle transition recommendation."""

    strategy_id: str
    current_state: LifecycleState
    new_state: LifecycleState
    reason: str


def evaluate_lifecycle(
    strategy_id: str,
    net_edge_bps: float,
    drawdown_pct: float,
    stability_score: float,
    current_state: LifecycleState = LifecycleState.SHADOW,
) -> StrategyLifecycleDecision:
    """Determine lifecycle transition based on edge, drawdown, and stability.

    Rules (evaluated in order):
        - net_edge > 5 and stability > 1.0  -> LIVE_FULL
        - net_edge > 0                      -> LIVE_SMALL
        - net_edge < -5                     -> SUSPENDED
        - else                              -> SHADOW
    """
    if net_edge_bps > 5.0 and stability_score > 1.0:
        new_state = LifecycleState.LIVE_FULL
        reason = (
            f"net_edge={net_edge_bps:.1f}bps > 5 and "
            f"stability={stability_score:.2f} > 1.0"
        )
    elif net_edge_bps > 0.0:
        new_state = LifecycleState.LIVE_SMALL
        reason = f"net_edge={net_edge_bps:.1f}bps > 0 (not yet LIVE_FULL)"
    elif net_edge_bps < -5.0:
        new_state = LifecycleState.SUSPENDED
        reason = f"net_edge={net_edge_bps:.1f}bps < -5 — suspending"
    else:
        new_state = LifecycleState.SHADOW
        reason = f"net_edge={net_edge_bps:.1f}bps in [-5, 0] — staying in shadow"

    return StrategyLifecycleDecision(
        strategy_id=strategy_id,
        current_state=current_state,
        new_state=new_state,
        reason=reason,
    )
