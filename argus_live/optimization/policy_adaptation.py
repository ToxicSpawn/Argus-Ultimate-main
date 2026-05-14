from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class PolicyAdaptationSuggestion:
    parameter: str
    current_value: float
    proposed_value: float
    reason: str
    advisory_only: bool = True


@dataclass(frozen=True)
class PolicyAdaptationPack:
    suggestions: List[PolicyAdaptationSuggestion]
    reason: str


def build_policy_adaptation(
    net_edge_bps: float,
    slippage_bps: float,
    slippage_target_bps: float,
    turnover: float,
    turnover_target: float,
    current_rebalance_threshold: float = 0.02,
    current_slippage_limit: float = 5.0,
    current_turnover_limit: float = 0.10,
) -> PolicyAdaptationPack:
    suggestions: List[PolicyAdaptationSuggestion] = []

    if net_edge_bps < 0:
        suggestions.append(
            PolicyAdaptationSuggestion(
                parameter="rebalance_threshold",
                current_value=current_rebalance_threshold,
                proposed_value=current_rebalance_threshold * 1.5,
                reason=f"net edge negative ({net_edge_bps:.1f} bps); widen rebalance threshold to reduce churn",
            )
        )

    if slippage_bps > slippage_target_bps:
        suggestions.append(
            PolicyAdaptationSuggestion(
                parameter="slippage_limit",
                current_value=current_slippage_limit,
                proposed_value=current_slippage_limit * 0.8,
                reason=f"slippage {slippage_bps:.1f} bps exceeds target {slippage_target_bps:.1f}; tighten limit",
            )
        )

    if turnover > turnover_target:
        suggestions.append(
            PolicyAdaptationSuggestion(
                parameter="turnover_limit",
                current_value=current_turnover_limit,
                proposed_value=current_turnover_limit * 0.8,
                reason=f"turnover {turnover:.4f} exceeds target {turnover_target:.4f}; reduce limit",
            )
        )

    reason = (
        f"{len(suggestions)} adaptation suggestion(s)"
        if suggestions
        else "no adaptations needed"
    )
    return PolicyAdaptationPack(suggestions=suggestions, reason=reason)
