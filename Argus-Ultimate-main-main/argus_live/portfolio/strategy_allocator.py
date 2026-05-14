from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyAllocation:
    strategy_id: str
    weight: float
    target_capital: float


def allocate_capital(
    total_capital: float,
    strategy_scores: dict[str, float],
) -> list[StrategyAllocation]:
    """Allocate *total_capital* across strategies weighted by score.

    *strategy_scores* maps ``strategy_id`` to its score.
    When all scores are ``<= 0`` the allocation falls back to equal weight.

    Returns a list of :class:`StrategyAllocation`.
    """
    if not strategy_scores:
        return []

    strategies = list(strategy_scores.items())
    total_score = sum(max(s, 0.0) for _, s in strategies)

    if total_score <= 0.0:
        # Equal-weight fallback
        weight = 1.0 / len(strategies)
        return [
            StrategyAllocation(
                strategy_id=sid,
                weight=weight,
                target_capital=total_capital * weight,
            )
            for sid, _ in strategies
        ]

    allocations: list[StrategyAllocation] = []
    for sid, score in strategies:
        w = max(score, 0.0) / total_score
        allocations.append(
            StrategyAllocation(
                strategy_id=sid,
                weight=w,
                target_capital=total_capital * w,
            )
        )
    return allocations
