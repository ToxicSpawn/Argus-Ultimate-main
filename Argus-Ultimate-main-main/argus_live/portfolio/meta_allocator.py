from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class FamilyAllocation:
    family: str
    capital: float
    weight: float
    reason: str


@dataclass(frozen=True)
class StrategyFamilyMap:
    strategy_to_family: Dict[str, str]


def allocate_by_family(
    total_capital: float,
    family_scores: Dict[str, float],
) -> List[FamilyAllocation]:
    """Score-weighted capital allocation across strategy families.

    Falls back to equal weighting when all scores are zero or the map is empty.
    """
    if not family_scores:
        return []

    total_score = sum(family_scores.values())

    allocations: List[FamilyAllocation] = []
    if total_score <= 0.0:
        # Equal fallback
        equal_weight = 1.0 / len(family_scores)
        for family in sorted(family_scores):
            allocations.append(
                FamilyAllocation(
                    family=family,
                    capital=total_capital * equal_weight,
                    weight=equal_weight,
                    reason="equal fallback (all scores zero)",
                )
            )
    else:
        for family in sorted(family_scores):
            weight = family_scores[family] / total_score
            allocations.append(
                FamilyAllocation(
                    family=family,
                    capital=total_capital * weight,
                    weight=weight,
                    reason=f"score-weighted: score={family_scores[family]:.4f}",
                )
            )
    return allocations
