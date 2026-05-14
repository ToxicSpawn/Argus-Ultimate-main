from __future__ import annotations

from dataclasses import dataclass

from argus_live.promotion.promotion_bundle import PromotionBundle
from argus_live.promotion.promotion_gate import promotion_allowed


@dataclass(frozen=True)
class PromotionDecision:
    strategy_id: str
    promote: bool
    reason: str


def evaluate_for_promotion(bundle: PromotionBundle) -> PromotionDecision:
    """Evaluate whether a strategy bundle qualifies for autonomous promotion.

    Requirements:
    1. ``promotion_allowed(bundle)`` must pass.
    2. ``walk_forward_score >= 1.0``
    3. ``stress_score >= 1.0``
    """
    if not promotion_allowed(bundle):
        return PromotionDecision(
            strategy_id=bundle.strategy_id,
            promote=False,
            reason="Base promotion gate rejected the bundle",
        )

    if bundle.walk_forward_score < 1.0:
        return PromotionDecision(
            strategy_id=bundle.strategy_id,
            promote=False,
            reason=(
                f"walk_forward_score {bundle.walk_forward_score:.4f} < 1.0"
            ),
        )

    if bundle.stress_score < 1.0:
        return PromotionDecision(
            strategy_id=bundle.strategy_id,
            promote=False,
            reason=f"stress_score {bundle.stress_score:.4f} < 1.0",
        )

    return PromotionDecision(
        strategy_id=bundle.strategy_id,
        promote=True,
        reason="All promotion criteria met",
    )
