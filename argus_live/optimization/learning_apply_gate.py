from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LearningApplyDecision:
    """Whether a learning update should be applied."""

    apply: bool
    reason: str


def should_apply_learning_update(
    replay_passed: bool,
    promotion_passed: bool,
    operator_approved: bool,
) -> LearningApplyDecision:
    """Gate that requires all three conditions to apply a learning update.

    Returns LearningApplyDecision(apply=True) only when replay validation,
    promotion gate, and operator approval all pass.
    """
    failures: list[str] = []
    if not replay_passed:
        failures.append("replay validation failed")
    if not promotion_passed:
        failures.append("promotion gate rejected")
    if not operator_approved:
        failures.append("operator has not approved")

    if failures:
        return LearningApplyDecision(
            apply=False,
            reason="blocked: " + "; ".join(failures),
        )

    return LearningApplyDecision(
        apply=True,
        reason="all gates passed (replay, promotion, operator)",
    )
