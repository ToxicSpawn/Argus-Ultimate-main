from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LearningUpdate:
    """Suggested parameter adjustments from the learning loop."""

    strategy_weight_suggestions: dict[str, float]
    venue_penalty_suggestions: dict[str, float]
    reason: str


def compute_learning_update(
    strategy_edges: dict[str, float],
    venue_slippage: dict[str, float],
) -> LearningUpdate:
    """Suggest strategy weights proportional to edges and venue penalties for high slippage.

    Strategy weights are normalised so they sum to 1.0 (or empty if no strategies).
    Venue penalties are slippage_bps / 100 clamped to [0.0, 1.0].
    """
    # --- strategy weight suggestions (proportional to edge, floored at 0) ---
    positive_edges = {k: max(v, 0.0) for k, v in strategy_edges.items()}
    total = sum(positive_edges.values())
    if total > 0:
        weights = {k: v / total for k, v in positive_edges.items()}
    else:
        weights = {k: 1.0 / len(strategy_edges) for k in strategy_edges} if strategy_edges else {}

    # --- venue penalty suggestions (higher slippage → higher penalty) ---
    penalties: dict[str, float] = {}
    for venue, slip in venue_slippage.items():
        penalties[venue] = max(0.0, min(slip / 100.0, 1.0))

    reasons: list[str] = []
    if weights:
        best = max(weights, key=lambda k: weights[k])
        reasons.append(f"highest-edge strategy: {best}")
    if penalties:
        worst = max(penalties, key=lambda k: penalties[k])
        reasons.append(f"highest-slippage venue: {worst}")

    return LearningUpdate(
        strategy_weight_suggestions=weights,
        venue_penalty_suggestions=penalties,
        reason="; ".join(reasons) if reasons else "no data",
    )
