from __future__ import annotations

import logging
from dataclasses import dataclass

from argus_live.execution.execution_alpha_engine import Aggression, ExecutionAlphaDecision

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdaptiveSlicePlan:
    slice_count: int
    slice_quantity: float
    spacing_seconds: float
    reason: str


def build_adaptive_slice_plan(
    total_quantity: float,
    alpha: ExecutionAlphaDecision,
    volatility_bps: float,
) -> AdaptiveSlicePlan:
    """Build an adaptive slice plan from the execution alpha decision.

    If not should_slice -> 1 slice (full quantity).
    Otherwise: base_slices = 4 if vol > 100 else 2.
    Spacing: HIGH aggression -> 5s, MEDIUM -> 10s, LOW -> 20s.
    """
    if not alpha.should_slice:
        return AdaptiveSlicePlan(
            slice_count=1,
            slice_quantity=total_quantity,
            spacing_seconds=0.0,
            reason="no_slicing_needed",
        )

    base_slices = 4 if volatility_bps > 100 else 2
    slice_qty = total_quantity / base_slices

    if alpha.aggression == Aggression.HIGH:
        spacing = 5.0
    elif alpha.aggression == Aggression.MEDIUM:
        spacing = 10.0
    else:
        spacing = 20.0

    return AdaptiveSlicePlan(
        slice_count=base_slices,
        slice_quantity=round(slice_qty, 8),
        spacing_seconds=spacing,
        reason=f"sliced_{base_slices}_at_{spacing}s",
    )
