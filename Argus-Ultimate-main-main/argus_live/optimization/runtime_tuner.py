from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeTuningSuggestion:
    parameter: str
    current_value: float
    proposed_value: float
    reason: str


def suggest_tuning(
    current_spread_threshold_bps: float,
    observed_slippage_bps: float,
    target_slippage_bps: float,
) -> RuntimeTuningSuggestion | None:
    """Suggest a tighter spread threshold when observed slippage exceeds target.

    Returns ``None`` if no adjustment is needed.
    """
    if observed_slippage_bps <= target_slippage_bps:
        return None

    overshoot = observed_slippage_bps - target_slippage_bps
    proposed = max(current_spread_threshold_bps - overshoot, 0.0)

    return RuntimeTuningSuggestion(
        parameter="spread_threshold_bps",
        current_value=current_spread_threshold_bps,
        proposed_value=proposed,
        reason=(
            f"Observed slippage {observed_slippage_bps:.2f} bps exceeds "
            f"target {target_slippage_bps:.2f} bps; tightening spread "
            f"threshold from {current_spread_threshold_bps:.2f} to {proposed:.2f}"
        ),
    )
