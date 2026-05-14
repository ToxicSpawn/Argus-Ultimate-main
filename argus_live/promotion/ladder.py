from __future__ import annotations

from dataclasses import dataclass


LADDER_ORDER = ["PAPER", "SHADOW", "LIVE_SAFE_MICRO", "LIVE_SAFE", "SCALE"]


@dataclass(frozen=True)
class LadderTransitionDecision:
    current_stage: str
    next_stage: str
    transition_allowed: bool
    blockers: list[str]
    cooldown_hours: int = 0
    recommended_action: str = "HOLD"
    scale_ready: bool = False


def normalize_stage(stage: str | None) -> str:
    value = str(stage or "PAPER").upper()
    return value if value in LADDER_ORDER else "PAPER"


def next_stage(current_stage: str) -> str:
    current_stage = normalize_stage(current_stage)
    idx = LADDER_ORDER.index(current_stage)
    return LADDER_ORDER[min(idx + 1, len(LADDER_ORDER) - 1)]


def previous_stage(current_stage: str) -> str:
    current_stage = normalize_stage(current_stage)
    idx = LADDER_ORDER.index(current_stage)
    return LADDER_ORDER[max(idx - 1, 0)]


def evaluate_ladder_transition(
    *,
    current_stage: str,
    promotion_decision: str,
    replay_ok: bool,
    critical_incident_count: int,
    regression_pass_rate: float,
    scale_readiness_score: float = 100.0,
    active_blockers: list[str] | None = None,
) -> LadderTransitionDecision:
    stage = normalize_stage(current_stage)
    blockers = list(active_blockers or [])
    if not replay_ok and "replay mismatch present" not in blockers:
        blockers.append("replay mismatch present")
    if critical_incident_count > 0 and "critical incidents present" not in blockers:
        blockers.append("critical incidents present")
    if stage in {"LIVE_SAFE", "SCALE"} and regression_pass_rate < 100.0 and "regression suite did not fully pass" not in blockers:
        blockers.append("regression suite did not fully pass")
    if stage in {"LIVE_SAFE", "SCALE"} and scale_readiness_score < 85.0 and "scale readiness below 85" not in blockers:
        blockers.append("scale readiness below 85")
    if promotion_decision != "GO" and not blockers:
        blockers.append(f"promotion decision is {promotion_decision}")

    scale_ready = scale_readiness_score >= 85.0 and replay_ok and critical_incident_count == 0 and regression_pass_rate >= 100.0
    allowed = promotion_decision == "GO" and not blockers and stage != "SCALE"
    cooldown_hours = 24 if promotion_decision == "HOLD" else (48 if promotion_decision == "NO_GO" else 0)

    if promotion_decision == "NO_GO" and stage not in {"PAPER", "SHADOW"}:
        action = f"DEMOTE_TO_{previous_stage(stage)}"
    elif blockers and stage == "SCALE":
        action = "FREEZE_SCALE"
    elif allowed:
        action = f"PROMOTE_TO_{next_stage(stage)}"
    else:
        action = "HOLD"

    return LadderTransitionDecision(
        current_stage=stage,
        next_stage=next_stage(stage),
        transition_allowed=allowed,
        blockers=blockers,
        cooldown_hours=cooldown_hours,
        recommended_action=action,
        scale_ready=scale_ready,
    )
