from __future__ import annotations

from argus_live.proving.day_review import DayReview


def build_control_surface(review: DayReview) -> dict:
    health = "GREEN"
    if review.promotion_decision == "NO_GO" or not review.replay_ok or review.scale_blocked:
        health = "RED"
    elif review.promotion_decision == "HOLD" or review.capital_throttled:
        health = "AMBER"

    return {
        "status": health,
        "capital_scaling_ready": bool(review.scale_ready and not review.scale_blocked and review.promotion_decision == 'GO'),
        "recommended_action": review.ladder_recommended_action,
        "blockers": review.active_blockers,
        "ladder": {
            "current": review.ladder_stage,
            "next": review.ladder_next_stage,
            "transition_allowed": review.ladder_transition_allowed,
            "cooldown_hours": review.ladder_cooldown_hours,
        },
        "replay": {
            "ok": review.replay_ok,
            "mismatch_count": review.replay_mismatch_count,
        },
        "regression": {
            "overall_score": review.regression_overall_score,
            "pass_rate": review.regression_pass_rate,
        },
        "scaling": {
            "readiness_score": review.scale_readiness_score,
            "blocked": review.scale_blocked,
            "capital_throttled": review.capital_throttled,
            "throttle_reason": review.throttle_reason,
            "max_safe_aum": review.max_safe_aum,
        },
        "performance": {
            "net_pnl": review.net_pnl,
            "execution_alpha_bps": review.execution_alpha_bps,
            "max_drawdown_pct": review.max_drawdown_pct,
            "slippage_tail_bps": review.slippage_tail_bps,
            "reject_rate_pct": review.reject_rate_pct,
        },
    }
