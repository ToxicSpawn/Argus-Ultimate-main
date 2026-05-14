from __future__ import annotations

import json
from pathlib import Path

from argus_live.proving.control_surface import build_control_surface
from argus_live.proving.day_review import DayReview


def build_operator_summary(review: DayReview) -> dict:
    base = {
        "run_id": review.run_id,
        "config_hash": review.config_hash,
        "promotion_decision": review.promotion_decision,
        "ladder_stage": review.ladder_stage,
        "ladder_next_stage": review.ladder_next_stage,
        "ladder_transition_allowed": review.ladder_transition_allowed,
        "ladder_cooldown_hours": review.ladder_cooldown_hours,
        "ladder_recommended_action": review.ladder_recommended_action,
        "scale_ready": review.scale_ready,
        "scale_readiness_score": review.scale_readiness_score,
        "scale_blocked": review.scale_blocked,
        "capital_throttled": review.capital_throttled,
        "throttle_reason": review.throttle_reason,
        "max_safe_aum": review.max_safe_aum,
        "active_blockers": review.active_blockers,
        "trade_count": review.trade_count,
        "net_pnl": review.net_pnl,
        "execution_alpha_bps": review.execution_alpha_bps,
        "max_drawdown_pct": review.max_drawdown_pct,
        "reject_rate_pct": review.reject_rate_pct,
        "slippage_tail_bps": review.slippage_tail_bps,
        "replay_ok": review.replay_ok,
        "replay_mismatch_count": review.replay_mismatch_count,
        "critical_incident_count": review.critical_incident_count,
        "regression_overall_score": review.regression_overall_score,
        "regression_pass_rate": review.regression_pass_rate,
        "regression_summary_path": review.regression_summary_path,
        "config_quarantined": review.config_quarantined,
        "rollback_tag": review.rollback_tag,
        "comparative_summary_path": review.comparative_summary_path,
        "comparative_score_delta": review.comparative_score_delta,
        "config_change_count": review.config_change_count,
    }
    base["control_surface"] = build_control_surface(review)
    return base


def write_operator_summary(review: DayReview, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(build_operator_summary(review), indent=2, sort_keys=True), encoding="utf-8")
