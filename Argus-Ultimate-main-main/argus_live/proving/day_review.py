from __future__ import annotations

from dataclasses import dataclass, asdict

from argus_live.promotion.promotion_gate import PromotionGateDecision, evaluate_promotion_gate


@dataclass(frozen=True)
class DayReview:
    run_id: str
    config_hash: str
    trade_count: int
    net_pnl: float
    execution_alpha_bps: float
    max_drawdown_pct: float
    reject_rate_pct: float
    slippage_tail_bps: float
    replay_ok: bool
    replay_mismatch_count: int
    critical_incident_count: int
    regression_overall_score: float
    regression_pass_rate: float
    regression_summary_path: str
    active_blockers: list[str]
    promotion_decision: str
    ladder_stage: str
    ladder_next_stage: str
    ladder_transition_allowed: bool
    ladder_cooldown_hours: int
    ladder_recommended_action: str
    scale_ready: bool
    scale_readiness_score: float
    scale_blocked: bool
    capital_throttled: bool
    throttle_reason: str
    max_safe_aum: float
    reasons: list[str]
    config_quarantined: bool = False
    rollback_tag: str = ""
    comparative_summary_path: str = ""
    comparative_score_delta: float = 0.0
    config_change_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def build_day_review(
    *,
    run_id: str,
    config_hash: str,
    trade_count: int,
    net_pnl: float,
    execution_alpha_bps: float,
    max_drawdown_pct: float,
    reject_rate_pct: float,
    slippage_tail_bps: float,
    replay_ok: bool,
    replay_mismatch_count: int,
    critical_incident_count: int,
    regression_overall_score: float = 0.0,
    regression_pass_rate: float = 0.0,
    regression_summary_path: str = "",
    ladder_stage: str = "PAPER",
    scale_readiness_score: float = 100.0,
    scale_blocked: bool = False,
    capital_throttled: bool = False,
    throttle_reason: str = "",
    max_safe_aum: float = 0.0,
    config_quarantined: bool = False,
    rollback_tag: str = "",
    comparative_summary_path: str = "",
    comparative_score_delta: float = 0.0,
    config_change_count: int = 0,
) -> DayReview:
    gate: PromotionGateDecision = evaluate_promotion_gate(
        net_pnl=net_pnl,
        execution_alpha_bps=execution_alpha_bps,
        max_drawdown_pct=max_drawdown_pct,
        replay_ok=replay_ok,
        critical_incident_count=critical_incident_count,
        reject_rate_pct=reject_rate_pct,
        slippage_tail_bps=slippage_tail_bps,
        regression_overall_score=regression_overall_score,
        regression_pass_rate=regression_pass_rate,
        current_stage=ladder_stage,
        scale_readiness_score=scale_readiness_score,
    )
    return DayReview(
        run_id=run_id,
        config_hash=config_hash,
        trade_count=trade_count,
        net_pnl=net_pnl,
        execution_alpha_bps=execution_alpha_bps,
        max_drawdown_pct=max_drawdown_pct,
        reject_rate_pct=reject_rate_pct,
        slippage_tail_bps=slippage_tail_bps,
        replay_ok=replay_ok,
        replay_mismatch_count=replay_mismatch_count,
        critical_incident_count=critical_incident_count,
        regression_overall_score=regression_overall_score,
        regression_pass_rate=regression_pass_rate,
        regression_summary_path=regression_summary_path,
        active_blockers=list(gate.blockers) if gate.decision != "GO" else [],
        promotion_decision=gate.decision,
        ladder_stage=(gate.ladder_transition.current_stage if gate.ladder_transition else ladder_stage),
        ladder_next_stage=(gate.ladder_transition.next_stage if gate.ladder_transition else ladder_stage),
        ladder_transition_allowed=(gate.ladder_transition.transition_allowed if gate.ladder_transition else False),
        ladder_cooldown_hours=(gate.ladder_transition.cooldown_hours if gate.ladder_transition else 0),
        ladder_recommended_action=(gate.ladder_transition.recommended_action if gate.ladder_transition else 'HOLD'),
        scale_ready=(gate.ladder_transition.scale_ready if gate.ladder_transition else False),
        scale_readiness_score=gate.scale_readiness_score,
        scale_blocked=scale_blocked,
        capital_throttled=capital_throttled,
        throttle_reason=throttle_reason,
        max_safe_aum=max_safe_aum,
        reasons=gate.reasons,
        config_quarantined=config_quarantined,
        rollback_tag=rollback_tag,
        comparative_summary_path=comparative_summary_path,
        comparative_score_delta=comparative_score_delta,
        config_change_count=config_change_count,
    )
