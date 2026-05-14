from argus_live.promotion.ladder import evaluate_ladder_transition
from argus_live.proving.day_review import build_day_review
from argus_live.risk.capital_scaling import derive_capital_scaling_policy


def test_ladder_transition_blocks_on_hold():
    decision = evaluate_ladder_transition(
        current_stage="LIVE_SAFE_MICRO",
        promotion_decision="HOLD",
        replay_ok=True,
        critical_incident_count=0,
        regression_pass_rate=100.0,
        active_blockers=["net pnl non-positive"],
    )
    assert decision.transition_allowed is False
    assert decision.next_stage == "LIVE_SAFE"
    assert "net pnl non-positive" in decision.blockers


def test_capital_scaling_blocks_over_max_safe_aum():
    policy = derive_capital_scaling_policy(
        equity=250000.0,
        ladder_stage="LIVE_SAFE",
        replay_ok=True,
        execution_alpha_bps=2.0,
        critical_incident_count=0,
        max_safe_aum=100000.0,
    )
    assert policy.blocked is True
    assert any("max_safe_aum" in b for b in policy.blockers)


def test_day_review_carries_ladder_fields():
    review = build_day_review(
        run_id="r1",
        config_hash="cfg",
        trade_count=5,
        net_pnl=10.0,
        execution_alpha_bps=1.0,
        max_drawdown_pct=0.5,
        reject_rate_pct=0.0,
        slippage_tail_bps=5.0,
        replay_ok=True,
        replay_mismatch_count=0,
        critical_incident_count=0,
        regression_overall_score=90.0,
        regression_pass_rate=100.0,
        ladder_stage="SHADOW",
    )
    assert review.ladder_stage == "SHADOW"
    assert review.ladder_next_stage == "LIVE_SAFE_MICRO"
