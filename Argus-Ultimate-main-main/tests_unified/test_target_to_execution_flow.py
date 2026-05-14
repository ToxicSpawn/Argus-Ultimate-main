from argus_live.execution.plan_to_intent import plan_to_intent
from argus_live.portfolio.delta_planner import build_execution_plan
from argus_live.portfolio.target_engine import evaluate_target
from argus_live.portfolio.target_models import TargetProposal


def test_target_delta_produces_intent() -> None:
    proposal = TargetProposal.new(
        strategy_id="target_engine_test",
        symbol="BTC/AUD",
        target_weight=0.10,
        current_weight=0.02,
        reference_price=50000.0,
        manifest_hash="sha256:test",
    )
    decision = evaluate_target(proposal=proposal, portfolio_equity=100000.0)
    assert decision.accepted is True
    assert decision.delta is not None
    plan = build_execution_plan(decision.delta)
    intent = plan_to_intent(plan)
    assert intent.symbol == "BTC/AUD"
    assert intent.side == "buy"
    assert intent.quantity > 0
