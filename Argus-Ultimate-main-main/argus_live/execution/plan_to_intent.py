from __future__ import annotations

from argus_live.execution.intent_builder import build_intent
from argus_live.portfolio.delta_planner import ExecutionPlan


def plan_to_intent(plan: ExecutionPlan):
    return build_intent(symbol=plan.symbol, side=plan.side, quantity=plan.quantity, strategy_id=plan.strategy_id, manifest_hash=plan.manifest_hash, limit_price=plan.limit_price)
