from __future__ import annotations

from dataclasses import dataclass

from argus_live.portfolio.target_models import TargetDelta


@dataclass(frozen=True)
class ExecutionPlan:
    symbol: str
    side: str
    quantity: float
    limit_price: float
    strategy_id: str
    source_proposal_id: str
    source_delta_weight: float
    source_delta_notional: float
    manifest_hash: str


def build_execution_plan(delta: TargetDelta) -> ExecutionPlan:
    quantity = delta.delta_notional / delta.reference_price
    return ExecutionPlan(delta.symbol, delta.side, quantity, delta.reference_price, delta.strategy_id, delta.proposal_id, delta.delta_weight, delta.delta_notional, delta.manifest_hash)
