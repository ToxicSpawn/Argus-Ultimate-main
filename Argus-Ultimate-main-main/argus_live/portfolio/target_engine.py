from __future__ import annotations

from dataclasses import dataclass

from argus_live.portfolio.target_models import TargetDelta, TargetProposal


@dataclass(frozen=True)
class TargetDecision:
    accepted: bool
    reason: str
    delta: TargetDelta | None


def evaluate_target(*, proposal: TargetProposal, portfolio_equity: float, min_rebalance_weight: float = 0.0025) -> TargetDecision:
    delta_weight = proposal.target_weight - proposal.current_weight
    if abs(delta_weight) < min_rebalance_weight:
        return TargetDecision(False, "rebalance suppression threshold", None)
    delta_notional = abs(delta_weight) * portfolio_equity
    side = "buy" if delta_weight > 0 else "sell"
    return TargetDecision(True, "target delta accepted", TargetDelta(proposal.proposal_id, proposal.strategy_id, proposal.symbol, delta_weight, delta_notional, side, proposal.reference_price, proposal.manifest_hash))
