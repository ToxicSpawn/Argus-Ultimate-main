from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyTargetProposal:
    strategy_id: str
    symbol: str
    target_weight: float
    current_weight: float
    reference_price: float
    reason: str


class TargetOnlyStrategyMixin:
    """Mixin documenting the target-only strategy contract."""

    def produce_target_proposal(self, *args, **kwargs) -> StrategyTargetProposal:
        raise NotImplementedError
