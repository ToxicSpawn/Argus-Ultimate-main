from __future__ import annotations

from dataclasses import dataclass

from argus_live.execution.slippage_model import SlippageEstimate, estimate_slippage


@dataclass(frozen=True)
class SmartExecutionInput:
    spread_bps: float
    volatility_bps: float
    participation_ratio: float


class SmartExecutionCore:
    def estimate(self, data: SmartExecutionInput) -> SlippageEstimate:
        return estimate_slippage(
            spread_bps=data.spread_bps,
            volatility_bps=data.volatility_bps,
            participation_ratio=data.participation_ratio,
        )
