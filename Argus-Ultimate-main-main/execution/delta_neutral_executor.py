from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HedgeSuggestion:
    primary_symbol: str
    hedge_symbol: str
    primary_notional: float
    hedge_notional: float
    reason: str


class DeltaNeutralExecutor:
    def suggest_hedge(self, *, primary_symbol: str, hedge_symbol: str, primary_notional: float, hedge_ratio: float) -> HedgeSuggestion:
        if primary_notional <= 0:
            raise ValueError("primary_notional must be > 0")
        if hedge_ratio < 0:
            raise ValueError("hedge_ratio must be >= 0")
        return HedgeSuggestion(primary_symbol=primary_symbol, hedge_symbol=hedge_symbol, primary_notional=primary_notional, hedge_notional=primary_notional * hedge_ratio, reason="delta-neutral advisory suggestion")
