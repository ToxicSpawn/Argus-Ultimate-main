"""Profit Target Stop - tiered profit taking with partial exits."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ProfitTier:
    target_pct: float  # e.g. 0.02 = 2%
    exit_fraction: float  # e.g. 0.33 = close 33% of position
    triggered: bool = False


class ProfitTargetStop:
    """Multi-tier profit taking: partial exits at defined profit levels."""
    def __init__(self, tiers: List[Tuple[float, float]] = None, atr_based: bool = True, atr_tp_mult: float = 3.0):
        if tiers is None:
            tiers = [(0.01, 0.25), (0.02, 0.25), (0.03, 0.25), (0.05, 0.25)]
        self.tiers = [ProfitTier(t, f) for t, f in tiers]
        self.atr_based = atr_based
        self.atr_tp_mult = atr_tp_mult

    def reset(self) -> None:
        for t in self.tiers:
            t.triggered = False

    def calculate(self, entry_price: float, current_price: float, side: str, current_atr: float = 0.0) -> dict:
        if side == "long":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        exits = []
        total_exit_fraction = 0.0
        for tier in self.tiers:
            if not tier.triggered and pnl_pct >= tier.target_pct:
                tier.triggered = True
                exits.append({"target_pct": tier.target_pct, "exit_fraction": tier.exit_fraction})
                total_exit_fraction += tier.exit_fraction
        # ATR-based overall target
        atr_target_price = None
        if self.atr_based and current_atr > 0:
            if side == "long":
                atr_target_price = entry_price + self.atr_tp_mult * current_atr
            else:
                atr_target_price = entry_price - self.atr_tp_mult * current_atr
        return {"pnl_pct": pnl_pct, "exits": exits, "total_exit_fraction": total_exit_fraction, "atr_target_price": atr_target_price, "method": "profit_target_stop"}
