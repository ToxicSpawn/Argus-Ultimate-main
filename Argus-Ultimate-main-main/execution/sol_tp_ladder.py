"""
SOL Take-Profit Ladder — tiered exit to capture SOL's volatile upswings.

Problem with 3Commas DCA flat TP:
  A single TP at e.g. +3% leaves money on the table when SOL extends 6-10%.
  It also forces the full position to wait for recovery after a dip.

This ladder:
  Tier 1: exit 40% of position at avg_entry * (1 + TP1_PCT)  → bank quick gains
  Tier 2: exit 40% of position at avg_entry * (1 + TP2_PCT)  → ride the move
  Tier 3: trail remaining 20% with ATR stop                  → capture extended runs

If price reverses before a tier is hit, the deal stop in sol_dca_superior
closes the remaining position.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


TP1_PCT = 0.015   # 1.5%
TP2_PCT = 0.030   # 3.0%
TP3_TRAIL_ATR_MULT = 1.8  # trail with 1.8x ATR for tier 3


@dataclass
class TpLadderState:
    avg_entry: float
    atr_at_open: float
    total_qty: float
    tier1_filled: bool = False
    tier2_filled: bool = False
    tier3_filled: bool = False
    tier3_high_water: float = 0.0
    remaining_qty: float = 0.0

    def __post_init__(self) -> None:
        self.remaining_qty = self.total_qty
        self.tier3_high_water = self.avg_entry

    @property
    def tp1_price(self) -> float:
        return self.avg_entry * (1 + TP1_PCT)

    @property
    def tp2_price(self) -> float:
        return self.avg_entry * (1 + TP2_PCT)

    def tier3_trail_stop(self) -> float:
        return self.tier3_high_water - (self.atr_at_open * TP3_TRAIL_ATR_MULT)


class SolTpLadder:
    """
    Manages tiered exits for an open SOL position.

    Usage:
        ladder = SolTpLadder()
        ladder.open_deal(avg_entry=150.0, atr=2.1, qty=1.0)
        orders = ladder.on_tick(price=152.5)  # returns exit orders when triggered
    """

    def __init__(self) -> None:
        self.state: Optional[TpLadderState] = None
        self.history: List[dict] = []

    def open_deal(
        self,
        avg_entry: float,
        atr: float,
        qty: float,
    ) -> None:
        """Register a new deal to manage."""
        self.state = TpLadderState(
            avg_entry=avg_entry,
            atr_at_open=atr,
            total_qty=qty,
        )

    def on_tick(self, price: float) -> List[dict]:
        """
        Call on each price update. Returns a list of exit orders (may be empty).
        Each order: {'tier': int, 'qty': float, 'price': float, 'reason': str}
        """
        if self.state is None:
            return []

        s = self.state
        orders: List[dict] = []

        # Tier 1: 40% at TP1
        if not s.tier1_filled and price >= s.tp1_price:
            qty = s.total_qty * 0.40
            s.remaining_qty -= qty
            s.tier1_filled = True
            orders.append({
                "tier": 1, "qty": qty, "price": price,
                "reason": f"tp_ladder_tier1_{price:.4f}",
            })

        # Tier 2: 40% at TP2
        if not s.tier2_filled and price >= s.tp2_price:
            qty = s.total_qty * 0.40
            s.remaining_qty -= qty
            s.tier2_filled = True
            orders.append({
                "tier": 2, "qty": qty, "price": price,
                "reason": f"tp_ladder_tier2_{price:.4f}",
            })

        # Tier 3: trail remaining 20%
        if s.tier1_filled and s.tier2_filled and not s.tier3_filled:
            s.tier3_high_water = max(s.tier3_high_water, price)
            if price <= s.tier3_trail_stop():
                qty = s.remaining_qty
                s.remaining_qty = 0.0
                s.tier3_filled = True
                self.history.append({
                    "avg_entry": s.avg_entry, "tier3_exit": price,
                    "total_qty": s.total_qty,
                })
                self.state = None
                orders.append({
                    "tier": 3, "qty": qty, "price": price,
                    "reason": f"tp_ladder_tier3_trail_{price:.4f}",
                })

        return orders

    def force_close(self, price: float) -> Optional[dict]:
        """Close all remaining quantity (called on deal stop or emergency)."""
        if self.state is None or self.state.remaining_qty <= 0:
            return None
        qty = self.state.remaining_qty
        self.state = None
        return {
            "tier": 0, "qty": qty, "price": price,
            "reason": "force_close",
        }
