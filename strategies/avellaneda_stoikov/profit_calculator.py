"""PnL accounting helpers for market making."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FillRecord:
    side: str
    price: float
    quantity: float
    reference_mid: float
    timestamp: float = field(default_factory=time.time)
    fee: float = 0.0


@dataclass(slots=True)
class PnLSnapshot:
    realized_pnl: float
    unrealized_pnl: float
    mark_to_market: float
    spread_capture: float
    adverse_selection_cost: float
    inventory: float
    average_cost: float
    total_pnl: float


class ProfitCalculator:
    """Tracks realized/unrealized PnL and market-making edge decomposition."""

    def __init__(self) -> None:
        self.inventory: float = 0.0
        self.average_cost: float = 0.0
        self.realized_pnl: float = 0.0
        self.spread_capture_total: float = 0.0
        self.adverse_selection_total: float = 0.0
        self.fees_total: float = 0.0
        self._fills: List[FillRecord] = []

    @property
    def fills(self) -> List[FillRecord]:
        return list(self._fills)

    def process_fill(self, fill: FillRecord) -> None:
        if fill.quantity <= 0 or fill.price <= 0 or fill.reference_mid <= 0:
            raise ValueError("fill values must be positive")

        signed_qty = fill.quantity if fill.side.lower() == "buy" else -fill.quantity
        old_inventory = self.inventory
        new_inventory = old_inventory + signed_qty

        edge = (fill.reference_mid - fill.price) if fill.side.lower() == "buy" else (fill.price - fill.reference_mid)
        self.spread_capture_total += edge * fill.quantity
        self.adverse_selection_total += abs(fill.reference_mid - fill.price) * fill.quantity
        self.fees_total += fill.fee

        if old_inventory == 0 or (old_inventory > 0) == (signed_qty > 0):
            total_qty = abs(old_inventory) + fill.quantity
            weighted_cost = self.average_cost * abs(old_inventory) + fill.price * fill.quantity
            self.average_cost = weighted_cost / total_qty if total_qty else 0.0
        else:
            closed_qty = min(abs(old_inventory), fill.quantity)
            if old_inventory > 0:
                self.realized_pnl += (fill.price - self.average_cost) * closed_qty
            else:
                self.realized_pnl += (self.average_cost - fill.price) * closed_qty
            if fill.quantity > abs(old_inventory):
                self.average_cost = fill.price
            elif new_inventory == 0:
                self.average_cost = 0.0

        self.inventory = new_inventory
        self._fills.append(fill)

    def mark_to_market(self, current_mid: float) -> float:
        if current_mid <= 0:
            raise ValueError("current_mid must be positive")
        return self.inventory * current_mid

    def unrealized_pnl(self, current_mid: float) -> float:
        if self.inventory == 0:
            return 0.0
        return self.inventory * (current_mid - self.average_cost)

    def adverse_selection_cost(self) -> float:
        return self.adverse_selection_total

    def snapshot(self, current_mid: Optional[float] = None) -> PnLSnapshot:
        if current_mid is None:
            current_mid = self.average_cost if self.average_cost > 0 else 1.0
        unrealized = self.unrealized_pnl(current_mid)
        total = self.realized_pnl + unrealized + self.spread_capture_total - self.adverse_selection_total - self.fees_total
        return PnLSnapshot(
            realized_pnl=self.realized_pnl,
            unrealized_pnl=unrealized,
            mark_to_market=self.mark_to_market(current_mid),
            spread_capture=self.spread_capture_total,
            adverse_selection_cost=self.adverse_selection_total,
            inventory=self.inventory,
            average_cost=self.average_cost,
            total_pnl=total,
        )
