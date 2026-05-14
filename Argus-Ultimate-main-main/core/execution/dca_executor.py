"""Push 67 — DCA (Dollar-Cost Averaging) Executor.

Splits a target position into N tranches at descending price levels.
Ideal for:
  - Building into a position during dips
  - Reducing average entry price
  - Combining with RL signals for gradual exposure scaling
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import time


class DCAStatus(str, Enum):
    PENDING   = "PENDING"
    PARTIAL   = "PARTIAL"   # some tranches filled
    COMPLETE  = "COMPLETE"  # all tranches filled
    CANCELLED = "CANCELLED"


@dataclass
class DCALevel:
    level_idx: int
    price_target: float    # entry price for this tranche
    size_usd: float        # USD notional for this tranche
    filled: bool = False
    fill_price: Optional[float] = None
    fill_time: Optional[float] = None


@dataclass
class DCAPlan:
    """A multi-tranche DCA entry plan."""

    symbol: str
    side: str              # "buy" | "sell"
    total_usd: float
    n_levels: int = 3
    level_spread: float = 0.005   # 0.5% between levels
    strategy_name: str = "unknown"
    plan_id: str = field(default_factory=lambda: str(int(time.time() * 1e6)))

    # Runtime
    status: DCAStatus = DCAStatus.PENDING
    levels: List[DCALevel] = field(default_factory=list)
    reference_price: float = field(default=0.0, repr=False)
    created_at: float = field(default_factory=time.time, repr=False)

    def __post_init__(self):
        if not self.levels and self.reference_price > 0:
            self._build_levels()

    def build(self, reference_price: float) -> None:
        """Build tranche levels from a reference price."""
        self.reference_price = reference_price
        self._build_levels()

    def _build_levels(self) -> None:
        size_per_level = self.total_usd / self.n_levels
        self.levels = []
        for i in range(self.n_levels):
            if self.side == "buy":
                # Buy lower: each level is deeper discount
                price = self.reference_price * (1.0 - i * self.level_spread)
            else:
                # Sell higher: each level is higher premium
                price = self.reference_price * (1.0 + i * self.level_spread)
            self.levels.append(DCALevel(
                level_idx=i,
                price_target=price,
                size_usd=size_per_level,
            ))

    def evaluate(self, current_price: float) -> List[DCALevel]:
        """Check which levels should be filled at current_price.
        Returns newly filled levels."""
        newly_filled = []
        for lv in self.levels:
            if lv.filled:
                continue
            should_fill = (
                (self.side == "buy" and current_price <= lv.price_target) or
                (self.side == "sell" and current_price >= lv.price_target)
            )
            if should_fill:
                lv.filled = True
                lv.fill_price = current_price
                lv.fill_time = time.time()
                newly_filled.append(lv)

        filled_count = sum(1 for lv in self.levels if lv.filled)
        if filled_count == self.n_levels:
            self.status = DCAStatus.COMPLETE
        elif filled_count > 0:
            self.status = DCAStatus.PARTIAL

        return newly_filled

    @property
    def avg_fill_price(self) -> float:
        filled = [lv for lv in self.levels if lv.filled and lv.fill_price]
        if not filled:
            return 0.0
        total_usd = sum(lv.size_usd for lv in filled)
        weighted = sum(lv.size_usd / lv.fill_price for lv in filled)
        return total_usd / weighted if weighted > 0 else 0.0

    @property
    def filled_usd(self) -> float:
        return sum(lv.size_usd for lv in self.levels if lv.filled)

    @property
    def pending_usd(self) -> float:
        return sum(lv.size_usd for lv in self.levels if not lv.filled)

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "symbol": self.symbol,
            "side": self.side,
            "total_usd": self.total_usd,
            "n_levels": self.n_levels,
            "status": self.status.value,
            "avg_fill_price": self.avg_fill_price,
            "filled_usd": self.filled_usd,
            "pending_usd": self.pending_usd,
            "levels": [
                {"idx": lv.level_idx, "target": lv.price_target,
                 "size": lv.size_usd, "filled": lv.filled}
                for lv in self.levels
            ],
        }


class DCAExecutorEngine:
    """Manages active DCA plans across symbols."""

    def __init__(self):
        self._plans: Dict[str, DCAPlan] = {}

    def add_plan(self, plan: DCAPlan) -> None:
        self._plans[plan.plan_id] = plan

    def evaluate_all(self, prices: Dict[str, float]) -> List[DCALevel]:
        """Evaluate all pending plans. Returns all newly filled levels."""
        all_filled = []
        for plan in self._plans.values():
            if plan.status == DCAStatus.CANCELLED:
                continue
            price = prices.get(plan.symbol)
            if price is None:
                continue
            filled = plan.evaluate(price)
            all_filled.extend(filled)
        return all_filled

    def cancel(self, plan_id: str) -> bool:
        if plan_id in self._plans:
            self._plans[plan_id].status = DCAStatus.CANCELLED
            return True
        return False

    def get_active(self) -> List[DCAPlan]:
        return [
            p for p in self._plans.values()
            if p.status in (DCAStatus.PENDING, DCAStatus.PARTIAL)
        ]

    def get_all(self) -> List[DCAPlan]:
        return list(self._plans.values())
