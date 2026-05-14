"""Inventory risk management utilities for Avellaneda-Stoikov market making."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class InventoryRiskConfig:
    """Configuration for inventory-aware quote skewing and hard limits."""

    gamma: float = 0.1
    max_position: float = 5.0
    soft_limit_fraction: float = 0.7
    inventory_target: float = 0.0
    max_skew_bps: float = 20.0

    def __post_init__(self) -> None:
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.max_position <= 0:
            raise ValueError("max_position must be positive")
        if not 0.0 < self.soft_limit_fraction <= 1.0:
            raise ValueError("soft_limit_fraction must be in (0, 1]")
        if self.max_skew_bps < 0:
            raise ValueError("max_skew_bps must be non-negative")


@dataclass(slots=True)
class InventoryState:
    """Current inventory state tracked by the market maker."""

    position: float = 0.0
    average_cost: float = 0.0
    realized_pnl: float = 0.0
    fills: int = 0


class InventoryRiskManager:
    """Tracks inventory and computes quote skew/penalty terms."""

    def __init__(self, config: InventoryRiskConfig):
        self.config = config
        self.state = InventoryState()

    @property
    def position(self) -> float:
        return self.state.position

    @property
    def gamma(self) -> float:
        return self.config.gamma

    def inventory_ratio(self) -> float:
        return max(-1.0, min(1.0, self.state.position / self.config.max_position))

    def within_limits(self) -> bool:
        return abs(self.state.position) <= self.config.max_position

    def is_soft_limit_breached(self) -> bool:
        return abs(self.inventory_ratio()) >= self.config.soft_limit_fraction

    def remaining_capacity(self) -> float:
        return max(0.0, self.config.max_position - abs(self.state.position))

    def inventory_penalty(self, volatility: float, time_remaining: float) -> float:
        return self.gamma * (volatility ** 2) * max(time_remaining, 0.0) * (self.state.position ** 2)

    def reservation_price_adjustment(self, volatility: float, time_remaining: float) -> float:
        return -self.state.position * self.gamma * (volatility ** 2) * max(time_remaining, 0.0)

    def quote_skew(self, mid_price: float) -> tuple[float, float]:
        if mid_price <= 0:
            return 0.0, 0.0

        ratio = self.inventory_ratio()
        max_skew = mid_price * self.config.max_skew_bps / 10_000.0
        skew = max_skew * ratio
        bid_adjustment = -max(skew, 0.0) + min(skew, 0.0) * 0.5
        ask_adjustment = -max(skew, 0.0) * 0.5 + min(skew, 0.0)
        return bid_adjustment, ask_adjustment

    def clamp_order_size(self, requested_size: float, side: str) -> float:
        size = max(0.0, requested_size)
        if side.lower() == "buy":
            max_size = self.config.max_position - self.state.position
        elif side.lower() == "sell":
            max_size = self.config.max_position + self.state.position
        else:
            raise ValueError(f"unsupported side: {side}")
        return max(0.0, min(size, max_size))

    def update_fill(self, side: str, quantity: float, price: float) -> None:
        if quantity <= 0 or price <= 0:
            raise ValueError("fill quantity and price must be positive")

        signed_qty = quantity if side.lower() == "buy" else -quantity
        old_position = self.state.position
        new_position = old_position + signed_qty

        if old_position == 0 or (old_position > 0) == (signed_qty > 0):
            total_qty = abs(old_position) + quantity
            weighted_cost = self.state.average_cost * abs(old_position) + price * quantity
            self.state.average_cost = weighted_cost / total_qty if total_qty else 0.0
        else:
            closed_qty = min(abs(old_position), quantity)
            if old_position > 0:
                self.state.realized_pnl += (price - self.state.average_cost) * closed_qty
            else:
                self.state.realized_pnl += (self.state.average_cost - price) * closed_qty
            if abs(quantity) > abs(old_position):
                self.state.average_cost = price
            elif new_position == 0:
                self.state.average_cost = 0.0

        self.state.position = new_position
        self.state.fills += 1

        if not self.within_limits():
            logger.warning(
                "Inventory limit breached: position=%.6f max=%.6f",
                self.state.position,
                self.config.max_position,
            )
