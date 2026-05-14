from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OrderWithStopPlan:
    entry_price: float
    stop_price: float
    take_profit_price: float | None
    risk_fraction: float
    reason: str


def build_order_with_stop_plan(*, entry_price: float, stop_distance_pct: float, take_profit_distance_pct: float | None = None) -> OrderWithStopPlan:
    if entry_price <= 0:
        raise ValueError("entry_price must be > 0")
    if stop_distance_pct <= 0:
        raise ValueError("stop_distance_pct must be > 0")
    stop_price = round(entry_price * (1.0 - stop_distance_pct), 10)
    take_profit_price = None
    if take_profit_distance_pct is not None:
        if take_profit_distance_pct <= 0:
            raise ValueError("take_profit_distance_pct must be > 0")
        take_profit_price = round(entry_price * (1.0 + take_profit_distance_pct), 10)
    return OrderWithStopPlan(entry_price=entry_price, stop_price=stop_price, take_profit_price=take_profit_price, risk_fraction=stop_distance_pct, reason="protective order planning helper")
