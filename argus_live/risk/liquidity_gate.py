from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LiquidityDecision:
    approved_quantity: float
    approved_notional: float
    haircut_ratio: float
    reason: str


def apply_liquidity_haircut(*, requested_quantity: float, reference_price: float, top_of_book_notional: float, max_book_take_ratio: float) -> LiquidityDecision:
    if requested_quantity <= 0:
        return LiquidityDecision(0.0, 0.0, 0.0, "non-positive requested quantity")
    requested_notional = requested_quantity * reference_price
    allowed_notional = top_of_book_notional * max_book_take_ratio
    if allowed_notional <= 0:
        return LiquidityDecision(0.0, 0.0, 0.0, "no usable liquidity")
    approved_notional = min(requested_notional, allowed_notional)
    approved_quantity = approved_notional / reference_price
    haircut_ratio = approved_notional / requested_notional
    return LiquidityDecision(approved_quantity, approved_notional, haircut_ratio, "liquidity haircut applied")
