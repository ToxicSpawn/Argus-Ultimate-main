from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PnLAttribution:
    gross_pnl: float
    fees: float
    slippage_cost: float
    net_pnl: float


def attribute_trade_pnl(*, fill_qty: float, entry_price: float, mark_price: float, fee_rate: float, expected_slippage_bps: float) -> PnLAttribution:
    gross = (mark_price - entry_price) * fill_qty
    notional = fill_qty * entry_price
    fees = notional * fee_rate
    slippage_cost = notional * (expected_slippage_bps / 10_000.0)
    net = gross - fees - slippage_cost
    return PnLAttribution(gross, fees, slippage_cost, net)
