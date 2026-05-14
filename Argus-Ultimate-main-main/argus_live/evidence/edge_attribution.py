from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EdgeAttribution:
    """Immutable record of edge attribution for a single trade."""

    strategy_id: str
    symbol: str
    expected_edge_bps: float
    realized_pnl_bps: float
    slippage_bps: float
    fee_bps: float
    net_edge_bps: float


def attribute_trade(
    strategy_id: str,
    symbol: str,
    expected_edge_bps: float,
    realized_pnl_bps: float,
    slippage_bps: float,
    fee_bps: float,
) -> EdgeAttribution:
    """Compute net edge after subtracting execution costs.

    net_edge_bps = realized_pnl_bps - slippage_bps - fee_bps
    """
    net_edge_bps = realized_pnl_bps - slippage_bps - fee_bps
    return EdgeAttribution(
        strategy_id=strategy_id,
        symbol=symbol,
        expected_edge_bps=expected_edge_bps,
        realized_pnl_bps=realized_pnl_bps,
        slippage_bps=slippage_bps,
        fee_bps=fee_bps,
        net_edge_bps=net_edge_bps,
    )
