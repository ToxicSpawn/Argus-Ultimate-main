from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioEdgeReport:
    gross_edge_bps: float
    total_fee_bps: float
    total_slippage_bps: float
    turnover_penalty_bps: float
    net_portfolio_edge_bps: float
    reason: str


def compute_portfolio_edge(
    gross_edge_bps: float,
    total_fee_bps: float,
    total_slippage_bps: float,
    turnover_penalty_bps: float,
) -> PortfolioEdgeReport:
    net = gross_edge_bps - total_fee_bps - total_slippage_bps - turnover_penalty_bps
    return PortfolioEdgeReport(
        gross_edge_bps=gross_edge_bps,
        total_fee_bps=total_fee_bps,
        total_slippage_bps=total_slippage_bps,
        turnover_penalty_bps=turnover_penalty_bps,
        net_portfolio_edge_bps=net,
        reason=(
            f"net edge {net:.1f} bps = "
            f"{gross_edge_bps:.1f} gross - {total_fee_bps:.1f} fee "
            f"- {total_slippage_bps:.1f} slip - {turnover_penalty_bps:.1f} turnover"
        ),
    )
