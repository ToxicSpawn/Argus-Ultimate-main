from __future__ import annotations

from dataclasses import dataclass

from argus_live.execution.router_policy import RouteDecision, select_route


@dataclass(frozen=True)
class SmartRouterInput:
    symbol: str
    spread_bps: float
    volatility_bps: float
    allow_market_orders: bool


class SmartOrderRouter:
    def route(self, data: SmartRouterInput) -> RouteDecision:
        return select_route(
            symbol=data.symbol,
            spread_bps=data.spread_bps,
            volatility_bps=data.volatility_bps,
            allow_market_orders=data.allow_market_orders,
        )
