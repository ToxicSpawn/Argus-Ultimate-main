from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, List


@dataclass(frozen=True)
class MarketState:
    symbol: str
    mid_price: float
    spread_bps: float
    volatility_bps: float
    top_of_book_notional: float
    reject_probability: float = 0.0
    latency_ms: float = 25.0
    fill_probability_multiplier: float = 1.0
    venue_quality: float = 1.0
    stale_quote: bool = False


@dataclass(frozen=True)
class ShockWindow:
    name: str
    start_step: int
    end_step: int
    spread_multiplier: float = 1.0
    volatility_multiplier: float = 1.0
    liquidity_multiplier: float = 1.0
    reject_probability_add: float = 0.0
    latency_ms_add: float = 0.0
    fill_probability_multiplier: float = 1.0
    venue_quality_multiplier: float = 1.0
    stale_quote: bool = False

    def active(self, step: int) -> bool:
        return self.start_step <= step <= self.end_step


@dataclass(frozen=True)
class ScenarioPlan:
    name: str
    base_state: MarketState
    shocks: List[ShockWindow]


class HostileScenarioInjector:
    def __init__(self, plan: ScenarioPlan) -> None:
        self.plan = plan

    def state_at(self, step: int) -> MarketState:
        state = self.plan.base_state
        spread = state.spread_bps
        vol = state.volatility_bps
        liq = state.top_of_book_notional
        reject = state.reject_probability
        latency = state.latency_ms
        fill_mult = state.fill_probability_multiplier
        venue_quality = state.venue_quality
        stale = state.stale_quote
        for shock in self.plan.shocks:
            if not shock.active(step):
                continue
            spread *= shock.spread_multiplier
            vol *= shock.volatility_multiplier
            liq *= shock.liquidity_multiplier
            reject = min(1.0, reject + shock.reject_probability_add)
            latency += shock.latency_ms_add
            fill_mult *= shock.fill_probability_multiplier
            venue_quality *= shock.venue_quality_multiplier
            stale = stale or shock.stale_quote
        return MarketState(
            symbol=state.symbol,
            mid_price=state.mid_price,
            spread_bps=spread,
            volatility_bps=vol,
            top_of_book_notional=max(1.0, liq),
            reject_probability=min(1.0, reject),
            latency_ms=max(0.0, latency),
            fill_probability_multiplier=max(0.05, fill_mult),
            venue_quality=max(0.05, venue_quality),
            stale_quote=stale,
        )
