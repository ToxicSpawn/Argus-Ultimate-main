from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .hostile_replay import ScenarioOrder
from .hostile_scenarios import MarketState, ScenarioPlan, ShockWindow


@dataclass(frozen=True)
class NamedScenario:
    name: str
    description: str
    plan: ScenarioPlan
    orders: List[ScenarioOrder]


def _default_orders(symbol: str, base_price: float) -> List[ScenarioOrder]:
    return [
        ScenarioOrder(symbol=symbol, side='buy', quantity=0.010, strategy_id='mr_1', price=base_price),
        ScenarioOrder(symbol=symbol, side='buy', quantity=0.018, strategy_id='mr_1', price=base_price * 1.0008),
        ScenarioOrder(symbol=symbol, side='sell', quantity=0.012, strategy_id='trend_1', price=base_price * 0.9992),
        ScenarioOrder(symbol=symbol, side='buy', quantity=0.022, strategy_id='trend_1', price=base_price * 1.0012),
    ]


def build_regression_library(symbol: str = 'BTC/AUD', base_price: float = 100_000.0) -> List[NamedScenario]:
    base = MarketState(
        symbol=symbol,
        mid_price=base_price,
        spread_bps=5.0,
        volatility_bps=8.0,
        top_of_book_notional=8_000.0,
    )
    return [
        NamedScenario(
            name='clean_baseline',
            description='Reference clean-path session with modest spread and healthy liquidity.',
            plan=ScenarioPlan(name='clean_baseline', base_state=base, shocks=[]),
            orders=_default_orders(symbol, base_price),
        ),
        NamedScenario(
            name='latency_spike',
            description='ACK/fetch latency spikes while market remains otherwise tradeable.',
            plan=ScenarioPlan(
                name='latency_spike',
                base_state=base,
                shocks=[ShockWindow('latency_open', 1, 2, latency_ms_add=120.0, spread_multiplier=1.25)],
            ),
            orders=_default_orders(symbol, base_price),
        ),
        NamedScenario(
            name='thin_book',
            description='Liquidity collapses and spread widens, creating impact risk and partial fills.',
            plan=ScenarioPlan(
                name='thin_book',
                base_state=base,
                shocks=[ShockWindow('book_vanish', 1, 3, spread_multiplier=2.5, liquidity_multiplier=0.18, fill_probability_multiplier=0.55)],
            ),
            orders=_default_orders(symbol, base_price),
        ),
        NamedScenario(
            name='reject_burst',
            description='Venue reliability degrades and rejects cluster in a short window.',
            plan=ScenarioPlan(
                name='reject_burst',
                base_state=base,
                shocks=[ShockWindow('rejects', 1, 2, reject_probability_add=0.45, latency_ms_add=40.0, venue_quality_multiplier=0.7)],
            ),
            orders=_default_orders(symbol, base_price),
        ),
        NamedScenario(
            name='stale_quote_news_shock',
            description='Fast volatility expansion with stale quotes and widened spread.',
            plan=ScenarioPlan(
                name='stale_quote_news_shock',
                base_state=base,
                shocks=[
                    ShockWindow(
                        'news_burst',
                        0,
                        2,
                        spread_multiplier=3.2,
                        volatility_multiplier=3.0,
                        liquidity_multiplier=0.4,
                        latency_ms_add=60.0,
                        stale_quote=True,
                        fill_probability_multiplier=0.65,
                    ),
                ],
            ),
            orders=_default_orders(symbol, base_price),
        ),
        NamedScenario(
            name='correlated_bad_day',
            description='Compounded stress: spread blowout, thinner book, rejects, and stale quoting.',
            plan=ScenarioPlan(
                name='correlated_bad_day',
                base_state=base,
                shocks=[
                    ShockWindow('spread_regime_break', 0, 3, spread_multiplier=2.8, volatility_multiplier=2.2),
                    ShockWindow('book_thin', 1, 3, liquidity_multiplier=0.22, fill_probability_multiplier=0.6),
                    ShockWindow('venue_degrade', 2, 3, reject_probability_add=0.30, latency_ms_add=90.0, venue_quality_multiplier=0.6, stale_quote=True),
                ],
            ),
            orders=_default_orders(symbol, base_price),
        ),
        NamedScenario(
            name='venue_desync',
            description='Venue quality collapses while latency and rejects rise unevenly, simulating route desynchronization.',
            plan=ScenarioPlan(
                name='venue_desync',
                base_state=base,
                shocks=[ShockWindow('desync', 1, 3, latency_ms_add=140.0, reject_probability_add=0.25, venue_quality_multiplier=0.45, fill_probability_multiplier=0.7)],
            ),
            orders=_default_orders(symbol, base_price),
        ),
        NamedScenario(
            name='liquidity_vacuum_open',
            description='Order book disappears at the open and spread blows out before recovering.',
            plan=ScenarioPlan(
                name='liquidity_vacuum_open',
                base_state=base,
                shocks=[
                    ShockWindow('vacuum', 0, 1, spread_multiplier=4.5, liquidity_multiplier=0.08, volatility_multiplier=2.5, fill_probability_multiplier=0.45),
                    ShockWindow('aftershock', 2, 3, spread_multiplier=1.8, liquidity_multiplier=0.35),
                ],
            ),
            orders=_default_orders(symbol, base_price),
        ),
        NamedScenario(
            name='adverse_selection_spiral',
            description='Fills become increasingly toxic as stale quotes and volatility stack together.',
            plan=ScenarioPlan(
                name='adverse_selection_spiral',
                base_state=base,
                shocks=[ShockWindow('toxicity', 0, 3, spread_multiplier=2.2, volatility_multiplier=2.8, stale_quote=True, venue_quality_multiplier=0.75, fill_probability_multiplier=0.72)],
            ),
            orders=_default_orders(symbol, base_price),
        ),
        NamedScenario(
            name='partial_fill_stall',
            description='Repeated partial fills with rising latency and deteriorating queue quality.',
            plan=ScenarioPlan(
                name='partial_fill_stall',
                base_state=base,
                shocks=[ShockWindow('stall', 1, 3, latency_ms_add=180.0, liquidity_multiplier=0.25, fill_probability_multiplier=0.5, venue_quality_multiplier=0.7)],
            ),
            orders=_default_orders(symbol, base_price),
        ),
        NamedScenario(
            name='venue_degradation_cluster',
            description='Compounded venue weakness with rejects, latency, stale quoting, and thin liquidity.',
            plan=ScenarioPlan(
                name='venue_degradation_cluster',
                base_state=base,
                shocks=[ShockWindow('cluster', 0, 3, spread_multiplier=2.4, liquidity_multiplier=0.2, reject_probability_add=0.35, latency_ms_add=110.0, stale_quote=True, venue_quality_multiplier=0.5, fill_probability_multiplier=0.55)],
            ),
            orders=_default_orders(symbol, base_price),
        ),
    ]


def get_named_scenario(name: str, *, symbol: str = 'BTC/AUD', base_price: float = 100_000.0) -> NamedScenario:
    for scenario in build_regression_library(symbol=symbol, base_price=base_price):
        if scenario.name == name:
            return scenario
    raise KeyError(f'unknown scenario: {name}')
