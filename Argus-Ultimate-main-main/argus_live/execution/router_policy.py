from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RouteDecision:
    venue: str
    order_type: str
    tif: str
    maker_preferred: bool
    reason: str


def _default_venues(symbol: str) -> list[str]:
    return ["kraken", "coinbase_advanced"] if symbol.endswith("/AUD") else ["coinbase_advanced", "kraken"]


def select_route(
    *,
    symbol: str,
    spread_bps: float,
    volatility_bps: float,
    allow_market_orders: bool,
    preferred_mode: str | None = None,
    venue_health: Iterable[object] | None = None,
) -> RouteDecision:
    default_priority = _default_venues(symbol)
    snapshots = list(venue_health or [])
    ranked: list[tuple[float, str, str]] = []
    seen = set()
    for snap in snapshots:
        venue = str(getattr(snap, 'venue_id', ''))
        if not venue:
            continue
        seen.add(venue)
        healthy = bool(getattr(snap, 'healthy', True))
        score = float(getattr(snap, 'score', 50.0))
        penalty = 0.0 if healthy else 100.0
        ranked.append((penalty - score, venue, str(getattr(snap, 'reason', 'venue health'))))
    for venue in default_priority:
        if venue not in seen:
            ranked.append((999.0 + default_priority.index(venue), venue, 'default priority'))
    ranked.sort(key=lambda item: item[0])
    venue = ranked[0][1]
    venue_reason = ranked[0][2]

    mode = (preferred_mode or '').upper()
    force_taker = mode == 'TAKER'
    force_maker = mode == 'MAKER'

    if force_maker:
        return RouteDecision(venue, 'limit', 'GTC', True, f'governance preferred maker; {venue_reason}')
    if force_taker and allow_market_orders:
        return RouteDecision(venue, 'market', 'IOC', False, f'governance preferred taker; {venue_reason}')
    if force_taker:
        return RouteDecision(venue, 'limit', 'IOC', False, f'governance taker preference via limit IOC; {venue_reason}')

    if spread_bps <= 8 and volatility_bps <= 25:
        return RouteDecision(venue, 'limit', 'GTC', True, f'tight spread; {venue_reason}')
    if allow_market_orders:
        return RouteDecision(venue, 'market', 'IOC', False, f'volatile market; {venue_reason}')
    return RouteDecision(venue, 'limit', 'IOC', False, f'fallback limit IOC; {venue_reason}')
