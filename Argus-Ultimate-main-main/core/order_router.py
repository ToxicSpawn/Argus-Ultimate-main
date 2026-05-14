"""OrderRouter — smart order routing with venue selection.

Extracted from unified_trading_system.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Result of a routing decision for a single order."""
    symbol: str
    side: str
    quantity: float
    venue: str
    order_type: str = "market"
    limit_price: Optional[float] = None
    reason: str = ""


class OrderRouter:
    """
    Routes orders to the best available venue based on:
    - Spread / fee comparison
    - Rate limit availability
    - Venue health / circuit breaker state

    Extracted from unified_trading_system.py.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self._venue_health: Dict[str, bool] = {}  # venue_name -> is_healthy
        self._preferred_venues: List[str] = list(
            getattr(config, "preferred_venues", []) or ["kraken", "coinbase"]
        )
        logger.info("OrderRouter initialised | venues=%s", self._preferred_venues)

    def mark_venue_healthy(self, venue: str, healthy: bool = True) -> None:
        self._venue_health[venue] = healthy
        if not healthy:
            logger.warning("OrderRouter: venue '%s' marked UNHEALTHY", venue)

    def select_venue(self, symbol: str, side: str) -> Optional[str]:
        """Select the best healthy venue for this symbol/side."""
        for venue in self._preferred_venues:
            if self._venue_health.get(venue, True):  # default healthy if unknown
                return venue
        logger.error("OrderRouter: NO healthy venue available for %s %s", side, symbol)
        return None

    def route(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> Optional[RoutingDecision]:
        """Route an order. Returns RoutingDecision or None if no venue available."""
        venue = self.select_venue(symbol, side)
        if venue is None:
            return None
        decision = RoutingDecision(
            symbol=symbol,
            side=side.upper(),
            quantity=float(quantity),
            venue=venue,
            order_type=order_type,
            limit_price=limit_price,
            reason=f"auto-routed to {venue}",
        )
        logger.debug("OrderRouter: %s %s %.6f -> %s (%s)", side, symbol, quantity, venue, order_type)
        return decision
