"""OrderRouter — symbol-level smart venue routing — Push 56."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Venue:
    """Trading venue descriptor."""
    name: str
    maker_fee_bps: float = 0.0
    taker_fee_bps: float = 2.0
    supports_market: bool = True
    supports_limit: bool = True
    supports_stop: bool = False
    enabled: bool = True
    priority: int = 0       # lower = higher priority

    @property
    def effective_fee(self) -> float:
        return (self.maker_fee_bps + self.taker_fee_bps) / 2


class OrderRouter:
    """Routes orders to the lowest-fee enabled venue.

    Parameters
    ----------
    venues : list of Venue
        Available trading venues, sorted by priority then fee.
    """

    # Default Argus venue roster
    DEFAULT_VENUES: List[Venue] = [
        Venue("bybit",   maker_fee_bps=0.0, taker_fee_bps=2.0, priority=1, supports_stop=True),
        Venue("mexc",    maker_fee_bps=0.0, taker_fee_bps=0.0, priority=2, supports_stop=False),
        Venue("woo_x",   maker_fee_bps=-1.5, taker_fee_bps=3.0, priority=3, supports_stop=False),
        Venue("btcmarkets", maker_fee_bps=-5.0, taker_fee_bps=8.5, priority=4, supports_stop=False),
        Venue("sim",     maker_fee_bps=0.0, taker_fee_bps=0.0, priority=99, supports_stop=True),
    ]

    def __init__(self, venues: Optional[List[Venue]] = None) -> None:
        self._venues = sorted(
            venues or self.DEFAULT_VENUES,
            key=lambda v: (v.priority, v.effective_fee),
        )

    def route(
        self,
        symbol: str,
        order_type: str = "market",
        prefer_maker: bool = True,
    ) -> Optional[Venue]:
        """Return the best enabled venue for a given order type."""
        needs_stop = order_type in {"stop", "stop_limit"}
        candidates = [
            v for v in self._venues
            if v.enabled
            and (not needs_stop or v.supports_stop)
        ]
        if not candidates:
            logger.warning("OrderRouter: no venue available for %s %s", symbol, order_type)
            return None
        if prefer_maker:
            candidates.sort(key=lambda v: (v.priority, v.maker_fee_bps))
        else:
            candidates.sort(key=lambda v: (v.priority, v.taker_fee_bps))
        chosen = candidates[0]
        logger.debug("OrderRouter: %s %s -> %s", symbol, order_type, chosen.name)
        return chosen

    def disable_venue(self, name: str) -> None:
        for v in self._venues:
            if v.name == name:
                v.enabled = False
                logger.info("OrderRouter: disabled venue %s", name)

    def enable_venue(self, name: str) -> None:
        for v in self._venues:
            if v.name == name:
                v.enabled = True
                logger.info("OrderRouter: enabled venue %s", name)

    @property
    def active_venues(self) -> List[str]:
        return [v.name for v in self._venues if v.enabled]
