#!/usr/bin/env python3
"""
Transaction cost model — spread, commission, market impact, slippage.

Provides a unified CostEstimate for any proposed trade, used by the
portfolio rebalancer and execution layer to make cost-aware decisions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CostEstimate:
    """Breakdown of estimated transaction costs."""

    spread_bps: float
    commission_bps: float
    market_impact_bps: float
    slippage_bps: float
    total_bps: float
    total_usd: float


class TransactionCostModel:
    """Estimates all-in transaction cost for a proposed trade."""

    def __init__(self, config: Any = None) -> None:
        self._config = config

        self._default_spread_bps: float = float(
            _cfg(config, "transaction_cost_model.default_spread_bps", 5.0)
        )
        self._commission_bps: float = float(
            _cfg(config, "transaction_cost_model.commission_bps", 26.0)
        )
        self._slippage_pct: float = float(
            _cfg(config, "transaction_cost_model.slippage_pct", 0.001)
        )
        self._impact_coeff: float = float(
            _cfg(config, "transaction_cost_model.market_impact_coefficient", 0.1)
        )

        # Optional L2 spread cache: symbol -> spread_bps
        self._l2_spreads: dict[str, float] = {}

        # Optional ADV (average daily volume in USD) cache: symbol -> adv
        self._adv_cache: dict[str, float] = {}

        logger.info(
            "TransactionCostModel initialised  commission=%.1f bps  spread_default=%.1f bps",
            self._commission_bps,
            self._default_spread_bps,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_cost(
        self,
        symbol: str,
        quantity: float,
        side: str,
        current_price: float,
    ) -> CostEstimate:
        """Return a full cost breakdown for the proposed trade."""

        notional = abs(quantity * current_price) if current_price > 0 else 0.0

        # Spread
        spread_bps = self._l2_spreads.get(symbol, self._default_spread_bps)

        # Commission
        commission_bps = self._commission_bps

        # Market impact — simplified Almgren: impact = coeff * sqrt(notional / ADV) * 10000
        market_impact_bps = 0.0
        adv = self._adv_cache.get(symbol)
        if adv and adv > 0 and notional > 0:
            try:
                market_impact_bps = self._impact_coeff * math.sqrt(notional / adv) * 10000.0
            except (ValueError, ZeroDivisionError):
                market_impact_bps = 0.0

        # Slippage
        slippage_bps = self._slippage_pct * 10000.0  # convert pct to bps

        total_bps = spread_bps + commission_bps + market_impact_bps + slippage_bps
        total_usd = (total_bps / 10000.0) * notional if notional > 0 else 0.0

        return CostEstimate(
            spread_bps=round(spread_bps, 2),
            commission_bps=round(commission_bps, 2),
            market_impact_bps=round(market_impact_bps, 2),
            slippage_bps=round(slippage_bps, 2),
            total_bps=round(total_bps, 2),
            total_usd=round(total_usd, 4),
        )

    # ------------------------------------------------------------------
    # L2 / ADV injection
    # ------------------------------------------------------------------

    def update_l2_spread(self, symbol: str, spread_bps: float) -> None:
        """Inject live L2 spread for a symbol."""
        self._l2_spreads[symbol] = spread_bps

    def update_adv(self, symbol: str, adv_usd: float) -> None:
        """Inject average daily volume for market impact calculation."""
        self._adv_cache[symbol] = adv_usd

    def clear_l2_spread(self, symbol: str) -> None:
        """Remove L2 spread override, falling back to default."""
        self._l2_spreads.pop(symbol, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(config: Any, dotted_key: str, default: Any) -> Any:
    """Safely navigate nested config via dotted key."""
    if config is None:
        return default
    parts = dotted_key.split(".")
    obj = config
    for part in parts:
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            obj = getattr(obj, part, None)
        if obj is None:
            return default
    return obj
