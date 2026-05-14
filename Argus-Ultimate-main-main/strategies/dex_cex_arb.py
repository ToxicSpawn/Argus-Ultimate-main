"""
DEX-CEX Arbitrage Strategy — exploits price differences between decentralised
and centralised exchanges.

Block times (not network latency) are the bottleneck for DEX trades, making
this strategy viable from any geographic location.

Flow:
  1. Receive live CEX prices from existing exchange feeds
  2. Receive live DEX pool prices from DEX connector
  3. Compare prices across venues, accounting for:
     - Gas costs (DEX side)
     - Trading fees (both sides)
     - Slippage / price impact
     - Minimum profit threshold
  4. Emit TradingSignal when a profitable opportunity is found

All features are disabled by default (``dex.enabled: false``).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

from core.connectors.dex_base import DEXConnector

logger = logging.getLogger(__name__)

# Prices older than this are stale
_STALE_SECONDS = 10.0

# Max history per pair
_MAX_HISTORY = 200


@dataclass
class DEXPriceQuote:
    """A single DEX pool price observation."""

    pool_address: str
    price: float
    reserves: Dict[str, float]
    timestamp: float  # time.time()

    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def is_stale(self) -> bool:
        return self.age_seconds() > _STALE_SECONDS


@dataclass
class CEXPriceQuote:
    """A single CEX price observation."""

    exchange: str
    symbol: str
    price: float
    timestamp: float

    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def is_stale(self) -> bool:
        return self.age_seconds() > _STALE_SECONDS


@dataclass
class ArbOpportunity:
    """A detected DEX-CEX arbitrage opportunity."""

    symbol: str
    direction: str  # 'buy_dex_sell_cex' or 'buy_cex_sell_dex'
    dex_price: float
    cex_price: float
    cex_exchange: str
    pool_address: str
    profit_bps: float
    gas_cost_usd: float
    optimal_size: float
    confidence: float
    timestamp: datetime

    def expected_profit_usd(self, position_usd: float) -> float:
        """Estimated USD profit for a given position size."""
        return position_usd * self.profit_bps / 10_000.0 - self.gas_cost_usd


@dataclass
class TradingSignal:
    """Signal emitted by the DEX-CEX arb strategy."""

    symbol: str
    action: str  # 'buy' or 'sell'
    confidence: float
    strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DEXCEXArbitrage:
    """
    Strategy that finds and exploits price differences between DEX and CEX.

    Parameters
    ----------
    min_profit_bps : float
        Minimum profit in basis points after all costs.
    max_gas_cost_usd : float
        Maximum acceptable gas cost in USD.
    max_position_usd : float
        Maximum position size per arb trade.
    cex_fee_bps : float
        Assumed CEX taker fee in basis points per side.
    dex_fee_bps : float
        Default DEX pool fee in basis points.
    slippage_buffer_bps : float
        Extra slippage buffer added to cost calculations.
    """

    def __init__(
        self,
        min_profit_bps: float = 10.0,
        max_gas_cost_usd: float = 5.0,
        max_position_usd: float = 500.0,
        cex_fee_bps: float = 4.0,
        dex_fee_bps: float = 30.0,
        slippage_buffer_bps: float = 5.0,
    ) -> None:
        if min_profit_bps < 0:
            raise ValueError("min_profit_bps must be non-negative")
        if max_gas_cost_usd < 0:
            raise ValueError("max_gas_cost_usd must be non-negative")
        if max_position_usd <= 0:
            raise ValueError("max_position_usd must be positive")

        self.min_profit_bps = min_profit_bps
        self.max_gas_cost_usd = max_gas_cost_usd
        self.max_position_usd = max_position_usd
        self.cex_fee_bps = cex_fee_bps
        self.dex_fee_bps = dex_fee_bps
        self.slippage_buffer_bps = slippage_buffer_bps

        # Latest prices
        self._cex_prices: Dict[str, CEXPriceQuote] = {}  # (exchange, symbol) key as str
        self._dex_prices: Dict[str, DEXPriceQuote] = {}  # pool_address key

        # Mapping: symbol → list of pool addresses
        self._symbol_pools: Dict[str, List[str]] = {}

        # Mapping: pool_address → symbol
        self._pool_symbols: Dict[str, str] = {}

        # Current gas cost estimate
        self._gas_cost_usd: float = 0.5  # default estimate

        # Opportunity history
        self._history: Deque[ArbOpportunity] = deque(maxlen=_MAX_HISTORY)

        logger.info(
            "DEXCEXArbitrage init: min_profit=%dbps max_gas=$%.2f max_pos=$%.0f",
            min_profit_bps,
            max_gas_cost_usd,
            max_position_usd,
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def register_pool(self, symbol: str, pool_address: str) -> None:
        """Associate a DEX pool address with a trading symbol."""
        symbol = symbol.upper()
        pool_address = pool_address.lower()
        if symbol not in self._symbol_pools:
            self._symbol_pools[symbol] = []
        if pool_address not in self._symbol_pools[symbol]:
            self._symbol_pools[symbol].append(pool_address)
        self._pool_symbols[pool_address] = symbol
        logger.debug("Registered pool %s for %s", pool_address[:10], symbol)

    def update_gas_cost(self, gas_cost_usd: float) -> None:
        """Update the current estimated gas cost for DEX transactions."""
        self._gas_cost_usd = max(0.0, gas_cost_usd)

    # ------------------------------------------------------------------
    # Price updates
    # ------------------------------------------------------------------

    def update_cex_price(self, symbol: str, price: float, exchange: str) -> None:
        """
        Called with live CEX prices from existing feeds.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g. "ETH", "BTC").
        price : float
            Current price in USD.
        exchange : str
            Exchange name (e.g. "kraken", "coinbase").
        """
        if price <= 0:
            return
        key = f"{exchange.lower()}:{symbol.upper()}"
        self._cex_prices[key] = CEXPriceQuote(
            exchange=exchange.lower(),
            symbol=symbol.upper(),
            price=price,
            timestamp=time.time(),
        )

    def update_dex_price(
        self,
        pool_address: str,
        price: float,
        reserves: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Called with live DEX pool prices.

        Parameters
        ----------
        pool_address : str
            The pool contract address.
        price : float
            Current pool price (token0 in terms of token1 / USD).
        reserves : dict, optional
            Pool reserves {token0_reserve, token1_reserve}.
        """
        if price <= 0:
            return
        pool_address = pool_address.lower()
        self._dex_prices[pool_address] = DEXPriceQuote(
            pool_address=pool_address,
            price=price,
            reserves=reserves or {},
            timestamp=time.time(),
        )

    # ------------------------------------------------------------------
    # Opportunity detection
    # ------------------------------------------------------------------

    def find_opportunities(self) -> List[ArbOpportunity]:
        """
        Compare all CEX vs DEX prices and return profitable arbs.

        Filters:
        - Price difference must exceed min_profit_bps + gas cost + slippage
        - Gas cost must be below max_gas_cost_usd
        - Prices must not be stale

        Returns
        -------
        list of ArbOpportunity
            Sorted by profit_bps descending.
        """
        opportunities: List[ArbOpportunity] = []
        now = datetime.now(timezone.utc)

        for pool_addr, dex_quote in self._dex_prices.items():
            if dex_quote.is_stale():
                continue

            symbol = self._pool_symbols.get(pool_addr)
            if not symbol:
                continue

            # Check against all CEX prices for this symbol
            for key, cex_quote in self._cex_prices.items():
                if cex_quote.symbol != symbol:
                    continue
                if cex_quote.is_stale():
                    continue

                opp = self._evaluate_pair(dex_quote, cex_quote, now)
                if opp is not None:
                    opportunities.append(opp)
                    self._history.append(opp)

        opportunities.sort(key=lambda o: o.profit_bps, reverse=True)
        return opportunities

    def _evaluate_pair(
        self,
        dex: DEXPriceQuote,
        cex: CEXPriceQuote,
        now: datetime,
    ) -> Optional[ArbOpportunity]:
        """Evaluate a single DEX-CEX price pair for arbitrage."""
        dex_price = dex.price
        cex_price = cex.price

        if dex_price <= 0 or cex_price <= 0:
            return None

        # Calculate raw spread in bps
        spread_bps = abs(dex_price - cex_price) / min(dex_price, cex_price) * 10_000.0

        # Total cost in bps: CEX fee + DEX fee + slippage buffer
        total_cost_bps = self.cex_fee_bps + self.dex_fee_bps + self.slippage_buffer_bps

        # Gas cost as bps of max position
        gas_cost_bps = (self._gas_cost_usd / self.max_position_usd) * 10_000.0 if self.max_position_usd > 0 else 0

        net_profit_bps = spread_bps - total_cost_bps - gas_cost_bps

        if net_profit_bps < self.min_profit_bps:
            return None

        if self._gas_cost_usd > self.max_gas_cost_usd:
            return None

        # Determine direction
        if dex_price < cex_price:
            direction = "buy_dex_sell_cex"
        else:
            direction = "buy_cex_sell_dex"

        # Calculate optimal size — bounded by max position and reserves
        optimal_size = self.max_position_usd
        reserves = dex.reserves
        if reserves:
            # Limit size to avoid excessive price impact (< 1% of reserves)
            reserve_val = (
                reserves.get("token0_reserve", 0) * dex_price
                + reserves.get("token1_reserve", 0)
            )
            if reserve_val > 0:
                max_from_reserves = reserve_val * 0.01  # 1% of pool
                optimal_size = min(optimal_size, max_from_reserves)

        confidence = min(1.0, net_profit_bps / 30.0)

        return ArbOpportunity(
            symbol=cex.symbol,
            direction=direction,
            dex_price=dex_price,
            cex_price=cex_price,
            cex_exchange=cex.exchange,
            pool_address=dex.pool_address,
            profit_bps=net_profit_bps,
            gas_cost_usd=self._gas_cost_usd,
            optimal_size=optimal_size,
            confidence=confidence,
            timestamp=now,
        )

    def generate_signal(self, opportunity: ArbOpportunity) -> TradingSignal:
        """
        Convert an ArbOpportunity to a TradingSignal for the execution pipeline.

        Parameters
        ----------
        opportunity : ArbOpportunity
            The arbitrage opportunity to convert.

        Returns
        -------
        TradingSignal
            Signal compatible with the ARGUS execution pipeline.
        """
        if opportunity.direction == "buy_dex_sell_cex":
            action = "buy"
            entry = opportunity.dex_price
            # Stop loss: if DEX price rises above CEX price, arb is gone
            stop_loss = entry * 1.005  # 0.5% above entry
            take_profit = entry * (1.0 + opportunity.profit_bps / 10_000.0)
        else:
            action = "sell"
            entry = opportunity.cex_price
            stop_loss = entry * 0.995
            take_profit = entry * (1.0 - opportunity.profit_bps / 10_000.0)

        reasoning = (
            f"DEX-CEX arb: {opportunity.direction} "
            f"spread={opportunity.profit_bps:.1f}bps "
            f"gas=${opportunity.gas_cost_usd:.2f} "
            f"size=${opportunity.optimal_size:.0f} "
            f"DEX={opportunity.dex_price:.2f} CEX={opportunity.cex_price:.2f} "
            f"on {opportunity.cex_exchange}"
        )

        return TradingSignal(
            symbol=opportunity.symbol,
            action=action,
            confidence=opportunity.confidence,
            strength=min(1.0, opportunity.profit_bps / 50.0),
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            metadata={
                "strategy": "dex_cex_arb",
                "direction": opportunity.direction,
                "profit_bps": opportunity.profit_bps,
                "gas_cost_usd": opportunity.gas_cost_usd,
                "pool_address": opportunity.pool_address,
                "cex_exchange": opportunity.cex_exchange,
            },
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_history(self) -> List[ArbOpportunity]:
        """Return recent opportunity history."""
        return list(self._history)

    def get_stats(self) -> Dict[str, Any]:
        """Return strategy statistics."""
        return {
            "cex_prices_tracked": len(self._cex_prices),
            "dex_pools_tracked": len(self._dex_prices),
            "registered_pools": len(self._pool_symbols),
            "opportunities_found": len(self._history),
            "current_gas_usd": self._gas_cost_usd,
        }
