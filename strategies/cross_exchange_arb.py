"""
Cross-Exchange Arbitrage Strategy — exploits price discrepancies between venues.

Trades when the same asset has a meaningful price difference across exchanges
after accounting for fees and transfer costs.

Example:
  BTC on Kraken: $65,000
  BTC on Bybit:  $65,120
  Spread: $120 = 18.5 bps
  Total fees (maker each side): ~4 bps × 2 = 8 bps
  Net profit: ~10.5 bps per BTC traded

This strategy does NOT execute physical cross-exchange transfers (too slow).
Instead it uses the price discrepancy as an alpha signal for the cheaper venue.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Maximum number of ArbitrageOpportunity records kept per symbol
_MAX_OPPORTUNITY_HISTORY = 100

# Prices older than this many seconds are considered stale
_STALE_PRICE_SECONDS = 10.0


@dataclass
class _PriceQuote:
    """A single price observation from one exchange."""
    exchange: str
    symbol: str
    price: float
    timestamp: datetime

    def age_seconds(self, now: Optional[datetime] = None) -> float:
        if now is None:
            now = datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        delta = now - self.timestamp
        return delta.total_seconds()

    def is_stale(self, now: Optional[datetime] = None) -> bool:
        return self.age_seconds(now) > _STALE_PRICE_SECONDS


@dataclass
class ArbitrageOpportunity:
    """
    A detected cross-exchange price discrepancy for a single symbol.

    Fields
    ------
    symbol : str
        Asset symbol, e.g. ``"BTC"``.
    cheap_exchange : str
        Exchange where the asset is cheapest (buy side).
    expensive_exchange : str
        Exchange where the asset is most expensive (sell side).
    cheap_price : float
        Best ask / mid price on the cheap exchange.
    expensive_price : float
        Best bid / mid price on the expensive exchange.
    spread_bps : float
        Raw price spread in basis points.
    net_spread_bps : float
        Spread remaining after deducting round-trip fees.
    confidence : float
        Confidence score in [0, 1].  Scaled linearly up to 20 bps net spread.
    timestamp : datetime
        Time the opportunity was identified.
    """
    symbol: str
    cheap_exchange: str
    expensive_exchange: str
    cheap_price: float
    expensive_price: float
    spread_bps: float
    net_spread_bps: float
    confidence: float
    timestamp: datetime

    def __post_init__(self) -> None:
        if self.cheap_price <= 0 or self.expensive_price <= 0:
            raise ValueError("Prices must be positive")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1]; got {self.confidence}")

    def implied_profit_usd(self, position_usd: float) -> float:
        """Rough USD P&L for a given notional position size."""
        return position_usd * self.net_spread_bps / 10_000.0


class CrossExchangeArbStrategy:
    """
    Detects cross-exchange price discrepancies and emits arbitrage signals.

    The strategy maintains the latest price quote per ``(exchange, symbol)``
    pair.  On each call to :meth:`find_opportunity` or
    :meth:`generate_signals`, stale quotes are skipped and only
    opportunities whose *net* spread exceeds ``min_net_spread_bps`` are
    returned.

    Parameters
    ----------
    fee_bps_per_side : float
        Assumed taker fee in basis points for one side of the trade.
        Round-trip cost = 2 × fee_bps_per_side.  Default 4.0.
    min_net_spread_bps : float
        Minimum net spread (after fees) required to flag an opportunity.
        Default 5.0.
    max_position_usd : float
        Maximum notional position size per opportunity.  Used only for
        informational purposes in :meth:`ArbitrageOpportunity.implied_profit_usd`.
        Default 500.0.
    """

    MIN_NET_SPREAD_BPS: float = 5.0
    MAX_SPREAD_BPS: float = 200.0  # above this likely a data quality issue

    def __init__(
        self,
        fee_bps_per_side: float = 4.0,
        min_net_spread_bps: float = 5.0,
        max_position_usd: float = 500.0,
    ) -> None:
        if fee_bps_per_side < 0:
            raise ValueError("fee_bps_per_side must be non-negative")
        if min_net_spread_bps < 0:
            raise ValueError("min_net_spread_bps must be non-negative")

        self.fee_bps_per_side = fee_bps_per_side
        self.min_net_spread_bps = min_net_spread_bps
        self.max_position_usd = max_position_usd

        # (exchange, symbol) → _PriceQuote
        self._quotes: Dict[Tuple[str, str], _PriceQuote] = {}

        # symbol → deque of ArbitrageOpportunity (history)
        self._opportunity_history: Dict[str, Deque[ArbitrageOpportunity]] = {}

        logger.info(
            "CrossExchangeArbStrategy initialised: fee_bps=%.1f "
            "min_net_spread_bps=%.1f max_position_usd=%.0f",
            fee_bps_per_side,
            min_net_spread_bps,
            max_position_usd,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_price(
        self,
        exchange: str,
        symbol: str,
        price: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Record the latest mid/last price from *exchange* for *symbol*.

        Parameters
        ----------
        exchange : str
            Exchange name, e.g. ``"kraken"`` or ``"bybit"``.
        symbol : str
            Asset symbol, e.g. ``"BTC"``.
        price : float
            Current price in USD (or quote currency).
        timestamp : datetime, optional
            Observation time.  Defaults to ``datetime.now(timezone.utc)``.
        """
        if price <= 0:
            logger.warning(
                "update_price: non-positive price %.6f from %s/%s — ignored",
                price, exchange, symbol,
            )
            return

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        key = (exchange.lower(), symbol.upper())
        self._quotes[key] = _PriceQuote(
            exchange=exchange.lower(),
            symbol=symbol.upper(),
            price=price,
            timestamp=timestamp,
        )

        logger.debug(
            "update_price: %s/%s price=%.4f ts=%s",
            exchange, symbol, price, timestamp.isoformat(),
        )

    def find_opportunity(
        self,
        symbol: str,
        now: Optional[datetime] = None,
    ) -> Optional[ArbitrageOpportunity]:
        """
        Check all tracked exchanges for a profitable spread on *symbol*.

        Returns the best (highest net spread) :class:`ArbitrageOpportunity`
        found, or ``None`` if no valid opportunity exists.
        """
        symbol = symbol.upper()
        if now is None:
            now = datetime.now(timezone.utc)

        # Gather fresh quotes for this symbol across all exchanges
        quotes = self._fresh_quotes_for_symbol(symbol, now)

        if len(quotes) < 2:
            logger.debug(
                "find_opportunity: need at least 2 fresh quotes for %s; got %d",
                symbol, len(quotes),
            )
            return None

        # Find cheapest and most expensive
        cheapest = min(quotes, key=lambda q: q.price)
        priciest = max(quotes, key=lambda q: q.price)

        if cheapest.exchange == priciest.exchange:
            return None  # same exchange — no arb

        spread_bps = self._raw_spread_bps(cheapest.price, priciest.price)

        # Sanity check: ignore likely data errors
        if spread_bps > self.MAX_SPREAD_BPS:
            logger.warning(
                "find_opportunity: spread %.1f bps for %s exceeds MAX_SPREAD_BPS %.1f "
                "— possible data quality issue (cheap=%s/%.4f expensive=%s/%.4f)",
                spread_bps, symbol, self.MAX_SPREAD_BPS,
                cheapest.exchange, cheapest.price,
                priciest.exchange, priciest.price,
            )
            return None

        net_spread_bps = self._compute_net_spread(cheapest.price, priciest.price)

        if net_spread_bps < self.min_net_spread_bps:
            logger.debug(
                "find_opportunity: net_spread %.2f bps < threshold %.2f bps for %s",
                net_spread_bps, self.min_net_spread_bps, symbol,
            )
            return None

        confidence = min(1.0, net_spread_bps / 20.0)

        opp = ArbitrageOpportunity(
            symbol=symbol,
            cheap_exchange=cheapest.exchange,
            expensive_exchange=priciest.exchange,
            cheap_price=cheapest.price,
            expensive_price=priciest.price,
            spread_bps=spread_bps,
            net_spread_bps=net_spread_bps,
            confidence=confidence,
            timestamp=now,
        )

        # Store in history
        if symbol not in self._opportunity_history:
            self._opportunity_history[symbol] = deque(maxlen=_MAX_OPPORTUNITY_HISTORY)
        self._opportunity_history[symbol].append(opp)

        logger.info(
            "ArbitrageOpportunity: %s %s→%s spread=%.2f bps net=%.2f bps confidence=%.3f",
            symbol,
            cheapest.exchange,
            priciest.exchange,
            spread_bps,
            net_spread_bps,
            confidence,
        )

        return opp

    def generate_signals(
        self,
        now: Optional[datetime] = None,
    ) -> List[ArbitrageOpportunity]:
        """
        Scan every tracked symbol and return all valid opportunities.

        Results are sorted by ``net_spread_bps`` descending.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        tracked_symbols = {key[1] for key in self._quotes.keys()}
        results: List[ArbitrageOpportunity] = []

        for symbol in tracked_symbols:
            opp = self.find_opportunity(symbol, now=now)
            if opp is not None:
                results.append(opp)

        results.sort(key=lambda o: o.net_spread_bps, reverse=True)
        return results

    def opportunity_history(self, symbol: str) -> List[ArbitrageOpportunity]:
        """Return the last up-to-100 opportunities detected for *symbol*."""
        symbol = symbol.upper()
        if symbol not in self._opportunity_history:
            return []
        return list(self._opportunity_history[symbol])

    def tracked_exchanges(self) -> List[str]:
        """Return the unique exchange names currently in the quote book."""
        return sorted({key[0] for key in self._quotes.keys()})

    def tracked_symbols(self) -> List[str]:
        """Return the unique symbols currently in the quote book."""
        return sorted({key[1] for key in self._quotes.keys()})

    def stale_quotes(self, now: Optional[datetime] = None) -> List[_PriceQuote]:
        """Return all quotes that are currently considered stale."""
        if now is None:
            now = datetime.now(timezone.utc)
        return [q for q in self._quotes.values() if q.is_stale(now)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fresh_quotes_for_symbol(
        self,
        symbol: str,
        now: datetime,
    ) -> List[_PriceQuote]:
        """Return non-stale quotes across all exchanges for *symbol*."""
        result: List[_PriceQuote] = []
        for (exch, sym), quote in self._quotes.items():
            if sym != symbol:
                continue
            if quote.is_stale(now):
                logger.debug(
                    "_fresh_quotes: stale quote from %s/%s (age=%.1fs)",
                    exch, sym, quote.age_seconds(now),
                )
                continue
            result.append(quote)
        return result

    @staticmethod
    def _raw_spread_bps(cheap: float, expensive: float) -> float:
        """Gross spread in basis points: (expensive - cheap) / cheap × 10000."""
        if cheap == 0:
            return 0.0
        return (expensive - cheap) / cheap * 10_000.0

    def _compute_net_spread(self, cheap: float, expensive: float) -> float:
        """
        Net spread in basis points after deducting round-trip fees.

        Round-trip fee = 2 × fee_bps_per_side.
        """
        gross = self._raw_spread_bps(cheap, expensive)
        round_trip_fee = 2.0 * self.fee_bps_per_side
        return gross - round_trip_fee
