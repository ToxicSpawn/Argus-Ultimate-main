"""
Futures Basis Arbitrage — captures the futures premium over spot.

Perpetual futures often trade at a premium/discount to spot due to funding.
When futures trade at a premium (positive basis), a market-neutral strategy:
  - Buys spot BTC
  - Sells BTC perpetual futures
  - Captures the premium decay (or funding payments)

This is NOT a directional bet — pure basis/carry trade.

Key metrics:
  - Annualised basis: (futures_price/spot_price - 1) × 365/days_to_expiry
  - For perps: use funding rate × 3 × 365 as annualised carry
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Number of funding periods per day for 8-hour funding cycles (standard in crypto)
_FUNDING_PERIODS_PER_DAY = 3
_DAYS_PER_YEAR = 365

# Confidence scaling: 1.0 confidence at this annualised basis level
_CONFIDENCE_SCALE_PCT = 20.0


@dataclass
class BasisOpportunity:
    """
    A detected basis arbitrage opportunity between spot and perpetual futures.

    Fields
    ------
    symbol : str
        Asset symbol, e.g. ``"BTC"``.
    spot_price : float
        Current spot mid price.
    futures_price : float
        Current perpetual futures mid price.
    basis_pct : float
        Raw basis as a percentage: ``(futures/spot - 1) × 100``.
    annualised_basis_pct : float
        Annualised carry assuming perpetual funding dynamics.
    funding_rate : float
        Latest 8-hour funding rate (signed).
    action : str
        ``"LONG_BASIS"`` — futures at premium, sell futures / buy spot.
        ``"SHORT_BASIS"`` — futures at discount, buy futures / sell spot.
        ``"NEUTRAL"`` — basis below threshold.
    confidence : float
        Signal confidence in [0, 1].
    """
    symbol: str
    spot_price: float
    futures_price: float
    basis_pct: float
    annualised_basis_pct: float
    funding_rate: float
    action: str    # "LONG_BASIS", "SHORT_BASIS", or "NEUTRAL"
    confidence: float

    def __post_init__(self) -> None:
        valid_actions = ("LONG_BASIS", "SHORT_BASIS", "NEUTRAL")
        if self.action not in valid_actions:
            raise ValueError(
                f"action must be one of {valid_actions}; got {self.action!r}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0, 1]; got {self.confidence}"
            )
        if self.spot_price <= 0 or self.futures_price <= 0:
            raise ValueError("Prices must be positive")

    def implied_annual_return_pct(self) -> float:
        """Return the annualised return estimate after fees."""
        return self.annualised_basis_pct

    def is_actionable(self) -> bool:
        """Return True when action is not NEUTRAL."""
        return self.action != "NEUTRAL"


class FuturesBasisArbStrategy:
    """
    Identifies basis arbitrage opportunities between spot and perpetual futures.

    The strategy requires spot and futures prices to be fed independently via
    :meth:`update_spot` and :meth:`update_futures`.  Call :meth:`compute_basis`
    to evaluate the opportunity for a given symbol, or
    :meth:`generate_signal` which only returns a :class:`BasisOpportunity`
    when the annualised basis exceeds the configured threshold.

    Parameters
    ----------
    min_annual_basis_pct : float
        Minimum annualised basis (%) required to emit a signal.  Default 5.0.
    max_position_usd : float
        Maximum notional USD size per leg.  Default 300.0.
    fee_bps : float
        Round-trip fee in basis points (both legs combined).  Default 6.0.
    """

    MIN_ANNUAL_BASIS_PCT: float = 5.0
    MAX_ANNUAL_BASIS_PCT: float = 200.0  # above this is a data error

    def __init__(
        self,
        min_annual_basis_pct: float = 5.0,
        max_position_usd: float = 300.0,
        fee_bps: float = 6.0,
    ) -> None:
        if min_annual_basis_pct <= 0:
            raise ValueError("min_annual_basis_pct must be positive")
        if max_position_usd <= 0:
            raise ValueError("max_position_usd must be positive")
        if fee_bps < 0:
            raise ValueError("fee_bps must be non-negative")

        self.min_annual_basis_pct = min_annual_basis_pct
        self.max_position_usd = max_position_usd
        self.fee_bps = fee_bps

        # Latest spot price per symbol
        self._spot: Dict[str, float] = {}
        # Latest futures price per symbol
        self._futures: Dict[str, float] = {}
        # Latest funding rate per symbol
        self._funding_rate: Dict[str, float] = {}
        # Last update timestamps
        self._spot_ts: Dict[str, datetime] = {}
        self._futures_ts: Dict[str, datetime] = {}

        logger.info(
            "FuturesBasisArbStrategy initialised: min_annual_basis=%.1f%% "
            "max_position_usd=%.0f fee_bps=%.1f",
            min_annual_basis_pct,
            max_position_usd,
            fee_bps,
        )

    # ------------------------------------------------------------------
    # Public API — data ingestion
    # ------------------------------------------------------------------

    def update_spot(
        self,
        symbol: str,
        price: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Feed the latest spot price for *symbol*.

        Parameters
        ----------
        symbol : str
            Asset symbol, e.g. ``"BTC"``.
        price : float
            Spot mid price in USD.
        timestamp : datetime, optional
            Defaults to ``datetime.now(timezone.utc)``.
        """
        if price <= 0:
            logger.warning(
                "update_spot: non-positive price %.6f for %s — ignored",
                price, symbol,
            )
            return

        symbol = symbol.upper()
        self._spot[symbol] = price
        self._spot_ts[symbol] = timestamp or datetime.now(timezone.utc)

        logger.debug("update_spot: %s price=%.4f", symbol, price)

    def update_futures(
        self,
        symbol: str,
        price: float,
        funding_rate: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Feed the latest perpetual futures price and funding rate for *symbol*.

        Parameters
        ----------
        symbol : str
            Asset symbol, e.g. ``"BTC"``.
        price : float
            Futures mid price in USD.
        funding_rate : float
            Signed 8-hour funding rate (e.g. 0.0001 = 0.01 %).
        timestamp : datetime, optional
            Defaults to ``datetime.now(timezone.utc)``.
        """
        if price <= 0:
            logger.warning(
                "update_futures: non-positive price %.6f for %s — ignored",
                price, symbol,
            )
            return

        symbol = symbol.upper()
        self._futures[symbol] = price
        self._funding_rate[symbol] = funding_rate
        self._futures_ts[symbol] = timestamp or datetime.now(timezone.utc)

        logger.debug(
            "update_futures: %s price=%.4f funding=%.6f",
            symbol, price, funding_rate,
        )

    # ------------------------------------------------------------------
    # Public API — signal generation
    # ------------------------------------------------------------------

    def compute_basis(self, symbol: str) -> Optional[BasisOpportunity]:
        """
        Compute the current basis opportunity for *symbol*.

        Returns ``None`` if spot or futures data has not yet been provided.
        Returns a :class:`BasisOpportunity` with action ``"NEUTRAL"`` when
        the basis is below threshold.
        """
        symbol = symbol.upper()

        if symbol not in self._spot:
            logger.debug("compute_basis: no spot data for %s", symbol)
            return None
        if symbol not in self._futures:
            logger.debug("compute_basis: no futures data for %s", symbol)
            return None

        spot = self._spot[symbol]
        futures = self._futures[symbol]
        funding_rate = self._funding_rate.get(symbol, 0.0)

        if spot <= 0:
            logger.warning("compute_basis: non-positive spot price for %s", symbol)
            return None

        # Raw basis as a percentage
        basis_pct = (futures / spot - 1.0) * 100.0

        # Annualised carry from funding rate for perpetual contracts
        # funding_rate is per 8h period; multiply by periods/day × days/year
        annualised_from_funding = (
            funding_rate * _FUNDING_PERIODS_PER_DAY * _DAYS_PER_YEAR * 100.0
        )

        # Blend: use funding-implied annualised rate as primary signal
        # Deduct fee drag (fee_bps = round-trip, annualised)
        fee_drag_pct = self.fee_bps / 100.0  # bps → pct, already annual drag
        annualised_basis_pct = abs(annualised_from_funding) - fee_drag_pct

        # Determine action based on sign of basis and funding
        action, confidence = self._classify_basis(
            basis_pct, annualised_basis_pct, funding_rate
        )

        opportunity = BasisOpportunity(
            symbol=symbol,
            spot_price=spot,
            futures_price=futures,
            basis_pct=basis_pct,
            annualised_basis_pct=annualised_basis_pct,
            funding_rate=funding_rate,
            action=action,
            confidence=confidence,
        )

        logger.debug(
            "compute_basis: %s basis=%.3f%% annual=%.2f%% action=%s confidence=%.3f",
            symbol, basis_pct, annualised_basis_pct, action, confidence,
        )

        return opportunity

    def generate_signal(self, symbol: str) -> Optional[BasisOpportunity]:
        """
        Return a :class:`BasisOpportunity` only when the annualised basis
        exceeds ``min_annual_basis_pct`` and action is not ``"NEUTRAL"``.

        Returns ``None`` otherwise.
        """
        opp = self.compute_basis(symbol)
        if opp is None:
            return None

        if opp.action == "NEUTRAL":
            return None

        if abs(opp.annualised_basis_pct) < self.min_annual_basis_pct:
            logger.debug(
                "generate_signal: basis %.2f%% below threshold %.2f%% for %s",
                opp.annualised_basis_pct, self.min_annual_basis_pct, symbol,
            )
            return None

        if abs(opp.annualised_basis_pct) > self.MAX_ANNUAL_BASIS_PCT:
            logger.warning(
                "generate_signal: basis %.2f%% exceeds MAX for %s — "
                "possible data error",
                opp.annualised_basis_pct, symbol,
            )
            return None

        logger.info(
            "BasisOpportunity: %s action=%s annual_basis=%.2f%% confidence=%.3f",
            symbol, opp.action, opp.annualised_basis_pct, opp.confidence,
        )

        return opp

    def analyze(
        self,
        symbol: str,
        spot_price: float,
        futures_price: float,
        funding_rate: float,
        spot_exchange: str = "kraken",
        perp_exchange: str = "bybit",
    ) -> Optional[Dict]:
        """
        Convenience method: feed prices, generate signal with both-legs info.

        Returns a dict suitable for execution with spot and perp legs specified,
        or None if no actionable signal.

        Args:
            symbol: Asset symbol, e.g. "BTC"
            spot_price: Current spot mid price
            futures_price: Current perp futures mid price
            funding_rate: Latest 8-hour funding rate
            spot_exchange: Spot venue name
            perp_exchange: Perp venue name

        Returns:
            Signal dict with both legs or None
        """
        self.update_spot(symbol, spot_price)
        self.update_futures(symbol, futures_price, funding_rate)

        opp = self.generate_signal(symbol)
        if opp is None:
            return None

        # Map symbol to CCXT perp format
        perp_symbol_map = {
            "BTC": "BTC/USDT:USDT",
            "ETH": "ETH/USDT:USDT",
            "SOL": "SOL/USDT:USDT",
            "XRP": "XRP/USDT:USDT",
        }
        perp_symbol = perp_symbol_map.get(symbol.upper(), f"{symbol.upper()}/USDT:USDT")
        spot_symbol_map = {
            "BTC": "BTC/USD",
            "ETH": "ETH/USD",
            "SOL": "SOL/USD",
            "XRP": "XRP/USD",
        }
        spot_symbol = spot_symbol_map.get(symbol.upper(), f"{symbol.upper()}/USD")

        if opp.action == "LONG_BASIS":
            # Futures at premium: buy spot, sell perp
            spot_side = "BUY"
            perp_side = "SELL"
        else:
            # SHORT_BASIS: sell spot, buy perp
            spot_side = "SELL"
            perp_side = "BUY"

        return {
            "symbol": symbol.upper(),
            "action": opp.action,
            "confidence": opp.confidence,
            "source": "futures_basis_arb",
            "spot_exchange": spot_exchange,
            "spot_symbol": spot_symbol,
            "spot_side": spot_side,
            "spot_price": opp.spot_price,
            "perp_exchange": perp_exchange,
            "perp_symbol": perp_symbol,
            "perp_side": perp_side,
            "perp_price": opp.futures_price,
            "basis_pct": opp.basis_pct,
            "annualised_basis_pct": opp.annualised_basis_pct,
            "funding_rate": opp.funding_rate,
            "max_position_usd": self.max_position_usd,
        }

    def tracked_symbols(self) -> list:
        """Return symbols for which both spot and futures data are available."""
        return sorted(set(self._spot.keys()) & set(self._futures.keys()))

    def all_signals(self) -> list:
        """Return valid signals for all tracked symbols."""
        signals = []
        for symbol in self.tracked_symbols():
            sig = self.generate_signal(symbol)
            if sig is not None:
                signals.append(sig)
        return signals

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_basis(
        self,
        basis_pct: float,
        annualised_basis_pct: float,
        funding_rate: float,
    ) -> tuple:
        """
        Determine the action and confidence for a given basis.

        LONG_BASIS : futures at premium (positive basis) → sell futures /
                     buy spot to capture decay.
        SHORT_BASIS: futures at discount (negative basis) → buy futures /
                     sell spot.
        NEUTRAL    : insufficient basis to trade.

        Returns
        -------
        (action, confidence) : (str, float)
        """
        abs_annual = abs(annualised_basis_pct)

        if abs_annual < self.min_annual_basis_pct:
            return "NEUTRAL", 0.0

        if abs_annual > self.MAX_ANNUAL_BASIS_PCT:
            # Data quality guard — treat as neutral
            return "NEUTRAL", 0.0

        # Confidence scales linearly to 1.0 at _CONFIDENCE_SCALE_PCT
        confidence = min(1.0, abs_annual / _CONFIDENCE_SCALE_PCT)

        # LONG_BASIS when futures at premium (basis > 0 or positive funding)
        if basis_pct > 0 or funding_rate > 0:
            action = "LONG_BASIS"
        else:
            action = "SHORT_BASIS"

        return action, confidence
