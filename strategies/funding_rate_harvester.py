"""
Delta-Neutral Funding Rate Harvesting Strategy.

Captures perpetual futures funding payments by holding equal and opposite
positions on spot and perpetual markets:
  - Long spot BTC on Kraken/Coinbase
  - Short BTC-USDT-SWAP perp on Bybit or OKX

When funding rate is positive (longs pay shorts), the short perp leg
earns funding every 8 hours. Net delta = 0, so price moves cancel out.

Expected returns: 8-100%+ APR in bull markets, near-zero in neutral markets.

Supported perp venues (AU-compatible): Bybit, OKX
NOTE: Binance API is NOT available to Australian residents — use Bybit/OKX.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Funding rate thresholds (per 8-hour period)
DEFAULT_OPEN_THRESHOLD = 0.0005   # 0.05% per 8h = ~54% APR — open harvest
DEFAULT_CLOSE_THRESHOLD = 0.0001  # 0.01% per 8h — close harvest (edge gone)
DEFAULT_STOP_THRESHOLD = -0.0003  # -0.03% — funding turned negative, stop out

# Symbol mappings: spot symbol -> perp symbol per venue
SPOT_TO_PERP: Dict[str, Dict[str, str]] = {
    "BTC/USD": {
        "bybit": "BTC/USDT:USDT",
        "okx": "BTC-USDT-SWAP",
        "dydx": "BTC-USD",
    },
    "ETH/USD": {
        "bybit": "ETH/USDT:USDT",
        "okx": "ETH-USDT-SWAP",
        "dydx": "ETH-USD",
    },
    "SOL/USD": {
        "bybit": "SOL/USDT:USDT",
        "okx": "SOL-USDT-SWAP",
        "dydx": "SOL-USD",
    },
    "XRP/USD": {
        "bybit": "XRP/USDT:USDT",
        "okx": "XRP-USDT-SWAP",
    },
}


class FundingRateHarvester:
    """
    Delta-neutral funding rate harvesting strategy.

    Monitors funding rates across Bybit and OKX, opens/closes
    delta-neutral positions to capture funding payments.
    """

    def __init__(
        self,
        open_threshold: float = DEFAULT_OPEN_THRESHOLD,
        close_threshold: float = DEFAULT_CLOSE_THRESHOLD,
        stop_threshold: float = DEFAULT_STOP_THRESHOLD,
        max_concurrent: int = 3,
        preferred_venues: Optional[List[str]] = None,
    ):
        self.open_threshold = open_threshold
        self.close_threshold = close_threshold
        self.stop_threshold = stop_threshold
        self.max_concurrent = max_concurrent
        self.preferred_venues = preferred_venues or ["bybit", "okx"]

        # Active harvest positions: spot_symbol -> position info
        self._active_harvests: Dict[str, Dict[str, Any]] = {}
        # Cumulative funding received per symbol
        self._cumulative_funding: Dict[str, float] = {}
        # Last known funding rates: symbol -> {venue -> rate}
        self._last_rates: Dict[str, Dict[str, float]] = {}

    def update_funding_rates(self, symbol: str, rates_by_venue: Dict[str, float]) -> None:
        """Update cached funding rates. Call this before analyze()."""
        self._last_rates[symbol] = rates_by_venue

    def _best_venue_and_rate(self, symbol: str) -> Tuple[Optional[str], float]:
        """Return (venue, rate) with highest funding rate for symbol."""
        rates = self._last_rates.get(symbol, {})
        if not rates:
            return None, 0.0
        best_venue = max(rates, key=lambda v: rates[v])
        return best_venue, rates[best_venue]

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyse current market state and generate harvest signal if appropriate.

        Args:
            market_data: {"symbol": str, "price": float,
                          "funding_rates": {venue: rate},
                          "spot_exchange": str}

        Returns:
            Signal dict or None.
        """
        symbol: str = market_data.get("symbol", "")
        price: float = float(market_data.get("price", 0.0))
        funding_rates: Dict[str, float] = market_data.get("funding_rates", {})
        spot_exchange: str = market_data.get("spot_exchange", "kraken")

        if not symbol or price <= 0:
            return None

        # Update cached rates
        if funding_rates:
            self.update_funding_rates(symbol, funding_rates)

        best_venue, best_rate = self._best_venue_and_rate(symbol)
        is_active = symbol in self._active_harvests

        # --- STOP LOSS: funding turned strongly negative ---
        if is_active and best_rate < self.stop_threshold:
            logger.warning(
                "Funding rate STOP for %s: %.4f%% (venue=%s) — closing harvest",
                symbol, best_rate * 100, best_venue,
            )
            self._active_harvests.pop(symbol, None)
            return self._close_signal(symbol, price, best_rate, best_venue or "", "stop_loss")

        # --- CLOSE: funding rate dropped below close threshold ---
        if is_active and best_rate < self.close_threshold:
            logger.info(
                "Funding rate below close threshold for %s: %.4f%% — closing harvest",
                symbol, best_rate * 100,
            )
            self._active_harvests.pop(symbol, None)
            return self._close_signal(symbol, price, best_rate, best_venue or "", "rate_normalized")

        # --- OPEN: funding rate above open threshold, symbol not active ---
        if (
            not is_active
            and best_rate >= self.open_threshold
            and len(self._active_harvests) < self.max_concurrent
            and best_venue in self.preferred_venues
            and symbol in SPOT_TO_PERP
        ):
            perp_symbol = SPOT_TO_PERP[symbol].get(best_venue, "")
            if not perp_symbol:
                return None

            apr_estimate = best_rate * 3 * 365 * 100  # 3 payments/day * 365
            confidence = min(best_rate / (self.open_threshold * 4), 0.95)

            self._active_harvests[symbol] = {
                "opened_at": time.time(),
                "open_price": price,
                "perp_venue": best_venue,
                "perp_symbol": perp_symbol,
                "open_rate": best_rate,
            }
            self._cumulative_funding.setdefault(symbol, 0.0)

            logger.info(
                "HARVEST OPEN %s: rate=%.4f%% (%.1f%% APR est), venue=%s, perp=%s",
                symbol, best_rate * 100, apr_estimate, best_venue, perp_symbol,
            )

            return {
                "symbol": symbol,
                "action": "HARVEST_OPEN",
                "confidence": confidence,
                "price": price,
                "source": "funding_rate_harvester",
                "spot_side": "BUY",
                "spot_exchange": spot_exchange,
                "perp_side": "SELL",
                "perp_symbol": perp_symbol,
                "perp_exchange": best_venue,
                "funding_rate": best_rate,
                "apr_estimate_pct": apr_estimate,
            }

        return None

    def _close_signal(
        self, symbol: str, price: float, rate: float, venue: str, reason: str
    ) -> Dict[str, Any]:
        pos = self._active_harvests.get(symbol, {})
        perp_symbol = pos.get("perp_symbol", SPOT_TO_PERP.get(symbol, {}).get(venue, ""))
        return {
            "symbol": symbol,
            "action": "HARVEST_CLOSE",
            "confidence": 0.9,
            "price": price,
            "source": "funding_rate_harvester",
            "spot_side": "SELL",
            "perp_side": "BUY",
            "perp_symbol": perp_symbol,
            "perp_exchange": venue,
            "funding_rate": rate,
            "close_reason": reason,
            "cumulative_funding": self._cumulative_funding.get(symbol, 0.0),
        }

    def record_funding_payment(self, symbol: str, amount_usd: float) -> None:
        """Record a received funding payment for accounting."""
        self._cumulative_funding[symbol] = self._cumulative_funding.get(symbol, 0.0) + amount_usd
        logger.info("Funding received: %s +$%.4f (total: $%.4f)",
                    symbol, amount_usd, self._cumulative_funding[symbol])

    def get_status(self) -> Dict[str, Any]:
        """Return current harvest status."""
        return {
            "active_harvests": dict(self._active_harvests),
            "cumulative_funding_usd": dict(self._cumulative_funding),
            "total_funding_usd": sum(self._cumulative_funding.values()),
            "last_rates": dict(self._last_rates),
            "n_active": len(self._active_harvests),
            "capacity_remaining": self.max_concurrent - len(self._active_harvests),
        }

    def get_active_symbols(self) -> List[str]:
        return list(self._active_harvests.keys())

    def get_optimal_entry(self) -> Optional[Dict[str, Any]]:
        """
        Find the best funding rate opportunity across all tracked symbols.

        Scans cached rates for all symbols in SPOT_TO_PERP and returns the
        one with the highest rate that meets the open_threshold, is not
        already active, and has capacity.

        Returns:
            Dict with symbol, venue, rate, apr_estimate, perp_symbol or None
        """
        if len(self._active_harvests) >= self.max_concurrent:
            return None

        best: Optional[Dict[str, Any]] = None
        best_rate = 0.0

        for symbol in SPOT_TO_PERP:
            if symbol in self._active_harvests:
                continue

            venue, rate = self._best_venue_and_rate(symbol)
            if venue is None or rate < self.open_threshold:
                continue
            if venue not in self.preferred_venues:
                continue

            if rate > best_rate:
                perp_symbol = SPOT_TO_PERP[symbol].get(venue, "")
                if perp_symbol:
                    apr = rate * 3 * 365 * 100
                    best_rate = rate
                    best = {
                        "symbol": symbol,
                        "venue": venue,
                        "funding_rate": rate,
                        "apr_estimate_pct": apr,
                        "perp_symbol": perp_symbol,
                        "confidence": min(rate / (self.open_threshold * 4), 0.95),
                    }

        return best

    @staticmethod
    def calculate_carry_pnl(position_size: float, funding_rate: float, hours: float) -> float:
        """
        Calculate expected carry P&L from funding rate payments.

        Args:
            position_size: Notional position size in USD
            funding_rate: Per-8-hour funding rate (e.g. 0.0005 = 0.05%)
            hours: Number of hours to project

        Returns:
            Expected P&L in USD (positive = earning funding)
        """
        # Number of funding settlements in the given timeframe
        settlements = hours / 8.0
        return position_size * funding_rate * settlements

    @staticmethod
    def find_best_funding_opportunity(
        rates: Dict[str, Dict[str, float]],
        min_spread_bps: float = 5.0,
        max_position_pct: float = 0.25,
        capital: float = 1000.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Compare funding rates across Bybit, OKX, dYdX and find the best arb.

        The core of funding rate arbitrage: go long on the venue with the
        lowest (most negative) rate and short on the venue with the highest
        (most positive) rate. Delta-neutral across venues.

        Args:
            rates: {symbol: {venue: rate_per_8h}}
                   e.g. {"BTC/USD": {"bybit": 0.0003, "okx": 0.0008, "dydx": -0.0001}}
            min_spread_bps: Minimum spread in basis points to consider (default 5 bps)
            max_position_pct: Max fraction of capital per leg (default 25%)
            capital: Total available capital in USD

        Returns:
            Best opportunity dict or None if no spread exceeds min_spread_bps.
            Dict keys: symbol, long_venue, short_venue, long_rate, short_rate,
                       spread_bps, annualized_apr, recommended_size
        """
        best: Optional[Dict[str, Any]] = None
        best_spread = 0.0

        for symbol, venue_rates in rates.items():
            if len(venue_rates) < 2:
                continue

            # Find min and max rate venues
            min_venue = min(venue_rates, key=lambda v: venue_rates[v])
            max_venue = max(venue_rates, key=lambda v: venue_rates[v])

            if min_venue == max_venue:
                continue

            spread = venue_rates[max_venue] - venue_rates[min_venue]
            spread_bps = spread * 10_000

            if spread_bps < min_spread_bps:
                continue

            if spread > best_spread:
                best_spread = spread
                # 3 settlements/day * 365 days
                annualized_apr = spread * 3 * 365 * 100
                # Recommended size: fraction of capital, scaled by confidence
                confidence = min(spread_bps / 50.0, 1.0)  # Max confidence at 50bps
                recommended_size = capital * max_position_pct * confidence

                best = {
                    "symbol": symbol,
                    "long_venue": min_venue,
                    "short_venue": max_venue,
                    "long_rate": venue_rates[min_venue],
                    "short_rate": venue_rates[max_venue],
                    "spread_bps": round(spread_bps, 2),
                    "annualized_apr": round(annualized_apr, 2),
                    "recommended_size": round(recommended_size, 2),
                    "all_venue_rates": dict(venue_rates),
                }

        if best:
            logger.info(
                "FUNDING ARB: %s long@%s(%.4f%%) short@%s(%.4f%%) spread=%.1fbps APR=%.1f%%",
                best["symbol"], best["long_venue"], best["long_rate"] * 100,
                best["short_venue"], best["short_rate"] * 100,
                best["spread_bps"], best["annualized_apr"],
            )

        return best
