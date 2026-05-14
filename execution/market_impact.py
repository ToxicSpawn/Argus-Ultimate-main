"""
Market Impact Model — Almgren-Chriss square-root model for realistic slippage estimation.

Used by the backtester, position sizer, and algo executor to estimate true
execution cost before and after trades.

Models:
  1. Square-root impact: impact_bps = eta * sigma * sqrt(Q / ADV) * 10000
  2. Linear temporary impact: temp_impact = gamma * (Q / ADV)
  3. Fixed costs: spread + exchange fee

Reference: Almgren & Chriss (2000), "Optimal execution of portfolio transactions"
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DEFAULT_ETA: float = 0.1          # permanent impact coefficient
DEFAULT_GAMMA: float = 0.314      # temporary impact coefficient
DEFAULT_SPREAD_BPS: float = 5.0   # half-spread in bps
DEFAULT_FEE_TAKER_BPS: float = 6.0  # Kraken taker fee (0.06%)
DEFAULT_FEE_MAKER_BPS: float = 2.0  # Kraken maker fee (0.02%)
DEFAULT_ADV_USD: float = 50_000_000.0  # default 24h ADV if unknown (BTC/USD approx)

# ---------------------------------------------------------------------------
# Exchange-specific impact profiles
# ---------------------------------------------------------------------------

EXCHANGE_PROFILES: Dict[str, Dict] = {
    "kraken": {
        "spread_bps": 4.0,
        "fee_taker_bps": 6.0,
        "fee_maker_bps": 2.0,
        "eta": 0.08,
    },
    "bybit": {
        "spread_bps": 2.0,
        "fee_taker_bps": 5.5,
        "fee_maker_bps": 1.0,
        "eta": 0.06,
    },
    "okx": {
        "spread_bps": 2.0,
        "fee_taker_bps": 5.0,
        "fee_maker_bps": 2.0,
        "eta": 0.06,
    },
    "coinbase": {
        "spread_bps": 5.0,
        "fee_taker_bps": 6.0,
        "fee_maker_bps": 4.0,
        "eta": 0.10,
    },
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ImpactEstimate:
    """Complete cost breakdown for a single execution estimate."""

    symbol: str
    side: str                     # "buy" or "sell"
    quantity_usd: float
    spread_bps: float
    fee_bps: float
    temporary_impact_bps: float
    permanent_impact_bps: float
    total_impact_bps: float
    total_cost_usd: float
    effective_price_adj: float    # price delta from all costs; positive = worse for buyer
    is_maker: bool


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

class MarketImpactModel:
    """
    Estimates realistic execution costs using Almgren-Chriss square-root impact.

    Parameters
    ----------
    eta : float
        Permanent impact coefficient. Higher = more price footprint.
    gamma : float
        Temporary impact coefficient. Higher = more immediate slippage.
    spread_bps : float
        Half-spread of the instrument (exchange-level bid-ask half-spread).
    adv_usd : float
        Average daily volume in USD. Used to normalize trade size.
    sigma_daily : float
        Daily volatility of the asset (e.g., 0.03 for 3% daily vol).
    fee_taker_bps : float
        Taker fee in basis points.
    fee_maker_bps : float
        Maker fee in basis points.
    """

    def __init__(
        self,
        eta: float = DEFAULT_ETA,
        gamma: float = DEFAULT_GAMMA,
        spread_bps: float = DEFAULT_SPREAD_BPS,
        adv_usd: float = DEFAULT_ADV_USD,
        sigma_daily: float = 0.03,
        fee_taker_bps: float = DEFAULT_FEE_TAKER_BPS,
        fee_maker_bps: float = DEFAULT_FEE_MAKER_BPS,
    ) -> None:
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.spread_bps = float(spread_bps)
        self.adv_usd = max(1_000.0, float(adv_usd))
        self.sigma_daily = max(0.001, float(sigma_daily))
        self.fee_taker_bps = float(fee_taker_bps)
        self.fee_maker_bps = float(fee_maker_bps)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        symbol: str,
        side: str,
        quantity_usd: float,
        price: float,
        is_maker: bool = False,
        adv_override: Optional[float] = None,
        sigma_override: Optional[float] = None,
    ) -> ImpactEstimate:
        """
        Estimate total execution cost for a given order.

        Formulas
        --------
        participation  = quantity_usd / adv_usd
        sqrt_part      = sqrt(participation)

        temporary_impact_bps  = gamma * sigma_daily * sqrt_part * 10_000
        permanent_impact_bps  = eta   * sigma_daily * sqrt_part * 10_000 * 0.5
        spread_cost_bps       = spread_bps   (half-spread, one-way)
        fee_bps               = fee_maker_bps if is_maker else fee_taker_bps
        total_impact_bps      = spread_cost_bps + fee_bps
                                + temporary_impact_bps + permanent_impact_bps
        total_cost_usd        = quantity_usd * total_impact_bps / 10_000
        effective_price_adj   = price * (total_impact_bps / 10_000)
                                * (1 if side=="buy" else -1)
        """
        adv = max(1_000.0, float(adv_override)) if adv_override is not None else self.adv_usd
        sigma = max(0.001, float(sigma_override)) if sigma_override is not None else self.sigma_daily

        qty = max(0.0, float(quantity_usd))
        p = max(1e-12, float(price))

        participation = qty / adv
        sqrt_part = math.sqrt(participation)

        temporary_impact_bps = self.gamma * sigma * sqrt_part * 10_000.0
        permanent_impact_bps = self.eta * sigma * sqrt_part * 10_000.0 * 0.5
        spread_cost_bps = self.spread_bps
        fee_bps = self.fee_maker_bps if is_maker else self.fee_taker_bps

        total_impact_bps = (
            spread_cost_bps
            + fee_bps
            + temporary_impact_bps
            + permanent_impact_bps
        )

        total_cost_usd = qty * total_impact_bps / 10_000.0
        direction = 1.0 if side.lower() == "buy" else -1.0
        effective_price_adj = p * (total_impact_bps / 10_000.0) * direction

        return ImpactEstimate(
            symbol=symbol,
            side=side.lower(),
            quantity_usd=qty,
            spread_bps=spread_cost_bps,
            fee_bps=fee_bps,
            temporary_impact_bps=temporary_impact_bps,
            permanent_impact_bps=permanent_impact_bps,
            total_impact_bps=total_impact_bps,
            total_cost_usd=total_cost_usd,
            effective_price_adj=effective_price_adj,
            is_maker=is_maker,
        )

    def adjust_price(
        self,
        price: float,
        side: str,
        quantity_usd: float,
        **kwargs,
    ) -> float:
        """Return the expected execution price after all impact costs."""
        est = self.estimate("", side, quantity_usd, price, **kwargs)
        if side.lower() == "buy":
            return price + est.effective_price_adj
        return price - abs(est.effective_price_adj)

    def is_cost_acceptable(
        self,
        quantity_usd: float,
        price: float,
        max_impact_bps: float = 30.0,
        **kwargs,
    ) -> bool:
        """Returns True if estimated total impact is at or below the threshold."""
        est = self.estimate("", "buy", quantity_usd, price, **kwargs)
        return est.total_impact_bps <= max_impact_bps

    def update_adv(self, adv_usd: float) -> None:
        """Update the ADV estimate (call periodically with fresh market data)."""
        self.adv_usd = max(1_000.0, float(adv_usd))

    def update_sigma(self, sigma_daily: float) -> None:
        """Update the daily volatility estimate."""
        self.sigma_daily = max(0.001, float(sigma_daily))

    def __repr__(self) -> str:
        return (
            f"MarketImpactModel(eta={self.eta}, gamma={self.gamma}, "
            f"spread_bps={self.spread_bps}, adv_usd={self.adv_usd:.0f}, "
            f"sigma_daily={self.sigma_daily:.4f})"
        )


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def estimate_fill_price(
    side: str,
    mid_price: float,
    quantity_usd: float,
    adv_usd: float = DEFAULT_ADV_USD,
    sigma_daily: float = 0.03,
    spread_bps: float = DEFAULT_SPREAD_BPS,
) -> float:
    """
    Convenience function: returns estimated fill price including all impact costs.

    Parameters
    ----------
    side : str
        "buy" or "sell"
    mid_price : float
        Current mid-market price of the instrument.
    quantity_usd : float
        Order size in USD notional.
    adv_usd : float
        Average daily volume in USD (default: BTC/USD approximation).
    sigma_daily : float
        Daily volatility (e.g., 0.03 for 3%).
    spread_bps : float
        Half-spread in basis points.

    Returns
    -------
    float
        Expected fill price after spread, fees, and market impact.
    """
    model = MarketImpactModel(
        adv_usd=adv_usd,
        sigma_daily=sigma_daily,
        spread_bps=spread_bps,
    )
    return model.adjust_price(mid_price, side, quantity_usd)


def get_model_for_exchange(exchange_id: str, **overrides) -> MarketImpactModel:
    """
    Create a MarketImpactModel pre-configured with exchange-specific defaults.

    Parameters
    ----------
    exchange_id : str
        One of "kraken", "bybit", "okx", "coinbase".  Falls back to "kraken"
        if the exchange is not in EXCHANGE_PROFILES.
    **overrides
        Any MarketImpactModel constructor keyword arguments that should override
        the profile defaults (e.g., adv_usd=80_000_000, sigma_daily=0.04).

    Returns
    -------
    MarketImpactModel
        Configured model instance.
    """
    profile = EXCHANGE_PROFILES.get(exchange_id.lower(), EXCHANGE_PROFILES["kraken"])
    params: Dict = {**profile, **overrides}
    return MarketImpactModel(**params)
