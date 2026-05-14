"""
Smart Fee Optimisation — Exchange Fee Schedule Management.

Hardcoded fee schedules for Kraken, Coinbase, Bybit, OKX, and Binance.
Determines optimal order type (limit vs market) based on spread, urgency,
and fee differentials.

Usage::

    opt = FeeOptimizer()
    tier = opt.get_fee_tier("kraken", volume_30d_usd=500_000)
    use_limit = opt.should_use_limit(spread_bps=3.0, urgency=0.4, fee_saving_bps=tier.maker_bps)
    to_next = opt.get_volume_to_next_tier("kraken", current_volume_usd=500_000)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FeeTier:
    """Fee tier details for an exchange."""

    exchange: str
    tier_name: str
    maker_bps: float              # maker fee in basis points
    taker_bps: float              # taker fee in basis points
    min_volume_usd: float         # minimum 30d volume for this tier
    max_volume_usd: float         # upper bound (inf for top tier)


# ---------------------------------------------------------------------------
# Fee schedules (as of March 2026, sourced from exchange documentation)
# Each entry: (tier_name, maker_bps, taker_bps, min_vol_usd, max_vol_usd)
# ---------------------------------------------------------------------------

_FEE_SCHEDULES: Dict[str, List[Tuple[str, float, float, float, float]]] = {
    "kraken": [
        ("Starter",    16.0, 26.0,              0,        50_000),
        ("Intermediate", 14.0, 24.0,       50_000,       100_000),
        ("Pro",        12.0, 22.0,        100_000,       250_000),
        ("Advanced",   10.0, 20.0,        250_000,       500_000),
        ("Expert",      8.0, 18.0,        500_000,     1_000_000),
        ("Champion",    6.0, 16.0,      1_000_000,     5_000_000),
        ("Master",      4.0, 14.0,      5_000_000,    10_000_000),
        ("Grandmaster", 2.0, 12.0,     10_000_000,    float("inf")),
    ],
    "coinbase": [
        ("Intro",      40.0, 60.0,              0,        10_000),
        ("Level 1",    25.0, 40.0,         10_000,        50_000),
        ("Level 2",    15.0, 25.0,         50_000,       100_000),
        ("Level 3",    10.0, 20.0,        100_000,     1_000_000),
        ("Level 4",     8.0, 18.0,      1_000_000,    15_000_000),
        ("Level 5",     5.0, 10.0,     15_000_000,    75_000_000),
        ("Level 6",     0.0,  5.0,     75_000_000,   float("inf")),
    ],
    "bybit": [
        ("Regular",    10.0, 10.0,              0,       100_000),
        ("VIP 1",       6.0,  8.0,        100_000,       250_000),
        ("VIP 2",       4.0,  6.0,        250_000,       500_000),
        ("VIP 3",       2.0,  5.0,        500_000,     1_000_000),
        ("VIP 4",       1.0,  4.0,      1_000_000,     2_000_000),
        ("VIP 5",       0.0,  3.0,      2_000_000,    float("inf")),
    ],
    "okx": [
        ("Regular",     8.0, 10.0,              0,       100_000),
        ("VIP 1",       6.0,  8.0,        100_000,       500_000),
        ("VIP 2",       5.0,  7.0,        500_000,     1_000_000),
        ("VIP 3",       3.0,  6.0,      1_000_000,     5_000_000),
        ("VIP 4",       2.0,  5.0,      5_000_000,    10_000_000),
        ("VIP 5",       0.0,  4.0,     10_000_000,    float("inf")),
    ],
    "binance": [
        ("Regular",    10.0, 10.0,              0,     1_000_000),
        ("VIP 1",       9.0,  9.0,      1_000_000,     5_000_000),
        ("VIP 2",       8.0,  8.0,      5_000_000,    10_000_000),
        ("VIP 3",       7.0,  7.0,     10_000_000,    25_000_000),
        ("VIP 4",       5.0,  5.5,     25_000_000,    50_000_000),
        ("VIP 5",       4.0,  4.5,     50_000_000,   100_000_000),
        ("VIP 6",       3.0,  4.0,    100_000_000,   250_000_000),
        ("VIP 7",       2.0,  3.0,    250_000_000,   500_000_000),
        ("VIP 8",       1.1,  2.3,    500_000_000,  1_000_000_000),
        ("VIP 9",       0.0,  2.0,  1_000_000_000,   float("inf")),
    ],
}


class FeeOptimizer:
    """Smart fee optimisation across multiple exchanges.

    Provides fee-tier lookup, limit-vs-market decision logic, and
    volume-to-next-tier calculations.

    Parameters
    ----------
    custom_schedules : dict | None
        Additional or overridden fee schedules.  Same format as the
        built-in ``_FEE_SCHEDULES``.
    """

    def __init__(
        self,
        custom_schedules: Optional[Dict[str, List[Tuple[str, float, float, float, float]]]] = None,
    ) -> None:
        self._schedules = dict(_FEE_SCHEDULES)
        if custom_schedules:
            self._schedules.update(custom_schedules)
        logger.info("FeeOptimizer initialised — %d exchanges loaded", len(self._schedules))

    # ------------------------------------------------------------------
    # Fee tier lookup
    # ------------------------------------------------------------------

    def get_fee_tier(
        self,
        exchange: str,
        volume_30d_usd: float,
    ) -> FeeTier:
        """Return the fee tier for *exchange* at *volume_30d_usd*.

        Parameters
        ----------
        exchange : str
            Exchange name (case-insensitive).
        volume_30d_usd : float
            Trailing 30-day trading volume in USD.

        Returns
        -------
        FeeTier

        Raises
        ------
        ValueError
            If the exchange is not in the schedule.
        """
        key = exchange.lower()
        schedule = self._schedules.get(key)
        if schedule is None:
            raise ValueError(f"Unknown exchange: {exchange!r}. "
                             f"Available: {sorted(self._schedules.keys())}")

        vol = max(0.0, volume_30d_usd)
        for tier_name, maker, taker, min_v, max_v in schedule:
            if min_v <= vol < max_v:
                return FeeTier(
                    exchange=key,
                    tier_name=tier_name,
                    maker_bps=maker,
                    taker_bps=taker,
                    min_volume_usd=min_v,
                    max_volume_usd=max_v,
                )

        # Fallback to last tier.
        last = schedule[-1]
        return FeeTier(
            exchange=key,
            tier_name=last[0],
            maker_bps=last[1],
            taker_bps=last[2],
            min_volume_usd=last[3],
            max_volume_usd=last[4],
        )

    # ------------------------------------------------------------------
    # Limit vs market decision
    # ------------------------------------------------------------------

    def should_use_limit(
        self,
        spread_bps: float,
        urgency: float,
        fee_saving_bps: float,
    ) -> bool:
        """Decide whether to use a limit order instead of a market order.

        Uses a simple cost/benefit analysis:

        * **Limit** saves ``fee_saving_bps`` (maker-taker delta) but risks
          non-fill, which costs an estimated ``spread_bps * urgency``.
        * Higher urgency penalises limit orders (risk of not being filled).

        Parameters
        ----------
        spread_bps : float
            Current bid-ask spread in basis points.
        urgency : float
            Urgency score in ``[0, 1]``.  0 = no rush, 1 = immediate fill required.
        fee_saving_bps : float
            Maker-taker fee differential in basis points.

        Returns
        -------
        bool
            ``True`` if a limit order is recommended.
        """
        urgency = max(0.0, min(1.0, urgency))
        # Opportunity cost of missing the fill grows with urgency and spread.
        non_fill_cost = spread_bps * urgency * 2.0
        net_benefit = fee_saving_bps - non_fill_cost

        use_limit = net_benefit > 0
        logger.debug("should_use_limit: spread=%.1f bps urgency=%.2f fee_save=%.1f bps "
                      "non_fill_cost=%.1f net=%.1f -> %s",
                      spread_bps, urgency, fee_saving_bps, non_fill_cost, net_benefit, use_limit)
        return use_limit

    # ------------------------------------------------------------------
    # Volume to next tier
    # ------------------------------------------------------------------

    def get_volume_to_next_tier(
        self,
        exchange: str,
        current_volume_usd: float,
    ) -> float:
        """Return the additional USD volume needed to reach the next fee tier.

        Returns ``0.0`` if the trader is already at the top tier.
        """
        key = exchange.lower()
        schedule = self._schedules.get(key)
        if schedule is None:
            raise ValueError(f"Unknown exchange: {exchange!r}")

        vol = max(0.0, current_volume_usd)
        for tier_name, maker, taker, min_v, max_v in schedule:
            if min_v <= vol < max_v:
                if max_v == float("inf"):
                    return 0.0
                return max_v - vol

        return 0.0

    # ------------------------------------------------------------------
    # Monthly savings estimation
    # ------------------------------------------------------------------

    def estimate_monthly_fee_savings(
        self,
        trades_per_day: float,
        avg_size_usd: float,
        exchange: str = "kraken",
        limit_order_pct: float = 0.5,
    ) -> float:
        """Estimate monthly fee savings from optimal order-type selection.

        Computes the savings (in USD) from placing ``limit_order_pct`` of
        trades as limit orders (paying maker fee) versus all as market
        orders (paying taker fee).

        Parameters
        ----------
        trades_per_day : float
            Average number of trades per day.
        avg_size_usd : float
            Average trade notional in USD.
        exchange : str
            Exchange to use for fee lookup.
        limit_order_pct : float
            Fraction of trades placed as limit orders (0–1).

        Returns
        -------
        float
            Estimated monthly savings in USD.
        """
        monthly_volume = trades_per_day * avg_size_usd * 30
        tier = self.get_fee_tier(exchange, monthly_volume)

        # All-market baseline.
        baseline_cost = monthly_volume * tier.taker_bps / 10_000.0

        # Optimised: mix of maker and taker.
        maker_vol = monthly_volume * limit_order_pct
        taker_vol = monthly_volume * (1.0 - limit_order_pct)
        optimised_cost = (maker_vol * tier.maker_bps + taker_vol * tier.taker_bps) / 10_000.0

        savings = baseline_cost - optimised_cost
        logger.info("Monthly fee savings estimate (%s): vol=$%.0f tier=%s "
                     "baseline=$%.2f optimised=$%.2f savings=$%.2f",
                     exchange, monthly_volume, tier.tier_name,
                     baseline_cost, optimised_cost, savings)
        return savings

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def exchanges(self) -> List[str]:
        """Return available exchange names."""
        return sorted(self._schedules.keys())

    def all_tiers(self, exchange: str) -> List[FeeTier]:
        """Return all tiers for *exchange*."""
        key = exchange.lower()
        schedule = self._schedules.get(key)
        if schedule is None:
            raise ValueError(f"Unknown exchange: {exchange!r}")
        return [
            FeeTier(exchange=key, tier_name=t, maker_bps=m, taker_bps=tk,
                    min_volume_usd=lo, max_volume_usd=hi)
            for t, m, tk, lo, hi in schedule
        ]
