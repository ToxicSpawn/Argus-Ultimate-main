"""
fee_tier_optimizer.py
=====================
Tracks per-exchange trading volume and optimises strategy to unlock better
fee tiers (lower taker fees, higher maker rebates) across MEXC, BTC Markets,
Bybit, and WOO X.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FeeTier:
    """Describes one fee tier on a given exchange."""

    exchange: str
    tier_name: str
    maker_fee: float       # Negative = rebate
    taker_fee: float
    min_volume_usd: float  # Minimum 30-day (or monthly) volume in USD to qualify
    is_rebate: bool        # True when maker_fee < 0


@dataclass
class TradeRecord:
    """A single trade recorded for volume tracking."""

    exchange: str
    symbol: str
    side: str           # "buy" or "sell"
    size_usd: float
    fee_usd: float
    timestamp_ns: int   # Unix timestamp in nanoseconds


# ---------------------------------------------------------------------------
# Fee tier tables
# ---------------------------------------------------------------------------

# MEXC fee tiers (monthly volume in USD)
_MEXC_TIERS: List[FeeTier] = [
    FeeTier("MEXC", "Lv0", 0.0000,  0.0005, 0.0,          is_rebate=False),
    FeeTier("MEXC", "Lv1", 0.0000,  0.0004, 1_000_000.0,  is_rebate=False),
    FeeTier("MEXC", "Lv2", -0.0001, 0.0002, 5_000_000.0,  is_rebate=True),
    FeeTier("MEXC", "Lv3", -0.0002, 0.0001, 10_000_000.0, is_rebate=True),
]

# BTC Markets – always at best tier for makers
_BTCMARKETS_TIERS: List[FeeTier] = [
    FeeTier("BTCMarkets", "Standard", -0.0005, 0.0020, 0.0, is_rebate=True),
]

# Bybit fee tiers (monthly volume in USD)
_BYBIT_TIERS: List[FeeTier] = [
    FeeTier("Bybit", "Lv0",  0.0010, 0.0010, 0.0,          is_rebate=False),
    FeeTier("Bybit", "VIP1", 0.0008, 0.0008, 500_000.0,    is_rebate=False),
    FeeTier("Bybit", "VIP2", 0.0006, 0.0006, 2_000_000.0,  is_rebate=False),
    FeeTier("Bybit", "MM",   0.0000, 0.0005, 0.0,          is_rebate=False),  # MM programme, applied separately
]

# WOO X – always 0% maker for eligible pairs
_WOOX_TIERS: List[FeeTier] = [
    FeeTier("WOOX", "Standard", 0.0000, 0.0005, 0.0, is_rebate=False),
]

_ALL_TIERS: Dict[str, List[FeeTier]] = {
    "MEXC":       _MEXC_TIERS,
    "BTCMarkets": _BTCMARKETS_TIERS,
    "Bybit":      _BYBIT_TIERS,
    "WOOX":       _WOOX_TIERS,
}

# Normalise exchange key lookups
_EXCHANGE_ALIASES: Dict[str, str] = {
    "mexc":       "MEXC",
    "btcmarkets": "BTCMarkets",
    "bybit":      "Bybit",
    "woox":       "WOOX",
    "woo":        "WOOX",
    "woo x":      "WOOX",
}


def _normalise_exchange(exchange: str) -> str:
    return _EXCHANGE_ALIASES.get(exchange.lower(), exchange)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class FeeTierOptimizer:
    """
    Tracks per-exchange 30-day rolling trading volume and recommends actions
    to reach the next fee tier.

    Thread-safety: single-threaded use. For concurrent access, wrap in a lock.
    """

    def __init__(self) -> None:
        # Per-exchange deque of (timestamp_ns, size_usd) tuples
        self._volume_log: Dict[str, Deque[Tuple[int, float]]] = defaultdict(deque)
        # Per-exchange trade records (full)
        self._trades: List[TradeRecord] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_trade(
        self,
        exchange: str,
        symbol: str,
        side: str,
        size_usd: float,
        fee_usd: float,
        timestamp_ns: int,
    ) -> None:
        """
        Record a completed trade for volume tracking.

        Parameters
        ----------
        exchange: Exchange name (case-insensitive)
        symbol: Trading pair, e.g. "BTC/USDT"
        side: "buy" or "sell"
        size_usd: Notional trade value in USD
        fee_usd: Fee paid (negative = rebate received)
        timestamp_ns: Execution timestamp as nanoseconds since Unix epoch
        """
        exc = _normalise_exchange(exchange)
        trade = TradeRecord(
            exchange=exc,
            symbol=symbol,
            side=side,
            size_usd=size_usd,
            fee_usd=fee_usd,
            timestamp_ns=timestamp_ns,
        )
        self._trades.append(trade)
        self._volume_log[exc].append((timestamp_ns, size_usd))
        self._prune_old_volume(exc, timestamp_ns)
        logger.debug(
            "Trade recorded: %s %s %s size_usd=%.2f ts_ns=%d",
            exc, symbol, side, size_usd, timestamp_ns,
        )

    def _prune_old_volume(self, exchange: str, current_ts_ns: int) -> None:
        """Remove volume entries older than 30 days from the deque."""
        cutoff_ns = current_ts_ns - int(30 * 24 * 3600 * 1e9)
        dq = self._volume_log[exchange]
        while dq and dq[0][0] < cutoff_ns:
            dq.popleft()

    # ------------------------------------------------------------------
    # Volume queries
    # ------------------------------------------------------------------

    def get_30d_volume(self, exchange: str) -> float:
        """Return rolling 30-day trading volume in USD for the given exchange."""
        exc = _normalise_exchange(exchange)
        dq = self._volume_log.get(exc)
        if not dq:
            return 0.0
        # Prune relative to the most recent trade
        self._prune_old_volume(exc, dq[-1][0])
        return sum(size for _ts, size in dq)

    # ------------------------------------------------------------------
    # Tier queries
    # ------------------------------------------------------------------

    def get_current_tier(self, exchange: str) -> FeeTier:
        """Return the active FeeTier for the given exchange based on 30d volume."""
        exc = _normalise_exchange(exchange)
        tiers = _ALL_TIERS.get(exc, [])
        if not tiers:
            raise ValueError(f"Unknown exchange: {exchange!r}")

        volume = self.get_30d_volume(exc)
        # Walk from highest tier down; first match wins
        active = tiers[0]
        for tier in tiers:
            if volume >= tier.min_volume_usd:
                active = tier
        return active

    def get_next_tier(self, exchange: str) -> Optional[FeeTier]:
        """
        Return the next achievable FeeTier above the current one.
        Returns None if already at the maximum tier.
        """
        exc = _normalise_exchange(exchange)
        tiers = _ALL_TIERS.get(exc, [])
        if not tiers:
            return None

        current = self.get_current_tier(exc)
        for i, tier in enumerate(tiers):
            if tier.tier_name == current.tier_name and i + 1 < len(tiers):
                return tiers[i + 1]
        return None

    def volume_to_next_tier(self, exchange: str) -> Optional[float]:
        """
        Return USD 30-day volume required to reach the next tier.
        Returns None if already at max tier.
        """
        exc = _normalise_exchange(exchange)
        next_tier = self.get_next_tier(exc)
        if next_tier is None:
            return None
        current_vol = self.get_30d_volume(exc)
        return max(0.0, next_tier.min_volume_usd - current_vol)

    def days_to_next_tier(
        self,
        exchange: str,
        current_daily_vol: float,
    ) -> Optional[float]:
        """
        Estimate calendar days until the next tier is reached.

        Parameters
        ----------
        current_daily_vol: Current average daily trading volume in USD.

        Returns None if already at max tier or daily volume is zero.
        """
        gap = self.volume_to_next_tier(exchange)
        if gap is None:
            return None
        if current_daily_vol <= 0:
            return None
        return gap / current_daily_vol

    # ------------------------------------------------------------------
    # Fee savings
    # ------------------------------------------------------------------

    def get_tier_benefit(self, exchange: str, tier: str) -> float:
        """
        Estimate annual fee savings (in USD) of upgrading to `tier` vs current tier.

        Savings are computed using the 30-day volume extrapolated to 365 days.
        Positive return = fee reduction; negative = cost increase (shouldn't happen).
        """
        exc = _normalise_exchange(exchange)
        tiers = _ALL_TIERS.get(exc, [])
        if not tiers:
            return 0.0

        current = self.get_current_tier(exc)
        target = next((t for t in tiers if t.tier_name == tier), None)
        if target is None:
            raise ValueError(f"Unknown tier {tier!r} for exchange {exc!r}")

        annual_vol = self.get_30d_volume(exc) * (365.0 / 30.0)
        # Assume 50% maker / 50% taker split
        maker_split = 0.5
        taker_split = 0.5

        current_fee = (
            annual_vol * maker_split * current.maker_fee
            + annual_vol * taker_split * current.taker_fee
        )
        target_fee = (
            annual_vol * maker_split * target.maker_fee
            + annual_vol * taker_split * target.taker_fee
        )
        # current_fee - target_fee = money saved (less fee paid / more rebate)
        return round(current_fee - target_fee, 4)

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def get_recommendations(self) -> List[str]:
        """
        Generate actionable recommendations for each exchange.

        Examples
        --------
        "MEXC: $4,500,000 more 30d volume unlocks Lv2 (-0.01% maker rebate) → saves $4,500/yr"
        "BTCMarkets: already at best tier (Standard, -0.05% maker rebate)"
        """
        recs: List[str] = []
        for exc in _ALL_TIERS:
            current = self.get_current_tier(exc)
            next_tier = self.get_next_tier(exc)

            if next_tier is None:
                # Already at max
                rebate_str = (
                    f"{current.maker_fee * 100:.4f}% maker rebate"
                    if current.is_rebate
                    else f"{current.maker_fee * 100:.4f}% maker fee"
                )
                recs.append(
                    f"{exc}: already at best tier ({current.tier_name}, {rebate_str})"
                )
                continue

            gap = self.volume_to_next_tier(exc)
            savings = self.get_tier_benefit(exc, next_tier.tier_name)
            rebate_str = (
                f"{abs(next_tier.maker_fee) * 100:.4f}% maker rebate"
                if next_tier.is_rebate
                else f"{next_tier.maker_fee * 100:.4f}% maker fee"
            )
            recs.append(
                f"{exc}: ${gap:,.0f} more 30d volume unlocks {next_tier.tier_name} "
                f"({rebate_str}) → saves ${savings:,.0f}/yr"
            )

        return recs

    # ------------------------------------------------------------------
    # Stats overview
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Dict]:
        """
        Return per-exchange stats: 30d volume, current tier, next tier, volume needed.
        """
        stats: Dict[str, Dict] = {}
        for exc in _ALL_TIERS:
            current = self.get_current_tier(exc)
            next_tier = self.get_next_tier(exc)
            vol_needed = self.volume_to_next_tier(exc)

            stats[exc] = {
                "30d_volume_usd": round(self.get_30d_volume(exc), 2),
                "current_tier": current.tier_name,
                "current_maker_fee": current.maker_fee,
                "current_taker_fee": current.taker_fee,
                "current_is_rebate": current.is_rebate,
                "next_tier": next_tier.tier_name if next_tier else None,
                "next_tier_maker_fee": next_tier.maker_fee if next_tier else None,
                "volume_to_next_tier_usd": (
                    round(vol_needed, 2) if vol_needed is not None else None
                ),
            }
        return stats

    # ------------------------------------------------------------------
    # Utility: total fees paid per exchange
    # ------------------------------------------------------------------

    def get_total_fees_paid(self, exchange: Optional[str] = None) -> Dict[str, float]:
        """
        Return total fees paid (or rebates received) per exchange.
        Positive = fees paid; negative = net rebate received.
        """
        totals: Dict[str, float] = {}
        for trade in self._trades:
            if exchange and _normalise_exchange(exchange) != trade.exchange:
                continue
            totals[trade.exchange] = totals.get(trade.exchange, 0.0) + trade.fee_usd
        return {k: round(v, 6) for k, v in totals.items()}

    def get_trade_count(self, exchange: Optional[str] = None) -> Dict[str, int]:
        """Return number of trades per exchange."""
        counts: Dict[str, int] = {}
        for trade in self._trades:
            if exchange and _normalise_exchange(exchange) != trade.exchange:
                continue
            counts[trade.exchange] = counts.get(trade.exchange, 0) + 1
        return counts
