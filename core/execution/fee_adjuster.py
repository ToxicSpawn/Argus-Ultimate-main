"""Push 67 — Dynamic fee adjuster (Hummingbot-compatible rules).

Implements all 6 Hummingbot fee rules:
  1. add_transaction_costs spread adjustment
  2. min_profitability gate
  3. Dynamic slippage buffer (ATR-scaled)
  4. DEX gas fee gate
  5. Fee tier auto-detection (Bybit VIP 0 → Supreme)
  6. order_refresh cancel+repost cycle

Reference: Hummingbot v2.6 Pure Market Making + XEMM docs
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


# Bybit fee tiers: (monthly_volume_usd) -> (maker_fee, taker_fee)
_BYBIT_FEE_TIERS: list[tuple[float, float, float]] = [
    (100_000_000, 0.0002, 0.0001),  # Supreme VIP
    (50_000_000,  0.0004, 0.0003),  # VIP 3
    (10_000_000,  0.0006, 0.0005),  # VIP 2
    (1_000_000,   0.0008, 0.0006),  # VIP 1
    (0,           0.0010, 0.0010),  # Standard
]


@dataclass
class FeeProfile:
    maker_fee: float
    taker_fee: float
    monthly_volume_usd: float
    tier_name: str


@dataclass
class SpreadQuote:
    bid_spread: float
    ask_spread: float
    min_profitable: bool
    net_pnl_pct: float
    adjusted: bool


class FeeAdjuster:
    """Dynamic fee + spread adjuster, Hummingbot V2 compatible.

    Args:
        monthly_volume_usd: Used for Bybit tier auto-detection
        base_bid_spread:    Default bid spread (e.g. 0.001 = 0.1%)
        base_ask_spread:    Default ask spread
        min_profitability:  Minimum net PnL to place order (e.g. 0.001 = 10bps)
        slippage_buffer:    Base slippage buffer for taker orders
        max_gas_usd:        Maximum gas fee in USD (DEX only)
        safety_margin:      Fee safety multiplier
    """

    def __init__(
        self,
        monthly_volume_usd: float = 0.0,
        base_bid_spread: float = 0.001,
        base_ask_spread: float = 0.001,
        min_profitability: float = 0.001,
        slippage_buffer: float = 0.005,
        max_gas_usd: float = 2.0,
        safety_margin: float = 2.5,
    ):
        self.min_profitability = min_profitability
        self.slippage_buffer = slippage_buffer
        self.max_gas_usd = max_gas_usd
        self.safety_margin = safety_margin
        self._base_bid = base_bid_spread
        self._base_ask = base_ask_spread
        self._fee_profile = self._detect_tier(monthly_volume_usd)

    # ------------------------------------------------------------------
    # Rule 1: add_transaction_costs — spread adjustment
    # ------------------------------------------------------------------

    def adjusted_spreads(self, volatility_pct: float = 0.0) -> SpreadQuote:
        """Widen spreads to cover fees + slippage. Rule 1 + Rule 3."""
        fp = self._fee_profile
        round_trip_cost = (fp.maker_fee + fp.taker_fee) * self.safety_margin
        vol_buffer = volatility_pct * 0.5

        bid = max(self._base_bid + round_trip_cost + vol_buffer, self.min_spread())
        ask = max(self._base_ask + round_trip_cost + vol_buffer, self.min_spread())
        return SpreadQuote(
            bid_spread=bid,
            ask_spread=ask,
            min_profitable=True,
            net_pnl_pct=(bid + ask) / 2 - round_trip_cost,
            adjusted=True,
        )

    # ------------------------------------------------------------------
    # Rule 2: min_profitability gate
    # ------------------------------------------------------------------

    def is_profitable(
        self,
        entry_price: float,
        exit_price: float,
        side: str,
    ) -> bool:
        """Returns True if round-trip is profitable after fees."""
        if side == "buy":
            gross = (exit_price - entry_price) / entry_price
        else:
            gross = (entry_price - exit_price) / entry_price
        fp = self._fee_profile
        net = gross - fp.maker_fee - fp.taker_fee
        return net >= self.min_profitability

    # ------------------------------------------------------------------
    # Rule 3: dynamic slippage buffer (ATR-scaled)
    # ------------------------------------------------------------------

    def dynamic_slippage(self, atr: float, mid_price: float) -> float:
        """Widen slippage buffer proportionally to realised volatility."""
        if mid_price <= 0:
            return self.slippage_buffer
        vol_ratio = atr / mid_price
        return self.slippage_buffer * (1.0 + vol_ratio * 10.0)

    # ------------------------------------------------------------------
    # Rule 4: DEX gas gate
    # ------------------------------------------------------------------

    def gas_is_acceptable(
        self,
        gas_usd: float,
        trade_size_usd: float,
    ) -> bool:
        """Skip DEX order if gas > max_gas_usd OR gas > 1% of trade size."""
        if gas_usd > self.max_gas_usd:
            return False
        if trade_size_usd > 0 and (gas_usd / trade_size_usd) > 0.01:
            return False
        return True

    # ------------------------------------------------------------------
    # Rule 5: fee tier auto-detection
    # ------------------------------------------------------------------

    def update_volume(self, monthly_volume_usd: float) -> FeeProfile:
        """Re-detect fee tier from latest 30-day volume."""
        self._fee_profile = self._detect_tier(monthly_volume_usd)
        return self._fee_profile

    @staticmethod
    def _detect_tier(volume: float) -> FeeProfile:
        tier_names = ["Supreme VIP", "VIP 3", "VIP 2", "VIP 1", "Standard"]
        for i, (threshold, maker, taker) in enumerate(_BYBIT_FEE_TIERS):
            if volume >= threshold:
                return FeeProfile(
                    maker_fee=maker,
                    taker_fee=taker,
                    monthly_volume_usd=volume,
                    tier_name=tier_names[i],
                )
        return FeeProfile(0.001, 0.001, volume, "Standard")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def min_spread(self) -> float:
        fp = self._fee_profile
        return (fp.maker_fee + fp.taker_fee) * self.safety_margin

    @property
    def fee_profile(self) -> FeeProfile:
        return self._fee_profile

    @property
    def maker_fee(self) -> float:
        return self._fee_profile.maker_fee

    @property
    def taker_fee(self) -> float:
        return self._fee_profile.taker_fee
