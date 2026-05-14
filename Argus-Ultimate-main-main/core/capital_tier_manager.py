"""
Capital Tier Manager — auto-switches ARGUS configuration based on portfolio value.

As capital grows, different strategies and risk parameters become viable.
This module monitors portfolio value and automatically transitions between
tier configurations, ramping changes gradually to avoid sudden behavior shifts.

Tiers:
  micro   ($0 - $5k)      — single venue, conservative, 5 strategies
  small   ($5k - $50k)    — 2 venues, moderate, 10 strategies
  medium  ($50k - $500k)  — 4 venues, full sizing, all strategies, HFT enabled
  large   ($500k - $5M)   — all venues, market making, FPGA optional
  fund    ($5M+)          — institutional setup, multi-asset, prime broker

Promotion is automatic. Demotion requires manual approval (so a temporary
drawdown doesn't reset the system to a smaller config).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CapitalTier(Enum):
    MICRO = "micro"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    FUND = "fund"


@dataclass(frozen=True)
class TierDefinition:
    """Definition of a capital tier — bounds, profile path, characteristics."""
    name: CapitalTier
    min_aud: float
    max_aud: float
    profile_path: str
    description: str
    max_concurrent_positions: int
    max_position_pct: float
    enabled_venues: List[str] = field(default_factory=list)
    enabled_strategies: List[str] = field(default_factory=list)
    enable_hft: bool = False
    enable_fpga: bool = False
    enable_market_making: bool = False
    enable_options: bool = False


# Tier definitions — these are the canonical ARGUS capital tiers
TIERS: Dict[CapitalTier, TierDefinition] = {
    CapitalTier.MICRO: TierDefinition(
        name=CapitalTier.MICRO,
        min_aud=0,
        max_aud=5_000,
        profile_path="config/profiles/tier_micro.yaml",
        description="Conservative single-venue config for $0-$5k",
        max_concurrent_positions=5,
        max_position_pct=0.30,
        enabled_venues=["kraken"],
        enabled_strategies=[
            "momentum", "mean_reversion", "breakout",
            "funding_rate_harvester", "bb_squeeze",
        ],
    ),
    CapitalTier.SMALL: TierDefinition(
        name=CapitalTier.SMALL,
        min_aud=5_000,
        max_aud=50_000,
        profile_path="config/profiles/tier_small.yaml",
        description="Moderate dual-venue config for $5k-$50k",
        max_concurrent_positions=10,
        max_position_pct=0.25,
        enabled_venues=["kraken", "coinbase"],
        enabled_strategies=[
            "momentum", "mean_reversion", "breakout",
            "funding_rate_harvester", "bb_squeeze",
            "market_maker", "cross_exchange_arb",
            "liquidation_cascade", "kalman_pairs",
            "grid_trader",
        ],
        enable_market_making=True,
    ),
    CapitalTier.MEDIUM: TierDefinition(
        name=CapitalTier.MEDIUM,
        min_aud=50_000,
        max_aud=500_000,
        profile_path="config/profiles/tier_medium.yaml",
        description="Full-stack four-venue config for $50k-$500k",
        max_concurrent_positions=20,
        max_position_pct=0.20,
        enabled_venues=["kraken", "coinbase", "bybit", "okx"],
        enabled_strategies=["all"],
        enable_market_making=True,
        enable_hft=True,
    ),
    CapitalTier.LARGE: TierDefinition(
        name=CapitalTier.LARGE,
        min_aud=500_000,
        max_aud=5_000_000,
        profile_path="config/profiles/tier_large.yaml",
        description="Institutional all-venue config for $500k-$5M",
        max_concurrent_positions=40,
        max_position_pct=0.15,
        enabled_venues=["kraken", "coinbase", "bybit", "okx", "deribit", "dydx"],
        enabled_strategies=["all"],
        enable_market_making=True,
        enable_hft=True,
        enable_fpga=True,
        enable_options=True,
    ),
    CapitalTier.FUND: TierDefinition(
        name=CapitalTier.FUND,
        min_aud=5_000_000,
        max_aud=float("inf"),
        profile_path="config/profiles/tier_fund.yaml",
        description="Fund-tier institutional config for $5M+",
        max_concurrent_positions=100,
        max_position_pct=0.10,
        enabled_venues=["all"],
        enabled_strategies=["all"],
        enable_market_making=True,
        enable_hft=True,
        enable_fpga=True,
        enable_options=True,
    ),
}


@dataclass
class TransitionEvent:
    """Record of a tier transition."""
    timestamp: float
    from_tier: CapitalTier
    to_tier: CapitalTier
    trigger_capital_aud: float
    direction: str  # "promote" or "demote"
    applied: bool = False
    reason: str = ""


class CapitalTierManager:
    """
    Monitors portfolio value and auto-promotes ARGUS configuration to
    larger tiers as capital grows. Demotion requires manual approval.

    Usage::

        mgr = CapitalTierManager(starting_capital_aud=1000.0)
        # Each cycle:
        result = mgr.check_and_transition(current_portfolio_value=1500.0)
        if result.transitioned:
            apply_new_config(result.new_tier_definition)
    """

    # Hysteresis: must exceed next-tier minimum by this fraction to promote
    # (prevents oscillation around boundaries)
    PROMOTION_HYSTERESIS = 1.10  # 10% buffer

    # Number of consecutive cycles above threshold required to promote
    PROMOTION_CONFIRMATION_CYCLES = 100

    # Cycles to ramp config changes over (smooth transition)
    RAMP_CYCLES = 360  # 1 hour at 10s/cycle

    def __init__(
        self,
        starting_capital_aud: float = 1000.0,
        manual_override_tier: Optional[CapitalTier] = None,
    ) -> None:
        self._starting_capital = starting_capital_aud
        self._current_tier = manual_override_tier or self._compute_tier(starting_capital_aud)
        self._target_tier = self._current_tier
        self._cycles_above_threshold: int = 0
        self._ramp_cycles_remaining: int = 0
        self._ramp_progress: float = 1.0
        self._transitions: List[TransitionEvent] = []
        self._last_capital_seen: float = starting_capital_aud
        self._cycle_count: int = 0

        logger.info(
            "CapitalTierManager: initialized at tier=%s (capital=$%.2f)",
            self._current_tier.value, starting_capital_aud,
        )

    @property
    def current_tier(self) -> CapitalTier:
        return self._current_tier

    @property
    def current_definition(self) -> TierDefinition:
        return TIERS[self._current_tier]

    @property
    def is_ramping(self) -> bool:
        return self._ramp_cycles_remaining > 0

    @property
    def ramp_progress(self) -> float:
        return self._ramp_progress

    @staticmethod
    def _compute_tier(capital_aud: float) -> CapitalTier:
        """Compute which tier a capital value falls into."""
        for tier_enum, defn in TIERS.items():
            if defn.min_aud <= capital_aud < defn.max_aud:
                return tier_enum
        return CapitalTier.FUND  # fallback for very large

    def check_and_transition(
        self,
        current_portfolio_value: float,
    ) -> Dict[str, Any]:
        """
        Check if a tier transition is needed.
        Returns a dict describing what happened (and what config to apply).
        """
        self._cycle_count += 1
        self._last_capital_seen = current_portfolio_value

        natural_tier = self._compute_tier(current_portfolio_value)
        result: Dict[str, Any] = {
            "cycle": self._cycle_count,
            "current_tier": self._current_tier.value,
            "natural_tier": natural_tier.value,
            "capital_aud": current_portfolio_value,
            "transitioned": False,
            "ramping": self.is_ramping,
            "ramp_progress": self._ramp_progress,
        }

        # If currently ramping, advance the ramp
        if self.is_ramping:
            self._advance_ramp()
            result["ramping"] = True
            result["ramp_progress"] = self._ramp_progress
            return result

        # PROMOTION path
        if natural_tier.value != self._current_tier.value and self._is_promotion(natural_tier):
            # Need hysteresis: must exceed next tier by 10% buffer
            next_tier_def = TIERS[natural_tier]
            threshold = next_tier_def.min_aud * self.PROMOTION_HYSTERESIS

            if current_portfolio_value >= threshold:
                self._cycles_above_threshold += 1

                if self._cycles_above_threshold >= self.PROMOTION_CONFIRMATION_CYCLES:
                    self._begin_promotion(natural_tier, current_portfolio_value)
                    result["transitioned"] = True
                    result["new_tier"] = natural_tier.value
                    result["new_definition"] = self._tier_definition_dict(natural_tier)
                    return result
            else:
                self._cycles_above_threshold = 0

        # DEMOTION path — never automatic, just log warning
        elif natural_tier.value != self._current_tier.value and self._is_demotion(natural_tier):
            logger.warning(
                "CapitalTierManager: portfolio dropped to natural tier=%s but current=%s "
                "(demotion requires manual override)",
                natural_tier.value, self._current_tier.value,
            )
            result["demotion_warning"] = True
            self._cycles_above_threshold = 0
        else:
            self._cycles_above_threshold = 0

        return result

    def _is_promotion(self, target_tier: CapitalTier) -> bool:
        order = [
            CapitalTier.MICRO, CapitalTier.SMALL, CapitalTier.MEDIUM,
            CapitalTier.LARGE, CapitalTier.FUND,
        ]
        return order.index(target_tier) > order.index(self._current_tier)

    def _is_demotion(self, target_tier: CapitalTier) -> bool:
        order = [
            CapitalTier.MICRO, CapitalTier.SMALL, CapitalTier.MEDIUM,
            CapitalTier.LARGE, CapitalTier.FUND,
        ]
        return order.index(target_tier) < order.index(self._current_tier)

    def _begin_promotion(
        self,
        new_tier: CapitalTier,
        trigger_capital: float,
    ) -> None:
        """Start the promotion process — begins ramping over RAMP_CYCLES."""
        old_tier = self._current_tier
        event = TransitionEvent(
            timestamp=time.time(),
            from_tier=old_tier,
            to_tier=new_tier,
            trigger_capital_aud=trigger_capital,
            direction="promote",
            applied=True,
            reason=f"capital reached ${trigger_capital:.2f}",
        )
        self._transitions.append(event)

        self._current_tier = new_tier
        self._target_tier = new_tier
        self._cycles_above_threshold = 0
        self._ramp_cycles_remaining = self.RAMP_CYCLES
        self._ramp_progress = 0.0

        logger.info(
            "CapitalTierManager: PROMOTING %s → %s (capital=$%.2f, ramp=%d cycles)",
            old_tier.value, new_tier.value, trigger_capital, self.RAMP_CYCLES,
        )

    def _advance_ramp(self) -> None:
        """Advance the ramp by one cycle."""
        self._ramp_cycles_remaining -= 1
        self._ramp_progress = 1.0 - (self._ramp_cycles_remaining / self.RAMP_CYCLES)
        if self._ramp_cycles_remaining <= 0:
            self._ramp_progress = 1.0
            logger.info(
                "CapitalTierManager: ramp complete — fully on tier %s",
                self._current_tier.value,
            )

    def manual_override(self, target_tier: CapitalTier) -> None:
        """Manually force a tier transition (for demotions or testing)."""
        if target_tier == self._current_tier:
            return
        old = self._current_tier
        self._current_tier = target_tier
        self._target_tier = target_tier
        self._ramp_cycles_remaining = self.RAMP_CYCLES
        self._ramp_progress = 0.0

        self._transitions.append(TransitionEvent(
            timestamp=time.time(),
            from_tier=old,
            to_tier=target_tier,
            trigger_capital_aud=self._last_capital_seen,
            direction="manual",
            applied=True,
            reason="manual override",
        ))
        logger.warning(
            "CapitalTierManager: MANUAL override %s → %s",
            old.value, target_tier.value,
        )

    def _tier_definition_dict(self, tier: CapitalTier) -> Dict[str, Any]:
        defn = TIERS[tier]
        return {
            "name": defn.name.value,
            "min_aud": defn.min_aud,
            "max_aud": defn.max_aud,
            "profile_path": defn.profile_path,
            "description": defn.description,
            "max_concurrent_positions": defn.max_concurrent_positions,
            "max_position_pct": defn.max_position_pct,
            "enabled_venues": defn.enabled_venues,
            "enabled_strategies": defn.enabled_strategies,
            "enable_hft": defn.enable_hft,
            "enable_fpga": defn.enable_fpga,
            "enable_market_making": defn.enable_market_making,
            "enable_options": defn.enable_options,
        }

    def snapshot(self) -> Dict[str, Any]:
        """Return current state for advisory dict."""
        return {
            "tier": self._current_tier.value,
            "tier_min_aud": TIERS[self._current_tier].min_aud,
            "tier_max_aud": TIERS[self._current_tier].max_aud,
            "capital_aud": self._last_capital_seen,
            "is_ramping": self.is_ramping,
            "ramp_progress": round(self._ramp_progress, 3),
            "transitions_count": len(self._transitions),
            "cycles_above_threshold": self._cycles_above_threshold,
            "next_tier_threshold": self._next_tier_threshold(),
        }

    def _next_tier_threshold(self) -> Optional[float]:
        """Capital needed for next tier promotion (with hysteresis)."""
        order = [
            CapitalTier.MICRO, CapitalTier.SMALL, CapitalTier.MEDIUM,
            CapitalTier.LARGE, CapitalTier.FUND,
        ]
        idx = order.index(self._current_tier)
        if idx >= len(order) - 1:
            return None
        next_tier = order[idx + 1]
        return TIERS[next_tier].min_aud * self.PROMOTION_HYSTERESIS

    def get_transition_history(self) -> List[Dict[str, Any]]:
        return [
            {
                "timestamp": e.timestamp,
                "from": e.from_tier.value,
                "to": e.to_tier.value,
                "capital": e.trigger_capital_aud,
                "direction": e.direction,
                "reason": e.reason,
            }
            for e in self._transitions
        ]
