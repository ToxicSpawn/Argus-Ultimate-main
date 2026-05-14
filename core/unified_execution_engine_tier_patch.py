"""
core/unified_execution_engine_tier_patch.py

Monkey-patch capital_tier awareness into UnifiedExecutionEngine /
UnifiedTradingSystem at startup.

Apply at boot (e.g. in full_wiring.py or main entrypoint):

    from core.unified_execution_engine_tier_patch import apply_tier_patch
    apply_tier_patch(unified_trading_system_instance)

What this adds
--------------
1. _get_capital_tier()        — resolves current CapitalTier from equity
2. _tier_slicing_gate()       — blocks orders below min notional / fee drag
3. _tier_slice_count()        — returns max_slices for current tier
4. _check_fee_gate()          — returns (allowed: bool, reason: str)
"""
from __future__ import annotations

import logging
from typing import Tuple

from core.capital_tier import CapitalTier, classify_tier
from core.tier_config_patch import get_fee_gate, get_slicing_params

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Patch methods
# ─────────────────────────────────────────────────────────────────────

def _get_capital_tier(self) -> CapitalTier:
    """Return current CapitalTier based on portfolio AUD equity."""
    equity = float(getattr(self, "portfolio_value_aud", 0.0) or 0.0)
    return classify_tier(equity)


def _tier_slicing_gate(
    self,
    notional_aud: float,
    symbol: str = "",
) -> Tuple[bool, str]:
    """
    Return (allowed, reason).
    Blocks trade if notional is below tier min_notional_aud.
    """
    tier   = self._get_capital_tier()
    params = get_fee_gate(tier)
    min_n  = params["min_notional_aud"]

    if notional_aud < min_n:
        reason = (
            f"tier_gate: notional {notional_aud:.2f} AUD < "
            f"min {min_n:.2f} AUD for tier {tier.value} [{symbol}]"
        )
        logger.info(reason)
        return False, reason
    return True, ""


def _tier_slice_count(self) -> int:
    """Return the max number of order slices for the current tier."""
    tier = self._get_capital_tier()
    return get_slicing_params(tier)["max_slices"]


def _check_fee_gate(
    self,
    notional_aud: float,
    fee_paid_aud: float,
) -> Tuple[bool, float]:
    """
    Return (passes, fee_drag_bps).
    fee_drag_bps = (fee_paid / notional) * 10_000.
    Blocks if fee_drag_bps > tier max_fee_drag_bps.
    """
    if notional_aud <= 0:
        return False, 0.0

    tier            = self._get_capital_tier()
    params          = get_fee_gate(tier)
    fee_drag_bps    = (fee_paid_aud / notional_aud) * 10_000.0
    max_drag        = params["max_fee_drag_bps"]

    if fee_drag_bps > max_drag:
        logger.info(
            "_check_fee_gate: fee_drag %.1f bps > max %.1f bps for tier %s",
            fee_drag_bps, max_drag, tier.value,
        )
        return False, fee_drag_bps
    return True, fee_drag_bps


# ─────────────────────────────────────────────────────────────────────
# Apply patch
# ─────────────────────────────────────────────────────────────────────

def apply_tier_patch(engine: object) -> None:
    """
    Bind all tier-aware methods onto `engine` instance (or class).
    Safe to call multiple times — idempotent.
    """
    import types

    cls = type(engine)

    if not hasattr(cls, "_get_capital_tier"):
        cls._get_capital_tier   = _get_capital_tier
        cls._tier_slicing_gate  = _tier_slicing_gate
        cls._tier_slice_count   = _tier_slice_count
        cls._check_fee_gate     = _check_fee_gate
        logger.info("apply_tier_patch: tier methods bound to %s", cls.__name__)
    else:
        logger.debug("apply_tier_patch: already patched — skipped")
