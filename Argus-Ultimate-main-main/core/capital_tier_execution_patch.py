"""
core/capital_tier_execution_patch.py

Wires CapitalTier into the UnifiedExecutionEngine:
  • Slicing thresholds from TIER_CONFIG
  • Min-notional gate
  • Fee gate (blocks trade if estimated fee exceeds tier max_fee_bps)

Monkey-patch at startup (in full_wiring.py or main.py):

    from core.capital_tier_execution_patch import apply_tier_execution_patch
    apply_tier_execution_patch(unified_execution_engine_instance)
"""
from __future__ import annotations

import logging
from typing import Any

from core.capital_tier import classify_tier
from core.tier_config_extension import get_tier_cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patched _pre_order_gate
# ---------------------------------------------------------------------------

def _tier_pre_order_gate(
    self,
    symbol:       str,
    side:         str,
    notional_usd: float,
    est_fee_usd:  float,
) -> tuple[bool, str]:
    """
    Tier-aware pre-order gate.
    Returns (allowed: bool, reason: str).
    """
    equity_aud  = float(getattr(self, "portfolio_value_aud", 0) or 0)
    tier        = classify_tier(equity_aud)
    tcfg        = get_tier_cfg(tier)

    # ── Min-notional gate ─────────────────────────────────────────────
    min_notional = float(tcfg.get("min_slice_usd", 1.0))
    if notional_usd < min_notional:
        reason = (f"[TierGate:{tier.value}] notional ${notional_usd:.2f} "
                  f"< min ${min_notional:.2f}")
        logger.info(reason)
        return False, reason

    # ── Fee gate ──────────────────────────────────────────────────────
    max_fee_bps  = float(tcfg.get("max_fee_bps", 30))
    if notional_usd > 0:
        est_fee_bps = (est_fee_usd / notional_usd) * 10_000
        if est_fee_bps > max_fee_bps:
            reason = (f"[TierGate:{tier.value}] est_fee {est_fee_bps:.1f} bps "
                      f"> max {max_fee_bps} bps — trade blocked")
            logger.warning(reason)
            return False, reason

    # ── Slice count enforcement ───────────────────────────────────────
    max_slice_usd = float(tcfg.get("max_slice_usd", 9999))
    if notional_usd > max_slice_usd:
        # Caller should have already sliced; warn but do not block
        logger.debug(
            "[TierGate:%s] notional $%.2f > max_slice $%.2f — check slicer",
            tier.value, notional_usd, max_slice_usd,
        )

    return True, "ok"


# ---------------------------------------------------------------------------
# Patched slice_order helper
# ---------------------------------------------------------------------------

def _tier_slice_order(
    self,
    symbol:       str,
    side:         str,
    total_usd:    float,
) -> list[float]:
    """
    Split total_usd into tier-appropriate slices.
    Returns list of notional USD per slice.
    """
    equity_aud    = float(getattr(self, "portfolio_value_aud", 0) or 0)
    tier          = classify_tier(equity_aud)
    tcfg          = get_tier_cfg(tier)

    slice_count   = int(tcfg.get("slice_count", 1))
    max_slice_usd = float(tcfg.get("max_slice_usd", total_usd))
    min_slice_usd = float(tcfg.get("min_slice_usd", 1.0))

    if slice_count <= 1 or total_usd <= min_slice_usd:
        return [total_usd]

    base_slice = min(total_usd / slice_count, max_slice_usd)
    slices     = []
    remaining  = total_usd
    while remaining >= min_slice_usd:
        s          = min(base_slice, remaining)
        slices.append(round(s, 8))
        remaining  -= s
        if len(slices) >= slice_count:
            # Absorb remainder into last slice
            if remaining > 0:
                slices[-1] += remaining
            break

    return slices if slices else [total_usd]


# ---------------------------------------------------------------------------
# Apply patch
# ---------------------------------------------------------------------------

def apply_tier_execution_patch(engine: Any) -> None:
    """
    Bind tier-aware methods onto an existing execution engine instance.
    Safe to call multiple times (idempotent).
    """
    import types
    engine._tier_pre_order_gate = types.MethodType(_tier_pre_order_gate, engine)
    engine._tier_slice_order    = types.MethodType(_tier_slice_order,    engine)
    logger.info(
        "[TierExecutionPatch] applied to %s", type(engine).__name__
    )
