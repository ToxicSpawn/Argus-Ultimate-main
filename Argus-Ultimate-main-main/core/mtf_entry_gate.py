"""MTF Entry Gate — blocks counter-trend entries when higher timeframes disagree.

Uses the existing MTFResult from MultiTimeframeFeatures to gate entries:
- Only allows LONG entries when 4h bias is bullish (or neutral with soft mode)
- Only allows SHORT entries when 4h bias is bearish (or neutral with soft mode)
- Returns a GateResult with allow/block decision + reason string for audit log

Two modes:
  strict: block if 4h disagrees regardless of other TFs
  soft:   block only if BOTH 4h AND 1h disagree (more trades, less filtering)

Small-capital enhancements (2026-04):
  - Minimum R:R gate (default 2.0) — filters low-quality scalps
  - Volume surge filter — requires current vol > surge_multiplier * 20-period SMA
  - Session filter — restricts scalp entries to London/NY overlap (13-17 UTC)
  - RSI overbought/oversold block — no longs above rsi_ob, no shorts below rsi_os

Integrate by calling gate.check() before submitting any order:
    gate = MTFEntryGate(mode="strict")
    result = gate.check(action="BUY", mtf=mtf_result)
    if not result.allow:
        logger.info("Entry blocked: %s", result.reason)
        return
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Bias threshold above which a TF is considered directional (not neutral)
BIAS_THRESHOLD = 0.15

# London/NY overlap session window (UTC hours, inclusive)
SESSION_START_UTC = 13
SESSION_END_UTC = 17


@dataclass(frozen=True)
class GateResult:
    allow:      bool
    reason:     str
    action:     str
    bias_4h:    float
    bias_1h:    float
    agg_bias:   float
    mode:       str


class MTFEntryGate:
    """
    Higher-timeframe entry filter with small-capital quality gates.

    Parameters
    ----------
    mode : "strict" (default) or "soft"
        strict — block if 4h bias opposes entry direction
        soft   — block only if both 4h AND 1h oppose entry direction
    bias_threshold : float
        Minimum absolute bias to consider a TF directional (default 0.15)
    min_rr : float
        Minimum required reward-to-risk ratio (default 2.0 for small capital)
    require_volume_surge : bool
        If True, require current volume > surge_multiplier * vol_sma20
    surge_multiplier : float
        Volume surge threshold multiplier (default 1.5x SMA)
    session_filter : bool
        If True, only allow entries during London/NY overlap (13-17 UTC)
    rsi_overbought : float
        Block long entries if RSI >= this level (default 70)
    rsi_oversold : float
        Block short entries if RSI <= this level (default 30)
    """

    def __init__(
        self,
        mode: str = "strict",
        bias_threshold: float = BIAS_THRESHOLD,
        min_rr: float = 2.0,
        require_volume_surge: bool = False,
        surge_multiplier: float = 1.5,
        session_filter: bool = False,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
    ) -> None:
        if mode not in ("strict", "soft"):
            raise ValueError(f"MTFEntryGate: mode must be 'strict' or 'soft', got '{mode}'")
        self._mode = mode
        self._threshold = bias_threshold
        self._min_rr = min_rr
        self._require_volume_surge = require_volume_surge
        self._surge_multiplier = surge_multiplier
        self._session_filter = session_filter
        self._rsi_ob = rsi_overbought
        self._rsi_os = rsi_oversold
        self._total_checked = 0
        self._total_blocked = 0

    def check(
        self,
        action: str,                          # "BUY" or "SELL"
        mtf,                                  # MTFResult from MultiTimeframeFeatures
        override: bool = False,               # True = bypass gate (e.g. forced close)
        rr_ratio: Optional[float] = None,     # expected reward:risk for this trade
        current_volume: Optional[float] = None,
        volume_sma20: Optional[float] = None,
        rsi: Optional[float] = None,
        timestamp: Optional[datetime] = None, # UTC datetime for session filter
    ) -> GateResult:
        """
        Decide whether to allow this entry.

        Returns GateResult with allow=True/False and reason string.
        """
        self._total_checked += 1

        if override:
            return GateResult(
                allow=True, reason="override", action=action,
                bias_4h=0.0, bias_1h=0.0, agg_bias=0.0, mode=self._mode,
            )

        bias_4h  = mtf.timeframe_biases.get("4h",  0.0)
        bias_1h  = mtf.timeframe_biases.get("1h",  0.0)
        agg_bias = mtf.aggregate_bias
        is_long  = action.upper() in ("BUY", "LONG")

        # --- Session filter (small-capital: London/NY overlap only) ---
        if self._session_filter:
            now_utc = timestamp or datetime.now(timezone.utc)
            hour = now_utc.hour
            if not (SESSION_START_UTC <= hour < SESSION_END_UTC):
                self._total_blocked += 1
                reason = (
                    f"Blocked {action}: outside session window "
                    f"({SESSION_START_UTC}-{SESSION_END_UTC} UTC), current hour={hour}"
                )
                logger.info("MTFEntryGate: %s", reason)
                return GateResult(
                    allow=False, reason=reason, action=action,
                    bias_4h=bias_4h, bias_1h=bias_1h, agg_bias=agg_bias,
                    mode=self._mode,
                )

        # --- Minimum R:R gate (small-capital: default 2.0) ---
        if rr_ratio is not None and rr_ratio < self._min_rr:
            self._total_blocked += 1
            reason = (
                f"Blocked {action}: R:R {rr_ratio:.2f} < minimum {self._min_rr:.2f}"
            )
            logger.info("MTFEntryGate: %s", reason)
            return GateResult(
                allow=False, reason=reason, action=action,
                bias_4h=bias_4h, bias_1h=bias_1h, agg_bias=agg_bias,
                mode=self._mode,
            )

        # --- Volume surge filter ---
        if self._require_volume_surge and current_volume is not None and volume_sma20 is not None:
            if volume_sma20 > 0 and current_volume < self._surge_multiplier * volume_sma20:
                self._total_blocked += 1
                ratio = current_volume / volume_sma20 if volume_sma20 > 0 else 0
                reason = (
                    f"Blocked {action}: volume surge insufficient "
                    f"(current={current_volume:.0f}, sma20={volume_sma20:.0f}, "
                    f"ratio={ratio:.2f}x < {self._surge_multiplier}x)"
                )
                logger.info("MTFEntryGate: %s", reason)
                return GateResult(
                    allow=False, reason=reason, action=action,
                    bias_4h=bias_4h, bias_1h=bias_1h, agg_bias=agg_bias,
                    mode=self._mode,
                )

        # --- RSI overbought/oversold block ---
        if rsi is not None:
            if is_long and rsi >= self._rsi_ob:
                self._total_blocked += 1
                reason = f"Blocked {action}: RSI {rsi:.1f} >= overbought {self._rsi_ob}"
                logger.info("MTFEntryGate: %s", reason)
                return GateResult(
                    allow=False, reason=reason, action=action,
                    bias_4h=bias_4h, bias_1h=bias_1h, agg_bias=agg_bias,
                    mode=self._mode,
                )
            if not is_long and rsi <= self._rsi_os:
                self._total_blocked += 1
                reason = f"Blocked {action}: RSI {rsi:.1f} <= oversold {self._rsi_os}"
                logger.info("MTFEntryGate: %s", reason)
                return GateResult(
                    allow=False, reason=reason, action=action,
                    bias_4h=bias_4h, bias_1h=bias_1h, agg_bias=agg_bias,
                    mode=self._mode,
                )

        # --- MTF bias gate (original logic) ---
        opposes_4h = (
            bias_4h < -self._threshold if is_long
            else bias_4h > self._threshold
        )
        opposes_1h = (
            bias_1h < -self._threshold if is_long
            else bias_1h > self._threshold
        )

        if self._mode == "strict":
            blocked = opposes_4h
            reason_tf = "4h"
        else:  # soft
            blocked = opposes_4h and opposes_1h
            reason_tf = "4h+1h"

        if blocked:
            self._total_blocked += 1
            reason = (
                f"Blocked {action}: {reason_tf} opposes entry "
                f"(4h={bias_4h:+.3f}, 1h={bias_1h:+.3f}, agg={agg_bias:+.3f})"
            )
            logger.info("MTFEntryGate: %s", reason)
            return GateResult(
                allow=False, reason=reason, action=action,
                bias_4h=bias_4h, bias_1h=bias_1h, agg_bias=agg_bias,
                mode=self._mode,
            )

        reason = (
            f"Allowed {action}: all gates passed "
            f"(4h={bias_4h:+.3f}, 1h={bias_1h:+.3f}, agg={agg_bias:+.3f})"
        )
        return GateResult(
            allow=True, reason=reason, action=action,
            bias_4h=bias_4h, bias_1h=bias_1h, agg_bias=agg_bias,
            mode=self._mode,
        )

    def get_stats(self) -> dict:
        block_rate = (
            self._total_blocked / self._total_checked
            if self._total_checked > 0 else 0.0
        )
        return {
            "mode":                self._mode,
            "bias_threshold":      self._threshold,
            "min_rr":              self._min_rr,
            "require_volume_surge": self._require_volume_surge,
            "surge_multiplier":    self._surge_multiplier,
            "session_filter":      self._session_filter,
            "rsi_overbought":      self._rsi_ob,
            "rsi_oversold":        self._rsi_os,
            "total_checked":       self._total_checked,
            "total_blocked":       self._total_blocked,
            "block_rate":          round(block_rate, 4),
        }
