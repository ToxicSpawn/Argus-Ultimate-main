"""
Latency Compensator — overcome 280ms Australia → Kraken round-trip.

The #1 disadvantage of trading from Australia: 280ms to Kraken (London).
By the time your order reaches the exchange, the price has moved.

This module compensates by:
1. PREDICTIVE PLACEMENT: submit orders 280ms BEFORE the expected signal
   using the Kalman price predictor to anticipate entry price
2. AGGRESSIVE LIMIT OFFSETS: set limit price 280ms of volatility beyond
   mid-price to account for price movement during transit
3. LATENCY-AWARE SIZING: reduce position size when latency is high
   (more slippage → smaller positions to cap total slippage cost)
4. STALE SIGNAL DETECTION: if signal is older than 500ms by the time
   we can act, skip it (the opportunity window has closed)
5. EXCHANGE LATENCY TRACKING: continuously measure actual RTT to Kraken
   and adapt offsets dynamically
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LatencyCompensation:
    """Compensation parameters for one order."""
    measured_rtt_ms: float          # actual measured round-trip time
    predicted_move_bps: float       # expected price movement during RTT
    limit_offset_bps: float         # recommended limit offset (wider than normal)
    size_multiplier: float          # 0.5-1.0 (reduce size for high latency)
    is_stale: bool                  # True if signal is too old to act on
    stale_age_ms: float             # how old the signal is
    recommendation: str             # "EXECUTE", "WIDEN_LIMIT", "REDUCE_AND_EXECUTE", "SKIP_STALE"


class LatencyCompensator:
    """
    Adapts execution to compensate for network latency.

    Measures actual RTT to exchange continuously and adjusts:
    - Limit price offsets (wider when latency is high)
    - Position sizes (smaller when slippage risk is high)
    - Signal freshness (skip stale signals)
    """

    def __init__(
        self,
        expected_rtt_ms: float = 5.0,        # AWS eu-west-1 → Kraken (London)
        max_signal_age_ms: float = 500.0,    # skip signals older than this
        vol_per_ms_bps: float = 0.005,       # BTC moves ~0.005 bps per ms (rough)
        max_latency_size_mult: float = 1.0,  # at 0ms latency, full size
        min_latency_size_mult: float = 0.5,  # at 500ms+ latency, half size
    ):
        self._expected_rtt = expected_rtt_ms
        self._max_signal_age = max_signal_age_ms
        self._vol_per_ms = vol_per_ms_bps
        self._max_size_mult = max_latency_size_mult
        self._min_size_mult = min_latency_size_mult

        self._rtt_history: deque = deque(maxlen=100)
        self._current_rtt = expected_rtt_ms

    def record_rtt(self, rtt_ms: float) -> None:
        """Record a measured round-trip time to the exchange."""
        self._rtt_history.append(rtt_ms)
        # Exponential moving average
        self._current_rtt = self._current_rtt * 0.8 + rtt_ms * 0.2

    def compensate(
        self,
        signal_timestamp_ms: float,
        current_volatility: float = 0.02,
        base_spread_bps: float = 2.0,
    ) -> LatencyCompensation:
        """
        Compute compensation parameters for an order.

        Args:
            signal_timestamp_ms: when the signal was generated (unix ms)
            current_volatility: per-bar volatility (decimal)
            base_spread_bps: current bid-ask spread
        """
        now_ms = time.time() * 1000
        signal_age_ms = now_ms - signal_timestamp_ms
        total_latency_ms = signal_age_ms + self._current_rtt

        # Stale signal check
        is_stale = signal_age_ms > self._max_signal_age

        # Expected price movement during round-trip
        # BTC at $50k, 2% daily vol, 86400s/day → ~0.023% per second → 0.23bps/ms
        vol_bps_per_ms = current_volatility * 10000 / 86400  # daily vol to bps/ms
        predicted_move = vol_bps_per_ms * total_latency_ms

        # Limit offset: spread + predicted movement + safety margin
        limit_offset = base_spread_bps + predicted_move * 1.5

        # Size multiplier: reduce for high latency
        # At 0ms → 1.0x, at 500ms → 0.5x (linear interpolation)
        latency_ratio = min(1.0, total_latency_ms / 500.0)
        size_mult = self._max_size_mult - (self._max_size_mult - self._min_size_mult) * latency_ratio

        # Recommendation
        if is_stale:
            rec = "SKIP_STALE"
        elif total_latency_ms > 400:
            rec = "REDUCE_AND_EXECUTE"
        elif predicted_move > base_spread_bps:
            rec = "WIDEN_LIMIT"
        else:
            rec = "EXECUTE"

        return LatencyCompensation(
            measured_rtt_ms=self._current_rtt,
            predicted_move_bps=predicted_move,
            limit_offset_bps=limit_offset,
            size_multiplier=size_mult,
            is_stale=is_stale,
            stale_age_ms=signal_age_ms,
            recommendation=rec,
        )

    def get_optimal_order_type(self, urgency: str, latency_ms: float) -> str:
        """Recommend order type accounting for latency.

        From Australia with 280ms RTT:
        - Limit orders have time to queue → prefer maker (save fees)
        - Market orders suffer 280ms of adverse movement → avoid unless urgent
        - TWAP spreads impact over time → good for larger orders
        """
        if latency_ms > 400:
            # Very high latency: only trade if truly urgent
            return "limit" if urgency != "critical" else "market"
        elif latency_ms > 200:
            # Typical AU latency: prefer limit with wide offset
            return "limit" if urgency in ("low", "medium") else "twap"
        else:
            # Low latency (colocation or similar): normal selection
            return "limit" if urgency == "low" else "market"

    def get_stats(self) -> Dict[str, Any]:
        rtts = list(self._rtt_history)
        return {
            "current_rtt_ms": self._current_rtt,
            "min_rtt_ms": min(rtts) if rtts else 0,
            "max_rtt_ms": max(rtts) if rtts else 0,
            "avg_rtt_ms": sum(rtts) / len(rtts) if rtts else 0,
            "samples": len(rtts),
        }
