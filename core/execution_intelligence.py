"""
ExecutionIntelligence — Decides HOW to execute each order.

Selects order type (market / limit / twap), urgency, limit offset, and
number of slices based on system health, volatility, latency, anomaly
detection, and timing advisory.

Decision matrix
---------------
health ≥ floor, LOW vol, slippage OK, latency OK        → LIMIT at mid
health ≥ floor, moderate conditions                     → LIMIT + small offset
health < floor, OR ELEVATED vol, OR anomaly detected    → MARKET urgency=1.0
timing_intelligence == DEFER                            → TWAP 3 slices
timing_intelligence == BLOCK                            → MARKET (forced, mark reason)

Output: advisory["execution_intelligence"]
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExecutionDirective:
    order_type: str          # "market" | "limit" | "twap"
    urgency: float           # 0.0–1.0 (1.0 = fill immediately)
    limit_offset_bps: float  # bps from mid for limit orders (0 = at mid)
    num_slices: int          # TWAP slices (1 = single fill)
    reason: str
    side: str                # "buy" | "sell" | "unknown"
    size_usd: float
    ts: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# ExecutionIntelligence
# ---------------------------------------------------------------------------

class ExecutionIntelligence:
    """
    Decides HOW to execute each order.

    Parameters
    ----------
    prefer_limit : bool
        Prefer limit orders when conditions allow (default True).
    health_limit_floor : int
        Minimum health score to consider limit orders (default 60).
    vol_market_threshold : float
        Volatility (as fraction, e.g. 0.0025) above which we switch to
        market orders (default 0.0025).
    high_latency_ms : float
        p99 latency above this (ms) triggers DEFER/TWAP (default 500).
    anomaly_market_threshold : float
        anomaly severity above this triggers MARKET (default 0.6).
    config : optional config object
    """

    def __init__(
        self,
        prefer_limit: bool = True,
        health_limit_floor: int = 60,
        vol_market_threshold: float = 0.0025,
        high_latency_ms: float = 500.0,
        anomaly_market_threshold: float = 0.6,
        config: Optional[Any] = None,
    ) -> None:
        self.prefer_limit             = bool(prefer_limit)
        self.health_limit_floor       = int(health_limit_floor)
        self.vol_market_threshold     = float(vol_market_threshold)
        self.high_latency_ms          = float(high_latency_ms)
        self.anomaly_market_threshold = float(anomaly_market_threshold)
        self.config = config
        self._last: Optional[ExecutionDirective] = None

    # ── Public API ─────────────────────────────────────────────────────────

    def evaluate(
        self,
        advisory: Dict[str, Any],
        side: str = "buy",
        size_usd: float = 0.0,
    ) -> ExecutionDirective:
        """
        Evaluate execution approach for the given order.

        Parameters
        ----------
        advisory : full advisory dict from current cycle
        side     : "buy" or "sell"
        size_usd : notional USD size of the order
        """
        side_str = str(side).lower()

        # ── Read advisory signals ─────────────────────────────────────────
        _hs = advisory.get("health_score") or {}
        health = int(_hs.get("score", 70) or 70)

        # Volatility
        _vf = advisory.get("vol_forecasts") or {}
        vol = float(_vf.get("predicted_vol", 0.0) or 0.0)

        # Adaptive slippage
        _as = advisory.get("adaptive_slippage") or {}
        predicted_slippage_bps = float(_as.get("predicted_bps", 5.0) or 5.0)

        # Latency
        _lat = advisory.get("latency") or {}
        p99_ms = float(_lat.get("p99_ms", 0.0) or 0.0)

        # Market anomaly
        _ma = advisory.get("market_anomaly") or {}
        anomaly_severity = float(_ma.get("severity", 0.0) or 0.0)

        # Timing intelligence
        _ti = advisory.get("timing_intelligence") or {}
        timing_action = str(_ti.get("action", "ok")).lower()

        # Intelligence directive (from orchestrator)
        _id = advisory.get("intelligence_directive") or {}
        exec_mode_str = str(_id.get("execution_mode", "normal")).lower()

        # ── Decision logic ────────────────────────────────────────────────

        # Force MARKET conditions
        high_vol      = vol > self.vol_market_threshold
        high_anomaly  = anomaly_severity > self.anomaly_market_threshold
        health_too_low = health < self.health_limit_floor
        timing_block  = timing_action == "block"
        exec_aggressive = exec_mode_str == "aggressive"

        # Force TWAP / DEFER conditions
        timing_defer = timing_action == "defer"
        high_latency = p99_ms > self.high_latency_ms

        # Choose order type
        if high_anomaly or timing_block:
            order_type        = "market"
            urgency           = 1.0
            limit_offset_bps  = 0.0
            num_slices        = 1
            reason = (
                f"anomaly={anomaly_severity:.2f} > {self.anomaly_market_threshold}"
                if high_anomaly
                else "timing_block"
            )

        elif timing_defer or (high_latency and not exec_aggressive):
            order_type        = "twap"
            urgency           = 0.3
            limit_offset_bps  = 0.0
            num_slices        = 3
            reason = (
                f"timing_defer"
                if timing_defer
                else f"latency_high p99={p99_ms:.0f}ms"
            )

        elif not self.prefer_limit or high_vol or health_too_low or exec_aggressive:
            order_type        = "market"
            urgency           = 0.8 if exec_aggressive else 1.0
            limit_offset_bps  = 0.0
            num_slices        = 1
            reason = (
                "exec_mode_aggressive" if exec_aggressive
                else f"high_vol={vol:.4f}" if high_vol
                else f"health_low={health}"
            )

        else:
            # Healthy + low vol + limit preferred
            base_offset = 0.0
            if predicted_slippage_bps > 8.0 or exec_mode_str == "patient":
                base_offset = 2.0  # limit 2bps from mid
            elif predicted_slippage_bps > 15.0:
                base_offset = 5.0

            # SELL limits slightly above mid, BUY limits slightly below
            offset = base_offset if side_str == "buy" else -base_offset
            order_type        = "limit"
            urgency           = 0.4
            limit_offset_bps  = offset
            num_slices        = 1
            reason = (
                f"health={health}, vol={vol:.4f}, slip={predicted_slippage_bps:.1f}bps"
            )

        directive = ExecutionDirective(
            order_type       = order_type,
            urgency          = urgency,
            limit_offset_bps = limit_offset_bps,
            num_slices       = num_slices,
            reason           = reason,
            side             = side_str,
            size_usd         = float(size_usd),
        )
        self._last = directive
        return directive

    def snapshot(self) -> Dict[str, Any]:
        d = self._last
        if d is None:
            return {
                "order_type": "market",
                "urgency": 1.0,
                "limit_offset_bps": 0.0,
                "num_slices": 1,
                "reason": "no_evaluation_yet",
            }
        return {
            "order_type":       d.order_type,
            "urgency":          d.urgency,
            "limit_offset_bps": d.limit_offset_bps,
            "num_slices":       d.num_slices,
            "reason":           d.reason,
            "side":             d.side,
            "size_usd":         d.size_usd,
            "ts":               d.ts,
        }
