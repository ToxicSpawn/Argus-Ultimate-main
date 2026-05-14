"""
Intelligent Execution Engine — world-class order management.

This replaces the hardcoded order type selection with an intelligent
system that chooses the optimal execution strategy for each trade.

Decision matrix:
┌───────────────┬──────────────┬───────────────────────────────┐
│ Fill Prob     │ Edge Urgency │ Execution Strategy            │
├───────────────┼──────────────┼───────────────────────────────┤
│ > 70%         │ Low          │ Limit order (passive maker)   │
│ > 70%         │ High         │ Limit + aggressive reprice    │
│ 40-70%        │ Low          │ TWAP with adaptive slicing    │
│ 40-70%        │ High         │ VWAP with acceleration        │
│ < 40%         │ Low          │ Cancel — conditions too poor  │
│ < 40%         │ High         │ Market order (pay the spread) │
│ Any           │ Toxic > 0.7  │ PAUSE — adverse selection     │
└───────────────┴──────────────┴───────────────────────────────┘

Components:
1. FillProbabilityEstimator — ML model predicting P(fill) from market state
2. AdaptiveSlicer — TWAP/VWAP with feedback-driven slice adjustment
3. ExecutionAlphaTracker — measures execution quality vs benchmarks
4. ToxicityGate — integrates OrderFlowToxicity into execution flow
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Fill Probability Estimator
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FillProbability:
    """Estimated fill probability and recommended action."""
    probability: float          # 0.0 to 1.0
    confidence: float           # how confident we are in the estimate
    recommended_type: str       # "market", "limit", "twap", "vwap", "cancel", "pause"
    limit_offset_bps: float     # recommended limit price offset
    max_slice_pct: float        # max % of book depth per slice
    reason: str


class FillProbabilityEstimator:
    """
    Estimates the probability of an order being filled based on:
    - Order size relative to visible book depth
    - Current spread (wider = harder to fill passively)
    - Recent fill rate for this symbol/venue
    - Volatility (higher vol = more fills but more adverse selection)
    - Time of day (Asian hours = thinner books)
    - Queue position proxy (estimated from order flow)

    Uses k-NN regression from historical fills (no sklearn dependency).
    """

    def __init__(self, history_capacity: int = 500):
        self._history: List[Dict[str, float]] = []
        self._capacity = history_capacity

    def record_fill(
        self,
        symbol: str,
        side: str,
        size_vs_depth: float,
        spread_bps: float,
        volatility: float,
        was_filled: bool,
        fill_time_s: float,
        slippage_bps: float,
    ) -> None:
        """Record a fill attempt for learning."""
        self._history.append({
            "size_vs_depth": size_vs_depth,
            "spread_bps": spread_bps,
            "volatility": volatility,
            "filled": 1.0 if was_filled else 0.0,
            "fill_time_s": fill_time_s,
            "slippage_bps": slippage_bps,
        })
        if len(self._history) > self._capacity:
            self._history.pop(0)

    def estimate(
        self,
        size_usd: float,
        book_depth_usd: float,
        spread_bps: float,
        volatility: float,
        edge_bps: float = 0.0,
        toxicity: float = 0.0,
    ) -> FillProbability:
        """Estimate fill probability and recommend execution strategy."""
        size_vs_depth = size_usd / max(book_depth_usd, 1.0)

        # Base probability from heuristics (when insufficient history)
        if spread_bps < 2:
            base_prob = 0.85 - size_vs_depth * 0.3
        elif spread_bps < 5:
            base_prob = 0.70 - size_vs_depth * 0.4
        elif spread_bps < 10:
            base_prob = 0.50 - size_vs_depth * 0.5
        else:
            base_prob = 0.30 - size_vs_depth * 0.6

        # Volatility boost: higher vol = more fills (but riskier)
        vol_boost = min(0.2, volatility * 5)
        base_prob += vol_boost

        # Toxicity penalty
        if toxicity > 0.7:
            base_prob *= 0.3

        # k-NN correction from history
        if len(self._history) >= 10:
            similar = self._find_similar(size_vs_depth, spread_bps, volatility)
            if similar:
                hist_prob = sum(s["filled"] for s in similar) / len(similar)
                base_prob = base_prob * 0.4 + hist_prob * 0.6  # blend

        prob = max(0.01, min(0.99, base_prob))

        # Decision matrix
        is_urgent = edge_bps > 10
        if toxicity > 0.7:
            rec = "pause"
            reason = f"toxicity={toxicity:.2f} — adverse selection risk"
        elif prob > 0.70 and not is_urgent:
            rec = "limit"
            reason = f"high fill prob ({prob:.0%}) + low urgency"
        elif prob > 0.70 and is_urgent:
            rec = "limit"
            reason = f"high fill prob ({prob:.0%}) + urgent — aggressive reprice"
        elif prob > 0.40 and not is_urgent:
            rec = "twap"
            reason = f"moderate fill prob ({prob:.0%}) — slice over time"
        elif prob > 0.40 and is_urgent:
            rec = "vwap"
            reason = f"moderate fill prob ({prob:.0%}) + urgent — accelerate"
        elif is_urgent:
            rec = "market"
            reason = f"low fill prob ({prob:.0%}) + urgent — pay spread"
        else:
            rec = "cancel"
            reason = f"low fill prob ({prob:.0%}) + no urgency — skip"

        # Limit offset: tighter when prob is high, wider when low
        offset = max(0.5, spread_bps * (1.5 - prob))

        # Max slice: smaller when depth is thin
        max_slice = min(0.10, 0.02 / max(size_vs_depth, 0.01))

        return FillProbability(
            probability=prob, confidence=min(1.0, len(self._history) / 50),
            recommended_type=rec, limit_offset_bps=offset,
            max_slice_pct=max_slice, reason=reason,
        )

    def _find_similar(self, svd: float, spread: float, vol: float, k: int = 7) -> List[Dict]:
        distances = []
        for obs in self._history:
            d = ((obs["size_vs_depth"] - svd) ** 2 * 4
                 + (obs["spread_bps"] - spread) ** 2 * 0.01
                 + (obs["volatility"] - vol) ** 2 * 100) ** 0.5
            distances.append((d, obs))
        distances.sort(key=lambda x: x[0])
        return [obs for _, obs in distances[:k]]

    def get_stats(self) -> Dict[str, Any]:
        if not self._history:
            return {"observations": 0, "avg_fill_rate": 0}
        return {
            "observations": len(self._history),
            "avg_fill_rate": sum(h["filled"] for h in self._history) / len(self._history),
            "avg_slippage_bps": sum(h["slippage_bps"] for h in self._history) / len(self._history),
        }


# ════════════════════════════════════════════════════════════════════════════
# Adaptive Slicer (intelligent TWAP/VWAP)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SliceResult:
    """Result of one slice execution."""
    slice_idx: int
    intended_qty: float
    filled_qty: float
    fill_price: float
    slippage_bps: float
    fill_time_s: float
    market_moved_bps: float     # price change during slice

@dataclass
class AdaptiveSlicePlan:
    """Adaptive slicing plan that adjusts in real-time."""
    total_qty: float
    remaining_qty: float
    slices_completed: int
    slices_planned: int
    current_slice_qty: float
    current_spacing_s: float
    aggression: float           # 0.0 (patient) to 1.0 (aggressive)
    status: str                 # "active", "paused", "completed", "cancelled"


class AdaptiveSlicer:
    """
    Intelligent TWAP/VWAP with feedback-driven slice adjustment.

    After each slice:
    - If fill rate < 80%: increase aggression (wider offset, larger slices)
    - If slippage > threshold: decrease aggression (smaller slices, more spacing)
    - If toxicity spikes: pause execution
    - If volatility increases: reduce slice size (less market impact)
    """

    def __init__(
        self,
        max_aggression: float = 1.0,
        min_spacing_s: float = 2.0,
        max_spacing_s: float = 30.0,
        toxicity_pause_threshold: float = 0.7,
        slippage_threshold_bps: float = 5.0,
    ):
        self._max_aggr = max_aggression
        self._min_spacing = min_spacing_s
        self._max_spacing = max_spacing_s
        self._tox_threshold = toxicity_pause_threshold
        self._slip_threshold = slippage_threshold_bps

    def create_plan(
        self,
        total_qty: float,
        target_duration_s: float,
        initial_aggression: float = 0.5,
        book_depth_qty: float = 0.0,
    ) -> AdaptiveSlicePlan:
        """Create an adaptive slicing plan."""
        # Estimate number of slices
        avg_spacing = (self._min_spacing + self._max_spacing) / 2
        n_slices = max(2, int(target_duration_s / avg_spacing))

        # Initial slice size: limited by book depth
        slice_qty = total_qty / n_slices
        if book_depth_qty > 0:
            max_slice = book_depth_qty * 0.05  # max 5% of visible depth
            slice_qty = min(slice_qty, max_slice)
            n_slices = max(2, int(total_qty / slice_qty))

        return AdaptiveSlicePlan(
            total_qty=total_qty,
            remaining_qty=total_qty,
            slices_completed=0,
            slices_planned=n_slices,
            current_slice_qty=slice_qty,
            current_spacing_s=target_duration_s / n_slices,
            aggression=initial_aggression,
            status="active",
        )

    def adapt(
        self,
        plan: AdaptiveSlicePlan,
        last_result: SliceResult,
        current_toxicity: float = 0.0,
        current_volatility: float = 0.0,
    ) -> AdaptiveSlicePlan:
        """Adapt the plan based on the last slice result."""
        plan.remaining_qty -= last_result.filled_qty
        plan.slices_completed += 1

        if plan.remaining_qty <= 0:
            plan.status = "completed"
            return plan

        # Toxicity check
        if current_toxicity > self._tox_threshold:
            plan.status = "paused"
            plan.aggression = max(0.1, plan.aggression - 0.3)
            logger.info("AdaptiveSlicer: PAUSED — toxicity %.2f", current_toxicity)
            return plan

        plan.status = "active"

        # Fill rate feedback
        fill_rate = last_result.filled_qty / max(last_result.intended_qty, 1e-9)
        if fill_rate < 0.80:
            # Low fill rate → increase aggression
            plan.aggression = min(self._max_aggr, plan.aggression + 0.1)
            plan.current_spacing_s = max(self._min_spacing, plan.current_spacing_s * 0.8)
        elif fill_rate > 0.95 and last_result.slippage_bps < self._slip_threshold * 0.5:
            # Good fills + low slippage → we can be more patient
            plan.aggression = max(0.1, plan.aggression - 0.05)
            plan.current_spacing_s = min(self._max_spacing, plan.current_spacing_s * 1.1)

        # Slippage feedback
        if last_result.slippage_bps > self._slip_threshold:
            # Too much slippage → smaller slices, more spacing
            plan.current_slice_qty *= 0.7
            plan.current_spacing_s = min(self._max_spacing, plan.current_spacing_s * 1.3)
            plan.aggression = max(0.1, plan.aggression - 0.15)

        # Volatility feedback
        if current_volatility > 0.03:  # high vol
            plan.current_slice_qty *= 0.8  # smaller slices in high vol
        elif current_volatility < 0.01:  # low vol
            plan.current_slice_qty *= 1.1  # can afford bigger slices

        # Recalculate remaining slices
        if plan.current_slice_qty > 0:
            plan.slices_planned = plan.slices_completed + max(
                1, int(plan.remaining_qty / plan.current_slice_qty))

        return plan


# ════════════════════════════════════════════════════════════════════════════
# Execution Alpha Tracker
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ExecutionQuality:
    """Execution quality measurement for one trade."""
    order_id: str
    symbol: str
    side: str
    strategy: str
    # Prices
    decision_price: float       # price when signal was generated
    arrival_price: float        # price when order was submitted
    fill_price: float           # actual fill price
    vwap_price: float           # market VWAP during execution
    # Metrics
    implementation_shortfall_bps: float  # fill vs decision price
    arrival_slippage_bps: float         # fill vs arrival price
    vs_vwap_bps: float                  # fill vs market VWAP
    market_impact_bps: float            # price moved because of our order
    spread_cost_bps: float              # half-spread at time of order
    # Quality grade
    grade: str                          # "EXCELLENT", "GOOD", "FAIR", "POOR"


class ExecutionAlphaTracker:
    """
    Measures execution quality vs multiple benchmarks.

    For every fill, computes:
    1. Implementation Shortfall (IS): fill price vs decision price
    2. Arrival slippage: fill price vs arrival price
    3. vs VWAP: fill price vs market VWAP during execution window
    4. Market impact: how much did our order move the price

    Tracks per-strategy, per-symbol, per-venue execution quality.
    Feeds back into slippage model and order type selection.
    """

    def __init__(self):
        self._trades: List[ExecutionQuality] = []
        self._by_strategy: Dict[str, List[ExecutionQuality]] = defaultdict(list)
        self._by_symbol: Dict[str, List[ExecutionQuality]] = defaultdict(list)

    def record(
        self,
        order_id: str,
        symbol: str,
        side: str,
        strategy: str,
        decision_price: float,
        arrival_price: float,
        fill_price: float,
        vwap_price: float = 0.0,
        spread_bps: float = 0.0,
    ) -> ExecutionQuality:
        """Record and measure execution quality."""
        if decision_price <= 0 or fill_price <= 0:
            return ExecutionQuality(
                order_id=order_id, symbol=symbol, side=side, strategy=strategy,
                decision_price=decision_price, arrival_price=arrival_price,
                fill_price=fill_price, vwap_price=vwap_price,
                implementation_shortfall_bps=0, arrival_slippage_bps=0,
                vs_vwap_bps=0, market_impact_bps=0, spread_cost_bps=spread_bps,
                grade="UNKNOWN",
            )

        sign = 1 if side.lower() == "buy" else -1

        # IS: how much worse is fill vs decision (positive = bad for buys)
        is_bps = sign * (fill_price / decision_price - 1) * 10000

        # Arrival slippage
        arr_slip = sign * (fill_price / max(arrival_price, 1e-9) - 1) * 10000 if arrival_price > 0 else 0

        # vs VWAP
        vs_vwap = sign * (fill_price / max(vwap_price, 1e-9) - 1) * 10000 if vwap_price > 0 else 0

        # Market impact (simplified: IS minus spread cost)
        impact = max(0, is_bps - spread_bps / 2)

        # Grade
        total_cost = abs(is_bps)
        if total_cost < 2:
            grade = "EXCELLENT"
        elif total_cost < 5:
            grade = "GOOD"
        elif total_cost < 10:
            grade = "FAIR"
        else:
            grade = "POOR"

        eq = ExecutionQuality(
            order_id=order_id, symbol=symbol, side=side, strategy=strategy,
            decision_price=decision_price, arrival_price=arrival_price,
            fill_price=fill_price, vwap_price=vwap_price,
            implementation_shortfall_bps=is_bps, arrival_slippage_bps=arr_slip,
            vs_vwap_bps=vs_vwap, market_impact_bps=impact,
            spread_cost_bps=spread_bps, grade=grade,
        )

        self._trades.append(eq)
        self._by_strategy[strategy].append(eq)
        self._by_symbol[symbol].append(eq)

        if len(self._trades) > 1000:
            self._trades = self._trades[-1000:]

        return eq

    def get_strategy_quality(self, strategy: str) -> Dict[str, float]:
        """Get average execution quality for a strategy."""
        trades = self._by_strategy.get(strategy, [])[-50:]
        if not trades:
            return {"avg_is_bps": 0, "avg_slippage_bps": 0, "grade_distribution": {}}
        return {
            "avg_is_bps": sum(t.implementation_shortfall_bps for t in trades) / len(trades),
            "avg_slippage_bps": sum(t.arrival_slippage_bps for t in trades) / len(trades),
            "avg_impact_bps": sum(t.market_impact_bps for t in trades) / len(trades),
            "trade_count": len(trades),
            "excellent_pct": sum(1 for t in trades if t.grade == "EXCELLENT") / len(trades),
        }

    def get_overall_quality(self) -> Dict[str, Any]:
        """Get overall execution quality metrics."""
        trades = self._trades[-100:]
        if not trades:
            return {"avg_is_bps": 0, "trades": 0}
        return {
            "avg_is_bps": sum(t.implementation_shortfall_bps for t in trades) / len(trades),
            "avg_slippage_bps": sum(t.arrival_slippage_bps for t in trades) / len(trades),
            "avg_impact_bps": sum(t.market_impact_bps for t in trades) / len(trades),
            "trades": len(trades),
            "excellent_pct": sum(1 for t in trades if t.grade == "EXCELLENT") / len(trades),
            "good_pct": sum(1 for t in trades if t.grade in ("EXCELLENT", "GOOD")) / len(trades),
            "strategies": list(self._by_strategy.keys()),
        }
