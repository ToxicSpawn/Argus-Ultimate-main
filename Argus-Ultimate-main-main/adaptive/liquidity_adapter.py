"""
Real-time liquidity adaptation for execution.

Provides orderbook analysis, liquidity classification, execution adjustment,
and time-of-day optimization to adapt trading behavior to current liquidity conditions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LiquidityMetrics:
    """Snapshot of current liquidity conditions."""
    bid_ask_spread: float = 0.0
    order_book_depth: float = 0.0  # total volume within 1%
    volume_24h: float = 0.0
    volume_profile: Dict[str, float] = field(default_factory=dict)  # hourly volumes
    impact_cost: float = 0.0  # estimated cost for standard order
    liquidity_score: float = 0.0  # 0-100


@dataclass
class LiquiditySnapshot:
    """Historical liquidity snapshot for analysis."""
    timestamp: float
    metrics: LiquidityMetrics
    regime: LiquidityRegime
    hour: int


@dataclass
class AdaptedOrder:
    """Order adjusted for current liquidity conditions."""
    original_size: float
    adjusted_size: float
    num_slices: int
    slice_sizes: List[float]
    time_between_slices: float
    urgency_adjustment: float
    spread_adjustment: float
    reason: str


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LiquidityRegime(str, Enum):
    """Liquidity regime classification."""
    HIGH_LIQUIDITY = "high_liquidity"
    NORMAL_LIQUIDITY = "normal_liquidity"
    LOW_LIQUIDITY = "low_liquidity"
    ILLIQUID = "illiquid"
    STRESSED = "stressed"


# ---------------------------------------------------------------------------
# OrderBookAnalyzer
# ---------------------------------------------------------------------------

class OrderBookAnalyzer:
    """Analyzes orderbook data to extract liquidity metrics."""

    @staticmethod
    def compute_spread(orderbook: Dict) -> float:
        """Compute bid-ask spread in basis points."""
        if not orderbook:
            return 0.0
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        if not bids or not asks:
            return 0.0
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        if best_bid <= 0 or best_ask <= 0:
            return 0.0
        mid_price = (best_bid + best_ask) / 2.0
        spread_bps = ((best_ask - best_bid) / mid_price) * 10000.0
        return float(spread_bps)

    @staticmethod
    def compute_depth(orderbook: Dict, depth_pct: float = 0.01) -> float:
        """Compute total volume within depth_pct of mid price."""
        if not orderbook:
            return 0.0
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        if not bids and not asks:
            return 0.0
        best_bid = float(bids[0][0]) if bids else 0.0
        best_ask = float(asks[0][0]) if asks else 0.0
        if best_bid <= 0 or best_ask <= 0:
            return 0.0
        mid_price = (best_bid + best_ask) / 2.0
        lower_bound = mid_price * (1.0 - depth_pct)
        upper_bound = mid_price * (1.0 + depth_pct)
        bid_depth = sum(
            float(b[1]) for b in bids if float(b[0]) >= lower_bound
        )
        ask_depth = sum(
            float(a[1]) for a in asks if float(a[0]) <= upper_bound
        )
        return float(bid_depth + ask_depth)

    @staticmethod
    def compute_imbalance(orderbook: Dict) -> float:
        """Compute orderbook imbalance ratio [-1, 1].

        Positive values indicate bid pressure, negative indicate ask pressure.
        """
        if not orderbook:
            return 0.0
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        if not bids and not asks:
            return 0.0
        bid_volume = sum(float(b[1]) for b in bids)
        ask_volume = sum(float(a[1]) for a in asks)
        total = bid_volume + ask_volume
        if total <= 0:
            return 0.0
        imbalance = (bid_volume - ask_volume) / total
        return float(np.clip(imbalance, -1.0, 1.0))

    @staticmethod
    def estimate_impact(orderbook: Dict, order_size: float) -> float:
        """Estimate market impact cost in basis points for given order size."""
        if not orderbook or order_size <= 0:
            return 0.0
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        if not bids or not asks:
            return 0.0
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid_price = (best_bid + best_ask) / 2.0
        if mid_price <= 0:
            return 0.0
        # Compute average price after consuming order_size from book
        avg_price = OrderBookAnalyzer._compute_vwap(asks, order_size)
        if avg_price <= 0:
            return 0.0
        impact_bps = ((avg_price - mid_price) / mid_price) * 10000.0
        return float(max(0.0, impact_bps))

    @staticmethod
    def detect_icebergs(orderbook: Dict) -> bool:
        """Detect potential iceberg orders by analyzing size patterns."""
        if not orderbook:
            return False
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        # Check for repeated identical sizes (iceberg signature)
        for side in [bids, asks]:
            if len(side) < 5:
                continue
            sizes = [float(entry[1]) for entry in side[:10]]
            if not sizes:
                continue
            # Look for clusters of similar sizes
            size_counts: Dict[float, int] = {}
            for s in sizes:
                rounded = round(s, 2)
                size_counts[rounded] = size_counts.get(rounded, 0) + 1
            # If any size appears 3+ times in top 10, likely iceberg
            if any(count >= 3 for count in size_counts.values()):
                return True
        return False

    @staticmethod
    def _compute_vwap(levels: List, target_size: float) -> float:
        """Compute volume-weighted average price for consuming target_size."""
        remaining = target_size
        total_cost = 0.0
        total_volume = 0.0
        for price, size in levels:
            p = float(price)
            s = float(size)
            if s <= 0 or p <= 0:
                continue
            fill = min(remaining, s)
            total_cost += fill * p
            total_volume += fill
            remaining -= fill
            if remaining <= 0:
                break
        if total_volume <= 0:
            return 0.0
        return total_cost / total_volume


# ---------------------------------------------------------------------------
# LiquidityClassifier
# ---------------------------------------------------------------------------

class LiquidityClassifier:
    """Classifies liquidity into regimes based on metrics."""

    def classify(self, metrics: LiquidityMetrics) -> LiquidityRegime:
        """Classify current liquidity into a regime."""
        thresholds = self.get_regime_thresholds()
        spread = metrics.bid_ask_spread
        depth = metrics.order_book_depth
        score = metrics.liquidity_score

        # Check for stressed conditions first (unusual patterns)
        if self._is_stressed(metrics):
            return LiquidityRegime.STRESSED

        # Classify based on score and thresholds
        if score >= thresholds[LiquidityRegime.HIGH_LIQUIDITY][0]:
            return LiquidityRegime.HIGH_LIQUIDITY
        elif score >= thresholds[LiquidityRegime.NORMAL_LIQUIDITY][0]:
            return LiquidityRegime.NORMAL_LIQUIDITY
        elif score >= thresholds[LiquidityRegime.LOW_LIQUIDITY][0]:
            return LiquidityRegime.LOW_LIQUIDITY
        elif score >= thresholds[LiquidityRegime.ILLIQUID][0]:
            return LiquidityRegime.ILLIQUID
        else:
            return LiquidityRegime.ILLIQUID

    def get_regime_thresholds(self) -> Dict[LiquidityRegime, Tuple[float, float]]:
        """Return score thresholds for each regime.

        Returns dict of regime -> (min_score, max_score) on 0-100 scale.
        """
        return {
            LiquidityRegime.HIGH_LIQUIDITY: (80.0, 100.0),
            LiquidityRegime.NORMAL_LIQUIDITY: (60.0, 79.9),
            LiquidityRegime.LOW_LIQUIDITY: (30.0, 59.9),
            LiquidityRegime.ILLIQUID: (10.0, 29.9),
            LiquidityRegime.STRESSED: (0.0, 100.0),  # detected separately
        }

    def detect_liquidity_change(self, current: LiquidityMetrics, previous: Optional[LiquidityMetrics]) -> bool:
        """Detect significant liquidity change between snapshots."""
        if previous is None:
            return True
        score_change = abs(current.liquidity_score - previous.liquidity_score)
        spread_change = abs(current.bid_ask_spread - previous.bid_ask_spread)
        depth_change = abs(current.order_book_depth - previous.order_book_depth)

        # Significant change thresholds
        score_threshold = 15.0
        spread_threshold = 5.0  # bps
        depth_threshold = 0.3  # 30% change

        prev_depth = previous.order_book_depth
        depth_pct_change = abs(depth_change / prev_depth) if prev_depth > 0 else 0.0

        return (
            score_change >= score_threshold
            or spread_change >= spread_threshold
            or depth_pct_change >= depth_threshold
        )

    @staticmethod
    def _is_stressed(metrics: LiquidityMetrics) -> bool:
        """Detect stressed market conditions."""
        # Wide spread + low depth = stress
        if metrics.bid_ask_spread > 50.0 and metrics.order_book_depth < 100.0:
            return True
        # Extreme imbalance
        if metrics.impact_cost > 100.0:  # >100 bps impact
            return True
        return False


# ---------------------------------------------------------------------------
# ExecutionAdjuster
# ---------------------------------------------------------------------------

class ExecutionAdjuster:
    """Adjusts execution parameters based on liquidity conditions."""

    @staticmethod
    def adjust_order_size(base_size: float, liquidity: LiquidityMetrics) -> float:
        """Adjust order size based on liquidity score."""
        score = float(np.clip(liquidity.liquidity_score, 0.0, 100.0))
        if score >= 80.0:
            return base_size
        elif score >= 60.0:
            return base_size * 0.85
        elif score >= 40.0:
            return base_size * 0.65
        elif score >= 20.0:
            return base_size * 0.40
        else:
            return base_size * 0.20

    @staticmethod
    def adjust_urgency(base_urgency: float, liquidity: LiquidityMetrics) -> float:
        """Adjust execution urgency [0, 1] based on liquidity.

        Higher urgency in high liquidity, lower in low liquidity.
        """
        score = float(np.clip(liquidity.liquidity_score, 0.0, 100.0))
        urgency_scalar = score / 100.0
        return float(np.clip(base_urgency * urgency_scalar, 0.0, 1.0))

    @staticmethod
    def adjust_spread(base_spread: float, liquidity: LiquidityMetrics) -> float:
        """Adjust limit order spread based on liquidity.

        Wider spreads in low liquidity to avoid adverse selection.
        """
        score = float(np.clip(liquidity.liquidity_score, 0.0, 100.0))
        if score >= 80.0:
            return base_spread
        elif score >= 60.0:
            return base_spread * 1.2
        elif score >= 40.0:
            return base_spread * 1.5
        elif score >= 20.0:
            return base_spread * 2.0
        else:
            return base_spread * 3.0

    @staticmethod
    def should_split_order(order_size: float, liquidity: LiquidityMetrics) -> bool:
        """Determine if order should be split into slices."""
        score = liquidity.liquidity_score
        depth = liquidity.order_book_depth
        if depth <= 0:
            return True
        # Split if order is >10% of book depth or liquidity is low
        if order_size > depth * 0.10:
            return True
        if score < 50.0:
            return True
        return False

    @staticmethod
    def compute_optimal_slices(order_size: float, liquidity: LiquidityMetrics) -> List[float]:
        """Compute optimal slice sizes for order execution."""
        score = float(np.clip(liquidity.liquidity_score, 0.0, 100.0))
        depth = liquidity.order_book_depth

        # Determine number of slices based on liquidity
        if score >= 80.0:
            num_slices = max(2, int(np.ceil(order_size / (depth * 0.25))))
            num_slices = min(num_slices, 4)
        elif score >= 60.0:
            num_slices = max(3, int(np.ceil(order_size / (depth * 0.15))))
            num_slices = min(num_slices, 6)
        elif score >= 40.0:
            num_slices = max(4, int(np.ceil(order_size / (depth * 0.10))))
            num_slices = min(num_slices, 8)
        elif score >= 20.0:
            num_slices = max(5, int(np.ceil(order_size / (depth * 0.05))))
            num_slices = min(num_slices, 12)
        else:
            num_slices = max(6, int(np.ceil(order_size / (depth * 0.03))))
            num_slices = min(num_slices, 20)

        num_slices = max(2, min(num_slices, 20))

        # Compute slice sizes (front-loaded for urgency)
        if num_slices <= 2:
            slices = [order_size / num_slices] * num_slices
        else:
            # Exponential decay: larger first slices
            weights = np.exp(-np.linspace(0, 2, num_slices))
            weights = weights / weights.sum()
            slices = (weights * order_size).tolist()

        return [float(s) for s in slices]


# ---------------------------------------------------------------------------
# TimeOfDayAdjuster
# ---------------------------------------------------------------------------

class TimeOfDayAdjuster:
    """Adjusts for time-of-day liquidity patterns."""

    # Typical crypto liquidity curve (UTC-based, 0-23 hours)
    # Peak during US/Europe overlap (13-16 UTC), lowest during Asian lunch
    _LIQUIDITY_CURVE: Dict[int, float] = {
        0: 0.65,   1: 0.55,   2: 0.50,   3: 0.45,   4: 0.48,   5: 0.52,
        6: 0.60,   7: 0.70,   8: 0.78,   9: 0.82,   10: 0.85,  11: 0.88,
        12: 0.90,  13: 0.95,  14: 0.98,  15: 1.00,  16: 0.97,  17: 0.92,
        18: 0.88,  19: 0.85,  20: 0.82,  21: 0.78,  22: 0.72,  23: 0.68,
    }

    def get_liquidity_multiplier(self, hour: int) -> float:
        """Get liquidity multiplier for given hour (0-23)."""
        hour = int(np.clip(hour, 0, 23))
        return self._LIQUIDITY_CURVE.get(hour, 0.75)

    def is_optimal_trading_window(self, hour: int) -> bool:
        """Check if hour falls within optimal trading window."""
        return self.get_liquidity_multiplier(hour) >= 0.85

    def get_liquidity_curve(self) -> Dict[int, float]:
        """Return full 24-hour liquidity curve."""
        return dict(self._LIQUIDITY_CURVE)


# ---------------------------------------------------------------------------
# LiquidityAdapter
# ---------------------------------------------------------------------------

class LiquidityAdapter:
    """Main liquidity adaptation engine.

    Coordinates orderbook analysis, regime classification, execution adjustment,
    and time-of-day optimization to adapt trading to current liquidity.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.orderbook_analyzer = OrderBookAnalyzer()
        self.classifier = LiquidityClassifier()
        self.execution_adjuster = ExecutionAdjuster()
        self.time_adjuster = TimeOfDayAdjuster()

        self._current_metrics: Optional[LiquidityMetrics] = None
        self._previous_metrics: Optional[LiquidityMetrics] = None
        self._current_regime: LiquidityRegime = LiquidityRegime.NORMAL_LIQUIDITY
        self._history: List[LiquiditySnapshot] = []
        self._max_history: int = int(self.config.get("max_history", 1000))

    def update(self, orderbook: Dict, volume_data: Optional[Dict] = None) -> LiquidityMetrics:
        """Update liquidity metrics from orderbook and volume data.

        Args:
            orderbook: Orderbook dict with 'bids' and 'asks' lists.
            volume_data: Optional dict with 'volume_24h', 'volume_profile' (hourly).

        Returns:
            Updated LiquidityMetrics.
        """
        spread = self.orderbook_analyzer.compute_spread(orderbook)
        depth = self.orderbook_analyzer.compute_depth(orderbook)
        impact_cost = self.orderbook_analyzer.estimate_impact(orderbook, 1000.0)  # standard order

        volume_24h = 0.0
        volume_profile: Dict[str, float] = {}
        if volume_data:
            volume_24h = float(volume_data.get("volume_24h", 0.0))
            volume_profile = {
                str(k): float(v) for k, v in volume_data.get("volume_profile", {}).items()
            }

        # Compute composite liquidity score (0-100)
        liquidity_score = self._compute_liquidity_score(spread, depth, impact_cost, volume_24h)

        self._previous_metrics = self._current_metrics
        self._current_metrics = LiquidityMetrics(
            bid_ask_spread=spread,
            order_book_depth=depth,
            volume_24h=volume_24h,
            volume_profile=volume_profile,
            impact_cost=impact_cost,
            liquidity_score=liquidity_score,
        )

        # Update regime
        self._current_regime = self.classifier.classify(self._current_metrics)

        # Record snapshot
        snapshot = LiquiditySnapshot(
            timestamp=time.time(),
            metrics=self._current_metrics,
            regime=self._current_regime,
            hour=datetime.now().hour,
        )
        self._history.append(snapshot)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        logger.debug(
            "Liquidity updated: spread=%.1f bps depth=%.0f score=%.1f regime=%s",
            spread, depth, liquidity_score, self._current_regime.value,
        )

        return self._current_metrics

    def get_current_liquidity(self) -> Optional[LiquidityMetrics]:
        """Return current liquidity metrics."""
        return self._current_metrics

    def get_liquidity_regime(self) -> LiquidityRegime:
        """Return current liquidity regime."""
        return self._current_regime

    def adapt_order(self, order: Dict, market_conditions: Optional[Dict] = None) -> AdaptedOrder:
        """Adapt an order for current liquidity conditions.

        Args:
            order: Order dict with at least 'size'.
            market_conditions: Optional dict with 'hour' for time-of-day adjustment.

        Returns:
            AdaptedOrder with adjusted parameters.
        """
        if self._current_metrics is None:
            return AdaptedOrder(
                original_size=float(order.get("size", 0)),
                adjusted_size=float(order.get("size", 0)),
                num_slices=1,
                slice_sizes=[float(order.get("size", 0))],
                time_between_slices=0.0,
                urgency_adjustment=1.0,
                spread_adjustment=1.0,
                reason="no_liquidity_data",
            )

        original_size = float(order.get("size", 0))
        base_urgency = float(order.get("urgency", 0.5))
        base_spread = float(order.get("spread", 0.01))

        # Apply execution adjustments
        adjusted_size = self.execution_adjuster.adjust_order_size(original_size, self._current_metrics)
        urgency_adj = self.execution_adjuster.adjust_urgency(base_urgency, self._current_metrics)
        spread_adj = self.execution_adjuster.adjust_spread(base_spread, self._current_metrics)

        # Apply time-of-day adjustment
        hour = 12
        if market_conditions:
            hour = int(market_conditions.get("hour", 12))
        time_multiplier = self.time_adjuster.get_liquidity_multiplier(hour)
        adjusted_size *= time_multiplier

        # Determine slicing
        should_split = self.execution_adjuster.should_split_order(adjusted_size, self._current_metrics)
        if should_split:
            slice_sizes = self.execution_adjuster.compute_optimal_slices(adjusted_size, self._current_metrics)
            num_slices = len(slice_sizes)
            # Time between slices increases in low liquidity
            base_interval = 5.0  # seconds
            time_between_slices = base_interval * (100.0 / max(self._current_metrics.liquidity_score, 1.0))
        else:
            slice_sizes = [adjusted_size]
            num_slices = 1
            time_between_slices = 0.0

        reason = self._build_adaptation_reason()

        return AdaptedOrder(
            original_size=original_size,
            adjusted_size=adjusted_size,
            num_slices=num_slices,
            slice_sizes=slice_sizes,
            time_between_slices=time_between_slices,
            urgency_adjustment=urgency_adj,
            spread_adjustment=spread_adj,
            reason=reason,
        )

    def get_liquidity_history(self, limit: Optional[int] = None) -> List[LiquiditySnapshot]:
        """Return liquidity history, optionally limited to last N entries."""
        if limit is not None:
            return self._history[-limit:]
        return list(self._history)

    def _compute_liquidity_score(
        self,
        spread: float,
        depth: float,
        impact_cost: float,
        volume_24h: float,
    ) -> float:
        """Compute composite liquidity score (0-100).

        Higher score = better liquidity.
        """
        # Spread component (0-30 points): tighter spread = better
        if spread <= 1.0:
            spread_score = 30.0
        elif spread <= 5.0:
            spread_score = 25.0
        elif spread <= 10.0:
            spread_score = 20.0
        elif spread <= 25.0:
            spread_score = 10.0
        else:
            spread_score = 0.0

        # Depth component (0-30 points): deeper book = better
        if depth >= 100000.0:
            depth_score = 30.0
        elif depth >= 50000.0:
            depth_score = 25.0
        elif depth >= 10000.0:
            depth_score = 20.0
        elif depth >= 1000.0:
            depth_score = 10.0
        else:
            depth_score = 0.0

        # Impact cost component (0-20 points): lower impact = better
        if impact_cost <= 1.0:
            impact_score = 20.0
        elif impact_cost <= 5.0:
            impact_score = 15.0
        elif impact_cost <= 10.0:
            impact_score = 10.0
        elif impact_cost <= 25.0:
            impact_score = 5.0
        else:
            impact_score = 0.0

        # Volume component (0-20 points): higher volume = better
        if volume_24h >= 100000000.0:  # $100M
            volume_score = 20.0
        elif volume_24h >= 10000000.0:  # $10M
            volume_score = 15.0
        elif volume_24h >= 1000000.0:  # $1M
            volume_score = 10.0
        elif volume_24h >= 100000.0:  # $100K
            volume_score = 5.0
        else:
            volume_score = 0.0

        total = spread_score + depth_score + impact_score + volume_score
        return float(np.clip(total, 0.0, 100.0))

    def _build_adaptation_reason(self) -> str:
        """Build human-readable reason for order adaptation."""
        if self._current_metrics is None:
            return "no_liquidity_data"
        regime = self._current_regime.value
        score = self._current_metrics.liquidity_score
        spread = self._current_metrics.bid_ask_spread
        depth = self._current_metrics.order_book_depth
        return f"regime={regime} score={score:.1f} spread={spread:.1f}bps depth={depth:.0f}"
