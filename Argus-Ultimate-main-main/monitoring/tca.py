"""
Transaction Cost Analysis (TCA) — systematic analysis of execution quality
and identification of money leaks.

Records every execution and provides detailed breakdowns of slippage,
venue quality, order type efficiency, and improvement recommendations.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Size buckets for analysis
_SIZE_SMALL = 500.0      # < $500
_SIZE_MEDIUM = 5000.0    # $500 - $5000
# > $5000 = large


@dataclass
class ExecutionRecord:
    """Single execution record for TCA."""
    order_id: str
    symbol: str
    side: str  # "buy" | "sell"
    intended_price: float
    fill_price: float
    quantity: float
    venue: str
    order_type: str  # "market" | "limit" | "vwap" | "twap"
    latency_ms: float
    timestamp: float = field(default_factory=time.time)

    @property
    def slippage_bps(self) -> float:
        """Slippage in basis points. Positive = worse than intended."""
        if self.intended_price == 0:
            return 0.0
        if self.side.lower() == "buy":
            # Buying: fill > intended is bad
            return (self.fill_price - self.intended_price) / self.intended_price * 10_000
        else:
            # Selling: fill < intended is bad
            return (self.intended_price - self.fill_price) / self.intended_price * 10_000

    @property
    def notional_usd(self) -> float:
        return self.fill_price * self.quantity

    @property
    def size_bucket(self) -> str:
        notional = self.notional_usd
        if notional < _SIZE_SMALL:
            return "small"
        elif notional < _SIZE_MEDIUM:
            return "medium"
        else:
            return "large"


class TransactionCostAnalyzer:
    """
    Systematic analysis of where money leaks in execution.

    Records every execution and provides breakdowns by venue, symbol,
    order type, and size. Generates an overall execution quality score.
    """

    def __init__(self, max_records: int = 10000) -> None:
        self._records: List[ExecutionRecord] = []
        self._max_records = max_records

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_execution(
        self,
        order_id: str,
        symbol: str,
        side: str,
        intended_price: float,
        fill_price: float,
        quantity: float,
        venue: str,
        order_type: str = "market",
        latency_ms: float = 0.0,
    ) -> None:
        """Record a single execution for TCA."""
        record = ExecutionRecord(
            order_id=order_id,
            symbol=symbol,
            side=side.lower(),
            intended_price=intended_price,
            fill_price=fill_price,
            quantity=quantity,
            venue=venue.lower(),
            order_type=order_type.lower(),
            latency_ms=latency_ms,
        )
        self._records.append(record)

        # Trim oldest if over capacity
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records:]

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(self, period_hours: float = 24) -> Dict[str, Any]:
        """
        Analyse execution quality over the given period.

        Returns a comprehensive breakdown of execution costs.
        """
        cutoff = time.time() - period_hours * 3600
        recent = [r for r in self._records if r.timestamp >= cutoff]

        if not recent:
            return {
                "total_slippage_bps": 0.0,
                "trade_count": 0,
                "slippage_by_venue": {},
                "slippage_by_symbol": {},
                "slippage_by_order_type": {},
                "slippage_by_size": {},
                "maker_vs_taker_savings": 0.0,
                "optimal_venue_recommendation": "insufficient_data",
                "cost_reduction_opportunity_bps": 0.0,
                "avg_latency_ms": 0.0,
            }

        # Total slippage
        all_slippage = [r.slippage_bps for r in recent]
        total_slippage = sum(all_slippage) / len(all_slippage)

        # By venue
        by_venue: Dict[str, List[float]] = defaultdict(list)
        for r in recent:
            by_venue[r.venue].append(r.slippage_bps)
        slippage_by_venue = {
            v: round(sum(s) / len(s), 2) for v, s in by_venue.items()
        }

        # By symbol
        by_symbol: Dict[str, List[float]] = defaultdict(list)
        for r in recent:
            by_symbol[r.symbol].append(r.slippage_bps)
        slippage_by_symbol = {
            s: round(sum(v) / len(v), 2) for s, v in by_symbol.items()
        }

        # By order type
        by_type: Dict[str, List[float]] = defaultdict(list)
        for r in recent:
            by_type[r.order_type].append(r.slippage_bps)
        slippage_by_type = {
            t: round(sum(v) / len(v), 2) for t, v in by_type.items()
        }

        # By size bucket
        by_size: Dict[str, List[float]] = defaultdict(list)
        for r in recent:
            by_size[r.size_bucket].append(r.slippage_bps)
        slippage_by_size = {
            b: round(sum(v) / len(v), 2) for b, v in by_size.items()
        }

        # Maker vs taker savings
        limit_slippage = by_type.get("limit", [])
        market_slippage = by_type.get("market", [])
        maker_savings = 0.0
        if limit_slippage and market_slippage:
            avg_limit = sum(limit_slippage) / len(limit_slippage)
            avg_market = sum(market_slippage) / len(market_slippage)
            maker_savings = avg_market - avg_limit  # positive = limit is better

        # Optimal venue
        if slippage_by_venue:
            best_venue = min(slippage_by_venue, key=slippage_by_venue.get)  # type: ignore
        else:
            best_venue = "unknown"

        # Cost reduction opportunity: difference between worst and best venue
        if len(slippage_by_venue) >= 2:
            venue_vals = list(slippage_by_venue.values())
            cost_reduction = max(venue_vals) - min(venue_vals)
        else:
            cost_reduction = 0.0

        # Average latency
        avg_latency = sum(r.latency_ms for r in recent) / len(recent)

        return {
            "total_slippage_bps": round(total_slippage, 2),
            "trade_count": len(recent),
            "slippage_by_venue": slippage_by_venue,
            "slippage_by_symbol": slippage_by_symbol,
            "slippage_by_order_type": slippage_by_type,
            "slippage_by_size": slippage_by_size,
            "maker_vs_taker_savings": round(maker_savings, 2),
            "optimal_venue_recommendation": best_venue,
            "cost_reduction_opportunity_bps": round(cost_reduction, 2),
            "avg_latency_ms": round(avg_latency, 2),
        }

    def get_execution_score(self, period_hours: float = 24) -> float:
        """
        Compute execution quality score: 0-100.

        100 = perfect execution (zero slippage, low latency).
        0 = terrible execution (high slippage, high latency).
        """
        analysis = self.analyze(period_hours)

        if analysis["trade_count"] == 0:
            return 50.0  # No data = neutral

        # Slippage component (0-60 points)
        # 0 bps = 60 pts, 10 bps = 0 pts
        slip = abs(analysis["total_slippage_bps"])
        slip_score = max(0.0, 60.0 * (1.0 - slip / 10.0))

        # Latency component (0-20 points)
        # 0ms = 20 pts, 500ms = 0 pts
        lat = analysis["avg_latency_ms"]
        lat_score = max(0.0, 20.0 * (1.0 - lat / 500.0))

        # Maker usage component (0-20 points)
        # High maker savings = more points
        maker = analysis["maker_vs_taker_savings"]
        maker_score = min(20.0, max(0.0, maker * 4.0))  # 5 bps savings = 20 pts

        return round(min(100.0, slip_score + lat_score + maker_score), 1)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_records(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return the last N execution records as dicts."""
        recent = self._records[-limit:]
        return [
            {
                "order_id": r.order_id,
                "symbol": r.symbol,
                "side": r.side,
                "intended_price": r.intended_price,
                "fill_price": r.fill_price,
                "quantity": r.quantity,
                "venue": r.venue,
                "order_type": r.order_type,
                "latency_ms": r.latency_ms,
                "slippage_bps": round(r.slippage_bps, 2),
                "notional_usd": round(r.notional_usd, 2),
                "size_bucket": r.size_bucket,
                "timestamp": r.timestamp,
            }
            for r in recent
        ]

    @property
    def record_count(self) -> int:
        return len(self._records)
