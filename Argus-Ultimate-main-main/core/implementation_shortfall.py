"""
Implementation Shortfall Tracker.

Measures the cost of execution by comparing decision price (when signal
was generated) to actual fill price. Tracks per strategy and symbol.

Implementation shortfall = actual fill price - decision price
(positive = slippage cost, negative = price improvement)

Feeds back into:
1. Order type selection (high IS → use limit orders, low IS → continue market)
2. Strategy evaluation (high IS strategies are less profitable than they appear)
3. Venue routing (route to venues with lower historical IS)
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ISRecord:
    """Single implementation shortfall observation."""
    symbol: str
    strategy: str
    side: str
    decision_price: float
    fill_price: float
    quantity: float
    shortfall_bps: float
    venue: str
    order_type: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ISStats:
    """Aggregated IS statistics for a strategy/symbol/venue."""
    key: str
    n_trades: int
    avg_shortfall_bps: float
    median_shortfall_bps: float
    p95_shortfall_bps: float
    total_cost_usd: float
    recommended_order_type: str  # "limit" or "market" based on IS history


class ImplementationShortfallTracker:
    """
    Tracks execution quality and recommends order type improvements.

    For each trade, compares the decision price (from signal) to the actual
    fill price. High IS → switch to limit orders. Low IS → continue with
    current approach.

    Recommendations:
    - avg IS > 5 bps → use limit orders (more patient)
    - avg IS > 15 bps → use TWAP (split execution)
    - avg IS < 2 bps → market orders fine (fast execution)
    """

    def __init__(self, lookback: int = 500, limit_threshold_bps: float = 5.0,
                 twap_threshold_bps: float = 15.0):
        self._records: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback))
        self._limit_threshold = limit_threshold_bps
        self._twap_threshold = twap_threshold_bps
        self._total_tracked = 0

    def record(self, symbol: str, strategy: str, side: str,
               decision_price: float, fill_price: float, quantity: float,
               venue: str = "kraken", order_type: str = "market") -> ISRecord:
        """Record an implementation shortfall observation."""
        if decision_price <= 0:
            return ISRecord(symbol=symbol, strategy=strategy, side=side,
                            decision_price=decision_price, fill_price=fill_price,
                            quantity=quantity, shortfall_bps=0.0, venue=venue,
                            order_type=order_type)

        # Calculate shortfall in bps
        if side.lower() in ("buy", "long"):
            shortfall_bps = (fill_price - decision_price) / decision_price * 10000.0
        else:
            shortfall_bps = (decision_price - fill_price) / decision_price * 10000.0

        rec = ISRecord(
            symbol=symbol, strategy=strategy, side=side,
            decision_price=decision_price, fill_price=fill_price,
            quantity=quantity, shortfall_bps=shortfall_bps,
            venue=venue, order_type=order_type,
        )

        # Store by multiple keys for different aggregations
        self._records[f"strategy:{strategy}"].append(rec)
        self._records[f"symbol:{symbol}"].append(rec)
        self._records[f"venue:{venue}"].append(rec)
        self._records[f"pair:{strategy}:{symbol}"].append(rec)
        self._records["global"].append(rec)
        self._total_tracked += 1

        return rec

    def get_stats(self, key: str = "global") -> Optional[ISStats]:
        """Get IS statistics for a key (strategy:X, symbol:X, venue:X, or global)."""
        records = list(self._records.get(key, []))
        if not records:
            return None

        shortfalls = [r.shortfall_bps for r in records]
        shortfalls.sort()
        n = len(shortfalls)

        avg = sum(shortfalls) / n
        median = shortfalls[n // 2]
        p95 = shortfalls[int(n * 0.95)] if n >= 5 else max(shortfalls)
        total_cost = sum(r.quantity * r.fill_price * r.shortfall_bps / 10000.0 for r in records)

        # Recommend order type based on avg IS
        if avg > self._twap_threshold:
            rec_type = "twap"
        elif avg > self._limit_threshold:
            rec_type = "limit"
        else:
            rec_type = "market"

        return ISStats(
            key=key, n_trades=n, avg_shortfall_bps=round(avg, 2),
            median_shortfall_bps=round(median, 2), p95_shortfall_bps=round(p95, 2),
            total_cost_usd=round(total_cost, 2), recommended_order_type=rec_type,
        )

    def get_recommended_order_type(self, strategy: str, symbol: str) -> str:
        """Get recommended order type based on historical IS."""
        stats = self.get_stats(f"pair:{strategy}:{symbol}")
        if stats and stats.n_trades >= 10:
            return stats.recommended_order_type
        # Fall back to strategy-level
        stats = self.get_stats(f"strategy:{strategy}")
        if stats and stats.n_trades >= 10:
            return stats.recommended_order_type
        return "limit"  # default: be patient

    def get_venue_ranking(self) -> List[Dict[str, Any]]:
        """Rank venues by execution quality (lower IS = better)."""
        venues = []
        for key, records in self._records.items():
            if key.startswith("venue:"):
                venue_name = key.split(":", 1)[1]
                stats = self.get_stats(key)
                if stats:
                    venues.append({
                        "venue": venue_name,
                        "avg_is_bps": stats.avg_shortfall_bps,
                        "trades": stats.n_trades,
                        "total_cost_usd": stats.total_cost_usd,
                    })
        venues.sort(key=lambda x: x["avg_is_bps"])
        return venues

    def get_advisory(self) -> Dict[str, Any]:
        """Get IS advisory for the trading loop."""
        global_stats = self.get_stats("global")
        return {
            "total_tracked": self._total_tracked,
            "global_avg_is_bps": global_stats.avg_shortfall_bps if global_stats else 0.0,
            "global_recommended_type": global_stats.recommended_order_type if global_stats else "limit",
            "venue_ranking": self.get_venue_ranking(),
        }
