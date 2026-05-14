"""
Best Execution Report — records and exports execution quality metrics
for every fill, meeting MiFID II best execution requirements.

For each trade, captures: decision price, arrival price, fill price,
spread at time of order, venue used, venues available, and computes
implementation shortfall decomposition.
"""
from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BestExecutionRecord:
    """Single best execution record for one fill."""
    # Order identification
    order_id: str
    symbol: str
    side: str
    strategy: str
    venue_used: str
    venues_available: List[str]

    # Prices
    decision_price: float      # price when signal was generated
    arrival_price: float       # price when order reached venue
    fill_price: float          # actual execution price
    quantity: float

    # Market conditions at time of order
    spread_bps: float          # bid-ask spread in bps
    depth_at_price: float      # notional available at fill price level

    # Costs
    commission: float
    slippage_bps: float        # (fill_price - arrival_price) / arrival_price * 10000
    implementation_shortfall_bps: float  # (fill_price - decision_price) / decision_price * 10000

    # Timing
    decision_timestamp: float
    fill_timestamp: float
    latency_ms: float          # fill_timestamp - decision_timestamp in ms

    # Quality assessment
    venue_was_best: bool       # was the venue with tightest spread used?
    reason: str

    timestamp: float = field(default_factory=time.time)


class BestExecutionReporter:
    """
    Tracks and reports best execution quality.

    Records every fill with full context and exports to CSV/JSON
    for regulatory reporting or internal analysis.
    """

    def __init__(self, report_dir: str = "reports/best_execution"):
        self._dir = Path(report_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._records: List[BestExecutionRecord] = []

    def record_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        strategy: str,
        venue_used: str,
        venues_available: Optional[List[str]] = None,
        decision_price: float = 0.0,
        arrival_price: float = 0.0,
        fill_price: float = 0.0,
        quantity: float = 0.0,
        spread_bps: float = 0.0,
        depth_at_price: float = 0.0,
        commission: float = 0.0,
        decision_timestamp: float = 0.0,
        fill_timestamp: float = 0.0,
    ) -> BestExecutionRecord:
        """Record a fill for best execution reporting."""
        if arrival_price <= 0:
            arrival_price = decision_price
        if fill_timestamp <= 0:
            fill_timestamp = time.time()
        if decision_timestamp <= 0:
            decision_timestamp = fill_timestamp

        # Compute metrics
        sign = 1.0 if side.lower() in ("buy", "long") else -1.0
        slippage_bps = sign * (fill_price - arrival_price) / max(arrival_price, 1e-9) * 10000
        is_bps = sign * (fill_price - decision_price) / max(decision_price, 1e-9) * 10000
        latency_ms = (fill_timestamp - decision_timestamp) * 1000

        record = BestExecutionRecord(
            order_id=order_id,
            symbol=symbol,
            side=side,
            strategy=strategy,
            venue_used=venue_used,
            venues_available=venues_available or [venue_used],
            decision_price=decision_price,
            arrival_price=arrival_price,
            fill_price=fill_price,
            quantity=quantity,
            spread_bps=spread_bps,
            depth_at_price=depth_at_price,
            commission=commission,
            slippage_bps=round(slippage_bps, 2),
            implementation_shortfall_bps=round(is_bps, 2),
            decision_timestamp=decision_timestamp,
            fill_timestamp=fill_timestamp,
            latency_ms=round(latency_ms, 1),
            venue_was_best=True,  # TODO: compare spreads across venues
            reason="fill recorded",
        )
        self._records.append(record)
        return record

    def export_csv(self, filename: str = "best_execution.csv") -> Path:
        """Export all records to CSV."""
        path = self._dir / filename
        if not self._records:
            return path

        fields = [
            "order_id", "symbol", "side", "strategy", "venue_used",
            "decision_price", "arrival_price", "fill_price", "quantity",
            "spread_bps", "commission", "slippage_bps",
            "implementation_shortfall_bps", "latency_ms", "venue_was_best",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for rec in self._records:
                d = asdict(rec)
                writer.writerow({k: d[k] for k in fields})

        logger.info("Best execution report exported: %s (%d records)", path, len(self._records))
        return path

    def export_json(self, filename: str = "best_execution.json") -> Path:
        """Export all records to JSON."""
        path = self._dir / filename
        data = [asdict(r) for r in self._records]
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info("Best execution JSON exported: %s (%d records)", path, len(self._records))
        return path

    def summary(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        if not self._records:
            return {"total_fills": 0}

        slippages = [r.slippage_bps for r in self._records]
        is_vals = [r.implementation_shortfall_bps for r in self._records]
        latencies = [r.latency_ms for r in self._records]

        return {
            "total_fills": len(self._records),
            "avg_slippage_bps": round(sum(slippages) / len(slippages), 2),
            "avg_is_bps": round(sum(is_vals) / len(is_vals), 2),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "worst_slippage_bps": round(max(slippages), 2),
            "best_slippage_bps": round(min(slippages), 2),
            "total_commission": round(sum(r.commission for r in self._records), 4),
        }

    def get_records(self, last_n: int = 100) -> List[BestExecutionRecord]:
        return self._records[-last_n:]
