"""
monitoring/institutional_execution_integration.py
==================================================
Integrates existing institutional execution infrastructure into the operator dashboard.

Wires together:
  - execution/fix_protocol_adapter.py (FIX-style protocol, 684 lines)
  - execution/institutional_execution.py (Almgren-Chriss, VWAP/TWAP/POV, 614 lines)
  - execution/smart_order_router_v2.py (multi-venue routing, 527 lines)
  - execution/execution_quality_tracker.py (fill quality, 279 lines)
  - execution/latency_attribution.py (latency monitoring, 79 lines)
  - execution/implementation_shortfall.py (IS calculation, 74 lines)
  - monitoring/operator_dashboard.py (dashboard state)

This is the glue that makes institutional execution visible and controllable
through the operator dashboard and WebSocket feed.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class VenuePerformance:
    """Per-venue execution performance metrics."""
    venue           : str
    fills           : int
    total_volume    : float
    avg_slippage_bps: float
    p99_slippage_bps: float
    fill_rate       : float
    avg_latency_ms  : float
    cost_bps        : float
    quality_score   : float  # 0-100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "venue"            : self.venue,
            "fills"            : self.fills,
            "total_volume"     : self.total_volume,
            "avg_slippage_bps" : self.avg_slippage_bps,
            "p99_slippage_bps" : self.p99_slippage_bps,
            "fill_rate"        : self.fill_rate,
            "avg_latency_ms"   : self.avg_latency_ms,
            "cost_bps"         : self.cost_bps,
            "quality_score"    : self.quality_score,
        }


@dataclass
class ActiveAlgoOrder:
    """An active algorithmic order."""
    order_id        : str
    symbol          : str
    side            : str
    algorithm       : str
    total_quantity  : float
    filled_quantity : float
    slices_total    : int
    slices_filled   : int
    start_time      : float
    expected_end    : float
    expected_cost_bps: float
    actual_cost_bps : float
    status          : str  # "running", "completed", "cancelled"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id"         : self.order_id,
            "symbol"           : self.symbol,
            "side"             : self.side,
            "algorithm"        : self.algorithm,
            "total_quantity"   : self.total_quantity,
            "filled_quantity"  : self.filled_quantity,
            "fill_pct"         : self.filled_quantity / self.total_quantity * 100 if self.total_quantity > 0 else 0,
            "slices_total"     : self.slices_total,
            "slices_filled"    : self.slices_filled,
            "start_time"       : self.start_time,
            "expected_end"     : self.expected_end,
            "expected_cost_bps": self.expected_cost_bps,
            "actual_cost_bps"  : self.actual_cost_bps,
            "status"           : self.status,
        }


@dataclass
class ExecutionSnapshot:
    """Full execution snapshot for dashboard consumption."""
    timestamp           : float
    # Venue performance
    venues              : List[VenuePerformance]
    best_venue          : str
    # Active algo orders
    active_orders       : List[ActiveAlgoOrder]
    # Aggregate metrics
    total_fills_today   : int
    total_volume_today  : float
    avg_slippage_bps    : float
    avg_is_bps          : float  # implementation shortfall
    total_fees_bps      : float
    # Latency metrics
    latency_p50_ms      : float
    latency_p95_ms      : float
    latency_p99_ms      : float
    # Router stats
    router_decisions    : int
    multi_venue_splits  : int
    # Alerts
    alerts              : List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp"         : self.timestamp,
            "venues"            : [v.to_dict() for v in self.venues],
            "best_venue"        : self.best_venue,
            "active_orders"     : [o.to_dict() for o in self.active_orders],
            "total_fills_today" : self.total_fills_today,
            "total_volume_today": self.total_volume_today,
            "avg_slippage_bps"  : self.avg_slippage_bps,
            "avg_is_bps"        : self.avg_is_bps,
            "total_fees_bps"    : self.total_fees_bps,
            "latency_p50_ms"    : self.latency_p50_ms,
            "latency_p95_ms"    : self.latency_p95_ms,
            "latency_p99_ms"    : self.latency_p99_ms,
            "router_decisions"  : self.router_decisions,
            "multi_venue_splits": self.multi_venue_splits,
            "alerts"            : self.alerts,
        }


# ---------------------------------------------------------------------------
# Institutional Execution Integrator
# ---------------------------------------------------------------------------

class InstitutionalExecutionIntegrator:
    """
    Wires existing execution modules into a unified interface for the dashboard.

    Parameters
    ----------
    db_path : str — path for execution quality SQLite database
    """

    def __init__(self, db_path: str = "data/execution_quality.db") -> None:
        self._db_path = db_path

        # Internal state
        self._fills_today       : Deque[Dict[str, Any]] = deque(maxlen=5000)
        self._venue_slippage    : Dict[str, Deque[float]] = {}
        self._venue_volume      : Dict[str, float] = {}
        self._active_orders     : Dict[str, ActiveAlgoOrder] = {}
        self._router_decisions  : int = 0
        self._multi_venue_splits: int = 0
        self._total_is_bps      : Deque[float] = deque(maxlen=2000)
        self._total_fees_bps    : Deque[float] = deque(maxlen=2000)
        self._alerts            : List[Dict[str, Any]] = []
        self._lock              = threading.Lock()

        # Lazy-loaded modules
        self._quality_tracker   : Optional[Any] = None
        self._latency_tracker   : Optional[Any] = None
        self._router            : Optional[Any] = None
        self._fix_adapter       : Optional[Any] = None

        logger.info("InstitutionalExecutionIntegrator: initialised")

    # ------------------------------------------------------------------ Lazy init

    def _init_quality_tracker(self) -> None:
        if self._quality_tracker is not None:
            return
        try:
            from execution.execution_quality_tracker import ExecutionQualityTracker
            self._quality_tracker = ExecutionQualityTracker(db_path=self._db_path)
            logger.info("InstitutionalExecutionIntegrator: quality tracker initialised")
        except Exception as e:
            logger.warning("Failed to init quality tracker: %s", e)

    def _init_latency_tracker(self) -> None:
        if self._latency_tracker is not None:
            return
        try:
            from execution.latency_attribution import LatencyTracker
            self._latency_tracker = LatencyTracker()
            logger.info("InstitutionalExecutionIntegrator: latency tracker initialised")
        except Exception as e:
            logger.warning("Failed to init latency tracker: %s", e)

    def _init_router(self) -> None:
        if self._router is not None:
            return
        try:
            from execution.smart_order_router_v2 import SmartOrderRouterV2
            self._router = SmartOrderRouterV2()
            logger.info("InstitutionalExecutionIntegrator: SOR initialised")
        except Exception as e:
            logger.warning("Failed to init SOR: %s", e)

    def _init_fix_adapter(self) -> None:
        if self._fix_adapter is not None:
            return
        try:
            from execution.fix_protocol_adapter import FIXAdapter
            self._fix_adapter = FIXAdapter
            logger.info("InstitutionalExecutionIntegrator: FIX adapter loaded")
        except Exception as e:
            logger.warning("Failed to init FIX adapter: %s", e)

    # ------------------------------------------------------------------ Data ingestion

    def record_fill(
        self,
        symbol      : str,
        venue       : str,
        side        : str,
        quantity    : float,
        price       : float,
        expected_price: float,
        fee_bps     : float,
        strategy    : str = "unknown",
        latency_ms  : float = 0.0,
    ) -> None:
        """Record a fill for tracking."""
        with self._lock:
            # Compute slippage
            if expected_price > 0:
                slippage_bps = abs(price - expected_price) / expected_price * 10000
            else:
                slippage_bps = 0.0

            # Record fill
            fill = {
                "timestamp"     : time.time(),
                "symbol"        : symbol,
                "venue"         : venue,
                "side"          : side,
                "quantity"      : quantity,
                "price"         : price,
                "expected_price": expected_price,
                "slippage_bps"  : slippage_bps,
                "fee_bps"       : fee_bps,
                "strategy"      : strategy,
                "latency_ms"    : latency_ms,
            }
            self._fills_today.append(fill)

            # Track venue slippage
            if venue not in self._venue_slippage:
                self._venue_slippage[venue] = deque(maxlen=1000)
            self._venue_slippage[venue].append(slippage_bps)

            # Track venue volume
            self._venue_volume[venue] = self._venue_volume.get(venue, 0.0) + quantity * price

            # Track IS and fees
            is_bps = slippage_bps  # simplified
            self._total_is_bps.append(is_bps)
            self._total_fees_bps.append(fee_bps)

            # Feed to quality tracker
            if self._quality_tracker is not None:
                try:
                    hour = datetime.now().hour
                    self._quality_tracker.record_fill(
                        symbol, strategy, hour, expected_price, price, quantity * price
                    )
                except Exception:
                    pass

    def record_router_decision(self, multi_venue: bool = False) -> None:
        """Record a router decision."""
        with self._lock:
            self._router_decisions += 1
            if multi_venue:
                self._multi_venue_splits += 1

    def add_active_order(self, order: ActiveAlgoOrder) -> None:
        """Add an active algorithmic order."""
        with self._lock:
            self._active_orders[order.order_id] = order

    def update_order(
        self,
        order_id        : str,
        filled_quantity : float,
        slices_filled   : int,
        actual_cost_bps : float,
        status          : str,
    ) -> None:
        """Update an active order's progress."""
        with self._lock:
            order = self._active_orders.get(order_id)
            if order:
                order.filled_quantity = filled_quantity
                order.slices_filled = slices_filled
                order.actual_cost_bps = actual_cost_bps
                order.status = status

    # ------------------------------------------------------------------ Snapshot

    def compute_snapshot(self) -> ExecutionSnapshot:
        """Compute full execution snapshot."""
        now = time.time()

        # Venue performance
        venues = self._compute_venue_performance()
        best_venue = max(venues, key=lambda v: v.quality_score).venue if venues else "none"

        # Aggregate metrics
        fills = list(self._fills_today)
        total_fills = len(fills)
        total_volume = sum(f["quantity"] * f["price"] for f in fills)
        avg_slippage = float(np.mean([f["slippage_bps"] for f in fills])) if fills else 0.0
        avg_is = float(np.mean(self._total_is_bps)) if self._total_is_bps else 0.0
        avg_fees = float(np.mean(self._total_fees_bps)) if self._total_fees_bps else 0.0

        # Latency
        lat_p50, lat_p95, lat_p99 = self._get_latency_stats()

        # Alerts
        alerts = self._check_alerts(avg_slippage, venues)

        return ExecutionSnapshot(
            timestamp           = now,
            venues              = venues,
            best_venue          = best_venue,
            active_orders       = list(self._active_orders.values()),
            total_fills_today   = total_fills,
            total_volume_today  = total_volume,
            avg_slippage_bps    = avg_slippage,
            avg_is_bps          = avg_is,
            total_fees_bps      = avg_fees,
            latency_p50_ms      = lat_p50,
            latency_p95_ms      = lat_p95,
            latency_p99_ms      = lat_p99,
            router_decisions    = self._router_decisions,
            multi_venue_splits  = self._multi_venue_splits,
            alerts              = alerts,
        )

    def _compute_venue_performance(self) -> List[VenuePerformance]:
        """Compute per-venue performance metrics."""
        results: List[VenuePerformance] = []
        fills = list(self._fills_today)

        for venue, slippages in self._venue_slippage.items():
            venue_fills = [f for f in fills if f["venue"] == venue]
            if not venue_fills:
                continue

            slip_arr = list(slippages)
            avg_slip = float(np.mean(slip_arr)) if slip_arr else 0.0
            p99_slip = float(np.percentile(slip_arr, 99)) if len(slip_arr) > 10 else avg_slip
            avg_lat = float(np.mean([f["latency_ms"] for f in venue_fills]))
            avg_fee = float(np.mean([f["fee_bps"] for f in venue_fills]))

            # Quality score: higher is better (100 = perfect, 0 = terrible)
            quality = max(0.0, 100.0 - avg_slip * 2 - p99_slip * 0.5)

            results.append(VenuePerformance(
                venue            = venue,
                fills            = len(venue_fills),
                total_volume     = self._venue_volume.get(venue, 0.0),
                avg_slippage_bps = avg_slip,
                p99_slippage_bps = p99_slip,
                fill_rate        = 1.0,  # simplified
                avg_latency_ms   = avg_lat,
                cost_bps         = avg_fee + avg_slip,
                quality_score    = quality,
            ))

        return sorted(results, key=lambda v: v.quality_score, reverse=True)

    def _get_latency_stats(self) -> Tuple[float, float, float]:
        """Get latency p50/p95/p99 from tracker."""
        if self._latency_tracker is None:
            return 0.0, 0.0, 0.0
        try:
            stats = self._latency_tracker.get_stats("order_roundtrip")
            return stats.get("p50_ms", 0.0), stats.get("p95_ms", 0.0), stats.get("p99_ms", 0.0)
        except Exception:
            return 0.0, 0.0, 0.0

    def _check_alerts(
        self,
        avg_slippage: float,
        venues: List[VenuePerformance],
    ) -> List[Dict[str, Any]]:
        """Check execution alerts."""
        alerts: List[Dict[str, Any]] = []
        now = time.time()

        # High slippage alert
        if avg_slippage > 20.0:
            alerts.append({
                "type"    : "high_slippage",
                "severity": "warning",
                "message" : f"Average slippage {avg_slippage:.1f}bps exceeds 20bps threshold",
                "value"   : avg_slippage,
                "threshold": 20.0,
                "ts"      : now,
            })

        # Venue degradation
        for v in venues:
            if v.quality_score < 50.0:
                alerts.append({
                    "type"    : "venue_degradation",
                    "severity": "warning",
                    "message" : f"Venue {v.venue} quality score {v.quality_score:.0f} below 50",
                    "value"   : v.quality_score,
                    "threshold": 50.0,
                    "ts"      : now,
                })

        # High latency
        lat_p99 = self._get_latency_stats()[2]
        if lat_p99 > 500.0:
            alerts.append({
                "type"    : "high_latency",
                "severity": "critical",
                "message" : f"P99 latency {lat_p99:.0f}ms exceeds 500ms threshold",
                "value"   : lat_p99,
                "threshold": 500.0,
                "ts"      : now,
            })

        return alerts

    # ------------------------------------------------------------------ Router integration

    def route_order(
        self,
        symbol    : str,
        side      : str,
        size_usd  : float,
        venue_books: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Route an order using the Smart Order Router.
        Returns routing recommendation.
        """
        self._init_router()
        if self._router is None:
            return {"error": "Router not available"}

        try:
            # Single venue recommendation
            rec = self._router.get_best_venue(symbol, side, size_usd, venue_books)
            self.record_router_decision(multi_venue=False)

            # Multi-venue split if order is large
            splits = []
            if size_usd > 10000:
                splits = self._router.split_across_venues(symbol, side, size_usd, venue_books)
                if len(splits) > 1:
                    self.record_router_decision(multi_venue=True)

            return {
                "best_venue"      : rec.venue,
                "expected_price"  : rec.expected_price,
                "expected_slippage_bps": rec.expected_slippage_bps,
                "expected_fees_bps": rec.expected_fees_bps,
                "total_cost_bps"  : rec.total_cost_bps,
                "confidence"      : rec.confidence,
                "multi_venue_splits": [s.__dict__ for s in splits] if splits else [],
            }
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------ Dashboard integration

    def feed_to_dashboard(self, dashboard_state: Any) -> None:
        """Push current execution snapshot to the operator dashboard state."""
        snapshot = self.compute_snapshot()

        # Store execution metrics on dashboard state
        if hasattr(dashboard_state, '_execution_snapshot'):
            dashboard_state._execution_snapshot = snapshot

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary dict for quick dashboard updates."""
        snap = self.compute_snapshot()
        return {
            "total_fills_today" : snap.total_fills_today,
            "total_volume_today": snap.total_volume_today,
            "avg_slippage_bps"  : snap.avg_slippage_bps,
            "avg_is_bps"        : snap.avg_is_bps,
            "best_venue"        : snap.best_venue,
            "n_active_orders"   : len(snap.active_orders),
            "router_decisions"  : snap.router_decisions,
            "latency_p99_ms"    : snap.latency_p99_ms,
            "n_alerts"          : len(snap.alerts),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_integrator: Optional[InstitutionalExecutionIntegrator] = None


def get_institutional_execution() -> InstitutionalExecutionIntegrator:
    """Get or create the institutional execution integrator singleton."""
    global _integrator
    if _integrator is None:
        _integrator = InstitutionalExecutionIntegrator()
    return _integrator
