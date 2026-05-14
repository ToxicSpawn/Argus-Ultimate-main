"""
core/execution_engine.py
========================
ExecutionEngine — extracted from unified_trading_system.py (H01, Phase 1).

Responsibilities:
- Accept validated TradingSignal objects from the signal pipeline.
- Route orders through the registered OrderRouter.
- Apply pre-execution risk checks via RiskFacade.
- Record fills in PositionTracker.
- Emit structured execution events to the audit bus.

This module owns *no* strategy logic and *no* market-data fetching.
Those concerns belong to signal_pipeline.py and the data layer respectively.

Batch-3 additions
-----------------
* SOR (Smart Order Router) bid-level splitting across multiple venues.
* Adaptive exponential backoff retry on transient exchange errors.
* Explicit dry-run guard — synthetic fill is clearly flagged in logs.
* Per-request latency histogram bucket tracking.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("argus.core.execution_engine")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RETRY_BASE_DELAY: float = 0.25   # seconds
_RETRY_MAX_DELAY: float = 8.0
_RETRY_MAX_ATTEMPTS: int = 5
_TRANSIENT_ERRORS: frozenset = frozenset({
    "rate_limit", "timeout", "503", "429", "nonce",
    "connection", "temporary", "overloaded",
})

# Latency histogram buckets (ms)
_LAT_BUCKETS: Tuple[float, ...] = (1, 5, 10, 25, 50, 100, 250, 500, 1000)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ExecutionRequest:
    """Validated trade intent handed to the engine."""
    symbol: str
    side: str                     # "buy" | "sell"
    quantity: float
    price: Optional[float]        # None → market order
    strategy_name: str
    signal_confidence: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    # SOR: optional list of (venue_name, max_qty_fraction) tuples.
    # If None, the primary router handles everything.
    sor_venues: Optional[List[Tuple[str, float]]] = None
    meta: dict = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Outcome of a single order attempt."""
    success: bool
    request: ExecutionRequest
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    fee: float = 0.0
    order_id: str = ""
    latency_ms: float = 0.0
    error: Optional[str] = None
    venue: str = "primary"
    retry_count: int = 0
    dry_run: bool = False

    @property
    def cost(self) -> float:
        return self.filled_quantity * self.filled_price


@dataclass
class SORSlice:
    """One venue slice produced by the Smart Order Router."""
    venue: str
    quantity: float
    router: Any   # the venue-specific order-router object


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ExecutionEngine:
    """
    Thin, stateless execution layer.

    Parameters
    ----------
    order_router:
        Any object with an ``async place_order(request) -> dict`` method.
        Used as the default / primary venue.
    risk_facade:
        Optional object with a ``check(request) -> bool`` method.
        If *None* all requests pass through (useful in paper/test mode).
    position_tracker:
        Optional object with ``record_fill(result)``.
    dry_run:
        When *True* the order router is bypassed; a synthetic fill at the
        requested price (or 0.0) is returned immediately.
    venue_routers:
        Optional dict mapping venue name -> order-router for SOR splits.
    max_retry_attempts:
        Override the default transient-error retry limit (default 5).
    """

    def __init__(
        self,
        order_router: Any = None,
        risk_facade: Any = None,
        position_tracker: Any = None,
        dry_run: bool = False,
        venue_routers: Optional[Dict[str, Any]] = None,
        max_retry_attempts: int = _RETRY_MAX_ATTEMPTS,
    ) -> None:
        self._router = order_router
        self._risk = risk_facade
        self._tracker = position_tracker
        self._dry_run = dry_run
        self._venue_routers: Dict[str, Any] = venue_routers or {}
        self._max_retry = max_retry_attempts

        self._orders_placed: int = 0
        self._orders_rejected: int = 0
        self._orders_failed: int = 0
        self._total_retries: int = 0
        self._lat_histogram: Dict[str, int] = {f"<{b}ms": 0 for b in _LAT_BUCKETS}
        self._lat_histogram[">=1000ms"] = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute a single trade request end-to-end."""
        t0 = time.monotonic()

        # 1. Explicit dry-run guard — must be checked before risk so that test
        #    harnesses never accidentally hit a real exchange.
        if self._dry_run:
            result = self._synthetic_fill(request, time.monotonic() - t0)
            self._record_result(result)
            return result

        # 2. Pre-flight risk check
        if self._risk is not None:
            try:
                allowed = self._risk.check(request)
            except Exception:
                logger.exception(
                    "Risk facade raised during check — blocking order %s %s",
                    request.side, request.symbol,
                )
                allowed = False
            if not allowed:
                self._orders_rejected += 1
                logger.warning(
                    "Order REJECTED by risk facade: %s %s qty=%.6f conf=%.2f",
                    request.side, request.symbol,
                    request.quantity, request.signal_confidence,
                )
                return ExecutionResult(
                    success=False,
                    request=request,
                    error="blocked_by_risk_facade",
                )

        # 3. SOR split or single-venue placement
        if request.sor_venues and len(request.sor_venues) > 1:
            result = await self._execute_sor(request, t0)
        else:
            result = await self._place_with_retry(request, self._router, t0)

        # 4. Record fill
        if result.success and self._tracker is not None:
            try:
                self._tracker.record_fill(result)
            except Exception:
                logger.exception(
                    "PositionTracker raised during record_fill for %s",
                    request.symbol,
                )

        self._record_result(result)
        return result

    async def execute_batch(
        self, requests: List[ExecutionRequest], *, max_concurrency: int = 4
    ) -> List[ExecutionResult]:
        """Execute multiple requests with bounded concurrency."""
        sem = asyncio.Semaphore(max_concurrency)

        async def _bounded(req: ExecutionRequest) -> ExecutionResult:
            async with sem:
                return await self.execute(req)

        return list(
            await asyncio.gather(
                *[_bounded(r) for r in requests], return_exceptions=False
            )
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        total = self._orders_placed + self._orders_rejected + self._orders_failed
        return {
            "placed": self._orders_placed,
            "rejected": self._orders_rejected,
            "failed": self._orders_failed,
            "total_attempted": total,
            "total_retries": self._total_retries,
            "latency_histogram": dict(self._lat_histogram),
        }

    def reset_stats(self) -> None:
        self._orders_placed = 0
        self._orders_rejected = 0
        self._orders_failed = 0
        self._total_retries = 0
        for k in self._lat_histogram:
            self._lat_histogram[k] = 0

    # ------------------------------------------------------------------
    # Smart Order Router
    # ------------------------------------------------------------------

    async def _execute_sor(
        self, request: ExecutionRequest, t0: float
    ) -> ExecutionResult:
        """
        Split *request* across multiple venues by bid-level allocation.

        Each (venue, fraction) pair in request.sor_venues drives how much of
        the total quantity goes to that venue. Fractions are re-normalised so
        they always sum to 1.0. Slices execute concurrently; results are
        aggregated into a single ExecutionResult (VWAP fill price).
        """
        venues = request.sor_venues or []
        total_frac = sum(f for _, f in venues) or 1.0
        slices: List[SORSlice] = []
        remaining = request.quantity

        for i, (vname, frac) in enumerate(venues):
            router = self._venue_routers.get(vname) or self._router
            if router is None:
                logger.warning("SOR: no router for venue '%s', skipping", vname)
                continue
            # Last slice gets any rounding remainder
            if i == len(venues) - 1:
                qty = remaining
            else:
                qty = round(request.quantity * (frac / total_frac), 8)
                remaining = round(remaining - qty, 8)
            if qty > 0:
                slices.append(SORSlice(venue=vname, quantity=qty, router=router))

        if not slices:
            return ExecutionResult(
                success=False, request=request,
                error="sor_no_viable_venues",
            )

        # Build per-slice requests
        slice_requests = [
            ExecutionRequest(
                symbol=request.symbol,
                side=request.side,
                quantity=sl.quantity,
                price=request.price,
                strategy_name=request.strategy_name,
                signal_confidence=request.signal_confidence,
                stop_loss=request.stop_loss,
                take_profit=request.take_profit,
                meta={**request.meta, "sor_venue": sl.venue},
            )
            for sl in slices
        ]

        tasks = [
            self._place_with_retry(sreq, sl.router, t0)
            for sreq, sl in zip(slice_requests, slices)
        ]
        results: List[ExecutionResult] = await asyncio.gather(*tasks)

        # Aggregate
        total_filled = sum(r.filled_quantity for r in results if r.success)
        total_cost = sum(r.filled_quantity * r.filled_price for r in results if r.success)
        total_fee = sum(r.fee for r in results if r.success)
        any_success = any(r.success for r in results)
        vwap = total_cost / total_filled if total_filled > 0 else 0.0
        latency_ms = (time.monotonic() - t0) * 1000

        venue_names = "+".join(sl.venue for sl in slices)
        logger.info(
            "SOR FILL %s %s total_qty=%.6f vwap=%.4f fee=%.4f latency=%.1fms venues=%s",
            request.side, request.symbol,
            total_filled, vwap, total_fee, latency_ms, venue_names,
        )
        return ExecutionResult(
            success=any_success,
            request=request,
            filled_quantity=total_filled,
            filled_price=vwap,
            fee=total_fee,
            order_id="sor_" + "_".join(r.order_id for r in results if r.success),
            latency_ms=latency_ms,
            venue=venue_names,
        )

    # ------------------------------------------------------------------
    # Retry wrapper
    # ------------------------------------------------------------------

    async def _place_with_retry(
        self,
        request: ExecutionRequest,
        router: Any,
        t0: float,
    ) -> ExecutionResult:
        """
        Attempt order placement with adaptive exponential backoff.

        Only retries on transient errors (rate-limit, timeout, 503, etc.).
        Permanent errors (insufficient funds, bad params) fail immediately.
        """
        delay = _RETRY_BASE_DELAY
        last_error: Optional[str] = None

        for attempt in range(self._max_retry):
            result = await self._place_via_router(request, router, t0)
            if result.success:
                result.retry_count = attempt
                if attempt:
                    self._total_retries += attempt
                    logger.info(
                        "Order succeeded after %d retries: %s %s",
                        attempt, request.side, request.symbol,
                    )
                return result

            last_error = result.error or ""
            if not self._is_transient(last_error):
                logger.warning(
                    "Permanent error — not retrying: %s %s err=%s",
                    request.side, request.symbol, last_error,
                )
                result.retry_count = attempt
                return result

            if attempt < self._max_retry - 1:
                jitter = random.uniform(0, delay * 0.2)
                wait = min(delay + jitter, _RETRY_MAX_DELAY)
                logger.warning(
                    "Transient error (attempt %d/%d), retrying in %.2fs: %s",
                    attempt + 1, self._max_retry, wait, last_error,
                )
                await asyncio.sleep(wait)
                delay = min(delay * 2, _RETRY_MAX_DELAY)

        self._total_retries += self._max_retry
        return ExecutionResult(
            success=False,
            request=request,
            error=f"max_retries_exceeded:{last_error}",
            retry_count=self._max_retry,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    @staticmethod
    def _is_transient(error_str: str) -> bool:
        el = error_str.lower()
        return any(t in el for t in _TRANSIENT_ERRORS)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _place_via_router(
        self, request: ExecutionRequest, router: Any, t0: float
    ) -> ExecutionResult:
        if router is None:
            return ExecutionResult(
                success=False,
                request=request,
                error="no_order_router_configured",
            )
        try:
            raw = await router.place_order(request)
            latency_ms = (time.monotonic() - t0) * 1000
            return ExecutionResult(
                success=True,
                request=request,
                filled_quantity=float(raw.get("filled", request.quantity)),
                filled_price=float(raw.get("price", request.price or 0.0)),
                fee=float(raw.get("fee", 0.0)),
                order_id=str(raw.get("id", "")),
                latency_ms=latency_ms,
                venue=str(raw.get("venue", "primary")),
            )
        except Exception as exc:
            logger.exception(
                "OrderRouter raised for %s %s", request.side, request.symbol
            )
            return ExecutionResult(
                success=False,
                request=request,
                error=str(exc),
                latency_ms=(time.monotonic() - t0) * 1000,
            )

    def _synthetic_fill(self, request: ExecutionRequest, elapsed: float) -> ExecutionResult:
        """Return a paper fill without touching any external system."""
        price = request.price or 0.0
        oid = f"dry_{int(time.time() * 1000)}"
        logger.debug(
            "DRY-RUN synthetic fill: %s %s qty=%.6f @ %.4f id=%s",
            request.side, request.symbol, request.quantity, price, oid,
        )
        return ExecutionResult(
            success=True,
            request=request,
            filled_quantity=request.quantity,
            filled_price=price,
            fee=0.0,
            order_id=oid,
            latency_ms=elapsed * 1000,
            venue="dry_run",
            dry_run=True,
        )

    def _record_result(self, result: ExecutionResult) -> None:
        """Update counters and latency histogram."""
        if result.success:
            self._orders_placed += 1
            logger.info(
                "FILL  %s %s qty=%.6f @ %.4f fee=%.4f lat=%.1fms venue=%s retries=%d%s [%s]",
                result.request.side, result.request.symbol,
                result.filled_quantity, result.filled_price,
                result.fee, result.latency_ms,
                result.venue, result.retry_count,
                " [DRY]" if result.dry_run else "",
                result.request.strategy_name,
            )
        else:
            self._orders_failed += 1

        # Bucket the latency
        ms = result.latency_ms
        for b in _LAT_BUCKETS:
            if ms < b:
                self._lat_histogram[f"<{b}ms"] += 1
                return
        self._lat_histogram[">=1000ms"] += 1
