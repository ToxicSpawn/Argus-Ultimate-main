"""
Execution Bridge — routes order execution through the sealed runtime.

The existing system places orders via:
- unified_trading_system._execute_signals() → direct exchange calls
- unified_execution_engine.execute_signals() → VWAP/limit/market

This bridge intercepts execution requests and routes them through:
ExecutionIntent → IntentRuntime → AdapterRegistry → VenueAdapter

In paper mode, it delegates to the existing paper wrapper.
In live mode, it enforces the sealed execution pipeline.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionRequest:
    """A request to execute a trade, produced by the assimilation layer."""
    symbol: str
    side: str              # "buy" or "sell"
    quantity: float
    price: float           # decision price
    order_type: str        # "market", "limit", "twap"
    strategy: str
    confidence: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExecutionResult:
    """Result of an execution request."""
    request: ExecutionRequest
    status: str            # "filled", "rejected", "pending", "error"
    fill_price: float = 0.0
    fill_qty: float = 0.0
    commission: float = 0.0
    order_id: str = ""
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


class ExecutionBridge:
    """
    Routes execution through the sealed runtime.

    In paper mode: delegates to existing paper trading path (preserves current behavior).
    In live mode: routes through argus_live execution pipeline with full safety gates.

    This is the ONLY path that can submit orders in live mode.
    """

    def __init__(
        self,
        mode: str = "paper",
        constitution: Optional[Dict[str, Any]] = None,
        max_single_exposure_pct: float = 0.08,
        max_gross_exposure_pct: float = 0.25,
        max_daily_loss_pct: float = 0.02,
    ):
        self._mode = mode.lower()
        self._constitution = constitution or {}
        self._max_single = max_single_exposure_pct
        self._max_gross = max_gross_exposure_pct
        self._max_daily_loss = max_daily_loss_pct
        self._requests: List[ExecutionRequest] = []
        self._results: List[ExecutionResult] = []
        self._blocked_count = 0
        self._executed_count = 0

        # Load limits from constitution if available
        limits = self._constitution.get("constitution", {}).get("limits", {})
        if limits:
            self._max_single = float(limits.get("max_single_symbol_exposure_pct", self._max_single))
            self._max_gross = float(limits.get("max_gross_exposure_pct", self._max_gross))
            self._max_daily_loss = float(limits.get("max_daily_loss_pct", self._max_daily_loss))

        logger.info(
            "ExecutionBridge initialized: mode=%s, max_single=%.1f%%, max_gross=%.1f%%, max_daily_loss=%.1f%%",
            self._mode, self._max_single * 100, self._max_gross * 100, self._max_daily_loss * 100,
        )

    def check_constitution(self, request: ExecutionRequest, portfolio_value: float,
                            current_exposure: float) -> Optional[str]:
        """
        Check if an execution request passes constitutional limits.

        Returns None if allowed, or a rejection reason string.
        """
        if portfolio_value <= 0:
            return "portfolio_value_zero"

        request_value = request.quantity * request.price
        request_pct = request_value / portfolio_value

        if request_pct > self._max_single:
            return f"single_symbol_exposure {request_pct:.1%} > {self._max_single:.1%}"

        projected_exposure = current_exposure + request_value
        projected_pct = projected_exposure / portfolio_value

        if projected_pct > self._max_gross:
            return f"gross_exposure {projected_pct:.1%} > {self._max_gross:.1%}"

        return None

    def submit(self, request: ExecutionRequest, portfolio_value: float = 1000.0,
               current_exposure: float = 0.0) -> ExecutionResult:
        """
        Submit an execution request.

        In paper mode: records the request and returns a simulated fill.
        In live mode: checks constitution, then routes through sealed pipeline.
        """
        self._requests.append(request)

        # Constitutional check (applies to ALL modes)
        rejection = self.check_constitution(request, portfolio_value, current_exposure)
        if rejection:
            self._blocked_count += 1
            result = ExecutionResult(
                request=request, status="rejected", reason=f"constitution: {rejection}",
            )
            self._results.append(result)
            logger.info("ExecutionBridge: REJECTED %s %s — %s", request.side, request.symbol, rejection)
            return result

        if self._mode == "paper":
            return self._paper_fill(request)
        elif self._mode == "live":
            return self._live_submit(request)
        else:
            # Shadow mode — record but don't execute
            result = ExecutionResult(
                request=request, status="shadow", reason="shadow mode — recorded only",
            )
            self._results.append(result)
            return result

    def _paper_fill(self, request: ExecutionRequest) -> ExecutionResult:
        """Simulate a paper fill with realistic slippage."""
        slippage_bps = 5.0
        slippage_mult = 1.0 + (slippage_bps / 10000.0) if request.side == "buy" else 1.0 - (slippage_bps / 10000.0)
        fill_price = request.price * slippage_mult
        commission = request.quantity * fill_price * 0.0026  # Kraken taker fee

        result = ExecutionResult(
            request=request,
            status="filled",
            fill_price=fill_price,
            fill_qty=request.quantity,
            commission=commission,
            order_id=f"paper_{request.symbol}_{int(time.time() * 1000)}",
        )
        self._results.append(result)
        self._executed_count += 1
        logger.info(
            "ExecutionBridge: PAPER FILL %s %s qty=%.8f @ %.2f (commission=%.4f)",
            request.side, request.symbol, request.quantity, fill_price, commission,
        )
        return result

    def _live_submit(self, request: ExecutionRequest) -> ExecutionResult:
        """
        Route through sealed live execution pipeline.

        This is where argus_live/execution/intent_runtime would be called.
        For now, returns an error — live execution requires full pipeline wiring.
        """
        # TODO: Wire through IntentRuntime → AdapterRegistry → VenueAdapter
        result = ExecutionResult(
            request=request, status="error",
            reason="live execution pipeline not yet wired — use paper mode",
        )
        self._results.append(result)
        return result

    def get_stats(self) -> Dict[str, Any]:
        return {
            "mode": self._mode,
            "total_requests": len(self._requests),
            "executed": self._executed_count,
            "blocked": self._blocked_count,
            "results": len(self._results),
        }
