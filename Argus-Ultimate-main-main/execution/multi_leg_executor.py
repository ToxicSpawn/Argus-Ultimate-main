"""
execution/multi_leg_executor.py
================================
Simultaneous multi-leg order submission with best-effort cancellation on
partial failures.  Designed for cross-exchange arbitrage and delta-neutral
hedging in HFT contexts.

Provides:
  - LegSpec            — dataclass describing a single order leg
  - MultiLegExecutor   — async executor for multi-leg strategies
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LegSpec dataclass
# ---------------------------------------------------------------------------

@dataclass
class LegSpec:
    """
    Specification for a single order leg.

    Fields
    ------
    exchange_name : str   — key into MultiLegExecutor.exchanges dict
    symbol        : str   — instrument symbol, e.g. "BTC-USD"
    side          : str   — "buy" or "sell"
    size          : float — order quantity in base units
    price         : float — limit price (0 for market orders)
    order_type    : str   — "limit" or "market"
    leg_id        : str   — unique identifier (auto-generated if not provided)
    """
    exchange_name : str
    symbol        : str
    side          : str
    size          : float
    price         : float
    order_type    : str   = "limit"
    leg_id        : str   = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if self.side.lower() not in ("buy", "sell"):
            raise ValueError(f"LegSpec.side must be 'buy' or 'sell', got {self.side!r}")
        if self.size <= 0:
            raise ValueError(f"LegSpec.size must be positive, got {self.size}")
        if self.order_type.lower() not in ("limit", "market"):
            raise ValueError(f"LegSpec.order_type must be 'limit' or 'market', got {self.order_type!r}")
        self.side       = self.side.lower()
        self.order_type = self.order_type.lower()


# ---------------------------------------------------------------------------
# Internal result containers
# ---------------------------------------------------------------------------

@dataclass
class LegResult:
    leg_id       : str
    exchange_name: str
    symbol       : str
    side         : str
    success      : bool
    order_dict   : Optional[Dict[str, Any]]
    error        : Optional[Exception]
    submit_ns    : int
    ack_ns       : Optional[int]

    @property
    def latency_ms(self) -> Optional[float]:
        if self.ack_ns and self.submit_ns:
            return (self.ack_ns - self.submit_ns) / 1_000_000.0
        return None

    def to_dict(self) -> dict:
        return {
            "leg_id":        self.leg_id,
            "exchange_name": self.exchange_name,
            "symbol":        self.symbol,
            "side":          self.side,
            "success":       self.success,
            "order":         self.order_dict,
            "error":         str(self.error) if self.error else None,
            "latency_ms":    self.latency_ms,
        }


# ---------------------------------------------------------------------------
# MultiLegExecutor
# ---------------------------------------------------------------------------

class MultiLegExecutor:
    """
    Async executor for simultaneous multi-leg order submission.

    Parameters
    ----------
    exchanges : dict[str, Any]
        Mapping of exchange_name → exchange client object.
        Each client must expose:
          - async submit_order(symbol, side, size, price, order_type) -> dict
          - async cancel_order(order_id) -> bool   (best-effort)

    Attributes (statistics)
    -----------------------
    _stats : dict tracking cumulative metrics across all submit_legs calls
    """

    def __init__(self, exchanges: Dict[str, Any]) -> None:
        if not exchanges:
            raise ValueError("exchanges dict must not be empty.")
        self.exchanges: Dict[str, Any] = exchanges

        # Cumulative session statistics
        self._stats: Dict[str, Any] = {
            "legs_submitted":     0,
            "legs_filled":        0,
            "legs_cancelled":     0,
            "legs_failed":        0,
            "total_fill_latency_ms": 0.0,
            "fill_latency_samples":  0,
            "calls":              0,    # number of submit_legs invocations
        }

    # ------------------------------------------------------------------
    # Core: submit all legs simultaneously
    # ------------------------------------------------------------------

    async def submit_legs(self, legs: List[LegSpec]) -> List[dict]:
        """
        Submit all *legs* simultaneously via asyncio.gather.

        On any leg failure, attempts to cancel already-acknowledged legs
        (best effort — exchange may reject the cancel).

        Parameters
        ----------
        legs : list[LegSpec]
            One or more leg specifications.

        Returns
        -------
        list[dict]
            One dict per leg (same order as input).  Failed legs include
            an "error" key with the exception string; successful legs
            include an "order" key with the exchange response.
        """
        if not legs:
            return []

        self._stats["calls"] += 1
        self._stats["legs_submitted"] += len(legs)

        coroutines = [self._submit_single_leg(leg) for leg in legs]
        raw_results = await asyncio.gather(*coroutines, return_exceptions=True)

        leg_results: List[LegResult] = []
        for leg, raw in zip(legs, raw_results):
            if isinstance(raw, Exception):
                lr = LegResult(
                    leg_id       = leg.leg_id,
                    exchange_name= leg.exchange_name,
                    symbol       = leg.symbol,
                    side         = leg.side,
                    success      = False,
                    order_dict   = None,
                    error        = raw,
                    submit_ns    = 0,
                    ack_ns       = None,
                )
                self._stats["legs_failed"] += 1
                logger.error(
                    "Leg %s on %s failed: %s",
                    leg.leg_id, leg.exchange_name, raw,
                )
            else:
                lr = raw
                if lr.success:
                    self._stats["legs_filled"] += 1
                    if lr.latency_ms is not None:
                        self._stats["total_fill_latency_ms"] += lr.latency_ms
                        self._stats["fill_latency_samples"]  += 1
                    logger.info(
                        "Leg %s on %s submitted OK | order_id=%s | latency=%.3f ms",
                        lr.leg_id, lr.exchange_name,
                        (lr.order_dict or {}).get("order_id", "N/A"),
                        lr.latency_ms or 0.0,
                    )
                else:
                    self._stats["legs_failed"] += 1
                    logger.error(
                        "Leg %s on %s returned failure: %s",
                        lr.leg_id, lr.exchange_name, lr.error,
                    )
            leg_results.append(lr)

        # --- Partial-failure handling ---
        failed = [lr for lr in leg_results if not lr.success]
        succeeded = [lr for lr in leg_results if lr.success]

        if failed and succeeded:
            logger.warning(
                "%d leg(s) failed; attempting to cancel %d successful leg(s).",
                len(failed), len(succeeded),
            )
            cancel_tasks = [
                self._cancel_leg(lr)
                for lr in succeeded
                if lr.order_dict and lr.order_dict.get("order_id")
            ]
            if cancel_tasks:
                cancel_results = await asyncio.gather(*cancel_tasks, return_exceptions=True)
                for lr, cr in zip(succeeded, cancel_results):
                    if isinstance(cr, Exception):
                        logger.warning(
                            "Best-effort cancel of leg %s failed: %s", lr.leg_id, cr
                        )
                    else:
                        self._stats["legs_cancelled"] += 1
                        logger.info("Leg %s cancelled (partial failure rollback).", lr.leg_id)

        return [lr.to_dict() for lr in leg_results]

    # ------------------------------------------------------------------
    # Convenience: 2-leg arbitrage
    # ------------------------------------------------------------------

    async def submit_arb_pair(
        self,
        buy_exchange: str,
        sell_exchange: str,
        symbol: str,
        size: float,
        buy_price: float,
        sell_price: float,
    ) -> Tuple[dict, dict]:
        """
        Submit a two-leg arbitrage: simultaneous buy on *buy_exchange* and
        sell on *sell_exchange*.

        Returns
        -------
        tuple[dict, dict]
            (buy_leg_result, sell_leg_result)
        """
        buy_leg = LegSpec(
            exchange_name=buy_exchange,
            symbol=symbol,
            side="buy",
            size=size,
            price=buy_price,
            order_type="limit",
        )
        sell_leg = LegSpec(
            exchange_name=sell_exchange,
            symbol=symbol,
            side="sell",
            size=size,
            price=sell_price,
            order_type="limit",
        )
        results = await self.submit_legs([buy_leg, sell_leg])
        return results[0], results[1]

    # ------------------------------------------------------------------
    # Convenience: hedge leg
    # ------------------------------------------------------------------

    async def submit_hedge(
        self,
        primary_order: dict,
        hedge_exchange: str,
        hedge_symbol: str,
        hedge_size: float,
        hedge_side: str,
    ) -> dict:
        """
        Submit a single hedge leg immediately after a primary fill.

        Parameters
        ----------
        primary_order  : dict  — filled primary order (must contain 'fill_price')
        hedge_exchange : str   — exchange name for the hedge
        hedge_symbol   : str   — symbol for the hedge instrument
        hedge_size     : float — size of the hedge
        hedge_side     : str   — "buy" or "sell"

        Returns
        -------
        dict — hedge leg result
        """
        hedge_price = float(primary_order.get("fill_price", 0.0))
        leg = LegSpec(
            exchange_name=hedge_exchange,
            symbol=hedge_symbol,
            side=hedge_side,
            size=hedge_size,
            price=hedge_price,
            order_type="limit" if hedge_price > 0 else "market",
        )
        results = await self.submit_legs([leg])
        return results[0]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return a snapshot of cumulative execution statistics."""
        samples = self._stats["fill_latency_samples"]
        avg_latency = (
            self._stats["total_fill_latency_ms"] / samples
            if samples > 0 else 0.0
        )
        return {
            "legs_submitted":         self._stats["legs_submitted"],
            "legs_filled":            self._stats["legs_filled"],
            "legs_cancelled":         self._stats["legs_cancelled"],
            "legs_failed":            self._stats["legs_failed"],
            "avg_fill_latency_ms":    avg_latency,
            "fill_latency_samples":   samples,
            "submit_legs_calls":      self._stats["calls"],
            "success_rate":           (
                self._stats["legs_filled"] / self._stats["legs_submitted"]
                if self._stats["legs_submitted"] > 0 else 0.0
            ),
        }

    def reset_stats(self) -> None:
        """Reset all statistics counters (e.g. at session start)."""
        for k in self._stats:
            self._stats[k] = 0 if isinstance(self._stats[k], int) else 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _submit_single_leg(self, leg: LegSpec) -> LegResult:
        """Submit one leg to its exchange client and return a LegResult."""
        exchange = self._get_exchange(leg.exchange_name)
        submit_ns = time.perf_counter_ns()
        try:
            order_dict = await exchange.submit_order(
                symbol     = leg.symbol,
                side       = leg.side,
                size       = leg.size,
                price      = leg.price,
                order_type = leg.order_type,
            )
            ack_ns = time.perf_counter_ns()
            return LegResult(
                leg_id        = leg.leg_id,
                exchange_name = leg.exchange_name,
                symbol        = leg.symbol,
                side          = leg.side,
                success       = True,
                order_dict    = order_dict,
                error         = None,
                submit_ns     = submit_ns,
                ack_ns        = ack_ns,
            )
        except Exception as exc:  # noqa: BLE001
            return LegResult(
                leg_id        = leg.leg_id,
                exchange_name = leg.exchange_name,
                symbol        = leg.symbol,
                side          = leg.side,
                success       = False,
                order_dict    = None,
                error         = exc,
                submit_ns     = submit_ns,
                ack_ns        = None,
            )

    async def _cancel_leg(self, lr: LegResult) -> bool:
        """Best-effort cancel of an already-submitted leg."""
        exchange = self._get_exchange(lr.exchange_name)
        order_id = (lr.order_dict or {}).get("order_id")
        if order_id is None:
            logger.warning("Cannot cancel leg %s — no order_id in response.", lr.leg_id)
            return False
        try:
            result = await exchange.cancel_order(order_id)
            return bool(result)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Cancel of order %s on %s raised: %s",
                order_id, lr.exchange_name, exc,
            )
            return False

    def _get_exchange(self, name: str) -> Any:
        """Retrieve exchange client by name; raise on unknown exchange."""
        client = self.exchanges.get(name)
        if client is None:
            raise KeyError(
                f"Unknown exchange {name!r}. "
                f"Available: {list(self.exchanges.keys())}"
            )
        return client
