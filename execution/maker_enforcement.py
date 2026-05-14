"""
Maker Order Enforcement — enforces post-only (maker) orders where beneficial.

On Kraken:
  - Taker fee: 0.06% (6 bps)
  - Maker fee: 0.02% (2 bps)
  - Saving: 4 bps per side = 8 bps round-trip

This module wraps order placement to:
  1. Determine if maker enforcement is worthwhile (spread > 2x fee difference)
  2. Submit with post-only flag
  3. On rejection (immediate fill would cross the book), either:
     a. Retry at a more passive price (default), or
     b. Fall back to taker (if urgency is high)

Usage:
    enforcer = MakerEnforcement(exchange_connector, enabled=True)
    result = await enforcer.place_order(symbol, side, qty, mid_price, urgency=0.3)
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MakerResult:
    """Outcome of a maker-enforcement order attempt."""

    success: bool
    fill_price: float
    fill_qty: float               # quantity in USD filled
    is_maker: bool                # True if filled as maker (post-only)
    order_id: str
    attempts: int                 # number of post-only attempts made
    fee_bps: float                # actual fee paid in bps


# ---------------------------------------------------------------------------
# MakerEnforcement
# ---------------------------------------------------------------------------


class MakerEnforcement:
    """
    Wraps raw exchange order placement to preferentially use maker (post-only)
    orders, falling back to taker only when urgency is high or retries are
    exhausted.

    When no real exchange connector is provided (``connector=None``) the class
    operates in simulation mode: post-only orders are assumed to succeed
    immediately at the posted price.  This makes the class safe to use in
    backtesting and unit tests without a live exchange connection.

    Thread/async-safety:
        The class itself holds no mutable state beyond ``__init__`` fields, so
        concurrent ``await place_order()`` calls are safe as long as the
        underlying connector is async-safe.
    """

    FEE_SAVINGS_BPS: float = 4.0          # maker vs taker saving per side
    MIN_SPREAD_BPS: float = 0.5           # post-only is free to attempt
    MAX_PASSIVE_OFFSET_BPS: float = 3.0   # max distance inside book to post
    MAX_RETRIES: int = 5
    RETRY_DELAY: float = 0.2              # seconds between retry attempts

    # Fee constants (Kraken defaults)
    TAKER_FEE_BPS: float = 6.0
    MAKER_FEE_BPS: float = 2.0

    # Exchange-specific fee schedules (maker_bps, taker_bps)
    EXCHANGE_FEES: dict = None  # set in __init__

    def __init__(
        self,
        connector: Optional[Any] = None,
        enabled: bool = True,
        fallback_to_taker: bool = True,
        urgency_threshold: float = 0.4,
        fallback_urgency_low: float = 0.3,
        fallback_urgency_high: float = 0.5,
    ) -> None:
        """
        Parameters
        ----------
        connector:
            Exchange connector object.  Must expose an async method
            ``place_order(symbol, side, quantity_usd, price, post_only)``
            that returns a dict with at least ``{"order_id": ..., "fill_price": ...,
            "fill_qty": ...}`` on success, or raises an exception / returns None
            if a post-only order would cross the book.
            Pass ``None`` to run in simulation mode.
        enabled:
            Master switch.  When False, all orders fall through to taker
            immediately.
        fallback_to_taker:
            When True (default), exhausted retries cause a market/taker order.
            When False, the order is abandoned and MakerResult.success=False.
        urgency_threshold:
            Urgency value (0-1) above which maker enforcement is bypassed.
            0.7 means "use maker unless urgency > 70%".
        """
        self.connector = connector
        self.enabled = enabled
        self.fallback_to_taker = fallback_to_taker
        self.urgency_threshold = urgency_threshold

        # Conditional fallback thresholds
        self.fallback_urgency_low = fallback_urgency_low
        self.fallback_urgency_high = fallback_urgency_high

        # Exchange-specific fee schedules: {exchange: (maker_bps, taker_bps)}
        self.EXCHANGE_FEES = {
            "kraken": (2.0, 6.0),
            "coinbase": (4.0, 6.0),
            "binance": (1.0, 1.0),
            "bybit": (1.0, 6.0),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity_usd: float,
        mid_price: float,
        urgency: float = 0.5,
        spread_bps: Optional[float] = None,
    ) -> MakerResult:
        """
        Place an order using maker enforcement where appropriate.

        Parameters
        ----------
        symbol:        Trading pair, e.g. "BTC/USD".
        side:          "buy" or "sell".
        quantity_usd:  Notional value to trade in USD.
        mid_price:     Current mid-market price.
        urgency:       0=no urgency (always prefer maker), 1=maximum urgency.
        spread_bps:    Current spread in bps.  If None, MIN_SPREAD_BPS is
                       assumed (conservative — maker will be attempted).

        Returns
        -------
        MakerResult with fill details.
        """
        if quantity_usd <= 0:
            raise ValueError(f"quantity_usd must be positive, got {quantity_usd}")
        if mid_price <= 0:
            raise ValueError(f"mid_price must be positive, got {mid_price}")

        effective_spread_bps = spread_bps if spread_bps is not None else self.MIN_SPREAD_BPS

        # Fast-path: maker enforcement disabled or urgency too high
        if not self.enabled or not self.should_use_maker(urgency, effective_spread_bps):
            reason = "disabled" if not self.enabled else f"urgency={urgency:.2f}"
            logger.debug(
                "MakerEnforcement bypassed (%s): placing taker order %s %s qty_usd=%.2f",
                reason, side, symbol, quantity_usd,
            )
            return await self._place_taker(symbol, side, quantity_usd, mid_price)

        # Compute the price at which to post
        post_price = self._compute_post_price(side, mid_price)

        attempts = 0
        passive_offset_bps = 0.0  # additional offset accumulated across retries

        while attempts < self.MAX_RETRIES:
            attempts += 1
            adjusted_price = self._apply_passive_offset(
                side, post_price, mid_price, passive_offset_bps
            )
            logger.debug(
                "MakerEnforcement attempt %d/%d: %s %s @ %.6f (offset=%.2f bps)",
                attempts, self.MAX_RETRIES, side, symbol, adjusted_price, passive_offset_bps,
            )

            result = await self._try_post_only(symbol, side, quantity_usd, adjusted_price)
            if result is not None:
                logger.info(
                    "Maker fill: %s %s qty_usd=%.2f price=%.6f attempt=%d savings_usd=%.4f",
                    side, symbol, quantity_usd, result.fill_price,
                    attempts, self.estimate_savings_usd(quantity_usd),
                )
                result.attempts = attempts
                return result

            # Post-only rejected: move further inside book on next attempt
            passive_offset_bps += self.MAX_PASSIVE_OFFSET_BPS / self.MAX_RETRIES
            logger.debug(
                "Post-only rejected, increasing passive offset to %.2f bps", passive_offset_bps
            )

            if attempts < self.MAX_RETRIES:
                await asyncio.sleep(self.RETRY_DELAY)

        # Exhausted retries — conditional fallback based on urgency
        should_fallback = self._should_fallback_to_taker(urgency)
        if should_fallback:
            logger.warning(
                "MakerEnforcement: max retries (%d) exhausted for %s %s — "
                "falling back to taker order (urgency=%.2f)",
                self.MAX_RETRIES, side, symbol, urgency,
            )
            return await self._place_taker(symbol, side, quantity_usd, mid_price, attempts=attempts)

        logger.error(
            "MakerEnforcement: max retries (%d) exhausted for %s %s and "
            "fallback disabled at urgency=%.2f — order abandoned",
            self.MAX_RETRIES, side, symbol, urgency,
        )
        return MakerResult(
            success=False,
            fill_price=0.0,
            fill_qty=0.0,
            is_maker=False,
            order_id="",
            attempts=attempts,
            fee_bps=0.0,
        )

    def should_use_maker(self, urgency: float, spread_bps: float) -> bool:
        """
        True if maker enforcement is worthwhile given current conditions.

        Conditions for maker use:
          - urgency is below the threshold (not time-sensitive)
          - current spread is wide enough to justify posting passively
        """
        return urgency < self.urgency_threshold and spread_bps >= self.MIN_SPREAD_BPS

    def estimate_savings_usd(self, quantity_usd: float) -> float:
        """
        Estimated fee savings from using maker vs taker for this notional.

        savings = quantity_usd * FEE_SAVINGS_BPS / 10000
        """
        return quantity_usd * self.FEE_SAVINGS_BPS / 10_000.0

    def estimate_fee_savings(self, is_maker: bool, notional: float, exchange: str = "kraken") -> float:
        """
        Calculate actual fee difference between maker and taker for the exchange.

        Parameters
        ----------
        is_maker : bool
            Whether the fill was a maker fill.
        notional : float
            Trade notional value in USD.
        exchange : str
            Exchange name (lowercase).

        Returns
        -------
        float
            Fee savings in USD compared to taker.  Positive means savings,
            zero if filled as taker.
        """
        maker_bps, taker_bps = self.EXCHANGE_FEES.get(
            exchange.lower(), (self.MAKER_FEE_BPS, self.TAKER_FEE_BPS)
        )
        if is_maker:
            return notional * (taker_bps - maker_bps) / 10_000.0
        return 0.0

    def _should_fallback_to_taker(self, urgency: float) -> bool:
        """
        Determine whether to fall back to taker after exhausting maker retries.

        - urgency < fallback_urgency_low (0.3): never fall back — wait for maker
        - urgency > fallback_urgency_high (0.5): always fall back to taker
        - between: use the configured fallback_to_taker default
        """
        if urgency < self.fallback_urgency_low:
            return False
        if urgency > self.fallback_urgency_high:
            return True
        return self.fallback_to_taker

    # ------------------------------------------------------------------
    # FIX 17: Maker fill rate tracking
    # ------------------------------------------------------------------

    def __init_fill_tracker(self) -> None:
        """Lazily initialise the per-symbol fill tracker."""
        if not hasattr(self, "_fill_tracker"):
            self._fill_tracker: dict = {}

    def record_fill(self, symbol: str, is_maker: bool) -> None:
        """Record a fill outcome for the given symbol."""
        self.__init_fill_tracker()
        tracker = self._fill_tracker.setdefault(symbol, {
            "maker_attempts": 0,
            "maker_fills": 0,
            "taker_fallbacks": 0,
        })
        tracker["maker_attempts"] += 1
        if is_maker:
            tracker["maker_fills"] += 1
        else:
            tracker["taker_fallbacks"] += 1

    def get_maker_fill_rate(self, symbol: str) -> float:
        """Return maker fill rate (0.0–1.0) for the symbol."""
        self.__init_fill_tracker()
        tracker = self._fill_tracker.get(symbol)
        if tracker is None or tracker["maker_attempts"] == 0:
            return 1.0
        return tracker["maker_fills"] / tracker["maker_attempts"]

    def should_auto_taker(self, symbol: str, min_trades: int = 50) -> bool:
        """True if maker fill rate is too low and we should switch to taker."""
        self.__init_fill_tracker()
        tracker = self._fill_tracker.get(symbol)
        if tracker is None or tracker["maker_attempts"] < min_trades:
            return False
        rate = tracker["maker_fills"] / max(tracker["maker_attempts"], 1)
        if rate < 0.30:
            logger.warning(
                "MakerEnforcement: fill rate for %s is %.1f%% over %d trades — recommending taker",
                symbol, rate * 100.0, tracker["maker_attempts"],
            )
            return True
        return False

    def get_fill_stats(self) -> dict:
        """Return copy of all per-symbol fill stats."""
        self.__init_fill_tracker()
        return dict(self._fill_tracker)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _try_post_only(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
    ) -> Optional[MakerResult]:
        """
        Attempt a single post-only order.

        Returns a MakerResult on success, or None if the order was rejected
        (would cross the book) or if no connector is available (simulation).

        When the connector is None this method simulates a successful maker
        fill at the posted price — useful for backtesting.
        """
        if self.connector is None:
            # Simulation mode: assume post-only always succeeds
            return MakerResult(
                success=True,
                fill_price=price,
                fill_qty=qty,
                is_maker=True,
                order_id=str(uuid.uuid4()),
                attempts=1,
                fee_bps=self.MAKER_FEE_BPS,
            )

        try:
            response = await self.connector.place_order(
                symbol=symbol,
                side=side,
                quantity_usd=qty,
                price=price,
                post_only=True,
            )
            if response is None:
                # Connector signals post-only rejection by returning None
                return None

            return MakerResult(
                success=True,
                fill_price=float(response.get("fill_price", price)),
                fill_qty=float(response.get("fill_qty", qty)),
                is_maker=True,
                order_id=str(response.get("order_id", str(uuid.uuid4()))),
                attempts=1,
                fee_bps=self.MAKER_FEE_BPS,
            )
        except Exception:
            # Any exception from the connector is treated as a post-only rejection
            logger.debug(
                "_try_post_only: connector raised for %s %s @ %.6f",
                side, symbol, price, exc_info=True,
            )
            return None

    async def _place_taker(
        self,
        symbol: str,
        side: str,
        quantity_usd: float,
        mid_price: float,
        attempts: int = 1,
    ) -> MakerResult:
        """
        Place a market/taker order.  Falls back to simulated fill at mid_price
        when no connector is configured.
        """
        if self.connector is None:
            # Simulation: taker fills at mid (no market impact modelled here)
            return MakerResult(
                success=True,
                fill_price=mid_price,
                fill_qty=quantity_usd,
                is_maker=False,
                order_id=str(uuid.uuid4()),
                attempts=attempts,
                fee_bps=self.TAKER_FEE_BPS,
            )

        try:
            response = await self.connector.place_order(
                symbol=symbol,
                side=side,
                quantity_usd=quantity_usd,
                price=mid_price,
                post_only=False,
            )
            if response is None:
                logger.error(
                    "_place_taker: connector returned None for %s %s", side, symbol
                )
                return MakerResult(
                    success=False,
                    fill_price=0.0,
                    fill_qty=0.0,
                    is_maker=False,
                    order_id="",
                    attempts=attempts,
                    fee_bps=self.TAKER_FEE_BPS,
                )

            return MakerResult(
                success=True,
                fill_price=float(response.get("fill_price", mid_price)),
                fill_qty=float(response.get("fill_qty", quantity_usd)),
                is_maker=False,
                order_id=str(response.get("order_id", str(uuid.uuid4()))),
                attempts=attempts,
                fee_bps=self.TAKER_FEE_BPS,
            )
        except Exception:
            logger.exception(
                "_place_taker: connector raised for %s %s", side, symbol
            )
            return MakerResult(
                success=False,
                fill_price=0.0,
                fill_qty=0.0,
                is_maker=False,
                order_id="",
                attempts=attempts,
                fee_bps=self.TAKER_FEE_BPS,
            )

    def _compute_post_price(self, side: str, mid_price: float) -> float:
        """
        Compute the initial price to post.

        buy:  post at mid_price - (MIN_SPREAD_BPS/2 / 10000) * mid_price
              (inside the bid, slightly below mid)
        sell: post at mid_price + (MIN_SPREAD_BPS/2 / 10000) * mid_price
              (inside the ask, slightly above mid)
        """
        half_spread_fraction = (self.MIN_SPREAD_BPS / 2.0) / 10_000.0
        offset = mid_price * half_spread_fraction
        if side.lower() == "buy":
            return mid_price - offset
        return mid_price + offset

    def _apply_passive_offset(
        self,
        side: str,
        post_price: float,
        mid_price: float,
        extra_offset_bps: float,
    ) -> float:
        """
        Apply an additional passive offset to the post price for retry attempts.

        buy:  move price further down (more passive)
        sell: move price further up   (more passive)

        The total passive offset is capped at MAX_PASSIVE_OFFSET_BPS to avoid
        posting so far inside the book that the order is unlikely to fill.
        """
        capped_bps = min(extra_offset_bps, self.MAX_PASSIVE_OFFSET_BPS)
        additional_offset = mid_price * capped_bps / 10_000.0
        if side.lower() == "buy":
            return post_price - additional_offset
        return post_price + additional_offset
