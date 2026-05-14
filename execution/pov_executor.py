"""POV (Percentage of Volume) execution algorithm.

This module implements a lightweight async POV executor that integrates with the
existing ``SmartOrderRouter`` while remaining usable with richer market-data and
volatility providers when available.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from execution.smart_order_router import SmartOrderRouter
from core.types import OrderRequest

logger = logging.getLogger(__name__)

_DEFAULT_VOLUME_CURVE = (
    1.30, 1.22, 1.16, 1.08, 1.02, 0.96,
    0.90, 0.86, 0.82, 0.78, 0.76, 0.74,
    0.76, 0.80, 0.86, 0.94, 1.02, 1.12,
    1.22, 1.30, 1.34, 1.28, 1.20, 1.12,
)


@dataclass
class POVConfig:
    """Runtime configuration for the POV executor."""

    participation_rate: float = 0.10
    min_participation_rate: float = 0.05
    max_participation_rate: float = 0.30
    urgency: str = "medium"
    max_duration_minutes: int = 60
    volume_threshold: float = 1.0

    def __post_init__(self) -> None:
        self.participation_rate = float(self.participation_rate)
        self.min_participation_rate = float(self.min_participation_rate)
        self.max_participation_rate = float(self.max_participation_rate)
        self.max_duration_minutes = int(self.max_duration_minutes)
        self.volume_threshold = float(self.volume_threshold)
        self.urgency = str(self.urgency).lower()

        if not 0.05 <= self.participation_rate <= 0.30:
            raise ValueError("participation_rate must be between 0.05 and 0.30")
        if self.min_participation_rate <= 0 or self.max_participation_rate <= 0:
            raise ValueError("participation limits must be positive")
        if self.min_participation_rate > self.max_participation_rate:
            raise ValueError("min_participation_rate cannot exceed max_participation_rate")
        if self.urgency not in {"low", "medium", "high"}:
            raise ValueError("urgency must be one of: low, medium, high")
        if self.max_duration_minutes <= 0:
            raise ValueError("max_duration_minutes must be positive")
        if self.volume_threshold < 0:
            raise ValueError("volume_threshold cannot be negative")


@dataclass
class FillResult:
    """Normalised POV child-order fill result."""

    symbol: str
    side: str
    quantity: float
    price: float
    venue: str
    status: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    participation_rate: float = 0.0
    market_volume: float = 0.0
    spread_bps: float = 0.0
    volatility: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0
    child_order_id: str = ""
    accelerated: bool = False
    anti_gaming_applied: bool = False


class POVExecutor:
    """Async percentage-of-volume execution controller."""

    def __init__(
        self,
        config: Optional[POVConfig] = None,
        smart_order_router: Optional[SmartOrderRouter] = None,
        market_data_feed: Optional[Any] = None,
        volatility_provider: Optional[Any] = None,
        volume_curve: Optional[Sequence[float]] = None,
        sleep_func: Optional[Callable[[float], Awaitable[None]]] = None,
        anti_gaming_seed: Optional[int] = None,
    ) -> None:
        self.config = config or POVConfig()
        self.smart_order_router = smart_order_router or SmartOrderRouter()
        self.market_data_feed = market_data_feed
        self.volatility_provider = volatility_provider
        self.sleep_func = sleep_func or asyncio.sleep
        self.volume_curve = list(volume_curve or _DEFAULT_VOLUME_CURVE)
        self._rng = random.Random(anti_gaming_seed)

        self._fills: List[FillResult] = []
        self._recent_slice_sizes: List[float] = []
        self._execution_state: Dict[str, Any] = {
            "symbol": None,
            "side": None,
            "initial_shares": 0.0,
            "remaining_shares": 0.0,
            "filled_shares": 0.0,
            "child_orders": 0,
            "accelerated_slices": 0,
            "anti_gaming_adjustments": 0,
            "average_participation_rate": 0.0,
            "status": "idle",
            "started_at": None,
            "completed_at": None,
        }

    def calculate_order_size(
        self,
        total_shares: float,
        market_volume: float,
        elapsed_time: float,
    ) -> float:
        """Calculate the next POV child-order size."""
        remaining = max(0.0, float(total_shares))
        observed_volume = max(0.0, float(market_volume))
        if remaining <= 0 or observed_volume < self.config.volume_threshold:
            return 0.0

        base_size = observed_volume * max(
            self.config.min_participation_rate,
            min(self.config.max_participation_rate, self.config.participation_rate),
        )

        curve_multiplier = self._volume_curve_multiplier(elapsed_time)
        target_size = base_size * curve_multiplier

        remaining_time = max(self.config.max_duration_minutes - float(elapsed_time), 0.0)
        if self.should_accelerate(remaining_time, remaining):
            target_size *= 1.35

        target_size = min(remaining, target_size)
        if remaining > 0:
            target_size = max(1.0, target_size)
        return min(remaining, target_size)

    def adaptive_participation(self, current_volatility: float, spread_bps: float) -> float:
        """Adapt participation based on current volatility and spread."""
        rate = self.config.participation_rate
        urgency_boost = {"low": -0.02, "medium": 0.0, "high": 0.03}[self.config.urgency]
        rate += urgency_boost

        volatility = max(0.0, float(current_volatility))
        spread = max(0.0, float(spread_bps))

        if volatility > 0.03:
            rate -= 0.03
        elif volatility < 0.01:
            rate += 0.01

        if spread > 12:
            rate -= 0.03
        elif spread < 4:
            rate += 0.01

        return max(self.config.min_participation_rate, min(self.config.max_participation_rate, rate))

    async def execute_pov(self, total_shares: float, symbol: str, side: str) -> List[FillResult]:
        """Execute an order using POV scheduling and smart routing."""
        total_shares = float(total_shares)
        if total_shares <= 0:
            raise ValueError("total_shares must be positive")
        if side.lower() not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")

        self._reset_execution_state(total_shares, symbol, side)
        start_ts = time.monotonic()
        average_participation_samples: List[float] = []
        remaining = total_shares

        logger.info(
            "Starting POV execution for %s %s %.2f shares at %.2f%% participation",
            side.upper(),
            symbol,
            total_shares,
            self.config.participation_rate * 100,
        )

        for minute_index in range(self.config.max_duration_minutes):
            if remaining <= 0:
                break

            snapshot = await self._get_market_snapshot(symbol)
            market_volume = max(0.0, float(snapshot.get("volume", 0.0)))
            spread_bps = max(0.0, float(snapshot.get("spread_bps", 0.0)))
            volatility = await self._get_current_volatility(symbol, snapshot)

            self.config.participation_rate = self.adaptive_participation(volatility, spread_bps)
            average_participation_samples.append(self.config.participation_rate)

            raw_size = self.calculate_order_size(remaining, market_volume, float(minute_index))
            slice_size, anti_gaming_applied = self._apply_anti_gaming_jitter(raw_size, remaining)

            if slice_size <= 0:
                logger.debug(
                    "Skipping POV slice for %s at minute %d due to low volume %.4f",
                    symbol,
                    minute_index,
                    market_volume,
                )
                if minute_index < self.config.max_duration_minutes - 1:
                    await self.sleep_func(0)
                continue

            remaining_time = self.config.max_duration_minutes - minute_index
            accelerated = self.should_accelerate(remaining_time, remaining)
            child_quantity = max(1, int(math.ceil(min(slice_size, remaining))))

            fills = await self._execute_child_order(
                symbol=symbol,
                side=side.lower(),
                quantity=child_quantity,
                snapshot=snapshot,
                market_volume=market_volume,
                spread_bps=spread_bps,
                volatility=volatility,
                accelerated=accelerated,
                anti_gaming_applied=anti_gaming_applied,
            )

            filled_qty = sum(fill.quantity for fill in fills if fill.status in {"filled", "partial"})
            remaining = max(0.0, remaining - filled_qty)

            self._execution_state["filled_shares"] = total_shares - remaining
            self._execution_state["remaining_shares"] = remaining
            self._execution_state["child_orders"] += 1
            if accelerated:
                self._execution_state["accelerated_slices"] += 1
            if anti_gaming_applied:
                self._execution_state["anti_gaming_adjustments"] += 1

            if remaining <= 0:
                break

            if minute_index < self.config.max_duration_minutes - 1:
                await self.sleep_func(0)

        self._execution_state["average_participation_rate"] = (
            sum(average_participation_samples) / len(average_participation_samples)
            if average_participation_samples
            else 0.0
        )
        self._execution_state["completed_at"] = datetime.utcnow()
        self._execution_state["status"] = "completed" if remaining <= 0 else "expired"

        logger.info(
            "Completed POV execution for %s: filled=%.2f remaining=%.2f status=%s duration=%.2fs",
            symbol,
            self._execution_state["filled_shares"],
            remaining,
            self._execution_state["status"],
            time.monotonic() - start_ts,
        )
        return list(self._fills)

    def should_accelerate(self, remaining_time: float, remaining_shares: float) -> bool:
        """Determine whether the order should accelerate near deadline."""
        if remaining_time <= 0:
            return True
        time_pressure = remaining_time <= max(3, self.config.max_duration_minutes * 0.2)
        size_pressure = remaining_shares > max(1.0, self._execution_state.get("initial_shares", 0.0) * 0.25)
        return bool(time_pressure and size_pressure)

    def get_execution_summary(self) -> Dict[str, Any]:
        """Return summary metrics for the most recent POV execution."""
        fill_count = len(self._fills)
        total_filled = sum(fill.quantity for fill in self._fills if fill.status in {"filled", "partial"})
        notional = sum(fill.quantity * fill.price for fill in self._fills if fill.status in {"filled", "partial"})
        avg_price = notional / total_filled if total_filled > 0 else 0.0
        avg_slippage = (
            sum(fill.slippage for fill in self._fills) / fill_count if fill_count else 0.0
        )
        avg_market_impact = (
            sum(fill.market_impact for fill in self._fills) / fill_count if fill_count else 0.0
        )

        return {
            "symbol": self._execution_state["symbol"],
            "side": self._execution_state["side"],
            "status": self._execution_state["status"],
            "initial_shares": self._execution_state["initial_shares"],
            "filled_shares": total_filled,
            "remaining_shares": self._execution_state["remaining_shares"],
            "fill_count": fill_count,
            "child_orders": self._execution_state["child_orders"],
            "average_fill_price": avg_price,
            "average_participation_rate": self._execution_state["average_participation_rate"],
            "average_slippage": avg_slippage,
            "average_market_impact": avg_market_impact,
            "accelerated_slices": self._execution_state["accelerated_slices"],
            "anti_gaming_adjustments": self._execution_state["anti_gaming_adjustments"],
            "started_at": self._execution_state["started_at"],
            "completed_at": self._execution_state["completed_at"],
        }

    async def _execute_child_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        snapshot: Dict[str, float],
        market_volume: float,
        spread_bps: float,
        volatility: float,
        accelerated: bool,
        anti_gaming_applied: bool,
    ) -> List[FillResult]:
        self._ensure_router_support(symbol)
        self._update_router_market_data(symbol, snapshot)
        order_request = OrderRequest(
            order_id=f"pov_{symbol.replace('/', '_')}_{int(time.time() * 1000)}_{self._execution_state['child_orders']}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type="market",
            strategy="smart_routing",
            constraints={
                "execution_algo": "pov",
                "participation_rate": self.config.participation_rate,
                "accelerated": accelerated,
            },
        )

        plan = await self.smart_order_router.submit_order(order_request)
        results = await self.smart_order_router.execute_plan(plan)
        fills: List[FillResult] = []

        for result in results:
            fill = FillResult(
                symbol=symbol,
                side=side,
                quantity=float(result.executed_quantity),
                price=float(result.executed_price),
                venue=result.venue,
                status=result.status,
                timestamp=result.execution_time,
                participation_rate=self.config.participation_rate,
                market_volume=market_volume,
                spread_bps=spread_bps,
                volatility=volatility,
                slippage=float(result.slippage),
                market_impact=float(result.market_impact),
                child_order_id=result.execution_id,
                accelerated=accelerated,
                anti_gaming_applied=anti_gaming_applied,
            )
            fills.append(fill)
            self._fills.append(fill)

        return fills

    async def _get_market_snapshot(self, symbol: str) -> Dict[str, float]:
        snapshot = await self._maybe_call_provider(self.market_data_feed, symbol)
        if isinstance(snapshot, dict):
            return self._normalise_snapshot(symbol, snapshot)

        router_data = getattr(self.smart_order_router, "market_data", None)
        if router_data is not None:
            volume = 0.0
            best_bid = 0.0
            best_ask = 0.0
            for venue_data in getattr(router_data, "market_data", {}).values():
                venue_snapshot = venue_data.get(symbol)
                if not venue_snapshot:
                    continue
                volume += float(venue_snapshot.get("volume", 0.0))
                bid = float(venue_snapshot.get("bid", 0.0))
                ask = float(venue_snapshot.get("ask", 0.0))
                if bid > best_bid:
                    best_bid = bid
                if best_ask == 0.0 or (ask > 0 and ask < best_ask):
                    best_ask = ask
            if volume > 0:
                spread_bps = 0.0
                if best_bid > 0 and best_ask > 0:
                    mid = (best_bid + best_ask) / 2.0
                    spread_bps = ((best_ask - best_bid) / max(mid, 1e-9)) * 10000.0
                return {
                    "volume": volume,
                    "spread_bps": spread_bps,
                    "bid": best_bid,
                    "ask": best_ask,
                    "last": (best_bid + best_ask) / 2.0 if best_bid and best_ask else max(best_bid, best_ask),
                }

        return {
            "volume": max(self.config.volume_threshold, 1_000.0),
            "spread_bps": 5.0,
            "bid": 100.0,
            "ask": 100.05,
            "last": 100.025,
        }

    async def _get_current_volatility(self, symbol: str, snapshot: Dict[str, float]) -> float:
        provider_value = await self._maybe_call_provider(self.volatility_provider, symbol, snapshot)
        if provider_value is not None:
            try:
                return max(0.0, float(provider_value))
            except (TypeError, ValueError):
                logger.debug("Volatility provider returned non-numeric value for %s", symbol)
        return 0.02

    async def _maybe_call_provider(self, provider: Optional[Any], *args: Any) -> Any:
        if provider is None:
            return None

        candidate = provider
        if hasattr(provider, "get_market_snapshot"):
            candidate = provider.get_market_snapshot
        elif hasattr(provider, "get_realtime_volume"):
            candidate = provider.get_realtime_volume
        elif hasattr(provider, "get_volatility"):
            candidate = provider.get_volatility
        elif hasattr(provider, "get_current_volatility"):
            candidate = provider.get_current_volatility

        result = candidate(*args) if callable(candidate) else candidate
        if inspect.isawaitable(result):
            return await result
        return result

    def _normalise_snapshot(self, symbol: str, snapshot: Dict[str, Any]) -> Dict[str, float]:
        bid = float(snapshot.get("bid", snapshot.get("best_bid", 0.0)) or 0.0)
        ask = float(snapshot.get("ask", snapshot.get("best_ask", 0.0)) or 0.0)
        last = float(snapshot.get("last", snapshot.get("mid_price", 0.0)) or 0.0)
        volume = float(
            snapshot.get("volume", snapshot.get("market_volume", snapshot.get("interval_volume", 0.0))) or 0.0
        )
        spread_bps = snapshot.get("spread_bps")
        if spread_bps is None and bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
            spread_bps = ((ask - bid) / max(mid, 1e-9)) * 10000.0
        return {
            "volume": volume,
            "spread_bps": float(spread_bps or 0.0),
            "bid": bid,
            "ask": ask,
            "last": last or ((bid + ask) / 2.0 if bid and ask else 0.0),
        }

    def _volume_curve_multiplier(self, elapsed_time: float) -> float:
        if not self.volume_curve:
            return 1.0
        index = int(max(0, elapsed_time)) % len(self.volume_curve)
        average_weight = sum(self.volume_curve) / len(self.volume_curve)
        return self.volume_curve[index] / max(average_weight, 1e-9)

    def _apply_anti_gaming_jitter(self, target_size: float, remaining: float) -> tuple[float, bool]:
        if target_size <= 0:
            return 0.0, False

        predictable = False
        if len(self._recent_slice_sizes) >= 2:
            latest = self._recent_slice_sizes[-1]
            previous = self._recent_slice_sizes[-2]
            predictable = abs(latest - previous) < 1e-9 and abs(latest - target_size) < 1e-9

        adjusted = min(remaining, target_size)
        if predictable:
            jitter = self._rng.uniform(0.92, 1.08)
            adjusted = min(remaining, max(1.0, adjusted * jitter))

        self._recent_slice_sizes.append(round(adjusted, 6))
        if len(self._recent_slice_sizes) > 5:
            self._recent_slice_sizes = self._recent_slice_sizes[-5:]
        return adjusted, predictable

    def _ensure_router_support(self, symbol: str) -> None:
        for venue in getattr(self.smart_order_router, "venues", []):
            supported_assets = getattr(venue, "supported_assets", None)
            if isinstance(supported_assets, list) and symbol not in supported_assets:
                supported_assets.append(symbol)

        market_data = getattr(self.smart_order_router, "market_data", None)
        if market_data is None:
            return
        venues = getattr(self.smart_order_router, "venues", [])
        for venue in venues:
            venue_snapshot = getattr(market_data, "market_data", {}).setdefault(venue.venue_id, {})
            venue_snapshot.setdefault(
                symbol,
                {
                    "bid": 100.0,
                    "ask": 100.05,
                    "last": 100.025,
                    "volume": max(1_000.0, self.config.volume_threshold),
                    "spread": 0.05,
                    "liquidity_score": getattr(venue, "liquidity_score", 0.5),
                    "timestamp": datetime.utcnow(),
                },
            )

    def _update_router_market_data(self, symbol: str, snapshot: Dict[str, float]) -> None:
        market_data = getattr(self.smart_order_router, "market_data", None)
        venues = getattr(self.smart_order_router, "venues", [])
        if market_data is None or not venues:
            return

        bid = float(snapshot.get("bid", 100.0) or 100.0)
        ask = float(snapshot.get("ask", max(bid, 100.0)) or max(bid, 100.0))
        last = float(snapshot.get("last", (bid + ask) / 2.0) or (bid + ask) / 2.0)
        volume = float(snapshot.get("volume", max(self.config.volume_threshold, 1_000.0)) or max(self.config.volume_threshold, 1_000.0))
        spread = max(0.0, ask - bid)
        per_venue_volume = volume / max(len(venues), 1)

        for venue in venues:
            venue_snapshot = getattr(market_data, "market_data", {}).setdefault(venue.venue_id, {})
            venue_snapshot[symbol] = {
                "bid": bid,
                "ask": ask,
                "last": last,
                "volume": per_venue_volume,
                "spread": spread,
                "liquidity_score": getattr(venue, "liquidity_score", 0.5),
                "timestamp": datetime.utcnow(),
            }

    def _reset_execution_state(self, total_shares: float, symbol: str, side: str) -> None:
        self._fills = []
        self._recent_slice_sizes = []
        self._execution_state.update(
            {
                "symbol": symbol,
                "side": side.lower(),
                "initial_shares": total_shares,
                "remaining_shares": total_shares,
                "filled_shares": 0.0,
                "child_orders": 0,
                "accelerated_slices": 0,
                "anti_gaming_adjustments": 0,
                "average_participation_rate": 0.0,
                "status": "running",
                "started_at": datetime.utcnow(),
                "completed_at": None,
            }
        )


__all__ = ["POVConfig", "FillResult", "POVExecutor"]
