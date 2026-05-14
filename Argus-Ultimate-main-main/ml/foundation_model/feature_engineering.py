"""Scale-invariant feature engineering for market microstructure."""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FeatureVector:
    """Feature vector derived from a market event."""

    ew_vwap: float
    relative_price: float
    normalized_size: float
    spread_bps: float
    order_flow_imbalance: float
    mid_price: float
    timestamp: float
    symbol: str = "UNKNOWN"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float | str | Dict[str, Any]]:
        return {
            "ew_vwap": self.ew_vwap,
            "relative_price": self.relative_price,
            "normalized_size": self.normalized_size,
            "spread_bps": self.spread_bps,
            "order_flow_imbalance": self.order_flow_imbalance,
            "mid_price": self.mid_price,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class FeatureEngineeringConfig:
    """Configuration for microstructure feature extraction."""

    ew_decay: float = 0.94
    history_size: int = 512
    price_floor: float = 1e-8

    def __post_init__(self) -> None:
        self.ew_decay = float(min(0.999, max(0.5, self.ew_decay)))
        self.history_size = max(16, int(self.history_size))
        self.price_floor = float(max(1e-12, self.price_floor))


class MicrostructureFeatureEngineer:
    """Computes TradeFM-style scale-invariant microstructure features."""

    def __init__(self, ew_decay: float = 0.94, history_size: int = 512, *, config: FeatureEngineeringConfig | None = None) -> None:
        self.config = config or FeatureEngineeringConfig(ew_decay=ew_decay, history_size=history_size)
        self.ew_decay: float = self.config.ew_decay
        self.history_size: int = self.config.history_size
        self._price_volume: deque[tuple[float, float]] = deque(maxlen=self.history_size)
        self._sizes: deque[float] = deque(maxlen=self.history_size)
        self._signed_volume: deque[float] = deque(maxlen=self.history_size)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return number if math.isfinite(number) else default

    def _average_trade_size(self) -> float:
        if not self._sizes:
            return 1.0
        return max(sum(self._sizes) / len(self._sizes), 1e-8)

    def compute_ew_vwap(self, history: Iterable[Mapping[str, object]] | None = None) -> float:
        """Exponentially weighted VWAP over recent trades."""
        if history is not None:
            price_volume = [
                (self._safe_float(item.get("price", 0.0)), self._safe_float(item.get("size", 0.0)))
                for item in history
            ]
        else:
            price_volume = list(self._price_volume)

        if not price_volume:
            return 0.0

        weighted_notional = 0.0
        weighted_volume = 0.0
        weight = 1.0
        for price, volume in reversed(price_volume):
            if volume <= 0.0:
                continue
            weighted_notional += weight * price * volume
            weighted_volume += weight * volume
            weight *= self.ew_decay
        if weighted_volume <= 0.0:
            return 0.0
        return weighted_notional / weighted_volume

    def compute_order_flow_imbalance(self) -> float:
        if not self._signed_volume:
            return 0.0
        buy_volume = sum(max(volume, 0.0) for volume in self._signed_volume)
        sell_volume = abs(sum(min(volume, 0.0) for volume in self._signed_volume))
        total = buy_volume + sell_volume
        if total <= 0.0:
            return 0.0
        return (buy_volume - sell_volume) / total

    def compute_book_imbalance(self, event: Mapping[str, object]) -> float:
        bid_size = max(0.0, self._safe_float(event.get("bid_size", event.get("buy_depth", 0.0))))
        ask_size = max(0.0, self._safe_float(event.get("ask_size", event.get("sell_depth", 0.0))))
        total = bid_size + ask_size
        if total <= 0.0:
            return 0.0
        return (bid_size - ask_size) / total

    def update_state(self, event: Mapping[str, object]) -> None:
        price = self._safe_float(event.get("price", 0.0))
        size = max(0.0, self._safe_float(event.get("size", 0.0)))
        side = str(event.get("side", "buy")).strip().lower()
        sign = 1.0 if side in {"buy", "bid", "long"} else -1.0
        self._price_volume.append((price, size))
        self._sizes.append(size)
        self._signed_volume.append(sign * size)

    def transform(self, event: Mapping[str, object]) -> FeatureVector:
        """Create a scale-invariant feature vector for a single market event."""
        try:
            price = self._safe_float(event.get("price", 0.0))
            bid = self._safe_float(event.get("bid", event.get("best_bid", price)))
            ask = self._safe_float(event.get("ask", event.get("best_ask", price)))
            if bid > 0.0 and ask > 0.0 and ask >= bid:
                mid_price = (bid + ask) / 2.0
            else:
                mid_price = max(price, self.config.price_floor)
            spread = max(0.0, ask - bid) if ask >= bid and bid > 0.0 else 0.0
            avg_size = self._average_trade_size()
            self.update_state(event)
            ew_vwap = self.compute_ew_vwap()
            normalized_size = max(0.0, self._safe_float(event.get("size", 0.0)) / avg_size)
            relative_price = (price - mid_price) / max(mid_price, self.config.price_floor)
            spread_bps = (spread / max(mid_price, self.config.price_floor)) * 10000.0
            trade_imbalance = self.compute_order_flow_imbalance()
            book_imbalance = self.compute_book_imbalance(event)
            imbalance = trade_imbalance if abs(trade_imbalance) >= abs(book_imbalance) else book_imbalance

            return FeatureVector(
                ew_vwap=ew_vwap,
                relative_price=relative_price,
                normalized_size=normalized_size,
                spread_bps=spread_bps,
                order_flow_imbalance=imbalance,
                mid_price=mid_price,
                timestamp=self._safe_float(event.get("timestamp", 0.0)),
                symbol=str(event.get("symbol", "UNKNOWN")),
                metadata={
                    "order_type": str(event.get("order_type", "unknown")),
                    "trade_imbalance": trade_imbalance,
                    "book_imbalance": book_imbalance,
                },
            )
        except Exception as exc:
            logger.exception("Failed to engineer features from event: %s", event)
            raise ValueError(f"Invalid market event for feature engineering: {exc}") from exc

    def transform_sequence(self, events: Iterable[Mapping[str, object]]) -> list[FeatureVector]:
        return [self.transform(event) for event in events]
