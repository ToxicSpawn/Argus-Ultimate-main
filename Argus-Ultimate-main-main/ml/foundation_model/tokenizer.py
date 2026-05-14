"""Universal tokenisation for market microstructure event streams."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Token:
    """Discrete market token with timing metadata."""

    token_type: str
    value: int
    timestamp: float
    raw_value: float | str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TokenizerConfig:
    """Configuration for a scale-invariant microstructure tokenizer."""

    num_price_bins: int = 256
    num_size_bins: int = 128
    num_spread_bins: int = 128
    num_time_bins: int = 64
    log_clip: float = 8.0
    max_time_delta_seconds: float = 60.0
    average_trade_size: float = 1.0
    price_scale_floor: float = 1e-8
    size_scale_floor: float = 1e-8
    mask_token_id: int = 0

    def __post_init__(self) -> None:
        self.num_price_bins = max(8, int(self.num_price_bins))
        self.num_size_bins = max(8, int(self.num_size_bins))
        self.num_spread_bins = max(8, int(self.num_spread_bins))
        self.num_time_bins = max(8, int(self.num_time_bins))
        self.log_clip = float(max(0.5, self.log_clip))
        self.max_time_delta_seconds = float(max(1.0, self.max_time_delta_seconds))
        self.average_trade_size = float(max(self.size_scale_floor, self.average_trade_size))


class UniversalMicrostructureTokenizer:
    """Converts raw market events into discrete, scale-invariant tokens."""

    ORDER_TYPE_TO_ID = {"limit": 0, "market": 1, "cancel": 2, "modify": 3, "unknown": 4}

    def __init__(self, config: Optional[TokenizerConfig] = None) -> None:
        self.config = config or TokenizerConfig()
        self._last_timestamp: Optional[float] = None

        self._offsets = {
            "price": 0,
            "size": self.config.num_price_bins,
            "spread": self.config.num_price_bins + self.config.num_size_bins,
            "order_type": self.config.num_price_bins + self.config.num_size_bins + self.config.num_spread_bins,
            "time": self.config.num_price_bins + self.config.num_size_bins + self.config.num_spread_bins + len(self.ORDER_TYPE_TO_ID),
        }
        self.vocab_size = self._offsets["time"] + self.config.num_time_bins

    @staticmethod
    def _coerce_timestamp(timestamp: Any) -> float:
        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            return float(timestamp.timestamp())
        return float(timestamp)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(number):
            return default
        return number

    def reset(self) -> None:
        self._last_timestamp = None

    def validate_event(self, event: Mapping[str, object]) -> None:
        if not isinstance(event, Mapping):
            raise TypeError("event must be a mapping")
        if "price" not in event and "mid_price" not in event:
            raise ValueError("event must contain price or mid_price")

    def normalize_scale_invariant(self, value: float, reference: float, *, signed: bool = True) -> float:
        """Apply log-ratio normalisation to preserve scale invariance."""
        reference = max(abs(reference), self.config.price_scale_floor)
        value = self._safe_float(value)
        if signed:
            scaled = math.copysign(math.log1p(abs(value) / reference), value)
        else:
            scaled = math.log1p(max(0.0, value) / reference)
        return max(-self.config.log_clip, min(self.config.log_clip, scaled))

    def discretize_continuous(self, value: float, *, num_bins: int, low: float, high: float) -> int:
        """Map a continuous value into an integer token bin."""
        if high <= low:
            raise ValueError("high must be greater than low")
        clipped = max(low, min(high, float(value)))
        ratio = (clipped - low) / (high - low)
        index = int(round(ratio * (num_bins - 1)))
        return max(0, min(num_bins - 1, index))

    def encode_order_type(self, order_type: str) -> int:
        key = str(order_type or "unknown").strip().lower()
        return self._offsets["order_type"] + self.ORDER_TYPE_TO_ID.get(key, self.ORDER_TYPE_TO_ID["unknown"])

    def encode_time(self, timestamp: float) -> int:
        timestamp = self._safe_float(timestamp)
        if self._last_timestamp is None:
            delta = 0.0
        else:
            delta = max(0.0, timestamp - self._last_timestamp)
        self._last_timestamp = timestamp
        normalized = math.log1p(delta) / math.log1p(self.config.max_time_delta_seconds)
        time_bin = self.discretize_continuous(normalized, num_bins=self.config.num_time_bins, low=0.0, high=1.0)
        return self._offsets["time"] + time_bin

    def tokenize_event(self, event: Mapping[str, Any]) -> List[Token]:
        """Tokenise a market event into price, size, spread, type, and time tokens."""
        self.validate_event(event)
        timestamp = self._coerce_timestamp(event.get("timestamp", 0.0))
        mid_price = self._safe_float(event.get("mid_price", event.get("price", 0.0)), default=1.0)
        price = self._safe_float(event.get("price", mid_price), default=mid_price)
        size = self._safe_float(event.get("size", 0.0))
        spread = self._safe_float(event.get("spread", 0.0))
        if spread <= 0.0:
            bid = self._safe_float(event.get("bid", event.get("best_bid", mid_price)), default=mid_price)
            ask = self._safe_float(event.get("ask", event.get("best_ask", mid_price)), default=mid_price)
            if ask >= bid > 0.0:
                spread = ask - bid
        average_trade_size = self._safe_float(
            event.get("average_trade_size", self.config.average_trade_size),
            default=self.config.average_trade_size,
        )

        relative_price = self.normalize_scale_invariant(price - mid_price, max(mid_price, self.config.price_scale_floor))
        normalized_size = self.normalize_scale_invariant(size, max(average_trade_size, self.config.size_scale_floor), signed=False)
        normalized_spread = self.normalize_scale_invariant(spread, max(mid_price, self.config.price_scale_floor), signed=False)

        price_token = Token(
            token_type="price",
            value=self._offsets["price"] + self.discretize_continuous(
                relative_price,
                num_bins=self.config.num_price_bins,
                low=-self.config.log_clip,
                high=self.config.log_clip,
            ),
            timestamp=timestamp,
            raw_value=price,
        )
        size_token = Token(
            token_type="size",
            value=self._offsets["size"] + self.discretize_continuous(
                normalized_size,
                num_bins=self.config.num_size_bins,
                low=0.0,
                high=self.config.log_clip,
            ),
            timestamp=timestamp,
            raw_value=size,
        )
        spread_token = Token(
            token_type="spread",
            value=self._offsets["spread"] + self.discretize_continuous(
                normalized_spread,
                num_bins=self.config.num_spread_bins,
                low=0.0,
                high=self.config.log_clip,
            ),
            timestamp=timestamp,
            raw_value=spread,
        )
        order_token = Token(
            token_type="order_type",
            value=self.encode_order_type(str(event.get("order_type", "unknown"))),
            timestamp=timestamp,
            raw_value=str(event.get("order_type", "unknown")),
        )
        time_token = Token(
            token_type="time",
            value=self.encode_time(timestamp),
            timestamp=timestamp,
            raw_value=timestamp,
        )
        return [price_token, size_token, spread_token, order_token, time_token]

    def tokenize_sequence(self, events: Iterable[Mapping[str, Any]], *, reset: bool = True) -> List[Token]:
        if reset:
            self.reset()
        tokens: List[Token] = []
        for event in events:
            try:
                tokens.extend(self.tokenize_event(event))
            except Exception:
                logger.exception("Failed to tokenize event: %s", event)
        return tokens

    def to_token_ids(self, events: Iterable[Mapping[str, Any]], *, reset: bool = True) -> List[int]:
        return [token.value for token in self.tokenize_sequence(events, reset=reset)]
