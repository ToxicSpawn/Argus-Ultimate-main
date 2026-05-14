"""Real-time streaming feature computation for market, trade, and order book data."""

from __future__ import annotations

import logging
import math
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, List, Tuple
from typing import Any
from collections.abc import Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

_WINDOWS: Dict[str, timedelta] = {
    "1min": timedelta(minutes=1),
    "5min": timedelta(minutes=5),
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class FeatureFrame:
    entity_id: str
    timestamp: datetime
    features: Dict[str, float]
    source: str


@dataclass(slots=True)
class OrderBookSnapshot:
    entity_id: str
    timestamp: datetime
    bids: List[Tuple[float, float]] = field(default_factory=list)
    asks: List[Tuple[float, float]] = field(default_factory=list)


class FeatureComputationEngine:
    """Computes sub-second streaming features across configurable windows."""

    def __init__(self, *, max_points_per_entity: int = 20000) -> None:
        self.max_points_per_entity = max(500, int(max_points_per_entity))
        self._price_history: Dict[str, Deque[Tuple[datetime, float]]] = defaultdict(
            lambda: deque(maxlen=self.max_points_per_entity)
        )
        self._trade_history: Dict[str, Deque[Tuple[datetime, float, float]]] = defaultdict(
            lambda: deque(maxlen=self.max_points_per_entity)
        )
        self._volume_history: Dict[str, Deque[Tuple[datetime, float]]] = defaultdict(
            lambda: deque(maxlen=self.max_points_per_entity)
        )
        self._order_books: Dict[str, OrderBookSnapshot] = {}
        self._lock = threading.RLock()

    def process_market_data(self, payload: Mapping[str, Any]) -> FeatureFrame:
        entity_id = str(payload.get("symbol") or payload.get("entity_id") or "UNKNOWN")
        timestamp = self._coerce_timestamp(payload.get("timestamp"))
        price = float(payload.get("price", payload.get("close", 0.0)))
        volume = float(payload.get("volume", 0.0))

        with self._lock:
            self._price_history[entity_id].append((timestamp, price))
            if volume > 0.0:
                self._volume_history[entity_id].append((timestamp, volume))
            self._trim(entity_id, timestamp)
            features = self._compute_market_features(entity_id, timestamp)
        return FeatureFrame(entity_id=entity_id, timestamp=timestamp, features=features, source="market_data")

    def process_trade_event(self, payload: Mapping[str, Any]) -> FeatureFrame:
        entity_id = str(payload.get("symbol") or payload.get("entity_id") or "UNKNOWN")
        timestamp = self._coerce_timestamp(payload.get("timestamp"))
        price = float(payload.get("price", 0.0))
        size = float(payload.get("size", payload.get("quantity", 0.0)))

        with self._lock:
            self._trade_history[entity_id].append((timestamp, price, size))
            if price > 0.0:
                self._price_history[entity_id].append((timestamp, price))
            if size > 0.0:
                self._volume_history[entity_id].append((timestamp, size))
            self._trim(entity_id, timestamp)
            features = self._compute_market_features(entity_id, timestamp)
            features.update(self._compute_trade_features(entity_id, timestamp))
        return FeatureFrame(entity_id=entity_id, timestamp=timestamp, features=features, source="trade_events")

    def process_order_book(self, payload: Mapping[str, Any]) -> FeatureFrame:
        entity_id = str(payload.get("symbol") or payload.get("entity_id") or "UNKNOWN")
        timestamp = self._coerce_timestamp(payload.get("timestamp"))
        bids = self._coerce_levels(payload.get("bids", []), reverse=True)
        asks = self._coerce_levels(payload.get("asks", []), reverse=False)
        snapshot = OrderBookSnapshot(entity_id=entity_id, timestamp=timestamp, bids=bids, asks=asks)

        with self._lock:
            self._order_books[entity_id] = snapshot
            features = self._compute_order_book_features(snapshot)
        return FeatureFrame(entity_id=entity_id, timestamp=timestamp, features=features, source="order_book")

    def _compute_market_features(self, entity_id: str, timestamp: datetime) -> dict[str, float]:
        features: dict[str, float] = {}
        prices = list(self._price_history[entity_id])
        volumes = list(self._volume_history[entity_id])
        if not prices:
            return features

        price_values = [value for _, value in prices]
        features.update(self._technical_indicators(price_values))

        for window_name, delta in _WINDOWS.items():
            window_prices = self._window_values(prices, timestamp - delta)
            window_volumes = self._window_values(volumes, timestamp - delta)
            if not window_prices:
                continue
            prefix = f"{window_name}_"
            features[f"{prefix}mean"] = self._safe_mean(window_prices)
            features[f"{prefix}std"] = self._safe_std(window_prices)
            features[f"{prefix}p25"] = self._safe_percentile(window_prices, 25)
            features[f"{prefix}p50"] = self._safe_percentile(window_prices, 50)
            features[f"{prefix}p75"] = self._safe_percentile(window_prices, 75)
            features[f"{prefix}return"] = self._window_return(window_prices)
            features[f"{prefix}high"] = float(max(window_prices))
            features[f"{prefix}low"] = float(min(window_prices))
            features[f"{prefix}volume_sum"] = float(sum(window_volumes)) if window_volumes else 0.0
            features[f"{prefix}volume_mean"] = self._safe_mean(window_volumes)
        return features

    def _compute_trade_features(self, entity_id: str, timestamp: datetime) -> dict[str, float]:
        features: dict[str, float] = {}
        trades = list(self._trade_history[entity_id])
        if not trades:
            return features

        for window_name, delta in _WINDOWS.items():
            min_ts = timestamp - delta
            trade_slice = [(price, size) for ts, price, size in trades if ts >= min_ts]
            if not trade_slice:
                continue
            prices = [price for price, _ in trade_slice]
            sizes = [size for _, size in trade_slice]
            prefix = f"trades_{window_name}_"
            features[f"{prefix}count"] = float(len(trade_slice))
            notionals = [price * size for price, size in trade_slice]
            features[f"{prefix}notional"] = float(sum(notionals))
            features[f"{prefix}vwap"] = float(sum(notionals) / max(sum(sizes), 1e-12))
            features[f"{prefix}size_mean"] = self._safe_mean(sizes)
            features[f"{prefix}size_std"] = self._safe_std(sizes)
        return features

    def _compute_order_book_features(self, snapshot: OrderBookSnapshot) -> dict[str, float]:
        features: dict[str, float] = {}
        if not snapshot.bids or not snapshot.asks:
            return features

        best_bid = snapshot.bids[0][0]
        best_ask = snapshot.asks[0][0]
        bid_depth = float(sum(size for _, size in snapshot.bids[:10]))
        ask_depth = float(sum(size for _, size in snapshot.asks[:10]))
        spread = max(best_ask - best_bid, 0.0)
        mid = (best_bid + best_ask) / 2.0 if best_bid > 0.0 and best_ask > 0.0 else 0.0
        imbalance = (bid_depth - ask_depth) / max(bid_depth + ask_depth, 1e-12)

        features["order_book_spread"] = spread
        features["order_book_spread_bps"] = spread / max(mid, 1e-12) * 10000.0 if mid > 0.0 else 0.0
        features["order_book_mid_price"] = mid
        features["order_book_imbalance"] = imbalance
        features["order_book_bid_depth_10"] = bid_depth
        features["order_book_ask_depth_10"] = ask_depth
        features["order_book_depth_ratio"] = bid_depth / max(ask_depth, 1e-12)
        return features

    def _technical_indicators(self, prices: Sequence[float]) -> dict[str, float]:
        if not prices:
            return {}
        features: dict[str, float] = {}
        features["rsi_14"] = self._compute_rsi(prices, 14)
        macd, signal, hist = self._compute_macd(prices)
        features["macd"] = macd
        features["macd_signal"] = signal
        features["macd_hist"] = hist
        bb_mid, bb_upper, bb_lower = self._compute_bollinger(prices, window=20, num_std=2.0)
        features["bollinger_mid"] = bb_mid
        features["bollinger_upper"] = bb_upper
        features["bollinger_lower"] = bb_lower
        features["bollinger_width"] = bb_upper - bb_lower
        return features

    def _trim(self, entity_id: str, now: datetime) -> None:
        cutoff = now - _WINDOWS["1d"]
        for bucket in (self._price_history[entity_id], self._trade_history[entity_id], self._volume_history[entity_id]):
            while bucket and bucket[0][0] < cutoff:
                _ = bucket.popleft()

    def _window_values(self, series: Sequence[tuple[datetime, float]], min_ts: datetime) -> list[float]:
        return [float(value) for ts, value in series if ts >= min_ts]

    def _coerce_levels(self, levels: Iterable[Any], *, reverse: bool) -> List[Tuple[float, float]]:
        result: List[Tuple[float, float]] = []
        for level in levels:
            try:
                if isinstance(level, Mapping):
                    raw_price = level.get("price", 0.0)
                    raw_size = level.get("size", 0.0)
                    price = float(raw_price) if isinstance(raw_price, (int, float, str)) else 0.0
                    size = float(raw_size) if isinstance(raw_size, (int, float, str)) else 0.0
                else:
                    price = float(level[0])
                    size = float(level[1])
            except Exception:
                continue
            if price > 0.0 and size >= 0.0:
                result.append((price, size))
        result.sort(key=lambda item: item[0], reverse=reverse)
        return result

    def _coerce_timestamp(self, raw: Any) -> datetime:
        if isinstance(raw, datetime):
            return raw if raw.tzinfo is not None else raw.replace(tzinfo=timezone.utc)
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        if isinstance(raw, str) and raw:
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                logger.debug("Invalid timestamp %s - using current UTC", raw)
        return _utc_now()

    def _safe_mean(self, values: Sequence[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    def _safe_std(self, values: Sequence[float]) -> float:
        if not values:
            return 0.0
        mean = self._safe_mean(values)
        variance = sum((float(value) - mean) ** 2 for value in values) / len(values)
        return float(math.sqrt(max(variance, 0.0)))

    def _safe_percentile(self, values: Sequence[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(value) for value in values)
        if len(ordered) == 1:
            return ordered[0]
        rank = (len(ordered) - 1) * (percentile / 100.0)
        lower = math.floor(rank)
        upper = math.ceil(rank)
        if lower == upper:
            return ordered[lower]
        weight = rank - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    def _window_return(self, values: Sequence[float]) -> float:
        if len(values) < 2:
            return 0.0
        start = float(values[0])
        end = float(values[-1])
        return (end - start) / max(abs(start), 1e-12)

    def _compute_rsi(self, prices: Sequence[float], period: int) -> float:
        if len(prices) <= period:
            return 50.0
        deltas = [float(prices[idx]) - float(prices[idx - 1]) for idx in range(1, len(prices))]
        gains = [max(delta, 0.0) for delta in deltas]
        losses = [max(-delta, 0.0) for delta in deltas]
        gain_slice = gains[-period:]
        loss_slice = losses[-period:]
        avg_gain = sum(gain_slice) / max(len(gain_slice), 1)
        avg_loss = sum(loss_slice) / max(len(loss_slice), 1)
        if avg_loss <= 1e-12:
            return 100.0 if avg_gain > 0.0 else 50.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    def _compute_macd(self, prices: Sequence[float]) -> tuple[float, float, float]:
        if len(prices) < 2:
            return 0.0, 0.0, 0.0
        ema_fast = self._ema(prices, 12)
        ema_slow = self._ema(prices, 26)
        macd_series = [fast - slow for fast, slow in zip(ema_fast, ema_slow)]
        signal_series = self._ema(macd_series, 9)
        macd_value = macd_series[-1]
        signal_value = signal_series[-1]
        return macd_value, signal_value, macd_value - signal_value

    def _compute_bollinger(self, prices: Sequence[float], *, window: int, num_std: float) -> tuple[float, float, float]:
        if not prices:
            return 0.0, 0.0, 0.0
        tail = list(prices[-window:] if len(prices) >= window else prices)
        mean = self._safe_mean(tail)
        std = self._safe_std(tail)
        return mean, mean + num_std * std, mean - num_std * std

    def _ema(self, values: Sequence[float], span: int) -> list[float]:
        alpha = 2.0 / (span + 1.0)
        ema = [0.0 for _ in values]
        ema[0] = float(values[0])
        for index in range(1, len(values)):
            ema[index] = alpha * float(values[index]) + (1.0 - alpha) * ema[index - 1]
        return ema
