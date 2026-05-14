"""
Real-time feature store for ARGUS.

Provides low-latency feature caching, history, derived feature computation,
quality scoring, and lightweight drift/importance analytics for live ML flows.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from core.feature_store import FeatureStore as InMemoryFeatureStore
except Exception:  # pragma: no cover - optional integration fallback
    InMemoryFeatureStore = None


_VALID_FEATURE_TYPES = {"numeric", "categorical", "binary"}
_VALID_SOURCES = {"ohlcv", "orderbook", "sentiment", "derived"}


@dataclass(slots=True)
class FeatureConfig:
    feature_name: str
    feature_type: str
    source: str
    ttl_seconds: int
    refresh_interval: int

    def __post_init__(self) -> None:
        self.feature_name = str(self.feature_name).strip()
        self.feature_type = str(self.feature_type).strip().lower()
        self.source = str(self.source).strip().lower()
        self.ttl_seconds = int(self.ttl_seconds)
        self.refresh_interval = int(self.refresh_interval)

        if not self.feature_name:
            raise ValueError("feature_name must be non-empty")
        if self.feature_type not in _VALID_FEATURE_TYPES:
            raise ValueError(f"Invalid feature_type: {self.feature_type!r}")
        if self.source not in _VALID_SOURCES:
            raise ValueError(f"Invalid source: {self.source!r}")
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        if self.refresh_interval <= 0:
            raise ValueError("refresh_interval must be > 0")


@dataclass(slots=True)
class FeatureValue:
    feature_name: str
    symbol: str
    value: float
    timestamp: datetime
    quality_score: float

    def __post_init__(self) -> None:
        self.feature_name = str(self.feature_name).strip()
        self.symbol = str(self.symbol).strip()
        self.value = float(self.value)
        self.quality_score = float(min(1.0, max(0.0, self.quality_score)))
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)


class RealTimeFeatureStore:
    """TTL-aware, async-capable real-time feature store for live trading."""

    def __init__(
        self,
        *,
        existing_store: Optional[Any] = None,
        max_history: int = 512,
        drift_threshold: float = 2.0,
    ) -> None:
        self.max_history = max(10, int(max_history))
        self.drift_threshold = float(drift_threshold)
        self._lock = threading.RLock()

        self._feature_configs: Dict[str, FeatureConfig] = {}
        self._cache: Dict[str, Dict[str, FeatureValue]] = defaultdict(dict)
        self._feature_versions: Dict[Tuple[str, str], int] = defaultdict(int)
        self._latest_refresh: Dict[Tuple[str, str], datetime] = {}
        self._history: Dict[Tuple[str, str], Deque[FeatureValue]] = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )
        self._derived_dependencies: Dict[str, List[str]] = {
            "returns": ["close", "prev_close"],
            "mid_price": ["bid", "ask"],
            "spread_bps": ["bid", "ask"],
            "orderbook_imbalance": ["bid_size", "ask_size"],
            "sentiment_momentum": ["sentiment_score", "sentiment_prev"],
            "volume_intensity": ["volume", "volume_ma"],
            "price_range_pct": ["high", "low", "close"],
            "composite_alpha": ["close", "prev_close", "bid_size", "ask_size", "sentiment_score", "sentiment_prev"],
        }

        self._store = existing_store
        if self._store is None and InMemoryFeatureStore is not None:
            try:
                self._store = InMemoryFeatureStore(background=True, max_history=1)
            except TypeError:
                self._store = InMemoryFeatureStore(background=True)
            logger.info("RealTimeFeatureStore connected to core.feature_store.FeatureStore")
        elif self._store is None:
            logger.info("RealTimeFeatureStore running without external FeatureStore integration")

        self._loop = asyncio.new_event_loop()
        self._queue: asyncio.Queue[Tuple[str, str]] = asyncio.Queue()
        self._loop_thread = threading.Thread(
            target=self._run_event_loop,
            name="RealTimeFeatureStore-loop",
            daemon=True,
        )
        self._loop_thread.start()
        self._worker_future = asyncio.run_coroutine_threadsafe(self._feature_worker(), self._loop)

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _feature_worker(self) -> None:
        logger.debug("RealTimeFeatureStore async worker started")
        while True:
            symbol, feature_name = await self._queue.get()
            try:
                if symbol == "__shutdown__" and feature_name == "__shutdown__":
                    return
                await self._refresh_feature(symbol, feature_name)
            except Exception:
                logger.exception(
                    "Feature refresh failed for %s/%s", symbol, feature_name,
                )
            finally:
                self._queue.task_done()

    async def _refresh_feature(self, symbol: str, feature_name: str) -> None:
        config = self._feature_configs.get(feature_name)
        if config is None or config.source != "derived":
            return

        base_feature_names = self._derived_dependencies.get(feature_name)
        if not base_feature_names:
            return

        base_features = self.get_features_batch(symbol, base_feature_names)
        if len(base_features) != len(base_feature_names):
            return

        await asyncio.sleep(0)
        derived = self.compute_derived_features(symbol, base_features)
        if feature_name in derived:
            self.update_feature(symbol, feature_name, derived[feature_name])

    def _enqueue_refresh(self, symbol: str, feature_name: str) -> None:
        if feature_name not in self._feature_configs:
            return
        asyncio.run_coroutine_threadsafe(self._queue.put((symbol, feature_name)), self._loop)

    def _is_expired(self, feature_value: FeatureValue, ttl_seconds: int) -> bool:
        return datetime.now(timezone.utc) > feature_value.timestamp + timedelta(seconds=ttl_seconds)

    def _validate_feature_value(self, config: FeatureConfig, value: float) -> float:
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"Non-finite feature value for {config.feature_name}")
        if config.feature_type == "binary" and numeric_value not in (0.0, 1.0):
            raise ValueError(f"Binary feature {config.feature_name} must be 0.0 or 1.0")
        return numeric_value

    def _compute_quality_score(self, config: FeatureConfig, value: float, timestamp: datetime) -> float:
        age_seconds = max(0.0, (datetime.now(timezone.utc) - timestamp).total_seconds())
        freshness = max(0.0, 1.0 - (age_seconds / max(config.ttl_seconds, 1)))
        validity = 1.0 if math.isfinite(value) else 0.0
        type_score = 1.0
        if config.feature_type == "binary":
            type_score = 1.0 if value in (0.0, 1.0) else 0.0
        elif config.feature_type == "categorical":
            type_score = 1.0 if abs(value - round(value)) < 1e-9 else 0.7
        return round(max(0.0, min(1.0, freshness * 0.6 + validity * 0.25 + type_score * 0.15)), 6)

    def _record_value(self, symbol: str, feature_name: str, feature_value: FeatureValue) -> None:
        key = (symbol, feature_name)
        self._cache[symbol][feature_name] = feature_value
        self._history[key].append(feature_value)
        self._feature_versions[key] += 1
        self._latest_refresh[key] = feature_value.timestamp

    def _mirror_to_existing_store(self, symbol: str, feature_name: str, feature_value: FeatureValue) -> None:
        if self._store is None:
            return
        ttl_seconds = self._feature_configs[feature_name].ttl_seconds
        try:
            self._store.set(symbol, feature_name, feature_value.value, ttl_s=ttl_seconds)
        except AttributeError:
            logger.debug("Existing feature store does not support set(symbol, feature, value, ttl_s)")

    def _extract_numeric_value(self, item: Any) -> float:
        if isinstance(item, FeatureValue):
            return float(item.value)
        return float(item)

    def _trigger_dependent_refreshes(self, symbol: str, updated_feature: str) -> None:
        for feature_name, deps in self._derived_dependencies.items():
            if updated_feature in deps and feature_name in self._feature_configs:
                self._enqueue_refresh(symbol, feature_name)

    def register_feature(self, config: FeatureConfig) -> None:
        with self._lock:
            self._feature_configs[config.feature_name] = config
        logger.info(
            "Registered realtime feature %s type=%s source=%s ttl=%ss refresh=%ss",
            config.feature_name,
            config.feature_type,
            config.source,
            config.ttl_seconds,
            config.refresh_interval,
        )

    def get_feature(self, symbol: str, feature_name: str) -> FeatureValue:
        with self._lock:
            config = self._feature_configs.get(feature_name)
            if config is None:
                raise KeyError(f"Feature {feature_name!r} is not registered")

            feature_value = self._cache.get(symbol, {}).get(feature_name)
            if feature_value is None and self._store is not None:
                try:
                    mirrored_value = self._store.get(symbol, feature_name)
                except AttributeError:
                    mirrored_value = None
                if mirrored_value is not None:
                    timestamp = datetime.now(timezone.utc)
                    quality_score = self._compute_quality_score(config, float(mirrored_value), timestamp)
                    feature_value = FeatureValue(
                        feature_name=feature_name,
                        symbol=symbol,
                        value=float(mirrored_value),
                        timestamp=timestamp,
                        quality_score=quality_score,
                    )
                    self._record_value(symbol, feature_name, feature_value)

            if feature_value is None:
                raise KeyError(f"Feature {feature_name!r} not available for symbol {symbol!r}")

            if self._is_expired(feature_value, config.ttl_seconds):
                self._cache[symbol].pop(feature_name, None)
                self._enqueue_refresh(symbol, feature_name)
                raise KeyError(f"Feature {feature_name!r} expired for symbol {symbol!r}")

            elapsed = (datetime.now(timezone.utc) - feature_value.timestamp).total_seconds()
            if elapsed >= config.refresh_interval:
                self._enqueue_refresh(symbol, feature_name)
            return feature_value

    def get_features_batch(self, symbol: str, feature_names: List[str]) -> Dict[str, FeatureValue]:
        features: Dict[str, FeatureValue] = {}
        for feature_name in feature_names:
            try:
                features[feature_name] = self.get_feature(symbol, feature_name)
            except KeyError:
                logger.debug("Batch get missed feature %s for %s", feature_name, symbol)
        return features

    def compute_derived_features(self, symbol: str, base_features: Dict[str, Any]) -> Dict[str, float]:
        del symbol  # reserved for future symbol-specific derived logic
        values = {name: self._extract_numeric_value(item) for name, item in base_features.items()}
        derived: Dict[str, float] = {}

        close = values.get("close")
        prev_close = values.get("prev_close")
        bid = values.get("bid")
        ask = values.get("ask")
        bid_size = values.get("bid_size")
        ask_size = values.get("ask_size")
        sentiment_score = values.get("sentiment_score")
        sentiment_prev = values.get("sentiment_prev")
        high = values.get("high")
        low = values.get("low")
        volume = values.get("volume")
        volume_ma = values.get("volume_ma")

        if close is not None and prev_close not in (None, 0.0):
            derived["returns"] = float((close - prev_close) / prev_close)
        if bid is not None and ask is not None:
            mid = (bid + ask) / 2.0
            derived["mid_price"] = float(mid)
            if mid != 0.0:
                derived["spread_bps"] = float(((ask - bid) / mid) * 10000.0)
        if bid_size is not None and ask_size is not None:
            depth_total = bid_size + ask_size
            if depth_total != 0.0:
                derived["orderbook_imbalance"] = float((bid_size - ask_size) / depth_total)
        if sentiment_score is not None and sentiment_prev is not None:
            derived["sentiment_momentum"] = float(sentiment_score - sentiment_prev)
        if volume is not None and volume_ma not in (None, 0.0):
            derived["volume_intensity"] = float(volume / volume_ma)
        if high is not None and low is not None and close not in (None, 0.0):
            derived["price_range_pct"] = float((high - low) / close)
        if {"returns", "orderbook_imbalance", "sentiment_momentum"}.issubset(derived):
            derived["composite_alpha"] = float(
                0.5 * derived["returns"]
                + 0.3 * derived["orderbook_imbalance"]
                + 0.2 * derived["sentiment_momentum"]
            )

        return derived

    def update_feature(self, symbol: str, feature_name: str, value: float) -> None:
        with self._lock:
            config = self._feature_configs.get(feature_name)
            if config is None:
                raise KeyError(f"Feature {feature_name!r} is not registered")

            validated_value = self._validate_feature_value(config, value)
            timestamp = datetime.now(timezone.utc)
            quality_score = self._compute_quality_score(config, validated_value, timestamp)
            feature_value = FeatureValue(
                feature_name=feature_name,
                symbol=symbol,
                value=validated_value,
                timestamp=timestamp,
                quality_score=quality_score,
            )
            self._record_value(symbol, feature_name, feature_value)
            self._mirror_to_existing_store(symbol, feature_name, feature_value)

        logger.debug(
            "Updated realtime feature %s/%s version=%d quality=%.3f",
            symbol,
            feature_name,
            self._feature_versions[(symbol, feature_name)],
            quality_score,
        )
        self._trigger_dependent_refreshes(symbol, feature_name)

    def get_feature_vector(self, symbol: str, lookback: int) -> np.ndarray:
        lookback = max(1, int(lookback))
        with self._lock:
            ordered_names = list(self._feature_configs.keys())
            if not ordered_names:
                return np.empty((lookback, 0), dtype=float)

            matrix = np.full((lookback, len(ordered_names)), np.nan, dtype=float)
            for col_idx, feature_name in enumerate(ordered_names):
                history = list(self._history.get((symbol, feature_name), []))[-lookback:]
                if not history:
                    continue
                values = np.array([item.value for item in history], dtype=float)
                matrix[-len(values):, col_idx] = values
            return matrix

    def calculate_feature_importance(self, features: Dict, targets: np.ndarray) -> Dict[str, float]:
        targets_arr = np.asarray(targets, dtype=float).ravel()
        if targets_arr.size == 0:
            return {}

        raw_scores: Dict[str, float] = {}
        for feature_name, values in features.items():
            feature_arr = np.asarray(values, dtype=float).ravel()
            usable = min(feature_arr.size, targets_arr.size)
            if usable < 2:
                raw_scores[feature_name] = 0.0
                continue
            x = feature_arr[-usable:]
            y = targets_arr[-usable:]
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 2 or np.std(x[mask]) < 1e-12 or np.std(y[mask]) < 1e-12:
                raw_scores[feature_name] = 0.0
                continue
            raw_scores[feature_name] = float(abs(np.corrcoef(x[mask], y[mask])[0, 1]))

        total = sum(raw_scores.values()) or 1.0
        return {name: round(score / total, 6) for name, score in raw_scores.items()}

    def detect_feature_drift(self, feature_name: str, window_days: int) -> dict:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(window_days)))
        observations: List[FeatureValue] = []
        with self._lock:
            for (symbol, name), history in self._history.items():
                if name != feature_name:
                    continue
                observations.extend(item for item in history if item.timestamp >= cutoff)

        observations.sort(key=lambda item: item.timestamp)
        if len(observations) < 4:
            return {
                "feature_name": feature_name,
                "window_days": int(window_days),
                "observation_count": len(observations),
                "drift_detected": False,
                "drift_score": 0.0,
                "baseline_mean": None,
                "recent_mean": None,
                "baseline_quality": None,
                "recent_quality": None,
            }

        midpoint = len(observations) // 2
        baseline = observations[:midpoint]
        recent = observations[midpoint:]
        baseline_values = np.array([item.value for item in baseline], dtype=float)
        recent_values = np.array([item.value for item in recent], dtype=float)
        baseline_quality = float(np.mean([item.quality_score for item in baseline]))
        recent_quality = float(np.mean([item.quality_score for item in recent]))
        baseline_mean = float(np.mean(baseline_values))
        recent_mean = float(np.mean(recent_values))
        baseline_std = float(np.std(baseline_values))
        denom = max(baseline_std, abs(baseline_mean) * 0.1, 1e-9)
        drift_score = float(abs(recent_mean - baseline_mean) / denom)
        quality_shift = abs(recent_quality - baseline_quality)
        drift_detected = drift_score >= self.drift_threshold or quality_shift >= 0.25

        if drift_detected:
            logger.warning(
                "Feature drift detected for %s score=%.3f quality_shift=%.3f observations=%d",
                feature_name,
                drift_score,
                quality_shift,
                len(observations),
            )

        return {
            "feature_name": feature_name,
            "window_days": int(window_days),
            "observation_count": len(observations),
            "drift_detected": drift_detected,
            "drift_score": round(drift_score, 6),
            "quality_shift": round(quality_shift, 6),
            "baseline_mean": round(baseline_mean, 6),
            "recent_mean": round(recent_mean, 6),
            "baseline_quality": round(baseline_quality, 6),
            "recent_quality": round(recent_quality, 6),
        }

    def get_feature_version(self, symbol: str, feature_name: str) -> int:
        return self._feature_versions.get((symbol, feature_name), 0)

    def close(self) -> None:
        if self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._queue.put(("__shutdown__", "__shutdown__")),
                self._loop,
            )
            if hasattr(self, "_worker_future"):
                try:
                    self._worker_future.result(timeout=1.0)
                except Exception:
                    logger.debug("Realtime feature worker did not stop cleanly", exc_info=True)
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread.is_alive():
            self._loop_thread.join(timeout=1.0)
