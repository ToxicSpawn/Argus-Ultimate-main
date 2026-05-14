"""Advanced Adaptive Features Integration.

This module integrates all adaptive components into a unified system
with enhanced features for maximum adaptability.
"""

from __future__ import annotations

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class SentimentSource(Enum):
    NEWS = "news"
    SOCIAL = "social"
    ONCHAIN = "onchain"
    DERIVATIVES = "derivatives"
    API = "api"


@dataclass
class SentimentData:
    source: SentimentSource
    symbol: str
    sentiment: float
    confidence: float
    timestamp: float


class SentimentIntegration:
    def __init__(self):
        self._sentiments: Dict[SentimentSource, deque] = {
            source: deque(maxlen=100) for source in SentimentSource
        }
        self._weights = {
            SentimentSource.NEWS: 0.25,
            SentimentSource.SOCIAL: 0.20,
            SentimentSource.ONCHAIN: 0.20,
            SentimentSource.DERIVATIVES: 0.25,
            SentimentSource.API: 0.10,
        }

    def add_sentiment(
        self,
        source: SentimentSource,
        symbol: str,
        sentiment: float,
        confidence: float = 0.5,
    ) -> None:
        data = SentimentData(
            source=source,
            symbol=symbol,
            sentiment=sentiment,
            confidence=confidence,
            timestamp=time.time(),
        )
        self._sentiments[source].append(data)

    def get_aggregated_sentiment(self, symbol: str) -> float:
        weighted_sum = 0.0
        weight_total = 0.0

        for source, weight in self._weights.items():
            if source not in self._sentiments:
                continue

            recent = [
                s for s in self._sentiments[source]
                if s.symbol == symbol
            ][-10:] if any(s.symbol == symbol for s in self._sentiments[source]) else []

            if not recent:
                continue

            avg_sentiment = np.mean([s.sentiment for s in recent])
            avg_confidence = np.mean([s.confidence for s in recent])

            effective_weight = weight * avg_confidence

            weighted_sum += avg_sentiment * effective_weight
            weight_total += effective_weight

        if weight_total <= 0:
            return 0.0

        return weighted_sum / weight_total

    def get_sentiment_trend(self, symbol: str, source: SentimentSource = None) -> str:
        if source:
            sentiments = [s.sentiment for s in self._sentiments[source] if s.symbol == symbol][-10:]
        else:
            all_sentiments = []
            for src in SentimentSource:
                all_sentiments.extend([
                    s.sentiment for s in self._sentiments[src] if s.symbol == symbol
                ][-5:])
            sentiments = all_sentiments[-10:]

        if len(sentiments) < 2:
            return "neutral"

        recent = np.mean(sentiments[-3:])
        older = np.mean(sentiments[:-3]) if len(sentiments) > 3 else recent

        diff = recent - older

        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"

        return "stable"


class AdaptiveExecutionOptimizer:
    def __init__(self):
        self._execution_history: deque = deque(maxlen=500)
        self._slippage_history: deque = deque(maxlen=200)
        self._fee_history: deque = deque(maxlen=200)

    def record_execution(
        self,
        symbol: str,
        order_type: str,
        requested_price: float,
        filled_price: float,
        fee: float,
    ) -> None:
        slippage = abs(filled_price - requested_price) / requested_price

        self._execution_history.append({
            "symbol": symbol,
            "order_type": order_type,
            "slippage": slippage,
            "fee": fee,
            "timestamp": time.time(),
        })

        self._slippage_history.append(slippage)
        self._fee_history.append(fee)

    def get_optimal_order_type(
        self,
        symbol: str,
        urgency: float,
        volatility: float,
    ) -> str:
        recent = [
            e["slippage"] for e in self._execution_history
            if e["symbol"] == symbol
        ][-20:] if any(e["symbol"] == symbol for e in self._execution_history) else list(self._slippage_history)[-20:]

        if not recent:
            return "market"

        avg_slippage = np.mean(recent)

        if volatility > 1.0 or avg_slippage > 0.001:
            return "limit"

        if urgency > 0.8:
            return "market"

        return "limit"

    def get_expected_slippage(self, symbol: str) -> float:
        recent = list(self._slippage_history)[-20:]
        return np.mean(recent) if recent else 0.0

    def get_optimal_split_count(
        self,
        order_size: float,
        volatility: float,
    ) -> int:
        if volatility > 1.0:
            return min(10, max(3, int(order_size / 10000)))
        elif volatility > 0.5:
            return min(5, max(2, int(order_size / 50000)))
        else:
            return 1


class AdaptiveSignalGenerator:
    def __init__(self):
        self._signal_history: deque = deque(maxlen=500)
        self._thresholds = {
            "strong_buy": 0.7,
            "buy": 0.3,
            "sell": -0.3,
            "strong_sell": -0.7,
        }

    def generate_signal(
        self,
        confidence: float,
        regime: str,
        sentiment: float,
        anomaly_score: float,
    ) -> Dict[str, Any]:
        base_signal = confidence

        sentiment_boost = sentiment * 0.2

        if anomaly_score > 0.7:
            signal_reduction = 0.5
        elif anomaly_score > 0.5:
            signal_reduction = 0.8
        else:
            signal_reduction = 1.0

        if regime in ["volatile"]:
            signal_reduction *= 0.7

        final_signal = (base_signal + sentiment_boost) * signal_reduction

        if final_signal > self._thresholds["strong_buy"]:
            action = "strong_buy"
        elif final_signal > self._thresholds["buy"]:
            action = "buy"
        elif final_signal < self._thresholds["strong_sell"]:
            action = "strong_sell"
        elif final_signal < self._thresholds["sell"]:
            action = "sell"
        else:
            action = "neutral"

        self._signal_history.append({
            "signal": final_signal,
            "action": action,
            "timestamp": time.time(),
        })

        return {
            "signal": final_signal,
            "action": action,
            "confidence": abs(final_signal),
        }


class AdaptivePortfolioRebalancer:
    def __init__(self):
        self._target_allocations: Dict[str, float] = {}
        self._current_allocations: Dict[str, float] = {}
        self._rebalance_threshold = 0.1

    def set_target_allocation(self, symbol: str, allocation: float) -> None:
        self._target_allocations[symbol] = allocation

    def update_current_allocation(self, symbol: str, value: float, total_value: float) -> None:
        if total_value > 0:
            self._current_allocations[symbol] = value / total_value

    def should_rebalance(self) -> bool:
        for symbol, target in self._target_allocations.items():
            current = self._current_allocations.get(symbol, 0.0)
            if abs(current - target) > self._rebalance_threshold:
                return True
        return False

    def get_rebalance_orders(self) -> List[Dict[str, Any]]:
        orders = []

        for symbol, target in self._target_allocations.items():
            current = self._current_allocations.get(symbol, 0.0)
            diff = target - current

            if abs(diff) > self._rebalance_threshold:
                orders.append({
                    "symbol": symbol,
                    "action": "buy" if diff > 0 else "sell",
                    "allocation_change": diff,
                })

        return orders


class AdaptiveRiskManager:
    def __init__(self):
        self._max_risk_per_trade = 0.02
        self._max_daily_risk = 0.06
        self._current_risk = 0.0
        self._daily_risk = 0.0

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        risk_amount: float,
    ) -> float:
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share <= 0:
            return 0.0

        max_size = risk_amount / risk_per_share

        return max_size

    def check_daily_risk_limit(self, additional_risk: float) -> bool:
        return (self._daily_risk + additional_risk) <= self._max_daily_risk

    def record_trade_risk(self, risk: float) -> None:
        self._current_risk = risk
        self._daily_risk += risk

    def reset_daily_risk(self) -> None:
        self._daily_risk = 0.0


class AdvancedAdaptiveEngine:
    """Ultimate adaptive engine combining all features."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        from adaptive.max_adaptive_system import MaxAdaptiveSystem
        self._max_adaptive = MaxAdaptiveSystem(config)

        self._sentiment = SentimentIntegration()
        self._execution = AdaptiveExecutionOptimizer()
        self._signals = AdaptiveSignalGenerator()
        self._rebalancer = AdaptivePortfolioRebalancer()
        self._risk_manager = AdaptiveRiskManager()

        self._enabled = True
        self._last_update = 0.0

    def update(
        self,
        symbol: str,
        price: float,
        volume: float,
        bid: float = 0.0,
        ask: float = 0.0,
        bid_depth: float = 0.0,
        ask_depth: float = 0.0,
        **kwargs,
    ) -> Dict[str, Any]:
        if not self._enabled:
            return {}

        regime_data = self._max_adaptive.update(
            symbol=symbol,
            price=price,
            volume=volume,
            bid=bid,
            ask=ask,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
        )

        sentiment = self._sentiment.get_aggregated_sentiment(symbol)

        signal_data = self._signals.generate_signal(
            confidence=kwargs.get("confidence", 0.5),
            regime=regime_data.get("regime", "unknown"),
            sentiment=sentiment,
            anomaly_score=regime_data.get("volatility", 0.0),
        )

        order_type = self._execution.get_optimal_order_type(
            symbol=symbol,
            urgency=kwargs.get("urgency", 0.5),
            volatility=regime_data.get("volatility", 0.0),
        )

        result = {
            **regime_data,
            "sentiment": sentiment,
            "signal": signal_data,
            "order_type": order_type,
            "params": {
                **regime_data.get("params", {}),
                "position_size_mult": regime_data.get("params", {}).get("position_size_mult", 1.0),
            },
        }

        self._last_update = time.time()

        return result

    def add_sentiment(
        self,
        source: SentimentSource,
        symbol: str,
        sentiment: float,
        confidence: float = 0.5,
    ) -> None:
        self._sentiment.add_sentiment(source, symbol, sentiment, confidence)

    def record_execution(
        self,
        symbol: str,
        order_type: str,
        requested_price: float,
        filled_price: float,
        fee: float,
    ) -> None:
        self._execution.record_execution(
            symbol, order_type, requested_price, filled_price, fee
        )

    def get_adapted_params(self) -> Dict[str, Any]:
        return self._max_adaptive.get_current_params()

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False


def create_adaptive_engine(config: Optional[Dict] = None) -> AdvancedAdaptiveEngine:
    return AdvancedAdaptiveEngine(config)
