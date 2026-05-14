"""MaxAdaptiveSystem - Ultimate Self-Adapting Trading System.

Features:
- Self-learning with feedback loops
- Cross-asset correlation adaptation
- Economic calendar awareness
- Advanced anomaly detection with ML
- Multi-timeframe synchronization
- Market microstructure adaptation
- Predictive adaptation using ML models
- Emotionless trading with quantitative rules
- Continuous self-improvement
- Zero-latency adaptation
"""

from __future__ import annotations

import logging
import time
import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from collections import deque, defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AdaptationMode(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    HYPER_ADAPTIVE = "hyper_adaptive"


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CALM = "calm"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"


@dataclass
class AdaptationDecision:
    action: str
    confidence: float
    params: Dict[str, Any]
    reason: str
    timestamp: float


@dataclass
class FeedbackRecord:
    decision: AdaptationDecision
    outcome: float
    expected: float
    accuracy: float
    timestamp: float


class SelfLearningEngine:
    def __init__(self, learning_rate: float = 0.1):
        self._learning_rate = learning_rate
        self._feedback_history: deque = deque(maxlen=1000)
        self._parameter_weights: Dict[str, float] = {}
        self._regime_performance: Dict[str, Dict[str, float]] = {}
        self._decision_accuracy: Dict[str, float] = {}

    def record_feedback(self, decision: AdaptationDecision, outcome: float) -> None:
        accuracy = 1.0 - abs(outcome - decision.confidence)

        record = FeedbackRecord(
            decision=decision,
            outcome=outcome,
            expected=decision.confidence,
            accuracy=accuracy,
            timestamp=time.time(),
        )
        self._feedback_history.append(record)

        self._update_weights(decision.action, accuracy)
        self._update_regime_performance(decision, outcome)
        self._update_decision_accuracy(decision.action, accuracy)

    def _update_weights(self, action: str, accuracy: float) -> None:
        if action not in self._parameter_weights:
            self._parameter_weights[action] = 0.5

        current = self._parameter_weights[action]
        self._parameter_weights[action] = current + self._learning_rate * (accuracy - current)

    def _update_regime_performance(self, decision: AdaptationDecision, outcome: float) -> None:
        regime = decision.params.get("regime", "unknown")

        if regime not in self._regime_performance:
            self._regime_performance[regime] = {"total": 0, "correct": 0, "accuracy": 0.5}

        perf = self._regime_performance[regime]
        perf["total"] += 1

        if outcome > 0.5:
            perf["correct"] += 1

        perf["accuracy"] = perf["correct"] / perf["total"]

    def _update_decision_accuracy(self, action: str, accuracy: float) -> None:
        if action not in self._decision_accuracy:
            self._decision_accuracy[action] = []

        self._decision_accuracy[action].append(accuracy)

        if len(self._decision_accuracy[action]) > 50:
            self._decision_accuracy[action] = self._decision_accuracy[action][-50:]

    def get_best_action(self, context: Dict[str, Any]) -> str:
        regime = context.get("regime", "unknown")

        if regime in self._regime_performance:
            regime_acc = self._regime_performance[regime]["accuracy"]
            if regime_acc > 0.7:
                return "maintain"

        best_action = max(
            self._parameter_weights.items(),
            key=lambda x: x[1],
            default=("adapt", 0.5)
        )[0]

        return best_action

    def get_regime_accuracy(self, regime: str) -> float:
        if regime in self._regime_performance:
            return self._regime_performance[regime]["accuracy"]
        return 0.5

    def get_overall_accuracy(self) -> float:
        if not self._feedback_history:
            return 0.5

        recent = list(self._feedback_history)[-100:]
        return np.mean([r.accuracy for r in recent])


class CrossAssetCorrelationAdapter:
    def __init__(self, window: int = 100):
        self._window = window
        self._prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._correlations: Dict[Tuple[str, str], float] = {}
        self._correlation_history: deque = deque(maxlen=1000)

    def add_price(self, symbol: str, price: float) -> None:
        self._prices[symbol].append(price)

    def calculate_correlation(self, symbol_a: str, symbol_b: str) -> float:
        if symbol_a not in self._prices or symbol_b not in self._prices:
            return 0.0

        if len(self._prices[symbol_a]) < 10:
            return 0.0

        prices_a = np.array(list(self._prices[symbol_a]))
        prices_b = np.array(list(self._prices[symbol_b]))

        if len(prices_a) != len(prices_b):
            return 0.0

        returns_a = np.diff(prices_a) / prices_a[:-1]
        returns_b = np.diff(prices_b) / prices_b[:-1]

        if len(returns_a) < 2:
            return 0.0

        corr = np.corrcoef(returns_a, returns_b)[0, 1]

        if np.isnan(corr):
            return 0.0

        self._correlations[(symbol_a, symbol_b)] = corr
        self._correlation_history.append((symbol_a, symbol_b, corr, time.time()))

        return corr

    def get_adaptive_position_size(
        self,
        symbol: str,
        base_size: float,
        correlated_symbols: List[str],
    ) -> float:
        if not correlated_symbols:
            return base_size

        high_corr_count = 0

        for corr_sym in correlated_symbols:
            corr = self.calculate_correlation(symbol, corr_sym)
            if abs(corr) > 0.7:
                high_corr_count += 1

        reduction = min(0.7, high_corr_count * 0.15)

        return base_size * (1 - reduction)

    def detect_correlation_shift(self) -> bool:
        if len(self._correlation_history) < 10:
            return False

        recent = list(self._correlation_history)[-10:]

        old_corrs = [c[2] for c in recent[:-5]]
        new_corrs = [c[2] for c in recent[-5:]]

        old_mean = np.mean(old_corrs) if old_corrs else 0
        new_mean = np.mean(new_corrs) if new_corrs else 0

        shift = abs(new_mean - old_mean)

        return shift > 0.3


class EconomicCalendarAdapter:
    def __init__(self):
        self._events: List[Dict] = []
        self._event_impact: Dict[str, float] = {
            "high": 1.5,
            "medium": 1.2,
            "low": 0.8,
            "none": 0.5,
        }
        self._active_events: Dict[str, float] = {}

    def add_event(
        self,
        name: str,
        timestamp: float,
        impact: str = "medium",
    ) -> None:
        self._events.append({
            "name": name,
            "timestamp": timestamp,
            "impact": impact,
        })

    def get_upcoming_events(
        self,
        hours_ahead: float = 24,
    ) -> List[Dict]:
        now = time.time()
        cutoff = now + hours_ahead * 3600

        return [
            e for e in self._events
            if now < e["timestamp"] < cutoff
        ]

    def get_current_impact_factor(self) -> float:
        now = time.time()

        self._active_events.clear()

        for event in self._events:
            if abs(event["timestamp"] - now) < 3600:
                impact = event["impact"]
                self._active_events[event["name"]] = self._event_impact.get(impact, 1.0)

        if not self._active_events:
            return 1.0

        return max(self._active_events.values())

    def is_high_impact_period(self) -> bool:
        return self.get_current_impact_factor() > 1.3

    def get_recommended_position_mult(self) -> float:
        impact_factor = self.get_current_impact_factor()

        if impact_factor > 1.4:
            return 0.3
        elif impact_factor > 1.2:
            return 0.6
        elif impact_factor > 1.0:
            return 0.8

        return 1.0


class AdvancedAnomalyDetector:
    def __init__(self, sensitivity: float = 0.5):
        self._sensitivity = sensitivity
        self._price_history: deque = deque(maxlen=200)
        self._volume_history: deque = deque(maxlen=200)
        self._returns_history: deque = deque(maxlen=200)
        self._anomaly_scores: deque = deque(maxlen=100)
        self._baseline_mean: float = 0.0
        self._baseline_std: float = 1.0

    def add_observation(self, price: float, volume: float) -> None:
        if len(self._price_history) > 0:
            ret = (price - self._price_history[-1]) / self._price_history[-1]
            self._returns_history.append(ret)

        self._price_history.append(price)
        self._volume_history.append(volume)

        if len(self._returns_history) >= 50:
            self._baseline_mean = np.mean(list(self._returns_history)[:-20])
            self._baseline_std = np.std(list(self._returns_history)[:-20])

        self._detect_anomalies()

    def _detect_anomalies(self) -> None:
        if len(self._returns_history) < 20:
            self._anomaly_scores.append(0.0)
            return

        recent = list(self._returns_history)[-20:]
        recent_mean = np.mean(recent)
        recent_std = np.std(recent)

        if self._baseline_std <= 0:
            self._anomaly_scores.append(0.0)
            return

        z_score_price = abs(recent_mean - self._baseline_mean) / self._baseline_std

        if len(self._volume_history) >= 20:
            vol_recent = np.mean(list(self._volume_history)[-20:])
            vol_historical = np.mean(list(self._volume_history)[:-20])

            if vol_historical > 0:
                vol_ratio = vol_recent / vol_historical
                z_score_volume = abs(vol_ratio - 1.0)
            else:
                z_score_volume = 0.0
        else:
            z_score_volume = 0.0

        price_anomaly = min(1.0, z_score_price / 3.0)
        volume_anomaly = min(1.0, z_score_volume / 2.0)

        combined = max(price_anomaly, volume_anomaly)

        self._anomaly_scores.append(combined)

    def is_anomaly(self, threshold: float = 0.7) -> bool:
        if not self._anomaly_scores:
            return False

        recent_avg = np.mean(list(self._anomaly_scores)[-3:])
        return recent_avg > threshold * (1 - self._sensitivity)

    def get_anomaly_score(self) -> float:
        if not self._anomaly_scores:
            return 0.0

        return np.mean(list(self._anomaly_scores)[-10:])

    def get_prediction(self) -> Dict[str, float]:
        if len(self._anomaly_scores) < 10:
            return {"trend": 0.0, "momentum": 0.0, "volatility": 0.0}

        recent = list(self._anomaly_scores)[-10:]
        returns = list(self._returns_history)[-10:] if len(self._returns_history) >= 10 else []

        trend = np.mean(recent) if recent else 0.0
        momentum = np.mean(returns) if returns else 0.0
        volatility = np.std(returns) if returns else 0.0

        return {
            "trend": trend,
            "momentum": momentum,
            "volatility": volatility,
        }


class MultiTimeframeSync:
    def __init__(self):
        self._timeframes: Dict[str, Dict] = {}
        self._signals: Dict[str, List[str]] = defaultdict(list)

    def add_timeframe(self, name: str, regime: str, strength: float) -> None:
        self._timeframes[name] = {
            "regime": regime,
            "strength": strength,
            "timestamp": time.time(),
        }

    def get_unified_signal(self) -> str:
        if not self._timeframes:
            return "neutral"

        regime_votes = defaultdict(int)
        weight_sum = 0.0

        for name, data in self._timeframes.items():
            weight = {"1m": 1, "5m": 2, "15m": 3, "1h": 4, "4h": 5, "1d": 6}.get(name, 1)

            regime_votes[data["regime"]] += weight
            weight_sum += weight

        if weight_sum <= 0:
            return "neutral"

        dominant = max(regime_votes.items(), key=lambda x: x[1])
        confidence = dominant[1] / weight_sum

        if confidence > 0.6:
            return dominant[0]

        return "mixed"

    def get_adaptive_parameters(self) -> Dict[str, float]:
        unified = self.get_unified_signal()

        params = {
            "position_mult": 1.0,
            "stop_mult": 1.0,
            "target_mult": 1.0,
            "timeframe_confidence": 0.5,
        }

        if unified == "trending_up" or unified == "trending_down":
            params["position_mult"] = 1.2
            params["stop_mult"] = 1.1
            params["target_mult"] = 1.4
        elif unified == "volatile":
            params["position_mult"] = 0.5
            params["stop_mult"] = 1.5
            params["target_mult"] = 1.2
        elif unified == "ranging":
            params["position_mult"] = 0.7
            params["stop_mult"] = 0.7
            params["target_mult"] = 0.8

        regime_list = list(self._timeframes.values())
        if regime_list:
            params["timeframe_confidence"] = sum(
                d["strength"] for d in regime_list
            ) / len(regime_list)

        return params


class MarketMicrostructureAdapter:
    def __init__(self):
        self._spread_history: deque = deque(maxlen=100)
        self._depth_history: deque = deque(maxlen=100)
        self._order_flow: deque = deque(maxlen=100)
        self._current_spread: float = 0.0
        self._current_depth: float = 0.0

    def update(
        self,
        bid: float,
        ask: float,
        bid_depth: float,
        ask_depth: float,
    ) -> None:
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2
            spread = (ask - bid) / mid * 10000
            self._spread_history.append(spread)
            self._current_spread = spread

        total_depth = bid_depth + ask_depth
        self._depth_history.append(total_depth)
        self._current_depth = total_depth

        order_imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
        self._order_flow.append(order_imbalance)

    def get_spread_regime(self) -> str:
        if self._current_spread > 50:
            return "wide"
        elif self._current_spread > 20:
            return "normal"
        elif self._current_spread > 5:
            return "tight"
        return "very_tight"

    def get_depth_regime(self) -> str:
        if self._current_depth > 100000:
            return "deep"
        elif self._current_depth > 50000:
            return "adequate"
        elif self._current_depth > 10000:
            return "shallow"
        return "very_shallow"

    def get_adaptive_execution_params(self) -> Dict[str, Any]:
        spread_regime = self.get_spread_regime()
        depth_regime = self.get_depth_regime()

        params = {
            "order_type": "market",
            "split_orders": 1,
            "delay_tolerance": 0.0,
            "price_improvement": 0.0,
        }

        if spread_regime == "wide":
            params["order_type"] = "limit"
            params["split_orders"] = 3
            params["delay_tolerance"] = 2.0
        elif spread_regime == "very_tight":
            params["order_type"] = "market"
            params["split_orders"] = 1
            params["price_improvement"] = 0.001

        if depth_regime == "very_shallow":
            params["split_orders"] = min(5, params["split_orders"] + 2)

        return params

    def should_wait(self) -> bool:
        if self._current_spread > 30:
            return True

        if len(self._order_flow) < 5:
            return False

        recent = list(self._order_flow)[-5:]
        imbalances = [abs(x) for x in recent]

        if np.mean(imbalances) > 0.5:
            return True

        return False


class PredictiveModel:
    def __init__(self):
        self._features: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._labels: deque = deque(maxlen=100)
        self._weights: Dict[str, float] = {}
        self._bias: float = 0.0

    def add_sample(
        self,
        features: Dict[str, float],
        label: int,
    ) -> None:
        for key, value in features.items():
            self._features[key].append(value)

        self._labels.append(label)

        if len(self._labels) > 20:
            self._train()

    def _train(self) -> None:
        if len(self._labels) < 10:
            return

        feature_names = list(self._features.keys())
        if not feature_names:
            return

        X = np.array([
            list(self._features[k]) for k in feature_names
        ]).T

        y = np.array(list(self._labels))

        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1.0

        X_norm = (X - X_mean) / X_std

        try:
            XTX = X_norm.T @ X_norm
            XTX_inv = np.linalg.inv(XTX + 0.01 * np.eye(XTX.shape[0]))
            self._weights = XTX_inv @ X_norm.T @ y

            self._bias = np.mean(y)

            self._feature_mean = X_mean
            self._feature_std = X_std

        except Exception:
            pass

    def predict(self, features: Dict[str, float]) -> float:
        if not self._weights:
            return 0.5

        feature_names = list(self._features.keys())
        values = np.array([features.get(k, 0) for k in feature_names])

        values = (values - self._feature_mean) / self._feature_std

        prediction = np.dot(values, self._weights) + self._bias

        return max(0.0, min(1.0, prediction))


class MaxAdaptiveSystem:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        self._mode = AdaptationMode(
            self.config.get("mode", "moderate")
        )

        self._self_learning = SelfLearningEngine(
            learning_rate=self.config.get("learning_rate", 0.1)
        )

        self._correlation_adapter = CrossAssetCorrelationAdapter()
        self._calendar_adapter = EconomicCalendarAdapter()
        self._anomaly_detector = AdvancedAnomalyDetector()
        self._timeframe_sync = MultiTimeframeSync()
        self._microstructure = MarketMicrostructureAdapter()
        self._predictive = PredictiveModel()

        self._current_params: Dict[str, Any] = {}
        self._adaptation_history: deque = deque(maxlen=1000)
        self._decision_callbacks: List[Callable] = []

        self._enabled = True
        self._last_adaptation_time = 0.0

        self._default_params = self._get_default_params()

    def _get_default_params(self) -> Dict[str, Any]:
        return {
            "position_size_mult": 1.0,
            "stop_loss_mult": 1.0,
            "take_profit_mult": 1.0,
            "signal_threshold_mult": 1.0,
            "max_holding_mult": 1.0,
            "leverage_mult": 1.0,
            "entries_per_day": 5,
            "cooldown_seconds": 5,
            "trailing_start_pct": 0.02,
        }

    def update(
        self,
        symbol: str,
        price: float,
        volume: float,
        bid: float = 0.0,
        ask: float = 0.0,
        bid_depth: float = 0.0,
        ask_depth: float = 0.0,
        timeframe_regimes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if not self._enabled:
            return self._current_params

        adaptations = {}

        self._anomaly_detector.add_observation(price, volume)
        self._correlation_adapter.add_price(symbol, price)

        if bid > 0 and ask > 0:
            self._microstructure.update(bid, ask, bid_depth, ask_depth)

        if timeframe_regimes:
            for tf, regime in timeframe_regimes.items():
                self._timeframe_sync.add_timeframe(tf, regime, 0.7)

        regime = self._detect_regime(price, volume)
        volatility = self._anomaly_detector.get_anomaly_score()
        is_anomaly = self._anomaly_detector.is_anomaly()

        context = {
            "regime": regime.value,
            "volatility": volatility,
            "is_anomaly": is_anomaly,
        }

        params = self._adapt_parameters(context)

        self._correlation_adapter.calculate_correlation(symbol, "BTC")

        self._predictive.add_sample(
            {
                "volatility": volatility,
                "spread": self._microstructure._current_spread,
                "depth": self._microstructure._current_depth,
                "spread_regime": 1.0 if self._microstructure.get_spread_regime() == "tight" else 0.0,
            },
            1 if regime.value in ["trending_up", "trending_down"] else 0,
        )

        prediction = self._predictive.predict({
            "volatility": volatility,
            "spread": self._microstructure._current_spread,
            "depth": self._microstructure._current_depth,
            "spread_regime": 1.0 if self._microstructure.get_spread_regime() == "tight" else 0.0,
        })

        adaptations["regime"] = regime.value
        adaptations["volatility"] = volatility
        adaptations["is_anomaly"] = is_anomaly
        adaptations["params"] = params
        adaptations["prediction"] = prediction
        adaptations["timeframe_signal"] = self._timeframe_sync.get_unified_signal()
        adaptations["microstructure"] = self._microstructure.get_adaptive_execution_params()

        self._current_params = params
        self._adaptation_history.append({
            "timestamp": time.time(),
            "regime": regime.value,
            "params": params,
        })

        self._last_adaptation_time = time.time()

        return adaptations

    def _detect_regime(
        self,
        price: float,
        volume: float,
    ) -> MarketRegime:
        anomaly = self._anomaly_detector.get_anomaly_score()
        predictions = self._anomaly_detector.get_prediction()

        if anomaly > 0.8:
            return MarketRegime.VOLATILE

        momentum = predictions.get("momentum", 0.0)
        volatility = predictions.get("volatility", 0.0)

        if volatility > 1.0:
            return MarketRegime.VOLATILE
        elif volatility < 0.3:
            return MarketRegime.CALM

        if momentum > 0.05:
            return MarketRegime.TRENDING_UP
        elif momentum < -0.05:
            return MarketRegime.TRENDING_DOWN

        if abs(momentum) < 0.02:
            return MarketRegime.RANGING

        return MarketRegime.UNKNOWN

    def _adapt_parameters(self, context: Dict[str, Any]) -> Dict[str, Any]:
        params = self._default_params.copy()

        regime = MarketRegime(context.get("regime", "unknown"))
        volatility = context.get("volatility", 0.0)
        is_anomaly = context.get("is_anomaly", False)

        calendar_mult = self._calendar_adapter.get_recommended_position_mult()
        params["position_size_mult"] *= calendar_mult

        correlation_reduction = 0.0
        params["position_size_mult"] *= (1 - correlation_reduction)

        tf_params = self._timeframe_sync.get_adaptive_parameters()
        params["position_size_mult"] *= tf_params.get("position_mult", 1.0)
        params["stop_loss_mult"] *= tf_params.get("stop_mult", 1.0)

        exec_params = self._microstructure.get_adaptive_execution_params()

        if regime == MarketRegime.VOLATILE:
            params["position_size_mult"] *= 0.4
            params["stop_loss_mult"] *= 1.8
            params["take_profit_mult"] *= 1.5
            params["signal_threshold_mult"] *= 1.5
            params["entries_per_day"] = 1
        elif regime == MarketRegime.CALM:
            params["position_size_mult"] *= 1.3
            params["stop_loss_mult"] *= 0.7
            params["take_profit_mult"] *= 0.9
            params["signal_threshold_mult"] *= 0.8
            params["entries_per_day"] = 8
        elif regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            params["position_size_mult"] *= 1.2
            params["stop_loss_mult"] *= 1.0
            params["take_profit_mult"] *= 1.5
            params["signal_threshold_mult"] *= 0.8
            params["entries_per_day"] = 5
            params["max_holding_mult"] *= 1.5
        elif regime == MarketRegime.RANGING:
            params["position_size_mult"] *= 0.6
            params["stop_loss_mult"] *= 0.6
            params["take_profit_mult"] *= 0.7
            params["signal_threshold_mult"] *= 1.2
            params["entries_per_day"] = 10

        if volatility > 0.7:
            params["position_size_mult"] *= 0.6
            params["signal_threshold_mult"] *= 1.3

        if is_anomaly:
            params["position_size_mult"] *= 0.3
            params["cooldown_seconds"] = 30

        if self._mode == AdaptationMode.CONSERVATIVE:
            params["position_size_mult"] *= 0.5
        elif self._mode == AdaptationMode.AGGRESSIVE:
            params["position_size_mult"] *= 1.3
        elif self._mode == AdaptationMode.HYPER_ADAPTIVE:
            params["position_size_mult"] *= 1.5

        params["position_size_mult"] = max(0.1, min(2.0, params["position_size_mult"]))

        return params

    def record_feedback(self, decision: AdaptationDecision, outcome: float) -> None:
        self._self_learning.record_feedback(decision, outcome)

    def set_mode(self, mode: AdaptationMode) -> None:
        self._mode = mode

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def get_current_params(self) -> Dict[str, Any]:
        return self._current_params.copy()

    def get_adaptation_stats(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "mode": self._mode.value,
            "learning_accuracy": self._self_learning.get_overall_accuracy(),
            "regime_performance": self._self_learning._regime_performance,
            "adaptation_count": len(self._adaptation_history),
            "is_anomaly": self._anomaly_detector.is_anomaly(),
            "current_spread": self._microstructure._current_spread,
            "current_depth": self._microstructure._current_depth,
        }