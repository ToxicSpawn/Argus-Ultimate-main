"""Market Condition Monitor and Unified Adaptive Orchestrator.

Features:
- Real-time market data monitoring
- Multi-exchange price aggregation
- Alert system
- Anomaly detection
- Liquidity monitoring
- Order book depth analysis
- Spread monitoring
- Unified adaptive system orchestration
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

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class MarketCondition(Enum):
    NORMAL = "normal"
    VOLATILE = "volatile"
    ILLIQUID = "illiquid"
    TRENDING = "trending"
    CONSOLIDATING = "consolidating"
    ANOMALY = "anomaly"


@dataclass
class MarketAlert:
    level: AlertLevel
    message: str
    symbol: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketConditionSnapshot:
    condition: MarketCondition
    volatility: float
    spread: float
    depth: float
    volume_ratio: float
    trend_strength: float
    anomaly_score: float
    timestamp: float


class PriceMonitor:
    def __init__(self, symbols: List[str] = None):
        self._symbols = symbols or []
        self._prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._volumes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._timestamps: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

    def add_symbol(self, symbol: str) -> None:
        if symbol not in self._symbols:
            self._symbols.append(symbol)

    def update(
        self,
        symbol: str,
        price: float,
        volume: float = 0.0,
    ) -> None:
        self._prices[symbol].append(price)
        self._volumes[symbol].append(volume)
        self._timestamps[symbol].append(time.time())

    def get_price(self, symbol: str) -> Optional[float]:
        if symbol in self._prices and len(self._prices[symbol]) > 0:
            return self._prices[symbol][-1]
        return None

    def get_price_change(
        self,
        symbol: str,
        periods: int = 1,
    ) -> float:
        if symbol not in self._prices or len(self._prices[symbol]) < periods + 1:
            return 0.0

        current = self._prices[symbol][-1]
        previous = self._prices[symbol][-(periods + 1)]

        if previous <= 0:
            return 0.0

        return (current - previous) / previous * 100

    def get_volatility(
        self,
        symbol: str,
        window: int = 20,
    ) -> float:
        if symbol not in self._prices or len(self._prices[symbol]) < window:
            return 0.0

        prices = list(self._prices[symbol])[-window:]
        returns = np.diff(prices) / prices[:-1]

        return float(np.std(returns)) * np.sqrt(252) if len(returns) > 1 else 0.0

    def get_all_prices(self) -> Dict[str, float]:
        return {
            sym: self._prices[sym][-1] if len(self._prices[sym]) > 0 else 0.0
            for sym in self._symbols
        }


class SpreadMonitor:
    def __init__(self, symbols: List[str] = None):
        self._symbols = symbols or []
        self._bids: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._asks: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    def update(self, symbol: str, bid: float, ask: float) -> None:
        self._bids[symbol].append(bid)
        self._asks[symbol].append(ask)

    def get_spread_bps(self, symbol: str) -> float:
        if symbol not in self._bids or symbol not in self._asks:
            return 0.0

        if len(self._bids[symbol]) == 0 or len(self._asks[symbol]) == 0:
            return 0.0

        bid = self._bids[symbol][-1]
        ask = self._asks[symbol][-1]

        if bid <= 0 or ask <= 0:
            return 0.0

        spread = ask - bid
        mid = (bid + ask) / 2

        return (spread / mid) * 10000

    def get_average_spread(self, symbol: str, window: int = 20) -> float:
        spreads = []
        for i in range(min(window, len(self._bids.get(symbol, [])), len(self._asks.get(symbol, [])))):
            bid = self._bids[symbol][-(i + 1)]
            ask = self._asks[symbol][-(i + 1)]
            mid = (bid + ask) / 2
            spread = (ask - bid) / mid * 10000
            spreads.append(spread)

        return np.mean(spreads) if spreads else 0.0


class VolumeMonitor:
    def __init__(self, symbols: List[str] = None):
        self._symbols = symbols or []
        self._volumes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._avg_volume: Dict[str, float] = {}

    def update(self, symbol: str, volume: float) -> None:
        self._volumes[symbol].append(volume)

        if len(self._volumes[symbol]) >= 20:
            self._avg_volume[symbol] = np.mean(list(self._volumes[symbol])[-20:])

    def get_volume_ratio(self, symbol: str) -> float:
        if symbol not in self._volumes or len(self._volumes[symbol]) == 0:
            return 1.0

        current = self._volumes[symbol][-1]
        avg = self._avg_volume.get(symbol, current)

        return current / avg if avg > 0 else 1.0

    def is_volume_spike(self, symbol: str, threshold: float = 2.0) -> bool:
        return self.get_volume_ratio(symbol) > threshold


class AnomalyDetector:
    def __init__(self, window: int = 100):
        self._window = window
        self._prices: deque = deque(maxlen=window)
        self._volumes: deque = deque(maxlen=window)

    def add_observation(
        self,
        price: float,
        volume: float = 0.0,
    ) -> None:
        self._prices.append(price)
        self._volumes.append(volume)

    def detect_price_anomaly(self) -> float:
        if len(self._prices) < 20:
            return 0.0

        recent = list(self._prices)[-20:]
        historical = list(self._prices)[:-20]

        if len(historical) < 10:
            return 0.0

        recent_mean = np.mean(recent)
        historical_mean = np.mean(historical)
        historical_std = np.std(historical)

        if historical_std <= 0:
            return 0.0

        z_score = abs(recent_mean - historical_mean) / historical_std

        return min(1.0, z_score / 3)

    def detect_volume_anomaly(self) -> float:
        if len(self._volumes) < 20:
            return 0.0

        recent = list(self._volumes)[-20:]
        historical = list(self._volumes)[:-20]

        if len(historical) < 10:
            return 0.0

        recent_mean = np.mean(recent)
        historical_mean = np.mean(historical)

        if historical_mean <= 0:
            return 0.0

        ratio = recent_mean / historical_mean

        return min(1.0, abs(ratio - 1) / 2)

    def get_anomaly_score(self) -> float:
        price_anomaly = self.detect_price_anomaly()
        volume_anomaly = self.detect_volume_anomaly()

        return max(price_anomaly, volume_anomaly)


class LiquidityMonitor:
    def __init__(self, symbols: List[str] = None):
        self._symbols = symbols or []
        self._order_book_depth: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    def update_depth(
        self,
        symbol: str,
        bid_depth: float,
        ask_depth: float,
    ) -> None:
        total_depth = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
        self._order_book_depth[symbol].append({
            "total": total_depth,
            "imbalance": imbalance,
        })

    def get_depth(self, symbol: str) -> Dict[str, float]:
        if symbol not in self._order_book_depth or len(self._order_book_depth[symbol]) == 0:
            return {"total": 0.0, "imbalance": 0.0}

        recent = list(self._order_book_depth[symbol])[-5:]
        return {
            "total": np.mean([r["total"] for r in recent]),
            "imbalance": np.mean([r["imbalance"] for r in recent]),
        }

    def is_illiquid(self, symbol: str, threshold: float = 10000.0) -> bool:
        depth = self.get_depth(symbol)
        return depth["total"] < threshold


class MarketConditionMonitor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        self._price_monitor = PriceMonitor()
        self._spread_monitor = SpreadMonitor()
        self._volume_monitor = VolumeMonitor()
        self._anomaly_detector = AnomalyDetector()
        self._liquidity_monitor = LiquidityMonitor()

        self._alerts: deque = deque(maxlen=500)
        self._alert_callbacks: List[Callable] = []

        self._symbols: List[str] = []

    def add_symbol(self, symbol: str) -> None:
        self._symbols.append(symbol)
        self._price_monitor.add_symbol(symbol)
        self._spread_monitor._symbols.append(symbol)
        self._volume_monitor._symbols.append(symbol)
        self._liquidity_monitor._symbols.append(symbol)

    def update(
        self,
        symbol: str,
        price: float,
        volume: float = 0.0,
        bid: float = 0.0,
        ask: float = 0.0,
        bid_depth: float = 0.0,
        ask_depth: float = 0.0,
    ) -> None:
        self._price_monitor.update(symbol, price, volume)

        if bid > 0 and ask > 0:
            self._spread_monitor.update(symbol, bid, ask)

        if volume > 0:
            self._volume_monitor.update(symbol, volume)

        if bid_depth > 0 or ask_depth > 0:
            self._liquidity_monitor.update_depth(symbol, bid_depth, ask_depth)

        self._anomaly_detector.add_observation(price, volume)

        self._check_alerts(symbol)

    def _check_alerts(self, symbol: str) -> None:
        volatility = self._price_monitor.get_volatility(symbol)
        if volatility > 1.5:
            self._create_alert(
                AlertLevel.WARNING,
                f"High volatility detected: {volatility:.2%}",
                symbol,
            )

        spread = self._spread_monitor.get_spread_bps(symbol)
        if spread > 50:
            self._create_alert(
                AlertLevel.WARNING,
                f"Wide spread: {spread:.1f} bps",
                symbol,
            )

        if self._volume_monitor.is_volume_spike(symbol, 3.0):
            self._create_alert(
                AlertLevel.INFO,
                f"Volume spike detected",
                symbol,
            )

        anomaly_score = self._anomaly_detector.get_anomaly_score()
        if anomaly_score > 0.7:
            self._create_alert(
                AlertLevel.CRITICAL,
                f"Anomaly detected: {anomaly_score:.2f}",
                symbol,
            )

        if self._liquidity_monitor.is_illiquid(symbol):
            self._create_alert(
                AlertLevel.WARNING,
                "Low liquidity",
                symbol,
            )

    def _create_alert(
        self,
        level: AlertLevel,
        message: str,
        symbol: str,
    ) -> None:
        alert = MarketAlert(
            level=level,
            message=message,
            symbol=symbol,
            timestamp=time.time(),
        )
        self._alerts.append(alert)

        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.warning(f"Alert callback error: {e}")

    def register_alert_callback(self, callback: Callable[[MarketAlert], None]) -> None:
        self._alert_callbacks.append(callback)

    def get_condition(self, symbol: str) -> MarketConditionSnapshot:
        volatility = self._price_monitor.get_volatility(symbol)
        spread = self._spread_monitor.get_spread_bps(symbol)
        volume_ratio = self._volume_monitor.get_volume_ratio(symbol)
        depth = self._liquidity_monitor.get_depth(symbol)
        anomaly_score = self._anomaly_detector.get_anomaly_score()

        price_change = self._price_monitor.get_price_change(symbol, 10)

        if anomaly_score > 0.7:
            condition = MarketCondition.ANOMALY
        elif volatility > 1.0:
            condition = MarketCondition.VOLATILE
        elif depth["total"] < 10000:
            condition = MarketCondition.ILLIQUID
        elif abs(price_change) > 5:
            condition = MarketCondition.TRENDING
        elif spread < 5 and volume_ratio < 1.5:
            condition = MarketCondition.CONSOLIDATING
        else:
            condition = MarketCondition.NORMAL

        return MarketConditionSnapshot(
            condition=condition,
            volatility=volatility,
            spread=spread,
            depth=depth["total"],
            volume_ratio=volume_ratio,
            trend_strength=abs(price_change),
            anomaly_score=anomaly_score,
            timestamp=time.time(),
        )

    def get_all_conditions(self) -> Dict[str, MarketConditionSnapshot]:
        return {
            symbol: self.get_condition(symbol)
            for symbol in self._symbols
        }

    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        since: Optional[float] = None,
    ) -> List[MarketAlert]:
        alerts = list(self._alerts)

        if level:
            alerts = [a for a in alerts if a.level == level]

        if since:
            alerts = [a for a in alerts if a.timestamp >= since]

        return alerts


class UnifiedAdaptiveOrchestrator:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        self._regime_detector = None
        self._param_optimizer = None
        self._strategy_switcher = None
        self._risk_adapter = None
        self._market_monitor = None

        self._adaptation_enabled = True

        self._state: Dict[str, Any] = {}
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)

    def set_components(
        self,
        regime_detector: Any = None,
        param_optimizer: Any = None,
        strategy_switcher: Any = None,
        risk_adapter: Any = None,
        market_monitor: Any = None,
    ) -> None:
        self._regime_detector = regime_detector
        self._param_optimizer = param_optimizer
        self._strategy_switcher = strategy_switcher
        self._risk_adapter = risk_adapter
        self._market_monitor = market_monitor

    def update(
        self,
        symbol: str,
        price: float,
        volume: float = 0.0,
        bid: float = 0.0,
        ask: float = 0.0,
    ) -> Dict[str, Any]:
        if not self._adaptation_enabled:
            return self._state

        adaptations = {}

        if self._market_monitor:
            self._market_monitor.update(symbol, price, volume, bid, ask)
            condition = self._market_monitor.get_condition(symbol)
            adaptations["market_condition"] = condition.condition.value
            adaptations["volatility"] = condition.volatility

            self._state["last_condition"] = condition

        regime_metrics = None
        if self._regime_detector:
            regime_metrics = self._regime_detector.update(price, volume)
            adaptations["regime"] = regime_metrics.regime.value
            self._state["last_regime"] = regime_metrics

        if self._risk_adapter and regime_metrics:
            risk_state = self._risk_adapter.adapt_to_market(
                regime=regime_metrics.regime.value,
                volatility=regime_metrics.volatility_level,
                drawdown=0.0,
            )
            adaptations["risk_level"] = risk_state.risk_level.value
            self._state["risk_state"] = risk_state

        if self._param_optimizer and regime_metrics:
            params = self._param_optimizer.get_current_params()
            adaptations["parameters"] = params
            self._state["current_params"] = params

        if self._strategy_switcher and regime_metrics:
            active = self._strategy_switcher.select_strategy(
                market_regime=regime_metrics.regime.value
            )
            adaptations["active_strategy"] = active
            self._state["active_strategy"] = active

        self._state["last_update"] = time.time()
        self._state["adaptations"] = adaptations

        return adaptations

    def get_adapted_parameters(
        self,
        symbol: str,
        signal_confidence: float,
    ) -> Dict[str, Any]:
        result = {
            "position_size": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "leverage": 1.0,
        }

        if self._risk_adapter and self._state.get("risk_state"):
            risk_state = self._state["risk_state"]
            result["leverage"] = risk_state.leverage_mult

        if self._param_optimizer and self._state.get("current_params"):
            params = self._state["current_params"]
            result["position_size"] = signal_confidence * params.get("position_size", 1.0)

        return result

    def enable_adaptation(self) -> None:
        self._adaptation_enabled = True

    def disable_adaptation(self) -> None:
        self._adaptation_enabled = False

    def register_callback(
        self,
        event: str,
        callback: Callable,
    ) -> None:
        self._callbacks[event].append(callback)

    def get_state(self) -> Dict[str, Any]:
        return self._state.copy()

    def reset(self) -> None:
        self._state.clear()
        self._adaptation_enabled = True
