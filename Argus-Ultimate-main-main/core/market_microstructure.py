"""Market Microstructure Analysis Module.

Features:
- Order book analysis
- Spread and depth monitoring
- Market impact modeling
- Liquidity detection
- Price impact estimation
- Microstructure regime detection
- Optimal execution strategies
"""

from __future__ import annotations

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class MicrostructureRegime(Enum):
    NORMAL = "normal"
    TURBULENT = "turbulent"
    ILLIQUID = "illiquid"
    FLASH = "flash"
    AUCTION = "auction"


@dataclass
class OrderBookLevel:
    price: float
    quantity: float
    orders: int = 0


@dataclass
class OrderBook:
    symbol: str
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class MicrostructureMetrics:
    spread_bps: float = 0.0
    mid_price: float = 0.0
    depth_imbalance: float = 0.0
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    realized_spread: float = 0.0
    effective_spread: float = 0.0
    price_impact: float = 0.0
    order_flow_imbalance: float = 0.0
    volatility: float = 0.0
    regime: MicrostructureRegime = MicrostructureRegime.NORMAL


class OrderBookAnalyzer:
    def __init__(self, n_levels: int = 10):
        self._n_levels = n_levels
        self._current_book: Optional[OrderBook] = None
        self._prev_book: Optional[OrderBook] = None
        self._book_history: deque = deque(maxlen=1000)

    def update_book(
        self,
        symbol: str,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> OrderBook:
        book = OrderBook(
            symbol=symbol,
            bids=[OrderBookLevel(p, q) for p, q in bids[:self._n_levels]],
            asks=[OrderBookLevel(p, q) for p, q in asks[:self._n_levels]],
            timestamp=time.time(),
        )
        
        self._prev_book = self._current_book
        self._current_book = book
        self._book_history.append(book)
        
        return book

    def calculate_spread(self) -> float:
        if not self._current_book or not self._current_book.bids or not self._current_book.asks:
            return 0.0
        
        best_bid = self._current_book.bids[0].price
        best_ask = self._current_book.asks[0].price
        
        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2
        
        return (spread / mid) * 10000 if mid > 0 else 0.0

    def calculate_mid_price(self) -> float:
        if not self._current_book or not self._current_book.bids or not self._current_book.asks:
            return 0.0
        
        best_bid = self._current_book.bids[0].price
        best_ask = self._current_book.asks[0].price
        
        return (best_bid + best_ask) / 2

    def calculate_depth_imbalance(
        self,
        depth_levels: int = 5,
    ) -> float:
        if not self._current_book:
            return 0.5
        
        bid_depth = sum(
            level.quantity for level in self._current_book.bids[:depth_levels]
        )
        ask_depth = sum(
            level.quantity for level in self._current_book.asks[:depth_levels]
        )
        
        total = bid_depth + ask_depth
        if total == 0:
            return 0.5
        
        return bid_depth / total

    def calculate_depth(self) -> Tuple[float, float]:
        if not self._current_book:
            return 0.0, 0.0
        
        bid_depth = sum(level.quantity for level in self._current_book.bids)
        ask_depth = sum(level.quantity for level in self._current_book.asks)
        
        return bid_depth, ask_depth

    def calculate_realized_spread(
        self,
        trade_direction: int,
        trade_price: float,
        horizon_seconds: int = 60,
    ) -> float:
        if not self._prev_book:
            return 0.0
        
        old_mid = (self._prev_book.bids[0].price + self._prev_book.asks[0].price) / 2
        
        effective = (trade_price - old_mid) / old_mid * 10000
        
        realized = effective * np.sign(trade_direction)
        
        return realized

    def calculate_order_flow_imbalance(
        self,
        window: int = 10,
    ) -> float:
        if len(self._book_history) < window:
            return 0.0
        
        bids_total = 0.0
        asks_total = 0.0
        
        for book in list(self._book_history)[-window:]:
            bids_total += sum(level.quantity for level in book.bids)
            asks_total += sum(level.quantity for level in book.asks)
        
        total = bids_total + asks_total
        if total == 0:
            return 0.0
        
        return (bids_total - asks_total) / total


class MarketImpactModel:
    def __init__(
        self,
        decay: float = 0.6,
        impact_coefficient: float = 0.1,
    ):
        self._decay = decay
        self._impact_coefficient = impact_coefficient
        self._impact_history: deque = deque(maxlen=500)

    def estimate_impact(
        self,
        order_size: float,
        ADV: float,
        execution_time_minutes: int = 1,
    ) -> float:
        participation_rate = order_size / (ADV * execution_time_minutes * 60)
        
        instantaneous = self._impact_coefficient * (participation_rate ** 2)
        
        permanent = 2 * self._impact_coefficient * participation_rate
        
        return instantaneous + permanent

    def calculate_optimal_execution(
        self,
        order_size: float,
        ADV: float,
        target_vwap: float,
        risk_aversion: float = 0.5,
    ) -> Dict[str, Any]:
        participation = np.linspace(0.01, 0.3, 30)
        
        best_schedule = None
        best_score = float("inf")
        
        for p in participation:
            cost = self.estimate_impact(order_size, ADV)
            vwap_diff = abs(p - target_vwap)
            
            score = cost + risk_aversion * vwap_diff
            
            if score < best_score:
                best_score = score
                best_schedule = {
                    "participation_rate": p,
                    "estimated_impact": cost,
                    "num_tranches": int(order_size / (ADV * p / 60)),
                }
        
        return best_schedule


class VolatilityEstimator:
    def __init__(self, window: int = 100):
        self._window = window
        self._returns: deque = deque(maxlen=window)

    def add_return(self, return_pct: float) -> None:
        self._returns.append(return_pct)

    def estimate_realized_volatility(self) -> float:
        if len(self._returns) < 2:
            return 0.0
        
        returns = np.array(self._returns)
        return float(np.std(returns)) * np.sqrt(252)

    def estimate_intraday_volatility(
        self,
        bar_returns: List[float],
    ) -> float:
        if len(bar_returns) < 2:
            return 0.0
        
        return float(np.std(bar_returns)) * np.sqrt(252 * len(bar_returns))

    def estimate_volatility_from_book(
        self,
        bid_depth: float,
        ask_depth: float,
        spread_bps: float,
    ) -> float:
        if spread_bps <= 0 or bid_depth + ask_depth <= 0:
            return 0.0
        
        depth_factor = (bid_depth + ask_depth) / max(bid_depth, ask_depth)
        
        implied_vol = spread_bps * depth_factor * 0.01
        
        return implied_vol


class MicrostructureRegimeDetector:
    def __init__(self):
        self._vol_estimator = VolatilityEstimator()
        self._book_analyzer = OrderBookAnalyzer()
        self._regime_history: deque = deque(maxlen=100)

    def detect_regime(
        self,
        order_book: Optional[OrderBook] = None,
        recent_volatility: Optional[float] = None,
    ) -> MicrostructureRegime:
        if order_book:
            self._book_analyzer.update_book(
                order_book.symbol,
                [(b.price, b.quantity) for b in order_book.bids],
                [(a.price, a.quantity) for a in order_book.asks],
            )
        
        spread = self._book_analyzer.calculate_spread()
        bid_depth, ask_depth = self._book_analyzer.calculate_depth()
        
        total_depth = bid_depth + ask_depth
        
        if spread > 50:
            return MicrostructureRegime.ILLIQUID
        
        if recent_volatility and recent_volatility > 0.50:
            return MicrostructureRegime.TURBULENT
        
        if spread < 1 and total_depth > 1000000:
            return MicrostructureRegime.FLASH
        
        if spread < 5 and total_depth < 10000:
            return MicrostructureRegime.ILLIQUID
        
        return MicrostructureRegime.NORMAL


class MarketMicrostructureAnalyzer:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        self._book_analyzer = OrderBookAnalyzer(
            n_levels=self.config.get("n_levels", 10),
        )
        self._impact_model = MarketImpactModel(
            decay=self.config.get("decay", 0.6),
            impact_coefficient=self.config.get("impact_coefficient", 0.1),
        )
        self._vol_estimator = VolatilityEstimator(
            window=self.config.get("vol_window", 100),
        )
        self._regime_detector = MicrostructureRegimeDetector()
        
        self._metrics_history: deque = deque(maxlen=1000)

    def analyze(
        self,
        symbol: str,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        trade_price: Optional[float] = None,
        trade_direction: Optional[int] = None,
    ) -> MicrostructureMetrics:
        book = self._book_analyzer.update_book(symbol, bids, asks)
        
        metrics = MicrostructureMetrics()
        
        metrics.spread_bps = self._book_analyzer.calculate_spread()
        metrics.mid_price = self._book_analyzer.calculate_mid_price()
        metrics.depth_imbalance = self._book_analyzer.calculate_depth_imbalance()
        metrics.bid_depth, metrics.ask_depth = self._book_analyzer.calculate_depth()
        
        if trade_price and trade_direction is not None:
            metrics.realized_spread = self._book_analyzer.calculate_realized_spread(
                trade_direction, trade_price
            )
        
        metrics.volatility = self._vol_estimator.estimate_realized_volatility()
        metrics.regime = self._regime_detector.detect_regime(book, metrics.volatility)
        
        self._metrics_history.append(metrics)
        
        return metrics

    def add_return(self, return_pct: float) -> None:
        self._vol_estimator.add_return(return_pct)

    def estimate_impact(
        self,
        order_size: float,
        ADV: float,
    ) -> float:
        return self._impact_model.estimate_impact(order_size, ADV)

    def get_optimal_execution(
        self,
        order_size: float,
        ADV: float,
        target_vwap: float,
    ) -> Dict[str, Any]:
        return self._impact_model.calculate_optimal_execution(
            order_size, ADV, target_vwap
        )

    def get_current_regime(self) -> MicrostructureRegime:
        return self._regime_detector.detect_regime()

    def get_metrics_history(self) -> List[MicrostructureMetrics]:
        return list(self._metrics_history)

    def get_average_spread(self, window: int = 100) -> float:
        recent = list(self._metrics_history)[-window:]
        if not recent:
            return 0.0
        return np.mean([m.spread_bps for m in recent])

    def get_average_depth_imbalance(self, window: int = 100) -> float:
        recent = list(self._metrics_history)[-window:]
        if not recent:
            return 0.5
        return np.mean([m.depth_imbalance for m in recent])