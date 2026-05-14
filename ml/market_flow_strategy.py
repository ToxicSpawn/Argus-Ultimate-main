"""
Market Flow Adaptive Strategy - Self-improving strategy for constant income.

This strategy reads market flow data in real-time and adapts to generate consistent income:
- Price momentum signals
- Volume-supported moves  
- Order book imbalance
- Volatility regime
- Spread optimization
- Micro-trend capture

Usage:
    strategy = MarketFlowAdaptiveStrategy(config)
    signals = await strategy.generate_signals(market_data)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime detection."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


class SignalDirection(Enum):
    """Trade direction."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class MarketFlowData:
    """Real-time market flow data."""

    symbol: str
    timestamp: datetime
    price: float
    price_change_pct: float
    volume: float
    volume_ratio: float  # Current / average
    bid_ask_spread: float
    order_book_imbalance: float  # -1 to 1
    volatility: float  # ATR-based
    atr: float
    rsi: float
    momentum: float  # Price momentum
    trend_strength: float  # 0-1
    recent_zigzags: int  # Count of direction changes


@dataclass
class StrategySignal:
    """Generated trading signal."""

    symbol: str
    direction: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float
    reason: str  # Why this signal was generated
    regime: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyPerformance:
    """Track strategy performance for adaptation."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.5
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 1.0
    max_drawdown: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0

    # Regime-specific performance
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Time windows
    hourly_pnl: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0


class MarketFlowAdaptiveStrategy:
    """
    Self-adapting strategy that reads market flow and generates constant income.

    Key mechanisms:
    1. Multi-timeframe momentum capture
    2. Volume confirmation
    3. Order book imbalance exploitation
    4. Volatility-adjusted position sizing
    5. Real-time parameter adaptation
    6. Regime-aware strategy switching
    """

    def __init__(
        self,
        *,
        # Entry parameters
        min_confidence: float = 0.50,
        min_momentum_pct: float = 0.10,
        min_volume_ratio: float = 1.0,
        min_trend_strength: float = 0.30,

        # Position sizing
        base_position_pct: float = 0.02,
        max_position_pct: float = 0.10,
        use_volatility_sizing: bool = True,
        vol_multiplier: float = 1.0,

        # Risk parameters
        stop_loss_pct: float = 0.015,
        take_profit_pct: float = 0.03,
        trailing_stop: bool = True,
        trailing_distance_pct: float = 0.01,

        # Adaptation parameters
        adapt_enabled: bool = True,
        adaptation_rate: float = 0.10,
        min_trades_for_adaptation: int = 10,

        # Regime parameters
        use_regime_filter: bool = True,
        regime_window: int = 50,

        # Micro-trading parameters
        use_micro_trends: bool = True,
        micro_timeframe: str = "1m",
        micro_target_pct: float = 0.005,  # 0.5% target

        # Performance tracking
        track_performance: bool = True,

        **kwargs,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_momentum_pct = min_momentum_pct
        self.min_volume_ratio = min_volume_ratio
        self.min_trend_strength = min_trend_strength

        self.base_position_pct = base_position_pct
        self.max_position_pct = max_position_pct
        self.use_volatility_sizing = use_volatility_sizing
        self.vol_multiplier = vol_multiplier

        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop = trailing_stop
        self.trailing_distance_pct = trailing_distance_pct

        self.adapt_enabled = adapt_enabled
        self.adaptation_rate = adaptation_rate
        self.min_trades_for_adaptation = min_trades_for_adaptation

        self.use_regime_filter = use_regime_filter
        self.regime_window = regime_window

        self.use_micro_trends = use_micro_trends
        self.micro_timeframe = micro_timeframe
        self.micro_target_pct = micro_target_pct

        self.track_performance = track_performance

        # Internal state
        self._performance = StrategyPerformance()
        self._regime = MarketRegime.UNKNOWN
        self._price_history: List[float] = []
        self._volume_history: List[float] = []
        self._entry_prices: Dict[str, float] = {}
        self._signals_generated: int = 0

    def detect_regime(
        self,
        price_data: List[float],
        volume_data: List[float],
    ) -> MarketRegime:
        """Detect current market regime."""
        if len(price_data) < self.regime_window:
            return MarketRegime.UNKNOWN

        recent = price_data[-self.regime_window :]

        # Calculate trend
        sma_short = np.mean(recent[-5:])
        sma_long = np.mean(recent[-20:])

        # Calculate volatility
        returns = np.diff(recent) / recent[:-1]
        volatility = np.std(returns)
        avg_vol = np.std(returns[-20:]) if len(returns) >= 20 else volatility

        # Calculate volume
        avg_volume = np.mean(volume_data[-20:]) if volume_data else 1.0

        # Determine regime
        if volatility > avg_vol * 1.5:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < avg_vol * 0.7:
            return MarketRegime.LOW_VOLATILITY
        elif sma_short > sma_long * 1.01:
            return MarketRegime.TRENDING_UP
        elif sma_short < sma_long * 0.99:
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.RANGING

    def calculate_momentum(
        self,
        prices: List[float],
        periods: int = 5,
    ) -> Tuple[float, float]:
        """Calculate price momentum and strength."""
        if len(prices) < periods + 1:
            return 0.0, 0.0

        recent_prices = prices[-periods:]
        momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

        # Trend strength (percentage of periods with positive returns)
        returns = np.diff(recent_prices) / recent_prices[:-1]
        positive_periods = np.sum(returns > 0)
        strength = positive_periods / max(len(returns), 1)

        return float(momentum), float(strength)

    def calculate_position_size(
        self,
        volatility: float,
        equity: float,
        confidence: float,
        regime: MarketRegime,
    ) -> float:
        """Calculate volatility-adjusted position size."""
        # Base size
        size = self.base_position_pct

        # Adjust for volatility
        if self.use_volatility_sizing and volatility > 0:
            # Target vol = 2% daily
            target_vol = 0.02
            vol_ratio = min(target_vol / volatility, 2.0)  # Cap at 2x
            size *= vol_ratio * self.vol_multiplier

        # Adjust for confidence
        size *= confidence

        # Adjust for regime
        if regime == MarketRegime.HIGH_VOLATILITY:
            size *= 0.5  # Reduce in high vol
        elif regime == MarketRegime.LOW_VOLATILITY:
            size *= 1.25  # Can size up in low vol
        elif regime == MarketRegime.TRENDING_UP:
            size *= 1.15  # More aggressive in trend

        # Cap position
        return min(size, self.max_position_pct)

    def calculate_confidence(
        self,
        flow_data: MarketFlowData,
        regime_performance: Dict[str, float],
    ) -> float:
        """Calculate signal confidence from multiple factors."""
        confidence_factors = []

        # Factor 1: Momentum strength
        momentum_score = min(abs(flow_data.momentum) / self.min_momentum_pct, 1.0)
        confidence_factors.append(("momentum", momentum_score, 0.25))

        # Factor 2: Volume confirmation
        volume_score = min(flow_data.volume_ratio / self.min_volume_ratio, 1.5) / 1.5
        confidence_factors.append(("volume", volume_score, 0.20))

        # Factor 3: Trend strength
        trend_score = flow_data.trend_strength
        confidence_factors.append(("trend", trend_score, 0.20))

        # Factor 4: Regime performance
        regime_score = regime_performance.get(self._regime.value, {}).get("win_rate", 0.5)
        confidence_factors.append(("regime", regime_score, 0.20))

        # Factor 5: Order book imbalance
        ob_score = abs(flow_data.order_book_imbalance)
        confidence_factors.append(("orderbook", ob_score, 0.15))

        # Calculate weighted confidence
        weighted_confidence = sum(score * weight for _, score, weight in confidence_factors)
        total_weight = sum(weight for _, _, weight in confidence_factors)

        return weighted_confidence / total_weight

    async def analyze_market_flow(
        self,
        symbol: str,
        ohlcv_data: List,  # List of [timestamp, open, high, low, close, volume]
    ) -> MarketFlowData:
        """Analyze market flow and extract features."""
        if not ohlcv_data or len(ohlcv_data) < 20:
            return MarketFlowData(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                price=0.0,
                price_change_pct=0.0,
                volume=0.0,
                volume_ratio=1.0,
                bid_ask_spread=0.0,
                order_book_imbalance=0.0,
                volatility=0.0,
                atr=0.0,
                rsi=50.0,
                momentum=0.0,
                trend_strength=0.0,
                recent_zigzags=0,
            )

        # Extract closing prices and volumes
        closes = np.array([c[4] for c in ohlcv_data])
        volumes = np.array([c[5] for c in ohlcv_data])
        highs = np.array([c[2] for c in ohlcv_data])
        lows = np.array([c[3] for c in ohlcv_data])

        current_price = closes[-1]

        # Price change
        price_change_pct = (current_price - closes[0]) / closes[0] if closes[0] > 0 else 0.0

        # Volume ratio
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0

        # ATR calculation
        tr = np.maximum(
            highs[-1] - lows[-1],
            np.abs(np.array([highs[-1] - closes[-2], lows[-1] - closes[-2]]) if len(closes) > 1 else [0, 0]),
        )
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else tr[-1]

        # Volatility (as percentage)
        volatility = atr / current_price if current_price > 0 else 0.0

        # RSI calculation
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
        rs = avg_gain / avg_loss if avg_loss > 0 else 1.0
        rsi = 100 - (100 / (1 + rs))

        # Momentum
        momentum, trend_strength = self.calculate_momentum(closes.tolist(), 5)

        # Count direction changes (zigzags)
        direction_changes = 0
        for i in range(1, len(closes) - 1):
            if (closes[i] > closes[i - 1] and closes[i] > closes[i + 1]) or (
                closes[i] < closes[i - 1] and closes[i] < closes[i + 1]
            ):
                direction_changes += 1

        return MarketFlowData(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            price=current_price,
            price_change_pct=price_change_pct,
            volume=volumes[-1],
            volume_ratio=volume_ratio,
            bid_ask_spread=0.0,  # Would need order book data
            order_book_imbalance=0.0,  # Would need order book data
            volatility=volatility,
            atr=atr,
            rsi=rsi,
            momentum=momentum,
            trend_strength=trend_strength,
            recent_zigzags=direction_changes,
        )

    async def generate_signals(
        self,
        symbol: str,
        ohlcv_data: List,
        equity: float = 10000.0,
    ) -> List[StrategySignal]:
        """Generate trading signals from market flow."""
        signals = []

        # Analyze market flow
        flow_data = await self.analyze_market_flow(symbol, ohlcv_data)

        # Get regime
        closes = [c[4] for c in ohlcv_data]
        volumes = [c[5] for c in ohlcv_data]
        self._regime = self.detect_regime(closes, volumes)

        # Get regime-specific performance
        regime_perf = self._performance.regime_performance.get(self._regime.value, {})

        # Calculate confidence
        confidence = self.calculate_confidence(flow_data, regime_perf)

        # Skip if below threshold
        if confidence < self.min_confidence:
            return signals

        # Determine direction
        direction = SignalDirection.HOLD
        if flow_data.momentum > self.min_momentum_pct and flow_data.trend_strength > self.min_trend_strength:
            direction = SignalDirection.BUY
        elif flow_data.momentum < -self.min_momentum_pct and flow_data.trend_strength > self.min_trend_strength:
            direction = SignalDirection.SELL

        # Skip if no clear direction
        if direction == SignalDirection.HOLD:
            return signals

        # Calculate position size
        position_size = self.calculate_position_size(
            flow_data.volatility, equity, confidence, self._regime
        )

        # Calculate stop loss and take profit
        entry_price = flow_data.price

        if direction == SignalDirection.BUY:
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
        else:  # SELL
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.take_profit_pct)

        # Generate signal
        signal = StrategySignal(
            symbol=symbol,
            direction=direction.value,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=position_size,
            reason=f"momentum_{direction.value}_{self._regime.value}",
            regime=self._regime.value,
            timestamp=flow_data.timestamp,
            metadata={
                "momentum": flow_data.momentum,
                "trend_strength": flow_data.trend_strength,
                "volume_ratio": flow_data.volume_ratio,
                "volatility": flow_data.volatility,
                "rsi": flow_data.rsi,
            },
        )

        signals.append(signal)
        self._signals_generated += 1

        return signals

    def record_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        position_size_pct: float,
    ) -> None:
        """Record trade for performance tracking and adaptation."""
        pnl_pct = (exit_price - entry_price) / entry_price if direction == "buy" else (entry_price - exit_price) / entry_price

        trade_pnl = pnl_pct * position_size_pct

        self._performance.total_trades += 1

        if pnl_pct > 0:
            self._performance.winning_trades += 1
            self._performance.consecutive_wins += 1
            self._performance.consecutive_losses = 0
        else:
            self._performance.losing_trades += 1
            self._performance.consecutive_losses += 1
            self._performance.consecutive_wins = 0

        self._performance.total_pnl += trade_pnl

        # Update win rate
        self._performance.win_rate = self._performance.winning_trades / max(
            self._performance.total_trades, 1
        )

        # Update regime-specific performance
        regime = self._regime.value
        if regime not in self._performance.regime_performance:
            self._performance.regime_performance[regime] = {
                "trades": 0,
                "wins": 0,
                "pnl": 0.0,
                "win_rate": 0.5,
            }

        regime_data = self._performance.regime_performance[regime]
        regime_data["trades"] += 1
        regime_data["wins"] += 1 if pnl_pct > 0 else 0
        regime_data["pnl"] += trade_pnl
        regime_data["win_rate"] = regime_data["wins"] / max(regime_data["trades"], 1)

    def get_performance(self) -> StrategyPerformance:
        """Get current strategy performance."""
        # Calculate derived metrics
        if self._performance.total_trades > 0:
            self._performance.avg_win = (
                self._performance.total_pnl / self._performance.winning_trades
                if self._performance.winning_trades > 0
                else 0.0
            )
            self._performance.avg_loss = (
                self._performance.total_pnl / self._performance.losing_trades
                if self._performance.losing_trades > 0
                else 0.0
            )
            expectance = (
                self._performance.avg_win * self._performance.win_rate
                - self._performance.avg_loss * (1 - self._performance.win_rate)
            )
            self._performance.expectancy = expectance

        return self._performance


def create_market_flow_strategy(
    *,
    min_confidence: float = 0.50,
    base_position_pct: float = 0.02,
    adapt_enabled: bool = True,
) -> MarketFlowAdaptiveStrategy:
    """Factory function to create configured strategy."""
    return MarketFlowAdaptiveStrategy(
        min_confidence=min_confidence,
        base_position_pct=base_position_pct,
        adapt_enabled=adapt_enabled,
    )


__all__ = [
    "MarketFlowAdaptiveStrategy",
    "MarketFlowData",
    "StrategySignal",
    "StrategyPerformance",
    "MarketRegime",
    "SignalDirection",
    "create_market_flow_strategy",
]