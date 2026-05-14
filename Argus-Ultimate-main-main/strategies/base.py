"""
Argus Trading System - Strategy Base Class
==========================================

Abstract base class for all trading strategies.
All strategy implementations must inherit from this class.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from core.types import (
    Signal,
    SignalAction,
    MarketRegime,
    Position,
)


@dataclass
class StrategyConfig:
    """Configuration for a strategy."""
    # Identification
    name: str
    version: str = "1.0.0"
    description: str = ""

    # Timeframe
    timeframe: str = "1h"  # 1m, 5m, 15m, 1h, 4h, 1d
    min_lookback_bars: int = 50

    # Signal thresholds
    min_confidence: float = 0.6
    min_strength: float = 0.3

    # Risk parameters
    max_position_pct: float = 0.10  # Max 10% of capital per position
    default_stop_loss_pct: float = 0.02  # 2% stop loss
    default_take_profit_pct: float = 0.04  # 4% take profit (2:1 R:R)

    # Regime preferences (which regimes this strategy works best in)
    preferred_regimes: List[MarketRegime] = field(
        default_factory=lambda: [
            MarketRegime.TREND_UP,
            MarketRegime.TREND_DOWN,
            MarketRegime.RANGE,
        ]
    )

    # Strategy-specific parameters
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyState:
    """
    Mutable state for a strategy.
    Used for tracking performance and adaptation.
    """
    # Performance tracking
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0

    # Recent performance (for adaptation)
    recent_pnl: List[float] = field(default_factory=list)
    recent_window: int = 20

    # Timestamps
    last_signal_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None

    # Cooldown tracking
    cooldown_until: Optional[datetime] = None

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def avg_pnl_pct(self) -> float:
        """Average PnL per trade."""
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl_pct / self.total_trades

    @property
    def recent_avg_pnl(self) -> float:
        """Average of recent PnL values."""
        if not self.recent_pnl:
            return 0.0
        return sum(self.recent_pnl) / len(self.recent_pnl)

    def record_trade(self, pnl_pct: float) -> None:
        """Record a completed trade."""
        self.total_trades += 1
        self.total_pnl_pct += pnl_pct

        if pnl_pct >= 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Update recent PnL
        self.recent_pnl.append(pnl_pct)
        if len(self.recent_pnl) > self.recent_window:
            self.recent_pnl.pop(0)

        self.last_trade_time = datetime.utcnow()

    def is_in_cooldown(self) -> bool:
        """Check if strategy is in cooldown period."""
        if self.cooldown_until is None:
            return False
        return datetime.utcnow() < self.cooldown_until


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.

    Strategies generate trading signals based on market data.
    They should be stateless with respect to market data -
    all state should be tracked in StrategyState.
    """

    def __init__(self, config: Optional[StrategyConfig] = None) -> None:
        """
        Initialize strategy.

        Args:
            config: Strategy configuration. If None, uses defaults.
        """
        self._config = config or self._default_config()
        self._state = StrategyState()

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier."""
        pass

    @property
    def version(self) -> str:
        """Strategy version."""
        return self._config.version

    @property
    def config(self) -> StrategyConfig:
        """Get strategy configuration."""
        return self._config

    @property
    def state(self) -> StrategyState:
        """Get strategy state."""
        return self._state

    @property
    def required_lookback(self) -> int:
        """Minimum number of bars required for signal generation."""
        return self._config.min_lookback_bars

    def _default_config(self) -> StrategyConfig:
        """Get default configuration. Override in subclasses."""
        return StrategyConfig(name=self.name)

    # =========================================================================
    # Signal Generation (Must Implement)
    # =========================================================================

    @abstractmethod
    async def generate_signal(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        regime: MarketRegime = MarketRegime.UNKNOWN,
        **kwargs,
    ) -> Optional[Signal]:
        """
        Generate a trading signal from market data.

        Args:
            symbol: Trading pair (e.g., 'BTC/AUD')
            ohlcv: DataFrame with columns [timestamp, open, high, low, close, volume]
            regime: Current market regime
            **kwargs: Additional context (e.g., order_book, sentiment)

        Returns:
            Signal if conditions are met, None otherwise
        """
        pass

    # =========================================================================
    # Optional Hooks
    # =========================================================================

    def on_trade_opened(self, position: Position) -> None:
        """
        Called when a trade is opened based on this strategy's signal.

        Override to implement custom logic.
        """
        self._state.last_signal_time = datetime.utcnow()

    def on_trade_closed(self, position: Position, pnl_pct: float) -> None:
        """
        Called when a trade is closed.

        Override to implement custom adaptation logic.
        """
        self._state.record_trade(pnl_pct)

    def on_regime_change(self, old_regime: MarketRegime, new_regime: MarketRegime) -> None:
        """
        Called when market regime changes.

        Override to implement regime-specific adaptations.
        """
        pass

    # =========================================================================
    # Regime Compatibility
    # =========================================================================

    def is_compatible_with_regime(self, regime: MarketRegime) -> bool:
        """Check if strategy is suitable for the current regime."""
        if regime == MarketRegime.UNKNOWN:
            return True
        return regime in self._config.preferred_regimes

    def get_regime_adjustment(self, regime: MarketRegime) -> float:
        """
        Get confidence adjustment factor for a regime.

        Returns a multiplier (0.5 - 1.5) to adjust signal confidence.
        Override for custom regime handling.
        """
        if regime == MarketRegime.UNKNOWN:
            return 1.0
        if regime in self._config.preferred_regimes:
            return 1.0
        # Reduce confidence for non-preferred regimes
        return 0.7

    # =========================================================================
    # Risk Management Helpers
    # =========================================================================

    def calculate_stop_loss(
        self,
        entry_price: float,
        side: SignalAction,
        atr: Optional[float] = None,
    ) -> float:
        """
        Calculate stop loss price.

        Args:
            entry_price: Entry price
            side: BUY or SELL
            atr: Average True Range (optional, for ATR-based stops)

        Returns:
            Stop loss price
        """
        # Use ATR if provided, otherwise use percentage
        if atr is not None:
            stop_distance = atr * 2.0  # 2x ATR
        else:
            stop_distance = entry_price * self._config.default_stop_loss_pct

        if side == SignalAction.BUY:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def calculate_take_profit(
        self,
        entry_price: float,
        side: SignalAction,
        stop_loss: Optional[float] = None,
        risk_reward_ratio: float = 2.0,
    ) -> float:
        """
        Calculate take profit price.

        Args:
            entry_price: Entry price
            side: BUY or SELL
            stop_loss: Stop loss price (for risk-reward calculation)
            risk_reward_ratio: Target risk-reward ratio

        Returns:
            Take profit price
        """
        if stop_loss is not None:
            # Calculate based on risk-reward ratio
            risk = abs(entry_price - stop_loss)
            reward = risk * risk_reward_ratio
        else:
            # Use default percentage
            reward = entry_price * self._config.default_take_profit_pct

        if side == SignalAction.BUY:
            return entry_price + reward
        else:
            return entry_price - reward

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def validate_data(self, ohlcv: pd.DataFrame) -> bool:
        """
        Validate that OHLCV data is sufficient for signal generation.

        Returns:
            True if data is valid and sufficient
        """
        if ohlcv is None or ohlcv.empty:
            return False

        required_columns = {"open", "high", "low", "close"}
        if not required_columns.issubset(set(ohlcv.columns)):
            return False

        if len(ohlcv) < self.required_lookback:
            return False

        return True

    def create_signal(
        self,
        symbol: str,
        action: SignalAction,
        confidence: float,
        strength: float,
        entry_price: float,
        regime: MarketRegime = MarketRegime.UNKNOWN,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        reasoning: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Signal:
        """
        Create a Signal with proper defaults and adjustments.

        Applies regime-based confidence adjustment automatically.
        """
        # Apply regime adjustment
        adjusted_confidence = confidence * self.get_regime_adjustment(regime)
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))

        # Calculate stops if not provided
        if stop_loss is None and action in (SignalAction.BUY, SignalAction.SELL):
            stop_loss = self.calculate_stop_loss(entry_price, action)

        if take_profit is None and action in (SignalAction.BUY, SignalAction.SELL):
            take_profit = self.calculate_take_profit(entry_price, action, stop_loss)

        # Generate unique signal ID
        signal_id = f"{self.name[:4]}_{uuid.uuid4().hex[:8]}"

        return Signal(
            symbol=symbol,
            action=action,
            confidence=adjusted_confidence,
            strength=strength,
            entry_price=entry_price,
            signal_id=signal_id,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=self.name,
            reasoning=reasoning,
            regime=regime,
            metadata=metadata or {},
        )

    def __repr__(self) -> str:
        return f"<Strategy: {self.name} v{self.version}>"
