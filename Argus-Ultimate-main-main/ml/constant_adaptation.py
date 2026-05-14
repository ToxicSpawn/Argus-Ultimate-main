"""
Constant Adaptation Loop - Real-time self-improving trading engine.

This module provides the continuous adaptation loop that:
- Monitors market data in real-time
- Evaluates strategy performance continuously  
- Adapts parameters based on recent results
- Generates signals with optimized parameters
- Records feedback for next iteration

Usage:
    engine = AdaptationLoop(config)
    await engine.start()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ml.market_flow_strategy import (
    MarketFlowAdaptiveStrategy,
    StrategySignal,
    StrategyPerformance,
    create_market_flow_strategy,
)

logger = logging.getLogger(__name__)


@dataclass
class AdaptationState:
    """Current state of the adaptation engine."""

    cycle: int = 0
    total_signals: int = 0
    total_trades: int = 0
    active_positions: int = 0
    equity: float = 10000.0
    starting_equity: float = 10000.0
    daily_pnl: float = 0.0
    hourly_pnl: float = 0.0
    current_drawdown: float = 0.0
    peak_equity: float = 10000.0


@dataclass
class AdaptationConfig:
    """Configuration for the adaptation loop."""

    # Cycle timing
    cycle_interval_seconds: float = 15.0  # Main analysis interval
    micro_interval_seconds: float = 1.0   # High-frequency check

    # Data config
    lookback_bars: int = 200  # Bars to keep in history
    warmup_bars: int = 50   # Bars needed before generating signals

    # Generation config
    max_signals_per_cycle: int = 3
    max_concurrent_positions: int = 5

    # Adaptation config
    adaptation_enabled: bool = True
    adaptation_interval_trades: int = 10  # Adapt every N trades
    adaptation_rate: float = 0.10  # How much to adjust (10%)

    # Performance thresholds
    min_win_rate: float = 0.40  # Below this, tighten parameters
    max_drawdown: float = 0.10  # Stop trading if exceeded

    # Position limits
    max_position_pct: float = 0.10
    max_correlation: float = 0.70

    # Emergency config
    emergency_stop_enabled: bool = True
    emergency_min_trades: int = 20


class ConstantAdaptationLoop:
    """
    Real-time adaptation loop for constant income generation.

    Key mechanisms:
    1. Market data ingestion (continuous)
    2. Signal generation (every cycle)
    3. Trade execution feedback
    4. Parameter adaptation (after N trades)
    5. Emergency stops (drawdown protection)
    """

    def __init__(
        self,
        config: Optional[AdaptationConfig] = None,
        strategy: Optional[MarketFlowAdaptiveStrategy] = None,
    ) -> None:
        self.config = config or AdaptationConfig()
        self.strategy = strategy or create_market_flow_strategy()

        # State
        self._state = AdaptationState()
        self._running = False
        self._paused = False
        self._market_data_cache: Dict[str, List] = {}

        # Performance history
        self._performance_history: List[StrategyPerformance] = []
        self._adaptation_history: List[Dict[str, Any]] = []

        # Execution callbacks
        self._on_signal: Optional[Callable] = None
        self._on_trade: Optional[Callable] = None
        self._on_adaptation: Optional[Callable] = None

    def set_signal_callback(self, callback: Callable[[List[StrategySignal]], None]) -> None:
        """Set callback for signal generation."""
        self._on_signal = callback

    def set_trade_callback(
        self, callback: Callable[[str, str, float, float, float], None]
    ) -> None:
        """Set callback for trade execution."""
        self._on_trade = callback

    def set_adaptation_callback(
        self, callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Set callback for adaptation events."""
        self._on_adaptation = callback

    async def ingest_market_data(
        self,
        symbol: str,
        ohlcv_data: List,
    ) -> None:
        """Ingest market data for a symbol."""
        if symbol not in self._market_data_cache:
            self._market_data_cache[symbol] = []

        # Add new data
        self._market_data_cache[symbol].extend(ohlcv_data)

        # Trim to lookback
        if len(self._market_data_cache[symbol]) > self.config.lookback_bars:
            self._market_data_cache[symbol] = self._market_data_cache[symbol][
                -self.config.lookback_bars :
            ]

    async def generate_signals_cycle(
        self,
        symbols: List[str],
    ) -> List[StrategySignal]:
        """Generate signals for all symbols."""
        all_signals = []

        # Check minimum warms
        for symbol in symbols:
            if symbol not in self._market_data_cache:
                continue

            data = self._market_data_cache[symbol]
            if len(data) < self.config.warmup_bars:
                continue

            # Generate signals for this symbol
            try:
                signals = await self.strategy.generate_signals(
                    symbol,
                    data,
                    self._state.equity,
                )
                all_signals.extend(signals)
            except Exception as e:
                logger.warning(f"Signal generation failed for {symbol}: {e}")

        # Limit signals
        all_signals = all_signals[: self.config.max_signals_per_cycle]

        # Update state
        self._state.total_signals += len(all_signals)
        self._state.cycle += 1

        return all_signals

    def record_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        position_size_pct: float,
    ) -> None:
        """Record trade and update performance."""
        # Record in strategy
        self.strategy.record_trade(
            symbol, direction, entry_price, exit_price, position_size_pct
        )

        # Update state
        self._state.total_trades += 1
        pnl = (exit_price - entry_price) / entry_price * position_size_pct
        self._state.daily_pnl += pnl
        self._state.equity *= 1 + pnl

        # Track peak equity
        if self._state.equity > self._state.peak_equity:
            self._state.peak_equity = self._state.equity

        # Calculate drawdown
        self._state.current_drawdown = (
            self._state.peak_equity - self._state.equity
        ) / self._state.peak_equity

    def adapt_parameters(self) -> Dict[str, Any]:
        """Adapt strategy parameters based on recent performance."""
        if not self.config.adaptation_enabled:
            return {}

        perf = self.strategy.get_performance()

        # Store performance
        self._performance_history.append(perf)

        # Only adapt after minimum trades
        if perf.total_trades < self.config.adaptation_interval_trades:
            return {"action": "wait", "reason": "insufficient_trades"}

        # Determine adaptation action
        adaptations = {}

        # 1. Check win rate
        if perf.win_rate < self.config.min_win_rate:
            # Tighten parameters
            adaptations["action"] = "tighten"
            adaptations["reason"] = f"low_win_rate_{perf.win_rate:.2%}"
            adaptations["confidence_mult"] = 1 - self.config.adaptation_rate
            adaptations["position_mult"] = 1 - self.config.adaptation_rate

        elif perf.win_rate > 0.55:
            # Can relax slightly
            adaptations["action"] = "relax"
            adaptations["reason"] = f"good_win_rate_{perf.win_rate:.2%}"
            adaptations["confidence_mult"] = 1 + self.config.adaptation_rate * 0.5
            adaptations["position_mult"] = 1 + self.config.adaptation_rate * 0.5

        # 2. Check drawdown
        if self._state.current_drawdown > self.config.max_drawdown * 0.8:
            adaptations["drawdown_warning"] = True
            current_mult = adaptations.get("position_mult", 1.0)
            adaptations["position_mult"] = min(current_mult * 0.7, self.config.max_position_pct)

        # 3. Check streak
        if perf.consecutive_losses >= 3:
            adaptations["loss_streak"] = perf.consecutive_losses
            current_mult = adaptations.get("position_mult", 1.0)
            adaptations["position_mult"] = min(current_mult * 0.5, 0.05)
        elif perf.consecutive_wins >= 5:
            adaptations["win_streak"] = perf.consecutive_wins

        # Store adaptation
        self._adaptation_history.append(adaptations)

        return adaptations

    def check_emergency_stop(self) -> bool:
        """Check if emergency stop should trigger."""
        if not self.config.emergency_stop_enabled:
            return False

        # Need minimum trades to evaluate
        if self._state.total_trades < self.config.emergency_min_trades:
            return False

        # Check drawdown
        if self._state.current_drawdown > self.config.max_drawdown:
            logger.critical(
                f"EMERGENCY STOP: Drawdown {self._state.current_drawdown:.2%} exceeds max"
            )
            return True

        # Check win rate (if enough trades)
        perf = self.strategy.get_performance()
        if perf.total_trades >= 30 and perf.win_rate < 0.35:
            logger.critical(f"EMERGENCY STOP: Win rate {perf.win_rate:.2%} too low")
            return True

        return False

    async def run_cycle(self, symbols: List[str]) -> Dict[str, Any]:
        """Run one adaptation cycle."""
        result = {
            "cycle": self._state.cycle,
            "signals": [],
            "adaptations": {},
            "emergency_stop": False,
            "equity": self._state.equity,
        }

        # Generate signals
        signals = await self.generate_signals_cycle(symbols)
        result["signals"] = signals

        # Execute callback
        if signals and self._on_signal:
            await self._on_signal(signals)

        # Check for adaptation
        if self._state.total_trades > 0 and self._state.total_trades % self.config.adaptation_interval_trades == 0:
            adaptations = self.adapt_parameters()
            result["adaptations"] = adaptations

            if adaptations and self._on_adaptation:
                await self._on_adaptation(adaptations)

        # Check emergency
        if self.check_emergency_stop():
            result["emergency_stop"] = True
            self._paused = True

        return result

    def get_state(self) -> AdaptationState:
        """Get current adaptation state."""
        return self._state

    def get_performance(self) -> StrategyPerformance:
        """Get strategy performance."""
        return self.strategy.get_performance()


async def create_default_loop() -> ConstantAdaptationLoop:
    """Factory function to create default loop."""
    config = AdaptationConfig(
        cycle_interval_seconds=15.0,
        adaptation_enabled=True,
        min_win_rate=0.40,
        max_drawdown=0.10,
    )
    strategy = create_market_flow_strategy()

    return ConstantAdaptationLoop(config=config, strategy=strategy)


__all__ = [
    "ConstantAdaptationLoop",
    "AdaptationState",
    "AdaptationConfig",
    "create_default_loop",
]