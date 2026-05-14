"""Push 75 — BaseStrategy: abstract base for all Argus strategies.

All concrete strategies inherit from BaseStrategy and implement:
  - tick(price, volume, timestamp) -> Optional[Signal]
  - Optional: on_fill(fill_event)

BaseStrategy handles:
  - Kelly position sizing
  - Risk gate (drawdown / daily loss)
  - Per-strategy metrics accumulation
  - Lifecycle: start / stop / reset
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.strategy.signal import Signal, SignalSide


@dataclass
class StrategyConfig:
    strategy_id:      str
    symbol:           str
    kelly_fraction:   float = 0.25
    max_drawdown_pct: float = 5.0
    daily_loss_limit: float = 500.0
    initial_equity:   float = 10_000.0
    min_signal_strength: float = 0.3
    params:           Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyMetrics:
    total_signals:  int   = 0
    total_trades:   int   = 0
    winning_trades: int   = 0
    total_pnl:      float = 0.0
    peak_equity:    float = 0.0
    daily_pnl:      float = 0.0
    last_reset_ts:  float = field(default_factory=time.time)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        current = self.peak_equity + self.total_pnl
        return max(0.0, (self.peak_equity - current) / self.peak_equity * 100)


class BaseStrategy(ABC):
    """Abstract base for all Argus trading strategies.

    Args:
        config: StrategyConfig instance
    """

    def __init__(self, config: StrategyConfig):
        self.config  = config
        self.metrics = StrategyMetrics(peak_equity=config.initial_equity)
        self._running = False
        self._signal_history: List[Signal] = []

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def tick(
        self,
        price: float,
        volume: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> Optional[Signal]:
        """Process one price tick. Return a Signal or None."""

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def on_fill(self, fill_event: Any) -> None:
        """Called by order manager on fill. Update metrics."""
        self.metrics.total_trades += 1
        pnl = getattr(fill_event, "pnl", 0.0)
        self.metrics.total_pnl += pnl
        self.metrics.daily_pnl += pnl
        if pnl > 0:
            self.metrics.winning_trades += 1
        current_equity = self.config.initial_equity + self.metrics.total_pnl
        if current_equity > self.metrics.peak_equity:
            self.metrics.peak_equity = current_equity

    def on_signal(self, signal: Signal) -> None:
        """Called after generate_signal emits. Default: log to history."""
        self._signal_history.append(signal)
        if len(self._signal_history) > 500:
            self._signal_history.pop(0)

    # ------------------------------------------------------------------
    # Risk gate
    # ------------------------------------------------------------------

    def _risk_gate_open(self) -> bool:
        """Return True if strategy is allowed to trade (risk limits OK)."""
        if self.metrics.drawdown_pct >= self.config.max_drawdown_pct:
            return False
        if self.metrics.daily_pnl <= -abs(self.config.daily_loss_limit):
            return False
        return True

    # ------------------------------------------------------------------
    # Kelly position sizing
    # ------------------------------------------------------------------

    def kelly_size(
        self,
        equity: float,
        signal_strength: float,
        price: float,
    ) -> float:
        """Return position size in base units using fractional Kelly.

        size = (equity * kelly_fraction * signal_strength) / price
        """
        if price <= 0:
            return 0.0
        raw = equity * self.config.kelly_fraction * signal_strength
        return raw / price

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def reset(self) -> None:
        """Reset metrics and signal history (e.g. daily reset)."""
        self.metrics = StrategyMetrics(peak_equity=self.config.initial_equity)
        self._signal_history.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_signal(
        self,
        side: SignalSide,
        strength: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        order_type: str = "Market",
        **meta,
    ) -> Optional[Signal]:
        """Build and gate-check a Signal."""
        if not self._running:
            return None
        if not self._risk_gate_open():
            return None
        if strength < self.config.min_signal_strength:
            return None
        sig = Signal(
            symbol=self.config.symbol,
            side=side,
            strength=min(max(strength, 0.0), 1.0),
            strategy_id=self.config.strategy_id,
            order_type=order_type,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=meta,
        )
        self.metrics.total_signals += 1
        self.on_signal(sig)
        return sig

    @property
    def strategy_id(self) -> str:
        return self.config.strategy_id

    @property
    def symbol(self) -> str:
        return self.config.symbol

    @property
    def is_running(self) -> bool:
        return self._running

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.strategy_id}, symbol={self.symbol})"
