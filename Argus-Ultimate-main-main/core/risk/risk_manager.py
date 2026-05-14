"""Push 78 — RiskManager: portfolio-level risk gating.

Checks performed on every check() call:
  1. Per-symbol drawdown vs max_symbol_drawdown_pct
  2. Portfolio heat (sum open notionals / equity) vs max_heat
  3. Historical VaR (95 + 99) vs var_limit_pct
  4. Daily loss vs daily_loss_limit_pct * initial_equity
  5. Kill-switch state

Kill-switch:
  Activated when any hard limit breached, or manually via
  activate_kill_switch(). Blocks all new orders until
  reset_kill_switch() is called (or auto-reset on new day).

VaR / CVaR:
  Historical simulation from a rolling returns window
  (default 252 bars). VaR_95 = 5th percentile of losses.
  CVaR/ES = mean of losses beyond VaR_95.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from core.risk.risk_event import RiskEvent, RiskEventBus, RiskEventType


@dataclass
class RiskConfig:
    initial_equity:          float = 10_000.0
    max_symbol_drawdown_pct: float = 10.0    # % per symbol
    max_portfolio_heat:      float = 0.80    # fraction of equity in open notionals
    var_limit_pct:           float = 5.0     # daily VaR % limit (95th)
    daily_loss_limit_pct:    float = 3.0     # % of initial equity
    var_window:              int   = 252     # bars for VaR lookback
    var_confidence:          float = 0.95
    kill_on_var_breach:      bool  = False   # True = hard kill on VaR breach
    kill_on_daily_loss:      bool  = True


@dataclass
class SymbolRiskState:
    symbol:        str
    peak_equity:   float = 0.0
    current_pnl:   float = 0.0

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        current = self.peak_equity + self.current_pnl
        return max(0.0, (self.peak_equity - current) / self.peak_equity * 100)


class RiskManager:
    """Portfolio-level risk manager.

    Args:
        config:    RiskConfig
        event_bus: RiskEventBus for emitting alerts
    """

    def __init__(
        self,
        config:    Optional[RiskConfig]    = None,
        event_bus: Optional[RiskEventBus]  = None,
    ):
        self.config    = config or RiskConfig()
        self.event_bus = event_bus or RiskEventBus()

        self._kill_switch:   bool  = False
        self._daily_pnl:     float = 0.0
        self._day_start:     float = time.time()
        self._equity:        float = self.config.initial_equity
        self._returns:       Deque[float] = deque(maxlen=self.config.var_window)
        self._symbol_states: Dict[str, SymbolRiskState] = {}
        self._open_notionals: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_order_allowed(self, symbol: str, notional: float) -> tuple[bool, str]:
        """Return (allowed, reason). Call before submitting any order."""
        if self._kill_switch:
            return False, "Kill switch active — trading halted"

        self._auto_reset_daily()

        # Daily loss
        loss_limit = self.config.initial_equity * self.config.daily_loss_limit_pct / 100
        if self._daily_pnl <= -loss_limit:
            msg = f"Daily loss limit reached: {self._daily_pnl:.2f}"
            self.event_bus.emit(RiskEvent(
                RiskEventType.DAILY_LOSS_LIMIT, msg,
                symbol=symbol, value=self._daily_pnl, threshold=-loss_limit
            ))
            if self.config.kill_on_daily_loss:
                self.activate_kill_switch(f"Daily loss limit: {self._daily_pnl:.2f}")
            return False, msg

        # Portfolio heat
        heat = self._portfolio_heat(notional)
        if heat > self.config.max_portfolio_heat:
            msg = f"Portfolio heat {heat:.2%} > limit {self.config.max_portfolio_heat:.2%}"
            self.event_bus.emit(RiskEvent(
                RiskEventType.PORTFOLIO_HEAT, msg,
                value=heat, threshold=self.config.max_portfolio_heat
            ))
            return False, msg

        # Symbol drawdown
        state = self._symbol_states.get(symbol)
        if state and state.drawdown_pct >= self.config.max_symbol_drawdown_pct:
            msg = f"{symbol} drawdown {state.drawdown_pct:.2f}% >= limit"
            self.event_bus.emit(RiskEvent(
                RiskEventType.DRAWDOWN_BREACH, msg,
                symbol=symbol,
                value=state.drawdown_pct,
                threshold=self.config.max_symbol_drawdown_pct,
            ))
            return False, msg

        # VaR check
        var95, _ = self.compute_var()
        if var95 > self.config.var_limit_pct:
            msg = f"VaR95 {var95:.2f}% > limit {self.config.var_limit_pct:.2f}%"
            self.event_bus.emit(RiskEvent(
                RiskEventType.VAR_BREACH, msg,
                value=var95, threshold=self.config.var_limit_pct
            ))
            if self.config.kill_on_var_breach:
                self.activate_kill_switch(msg)
            return False, msg

        return True, "OK"

    def record_return(self, daily_return_pct: float) -> None:
        """Feed daily return (%) into VaR window."""
        self._returns.append(daily_return_pct)
        self._daily_pnl += self._equity * daily_return_pct / 100

    def update_symbol_pnl(self, symbol: str, pnl_delta: float) -> None:
        """Update per-symbol PnL (called on fill)."""
        state = self._symbol_states.setdefault(
            symbol, SymbolRiskState(symbol=symbol, peak_equity=self.config.initial_equity)
        )
        state.current_pnl += pnl_delta
        self._daily_pnl   += pnl_delta
        current = state.peak_equity + state.current_pnl
        if current > state.peak_equity:
            state.peak_equity = current

    def update_open_notional(self, symbol: str, notional: float) -> None:
        """Set current open notional for symbol (used for heat calc)."""
        self._open_notionals[symbol] = max(0.0, notional)

    def update_equity(self, equity: float) -> None:
        self._equity = equity

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def activate_kill_switch(self, reason: str = "") -> None:
        self._kill_switch = True
        self.event_bus.emit(RiskEvent(
            RiskEventType.KILL_SWITCH,
            f"Kill switch activated: {reason}",
        ))

    def reset_kill_switch(self) -> None:
        self._kill_switch = False
        self.event_bus.emit(RiskEvent(
            RiskEventType.KILL_SWITCH_RESET,
            "Kill switch reset — trading resumed",
        ))

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch

    # ------------------------------------------------------------------
    # VaR / CVaR
    # ------------------------------------------------------------------

    def compute_var(
        self,
        confidence: Optional[float] = None,
    ) -> tuple[float, float]:
        """Return (VaR_pct, CVaR_pct) at given confidence.

        VaR  = -percentile(losses, 1-confidence)
        CVaR = mean of returns worse than VaR

        Returns (0, 0) if insufficient data.
        """
        if len(self._returns) < 10:
            return 0.0, 0.0
        conf   = confidence or self.config.var_confidence
        rets   = sorted(self._returns)
        idx    = int((1 - conf) * len(rets))
        idx    = max(0, min(idx, len(rets) - 1))
        var    = -rets[idx]          # positive number = loss
        tail   = [r for r in rets if r <= rets[idx]]
        cvar   = -sum(tail) / len(tail) if tail else 0.0
        return max(0.0, var), max(0.0, cvar)

    def compute_var_99(self) -> tuple[float, float]:
        return self.compute_var(confidence=0.99)

    # ------------------------------------------------------------------
    # Portfolio heat
    # ------------------------------------------------------------------

    def _portfolio_heat(self, additional_notional: float = 0.0) -> float:
        total = sum(self._open_notionals.values()) + additional_notional
        return total / self._equity if self._equity > 0 else 0.0

    @property
    def portfolio_heat(self) -> float:
        return self._portfolio_heat()

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def _auto_reset_daily(self) -> None:
        now = time.time()
        if now - self._day_start >= 86_400:
            self._daily_pnl  = 0.0
            self._day_start  = now
            if self._kill_switch and self.config.kill_on_daily_loss:
                self.reset_kill_switch()

    def force_daily_reset(self) -> None:
        """Manual daily reset (e.g. for testing or EOD job)."""
        self._daily_pnl = 0.0
        self._day_start = time.time()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        var95, cvar95 = self.compute_var()
        var99, cvar99 = self.compute_var_99()
        return {
            "kill_switch":    self._kill_switch,
            "daily_pnl":      round(self._daily_pnl, 4),
            "portfolio_heat": round(self.portfolio_heat, 4),
            "var_95":         round(var95, 4),
            "cvar_95":        round(cvar95, 4),
            "var_99":         round(var99, 4),
            "cvar_99":        round(cvar99, 4),
            "equity":         round(self._equity, 2),
            "returns_window": len(self._returns),
        }
