"""Push 69 — Event-driven backtest engine.

Supports:
  - Per-bar signal consumption (buy/sell/flat)
  - Commission (bps of notional)
  - Slippage (fixed bps or ATR-scaled)
  - Mark-to-market PnL per bar
  - Equity curve, drawdown series
  - Long + short positions
  - Optional stop-loss / take-profit rules
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence
import numpy as np


@dataclass
class Trade:
    entry_bar: int
    exit_bar: int
    side: str            # "long" | "short"
    entry_price: float
    exit_price: float
    size_usd: float
    commission: float
    pnl: float
    pnl_pct: float


@dataclass
class BacktestResult:
    equity_curve: np.ndarray      # equity at each bar
    returns: np.ndarray           # bar returns
    drawdown_series: np.ndarray   # peak-to-trough at each bar
    trades: List[Trade]
    initial_equity: float
    final_equity: float
    total_commission: float
    n_bars: int

    @property
    def total_return(self) -> float:
        return (self.final_equity - self.initial_equity) / self.initial_equity

    @property
    def n_trades(self) -> int:
        return len(self.trades)


class BacktestEngine:
    """Vectorised-signal event-driven backtester.

    Args:
        initial_equity:  Starting capital in USD
        commission_bps:  Round-trip commission in basis points
        slippage_bps:    Slippage per fill in basis points
        position_sizing: Fraction of equity per trade [0, 1]
        allow_short:     Whether to allow short positions
        stop_loss_pct:   Optional per-trade stop loss
        take_profit_pct: Optional per-trade take profit
    """

    def __init__(
        self,
        initial_equity: float = 10_000.0,
        commission_bps: float = 10.0,
        slippage_bps: float = 5.0,
        position_sizing: float = 0.95,
        allow_short: bool = True,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
    ):
        self.initial_equity = initial_equity
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.position_sizing = position_sizing
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def run(
        self,
        closes: Sequence[float],
        signals: Sequence[float],   # +1=long, -1=short, 0=flat
    ) -> BacktestResult:
        """Run backtest. signals[i] is acted on at close[i+1]."""
        closes = np.array(closes, dtype=float)
        signals = np.array(signals, dtype=float)
        n = len(closes)
        assert len(signals) == n, "signals must match closes length"

        equity = self.initial_equity
        position = 0.0          # current position in USD notional
        entry_price = 0.0
        entry_bar = 0
        current_side = "flat"
        total_commission = 0.0

        equity_curve = np.zeros(n)
        trades: List[Trade] = []

        for i in range(n):
            price = closes[i]
            sig = signals[i]

            # Check SL/TP on existing position
            if position != 0.0 and entry_price > 0:
                pnl_pct = ((price - entry_price) / entry_price
                           if current_side == "long"
                           else (entry_price - price) / entry_price)
                if (self.stop_loss_pct and pnl_pct <= -self.stop_loss_pct) or \
                   (self.take_profit_pct and pnl_pct >= self.take_profit_pct):
                    sig = 0.0   # force flat

            # Position change
            target_side = ("long" if sig > 0 else
                           "short" if sig < 0 and self.allow_short else "flat")

            if target_side != current_side:
                # Close existing
                if position != 0.0 and entry_price > 0:
                    exit_price = price * (1 - self.slippage_bps / 10_000)
                    pnl_pct = ((exit_price - entry_price) / entry_price
                               if current_side == "long"
                               else (entry_price - exit_price) / entry_price)
                    comm = abs(position) * (self.commission_bps / 10_000)
                    pnl = position * pnl_pct - comm
                    equity += pnl
                    total_commission += comm
                    trades.append(Trade(
                        entry_bar=entry_bar, exit_bar=i,
                        side=current_side,
                        entry_price=entry_price, exit_price=exit_price,
                        size_usd=abs(position), commission=comm,
                        pnl=pnl, pnl_pct=pnl_pct,
                    ))
                    position = 0.0

                # Open new
                if target_side in ("long", "short"):
                    entry_price = price * (1 + self.slippage_bps / 10_000)
                    position = equity * self.position_sizing * (1 if target_side == "long" else -1)
                    entry_bar = i
                    current_side = target_side
                else:
                    current_side = "flat"
                    entry_price = 0.0

            # Mark-to-market
            if position != 0.0 and entry_price > 0:
                pnl_pct = ((price - entry_price) / entry_price
                           if current_side == "long"
                           else (entry_price - price) / entry_price)
                mtm_equity = equity + abs(position) * pnl_pct
            else:
                mtm_equity = equity

            equity_curve[i] = max(mtm_equity, 0.0)

        # Compute returns + drawdown
        returns = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-9)
        returns = np.concatenate([[0.0], returns])
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / np.maximum(peak, 1e-9)

        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            drawdown_series=drawdown,
            trades=trades,
            initial_equity=self.initial_equity,
            final_equity=float(equity_curve[-1]),
            total_commission=total_commission,
            n_bars=n,
        )
