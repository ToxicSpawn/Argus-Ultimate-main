"""Session report generator — accumulates metrics and produces a summary."""
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionReport:
    start_time: float
    end_time: float
    duration_seconds: float
    cycles: int
    trades_executed: int
    signals_generated: int
    win_rate: float
    total_pnl: float
    max_drawdown_pct: float
    sharpe_estimate: float
    avg_slippage_bps: float
    regime_changes: int
    strategies_used: List[str]
    symbols_traded: List[str]
    summary_text: str


class SessionReportGenerator:
    """Accumulates per-session trading metrics and generates a report."""

    def __init__(self) -> None:
        self._start_time: float = time.time()
        self._cycles: int = 0

        # Trade tracking
        self._pnls: List[float] = []
        self._slippages: List[float] = []
        self._trade_symbols: set = set()
        self._trade_strategies: set = set()
        self._wins: int = 0
        self._losses: int = 0

        # Signal tracking
        self._signal_count: int = 0
        self._signal_symbols: set = set()
        self._signal_strategies: set = set()

        # Regime tracking
        self._regime_changes: int = 0
        self._regime_log: List[Dict[str, str]] = []

        # Drawdown tracking
        self._cumulative_pnl: float = 0.0
        self._peak_pnl: float = 0.0
        self._max_drawdown_pct: float = 0.0

        logger.info("SessionReportGenerator initialised")

    # ------------------------------------------------------------------
    def record_trade(
        self,
        symbol: str,
        side: str,
        pnl: float,
        slippage_bps: float,
        strategy: str,
    ) -> None:
        """Accumulate a single trade."""
        self._pnls.append(pnl)
        self._slippages.append(slippage_bps)
        self._trade_symbols.add(symbol)
        self._trade_strategies.add(strategy)
        if pnl >= 0:
            self._wins += 1
        else:
            self._losses += 1

        # Update drawdown
        self._cumulative_pnl += pnl
        if self._cumulative_pnl > self._peak_pnl:
            self._peak_pnl = self._cumulative_pnl
        if self._peak_pnl > 0:
            dd = (self._peak_pnl - self._cumulative_pnl) / self._peak_pnl * 100.0
            if dd > self._max_drawdown_pct:
                self._max_drawdown_pct = dd

    # ------------------------------------------------------------------
    def record_signal(self, symbol: str, strategy: str) -> None:
        """Accumulate a generated signal."""
        self._signal_count += 1
        self._signal_symbols.add(symbol)
        self._signal_strategies.add(strategy)

    # ------------------------------------------------------------------
    def record_regime_change(self, from_regime: str, to_regime: str) -> None:
        """Record a regime transition."""
        self._regime_changes += 1
        self._regime_log.append({
            "from": from_regime,
            "to": to_regime,
            "time": time.time(),
        })
        logger.debug("Regime change #%d: %s -> %s", self._regime_changes, from_regime, to_regime)

    # ------------------------------------------------------------------
    def record_cycle(self) -> None:
        """Increment the cycle counter."""
        self._cycles += 1

    # ------------------------------------------------------------------
    def generate(self) -> SessionReport:
        """Compute all metrics and return a frozen SessionReport."""
        end_time = time.time()
        duration = end_time - self._start_time
        trades = len(self._pnls)
        total_pnl = sum(self._pnls)

        # Win rate
        win_rate = 0.0
        if trades > 0:
            win_rate = round(self._wins / trades, 4)

        # Avg slippage
        avg_slippage = 0.0
        if self._slippages:
            avg_slippage = round(sum(self._slippages) / len(self._slippages), 2)

        # Sharpe estimate (annualised from per-trade returns)
        sharpe = 0.0
        if len(self._pnls) >= 2:
            mean_r = sum(self._pnls) / len(self._pnls)
            var_r = sum((p - mean_r) ** 2 for p in self._pnls) / (len(self._pnls) - 1)
            std_r = math.sqrt(var_r) if var_r > 0 else 0.0
            if std_r > 0:
                # Assume ~3 trades/day, 365 days
                trades_per_year = 3 * 365
                sharpe = round((mean_r / std_r) * math.sqrt(trades_per_year), 2)

        strategies = sorted(self._trade_strategies | self._signal_strategies)
        symbols = sorted(self._trade_symbols | self._signal_symbols)

        summary = self._build_summary_text(
            duration, trades, total_pnl, win_rate, sharpe,
            avg_slippage, strategies, symbols,
        )

        report = SessionReport(
            start_time=self._start_time,
            end_time=end_time,
            duration_seconds=round(duration, 1),
            cycles=self._cycles,
            trades_executed=trades,
            signals_generated=self._signal_count,
            win_rate=win_rate,
            total_pnl=round(total_pnl, 4),
            max_drawdown_pct=round(self._max_drawdown_pct, 2),
            sharpe_estimate=sharpe,
            avg_slippage_bps=avg_slippage,
            regime_changes=self._regime_changes,
            strategies_used=strategies,
            symbols_traded=symbols,
            summary_text=summary,
        )
        logger.info("Session report generated: %d cycles, %d trades, PnL=%.4f",
                     self._cycles, trades, total_pnl)
        return report

    # ------------------------------------------------------------------
    def export_json(self, path: str) -> Path:
        """Generate the report and save as JSON. Returns the output path."""
        report = self.generate()
        out = Path(path)
        os.makedirs(out.parent, exist_ok=True)

        data = {
            "start_time": report.start_time,
            "end_time": report.end_time,
            "duration_seconds": report.duration_seconds,
            "cycles": report.cycles,
            "trades_executed": report.trades_executed,
            "signals_generated": report.signals_generated,
            "win_rate": report.win_rate,
            "total_pnl": report.total_pnl,
            "max_drawdown_pct": report.max_drawdown_pct,
            "sharpe_estimate": report.sharpe_estimate,
            "avg_slippage_bps": report.avg_slippage_bps,
            "regime_changes": report.regime_changes,
            "strategies_used": report.strategies_used,
            "symbols_traded": report.symbols_traded,
            "summary_text": report.summary_text,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Session report exported to %s", out)
        return out

    # ------------------------------------------------------------------
    def export_text(self) -> str:
        """Generate the report and return a human-readable string."""
        report = self.generate()
        return report.summary_text

    # ------------------------------------------------------------------
    @staticmethod
    def _build_summary_text(
        duration: float,
        trades: int,
        total_pnl: float,
        win_rate: float,
        sharpe: float,
        avg_slippage: float,
        strategies: List[str],
        symbols: List[str],
    ) -> str:
        hours = duration / 3600
        lines = [
            "=== ARGUS Session Report ===",
            f"Duration        : {hours:.1f} hours",
            f"Trades executed : {trades}",
            f"Win rate        : {win_rate:.1%}",
            f"Total PnL       : {total_pnl:+.4f}",
            f"Sharpe estimate : {sharpe:.2f}",
            f"Avg slippage    : {avg_slippage:.1f} bps",
            f"Strategies      : {', '.join(strategies) if strategies else 'none'}",
            f"Symbols         : {', '.join(symbols) if symbols else 'none'}",
            "============================",
        ]
        return "\n".join(lines)
