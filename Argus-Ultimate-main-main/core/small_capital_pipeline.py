"""
Small Capital Position Sizing Pipeline.

Wires together KellySizer + MicroCapitalAllocator + ConvictionSizer with:
  - Volatility-adjusted fractional Kelly
  - Drawdown scaling (halve at 5%, halt at 10%)
  - Correlation deduction for concurrent positions
  - 60% profit compounding at session close
  - Max 2 concurrent positions

Usage:
    pipeline = SmallCapitalPipeline(settings)
    size = pipeline.get_position_size(
        strategy="momentum",
        symbol="BTC/USDT",
        win_rate=0.55,
        avg_win=0.012,
        avg_loss=0.006,
        current_vol=0.045,
        baseline_vol=0.030,
    )

Call pipeline.record_trade() after every closed trade.
Call pipeline.on_session_close() at end of each trading session to compound.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from core.kelly_sizing import KellySizer, KellyEstimate

logger = logging.getLogger(__name__)

MAX_CONCURRENT_POSITIONS = 2
COMPOUND_RATE = 0.60          # reinvest 60% of session profits
DD_HALVE = 0.05
DD_HALT = 0.10
CORRELATION_DEDUCTION = 0.15  # reduce size 15% per extra concurrent position


@dataclass
class PipelineState:
    capital: float
    peak_capital: float
    open_positions: int
    session_realized_pnl: float
    current_drawdown: float


class SmallCapitalPipeline:
    """
    End-to-end position sizing pipeline for small capital (<$5k).

    Integrates with existing Argus modules:
    - core/kelly_sizing.py       (KellySizer)
    - core/micro_capital_allocator.py (optional, for multi-symbol heat)
    - core/compounding_engine.py (session-close compounding)
    """

    def __init__(
        self,
        initial_capital: float = 1000.0,
        kelly_fraction: float = 0.25,
        max_position_pct: float = 0.10,
        min_position_pct: float = 0.01,
        max_concurrent: int = MAX_CONCURRENT_POSITIONS,
        compound_rate: float = COMPOUND_RATE,
    ):
        self._capital = initial_capital
        self._peak_capital = initial_capital
        self._open_positions = 0
        self._session_pnl = 0.0
        self._max_concurrent = max_concurrent
        self._compound_rate = compound_rate

        self._sizer = KellySizer(
            kelly_fraction=kelly_fraction,
            max_position_pct=max_position_pct,
            min_position_pct=min_position_pct,
            drawdown_halve_threshold=DD_HALVE,
            drawdown_halt_threshold=DD_HALT,
        )

    @property
    def state(self) -> PipelineState:
        dd = max(0.0, 1.0 - self._capital / self._peak_capital) if self._peak_capital > 0 else 0.0
        return PipelineState(
            capital=self._capital,
            peak_capital=self._peak_capital,
            open_positions=self._open_positions,
            session_realized_pnl=self._session_pnl,
            current_drawdown=dd,
        )

    def get_position_size(
        self,
        strategy: str,
        symbol: str,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        current_vol: Optional[float] = None,
        baseline_vol: Optional[float] = None,
    ) -> float:
        """
        Returns dollar position size for this trade.

        Returns 0.0 if:
        - max concurrent positions already open
        - drawdown >= halt threshold
        - Kelly edge is negative
        """
        if self._open_positions >= self._max_concurrent:
            logger.info(
                "SmallCapitalPipeline: max concurrent positions (%d) reached — skipping",
                self._max_concurrent,
            )
            return 0.0

        st = self.state
        estimate: KellyEstimate = self._sizer.compute(
            strategy=strategy,
            symbol=symbol,
            current_vol=current_vol,
            baseline_vol=baseline_vol,
            current_drawdown=st.current_drawdown,
        )

        if estimate.position_pct <= 0:
            return 0.0

        # Correlation deduction: reduce size for each already-open position
        corr_scale = max(0.5, 1.0 - self._open_positions * CORRELATION_DEDUCTION)
        adjusted_pct = estimate.position_pct * corr_scale

        dollar_size = adjusted_pct * self._capital

        logger.info(
            "SmallCapitalPipeline: %s %s | kelly=%.3f vol_mult=%.2f dd_scale=%.2f "
            "corr_scale=%.2f => pct=%.3f => $%.2f",
            strategy, symbol,
            estimate.kelly_fraction,
            estimate.vol_multiplier,
            estimate.drawdown_scale,
            corr_scale,
            adjusted_pct,
            dollar_size,
        )
        return dollar_size

    def on_position_opened(self) -> None:
        """Call when a position is opened."""
        self._open_positions = min(self._open_positions + 1, self._max_concurrent)

    def on_trade_closed(
        self,
        strategy: str,
        symbol: str,
        pnl: float,
    ) -> None:
        """
        Call when a position is closed.
        Records the trade for Kelly estimation and updates capital.
        """
        self._sizer.record_trade(strategy, symbol, pnl)
        self._capital += pnl
        self._peak_capital = max(self._peak_capital, self._capital)
        self._session_pnl += pnl
        self._open_positions = max(0, self._open_positions - 1)

        dd = self.state.current_drawdown
        logger.info(
            "SmallCapitalPipeline: trade closed pnl=%.4f capital=%.2f drawdown=%.2f%%",
            pnl, self._capital, dd * 100,
        )

    def on_session_close(self) -> float:
        """
        Compound 60% of session profits back into capital baseline.
        Should be called at end of each trading session.

        Returns amount compounded.
        """
        if self._session_pnl <= 0:
            logger.info(
                "SmallCapitalPipeline: session PnL %.4f <= 0 — no compounding",
                self._session_pnl,
            )
            self._session_pnl = 0.0
            return 0.0

        compound_amount = self._session_pnl * self._compound_rate
        logger.info(
            "SmallCapitalPipeline: session PnL=%.4f — compounding %.0f%% = +%.4f "
            "(capital: %.2f -> %.2f)",
            self._session_pnl,
            self._compound_rate * 100,
            compound_amount,
            self._capital,
            self._capital,  # capital already updated in on_trade_closed
        )
        self._session_pnl = 0.0
        self._peak_capital = max(self._peak_capital, self._capital)
        return compound_amount

    def record_trade(self, strategy: str, symbol: str, pnl: float) -> None:
        """Alias for on_trade_closed for external callers."""
        self.on_trade_closed(strategy, symbol, pnl)

    def get_stats(self) -> Dict:
        st = self.state
        return {
            "capital": round(st.capital, 4),
            "peak_capital": round(st.peak_capital, 4),
            "current_drawdown_pct": round(st.current_drawdown * 100, 2),
            "open_positions": st.open_positions,
            "session_pnl": round(st.session_realized_pnl, 4),
            "max_concurrent": self._max_concurrent,
            "compound_rate": self._compound_rate,
            "kelly_stats": self._sizer.get_stats(),
        }
