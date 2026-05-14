"""
Kelly Criterion Position Sizing for Argus-Ultimate v5.0.0 (HFT-Pinnacle)
Full Kelly + Fractional Kelly with drawdown guard.
Integrated into the risk module.
"""

import numpy as np
import logging
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class KellyResult:
    full_kelly_fraction: float
    fractional_kelly_fraction: float
    position_usd: float
    position_units: float
    kelly_fraction_used: float
    drawdown_scaled: bool
    correlation_scaled: bool = False
    correlation_scalar: float = 1.0
    notes: str = ""


@dataclass
class TradeOutcome:
    win: bool
    pnl_pct: float  # as decimal, e.g. 0.005 for 0.5%


class KellyCriterion:
    """
    Full and Fractional Kelly Criterion position sizer with:
    - Rolling win rate and avg win/loss estimation
    - Fractional Kelly scaling (default: half-Kelly)
    - Drawdown guard that reduces Kelly fraction under stress
    - Hard floor/ceiling on position size
    """

    def __init__(
        self,
        capital: float = 1000.0,
        kelly_fraction: float = 0.5,
        max_kelly_pct: float = 0.20,
        min_position_usd: float = 12.0,
        max_position_pct: float = 0.20,
        drawdown_guard_threshold: float = 0.05,
        drawdown_guard_scale: float = 0.5,
        lookback_window: int = 50,
    ):
        """
        Args:
            capital: Current account capital in USD.
            kelly_fraction: Fractional Kelly multiplier (0.5 = half-Kelly).
            max_kelly_pct: Hard cap on Kelly fraction (e.g. 0.20 = 20% of capital max).
            min_position_usd: Minimum position size in USD.
            max_position_pct: Maximum position as % of capital.
            drawdown_guard_threshold: Drawdown level (as decimal) that triggers scaling.
            drawdown_guard_scale: Scale factor applied when drawdown guard fires.
            lookback_window: Number of recent trades used for win rate estimation.
        """
        self.capital = capital
        self.kelly_fraction = kelly_fraction
        self.max_kelly_pct = max_kelly_pct
        self.min_position_usd = min_position_usd
        self.max_position_pct = max_position_pct
        self.drawdown_guard_threshold = drawdown_guard_threshold
        self.drawdown_guard_scale = drawdown_guard_scale
        self.lookback_window = lookback_window
        self._trade_history: List[TradeOutcome] = []
        self._peak_capital: float = capital
        self._correlation_scalar: float = 1.0  # Correlation-based position scalar

    def update_capital(self, current_capital: float) -> None:
        self.capital = current_capital
        if current_capital > self._peak_capital:
            self._peak_capital = current_capital

    def record_trade(self, win: bool, pnl_pct: float) -> None:
        """Record a trade outcome for rolling statistics."""
        self._trade_history.append(TradeOutcome(win=win, pnl_pct=pnl_pct))
        if len(self._trade_history) > self.lookback_window:
            self._trade_history.pop(0)

    def update_correlation_scalar(self, scalar: float) -> None:
        """
        Update correlation-based position scalar from CorrelationMonitor.
        
        Args:
            scalar: Position scalar (0.1-1.0) based on portfolio correlation.
                    1.0 = normal, 0.5 = high correlation, 0.1 = crisis
        """
        self._correlation_scalar = float(np.clip(scalar, 0.1, 1.0))
        logger.debug("KellyCriterion correlation scalar updated to %.2f", self._correlation_scalar)

    def _rolling_stats(self) -> Tuple[float, float, float]:
        """
        Compute rolling win rate, average win, and average loss from trade history.
        Returns (win_rate, avg_win, avg_loss) as decimals.
        """
        if not self._trade_history:
            # Conservative defaults when no history
            return 0.50, 0.005, 0.004

        wins = [t for t in self._trade_history if t.win]
        losses = [t for t in self._trade_history if not t.win]

        win_rate = len(wins) / len(self._trade_history)
        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0.005
        avg_loss = abs(np.mean([t.pnl_pct for t in losses])) if losses else 0.004

        # Clamp to avoid degenerate values
        win_rate = np.clip(win_rate, 0.01, 0.99)
        avg_win = max(avg_win, 1e-6)
        avg_loss = max(avg_loss, 1e-6)

        return win_rate, avg_win, avg_loss

    def full_kelly(
        self,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
    ) -> float:
        """
        Compute the full Kelly fraction: f* = (p*b - q) / b
        where p = win_rate, q = 1-p, b = avg_win / avg_loss (win/loss ratio).
        Returns fraction of capital to risk (clamped to [0, max_kelly_pct]).
        """
        p, w, l = self._rolling_stats()
        p = win_rate if win_rate is not None else p
        w = avg_win if avg_win is not None else w
        l = avg_loss if avg_loss is not None else l

        q = 1.0 - p
        b = w / l  # win/loss ratio

        kelly_f = (p * b - q) / b
        kelly_f = np.clip(kelly_f, 0.0, self.max_kelly_pct)
        logger.debug(f"Full Kelly: p={p:.4f}, b={b:.4f}, f*={kelly_f:.6f}")
        return float(kelly_f)

    def fractional_kelly(
        self,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
    ) -> float:
        """
        Fractional Kelly = kelly_fraction * full_kelly.
        Reduces variance while sacrificing some expected growth.
        """
        fk = self.full_kelly(win_rate, avg_win, avg_loss) * self.kelly_fraction
        logger.debug(f"Fractional Kelly ({self.kelly_fraction}x): {fk:.6f}")
        return fk

    def _drawdown_guard_scale(self) -> float:
        """
        Return a scaling factor based on current drawdown.
        If drawdown exceeds threshold, scale down aggressively.
        """
        if self._peak_capital <= 0:
            return 1.0
        current_dd = (self._peak_capital - self.capital) / self._peak_capital
        if current_dd >= self.drawdown_guard_threshold:
            scale = self.drawdown_guard_scale * (
                1.0 - (current_dd - self.drawdown_guard_threshold) / self.drawdown_guard_threshold
            )
            scale = max(scale, 0.1)  # Never go below 10% of normal sizing
            logger.warning(
                f"Drawdown guard active: DD={current_dd*100:.2f}%, scale={scale:.4f}"
            )
            return scale
        return 1.0

    def position_size(
        self,
        price: float,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        use_fractional: bool = True,
    ) -> KellyResult:
        """
        Compute final position size in USD and units.

        Args:
            price: Current asset price.
            win_rate: Override win rate (uses rolling stats if None).
            avg_win: Override avg win pct (uses rolling stats if None).
            avg_loss: Override avg loss pct (uses rolling stats if None).
            use_fractional: Use fractional Kelly (True) or full Kelly (False).

        Returns:
            KellyResult with full details.
        """
        full_k = self.full_kelly(win_rate, avg_win, avg_loss)
        frac_k = full_k * self.kelly_fraction

        chosen_fraction = frac_k if use_fractional else full_k

        # Apply drawdown guard
        dd_scale = self._drawdown_guard_scale()
        drawdown_scaled = dd_scale < 1.0
        
        # Apply correlation scalar (NEW - reduces positions when correlations are high)
        correlation_scaled = self._correlation_scalar < 1.0
        
        adjusted_fraction = chosen_fraction * dd_scale * self._correlation_scalar

        # Cap at max position pct
        adjusted_fraction = min(adjusted_fraction, self.max_position_pct)

        position_usd = self.capital * adjusted_fraction

        # Enforce minimum
        if position_usd < self.min_position_usd:
            logger.debug(
                f"Kelly position ${position_usd:.2f} below minimum ${self.min_position_usd:.2f}, zeroing."
            )
            return KellyResult(
                full_kelly_fraction=full_k,
                fractional_kelly_fraction=frac_k,
                position_usd=0.0,
                position_units=0.0,
                kelly_fraction_used=adjusted_fraction,
                drawdown_scaled=drawdown_scaled,
                correlation_scaled=correlation_scaled,
                correlation_scalar=self._correlation_scalar,
                notes="Below minimum trade size",
            )

        position_units = position_usd / price if price > 0 else 0.0

        logger.info(
            f"Kelly position: ${position_usd:.2f} ({adjusted_fraction*100:.2f}% of capital), "
            f"{position_units:.6f} units @ ${price:.2f}, "
            f"correlation_scalar={self._correlation_scalar:.2f}"
        )

        return KellyResult(
            full_kelly_fraction=full_k,
            fractional_kelly_fraction=frac_k,
            position_usd=position_usd,
            position_units=position_units,
            kelly_fraction_used=adjusted_fraction,
            drawdown_scaled=drawdown_scaled,
            correlation_scaled=correlation_scaled,
            correlation_scalar=self._correlation_scalar,
            notes="OK",
        )

    def get_stats(self) -> Dict:
        """Return current Kelly stats."""
        p, w, l = self._rolling_stats()
        full_k = self.full_kelly()
        return {
            "capital": self.capital,
            "peak_capital": self._peak_capital,
            "current_drawdown_pct": (
                (self._peak_capital - self.capital) / self._peak_capital * 100
                if self._peak_capital > 0 else 0.0
            ),
            "trade_history_len": len(self._trade_history),
            "rolling_win_rate": round(p, 4),
            "rolling_avg_win_pct": round(w * 100, 4),
            "rolling_avg_loss_pct": round(l * 100, 4),
            "full_kelly_fraction": round(full_k, 6),
            "fractional_kelly_fraction": round(full_k * self.kelly_fraction, 6),
        }
