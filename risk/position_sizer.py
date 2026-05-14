"""
position_sizer.py — ATR-based Kelly volatility-scaled position sizer.

Combines three sizing methodologies into a single composable sizer:

  1. Kelly Fraction    — f* = (p*b - q) / b  where b = avg_win/avg_loss
  2. ATR Volatility    — size = (account * risk_pct) / (atr_mult * ATR)
  3. Regime Scalar     — scales by CrossAssetRegime.get_scalar() (Push 31)
  4. Conviction Scalar — scales output by MatrixResult.conviction [0, 1]

Sizing pipeline (Push 31)
--------------------------
  base_f    = min(kelly_f, atr_f)               # conservative blend
  sized_f   = base_f * regime_scalar * conv_scalar
  final_f   = clamp(sized_f, min_position, max_position)

All outputs are fractions of account equity [0.0, 1.0].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ATR_PERIOD    = 14
DEFAULT_ATR_MULT      = 2.0    # stop distance in ATR units
DEFAULT_RISK_PER_TRADE= 0.01   # 1% account risk per trade
DEFAULT_KELLY_FRAC    = 0.5    # half-Kelly (conservative)
DEFAULT_MAX_POSITION  = 0.25   # never exceed 25% of account in one trade
DEFAULT_MIN_POSITION  = 0.01   # minimum meaningful size (1%)


@dataclass
class SizeResult:
    fraction: float          # position size as fraction of equity [0, 1]
    notional: float          # position size in quote currency
    kelly_f: float           # raw Kelly fraction
    atr_f: float             # ATR-derived fraction
    conviction_scalar: float # conviction multiplier applied
    regime_scalar: float     # regime-adaptive risk multiplier applied (Push 31)
    atr: float               # current ATR value
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_tradeable(self) -> bool:
        return self.fraction >= DEFAULT_MIN_POSITION


class PositionSizer:
    """
    ATR-based Kelly volatility-scaled position sizer.

    Parameters
    ----------
    account_equity   : float  current total equity in quote currency
    atr_period       : int    ATR lookback period (default 14)
    atr_mult         : float  stop distance in ATR multiples (default 2.0)
    risk_per_trade   : float  fraction of equity to risk per trade (default 0.01)
    kelly_fraction   : float  Kelly scaling factor, 0.5 = half-Kelly (default 0.5)
    max_position     : float  hard cap on position as fraction of equity (default 0.25)
    min_position     : float  minimum tradeable size fraction (default 0.01)
    use_conviction   : bool   scale by MatrixResult conviction (default True)

    Push 31 — regime_scalar wired in via size() parameter.
    """

    def __init__(
        self,
        account_equity: float = 10_000.0,
        atr_period: int = DEFAULT_ATR_PERIOD,
        atr_mult: float = DEFAULT_ATR_MULT,
        risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
        kelly_fraction: float = DEFAULT_KELLY_FRAC,
        max_position: float = DEFAULT_MAX_POSITION,
        min_position: float = DEFAULT_MIN_POSITION,
        use_conviction: bool = True,
    ) -> None:
        self._equity        = float(account_equity)
        self._atr_period    = atr_period
        self._atr_mult      = atr_mult
        self._risk_per_trade= risk_per_trade
        self._kelly_frac    = kelly_fraction
        self._max_pos       = max_position
        self._min_pos       = min_position
        self._use_conviction= use_conviction

        # Rolling trade history for Kelly estimation
        self._trade_returns: List[float] = []

    # ------------------------------------------------------------------
    # Primary sizing method
    # ------------------------------------------------------------------

    def size(
        self,
        candles: np.ndarray,
        conviction: float = 1.0,
        current_price: Optional[float] = None,
        regime_scalar: float = 1.0,
    ) -> SizeResult:
        """
        Calculate position size given candle history and signal conviction.

        Parameters
        ----------
        candles        : np.ndarray (N, 6) [ts, open, high, low, close, vol]
        conviction     : float [0, 1] from MatrixResult.conviction
        current_price  : override close price (optional)
        regime_scalar  : float  regime-adaptive risk multiplier from
                         CrossAssetRegime.get_scalar() (Push 31).
                         0.5 = high-vol/risk-off, 1.5 = trending, 1.0 = neutral.

        Sizing pipeline
        ---------------
          base_f  = min(kelly_f, atr_f)
          sized_f = base_f * regime_scalar * conv_scalar
          final_f = clamp(sized_f, min_pos, max_pos)

        Returns
        -------
        SizeResult with .fraction in [0, 1] of account equity
        """
        if len(candles) < self._atr_period + 1:
            logger.debug("PositionSizer: insufficient candles (%d)", len(candles))
            return SizeResult(
                fraction=0.0, notional=0.0, kelly_f=0.0,
                atr_f=0.0, conviction_scalar=conviction,
                regime_scalar=regime_scalar, atr=0.0,
                metadata={"reason": "insufficient_candles"},
            )

        price  = current_price or float(candles[-1, 4])
        atr    = self._calc_atr(candles)
        kelly_f= self._calc_kelly()
        atr_f  = self._calc_atr_size(atr, price)

        # Blend: take the more conservative of Kelly and ATR sizing
        base_f = min(kelly_f, atr_f)

        # Apply regime scalar (Push 31) then conviction scalar
        conv_scalar = float(conviction) if self._use_conviction else 1.0
        sized_f     = base_f * float(regime_scalar) * conv_scalar

        # Clamp to [min, max]
        if sized_f < self._min_pos:
            final_f = 0.0  # below minimum — skip trade
        else:
            final_f = min(sized_f, self._max_pos)

        notional = final_f * self._equity

        logger.debug(
            "PositionSizer: kelly=%.4f atr=%.4f regime=%.2f conv=%.3f -> final=%.4f (%.2f)",
            kelly_f, atr_f, regime_scalar, conv_scalar, final_f, notional,
        )

        return SizeResult(
            fraction=round(final_f, 6),
            notional=round(notional, 4),
            kelly_f=round(kelly_f, 6),
            atr_f=round(atr_f, 6),
            conviction_scalar=round(conv_scalar, 4),
            regime_scalar=round(float(regime_scalar), 4),
            atr=round(float(atr), 6),
            metadata={
                "price": price,
                "equity": self._equity,
                "atr_stop_distance": round(float(atr) * self._atr_mult, 4),
                "n_trades_history": len(self._trade_returns),
                "base_f": round(base_f, 6),
            },
        )

    # ------------------------------------------------------------------
    # Equity + trade tracking
    # ------------------------------------------------------------------

    def update_equity(self, new_equity: float) -> None:
        """Call after each closed trade to update account equity."""
        self._equity = float(new_equity)

    def record_trade(self, pnl_pct: float) -> None:
        """
        Record a completed trade return for Kelly estimation.

        Parameters
        ----------
        pnl_pct : float  P&L as fraction of position size (e.g. 0.02 = 2% win)
        """
        self._trade_returns.append(float(pnl_pct))
        if len(self._trade_returns) > 100:
            self._trade_returns.pop(0)

    def reset_history(self) -> None:
        self._trade_returns.clear()

    # ------------------------------------------------------------------
    # Kelly fraction calculation
    # ------------------------------------------------------------------

    def _calc_kelly(self) -> float:
        """
        Estimate Kelly fraction from trade history.
        Falls back to default risk_per_trade if insufficient history (<10 trades).
        """
        if len(self._trade_returns) < 10:
            return self._risk_per_trade

        wins  = [r for r in self._trade_returns if r > 0]
        losses= [r for r in self._trade_returns if r < 0]

        if not wins or not losses:
            return self._risk_per_trade

        p = len(wins) / len(self._trade_returns)
        q = 1.0 - p
        b = np.mean(wins) / abs(np.mean(losses))

        if b <= 0:
            return self._risk_per_trade

        kelly = (p * b - q) / b
        kelly = max(0.0, kelly)
        kelly *= self._kelly_frac
        return min(kelly, self._max_pos)

    # ------------------------------------------------------------------
    # ATR sizing
    # ------------------------------------------------------------------

    def _calc_atr(self, candles: np.ndarray) -> float:
        """Wilder's ATR over the configured period."""
        high  = candles[-self._atr_period - 1:, 2].astype(np.float64)
        low   = candles[-self._atr_period - 1:, 3].astype(np.float64)
        close = candles[-self._atr_period - 1:, 4].astype(np.float64)

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:]  - close[:-1]),
            ),
        )
        atr = float(tr[0])
        alpha = 1.0 / self._atr_period
        for t in tr[1:]:
            atr = atr * (1 - alpha) + float(t) * alpha
        return atr

    def _calc_atr_size(self, atr: float, price: float) -> float:
        """
        Position fraction based on volatility stop.
        fraction = risk_per_trade / (atr_mult * atr / price)
        """
        if price <= 0 or atr <= 0:
            return self._risk_per_trade
        stop_pct = (self._atr_mult * atr) / price
        if stop_pct <= 0:
            return self._risk_per_trade
        return min(self._risk_per_trade / stop_pct, self._max_pos)

    # ------------------------------------------------------------------
    # Convenience: batch size for backtest position arrays
    # ------------------------------------------------------------------

    def size_array(
        self,
        candles: np.ndarray,
        convictions: np.ndarray,
        obs_window: int = 30,
        regime_scalar: float = 1.0,
    ) -> np.ndarray:
        """
        Compute position size fractions for each bar in a candle array.
        Useful for vectorised backtest position scaling.

        Parameters
        ----------
        regime_scalar : float  applied uniformly across all bars (Push 31)

        Returns
        -------
        np.ndarray shape (N,) of size fractions
        """
        n = len(candles)
        sizes = np.zeros(n)
        for i in range(obs_window, n):
            window = candles[max(0, i - obs_window):i + 1]
            conv   = float(convictions[i]) if i < len(convictions) else 1.0
            result = self.size(window, conviction=conv, regime_scalar=regime_scalar)
            sizes[i] = result.fraction
        return sizes

    def __repr__(self) -> str:
        return (
            f"<PositionSizer equity={self._equity:.2f} "
            f"kelly_frac={self._kelly_frac} atr_period={self._atr_period} "
            f"max_pos={self._max_pos}>"
        )
