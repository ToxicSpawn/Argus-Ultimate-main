"""
Dynamic Kelly Position Sizing

Tracks rolling win rate and payoff ratio per strategy per regime,
then blends a fractional Kelly with a fixed-risk baseline.

Batch 2 upgrade: replaces static risk-per-trade with per-strategy,
per-regime Kelly fraction updated after each closed trade.

Usage:
    sizer = KellyPositionSizer()
    # After each closed trade:
    sizer.record_trade(strategy="momentum", regime="TREND_UP",
                       pnl=120.0, entry=50000, exit=50240)
    # On new signal:
    result = sizer.calculate_position_size(
        capital=10000, entry_price=50000, stop_loss=49000,
        confidence=0.75, regime="TREND_UP", strategy="momentum",
    )
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Kelly fraction cap (never bet more than 25% of capital on one trade)
_MAX_KELLY_FRACTION = 0.25
# Blend weight: 0 = pure Kelly, 1 = pure fixed-risk
_KELLY_BLEND = 0.5
# Minimum trades before Kelly activates (use fixed-risk before then)
_MIN_TRADES_FOR_KELLY = 10
# Fixed risk fallback (2% of capital)
_FIXED_RISK_PCT = 0.02
# Rolling window length
_WINDOW = 50


@dataclass
class SizeResult:
    quantity: float
    position_value: float
    kelly_fraction: float
    win_rate: float
    payoff_ratio: float
    strategy: str
    regime: str


class KellyPositionSizer:
    """
    Per-strategy, per-regime dynamic Kelly position sizer.

    Maintains a rolling deque of (win: bool, payoff: float) per
    (strategy, regime) key and recomputes Kelly fraction on each new signal.
    """

    def __init__(self) -> None:
        # {(strategy_key, regime_key): deque[(win, payoff)]}
        self._history: Dict[Tuple[str, str], Deque[Tuple[bool, float]]] = {}

    def _key(self, strategy: str, regime: Optional[str]) -> Tuple[str, str]:
        s = strategy.lower().strip() if strategy else "default"
        r = str(regime).upper().strip() if regime else "UNKNOWN"
        return (s, r)

    def record_trade(
        self,
        strategy: str,
        regime: Optional[str],
        pnl: float,
        entry: float,
        exit_price: float,
    ) -> None:
        """
        Record a closed trade for Kelly tracking.

        Parameters
        ----------
        pnl : float
            Net P&L of the trade (positive = win).
        entry, exit_price : float
            Used to compute payoff ratio = |gain| / |loss| proxy.
        """
        key = self._key(strategy, regime)
        if key not in self._history:
            self._history[key] = deque(maxlen=_WINDOW)

        win = pnl > 0
        if win:
            payoff = abs(exit_price - entry) / entry if entry > 0 else 0.01
        else:
            payoff = abs(exit_price - entry) / entry if entry > 0 else 0.01

        self._history[key].append((win, payoff))
        logger.debug(
            "KellySizer record: %s/%s win=%s payoff=%.4f  history_len=%d",
            strategy, regime, win, payoff, len(self._history[key]),
        )

    def _compute_kelly(self, key: Tuple[str, str]) -> Tuple[float, float, float]:
        """
        Returns (kelly_fraction, win_rate, payoff_ratio).
        Falls back to fixed-risk if insufficient history.
        """
        history = self._history.get(key, deque())
        n = len(history)
        if n < _MIN_TRADES_FOR_KELLY:
            return _FIXED_RISK_PCT, 0.0, 0.0

        wins = [h for h in history if h[0]]
        losses = [h for h in history if not h[0]]
        win_rate = len(wins) / n

        avg_win_payoff = sum(h[1] for h in wins) / len(wins) if wins else 0.0
        avg_loss_payoff = sum(h[1] for h in losses) / len(losses) if losses else 0.0

        b = avg_win_payoff / avg_loss_payoff if avg_loss_payoff > 0 else avg_win_payoff
        p = win_rate
        q = 1 - p

        # Full Kelly: f* = (b*p - q) / b
        full_kelly = (b * p - q) / b if b > 0 else 0.0
        full_kelly = max(0.0, full_kelly)

        # Blend with fixed-risk
        blended = _KELLY_BLEND * _FIXED_RISK_PCT + (1 - _KELLY_BLEND) * full_kelly
        capped = min(blended, _MAX_KELLY_FRACTION)

        return capped, win_rate, (b if b > 0 else 0.0)

    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
        confidence: float = 1.0,
        regime: Optional[str] = None,
        strategy: str = "default",
    ) -> SizeResult:
        """
        Calculate position size using dynamic Kelly fraction.

        Parameters
        ----------
        capital : float
            Total available capital in AUD.
        entry_price : float
            Proposed entry price.
        stop_loss : float
            Stop-loss price.
        confidence : float
            Signal confidence [0, 1] used to scale down Kelly fraction.
        regime : str | None
            Current market regime.
        strategy : str
            Strategy name for Kelly history lookup.

        Returns
        -------
        SizeResult
        """
        key = self._key(strategy, regime)
        kelly_fraction, win_rate, payoff_ratio = self._compute_kelly(key)

        # Scale by confidence
        adjusted_fraction = kelly_fraction * confidence
        risk_capital = capital * adjusted_fraction

        stop_dist = abs(entry_price - stop_loss)
        if stop_dist <= 0 or entry_price <= 0:
            quantity = 0.0
            position_value = 0.0
        else:
            quantity = risk_capital / stop_dist
            position_value = quantity * entry_price

        logger.debug(
            "KellySizer: %s/%s  kelly=%.4f conf=%.2f  qty=%.6f  val=%.2f AUD",
            strategy, regime, kelly_fraction, confidence, quantity, position_value,
        )

        return SizeResult(
            quantity=max(0.0, quantity),
            position_value=max(0.0, position_value),
            kelly_fraction=adjusted_fraction,
            win_rate=win_rate,
            payoff_ratio=payoff_ratio,
            strategy=strategy,
            regime=str(regime) if regime else "UNKNOWN",
        )
