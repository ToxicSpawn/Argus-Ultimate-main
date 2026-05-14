"""
ATR-Adaptive Stops + Take-Profit Ladder.

Places stops at ``atr_mult`` × ATR below/above entry price and scales out
at 1R / 2R / 3R (R = ATR × atr_mult) to lock in gains progressively.

Scale-out schedule (default, configurable):
  Tranche 0: close 33% of position at 1R profit
  Tranche 1: close 33% of position at 2R profit
  Tranche 2: close remaining at 3R profit (or stop is never hit)

ATR is computed from the last ``atr_period`` highs/lows/closes using
Wilder's smoothing (equivalent to EMA with alpha = 1/atr_period).

Usage::

    mgr = ATRStopManager(atr_period=14, atr_mult=1.5)

    # Record each price bar to keep ATR current
    mgr.update_bar(symbol="BTC/USD", high=71200, low=70500, close=71000)

    # Register an open position on entry
    mgr.record_entry(symbol="BTC/USD", side="long", entry_price=71000, quantity=0.1)

    # On each new bar / price update
    result = mgr.check(symbol="BTC/USD", current_price=72500)
    # result.stop_hit          → True if stop triggered
    # result.exit_fraction     → fraction of position to close (0 if no action)
    # result.tranche           → "stop" | "1R" | "2R" | "3R" | None
    # result.stop_price        → current stop level
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default scale-out fractions at 1R / 2R / 3R
_DEFAULT_TRANCHES: List[Tuple[float, float]] = [
    (1.0, 0.33),   # at 1R profit: close 33%
    (2.0, 0.33),   # at 2R profit: close 33%
    (3.0, 1.00),   # at 3R profit: close rest (100% of remaining)
]


@dataclass
class StopCheckResult:
    """Result of a stop/take-profit check for one position."""
    symbol: str
    stop_hit: bool
    exit_fraction: float  # fraction of original quantity to close; 0 if no action
    tranche: Optional[str]  # "stop" | "1R" | "2R" | "3R" | None
    stop_price: float      # current stop level
    current_price: float
    atr: float
    reason: str


@dataclass
class _PositionState:
    side: str                # "long" | "short"
    entry_price: float
    quantity: float          # original quantity
    remaining_fraction: float = 1.0   # fraction not yet closed by take-profit
    atr_at_entry: float = 0.0
    stop_price: float = 0.0
    tranches_hit: List[int] = field(default_factory=list)


@dataclass
class _BarState:
    """Wilder ATR state per symbol."""
    prev_close: Optional[float] = None
    atr: float = 0.0
    n_bars: int = 0


class ATRStopManager:
    """
    ATR-adaptive stop + take-profit ladder manager.

    Parameters
    ----------
    atr_period : int
        Wilder smoothing period for ATR (default 14).
    atr_mult : float
        Stop distance in ATR multiples (default 1.5).
    tranches : list of (r_multiple, fraction), optional
        Take-profit scale-out schedule. Each tuple is (R multiple, fraction of
        remaining position to close). If None, uses the default 1R/2R/3R schedule.
    """

    def __init__(
        self,
        atr_period: int = 14,
        atr_mult: float = 1.5,
        tranches: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        self.atr_period = int(atr_period)
        self.atr_mult   = float(atr_mult)
        self.tranches   = tranches if tranches is not None else _DEFAULT_TRANCHES
        self._bars: Dict[str, _BarState] = {}
        self._positions: Dict[str, _PositionState] = {}
        self._symbol_mults: Dict[str, float] = {}  # per-symbol ATR multiplier overrides

    def set_multiplier(self, symbol: str, mult: float) -> None:
        """Override the ATR multiplier for a specific symbol (e.g. driven by vol regime).

        Affects new ``record_entry()`` stop distances and R-multiple calculations in
        ``check()`` for that symbol.  Falls back to ``self.atr_mult`` when not set.
        """
        self._symbol_mults[symbol] = float(mult)

    # ── Price feed ────────────────────────────────────────────────────────────

    def update_bar(
        self,
        symbol: str,
        high: float,
        low: float,
        close: float,
    ) -> float:
        """
        Feed a new OHLC bar and update the ATR for ``symbol``.

        Returns the updated ATR.
        """
        if symbol not in self._bars:
            self._bars[symbol] = _BarState()
        state = self._bars[symbol]

        if state.prev_close is None:
            # First bar — can't compute True Range yet
            state.prev_close = float(close)
            return 0.0

        tr = max(
            float(high) - float(low),
            abs(float(high) - state.prev_close),
            abs(float(low)  - state.prev_close),
        )

        alpha = 1.0 / self.atr_period
        if state.n_bars == 0:
            state.atr = tr
        else:
            state.atr = state.atr * (1.0 - alpha) + tr * alpha

        state.n_bars += 1
        state.prev_close = float(close)
        return state.atr

    def current_atr(self, symbol: str) -> float:
        """Return the current ATR for ``symbol`` (0.0 if unknown)."""
        return float(self._bars.get(symbol, _BarState()).atr)

    # ── Position management ───────────────────────────────────────────────────

    def record_entry(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
    ) -> float:
        """
        Register a new open position.

        Returns the stop price set for this entry.
        """
        atr = self.current_atr(symbol)
        _mult = self._symbol_mults.get(symbol, self.atr_mult)
        stop_dist = atr * _mult if atr > 0 else entry_price * 0.02

        if side.lower() == "long":
            stop_price = entry_price - stop_dist
        else:
            stop_price = entry_price + stop_dist

        self._positions[symbol] = _PositionState(
            side=side.lower(),
            entry_price=float(entry_price),
            quantity=float(quantity),
            remaining_fraction=1.0,
            atr_at_entry=atr,
            stop_price=stop_price,
            tranches_hit=[],
        )
        logger.info(
            "ATRStopManager: entry %s %s @ %.4f  stop=%.4f  ATR=%.4f",
            symbol, side, entry_price, stop_price, atr,
        )
        return stop_price

    def close_position(self, symbol: str) -> None:
        """Remove a position from tracking (called after it has been fully closed)."""
        self._positions.pop(symbol, None)

    # ── Check ─────────────────────────────────────────────────────────────────

    def check(self, symbol: str, current_price: float) -> StopCheckResult:
        """
        Evaluate whether a stop or take-profit tranche should be triggered.

        Returns a ``StopCheckResult``; caller should act on ``stop_hit`` and
        ``exit_fraction`` (fraction of original quantity to close).
        """
        pos = self._positions.get(symbol)
        atr = self.current_atr(symbol)

        if pos is None:
            return StopCheckResult(
                symbol=symbol, stop_hit=False, exit_fraction=0.0,
                tranche=None, stop_price=0.0,
                current_price=float(current_price), atr=atr, reason="no position",
            )

        cp = float(current_price)
        r  = pos.atr_at_entry * self._symbol_mults.get(symbol, self.atr_mult)  # 1R in price units

        if pos.side == "long":
            profit = cp - pos.entry_price
            stop_hit = cp <= pos.stop_price
        else:
            profit = pos.entry_price - cp
            stop_hit = cp >= pos.stop_price

        # Stop hit — close everything remaining
        if stop_hit:
            exit_frac = pos.remaining_fraction
            logger.info(
                "ATRStopManager: STOP HIT %s %s @ %.4f  stop=%.4f",
                symbol, pos.side, cp, pos.stop_price,
            )
            self.close_position(symbol)
            return StopCheckResult(
                symbol=symbol, stop_hit=True,
                exit_fraction=exit_frac,
                tranche="stop",
                stop_price=pos.stop_price,
                current_price=cp, atr=atr,
                reason=f"Stop hit: price {cp:.4f} through stop {pos.stop_price:.4f}",
            )

        # Take-profit tranches
        if r > 0:
            for idx, (r_mult, close_frac) in enumerate(self.tranches):
                if idx in pos.tranches_hit:
                    continue
                if profit >= r_mult * r:
                    # Fraction of *original* quantity to close
                    actual_frac = close_frac * pos.remaining_fraction
                    pos.tranches_hit.append(idx)
                    pos.remaining_fraction = max(0.0, pos.remaining_fraction - actual_frac)
                    tranche_name = f"{r_mult:.0f}R"
                    reason = (
                        f"TP tranche {tranche_name}: profit {profit:.4f} ≥ "
                        f"{r_mult:.0f}×R ({r:.4f}) — close {actual_frac:.1%}"
                    )
                    logger.info("ATRStopManager: %s %s", symbol, reason)

                    # If all tranches exhausted, remove position
                    if pos.remaining_fraction <= 0.0:
                        self.close_position(symbol)

                    return StopCheckResult(
                        symbol=symbol, stop_hit=False,
                        exit_fraction=actual_frac,
                        tranche=tranche_name,
                        stop_price=pos.stop_price,
                        current_price=cp, atr=atr,
                        reason=reason,
                    )

        return StopCheckResult(
            symbol=symbol, stop_hit=False, exit_fraction=0.0,
            tranche=None, stop_price=pos.stop_price,
            current_price=cp, atr=atr, reason="",
        )

    # ── Inspect ───────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Return current state for diagnostics."""
        return {
            "positions": {
                sym: {
                    "side": p.side,
                    "entry_price": p.entry_price,
                    "stop_price": p.stop_price,
                    "remaining_fraction": round(p.remaining_fraction, 4),
                    "atr_at_entry": round(p.atr_at_entry, 4),
                    "tranches_hit": p.tranches_hit,
                }
                for sym, p in self._positions.items()
            },
            "atrs": {sym: round(b.atr, 4) for sym, b in self._bars.items()},
            "atr_mult": self.atr_mult,
            "symbol_mults": dict(self._symbol_mults),
        }
