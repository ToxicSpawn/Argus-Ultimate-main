"""
alpha/regime_scheduler.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Volatility regime scheduling — dynamically adjusts spreads by time-of-day
and detected market regime.

Integrates
----------
  - RegimeDetector (alpha/regime_detector.py) — live regime signal
  - SessionSpreadSchedule (execution/session_spread_schedule.py) — clock-based
    liquidity multiplier

Spread multiplier table
-----------------------
  HIGH_VOLATILITY          : ×2.0   (widen dramatically)
  TRENDING_UP/TRENDING_DOWN: ×1.5   + halt quoting
  MEAN_REVERTING (conf>0.7): ×0.85  (tighten for statistical edge)
  UNKNOWN / LOW_LIQUIDITY  : ×1.3   (cautious)

The session multiplier from SessionSpreadSchedule is applied *first*; the
regime multiplier compounds on top.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional

from alpha.regime_detector import MarketRegime, RegimeDetector
from execution.session_spread_schedule import SessionSpreadSchedule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class QuoteParams:
    """Complete set of quoting parameters returned per symbol.

    Attributes
    ----------
    effective_spread_bps : float
        Final spread in basis points after all multipliers applied.
    should_quote : bool
        Whether the bot should actively place quotes for this symbol.
    regime : MarketRegime
        Current detected regime.
    confidence : float
        Regime classification confidence in [0, 1].
    multiplier : float
        Total spread multiplier (session × regime combined).
    reason : str
        Human-readable explanation of the decision.
    """

    effective_spread_bps: float
    should_quote: bool
    regime: MarketRegime
    confidence: float
    multiplier: float
    reason: str


# ---------------------------------------------------------------------------
# Regime scheduler
# ---------------------------------------------------------------------------


class RegimeScheduler:
    """
    Combines session-based and regime-based spread adjustments.

    Parameters
    ----------
    regime_detector : RegimeDetector
        Live regime signal source.
    spread_schedule : SessionSpreadSchedule
        Clock-based spread schedule from execution layer.
    """

    # Regime-specific multiplier table (applied on top of session multiplier)
    _REGIME_MULTIPLIER: Dict[MarketRegime, float] = {
        MarketRegime.HIGH_VOLATILITY: 2.0,
        MarketRegime.TRENDING_UP: 1.5,
        MarketRegime.TRENDING_DOWN: 1.5,
        MarketRegime.UNKNOWN: 1.3,
        MarketRegime.LOW_LIQUIDITY: 1.3,
        MarketRegime.MEAN_REVERTING: 1.0,   # overridden by confidence check
    }

    # Mean-reverting tightening only when confidence is high
    _MR_HIGH_CONFIDENCE_MULTIPLIER: float = 0.85
    _MR_HIGH_CONFIDENCE_THRESHOLD: float = 0.7

    # Minimum confidence to quote at all
    _MIN_QUOTE_CONFIDENCE: float = 0.3

    # MEAN_REVERTING requires at least this confidence to quote
    _MR_MIN_QUOTE_CONFIDENCE: float = 0.4

    def __init__(
        self,
        regime_detector: RegimeDetector,
        spread_schedule: SessionSpreadSchedule,
    ) -> None:
        self._detector = regime_detector
        self._schedule = spread_schedule

        # Session-level statistics
        self._stats_start_ns: int = time.monotonic_ns()
        self._ticks_quoting: Dict[str, int] = defaultdict(int)
        self._ticks_halted: Dict[str, int] = defaultdict(int)
        self._regime_ticks: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    # ------------------------------------------------------------------
    # Core public methods
    # ------------------------------------------------------------------

    def get_effective_spread_multiplier(self, symbol: str) -> float:
        """Return the total spread multiplier for *symbol*.

        Combines:
        1. SessionSpreadSchedule clock-based multiplier.
        2. Regime-specific multiplier (may be tightened or widened).
        """
        session_mult = self._schedule.get_current_spread_multiplier()
        regime_mult = self._regime_multiplier(symbol)
        return session_mult * regime_mult

    def should_quote(self, symbol: str) -> bool:
        """Return True if it is safe to place passive MM quotes for *symbol*.

        Halted when:
          - Regime is TRENDING_UP or TRENDING_DOWN
          - Regime is HIGH_VOLATILITY
          - Regime confidence < MIN_QUOTE_CONFIDENCE (0.3)

        Allowed when:
          - Regime is MEAN_REVERTING with confidence >= 0.4
        """
        regime = self._detector.get_regime(symbol)
        confidence = self._detector.get_regime_confidence(symbol)

        # Hard halts
        if regime in (
            MarketRegime.TRENDING_UP,
            MarketRegime.TRENDING_DOWN,
            MarketRegime.HIGH_VOLATILITY,
        ):
            return False

        # Insufficient confidence
        if confidence < self._MIN_QUOTE_CONFIDENCE:
            return False

        # Mean-reverting requires a slightly higher bar
        if regime == MarketRegime.MEAN_REVERTING:
            return confidence >= self._MR_MIN_QUOTE_CONFIDENCE

        # LOW_LIQUIDITY / UNKNOWN — don't quote by default
        return False

    def get_quote_params(
        self, symbol: str, base_spread_bps: float
    ) -> QuoteParams:
        """Return full quoting parameters for *symbol*.

        Parameters
        ----------
        symbol : str
            Instrument identifier.
        base_spread_bps : float
            Caller-provided base spread before any multipliers.

        Returns
        -------
        QuoteParams
            All fields populated with current regime and spread information.
        """
        regime = self._detector.get_regime(symbol)
        confidence = self._detector.get_regime_confidence(symbol)
        session_mult = self._schedule.get_current_spread_multiplier()
        regime_mult = self._regime_multiplier(symbol)
        total_mult = session_mult * regime_mult
        effective_bps = base_spread_bps * total_mult
        quoting = self.should_quote(symbol)
        reason = self._build_reason(symbol, regime, confidence, session_mult, regime_mult, quoting)

        # Update statistics
        if quoting:
            self._ticks_quoting[symbol] += 1
        else:
            self._ticks_halted[symbol] += 1
        self._regime_ticks[symbol][regime.value] += 1

        return QuoteParams(
            effective_spread_bps=effective_bps,
            should_quote=quoting,
            regime=regime,
            confidence=confidence,
            multiplier=total_mult,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_session_stats(self) -> dict:
        """Return aggregate statistics for the current session.

        Keys
        ----
        time_quoting_pct : float
            Fraction of ticks (across all symbols) when quoting was active.
        time_halted_pct : float
            Fraction of ticks when quoting was halted.
        regime_breakdown : dict
            Per-regime tick counts summed across all symbols.
        symbols : list[str]
            Symbols that have been queried.
        elapsed_s : float
            Elapsed wall-clock seconds since the scheduler was created.
        """
        total_quoting = sum(self._ticks_quoting.values())
        total_halted = sum(self._ticks_halted.values())
        total = total_quoting + total_halted

        time_quoting_pct = total_quoting / total if total > 0 else 0.0
        time_halted_pct = total_halted / total if total > 0 else 0.0

        # Aggregate regime breakdown
        regime_breakdown: Dict[str, int] = defaultdict(int)
        for sym_dict in self._regime_ticks.values():
            for reg_name, count in sym_dict.items():
                regime_breakdown[reg_name] += count

        elapsed_s = (time.monotonic_ns() - self._stats_start_ns) / 1e9

        return {
            "time_quoting_pct": time_quoting_pct,
            "time_halted_pct": time_halted_pct,
            "regime_breakdown": dict(regime_breakdown),
            "symbols": list(self._ticks_quoting.keys()),
            "elapsed_s": elapsed_s,
            "total_ticks": total,
        }

    def reset_stats(self) -> None:
        """Reset session statistics counters."""
        self._ticks_quoting.clear()
        self._ticks_halted.clear()
        self._regime_ticks.clear()
        self._stats_start_ns = time.monotonic_ns()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _regime_multiplier(self, symbol: str) -> float:
        """Compute the regime component of the spread multiplier."""
        regime = self._detector.get_regime(symbol)
        confidence = self._detector.get_regime_confidence(symbol)

        base = self._REGIME_MULTIPLIER.get(regime, 1.3)

        # Special case: mean-reverting with high confidence → tighten
        if (
            regime == MarketRegime.MEAN_REVERTING
            and confidence >= self._MR_HIGH_CONFIDENCE_THRESHOLD
        ):
            return self._MR_HIGH_CONFIDENCE_MULTIPLIER

        return base

    @staticmethod
    def _build_reason(
        symbol: str,
        regime: MarketRegime,
        confidence: float,
        session_mult: float,
        regime_mult: float,
        quoting: bool,
    ) -> str:
        """Construct a human-readable explanation string."""
        action = "QUOTING" if quoting else "HALTED"
        parts = [
            f"{action} {symbol}",
            f"regime={regime.value}",
            f"confidence={confidence:.2f}",
            f"session_mult={session_mult:.3f}",
            f"regime_mult={regime_mult:.3f}",
        ]
        if not quoting:
            if regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
                parts.append("reason=trending_halt")
            elif regime == MarketRegime.HIGH_VOLATILITY:
                parts.append("reason=high_vol_halt")
            elif confidence < 0.3:
                parts.append("reason=low_confidence_halt")
            else:
                parts.append("reason=unknown_regime_halt")
        else:
            if regime == MarketRegime.MEAN_REVERTING and regime_mult < 1.0:
                parts.append("reason=mr_tightening")
            else:
                parts.append("reason=normal")
        return " | ".join(parts)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<RegimeScheduler detector={self._detector!r} "
            f"schedule={self._schedule!r}>"
        )
