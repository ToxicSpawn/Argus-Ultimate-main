"""
alpha/adverse_selection_filter.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pre-trade adverse selection filter — checks OBI/VPIN before every quote
placement to protect the market-maker from informed flow.

Filter pipeline (applied sequentially)
---------------------------------------
1. OBI z-score gate  — deny if |OBI z-score| > obi_zscore_threshold
2. VPIN gate         — deny if VPIN > vpin_threshold
3. Microprice drift  — deny if drift > microprice_drift_threshold
4. Own adverse rate  — deny (symbol halt) if our own fill adverse rate exceeds
                       max_own_adverse_rate

Integration
-----------
  Imports OFISignal from alpha/microstructure/live_ofi_stream.py and
  VPINSignal types from alpha/microstructure/live_vpin_stream.py for type
  hints; concrete signal values are passed as plain floats so this module
  is usable without a live feed in tests.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases / reference imports (for documentation / integration only)
# ---------------------------------------------------------------------------
# We import these for type-annotation and documentation purposes.
# The filter accepts plain float inputs so it can be tested standalone.
try:
    from alpha.microstructure.live_ofi_stream import OFISignal  # noqa: F401
    from alpha.microstructure.live_vpin_stream import _SymbolState as _VPINSymbolState  # noqa: F401
except ImportError:  # pragma: no cover
    pass  # Allow the filter to run in isolation if microstructure not installed


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ASFilterConfig:
    """Configuration for AdverseSelectionFilter.

    Parameters
    ----------
    obi_zscore_threshold : float
        Deny quoting if |OBI z-score| exceeds this value.
        OBI z-score is typically provided by LiveOFIStream.get_ofi_zscore().
    vpin_threshold : float
        Deny quoting if VPIN exceeds this value.
        VPIN from LiveVPINStream.get_vpin() in [0, 1].
    delay_on_signal_ms : float
        Milliseconds to wait before re-checking after a bad signal.
    microprice_drift_threshold : float
        Deny quoting if |microprice drift| in the adverse direction exceeds this.
        Microprice drift is in units of basis points per tick (or normalised [0,1]).
    lookback_fills : int
        Rolling window length for our own adverse fill tracking.
    max_own_adverse_rate : float
        Halt a symbol if our own fill adverse rate exceeds this fraction.
    """

    obi_zscore_threshold: float = 1.8
    vpin_threshold: float = 0.65
    delay_on_signal_ms: float = 500.0
    microprice_drift_threshold: float = 0.6
    lookback_fills: int = 20
    max_own_adverse_rate: float = 0.5


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ASCheckResult:
    """Result of a pre-trade adverse selection check.

    Attributes
    ----------
    allowed : bool
        True if all filters passed.
    reason : str
        "all_clear" if allowed; otherwise the name of the filter that denied.
    delay_ms : float
        0.0 if allowed; delay_on_signal_ms if denied.
    obi_zscore : float
        Absolute OBI z-score at the time of the check.
    vpin : float
        VPIN value at the time of the check.
    drift : float
        Microprice drift value at the time of the check.
    own_adverse_rate : float
        Our own fill adverse rate at the time of the check.
    """

    allowed: bool
    reason: str
    delay_ms: float
    obi_zscore: float
    vpin: float
    drift: float
    own_adverse_rate: float


# ---------------------------------------------------------------------------
# Internal per-symbol state
# ---------------------------------------------------------------------------


@dataclass
class _SymbolFilterState:
    """Mutable per-symbol filter state."""

    # Rolling window of our fill outcomes: True = adverse, False = benign
    fill_outcomes: Deque[bool] = field(default_factory=lambda: deque())

    # Per-filter denial counters
    denied_obi: int = 0
    denied_vpin: int = 0
    denied_drift: int = 0
    denied_own_adverse: int = 0
    total_checks: int = 0

    # Delay tracking
    total_delay_ms: float = 0.0
    delayed_count: int = 0

    # Halt state
    halted: bool = False
    last_check_ns: int = 0

    def adverse_rate(self) -> float:
        """Fraction of recent fills that were adverse."""
        outcomes = list(self.fill_outcomes)
        if not outcomes:
            return 0.0
        return sum(outcomes) / len(outcomes)

    def denial_rate(self) -> float:
        """Fraction of checks that were denied."""
        if self.total_checks == 0:
            return 0.0
        denied = self.denied_obi + self.denied_vpin + self.denied_drift + self.denied_own_adverse
        return denied / self.total_checks

    def avg_delay_ms(self) -> float:
        """Average delay issued per check (0 for allowed checks)."""
        if self.total_checks == 0:
            return 0.0
        return self.total_delay_ms / self.total_checks


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class AdverseSelectionFilter:
    """
    Pre-trade filter that blocks quote placement when adverse selection risk
    is elevated.

    Typical usage
    -------------
    ::

        asf = AdverseSelectionFilter(ASFilterConfig())

        # Before placing a quote:
        result = asf.pre_trade_check(
            symbol="BTC-USD",
            side="buy",
            ofi_signal=ofi_stream.get_signal("BTC-USD").ofi_zscore,
            vpin=vpin_stream.get_vpin("BTC-USD"),
            microprice_drift=drift_model.get_drift("BTC-USD"),
        )
        if result.allowed:
            place_quote(...)
        else:
            schedule_recheck_after(result.delay_ms)

        # After our fill resolves:
        asf.on_our_fill("BTC-USD", "buy", fill_price=29500.0, post_mid_500ms=29490.0)
    """

    def __init__(self, config: ASFilterConfig) -> None:
        self._cfg = config
        self._states: Dict[str, _SymbolFilterState] = {}

    # ------------------------------------------------------------------
    # Pre-trade gate
    # ------------------------------------------------------------------

    def pre_trade_check(
        self,
        symbol: str,
        side: str,
        ofi_signal: float,
        vpin: float,
        microprice_drift: float,
    ) -> ASCheckResult:
        """Run all adverse-selection filters for a prospective quote.

        Parameters
        ----------
        symbol : str
            Instrument identifier.
        side : str
            Quote side — "buy" (bid) or "sell" (ask).
        ofi_signal : float
            OBI z-score from LiveOFIStream.  Positive = buy pressure.
        vpin : float
            VPIN in [0, 1] from LiveVPINStream.
        microprice_drift : float
            Signed microprice drift; positive = upward drift.
            Provide from alpha/microstructure/microprice_drift.py.

        Returns
        -------
        ASCheckResult
        """
        state = self._get_or_create(symbol)
        state.total_checks += 1
        state.last_check_ns = time.monotonic_ns()

        cfg = self._cfg
        abs_obi = abs(ofi_signal)
        own_rate = state.adverse_rate()

        # Determine adverse drift direction:
        # - For a buy (bid), we're hurt if price goes *down* → negative drift
        # - For a sell (ask), we're hurt if price goes *up* → positive drift
        if side.lower() == "buy":
            adverse_drift = -microprice_drift  # negative drift adverse for bids
        else:
            adverse_drift = microprice_drift   # positive drift adverse for asks

        # --- Filter 1: OBI z-score ---
        if abs_obi > cfg.obi_zscore_threshold:
            state.denied_obi += 1
            state.total_delay_ms += cfg.delay_on_signal_ms
            state.delayed_count += 1
            logger.debug(
                "AS filter [%s] denied OBI: |zscore|=%.3f > %.3f",
                symbol, abs_obi, cfg.obi_zscore_threshold,
            )
            return ASCheckResult(
                allowed=False,
                reason="denied_obi_zscore",
                delay_ms=cfg.delay_on_signal_ms,
                obi_zscore=abs_obi,
                vpin=vpin,
                drift=microprice_drift,
                own_adverse_rate=own_rate,
            )

        # --- Filter 2: VPIN ---
        if vpin > cfg.vpin_threshold:
            state.denied_vpin += 1
            state.total_delay_ms += cfg.delay_on_signal_ms
            state.delayed_count += 1
            logger.debug(
                "AS filter [%s] denied VPIN: %.4f > %.4f",
                symbol, vpin, cfg.vpin_threshold,
            )
            return ASCheckResult(
                allowed=False,
                reason="denied_vpin",
                delay_ms=cfg.delay_on_signal_ms,
                obi_zscore=abs_obi,
                vpin=vpin,
                drift=microprice_drift,
                own_adverse_rate=own_rate,
            )

        # --- Filter 3: Microprice drift in adverse direction ---
        if adverse_drift > cfg.microprice_drift_threshold:
            state.denied_drift += 1
            state.total_delay_ms += cfg.delay_on_signal_ms
            state.delayed_count += 1
            logger.debug(
                "AS filter [%s] denied microprice drift: %.4f > %.4f (side=%s)",
                symbol, adverse_drift, cfg.microprice_drift_threshold, side,
            )
            return ASCheckResult(
                allowed=False,
                reason="denied_microprice_drift",
                delay_ms=cfg.delay_on_signal_ms,
                obi_zscore=abs_obi,
                vpin=vpin,
                drift=microprice_drift,
                own_adverse_rate=own_rate,
            )

        # --- Filter 4: Own adverse fill rate ---
        if own_rate > cfg.max_own_adverse_rate:
            state.denied_own_adverse += 1
            state.total_delay_ms += cfg.delay_on_signal_ms
            state.delayed_count += 1
            state.halted = True
            logger.warning(
                "AS filter [%s] denied own adverse rate: %.3f > %.3f — HALTING",
                symbol, own_rate, cfg.max_own_adverse_rate,
            )
            return ASCheckResult(
                allowed=False,
                reason="denied_own_adverse_rate",
                delay_ms=cfg.delay_on_signal_ms,
                obi_zscore=abs_obi,
                vpin=vpin,
                drift=microprice_drift,
                own_adverse_rate=own_rate,
            )

        # All filters passed
        logger.debug("AS filter [%s] all_clear (side=%s)", symbol, side)
        return ASCheckResult(
            allowed=True,
            reason="all_clear",
            delay_ms=0.0,
            obi_zscore=abs_obi,
            vpin=vpin,
            drift=microprice_drift,
            own_adverse_rate=own_rate,
        )

    # ------------------------------------------------------------------
    # Fill feedback
    # ------------------------------------------------------------------

    def on_our_fill(
        self,
        symbol: str,
        side: str,
        fill_price: float,
        post_mid_500ms: float,
    ) -> None:
        """Record a fill outcome for adverse selection tracking.

        A fill is considered *adverse* if the market moves against us
        within 500ms of the fill:
          - Long (buy fill): mid drops (post_mid < fill_price)
          - Short (sell fill): mid rises (post_mid > fill_price)

        Parameters
        ----------
        symbol : str
            Instrument identifier.
        side : str
            "buy" or "sell".
        fill_price : float
            Execution price.
        post_mid_500ms : float
            Mid-price 500ms after the fill.
        """
        state = self._get_or_create(symbol)

        if side.lower() == "buy":
            adverse = post_mid_500ms < fill_price
        else:  # sell / ask
            adverse = post_mid_500ms > fill_price

        outcomes = state.fill_outcomes
        outcomes.append(adverse)
        # Trim to lookback window
        while len(outcomes) > self._cfg.lookback_fills:
            outcomes.popleft()

        # Re-evaluate halt status
        new_rate = state.adverse_rate()
        if new_rate <= self._cfg.max_own_adverse_rate and state.halted:
            state.halted = False
            logger.info(
                "AS filter [%s] adverse rate recovered to %.3f — resuming",
                symbol, new_rate,
            )

        logger.debug(
            "AS filter [%s] fill recorded: side=%s adverse=%s rate=%.3f",
            symbol, side, adverse, new_rate,
        )

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_our_adverse_rate(self, symbol: str) -> float:
        """Return fraction of recent fills for *symbol* that were adverse."""
        state = self._states.get(symbol)
        if state is None:
            return 0.0
        return state.adverse_rate()

    def should_halt_symbol(self, symbol: str) -> bool:
        """Return True if the own adverse fill rate exceeds the threshold."""
        state = self._states.get(symbol)
        if state is None:
            return False
        return state.adverse_rate() > self._cfg.max_own_adverse_rate

    def get_filter_stats(self, symbol: str) -> dict:
        """Return per-symbol filter statistics.

        Keys
        ----
        total_checks, denied_obi, denied_vpin, denied_drift,
        denied_own_adverse, denial_rate, own_adverse_rate, avg_delay_ms,
        halted, lookback_fills
        """
        state = self._states.get(symbol)
        if state is None:
            return {
                "total_checks": 0,
                "denied_obi": 0,
                "denied_vpin": 0,
                "denied_drift": 0,
                "denied_own_adverse": 0,
                "denial_rate": 0.0,
                "own_adverse_rate": 0.0,
                "avg_delay_ms": 0.0,
                "halted": False,
                "lookback_fills": 0,
            }
        return {
            "total_checks": state.total_checks,
            "denied_obi": state.denied_obi,
            "denied_vpin": state.denied_vpin,
            "denied_drift": state.denied_drift,
            "denied_own_adverse": state.denied_own_adverse,
            "denial_rate": state.denial_rate(),
            "own_adverse_rate": state.adverse_rate(),
            "avg_delay_ms": state.avg_delay_ms(),
            "halted": state.halted,
            "lookback_fills": len(state.fill_outcomes),
        }

    def get_global_stats(self) -> dict:
        """Return aggregate statistics across all tracked symbols.

        Keys
        ----
        symbols, total_checks, total_denied_obi, total_denied_vpin,
        total_denied_drift, total_denied_own_adverse, overall_denial_rate,
        halted_symbols
        """
        symbols = list(self._states.keys())
        total_checks = 0
        total_denied_obi = 0
        total_denied_vpin = 0
        total_denied_drift = 0
        total_denied_own_adverse = 0
        halted_symbols: List[str] = []

        for sym, state in self._states.items():
            total_checks += state.total_checks
            total_denied_obi += state.denied_obi
            total_denied_vpin += state.denied_vpin
            total_denied_drift += state.denied_drift
            total_denied_own_adverse += state.denied_own_adverse
            if state.halted or state.adverse_rate() > self._cfg.max_own_adverse_rate:
                halted_symbols.append(sym)

        total_denied = (
            total_denied_obi + total_denied_vpin
            + total_denied_drift + total_denied_own_adverse
        )
        overall_denial_rate = total_denied / total_checks if total_checks > 0 else 0.0

        return {
            "symbols": symbols,
            "total_checks": total_checks,
            "total_denied_obi": total_denied_obi,
            "total_denied_vpin": total_denied_vpin,
            "total_denied_drift": total_denied_drift,
            "total_denied_own_adverse": total_denied_own_adverse,
            "overall_denial_rate": overall_denial_rate,
            "halted_symbols": halted_symbols,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, symbol: str) -> _SymbolFilterState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolFilterState()
        return self._states[symbol]

    def reset(self, symbol: str) -> None:
        """Clear all state for *symbol*."""
        if symbol in self._states:
            del self._states[symbol]
            logger.info("AdverseSelectionFilter: reset state for %s", symbol)

    def symbols(self) -> List[str]:
        """Return list of all tracked symbols."""
        return list(self._states.keys())

    def __repr__(self) -> str:  # pragma: no cover
        n = len(self._states)
        return f"<AdverseSelectionFilter symbols={n} cfg={self._cfg}>"
