"""
alpha/optimal_spread_calibrator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Per-symbol dynamic spread-width calibrator using live fill data.

Two spread estimates are blended
---------------------------------
1. **Avellaneda-Stoikov (AS) formula** — theory-driven:

       s* = γ × σ² × T + (2/γ) × ln(1 + γ/κ)

   where:
     γ = risk aversion (default 0.1)
     σ = current annualised volatility (estimated from recent returns)
     T = time horizon (5-minute bar = 1/12 hour)
     κ = fill-rate sensitivity (inferred from observed fill rate)

2. **Empirical fill-rate adjustment** — data-driven:

   If fill_rate > fill_rate_target  → spread is too tight  → widen by (1 + overshoot)
   If fill_rate < fill_rate_target  → spread is too wide   → tighten by (1 - undershoot)

Blend: 70% AS formula + 30% empirical, then clamped to [min_spread_bps, max_spread_bps].

Calibration is triggered every `update_interval_fills` fills.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CalibConfig:
    """Calibration configuration for OptimalSpreadCalibrator.

    Parameters
    ----------
    base_spread_bps : float
        Default starting spread in basis points.
    min_spread_bps : float
        Hard floor — never quote tighter than this.
    max_spread_bps : float
        Hard ceiling — never quote wider than this.
    vol_sensitivity : float
        Linear scale factor applied to AS formula volatility term.
    fill_rate_target : float
        Target fill rate per quote lifetime (fraction).
    calibration_window : int
        Number of fill events to include in calibration window.
    update_interval_fills : int
        Recalibrate every this many new fills.
    gamma : float
        Risk aversion parameter for Avellaneda-Stoikov formula.
    T : float
        Time horizon in hours (default: 5 minutes = 1/12 hour).
    vol_lookback : int
        Number of price ticks used for volatility estimation.
    annualise_factor : float
        Annualisation factor matching the tick frequency.
        Default: sqrt(252 * 24 * 12) for 5-minute bars.
    """

    base_spread_bps: float = 30.0
    min_spread_bps: float = 5.0
    max_spread_bps: float = 200.0
    vol_sensitivity: float = 1.5
    fill_rate_target: float = 0.4
    calibration_window: int = 200
    update_interval_fills: int = 20
    gamma: float = 0.1
    T: float = 1.0 / 12.0  # 5-minute horizon in hours
    vol_lookback: int = 50
    annualise_factor: float = math.sqrt(252 * 24 * 12)


# ---------------------------------------------------------------------------
# Internal records
# ---------------------------------------------------------------------------


@dataclass
class _QuoteRecord:
    """One sent quote."""

    bid: float
    ask: float
    spread_bps: float
    timestamp_ns: int
    filled: bool = False
    fill_side: Optional[str] = None     # "buy" or "sell"
    fill_spread_bps: Optional[float] = None
    was_adverse: Optional[bool] = None


@dataclass
class _FillRecord:
    """One fill event."""

    side: str
    fill_price: float
    spread_bps_at_fill: float
    was_adverse: bool
    timestamp_ns: int


# ---------------------------------------------------------------------------
# Per-symbol state
# ---------------------------------------------------------------------------


@dataclass
class _SymbolState:
    """All mutable calibration state for one symbol."""

    # Rolling price buffer for volatility estimation
    prices: Deque[float] = field(default_factory=lambda: deque(maxlen=200))

    # Quote and fill history
    quotes: Deque[_QuoteRecord] = field(default_factory=lambda: deque(maxlen=500))
    fills: Deque[_FillRecord] = field(default_factory=lambda: deque(maxlen=500))

    # Counters
    quotes_sent: int = 0
    fills_received: int = 0
    fills_since_calibration: int = 0

    # Calibrated values
    optimal_bps: float = 30.0       # initialised to base
    as_bps: float = 30.0
    empirical_bps: float = 30.0
    last_calibrated_ns: int = 0

    # Cached metrics
    cached_vol: float = 0.0
    cached_fill_rate: float = 0.0
    cached_adverse_rate: float = 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class OptimalSpreadCalibrator:
    """
    Dynamically calibrates per-symbol spread width using live fill data.

    Usage
    -----
    ::

        cfg = CalibConfig(base_spread_bps=30.0, fill_rate_target=0.4)
        calib = OptimalSpreadCalibrator(cfg)

        # Feed price ticks for volatility
        calib.on_price("BTC-USD", 29500.0)

        # Record quotes sent
        calib.on_quote_sent("BTC-USD", bid=29490.0, ask=29510.0,
                             spread_bps=13.56, timestamp_ns=ts)

        # Record fills
        calib.on_fill("BTC-USD", side="buy", fill_price=29490.0,
                      spread_bps_at_fill=13.56, was_adverse=False)

        # Get current optimal spread
        spread = calib.get_optimal_spread("BTC-USD")
    """

    def __init__(self, config: CalibConfig) -> None:
        self._cfg = config
        self._states: Dict[str, _SymbolState] = {}

    # ------------------------------------------------------------------
    # Feed methods
    # ------------------------------------------------------------------

    def on_price(self, symbol: str, price: float) -> None:
        """Feed a price tick for volatility estimation.

        Parameters
        ----------
        symbol : str
            Instrument identifier.
        price : float
            Mid-price or last trade price.
        """
        state = self._get_or_create(symbol)
        state.prices.append(price)

    def on_quote_sent(
        self,
        symbol: str,
        bid: float,
        ask: float,
        spread_bps: float,
        timestamp_ns: int,
    ) -> None:
        """Record that a two-sided quote was sent.

        Parameters
        ----------
        symbol : str
        bid : float
        ask : float
        spread_bps : float
            Spread in basis points at the time of quoting.
        timestamp_ns : int
            Nanosecond timestamp.
        """
        state = self._get_or_create(symbol)
        state.quotes.append(
            _QuoteRecord(
                bid=bid,
                ask=ask,
                spread_bps=spread_bps,
                timestamp_ns=timestamp_ns,
            )
        )
        state.quotes_sent += 1

    def on_fill(
        self,
        symbol: str,
        side: str,
        fill_price: float,
        spread_bps_at_fill: float,
        was_adverse: bool,
    ) -> None:
        """Record a fill event for calibration feedback.

        Parameters
        ----------
        symbol : str
        side : str
            "buy" or "sell".
        fill_price : float
        spread_bps_at_fill : float
            Spread that was being quoted when this fill occurred.
        was_adverse : bool
            True if price moved against us after the fill.
        """
        state = self._get_or_create(symbol)
        state.fills.append(
            _FillRecord(
                side=side,
                fill_price=fill_price,
                spread_bps_at_fill=spread_bps_at_fill,
                was_adverse=was_adverse,
                timestamp_ns=time.monotonic_ns(),
            )
        )
        state.fills_received += 1
        state.fills_since_calibration += 1

        # Trim fills to calibration window
        while len(state.fills) > self._cfg.calibration_window:
            state.fills.popleft()

        # Trigger recalibration if enough new fills have accumulated
        if state.fills_since_calibration >= self._cfg.update_interval_fills:
            self._calibrate(symbol, state)
            state.fills_since_calibration = 0

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_optimal_spread(self, symbol: str) -> float:
        """Return the current optimal spread in basis points for *symbol*.

        If no calibration has occurred yet, returns base_spread_bps.
        """
        state = self._states.get(symbol)
        if state is None:
            return self._cfg.base_spread_bps
        # If never calibrated, run it now
        if state.last_calibrated_ns == 0 and len(state.fills) > 0:
            self._calibrate(symbol, state)
        return state.optimal_bps

    def get_fill_rate(self, symbol: str) -> float:
        """Return recent fill rate (fills / quotes) for *symbol*."""
        state = self._states.get(symbol)
        if state is None:
            return 0.0
        return self._compute_fill_rate(state)

    def get_calibration_stats(self, symbol: str) -> dict:
        """Return calibration diagnostics for *symbol*.

        Keys
        ----
        current_optimal_bps, fill_rate, as_formula_bps, empirical_bps,
        quotes_sent, fills_received, adverse_fill_rate, last_calibrated_ns,
        cached_vol
        """
        state = self._states.get(symbol)
        if state is None:
            return {
                "current_optimal_bps": self._cfg.base_spread_bps,
                "fill_rate": 0.0,
                "as_formula_bps": self._cfg.base_spread_bps,
                "empirical_bps": self._cfg.base_spread_bps,
                "quotes_sent": 0,
                "fills_received": 0,
                "adverse_fill_rate": 0.0,
                "last_calibrated_ns": 0,
                "cached_vol": 0.0,
            }
        fills_list = list(state.fills)
        adverse_count = sum(1 for f in fills_list if f.was_adverse)
        adverse_rate = adverse_count / len(fills_list) if fills_list else 0.0

        return {
            "current_optimal_bps": state.optimal_bps,
            "fill_rate": state.cached_fill_rate,
            "as_formula_bps": state.as_bps,
            "empirical_bps": state.empirical_bps,
            "quotes_sent": state.quotes_sent,
            "fills_received": state.fills_received,
            "adverse_fill_rate": adverse_rate,
            "last_calibrated_ns": state.last_calibrated_ns,
            "cached_vol": state.cached_vol,
        }

    # ------------------------------------------------------------------
    # Calibration engine
    # ------------------------------------------------------------------

    def _calibrate(self, symbol: str, state: _SymbolState) -> None:
        """Run the full calibration and update state.optimal_bps."""
        cfg = self._cfg

        # 1. Estimate current volatility
        sigma = self._estimate_vol(state)
        state.cached_vol = sigma

        # 2. Compute fill rate
        fill_rate = self._compute_fill_rate(state)
        state.cached_fill_rate = fill_rate

        # 3. AS formula spread
        as_bps = self._avellaneda_stoikov(sigma, fill_rate)
        state.as_bps = as_bps

        # 4. Empirical fill-rate adjustment
        empirical_bps = self._empirical_adjustment(state, fill_rate)
        state.empirical_bps = empirical_bps

        # 5. Blend: 70% AS + 30% empirical
        blended = 0.70 * as_bps + 0.30 * empirical_bps

        # 6. Clamp
        optimal = max(cfg.min_spread_bps, min(cfg.max_spread_bps, blended))
        state.optimal_bps = optimal
        state.last_calibrated_ns = time.monotonic_ns()

        logger.debug(
            "Calibrator [%s]: AS=%.2f emp=%.2f blended=%.2f optimal=%.2f "
            "vol=%.4f fill_rate=%.3f",
            symbol, as_bps, empirical_bps, blended, optimal, sigma, fill_rate,
        )

    def _avellaneda_stoikov(self, sigma: float, fill_rate: float) -> float:
        """Compute the Avellaneda-Stoikov optimal spread.

        s* = γ × σ² × T + (2/γ) × ln(1 + γ/κ)

        κ is the fill-rate sensitivity parameter. We estimate κ from the
        observed fill rate using the approximation:

            κ ≈ -ln(fill_rate) / (base_spread_bps / 10_000)

        If fill_rate is 0 or too small, fall back to κ = 1.
        """
        cfg = self._cfg
        gamma = cfg.gamma
        T = cfg.T

        # Inventory risk component
        inventory_term = gamma * (sigma ** 2) * T * cfg.vol_sensitivity

        # Estimate κ from observed fill rate
        kappa = self._estimate_kappa(fill_rate)

        # Market impact component
        if kappa > 1e-9 and gamma > 1e-9:
            impact_term = (2.0 / gamma) * math.log(1.0 + gamma / kappa)
        else:
            impact_term = cfg.base_spread_bps / 10_000.0

        # Convert to basis points (sigma and inventory_term in fraction units)
        as_fraction = inventory_term + impact_term
        as_bps = as_fraction * 10_000.0

        # Guard against degenerate inputs
        if not math.isfinite(as_bps) or as_bps <= 0:
            as_bps = cfg.base_spread_bps

        return as_bps

    def _estimate_kappa(self, fill_rate: float) -> float:
        """Infer the fill-rate sensitivity κ from observed fill rate.

        Approximate inversion: κ = -ln(1 - fill_rate) / half_spread
        where half_spread = base_spread_bps / 20_000 (fraction).
        """
        cfg = self._cfg
        if fill_rate <= 0.0 or fill_rate >= 1.0:
            return 1.0
        half_spread = cfg.base_spread_bps / 20_000.0
        if half_spread < 1e-9:
            return 1.0
        try:
            kappa = -math.log(1.0 - fill_rate) / half_spread
            kappa = max(0.01, min(100.0, kappa))
        except (ValueError, ZeroDivisionError):
            kappa = 1.0
        return kappa

    def _empirical_adjustment(
        self, state: _SymbolState, fill_rate: float
    ) -> float:
        """Compute empirically adjusted spread based on fill rate deviation.

        If we're filling *more* than target: spread too tight → widen.
        If we're filling *less* than target: spread too wide → tighten.
        """
        cfg = self._cfg
        target = cfg.fill_rate_target

        # Grab recent spread values from fills
        fills = list(state.fills)
        if not fills:
            return cfg.base_spread_bps

        recent_spreads = [f.spread_bps_at_fill for f in fills[-cfg.update_interval_fills :]]
        avg_quoted_spread = float(np.mean(recent_spreads)) if recent_spreads else cfg.base_spread_bps

        if target <= 0:
            return avg_quoted_spread

        deviation = fill_rate - target  # positive = overfilling (too tight)

        # Proportional adjustment: max 50% swing
        adjustment_factor = 1.0 + deviation  # e.g. fill_rate 0.6 vs target 0.4 → ×1.2
        adjustment_factor = max(0.5, min(2.0, adjustment_factor))

        empirical_bps = avg_quoted_spread * adjustment_factor
        return empirical_bps

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    def _estimate_vol(self, state: _SymbolState) -> float:
        """Estimate annualised volatility from the price buffer.

        Returns vol as a fraction (e.g. 0.80 = 80% annualised).
        Returns 0.0 if insufficient data.
        """
        prices = list(state.prices)
        n = len(prices)
        if n < 4:
            return 0.0
        lookback = min(n - 1, self._cfg.vol_lookback)
        arr = np.asarray(prices[-(lookback + 1) :], dtype=float)
        arr = np.maximum(arr, 1e-12)
        log_rets = np.diff(np.log(arr))
        if len(log_rets) < 2:
            return 0.0
        sigma = float(np.std(log_rets, ddof=1)) * self._cfg.annualise_factor
        return sigma

    def _compute_fill_rate(self, state: _SymbolState) -> float:
        """Compute fill rate as fills / quotes over the calibration window."""
        window = self._cfg.calibration_window
        recent_fills = len(list(state.fills)[-window:])
        recent_quotes = min(state.quotes_sent, window)
        if recent_quotes == 0:
            return 0.0
        return min(1.0, recent_fills / recent_quotes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            state = _SymbolState()
            state.optimal_bps = self._cfg.base_spread_bps
            state.as_bps = self._cfg.base_spread_bps
            state.empirical_bps = self._cfg.base_spread_bps
            self._states[symbol] = state
        return self._states[symbol]

    def reset(self, symbol: str) -> None:
        """Clear all state for *symbol*."""
        if symbol in self._states:
            del self._states[symbol]
            logger.info("OptimalSpreadCalibrator: reset state for %s", symbol)

    def symbols(self) -> List[str]:
        """Return list of all tracked symbols."""
        return list(self._states.keys())

    def __repr__(self) -> str:  # pragma: no cover
        n = len(self._states)
        return f"<OptimalSpreadCalibrator symbols={n} cfg={self._cfg}>"
