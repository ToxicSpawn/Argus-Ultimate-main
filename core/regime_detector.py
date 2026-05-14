"""Isolated Regime Detector — pure signal, no side effects.

Improvements over the previous autonomous_brain.py embedded logic:
1. Outputs a strict RegimeState enum — every downstream component reads ONE source
   of truth instead of parsing strings from multiple places.
2. Hysteresis filter — regime must persist for MIN_HOLD_BARS consecutive bars
   before the transition is confirmed. Prevents thrashing on noisy transitions.
3. HV/RV ratio feature — realised vol / ATR ratio detects coiled-spring setups
   (low ATR + low HV = breakout imminent, bandit should favour momentum).
4. ADX-based trend strength added alongside ATR percentile.
5. Regime confidence score (0.0-1.0) passed to the contextual bandit so it can
   blend regime-specific and global distributions proportionally.
6. Fully stateless compute() for backtesting; stateful update() for live use.

Compatibility aliases (2026-04):
  LiveRegimeDetector     -> RegimeDetector   (preserved for existing callers)
  RegimeDetectorConfig   -> dataclass shim   (no-op config, thresholds now class-level)
  Regime                 -> RegimeState      (old string-enum alias)
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

# Minimum consecutive bars a candidate regime must hold before confirming
MIN_HOLD_BARS = 6

# ADX thresholds
ADX_TRENDING = 25.0
ADX_STRONG   = 40.0

# ATR percentile thresholds (vs rolling 50-bar window)
ATR_HIGH_VOL_PCT = 80   # top 20% ATR  => high-vol / crash regime
ATR_LOW_PCT      = 30   # bottom 30%   => ranging / coiled

# HV/ATR ratio below this suggests volatility compression (coiled spring)
HV_ATR_COIL_RATIO = 0.80


class RegimeState(str, Enum):
    """Canonical regime labels used by all downstream components."""
    TRENDING_UP    = "TRENDING_UP"
    TRENDING_DOWN  = "TRENDING_DOWN"
    RANGING        = "RANGING"
    HIGH_VOL_CRASH = "HIGH_VOL_CRASH"
    COILED         = "COILED"    # low ATR + low HV => imminent breakout
    UNKNOWN        = "UNKNOWN"


# ---------------------------------------------------------------------------
# Backward-compatibility alias: old code used Regime.HIGH_VOL etc.
# Map old names -> new RegimeState values.
# ---------------------------------------------------------------------------
class Regime(str, Enum):
    """Legacy regime enum — alias of RegimeState for backward compatibility."""
    TRENDING_UP    = "TRENDING_UP"
    TRENDING_DOWN  = "TRENDING_DOWN"
    RANGING        = "RANGING"
    HIGH_VOL       = "HIGH_VOL_CRASH"   # old name -> new
    HIGH_VOL_CRASH = "HIGH_VOL_CRASH"
    COILED         = "COILED"
    UNKNOWN        = "UNKNOWN"
    CRISIS         = "HIGH_VOL_CRASH"   # another old alias


@dataclass(frozen=True)
class RegimeReading:
    """Full regime snapshot for a single bar."""
    state: RegimeState
    confidence: float          # 0.0 - 1.0
    adx: float
    atr: float
    atr_pct_rank: float        # 0-100 percentile vs lookback window
    hv_atr_ratio: float        # realised vol / ATR
    trend_direction: int       # +1 up, -1 down, 0 flat
    hold_bars: int             # bars current candidate has been held
    confirmed: bool            # True once hysteresis satisfied


# ---------------------------------------------------------------------------
# Compatibility shim: old code may instantiate RegimeDetectorConfig(**kwargs)
# ---------------------------------------------------------------------------
@dataclass
class RegimeDetectorConfig:
    """Legacy config dataclass — fields accepted but ignored (now class-level constants).

    Kept so existing callers that do::

        cfg = RegimeDetectorConfig(high_vol_threshold=2.5, ...)
        detector = LiveRegimeDetector(config=cfg)

    continue to work without modification.
    """
    high_vol_threshold:  float = 2.5
    trend_threshold:     float = 0.003
    ranging_vol_max:     float = 1.2
    hysteresis_ticks:    int   = 5
    # Extended fields accepted silently.
    atr_lookback:        int   = 50
    hv_lookback:         int   = 20
    min_hold_bars:       int   = MIN_HOLD_BARS


class RegimeDetector:
    """
    Stateful regime detector for live trading.

    Usage::
        detector = RegimeDetector()
        for bar in stream:
            reading = detector.update(
                close=bar.close, high=bar.high, low=bar.low,
                atr=bar.atr, adx=bar.adx,
            )
            regime = reading.state
            conf   = reading.confidence
    """

    def __init__(
        self,
        atr_lookback: int = 50,
        hv_lookback:  int = 20,
        min_hold_bars: int = MIN_HOLD_BARS,
        config: Optional[RegimeDetectorConfig] = None,
    ) -> None:
        # Accept legacy config kwarg and extract fields from it.
        if config is not None:
            atr_lookback  = getattr(config, "atr_lookback", atr_lookback)
            hv_lookback   = getattr(config, "hv_lookback", hv_lookback)
            min_hold_bars = getattr(config, "min_hold_bars", min_hold_bars)

        self._atr_lookback   = atr_lookback
        self._hv_lookback    = hv_lookback
        self._min_hold       = min_hold_bars

        # Rolling windows
        self._atr_window:    deque = deque(maxlen=atr_lookback)
        self._close_window:  deque = deque(maxlen=hv_lookback + 1)

        # Hysteresis state
        self._candidate:     RegimeState = RegimeState.UNKNOWN
        self._candidate_hold: int = 0
        self._confirmed:     RegimeState = RegimeState.UNKNOWN

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        close: float,
        high: float,
        low: float,
        atr: float,
        adx: float,
        ema_fast: Optional[float] = None,
        ema_slow: Optional[float] = None,
    ) -> RegimeReading:
        """Ingest one bar and return the current regime reading."""
        self._close_window.append(close)
        self._atr_window.append(atr)

        atr_pct_rank = self._atr_percentile_rank(atr)
        hv = self._realised_vol()
        hv_atr_ratio = (hv / atr) if atr > 0 else 1.0

        trend_dir = 0
        if ema_fast is not None and ema_slow is not None:
            trend_dir = 1 if ema_fast > ema_slow else (-1 if ema_fast < ema_slow else 0)

        raw = self._classify(atr_pct_rank, hv_atr_ratio, adx, trend_dir)
        confidence = self._confidence(atr_pct_rank, hv_atr_ratio, adx)

        # Hysteresis
        if raw == self._candidate:
            self._candidate_hold += 1
        else:
            self._candidate       = raw
            self._candidate_hold  = 1

        if self._candidate_hold >= self._min_hold:
            self._confirmed = self._candidate

        reading = RegimeReading(
            state           = self._confirmed,
            confidence      = confidence,
            adx             = adx,
            atr             = atr,
            atr_pct_rank    = atr_pct_rank,
            hv_atr_ratio    = hv_atr_ratio,
            trend_direction = trend_dir,
            hold_bars       = self._candidate_hold,
            confirmed       = self._candidate_hold >= self._min_hold,
        )

        logger.debug(
            "Regime: confirmed=%s candidate=%s hold=%d/%d conf=%.2f adx=%.1f atr_pct=%.0f hv_atr=%.2f",
            self._confirmed.value, self._candidate.value,
            self._candidate_hold, self._min_hold,
            confidence, adx, atr_pct_rank, hv_atr_ratio,
        )
        return reading

    @staticmethod
    def compute(
        atr: float,
        adx: float,
        atr_pct_rank: float,
        hv_atr_ratio: float,
        trend_dir: int = 0,
    ) -> RegimeState:
        """Stateless single-bar classification — use in backtesting."""
        return RegimeDetector._classify_static(atr_pct_rank, hv_atr_ratio, adx, trend_dir)

    @property
    def confirmed_regime(self) -> RegimeState:
        return self._confirmed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify(
        self,
        atr_pct_rank: float,
        hv_atr_ratio: float,
        adx: float,
        trend_dir: int,
    ) -> RegimeState:
        return self._classify_static(atr_pct_rank, hv_atr_ratio, adx, trend_dir)

    @staticmethod
    def _classify_static(
        atr_pct_rank: float,
        hv_atr_ratio: float,
        adx: float,
        trend_dir: int,
    ) -> RegimeState:
        # 1. High-vol / crash first (most dangerous — override everything)
        if atr_pct_rank >= ATR_HIGH_VOL_PCT:
            return RegimeState.HIGH_VOL_CRASH

        # 2. Coiled spring — low vol + low HV => breakout imminent
        if atr_pct_rank <= ATR_LOW_PCT and hv_atr_ratio < HV_ATR_COIL_RATIO:
            return RegimeState.COILED

        # 3. Trending
        if adx >= ADX_TRENDING:
            if trend_dir >= 0:
                return RegimeState.TRENDING_UP
            else:
                return RegimeState.TRENDING_DOWN

        # 4. Default: ranging
        return RegimeState.RANGING

    def _atr_percentile_rank(self, current_atr: float) -> float:
        """0-100 percentile rank of current ATR vs rolling window."""
        window = list(self._atr_window)
        if len(window) < 2:
            return 50.0
        below = sum(1 for v in window if v <= current_atr)
        return 100.0 * below / len(window)

    def _realised_vol(self) -> float:
        """20-bar close-to-close realised volatility."""
        closes = list(self._close_window)
        if len(closes) < 2:
            return 0.0
        log_returns = [
            math.log(closes[i] / closes[i - 1])
            for i in range(1, len(closes))
            if closes[i - 1] > 0
        ]
        if not log_returns:
            return 0.0
        mean = sum(log_returns) / len(log_returns)
        variance = sum((r - mean) ** 2 for r in log_returns) / max(1, len(log_returns) - 1)
        return math.sqrt(variance)

    def _confidence(
        self,
        atr_pct_rank: float,
        hv_atr_ratio: float,
        adx: float,
    ) -> float:
        adx_conf = min(1.0, max(0.0, (adx - 10.0) / 40.0))
        atr_extremity = abs(atr_pct_rank - 50.0) / 50.0
        hv_conf = min(1.0, abs(hv_atr_ratio - 1.0) / 1.0)
        return round(0.5 * adx_conf + 0.3 * atr_extremity + 0.2 * hv_conf, 4)

    def reset(self) -> None:
        """Clear all state — use between backtesting runs."""
        self._atr_window.clear()
        self._close_window.clear()
        self._candidate      = RegimeState.UNKNOWN
        self._candidate_hold = 0
        self._confirmed      = RegimeState.UNKNOWN

    def get_stats(self) -> dict:
        return {
            "confirmed_regime":  self._confirmed.value,
            "candidate_regime":  self._candidate.value,
            "candidate_hold":    self._candidate_hold,
            "min_hold_bars":     self._min_hold,
            "atr_window_size":   len(self._atr_window),
        }


# ---------------------------------------------------------------------------
# Compatibility alias: LiveRegimeDetector -> RegimeDetector
# ---------------------------------------------------------------------------
LiveRegimeDetector = RegimeDetector


__all__ = [
    "RegimeState",
    "RegimeReading",
    "RegimeDetector",
    "RegimeDetectorConfig",
    "Regime",
    "LiveRegimeDetector",
    "MIN_HOLD_BARS",
    "ADX_TRENDING",
    "ADX_STRONG",
]
