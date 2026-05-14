"""
Multi-Horizon Signal Fusion — aggregates directional signals from multiple
timeframes (1m through 1d) into a single fused signal per symbol.

Uses attention-style weighting: each regime has learned weights per timeframe,
so the system can emphasise hourly signals in trending regimes and minute-level
signals in mean-reverting ones.  Weights are initialised uniformly and updated
via ``update_regime_weights()`` or automatic accuracy tracking.

Thread-safe: all mutable state protected by a threading lock.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_TIMEFRAMES = ("1m", "5m", "15m", "1h", "4h", "1d")
TIMEFRAME_RANK = {tf: i for i, tf in enumerate(VALID_TIMEFRAMES)}  # higher = longer
DEFAULT_REGIME = "normal"

# Default uniform weights per timeframe
_UNIFORM_WEIGHTS: Dict[str, float] = {tf: 1.0 / len(VALID_TIMEFRAMES) for tf in VALID_TIMEFRAMES}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SignalEntry:
    """A single directional signal from one timeframe."""
    timeframe: str
    symbol: str
    direction: str          # "long" or "short"
    confidence: float       # 0.0 – 1.0
    features: Dict[str, float]
    timestamp: float = field(default_factory=time.time)


@dataclass
class FusedSignal:
    """Result of fusing signals across timeframes for one symbol."""
    symbol: str
    direction: str               # "long", "short", or "neutral"
    confidence: float            # 0.0 – 1.0
    contributing_timeframes: List[str]
    agreement_pct: float         # fraction of timeframes that agree with fused direction
    dominant_timeframe: str      # timeframe with highest weighted contribution


@dataclass
class _AccuracyRecord:
    """Tracks whether a signal's direction was correct."""
    timeframe: str
    direction: str
    correct: Optional[bool] = None
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MultiHorizonFusion:
    """
    Fuses directional signals across 6 canonical timeframes into a single
    signal per symbol, with regime-dependent attention weights.

    Parameters
    ----------
    signal_ttl_s : float
        Seconds after which a signal is considered stale and excluded.
    max_accuracy_records : int
        Rolling window size for per-timeframe accuracy tracking.
    """

    def __init__(self, signal_ttl_s: float = 300.0,
                 max_accuracy_records: int = 1000) -> None:
        self._signal_ttl_s = signal_ttl_s
        self._max_accuracy_records = max_accuracy_records
        self._lock = threading.Lock()

        # symbol → timeframe → latest SignalEntry
        self._signals: Dict[str, Dict[str, SignalEntry]] = defaultdict(dict)

        # regime → timeframe → weight  (sum to 1.0 within each regime)
        self._regime_weights: Dict[str, Dict[str, float]] = defaultdict(
            lambda: dict(_UNIFORM_WEIGHTS)
        )
        self._current_regime: str = DEFAULT_REGIME

        # Per-timeframe accuracy tracking
        self._accuracy: Dict[str, Deque[_AccuracyRecord]] = {
            tf: deque(maxlen=max_accuracy_records) for tf in VALID_TIMEFRAMES
        }

        logger.info("MultiHorizonFusion initialised (ttl=%.0fs, regimes=%s)",
                     signal_ttl_s, DEFAULT_REGIME)

    # ------------------------------------------------------------------
    # Signal ingestion
    # ------------------------------------------------------------------

    def add_signal(self, timeframe: str, symbol: str, direction: str,
                   confidence: float, features: Optional[Dict[str, float]] = None) -> None:
        """
        Add or update a signal from a specific timeframe.

        Parameters
        ----------
        timeframe : str
            One of ``VALID_TIMEFRAMES``.
        symbol : str
            Trading pair, e.g. "BTC/AUD".
        direction : str
            "long" or "short".
        confidence : float
            Signal confidence in [0.0, 1.0].
        features : dict, optional
            Arbitrary feature dict attached to the signal.
        """
        if timeframe not in TIMEFRAME_RANK:
            logger.warning("add_signal: unknown timeframe '%s' — ignoring", timeframe)
            return

        direction = direction.lower()
        if direction not in ("long", "short"):
            logger.warning("add_signal: direction must be 'long' or 'short', got '%s'", direction)
            return

        confidence = max(0.0, min(1.0, float(confidence)))
        entry = SignalEntry(
            timeframe=timeframe,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            features=features or {},
        )

        with self._lock:
            self._signals[symbol][timeframe] = entry

        logger.debug("add_signal: %s %s %s conf=%.2f", symbol, timeframe, direction, confidence)

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def fuse(self, symbol: str) -> FusedSignal:
        """
        Fuse all active signals for *symbol* into a single directional view.

        Stale signals (older than ``signal_ttl_s``) are excluded.  The fused
        direction is the confidence-and-weight-weighted majority vote.

        Parameters
        ----------
        symbol : str

        Returns
        -------
        FusedSignal
        """
        now = time.time()

        with self._lock:
            raw = dict(self._signals.get(symbol, {}))
            weights = dict(self._regime_weights[self._current_regime])

        # Filter stale
        active: Dict[str, SignalEntry] = {}
        for tf, sig in raw.items():
            if now - sig.timestamp <= self._signal_ttl_s:
                active[tf] = sig

        if not active:
            return FusedSignal(
                symbol=symbol, direction="neutral", confidence=0.0,
                contributing_timeframes=[], agreement_pct=0.0,
                dominant_timeframe="",
            )

        # Weighted vote
        long_score = 0.0
        short_score = 0.0
        contrib_timeframes: List[str] = []
        best_tf = ""
        best_contribution = -1.0

        for tf, sig in active.items():
            w = weights.get(tf, 0.0)
            contribution = w * sig.confidence
            if sig.direction == "long":
                long_score += contribution
            else:
                short_score += contribution
            contrib_timeframes.append(tf)
            if contribution > best_contribution:
                best_contribution = contribution
                best_tf = tf

        total = long_score + short_score
        if total < 1e-12:
            return FusedSignal(
                symbol=symbol, direction="neutral", confidence=0.0,
                contributing_timeframes=contrib_timeframes,
                agreement_pct=0.0, dominant_timeframe=best_tf,
            )

        if long_score >= short_score:
            direction = "long"
            confidence = long_score / total
            agree_count = sum(1 for s in active.values() if s.direction == "long")
        else:
            direction = "short"
            confidence = short_score / total
            agree_count = sum(1 for s in active.values() if s.direction == "short")

        agreement_pct = agree_count / len(active)

        # Sort contributing timeframes by rank
        contrib_timeframes.sort(key=lambda t: TIMEFRAME_RANK.get(t, 99))

        fused = FusedSignal(
            symbol=symbol,
            direction=direction,
            confidence=round(confidence, 4),
            contributing_timeframes=contrib_timeframes,
            agreement_pct=round(agreement_pct, 4),
            dominant_timeframe=best_tf,
        )
        logger.debug("fuse(%s): %s conf=%.2f agreement=%.0f%% dominant=%s",
                     symbol, direction, confidence, agreement_pct * 100, best_tf)
        return fused

    # ------------------------------------------------------------------
    # Regime weights
    # ------------------------------------------------------------------

    def update_regime_weights(self, regime: str,
                              timeframe_weights: Dict[str, float]) -> None:
        """
        Set or update the attention weights for a given regime.

        Weights are normalised to sum to 1.0.

        Parameters
        ----------
        regime : str
            Regime label, e.g. "trending", "mean_reverting", "crisis".
        timeframe_weights : dict
            Mapping of timeframe → raw weight.
        """
        total = sum(timeframe_weights.values())
        if total <= 0:
            logger.warning("update_regime_weights: all weights zero — ignoring")
            return

        normalised = {tf: w / total for tf, w in timeframe_weights.items()
                      if tf in TIMEFRAME_RANK}

        with self._lock:
            self._regime_weights[regime] = normalised

        logger.info("update_regime_weights(%s): %s", regime,
                    {k: round(v, 3) for k, v in normalised.items()})

    def set_regime(self, regime: str) -> None:
        """Switch the active regime (affects fusion weights)."""
        with self._lock:
            self._current_regime = regime
        logger.info("set_regime → %s", regime)

    # ------------------------------------------------------------------
    # Accuracy tracking
    # ------------------------------------------------------------------

    def record_outcome(self, timeframe: str, direction: str, correct: bool) -> None:
        """
        Record whether a signal from *timeframe* was correct.

        Called by the system after a trade resolves so that
        ``get_timeframe_accuracy`` can report per-timeframe hit rate.
        """
        if timeframe not in TIMEFRAME_RANK:
            return
        with self._lock:
            self._accuracy[timeframe].append(
                _AccuracyRecord(timeframe=timeframe, direction=direction, correct=correct)
            )

    def get_timeframe_accuracy(self, timeframe: str, lookback_days: int = 7) -> float:
        """
        Rolling accuracy for a given timeframe.

        Parameters
        ----------
        timeframe : str
        lookback_days : int
            Only consider records from the last N days.

        Returns
        -------
        float
            Accuracy in [0.0, 1.0].  Returns 0.0 if no data.
        """
        if timeframe not in self._accuracy:
            return 0.0

        cutoff = time.time() - lookback_days * 86400
        with self._lock:
            records = [r for r in self._accuracy[timeframe]
                       if r.correct is not None and r.timestamp >= cutoff]

        if not records:
            return 0.0
        return sum(1 for r in records if r.correct) / len(records)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def active_signals_count(self) -> int:
        """Number of symbols with at least one signal."""
        with self._lock:
            return len(self._signals)

    def get_regime_weights(self, regime: Optional[str] = None) -> Dict[str, float]:
        """Return the weight dict for *regime* (default: current regime)."""
        with self._lock:
            r = regime or self._current_regime
            return dict(self._regime_weights[r])
