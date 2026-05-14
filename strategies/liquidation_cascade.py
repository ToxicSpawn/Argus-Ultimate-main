"""
Liquidation Cascade Strategy — trades momentum driven by forced liquidations.

When large amounts of open interest are liquidated (forced selling/buying),
prices move sharply. This strategy detects large OI drops → momentum trades
in the direction of liquidations.

Signal generation:
  1. Monitor open_interest change rate (>5% drop in 1h = liquidation cascade)
  2. Cross with funding_rate direction (negative = longs being squeezed)
  3. Enter momentum trade in liquidation direction
  4. Exit when OI stabilises (< 1% change per hour)

Returns BUY/SELL signals with confidence.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Maximum OI history entries kept per symbol (24 readings = 24h at 1h cadence)
_MAX_OI_HISTORY = 24

# Thresholds used in confidence calculation
_CONFIDENCE_OI_SCALE = 0.15      # 15% OI drop → full OI contribution
_CONFIDENCE_FUNDING_SCALE = 0.03  # 3% funding rate → full funding contribution
_CONFIDENCE_OI_WEIGHT = 0.6
_CONFIDENCE_FUNDING_WEIGHT = 0.4

# OI stabilisation threshold (exit signal when per-period change < this)
_STABILISATION_THRESHOLD = 0.01  # 1% change per reading


@dataclass
class _OIReading:
    """A single open-interest observation for one symbol."""
    open_interest_usd: float
    funding_rate: float
    price: float
    timestamp: datetime

    def __post_init__(self) -> None:
        if self.open_interest_usd < 0:
            raise ValueError("open_interest_usd must be non-negative")


@dataclass
class LiquidationSignal:
    """Signal emitted when a liquidation cascade is detected."""
    symbol: str
    direction: str                    # "BUY" or "SELL"
    confidence: float                 # 0.0 – 1.0
    oi_drop_pct: float                # fractional drop, e.g. 0.08 = 8%
    funding_rate: float               # current funding rate
    estimated_cascade_size_usd: float # rough USD volume of forced liquidations
    timestamp: datetime

    def __post_init__(self) -> None:
        if self.direction not in ("BUY", "SELL"):
            raise ValueError(f"direction must be 'BUY' or 'SELL', got {self.direction!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


class LiquidationCascadeStrategy:
    """
    Detects forced liquidation events from open-interest data and emits
    momentum signals in the direction of the cascade.

    Parameters
    ----------
    oi_drop_threshold : float
        Fractional OI drop (over ``lookback_hours``) required to declare a
        cascade.  Default 0.05 (5 %).
    funding_threshold : float
        Negative funding rate threshold below which longs are considered
        squeezed.  Default -0.01 (-1 %).
    min_confidence : float
        Minimum confidence score required to emit a signal.  Default 0.60.
    lookback_hours : int
        Number of historical OI readings to inspect when checking for a
        cascade.  Default 4.
    """

    def __init__(
        self,
        oi_drop_threshold: float = 0.05,
        funding_threshold: float = -0.01,
        min_confidence: float = 0.60,
        lookback_hours: int = 4,
    ) -> None:
        if oi_drop_threshold <= 0:
            raise ValueError("oi_drop_threshold must be positive")
        if not (0.0 < min_confidence <= 1.0):
            raise ValueError("min_confidence must be in (0, 1]")
        if lookback_hours < 1:
            raise ValueError("lookback_hours must be >= 1")

        self.oi_drop_threshold = oi_drop_threshold
        self.funding_threshold = funding_threshold
        self.min_confidence = min_confidence
        self.lookback_hours = lookback_hours

        # Rolling OI history: symbol → deque of _OIReading
        self._history: Dict[str, Deque[_OIReading]] = {}

        # Last emitted signal per symbol (for deduplication / logging)
        self._last_signal: Dict[str, Optional[LiquidationSignal]] = {}

        logger.info(
            "LiquidationCascadeStrategy initialised: oi_drop_threshold=%.3f "
            "funding_threshold=%.4f min_confidence=%.2f lookback_hours=%d",
            oi_drop_threshold,
            funding_threshold,
            min_confidence,
            lookback_hours,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        symbol: str,
        open_interest_usd: float,
        funding_rate: float,
        price: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Feed a new OI / funding-rate observation for *symbol*.

        Parameters
        ----------
        symbol : str
            Asset symbol, e.g. ``"BTC"``.
        open_interest_usd : float
            Total open interest denominated in USD.
        funding_rate : float
            Current perpetual funding rate (signed, per 8-hour period typically).
        price : float
            Mid-market price at the time of the reading.
        timestamp : datetime, optional
            Observation time.  Defaults to ``datetime.now(timezone.utc)``.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        reading = _OIReading(
            open_interest_usd=open_interest_usd,
            funding_rate=funding_rate,
            price=price,
            timestamp=timestamp,
        )

        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=_MAX_OI_HISTORY)
            self._last_signal[symbol] = None

        self._history[symbol].append(reading)

        logger.debug(
            "update: symbol=%s oi_usd=%.2f funding=%.6f price=%.2f ts=%s",
            symbol,
            open_interest_usd,
            funding_rate,
            price,
            timestamp.isoformat(),
        )

    def generate_signal(self, symbol: str) -> Optional[LiquidationSignal]:
        """
        Evaluate whether a liquidation cascade is in progress for *symbol*.

        Returns a :class:`LiquidationSignal` when the cascade and confidence
        thresholds are both exceeded; ``None`` otherwise.
        """
        if symbol not in self._history or len(self._history[symbol]) < 2:
            logger.debug("generate_signal: insufficient history for %s", symbol)
            return None

        cascade_detected, oi_drop_pct = self._detect_cascade(symbol)
        if not cascade_detected:
            return None

        history = self._history[symbol]
        latest = history[-1]
        funding_rate = latest.funding_rate

        confidence = self._compute_confidence(oi_drop_pct, funding_rate)
        if confidence < self.min_confidence:
            logger.debug(
                "generate_signal: confidence %.3f below threshold %.3f for %s",
                confidence,
                self.min_confidence,
                symbol,
            )
            return None

        # Direction: negative funding → longs squeezed → SELL cascade
        #            positive funding → shorts squeezed → BUY cascade
        direction = "SELL" if funding_rate <= 0 else "BUY"

        # Rough cascade size: OI drop × latest OI figure
        estimated_cascade_size_usd = oi_drop_pct * latest.open_interest_usd

        signal = LiquidationSignal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            oi_drop_pct=oi_drop_pct,
            funding_rate=funding_rate,
            estimated_cascade_size_usd=estimated_cascade_size_usd,
            timestamp=latest.timestamp,
        )

        self._last_signal[symbol] = signal

        logger.info(
            "LiquidationSignal emitted: symbol=%s direction=%s confidence=%.3f "
            "oi_drop_pct=%.2f%% funding=%.6f cascade_size_usd=%.0f",
            symbol,
            direction,
            confidence,
            oi_drop_pct * 100,
            funding_rate,
            estimated_cascade_size_usd,
        )

        return signal

    def is_stabilising(self, symbol: str) -> bool:
        """
        Return ``True`` when the OI change rate has dropped below the
        stabilisation threshold — indicating a potential exit point.
        """
        if symbol not in self._history or len(self._history[symbol]) < 2:
            return False

        history = self._history[symbol]
        recent = list(history)[-2:]  # last two readings
        prev_oi = recent[0].open_interest_usd
        curr_oi = recent[1].open_interest_usd

        if prev_oi == 0:
            return True

        change = abs(curr_oi - prev_oi) / prev_oi
        return change < _STABILISATION_THRESHOLD

    def all_signals(self) -> List[LiquidationSignal]:
        """Return active signals for all tracked symbols."""
        signals: List[LiquidationSignal] = []
        for symbol in list(self._history.keys()):
            sig = self.generate_signal(symbol)
            if sig is not None:
                signals.append(sig)
        return signals

    def symbols(self) -> List[str]:
        """Return list of symbols currently being tracked."""
        return list(self._history.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_cascade(self, symbol: str) -> Tuple[bool, float]:
        """
        Inspect the rolling OI history for *symbol* over ``lookback_hours``
        most recent readings and determine whether a cascade is underway.

        Returns
        -------
        (cascade_detected, oi_drop_pct) : Tuple[bool, float]
            ``cascade_detected`` is ``True`` when the OI has dropped by at
            least ``oi_drop_threshold`` from its peak within the window.
            ``oi_drop_pct`` is the measured fractional drop (0 if none).
        """
        history = self._history[symbol]
        window: List[_OIReading] = list(history)[-self.lookback_hours:]

        if len(window) < 2:
            return False, 0.0

        peak_oi = max(r.open_interest_usd for r in window)
        current_oi = window[-1].open_interest_usd

        if peak_oi == 0:
            return False, 0.0

        oi_drop_pct = (peak_oi - current_oi) / peak_oi

        if oi_drop_pct >= self.oi_drop_threshold:
            logger.debug(
                "_detect_cascade: symbol=%s peak_oi=%.2f current_oi=%.2f drop=%.2f%%",
                symbol,
                peak_oi,
                current_oi,
                oi_drop_pct * 100,
            )
            return True, oi_drop_pct

        return False, oi_drop_pct

    def generate_orders(
        self, signal: LiquidationSignal, portfolio_value: float
    ) -> List[Dict]:
        """Convert a cascade detection signal to contrarian entry orders.

        During a liquidation cascade, forced sellers drive prices below fair
        value.  This method generates contrarian orders that fade the cascade
        (buy when longs are liquidated / sell when shorts are liquidated).

        Parameters
        ----------
        signal : LiquidationSignal
            Signal from ``generate_signal()``.
        portfolio_value : float
            Current total portfolio value in USD.

        Returns
        -------
        List of order dicts with keys: symbol, side, quantity, order_type, reason.
        """
        # Contrarian: opposite of cascade direction
        contrarian_side = "BUY" if signal.direction == "SELL" else "SELL"

        # Size proportional to confidence, capped at 15% of portfolio
        size_fraction = min(0.15, signal.confidence * 0.20)
        notional = portfolio_value * size_fraction

        reason = (
            f"contrarian_{contrarian_side.lower()}_cascade_"
            f"oi_drop_{signal.oi_drop_pct:.2%}_conf_{signal.confidence:.2f}"
        )

        return [{
            "symbol": signal.symbol,
            "side": contrarian_side,
            "quantity": notional,
            "order_type": "limit",
            "reason": reason,
        }]

    def _compute_confidence(self, oi_drop: float, funding_rate: float) -> float:
        """
        Calculate a confidence score in [0, 1] for a detected cascade.

        Confidence has two additive components:
        - OI component:      scaled by ``_CONFIDENCE_OI_SCALE``, weighted 60 %
        - Funding component: scaled by ``_CONFIDENCE_FUNDING_SCALE``, weighted 40 %

        Parameters
        ----------
        oi_drop : float
            Fractional OI drop, e.g. 0.08 for 8 %.
        funding_rate : float
            Signed funding rate.

        Returns
        -------
        float
            Confidence in [0, 1].
        """
        oi_component = min(1.0, oi_drop / _CONFIDENCE_OI_SCALE)
        funding_component = min(1.0, abs(funding_rate) / _CONFIDENCE_FUNDING_SCALE)

        confidence = (
            oi_component * _CONFIDENCE_OI_WEIGHT
            + funding_component * _CONFIDENCE_FUNDING_WEIGHT
        )

        return min(1.0, confidence)
