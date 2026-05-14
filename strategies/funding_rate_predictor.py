"""
Funding Rate Prediction Strategy — predict next 8h funding rate and position
BEFORE settlement to capture extreme funding payments.

Uses:
  - Current funding rate trend (momentum)
  - Open interest change rate
  - Spot-perp basis (premium/discount)
  - Order book imbalance
  - Volume ratio (perp vs spot)

When predicted rate is extreme (>0.05% per 8h = ~228% APR):
  - Position BEFORE settlement
  - Capture the full funding payment
  - Exit after settlement if rate normalizing
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Funding periods: every 8 hours
_FUNDING_INTERVAL_S = 28800  # 8h in seconds

# Thresholds
_EXTREME_RATE_PCT = 0.05       # 0.05% per 8h = ~228% APR
_HIGH_RATE_PCT = 0.03          # 0.03% per 8h = ~137% APR
_ENTRY_WINDOW_HOURS = (1.0, 3.0)  # Sweet spot: 1-3h before settlement

# Feature weights for prediction
_W_RATE_MOMENTUM = 0.30
_W_OI_CHANGE = 0.15
_W_BASIS = 0.30
_W_OBI = 0.15
_W_VOLUME_RATIO = 0.10


@dataclass
class FundingPrediction:
    """Result of a funding rate prediction."""
    symbol: str
    predicted_rate_pct: float
    confidence: float
    direction: str  # "LONG_PAY" | "SHORT_PAY" | "NEUTRAL"
    magnitude: str  # "EXTREME" | "HIGH" | "NORMAL" | "LOW"
    position_recommendation: str  # "SHORT_PERP" | "LONG_PERP" | "NONE"
    expected_pnl_bps: float
    hours_to_settlement: float
    timestamp: float = field(default_factory=time.time)


class FundingRatePredictor:
    """
    Predict next 8h funding rate from multiple on-chain/market signals
    and generate actionable trading signals when rates are extreme.
    """

    def __init__(self, lookback_periods: int = 30) -> None:
        self._lookback = lookback_periods

        # Per-symbol historical data
        self._rate_history: Dict[str, Deque[Tuple[float, float]]] = {}  # symbol -> deque of (ts, rate)
        self._oi_history: Dict[str, Deque[Tuple[float, float]]] = {}   # symbol -> deque of (ts, oi)
        self._basis_history: Dict[str, Deque[Tuple[float, float]]] = {}  # symbol -> deque of (ts, basis_pct)
        self._volume_ratio_history: Dict[str, Deque[Tuple[float, float]]] = {}  # symbol -> deque of (ts, ratio)
        self._obi_history: Dict[str, Deque[Tuple[float, float]]] = {}  # symbol -> deque of (ts, obi)

        # Last known settlement times
        self._last_settlement_ts: Dict[str, float] = {}

        # Prediction cache
        self._last_prediction: Dict[str, FundingPrediction] = {}

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def update(
        self,
        symbol: str,
        funding_rate: float,
        open_interest: float,
        spot_price: float,
        perp_price: float,
        volume_spot: float = 0.0,
        volume_perp: float = 0.0,
    ) -> None:
        """Feed latest data for a symbol."""
        sym = symbol.upper()
        now = time.time()

        # Ensure deques exist
        if sym not in self._rate_history:
            self._rate_history[sym] = deque(maxlen=self._lookback)
            self._oi_history[sym] = deque(maxlen=self._lookback)
            self._basis_history[sym] = deque(maxlen=self._lookback)
            self._volume_ratio_history[sym] = deque(maxlen=self._lookback)
            self._obi_history[sym] = deque(maxlen=self._lookback)

        # Funding rate (as percentage)
        self._rate_history[sym].append((now, funding_rate * 100.0))

        # Open interest
        self._oi_history[sym].append((now, open_interest))

        # Basis (premium/discount)
        if spot_price > 0:
            basis_pct = (perp_price - spot_price) / spot_price * 100.0
        else:
            basis_pct = 0.0
        self._basis_history[sym].append((now, basis_pct))

        # Volume ratio
        if volume_spot > 0:
            vol_ratio = volume_perp / volume_spot
        else:
            vol_ratio = 1.0
        self._volume_ratio_history[sym].append((now, vol_ratio))

    def update_orderbook_imbalance(self, symbol: str, obi: float) -> None:
        """Feed order book imbalance for a symbol. OBI in [-1, +1]."""
        sym = symbol.upper()
        now = time.time()
        if sym not in self._obi_history:
            self._obi_history[sym] = deque(maxlen=self._lookback)
        self._obi_history[sym].append((now, max(-1.0, min(1.0, obi))))

    def set_last_settlement(self, symbol: str, ts: float) -> None:
        """Set the last known settlement timestamp."""
        self._last_settlement_ts[symbol.upper()] = ts

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_next_rate(self, symbol: str) -> FundingPrediction:
        """
        Predict the next funding rate for the symbol.

        Returns a FundingPrediction with predicted rate, confidence,
        direction, magnitude, and position recommendation.
        """
        sym = symbol.upper()

        # Compute feature signals
        rate_momentum = self._compute_rate_momentum(sym)
        oi_change = self._compute_oi_change(sym)
        basis_signal = self._compute_basis_signal(sym)
        obi_signal = self._compute_obi_signal(sym)
        volume_signal = self._compute_volume_signal(sym)

        # Weighted combination
        predicted_rate_pct = (
            _W_RATE_MOMENTUM * rate_momentum
            + _W_OI_CHANGE * oi_change * 0.1
            + _W_BASIS * basis_signal
            + _W_OBI * obi_signal * 0.05
            + _W_VOLUME_RATIO * volume_signal * 0.02
        )

        # Clamp to realistic range
        predicted_rate_pct = max(-0.5, min(0.5, predicted_rate_pct))

        # Confidence based on data availability and signal agreement
        n_sources = sum([
            len(self._rate_history.get(sym, [])) >= 3,
            len(self._oi_history.get(sym, [])) >= 2,
            len(self._basis_history.get(sym, [])) >= 3,
            len(self._obi_history.get(sym, [])) >= 2,
            len(self._volume_ratio_history.get(sym, [])) >= 2,
        ])
        confidence = min(1.0, n_sources / 5.0 * 0.7 + abs(predicted_rate_pct) / 0.1 * 0.3)
        confidence = max(0.0, min(1.0, confidence))

        # Direction
        if predicted_rate_pct > 0.005:
            direction = "LONG_PAY"
        elif predicted_rate_pct < -0.005:
            direction = "SHORT_PAY"
        else:
            direction = "NEUTRAL"

        # Magnitude
        abs_rate = abs(predicted_rate_pct)
        if abs_rate >= _EXTREME_RATE_PCT:
            magnitude = "EXTREME"
        elif abs_rate >= _HIGH_RATE_PCT:
            magnitude = "HIGH"
        elif abs_rate >= 0.01:
            magnitude = "NORMAL"
        else:
            magnitude = "LOW"

        # Position recommendation
        if magnitude in ("EXTREME", "HIGH") and confidence >= 0.5:
            if direction == "LONG_PAY":
                rec = "SHORT_PERP"  # Short perp to receive funding
            elif direction == "SHORT_PAY":
                rec = "LONG_PERP"   # Long perp to receive funding
            else:
                rec = "NONE"
        else:
            rec = "NONE"

        # Expected P&L in basis points
        expected_pnl_bps = abs(predicted_rate_pct) * 100.0  # 0.05% = 5 bps

        # Hours to settlement
        hours_to = self._hours_to_settlement(sym)

        pred = FundingPrediction(
            symbol=sym,
            predicted_rate_pct=predicted_rate_pct,
            confidence=confidence,
            direction=direction,
            magnitude=magnitude,
            position_recommendation=rec,
            expected_pnl_bps=expected_pnl_bps,
            hours_to_settlement=hours_to,
        )
        self._last_prediction[sym] = pred
        return pred

    def get_optimal_entry_timing(self, symbol: str) -> Dict[str, Any]:
        """
        When to enter for max funding capture.

        Returns timing guidance:
        - Too early: capital locked, opportunity cost
        - Too late: rate already priced in
        - Sweet spot: 1-3 hours before settlement
        """
        sym = symbol.upper()
        hours_to = self._hours_to_settlement(sym)

        if hours_to <= _ENTRY_WINDOW_HOURS[0]:
            timing = "LATE"
            recommendation = "Price likely already adjusted — reduced edge"
            urgency = 0.3
        elif hours_to <= _ENTRY_WINDOW_HOURS[1]:
            timing = "OPTIMAL"
            recommendation = "Enter now — best risk/reward before settlement"
            urgency = 1.0
        elif hours_to <= 5.0:
            timing = "EARLY"
            recommendation = "Wait — capital locked with opportunity cost"
            urgency = 0.5
        else:
            timing = "TOO_EARLY"
            recommendation = "Wait for closer to settlement window"
            urgency = 0.1

        return {
            "symbol": sym,
            "hours_to_settlement": hours_to,
            "timing": timing,
            "recommendation": recommendation,
            "urgency": urgency,
            "optimal_entry_window_hours": _ENTRY_WINDOW_HOURS,
        }

    def generate_signal(self, symbol: str, market_data: Optional[Dict] = None) -> Optional[Any]:
        """
        Generate an actionable TradingSignal when predicted rate is extreme.

        Returns None if no trade opportunity, or a TradingSignal dict.
        """
        pred = self.predict_next_rate(symbol)
        timing = self.get_optimal_entry_timing(symbol)

        # Only generate signal for extreme/high rates with good timing
        if pred.magnitude not in ("EXTREME", "HIGH"):
            return None
        if timing["timing"] not in ("OPTIMAL", "EARLY"):
            return None
        if pred.confidence < 0.4:
            return None

        # Build signal
        action = "SELL" if pred.direction == "LONG_PAY" else "BUY"
        price = (market_data or {}).get("price", 0.0)

        return {
            "symbol": symbol,
            "action": action,
            "confidence": pred.confidence,
            "strength": min(1.0, abs(pred.predicted_rate_pct) / _EXTREME_RATE_PCT),
            "entry_price": price,
            "reasoning": (
                f"Funding prediction: {pred.predicted_rate_pct:.4f}% ({pred.magnitude}) "
                f"— {pred.direction} — {timing['timing']} entry, "
                f"{pred.hours_to_settlement:.1f}h to settlement"
            ),
            "strategy": "funding_rate_predictor",
            "expected_pnl_bps": pred.expected_pnl_bps,
        }

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _compute_rate_momentum(self, symbol: str) -> float:
        """Linear trend of recent funding rates."""
        history = list(self._rate_history.get(symbol, []))
        if len(history) < 2:
            return 0.0
        rates = [r for _, r in history]
        return self._linear_slope(rates)

    def _compute_oi_change(self, symbol: str) -> float:
        """Rate of change in open interest."""
        history = list(self._oi_history.get(symbol, []))
        if len(history) < 2:
            return 0.0
        oi_vals = [oi for _, oi in history]
        if oi_vals[0] == 0:
            return 0.0
        return (oi_vals[-1] - oi_vals[0]) / max(abs(oi_vals[0]), 1e-10)

    def _compute_basis_signal(self, symbol: str) -> float:
        """Current premium/discount trend."""
        history = list(self._basis_history.get(symbol, []))
        if len(history) < 2:
            return 0.0
        bases = [b for _, b in history]
        return self._linear_slope(bases)

    def _compute_obi_signal(self, symbol: str) -> float:
        """Smoothed order book imbalance."""
        history = list(self._obi_history.get(symbol, []))
        if not history:
            return 0.0
        obis = [o for _, o in history]
        return sum(obis) / len(obis)

    def _compute_volume_signal(self, symbol: str) -> float:
        """Perp/spot volume ratio trend."""
        history = list(self._volume_ratio_history.get(symbol, []))
        if len(history) < 2:
            return 0.0
        ratios = [r for _, r in history]
        # High perp volume relative to spot → more speculative → higher funding
        return self._linear_slope(ratios)

    def _hours_to_settlement(self, symbol: str) -> float:
        """Estimate hours until next funding settlement."""
        last_ts = self._last_settlement_ts.get(symbol, time.time())
        elapsed = time.time() - last_ts
        remaining = _FUNDING_INTERVAL_S - (elapsed % _FUNDING_INTERVAL_S)
        return max(0.0, remaining / 3600.0)

    @staticmethod
    def _linear_slope(values: List[float]) -> float:
        """Simple linear regression slope."""
        n = len(values)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        num = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        if abs(den) < 1e-12:
            return 0.0
        return num / den
