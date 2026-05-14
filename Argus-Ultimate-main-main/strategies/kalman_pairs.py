"""
Kalman Filter Pairs Trading — dynamic hedge ratio using Kalman filter.

Unlike static cointegration with fixed hedge ratio, Kalman filter
continuously updates the hedge ratio as the relationship evolves.

State: [spread_mean, hedge_ratio]
Observation: price_A - hedge_ratio * price_B

Entry: z-score of spread > adaptive threshold (rolling 90th percentile)
Exit:  |z-score| < EXIT_ZSCORE (0.5) or > STOP_ZSCORE (3.5)

Enhanced with:
  - Adaptive z-score thresholds (rolling percentile instead of fixed 2.0)
  - Spread half-life estimation (Ornstein-Uhlenbeck fit, only trade 1-30 bars)
  - Dynamic position sizing based on z-score magnitude
  - Correlation regime filter (skip when correlation < 0.6)
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENTRY_ZSCORE: float = 2.0    # fallback entry when |z-score| exceeds this
EXIT_ZSCORE: float = 0.5     # exit when |z-score| falls below this
STOP_ZSCORE: float = 3.5     # stop-loss when |z-score| exceeds this
DELTA: float = 1e-5          # state covariance increment (process noise proxy)
VE: float = 0.001            # observation noise variance
MIN_HISTORY: int = 60        # minimum observations before signals are emitted

# Adaptive threshold defaults
ADAPTIVE_PERCENTILE: float = 0.90   # enter at 90th percentile of |z-score|
MIN_ADAPTIVE_ZSCORE: float = 1.5    # never enter below this even if 90th pctl is low
MAX_ADAPTIVE_ZSCORE: float = 3.0    # cap to avoid impossibly high threshold

# Half-life filter
HALF_LIFE_MIN: int = 1       # bars — faster is noise
HALF_LIFE_MAX: int = 30      # bars — slower won't converge in time

# Correlation filter
MIN_CORRELATION: float = 0.6  # skip trades when correlation drops below this
CORRELATION_WINDOW: int = 60   # bars for rolling correlation

# Dynamic sizing
BASE_SIZE: float = 1.0        # base position size multiplier at entry threshold
MAX_SIZE_MULTIPLIER: float = 2.0  # max multiplier at extreme z-scores


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PairState:
    """Full state snapshot of the Kalman pairs trader."""

    asset_a: str
    asset_b: str
    hedge_ratio: float           # current Kalman-estimated hedge ratio (θ)
    spread_mean: float           # rolling mean of the residual spread
    spread_std: float            # rolling std of the residual spread
    current_spread: float        # latest residual: price_A - θ * price_B
    z_score: float               # (current_spread - spread_mean) / spread_std
    position: str                # LONG_SPREAD | SHORT_SPREAD | FLAT
    opened_ts: Optional[float]   # unix timestamp when current position was opened
    half_life: Optional[float] = None       # current spread half-life in bars
    adaptive_threshold: Optional[float] = None  # current adaptive z-score entry
    correlation: Optional[float] = None     # current rolling correlation
    position_size_mult: float = 1.0         # sizing multiplier


@dataclass
class PairsSignal:
    """Trading signal emitted by KalmanPairsTrader."""

    asset_a: str
    asset_b: str
    action: str           # LONG_SPREAD | SHORT_SPREAD | EXIT | HOLD
    z_score: float
    hedge_ratio: float
    spread: float
    reason: str
    timestamp: float = field(default_factory=time.time)
    position_size_mult: float = 1.0   # how much to scale the base size


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class KalmanPairsTrader:
    """
    Dynamic hedge-ratio pairs trading using a 1-D Kalman filter.

    The Kalman filter treats the hedge ratio θ as the latent state and
    updates it on every new (price_A, price_B) observation.  The spread
    residual r = price_A - θ * price_B is then z-scored over a rolling
    window of length MIN_HISTORY to generate entry/exit signals.

    Enhanced features:
    - Adaptive z-score thresholds using rolling percentile of |z-score|
    - Spread half-life estimation via Ornstein-Uhlenbeck regression
    - Dynamic position sizing (linear scale with z-score magnitude)
    - Correlation regime filter (skip when pair breaks down)

    Thread-safe: a reentrant lock guards all mutable state.

    Usage
    -----
    trader = KalmanPairsTrader("BTC/USD", "ETH/USD")
    for price_a, price_b in stream:
        signal = trader.update(price_a, price_b, timestamp=time.time())
        if signal.action in ("LONG_SPREAD", "SHORT_SPREAD"):
            execute(signal)
    """

    def __init__(
        self,
        asset_a: str,
        asset_b: str,
        delta: float = DELTA,
        ve: float = VE,
        adaptive_percentile: float = ADAPTIVE_PERCENTILE,
        min_correlation: float = MIN_CORRELATION,
        half_life_min: int = HALF_LIFE_MIN,
        half_life_max: int = HALF_LIFE_MAX,
    ) -> None:
        self._asset_a = asset_a
        self._asset_b = asset_b
        self._delta = delta
        self._ve = ve
        self._lock = threading.RLock()

        # Configuration
        self._adaptive_percentile = adaptive_percentile
        self._min_correlation = min_correlation
        self._half_life_min = half_life_min
        self._half_life_max = half_life_max

        # Kalman filter internal state (1-D: tracking θ = hedge ratio)
        self._theta: float = 1.0     # hedge ratio estimate
        self._C: float = 0.0         # state error covariance (scalar in 1-D case)

        # Rolling spread history for z-score
        self._spreads: Deque[float] = deque(maxlen=MIN_HISTORY)

        # Price histories for correlation calculation
        self._prices_a: Deque[float] = deque(maxlen=CORRELATION_WINDOW)
        self._prices_b: Deque[float] = deque(maxlen=CORRELATION_WINDOW)

        # Rolling z-score history for adaptive threshold
        self._z_scores: Deque[float] = deque(maxlen=MIN_HISTORY)

        # Cached computed values
        self._last_half_life: Optional[float] = None
        self._last_correlation: Optional[float] = None
        self._last_adaptive_threshold: float = ENTRY_ZSCORE

        # Position tracking
        self._position: str = "FLAT"
        self._opened_ts: Optional[float] = None
        self._n_updates: int = 0
        self._n_trades: int = 0

        logger.info(
            "KalmanPairsTrader initialised: %s/%s delta=%g ve=%g "
            "adaptive_pctl=%.2f min_corr=%.2f hl_range=[%d,%d]",
            asset_a, asset_b, delta, ve,
            adaptive_percentile, min_correlation,
            half_life_min, half_life_max,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        price_a: float,
        price_b: float,
        timestamp: Optional[float] = None,
    ) -> PairsSignal:
        """
        Ingest a new price pair and return a signal.

        Parameters
        ----------
        price_a:
            Current price of asset A (USD).
        price_b:
            Current price of asset B (USD).
        timestamp:
            Unix timestamp; defaults to ``time.time()``.

        Returns
        -------
        PairsSignal
            action is HOLD until MIN_HISTORY observations have accumulated.
        """
        if timestamp is None:
            timestamp = time.time()

        if price_a <= 0 or price_b <= 0:
            logger.warning(
                "update: non-positive price pair (%.4f, %.4f) — skipped",
                price_a, price_b,
            )
            return PairsSignal(
                asset_a=self._asset_a,
                asset_b=self._asset_b,
                action="HOLD",
                z_score=0.0,
                hedge_ratio=self._theta,
                spread=0.0,
                reason="invalid_prices",
                timestamp=timestamp,
            )

        with self._lock:
            # Store prices for correlation
            self._prices_a.append(price_a)
            self._prices_b.append(price_b)

            self._kalman_update(price_a, price_b)
            residual = price_a - self._theta * price_b
            self._spreads.append(residual)
            self._n_updates += 1

            n = len(self._spreads)
            theta_snap = self._theta

            if n < MIN_HISTORY:
                return PairsSignal(
                    asset_a=self._asset_a,
                    asset_b=self._asset_b,
                    action="HOLD",
                    z_score=0.0,
                    hedge_ratio=theta_snap,
                    spread=residual,
                    reason=f"warming_up:{n}/{MIN_HISTORY}",
                    timestamp=timestamp,
                )

            spread_mean, spread_std = self._rolling_stats()
            if spread_std == 0.0:
                return PairsSignal(
                    asset_a=self._asset_a,
                    asset_b=self._asset_b,
                    action="HOLD",
                    z_score=0.0,
                    hedge_ratio=theta_snap,
                    spread=residual,
                    reason="spread_std_zero",
                    timestamp=timestamp,
                )

            z_score = (residual - spread_mean) / spread_std

            # Store z-score for adaptive threshold
            self._z_scores.append(z_score)

            # Update adaptive threshold
            self._last_adaptive_threshold = self._compute_adaptive_threshold()

            # Update half-life
            self._last_half_life = self._estimate_half_life()

            # Update correlation
            self._last_correlation = self._compute_correlation()

            return self._generate_signal(
                z_score, residual, theta_snap,
                spread_mean, spread_std, timestamp,
            )

    def get_state(self) -> Optional[PairState]:
        """Return the current pair state, or None before MIN_HISTORY updates."""
        with self._lock:
            if len(self._spreads) < MIN_HISTORY:
                return None

            spread_mean, spread_std = self._rolling_stats()
            residual = float(list(self._spreads)[-1]) if self._spreads else 0.0
            z_score = (residual - spread_mean) / spread_std if spread_std > 0 else 0.0

            return PairState(
                asset_a=self._asset_a,
                asset_b=self._asset_b,
                hedge_ratio=self._theta,
                spread_mean=spread_mean,
                spread_std=spread_std,
                current_spread=residual,
                z_score=z_score,
                position=self._position,
                opened_ts=self._opened_ts,
                half_life=self._last_half_life,
                adaptive_threshold=self._last_adaptive_threshold,
                correlation=self._last_correlation,
                position_size_mult=self._compute_size_multiplier(z_score),
            )

    def reset(self) -> None:
        """Reset filter state and position (does not close any live orders)."""
        with self._lock:
            self._theta = 1.0
            self._C = 0.0
            self._spreads.clear()
            self._prices_a.clear()
            self._prices_b.clear()
            self._z_scores.clear()
            self._position = "FLAT"
            self._opened_ts = None
            self._n_updates = 0
            self._n_trades = 0
            self._last_half_life = None
            self._last_correlation = None
            self._last_adaptive_threshold = ENTRY_ZSCORE
        logger.info(
            "KalmanPairsTrader reset: %s/%s", self._asset_a, self._asset_b
        )

    def get_stats(self) -> Dict:
        """Return diagnostic statistics."""
        with self._lock:
            state = self.get_state()
            return {
                "n_updates": self._n_updates,
                "current_hedge_ratio": round(self._theta, 6),
                "current_z": round(state.z_score, 4) if state else 0.0,
                "n_trades": self._n_trades,
                "position": self._position,
                "history_len": len(self._spreads),
                "adaptive_threshold": round(self._last_adaptive_threshold, 4),
                "half_life": round(self._last_half_life, 2) if self._last_half_life is not None else None,
                "correlation": round(self._last_correlation, 4) if self._last_correlation is not None else None,
            }

    # ------------------------------------------------------------------
    # Internal: Kalman filter
    # ------------------------------------------------------------------

    def _kalman_update(self, price_a: float, price_b: float) -> None:
        """
        1-D Kalman filter update step.

        State variable: θ (hedge ratio).
        Observation:    y = price_A  (with regressor price_B).

        Prediction
        ----------
        R = C + delta          # inflate state covariance by process noise

        Update
        ------
        yhat   = price_B * θ                      # predicted price_A
        Q      = price_B^2 * R + ve               # innovation variance
        K      = price_B * R / Q                  # Kalman gain (scalar)
        theta += K * (price_A - yhat)             # state update
        C      = (1 - K * price_B) * R            # covariance update

        All quantities are scalars because we track a single state (θ).
        """
        # Prediction: inflate covariance
        R = self._C + self._delta

        # Update
        yhat = price_b * self._theta
        Q = price_b ** 2 * R + self._ve
        K = price_b * R / Q
        self._theta = self._theta + K * (price_a - yhat)
        self._C = (1.0 - K * price_b) * R

    # ------------------------------------------------------------------
    # Internal: spread statistics and signal logic
    # ------------------------------------------------------------------

    def _rolling_stats(self) -> tuple[float, float]:
        """Return (mean, std) of the current spread deque."""
        data = list(self._spreads)
        n = len(data)
        if n == 0:
            return 0.0, 0.0
        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / max(n - 1, 1)
        return mean, math.sqrt(variance)

    def _compute_adaptive_threshold(self) -> float:
        """
        Compute adaptive entry threshold as the rolling percentile
        of absolute z-scores. Adapts to changing volatility regimes.

        Returns a z-score threshold clamped to [MIN_ADAPTIVE_ZSCORE, MAX_ADAPTIVE_ZSCORE].
        Falls back to ENTRY_ZSCORE if insufficient history.
        """
        z_list = list(self._z_scores)
        if len(z_list) < 20:
            return ENTRY_ZSCORE

        abs_zs = sorted(abs(z) for z in z_list)
        idx = int(len(abs_zs) * self._adaptive_percentile)
        idx = min(idx, len(abs_zs) - 1)
        threshold = abs_zs[idx]

        # Clamp
        threshold = max(MIN_ADAPTIVE_ZSCORE, min(MAX_ADAPTIVE_ZSCORE, threshold))
        return threshold

    def _estimate_half_life(self) -> Optional[float]:
        """
        Estimate spread mean-reversion half-life using Ornstein-Uhlenbeck fit.

        Regresses spread[t] - spread[t-1] on spread[t-1] to estimate the
        mean-reversion speed parameter λ, then:
            half_life = -ln(2) / λ

        Returns None if insufficient data or non-mean-reverting.
        """
        spreads = list(self._spreads)
        n = len(spreads)
        if n < 20:
            return None

        # OU regression: Δs = λ * s_{t-1} + ε
        # s_lagged = spread[t-1], ds = spread[t] - spread[t-1]
        s_lagged = []
        ds = []
        for i in range(1, n):
            s_lagged.append(spreads[i - 1])
            ds.append(spreads[i] - spreads[i - 1])

        # OLS: λ = Σ(s_lag * ds) / Σ(s_lag^2)
        sum_xy = sum(x * y for x, y in zip(s_lagged, ds))
        sum_xx = sum(x * x for x in s_lagged)

        if sum_xx < 1e-12:
            return None

        lam = sum_xy / sum_xx

        # λ must be negative for mean-reversion
        if lam >= 0:
            return None

        half_life = -math.log(2) / lam
        return half_life

    def _compute_correlation(self) -> Optional[float]:
        """
        Compute rolling Pearson correlation between price_a and price_b
        over the correlation window.

        Returns None if insufficient data.
        """
        pa = list(self._prices_a)
        pb = list(self._prices_b)
        n = min(len(pa), len(pb))

        if n < 20:
            return None

        # Use the last n prices
        pa = pa[-n:]
        pb = pb[-n:]

        mean_a = sum(pa) / n
        mean_b = sum(pb) / n

        cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(pa, pb))
        var_a = sum((a - mean_a) ** 2 for a in pa)
        var_b = sum((b - mean_b) ** 2 for b in pb)

        denom = math.sqrt(var_a * var_b)
        if denom < 1e-12:
            return None

        return cov / denom

    def _compute_size_multiplier(self, z_score: float) -> float:
        """
        Dynamic position sizing based on z-score magnitude.

        Linear scaling: at the adaptive threshold we use BASE_SIZE,
        at 2x threshold we use MAX_SIZE_MULTIPLIER.
        Bigger position at z=3.0 than z=2.0.
        """
        threshold = self._last_adaptive_threshold
        abs_z = abs(z_score)

        if abs_z <= threshold:
            return BASE_SIZE

        # Linear interpolation from threshold to 2*threshold
        extra = abs_z - threshold
        range_width = max(threshold, 0.5)  # avoid div-by-zero
        scale = min(1.0, extra / range_width)
        return BASE_SIZE + scale * (MAX_SIZE_MULTIPLIER - BASE_SIZE)

    def _check_filters(self, z_score: float, timestamp: float) -> Optional[PairsSignal]:
        """
        Check correlation regime filter and half-life filter.
        Returns a HOLD signal with reason if a filter blocks entry, else None.
        """
        # Correlation filter
        corr = self._last_correlation
        if corr is not None and corr < self._min_correlation:
            logger.debug(
                "FILTER: %s/%s correlation=%.3f < %.3f — skipping entry",
                self._asset_a, self._asset_b, corr, self._min_correlation,
            )
            return PairsSignal(
                asset_a=self._asset_a,
                asset_b=self._asset_b,
                action="HOLD",
                z_score=z_score,
                hedge_ratio=self._theta,
                spread=0.0,
                reason=f"correlation_too_low:{corr:.3f}<{self._min_correlation}",
                timestamp=timestamp,
            )

        # Half-life filter
        hl = self._last_half_life
        if hl is not None:
            if hl < self._half_life_min or hl > self._half_life_max:
                logger.debug(
                    "FILTER: %s/%s half_life=%.1f outside [%d,%d] — skipping entry",
                    self._asset_a, self._asset_b, hl,
                    self._half_life_min, self._half_life_max,
                )
                return PairsSignal(
                    asset_a=self._asset_a,
                    asset_b=self._asset_b,
                    action="HOLD",
                    z_score=z_score,
                    hedge_ratio=self._theta,
                    spread=0.0,
                    reason=f"half_life_out_of_range:{hl:.1f}",
                    timestamp=timestamp,
                )

        return None  # all filters pass

    def _generate_signal(
        self,
        z_score: float,
        spread: float,
        hedge_ratio: float,
        spread_mean: float,   # noqa: ARG002
        spread_std: float,    # noqa: ARG002
        timestamp: float,
    ) -> PairsSignal:
        """Apply entry/exit logic and update position state."""

        # ---- Stop-loss: z-score too extreme while in a position ----
        if self._position != "FLAT" and abs(z_score) > STOP_ZSCORE:
            logger.warning(
                "STOP_LOSS: %s/%s z=%.3f exceeds stop %.1f",
                self._asset_a, self._asset_b, z_score, STOP_ZSCORE,
            )
            self._position = "FLAT"
            self._opened_ts = None
            self._n_trades += 1
            return PairsSignal(
                asset_a=self._asset_a,
                asset_b=self._asset_b,
                action="EXIT",
                z_score=z_score,
                hedge_ratio=hedge_ratio,
                spread=spread,
                reason=f"stop_loss:z={z_score:.3f}>{STOP_ZSCORE}",
                timestamp=timestamp,
            )

        # ---- Exit: spread reverted ----
        if self._position != "FLAT" and abs(z_score) < EXIT_ZSCORE:
            logger.info(
                "EXIT: %s/%s z=%.3f below exit threshold %.1f",
                self._asset_a, self._asset_b, z_score, EXIT_ZSCORE,
            )
            self._position = "FLAT"
            self._opened_ts = None
            self._n_trades += 1
            return PairsSignal(
                asset_a=self._asset_a,
                asset_b=self._asset_b,
                action="EXIT",
                z_score=z_score,
                hedge_ratio=hedge_ratio,
                spread=spread,
                reason=f"spread_reverted:z={z_score:.3f}<{EXIT_ZSCORE}",
                timestamp=timestamp,
            )

        # ---- Already positioned — hold ----
        if self._position != "FLAT":
            return PairsSignal(
                asset_a=self._asset_a,
                asset_b=self._asset_b,
                action="HOLD",
                z_score=z_score,
                hedge_ratio=hedge_ratio,
                spread=spread,
                reason=f"holding:{self._position} z={z_score:.3f}",
                timestamp=timestamp,
            )

        # ---- Use adaptive threshold instead of fixed ENTRY_ZSCORE ----
        entry_threshold = self._last_adaptive_threshold

        # ---- Entry signals (with filters) ----
        if abs(z_score) > entry_threshold:
            # Check correlation + half-life filters before entering
            filter_signal = self._check_filters(z_score, timestamp)
            if filter_signal is not None:
                return filter_signal

            size_mult = self._compute_size_multiplier(z_score)

            if z_score > entry_threshold:
                # Spread too high → SHORT spread (sell A, buy B)
                logger.info(
                    "SHORT_SPREAD: %s/%s z=%.3f > adaptive_threshold=%.3f "
                    "size_mult=%.2f hl=%.1f corr=%.3f",
                    self._asset_a, self._asset_b, z_score, entry_threshold,
                    size_mult,
                    self._last_half_life or 0,
                    self._last_correlation or 0,
                )
                self._position = "SHORT_SPREAD"
                self._opened_ts = timestamp
                self._n_trades += 1
                return PairsSignal(
                    asset_a=self._asset_a,
                    asset_b=self._asset_b,
                    action="SHORT_SPREAD",
                    z_score=z_score,
                    hedge_ratio=hedge_ratio,
                    spread=spread,
                    reason=f"z={z_score:.3f}>{entry_threshold:.3f}",
                    timestamp=timestamp,
                    position_size_mult=size_mult,
                )

            if z_score < -entry_threshold:
                # Spread too low → LONG spread (buy A, sell B)
                logger.info(
                    "LONG_SPREAD: %s/%s z=%.3f < -adaptive_threshold=%.3f "
                    "size_mult=%.2f hl=%.1f corr=%.3f",
                    self._asset_a, self._asset_b, z_score, entry_threshold,
                    size_mult,
                    self._last_half_life or 0,
                    self._last_correlation or 0,
                )
                self._position = "LONG_SPREAD"
                self._opened_ts = timestamp
                self._n_trades += 1
                return PairsSignal(
                    asset_a=self._asset_a,
                    asset_b=self._asset_b,
                    action="LONG_SPREAD",
                    z_score=z_score,
                    hedge_ratio=hedge_ratio,
                    spread=spread,
                    reason=f"z={z_score:.3f}<-{entry_threshold:.3f}",
                    timestamp=timestamp,
                    position_size_mult=size_mult,
                )

        # ---- No signal ----
        return PairsSignal(
            asset_a=self._asset_a,
            asset_b=self._asset_b,
            action="HOLD",
            z_score=z_score,
            hedge_ratio=hedge_ratio,
            spread=spread,
            reason=f"no_signal:z={z_score:.3f}",
            timestamp=timestamp,
        )
