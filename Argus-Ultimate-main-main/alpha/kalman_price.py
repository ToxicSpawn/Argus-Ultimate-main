"""
KalmanPriceEstimator — 1-D Kalman filter mid-price estimator.

Models price as a random walk with Gaussian noise:
    x[t] = x[t-1] + w[t]        (process noise)
    z[t] = x[t]  + v[t]         (observation noise)

The filter outputs a smoothed mid-price estimate and uncertainty
(posterior variance) which can be used as an alpha signal or for
dynamic spread/sizing decisions.

Usage
-----
    kf = KalmanPriceEstimator(process_var=1e-4, obs_var=1e-2)
    for bid, ask in tick_stream:
        mid = (bid + ask) / 2
        est = kf.update(mid)
        print(est.price, est.uncertainty)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Deque, List, Optional
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class KalmanState:
    price: float            # posterior mean (filtered estimate)
    uncertainty: float      # posterior variance P
    innovation: float       # z - x_prior  (useful for anomaly detection)
    gain: float             # Kalman gain K


class KalmanPriceEstimator:
    """
    Univariate Kalman filter for mid-price smoothing.

    Parameters
    ----------
    process_var  : Q — variance of the hidden state transition (higher = faster tracking)
    obs_var      : R — variance of the observation noise   (higher = more smoothing)
    init_price   : optional seed price; uses first observation if None
    history_len  : length of rolling state history kept in memory
    """

    def __init__(
        self,
        process_var: float = 1e-4,
        obs_var: float = 1e-2,
        init_price: Optional[float] = None,
        history_len: int = 200,
    ) -> None:
        if process_var <= 0 or obs_var <= 0:
            raise ValueError("process_var and obs_var must be positive")

        self._Q = process_var
        self._R = obs_var
        self._history: Deque[KalmanState] = deque(maxlen=history_len)

        # State
        self._x: Optional[float] = init_price   # posterior mean
        self._P: float = 1.0                     # posterior variance
        self._initialised = init_price is not None

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, observation: float) -> KalmanState:
        """
        Ingest a new mid-price observation and return the updated state.
        On the first call (if no init_price was given) the filter
        bootstraps itself from the observation.
        """
        if not self._initialised:
            self._x = observation
            self._P = self._R
            self._initialised = True
            state = KalmanState(
                price=self._x,
                uncertainty=self._P,
                innovation=0.0,
                gain=1.0,
            )
            self._history.append(state)
            return state

        # --- Predict ---
        x_prior = self._x                   # state prediction (random walk: x[t|t-1] = x[t-1])
        P_prior = self._P + self._Q         # covariance prediction

        # --- Update ---
        innovation = observation - x_prior
        S = P_prior + self._R               # innovation covariance
        K = P_prior / S                     # Kalman gain

        self._x = x_prior + K * innovation
        self._P = (1.0 - K) * P_prior

        state = KalmanState(
            price=self._x,
            uncertainty=self._P,
            innovation=innovation,
            gain=K,
        )
        self._history.append(state)
        return state

    # ------------------------------------------------------------------
    # Derived signals
    # ------------------------------------------------------------------

    @property
    def current_price(self) -> Optional[float]:
        return self._x

    @property
    def current_uncertainty(self) -> float:
        return self._P

    def trend_signal(self, lookback: int = 10) -> float:
        """
        Simple trend signal derived from the Kalman-smoothed price series.
        Returns the slope of a linear regression over the last `lookback`
        states, normalised by the current price to give a dimensionless value.

        Positive -> uptrend, negative -> downtrend.
        Clamped to [-1, 1].
        """
        hist = list(self._history)
        if len(hist) < 2:
            return 0.0
        window = hist[-min(lookback, len(hist)):]
        n = len(window)
        prices = [s.price for s in window]
        mean_p = sum(prices) / n
        if mean_p == 0:
            return 0.0
        # OLS slope
        x_vals = list(range(n))
        mean_x = (n - 1) / 2.0
        num = sum((x_vals[i] - mean_x) * (prices[i] - mean_p) for i in range(n))
        den = sum((x_vals[i] - mean_x) ** 2 for i in range(n))
        slope = num / den if den != 0 else 0.0
        # Normalise: slope per bar as fraction of price
        normalised = slope / mean_p
        # Scale so that 0.1% per bar ~ 1.0 signal
        scaled = normalised / 0.001
        return max(-1.0, min(1.0, scaled))

    def innovation_zscore(self, window: int = 20) -> float:
        """
        Z-score of the latest innovation relative to recent history.
        Large positive z-score => price moved above filter => potential mean-reversion short.
        Large negative z-score => price moved below filter => potential mean-reversion long.
        """
        hist = list(self._history)
        if len(hist) < 2:
            return 0.0
        recent = hist[-min(window, len(hist)):]
        innovations = [s.innovation for s in recent]
        mean_inn = sum(innovations) / len(innovations)
        var_inn = sum((v - mean_inn) ** 2 for v in innovations) / len(innovations)
        std_inn = math.sqrt(var_inn) if var_inn > 0 else 1.0
        latest = hist[-1].innovation
        return (latest - mean_inn) / std_inn

    def get_history(self) -> List[KalmanState]:
        return list(self._history)

    def reset(self, init_price: Optional[float] = None) -> None:
        """Reset filter state (e.g. after a market halt or symbol change)."""
        self._x = init_price
        self._P = 1.0
        self._initialised = init_price is not None
        self._history.clear()
        logger.info("KalmanPriceEstimator reset (init_price=%s)", init_price)
