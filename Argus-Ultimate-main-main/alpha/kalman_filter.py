"""Batch 2 — Kalman filter for price trend extraction.

Implements a 2-state Kalman filter (level, velocity) that can be used
as a low-noise price estimator or spread tracker in pairs trading.

References
----------
* Welch & Bishop — An Introduction to the Kalman Filter (2006)
* Quantopian lecture on Kalman-filter pairs trading
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class KalmanState:
    mean: np.ndarray        # [level, velocity]
    covariance: np.ndarray  # 2x2
    innovation: float       # y - y_hat
    innovation_var: float


class KalmanPriceFilter:
    """Univariate Kalman filter tracking price level and velocity."""

    def __init__(
        self,
        observation_noise: float = 1.0,
        process_noise: float = 0.01,
        dt: float = 1.0,
    ) -> None:
        # State transition matrix  (level, velocity)
        self._F = np.array([[1, dt], [0, 1]], dtype=float)
        # Observation matrix
        self._H = np.array([[1, 0]], dtype=float)
        # Process noise covariance
        q = process_noise
        self._Q = q * np.array(
            [[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]], dtype=float
        )
        # Observation noise covariance
        self._R = np.array([[observation_noise]], dtype=float)
        # Initial state
        self._x = np.zeros((2, 1))
        self._P = np.eye(2) * 1e3
        self._initialised = False

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def update(self, price: float) -> KalmanState:
        """Process one observation; return posterior state."""
        y = np.array([[price]])

        if not self._initialised:
            self._x[0, 0] = price
            self._initialised = True

        # Predict
        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + self._Q

        # Update
        S = self._H @ P_pred @ self._H.T + self._R
        K = P_pred @ self._H.T @ np.linalg.inv(S)  # Kalman gain
        innovation = float((y - self._H @ x_pred)[0, 0])
        self._x = x_pred + K * innovation
        self._P = (np.eye(2) - K @ self._H) @ P_pred

        return KalmanState(
            mean=self._x.copy(),
            covariance=self._P.copy(),
            innovation=innovation,
            innovation_var=float(S[0, 0]),
        )

    def batch_update(self, prices: List[float]) -> List[KalmanState]:
        """Run the filter over a sequence of prices."""
        return [self.update(p) for p in prices]

    @property
    def level(self) -> float:
        return float(self._x[0, 0])

    @property
    def velocity(self) -> float:
        return float(self._x[1, 0])

    def reset(self) -> None:
        self._x = np.zeros((2, 1))
        self._P = np.eye(2) * 1e3
        self._initialised = False


class KalmanPairsSpread:
    """Kalman filter hedge-ratio estimator for pairs trading."""

    def __init__(
        self,
        delta: float = 1e-4,
        observation_noise: float = 1e-3,
    ) -> None:
        self._delta = delta
        self._R = observation_noise
        self._theta = np.zeros(2)  # [beta, alpha]
        self._P = np.zeros((2, 2))
        self._Vw = delta / (1 - delta) * np.eye(2)
        self._Ve = observation_noise

    def update(self, x: float, y: float) -> Tuple[float, float, float]:
        """Update hedge ratio; return (spread, beta, alpha)."""
        F = np.array([x, 1.0])

        # Predict
        P_pred = self._P + self._Vw

        # Update
        Q = F @ P_pred @ F + self._Ve
        K = P_pred @ F / Q
        innovation = y - F @ self._theta
        self._theta = self._theta + K * innovation
        self._P = (np.eye(2) - np.outer(K, F)) @ P_pred

        spread = y - self._theta[0] * x - self._theta[1]
        return spread, self._theta[0], self._theta[1]
