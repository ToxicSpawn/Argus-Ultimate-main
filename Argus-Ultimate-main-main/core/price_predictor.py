"""
Next-Bar Price Predictor — ARGUS anticipates instead of reacting.

This is the fundamental shift from reactive to predictive trading.
Instead of "price crossed above SMA, buy now", ARGUS predicts
"price will likely increase 0.3% in the next bar" and positions BEFORE.

Three prediction models (ensemble):
1. Kalman Filter — optimal linear state estimator, adapts online
2. Momentum Extrapolation — short-term trend continuation/mean-reversion
3. Microstructure Model — order flow imbalance → price direction

The ensemble prediction is injected into:
- Strategy generator as a new node type (PREDICTED_RETURN)
- Signal confidence weighting (agree with prediction = higher confidence)
- Execution timing (enter BEFORE predicted move, not after)
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PricePrediction:
    """Prediction for the next bar."""
    symbol: str
    predicted_return_pct: float     # expected return in %
    confidence: float               # 0-1
    direction: str                  # "UP", "DOWN", "FLAT"
    predicted_price: float          # absolute price prediction
    prediction_std: float           # uncertainty (1 std)
    models_agree: int               # how many of 3 models agree on direction
    # Per-model breakdown
    kalman_return: float = 0.0
    momentum_return: float = 0.0
    microstructure_return: float = 0.0


class KalmanPriceFilter:
    """
    Kalman filter for price estimation and prediction.

    State: [price, velocity, acceleration]
    Observation: close price
    Predicts next-bar price by extrapolating state.
    """

    def __init__(self, process_noise: float = 0.001, measurement_noise: float = 0.01):
        self._state = np.zeros(3)       # [price, velocity, acceleration]
        self._P = np.eye(3) * 100       # state covariance (high initial uncertainty)
        self._Q = np.eye(3) * process_noise   # process noise
        self._R = np.array([[measurement_noise]])  # measurement noise
        self._F = np.array([             # state transition
            [1, 1, 0.5],                 # price += velocity + 0.5*accel
            [0, 1, 1],                   # velocity += acceleration
            [0, 0, 0.95],               # acceleration decays (mean-reverting)
        ])
        self._H = np.array([[1, 0, 0]])  # observe price only
        self._initialized = False
        self._predictions: deque = deque(maxlen=100)

    def update(self, price: float) -> None:
        """Update filter with new price observation."""
        if not self._initialized:
            self._state[0] = price
            self._initialized = True
            return

        # Predict
        x_pred = self._F @ self._state
        P_pred = self._F @ self._P @ self._F.T + self._Q

        # Update
        y = np.array([price]) - self._H @ x_pred  # innovation
        S = self._H @ P_pred @ self._H.T + self._R  # innovation covariance
        K = P_pred @ self._H.T @ np.linalg.inv(S)  # Kalman gain

        self._state = x_pred + K @ y
        self._P = (np.eye(3) - K @ self._H) @ P_pred

        # Track prediction accuracy
        if self._predictions:
            last_pred = self._predictions[-1]
            error = abs(price - last_pred) / max(price, 1e-9)
            # Adapt process noise based on prediction error
            self._Q *= (1 + error * 0.1)  # increase noise if predictions are bad
            self._Q = np.clip(self._Q, 1e-6, 0.1)

    def predict(self) -> Tuple[float, float]:
        """Predict next-bar price and uncertainty.
        Returns (predicted_price, std_dev)."""
        if not self._initialized:
            return 0.0, 999.0

        x_pred = self._F @ self._state
        P_pred = self._F @ self._P @ self._F.T + self._Q
        predicted_price = x_pred[0]
        uncertainty = (P_pred[0, 0]) ** 0.5

        self._predictions.append(predicted_price)
        return float(predicted_price), float(uncertainty)


class MomentumPredictor:
    """
    Short-term momentum extrapolation.

    Uses exponential-weighted return to predict continuation or
    mean-reversion based on recent autocorrelation.
    """

    def __init__(self, fast_window: int = 5, slow_window: int = 20, decay: float = 0.94):
        self._prices: deque = deque(maxlen=slow_window + 5)
        self._fast = fast_window
        self._slow = slow_window
        self._decay = decay

    def update(self, price: float) -> None:
        self._prices.append(price)

    def predict(self) -> float:
        """Predict next-bar return in %. Positive = expect up."""
        if len(self._prices) < self._slow + 1:
            return 0.0

        prices = list(self._prices)
        returns = [(prices[i] / prices[i - 1] - 1) * 100
                    for i in range(1, len(prices))]

        # Fast momentum (recent trend)
        fast_rets = returns[-self._fast:]
        fast_mom = sum(r * (self._decay ** (len(fast_rets) - 1 - i))
                       for i, r in enumerate(fast_rets))
        fast_mom /= sum(self._decay ** i for i in range(len(fast_rets)))

        # Slow momentum (longer trend)
        slow_rets = returns[-self._slow:]
        slow_mom = sum(slow_rets) / len(slow_rets)

        # Autocorrelation: if positive, trend continues; if negative, mean-reverts
        if len(returns) >= 10:
            n = min(20, len(returns))
            rets = returns[-n:]
            mean_r = sum(rets) / n
            autocorr = sum((rets[i] - mean_r) * (rets[i - 1] - mean_r)
                           for i in range(1, n))
            var_r = sum((r - mean_r) ** 2 for r in rets)
            autocorr = autocorr / max(var_r, 1e-9)
        else:
            autocorr = 0.0

        # If autocorrelation > 0: trend continues → use fast momentum
        # If autocorrelation < 0: mean-reversion → fade fast momentum
        if autocorr > 0.1:
            return fast_mom * 0.7 + slow_mom * 0.3
        elif autocorr < -0.1:
            return -fast_mom * 0.3 + slow_mom * 0.2
        else:
            return slow_mom * 0.5


class MicrostructurePredictor:
    """
    Microstructure-based price direction prediction.

    Uses volume, spread, and volatility to predict next-bar direction.
    High volume + narrowing spread = institutional accumulation = UP.
    Low volume + widening spread = distribution = DOWN.
    """

    def __init__(self, window: int = 20):
        self._volumes: deque = deque(maxlen=window + 5)
        self._spreads: deque = deque(maxlen=window + 5)
        self._returns: deque = deque(maxlen=window + 5)
        self._window = window

    def update(self, volume: float, spread_bps: float, ret: float) -> None:
        self._volumes.append(volume)
        self._spreads.append(spread_bps)
        self._returns.append(ret)

    def predict(self) -> float:
        """Predict next-bar return in %. Based on microstructure signals."""
        if len(self._volumes) < self._window:
            return 0.0

        vols = list(self._volumes)
        spreads = list(self._spreads)
        rets = list(self._returns)

        # Volume trend: increasing volume with positive returns = accumulation
        avg_vol = sum(vols[-self._window:]) / self._window
        recent_vol = sum(vols[-5:]) / 5
        vol_ratio = recent_vol / max(avg_vol, 1e-9)

        # Spread trend: narrowing spread = confidence
        avg_spread = sum(spreads[-self._window:]) / self._window
        recent_spread = sum(spreads[-5:]) / 5
        spread_ratio = recent_spread / max(avg_spread, 1e-9)

        # Recent return direction
        recent_ret = sum(rets[-5:]) / 5

        # Signals:
        # High vol + narrow spread + positive return = accumulation = UP
        # High vol + wide spread + negative return = distribution = DOWN
        signal = 0.0
        if vol_ratio > 1.2 and spread_ratio < 0.9:
            signal = 0.3  # accumulation
        elif vol_ratio > 1.2 and spread_ratio > 1.1:
            signal = -0.2  # distribution
        elif vol_ratio < 0.8:
            signal = -0.1  # low conviction

        # Weight by recent direction
        if recent_ret > 0 and signal > 0:
            signal *= 1.5  # confirmation
        elif recent_ret < 0 and signal < 0:
            signal *= 1.5  # confirmation

        return signal


class PricePredictor:
    """
    Ensemble price predictor combining Kalman, Momentum, and Microstructure.

    Produces a PricePrediction for each symbol every cycle.
    """

    def __init__(self):
        self._kalman: Dict[str, KalmanPriceFilter] = {}
        self._momentum: Dict[str, MomentumPredictor] = {}
        self._micro: Dict[str, MicrostructurePredictor] = {}

    def update(
        self,
        symbol: str,
        price: float,
        volume: float = 0.0,
        spread_bps: float = 0.0,
    ) -> None:
        """Feed new data point."""
        if symbol not in self._kalman:
            self._kalman[symbol] = KalmanPriceFilter()
            self._momentum[symbol] = MomentumPredictor()
            self._micro[symbol] = MicrostructurePredictor()

        self._kalman[symbol].update(price)
        self._momentum[symbol].update(price)

        # Compute return for microstructure
        mom = self._momentum[symbol]
        if len(mom._prices) >= 2:
            ret = (list(mom._prices)[-1] / list(mom._prices)[-2] - 1) * 100
        else:
            ret = 0.0
        self._micro[symbol].update(volume, spread_bps, ret)

    def predict(self, symbol: str) -> PricePrediction:
        """Predict next-bar price for a symbol."""
        if symbol not in self._kalman:
            return PricePrediction(
                symbol=symbol, predicted_return_pct=0, confidence=0,
                direction="FLAT", predicted_price=0, prediction_std=999,
                models_agree=0,
            )

        # Kalman prediction
        k_price, k_std = self._kalman[symbol].predict()
        current = self._kalman[symbol]._state[0] if self._kalman[symbol]._initialized else k_price
        k_return = ((k_price / max(current, 1e-9)) - 1) * 100 if current > 0 else 0

        # Momentum prediction
        m_return = self._momentum[symbol].predict()

        # Microstructure prediction
        ms_return = self._micro[symbol].predict()

        # Ensemble: weighted average (Kalman most reliable, micro least)
        ensemble_return = k_return * 0.45 + m_return * 0.35 + ms_return * 0.20

        # Direction consensus
        directions = []
        for r in [k_return, m_return, ms_return]:
            if r > 0.05:
                directions.append("UP")
            elif r < -0.05:
                directions.append("DOWN")
            else:
                directions.append("FLAT")

        # Count agreement
        up_count = directions.count("UP")
        down_count = directions.count("DOWN")
        if up_count >= 2:
            direction = "UP"
            agree = up_count
        elif down_count >= 2:
            direction = "DOWN"
            agree = down_count
        else:
            direction = "FLAT"
            agree = 0

        # Confidence: based on agreement + uncertainty
        conf = agree / 3.0
        if k_std > 0 and current > 0:
            uncertainty_pct = k_std / current * 100
            if uncertainty_pct < abs(ensemble_return):
                conf *= 1.2  # prediction exceeds uncertainty = more confident
            else:
                conf *= 0.7  # prediction within noise

        conf = max(0.0, min(1.0, conf))
        predicted_price = current * (1 + ensemble_return / 100)

        return PricePrediction(
            symbol=symbol,
            predicted_return_pct=ensemble_return,
            confidence=conf,
            direction=direction,
            predicted_price=predicted_price,
            prediction_std=k_std,
            models_agree=agree,
            kalman_return=k_return,
            momentum_return=m_return,
            microstructure_return=ms_return,
        )

    def get_all_predictions(self) -> Dict[str, PricePrediction]:
        """Get predictions for all tracked symbols."""
        return {sym: self.predict(sym) for sym in self._kalman}

    def get_stats(self) -> Dict[str, Any]:
        return {
            "symbols_tracked": len(self._kalman),
            "predictions": {sym: self.predict(sym).predicted_return_pct
                           for sym in list(self._kalman.keys())[:5]},
        }
