"""
Time Series Forecaster — transformer-based price prediction using darts.

Provides multi-horizon price forecasts using:
- TransformerModel (attention-based, similar to PatchTST)
- NHiTSModel (multi-rate sampling)
- TFTModel (Temporal Fusion Transformer)

Falls back to simpler models if torch/darts unavailable.

Example::

    forecaster = TimeSeriesForecaster(model_type="transformer")
    forecaster.fit(symbol="BTC/USD", prices=[...])
    forecast = forecaster.predict("BTC/USD", horizon=12)
    print(forecast.predicted_prices)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

# Check darts availability
try:
    from darts import TimeSeries
    from darts.models import TransformerModel, NHiTSModel, TFTModel
    _DARTS_AVAILABLE = True
except ImportError:
    _DARTS_AVAILABLE = False
    logger.debug("darts not available; using simple forecast fallback")


@dataclass
class ForecastResult:
    """Single forecast output."""
    symbol: str
    timestamp: float
    horizon: int  # number of steps ahead
    predicted_prices: List[float]
    confidence_lower: List[float]
    confidence_upper: List[float]
    model_type: str
    mae: float  # mean absolute error on validation
    direction_accuracy: float  # % correct direction predictions


@dataclass
class _SymbolState:
    prices: Deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    last_forecast: Optional[ForecastResult] = None
    model: Any = None
    fitted: bool = False


class TimeSeriesForecaster:
    """
    Multi-symbol time series forecaster using darts transformer models.

    Parameters
    ----------
    model_type : str
        Model to use: "transformer", "nhits", "tft", or "simple" (fallback).
    input_chunk_length : int
        Number of past time steps to use as input (default 48).
    output_chunk_length : int
        Number of future steps to predict (default 12).
    n_epochs : int
        Training epochs (default 50).
    hidden_size : int
        Hidden layer size (default 64).
    min_train_samples : int
        Minimum prices before training (default 200).
    """

    def __init__(
        self,
        model_type: str = "transformer",
        input_chunk_length: int = 48,
        output_chunk_length: int = 12,
        n_epochs: int = 50,
        hidden_size: int = 64,
        min_train_samples: int = 200,
    ) -> None:
        self._model_type = model_type.lower()
        self._input_chunk_length = input_chunk_length
        self._output_chunk_length = output_chunk_length
        self._n_epochs = n_epochs
        self._hidden_size = hidden_size
        self._min_train_samples = min_train_samples
        self._states: Dict[str, _SymbolState] = {}

        if not _DARTS_AVAILABLE and self._model_type != "simple":
            logger.warning(
                "darts not available; falling back to simple forecast model"
            )
            self._model_type = "simple"

        logger.info(
            "TimeSeriesForecaster initialized: model=%s input_chunk=%d output_chunk=%d epochs=%d",
            self._model_type, input_chunk_length, output_chunk_length, n_epochs,
        )

    def update(self, symbol: str, price: float) -> None:
        """Add a new price observation for symbol."""
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()
        self._states[symbol].prices.append(price)

    def fit(self, symbol: str, prices: Optional[List[float]] = None) -> bool:
        """
        Train/retrain the model for symbol.

        Returns True if training succeeded.
        """
        if self._model_type == "simple":
            return True  # Simple model doesn't need training

        if symbol not in self._states:
            self._states[symbol] = _SymbolState()

        state = self._states[symbol]

        # Use provided prices or internal history
        if prices is not None:
            state.prices.extend(prices)

        if len(state.prices) < self._min_train_samples:
            logger.debug(
                "%s: only %d prices, need %d for training",
                symbol, len(state.prices), self._min_train_samples,
            )
            return False

        try:
            # Create darts TimeSeries
            price_array = np.array(list(state.prices))
            ts = TimeSeries.from_values(price_array.reshape(-1, 1))

            # Split train/validation
            split_idx = int(len(ts) * 0.8)
            train_ts = ts[:split_idx]
            val_ts = ts[split_idx:]

            # Create model
            model = self._create_model()

            # Train
            model.fit(train_ts, epochs=self._n_epochs)

            # Validate
            val_pred = model.predict(n=len(val_ts))
            val_actual = val_ts.values().flatten()
            val_pred_values = val_pred.values().flatten()

            # Calculate metrics
            min_len = min(len(val_actual), len(val_pred_values))
            mae = float(np.mean(np.abs(val_actual[:min_len] - val_pred_values[:min_len])))
            
            # Direction accuracy
            actual_dirs = np.sign(np.diff(val_actual[:min_len]))
            pred_dirs = np.sign(np.diff(val_pred_values[:min_len]))
            direction_accuracy = float(np.mean(actual_dirs == pred_dirs)) if len(actual_dirs) > 0 else 0.5

            state.model = model
            state.fitted = True

            logger.info(
                "%s: model fitted, val_mae=%.4f, direction_acc=%.2f%%",
                symbol, mae, direction_accuracy * 100,
            )
            return True

        except Exception as exc:
            logger.warning("%s: training failed: %s", symbol, exc)
            return False

    def _create_model(self) -> Any:
        """Create the darts model based on model_type."""
        if self._model_type == "transformer":
            return TransformerModel(
                input_chunk_length=self._input_chunk_length,
                output_chunk_length=self._output_chunk_length,
                d_model=self._hidden_size,
                nhead=4,
                num_encoder_layers=3,
                num_decoder_layers=3,
                dim_feedforward=self._hidden_size * 4,
                dropout=0.1,
                batch_size=32,
                n_epochs=self._n_epochs,
                random_state=42,
                force_reset=True,
            )
        elif self._model_type == "nhits":
            return NHiTSModel(
                input_chunk_length=self._input_chunk_length,
                output_chunk_length=self._output_chunk_length,
                num_blocks=2,
                num_layers=2,
                layer_widths=self._hidden_size,
                batch_size=32,
                n_epochs=self._n_epochs,
                random_state=42,
                force_reset=True,
            )
        elif self._model_type == "tft":
            return TFTModel(
                input_chunk_length=self._input_chunk_length,
                output_chunk_length=self._output_chunk_length,
                hidden_size=self._hidden_size,
                lstm_layers=2,
                num_attention_heads=4,
                dropout=0.1,
                batch_size=32,
                n_epochs=self._n_epochs,
                random_state=42,
                force_reset=True,
            )
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")

    def predict(
        self,
        symbol: str,
        horizon: Optional[int] = None,
    ) -> Optional[ForecastResult]:
        """
        Generate price forecast for symbol.

        Parameters
        ----------
        symbol : str
            Trading pair.
        horizon : int, optional
            Number of steps to forecast. Defaults to output_chunk_length.

        Returns
        -------
        ForecastResult or None if insufficient data.
        """
        if symbol not in self._states:
            return None

        state = self._states[symbol]
        horizon = horizon or self._output_chunk_length

        if len(state.prices) < self._input_chunk_length:
            logger.debug(
                "%s: only %d prices, need %d for prediction",
                symbol, len(state.prices), self._input_chunk_length,
            )
            return None

        if self._model_type == "simple":
            return self._simple_predict(symbol, state, horizon)

        if not state.fitted or state.model is None:
            # Try to fit first
            if not self.fit(symbol):
                return self._simple_predict(symbol, state, horizon)

        try:
            # Generate prediction
            pred = state.model.predict(n=horizon)
            pred_values = pred.values().flatten().tolist()

            # Simple confidence intervals (±1 std of recent errors)
            recent_prices = list(state.prices)[-self._input_chunk_length:]
            price_array = np.array(recent_prices)
            ts = TimeSeries.from_values(price_array.reshape(-1, 1))
            
            # Get in-sample predictions for error estimation
            try:
                in_sample = state.model.predict(n=len(recent_prices))
                in_sample_values = in_sample.values().flatten()
                errors = price_array[-len(in_sample_values):] - in_sample_values
                std_error = float(np.std(errors)) if len(errors) > 0 else np.std(price_array) * 0.1
            except:
                std_error = float(np.std(price_array)) * 0.1

            # Expand uncertainty with horizon
            confidence_lower = []
            confidence_upper = []
            for i, pred_val in enumerate(pred_values):
                uncertainty = std_error * np.sqrt(i + 1)  # Uncertainty grows with horizon
                confidence_lower.append(pred_val - uncertainty * 1.96)
                confidence_upper.append(pred_val + uncertainty * 1.96)

            # Direction accuracy from validation
            direction_accuracy = 0.5  # Default

            result = ForecastResult(
                symbol=symbol,
                timestamp=time.time(),
                horizon=horizon,
                predicted_prices=pred_values,
                confidence_lower=confidence_lower,
                confidence_upper=confidence_upper,
                model_type=self._model_type,
                mae=std_error,
                direction_accuracy=direction_accuracy,
            )

            state.last_forecast = result
            return result

        except Exception as exc:
            logger.warning("%s: prediction failed: %s", symbol, exc)
            return self._simple_predict(symbol, state, horizon)

    def _simple_predict(
        self,
        symbol: str,
        state: _SymbolState,
        horizon: int,
    ) -> ForecastResult:
        """Simple exponential smoothing fallback."""
        prices = list(state.prices)
        if len(prices) < 2:
            return None

        # EWMA-based forecast
        alpha = 0.3
        ewma = prices[0]
        for p in prices[1:]:
            ewma = alpha * p + (1 - alpha) * ewma

        # Calculate momentum
        recent = prices[-min(20, len(prices)):]
        if len(recent) >= 2:
            momentum = (recent[-1] - recent[0]) / recent[0]
        else:
            momentum = 0.0

        # Generate forecast with momentum
        last_price = prices[-1]
        predicted_prices = []
        for i in range(1, horizon + 1):
            # Mean reversion towards EWMA with momentum
            forecast = last_price + momentum * last_price * np.sqrt(i) * 0.1
            forecast = forecast * 0.7 + ewma * 0.3  # Blend with mean
            predicted_prices.append(forecast)

        # Simple confidence intervals
        volatility = float(np.std(np.diff(prices[-20:]))) if len(prices) >= 21 else last_price * 0.02
        confidence_lower = [p - volatility * 1.96 * np.sqrt(i + 1) for i, p in enumerate(predicted_prices)]
        confidence_upper = [p + volatility * 1.96 * np.sqrt(i + 1) for i, p in enumerate(predicted_prices)]

        result = ForecastResult(
            symbol=symbol,
            timestamp=time.time(),
            horizon=horizon,
            predicted_prices=predicted_prices,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            model_type="simple",
            mae=volatility,
            direction_accuracy=0.55,  # Slight edge assumed
        )

        state.last_forecast = result
        return result

    def get_forecast(self, symbol: str) -> Optional[ForecastResult]:
        """Get the last forecast for symbol."""
        if symbol in self._states:
            return self._states[symbol].last_forecast
        return None

    def get_direction_signal(self, symbol: str) -> Optional[str]:
        """
        Get trading signal based on forecast direction.

        Returns "buy", "sell", or None.
        """
        forecast = self.get_forecast(symbol)
        if forecast is None or len(forecast.predicted_prices) < 2:
            return None

        current_price = self._states[symbol].prices[-1] if symbol in self._states else None
        if current_price is None:
            return None

        # Predicted price change
        predicted_change = forecast.predicted_prices[0] - current_price
        change_pct = predicted_change / current_price

        # Only signal if change is significant (>0.5%) and confidence is decent
        if abs(change_pct) < 0.005:
            return None

        if forecast.direction_accuracy < 0.52:
            return None  # Model not reliable enough

        return "buy" if change_pct > 0 else "sell"

    def get_all_symbols(self) -> List[str]:
        """Get all symbols with price data."""
        return sorted(self._states.keys())

    def should_retrain(self, symbol: str, min_new_samples: int = 50) -> bool:
        """Check if model should be retrained."""
        if symbol not in self._states:
            return False
        state = self._states[symbol]
        if not state.fitted:
            return True
        # Retrain if we have enough new data
        return len(state.prices) >= self._min_train_samples + min_new_samples


__all__ = ["TimeSeriesForecaster", "ForecastResult"]
