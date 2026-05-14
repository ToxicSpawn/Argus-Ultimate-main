"""Funding rate predictor for perpetual futures."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FundingPrediction:
    """Funding rate prediction.
    
    Attributes
    ----------
    symbol : str
        Trading pair symbol
    predicted_rate : float
        Predicted funding rate
    confidence : float
        Prediction confidence [0, 1]
    direction : str
        "positive", "negative", or "neutral"
    timestamp : float
        Prediction timestamp
    """
    symbol: str = ""
    predicted_rate: float = 0.0
    confidence: float = 0.0
    direction: str = "neutral"
    timestamp: float = field(default_factory=time.time)


class FundingRatePredictor:
    """Predictor for perpetual futures funding rates.
    
    Uses historical funding rate patterns and market conditions
    to predict future funding rates.
    """
    
    def __init__(self, symbol: str = "BTC/USDT") -> None:
        self.symbol = symbol
        self._history: List[float] = []
        self._max_history = 1000
    
    def update(self, current_rate: float) -> None:
        """Update with current funding rate."""
        self._history.append(current_rate)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
    
    def predict(self) -> FundingPrediction:
        """Predict next funding rate."""
        if len(self._history) < 3:
            return FundingPrediction(symbol=self.symbol)
        
        # Simple moving average prediction
        recent = self._history[-3:]
        predicted = sum(recent) / len(recent)
        
        direction = "neutral"
        if predicted > 0.0001:
            direction = "positive"
        elif predicted < -0.0001:
            direction = "negative"
        
        return FundingPrediction(
            symbol=self.symbol,
            predicted_rate=predicted,
            confidence=min(len(self._history) / 100, 1.0),
            direction=direction,
        )
    
    def get_history(self, count: int = 100) -> List[float]:
        """Get recent funding rate history."""
        return self._history[-count:]
