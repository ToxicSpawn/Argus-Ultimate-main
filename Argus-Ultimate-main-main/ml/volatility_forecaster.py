"""
Volatility Forecaster — GARCH-style realized volatility prediction.

Uses EWMA (always available) with optional ARCH/GARCH via arch package.
Outputs annualised volatility forecast + regime classification.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from arch import arch_model  # type: ignore
    _ARCH_AVAILABLE = True
except ImportError:
    _ARCH_AVAILABLE = False
    logger.debug("arch package not available; using EWMA fallback")

# Annualise from 5-minute bars
BARS_PER_YEAR = 252 * 24 * 12  # ~5-min bars


@dataclass
class VolatilityForecast:
    symbol: str
    realized_vol_1d: float        # annualised, last 24h
    realized_vol_7d: float        # annualised, last 7d
    forecast_vol_1d: float        # annualised 1-day ahead
    forecast_vol_5d: float        # annualised 5-day ahead
    regime: str                   # LOW / NORMAL / ELEVATED / EXTREME
    method: str                   # ewma | garch
    confidence: float             # 0-1


@dataclass
class _SymbolState:
    returns: Deque[float] = field(default_factory=lambda: deque(maxlen=2016))  # 7d of 5m bars
    ewma_var: float = 0.0
    last_price: Optional[float] = None
    garch_params: Optional[Tuple[float, float, float]] = None  # omega, alpha, beta


# Volatility regime thresholds (annualised %)
_REGIMES = [
    (30.0, "LOW"),
    (60.0, "NORMAL"),
    (100.0, "ELEVATED"),
    (float("inf"), "EXTREME"),
]


class VolatilityForecaster:
    """
    Multi-symbol realised + GARCH volatility forecaster.

    Usage::

        vf = VolatilityForecaster(lambda_ewma=0.94)
        vf.update("BTC/USD", 50000.0)
        vf.update("BTC/USD", 50200.0)
        forecast = vf.forecast("BTC/USD")
    """

    def __init__(
        self,
        lambda_ewma: float = 0.94,
        min_bars_ewma: int = 30,
        min_bars_garch: int = 250,
        use_garch: bool = True,
    ) -> None:
        self._lambda = lambda_ewma
        self._min_ewma = min_bars_ewma
        self._min_garch = min_bars_garch
        self._use_garch = use_garch and _ARCH_AVAILABLE
        self._states: Dict[str, _SymbolState] = {}

    # ------------------------------------------------------------------
    def update(self, symbol: str, price: float) -> None:
        """Feed a new price tick/bar close."""
        if symbol not in self._states:
            self._states[symbol] = _SymbolState(last_price=price)
            return
        state = self._states[symbol]
        if state.last_price is None or state.last_price <= 0:
            state.last_price = price
            return
        ret = math.log(price / state.last_price)
        state.returns.append(ret)
        state.last_price = price
        # Update EWMA variance
        lam = self._lambda
        state.ewma_var = lam * state.ewma_var + (1 - lam) * ret ** 2

    # ------------------------------------------------------------------
    def forecast(self, symbol: str) -> Optional[VolatilityForecast]:
        """Return volatility forecast for symbol, or None if insufficient data."""
        state = self._states.get(symbol)
        if state is None or len(state.returns) < self._min_ewma:
            return None

        returns = np.array(state.returns)
        bars_1d = 288   # 24h × 12 bars/h
        bars_7d = bars_1d * 7

        # Realised vol windows
        r1d = returns[-min(bars_1d, len(returns)):]
        r7d = returns[-min(bars_7d, len(returns)):]
        rv_1d = float(np.std(r1d) * math.sqrt(BARS_PER_YEAR)) * 100
        rv_7d = float(np.std(r7d) * math.sqrt(BARS_PER_YEAR)) * 100

        method = "ewma"
        forecast_1d = math.sqrt(state.ewma_var * BARS_PER_YEAR) * 100
        forecast_5d = forecast_1d  # EWMA is flat beyond 1 step

        if self._use_garch and len(state.returns) >= self._min_garch:
            try:
                g1d, g5d = self._garch_forecast(symbol, returns)
                forecast_1d = g1d
                forecast_5d = g5d
                method = "garch"
            except Exception as exc:
                logger.warning("GARCH failed for %s: %s — using EWMA", symbol, exc)

        regime = self._classify_regime(forecast_1d)
        confidence = min(1.0, len(state.returns) / 500)

        return VolatilityForecast(
            symbol=symbol,
            realized_vol_1d=rv_1d,
            realized_vol_7d=rv_7d,
            forecast_vol_1d=forecast_1d,
            forecast_vol_5d=forecast_5d,
            regime=regime,
            method=method,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    def realized_vol(self, symbol: str, bars: int = 288) -> Optional[float]:
        """Return simple annualised realised vol over last `bars` bars (%)."""
        state = self._states.get(symbol)
        if not state or len(state.returns) < 2:
            return None
        r = np.array(list(state.returns)[-bars:])
        return float(np.std(r) * math.sqrt(BARS_PER_YEAR)) * 100

    def all_forecasts(self) -> Dict[str, VolatilityForecast]:
        result = {}
        for sym in self._states:
            f = self.forecast(sym)
            if f is not None:
                result[sym] = f
        return result

    # ------------------------------------------------------------------
    def _garch_forecast(
        self, symbol: str, returns: np.ndarray
    ) -> Tuple[float, float]:
        """Fit GARCH(1,1) and return 1-day and 5-day ahead annualised vol (%)."""
        state = self._states[symbol]
        # Scale returns to percent for numerical stability
        r_pct = returns * 100
        am = arch_model(r_pct, vol="Garch", p=1, q=1, rescale=False)
        res = am.fit(disp="off", show_warning=False)
        fc = res.forecast(horizon=5, reindex=False)
        var_1 = float(fc.variance.iloc[-1, 0])
        var_5 = float(fc.variance.iloc[-1, 4])
        # Convert % variance per bar → annualised vol
        vol_1d = math.sqrt(var_1 * BARS_PER_YEAR) / 100 * 100  # stays as %
        vol_5d = math.sqrt(var_5 * BARS_PER_YEAR) / 100 * 100
        # Store fitted params
        params = res.params
        state.garch_params = (
            float(params.get("omega", 0.0)),
            float(params.get("alpha[1]", 0.0)),
            float(params.get("beta[1]", 0.0)),
        )
        return vol_1d, vol_5d

    @staticmethod
    def _classify_regime(annualised_vol_pct: float) -> str:
        for threshold, label in _REGIMES:
            if annualised_vol_pct < threshold:
                return label
        return "EXTREME"
