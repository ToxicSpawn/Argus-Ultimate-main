"""Real-Time Risk Adapter.

Features:
- Dynamic position sizing based on risk
- Real-time VaR calculation
- Adaptive stop losses
- Dynamic leverage adjustment
- Risk per trade optimization
- Drawdown-based position reduction
- Correlation-based hedging
- Stress testing
"""

from __future__ import annotations

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class RiskLimits:
    max_position_size_pct: float = 0.10
    max_portfolio_exposure_pct: float = 0.80
    max_daily_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.15
    max_leverage: int = 1
    max_correlation: float = 0.70
    var_confidence: float = 0.95
    stop_loss_pct: float = 0.02


@dataclass
class RiskState:
    risk_level: RiskLevel
    position_size_mult: float
    leverage_mult: float
    stop_loss_mult: float
    exposure_mult: float
    timestamp: float


class DynamicPositionSizer:
    def __init__(self, initial_capital: float = 10000.0):
        self._initial_capital = initial_capital
        self._current_capital = initial_capital
        self._peak_capital = initial_capital
        self._drawdown_history: deque = deque(maxlen=100)

    def update_capital(self, capital: float) -> None:
        self._current_capital = capital
        if capital > self._peak_capital:
            self._peak_capital = capital

    def get_drawdown(self) -> float:
        if self._peak_capital <= 0:
            return 0.0
        return (self._peak_capital - self._current_capital) / self._peak_capital

    def calculate_position_size(
        self,
        signal_confidence: float,
        stop_loss_pct: float,
        risk_per_trade_pct: float = 0.02,
    ) -> float:
        drawdown = self.get_drawdown()

        dd_adjustment = 1.0 - min(0.7, drawdown * 2)

        kelly_fraction = self._calculate_kelly(signal_confidence)

        adjusted_fraction = kelly_fraction * dd_adjustment

        base_size = self._current_capital * adjusted_fraction

        max_size = self._current_capital * 0.10

        position_size = min(base_size, max_size)

        if stop_loss_pct > 0:
            risk_amount = position_size * stop_loss_pct
            max_risk = self._current_capital * risk_per_trade_pct

            if risk_amount > max_risk:
                position_size = max_risk / stop_loss_pct

        return position_size

    def _calculate_kelly(self, win_rate: float, avg_win: float = 1.5, avg_loss: float = 1.0) -> float:
        if win_rate <= 0 or win_rate >= 1:
            return 0.02

        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        kelly = max(0.01, min(0.25, kelly))

        return kelly * 0.5


class AdaptiveStopLoss:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._atr_multiplier = self.config.get("atr_multiplier", 2.0)
        self._trailing_start_pct = self.config.get("trailing_start", 0.02)

    def calculate_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        atr: Optional[float] = None,
        volatility: Optional[float] = None,
        regime: str = "normal",
    ) -> Tuple[float, bool]:
        is_trailing = False

        if regime == "volatile":
            multiplier = self._ATR_multiplier * 1.5
        elif regime == "calm":
            multiplier = self._ATR_multiplier * 0.7
        else:
            multiplier = self._ATR_multiplier

        if atr and atr > 0:
            stop_distance = atr * multiplier
        elif volatility:
            stop_distance = entry_price * volatility * multiplier
        else:
            stop_distance = entry_price * 0.02

        stop_price = entry_price - stop_distance

        if current_price > entry_price * (1 + self._trailing_start_pct):
            profit_pct = (current_price - entry_price) / entry_price

            trail_distance = stop_distance * (1 + profit_pct * 0.5)
            stop_price = max(stop_price, current_price - trail_distance)
            is_trailing = True

        return stop_price, is_trailing

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        regime: str = "normal",
        risk_reward_ratio: float = 2.0,
    ) -> float:
        risk = abs(entry_price - stop_loss)

        if regime == "trending":
            multiplier = risk_reward_ratio * 1.5
        elif regime == "volatile":
            multiplier = risk_reward_ratio * 0.8
        else:
            multiplier = risk_reward_ratio

        return entry_price + (risk * multiplier)


class RealTimeVaRCalculator:
    def __init__(self, window: int = 252):
        self._window = window
        self._returns: deque = deque(maxlen=window)
        self._positions: Dict[str, float] = {}

    def add_return(self, return_pct: float) -> None:
        self._returns.append(return_pct)

    def update_position(self, symbol: str, value: float) -> None:
        self._positions[symbol] = value

    def calculate_var(
        self,
        confidence: float = 0.95,
    ) -> float:
        if len(self._returns) < 10:
            return 0.0

        returns = sorted(self._returns)
        idx = int((1 - confidence) * len(returns))
        idx = min(max(0, idx), len(returns) - 1)

        return abs(returns[idx])

    def calculate_portfolio_var(
        self,
        confidence: float = 0.95,
    ) -> float:
        if not self._positions:
            return 0.0

        total_value = sum(self._positions.values())
        if total_value <= 0:
            return 0.0

        position_weights = {
            sym: val / total_value
            for sym, val in self._positions.items()
        }

        var = 0.0
        for sym, weight in position_weights.items():
            sym_var = self.calculate_var(confidence)
            var += (weight * sym_var) ** 2

        return np.sqrt(var) * total_value


class CorrelationManager:
    def __init__(self, window: int = 50):
        self._window = window
        self._prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._correlations: Dict[Tuple[str, str], float] = {}

    def add_price(self, symbol: str, price: float) -> None:
        self._prices[symbol].append(price)

    def calculate_correlation(
        self,
        symbol_a: str,
        symbol_b: str,
    ) -> float:
        if symbol_a not in self._prices or symbol_b not in self._prices:
            return 0.0

        if len(self._prices[symbol_a]) < 10:
            return 0.0

        prices_a = np.array(list(self._prices[symbol_a]))
        prices_b = np.array(list(self._prices[symbol_b]))

        if len(prices_a) != len(prices_b):
            return 0.0

        returns_a = np.diff(prices_a) / prices_a[:-1]
        returns_b = np.diff(prices_b) / prices_b[:-1]

        if len(returns_a) < 2:
            return 0.0

        corr = np.corrcoef(returns_a, returns_b)[0, 1]

        if np.isnan(corr):
            return 0.0

        self._correlations[(symbol_a, symbol_b)] = corr

        return corr

    def get_high_correlations(
        self,
        threshold: float = 0.7,
    ) -> List[Tuple[str, str, float]]:
        high_corr = []

        for sym_a in self._prices.keys():
            for sym_b in self._prices.keys():
                if sym_a >= sym_b:
                    continue

                corr = self.calculate_correlation(sym_a, sym_b)

                if abs(corr) >= threshold:
                    high_corr.append((sym_a, sym_b, corr))

        return high_corr


class StressTester:
    def __init__(self):
        self._scenarios = {
            "market_crash": -0.20,
            "flash_crash": -0.10,
            "volatility_spike": 0.50,
            "liquidity_crisis": -0.15,
            "correlation_breakdown": 0.0,
        }

    def stress_test_portfolio(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
    ) -> Dict[str, float]:
        results = {}

        for scenario, shock in self._scenarios.items():
            total_value = 0.0

            for symbol, qty in positions.items():
                price = prices.get(symbol, 0.0)
                position_value = qty * price
                stressed_value = position_value * (1 + shock)
                total_value += stressed_value

            results[scenario] = total_value

        return results

    def calculate_max_loss(
        self,
        stress_results: Dict[str, float],
    ) -> float:
        return min(stress_results.values())


class RealTimeRiskAdapter:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        self._limits = RiskLimits()

        self._position_sizer = DynamicPositionSizer(
            initial_capital=self.config.get("initial_capital", 10000.0)
        )
        self._stop_loss = AdaptiveStopLoss(config)
        self._var_calc = RealTimeVaRCalculator()
        self._correlation = CorrelationManager()
        self._stress = StressTester()

        self._risk_state = RiskState(
            risk_level=RiskLevel.MODERATE,
            position_size_mult=1.0,
            leverage_mult=1.0,
            stop_loss_mult=1.0,
            exposure_mult=1.0,
            timestamp=time.time(),
        )

        self._risk_history: deque = deque(maxlen=1000)

    def update_capital(self, capital: float) -> None:
        self._position_sizer.update_capital(capital)

    def calculate_position_size(
        self,
        signal_confidence: float,
        stop_loss_pct: float,
        regime: str = "normal",
    ) -> float:
        base_size = self._position_sizer.calculate_position_size(
            signal_confidence, stop_loss_pct
        )

        adjusted_size = base_size * self._risk_state.position_size_mult

        max_exposure = self._position_sizer._current_capital * self._limits.max_position_size_pct

        return min(adjusted_size, max_exposure)

    def calculate_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        atr: Optional[float] = None,
        regime: str = "normal",
    ) -> Tuple[float, bool]:
        stop, is_trailing = self._stop_loss.calculate_stop_loss(
            entry_price, current_price, atr, regime=regime
        )

        adjusted_stop = stop * self._risk_state.stop_loss_mult

        return adjusted_stop, is_trailing

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        regime: str = "normal",
    ) -> float:
        return self._stop_loss.calculate_take_profit(
            entry_price, stop_loss, regime
        )

    def check_risk_limits(
        self,
        symbol: str,
        position_value: float,
        total_exposure: float,
    ) -> Tuple[bool, str]:
        max_position = self._position_sizer._current_capital * self._limits.max_position_size_pct

        if position_value > max_position:
            return False, f"Position size exceeds limit: {position_value} > {max_position}"

        max_exposure = self._position_sizer._current_capital * self._limits.max_portfolio_exposure_pct

        if total_exposure > max_exposure:
            return False, f"Total exposure exceeds limit: {total_exposure} > {max_exposure}"

        drawdown = self._position_sizer.get_drawdown()

        if drawdown > self._limits.max_drawdown_pct:
            return False, f"Drawdown exceeds limit: {drawdown:.2%}"

        var = self._var_calc.calculate_portfolio_var(self._limits.var_confidence)

        max_var = self._position_sizer._current_capital * 0.05

        if var > max_var:
            return False, f"VaR exceeds limit: {var} > {max_var}"

        return True, "OK"

    def adapt_to_market(
        self,
        regime: str,
        volatility: float,
        drawdown: float,
    ) -> RiskState:
        position_mult = 1.0
        leverage_mult = 1.0
        stop_mult = 1.0
        exposure_mult = 1.0

        if regime == "volatile":
            position_mult = 0.5
            leverage_mult = 0.5
            stop_mult = 1.5
            exposure_mult = 0.6
            risk_level = RiskLevel.HIGH
        elif regime == "calm":
            position_mult = 1.2
            leverage_mult = 1.2
            stop_mult = 0.8
            exposure_mult = 1.1
            risk_level = RiskLevel.LOW
        elif regime == "trending_up" or regime == "trending_down":
            position_mult = 1.1
            leverage_mult = 1.0
            stop_mult = 1.0
            exposure_mult = 1.0
            risk_level = RiskLevel.MODERATE
        elif regime == "ranging":
            position_mult = 0.7
            leverage_mult = 0.7
            stop_mult = 0.7
            exposure_mult = 0.7
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.MODERATE

        if volatility > 1.0:
            position_mult *= 0.7
            exposure_mult *= 0.7
            risk_level = RiskLevel.HIGH
        elif volatility < 0.3:
            position_mult *= 1.1

        if drawdown > 0.10:
            reduction = min(0.8, drawdown * 3)
            position_mult *= (1 - reduction)
            exposure_mult *= (1 - reduction)
            risk_level = RiskLevel.HIGH
        elif drawdown > 0.05:
            position_mult *= 0.8
            exposure_mult *= 0.8
            risk_level = RiskLevel.MODERATE

        self._risk_state = RiskState(
            risk_level=risk_level,
            position_size_mult=position_mult,
            leverage_mult=leverage_mult,
            stop_loss_mult=stop_mult,
            exposure_mult=exposure_mult,
            timestamp=time.time(),
        )

        self._risk_history.append(self._risk_state)

        return self._risk_state

    def stress_test(self, positions: Dict[str, float]) -> Dict[str, float]:
        prices = {sym: 1.0 for sym in positions.keys()}
        return self._stress.stress_test_portfolio(positions, prices)

    def get_risk_state(self) -> RiskState:
        return self._risk_state

    def get_var(self, confidence: float = 0.95) -> float:
        return self._var_calc.calculate_portfolio_var(confidence)

    def update_correlation(self, symbol: str, price: float) -> None:
        self._correlation.add_price(symbol, price)

    def get_high_correlations(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        return self._correlation.get_high_correlations(threshold)

    def set_limits(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self._limits, key):
                setattr(self._limits, key, value)
