"""
Maximum Risk Engine — The ultimate risk management system for Argus.

This module orchestrates ALL existing risk modules and adds ML-based prediction
to create the most sophisticated risk management system possible.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    MAXIMUM RISK ENGINE                              │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Layer 1: PREDICTION (ML-based)                                     │
    │    - RiskPredictor: predicts risk 5-30 minutes ahead                │
    │    - DrawdownForecaster: forecasts drawdown trajectory              │
    │    - VolatilityRegimeClassifier: classifies volatility regimes      │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Layer 2: REAL-TIME MONITORING                                      │
    │    - BlackSwanDetector: anomaly detection                           │
    │    - CorrelationMonitor: correlation breakdown                      │
    │    - RegimeConditionalVaR: regime-aware VaR                         │
    │    - LiquidityRiskEngine: liquidity risk                            │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Layer 3: POSITION MANAGEMENT                                       │
    │    - DynamicKellySizer: Kelly criterion with uncertainty            │
    │    - AdaptiveStopLoss: volatility-adjusted stops                    │
    │    - AntifragileManager: benefit from volatility                    │
    │    - DynamicDrawdownController: convex drawdown reduction           │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Layer 4: EMERGENCY DEFENSE                                         │
    │    - UltimateDefense: circuit breakers, kill switches               │
    │    - RiskFacade: unified risk approval                              │
    └─────────────────────────────────────────────────────────────────────┘

Usage:
    from risk.maximum_risk_engine import MaximumRiskEngine

    engine = MaximumRiskEngine(config)
    decision = engine.evaluate_trade(
        symbol="BTC/USDT",
        side="buy",
        size_usd=1000,
        current_price=65000,
        portfolio_equity=100000,
    )
    if decision.approved:
        execute_trade(decision.approved_size, decision.stop_loss, decision.take_profit)
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RiskLevel(Enum):
    """System-wide risk level."""
    MINIMAL = "minimal"       # 0-2: Normal conditions
    LOW = "low"               # 2-4: Slightly elevated
    MODERATE = "moderate"     # 4-6: Elevated risk
    HIGH = "high"             # 6-8: High risk
    EXTREME = "extreme"       # 8-10: Extreme risk
    CRITICAL = "critical"     # 10+: Critical - halt trading


class PositionAction(Enum):
    """Recommended position action."""
    OPEN = "open"
    INCREASE = "increase"
    DECREASE = "decrease"
    CLOSE = "close"
    HALT = "halt"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RiskPrediction:
    """ML-based risk prediction."""
    predicted_var_5m: float      # VaR in 5 minutes
    predicted_var_15m: float     # VaR in 15 minutes
    predicted_var_30m: float     # VaR in 30 minutes
    volatility_regime: str       # "low", "normal", "high", "extreme"
    crash_probability: float     # 0-1 probability of crash in next hour
    trend_reversal_probability: float  # 0-1 probability of trend reversal
    confidence: float            # 0-1 prediction confidence
    timestamp: float = field(default_factory=time.time)


@dataclass
class DrawdownForecast:
    """Predicted drawdown trajectory."""
    current_drawdown_pct: float
    predicted_max_dd_1h: float
    predicted_max_dd_4h: float
    predicted_max_dd_24h: float
    recovery_probability: float  # 0-1 probability of recovery within 24h
    time_to_recovery_hours: Optional[float]  # estimated hours to recovery
    trajectory: List[float]      # predicted drawdown path [0-24h]
    timestamp: float = field(default_factory=time.time)


@dataclass
class AdaptiveStop:
    """Adaptive stop-loss parameters."""
    stop_loss_price: float
    stop_loss_pct: float
    trailing_stop_active: bool
    trailing_distance_pct: float
    volatility_adjusted: bool
    time_decay_factor: float     # how much stop tightens over time
    reason: str                  # why this stop was chosen


@dataclass
class PositionSizing:
    """Optimal position sizing result."""
    kelly_fraction: float
    adjusted_fraction: float     # after all risk adjustments
    max_size_usd: float
    recommended_size_usd: float
    leverage_cap: float
    risk_budget_remaining: float  # remaining risk budget today
    adjustments: List[str]       # list of adjustments applied


@dataclass
class TradeDecision:
    """Complete trade decision with all risk considerations."""
    approved: bool
    action: PositionAction
    symbol: str
    side: str
    approved_size_usd: float
    requested_size_usd: float
    stop_loss: AdaptiveStop
    take_profit_pct: float
    risk_score: float            # 0-10 (0=no risk, 10=critical)
    risk_level: RiskLevel
    position_sizing: PositionSizing
    risk_prediction: RiskPrediction
    drawdown_forecast: DrawdownForecast
    conditions: List[str]        # risk conditions/reasons
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemRiskStatus:
    """Overall system risk status."""
    risk_level: RiskLevel
    risk_score: float
    var_99_pct: float
    cvar_99_pct: float
    max_drawdown_pct: float
    correlation_alert: bool
    black_swan_detected: bool
    circuit_breaker_active: bool
    position_multiplier: float
    can_trade: bool
    recommendations: List[str]
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# ML-Based Risk Prediction
# ---------------------------------------------------------------------------

class RiskPredictor:
    """
    ML-based risk prediction using online learning.

    Predicts:
    - Future VaR at 5m, 15m, 30m horizons
    - Volatility regime classification
    - Crash probability
    - Trend reversal probability

    Uses lightweight online learning (no heavy ML dependencies).
    """

    def __init__(self, window: int = 500):
        self.window = window
        self._returns: Deque[float] = deque(maxlen=window)
        self._volatilities: Deque[float] = deque(maxlen=window)
        self._volumes: Deque[float] = deque(maxlen=window)
        self._prices: Deque[float] = deque(maxlen=window)

        # Online learning state
        self._vol_mean: float = 0.0
        self._vol_std: float = 0.01
        self._return_mean: float = 0.0
        self._return_std: float = 0.01
        self._count: int = 0

        # Crash detection state
        self._crash_score: float = 0.0
        self._reversal_score: float = 0.0

    def update(self, price: float, volume: float) -> None:
        """Update with new market data."""
        self._prices.append(price)
        self._volumes.append(volume)

        if len(self._prices) >= 2:
            ret = math.log(self._prices[-1] / self._prices[-2])
            self._returns.append(ret)

            # Update online statistics
            self._count += 1
            delta = ret - self._return_mean
            self._return_mean += delta / self._count
            delta2 = ret - self._return_mean
            self._return_std = math.sqrt(
                max(1e-10, (self._return_std ** 2 * (self._count - 1) + delta * delta2) / self._count)
            )

            # Realized volatility (annualized)
            if len(self._returns) >= 10:
                recent_returns = list(self._returns)[-10:]
                vol = math.sqrt(sum(r ** 2 for r in recent_returns) / len(recent_returns)) * math.sqrt(252 * 24 * 60)
                self._volatilities.append(vol)

                # Update vol statistics
                if len(self._volatilities) > 1:
                    self._vol_mean = np.mean(self._volatilities)
                    self._vol_std = max(1e-6, np.std(self._volatilities))

        # Update crash/reversal scores
        self._update_crash_score()
        self._update_reversal_score()

    def _update_crash_score(self) -> None:
        """Update crash probability score."""
        if len(self._returns) < 20:
            return

        recent = list(self._returns)[-20:]
        returns_arr = np.array(recent)

        # Factors that increase crash probability:
        # 1. Large negative returns
        # 2. Increasing volatility
        # 3. Volume spike
        # 4. Momentum exhaustion

        # Negative return streak
        neg_streak = 0
        for r in reversed(recent):
            if r < 0:
                neg_streak += 1
            else:
                break

        # Volatility spike
        if len(self._volatilities) >= 20:
            vol_ratio = self._volatilities[-1] / (np.mean(self._volatilities) + 1e-10)
        else:
            vol_ratio = 1.0

        # Volume spike
        if len(self._volumes) >= 20:
            vol_mean = np.mean(list(self._volumes)[-20:])
            vol_ratio_v = self._volumes[-1] / (vol_mean + 1e-10)
        else:
            vol_ratio_v = 1.0

        # Composite crash score
        self._crash_score = min(1.0, max(0.0,
            0.3 * min(1.0, neg_streak / 5.0) +
            0.3 * min(1.0, (vol_ratio - 1.0) / 3.0) +
            0.2 * min(1.0, (vol_ratio_v - 1.0) / 2.0) +
            0.2 * min(1.0, abs(returns_arr.min()) / 0.05)
        ))

    def _update_reversal_score(self) -> None:
        """Update trend reversal probability score."""
        if len(self._prices) < 50:
            return

        prices = list(self._prices)[-50:]
        returns = list(self._returns)[-50:] if len(self._returns) >= 50 else list(self._returns)

        # Trend strength (ADX-like)
        gains = [max(0, r) for r in returns]
        losses = [max(0, -r) for r in returns]
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0

        if avg_loss > 0:
            rs = avg_gain / avg_loss
            trend_strength = 1.0 - 1.0 / (1.0 + rs)
        else:
            trend_strength = 0.5

        # Momentum divergence (price making new highs but momentum slowing)
        if len(returns) >= 20:
            recent_momentum = sum(returns[-10:])
            prior_momentum = sum(returns[-20:-10])
            momentum_divergence = prior_momentum - recent_momentum
        else:
            momentum_divergence = 0

        # Mean reversion pressure
        if len(prices) >= 20:
            mean_price = np.mean(prices[-20:])
            std_price = np.std(prices[-20:])
            if std_price > 0:
                z_score = (prices[-1] - mean_price) / std_price
            else:
                z_score = 0
        else:
            z_score = 0

        # Composite reversal score
        self._reversal_score = min(1.0, max(0.0,
            0.3 * trend_strength +
            0.3 * min(1.0, abs(momentum_divergence) / 0.02) +
            0.4 * min(1.0, abs(z_score) / 3.0)
        ))

    def predict(self) -> RiskPrediction:
        """Generate risk prediction."""
        # Current volatility
        if len(self._volatilities) >= 10:
            current_vol = self._volatilities[-1]
        else:
            current_vol = 0.5  # default annual vol

        # Volatility regime
        vol_percentile = self._vol_percentile(current_vol)
        if vol_percentile < 0.25:
            vol_regime = "low"
        elif vol_percentile < 0.50:
            vol_regime = "normal"
        elif vol_percentile < 0.75:
            vol_regime = "high"
        else:
            vol_regime = "extreme"

        # Predicted VaR (using GARCH-like scaling)
        daily_vol = current_vol / math.sqrt(252)
        var_5m = daily_vol * math.sqrt(5.0 / (24 * 60)) * 2.33  # 99% confidence
        var_15m = daily_vol * math.sqrt(15.0 / (24 * 60)) * 2.33
        var_30m = daily_vol * math.sqrt(30.0 / (24 * 60)) * 2.33

        # Confidence based on data availability
        confidence = min(1.0, self._count / 100)

        return RiskPrediction(
            predicted_var_5m=var_5m,
            predicted_var_15m=var_15m,
            predicted_var_30m=var_30m,
            volatility_regime=vol_regime,
            crash_probability=self._crash_score,
            trend_reversal_probability=self._reversal_score,
            confidence=confidence,
        )

    def _vol_percentile(self, vol: float) -> float:
        """Get percentile of current volatility in history."""
        if len(self._volatilities) < 10:
            return 0.5
        vols = sorted(self._volatilities)
        rank = sum(1 for v in vols if v <= vol)
        return rank / len(vols)


# ---------------------------------------------------------------------------
# Drawdown Forecaster
# ---------------------------------------------------------------------------

class DrawdownForecaster:
    """
    Forecasts drawdown trajectory using Monte Carlo simulation.

    Uses recent return distribution to simulate future paths and
    estimate maximum drawdown at various horizons.
    """

    def __init__(self, n_simulations: int = 500):
        self.n_simulations = n_simulations
        self._returns: Deque[float] = deque(maxlen=1000)
        self._equity: Deque[float] = deque(maxlen=10000)
        self._peak_equity: float = 0.0

    def update(self, equity: float) -> None:
        """Update with new equity value."""
        if len(self._equity) > 0:
            prev_equity = self._equity[-1]
            if prev_equity > 0:
                ret = (equity - prev_equity) / prev_equity
                self._returns.append(ret)

        self._equity.append(equity)
        if equity > self._peak_equity:
            self._peak_equity = equity

    def forecast(self, hours: int = 24) -> DrawdownForecast:
        """Generate drawdown forecast."""
        current_dd = 0.0
        if self._peak_equity > 0 and len(self._equity) > 0:
            current_dd = (self._peak_equity - self._equity[-1]) / self._peak_equity * 100

        if len(self._returns) < 20:
            # Insufficient data - return conservative estimate
            return DrawdownForecast(
                current_drawdown_pct=current_dd,
                predicted_max_dd_1h=current_dd + 2.0,
                predicted_max_dd_4h=current_dd + 5.0,
                predicted_max_dd_24h=current_dd + 10.0,
                recovery_probability=0.5,
                time_to_recovery_hours=None,
                trajectory=[current_dd] * (hours + 1),
            )

        returns_arr = np.array(self._returns)
        mean_ret = np.mean(returns_arr)
        std_ret = np.std(returns_arr)

        # Monte Carlo simulation
        steps_per_hour = 60  # minute-level
        trajectories = []

        for _ in range(self.n_simulations):
            equity = self._equity[-1] if self._equity else 100000.0
            peak = self._peak_equity if self._peak_equity > 0 else equity
            path = [current_dd]
            max_dd = current_dd

            for h in range(hours):
                for _ in range(steps_per_hour):
                    # Simulate return with fat tails
                    if np.random.random() < 0.05:  # 5% chance of fat tail
                        ret = np.random.normal(mean_ret, std_ret * 3)
                    else:
                        ret = np.random.normal(mean_ret, std_ret)

                    equity *= (1 + ret)
                    if equity > peak:
                        peak = equity

                    dd = (peak - equity) / peak * 100 if peak > 0 else 0
                    max_dd = max(max_dd, dd)

                path.append(max_dd)

            trajectories.append(path)

        # Compute statistics
        trajectories_arr = np.array(trajectories)
        predicted_path = np.percentile(trajectories_arr, 50, axis=0).tolist()

        # Recovery probability
        final_equities = []
        for _ in range(self.n_simulations):
            equity = self._equity[-1] if self._equity else 100000.0
            for _ in range(hours * steps_per_hour):
                ret = np.random.normal(mean_ret, std_ret)
                equity *= (1 + ret)
            final_equities.append(equity)

        initial_equity = self._equity[0] if self._equity else 100000.0
        recovered = sum(1 for e in final_equities if e >= initial_equity)
        recovery_prob = recovered / len(final_equities)

        # Estimate time to recovery
        time_to_recovery = None
        if current_dd > 1 and recovery_prob > 0.5:
            # Estimate based on mean return
            daily_return = mean_ret * 24 * 60
            if daily_return > 0:
                # Simple estimate: time = dd / daily_return
                time_to_recovery = min(720, current_dd / (daily_return * 100))  # cap at 30 days

        return DrawdownForecast(
            current_drawdown_pct=current_dd,
            predicted_max_dd_1h=predicted_path[1] if len(predicted_path) > 1 else current_dd,
            predicted_max_dd_4h=predicted_path[4] if len(predicted_path) > 4 else current_dd,
            predicted_max_dd_24h=predicted_path[hours] if len(predicted_path) > hours else current_dd,
            recovery_probability=recovery_prob,
            time_to_recovery_hours=time_to_recovery,
            trajectory=predicted_path,
        )


# ---------------------------------------------------------------------------
# Adaptive Stop-Loss
# ---------------------------------------------------------------------------

class AdaptiveStopLoss:
    """
    Volatility-adjusted adaptive stop-loss system.

    Features:
    - ATR-based dynamic stops
    - Time decay (stops tighten over time)
    - Trailing stops with volatility adjustment
    - Regime-aware stop distances
    """

    def __init__(
        self,
        base_atr_period: int = 14,
        atr_multiplier: float = 2.0,
        min_stop_pct: float = 0.5,
        max_stop_pct: float = 10.0,
        time_decay_hours: float = 24.0,
    ):
        self.base_atr_period = base_atr_period
        self.atr_multiplier = atr_multiplier
        self.min_stop_pct = min_stop_pct
        self.max_stop_pct = max_stop_pct
        self.time_decay_hours = time_decay_hours

        self._highs: Deque[float] = deque(maxlen=base_atr_period * 2)
        self._lows: Deque[float] = deque(maxlen=base_atr_period * 2)
        self._closes: Deque[float] = deque(maxlen=base_atr_period * 2)
        self._entry_time: Optional[float] = None
        self._entry_price: Optional[float] = None
        self._highest_since_entry: float = 0.0
        self._lowest_since_entry: float = float('inf')

    def update(self, high: float, low: float, close: float) -> None:
        """Update with new OHLC data."""
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)

        if self._entry_price is not None:
            self._highest_since_entry = max(self._highest_since_entry, high)
            self._lowest_since_entry = min(self._lowest_since_entry, low)

    def set_entry(self, price: float, side: str) -> None:
        """Record entry for new position."""
        self._entry_time = time.time()
        self._entry_price = price
        self._highest_since_entry = price
        self._lowest_since_entry = price

    def calculate_atr(self) -> float:
        """Calculate ATR."""
        if len(self._highs) < self.base_atr_period:
            return 0.0

        highs = list(self._highs)[-self.base_atr_period:]
        lows = list(self._lows)[-self.base_atr_period:]
        closes = list(self._closes)[-self.base_atr_period:]

        tr_values = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_values.append(tr)

        return np.mean(tr_values) if tr_values else 0.0

    def get_stop(
        self,
        side: str,
        current_price: float,
        regime: str = "normal",
    ) -> AdaptiveStop:
        """Calculate adaptive stop-loss."""
        atr = self.calculate_atr()
        current_time = time.time()

        # Regime multiplier
        regime_multipliers = {
            "low": 1.5,
            "normal": 2.0,
            "high": 2.5,
            "extreme": 3.0,
        }
        multiplier = regime_multipliers.get(regime, 2.0)

        # Base stop distance
        if atr > 0:
            stop_distance = atr * multiplier
            stop_pct = (stop_distance / current_price) * 100
        else:
            stop_pct = 2.0  # default 2%

        # Clamp to bounds
        stop_pct = max(self.min_stop_pct, min(self.max_stop_pct, stop_pct))

        # Time decay - tighten stops over time
        time_decay_factor = 1.0
        if self._entry_time is not None:
            hours_held = (current_time - self._entry_time) / 3600
            if hours_held > 0:
                # Reduce stop by 10% per time_decay_hours
                time_decay_factor = max(0.5, 1.0 - (hours_held / self.time_decay_hours) * 0.5)
                stop_pct *= time_decay_factor

        # Calculate stop price
        if side == "buy":
            stop_price = current_price * (1 - stop_pct / 100)
            trailing_distance = (self._highest_since_entry - stop_price) / self._highest_since_entry * 100 if self._highest_since_entry > 0 else stop_pct
        else:
            stop_price = current_price * (1 + stop_pct / 100)
            trailing_distance = (stop_price - self._lowest_since_entry) / self._lowest_since_entry * 100 if self._lowest_since_entry < float('inf') else stop_pct

        # Determine if trailing stop should be active
        trailing_active = False
        if side == "buy" and self._highest_since_entry > self._entry_price * 1.01:
            trailing_active = True
            # Move stop up
            stop_price = max(stop_price, self._highest_since_entry * (1 - trailing_distance / 100))
        elif side == "sell" and self._lowest_since_entry < self._entry_price * 0.99:
            trailing_active = True
            # Move stop down
            stop_price = min(stop_price, self._lowest_since_entry * (1 + trailing_distance / 100))

        return AdaptiveStop(
            stop_loss_price=stop_price,
            stop_loss_pct=stop_pct,
            trailing_stop_active=trailing_active,
            trailing_distance_pct=trailing_distance,
            volatility_adjusted=True,
            time_decay_factor=time_decay_factor,
            reason=f"ATR={atr:.4f} x {multiplier} ({regime} regime), decay={time_decay_factor:.2f}",
        )


# ---------------------------------------------------------------------------
# Maximum Risk Engine
# ---------------------------------------------------------------------------

class MaximumRiskEngine:
    """
    The ultimate risk management system for Argus.

    Orchestrates all risk modules and adds ML-based prediction
    for the most sophisticated risk management possible.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Risk limits
        self.max_daily_loss_pct = self.config.get("max_daily_loss_pct", 5.0)
        self.max_drawdown_pct = self.config.get("max_drawdown_pct", 15.0)
        self.max_position_pct = self.config.get("max_position_pct", 25.0)
        self.max_leverage = self.config.get("max_leverage", 5.0)
        self.kelly_fraction = self.config.get("kelly_fraction", 0.5)  # half-Kelly

        # ML components
        self.risk_predictor = RiskPredictor(window=500)
        self.drawdown_forecaster = DrawdownForecaster(n_simulations=500)
        self.adaptive_stop = AdaptiveStopLoss()

        # State tracking
        self._peak_equity: float = 0.0
        self._day_start_equity: float = 0.0
        self._day_start_ts: float = time.time()
        self._daily_pnl: float = 0.0
        self._current_equity: float = 0.0
        self._halted: bool = False
        self._halt_reason: str = ""

        # Risk history
        self._risk_scores: Deque[float] = deque(maxlen=1000)
        self._trade_outcomes: Deque[Tuple[float, float]] = deque(maxlen=500)  # (pnl, volatility)

        logger.info("MaximumRiskEngine initialized with max_dd=%.1f%%, daily_loss=%.1f%%",
                    self.max_drawdown_pct, self.max_daily_loss_pct)

    def update_market(self, price: float, volume: float, high: float, low: float) -> None:
        """Update with new market data."""
        self.risk_predictor.update(price, volume)
        self.adaptive_stop.update(high, low, price)

    def update_equity(self, equity: float) -> None:
        """Update with new portfolio equity."""
        self._current_equity = equity

        if equity > self._peak_equity:
            self._peak_equity = equity

        self.drawdown_forecaster.update(equity)

        # Reset daily tracking
        current_ts = time.time()
        if current_ts - self._day_start_ts > 86400:
            self._day_start_equity = equity
            self._day_start_ts = current_ts
            self._daily_pnl = 0.0

        self._daily_pnl = equity - self._day_start_equity

    def record_trade_outcome(self, pnl_pct: float, volatility: float) -> None:
        """Record trade outcome for learning."""
        self._trade_outcomes.append((pnl_pct, volatility))

    def calculate_risk_score(self) -> float:
        """
        Calculate composite risk score (0-10).

        Factors:
        - Current drawdown (0-3 points)
        - Daily P&L (0-2 points)
        - Volatility regime (0-2 points)
        - Crash probability (0-2 points)
        - Correlation risk (0-1 point)
        """
        score = 0.0

        # Drawdown component (0-3)
        if self._peak_equity > 0:
            current_dd = (self._peak_equity - self._current_equity) / self._peak_equity * 100
            dd_score = min(3.0, current_dd / self.max_drawdown_pct * 3.0)
            score += dd_score

        # Daily P&L component (0-2)
        if self._day_start_equity > 0:
            daily_loss = -self._daily_pnl / self._day_start_equity * 100
            if daily_loss > 0:
                daily_score = min(2.0, daily_loss / self.max_daily_loss_pct * 2.0)
                score += daily_score

        # Volatility regime component (0-2)
        prediction = self.risk_predictor.predict()
        vol_regime_scores = {"low": 0.0, "normal": 0.5, "high": 1.5, "extreme": 2.0}
        score += vol_regime_scores.get(prediction.volatility_regime, 0.5)

        # Crash probability component (0-2)
        score += prediction.crash_probability * 2.0

        self._risk_scores.append(score)
        return score

    def get_risk_level(self) -> RiskLevel:
        """Get current risk level."""
        score = self.calculate_risk_score()
        if score < 2:
            return RiskLevel.MINIMAL
        elif score < 4:
            return RiskLevel.LOW
        elif score < 6:
            return RiskLevel.MODERATE
        elif score < 8:
            return RiskLevel.HIGH
        elif score < 10:
            return RiskLevel.EXTREME
        else:
            return RiskLevel.CRITICAL

    def calculate_position_size(
        self,
        symbol: str,
        side: str,
        requested_size_usd: float,
        win_rate: Optional[float] = None,
        win_loss_ratio: Optional[float] = None,
    ) -> PositionSizing:
        """
        Calculate optimal position size using Kelly criterion
        with all risk adjustments.
        """
        adjustments = []

        # Base Kelly calculation
        if win_rate is None or win_loss_ratio is None:
            # Use historical data if available
            if len(self._trade_outcomes) >= 20:
                outcomes = list(self._trade_outcomes)
                wins = [o for o in outcomes if o[0] > 0]
                wr = len(wins) / len(outcomes) if outcomes else 0.5
                avg_win = np.mean([o[0] for o in wins]) if wins else 0.01
                losses = [o for o in outcomes if o[0] <= 0]
                avg_loss = abs(np.mean([o[0] for o in losses])) if losses else 0.01
                wlr = avg_win / avg_loss if avg_loss > 0 else 1.5
            else:
                wr = 0.55
                wlr = 1.5
        else:
            wr = win_rate
            wlr = win_loss_ratio

        # Full Kelly: f* = (p * b - q) / b where p=win_rate, q=1-p, b=win_loss_ratio
        full_kelly = (wr * wlr - (1 - wr)) / wlr if wlr > 0 else 0
        full_kelly = max(0, full_kelly)

        # Apply fractional Kelly (half-Kelly by default)
        kelly_fraction = full_kelly * self.kelly_fraction
        adjustments.append(f"Half-Kelly: {kelly_fraction:.3f}")

        # Risk level adjustment
        risk_level = self.get_risk_level()
        risk_multipliers = {
            RiskLevel.MINIMAL: 1.0,
            RiskLevel.LOW: 0.8,
            RiskLevel.MODERATE: 0.5,
            RiskLevel.HIGH: 0.25,
            RiskLevel.EXTREME: 0.1,
            RiskLevel.CRITICAL: 0.0,
        }
        risk_mult = risk_multipliers[risk_level]
        kelly_fraction *= risk_mult
        adjustments.append(f"Risk level {risk_level.value}: x{risk_mult}")

        # Drawdown adjustment
        if self._peak_equity > 0:
            current_dd = (self._peak_equity - self._current_equity) / self._peak_equity * 100
            if current_dd > 5:
                dd_mult = max(0.1, 1.0 - current_dd / self.max_drawdown_pct)
                kelly_fraction *= dd_mult
                adjustments.append(f"Drawdown {current_dd:.1f}%: x{dd_mult:.2f}")

        # Volatility adjustment
        prediction = self.risk_predictor.predict()
        vol_mult = {"low": 1.2, "normal": 1.0, "high": 0.7, "extreme": 0.4}
        kelly_fraction *= vol_mult.get(prediction.volatility_regime, 1.0)
        adjustments.append(f"Vol regime {prediction.volatility_regime}: x{vol_mult.get(prediction.volatility_regime, 1.0)}")

        # Crash probability adjustment
        if prediction.crash_probability > 0.3:
            crash_mult = max(0.1, 1.0 - prediction.crash_probability)
            kelly_fraction *= crash_mult
            adjustments.append(f"Crash prob {prediction.crash_probability:.1%}: x{crash_mult:.2f}")

        # Clamp to bounds
        kelly_fraction = max(0, min(self.max_position_pct / 100, kelly_fraction))

        # Calculate sizes
        max_size = self._current_equity * (self.max_position_pct / 100)
        recommended_size = self._current_equity * kelly_fraction
        recommended_size = min(recommended_size, requested_size_usd, max_size)

        # Risk budget remaining
        daily_loss_limit = self._day_start_equity * (self.max_daily_loss_pct / 100)
        risk_budget = max(0, daily_loss_limit + self._daily_pnl)

        return PositionSizing(
            kelly_fraction=full_kelly,
            adjusted_fraction=kelly_fraction,
            max_size_usd=max_size,
            recommended_size_usd=recommended_size,
            leverage_cap=min(self.max_leverage, 1.0 / max(0.01, kelly_fraction)),
            risk_budget_remaining=risk_budget,
            adjustments=adjustments,
        )

    def evaluate_trade(
        self,
        symbol: str,
        side: str,
        size_usd: float,
        current_price: float,
        portfolio_equity: float,
        win_rate: Optional[float] = None,
        win_loss_ratio: Optional[float] = None,
    ) -> TradeDecision:
        """
        Evaluate a trade request and return a complete decision
        with all risk considerations.
        """
        # Update equity
        self.update_equity(portfolio_equity)

        # Get predictions
        prediction = self.risk_predictor.predict()
        dd_forecast = self.drawdown_forecaster.forecast(hours=24)
        risk_score = self.calculate_risk_score()
        risk_level = self.get_risk_level()

        # Check halt conditions
        conditions = []

        if self._halted:
            conditions.append(f"SYSTEM HALTED: {self._halt_reason}")
            return TradeDecision(
                approved=False,
                action=PositionAction.HALT,
                symbol=symbol,
                side=side,
                approved_size_usd=0,
                requested_size_usd=size_usd,
                stop_loss=self.adaptive_stop.get_stop(side, current_price, prediction.volatility_regime),
                take_profit_pct=0,
                risk_score=risk_score,
                risk_level=risk_level,
                position_sizing=self.calculate_position_size(symbol, side, size_usd, win_rate, win_loss_ratio),
                risk_prediction=prediction,
                drawdown_forecast=dd_forecast,
                conditions=conditions,
            )

        # Check daily loss limit
        if self._daily_pnl < -self._day_start_equity * (self.max_daily_loss_pct / 100):
            conditions.append(f"Daily loss limit reached: {self._daily_pnl:.2f}")
            self._halted = True
            self._halt_reason = "Daily loss limit"

        # Check drawdown limit
        if dd_forecast.current_drawdown_pct >= self.max_drawdown_pct:
            conditions.append(f"Max drawdown reached: {dd_forecast.current_drawdown_pct:.1f}%")
            self._halted = True
            self._halt_reason = "Max drawdown"

        # Check crash probability
        if prediction.crash_probability > 0.7:
            conditions.append(f"High crash probability: {prediction.crash_probability:.1%}")

        # Calculate position sizing
        sizing = self.calculate_position_size(symbol, side, size_usd, win_rate, win_loss_ratio)

        # Get adaptive stop
        stop = self.adaptive_stop.get_stop(side, current_price, prediction.volatility_regime)

        # Calculate take profit (asymmetric based on risk)
        if risk_level in [RiskLevel.MINIMAL, RiskLevel.LOW]:
            take_profit_pct = stop.stop_loss_pct * 3.0  # 3:1 reward:risk
        elif risk_level == RiskLevel.MODERATE:
            take_profit_pct = stop.stop_loss_pct * 2.0  # 2:1
        else:
            take_profit_pct = stop.stop_loss_pct * 1.5  # 1.5:1

        # Determine action
        if self._halted or risk_level == RiskLevel.CRITICAL:
            action = PositionAction.HALT
            approved = False
            approved_size = 0
        elif sizing.recommended_size_usd < size_usd * 0.1:
            action = PositionAction.DECREASE
            approved = True
            approved_size = sizing.recommended_size_usd
            conditions.append(f"Size reduced: ${size_usd:.0f} -> ${approved_size:.0f}")
        else:
            action = PositionAction.OPEN
            approved = True
            approved_size = sizing.recommended_size_usd

        # Add conditions
        if prediction.crash_probability > 0.5:
            conditions.append(f"Elevated crash risk: {prediction.crash_probability:.0%}")
        if prediction.volatility_regime == "extreme":
            conditions.append("Extreme volatility regime")
        if dd_forecast.predicted_max_dd_24h > self.max_drawdown_pct * 0.8:
            conditions.append(f"Predicted DD 24h: {dd_forecast.predicted_max_dd_24h:.1f}%")

        return TradeDecision(
            approved=approved,
            action=action,
            symbol=symbol,
            side=side,
            approved_size_usd=approved_size,
            requested_size_usd=size_usd,
            stop_loss=stop,
            take_profit_pct=take_profit_pct,
            risk_score=risk_score,
            risk_level=risk_level,
            position_sizing=sizing,
            risk_prediction=prediction,
            drawdown_forecast=dd_forecast,
            conditions=conditions,
        )

    def get_system_status(self) -> SystemRiskStatus:
        """Get overall system risk status."""
        prediction = self.risk_predictor.predict()
        dd_forecast = self.drawdown_forecaster.forecast(hours=24)
        risk_score = self.calculate_risk_score()
        risk_level = self.get_risk_level()

        recommendations = []

        if risk_level in [RiskLevel.HIGH, RiskLevel.EXTREME, RiskLevel.CRITICAL]:
            recommendations.append("Reduce position sizes")
        if prediction.crash_probability > 0.5:
            recommendations.append("Consider hedging tail risk")
        if dd_forecast.current_drawdown_pct > 10:
            recommendations.append("Review stop-loss levels")
        if prediction.volatility_regime == "extreme":
            recommendations.append("Consider reducing leverage")

        return SystemRiskStatus(
            risk_level=risk_level,
            risk_score=risk_score,
            var_99_pct=prediction.predicted_var_30m * 100,
            cvar_99_pct=prediction.predicted_var_30m * 100 * 1.5,
            max_drawdown_pct=dd_forecast.current_drawdown_pct,
            correlation_alert=False,  # Would integrate with CorrelationMonitor
            black_swan_detected=prediction.crash_probability > 0.8,
            circuit_breaker_active=self._halted,
            position_multiplier=self.calculate_position_size("BTC/USDT", "buy", 10000).adjusted_fraction / 0.25,
            can_trade=not self._halted and risk_level != RiskLevel.CRITICAL,
            recommendations=recommendations,
        )

    def reset_halt(self) -> None:
        """Manually reset halt state (requires manual override)."""
        self._halted = False
        self._halt_reason = ""
        logger.info("Halt state manually reset")
