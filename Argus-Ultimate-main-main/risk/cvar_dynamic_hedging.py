"""Enhanced Risk Management with CVaR-Based Dynamic Hedging.

Features:
- CVaR (Conditional Value at Risk) calculation
- Dynamic hedging strategies
- Tail risk protection
- Delta/Gamma/Vega hedging
- Options-based protection
- Cross-asset correlation hedging
- Real-time risk monitoring
"""

from __future__ import annotations

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class HedgeType(Enum):
    DELTA = "delta"
    GAMMA = "gamma"
    VEGA = "vega"
    DELTA_GAMMA = "delta_gamma"
    OPTIONS = "options"
    CORRELATION = "correlation"


@dataclass
class RiskMetrics:
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    expected_shortfall: float = 0.0
    max_drawdown: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0


@dataclass
class HedgePosition:
    hedge_type: HedgeType
    symbol: str
    quantity: float
    strike: Optional[float] = None
    expiry: Optional[float] = None
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0


@dataclass
class HedgingStrategy:
    target_cvar_pct: float
    max_hedge_cost_pct: float = 0.02
    rebalance_threshold: float = 0.10
    enabled_types: List[HedgeType] = field(default_factory=lambda: [HedgeType.DELTA])


class CVaRRiskEngine:
    def __init__(
        self,
        confidence_levels: List[float] = None,
        window_size: int = 252,
        method: str = "historical",
    ):
        self._confidence_levels = confidence_levels or [0.95, 0.99]
        self._window = window_size
        self._method = method
        
        self._returns_history: deque = deque(maxlen=window_size)
        self._pnl_history: deque = deque(maxlen=window_size)
        
        self._var_cache: Dict[float, float] = {}
        self._cvar_cache: Dict[float, float] = {}

    def add_return(self, return_pct: float) -> None:
        self._returns_history.append(return_pct)
        self._pnl_history.append(-return_pct)
        self._clear_cache()

    def _clear_cache(self) -> None:
        self._var_cache.clear()
        self._cvar_cache.clear()

    def compute_var(
        self,
        confidence: float = 0.95,
    ) -> float:
        if confidence in self._var_cache:
            return self._var_cache[confidence]
        
        if len(self._returns_history) < 10:
            return 0.0
        
        returns = sorted(self._returns_history)
        idx = int((1 - confidence) * len(returns))
        idx = min(max(0, idx), len(returns) - 1)
        
        var = -returns[idx]
        self._var_cache[confidence] = max(0.0, var)
        return self._var_cache[confidence]

    def compute_cvar(
        self,
        confidence: float = 0.95,
    ) -> float:
        if confidence in self._cvar_cache:
            return self._cvar_cache[confidence]
        
        if len(self._returns_history) < 10:
            return 0.0
        
        var = self.compute_var(confidence)
        returns = list(self._returns_history)
        
        tail_losses = [r for r in returns if r <= -var]
        
        if not tail_losses:
            self._cvar_cache[confidence] = var
            return self._cvar_cache[confidence]
        
        cvar = -np.mean(tail_losses)
        self._cvar_cache[confidence] = max(0.0, cvar)
        return self._cvar_cache[confidence]

    def compute_var_histogram(
        self,
        n_bins: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(self._returns_history) < n_bins:
            return np.array([]), np.array([])
        
        returns = np.array(self._returns_history)
        hist, edges = np.histogram(returns, bins=n_bins)
        return hist, edges

    def compute_risk_metrics(
        self,
        equity_curve: Optional[np.ndarray] = None,
    ) -> RiskMetrics:
        returns = np.array(self._returns_history)
        
        metrics = RiskMetrics()
        
        if len(returns) >= 10:
            metrics.var_95 = self.compute_var(0.95)
            metrics.var_99 = self.compute_var(0.99)
            metrics.cvar_95 = self.compute_cvar(0.95)
            metrics.cvar_99 = self.compute_cvar(0.99)
            metrics.skewness = float(np.mean(returns**3)) / (float(np.mean(returns**2))**1.5) if len(returns) > 0 else 0.0
            metrics.kurtosis = float(np.mean(returns**4)) / (float(np.mean(returns**2))**2) - 3 if len(returns) > 0 else 0.0
        
        if equity_curve is not None and len(equity_curve) > 1:
            cummax = np.maximum.accumulate(equity_curve)
            drawdowns = (equity_curve - cummax) / cummax
            metrics.max_drawdown = abs(np.min(drawdowns))
        
        return metrics


class DynamicHedger:
    def __init__(
        self,
        risk_engine: CVaRRiskEngine,
        strategy: HedgingStrategy,
    ):
        self._risk_engine = risk_engine
        self._strategy = strategy
        self._hedge_positions: Dict[str, HedgePosition] = {}
        self._hedge_history: deque = deque(maxlen=1000)

    def calculate_delta_hedge(
        self,
        position_value: float,
        beta: float = 1.0,
        risk_free_rate: float = 0.02,
    ) -> float:
        delta_target = -position_value * beta
        
        current_delta = sum(
            h.delta for h in self._hedge_positions.values()
        )
        
        return delta_target - current_delta

    def calculate_gamma_hedge(
        self,
        position_gamma: float,
        hedge_gamma: float = 0.0,
    ) -> float:
        if hedge_gamma == 0:
            return 0.0
        
        return -position_gamma / hedge_gamma

    def calculate_vega_hedge(
        self,
        position_vega: float,
        option_vega: float = 1.0,
    ) -> float:
        if option_vega == 0:
            return 0.0
        
        return -position_vega / option_vega

    def calculate_correlation_hedge(
        self,
        exposures: Dict[str, float],
        correlation_matrix: np.ndarray,
    ) -> Dict[str, float]:
        n = len(exposures)
        if n == 0:
            return {}
        
        assets = list(exposures.keys())
        weights = np.array([exposures[a] for a in assets])
        
        variance = np.dot(weights, np.dot(correlation_matrix, weights))
        
        hedges = {}
        for i, asset in enumerate(assets):
            hedges[asset] = -np.sum(correlation_matrix[i] * weights) / (correlation_matrix[i, i] + 1e-8)
        
        return hedges

    def calculate_options_hedge(
        self,
        position_value: float,
        current_var: float,
        target_var: float,
        option_cost: float = 0.02,
        risk_free_rate: float = 0.02,
    ) -> Tuple[Optional[float], Optional[float]]:
        if current_var <= target_var:
            return None, None
        
        protection_needed = current_var - target_var
        notional = protection_needed * position_value / (option_cost * position_value)
        
        strike_pct = 1 - protection_needed
        
        return strike_pct, notional

    def rebalance_needed(
        self,
        current_cvar: float,
    ) -> bool:
        if current_cvar > self._strategy.target_cvar_pct:
            excess_cvar = current_cvar - self._strategy.target_cvar_pct
            return excess_cvar > self._strategy.rebalance_threshold
        
        return False

    def execute_hedge(
        self,
        hedge_type: HedgeType,
        symbol: str,
        quantity: float,
        metadata: Optional[Dict] = None,
    ) -> HedgePosition:
        position = HedgePosition(
            hedge_type=hedge_type,
            symbol=symbol,
            quantity=quantity,
            delta=metadata.get("delta", 0.0) if metadata else 0.0,
            gamma=metadata.get("gamma", 0.0) if metadata else 0.0,
            vega=metadata.get("vega", 0.0) if metadata else 0.0,
        )
        
        self._hedge_positions[symbol] = position
        self._hedge_history.append({
            "type": hedge_type.value,
            "symbol": symbol,
            "quantity": quantity,
            "timestamp": time.time(),
        })
        
        logger.info(f"Executed {hedge_type.value} hedge: {symbol} qty={quantity}")
        return position

    def get_hedge_positions(self) -> Dict[str, HedgePosition]:
        return self._hedge_positions


class CVaRBasedDynamicHedger:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        self._risk_engine = CVaRRiskEngine(
            window_size=self.config.get("window_size", 252),
            method=self.config.get("method", "historical"),
        )
        
        strategy = HedgingStrategy(
            target_cvar_pct=self.config.get("target_cvar_pct", 5.0),
            max_hedge_cost_pct=self.config.get("max_hedge_cost_pct", 0.02),
            rebalance_threshold=self.config.get("rebalance_threshold", 0.10),
            enabled_types=[HedgeType(h) for h in self.config.get("enabled_hedges", ["delta"])],
        )
        
        self._hedger = DynamicHedger(self._risk_engine, strategy)
        
        self._positions: Dict[str, float] = {}
        self._correlation_matrix: Optional[np.ndarray] = None

    def update_position(
        self,
        symbol: str,
        quantity: float,
        delta: float = 0.0,
        gamma: float = 0.0,
        vega: float = 0.0,
    ) -> None:
        self._positions[symbol] = quantity

    def set_correlation_matrix(self, matrix: np.ndarray) -> None:
        self._correlation_matrix = matrix

    def calculate_hedges(
        self,
        portfolio_value: float,
    ) -> Dict[HedgeType, float]:
        hedges = {}
        
        current_cvar = self._risk_engine.compute_cvar(0.95)
        
        if self._hedger.rebalance_needed(current_cvar):
            for hedge_type in self._hedger._strategy.enabled_types:
                if hedge_type == HedgeType.DELTA:
                    total_exposure = sum(
                        abs(q * 100) for q in self._positions.values()
                    )
                    hedges[hedge_type] = self._hedger.calculate_delta_hedge(
                        total_exposure,
                        beta=1.0,
                    )
                
                elif hedge_type == HedgeType.GAMMA:
                    pass
                
                elif hedge_type == HedgeType.VEGA:
                    pass
        
        return hedges

    def add_return(self, return_pct: float) -> None:
        self._risk_engine.add_return(return_pct)

    def get_risk_metrics(self) -> RiskMetrics:
        return self._risk_engine.compute_risk_metrics()

    def get_current_cvar(self) -> float:
        return self._risk_engine.compute_cvar(0.95)

    def get_positions(self) -> Dict[str, HedgePosition]:
        return self._hedger.get_hedge_positions()