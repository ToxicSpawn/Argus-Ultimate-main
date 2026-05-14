"""
Monte Carlo Simulation Engine - Ultimate Edge Module

Provides comprehensive risk simulation:
- 1000+ scenario generation
- Historical simulation
- Portfolio distribution analysis
- VaR/CVaR calculation
- Stress testing scenarios

This module quantifies risk through massive scenario analysis.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    """Single simulation scenario."""
    scenario_id: int
    returns: List[float]
    final_value: float
    max_drawdown: float
    sharpe_ratio: float
    trade_count: int


@dataclass
class SimulationResult:
    """Monte Carlo simulation result."""
    n_scenarios: int
    scenarios: List[Scenario]
    percentiles: Dict[int, float]
    var_confidences: Dict[float, float]
    cvar_confidences: Dict[float, float]
    probability_of_ruin: float
    expected_return: float
    volatility: float
    skewness: float
    kurtosis: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StressTestScenario:
    """Stress test scenario."""
    name: str
    description: str
    price_shocks: Dict[str, float]
    correlation_shocks: Dict[Tuple[str, str], float]
    volatility_multiplier: float


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for risk quantification.

    Capabilities:
    - Historical bootstrap simulation
    - Parametric (normal) simulation
    - Fat-tailed (t-distribution) simulation
    - VaR/CVaR at multiple confidence levels
    - Stress testing with custom scenarios
    - Portfolio optimization via simulation
    """

    def __init__(
        self,
        n_scenarios: int = 1000,
        confidence_levels: Optional[List[float]] = None,
        time_horizon_days: int = 252,
    ):
        self.n_scenarios = n_scenarios
        self.confidence_levels = confidence_levels or [0.90, 0.95, 0.99]
        self.horizon_days = time_horizon_days

        self._returns_history: Deque[List[float]] = deque(maxlen=252 * 5)
        self._last_result: Optional[SimulationResult] = None

    def add_returns(self, returns: List[float]) -> None:
        """Add historical returns for simulation."""
        self._returns_history.append(returns)

    def run_historical_simulation(
        self,
        initial_value: float,
        n_years: float = 1.0,
        n_scenarios: Optional[int] = None,
    ) -> SimulationResult:
        """
        Run historical bootstrap simulation.

        Args:
            initial_value: Starting portfolio value
            n_years: Simulation horizon in years
            n_scenarios: Number of scenarios (default: self.n_scenarios)

        Returns:
            SimulationResult with all scenarios and metrics
        """
        if n_scenarios is None:
            n_scenarios = self.n_scenarios

        if len(self._returns_history) < 30:
            logger.warning("Insufficient historical data for simulation")
            return self._create_empty_result()

        all_returns = []
        for returns_list in self._returns_history:
            all_returns.extend(returns_list)

        n_days = int(n_years * 252)
        scenarios = []

        for i in range(n_scenarios):
            scenario_returns = []
            value = initial_value

            indices = np.random.choice(len(all_returns), size=n_days, replace=True)
            daily_returns = [all_returns[idx] for idx in indices]

            peak = value
            max_dd = 0.0

            for ret in daily_returns:
                value *= (1 + ret)
                peak = max(peak, value)
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)
                scenario_returns.append(ret)

            sharpe = self._calculate_sharpe(daily_returns)

            scenario = Scenario(
                scenario_id=i,
                returns=scenario_returns,
                final_value=value,
                max_drawdown=max_dd,
                sharpe_ratio=sharpe,
                trade_count=n_days,
            )
            scenarios.append(scenario)

        return self._compile_results(scenarios, initial_value)

    def run_parametric_simulation(
        self,
        initial_value: float,
        mean_return: float,
        std_return: float,
        n_years: float = 1.0,
        n_scenarios: Optional[int] = None,
    ) -> SimulationResult:
        """
        Run parametric (normal distribution) simulation.

        Args:
            initial_value: Starting portfolio value
            mean_return: Mean daily return
            std_return: Standard deviation of daily returns
            n_years: Simulation horizon in years
            n_scenarios: Number of scenarios

        Returns:
            SimulationResult with all scenarios and metrics
        """
        if n_scenarios is None:
            n_scenarios = self.n_scenarios

        n_days = int(n_years * 252)
        scenarios = []

        for i in range(n_scenarios):
            scenario_returns = []
            value = initial_value

            daily_returns = np.random.normal(mean_return, std_return, n_days).tolist()

            peak = value
            max_dd = 0.0

            for ret in daily_returns:
                value *= (1 + ret)
                peak = max(peak, value)
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)
                scenario_returns.append(ret)

            sharpe = self._calculate_sharpe(daily_returns)

            scenario = Scenario(
                scenario_id=i,
                returns=scenario_returns,
                final_value=value,
                max_drawdown=max_dd,
                sharpe_ratio=sharpe,
                trade_count=n_days,
            )
            scenarios.append(scenario)

        return self._compile_results(scenarios, initial_value)

    def run_fat_tailed_simulation(
        self,
        initial_value: float,
        mean_return: float,
        std_return: float,
        df: float = 5.0,
        n_years: float = 1.0,
        n_scenarios: Optional[int] = None,
    ) -> SimulationResult:
        """
        Run fat-tailed simulation using t-distribution.

        Args:
            initial_value: Starting portfolio value
            mean_return: Mean daily return
            std_return: Standard deviation of daily returns
            df: Degrees of freedom (lower = fatter tails)
            n_years: Simulation horizon in years
            n_scenarios: Number of scenarios

        Returns:
            SimulationResult with all scenarios and metrics
        """
        if n_scenarios is None:
            n_scenarios = self.n_scenarios

        n_days = int(n_years * 252)
        scenarios = []

        from scipy.stats import t as t_dist
        t_scale = std_return * math.sqrt(df / (df - 2))

        for i in range(n_scenarios):
            scenario_returns = []
            value = initial_value

            t_returns = t_dist.rvs(df=df, size=n_days)
            daily_returns = [(r * t_scale) + mean_return for r in t_returns]

            peak = value
            max_dd = 0.0

            for ret in daily_returns:
                value *= (1 + ret)
                peak = max(peak, value)
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)
                scenario_returns.append(ret)

            sharpe = self._calculate_sharpe(daily_returns)

            scenario = Scenario(
                scenario_id=i,
                returns=scenario_returns,
                final_value=value,
                max_drawdown=max_dd,
                sharpe_ratio=sharpe,
                trade_count=n_days,
            )
            scenarios.append(scenario)

        return self._compile_results(scenarios, initial_value)

    def _compile_results(
        self,
        scenarios: List[Scenario],
        initial_value: float,
    ) -> SimulationResult:
        """Compile simulation results into final output."""
        final_values = [s.final_value for s in scenarios]
        max_drawdowns = [s.max_drawdown for s in scenarios]
        all_returns = []
        for s in scenarios:
            all_returns.extend(s.returns)

        final_values_arr = np.array(final_values)
        percentiles = {
            p: float(np.percentile(final_values_arr, p))
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }

        returns_arr = np.array(all_returns)
        var_confidences = {}
        cvar_confidences = {}

        for conf in self.confidence_levels:
            alpha = (1 - conf) * 100
            var_pct = np.percentile(returns_arr, alpha)
            var_dollar = initial_value * var_pct
            var_confidences[conf] = var_dollar

            cvar_returns = returns_arr[returns_arr <= var_pct]
            cvar_pct = np.mean(cvar_returns) if len(cvar_returns) > 0 else var_pct
            cvar_dollar = initial_value * cvar_pct
            cvar_confidences[conf] = cvar_dollar

        ruin_count = sum(1 for v in final_values if v < initial_value * 0.5)
        prob_ruin = ruin_count / len(final_values) if final_values else 0.0

        expected_return = np.mean(final_values_arr)
        volatility = np.std(final_values_arr)
        skewness = float(np.mean(((final_values_arr - expected_return) / volatility) ** 3)) if volatility > 0 else 0.0
        kurtosis_val = float(np.mean(((final_values_arr - expected_return) / volatility) ** 4)) if volatility > 0 else 0.0

        result = SimulationResult(
            n_scenarios=len(scenarios),
            scenarios=scenarios[:100],
            percentiles=percentiles,
            var_confidences=var_confidences,
            cvar_confidences=cvar_confidences,
            probability_of_ruin=prob_ruin,
            expected_return=expected_return,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis_val,
        )

        self._last_result = result
        return result

    def calculate_var(
        self,
        initial_value: float,
        confidence: float = 0.95,
        method: str = "historical",
    ) -> float:
        """
        Calculate Value at Risk.

        Args:
            initial_value: Portfolio value
            confidence: Confidence level (0.95 = 95%)
            method: 'historical', 'parametric', or 'fat_tailed'

        Returns:
            VaR as positive number (loss)
        """
        if method == "historical" and len(self._returns_history) >= 30:
            result = self.run_historical_simulation(initial_value, n_years=1.0 / 252)
        elif method == "fat_tailed":
            all_returns = []
            for r in self._returns_history:
                all_returns.extend(r)
            mean_r = np.mean(all_returns)
            std_r = np.std(all_returns)
            result = self.run_fat_tailed_simulation(initial_value, mean_r, std_r, n_years=1.0 / 252)
        else:
            all_returns = []
            for r in self._returns_history:
                all_returns.extend(r)
            mean_r = np.mean(all_returns) if all_returns else 0.0
            std_r = np.std(all_returns) if all_returns else 0.01
            result = self.run_parametric_simulation(initial_value, mean_r, std_r, n_years=1.0 / 252)

        return abs(result.var_confidences.get(confidence, 0.0))

    def calculate_cvar(
        self,
        initial_value: float,
        confidence: float = 0.95,
    ) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if self._last_result is None:
            self.run_historical_simulation(initial_value)

        if self._last_result:
            return abs(self._last_result.cvar_confidences.get(confidence, 0.0))
        return 0.0

    def run_stress_test(
        self,
        initial_value: float,
        scenario: StressTestScenario,
    ) -> Dict[str, float]:
        """
        Run stress test scenario.

        Args:
            initial_value: Starting portfolio value
            scenario: Stress test scenario

        Returns:
            Dict with stress test results
        """
        final_value = initial_value

        for symbol, shock_pct in scenario.price_shocks.items():
            final_value *= (1 + shock_pct)

        max_drawdown = 0.0
        peak = final_value

        for _ in range(10):
            final_value *= (1 + np.random.uniform(-0.02, 0.02) * scenario.volatility_multiplier)
            peak = max(peak, final_value)
            dd = (peak - final_value) / peak
            max_drawdown = max(max_drawdown, dd)

        loss = initial_value - final_value
        loss_pct = loss / initial_value

        return {
            "initial_value": initial_value,
            "stressed_value": final_value,
            "loss": loss,
            "loss_pct": loss_pct,
            "max_drawdown": max_drawdown,
            "scenario_name": scenario.name,
        }

    def get_default_scenarios(self) -> List[StressTestScenario]:
        """Get standard stress test scenarios."""
        return [
            StressTestScenario(
                name="2008 Financial Crisis",
                description="Historical: 2008-like market crash",
                price_shocks={"BTC": -0.80, "ETH": -0.85, "SPY": -0.50},
                correlation_shocks={},
                volatility_multiplier=3.0,
            ),
            StressTestScenario(
                name="Flash Crash",
                description="Intraday flash crash",
                price_shocks={"BTC": -0.30, "ETH": -0.35},
                correlation_shocks={},
                volatility_multiplier=5.0,
            ),
            StressTestScenario(
                name="Black Swan",
                description="Tail risk event",
                price_shocks={"BTC": -0.50, "ETH": -0.55, "SPY": -0.30},
                correlation_shocks={},
                volatility_multiplier=4.0,
            ),
            StressTestScenario(
                name="Regulatory Crackdown",
                description="Crypto regulation event",
                price_shocks={"BTC": -0.40, "ETH": -0.50, "ALT": -0.70},
                correlation_shocks={},
                volatility_multiplier=2.5,
            ),
            StressTestScenario(
                name="Correlation Spike",
                description="All assets correlate during crisis",
                price_shocks={"BTC": -0.25, "ETH": -0.25, "SPY": -0.25},
                correlation_shocks={("BTC", "SPY"): 0.95},
                volatility_multiplier=2.0,
            ),
        ]

    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio from returns."""
        if not returns:
            return 0.0
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret == 0:
            return 0.0
        return (mean_ret / std_ret) * math.sqrt(252)

    def _create_empty_result(self) -> SimulationResult:
        """Create empty result when insufficient data."""
        return SimulationResult(
            n_scenarios=0,
            scenarios=[],
            percentiles={},
            var_confidences={},
            cvar_confidences={},
            probability_of_ruin=0.0,
            expected_return=0.0,
            volatility=0.0,
            skewness=0.0,
            kurtosis=0.0,
        )

    def reset(self) -> None:
        """Reset simulation engine."""
        self._returns_history.clear()
        self._last_result = None
        logger.info("MonteCarloEngine reset")
