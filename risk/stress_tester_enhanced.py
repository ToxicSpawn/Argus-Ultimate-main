"""
Enhanced stress testing framework for portfolio risk analysis.

This module extends the lightweight historical stress tester with:
- richer scenario metadata
- deterministic scenario path generation
- reverse stress testing
- Monte Carlo stress simulation using historical price data
- aggregate reporting suitable for risk dashboards
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


_DEFAULT_VOLATILITY = 0.04
_DEFAULT_CORRELATION = 0.55
_MIN_HISTORY_POINTS = 20


@dataclass(frozen=True)
class StressScenario:
    name: str
    description: str
    shock_type: str
    asset_shocks: Dict[str, float]
    volatility_multiplier: float
    correlation_shock: float
    duration_days: int


@dataclass(frozen=True)
class StressTestResult:
    scenario_name: str
    initial_portfolio_value: float
    stressed_portfolio_value: float
    max_drawdown: float
    var_99: float
    cvar_99: float
    worst_day_loss: float
    recovery_days: int
    positions_impact: Dict[str, float]


class StressTestEngine:
    """Enhanced stress testing engine for Argus portfolios."""

    def __init__(
        self,
        portfolio: Optional[Mapping[str, Any]] = None,
        historical_price_data: Optional[Mapping[str, Sequence[float]]] = None,
        risk_reporter: Optional[object] = None,
        random_seed: Optional[int] = 42,
    ) -> None:
        self.portfolio: Dict[str, Any] = dict(portfolio or {})
        self.historical_price_data: Dict[str, np.ndarray] = self._normalise_price_history(
            historical_price_data or {}
        )
        self.risk_reporter = risk_reporter
        self._rng = np.random.default_rng(random_seed)
        self._historical_scenarios = self.load_historical_scenarios()

    def load_historical_scenarios(self) -> List[StressScenario]:
        """Load built-in historical stress scenarios."""
        return [
            StressScenario(
                name="2008 Financial Crisis",
                description="Global deleveraging shock with extreme correlation breakdown across risk assets.",
                shock_type="historical",
                asset_shocks={
                    "BTC": -0.55,
                    "ETH": -0.60,
                    "SOL": -0.58,
                    "ADA": -0.52,
                    "XRP": -0.48,
                    "SPY": -0.37,
                    "QQQ": -0.42,
                    "GLD": 0.08,
                },
                volatility_multiplier=2.8,
                correlation_shock=0.30,
                duration_days=180,
            ),
            StressScenario(
                name="2020 COVID Crash",
                description="Pandemic liquidation event with violent cross-asset selloff and elevated volatility.",
                shock_type="historical",
                asset_shocks={
                    "BTC": -0.63,
                    "ETH": -0.75,
                    "SOL": -0.70,
                    "AVAX": -0.72,
                    "LINK": -0.78,
                    "DOT": -0.75,
                    "ADA": -0.74,
                    "XRP": -0.65,
                    "MATIC": -0.72,
                    "SPY": -0.34,
                },
                volatility_multiplier=3.5,
                correlation_shock=0.35,
                duration_days=32,
            ),
            StressScenario(
                name="2022 Crypto Winter",
                description="Extended crypto bear market with persistent drawdown and reduced liquidity.",
                shock_type="historical",
                asset_shocks={
                    "BTC": -0.65,
                    "ETH": -0.68,
                    "SOL": -0.85,
                    "AVAX": -0.78,
                    "LINK": -0.70,
                    "DOT": -0.76,
                    "ADA": -0.80,
                    "XRP": -0.58,
                    "MATIC": -0.74,
                },
                volatility_multiplier=2.2,
                correlation_shock=0.25,
                duration_days=240,
            ),
            StressScenario(
                name="LUNA/Terra Collapse",
                description="Algorithmic stablecoin death spiral and contagion across crypto beta exposures.",
                shock_type="historical",
                asset_shocks={
                    "BTC": -0.40,
                    "ETH": -0.45,
                    "SOL": -0.55,
                    "AVAX": -0.58,
                    "LINK": -0.50,
                    "DOT": -0.52,
                    "ADA": -0.60,
                    "XRP": -0.40,
                    "MATIC": -0.55,
                    "LUNA": -0.999,
                },
                volatility_multiplier=2.7,
                correlation_shock=0.28,
                duration_days=10,
            ),
            StressScenario(
                name="FTX Collapse",
                description="Exchange insolvency shock with concentrated downside in exchange-linked crypto assets.",
                shock_type="historical",
                asset_shocks={
                    "BTC": -0.25,
                    "ETH": -0.30,
                    "SOL": -0.60,
                    "AVAX": -0.35,
                    "LINK": -0.32,
                    "DOT": -0.33,
                    "ADA": -0.30,
                    "XRP": -0.20,
                    "MATIC": -0.35,
                    "FTT": -0.95,
                },
                volatility_multiplier=2.4,
                correlation_shock=0.22,
                duration_days=7,
            ),
            StressScenario(
                name="Flash Crash",
                description="Single-day liquidity vacuum with gap risk, forced liquidations, and intraday whipsaw.",
                shock_type="historical",
                asset_shocks={
                    "BTC": -0.50,
                    "ETH": -0.55,
                    "SOL": -0.50,
                    "AVAX": -0.50,
                    "LINK": -0.52,
                    "DOT": -0.50,
                    "ADA": -0.50,
                    "XRP": -0.45,
                    "MATIC": -0.50,
                    "SPY": -0.12,
                },
                volatility_multiplier=4.0,
                correlation_shock=0.40,
                duration_days=1,
            ),
        ]

    def create_hypothetical_scenario(self, shocks: dict) -> StressScenario:
        """Create a hypothetical scenario from user-defined shocks."""
        asset_shocks = {
            self._canonical_symbol(str(symbol)): float(change)
            for symbol, change in (shocks or {}).items()
            if symbol is not None
        }
        if not asset_shocks:
            raise ValueError("shocks must contain at least one asset shock")

        avg_magnitude = float(np.mean(np.abs(list(asset_shocks.values()))))
        duration_days = max(1, int(round(5 + (avg_magnitude * 20))))
        volatility_multiplier = max(1.0, min(5.0, 1.0 + avg_magnitude * 5.0))
        correlation_shock = max(0.0, min(0.75, avg_magnitude * 0.5))

        return StressScenario(
            name="Hypothetical Stress Scenario",
            description="User-defined hypothetical market shock for targeted portfolio analysis.",
            shock_type="hypothetical",
            asset_shocks=asset_shocks,
            volatility_multiplier=volatility_multiplier,
            correlation_shock=correlation_shock,
            duration_days=duration_days,
        )

    def run_stress_test(self, portfolio: dict, scenario: StressScenario) -> StressTestResult:
        """Run a deterministic stress scenario against a portfolio."""
        normalised_portfolio = self._normalise_portfolio(portfolio)
        self.portfolio = dict(portfolio)

        exposures = self._position_exposures(normalised_portfolio)
        initial_portfolio_value = float(sum(abs(value) for value in exposures.values()))
        if initial_portfolio_value <= 0:
            raise ValueError("portfolio must contain at least one non-zero position")

        impacts = self._calculate_position_impacts(exposures, scenario)
        stressed_portfolio_value = max(0.0, initial_portfolio_value + float(sum(impacts.values())))

        daily_portfolio_pnl = self._simulate_scenario_path(exposures, scenario)
        portfolio_path = initial_portfolio_value + np.cumsum(daily_portfolio_pnl)

        max_drawdown = self._calculate_max_drawdown(np.concatenate(([initial_portfolio_value], portfolio_path)))
        var_99, cvar_99 = self._calculate_tail_metrics(daily_portfolio_pnl)
        worst_day_loss = float(np.min(daily_portfolio_pnl)) if daily_portfolio_pnl.size else 0.0
        recovery_days = self._estimate_recovery_days(initial_portfolio_value, portfolio_path, scenario.duration_days)

        result = StressTestResult(
            scenario_name=scenario.name,
            initial_portfolio_value=initial_portfolio_value,
            stressed_portfolio_value=stressed_portfolio_value,
            max_drawdown=max_drawdown,
            var_99=var_99,
            cvar_99=cvar_99,
            worst_day_loss=worst_day_loss,
            recovery_days=recovery_days,
            positions_impact=impacts,
        )

        logger.info(
            "Stress test complete for '%s': initial=%.2f stressed=%.2f drawdown=%.2f%%",
            scenario.name,
            initial_portfolio_value,
            stressed_portfolio_value,
            max_drawdown * 100.0,
        )
        return result

    def run_reverse_stress_test(self, target_loss_pct: float) -> StressScenario:
        """Construct a reverse stress scenario that targets a portfolio loss threshold."""
        if not (0 < target_loss_pct < 1):
            raise ValueError("target_loss_pct must be between 0 and 1")

        exposures = self._position_exposures(self._normalise_portfolio(self.portfolio))
        if not exposures:
            raise ValueError("portfolio must be set before running reverse stress test")

        initial_value = float(sum(abs(value) for value in exposures.values()))
        if initial_value <= 0:
            raise ValueError("portfolio must contain at least one non-zero position")

        target_loss = initial_value * float(target_loss_pct)
        total_abs_exposure = float(sum(abs(value) for value in exposures.values()))
        base_shock = min(0.95, target_loss / total_abs_exposure) if total_abs_exposure > 0 else 0.0

        ranked = sorted(exposures.items(), key=lambda item: abs(item[1]), reverse=True)
        asset_shocks: Dict[str, float] = {}
        concentration_boost = 1.0
        for index, (symbol, _) in enumerate(ranked):
            if index == 0:
                concentration_boost = 1.2
            elif index >= 3:
                concentration_boost = 0.9
            canonical = self._canonical_symbol(symbol)
            historical_bias = abs(self._proxy_shock(canonical, self._historical_scenarios[0]))
            shock = min(0.95, max(base_shock * concentration_boost, historical_bias * 0.35))
            asset_shocks[canonical] = -shock

        return StressScenario(
            name=f"Reverse Stress {target_loss_pct:.0%} Loss",
            description="Synthetic reverse stress scenario calibrated to breach the requested portfolio loss threshold.",
            shock_type="reverse",
            asset_shocks=asset_shocks,
            volatility_multiplier=max(1.5, 1.0 + target_loss_pct * 6.0),
            correlation_shock=min(0.85, 0.2 + target_loss_pct * 0.8),
            duration_days=max(1, int(round(3 + target_loss_pct * 30))),
        )

    def run_monte_carlo_stress(self, n_simulations: int) -> List[StressTestResult]:
        """Run Monte Carlo stress testing using historical price data."""
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive")

        exposures = self._position_exposures(self._normalise_portfolio(self.portfolio))
        if not exposures:
            raise ValueError("portfolio must be set before running Monte Carlo stress tests")

        initial_portfolio_value = float(sum(abs(value) for value in exposures.values()))
        returns_matrix, symbols = self._build_returns_matrix(exposures)

        mean_vector = np.mean(returns_matrix, axis=0)
        covariance_matrix = np.cov(returns_matrix, rowvar=False)
        covariance_matrix = self._regularize_covariance(covariance_matrix, len(symbols))

        durations = [5, 10, 20]
        results: List[StressTestResult] = []
        for index in range(n_simulations):
            duration = durations[index % len(durations)]
            sampled_returns = self._rng.multivariate_normal(mean_vector, covariance_matrix, size=duration)
            daily_portfolio_pnl = self._portfolio_pnl_from_returns(exposures, symbols, sampled_returns)
            portfolio_path = initial_portfolio_value + np.cumsum(daily_portfolio_pnl)
            impacts = self._impacts_from_terminal_returns(
                exposures,
                symbols,
                sampled_returns,
            )
            stressed_value = max(0.0, initial_portfolio_value + float(sum(impacts.values())))
            max_drawdown = self._calculate_max_drawdown(np.concatenate(([initial_portfolio_value], portfolio_path)))
            var_99, cvar_99 = self._calculate_tail_metrics(daily_portfolio_pnl)

            results.append(
                StressTestResult(
                    scenario_name=f"Monte Carlo Stress #{index + 1}",
                    initial_portfolio_value=initial_portfolio_value,
                    stressed_portfolio_value=stressed_value,
                    max_drawdown=max_drawdown,
                    var_99=var_99,
                    cvar_99=cvar_99,
                    worst_day_loss=float(np.min(daily_portfolio_pnl)),
                    recovery_days=self._estimate_recovery_days(initial_portfolio_value, portfolio_path, duration),
                    positions_impact=impacts,
                )
            )

        logger.info("Monte Carlo stress run complete: %d simulations", n_simulations)
        return results

    def aggregate_results(self, results: List[StressTestResult]) -> dict:
        """Aggregate stress test results into a report-friendly dictionary."""
        if not results:
            return {
                "scenario_count": 0,
                "worst_scenario": None,
                "average_portfolio_change_pct": 0.0,
                "average_max_drawdown": 0.0,
                "max_var_99": 0.0,
                "max_cvar_99": 0.0,
                "worst_day_loss": 0.0,
                "average_recovery_days": 0.0,
                "position_impacts": {},
            }

        def portfolio_change_pct(result: StressTestResult) -> float:
            if result.initial_portfolio_value <= 0:
                return 0.0
            return (result.stressed_portfolio_value - result.initial_portfolio_value) / result.initial_portfolio_value

        worst_result = min(results, key=portfolio_change_pct)
        aggregated_position_impacts: Dict[str, float] = {}
        for result in results:
            for symbol, impact in result.positions_impact.items():
                aggregated_position_impacts[symbol] = aggregated_position_impacts.get(symbol, 0.0) + impact

        summary = {
            "scenario_count": len(results),
            "worst_scenario": worst_result.scenario_name,
            "average_portfolio_change_pct": float(np.mean([portfolio_change_pct(result) for result in results])),
            "average_max_drawdown": float(np.mean([result.max_drawdown for result in results])),
            "max_var_99": float(np.max([result.var_99 for result in results])),
            "max_cvar_99": float(np.max([result.cvar_99 for result in results])),
            "worst_day_loss": float(np.min([result.worst_day_loss for result in results])),
            "average_recovery_days": float(np.mean([result.recovery_days for result in results])),
            "position_impacts": aggregated_position_impacts,
        }

        self._publish_risk_report(summary)
        return summary

    def generate_stress_report(self, results: List[StressTestResult]) -> str:
        """Generate a human-readable stress report."""
        summary = self.aggregate_results(results)
        lines = [
            "ARGUS STRESS TEST REPORT",
            "========================",
            f"Scenarios analysed: {summary['scenario_count']}",
            f"Worst scenario: {summary['worst_scenario']}",
            f"Average stressed change: {summary['average_portfolio_change_pct']:.2%}",
            f"Average max drawdown: {summary['average_max_drawdown']:.2%}",
            f"Peak VaR(99): {summary['max_var_99']:.2f}",
            f"Peak CVaR(99): {summary['max_cvar_99']:.2f}",
            f"Worst day loss: {summary['worst_day_loss']:.2f}",
            f"Average recovery days: {summary['average_recovery_days']:.1f}",
            "",
            "Scenario details:",
        ]

        for result in results:
            pnl = result.stressed_portfolio_value - result.initial_portfolio_value
            pnl_pct = pnl / result.initial_portfolio_value if result.initial_portfolio_value > 0 else 0.0
            lines.append(
                "- "
                f"{result.scenario_name}: pnl={pnl:.2f} ({pnl_pct:.2%}), "
                f"drawdown={result.max_drawdown:.2%}, var99={result.var_99:.2f}, "
                f"cvar99={result.cvar_99:.2f}, recovery_days={result.recovery_days}"
            )

        report = "\n".join(lines)
        self._publish_risk_report({"stress_report": report, "summary": summary})
        return report

    def _normalise_portfolio(self, portfolio: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        normalised: Dict[str, Dict[str, Any]] = {}
        for symbol, raw_position in (portfolio or {}).items():
            if raw_position is None:
                continue

            if isinstance(raw_position, (int, float)):
                notional = float(raw_position)
                if notional == 0.0:
                    continue
                normalised[symbol] = {
                    "symbol": symbol,
                    "notional_value": notional,
                }
                continue

            if isinstance(raw_position, Mapping):
                notional = self._extract_position_notional(raw_position)
                if notional == 0.0:
                    continue
                normalised[symbol] = dict(raw_position)
                normalised[symbol]["symbol"] = raw_position.get("symbol", symbol)
                normalised[symbol]["notional_value"] = notional
                continue

            logger.debug("Unsupported position format for %s: %r", symbol, raw_position)

        return normalised

    def _extract_position_notional(self, position: Mapping[str, Any]) -> float:
        direct_keys = (
            "notional_value",
            "market_value",
            "value",
            "exposure_usd",
            "position_value",
            "qty_usd",
        )
        for key in direct_keys:
            value = position.get(key)
            if isinstance(value, (int, float)):
                notional = float(value)
                side = str(position.get("side", "")).lower()
                if side in {"sell", "short", "-1"} and notional > 0:
                    return -notional
                return notional

        quantity = position.get("quantity", position.get("qty", position.get("size")))
        price = position.get("current_price", position.get("price", position.get("mark_price")))
        if isinstance(quantity, (int, float)) and isinstance(price, (int, float)):
            notional = float(quantity) * float(price)
            side = str(position.get("side", "")).lower()
            if side in {"sell", "short", "-1"} and notional > 0:
                return -notional
            return notional

        return 0.0

    def _position_exposures(self, portfolio: Mapping[str, Dict[str, Any]]) -> Dict[str, float]:
        exposures: Dict[str, float] = {}
        for symbol, position in portfolio.items():
            notional = float(position.get("notional_value", 0.0))
            if notional != 0.0:
                exposures[symbol] = notional
        return exposures

    def _calculate_position_impacts(
        self,
        exposures: Mapping[str, float],
        scenario: StressScenario,
    ) -> Dict[str, float]:
        impacts: Dict[str, float] = {}
        for symbol, exposure in exposures.items():
            shock = self._proxy_shock(symbol, scenario)
            impacts[symbol] = float(exposure) * shock
        return impacts

    def _proxy_shock(self, symbol: str, scenario: StressScenario) -> float:
        canonical = self._canonical_symbol(symbol)
        if canonical in scenario.asset_shocks:
            return float(scenario.asset_shocks[canonical])

        if scenario.asset_shocks:
            proxy = float(np.median(np.array(list(scenario.asset_shocks.values()), dtype=float)))
            logger.debug(
                "Stress scenario '%s' missing shock for %s; using proxy %.4f",
                scenario.name,
                canonical,
                proxy,
            )
            return proxy

        return 0.0

    def _simulate_scenario_path(
        self,
        exposures: Mapping[str, float],
        scenario: StressScenario,
    ) -> np.ndarray:
        duration = max(1, int(scenario.duration_days))
        if not exposures:
            return np.zeros(duration, dtype=float)

        symbols = list(exposures.keys())
        common_factor = self._rng.normal(
            loc=0.0,
            scale=0.01 * max(1.0, scenario.volatility_multiplier),
            size=duration,
        )
        daily_portfolio_pnl = np.zeros(duration, dtype=float)

        for symbol in symbols:
            exposure = float(exposures[symbol])
            cumulative_shock = self._proxy_shock(symbol, scenario)
            base_log_return = math.log(max(1e-6, 1.0 + cumulative_shock)) / duration
            symbol_vol = self._estimated_volatility(symbol) * max(1.0, scenario.volatility_multiplier)
            idiosyncratic = self._rng.normal(loc=0.0, scale=symbol_vol / math.sqrt(duration), size=duration)
            correlated = scenario.correlation_shock * common_factor
            daily_returns = base_log_return + correlated + idiosyncratic

            target_total = math.log(max(1e-6, 1.0 + cumulative_shock))
            adjustment = (target_total - float(np.sum(daily_returns))) / duration
            daily_returns = daily_returns + adjustment

            arithmetic_returns = np.expm1(daily_returns)
            daily_portfolio_pnl += exposure * arithmetic_returns

        return daily_portfolio_pnl

    def _calculate_tail_metrics(self, daily_pnl: np.ndarray) -> tuple[float, float]:
        if daily_pnl.size == 0:
            return 0.0, 0.0

        percentile_1 = float(np.percentile(daily_pnl, 1))
        var_99 = abs(percentile_1)
        tail_losses = daily_pnl[daily_pnl <= percentile_1]
        cvar_99 = abs(float(np.mean(tail_losses))) if tail_losses.size else var_99
        return var_99, cvar_99

    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        if portfolio_values.size == 0:
            return 0.0

        running_peak = np.maximum.accumulate(portfolio_values)
        drawdowns = np.where(running_peak > 0, (running_peak - portfolio_values) / running_peak, 0.0)
        return float(np.max(drawdowns))

    def _estimate_recovery_days(
        self,
        initial_value: float,
        portfolio_path: np.ndarray,
        fallback_days: int,
    ) -> int:
        if portfolio_path.size == 0:
            return 0

        for index, value in enumerate(portfolio_path, start=1):
            if value >= initial_value:
                return index

        trough = float(np.min(portfolio_path))
        unrecovered_gap = max(0.0, initial_value - trough)
        if unrecovered_gap == 0.0:
            return 0

        estimated_extra_days = max(1, int(math.ceil(unrecovered_gap / max(initial_value * 0.01, 1.0))))
        return int(fallback_days + estimated_extra_days)

    def _normalise_price_history(
        self,
        historical_price_data: Mapping[str, Sequence[float]],
    ) -> Dict[str, np.ndarray]:
        normalised: Dict[str, np.ndarray] = {}
        for symbol, values in historical_price_data.items():
            prices = self._coerce_price_series(values)
            if prices.size >= 2:
                normalised[self._canonical_symbol(symbol)] = prices
        return normalised

    def _coerce_price_series(self, values: Sequence[float]) -> np.ndarray:
        extracted: List[float] = []
        for item in values:
            if isinstance(item, (int, float)):
                extracted.append(float(item))
            elif isinstance(item, Mapping):
                for key in ("close", "price", "value"):
                    price = item.get(key)
                    if isinstance(price, (int, float)):
                        extracted.append(float(price))
                        break

        array = np.array(extracted, dtype=float)
        return array[array > 0]

    def _estimated_volatility(self, symbol: str) -> float:
        prices = self.historical_price_data.get(self._canonical_symbol(symbol))
        if prices is None or prices.size < _MIN_HISTORY_POINTS:
            return _DEFAULT_VOLATILITY

        returns = np.diff(np.log(prices))
        if returns.size == 0:
            return _DEFAULT_VOLATILITY

        return float(max(np.std(returns), _DEFAULT_VOLATILITY / 4.0))

    def _build_returns_matrix(self, exposures: Mapping[str, float]) -> tuple[np.ndarray, List[str]]:
        symbols = [self._canonical_symbol(symbol) for symbol in exposures.keys()]
        series_list: List[np.ndarray] = []

        min_length: Optional[int] = None
        for symbol in symbols:
            prices = self.historical_price_data.get(symbol)
            if prices is None or prices.size < _MIN_HISTORY_POINTS:
                synthetic = self._synthetic_price_series(symbol)
                prices = synthetic
            returns = np.diff(np.log(prices))
            if returns.size == 0:
                returns = np.full(_MIN_HISTORY_POINTS, -_DEFAULT_VOLATILITY / 10.0)
            series_list.append(returns)
            min_length = returns.size if min_length is None else min(min_length, returns.size)

        if min_length is None or min_length <= 1:
            raise ValueError("insufficient historical data for Monte Carlo stress testing")

        aligned = np.column_stack([series[-min_length:] for series in series_list])
        return aligned, symbols

    def _synthetic_price_series(self, symbol: str) -> np.ndarray:
        seed_vol = self._estimated_volatility(symbol)
        returns = self._rng.normal(loc=-seed_vol / 8.0, scale=seed_vol, size=max(_MIN_HISTORY_POINTS, 60))
        return 100.0 * np.exp(np.cumsum(returns))

    def _regularize_covariance(self, covariance_matrix: np.ndarray, dimension: int) -> np.ndarray:
        if dimension == 1:
            variance = float(covariance_matrix) if np.ndim(covariance_matrix) == 0 else float(covariance_matrix[0, 0])
            return np.array([[max(variance, 1e-8)]], dtype=float)

        if covariance_matrix.shape != (dimension, dimension):
            covariance_matrix = np.eye(dimension, dtype=float) * (_DEFAULT_VOLATILITY ** 2)

        jitter = np.eye(dimension, dtype=float) * 1e-8
        return covariance_matrix + jitter

    def _portfolio_pnl_from_returns(
        self,
        exposures: Mapping[str, float],
        symbols: Sequence[str],
        sampled_returns: np.ndarray,
    ) -> np.ndarray:
        daily_pnl = np.zeros(sampled_returns.shape[0], dtype=float)
        for col_index, symbol in enumerate(symbols):
            exposure = float(exposures[self._matching_portfolio_symbol(exposures, symbol)])
            daily_pnl += exposure * np.expm1(sampled_returns[:, col_index])
        return daily_pnl

    def _impacts_from_terminal_returns(
        self,
        exposures: Mapping[str, float],
        symbols: Sequence[str],
        sampled_returns: np.ndarray,
    ) -> Dict[str, float]:
        impacts: Dict[str, float] = {}
        terminal_returns = np.expm1(np.sum(sampled_returns, axis=0))
        for index, symbol in enumerate(symbols):
            portfolio_symbol = self._matching_portfolio_symbol(exposures, symbol)
            impacts[portfolio_symbol] = float(exposures[portfolio_symbol]) * float(terminal_returns[index])
        return impacts

    def _matching_portfolio_symbol(self, exposures: Mapping[str, float], canonical_symbol: str) -> str:
        for symbol in exposures.keys():
            if self._canonical_symbol(symbol) == canonical_symbol:
                return symbol
        return canonical_symbol

    def _publish_risk_report(self, payload: dict) -> None:
        if self.risk_reporter is None:
            return

        for method_name in (
            "publish_stress_report",
            "record_stress_report",
            "ingest_stress_report",
            "update_risk_report",
        ):
            method = getattr(self.risk_reporter, method_name, None)
            if callable(method):
                try:
                    method(payload)
                    return
                except Exception as exc:
                    logger.warning("risk_reporter.%s failed: %s", method_name, exc)
                    return

        logger.debug("Risk reporter provided but no compatible publish method was found")

    def _canonical_symbol(self, symbol: str) -> str:
        upper = str(symbol).upper()
        for separator in ("/", "-", "_"):
            if separator in upper:
                upper = upper.split(separator)[0]
                break
        aliases = {
            "XBT": "BTC",
            "BTCUSDT": "BTC",
            "ETHUSDT": "ETH",
            "SOLUSDT": "SOL",
        }
        return aliases.get(upper, upper)


__all__ = [
    "StressScenario",
    "StressTestResult",
    "StressTestEngine",
]
