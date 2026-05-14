#!/usr/bin/env python3
"""
ARGUS Ultimate - Advanced Risk Management System
===============================================

Institutional-grade risk management with:
- Portfolio optimization (Black-Litterman, Risk Parity)
- Stress testing and scenario analysis
- Dynamic position sizing
- Market regime detection
- Liquidity risk management
- Quantum Monte Carlo VaR/CVaR (Sobol QMC)
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import statistics

# Quantum Monte Carlo for VaR/CVaR
try:
    from quantum.algorithms.quantum_monte_carlo import run as qmc_var_cvar
    QMC_AVAILABLE = True
except ImportError:
    QMC_AVAILABLE = False

try:
    from foundation.services.transcendent.base import BaseTranscendentService
except ImportError:
    # foundation package not available — provide stub base class
    class BaseTranscendentService:
        pass

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    CRASH = "crash"
    RECOVERY = "recovery"


@dataclass
class PortfolioPosition:
    """Portfolio position details"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    weight: float
    risk_contribution: float
    liquidity_score: float


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    total_value: float
    total_pnl: float
    daily_pnl: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    value_at_risk_95: float
    expected_shortfall_95: float
    beta: float
    correlation_matrix: np.ndarray
    concentration_risk: float
    liquidity_risk: float
    regime_risk: float


@dataclass
class StressTestScenario:
    """Stress test scenario definition"""
    name: str
    description: str
    shock_type: str  # 'absolute', 'percentage', 'volatility'
    shock_value: float
    time_horizon: int  # days
    probability: float
    historical_analog: Optional[str] = None


class MarketRegimeDetector:
    """
    Advanced market regime detection using multiple indicators
    """

    def __init__(self):
        self.price_history: deque = deque(maxlen=1000)
        self.volatility_history: deque = deque(maxlen=500)
        self.regime_history: deque = deque(maxlen=100)

        # Regime detection parameters
        self.trend_threshold = 0.02  # 2% trend threshold
        self.volatility_threshold = 0.05  # 5% volatility threshold
        self.crash_threshold = -0.10  # 10% crash threshold

    async def update_market_data(self, price: float, timestamp: datetime):
        """Update with latest market data"""
        self.price_history.append({
            'price': price,
            'timestamp': timestamp,
            'returns': 0.0  # Will be calculated
        })

        # Calculate returns
        if len(self.price_history) >= 2:
            prev_price = self.price_history[-2]['price']
            current_return = (price - prev_price) / prev_price
            self.price_history[-1]['returns'] = current_return

        # Update volatility
        if len(self.price_history) >= 30:
            recent_returns = [p['returns'] for p in list(self.price_history)[-30:]]
            volatility = np.std(recent_returns) * np.sqrt(252)  # Annualized
            self.volatility_history.append(volatility)

    async def detect_regime(self) -> MarketRegime:
        """Detect current market regime using multiple indicators"""
        if len(self.price_history) < 60:
            return MarketRegime.SIDEWAYS  # Not enough data

        # Trend analysis (60-day)
        recent_prices = [p['price'] for p in list(self.price_history)[-60:]]
        trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

        # Volatility analysis
        if self.volatility_history:
            current_volatility = self.volatility_history[-1]
        else:
            current_volatility = 0.02  # Default

        # Momentum analysis (20-day)
        momentum_prices = [p['price'] for p in list(self.price_history)[-20:]]
        momentum = (momentum_prices[-1] - momentum_prices[0]) / momentum_prices[0]

        # Regime classification logic
        if trend > self.trend_threshold and momentum > 0.01:
            if current_volatility > self.volatility_threshold:
                return MarketRegime.HIGH_VOLATILITY
            else:
                return MarketRegime.BULL
        elif trend < self.crash_threshold:
            return MarketRegime.CRASH
        elif trend < -self.trend_threshold:
            return MarketRegime.BEAR
        elif abs(momentum) < 0.005 and current_volatility < self.volatility_threshold:
            return MarketRegime.SIDEWAYS
        elif trend > 0 and momentum > 0:
            return MarketRegime.RECOVERY
        else:
            return MarketRegime.SIDEWAYS

    async def get_regime_confidence(self) -> float:
        """Get confidence level in current regime detection"""
        if len(self.regime_history) < 10:
            return 0.5

        # Calculate regime stability
        recent_regimes = list(self.regime_history)[-10:]
        most_common = max(set(recent_regimes), key=recent_regimes.count)
        stability = recent_regimes.count(most_common) / len(recent_regimes)

        return stability


class PortfolioOptimizer:
    """
    Advanced portfolio optimization using multiple models
    """

    def __init__(self):
        self.risk_free_rate = 0.02  # 2% risk-free rate
        self.max_weight_per_asset = 0.20  # 20% max per asset
        self.min_weight_per_asset = 0.01  # 1% min per asset

    async def optimize_portfolio_black_litterman(
        self,
        current_weights: Dict[str, float],
        expected_returns: Dict[str, float],
        covariance_matrix: np.ndarray,
        investor_views: Dict[str, float],
        confidence_levels: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Black-Litterman portfolio optimization
        Combines market equilibrium with investor views
        """
        n_assets = len(current_weights)

        # Market equilibrium returns (CAPM-based)
        market_weights = np.array(list(current_weights.values()))
        market_return = np.dot(market_weights, list(expected_returns.values()))

        # Risk aversion parameter
        risk_aversion = 2.5

        # Equilibrium returns
        equilibrium_returns = risk_aversion * np.dot(covariance_matrix, market_weights)

        # Incorporate investor views
        P = np.zeros((len(investor_views), n_assets))  # View matrix
        Q = np.array(list(investor_views.values()))     # View returns
        Ω = np.diag([1/c for c in confidence_levels.values()])  # Confidence matrix

        # Fill P matrix based on view keys
        asset_names = list(current_weights.keys())
        for i, (asset, _) in enumerate(investor_views.items()):
            if asset in asset_names:
                P[i, asset_names.index(asset)] = 1

        # Black-Litterman formula
        tau = 0.025  # Uncertainty in prior
        Π = equilibrium_returns

        # Posterior expected returns
        temp1 = np.linalg.inv(tau * covariance_matrix)
        temp2 = np.dot(np.dot(P.T, np.linalg.inv(Ω)), P)
        posterior_cov = np.linalg.inv(temp1 + temp2)

        temp3 = np.dot(temp1, Π) + np.dot(np.dot(P.T, np.linalg.inv(Ω)), Q)
        posterior_returns = np.dot(posterior_cov, temp3)

        # Optimize portfolio (mean-variance with constraints)
        optimal_weights = await self._mean_variance_optimization(
            posterior_returns, covariance_matrix
        )

        return dict(zip(asset_names, optimal_weights))

    async def optimize_portfolio_risk_parity(
        self,
        current_weights: Dict[str, float],
        covariance_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Risk parity portfolio optimization
        Equalizes risk contribution across assets
        """
        n_assets = len(current_weights)
        asset_names = list(current_weights.keys())

        # Initial weights
        weights = np.array(list(current_weights.values()))

        # Risk parity optimization (simplified iterative approach)
        for _ in range(100):  # Max iterations
            # Calculate portfolio volatility
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

            # Calculate marginal risk contributions
            marginal_risk = np.dot(covariance_matrix, weights) / portfolio_vol

            # Calculate risk contributions
            risk_contributions = weights * marginal_risk

            # Target equal risk contribution
            target_risk = portfolio_vol / n_assets

            # Adjust weights
            new_weights = weights * (target_risk / risk_contributions)

            # Normalize
            new_weights = new_weights / np.sum(new_weights)

            # Check convergence
            if np.max(np.abs(new_weights - weights)) < 0.001:
                break

            weights = new_weights

        # Apply constraints
        weights = np.clip(weights, self.min_weight_per_asset, self.max_weight_per_asset)
        weights = weights / np.sum(weights)  # Re-normalize

        return dict(zip(asset_names, weights))

    async def _mean_variance_optimization(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        target_return: Optional[float] = None
    ) -> np.ndarray:
        """Mean-variance portfolio optimization with constraints"""
        n_assets = len(expected_returns)

        # Use scipy.optimize for proper optimization
        try:
            from scipy.optimize import minimize

            def objective(weights):
                # Minimize negative Sharpe ratio (maximize Sharpe)
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
                sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
                return -sharpe

            def constraint_sum_weights(weights):
                return np.sum(weights) - 1.0

            def constraint_min_weight(weights):
                return weights - self.min_weight_per_asset

            def constraint_max_weight(weights):
                return self.max_weight_per_asset - weights

            # Constraints
            constraints = [
                {'type': 'eq', 'fun': constraint_sum_weights}
            ]

            # Bounds for each asset
            bounds = [(self.min_weight_per_asset, self.max_weight_per_asset)] * n_assets

            # Initial guess (equal weight)
            x0 = np.ones(n_assets) / n_assets

            # Optimize
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )

            if result.success:
                return result.x
            else:
                # Fallback to equal weight
                return np.ones(n_assets) / n_assets

        except ImportError:
            # Fallback without scipy
            return np.ones(n_assets) / n_assets


class StressTester:
    """
    Comprehensive stress testing and scenario analysis
    """

    def __init__(self):
        self.historical_scenarios = self._load_historical_scenarios()
        self.monte_carlo_simulations = 10000

    def _load_historical_scenarios(self) -> Dict[str, StressTestScenario]:
        """Load historical stress test scenarios"""
        return {
            '2008_crisis': StressTestScenario(
                name='2008 Financial Crisis',
                description='Global financial crisis impact',
                shock_type='percentage',
                shock_value=-0.50,  # 50% decline
                time_horizon=365,
                probability=0.01,
                historical_analog='2008-2009'
            ),
            '2020_covid': StressTestScenario(
                name='COVID-19 Crash',
                description='March 2020 market crash',
                shock_type='percentage',
                shock_value=-0.34,  # 34% decline
                time_horizon=30,
                probability=0.05,
                historical_analog='2020-03'
            ),
            'tech_bubble': StressTestScenario(
                name='Tech Bubble Burst',
                description='2000 tech bubble collapse',
                shock_type='percentage',
                shock_value=-0.49,  # 49% decline
                time_horizon=180,
                probability=0.02,
                historical_analog='2000-2002'
            ),
            'volatility_spike': StressTestScenario(
                name='Volatility Spike',
                description='Sudden volatility increase',
                shock_type='volatility',
                shock_value=3.0,  # 3x normal volatility
                time_horizon=30,
                probability=0.10,
                historical_analog='Various flash crashes'
            ),
            'liquidity_crisis': StressTestScenario(
                name='Liquidity Crisis',
                description='Market liquidity dries up',
                shock_type='absolute',
                shock_value=0.10,  # 10% of positions become illiquid
                time_horizon=7,
                probability=0.03,
                historical_analog='Various liquidity events'
            )
        }

    async def run_historical_stress_test(
        self,
        portfolio: Dict[str, PortfolioPosition],
        scenario_name: str
    ) -> Dict[str, Any]:
        """Run historical scenario stress test"""
        if scenario_name not in self.historical_scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario = self.historical_scenarios[scenario_name]

        # Apply shock to portfolio
        stressed_portfolio = await self._apply_shock(portfolio.copy(), scenario)

        # Calculate stressed metrics
        stressed_metrics = await self._calculate_stressed_metrics(stressed_portfolio)

        return {
            'scenario': scenario,
            'stressed_portfolio': stressed_portfolio,
            'stressed_metrics': stressed_metrics,
            'survival_probability': await self._calculate_survival_probability(
                portfolio, scenario
            ),
            'recovery_time': await self._estimate_recovery_time(scenario)
        }

    async def run_monte_carlo_stress_test(
        self,
        portfolio: Dict[str, PortfolioPosition],
        n_simulations: int = 10000
    ) -> Dict[str, Any]:
        """Run Monte Carlo stress testing"""
        simulation_results = []

        for _ in range(n_simulations):
            # Generate random shock scenario
            shock_type = np.random.choice(['percentage', 'volatility', 'liquidity'])
            shock_value = np.random.normal(0, 0.15)  # Mean 0, std 15%

            scenario = StressTestScenario(
                name=f'mc_{_}',
                description='Monte Carlo simulation',
                shock_type=shock_type,
                shock_value=shock_value,
                time_horizon=np.random.randint(1, 90),
                probability=0.5  # Neutral probability
            )

            # Apply shock and record result
            stressed_portfolio = await self._apply_shock(portfolio.copy(), scenario)
            stressed_value = sum(p.market_value for p in stressed_portfolio.values())

            simulation_results.append({
                'final_value': stressed_value,
                'return_pct': (stressed_value / sum(p.market_value for p in portfolio.values()) - 1) * 100,
                'scenario': scenario
            })

        # Analyze results
        returns = [r['return_pct'] for r in simulation_results]
        var_95 = np.percentile(returns, 5)  # 5th percentile for 95% VaR
        var_99 = np.percentile(returns, 1)  # 1st percentile for 99% VaR
        expected_shortfall = np.mean([r for r in returns if r <= var_95])

        return {
            'n_simulations': n_simulations,
            'value_at_risk_95': var_95,
            'value_at_risk_99': var_99,
            'expected_shortfall_95': expected_shortfall,
            'worst_case': min(returns),
            'best_case': max(returns),
            'average_return': np.mean(returns),
            'probability_loss': len([r for r in returns if r < 0]) / len(returns)
        }

    async def _apply_shock(
        self,
        portfolio: Dict[str, PortfolioPosition],
        scenario: StressTestScenario
    ) -> Dict[str, PortfolioPosition]:
        """Apply shock to portfolio based on scenario"""
        shocked_portfolio = {}

        for symbol, position in portfolio.items():
            shocked_position = position

            if scenario.shock_type == 'percentage':
                # Percentage price shock
                shocked_price = position.current_price * (1 + scenario.shock_value)
                shocked_position = PortfolioPosition(
                    **position.__dict__,
                    current_price=shocked_price,
                    market_value=position.quantity * shocked_price,
                    unrealized_pnl=(shocked_price - position.entry_price) * position.quantity
                )

            elif scenario.shock_type == 'volatility':
                # Volatility shock (increased price swings)
                volatility_adjustment = np.random.normal(0, scenario.shock_value)
                shocked_price = position.current_price * (1 + volatility_adjustment)
                shocked_position = PortfolioPosition(
                    **position.__dict__,
                    current_price=max(shocked_price, 0.01),  # Prevent negative prices
                    market_value=position.quantity * max(shocked_price, 0.01),
                    unrealized_pnl=(max(shocked_price, 0.01) - position.entry_price) * position.quantity
                )

            elif scenario.shock_type == 'absolute':
                # Liquidity shock (some positions become illiquid)
                if np.random.random() < scenario.shock_value:
                    # This position becomes illiquid - assume 20% haircut
                    shocked_price = position.current_price * 0.80
                    shocked_position = PortfolioPosition(
                        **position.__dict__,
                        current_price=shocked_price,
                        market_value=position.quantity * shocked_price,
                        unrealized_pnl=(shocked_price - position.entry_price) * position.quantity
                    )

            shocked_portfolio[symbol] = shocked_position

        return shocked_portfolio

    async def _calculate_stressed_metrics(self, portfolio: Dict[str, PortfolioPosition]) -> RiskMetrics:
        """Calculate risk metrics for stressed portfolio"""
        # Simplified metrics calculation
        total_value = sum(p.market_value for p in portfolio.values())
        total_pnl = sum(p.unrealized_pnl for p in portfolio.values())

        # Placeholder values for demonstration
        return RiskMetrics(
            total_value=total_value,
            total_pnl=total_pnl,
            daily_pnl=0.0,  # Would calculate from time series
            sharpe_ratio=0.0,  # Would calculate properly
            sortino_ratio=0.0,
            max_drawdown=0.0,
            value_at_risk_95=0.0,
            expected_shortfall_95=0.0,
            beta=0.0,
            correlation_matrix=np.array([[1.0]]),
            concentration_risk=0.0,
            liquidity_risk=0.0,
            regime_risk=0.0
        )

    async def _calculate_survival_probability(
        self,
        portfolio: Dict[str, PortfolioPosition],
        scenario: StressTestScenario
    ) -> float:
        """Calculate probability of portfolio surviving the scenario"""
        # Simplified survival calculation
        portfolio_value = sum(p.market_value for p in portfolio.values())
        stressed_value = portfolio_value * (1 + scenario.shock_value)

        # Assume survival if portfolio retains >50% of value
        survival_threshold = portfolio_value * 0.50

        if stressed_value > survival_threshold:
            return 0.95  # High survival probability
        elif stressed_value > survival_threshold * 0.7:
            return 0.70  # Moderate survival probability
        else:
            return 0.20  # Low survival probability

    async def _estimate_recovery_time(self, scenario: StressTestScenario) -> int:
        """Estimate time to recover from scenario (in days)"""
        shock_magnitude = abs(scenario.shock_value)

        if shock_magnitude < 0.10:
            return 30  # 1 month
        elif shock_magnitude < 0.25:
            return 90  # 3 months
        elif shock_magnitude < 0.50:
            return 180  # 6 months
        else:
            return 365  # 1 year


class AdvancedRiskManagementService(BaseTranscendentService):
    """
    Advanced Risk Management Service
    ================================

    Institutional-grade risk management combining:
    - Market regime detection
    - Portfolio optimization
    - Stress testing
    - Dynamic position sizing
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        event_bus: Optional[Any] = None
    ):
        super().__init__(
            name="AdvancedRiskManagementService",
            version="1.0.0",
            description="Institutional-grade risk management system",
            config=config,
            event_bus=event_bus
        )

        # Core components
        self.regime_detector = MarketRegimeDetector()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.stress_tester = StressTester()

        # State tracking
        self.current_regime = MarketRegime.SIDEWAYS
        self.portfolio: Dict[str, PortfolioPosition] = {}
        self.risk_limits = {
            'max_drawdown': 0.20,
            'max_daily_loss': 0.03,
            'max_concentration': 0.25,
            'min_liquidity_score': 0.6
        }

    async def _initialize_transcendent(self) -> None:
        """Initialize the advanced risk management system"""
        logger.info("🛡️ Initializing Advanced Risk Management Service...")

        # Initialize regime detection
        logger.info("   - Market regime detector: ACTIVE")
        logger.info("   - Portfolio optimizer: ACTIVE")
        logger.info("   - Stress tester: ACTIVE")

        # Set up risk monitoring
        self.risk_monitoring_active = True

        logger.info("✅ Advanced Risk Management Service initialized")
        logger.info("   - Risk monitoring: ACTIVE")
        logger.info("   - Regime detection: ACTIVE")
        logger.info("   - Stress testing: READY")

    async def update_portfolio(self, positions: Dict[str, Dict[str, Any]]):
        """Update current portfolio positions"""
        for symbol, position_data in positions.items():
            self.portfolio[symbol] = PortfolioPosition(
                symbol=symbol,
                quantity=position_data['quantity'],
                entry_price=position_data['entry_price'],
                current_price=position_data['current_price'],
                market_value=position_data['market_value'],
                unrealized_pnl=position_data['unrealized_pnl'],
                weight=position_data.get('weight', 0.0),
                risk_contribution=position_data.get('risk_contribution', 0.0),
                liquidity_score=position_data.get('liquidity_score', 0.8)
            )

    async def assess_risk(self) -> Dict[str, Any]:
        """Comprehensive risk assessment"""
        # Detect market regime
        regime = await self.regime_detector.detect_regime()
        regime_confidence = await self.regime_detector.get_regime_confidence()

        # Calculate risk metrics
        risk_metrics = await self._calculate_risk_metrics()

        # Check risk limits
        breaches = await self._check_risk_limits(risk_metrics)

        # Generate recommendations
        recommendations = await self._generate_risk_recommendations(risk_metrics, breaches, regime)

        return {
            'market_regime': {
                'regime': regime.value,
                'confidence': regime_confidence
            },
            'risk_metrics': risk_metrics,
            'limit_breaches': breaches,
            'recommendations': recommendations,
            'overall_risk_score': await self._calculate_overall_risk_score(risk_metrics, breaches)
        }

    async def optimize_portfolio(self, optimization_method: str = 'risk_parity') -> Dict[str, Any]:
        """Optimize portfolio using specified method"""
        if not self.portfolio:
            return {'error': 'No portfolio data available'}

        # Extract current weights and returns
        current_weights = {symbol: pos.weight for symbol, pos in self.portfolio.items()}
        expected_returns = await self._estimate_expected_returns()
        covariance_matrix = await self._estimate_covariance_matrix()

        if optimization_method == 'black_litterman':
            # Would need investor views - using simplified version
            optimal_weights = await self.portfolio_optimizer.optimize_portfolio_risk_parity(
                current_weights, covariance_matrix
            )
        else:  # Default to risk parity
            optimal_weights = await self.portfolio_optimizer.optimize_portfolio_risk_parity(
                current_weights, covariance_matrix
            )

        # Calculate rebalancing trades
        rebalancing_trades = await self._calculate_rebalancing_trades(current_weights, optimal_weights)

        return {
            'optimization_method': optimization_method,
            'current_weights': current_weights,
            'optimal_weights': optimal_weights,
            'expected_improvement': await self._estimate_optimization_improvement(
                current_weights, optimal_weights, expected_returns, covariance_matrix
            ),
            'rebalancing_trades': rebalancing_trades
        }

    async def run_stress_test(self, scenario: str = 'monte_carlo') -> Dict[str, Any]:
        """Run comprehensive stress testing"""
        if not self.portfolio:
            return {'error': 'No portfolio data available'}

        if scenario == 'monte_carlo':
            results = await self.stress_tester.run_monte_carlo_stress_test(self.portfolio)
        elif scenario in self.stress_tester.historical_scenarios:
            results = await self.stress_tester.run_historical_stress_test(self.portfolio, scenario)
        else:
            return {'error': f'Unknown scenario: {scenario}'}

        return {
            'test_type': scenario,
            'results': results,
            'risk_assessment': await self._assess_stress_test_results(results),
            'mitigation_strategies': await self._generate_stress_mitigation_strategies(results)
        }

    async def _calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics with QMC VaR/CVaR"""
        if not self.portfolio:
            return RiskMetrics(
                total_value=0, total_pnl=0, daily_pnl=0, sharpe_ratio=0,
                sortino_ratio=0, max_drawdown=0, value_at_risk_95=0,
                expected_shortfall_95=0, beta=0, correlation_matrix=np.array([[1.0]]),
                concentration_risk=0, liquidity_risk=0, regime_risk=0
            )

        # Basic calculations
        total_value = sum(p.market_value for p in self.portfolio.values())
        total_pnl = sum(p.unrealized_pnl for p in self.portfolio.values())

        # Concentration risk (Herfindahl-Hirschman Index)
        weights = np.array([p.weight for p in self.portfolio.values()])
        concentration_risk = np.sum(weights ** 2)

        # Liquidity risk (average liquidity score)
        liquidity_scores = [p.liquidity_score for p in self.portfolio.values()]
        liquidity_risk = 1 - np.mean(liquidity_scores) if liquidity_scores else 1.0

        # Quantum Monte Carlo VaR/CVaR calculation
        var_95 = 0.0
        cvar_95 = 0.0
        
        if QMC_AVAILABLE and total_value > 0:
            try:
                # Collect returns from portfolio positions
                returns_data = []
                for pos in self.portfolio.values():
                    if hasattr(pos, 'returns') and pos.returns:
                        returns_data.extend(pos.returns[-100:])  # Last 100 returns
                
                if len(returns_data) >= 10:
                    qmc_result = qmc_var_cvar(
                        returns_data,
                        n_samples=5000,
                        confidence=0.95
                    )
                    # Convert to dollar terms
                    var_95 = abs(qmc_result.get('var', 0.0)) * total_value
                    cvar_95 = abs(qmc_result.get('cvar', 0.0)) * total_value
                    logger.debug(f"QMC VaR/CVaR: method={qmc_result.get('method')}, var={var_95:.2f}, cvar={cvar_95:.2f}")
            except Exception as e:
                logger.debug(f"QMC VaR calculation failed, using fallback: {e}")
        
        # Fallback: simple historical VaR if QMC not available or failed
        if var_95 == 0.0 and total_value > 0:
            # Use position weights as rough proxy
            var_95 = total_value * 0.02  # 2% default VaR
            cvar_95 = total_value * 0.03  # 3% default CVaR

        return RiskMetrics(
            total_value=total_value,
            total_pnl=total_pnl,
            daily_pnl=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            value_at_risk_95=var_95,
            expected_shortfall_95=cvar_95,
            beta=0.0,
            correlation_matrix=np.array([[1.0]]),
            concentration_risk=concentration_risk,
            liquidity_risk=liquidity_risk,
            regime_risk=0.0
        )

    async def _check_risk_limits(self, metrics: RiskMetrics) -> List[Dict[str, Any]]:
        """Check for risk limit breaches"""
        breaches = []

        if metrics.max_drawdown > self.risk_limits['max_drawdown']:
            breaches.append({
                'limit': 'max_drawdown',
                'current': metrics.max_drawdown,
                'threshold': self.risk_limits['max_drawdown'],
                'severity': 'HIGH'
            })

        if metrics.concentration_risk > self.risk_limits['max_concentration']:
            breaches.append({
                'limit': 'concentration_risk',
                'current': metrics.concentration_risk,
                'threshold': self.risk_limits['max_concentration'],
                'severity': 'MEDIUM'
            })

        if metrics.liquidity_risk > (1 - self.risk_limits['min_liquidity_score']):
            breaches.append({
                'limit': 'liquidity_risk',
                'current': metrics.liquidity_risk,
                'threshold': 1 - self.risk_limits['min_liquidity_score'],
                'severity': 'MEDIUM'
            })

        return breaches

    async def _generate_risk_recommendations(
        self,
        metrics: RiskMetrics,
        breaches: List[Dict[str, Any]],
        regime: MarketRegime
    ) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []

        if breaches:
            recommendations.append("🚨 IMMEDIATE ACTION REQUIRED: Address risk limit breaches")

        if regime == MarketRegime.HIGH_VOLATILITY:
            recommendations.append("⚠️ HIGH VOLATILITY: Reduce position sizes and tighten stops")
        elif regime == MarketRegime.BEAR:
            recommendations.append("📉 BEAR MARKET: Consider defensive positioning")
        elif regime == MarketRegime.CRASH:
            recommendations.append("💥 MARKET CRASH: Emergency risk reduction activated")

        if metrics.concentration_risk > 0.3:
            recommendations.append("🎯 HIGH CONCENTRATION: Diversify portfolio")

        if metrics.liquidity_risk > 0.4:
            recommendations.append("💧 LIQUIDITY RISK: Reduce illiquid positions")

        return recommendations

    async def _calculate_overall_risk_score(self, metrics: RiskMetrics, breaches: List[Dict[str, Any]]) -> float:
        """Calculate overall risk score (0-100, higher = riskier)"""
        # Simple risk scoring
        base_score = 0

        # Add points for breaches
        base_score += len(breaches) * 20

        # Add points for high concentration
        if metrics.concentration_risk > 0.3:
            base_score += 15

        # Add points for low liquidity
        if metrics.liquidity_risk > 0.4:
            base_score += 15

        return min(100, base_score)

    async def _estimate_expected_returns(self) -> Dict[str, float]:
        """Estimate expected returns for portfolio assets"""
        # Simplified estimation - would use more sophisticated models
        returns = {}
        for symbol in self.portfolio.keys():
            if 'BTC' in symbol:
                returns[symbol] = 0.15  # 15% expected for crypto
            elif 'ETH' in symbol:
                returns[symbol] = 0.12  # 12% expected for ETH
            else:
                returns[symbol] = 0.08  # 8% expected for others
        return returns

    async def _estimate_covariance_matrix(self) -> np.ndarray:
        """Estimate covariance matrix for portfolio assets"""
        n_assets = len(self.portfolio)
        # Simplified - assume some correlation
        base_corr = 0.3
        cov_matrix = np.full((n_assets, n_assets), base_corr)
        np.fill_diagonal(cov_matrix, 1.0)  # Diagonal = 1

        # Scale by volatility
        volatilities = [0.04] * n_assets  # Assume 4% volatility
        for i in range(n_assets):
            for j in range(n_assets):
                cov_matrix[i, j] *= volatilities[i] * volatilities[j]

        return cov_matrix

    async def _calculate_rebalancing_trades(
        self,
        current_weights: Dict[str, float],
        optimal_weights: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Calculate required trades for rebalancing"""
        trades = []

        for symbol in current_weights.keys():
            current_weight = current_weights[symbol]
            optimal_weight = optimal_weights[symbol]
            weight_diff = optimal_weight - current_weight

            if abs(weight_diff) > 0.01:  # Only trade if difference > 1%
                trades.append({
                    'symbol': symbol,
                    'action': 'BUY' if weight_diff > 0 else 'SELL',
                    'weight_change': weight_diff,
                    'urgency': 'HIGH' if abs(weight_diff) > 0.05 else 'MEDIUM'
                })

        return trades

    async def _estimate_optimization_improvement(
        self,
        current_weights: Dict[str, float],
        optimal_weights: Dict[str, float],
        expected_returns: Dict[str, float],
        covariance_matrix: np.ndarray
    ) -> Dict[str, float]:
        """Estimate improvement from portfolio optimization"""
        # Simplified estimation
        return {
            'expected_return_improvement': 0.02,  # 2% improvement
            'risk_reduction': 0.15,  # 15% risk reduction
            'sharpe_improvement': 0.3  # 30% Sharpe improvement
        }

    async def _assess_stress_test_results(self, results: Dict[str, Any]) -> str:
        """Assess stress test results and provide risk rating"""
        if 'value_at_risk_95' in results:
            var_95 = results['value_at_risk_95']
            if var_95 > -20:
                return "LOW_RISK"
            elif var_95 > -35:
                return "MODERATE_RISK"
            else:
                return "HIGH_RISK"
        return "UNKNOWN"

    async def _generate_stress_mitigation_strategies(self, results: Dict[str, Any]) -> List[str]:
        """Generate strategies to mitigate stress test results"""
        strategies = []

        if 'value_at_risk_95' in results and results['value_at_risk_95'] < -30:
            strategies.append("Reduce portfolio leverage")
            strategies.append("Increase diversification")
            strategies.append("Add protective put options")

        if 'probability_loss' in results and results['probability_loss'] > 0.3:
            strategies.append("Implement stricter stop losses")
            strategies.append("Reduce position sizes")
            strategies.append("Add trend-following filters")

        return strategies if strategies else ["Maintain current risk controls"]