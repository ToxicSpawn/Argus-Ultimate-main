"""
Causal Intelligence Engine v2.0
================================
Causal inference for strategy attribution and market understanding.

Provides:
- Granger causality testing
- Counterfactual analysis
- Strategy attribution
- Lead-lag detection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class CausalRelationship:
    """Detected causal relationship."""
    cause: str
    effect: str
    lag: int
    strength: float  # 0 to 1
    p_value: float
    confidence: float
    method: str
    timestamp: datetime


@dataclass
class CounterfactualResult:
    """Result of counterfactual analysis."""
    scenario: str
    actual_outcome: float
    counterfactual_outcome: float
    treatment_effect: float
    confidence_interval: Tuple[float, float]
    method: str


@dataclass
class AttributionResult:
    """Strategy performance attribution."""
    strategy_name: str
    total_return: float
    factor_attributions: Dict[str, float]
    alpha: float
    beta: float
    r_squared: float
    residual: float
    timestamp: datetime


@dataclass
class LeadLagSignal:
    """Lead-lag relationship signal."""
    leading_asset: str
    lagging_asset: str
    optimal_lag: int
    correlation_at_lag: float
    lead_strength: float  # 0 to 1
    p_value: float


class GrangerCausalityTester:
    """
    Tests for Granger causality between time series.
    
    Granger causality: X Granger-causes Y if past values of X
    help predict Y beyond what past values of Y alone can predict.
    """
    
    def __init__(self, max_lag: int = 20) -> None:
        """
        Initialize Granger causality tester.
        
        Args:
            max_lag: Maximum lag to test
        """
        self.max_lag = max_lag
    
    def test_granger_causality(
        self,
        cause_series: np.ndarray,
        effect_series: np.ndarray,
        max_lag: Optional[int] = None
    ) -> CausalRelationship:
        """
        Test if cause_series Granger-causes effect_series.
        
        Args:
            cause_series: Potential cause time series
            effect_series: Effect time series
            max_lag: Maximum lag to test
            
        Returns:
            CausalRelationship with test results
        """
        if max_lag is None:
            max_lag = self.max_lag
        
        # Align series
        min_len = min(len(cause_series), len(effect_series))
        cause = cause_series[-min_len:]
        effect = effect_series[-min_len:]
        
        if min_len < max_lag + 10:
            return CausalRelationship(
                cause="unknown",
                effect="unknown",
                lag=0,
                strength=0.0,
                p_value=1.0,
                confidence=0.0,
                method="granger",
                timestamp=datetime.now()
            )
        
        best_lag = 0
        best_f_stat = 0.0
        best_p_value = 1.0
        
        # Test different lags
        for lag in range(1, min(max_lag, min_len // 4)):
            try:
                f_stat, p_value = self._granger_test_single_lag(cause, effect, lag)
                
                if p_value < best_p_value:
                    best_lag = lag
                    best_f_stat = f_stat
                    best_p_value = p_value
            except Exception:
                continue
        
        # Calculate strength (normalized F-statistic)
        strength = min(1.0, best_f_stat / 20.0)
        confidence = 1.0 - best_p_value
        
        return CausalRelationship(
            cause="cause_series",
            effect="effect_series",
            lag=best_lag,
            strength=strength,
            p_value=best_p_value,
            confidence=confidence,
            method="granger",
            timestamp=datetime.now()
        )
    
    def _granger_test_single_lag(
        self,
        cause: np.ndarray,
        effect: np.ndarray,
        lag: int
    ) -> Tuple[float, float]:
        """
        Perform Granger test for a single lag.
        
        Returns F-statistic and p-value.
        """
        n = len(effect)
        
        # Restricted model: AR(lag) on effect only
        Y = effect[lag:]
        X_restricted = np.column_stack([
            effect[lag - i - 1:n - i - 1] for i in range(lag)
        ])
        
        # Unrestricted model: AR(lag) on effect + lagged cause
        X_unrestricted = np.column_stack([
            X_restricted,
            *[cause[lag - i - 1:n - i - 1].reshape(-1, 1) for i in range(lag)]
        ])
        
        # Add constant
        X_restricted = np.column_stack([np.ones(len(Y)), X_restricted])
        X_unrestricted = np.column_stack([np.ones(len(Y)), X_unrestricted])
        
        # Fit models
        beta_restricted = np.linalg.lstsq(X_restricted, Y, rcond=None)[0]
        beta_unrestricted = np.linalg.lstsq(X_unrestricted, Y, rcond=None)[0]
        
        # Calculate residuals
        residuals_restricted = Y - X_restricted @ beta_restricted
        residuals_unrestricted = Y - X_unrestricted @ beta_unrestricted
        
        # Sum of squared residuals
        ssr_restricted = np.sum(residuals_restricted ** 2)
        ssr_unrestricted = np.sum(residuals_unrestricted ** 2)
        
        # F-test
        df1 = lag  # Number of restrictions
        df2 = n - 2 * lag - 1  # Degrees of freedom
        
        if ssr_unrestricted == 0 or df2 <= 0:
            return 0.0, 1.0
        
        f_stat = ((ssr_restricted - ssr_unrestricted) / df1) / (ssr_unrestricted / df2)
        p_value = 1.0 - stats.f.cdf(f_stat, df1, df2)
        
        return f_stat, p_value
    
    def find_lead_lag_relationships(
        self,
        series_dict: Dict[str, np.ndarray],
        max_lag: int = 10
    ) -> List[LeadLagSignal]:
        """
        Find lead-lag relationships between multiple series.
        
        Args:
            series_dict: Dictionary of series name -> values
            max_lag: Maximum lag to test
            
        Returns:
            List of LeadLagSignal
        """
        signals = []
        names = list(series_dict.keys())
        
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                # Test A -> B
                result_ab = self.test_granger_causality(
                    series_dict[name_a],
                    series_dict[name_b],
                    max_lag
                )
                
                if result_ab.p_value < 0.05:
                    signals.append(LeadLagSignal(
                        leading_asset=name_a,
                        lagging_asset=name_b,
                        optimal_lag=result_ab.lag,
                        correlation_at_lag=self._lagged_correlation(
                            series_dict[name_a],
                            series_dict[name_b],
                            result_ab.lag
                        ),
                        lead_strength=result_ab.strength,
                        p_value=result_ab.p_value
                    ))
                
                # Test B -> A
                result_ba = self.test_granger_causality(
                    series_dict[name_b],
                    series_dict[name_a],
                    max_lag
                )
                
                if result_ba.p_value < 0.05:
                    signals.append(LeadLagSignal(
                        leading_asset=name_b,
                        lagging_asset=name_a,
                        optimal_lag=result_ba.lag,
                        correlation_at_lag=self._lagged_correlation(
                            series_dict[name_b],
                            series_dict[name_a],
                            result_ba.lag
                        ),
                        lead_strength=result_ba.strength,
                        p_value=result_ba.p_value
                    ))
        
        return signals
    
    def _lagged_correlation(
        self,
        series_a: np.ndarray,
        series_b: np.ndarray,
        lag: int
    ) -> float:
        """Calculate correlation at a specific lag."""
        if lag == 0:
            min_len = min(len(series_a), len(series_b))
            if min_len < 2:
                return 0.0
            return float(np.corrcoef(series_a[-min_len:], series_b[-min_len:])[0, 1])
        
        if lag > 0:
            a = series_a[:-lag] if lag > 0 else series_a
            b = series_b[lag:]
        else:
            a = series_a[-lag:]
            b = series_b[:lag]
        
        min_len = min(len(a), len(b))
        if min_len < 2:
            return 0.0
        
        corr = np.corrcoef(a[-min_len:], b[-min_len:])[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0


class CounterfactualAnalyzer:
    """
    Performs counterfactual analysis for strategy evaluation.
    
    "What would have happened if we had done X instead of Y?"
    """
    
    def __init__(self) -> None:
        """Initialize counterfactual analyzer."""
        self._analyses: List[CounterfactualResult] = []
    
    def analyze_treatment_effect(
        self,
        treated_returns: np.ndarray,
        control_returns: np.ndarray,
        treatment_start: int,
        scenario_name: str = "treatment"
    ) -> CounterfactualResult:
        """
        Analyze treatment effect using difference-in-differences.
        
        Args:
            treated_returns: Returns for treated group/strategy
            control_returns: Returns for control group/strategy
            treatment_start: Index where treatment began
            scenario_name: Name of the scenario
            
        Returns:
            CounterfactualResult
        """
        # Pre-treatment period
        pre_treated = treated_returns[:treatment_start]
        pre_control = control_returns[:treatment_start]
        
        # Post-treatment period
        post_treated = treated_returns[treatment_start:]
        post_control = control_returns[treatment_start:]
        
        if len(pre_treated) < 2 or len(post_treated) < 2:
            return CounterfactualResult(
                scenario=scenario_name,
                actual_outcome=0.0,
                counterfactual_outcome=0.0,
                treatment_effect=0.0,
                confidence_interval=(0.0, 0.0),
                method="diff_in_diff"
            )
        
        # Difference-in-differences
        pre_diff = np.mean(pre_treated) - np.mean(pre_control)
        post_diff = np.mean(post_treated) - np.mean(post_control)
        did = post_diff - pre_diff
        
        # Actual outcome (post-treatment treated mean)
        actual = np.mean(post_treated)
        
        # Counterfactual (what would have happened without treatment)
        counterfactual = actual - did
        
        # Confidence interval (simplified)
        se = np.std(post_treated - post_control) / np.sqrt(len(post_treated))
        ci_lower = did - 1.96 * se
        ci_upper = did + 1.96 * se
        
        result = CounterfactualResult(
            scenario=scenario_name,
            actual_outcome=float(actual),
            counterfactual_outcome=float(counterfactual),
            treatment_effect=float(did),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            method="diff_in_diff"
        )
        
        self._analyses.append(result)
        return result
    
    def analyze_counterfactual_scenario(
        self,
        actual_returns: np.ndarray,
        counterfactual_returns: np.ndarray,
        scenario_name: str = "scenario"
    ) -> CounterfactualResult:
        """
        Analyze a counterfactual scenario.
        
        Args:
            actual_returns: Actual returns achieved
            counterfactual_returns: Returns under counterfactual scenario
            scenario_name: Name of the scenario
            
        Returns:
            CounterfactualResult
        """
        actual_total = np.sum(actual_returns)
        counterfactual_total = np.sum(counterfactual_returns)
        
        treatment_effect = actual_total - counterfactual_total
        
        # Confidence interval based on return variance
        if len(actual_returns) > 1:
            se = np.std(actual_returns - counterfactual_returns) / np.sqrt(len(actual_returns))
            ci_lower = treatment_effect - 1.96 * se * np.sqrt(len(actual_returns))
            ci_upper = treatment_effect + 1.96 * se * np.sqrt(len(actual_returns))
        else:
            ci_lower = ci_upper = treatment_effect
        
        result = CounterfactualResult(
            scenario=scenario_name,
            actual_outcome=float(actual_total),
            counterfactual_outcome=float(counterfactual_total),
            treatment_effect=float(treatment_effect),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            method="counterfactual"
        )
        
        self._analyses.append(result)
        return result


class StrategyAttributor:
    """
    Attributes strategy performance to various factors.
    """
    
    def __init__(self) -> None:
        """Initialize strategy attributor."""
        self._attributions: List[AttributionResult] = []
    
    def attribute_performance(
        self,
        strategy_returns: np.ndarray,
        factor_returns: Dict[str, np.ndarray],
        strategy_name: str = "strategy"
    ) -> AttributionResult:
        """
        Attribute strategy returns to factors.
        
        Uses regression to decompose returns into factor exposures.
        
        Args:
            strategy_returns: Strategy return series
            factor_returns: Dictionary of factor name -> returns
            strategy_name: Name of the strategy
            
        Returns:
            AttributionResult
        """
        # Align all series
        min_len = len(strategy_returns)
        for factor_rets in factor_returns.values():
            min_len = min(min_len, len(factor_rets))
        
        if min_len < 10:
            return AttributionResult(
                strategy_name=strategy_name,
                total_return=float(np.sum(strategy_returns)),
                factor_attributions={},
                alpha=0.0,
                beta=0.0,
                r_squared=0.0,
                residual=0.0,
                timestamp=datetime.now()
            )
        
        y = strategy_returns[-min_len:]
        
        # Build factor matrix
        factor_names = list(factor_returns.keys())
        X = np.column_stack([
            factor_returns[name][-min_len:] for name in factor_names
        ])
        
        # Add constant for alpha
        X = np.column_stack([np.ones(min_len), X])
        
        # Regression
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            
            # Predictions and residuals
            y_pred = X @ beta
            residuals = y - y_pred
            
            # R-squared
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            # Factor attributions
            alpha = beta[0]
            factor_betas = beta[1:]
            
            factor_attributions = {}
            for i, name in enumerate(factor_names):
                factor_contribution = factor_betas[i] * np.mean(factor_returns[name][-min_len:])
                factor_attributions[name] = float(factor_contribution)
            
            # Total factor contribution
            total_factor = sum(factor_attributions.values())
            
            attribution = AttributionResult(
                strategy_name=strategy_name,
                total_return=float(np.sum(y)),
                factor_attributions=factor_attributions,
                alpha=float(alpha * min_len),  # Annualized alpha contribution
                beta=float(np.mean(factor_betas)),
                r_squared=float(r_squared),
                residual=float(np.sum(residuals)),
                timestamp=datetime.now()
            )
            
            self._attributions.append(attribution)
            return attribution
            
        except Exception as e:
            logger.error("Attribution failed: %s", e)
            return AttributionResult(
                strategy_name=strategy_name,
                total_return=float(np.sum(y)),
                factor_attributions={},
                alpha=0.0,
                beta=0.0,
                r_squared=0.0,
                residual=0.0,
                timestamp=datetime.now()
            )
    
    def calculate_alpha(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        risk_free_rate: float = 0.05
    ) -> float:
        """
        Calculate alpha (excess return adjusted for market exposure).
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark/market returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Alpha value
        """
        min_len = min(len(strategy_returns), len(benchmark_returns))
        
        if min_len < 10:
            return 0.0
        
        strategy = strategy_returns[-min_len:]
        benchmark = benchmark_returns[-min_len:]
        
        # Daily risk-free rate
        rf_daily = risk_free_rate / 365
        
        # Excess returns
        strategy_excess = strategy - rf_daily
        benchmark_excess = benchmark - rf_daily
        
        # Regression: strategy_excess = alpha + beta * benchmark_excess
        X = np.column_stack([np.ones(min_len), benchmark_excess])
        beta = np.linalg.lstsq(X, strategy_excess, rcond=None)[0]
        
        # Alpha (daily)
        alpha_daily = beta[0]
        
        # Annualize
        alpha_annual = alpha_daily * 365
        
        return float(alpha_annual)


class CausalIntelligenceEngine:
    """
    Main causal intelligence engine for Argus.
    
    Combines Granger causality, counterfactual analysis,
    and strategy attribution for causal understanding.
    """
    
    def __init__(self) -> None:
        """Initialize causal intelligence engine."""
        self.granger_tester = GrangerCausalityTester()
        self.counterfactual_analyzer = CounterfactualAnalyzer()
        self.strategy_attributor = StrategyAttributor()
        
        logger.info("CausalIntelligenceEngine initialized")
    
    def analyze_causality(
        self,
        series_dict: Dict[str, np.ndarray],
        max_lag: int = 10
    ) -> List[LeadLagSignal]:
        """
        Analyze causal relationships between series.
        
        Args:
            series_dict: Dictionary of series name -> values
            max_lag: Maximum lag to test
            
        Returns:
            List of LeadLagSignal
        """
        return self.granger_tester.find_lead_lag_relationships(series_dict, max_lag)
    
    def what_if_analysis(
        self,
        actual_returns: np.ndarray,
        alternative_returns: np.ndarray,
        scenario_name: str = "alternative"
    ) -> CounterfactualResult:
        """
        Perform what-if analysis.
        
        Args:
            actual_returns: Actual returns
            alternative_returns: Returns under alternative scenario
            scenario_name: Name of scenario
            
        Returns:
            CounterfactualResult
        """
        return self.counterfactual_analyzer.analyze_counterfactual_scenario(
            actual_returns, alternative_returns, scenario_name
        )
    
    def attribute_strategy(
        self,
        strategy_returns: np.ndarray,
        factors: Dict[str, np.ndarray],
        strategy_name: str = "strategy"
    ) -> AttributionResult:
        """
        Attribute strategy performance to factors.
        
        Args:
            strategy_returns: Strategy returns
            factors: Factor returns
            strategy_name: Strategy name
            
        Returns:
            AttributionResult
        """
        return self.strategy_attributor.attribute_performance(
            strategy_returns, factors, strategy_name
        )
    
    def get_causal_insights(
        self,
        asset_returns: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Get comprehensive causal insights.
        
        Args:
            asset_returns: Dictionary of asset returns
            
        Returns:
            Causal insights summary
        """
        # Find lead-lag relationships
        lead_lag = self.analyze_causality(asset_returns)
        
        # Identify strongest relationships
        strong_signals = [s for s in lead_lag if s.lead_strength > 0.5]
        
        return {
            "n_relationships_found": len(lead_lag),
            "n_strong_relationships": len(strong_signals),
            "lead_lag_signals": [
                {
                    "leading": s.leading_asset,
                    "lagging": s.lagging_asset,
                    "optimal_lag": s.optimal_lag,
                    "strength": s.lead_strength,
                    "p_value": s.p_value
                }
                for s in strong_signals[:10]
            ],
            "timestamp": datetime.now().isoformat()
        }
