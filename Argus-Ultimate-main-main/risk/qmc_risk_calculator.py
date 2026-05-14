"""
Quantum Monte Carlo Risk Calculator for Argus.

Uses Sobol Quasi-Monte Carlo for faster-converging VaR/CVaR estimation.
This is quantum-inspired variance reduction, not hardware quantum computing.

Benefits over classical Monte Carlo:
- O(1/N) convergence vs O(1/√N)
- Better coverage of tail events
- More stable estimates with fewer samples
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import QMC VaR/CVaR
try:
    from quantum.algorithms.quantum_monte_carlo import run as _qmc_run
    QMC_AVAILABLE = True
except ImportError:
    QMC_AVAILABLE = False
    logger.warning("Quantum Monte Carlo not available - using classical fallback")


@dataclass
class RiskEstimate:
    """Risk estimation result."""
    var_95: float  # Value at Risk at 95% confidence
    cvar_95: float  # Conditional VaR (Expected Shortfall) at 95%
    var_99: float  # Value at Risk at 99% confidence
    cvar_99: float  # Conditional VaR at 99%
    method: str  # "sobol_qmc" or "classical"
    expected_shortfall_bps: float  # CVaR in basis points
    confidence_interval: tuple  # (lower, upper) for VaR estimate


class QMCRiskCalculator:
    """
    Quantum Monte Carlo Risk Calculator.
    
    Uses Sobol low-discrepancy sequences for VaR/CVaR estimation
    with better convergence than classical Monte Carlo.
    """
    
    def __init__(self, n_samples: int = 10000):
        """
        Initialize QMC Risk Calculator.
        
        Args:
            n_samples: Number of quasi-random samples (higher = more accurate)
        """
        self.n_samples = n_samples
        self._history: List[RiskEstimate] = []
    
    def calculate_var_cvar(
        self,
        returns: np.ndarray,
        portfolio_value: float = 1.0,
    ) -> RiskEstimate:
        """
        Calculate VaR and CVaR using Quasi-Monte Carlo.
        
        Args:
            returns: Array of historical returns (e.g., daily pct changes)
            portfolio_value: Current portfolio value for dollar conversion
            
        Returns:
            RiskEstimate with VaR/CVaR at 95% and 99% confidence
        """
        returns = np.asarray(returns, dtype=float).ravel()
        
        if len(returns) < 2:
            return RiskEstimate(
                var_95=0.0, cvar_95=0.0,
                var_99=0.0, cvar_99=0.0,
                method="insufficient_data",
                expected_shortfall_bps=0.0,
                confidence_interval=(0.0, 0.0),
            )
        
        # Calculate VaR/CVaR at 95%
        result_95 = self._run_qmc(returns, confidence=0.95)
        
        # Calculate VaR/CVaR at 99%
        result_99 = self._run_qmc(returns, confidence=0.99)
        
        estimate = RiskEstimate(
            var_95=abs(result_95['var']) * portfolio_value,
            cvar_95=abs(result_95['cvar']) * portfolio_value,
            var_99=abs(result_99['var']) * portfolio_value,
            cvar_99=abs(result_99['cvar']) * portfolio_value,
            method=result_95.get('method', 'unknown'),
            expected_shortfall_bps=result_95.get('expected_shortfall_bps', 0.0),
            confidence_interval=(
                abs(result_95.get('empirical_var', 0.0)) * portfolio_value,
                abs(result_95['var']) * portfolio_value,
            ),
        )
        
        self._history.append(estimate)
        return estimate
    
    def _run_qmc(self, returns: np.ndarray, confidence: float) -> Dict[str, Any]:
        """Run QMC VaR/CVaR calculation."""
        if QMC_AVAILABLE:
            try:
                return _qmc_run(
                    returns,
                    n_samples=self.n_samples,
                    confidence=confidence,
                )
            except Exception as e:
                logger.debug(f"QMC failed, using classical: {e}")
        
        # Classical fallback
        return self._classical_var_cvar(returns, confidence)
    
    def _classical_var_cvar(
        self,
        returns: np.ndarray,
        confidence: float,
    ) -> Dict[str, Any]:
        """Classical VaR/CVaR fallback."""
        alpha = 1.0 - confidence
        var = float(np.percentile(returns, alpha * 100.0))
        tail = returns[returns <= var]
        cvar = float(np.mean(tail)) if len(tail) > 0 else var
        
        return {
            'var': var,
            'cvar': cvar,
            'method': 'classical',
            'expected_shortfall_bps': -cvar * 1e4 if cvar < 0 else 0.0,
            'empirical_var': var,
        }
    
    def get_position_size(
        self,
        returns: np.ndarray,
        portfolio_value: float,
        max_risk_pct: float = 0.02,
    ) -> float:
        """
        Calculate safe position size based on QMC risk estimate.
        
        Args:
            returns: Historical returns for the asset
            portfolio_value: Current portfolio value
            max_risk_pct: Maximum risk as percentage of portfolio (default 2%)
            
        Returns:
            Maximum position size in dollars
        """
        estimate = self.calculate_var_cvar(returns, portfolio_value)
        
        if estimate.cvar_95 <= 0:
            return portfolio_value * max_risk_pct
        
        # Size position so that CVaR equals max_risk_pct of portfolio
        max_loss = portfolio_value * max_risk_pct
        position_size = max_loss / (estimate.cvar_95 / portfolio_value) if estimate.cvar_95 > 0 else 0
        
        return min(position_size, portfolio_value * 0.5)  # Cap at 50% of portfolio
    
    def get_risk_report(self, returns: np.ndarray, portfolio_value: float) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        estimate = self.calculate_var_cvar(returns, portfolio_value)
        
        return {
            'var_95_pct': (estimate.var_95 / portfolio_value * 100) if portfolio_value > 0 else 0,
            'var_95_dollar': estimate.var_95,
            'cvar_95_pct': (estimate.cvar_95 / portfolio_value * 100) if portfolio_value > 0 else 0,
            'cvar_95_dollar': estimate.cvar_95,
            'var_99_pct': (estimate.var_99 / portfolio_value * 100) if portfolio_value > 0 else 0,
            'cvar_99_pct': (estimate.cvar_99 / portfolio_value * 100) if portfolio_value > 0 else 0,
            'expected_shortfall_bps': estimate.expected_shortfall_bps,
            'method': estimate.method,
            'qmc_available': QMC_AVAILABLE,
            'samples_used': self.n_samples,
        }


# Singleton instance for easy use
_default_calculator: Optional[QMCRiskCalculator] = None


def get_qmc_risk_calculator(n_samples: int = 10000) -> QMCRiskCalculator:
    """Get or create the default QMC risk calculator."""
    global _default_calculator
    if _default_calculator is None:
        _default_calculator = QMCRiskCalculator(n_samples=n_samples)
    return _default_calculator
