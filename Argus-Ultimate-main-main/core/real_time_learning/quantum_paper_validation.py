"""
Quantum-Enhanced Paper Validation Engine

This component validates all adaptive changes with quantum-assisted paper trading
before live deployment. It extends the classical validation engine with:
- Quantum portfolio optimization (QAOA)
- Quantum risk analysis (QAE)
- Quantum strategy optimization (QNN)
- Quantum execution routing
"""

from __future__ import annotations
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# Quantum imports
from quantum.canonical import get_quantum_facade
from core.real_time_learning.paper_validation import (
    PaperValidationEngine as ClassicalPaperValidationEngine,
    ValidationResult as ClassicalValidationResult
)

logger = logging.getLogger(__name__)


@dataclass
class QuantumValidationResult(ClassicalValidationResult):
    """Extends classical validation result with quantum metadata"""
    quantum_metadata: Optional[Dict[str, Any]] = None
    
    def get_quantum_improvement(self) -> float:
        """Calculate quantum advantage over classical validation"""
        if not self.quantum_metadata or 'classical_baseline' not in self.quantum_metadata:
            return 0.0
            
        # Try different quantum metrics
        quantum_score = self.metrics.get('quantum_sharpe', 
                        self.metrics.get('quantum_score', 
                        self.metrics.get('quantum_improvement', 0)))
        
        classical_score = self.quantum_metadata['classical_baseline'].get('score', 0)
        
        if classical_score == 0:
            return 0.0
            
        # Handle case where quantum_score is already a relative improvement
        if 'quantum_improvement' in self.metrics:
            return self.metrics['quantum_improvement']
            
        return (quantum_score - classical_score) / abs(classical_score)


class QuantumPaperValidationEngine(ClassicalPaperValidationEngine):
    """Quantum-enhanced validation engine for adaptive changes"""
    
    def __init__(self):
        super().__init__()
        self.name = "quantum_paper_validation"
        self.version = "2.0"
        self.quantum_facade = get_quantum_facade()
        
        # Quantum-specific thresholds
        self.quantum_thresholds = {
            'min_quantum_improvement': 0.05,  # 5% improvement required
            'min_quantum_confidence': 0.7,   # 70% confidence in quantum advantage
            'max_quantum_risk': 0.02          # Max 2% additional risk from quantum
        }
    
    def _run_quantum_portfolio_optimization(self, component_name: str, proposed_changes: Dict[str, Any], 
                                           data: Dict[str, Any]) -> Dict[str, Any]:
        """Run quantum portfolio optimization for correlation matrix changes"""
        if component_name != 'correlation_matrix':
            return {}
            
        # Extract correlation matrix
        if 'current_matrix' not in proposed_changes:
            return {"error": "No correlation matrix in proposed changes"}
            
        matrix = proposed_changes['current_matrix']
        
        # Convert correlation matrix to covariance matrix for QAOA
        # This is a simplified approach - real implementation would use actual returns
        n_assets = len(matrix)
        std_devs = np.ones(n_assets)  # Assume unit std dev for demo
        
        # Convert dict correlation matrix to numpy array
        asset_pairs = list(matrix.keys())
        assets = set()
        for pair in asset_pairs:
            parts = pair.split(':')
            if len(parts) == 2:
                assets.add(parts[0])
                assets.add(parts[1])
        assets = sorted(list(assets))
        
        # Only proceed if we have assets
        if not assets:
            return {"error": "No assets found in correlation matrix"}
            
        # Create full correlation matrix
        corr_matrix = np.eye(len(assets))
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i != j:
                    key1 = f"{asset1}:{asset2}"
                    key2 = f"{asset2}:{asset1}"
                    corr_matrix[i,j] = matrix.get(key1, matrix.get(key2, 0.5))
                    corr_matrix[j,i] = corr_matrix[i,j]
        
        # Make sure std_devs has the right shape
        if len(std_devs) != len(assets):
            std_devs = np.ones(len(assets))
            
        # Convert to numpy arrays if they aren't already
        if not isinstance(std_devs, np.ndarray):
            std_devs = np.array(std_devs)
            
        if not isinstance(corr_matrix, np.ndarray):
            corr_matrix = np.array(corr_matrix)
            
        covariance = np.outer(std_devs, std_devs) * corr_matrix
        
        # Run quantum portfolio optimization
        try:
            result = self.quantum_facade.optimize_portfolio(
                expected_returns=np.ones(n_assets),  # Equal expected returns for demo
                covariance_matrix=covariance,
                method="qaoa_in_repo_simulator",
                n_layers=3,
                shots=1024
            )
            
            # Extract quantum results
            quantum_weights = result['optimal_weights']
            quantum_score = result['optimal_value']
            
            # Calculate diversification score
            if isinstance(quantum_weights, list):
                quantum_weights = np.array(quantum_weights)
                
            portfolio_var = np.dot(quantum_weights, np.dot(covariance, quantum_weights))
            diversification_score = 1.0 / (1.0 + portfolio_var)
            
            return {
                'quantum_weights': quantum_weights,
                'quantum_score': quantum_score,
                'diversification_score': diversification_score,
                'portfolio_variance': portfolio_var,
                'quantum_metadata': result.get('metadata', {})
            }
        except Exception as e:
            logger.warning("Quantum portfolio optimization failed: %s", str(e))
            return {"error": f"Quantum optimization failed: {str(e)}"}
    
    def _run_quantum_risk_analysis(self, component_name: str, proposed_changes: Dict[str, Any], 
                                 data: Dict[str, Any]) -> Dict[str, Any]:
        """Run quantum risk analysis for all component types"""
        # We need historical returns for quantum risk analysis
        # For this demo, we'll simulate some returns
        n_assets = 3  # BTC, ETH, SOL
        n_days = 100
        
        # Simulate returns based on regime
        regime = data.get('market_data', {}).get('regime', 'stable')
        
        if regime == 'volatile':
            returns = np.random.normal(0.001, 0.03, (n_days, n_assets))
        elif regime == 'trending':
            returns = np.random.normal(0.002, 0.02, (n_days, n_assets))
        else:  # stable or range
            returns = np.random.normal(0.0005, 0.01, (n_days, n_assets))
        
        try:
            # Run quantum VaR estimation
            var_result = self.quantum_facade.estimate_tail_risk_qmc(
                returns=returns,
                confidence_level=0.95,
                method="quantum_inspired_classical_sobol"
            )
            
            return {
                'quantum_var': var_result['var'],
                'quantum_cvar': var_result['cvar'],
                'quantum_risk_metadata': var_result.get('metadata', {})
            }
        except Exception as e:
            logger.warning("Quantum risk analysis failed: %s", str(e))
            return {"error": f"Quantum risk analysis failed: {str(e)}"}
    
    def _run_quantum_strategy_analysis(self, component_name: str, proposed_changes: Dict[str, Any], 
                                      data: Dict[str, Any]) -> Dict[str, Any]:
        """Run quantum strategy analysis for strategy allocator changes"""
        if component_name != 'strategy_allocator':
            return {}
            
        if 'strategy_weights' not in proposed_changes:
            return {"error": "No strategy weights in proposed changes"}
            
        weights = proposed_changes['strategy_weights']
        
        # Simulate strategy returns based on weights and regime
        regime = data.get('market_data', {}).get('regime', 'stable')
        n_days = 100
        
        # Create simulated returns for each strategy
        strategies = list(weights.keys())
        n_strategies = len(strategies)
        
        if regime == 'volatile':
            strategy_returns = np.random.normal(0.0015, 0.04, (n_days, n_strategies))
        elif regime == 'trending':
            strategy_returns = np.random.normal(0.0025, 0.03, (n_days, n_strategies))
        else:  # stable or range
            strategy_returns = np.random.normal(0.001, 0.02, (n_days, n_strategies))
        
        # Apply weights to get portfolio returns
        portfolio_returns = strategy_returns.dot(np.array(list(weights.values())))
        
        try:
            # Run quantum kernel analysis for strategy interactions
            kernel_result = self.quantum_facade.quantum_kernel_analysis(
                data=strategy_returns,
                method="quantum_kernel_svm"
            )
            
            # Calculate quantum-enhanced Sharpe ratio
            sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns)
            quantum_sharpe = sharpe_ratio * (1 + kernel_result.get('quantum_boost', 0))
            
            return {
                'quantum_sharpe': quantum_sharpe,
                'quantum_kernel_metadata': kernel_result.get('metadata', {})
            }
        except Exception as e:
            logger.warning("Quantum strategy analysis failed: %s", str(e))
            return {"error": f"Quantum strategy analysis failed: {str(e)}"}
    
    def learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate proposed parameter changes with quantum-enhanced paper trading"""
        
        # First run classical validation
        classical_result = super().learn(data)
        
        if classical_result["status"] != "success":
            return classical_result
            
        component_name = data['component_name']
        validation_result = classical_result["validation_result"]
        backtest_results = classical_result["backtest_results"]
        
        # Run quantum enhancements
        quantum_results = {}
        
        if component_name == 'correlation_matrix':
            quantum_results = self._run_quantum_portfolio_optimization(
                component_name, data['proposed_changes'], data
            )
        
        if component_name == 'strategy_allocator':
            quantum_results.update(self._run_quantum_strategy_analysis(
                component_name, data['proposed_changes'], data
            ))
        
        # Run quantum risk analysis for all components
        risk_results = self._run_quantum_risk_analysis(
            component_name, data['proposed_changes'], data
        )
        quantum_results.update(risk_results)

        # Update validation result with quantum enhancements
        if quantum_results and 'error' not in quantum_results:
            # Update metrics with quantum results
            updated_metrics = validation_result.metrics.copy()
            
            # Add quantum-specific metrics
            if 'quantum_score' in quantum_results:
                updated_metrics['quantum_score'] = quantum_results['quantum_score']
            
            if 'quantum_sharpe' in quantum_results:
                updated_metrics['quantum_sharpe'] = quantum_results['quantum_sharpe']
                # Calculate quantum improvement
                classical_sharpe = validation_result.metrics.get('sharpe_ratio', 1e-9)  # Avoid division by zero
                if classical_sharpe > 1e-9:
                    quantum_improvement = (
                        quantum_results['quantum_sharpe'] - classical_sharpe
                    ) / classical_sharpe
                    updated_metrics['quantum_improvement'] = quantum_improvement
                else:
                    updated_metrics['quantum_improvement'] = 0.0
            
            if 'quantum_var' in quantum_results:
                updated_metrics['quantum_var'] = quantum_results['quantum_var']
                # Compare with classical VaR if available
                if 'var' in validation_result.metrics:
                    classical_var = validation_result.metrics['var']
                    if classical_var != 0:
                        updated_metrics['quantum_var_improvement'] = (
                            abs(quantum_results['quantum_var']) - abs(classical_var)
                        ) / abs(classical_var)
            
            if 'diversification_score' in quantum_results:
                updated_metrics['quantum_diversification'] = quantum_results['diversification_score']
                # Compare with classical diversification if available
                if 'diversification_score' in validation_result.metrics:
                    classical_div = validation_result.metrics['diversification_score']
                    if classical_div > 0:
                        updated_metrics['quantum_div_improvement'] = (
                            quantum_results['diversification_score'] - classical_div
                        ) / classical_div

            # Create quantum-enhanced validation result
            quantum_validation = QuantumValidationResult(
                component=validation_result.component,
                parameter_changes=validation_result.parameter_changes,
                test_passed=validation_result.test_passed,
                metrics=updated_metrics,
                required_metrics=validation_result.required_metrics,
                backtest_results=validation_result.backtest_results,
                statistical_results=validation_result.statistical_results,
                quantum_metadata={
                    'quantum_results': quantum_results,
                    'classical_baseline': {
                        'score': validation_result.metrics.get('sharpe_ratio', 0),
                        'diversification': validation_result.metrics.get('diversification_score', 0)
                    }
                }
            )
            
            # Check quantum improvement thresholds
            quantum_improvement = quantum_validation.get_quantum_improvement()
            if isinstance(quantum_improvement, (int, float)) and quantum_improvement < self.quantum_thresholds['min_quantum_improvement']:
                logger.info("Quantum enhancement insufficient (%.2f%%) for %s", 
                          quantum_improvement * 100, component_name)
                quantum_validation.test_passed = False
            
            return {
                "status": "success" if quantum_validation.is_valid() else "failed",
                "validation_result": quantum_validation,
                "backtest_results": backtest_results,
                "quantum_enhanced": True
            }
        
        return classical_result
    
    def get_params(self) -> Dict[str, Any]:
        """Get current parameters including quantum settings"""
        params = super().get_params()
        params.update({
            'quantum_thresholds': self.quantum_thresholds,
            'quantum_facade_status': self.quantum_facade.status().to_dict()
        })
        return params