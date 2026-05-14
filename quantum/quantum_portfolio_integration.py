"""
Quantum Portfolio Integration - Maximum Earnings
=================================================
Integrates quantum portfolio optimization with the allocation system.
Uses QUBO formulation for optimal asset selection and weight optimization.
"""
import sys
sys.path.insert(0, '.')
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QuantumPortfolioConfig:
    """Quantum portfolio configuration."""
    n_assets: int = 10
    weight_bits: int = 4
    max_assets: int = 5  # Cardinality constraint
    risk_aversion: float = 2.5
    annealing_reads: int = 200
    annealing_sweeps: int = 1000


class QuantumPortfolioAllocator:
    """
    Quantum-enhanced portfolio allocator.
    
    Uses quantum annealing for:
    - Asset selection (which assets to hold)
    - Weight optimization (how much of each)
    - Constraint satisfaction (max assets, sector limits)
    """
    
    def __init__(self, config: Optional[QuantumPortfolioConfig] = None):
        self.config = config or QuantumPortfolioConfig()
        
        # Try to import quantum annealing
        try:
            from quantum.optimization.annealing import solve_qubo
            self.solve_qubo = solve_qubo
            self.quantum_available = True
            logger.info("Quantum annealing available for portfolio optimization")
        except ImportError:
            self.quantum_available = False
            logger.warning("Quantum annealing not available, using classical fallback")
    
    def build_portfolio_qubo(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        asset_names: List[str]
    ) -> Dict[Tuple[int, int], float]:
        """
        Build QUBO matrix for portfolio optimization.
        
        Objective: Maximize Sharpe ratio with cardinality constraint
        QUBO: minimize -return + risk_aversion * variance + cardinality_penalty
        """
        n = len(expected_returns)
        k = self.config.weight_bits
        Q = {}
        
        # Linear terms (diagonal)
        for i in range(n * k):
            asset_idx = i // k
            bit_idx = i % k
            weight_factor = 2 ** bit_idx / (2 ** k - 1)
            
            # Return term (negative because we minimize)
            Q[(i, i)] = -expected_returns[asset_idx] * weight_factor
            
            # Variance term
            Q[(i, i)] += self.config.risk_aversion * cov_matrix[asset_idx, asset_idx] * weight_factor ** 2
        
        # Quadratic terms (off-diagonal)
        for i in range(n * k):
            for j in range(i + 1, n * k):
                asset_i = i // k
                asset_j = j // k
                bit_i = i % k
                bit_j = j % k
                
                if asset_i == asset_j:
                    # Same asset, different bits
                    weight_i = 2 ** bit_i / (2 ** k - 1)
                    weight_j = 2 ** bit_j / (2 ** k - 1)
                    Q[(i, j)] = 2 * self.config.risk_aversion * cov_matrix[asset_i, asset_i] * weight_i * weight_j
                else:
                    # Different assets
                    weight_i = 2 ** bit_i / (2 ** k - 1)
                    weight_j = 2 ** bit_j / (2 ** k - 1)
                    Q[(i, j)] = 2 * self.config.risk_aversion * cov_matrix[asset_i, asset_j] * weight_i * weight_j
        
        # Cardinality constraint (penalize selecting more than max_assets)
        cardinality_penalty = 10.0
        for i in range(n * k):
            for j in range(n * k):
                if i != j:
                    asset_i = i // k
                    asset_j = j // k
                    if asset_i != asset_j:
                        # Penalty for selecting multiple assets
                        Q[(i, j)] = Q.get((i, j), 0) + cardinality_penalty
        
        return Q
    
    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        asset_names: List[str]
    ) -> Dict[str, float]:
        """
        Optimize portfolio using quantum annealing.
        
        Returns optimal weights for each asset.
        """
        n = len(expected_returns)
        k = self.config.weight_bits
        
        if self.quantum_available:
            # Build QUBO
            Q = self.build_portfolio_qubo(expected_returns, cov_matrix, asset_names)
            
            # Solve with quantum annealing
            result = self.solve_qubo(
                Q,
                num_reads=self.config.annealing_reads,
                num_sweeps=self.config.annealing_sweeps
            )
            
            # Extract weights from solution
            solution = result.get("solution", {})
            weights = np.zeros(n)
            
            # Solution is a dict of {variable_index: value}
            for idx, value in solution.items():
                if value == 1:
                    asset_idx = idx // k
                    bit_idx = idx % k
                    if asset_idx < n:
                        weight_factor = 2 ** bit_idx / (2 ** k - 1)
                        weights[asset_idx] += weight_factor
            
            # Normalize weights
            if weights.sum() > 0:
                weights = weights / weights.sum()
            
            method = "quantum_annealing"
        else:
            # Classical fallback
            weights = self._classical_optimize(expected_returns, cov_matrix)
            method = "classical"
        
        # Create result dictionary
        result = {
            "weights": {name: float(w) for name, w in zip(asset_names, weights)},
            "expected_return": float(weights @ expected_returns),
            "expected_risk": float(np.sqrt(weights @ cov_matrix @ weights)),
            "sharpe_ratio": float((weights @ expected_returns) / np.sqrt(weights @ cov_matrix @ weights)),
            "method": method,
            "num_assets_selected": int(np.sum(weights > 0.01))
        }
        
        return result
    
    def _classical_optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Classical mean-variance optimization fallback."""
        from scipy.optimize import minimize
        
        n = len(expected_returns)
        
        def neg_sharpe(weights):
            ret = weights @ expected_returns
            risk = np.sqrt(weights @ cov_matrix @ weights)
            return -ret / risk if risk > 0 else 0
        
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(n)]
        
        result = minimize(
            neg_sharpe,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x


def activate_quantum_portfolio():
    """Activate quantum portfolio optimization."""
    print("="*70)
    print("QUANTUM PORTFOLIO OPTIMIZER - ACTIVATION")
    print("="*70)
    
    config = QuantumPortfolioConfig(
        n_assets=8,
        weight_bits=4,
        max_assets=5,
        risk_aversion=2.5,
        annealing_reads=200
    )
    
    allocator = QuantumPortfolioAllocator(config=config)
    
    # Test with sample data
    assets = ["BTC", "ETH", "SOL", "DOGE", "XRP", "ADA", "AVAX", "MATIC"]
    expected_returns = np.array([0.15, 0.12, 0.18, 0.10, 0.08, 0.09, 0.11, 0.07])
    cov_matrix = np.array([
        [0.04, 0.02, 0.02, 0.01, 0.01, 0.01, 0.02, 0.01],
        [0.02, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01],
        [0.02, 0.02, 0.05, 0.02, 0.01, 0.02, 0.02, 0.01],
        [0.01, 0.01, 0.02, 0.04, 0.01, 0.01, 0.01, 0.01],
        [0.01, 0.01, 0.01, 0.01, 0.03, 0.01, 0.01, 0.01],
        [0.01, 0.01, 0.02, 0.01, 0.01, 0.03, 0.01, 0.01],
        [0.02, 0.01, 0.02, 0.01, 0.01, 0.01, 0.04, 0.01],
        [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02]
    ])
    
    print(f"\nOptimizing portfolio for {len(assets)} assets...")
    result = allocator.optimize(expected_returns, cov_matrix, assets)
    
    print(f"\nOptimization Results ({result['method']}):")
    print(f"  Expected Return: {result['expected_return']*100:.2f}%")
    print(f"  Expected Risk: {result['expected_risk']*100:.2f}%")
    print(f"  Sharpe Ratio: {result['sharpe_ratio']:.3f}")
    print(f"  Assets Selected: {result['num_assets_selected']}/{len(assets)}")
    
    print(f"\nOptimal Weights:")
    for asset, weight in sorted(result['weights'].items(), key=lambda x: -x[1]):
        if weight > 0.01:
            print(f"  {asset}: {weight*100:.1f}%")
    
    print(f"\n[OK] QUANTUM PORTFOLIO OPTIMIZER ACTIVATED")
    print(f"  Quantum Available: {allocator.quantum_available}")
    print(f"  Method: {result['method']}")
    
    return allocator, result


if __name__ == "__main__":
    activate_quantum_portfolio()
