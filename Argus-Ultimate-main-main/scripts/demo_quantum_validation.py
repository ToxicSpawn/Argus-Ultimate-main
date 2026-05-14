"""
Quantum-Enhanced Paper Validation Demo

This script demonstrates the quantum-enhanced validation system for adaptive trading components.
"""

import sys
import os
import logging

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.real_time_learning.quantum_paper_validation import QuantumValidationResult

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_quantum_validation():
    """Demonstrate quantum-enhanced validation workflow"""
    print("Starting Quantum-Enhanced Paper Validation Demo")
    logger.info("Starting Quantum-Enhanced Paper Validation Demo")
    
    # Demo 1: Strategy Allocator Validation
    print("\n=== Testing Strategy Allocator Validation ===")
    logger.info("\n=== Testing Strategy Allocator Validation ===")
    
    # Create a demo quantum validation result
    quantum_result = QuantumValidationResult(
        component="strategy_allocator",
        parameter_changes={"strategy_weights": {"momentum": 0.4, "mean_reversion": 0.6}},
        test_passed=True,
        metrics={
            "sharpe_ratio": 2.0,
            "win_rate": 0.6,
            "max_drawdown": 0.15,
            "quantum_sharpe": 2.2,
            "quantum_improvement": 0.1
        },
        required_metrics={
            "sharpe_ratio": (1.2, ">"),
            "win_rate": (0.5, ">="),
            "max_drawdown": (0.2, "<")
        },
        quantum_metadata={
            "quantum_results": {
                "quantum_sharpe": 2.2,
                "quantum_kernel_metadata": {"quantum_boost": 0.1}
            },
            "classical_baseline": {"score": 2.0}
        }
    )
    
    result = {
        "status": "success",
        "validation_result": quantum_result,
        "backtest_results": {
            "sharpe_ratio": 2.2,
            "total_trades": 50
        },
        "quantum_enhanced": True
    }
    
    print(f"Proposed changes: {{'momentum': 0.4, 'mean_reversion': 0.6}}")
    print(f"Validation passed: {result['validation_result'].test_passed}")
    print(f"Classical Sharpe: {result['validation_result'].metrics.get('sharpe_ratio', 'N/A')}")
    print(f"Quantum Sharpe: {result['validation_result'].metrics.get('quantum_sharpe', 'N/A')}")
    
    improvement = result['validation_result'].metrics.get('quantum_improvement', 'N/A')
    if isinstance(improvement, (int, float)):
        print(f"Quantum Improvement: {improvement*100:.1f}%")
    else:
        print(f"Quantum Improvement: {improvement}")
        
    print(f"Quantum Metadata: {result['validation_result'].quantum_metadata}")
    
    # Demo 2: Correlation Matrix Validation
    print("\n=== Testing Correlation Matrix Validation ===")
    logger.info("\n=== Testing Correlation Matrix Validation ===")
    
    # Create a demo quantum validation result for correlation matrix
    quantum_result = QuantumValidationResult(
        component="correlation_matrix",
        parameter_changes={"BTC:ETH": 0.75, "BTC:SOL": 0.6, "ETH:SOL": 0.65},
        test_passed=True,
        metrics={
            "diversification_score": 0.85,
            "quantum_diversification": 0.92,
            "quantum_div_improvement": 0.08
        },
        required_metrics={},
        quantum_metadata={
            "quantum_results": {
                "quantum_diversification": 0.92,
                "quantum_weights": [0.4, 0.3, 0.3]
            },
            "classical_baseline": {"diversification": 0.85}
        }
    )
    
    result = {
        "status": "success",
        "validation_result": quantum_result,
        "backtest_results": {},
        "quantum_enhanced": True
    }
    
    print(f"Proposed changes: {{'BTC:ETH': 0.75, 'BTC:SOL': 0.6, 'ETH:SOL': 0.65}}")
    print(f"Validation passed: {result['validation_result'].test_passed}")
    print(f"Diversification Score: {result['validation_result'].metrics.get('quantum_diversification', 'N/A')}")
    print(f"Quantum Metadata: {result['validation_result'].quantum_metadata}")
    
    print("\nQuantum-Enhanced Paper Validation Demo completed successfully!")
    logger.info("\nQuantum-Enhanced Paper Validation Demo completed successfully!")

if __name__ == "__main__":
    demo_quantum_validation()