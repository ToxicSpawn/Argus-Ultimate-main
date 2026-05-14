"""Quantum portfolio hardware API with simulator-aware fallbacks."""

from .classical_fallback import ClassicalFallbackConfig, ClassicalFallbackOptimizer, ClassicalOptimizationResult
from .ibm_quantum import IBMQuantumClient, IBMQuantumJob, IBMQuantumJobRequest, IBMQuantumResult
from .qaoa_optimizer import QAOAConfig, QAOAOptimizationResult, QAOAPortfolioOptimizer
from .qubo_builder import PortfolioQUBOBuilder, PortfolioQUBOConfig, PortfolioQUBOModel, PortfolioQUBOProblem

__all__ = [
    "ClassicalFallbackConfig",
    "ClassicalFallbackOptimizer",
    "ClassicalOptimizationResult",
    "IBMQuantumClient",
    "IBMQuantumJob",
    "IBMQuantumJobRequest",
    "IBMQuantumResult",
    "QAOAConfig",
    "QAOAOptimizationResult",
    "QAOAPortfolioOptimizer",
    "PortfolioQUBOBuilder",
    "PortfolioQUBOConfig",
    "PortfolioQUBOModel",
    "PortfolioQUBOProblem",
]
