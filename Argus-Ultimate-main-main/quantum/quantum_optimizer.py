"""Retired optimizer implementation with canonical-facade compatibility.

The previous file contained a duplicate QAOA/VQE implementation with broken
runtime paths and unsupported quantum-advantage language. Application code
should now use :func:`quantum.get_quantum_facade` directly. This module remains
only so older optional imports receive a small, honest adapter instead of the
retired implementation.
"""

from __future__ import annotations

from typing import Any

from .canonical import ArgusQuantumFacade, get_quantum_facade
from .retired import retired_placeholder


class QuantumOptimizer:
    """Compatibility adapter around the canonical quantum facade."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.facade: ArgusQuantumFacade = get_quantum_facade()

    def optimize_portfolio(
        self,
        expected_returns: Any,
        covariance_matrix: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Delegate portfolio subset optimization to the canonical QAOA API."""
        return self.facade.optimize_portfolio(
            expected_returns,
            covariance_matrix,
            **kwargs,
        )

    def estimate_tail_risk(
        self,
        returns: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Delegate VaR/CVaR estimation to the canonical Sobol QMC API."""
        return self.facade.estimate_tail_risk_qmc(returns, **kwargs)

    def status(self) -> dict[str, Any]:
        """Return an honest capability report as a plain dict."""
        return self.facade.status().to_dict()


def get_quantum_optimizer(*_args: Any, **_kwargs: Any) -> QuantumOptimizer:
    """Return the compatibility optimizer backed by the canonical facade."""
    return QuantumOptimizer()


def __getattr__(name: str) -> Any:
    """Return callable retirement placeholders for removed legacy symbols."""
    return retired_placeholder(f"quantum.quantum_optimizer.{name}")


__all__ = ["QuantumOptimizer", "get_quantum_optimizer"]
