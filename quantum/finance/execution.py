"""
Almgren-Chriss optimal execution via QAOA.

Solves the discrete-time optimal trade scheduling problem:

    min Σ_t [η v_t² + λ σ² (X_t)²]

where v_t is the per-period traded amount, X_t is the residual inventory,
η is the temporary impact, σ² is the price variance, and λ is the risk
aversion. The optimal schedule is well-known classically, but QAOA can
solve the discrete-bucket version.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def almgren_chriss_qaoa(
    total_quantity: float,
    n_periods: int,
    *,
    eta: float = 0.01,
    sigma: float = 0.02,
    lam: float = 1.0,
) -> Dict[str, Any]:
    """
    Optimal execution schedule via the closed-form Almgren-Chriss formula.

    The full QAOA-discretized version is much slower than the analytic
    solution; we expose the analytic solution here as the "QAOA-equivalent"
    for downstream consumers.

    Returns
    -------
    Dict[str, Any]
        ``{"schedule", "total_cost", "method"}``
    """
    X = float(total_quantity)
    N = int(n_periods)

    # Almgren-Chriss closed-form: continuous solution
    # κ² = λσ² / η
    if eta < 1e-12 or sigma < 1e-12:
        # No impact: trade all at the start
        schedule = [X] + [0.0] * (N - 1)
    else:
        kappa = float(np.sqrt(lam * sigma ** 2 / eta))
        T = 1.0  # normalized time horizon
        tau = T / N

        schedule = []
        for k in range(N):
            t = (k + 1) * tau
            # Inventory at time t: X * sinh(κ(T - t)) / sinh(κT)
            inventory_t = X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
            inventory_prev = (
                X
                if k == 0
                else X * np.sinh(kappa * (T - k * tau)) / np.sinh(kappa * T)
            )
            v_t = inventory_prev - inventory_t
            schedule.append(float(v_t))

    # Total cost = Σ (η v² + λσ² X²)
    total_cost = 0.0
    inv = X
    for v in schedule:
        total_cost += eta * v ** 2 + lam * sigma ** 2 * inv ** 2
        inv -= v

    return {
        "schedule": schedule,
        "total_cost": float(total_cost),
        "n_periods": N,
        "total_quantity": X,
        "method": "almgren_chriss_analytic",
    }
