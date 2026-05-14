"""
Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.

SPSA estimates the gradient of a noisy cost function using only **2 evaluations
per iteration** regardless of the parameter count. This is dramatically more
efficient than COBYLA (which needs N+1 evaluations) for high-dimensional
variational quantum circuits.

Reference
---------
J.C. Spall, "An overview of the simultaneous perturbation method for efficient
optimization," Johns Hopkins APL Tech. Dig. 19(4), 482 (1998).

Recommended hyperparameters (Spall 1998):
    a = 0.2, c = 0.1, A = 10, alpha = 0.602, gamma = 0.101

Update rule
-----------
    Δ_k ~ Bernoulli(±1)^N (random perturbation direction)
    g_k = (L(θ_k + c_k Δ_k) - L(θ_k - c_k Δ_k)) / (2 c_k Δ_k)
    θ_{k+1} = θ_k - a_k g_k

with a_k = a / (k + 1 + A)^alpha and c_k = c / (k + 1)^gamma.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# SPSA optimizer
# ═════════════════════════════════════════════════════════════════════════════


def spsa_optimize(
    cost_fn: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    *,
    n_iter: int = 200,
    a: float = 0.2,
    c: float = 0.1,
    A: float = 10.0,
    alpha: float = 0.602,
    gamma: float = 0.101,
    seed: Optional[int] = 42,
    callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
) -> Dict[str, Any]:
    """
    Minimize ``cost_fn`` over the parameter space using SPSA.

    Parameters
    ----------
    cost_fn : Callable[[np.ndarray], float]
        Function to minimize. Should be tolerant of measurement noise.
    initial_params : np.ndarray
        Starting parameter vector.
    n_iter : int
        Number of SPSA iterations.
    a, c, A, alpha, gamma : float
        Spall (1998) hyperparameters.
    seed : int, optional
        RNG seed for reproducibility.
    callback : Callable, optional
        Called as ``callback(iter_idx, params, cost)`` after each iteration.

    Returns
    -------
    Dict[str, Any]
        ``{"best_params", "best_cost", "history", "n_evaluations",
          "method", "elapsed_ms"}``
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    params = np.asarray(initial_params, dtype=float).copy()
    n_params = params.size

    best_params = params.copy()
    best_cost = float("inf")
    history: List[Dict[str, Any]] = []
    n_evaluations = 0

    for k in range(n_iter):
        a_k = a / ((k + 1 + A) ** alpha)
        c_k = c / ((k + 1) ** gamma)

        # Bernoulli ±1 perturbation
        delta = rng.choice([-1.0, 1.0], size=n_params)

        # Two evaluations per iteration (vs N+1 for COBYLA)
        params_plus = params + c_k * delta
        params_minus = params - c_k * delta

        cost_plus = float(cost_fn(params_plus))
        cost_minus = float(cost_fn(params_minus))
        n_evaluations += 2

        # SPSA gradient estimate
        g_k = (cost_plus - cost_minus) / (2.0 * c_k) / delta

        # Update parameters
        params = params - a_k * g_k

        # Track best
        cost_now = 0.5 * (cost_plus + cost_minus)
        if cost_now < best_cost:
            best_cost = cost_now
            best_params = params.copy()

        history.append({
            "iter": k,
            "cost": cost_now,
            "a_k": a_k,
            "c_k": c_k,
        })

        if callback is not None:
            try:
                callback(k, params.copy(), cost_now)
            except Exception:
                pass

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "best_params": best_params,
        "best_cost": float(best_cost),
        "history": history,
        "n_evaluations": n_evaluations,
        "method": "spsa",
        "elapsed_ms": elapsed_ms,
        "n_iter": n_iter,
    }
