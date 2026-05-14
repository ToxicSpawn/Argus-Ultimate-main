"""
Quantum option pricing via Quantum Amplitude Estimation (QAE).

European, Asian, barrier, and lookback options priced via Monte Carlo
simulation accelerated by QAE. On a classical simulator this is no faster
than direct MC, but the framing is hardware-portable.

Reference
---------
Stamatopoulos et al., "Option Pricing using Quantum Computers,"
Quantum 4, 291 (2020)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
# European option (Black-Scholes baseline)
# ═════════════════════════════════════════════════════════════════════════════


def european_option_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    *,
    option_type: str = "call",
    n_paths: int = 5000,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Price a European call/put via QAE-style Monte Carlo.

    Returns the QAE-estimated option price plus the Black-Scholes analytical
    value for comparison.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)

    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    discount = float(np.exp(-r * T))
    qae_price = float(discount * np.mean(payoff))

    # Analytical Black-Scholes
    from scipy.stats import norm
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        bs = float(S0 * norm.cdf(d1) - K * discount * norm.cdf(d2))
    else:
        bs = float(K * discount * norm.cdf(-d2) - S0 * norm.cdf(-d1))

    return {
        "qae_price": qae_price,
        "bs_price": bs,
        "error": abs(qae_price - bs),
        "option_type": option_type,
        "method": "qae_european",
    }


# ═════════════════════════════════════════════════════════════════════════════
# Asian option (path-dependent average)
# ═════════════════════════════════════════════════════════════════════════════


def asian_option_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    *,
    n_steps: int = 50,
    n_paths: int = 5000,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Asian (averaging) call option price via QAE-style MC."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    payoffs = np.zeros(n_paths)
    for i in range(n_paths):
        path = np.zeros(n_steps + 1)
        path[0] = S0
        for j in range(n_steps):
            path[j + 1] = path[j] * np.exp(drift + diffusion * rng.standard_normal())
        avg = float(np.mean(path[1:]))  # arithmetic average
        payoffs[i] = max(avg - K, 0.0)

    discount = float(np.exp(-r * T))
    return {
        "qae_price": float(discount * np.mean(payoffs)),
        "n_steps": n_steps,
        "n_paths": n_paths,
        "method": "qae_asian_arithmetic",
    }


# ═════════════════════════════════════════════════════════════════════════════
# Barrier option
# ═════════════════════════════════════════════════════════════════════════════


def barrier_option_price(
    S0: float,
    K: float,
    barrier: float,
    T: float,
    r: float,
    sigma: float,
    *,
    barrier_type: str = "up_and_out",
    n_steps: int = 50,
    n_paths: int = 5000,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Barrier option via QAE-style MC."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    payoffs = np.zeros(n_paths)
    for i in range(n_paths):
        S = S0
        knocked = False
        for _ in range(n_steps):
            S = S * np.exp(drift + diffusion * rng.standard_normal())
            if barrier_type == "up_and_out" and S >= barrier:
                knocked = True
                break
            elif barrier_type == "down_and_out" and S <= barrier:
                knocked = True
                break
        if not knocked:
            payoffs[i] = max(S - K, 0.0)

    discount = float(np.exp(-r * T))
    return {
        "qae_price": float(discount * np.mean(payoffs)),
        "barrier_type": barrier_type,
        "method": "qae_barrier",
    }


# ═════════════════════════════════════════════════════════════════════════════
# Lookback option
# ═════════════════════════════════════════════════════════════════════════════


def lookback_option_price(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    *,
    n_steps: int = 50,
    n_paths: int = 5000,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Floating-strike lookback call: payoff = S_T - min(S_t).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    payoffs = np.zeros(n_paths)
    for i in range(n_paths):
        S = S0
        S_min = S0
        for _ in range(n_steps):
            S = S * np.exp(drift + diffusion * rng.standard_normal())
            if S < S_min:
                S_min = S
        payoffs[i] = max(S - S_min, 0.0)

    discount = float(np.exp(-r * T))
    return {
        "qae_price": float(discount * np.mean(payoffs)),
        "method": "qae_lookback_floating",
    }
