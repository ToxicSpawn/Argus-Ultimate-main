"""
Probabilistic Error Cancellation (PEC).

PEC inverts a known noise channel by sampling from a quasi-probability
distribution over Pauli operations. The expectation value of an observable
under the noise-free state is recovered as a weighted average of noisy
runs, with weights drawn from the quasi-distribution.

Reference
---------
Temme, Bravyi, Gambetta, "Error mitigation for short-depth quantum circuits,"
PRL 119, 180509 (2017)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Quasi-probability decomposition for depolarizing noise inverse
# ═════════════════════════════════════════════════════════════════════════════


def build_quasi_probability(noise_strength: float) -> Dict[str, Any]:
    """
    Build the quasi-probability distribution for inverting a depolarizing
    channel of strength p.

    The inverse of D_p(ρ) = (1 - p) ρ + (p / 3)(X ρ X + Y ρ Y + Z ρ Z) is

        D_p^(-1)(ρ) = a₀ ρ - a₁ (X ρ X + Y ρ Y + Z ρ Z)

    where a₀ = (1 + 2 p / (3 (1 - p))) / 4 ... actually for unital depolarizing:
    The inverse has weights:
        c_I = (4 - p) / (4 - 4p)
        c_X = c_Y = c_Z = -p / (4 - 4p)

    The 1-norm γ = sum |c_i| measures the sampling overhead.
    """
    p = float(np.clip(noise_strength, 0.0, 0.999))
    denom = 4.0 - 4.0 * p
    if abs(denom) < 1e-12:
        return {"weights": {"I": 1.0}, "gamma": 1.0}
    c_I = (4.0 - p) / denom
    c_X = -p / denom
    c_Y = -p / denom
    c_Z = -p / denom
    weights = {"I": c_I, "X": c_X, "Y": c_Y, "Z": c_Z}
    gamma = sum(abs(w) for w in weights.values())
    return {"weights": weights, "gamma": gamma}


def probabilistic_error_cancellation(
    noisy_estimator: Callable[[str], float],
    noise_strength: float,
    *,
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Apply PEC to recover the noise-free expectation of an observable.

    Parameters
    ----------
    noisy_estimator : Callable[[str], float]
        Function that takes a Pauli prefix string ("I", "X", "Y", or "Z") and
        returns the noisy expectation value of the observable when that Pauli
        is prepended to the circuit.
    noise_strength : float
        Effective single-qubit depolarizing noise strength.
    n_samples : int
        Number of PEC samples.

    Returns
    -------
    Dict[str, Any]
        ``{"mitigated_value", "raw_noisy_value", "n_samples", "gamma",
          "method"}``
    """
    rng = np.random.default_rng(seed)
    qp = build_quasi_probability(noise_strength)
    weights = qp["weights"]
    gamma = qp["gamma"]
    if gamma <= 0:
        return {
            "mitigated_value": noisy_estimator("I"),
            "raw_noisy_value": noisy_estimator("I"),
            "n_samples": 1,
            "gamma": 0.0,
            "method": "pec_trivial",
        }

    # Build sampling distribution: |c_i| / gamma
    paulis = list(weights.keys())
    probs = np.array([abs(weights[p]) / gamma for p in paulis])
    signs = np.array([np.sign(weights[p]) for p in paulis])

    samples = []
    for _ in range(n_samples):
        idx = int(rng.choice(len(paulis), p=probs))
        pauli = paulis[idx]
        sign = signs[idx]
        # Sample the noisy estimator with this Pauli prepended
        noisy_val = noisy_estimator(pauli)
        # PEC sample contribution: gamma * sign * noisy_val
        samples.append(gamma * sign * noisy_val)

    mitigated = float(np.mean(samples))
    raw = noisy_estimator("I")  # raw noisy value with no Pauli prep

    return {
        "mitigated_value": mitigated,
        "raw_noisy_value": raw,
        "n_samples": n_samples,
        "gamma": gamma,
        "method": "pec_depolarizing_inverse",
    }
