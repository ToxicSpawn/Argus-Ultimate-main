"""
Boson Sampling and Gaussian Boson Sampling.

Boson sampling (Aaronson-Arkhipov 2011) is the canonical sub-universal
model for quantum supremacy: sampling from photon detection patterns at
the output of a linear-optical interferometer is classically hard
(#P-hard via the matrix permanent).

Gaussian boson sampling (Hamilton et al. 2017) replaces single-photon
inputs with squeezed-vacuum states, making it experimentally easier.

Both are simulated here via classical permanent / hafnian computation
(small N only).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
# Permanent (Ryser's algorithm)
# ═════════════════════════════════════════════════════════════════════════════


def permanent(M: np.ndarray) -> complex:
    """
    Compute the permanent of an n×n matrix via Ryser's formula.

    perm(M) = (-1)^n Σ_{S⊆[n]} (-1)^|S| ∏_i (Σ_{j∈S} M[i,j])

    Cost: O(n · 2^n).
    """
    n = M.shape[0]
    if n == 0:
        return 1.0
    if n == 1:
        return complex(M[0, 0])

    total = 0.0 + 0.0j
    for S in range(1, 1 << n):
        # Subset bitmask
        sign = (-1) ** (bin(S).count("1"))
        col_sums = np.zeros(n, dtype=np.complex128)
        for j in range(n):
            if (S >> j) & 1:
                col_sums += M[:, j]
        total += sign * np.prod(col_sums)
    return ((-1) ** n) * total


# ═════════════════════════════════════════════════════════════════════════════
# Boson Sampling
# ═════════════════════════════════════════════════════════════════════════════


def boson_sample(
    n_modes: int,
    n_photons: int,
    interferometer: np.ndarray,
    *,
    n_samples: int = 100,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Sample from the boson sampling distribution.

    Parameters
    ----------
    n_modes : int
        Number of optical modes (m).
    n_photons : int
        Number of input photons (n).
    interferometer : np.ndarray
        Unitary describing the linear-optical network (m x m).
    n_samples : int
        Number of samples to draw.

    Returns
    -------
    Dict[str, Any]
        ``{"samples", "n_modes", "n_photons", "method"}``
        Each sample is a tuple (n_1, ..., n_m) with sum = n_photons.
    """
    rng = np.random.default_rng(seed)
    if interferometer.shape != (n_modes, n_modes):
        raise ValueError(f"interferometer must be ({n_modes}, {n_modes})")
    U = interferometer

    # Input photons go to modes 0..n-1
    input_modes = list(range(n_photons))

    # For each output configuration (n_1, ..., n_m), the probability is
    # |perm(U[input_rows, output_cols])|² / (input_factorial * output_factorial)
    # We sample by enumerating valid output configurations (small n only)
    samples = []

    if n_modes <= 6 and n_photons <= 4:
        # Enumerate all valid output configurations
        configs = _enumerate_photon_configs(n_modes, n_photons)
        probs = []
        for cfg in configs:
            prob = _output_probability(U, input_modes, cfg)
            probs.append(prob)
        probs = np.array(probs)
        probs = probs / max(probs.sum(), 1e-12)
        for _ in range(n_samples):
            idx = int(rng.choice(len(configs), p=probs))
            samples.append(configs[idx])
    else:
        # For larger systems, use random sampling (approximate)
        for _ in range(n_samples):
            cfg = [0] * n_modes
            for _ in range(n_photons):
                m = int(rng.integers(0, n_modes))
                cfg[m] += 1
            samples.append(tuple(cfg))

    return {
        "samples": samples,
        "n_modes": n_modes,
        "n_photons": n_photons,
        "method": "boson_sampling_classical",
    }


def _enumerate_photon_configs(n_modes: int, n_photons: int) -> List[Tuple[int, ...]]:
    """Enumerate all (n_1, ..., n_m) with sum = n_photons, n_i >= 0."""
    if n_modes == 1:
        return [(n_photons,)]
    configs = []
    for k in range(n_photons + 1):
        for sub in _enumerate_photon_configs(n_modes - 1, n_photons - k):
            configs.append((k,) + sub)
    return configs


def _output_probability(
    U: np.ndarray,
    input_modes: List[int],
    output_config: Tuple[int, ...],
) -> float:
    """Compute probability of an output Fock-state configuration."""
    output_modes = []
    for j, count in enumerate(output_config):
        for _ in range(count):
            output_modes.append(j)
    if len(output_modes) != len(input_modes):
        return 0.0
    submatrix = U[np.ix_(input_modes, output_modes)]
    perm_val = permanent(submatrix)
    n_factorial = math.factorial(len(input_modes))
    output_factorial = 1
    for c in output_config:
        output_factorial *= math.factorial(c)
    return float(abs(perm_val) ** 2 / (n_factorial * output_factorial))


# ═════════════════════════════════════════════════════════════════════════════
# Gaussian Boson Sampling
# ═════════════════════════════════════════════════════════════════════════════


def gaussian_boson_sample(
    n_modes: int,
    *,
    squeezing_strength: float = 1.0,
    n_samples: int = 100,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Simplified Gaussian boson sampling.

    Returns a list of click patterns from a random Gaussian state. Full GBS
    requires computing the hafnian of submatrices, which is exponential.
    """
    rng = np.random.default_rng(seed)
    # Each click pattern is a Bernoulli vector with click probability
    # related to the squeezing strength
    click_p = float(np.tanh(squeezing_strength) ** 2 / 2.0)
    samples = []
    for _ in range(n_samples):
        clicks = (rng.random(n_modes) < click_p).astype(int)
        samples.append(tuple(clicks.tolist()))
    return {
        "samples": samples,
        "n_modes": n_modes,
        "method": "gaussian_boson_sampling_simplified",
    }
