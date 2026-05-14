"""
Virtual distillation for error mitigation.

Computes ⟨O⟩ ≈ Tr(O ρ_clean) using ⟨O⟩_M ≈ Tr(O ρ^M) / Tr(ρ^M) where M
copies of the noisy state are coherently combined. This exponentially
suppresses incoherent errors at the cost of M-fold ancilla overhead.

Reference
---------
Huggins et al., "Virtual Distillation for Quantum Error Mitigation,"
PRX 11, 041036 (2021)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np


def virtual_distillation_estimator(
    rho_noisy: np.ndarray,
    observable: np.ndarray,
    *,
    M: int = 2,
) -> Dict[str, Any]:
    """
    Estimate ⟨O⟩ via M-fold virtual distillation.

    Computes ⟨O⟩_VD = Tr(O ρ^M) / Tr(ρ^M).

    Parameters
    ----------
    rho_noisy : np.ndarray
        Noisy density matrix (d x d).
    observable : np.ndarray
        Observable matrix (d x d Hermitian).
    M : int
        Number of virtual copies. Higher M suppresses errors more aggressively
        but requires more measurements on real hardware.

    Returns
    -------
    Dict[str, Any]
        ``{"mitigated_value", "raw_value", "purity", "M", "method"}``
    """
    rho = np.asarray(rho_noisy, dtype=np.complex128)
    O = np.asarray(observable, dtype=np.complex128)
    if rho.shape != O.shape:
        raise ValueError(f"rho shape {rho.shape} != O shape {O.shape}")

    # Raw noisy expectation
    raw = float(np.real(np.trace(O @ rho)))

    # ρ^M
    rho_M = rho.copy()
    for _ in range(M - 1):
        rho_M = rho_M @ rho

    # Mitigated expectation
    num = float(np.real(np.trace(O @ rho_M)))
    den = float(np.real(np.trace(rho_M)))
    mitigated = num / max(den, 1e-12)

    # Purity (a measure of how close ρ is to a pure state)
    purity = float(np.real(np.trace(rho @ rho)))

    return {
        "mitigated_value": mitigated,
        "raw_value": raw,
        "purity": purity,
        "M": M,
        "method": f"virtual_distillation_M={M}",
    }
