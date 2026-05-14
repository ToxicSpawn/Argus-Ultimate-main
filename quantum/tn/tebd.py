"""
Time-Evolving Block Decimation (TEBD) for 1D quantum systems.

TEBD applies a Trotterized time-evolution operator to an MPS via local
2-site gates with SVD truncation. For 1D systems with low entanglement,
TEBD scales as O(n · χ³) per time step.

Reference
---------
Vidal, "Efficient classical simulation of slightly entangled quantum
computations," Phys. Rev. Lett. 91, 147902 (2003).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# TEBD evolver
# ═════════════════════════════════════════════════════════════════════════════


class TEBD:
    """
    Time-Evolving Block Decimation.

    Parameters
    ----------
    pauli_2site_terms : List[Tuple[int, np.ndarray]]
        List of (site_index, 4x4 matrix) pairs. Each entry specifies a
        2-site Hamiltonian term acting on sites (i, i+1).
    max_bond_dim : int
        Maximum bond dimension for SVD truncation.
    """

    def __init__(
        self,
        pauli_2site_terms: List[Tuple[int, np.ndarray]],
        max_bond_dim: int = 32,
    ) -> None:
        self.terms = list(pauli_2site_terms)
        self.max_bond_dim = int(max_bond_dim)

    def evolve(
        self,
        initial_mps: List[np.ndarray],
        time: float,
        *,
        dt: float = 0.05,
        order: int = 2,
    ) -> Dict[str, Any]:
        """
        Evolve the initial MPS for time t with time step dt.

        Returns the final MPS plus snapshots of the magnetization at each step.
        """
        t0 = time
        n_steps = int(np.ceil(t0 / dt))
        dt_actual = t0 / n_steps if n_steps > 0 else dt

        mps = [t.copy() for t in initial_mps]
        n = len(mps)
        snapshots: List[np.ndarray] = []

        # Build local 2-site evolution operators
        ops = []
        for i, H_local in self.terms:
            U_local = self._matrix_exp(-1j * dt_actual * H_local)
            ops.append((i, U_local))

        for step in range(n_steps):
            for i, U in ops:
                self._apply_2site(mps, i, U)
            snapshots.append(self._compute_magnetization(mps))

        return {
            "final_mps": mps,
            "snapshots": snapshots,
            "n_steps": n_steps,
            "dt": dt_actual,
            "method": "tebd",
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _matrix_exp(self, M: np.ndarray) -> np.ndarray:
        """Matrix exponential via eigendecomposition."""
        eigvals, eigvecs = np.linalg.eig(M)
        return eigvecs @ np.diag(np.exp(eigvals)) @ np.linalg.inv(eigvecs)

    def _apply_2site(self, mps: List[np.ndarray], i: int, U: np.ndarray) -> None:
        """Apply a 2-site gate U at sites (i, i+1) and SVD-split."""
        T_l = mps[i]
        T_r = mps[i + 1]
        Dl, _, Dm = T_l.shape
        Dm2, _, Dr = T_r.shape
        if Dm != Dm2:
            raise ValueError(f"MPS bond mismatch at site {i}")

        # Merge sites
        theta = np.einsum("amb,bnc->amnc", T_l, T_r)
        # Apply U on physical indices (m, n)
        U_mat = U.reshape(2, 2, 2, 2)  # (m', n', m, n)
        theta = np.einsum("MNmn,amnc->aMNc", U_mat, theta)

        # SVD-split
        M = theta.reshape(Dl * 2, 2 * Dr)
        U_mat, S, Vh = np.linalg.svd(M, full_matrices=False)
        chi_eff = min(self.max_bond_dim, len(S))
        U_mat = U_mat[:, :chi_eff]
        S = S[:chi_eff]
        S = S / max(float(np.linalg.norm(S)), 1e-12)
        Vh = Vh[:chi_eff]

        mps[i] = U_mat.reshape(Dl, 2, chi_eff)
        mps[i + 1] = (S.reshape(-1, 1) * Vh).reshape(chi_eff, 2, Dr)

    def _compute_magnetization(self, mps: List[np.ndarray]) -> np.ndarray:
        """Compute ⟨Z_i⟩ for each site (simplified — returns abs of first amplitude)."""
        n = len(mps)
        mags = np.zeros(n)
        for i in range(n):
            T = mps[i]
            # ⟨Z⟩ = |amplitude_0|² - |amplitude_1|²
            # Approximate by squaring tensor elements
            mags[i] = float(np.real(np.sum(np.abs(T[:, 0, :]) ** 2 - np.abs(T[:, 1, :]) ** 2)))
        return mags
