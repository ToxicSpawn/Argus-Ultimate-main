"""
Standard Hamiltonian builders for many-body quantum systems.

Provides:
- ``hubbard_model`` — Fermi-Hubbard on 1D chain or 2D lattice
- ``heisenberg_model`` — XXX, XXZ, XYZ Heisenberg
- ``transverse_field_ising`` — TFIM on a chain
- ``pauli_grouping`` — measurement-reduction by commuting Pauli grouping
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
# Hubbard model
# ═════════════════════════════════════════════════════════════════════════════


def hubbard_model(
    n_sites: int,
    *,
    t: float = 1.0,
    U: float = 4.0,
    mu: float = 0.0,
    geometry: str = "1d_chain",
) -> List[Tuple[str, float]]:
    """
    Fermi-Hubbard Hamiltonian:

        H = -t Σ_<i,j>,σ (c†_iσ c_jσ + h.c.)
            + U Σ_i n_i↑ n_i↓
            - μ Σ_iσ n_iσ

    Each site has two spin orbitals (up and down). Total qubits = 2 N.

    Parameters
    ----------
    n_sites : int
        Number of lattice sites.
    t, U, mu : float
        Hopping, on-site Coulomb repulsion, chemical potential.
    geometry : str
        "1d_chain" or "1d_ring".

    Returns
    -------
    List[Tuple[str, float]]
        Pauli decomposition of H. Each entry is (pauli_string, coefficient).
        Pauli strings have length 2 N (qubits 0..N-1 are spin-up, N..2N-1
        are spin-down).
    """
    n_qubits = 2 * n_sites
    terms: List[Tuple[str, float]] = []

    # Bonds list
    if geometry == "1d_chain":
        bonds = [(i, i + 1) for i in range(n_sites - 1)]
    elif geometry == "1d_ring":
        bonds = [(i, (i + 1) % n_sites) for i in range(n_sites)]
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    # Hopping term: -t (c†_iσ c_jσ + h.c.)
    # In JW: c†_i c_j → 0.5 (X_i X_j + Y_i Y_j) + Z chain between
    # For nearest neighbors with no qubits between, the Z chain is empty.
    for i, j in bonds:
        # Spin-up: qubits i, j
        for sigma_offset in (0, n_sites):
            qi = i + sigma_offset
            qj = j + sigma_offset
            # XX + YY hopping (with appropriate sign)
            xx = ["I"] * n_qubits
            yy = ["I"] * n_qubits
            xx[qi] = "X"
            xx[qj] = "X"
            yy[qi] = "Y"
            yy[qj] = "Y"
            terms.append(("".join(xx), -t * 0.5))
            terms.append(("".join(yy), -t * 0.5))

    # On-site Coulomb: U n_i↑ n_i↓ where n_iσ = (1 - Z_iσ) / 2
    # n_i↑ n_i↓ = 0.25 (1 - Z_i↑)(1 - Z_i↓)
    #          = 0.25 (1 - Z_i↑ - Z_i↓ + Z_i↑ Z_i↓)
    for i in range(n_sites):
        qi_up = i
        qi_dn = i + n_sites

        # Constant term (1)
        identity = "I" * n_qubits
        terms.append((identity, U * 0.25))

        # -Z_i↑
        z_up = ["I"] * n_qubits
        z_up[qi_up] = "Z"
        terms.append(("".join(z_up), -U * 0.25))

        # -Z_i↓
        z_dn = ["I"] * n_qubits
        z_dn[qi_dn] = "Z"
        terms.append(("".join(z_dn), -U * 0.25))

        # Z_i↑ Z_i↓
        zz = ["I"] * n_qubits
        zz[qi_up] = "Z"
        zz[qi_dn] = "Z"
        terms.append(("".join(zz), U * 0.25))

    # Chemical potential: -μ n_iσ = -μ (1 - Z_iσ) / 2 for each spin
    if abs(mu) > 1e-12:
        for i in range(n_sites):
            for sigma_offset in (0, n_sites):
                q = i + sigma_offset
                terms.append(("I" * n_qubits, -mu * 0.5))
                z_str = ["I"] * n_qubits
                z_str[q] = "Z"
                terms.append(("".join(z_str), mu * 0.5))

    return terms


# ═════════════════════════════════════════════════════════════════════════════
# Heisenberg model
# ═════════════════════════════════════════════════════════════════════════════


def heisenberg_model(
    n_sites: int,
    *,
    Jx: float = 1.0,
    Jy: float = 1.0,
    Jz: float = 1.0,
    h: float = 0.0,
    geometry: str = "1d_chain",
) -> List[Tuple[str, float]]:
    """
    Heisenberg model:

        H = Σ_<i,j> (Jx X_i X_j + Jy Y_i Y_j + Jz Z_i Z_j) + h Σ_i Z_i

    XXX: Jx = Jy = Jz
    XXZ: Jx = Jy ≠ Jz
    XYZ: all different

    Returns Pauli decomposition.
    """
    n = n_sites
    terms: List[Tuple[str, float]] = []

    if geometry == "1d_chain":
        bonds = [(i, i + 1) for i in range(n - 1)]
    elif geometry == "1d_ring":
        bonds = [(i, (i + 1) % n) for i in range(n)]
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    for i, j in bonds:
        for label, coupling in (("X", Jx), ("Y", Jy), ("Z", Jz)):
            if abs(coupling) < 1e-12:
                continue
            term = ["I"] * n
            term[i] = label
            term[j] = label
            terms.append(("".join(term), float(coupling)))

    if abs(h) > 1e-12:
        for i in range(n):
            term = ["I"] * n
            term[i] = "Z"
            terms.append(("".join(term), float(h)))

    return terms


# ═════════════════════════════════════════════════════════════════════════════
# Transverse-field Ising
# ═════════════════════════════════════════════════════════════════════════════


def transverse_field_ising(
    n_sites: int,
    *,
    J: float = 1.0,
    h: float = 1.0,
    geometry: str = "1d_chain",
) -> List[Tuple[str, float]]:
    """
    Transverse-field Ising model:

        H = -J Σ_<i,j> Z_i Z_j - h Σ_i X_i
    """
    n = n_sites
    terms: List[Tuple[str, float]] = []

    if geometry == "1d_chain":
        bonds = [(i, i + 1) for i in range(n - 1)]
    elif geometry == "1d_ring":
        bonds = [(i, (i + 1) % n) for i in range(n)]
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    for i, j in bonds:
        term = ["I"] * n
        term[i] = "Z"
        term[j] = "Z"
        terms.append(("".join(term), -float(J)))

    for i in range(n):
        term = ["I"] * n
        term[i] = "X"
        terms.append(("".join(term), -float(h)))

    return terms


# ═════════════════════════════════════════════════════════════════════════════
# Pauli grouping for measurement reduction
# ═════════════════════════════════════════════════════════════════════════════


def pauli_grouping(
    pauli_terms: List[Tuple[str, float]],
) -> List[List[Tuple[str, float]]]:
    """
    Group Pauli terms into commuting sets (qubit-wise).

    Two Pauli strings commute qubit-wise iff for every qubit position they
    share the same letter (or one of them is I). Grouping commuting terms
    means they can all be measured in the same basis, reducing the total
    number of measurements needed.

    Returns a list of groups, each group being a list of (pauli_string, coeff).
    """
    if not pauli_terms:
        return []

    groups: List[List[Tuple[str, float]]] = []
    for term in pauli_terms:
        pauli, coeff = term
        placed = False
        for grp in groups:
            if all(_qubitwise_commute(pauli, p2) for p2, _ in grp):
                grp.append(term)
                placed = True
                break
        if not placed:
            groups.append([term])
    return groups


def _qubitwise_commute(p1: str, p2: str) -> bool:
    """Check if two Pauli strings commute qubit-wise."""
    if len(p1) != len(p2):
        return False
    for c1, c2 in zip(p1, p2):
        if c1 == "I" or c2 == "I":
            continue
        if c1 != c2:
            return False
    return True
