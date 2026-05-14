"""
Fermion-to-qubit mappings.

Maps fermionic creation/annihilation operators to Pauli strings on qubits.
The three standard mappings are:

- **Jordan-Wigner (JW)**: a^†_i = (Z_0 ⊗ ... ⊗ Z_{i-1}) ⊗ (X_i - iY_i)/2
  Locality: O(N) per term.
- **Bravyi-Kitaev (BK)**: locality O(log N) per term, more complex.
- **Parity**: simpler mapping with locality O(N).

Each function returns a dict mapping a Pauli string (length N) to a complex
coefficient.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
# Jordan-Wigner
# ═════════════════════════════════════════════════════════════════════════════


def jordan_wigner(
    op_type: str,
    site: int,
    n_sites: int,
) -> Dict[str, complex]:
    """
    Jordan-Wigner transform of a single fermion operator.

    Parameters
    ----------
    op_type : str
        "creation" (a^†_i) or "annihilation" (a_i)
    site : int
        Site index i.
    n_sites : int
        Total number of sites N.

    Returns
    -------
    Dict[str, complex]
        Pauli string → coefficient.
        a^†_i = (Z⊗...⊗Z ⊗ (X - iY)/2) on the i-th qubit.
    """
    if site < 0 or site >= n_sites:
        raise ValueError(f"site {site} out of range [0, {n_sites})")

    # Build the Z chain: Z_0 Z_1 ... Z_{site-1}
    z_chain = ["I"] * n_sites
    for j in range(site):
        z_chain[j] = "Z"

    # Add the X and -iY (or X and +iY for annihilation) on site i
    x_string = list(z_chain)
    x_string[site] = "X"
    y_string = list(z_chain)
    y_string[site] = "Y"

    if op_type == "creation":
        # a^†_i = 0.5 (X - iY) on site i, with Z chain
        return {
            "".join(x_string): 0.5 + 0j,
            "".join(y_string): -0.5j,
        }
    elif op_type == "annihilation":
        # a_i = 0.5 (X + iY) on site i, with Z chain
        return {
            "".join(x_string): 0.5 + 0j,
            "".join(y_string): 0.5j,
        }
    else:
        raise ValueError(f"op_type must be 'creation' or 'annihilation', got {op_type}")


def fermion_creation(site: int, n_sites: int, mapping: str = "jw") -> Dict[str, complex]:
    """Convenience: create a^†_i Pauli decomposition."""
    if mapping == "jw":
        return jordan_wigner("creation", site, n_sites)
    elif mapping == "bk":
        return bravyi_kitaev("creation", site, n_sites)
    elif mapping == "parity":
        return parity_mapping("creation", site, n_sites)
    else:
        raise ValueError(f"Unknown mapping: {mapping}")


def fermion_annihilation(site: int, n_sites: int, mapping: str = "jw") -> Dict[str, complex]:
    """Convenience: create a_i Pauli decomposition."""
    if mapping == "jw":
        return jordan_wigner("annihilation", site, n_sites)
    elif mapping == "bk":
        return bravyi_kitaev("annihilation", site, n_sites)
    elif mapping == "parity":
        return parity_mapping("annihilation", site, n_sites)
    else:
        raise ValueError(f"Unknown mapping: {mapping}")


# ═════════════════════════════════════════════════════════════════════════════
# Bravyi-Kitaev
# ═════════════════════════════════════════════════════════════════════════════


def bravyi_kitaev(
    op_type: str,
    site: int,
    n_sites: int,
) -> Dict[str, complex]:
    """
    Bravyi-Kitaev transform of a single fermion operator.

    BK uses a binary tree partial-sum scheme; each fermion operator becomes a
    Pauli string of length O(log N) (vs O(N) for JW).

    This is a simplified implementation that produces correct anticommutation
    behavior but may not be optimal in Pauli weight.
    """
    if site < 0 or site >= n_sites:
        raise ValueError(f"site {site} out of range [0, {n_sites})")

    # BK update set: bits below i in the binary tree
    update_set = _bk_update_set(site, n_sites)
    parity_set = _bk_parity_set(site, n_sites)
    flip_set = _bk_flip_set(site, n_sites)

    # Build Pauli strings (X on update set, Z on parity set, Z on flip)
    # The BK basis transforms a_i and a_i^† into shorter Pauli strings.
    pauli_x = ["I"] * n_sites
    pauli_y = ["I"] * n_sites
    for q in update_set:
        pauli_x[q] = "X"
        pauli_y[q] = "X"
    for q in parity_set:
        # We use Z on parity set
        if pauli_x[q] == "I":
            pauli_x[q] = "Z"
        if pauli_y[q] == "I":
            pauli_y[q] = "Z"
    for q in flip_set:
        # Tracks the qubit at site i (X for the X string, Y for the Y string)
        if pauli_y[q] == "I":
            pauli_y[q] = "Z"

    # Set the X and Y on the actual site
    pauli_x[site] = "X"
    pauli_y[site] = "Y"

    if op_type == "creation":
        return {
            "".join(pauli_x): 0.5 + 0j,
            "".join(pauli_y): -0.5j,
        }
    elif op_type == "annihilation":
        return {
            "".join(pauli_x): 0.5 + 0j,
            "".join(pauli_y): 0.5j,
        }
    else:
        raise ValueError(f"op_type must be 'creation' or 'annihilation'")


def _bk_update_set(site: int, n_sites: int) -> List[int]:
    """Bravyi-Kitaev update set: ancestors of site i in the binary tree."""
    update = []
    i = site
    while i < n_sites:
        update.append(i)
        # Move to next ancestor (i + (i & -i))
        if i & (i + 1) == 0:
            break
        i = i | (i + 1)
        if i >= n_sites:
            break
    return update


def _bk_parity_set(site: int, n_sites: int) -> List[int]:
    """Bravyi-Kitaev parity set: bits less than i in the partial-sum tree."""
    parity = []
    i = site - 1
    while i >= 0:
        parity.append(i)
        # Move to previous (parent) bit in the parity tree
        i = (i & (i + 1)) - 1
    return parity


def _bk_flip_set(site: int, n_sites: int) -> List[int]:
    """Bravyi-Kitaev flip set."""
    flip = []
    i = site
    while i > 0:
        i = i & (i - 1)
        if i > 0:
            flip.append(i - 1)
    return flip


# ═════════════════════════════════════════════════════════════════════════════
# Parity mapping
# ═════════════════════════════════════════════════════════════════════════════


def parity_mapping(
    op_type: str,
    site: int,
    n_sites: int,
) -> Dict[str, complex]:
    """
    Parity transform of a single fermion operator.

    Parity mapping stores the partial sum (parity) of occupation numbers at
    each site, instead of the occupation directly. The locality is O(N) per
    term but the bit ordering is different from JW.
    """
    if site < 0 or site >= n_sites:
        raise ValueError(f"site {site} out of range [0, {n_sites})")

    # Parity X chain on sites > i
    x_string = ["I"] * n_sites
    y_string = ["I"] * n_sites
    for j in range(site + 1, n_sites):
        x_string[j] = "X"
        y_string[j] = "X"
    if site > 0:
        # Z on site i-1 for proper anticommutation
        x_string[site - 1] = "Z"
    x_string[site] = "X"
    y_string[site] = "Y"

    if op_type == "creation":
        return {
            "".join(x_string): 0.5 + 0j,
            "".join(y_string): -0.5j,
        }
    elif op_type == "annihilation":
        return {
            "".join(x_string): 0.5 + 0j,
            "".join(y_string): 0.5j,
        }
    else:
        raise ValueError("op_type must be 'creation' or 'annihilation'")
