"""
Symmetry verification for error mitigation.

If the noise-free state is known to obey a symmetry (e.g., parity
conservation, particle number conservation), we can simply discard
measurement shots that violate the symmetry. The remaining shots give
a more accurate estimate of the observable.

Reference
---------
Bonet-Monroig, Sagastizabal, Singh, O'Brien, "Low-cost error mitigation by
symmetry verification," PRA 98, 062339 (2018)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


def discard_invalid_shots(
    counts: Dict[str, int],
    is_valid: Callable[[str], bool],
) -> Dict[str, int]:
    """
    Filter measurement counts to keep only shots satisfying the symmetry.

    Parameters
    ----------
    counts : Dict[str, int]
        Original measurement counts (bitstring → count).
    is_valid : Callable[[str], bool]
        Predicate that returns True for valid (symmetry-respecting) bitstrings.

    Returns
    -------
    Dict[str, int]
        Filtered counts.
    """
    return {bs: c for bs, c in counts.items() if is_valid(bs)}


def parity_symmetry_filter(
    counts: Dict[str, int],
    parity: int = 0,
) -> Dict[str, int]:
    """
    Keep only bitstrings whose Hamming weight has the specified parity.

    parity=0 → even number of 1s; parity=1 → odd.
    """
    return discard_invalid_shots(
        counts,
        is_valid=lambda bs: bs.count("1") % 2 == parity,
    )


def particle_number_filter(
    counts: Dict[str, int],
    target_count: int,
) -> Dict[str, int]:
    """
    Keep only bitstrings with exactly ``target_count`` 1s.

    Useful for fermion systems where particle number is conserved.
    """
    return discard_invalid_shots(
        counts,
        is_valid=lambda bs: bs.count("1") == target_count,
    )
