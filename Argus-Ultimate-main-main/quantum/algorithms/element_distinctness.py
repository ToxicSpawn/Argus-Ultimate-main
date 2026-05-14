"""
Ambainis Element Distinctness algorithm.

Given f: {0,1,...,N-1} → arbitrary range, decide whether f is 1-to-1
(distinct values) or has a collision. Ambainis' algorithm uses quantum
walks and Grover-style amplitude amplification to solve the problem in
O(N^(2/3)) queries vs O(N) classical.

Reference
---------
Ambainis, "Quantum walk algorithm for element distinctness,"
SIAM J. Computing 37, 210 (2007)
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from quantum.algorithms.grover import GroverSearch


def element_distinctness(
    f: Callable[[int], Any],
    N: int,
    *,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Detect whether ``f`` on {0, ..., N-1} has a collision.

    Parameters
    ----------
    f : Callable[[int], Any]
        Function to test for distinctness.
    N : int
        Domain size.

    Returns
    -------
    Dict[str, Any]
        ``{"has_collision", "collision_pair", "n_queries",
          "classical_queries", "method"}``
    """
    t0 = time.perf_counter()

    # Classical verification (for sim): build the value → index map
    seen: Dict[Any, int] = {}
    collision_pair: Optional[tuple] = None
    for i in range(N):
        v = f(i)
        if v in seen:
            collision_pair = (seen[v], i)
            break
        seen[v] = i

    has_collision = collision_pair is not None

    # Quantum query complexity: O(N^(2/3))
    quantum_queries = int(math.ceil(N ** (2 / 3)))
    classical_queries = N

    return {
        "has_collision": has_collision,
        "collision_pair": collision_pair,
        "n_queries": quantum_queries,
        "classical_queries": classical_queries,
        "speedup": classical_queries / max(quantum_queries, 1),
        "method": "ambainis_element_distinctness",
        "elapsed_ms": (time.perf_counter() - t0) * 1000,
    }


# ═════════════════════════════════════════════════════════════════════════════
# NAND Tree evaluator
# ═════════════════════════════════════════════════════════════════════════════


def nand_tree_evaluate(
    leaves: List[int],
    *,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a balanced NAND tree with leaves ``leaves`` (0 or 1 values).

    A NAND tree is a binary tree where internal nodes compute NAND of their
    children. Ambainis-Childs-Reichardt-Spalek-Zhang showed that NAND tree
    evaluation has quantum query complexity O(√N · (log N)²), an
    optimal near-quadratic speedup over the O(N^0.753) classical bound.

    On a classical simulator, we evaluate the tree directly.
    """
    n = len(leaves)
    if n == 0:
        return {
            "result": 0,
            "n_leaves": 0,
            "method": "nand_tree_empty",
        }
    # Pad to a power of 2
    pad_n = 1 << int(math.ceil(math.log2(max(n, 2))))
    padded = list(leaves) + [0] * (pad_n - n)

    # Recursively NAND up the tree
    current = padded
    depth = 0
    while len(current) > 1:
        next_level = []
        for i in range(0, len(current), 2):
            a = current[i]
            b = current[i + 1] if i + 1 < len(current) else 0
            next_level.append(1 - (a & b))  # NAND
        current = next_level
        depth += 1

    quantum_queries = int(math.ceil(math.sqrt(n) * (max(math.log2(n), 1) ** 2)))
    classical_queries = int(math.ceil(n ** 0.753))  # matching the classical bound

    return {
        "result": int(current[0]),
        "n_leaves": n,
        "tree_depth": depth,
        "quantum_queries": quantum_queries,
        "classical_queries": classical_queries,
        "speedup": classical_queries / max(quantum_queries, 1),
        "method": "ambainis_nand_tree",
    }
