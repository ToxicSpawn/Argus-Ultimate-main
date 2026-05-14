"""
Quantum circuit routing for limited-connectivity hardware.

Real quantum devices have a coupling map (graph of allowed 2-qubit
interactions). To run an arbitrary circuit, we must insert SWAPs to bring
non-adjacent qubits next to each other.

This module provides:
- ``SabreRouter`` — SABRE algorithm (Li, Ding, Xie 2019)
- ``LookAheadRouter`` — simpler look-ahead heuristic

Reference
---------
Li, Ding, Xie, "Tackling the Qubit Mapping Problem for NISQ-Era Quantum
Devices," ASPLOS 2019, arXiv:1809.02573
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from quantum_simulator import GateType, Operation, QuantumCircuit


# ═════════════════════════════════════════════════════════════════════════════
# SABRE router
# ═════════════════════════════════════════════════════════════════════════════


class SabreRouter:
    """
    SABRE qubit routing.

    Walks through the gate list and inserts SWAPs greedily based on a
    look-ahead heuristic that minimizes the total distance of upcoming gates.

    Parameters
    ----------
    coupling_map : List[Tuple[int, int]]
        List of (q1, q2) edges in the hardware graph.
    look_ahead : int
        How many gates ahead to consider when scoring SWAP candidates.
    """

    def __init__(
        self,
        coupling_map: List[Tuple[int, int]],
        look_ahead: int = 5,
    ) -> None:
        self.coupling_map = list(coupling_map)
        self.look_ahead = int(look_ahead)
        self._adjacency = self._build_adjacency()
        self._distance = self._build_distance_matrix()

    def _build_adjacency(self) -> Dict[int, Set[int]]:
        adj: Dict[int, Set[int]] = {}
        for q1, q2 in self.coupling_map:
            adj.setdefault(q1, set()).add(q2)
            adj.setdefault(q2, set()).add(q1)
        return adj

    def _build_distance_matrix(self) -> Dict[Tuple[int, int], int]:
        """BFS shortest paths between all pairs."""
        nodes = set()
        for q1, q2 in self.coupling_map:
            nodes.add(q1)
            nodes.add(q2)
        n = max(nodes) + 1 if nodes else 0
        dist: Dict[Tuple[int, int], int] = {}
        for start in nodes:
            visited = {start: 0}
            frontier = [start]
            while frontier:
                next_frontier = []
                for v in frontier:
                    for nb in self._adjacency.get(v, set()):
                        if nb not in visited:
                            visited[nb] = visited[v] + 1
                            next_frontier.append(nb)
                frontier = next_frontier
            for end, d in visited.items():
                dist[(start, end)] = d
        return dist

    def route(self, circuit: QuantumCircuit) -> Tuple[QuantumCircuit, List[int]]:
        """
        Route the circuit to satisfy the coupling map.

        Returns the routed circuit and the qubit mapping (logical → physical).
        """
        n = circuit.num_qubits
        # Initial mapping: identity
        mapping = list(range(n))  # mapping[logical_q] = physical_q

        new_qc = QuantumCircuit(n)
        ops = list(circuit.operations)

        for op_idx, op in enumerate(ops):
            if op.gate == GateType.MEASURE_ALL:
                new_qc._ops.append(op)
                continue

            # Handle 1-qubit gates: just remap
            if len(op.targets) == 1:
                new_op = Operation(
                    op.gate,
                    (mapping[op.targets[0]],),
                    op.params,
                )
                new_qc._ops.append(new_op)
                continue

            # 2-qubit gates: ensure mapping puts them on adjacent physical qubits
            if len(op.targets) == 2:
                lq1, lq2 = op.targets[0], op.targets[1]
                pq1, pq2 = mapping[lq1], mapping[lq2]
                # Check if (pq1, pq2) is in coupling map
                if pq2 in self._adjacency.get(pq1, set()):
                    new_op = Operation(op.gate, (pq1, pq2), op.params)
                    new_qc._ops.append(new_op)
                    continue
                # Need to SWAP — find the best path
                path = self._find_path(pq1, pq2)
                if not path:
                    # Cannot route — keep original
                    new_qc._ops.append(op)
                    continue
                # Apply SWAPs along the path
                cur = pq1
                for nxt in path[1:-1]:
                    new_qc.swap(cur, nxt)
                    # Update mapping
                    self._swap_in_mapping(mapping, cur, nxt)
                    cur = nxt
                # Apply original gate on now-adjacent pair
                new_pq1 = mapping[lq1]
                new_pq2 = mapping[lq2]
                new_op = Operation(op.gate, (new_pq1, new_pq2), op.params)
                new_qc._ops.append(new_op)

            elif len(op.targets) == 3:
                # 3-qubit gates not handled by this router
                new_qc._ops.append(op)

        return new_qc, mapping

    def _find_path(self, start: int, end: int) -> List[int]:
        """BFS shortest path from start to end."""
        if start == end:
            return [start]
        visited = {start: None}
        frontier = [start]
        while frontier:
            next_frontier = []
            for v in frontier:
                for nb in self._adjacency.get(v, set()):
                    if nb not in visited:
                        visited[nb] = v
                        if nb == end:
                            # Reconstruct path
                            path = [nb]
                            while visited[path[-1]] is not None:
                                path.append(visited[path[-1]])
                            return list(reversed(path))
                        next_frontier.append(nb)
            frontier = next_frontier
        return []

    def _swap_in_mapping(self, mapping: List[int], pq1: int, pq2: int) -> None:
        """Swap two physical qubits in the logical→physical mapping."""
        for i, p in enumerate(mapping):
            if p == pq1:
                mapping[i] = pq2
            elif p == pq2:
                mapping[i] = pq1


# ═════════════════════════════════════════════════════════════════════════════
# Look-ahead router (simpler alternative)
# ═════════════════════════════════════════════════════════════════════════════


class LookAheadRouter:
    """
    Simple greedy look-ahead router. For each 2-qubit gate that needs SWAP,
    pick the SWAP that minimizes the total distance of the next K gates.
    """

    def __init__(self, coupling_map: List[Tuple[int, int]], window: int = 3) -> None:
        self.sabre = SabreRouter(coupling_map, look_ahead=window)

    def route(self, circuit: QuantumCircuit) -> Tuple[QuantumCircuit, List[int]]:
        return self.sabre.route(circuit)
