"""
Quantum Circuit Optimizer for NISQ devices.

Optimizes quantum circuits to reduce gate count and depth, making them
more resilient to noise on real hardware. Works with a gate-sequence
representation that is backend-agnostic.

Optimizations:
- Gate cancellation: H*H=I, X*X=I, CNOT*CNOT=I, Rz merging
- Commutation analysis: reorder gates to enable more cancellations
- Depth reduction: parallelize independent gates into layers
- Execution time estimation per hardware backend

Gate representation:
    Each gate is a dict: {"name": str, "qubits": list[int], "params": list[float]}
    Examples:
        {"name": "H", "qubits": [0], "params": []}
        {"name": "CNOT", "qubits": [0, 1], "params": []}
        {"name": "Rz", "qubits": [2], "params": [0.5]}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Gates that are self-inverse (applying twice = identity)
_SELF_INVERSE_GATES = {"H", "X", "Y", "Z", "CNOT", "CX", "SWAP"}

# Gates that can be merged by adding parameters
_MERGEABLE_GATES = {"Rz", "Rx", "Ry", "RZ", "RX", "RY", "P", "U1"}

# Pairs of gates that commute (can be swapped without changing result)
# when they act on non-overlapping qubits
_SINGLE_QUBIT_GATES = {"H", "X", "Y", "Z", "Rz", "Rx", "Ry", "RZ", "RX", "RY", "S", "T", "P", "U1"}
_TWO_QUBIT_GATES = {"CNOT", "CX", "CZ", "SWAP"}

# Hardware gate times in nanoseconds
_GATE_TIMES: Dict[str, Dict[str, float]] = {
    "ibm": {
        "single": 35.0,    # Single-qubit gate
        "cx": 300.0,        # CNOT / CX
        "measure": 500.0,   # Measurement
    },
    "ionq": {
        "single": 10_000.0,   # Single-qubit gate (~10 us)
        "cx": 200_000.0,      # MS gate (~200 us)
        "measure": 100_000.0, # Measurement (~100 us)
    },
    "rigetti": {
        "single": 40.0,
        "cx": 200.0,
        "measure": 600.0,
    },
}


class QuantumCircuitOptimizer:
    """
    Optimize quantum circuits for NISQ devices.

    Takes a list of gate dictionaries and produces an optimized sequence
    with fewer gates and/or reduced circuit depth.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Gate cancellation
    # ------------------------------------------------------------------

    def gate_cancellation(self, gate_sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Cancel adjacent inverse gates.

        Rules:
        - H*H = I (remove both)
        - X*X = I (remove both)
        - Y*Y = I (remove both)
        - Z*Z = I (remove both)
        - CNOT*CNOT = I on same qubits (remove both)
        - Rz(a)*Rz(b) = Rz(a+b) on same qubit (merge)
        - Rx(a)*Rx(b) = Rx(a+b) on same qubit (merge)
        - Ry(a)*Ry(b) = Ry(a+b) on same qubit (merge)
        - Remove Rz/Rx/Ry with angle ~0 (mod 2pi)

        Args:
            gate_sequence: List of gate dicts.

        Returns:
            Optimized gate sequence with cancellations applied.
        """
        if not gate_sequence:
            return []

        result: List[Dict[str, Any]] = []
        i = 0

        while i < len(gate_sequence):
            gate = gate_sequence[i]

            # Try to cancel/merge with the last gate in result
            if result:
                prev = result[-1]

                # Check same gate name and same qubits
                if (prev["name"] == gate["name"]
                        and prev["qubits"] == gate["qubits"]):

                    # Self-inverse cancellation
                    if gate["name"] in _SELF_INVERSE_GATES:
                        result.pop()
                        i += 1
                        continue

                    # Mergeable rotation gates
                    if gate["name"] in _MERGEABLE_GATES:
                        merged_angle = prev["params"][0] + gate["params"][0]
                        # Remove if angle is effectively 0 (mod 2pi)
                        reduced = merged_angle % (2 * np.pi)
                        if reduced < 1e-10 or abs(reduced - 2 * np.pi) < 1e-10:
                            result.pop()
                            i += 1
                            continue
                        else:
                            result[-1] = {
                                "name": gate["name"],
                                "qubits": list(gate["qubits"]),
                                "params": [merged_angle],
                            }
                            i += 1
                            continue

            result.append(gate)
            i += 1

        # Second pass: remove near-zero rotation gates
        final: List[Dict[str, Any]] = []
        for gate in result:
            if gate["name"] in _MERGEABLE_GATES and gate.get("params"):
                angle = gate["params"][0] % (2 * np.pi)
                if angle < 1e-10 or abs(angle - 2 * np.pi) < 1e-10:
                    continue
            final.append(gate)

        return final

    # ------------------------------------------------------------------
    # Commutation optimization
    # ------------------------------------------------------------------

    def commutation_optimization(self, gate_sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reorder commuting gates to enable more cancellations.

        Strategy:
        1. Gates on non-overlapping qubits always commute.
        2. Diagonal gates (Rz, Z, P, U1) on the same qubit commute.
        3. After reordering, apply gate_cancellation.

        Uses a bubble-sort-like approach: repeatedly try swapping adjacent
        gates that commute, then check if cancellation is possible.

        Args:
            gate_sequence: List of gate dicts.

        Returns:
            Optimized gate sequence after commutation + cancellation.
        """
        if len(gate_sequence) <= 1:
            return list(gate_sequence)

        gates = [dict(g) for g in gate_sequence]  # deep copy
        changed = True
        passes = 0
        max_passes = 3  # limit passes for efficiency

        while changed and passes < max_passes:
            changed = False
            passes += 1

            for i in range(len(gates) - 1):
                g1 = gates[i]
                g2 = gates[i + 1]

                # Check if they commute
                if not self._gates_commute(g1, g2):
                    continue

                # Check if swapping enables a cancellation
                # (look at what's before g1 or after g2)
                swap_beneficial = False

                # Would g2 cancel with gates[i-1]?
                if i > 0:
                    prev = gates[i - 1]
                    if self._can_cancel_or_merge(prev, g2):
                        swap_beneficial = True

                # Would g1 cancel with gates[i+2]?
                if i + 2 < len(gates):
                    nxt = gates[i + 2]
                    if self._can_cancel_or_merge(g1, nxt):
                        swap_beneficial = True

                if swap_beneficial:
                    gates[i], gates[i + 1] = gates[i + 1], gates[i]
                    changed = True

        # Apply gate cancellation after reordering
        return self.gate_cancellation(gates)

    def _gates_commute(self, g1: Dict[str, Any], g2: Dict[str, Any]) -> bool:
        """Check if two gates commute."""
        q1 = set(g1["qubits"])
        q2 = set(g2["qubits"])

        # Non-overlapping qubits always commute
        if not q1.intersection(q2):
            return True

        # Diagonal gates on the same qubit commute
        diagonal_gates = {"Rz", "RZ", "Z", "P", "U1", "S", "T"}
        if g1["name"] in diagonal_gates and g2["name"] in diagonal_gates:
            return True

        return False

    def _can_cancel_or_merge(self, g1: Dict[str, Any], g2: Dict[str, Any]) -> bool:
        """Check if two gates can cancel or merge."""
        if g1["name"] != g2["name"] or g1["qubits"] != g2["qubits"]:
            return False
        if g1["name"] in _SELF_INVERSE_GATES:
            return True
        if g1["name"] in _MERGEABLE_GATES:
            return True
        return False

    # ------------------------------------------------------------------
    # Depth reduction
    # ------------------------------------------------------------------

    def depth_reduction(
        self,
        gate_sequence: List[Dict[str, Any]],
        n_qubits: int,
    ) -> List[List[Dict[str, Any]]]:
        """
        Parallelize independent gates to reduce circuit depth.

        Gates on non-overlapping qubits can run simultaneously. This method
        groups gates into layers where all gates in a layer can execute
        in parallel.

        Args:
            gate_sequence: List of gate dicts.
            n_qubits: Total number of qubits in the circuit.

        Returns:
            List of layers, where each layer is a list of gates that can
            execute in parallel.
        """
        if not gate_sequence:
            return []

        layers: List[List[Dict[str, Any]]] = []
        # Track which qubit is available at which layer
        qubit_available_at: Dict[int, int] = {q: 0 for q in range(n_qubits)}

        for gate in gate_sequence:
            qubits = gate["qubits"]

            # Find the earliest layer where all qubits used by this gate are free
            earliest = max(qubit_available_at.get(q, 0) for q in qubits)

            # Place gate in this layer
            while len(layers) <= earliest:
                layers.append([])
            layers[earliest].append(gate)

            # Mark qubits as occupied through this layer
            for q in qubits:
                qubit_available_at[q] = earliest + 1

        return layers

    # ------------------------------------------------------------------
    # Full optimization pipeline
    # ------------------------------------------------------------------

    def optimize(
        self,
        gate_sequence: List[Dict[str, Any]],
        n_qubits: int,
    ) -> Dict[str, Any]:
        """
        Full optimization pipeline: commutation + cancellation + depth reduction.

        Args:
            gate_sequence: List of gate dicts.
            n_qubits: Total number of qubits.

        Returns:
            optimized_gates: list - optimized gate sequence
            original_depth: int - depth before optimization
            optimized_depth: int - depth after optimization
            original_gate_count: int
            optimized_gate_count: int
            gates_removed: int
            depth_reduction_pct: float
            gate_reduction_pct: float
        """
        original_count = len(gate_sequence)
        original_layers = self.depth_reduction(gate_sequence, n_qubits)
        original_depth = len(original_layers)

        # Apply commutation + cancellation
        optimized = self.commutation_optimization(gate_sequence)
        optimized_count = len(optimized)

        # Compute depth of optimized circuit
        optimized_layers = self.depth_reduction(optimized, n_qubits)
        optimized_depth = len(optimized_layers)

        gates_removed = original_count - optimized_count
        depth_reduction_pct = (
            (original_depth - optimized_depth) / max(original_depth, 1) * 100.0
        )
        gate_reduction_pct = (
            gates_removed / max(original_count, 1) * 100.0
        )

        return {
            "optimized_gates": optimized,
            "original_depth": original_depth,
            "optimized_depth": optimized_depth,
            "original_gate_count": original_count,
            "optimized_gate_count": optimized_count,
            "gates_removed": gates_removed,
            "depth_reduction_pct": depth_reduction_pct,
            "gate_reduction_pct": gate_reduction_pct,
        }

    # ------------------------------------------------------------------
    # Execution time estimation
    # ------------------------------------------------------------------

    def estimate_execution_time(
        self,
        gate_sequence: List[Dict[str, Any]],
        backend: str = "ibm",
    ) -> Dict[str, Any]:
        """
        Estimate execution time on target hardware.

        Accounts for parallelism (depth, not gate count) and backend-specific
        gate times. Includes measurement time.

        Args:
            gate_sequence: List of gate dicts.
            backend: Hardware backend ('ibm', 'ionq', 'rigetti').

        Returns:
            total_time_us: float - estimated wall-clock execution time
            gate_time_us: float - time for gates only
            measurement_time_us: float - time for final measurements
            n_layers: int - circuit depth
            bottleneck: str - what dominates execution time
            recommended_backend: str - which backend is faster
        """
        if backend not in _GATE_TIMES:
            backend = "ibm"  # default

        times = _GATE_TIMES[backend]

        # Determine depth (layers)
        n_qubits = 0
        for gate in gate_sequence:
            for q in gate["qubits"]:
                n_qubits = max(n_qubits, q + 1)

        layers = self.depth_reduction(gate_sequence, max(n_qubits, 1))
        n_layers = len(layers)

        # Compute time per layer (maximum gate time in each layer)
        gate_time_ns = 0.0
        for layer in layers:
            max_time_in_layer = 0.0
            for gate in layer:
                if gate["name"] in _TWO_QUBIT_GATES:
                    max_time_in_layer = max(max_time_in_layer, times["cx"])
                else:
                    max_time_in_layer = max(max_time_in_layer, times["single"])
            gate_time_ns += max_time_in_layer

        gate_time_us = gate_time_ns / 1000.0
        measurement_time_us = times["measure"] * max(n_qubits, 1) / 1000.0
        total_time_us = gate_time_us + measurement_time_us

        # Determine bottleneck
        if measurement_time_us > gate_time_us:
            bottleneck = "measurement"
        else:
            # Count 2-qubit gates
            two_qubit_count = sum(
                1 for g in gate_sequence if g["name"] in _TWO_QUBIT_GATES
            )
            if two_qubit_count > len(gate_sequence) * 0.3:
                bottleneck = "two_qubit_gates"
            else:
                bottleneck = "circuit_depth"

        # Compare backends to find recommended
        best_backend = backend
        best_time = total_time_us
        for alt_backend, alt_times in _GATE_TIMES.items():
            alt_gate_ns = 0.0
            for layer in layers:
                max_t = 0.0
                for gate in layer:
                    if gate["name"] in _TWO_QUBIT_GATES:
                        max_t = max(max_t, alt_times["cx"])
                    else:
                        max_t = max(max_t, alt_times["single"])
                alt_gate_ns += max_t
            alt_total = alt_gate_ns / 1000.0 + alt_times["measure"] * max(n_qubits, 1) / 1000.0
            if alt_total < best_time:
                best_time = alt_total
                best_backend = alt_backend

        return {
            "total_time_us": total_time_us,
            "gate_time_us": gate_time_us,
            "measurement_time_us": measurement_time_us,
            "n_layers": n_layers,
            "bottleneck": bottleneck,
            "recommended_backend": best_backend,
        }
