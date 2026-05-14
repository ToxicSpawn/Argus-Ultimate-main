"""
Stabilizer (Clifford-only) quantum simulator.

Uses the Aaronson-Gottesman tableau formalism to simulate Clifford circuits in
**O(n²) per gate**, regardless of qubit count. This is exponentially faster
than statevector simulation for circuits that only contain Clifford gates
(H, S, S†, CNOT, CZ, X, Y, Z) and Pauli measurements.

Reference
---------
Aaronson, Gottesman, "Improved simulation of stabilizer circuits,"
Physical Review A 70, 052328 (2004). arXiv:quant-ph/0406196

Use cases
---------
- **Error correction code testing**: stabilizer codes use only Clifford gates.
- **Large state preparation**: Bell, GHZ, cluster states.
- **ZNE reference**: noise-free Clifford reference for variational circuits.
- **Random Clifford sampling**: 50-qubit, 100-qubit Clifford circuits run in
  milliseconds.

Limitation
----------
Cannot simulate non-Clifford gates (T, T†, RX/RY/RZ except π/2 multiples).
For those, use the statevector backend in ``quantum_simulator.py``.

Tableau format
--------------
A stabilizer state is represented by a 2n × (2n + 1) binary matrix:
    rows 0..n-1     : destabilizer generators
    rows n..2n-1    : stabilizer generators
    columns 0..n-1  : X part
    columns n..2n-1 : Z part
    column 2n        : phase (sign bit, 0 = +1, 1 = -1)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# StabilizerSimulator
# ═════════════════════════════════════════════════════════════════════════════


class StabilizerSimulator:
    """
    Aaronson-Gottesman tableau simulator for Clifford circuits.

    Parameters
    ----------
    n_qubits : int
        Number of qubits. Can be very large (1000+) since cost is O(n²).

    Examples
    --------
    >>> sim = StabilizerSimulator(50)
    >>> sim.apply_h(0)
    >>> for i in range(49):
    ...     sim.apply_cnot(i, i + 1)
    >>> # 50-qubit GHZ state prepared in O(n²) = O(2500) ops
    >>> result = sim.measure_all(shots=1024)
    """

    def __init__(self, n_qubits: int) -> None:
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        self.n_qubits = int(n_qubits)
        n = self.n_qubits
        # Tableau: 2n rows, 2n + 1 columns
        # Row i in [0, n) is destabilizer i; row i in [n, 2n) is stabilizer i.
        # Cols 0..n-1 are X part; cols n..2n-1 are Z part; col 2n is phase.
        self.tab = np.zeros((2 * n, 2 * n + 1), dtype=np.int8)
        # Initialize to |0...0⟩: destabilizer i = X_i, stabilizer i = Z_i
        for i in range(n):
            self.tab[i, i] = 1  # destabilizer X_i
            self.tab[n + i, n + i] = 1  # stabilizer Z_i

    # ── Clifford gate operations ─────────────────────────────────────────────

    def apply_h(self, q: int) -> None:
        """Hadamard on qubit q. Swaps the X and Z parts."""
        n = self.n_qubits
        if not (0 <= q < n):
            raise ValueError(f"qubit {q} out of range")
        for i in range(2 * n):
            x = self.tab[i, q]
            z = self.tab[i, n + q]
            self.tab[i, 2 * n] ^= x & z  # phase flip for XZ → ZX
            self.tab[i, q] = z
            self.tab[i, n + q] = x

    def apply_s(self, q: int) -> None:
        """S gate (phase) on qubit q. Maps X → Y, Y → -X, Z → Z."""
        n = self.n_qubits
        if not (0 <= q < n):
            raise ValueError(f"qubit {q} out of range")
        for i in range(2 * n):
            x = self.tab[i, q]
            z = self.tab[i, n + q]
            self.tab[i, 2 * n] ^= x & z
            self.tab[i, n + q] = x ^ z

    def apply_sdg(self, q: int) -> None:
        """S† gate. Apply S three times (S^4 = I)."""
        self.apply_s(q)
        self.apply_s(q)
        self.apply_s(q)

    def apply_x(self, q: int) -> None:
        """X gate. Anticommutes with Z, so flip stabilizer phases that have Z."""
        n = self.n_qubits
        for i in range(2 * n):
            self.tab[i, 2 * n] ^= self.tab[i, n + q]

    def apply_z(self, q: int) -> None:
        """Z gate. Anticommutes with X, so flip phases that have X."""
        n = self.n_qubits
        for i in range(2 * n):
            self.tab[i, 2 * n] ^= self.tab[i, q]

    def apply_y(self, q: int) -> None:
        """Y = iXZ. Apply X then Z (with phase tracking via S/H)."""
        # Y = S X S†
        self.apply_z(q)
        self.apply_x(q)

    def apply_cnot(self, control: int, target: int) -> None:
        """CNOT(control, target). Maps X_c → X_c X_t, Z_t → Z_c Z_t."""
        n = self.n_qubits
        c = int(control)
        t = int(target)
        if c == t:
            raise ValueError("control and target must differ")
        if not (0 <= c < n and 0 <= t < n):
            raise ValueError("qubit out of range")
        for i in range(2 * n):
            xc = self.tab[i, c]
            xt = self.tab[i, t]
            zc = self.tab[i, n + c]
            zt = self.tab[i, n + t]
            # Phase update: r ⊕= xc·zt·(xt⊕zc⊕1)
            self.tab[i, 2 * n] ^= xc & zt & (xt ^ zc ^ 1)
            self.tab[i, t] ^= xc
            self.tab[i, n + c] ^= zt

    def apply_cz(self, q1: int, q2: int) -> None:
        """CZ gate. CZ = H_2 · CNOT(q1, q2) · H_2."""
        self.apply_h(q2)
        self.apply_cnot(q1, q2)
        self.apply_h(q2)

    # ── Measurement ───────────────────────────────────────────────────────────

    def measure(self, q: int, *, rng: Optional[np.random.Generator] = None) -> int:
        """
        Measure qubit q in the Z basis. Returns 0 or 1.

        Updates the tableau in place. If the outcome is random (qubit q in
        a +/- superposition), draws from the rng.
        """
        if rng is None:
            rng = np.random.default_rng()
        n = self.n_qubits

        # Find a stabilizer with X_q = 1 → outcome is random
        p = -1
        for i in range(n, 2 * n):
            if self.tab[i, q] == 1:
                p = i
                break

        if p >= 0:
            # Random outcome
            outcome = int(rng.integers(0, 2))
            # Set destabilizer p-n to the old stabilizer
            for j in range(2 * n + 1):
                self.tab[p - n, j] = self.tab[p, j]
            # Replace stabilizer p with Z_q × (-1)^outcome
            self.tab[p, :] = 0
            self.tab[p, n + q] = 1
            self.tab[p, 2 * n] = outcome
            # Eliminate other stabilizers/destabilizers with X_q = 1
            for i in range(2 * n):
                if i != p and self.tab[i, q] == 1:
                    self._row_mult(i, p)
            return outcome
        else:
            # Deterministic outcome — accumulate via destabilizers
            # Build a temporary "scratch" row at index 2n to compute the result
            scratch = np.zeros(2 * n + 1, dtype=np.int8)
            for i in range(n):
                if self.tab[i, q] == 1:
                    scratch = self._add_pauli(scratch, self.tab[n + i])
            return int(scratch[2 * n])

    def measure_all(
        self, *, shots: int = 1024, seed: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Measure all qubits and return counts. Re-runs the simulator from a
        fresh tableau for each shot to capture random outcomes correctly.

        Note: this is wasteful for circuits ending in deterministic states.
        For pure-state final tableaus, you can call ``measure(q)`` once per
        qubit.
        """
        rng = np.random.default_rng(seed)
        n = self.n_qubits
        counts: Dict[str, int] = {}

        # Save current tableau
        saved = self.tab.copy()

        for _ in range(shots):
            self.tab = saved.copy()
            bits = []
            for q in range(n):
                bit = self.measure(q, rng=rng)
                bits.append(bit)
            # Bitstring with qubit 0 = LSB (rightmost)
            bs = "".join(str(b) for b in reversed(bits))
            counts[bs] = counts.get(bs, 0) + 1

        # Restore
        self.tab = saved
        return counts

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _row_mult(self, i: int, h: int) -> None:
        """Multiply row i by row h (in symplectic Pauli arithmetic)."""
        n = self.n_qubits
        # Phase update via the g function (Aaronson-Gottesman 2004)
        phase = 0
        for j in range(n):
            x_i = self.tab[i, j]
            z_i = self.tab[i, n + j]
            x_h = self.tab[h, j]
            z_h = self.tab[h, n + j]
            phase += _g(x_i, z_i, x_h, z_h)
        new_phase = (
            2 * self.tab[i, 2 * n] + 2 * self.tab[h, 2 * n] + phase
        ) % 4
        self.tab[i, 2 * n] = new_phase // 2
        for j in range(2 * n):
            self.tab[i, j] ^= self.tab[h, j]

    def _add_pauli(self, scratch: np.ndarray, row: np.ndarray) -> np.ndarray:
        """Add a row to scratch with phase tracking."""
        n = self.n_qubits
        phase = 0
        for j in range(n):
            x_s = scratch[j]
            z_s = scratch[n + j]
            x_r = row[j]
            z_r = row[n + j]
            phase += _g(x_s, z_s, x_r, z_r)
        new_phase = (
            2 * scratch[2 * n] + 2 * row[2 * n] + phase
        ) % 4
        result = scratch.copy()
        result[2 * n] = new_phase // 2
        for j in range(2 * n):
            result[j] ^= row[j]
        return result


def _g(x1: int, z1: int, x2: int, z2: int) -> int:
    """
    Aaronson-Gottesman g function: contributes to the phase when multiplying
    two single-qubit Paulis encoded as (x, z) bits.
    """
    if x1 == 0 and z1 == 0:
        return 0
    if x1 == 1 and z1 == 1:
        return z2 - x2
    if x1 == 1 and z1 == 0:
        return z2 * (2 * x2 - 1)
    # x1 == 0, z1 == 1
    return x2 * (1 - 2 * z2)


# ═════════════════════════════════════════════════════════════════════════════
# Convenience: build a stabilizer simulator from a Clifford-only QuantumCircuit
# ═════════════════════════════════════════════════════════════════════════════


def stabilizer_simulate(circuit, *, shots: int = 1024, seed: Optional[int] = None) -> Dict:
    """
    Run a Clifford-only ``QuantumCircuit`` through the stabilizer simulator.

    Raises ValueError if any non-Clifford gate is encountered.
    """
    from quantum_simulator import GateType

    n = circuit.num_qubits
    sim = StabilizerSimulator(n)

    _CLIFFORD_GATES = {
        GateType.H, GateType.X, GateType.Y, GateType.Z,
        GateType.S, GateType.SDG, GateType.CNOT, GateType.CZ,
    }

    for op in circuit.operations:
        if op.gate == GateType.MEASURE_ALL:
            break
        if op.gate not in _CLIFFORD_GATES:
            raise ValueError(
                f"Non-Clifford gate {op.gate} cannot be simulated by stabilizer "
                "backend. Use the statevector backend instead."
            )

        if op.gate == GateType.H:
            sim.apply_h(op.targets[0])
        elif op.gate == GateType.X:
            sim.apply_x(op.targets[0])
        elif op.gate == GateType.Y:
            sim.apply_y(op.targets[0])
        elif op.gate == GateType.Z:
            sim.apply_z(op.targets[0])
        elif op.gate == GateType.S:
            sim.apply_s(op.targets[0])
        elif op.gate == GateType.SDG:
            sim.apply_sdg(op.targets[0])
        elif op.gate == GateType.CNOT:
            sim.apply_cnot(op.targets[0], op.targets[1])
        elif op.gate == GateType.CZ:
            sim.apply_cz(op.targets[0], op.targets[1])

    counts = sim.measure_all(shots=shots, seed=seed)
    return {
        "counts": counts,
        "shots": shots,
        "backend": "stabilizer",
        "num_qubits": n,
        "method": "aaronson_gottesman",
    }
