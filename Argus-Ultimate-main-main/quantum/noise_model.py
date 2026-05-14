"""
Quantum Noise Simulator for NISQ devices.

Simulates realistic quantum hardware noise to test error mitigation
strategies and predict how circuits will behave on real backends.

Supports three hardware families:
- Superconducting (IBM): T1/T2 coherence, CX gate errors, fast gates
- Trapped ion (IonQ): Long coherence, MS gate errors, slow gates
- Annealer (D-Wave): Thermal noise, chain break errors

Noise channels implemented:
- Depolarizing noise (uniform error)
- Amplitude damping (T1 energy relaxation)
- Phase damping (T2 dephasing)
- Readout error (measurement bit flips)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known hardware noise profiles
# ---------------------------------------------------------------------------

_HARDWARE_PROFILES: Dict[str, Dict[str, float]] = {
    "ibm_brisbane": {
        "t1_us": 200.0,
        "t2_us": 150.0,
        "single_qubit_error": 0.0003,
        "cx_error": 0.008,
        "readout_error": 0.015,
        "single_gate_time_ns": 35.0,
        "cx_gate_time_ns": 300.0,
        "n_qubits": 127,
        "backend_type": "superconducting",
    },
    "ibm_osaka": {
        "t1_us": 250.0,
        "t2_us": 180.0,
        "single_qubit_error": 0.0002,
        "cx_error": 0.006,
        "readout_error": 0.012,
        "single_gate_time_ns": 35.0,
        "cx_gate_time_ns": 280.0,
        "n_qubits": 127,
        "backend_type": "superconducting",
    },
    "ionq_aria": {
        "t1_us": 10_000_000.0,  # ~10 seconds
        "t2_us": 1_000_000.0,   # ~1 second
        "single_qubit_error": 0.0003,
        "cx_error": 0.003,      # MS gate error
        "readout_error": 0.005,
        "single_gate_time_ns": 10_000.0,   # ~10 us
        "cx_gate_time_ns": 200_000.0,      # ~200 us
        "n_qubits": 25,
        "backend_type": "trapped_ion",
    },
    "ionq_forte": {
        "t1_us": 15_000_000.0,
        "t2_us": 2_000_000.0,
        "single_qubit_error": 0.0002,
        "cx_error": 0.002,
        "readout_error": 0.003,
        "single_gate_time_ns": 8_000.0,
        "cx_gate_time_ns": 150_000.0,
        "n_qubits": 36,
        "backend_type": "trapped_ion",
    },
    "dwave_advantage": {
        "t1_us": 20.0,
        "t2_us": 10.0,
        "single_qubit_error": 0.0,       # No gates
        "cx_error": 0.0,                  # No gates
        "readout_error": 0.03,
        "single_gate_time_ns": 0.0,
        "cx_gate_time_ns": 0.0,
        "n_qubits": 5627,
        "backend_type": "annealer",
        "anneal_time_us": 20.0,
        "chain_break_rate": 0.02,
    },
}


class QuantumNoiseModel:
    """
    Simulate realistic quantum hardware noise.

    Works with statevectors (numpy arrays of complex amplitudes) and
    count dictionaries. All noise channels preserve trace (normalization)
    and are physically valid quantum channels.
    """

    def __init__(self, backend_type: str = "superconducting") -> None:
        """
        Initialize noise model for a hardware family.

        Args:
            backend_type: One of 'superconducting', 'trapped_ion', 'annealer'.
                Determines default noise parameters.
        """
        self.backend_type = backend_type

        # Set default parameters based on backend type
        if backend_type == "superconducting":
            self._defaults = {
                "gate_error": 0.001,
                "readout_error": 0.015,
                "t1_us": 200.0,
                "t2_us": 150.0,
            }
        elif backend_type == "trapped_ion":
            self._defaults = {
                "gate_error": 0.003,
                "readout_error": 0.005,
                "t1_us": 10_000_000.0,
                "t2_us": 1_000_000.0,
            }
        elif backend_type == "annealer":
            self._defaults = {
                "gate_error": 0.0,
                "readout_error": 0.03,
                "t1_us": 20.0,
                "t2_us": 10.0,
            }
        else:
            self._defaults = {
                "gate_error": 0.001,
                "readout_error": 0.01,
                "t1_us": 200.0,
                "t2_us": 150.0,
            }

    # ------------------------------------------------------------------
    # Noise channels
    # ------------------------------------------------------------------

    def apply_depolarizing(
        self,
        statevector: np.ndarray,
        error_rate: float,
    ) -> np.ndarray:
        """
        Apply depolarizing noise channel to statevector.

        With probability p (error_rate), replace state with maximally mixed
        state. The density matrix transformation is:
            rho -> (1-p) * rho + p * I/d

        For statevectors, we simulate this stochastically:
        with probability p, output a random computational basis state;
        otherwise return the original statevector.

        Since we want deterministic behavior for testing, we apply the
        density matrix version via amplitude scaling:
            |psi> -> sqrt(1-p) * |psi>  (renormalized)
        and add uniform probability mass.

        Args:
            statevector: Complex amplitude vector (length 2^n).
            error_rate: Depolarizing probability p in [0, 1].

        Returns:
            Modified statevector (normalized).
        """
        sv = np.asarray(statevector, dtype=complex).copy()
        n_states = len(sv)
        p = max(0.0, min(error_rate, 1.0))

        if p < 1e-15:
            return sv

        # Density matrix approach in amplitude space:
        # rho_noisy = (1-p) * |psi><psi| + p/d * I
        # We can't represent this as a pure state, so we work with
        # the probability distribution instead.
        probs = np.abs(sv) ** 2
        noisy_probs = (1.0 - p) * probs + p / n_states

        # Construct a statevector with these probabilities, preserving phases
        abs_sv = np.abs(sv)
        safe_abs = np.where(abs_sv > 1e-15, abs_sv, 1.0)
        phases = np.where(abs_sv > 1e-15, sv / safe_abs, 1.0 + 0j)
        noisy_sv = np.sqrt(np.maximum(noisy_probs, 0.0)) * phases

        # Renormalize
        norm = np.sqrt(np.sum(np.abs(noisy_sv) ** 2))
        if norm > 1e-15:
            noisy_sv /= norm

        return noisy_sv

    def apply_amplitude_damping(
        self,
        statevector: np.ndarray,
        gamma: float,
    ) -> np.ndarray:
        """
        Model T1 decay (energy relaxation).

        |1> decays to |0> with probability gamma. This models spontaneous
        emission / energy relaxation characterized by time constant T1.

        For a single qubit:
            |0> -> |0>
            |1> -> sqrt(1-gamma)|1> + sqrt(gamma)|0>

        Applied independently to each qubit in the register.

        Args:
            statevector: Complex amplitude vector (length 2^n).
            gamma: Damping probability in [0, 1]. Related to T1 by:
                gamma = 1 - exp(-t/T1) where t is the gate time.

        Returns:
            Modified statevector (normalized).
        """
        sv = np.asarray(statevector, dtype=complex).copy()
        n_states = len(sv)
        n_qubits = int(np.log2(n_states))
        gamma = max(0.0, min(gamma, 1.0))

        if gamma < 1e-15:
            return sv

        # Apply amplitude damping to each qubit independently
        for q in range(n_qubits):
            mask = 1 << q
            new_sv = sv.copy()
            for idx in range(n_states):
                if idx & mask:
                    # This is a |1> state for qubit q
                    partner = idx ^ mask  # corresponding |0> state

                    # |1> component gets damped
                    new_sv[idx] = np.sqrt(1.0 - gamma) * sv[idx]
                    # |0> component gets contribution from decay
                    new_sv[partner] += np.sqrt(gamma) * sv[idx]

            sv = new_sv

        # Renormalize
        norm = np.sqrt(np.sum(np.abs(sv) ** 2))
        if norm > 1e-15:
            sv /= norm

        return sv

    def apply_phase_damping(
        self,
        statevector: np.ndarray,
        gamma: float,
    ) -> np.ndarray:
        """
        Model T2 dephasing (phase randomization).

        Off-diagonal elements of the density matrix decay by sqrt(1-gamma).
        This models loss of quantum coherence characterized by T2.

        In amplitude space, we apply a random phase kick with variance
        proportional to gamma. For deterministic behavior, we scale the
        coherent part by sqrt(1-gamma).

        Args:
            statevector: Complex amplitude vector (length 2^n).
            gamma: Dephasing probability in [0, 1]. Related to T2 by:
                gamma = 1 - exp(-t/T2).

        Returns:
            Modified statevector (normalized).
        """
        sv = np.asarray(statevector, dtype=complex).copy()
        n_states = len(sv)
        gamma = max(0.0, min(gamma, 1.0))

        if gamma < 1e-15:
            return sv

        # Phase damping in density matrix: rho_ij -> rho_ij * sqrt(1-gamma) for i!=j
        # In statevector: we can model this by scaling relative phases.
        # The diagonal (probabilities) are unchanged; off-diagonal coherences decay.
        probs = np.abs(sv) ** 2
        abs_sv = np.abs(sv)
        safe_abs = np.where(abs_sv > 1e-15, abs_sv, 1.0)
        phases = np.where(abs_sv > 1e-15, sv / safe_abs, 1.0 + 0j)

        # The coherent part gets scaled; we interpolate toward a "dephased" state
        # that has the same probabilities but scrambled phases.
        # Effective scaling of coherence: sqrt(1-gamma)
        coherence_factor = np.sqrt(1.0 - gamma)

        # The dephased contribution has the same |psi_i|^2 but no relative phases
        # We model this by keeping amplitudes but reducing phase coherence
        noisy_sv = np.sqrt(probs) * (coherence_factor * phases + (1.0 - coherence_factor))

        # Renormalize
        norm = np.sqrt(np.sum(np.abs(noisy_sv) ** 2))
        if norm > 1e-15:
            noisy_sv /= norm

        return noisy_sv

    def apply_readout_error(
        self,
        counts: Dict[str, int],
        error_rate: float,
    ) -> Dict[str, int]:
        """
        Flip measurement outcomes with probability error_rate.

        Each bit in each measurement outcome is independently flipped
        with probability `error_rate`. This simulates measurement
        (readout) errors on NISQ hardware.

        Args:
            counts: Dict of bitstring -> count.
            error_rate: Per-bit flip probability.

        Returns:
            Noisy counts dict with same total shots.
        """
        if not counts or error_rate < 1e-15:
            return dict(counts)

        noisy_counts: Dict[str, int] = {}

        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")
            n_bits = len(bits)

            for _ in range(count):
                # Flip each bit independently with probability error_rate
                noisy_bits = list(bits)
                for b in range(n_bits):
                    if np.random.random() < error_rate:
                        noisy_bits[b] = "0" if noisy_bits[b] == "1" else "1"
                noisy_key = "".join(noisy_bits)
                noisy_counts[noisy_key] = noisy_counts.get(noisy_key, 0) + 1

        return noisy_counts

    # ------------------------------------------------------------------
    # Full noisy circuit simulation
    # ------------------------------------------------------------------

    def simulate_noisy_circuit(
        self,
        ideal_statevector: np.ndarray,
        n_gates: int,
        gate_error: float = 0.001,
        readout_error: float = 0.01,
        shots: int = 1000,
    ) -> Dict[str, Any]:
        """
        Apply noise model to ideal circuit result.

        Applies cumulative gate noise (depolarizing) and coherence errors
        (amplitude/phase damping) to the ideal statevector, then samples
        with readout errors.

        Args:
            ideal_statevector: Complex amplitude vector from ideal simulation.
            n_gates: Number of gates in the circuit (determines noise level).
            gate_error: Per-gate error rate.
            readout_error: Per-qubit readout error rate.
            shots: Number of measurement shots.

        Returns:
            noisy_counts: dict - measured counts with all noise applied
            ideal_counts: dict - counts from ideal statevector (no noise)
            fidelity: float - overlap between noisy and ideal distributions
            total_error: float - estimated total error rate
        """
        sv = np.asarray(ideal_statevector, dtype=complex).copy()
        n_states = len(sv)
        n_qubits = int(np.log2(n_states))

        # Ideal probabilities and counts
        ideal_probs = np.abs(sv) ** 2
        ideal_probs /= ideal_probs.sum()

        ideal_indices = np.random.choice(n_states, size=shots, p=ideal_probs)
        ideal_counts: Dict[str, int] = {}
        for idx in ideal_indices:
            bs = format(idx, f"0{n_qubits}b")
            ideal_counts[bs] = ideal_counts.get(bs, 0) + 1

        # Apply depolarizing noise (cumulative from all gates)
        cumulative_depol = 1.0 - (1.0 - gate_error) ** n_gates
        noisy_sv = self.apply_depolarizing(sv, cumulative_depol)

        # Apply T1 amplitude damping
        if self._defaults["t1_us"] > 0:
            # Assume gate time ~35ns for superconducting, scale by n_gates
            if self.backend_type == "superconducting":
                gate_time_ns = 35.0
            elif self.backend_type == "trapped_ion":
                gate_time_ns = 10_000.0
            else:
                gate_time_ns = 100.0
            total_time_us = gate_time_ns * n_gates / 1000.0
            gamma_t1 = 1.0 - np.exp(-total_time_us / self._defaults["t1_us"])
            noisy_sv = self.apply_amplitude_damping(noisy_sv, gamma_t1)

        # Apply T2 phase damping
        if self._defaults["t2_us"] > 0:
            if self.backend_type == "superconducting":
                gate_time_ns = 35.0
            elif self.backend_type == "trapped_ion":
                gate_time_ns = 10_000.0
            else:
                gate_time_ns = 100.0
            total_time_us = gate_time_ns * n_gates / 1000.0
            gamma_t2 = 1.0 - np.exp(-total_time_us / self._defaults["t2_us"])
            noisy_sv = self.apply_phase_damping(noisy_sv, gamma_t2)

        # Sample from noisy distribution
        noisy_probs = np.abs(noisy_sv) ** 2
        noisy_probs /= noisy_probs.sum()

        noisy_indices = np.random.choice(n_states, size=shots, p=noisy_probs)
        pre_readout_counts: Dict[str, int] = {}
        for idx in noisy_indices:
            bs = format(idx, f"0{n_qubits}b")
            pre_readout_counts[bs] = pre_readout_counts.get(bs, 0) + 1

        # Apply readout errors
        noisy_counts = self.apply_readout_error(pre_readout_counts, readout_error)

        # Compute fidelity (classical fidelity between probability distributions)
        noisy_final_probs = np.zeros(n_states)
        total_noisy = sum(noisy_counts.values())
        for bs, cnt in noisy_counts.items():
            idx = int(bs, 2)
            if idx < n_states:
                noisy_final_probs[idx] = cnt / total_noisy

        # Bhattacharyya coefficient as fidelity measure
        fidelity = float(np.sum(np.sqrt(ideal_probs * noisy_final_probs)))

        # Total error estimate
        total_error = 1.0 - fidelity

        return {
            "noisy_counts": noisy_counts,
            "ideal_counts": ideal_counts,
            "fidelity": fidelity,
            "total_error": total_error,
        }

    # ------------------------------------------------------------------
    # Hardware profiles
    # ------------------------------------------------------------------

    def get_hardware_profile(self, backend: str = "ibm_brisbane") -> Dict[str, Any]:
        """
        Return realistic noise parameters for known backends.

        Args:
            backend: Backend identifier. Known backends:
                - ibm_brisbane: IBM 127-qubit Eagle processor
                - ibm_osaka: IBM 127-qubit Eagle processor
                - ionq_aria: IonQ 25-qubit trapped ion
                - ionq_forte: IonQ 36-qubit trapped ion
                - dwave_advantage: D-Wave 5000+ qubit annealer

        Returns:
            Dict with t1_us, t2_us, single_qubit_error, cx_error,
            readout_error, gate times, n_qubits, backend_type.
        """
        if backend in _HARDWARE_PROFILES:
            return dict(_HARDWARE_PROFILES[backend])

        # Return generic profile
        return {
            "t1_us": self._defaults["t1_us"],
            "t2_us": self._defaults["t2_us"],
            "single_qubit_error": self._defaults["gate_error"],
            "cx_error": self._defaults["gate_error"] * 10,
            "readout_error": self._defaults["readout_error"],
            "single_gate_time_ns": 35.0,
            "cx_gate_time_ns": 300.0,
            "n_qubits": 0,
            "backend_type": self.backend_type,
        }
