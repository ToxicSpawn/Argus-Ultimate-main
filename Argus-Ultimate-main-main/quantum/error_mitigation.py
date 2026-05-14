"""
Quantum Error Mitigation for NISQ devices.

Implements real error mitigation techniques that make noisy quantum results
usable for trading decisions:

- Zero-Noise Extrapolation (ZNE): Richardson and exponential extrapolation
- Measurement Error Mitigation: inverse calibration matrix
- Probabilistic Error Cancellation (PEC): quasi-probability decomposition
- Twirled Readout Error mitigation: Pauli twirling for cheap readout correction
- Circuit Fidelity Estimation: predict whether hardware results are useful

All techniques work with raw count dictionaries from any quantum backend.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class QuantumErrorMitigator:
    """
    NISQ error mitigation toolkit.

    Provides multiple mitigation strategies that can be composed:
      1. Build calibration data (once per backend session)
      2. Run circuits at multiple noise levels
      3. Apply ZNE / MEM / PEC / twirling to extract cleaner results
      4. Estimate whether the circuit fidelity justifies hardware use
    """

    def __init__(self) -> None:
        self._calibration_data: Dict[str, Any] = {}
        self._noise_profiles: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Zero-Noise Extrapolation (ZNE)
    # ------------------------------------------------------------------

    def zero_noise_extrapolation(
        self,
        results_at_noise_levels: List[Tuple[float, float]],
    ) -> Dict[str, Any]:
        """
        ZNE: Run circuit at multiple noise levels, extrapolate to zero noise.

        Takes list of (noise_factor, expectation_value) tuples.
        noise_factor=1.0 is the native hardware noise; 2.0, 3.0 etc. are
        amplified via gate folding or pulse stretching.

        Fits both Richardson (polynomial) and exponential extrapolation,
        returns the one with better fit quality plus confidence intervals.

        Args:
            results_at_noise_levels: List of (noise_factor, measured_value).
                Must have at least 2 points. noise_factor >= 1.0.

        Returns:
            zne_richardson: float - Richardson extrapolation to zero noise
            zne_exponential: float - Exponential extrapolation to zero noise
            confidence_interval: (low, high) - 95% CI from fit residuals
            extrapolation_quality: float in [0, 1] - R^2 of the better fit
            method_used: str - which extrapolation was preferred
            fit_residuals: list - residuals of the preferred fit
        """
        if len(results_at_noise_levels) < 2:
            raise ValueError("ZNE requires at least 2 noise levels")

        # Sort by noise factor
        data = sorted(results_at_noise_levels, key=lambda x: x[0])
        noise_factors = np.array([d[0] for d in data])
        values = np.array([d[1] for d in data])

        # --- Richardson extrapolation (polynomial) ---
        # Fit polynomial of degree (n_points - 1)
        degree = min(len(data) - 1, 4)  # cap at degree 4
        poly_coeffs = np.polyfit(noise_factors, values, degree)
        poly = np.poly1d(poly_coeffs)
        zne_richardson = float(poly(0.0))

        # Residuals for Richardson
        richardson_fitted = poly(noise_factors)
        richardson_residuals = values - richardson_fitted
        richardson_ss_res = float(np.sum(richardson_residuals ** 2))
        ss_tot = float(np.sum((values - np.mean(values)) ** 2))
        richardson_r2 = 1.0 - richardson_ss_res / max(ss_tot, 1e-15)

        # --- Exponential extrapolation ---
        # Model: f(lambda) = a * exp(b * lambda) + c
        # For stability, fit log(values - min) when possible
        zne_exponential = zne_richardson  # default if exp fit fails
        exp_r2 = -1.0
        exp_residuals = richardson_residuals

        try:
            if len(data) >= 3:
                # Fit: f(x) = a * exp(b * x) + c via least-squares
                from scipy.optimize import curve_fit

                def exp_model(x, a, b, c):
                    return a * np.exp(b * x) + c

                # Initial guess
                a0 = float(values[-1] - values[0])
                b0 = 0.1
                c0 = float(values[0])

                try:
                    popt, _ = curve_fit(
                        exp_model, noise_factors, values,
                        p0=[a0, b0, c0],
                        maxfev=2000,
                    )
                    zne_exponential = float(exp_model(0.0, *popt))
                    exp_fitted = exp_model(noise_factors, *popt)
                    exp_residuals = values - exp_fitted
                    exp_ss_res = float(np.sum(exp_residuals ** 2))
                    exp_r2 = 1.0 - exp_ss_res / max(ss_tot, 1e-15)
                except Exception:
                    # curve_fit can fail; keep Richardson result
                    pass
            else:
                # With only 2 points, do simple linear extrapolation for exp
                slope = (values[1] - values[0]) / max(noise_factors[1] - noise_factors[0], 1e-15)
                zne_exponential = float(values[0] - slope * noise_factors[0])
        except ImportError:
            # No scipy, use linear extrapolation
            slope = (values[-1] - values[0]) / max(noise_factors[-1] - noise_factors[0], 1e-15)
            zne_exponential = float(values[0] - slope * noise_factors[0])

        # Select better method
        if exp_r2 > richardson_r2:
            method_used = "exponential"
            quality = max(0.0, min(1.0, exp_r2))
            residuals = exp_residuals
        else:
            method_used = "richardson"
            quality = max(0.0, min(1.0, richardson_r2))
            residuals = richardson_residuals

        # Confidence interval from residual std
        if len(residuals) > 1:
            std_err = float(np.std(residuals, ddof=1))
        else:
            std_err = abs(zne_richardson - zne_exponential)

        best_val = zne_exponential if method_used == "exponential" else zne_richardson
        ci_low = best_val - 1.96 * std_err
        ci_high = best_val + 1.96 * std_err

        return {
            "zne_richardson": zne_richardson,
            "zne_exponential": zne_exponential,
            "confidence_interval": (ci_low, ci_high),
            "extrapolation_quality": quality,
            "method_used": method_used,
            "fit_residuals": residuals.tolist(),
        }

    # ------------------------------------------------------------------
    # Measurement Error Mitigation
    # ------------------------------------------------------------------

    def build_calibration_matrix(
        self,
        n_qubits: int,
        error_rate: float = 0.01,
    ) -> np.ndarray:
        """
        Build measurement error calibration matrix.

        For simulation: constructs a synthetic error model where each qubit
        has independent readout error with probability `error_rate`.

        calibration_matrix[i][j] = P(measure bitstring i | prepared bitstring j)

        For real hardware, you would run 2^n calibration circuits (prepare each
        computational basis state and measure). This synthetic version is used
        for testing and when hardware calibration data is unavailable.

        Args:
            n_qubits: Number of qubits.
            error_rate: Per-qubit readout error probability.

        Returns:
            (2^n x 2^n) calibration matrix.
        """
        n_states = 2 ** n_qubits
        cal_matrix = np.zeros((n_states, n_states), dtype=float)

        for j in range(n_states):
            # Prepared state j
            prepared_bits = [(j >> q) & 1 for q in range(n_qubits)]

            for i in range(n_states):
                # Measured state i
                measured_bits = [(i >> q) & 1 for q in range(n_qubits)]

                # Probability = product of per-qubit probabilities
                prob = 1.0
                for q in range(n_qubits):
                    if measured_bits[q] == prepared_bits[q]:
                        prob *= (1.0 - error_rate)
                    else:
                        prob *= error_rate
                cal_matrix[i, j] = prob

        return cal_matrix

    def measurement_error_mitigation(
        self,
        raw_counts: Dict[str, int],
        calibration_matrix: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Apply inverse calibration matrix to raw measurement results.

        Solves the linear system: raw_probs = cal_matrix @ ideal_probs
        via least-squares (handles non-invertible and ill-conditioned cases).
        Clips negative probabilities to 0 and renormalizes.

        Args:
            raw_counts: Dict of bitstring -> count from measurement.
            calibration_matrix: (2^n x 2^n) calibration matrix from
                build_calibration_matrix() or hardware calibration.

        Returns:
            mitigated_counts: dict of bitstring -> mitigated count (float)
            fidelity_improvement: float - ratio of mitigated vs raw entropy
            raw_total: int - total raw shots
        """
        if not raw_counts:
            return {"mitigated_counts": {}, "fidelity_improvement": 0.0, "raw_total": 0}

        n_states = calibration_matrix.shape[0]
        n_qubits = int(np.log2(n_states))
        total_shots = sum(raw_counts.values())

        # Convert counts to probability vector
        raw_probs = np.zeros(n_states, dtype=float)
        for bitstring, count in raw_counts.items():
            bits = bitstring.replace(" ", "")
            # Ensure consistent length
            bits = bits.zfill(n_qubits)[-n_qubits:]
            idx = int(bits, 2)
            if idx < n_states:
                raw_probs[idx] = count / total_shots

        # Solve via least-squares: cal_matrix @ ideal_probs = raw_probs
        # This handles singular and ill-conditioned matrices
        ideal_probs, _, _, _ = np.linalg.lstsq(calibration_matrix, raw_probs, rcond=None)

        # Clip negative probabilities and renormalize
        ideal_probs = np.maximum(ideal_probs, 0.0)
        prob_sum = ideal_probs.sum()
        if prob_sum > 0:
            ideal_probs /= prob_sum

        # Convert back to counts dict
        mitigated_counts: Dict[str, float] = {}
        for idx in range(n_states):
            if ideal_probs[idx] > 1e-10:
                bitstring = format(idx, f"0{n_qubits}b")
                mitigated_counts[bitstring] = float(ideal_probs[idx] * total_shots)

        # Compute fidelity improvement via entropy comparison
        raw_nonzero = raw_probs[raw_probs > 0]
        ideal_nonzero = ideal_probs[ideal_probs > 0]

        raw_entropy = float(-np.sum(raw_nonzero * np.log2(raw_nonzero))) if len(raw_nonzero) > 0 else 0.0
        ideal_entropy = float(-np.sum(ideal_nonzero * np.log2(ideal_nonzero))) if len(ideal_nonzero) > 0 else 0.0

        # Lower entropy after mitigation means more concentrated = better
        max_entropy = np.log2(n_states)
        if max_entropy > 0 and raw_entropy > 0:
            fidelity_improvement = (raw_entropy - ideal_entropy) / raw_entropy
        else:
            fidelity_improvement = 0.0

        return {
            "mitigated_counts": mitigated_counts,
            "fidelity_improvement": fidelity_improvement,
            "raw_total": total_shots,
        }

    # ------------------------------------------------------------------
    # Probabilistic Error Cancellation (PEC)
    # ------------------------------------------------------------------

    def probabilistic_error_cancellation(
        self,
        ideal_result: float,
        noise_results: List[float],
    ) -> Dict[str, Any]:
        """
        PEC: Decompose noisy channel, sample correction terms.

        Estimates the corrected expectation value by using quasi-probability
        decomposition. The idea: express the ideal operation as a linear
        combination of noisy operations with quasi-probability coefficients.

        For a simple depolarizing channel with error rate p:
          ideal = (1/(1-p)) * noisy - (p/(1-p)) * maximally_mixed

        The overhead factor gamma = 1/(1-p) determines the variance cost.

        Args:
            ideal_result: The noiseless expected value (from simulator).
            noise_results: List of noisy measurements at different
                error rates or random circuit instances.

        Returns:
            corrected_value: float - PEC-corrected estimate
            overhead_factor: float - sampling overhead (gamma)
            variance: float - estimated variance of the corrected estimator
            n_samples: int - number of noise samples used
        """
        if not noise_results:
            return {
                "corrected_value": ideal_result,
                "overhead_factor": 1.0,
                "variance": 0.0,
                "n_samples": 0,
            }

        noisy_mean = float(np.mean(noise_results))
        noisy_var = float(np.var(noise_results)) if len(noise_results) > 1 else 0.0

        # Estimate the error rate from deviation
        if abs(ideal_result) > 1e-15:
            estimated_error = 1.0 - noisy_mean / ideal_result
        else:
            estimated_error = abs(noisy_mean)

        estimated_error = max(0.0, min(estimated_error, 0.99))

        # Quasi-probability overhead
        gamma = 1.0 / max(1.0 - estimated_error, 0.01)

        # PEC correction: corrected = gamma * noisy_mean - (gamma - 1) * mixed_state
        # For expectation values, mixed state contribution is 0.0
        corrected_value = gamma * noisy_mean

        # Variance of the PEC estimator scales with gamma^2
        pec_variance = (gamma ** 2) * noisy_var / max(len(noise_results), 1)

        return {
            "corrected_value": corrected_value,
            "overhead_factor": gamma,
            "variance": pec_variance,
            "n_samples": len(noise_results),
        }

    # ------------------------------------------------------------------
    # Twirled Readout Error Mitigation
    # ------------------------------------------------------------------

    def twirled_readout_error(
        self,
        counts: Dict[str, int],
        n_qubits: int,
        readout_error: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Simplified readout error mitigation via Pauli twirling.

        Symmetrizes readout errors so that P(0|1) = P(1|0), then
        corrects via simple rescaling: p_corrected = (p_raw - e/2) / (1 - e)
        where e is the symmetrized readout error rate.

        Much cheaper than full calibration matrix inversion: O(n) vs O(4^n).

        Args:
            counts: Dict of bitstring -> count.
            n_qubits: Number of qubits.
            readout_error: Per-qubit readout error rate (assumed symmetric
                after twirling).

        Returns:
            mitigated_counts: dict - corrected counts
            correction_factor: float - the rescaling factor applied
            effective_error: float - the symmetrized error rate used
        """
        if not counts:
            return {
                "mitigated_counts": {},
                "correction_factor": 1.0,
                "effective_error": readout_error,
            }

        total = sum(counts.values())
        if total <= 0:
            return {
                "mitigated_counts": {},
                "correction_factor": 1.0,
                "effective_error": readout_error,
            }

        # Symmetrized correction factor per qubit
        # After Pauli twirling, readout error becomes symmetric:
        # P(correct) = 1 - e, P(flip) = e
        # For n qubits, the global correction factor on probabilities:
        # p_ideal(x) = sum_y M_inv(x,y) * p_raw(y)
        # For symmetric errors, this simplifies to rescaling
        correction = 1.0 / max(1.0 - 2.0 * readout_error, 0.01)

        mitigated_counts: Dict[str, float] = {}
        uniform_prob = 1.0 / (2 ** n_qubits)

        for bitstring, count in counts.items():
            raw_prob = count / total
            # Correct: p_ideal = (p_raw - e * p_uniform) / (1 - e)
            # For multi-qubit: accumulated error per qubit
            effective_e = 1.0 - (1.0 - readout_error) ** n_qubits
            corrected_prob = (raw_prob - effective_e * uniform_prob) / max(1.0 - effective_e, 0.01)
            corrected_prob = max(corrected_prob, 0.0)
            mitigated_counts[bitstring] = corrected_prob * total

        # Renormalize
        mit_total = sum(mitigated_counts.values())
        if mit_total > 0:
            scale = total / mit_total
            mitigated_counts = {k: v * scale for k, v in mitigated_counts.items()}

        effective_e = 1.0 - (1.0 - readout_error) ** n_qubits

        return {
            "mitigated_counts": mitigated_counts,
            "correction_factor": correction,
            "effective_error": effective_e,
        }

    # ------------------------------------------------------------------
    # Circuit Fidelity Estimation
    # ------------------------------------------------------------------

    def estimate_circuit_fidelity(
        self,
        n_gates: int,
        n_qubits: int,
        gate_error: float = 0.001,
        readout_error: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Estimate expected fidelity of a quantum circuit on NISQ hardware.

        Uses the simple product model:
            F = (1 - gate_error)^n_gates * (1 - readout_error)^n_qubits

        This is a lower bound on actual fidelity (correlated errors can
        help or hurt). Used to decide whether to use hardware or simulator.

        Args:
            n_gates: Total number of gates in the circuit.
            n_qubits: Number of qubits.
            gate_error: Average gate error rate (2-qubit gates typically
                5-10x higher than 1-qubit).
            readout_error: Per-qubit readout error rate.

        Returns:
            expected_fidelity: float in [0, 1]
            useful_shots_pct: float - approximate % of useful measurements
            recommended_shots: int - shots needed for statistical significance
            recommendation: 'use_hardware' | 'use_simulator' | 'too_noisy'
        """
        # Gate fidelity: product of individual gate fidelities
        gate_fidelity = (1.0 - gate_error) ** n_gates

        # Readout fidelity: product of per-qubit readout fidelities
        readout_fidelity = (1.0 - readout_error) ** n_qubits

        # Total circuit fidelity
        total_fidelity = gate_fidelity * readout_fidelity

        # Useful shots: fraction of measurements that reflect the ideal distribution
        useful_shots_pct = total_fidelity * 100.0

        # Recommended shots: need enough that useful shots give statistical power
        # Want at least ~100 useful shots for meaningful statistics
        min_useful = 100
        if total_fidelity > 0:
            recommended_shots = max(1000, int(np.ceil(min_useful / total_fidelity)))
        else:
            recommended_shots = 100000

        # Decision thresholds
        if total_fidelity >= 0.5:
            recommendation = "use_hardware"
        elif total_fidelity >= 0.1:
            recommendation = "use_simulator"
        else:
            recommendation = "too_noisy"

        return {
            "expected_fidelity": total_fidelity,
            "gate_fidelity": gate_fidelity,
            "readout_fidelity": readout_fidelity,
            "useful_shots_pct": useful_shots_pct,
            "recommended_shots": recommended_shots,
            "recommendation": recommendation,
        }
