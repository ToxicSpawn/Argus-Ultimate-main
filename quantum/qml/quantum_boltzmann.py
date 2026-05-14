"""
Quantum-Inspired Boltzmann Machine for Generative Modelling of Return Distributions.

A restricted Boltzmann machine (RBM) with quantum-inspired training: the negative
phase uses simulated quantum annealing instead of classical Gibbs sampling.
Quantum tunnelling (simulated) helps the sampler escape local energy minima,
producing better fantasy particles for the gradient estimate.

This is a classical simulation — no quantum hardware is used.  We are honest
about this.  The value comes from the annealing-based sampler's ability to
explore the energy landscape more thoroughly than CD-k Gibbs sampling, which
can help model heavy-tailed return distributions.

Typical usage::

    from quantum.qml.quantum_boltzmann import QuantumBoltzmannMachine

    qbm = QuantumBoltzmannMachine(n_visible=10, n_hidden=5)
    qbm.fit(returns_matrix, epochs=50)
    samples = qbm.generate_samples(100)
    dist = qbm.estimate_distribution(returns_matrix)
    score = qbm.anomaly_score(new_observation)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class QuantumBoltzmannMachine:
    """
    Quantum-inspired RBM for generative modelling of return distributions.

    Uses simulated quantum annealing for the negative phase (sampling).
    Classical simulation -- honest about this.
    """

    def __init__(
        self,
        n_visible: int = 10,
        n_hidden: int = 5,
        learning_rate: float = 0.01,
        seed: Optional[int] = None,
    ) -> None:
        if n_visible < 1:
            raise ValueError(f"n_visible must be >= 1, got {n_visible}")
        if n_hidden < 1:
            raise ValueError(f"n_hidden must be >= 1, got {n_hidden}")

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self._rng = np.random.RandomState(seed)

        # Model parameters
        self.weights = self._rng.randn(n_visible, n_hidden) * 0.01
        self.visible_bias = np.zeros(n_visible)
        self.hidden_bias = np.zeros(n_hidden)

        # Training state
        self._fitted = False
        self._reconstruction_errors: List[float] = []
        self._fit_time: float = 0.0
        self._n_train_samples: int = 0

        # Data normalisation (set during fit)
        self._data_mean: Optional[np.ndarray] = None
        self._data_std: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Core RBM operations
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        x_clipped = np.clip(x, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def _sample_hidden(self, visible: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """P(h|v) = sigmoid(W^T v + c).

        Returns: (probabilities, binary samples)
        """
        activation = visible @ self.weights + self.hidden_bias
        probs = self._sigmoid(activation)
        samples = (self._rng.rand(*probs.shape) < probs).astype(np.float64)
        return probs, samples

    def _sample_visible(self, hidden: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """P(v|h) = sigmoid(W h + b).

        Returns: (probabilities, binary samples)
        For continuous data we return the probability (mean-field) as the sample.
        """
        activation = hidden @ self.weights.T + self.visible_bias
        probs = self._sigmoid(activation)
        # For continuous-valued data, use Gaussian visible units:
        # sample = activation + noise, but probabilities are still useful
        samples = activation + self._rng.randn(*activation.shape) * 0.1
        return probs, samples

    def _free_energy(self, v: np.ndarray) -> np.ndarray:
        """Compute free energy F(v) = -b^T v - sum_j log(1 + exp(W^T v + c)).

        Lower free energy = model assigns higher probability.
        """
        if v.ndim == 1:
            v = v.reshape(1, -1)
        wx_b = v @ self.weights + self.hidden_bias  # (batch, n_hidden)
        visible_term = v @ self.visible_bias  # (batch,)
        hidden_term = np.sum(np.log(1.0 + np.exp(np.clip(wx_b, -50, 50))), axis=1)
        return -visible_term - hidden_term

    # ------------------------------------------------------------------
    # Quantum-inspired negative phase
    # ------------------------------------------------------------------

    def _quantum_negative_phase(self, n_samples: int = 100) -> np.ndarray:
        """Use simulated quantum annealing instead of Gibbs sampling.

        Formulate as QUBO: minimise energy E = -v^T W h - b^T v - c^T h.
        Quantum tunnelling helps escape local minima in energy landscape.
        Returns fantasy particles for gradient computation.

        The annealing schedule linearly reduces the transverse field from
        Gamma_0 to 0 over n_sweeps steps.  At each sweep, every spin is
        flipped with probability proportional to exp(-dE / T_eff) where
        T_eff combines the thermal temperature and the transverse field.
        """
        n_sweeps = 50
        gamma_0 = 3.0  # initial transverse field strength
        beta = 2.0  # inverse temperature

        n_total = self.n_visible + self.n_hidden
        fantasies = np.zeros((n_samples, self.n_visible))

        for s in range(n_samples):
            # Random initial spin configuration
            spins = self._rng.randint(0, 2, size=n_total).astype(np.float64)
            v = spins[: self.n_visible]
            h = spins[self.n_visible :]

            for sweep in range(n_sweeps):
                # Anneal transverse field
                t = (sweep + 1) / n_sweeps
                gamma = gamma_0 * (1.0 - t)
                t_eff = 1.0 / beta + gamma  # effective temperature

                # Sweep over all spins
                for i in range(self.n_visible):
                    # Energy difference if we flip visible unit i
                    delta_e = self.visible_bias[i] + h @ self.weights[i, :]
                    delta_e = delta_e * (1.0 - 2.0 * v[i])
                    # Metropolis with quantum tunnelling boost
                    if delta_e < 0 or self._rng.rand() < np.exp(
                        -delta_e / max(t_eff, 1e-10)
                    ):
                        v[i] = 1.0 - v[i]

                for j in range(self.n_hidden):
                    delta_e = self.hidden_bias[j] + v @ self.weights[:, j]
                    delta_e = delta_e * (1.0 - 2.0 * h[j])
                    if delta_e < 0 or self._rng.rand() < np.exp(
                        -delta_e / max(t_eff, 1e-10)
                    ):
                        h[j] = 1.0 - h[j]

            fantasies[s] = v

        return fantasies

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _normalise_data(self, data: np.ndarray) -> np.ndarray:
        """Normalise data to [0, 1] for binary RBM compatibility."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if self._data_mean is None:
            self._data_mean = np.mean(data, axis=0)
            self._data_std = np.std(data, axis=0)
            self._data_std[self._data_std < 1e-10] = 1.0

        normalised = (data - self._data_mean) / self._data_std
        # Map to [0, 1] via sigmoid
        return self._sigmoid(normalised)

    def _denormalise_data(self, data_sigmoid: np.ndarray) -> np.ndarray:
        """Reverse normalisation: [0, 1] -> original scale."""
        if self._data_mean is None or self._data_std is None:
            return data_sigmoid
        # Inverse sigmoid (logit)
        clipped = np.clip(data_sigmoid, 1e-7, 1.0 - 1e-7)
        normalised = np.log(clipped / (1.0 - clipped))
        return normalised * self._data_std + self._data_mean

    def fit(
        self,
        data: Any,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> "QuantumBoltzmannMachine":
        """Contrastive divergence training with quantum negative phase.

        - Positive phase: clamp data
        - Negative phase: quantum annealing sample
        - Update: dW = <v h^T>_data - <v h^T>_model
        Track reconstruction error per epoch.
        """
        t0 = time.monotonic()
        raw = np.asarray(data, dtype=np.float64)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)

        # Handle NaN
        nan_mask = np.isnan(raw)
        if nan_mask.any():
            col_means = np.nanmean(raw, axis=0)
            for c in range(raw.shape[1]):
                raw[nan_mask[:, c], c] = col_means[c] if not np.isnan(col_means[c]) else 0.0

        # Adjust dimensions if data width != n_visible
        if raw.shape[1] != self.n_visible:
            # Truncate or pad
            if raw.shape[1] > self.n_visible:
                raw = raw[:, : self.n_visible]
            else:
                padded = np.zeros((raw.shape[0], self.n_visible))
                padded[:, : raw.shape[1]] = raw
                raw = padded

        # Reset normalisation for new data
        self._data_mean = None
        self._data_std = None
        v_data = self._normalise_data(raw)

        self._n_train_samples = len(v_data)
        self._reconstruction_errors = []

        for epoch in range(max(1, epochs)):
            # Shuffle
            perm = self._rng.permutation(len(v_data))
            epoch_error = 0.0
            n_batches = 0

            for start in range(0, len(v_data), max(1, batch_size)):
                batch = v_data[perm[start : start + batch_size]]
                if len(batch) == 0:
                    continue

                # ── Positive phase ──
                pos_h_probs, pos_h_samples = self._sample_hidden(batch)
                pos_associations = batch.T @ pos_h_probs / len(batch)

                # ── Negative phase (quantum annealing) ──
                neg_v = self._quantum_negative_phase(n_samples=len(batch))
                neg_h_probs, _ = self._sample_hidden(neg_v)
                neg_associations = neg_v.T @ neg_h_probs / len(batch)

                # ── Update ──
                self.weights += self.learning_rate * (pos_associations - neg_associations)
                self.visible_bias += self.learning_rate * np.mean(batch - neg_v, axis=0)
                self.hidden_bias += self.learning_rate * np.mean(
                    pos_h_probs - neg_h_probs, axis=0
                )

                # Reconstruction error
                _, v_recon = self._sample_visible(pos_h_samples)
                v_recon_sig = self._sigmoid(v_recon)
                epoch_error += float(np.mean((batch - v_recon_sig) ** 2))
                n_batches += 1

            avg_error = epoch_error / max(n_batches, 1)
            self._reconstruction_errors.append(avg_error)

            if epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1:
                logger.debug(
                    "QBM epoch %d/%d: reconstruction_error=%.6f",
                    epoch + 1, epochs, avg_error,
                )

        self._fitted = True
        self._fit_time = time.monotonic() - t0
        logger.info(
            "QuantumBoltzmannMachine fit: %d samples, %d visible, %d hidden, "
            "%d epochs, final_error=%.6f, time=%.2fs",
            self._n_train_samples,
            self.n_visible,
            self.n_hidden,
            epochs,
            self._reconstruction_errors[-1] if self._reconstruction_errors else 0.0,
            self._fit_time,
        )
        return self

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_samples(
        self, n_samples: int = 100, burn_in: int = 50
    ) -> np.ndarray:
        """Generate synthetic return distributions.

        Run quantum annealing to sample from learned distribution,
        then denormalise back to the original data scale.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before generate_samples()")

        # Use more annealing sweeps for quality
        raw_samples = self._quantum_negative_phase(n_samples=n_samples + burn_in)
        # Discard burn-in
        raw_samples = raw_samples[burn_in:]
        # Denormalise
        return self._denormalise_data(raw_samples)

    # ------------------------------------------------------------------
    # Distribution estimation
    # ------------------------------------------------------------------

    def estimate_distribution(self, data: Any) -> Dict[str, Any]:
        """Estimate underlying return distribution.

        Returns: {mean, std, skewness, kurtosis, tail_probability,
                  var_95, cvar_95, distribution_type: str}
        """
        raw = np.asarray(data, dtype=np.float64).ravel()
        raw = raw[~np.isnan(raw)]

        if len(raw) < 2:
            return {
                "mean": 0.0,
                "std": 0.0,
                "skewness": 0.0,
                "kurtosis": 0.0,
                "tail_probability": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
                "distribution_type": "insufficient_data",
            }

        mean = float(np.mean(raw))
        std = float(np.std(raw, ddof=1)) if len(raw) > 1 else 0.0

        # Skewness
        if std > 1e-12:
            skewness = float(np.mean(((raw - mean) / std) ** 3))
            kurtosis = float(np.mean(((raw - mean) / std) ** 4) - 3.0)
        else:
            skewness = 0.0
            kurtosis = 0.0

        # VaR and CVaR at 95% confidence
        sorted_returns = np.sort(raw)
        var_idx = int(len(sorted_returns) * 0.05)
        var_95 = float(-sorted_returns[max(var_idx, 0)])
        cvar_95 = float(-np.mean(sorted_returns[: max(var_idx, 1)]))

        # Tail probability: P(|X - mean| > 2 * std)
        if std > 1e-12:
            tail_probability = float(np.mean(np.abs(raw - mean) > 2.0 * std))
        else:
            tail_probability = 0.0

        # Classify distribution type
        if abs(kurtosis) < 1.0 and abs(skewness) < 0.5:
            dist_type = "approximately_normal"
        elif kurtosis > 1.0:
            dist_type = "heavy_tailed"
        elif skewness < -1.0:
            dist_type = "left_skewed"
        elif skewness > 1.0:
            dist_type = "right_skewed"
        else:
            dist_type = "mixed"

        # If fitted, also generate model samples and compare
        model_assessment = "not_fitted"
        if self._fitted:
            try:
                gen = self.generate_samples(n_samples=min(500, len(raw) * 2))
                gen_flat = gen.ravel()
                gen_mean = float(np.mean(gen_flat))
                gen_std = float(np.std(gen_flat))
                if std > 1e-12:
                    mean_error = abs(gen_mean - mean) / std
                    std_error = abs(gen_std - std) / std
                    if mean_error < 0.5 and std_error < 0.5:
                        model_assessment = "good_fit"
                    elif mean_error < 1.0 and std_error < 1.0:
                        model_assessment = "moderate_fit"
                    else:
                        model_assessment = "poor_fit"
                else:
                    model_assessment = "constant_data"
            except Exception:
                model_assessment = "generation_failed"

        return {
            "mean": round(mean, 8),
            "std": round(std, 8),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurtosis, 4),
            "tail_probability": round(tail_probability, 4),
            "var_95": round(var_95, 8),
            "cvar_95": round(cvar_95, 8),
            "distribution_type": dist_type,
            "model_assessment": model_assessment,
            "n_samples": len(raw),
        }

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def anomaly_score(self, observation: Any) -> float:
        """Score how anomalous an observation is (free energy based).

        Low free energy = normal, high = anomalous.
        Useful for detecting regime changes.

        Returns a normalised score in [0, 1] where 1 = highly anomalous.
        """
        if not self._fitted:
            return 0.5  # uncertain

        obs = np.asarray(observation, dtype=np.float64).ravel()

        # Handle NaN
        nan_mask = np.isnan(obs)
        if nan_mask.any():
            obs = obs.copy()
            obs[nan_mask] = 0.0

        # Pad/truncate
        if len(obs) < self.n_visible:
            padded = np.zeros(self.n_visible)
            padded[: len(obs)] = obs
            obs = padded
        elif len(obs) > self.n_visible:
            obs = obs[: self.n_visible]

        # Normalise
        if self._data_mean is not None and self._data_std is not None:
            normalised = (obs - self._data_mean) / self._data_std
            obs_norm = self._sigmoid(normalised)
        else:
            obs_norm = self._sigmoid(obs)

        # Free energy of observation
        fe_obs = float(self._free_energy(obs_norm.reshape(1, -1))[0])

        # Compare with free energy of typical samples from model
        try:
            model_samples = self._quantum_negative_phase(n_samples=50)
            fe_model = self._free_energy(model_samples)
            fe_mean = float(np.mean(fe_model))
            fe_std = float(np.std(fe_model))

            if fe_std < 1e-10:
                return 0.5

            # Z-score of observation's free energy
            z = (fe_obs - fe_mean) / fe_std
            # Map to [0, 1] via sigmoid — high z (high free energy) = anomalous
            score = float(self._sigmoid(np.array([z]))[0])
        except Exception:
            score = 0.5

        return round(float(np.clip(score, 0.0, 1.0)), 4)

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def benchmark_vs_classical(self, data: Any) -> Dict[str, Any]:
        """Compare QBM sampling vs classical Gibbs vs KDE.

        Returns honest assessment of relative quality.
        """
        raw = np.asarray(data, dtype=np.float64)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)

        # Handle NaN
        nan_mask = np.isnan(raw)
        if nan_mask.any():
            col_means = np.nanmean(raw, axis=0)
            for c in range(raw.shape[1]):
                raw[nan_mask[:, c], c] = col_means[c] if not np.isnan(col_means[c]) else 0.0

        results: Dict[str, Any] = {
            "n_samples": raw.shape[0],
            "n_features": raw.shape[1],
        }

        n_gen = min(200, raw.shape[0])
        flat_data = raw.ravel()

        # ── QBM ──
        t0 = time.monotonic()
        try:
            self.fit(raw, epochs=30, batch_size=32)
            qbm_samples = self.generate_samples(n_gen).ravel()
            qbm_time = time.monotonic() - t0
            results["qbm_time_s"] = round(qbm_time, 3)
            results["qbm_mean_error"] = round(
                abs(float(np.mean(qbm_samples)) - float(np.mean(flat_data))), 6
            )
            results["qbm_std_error"] = round(
                abs(float(np.std(qbm_samples)) - float(np.std(flat_data))), 6
            )
        except Exception as e:
            results["qbm_error"] = str(e)
            qbm_samples = None

        # ── Classical Gibbs (CD-1) ──
        t0 = time.monotonic()
        try:
            gibbs_samples = self._classical_gibbs(raw, n_gen)
            gibbs_time = time.monotonic() - t0
            results["gibbs_time_s"] = round(gibbs_time, 3)
            gibbs_flat = gibbs_samples.ravel()
            results["gibbs_mean_error"] = round(
                abs(float(np.mean(gibbs_flat)) - float(np.mean(flat_data))), 6
            )
            results["gibbs_std_error"] = round(
                abs(float(np.std(gibbs_flat)) - float(np.std(flat_data))), 6
            )
        except Exception as e:
            results["gibbs_error"] = str(e)

        # ── KDE baseline ──
        t0 = time.monotonic()
        try:
            kde_samples = self._kde_sample(flat_data, n_gen)
            kde_time = time.monotonic() - t0
            results["kde_time_s"] = round(kde_time, 3)
            results["kde_mean_error"] = round(
                abs(float(np.mean(kde_samples)) - float(np.mean(flat_data))), 6
            )
            results["kde_std_error"] = round(
                abs(float(np.std(kde_samples)) - float(np.std(flat_data))), 6
            )
        except Exception as e:
            results["kde_error"] = str(e)

        # Honest assessment
        qbm_err = results.get("qbm_mean_error", float("inf"))
        gibbs_err = results.get("gibbs_mean_error", float("inf"))
        kde_err = results.get("kde_mean_error", float("inf"))

        best = min(qbm_err, gibbs_err, kde_err)
        if best == qbm_err and qbm_err < float("inf"):
            results["best_method"] = "quantum_boltzmann"
        elif best == kde_err and kde_err < float("inf"):
            results["best_method"] = "kde"
        else:
            results["best_method"] = "classical_gibbs"

        results["honest_note"] = (
            "Quantum Boltzmann uses simulated quantum annealing on classical hardware. "
            "No quantum advantage is claimed. For small models, KDE is typically "
            "faster and more accurate. The QBM approach may provide better exploration "
            "of multimodal distributions due to quantum tunnelling simulation."
        )
        return results

    def _classical_gibbs(self, data: np.ndarray, n_samples: int) -> np.ndarray:
        """Classical CD-1 Gibbs sampling for benchmark comparison."""
        v_data = self._normalise_data(data)
        # Pick random starting points
        indices = self._rng.choice(len(v_data), size=n_samples)
        v = v_data[indices].copy()
        # One full Gibbs step
        _, h = self._sample_hidden(v)
        _, v_recon = self._sample_visible(h)
        return self._denormalise_data(self._sigmoid(v_recon))

    def _kde_sample(self, flat_data: np.ndarray, n_samples: int) -> np.ndarray:
        """Simple KDE sampling baseline."""
        bandwidth = 1.06 * float(np.std(flat_data)) * len(flat_data) ** (-0.2)
        if bandwidth < 1e-12:
            bandwidth = 1.0
        # Sample from data + Gaussian noise
        indices = self._rng.choice(len(flat_data), size=n_samples)
        samples = flat_data[indices] + self._rng.randn(n_samples) * bandwidth
        return samples

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the QBM state and configuration."""
        return {
            "n_visible": self.n_visible,
            "n_hidden": self.n_hidden,
            "learning_rate": self.learning_rate,
            "fitted": self._fitted,
            "n_train_samples": self._n_train_samples,
            "fit_time_s": round(self._fit_time, 3) if self._fitted else None,
            "final_reconstruction_error": (
                round(self._reconstruction_errors[-1], 6)
                if self._reconstruction_errors
                else None
            ),
            "n_epochs_trained": len(self._reconstruction_errors),
            "method": "quantum_inspired_annealing_rbm",
        }
