"""
Quantum-Inspired GAN for Synthetic Market Data Generation.

Generates realistic synthetic market data (returns, volatility, volume, spread,
momentum) for backtesting and data augmentation.  The generator uses a
parameterised quantum circuit (classically simulated) to map latent noise to
market features.  The discriminator is a simple feedforward network.

No quantum hardware is used -- this is an honest classical simulation of
quantum-circuit-based generative modelling.  The quantum generator naturally
produces complex correlations between output features due to entanglement
in the circuit, which can help capture cross-feature dependencies in market data.

Typical usage::

    from quantum.qml.quantum_gan import QuantumGAN

    qgan = QuantumGAN(n_features=5, latent_dim=3, n_qubits=4)
    qgan.train(real_data, epochs=100)
    synthetic = qgan.generate(n_samples=200)
    quality = qgan.evaluate_quality(real_data, synthetic)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class QuantumGAN:
    """
    Quantum-inspired GAN for synthetic market data generation.

    Generator uses parameterised quantum circuit (simulated).
    Discriminator is classical neural network (single hidden layer).
    """

    def __init__(
        self,
        n_features: int = 5,
        latent_dim: int = 3,
        n_qubits: int = 4,
        learning_rate: float = 0.01,
        seed: Optional[int] = None,
    ) -> None:
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if n_qubits < 1 or n_qubits > 12:
            raise ValueError(f"n_qubits must be in [1, 12], got {n_qubits}")

        self.n_features = n_features
        self.latent_dim = latent_dim
        self.n_qubits = n_qubits
        self.learning_rate = learning_rate
        self._rng = np.random.RandomState(seed)

        # Generator parameters: rotation angles for quantum circuit
        self._n_gen_layers = 2
        # Params: encoding rotations + variational rotations
        n_gen_params = self._n_gen_layers * n_qubits * 3 + latent_dim * n_qubits
        self._gen_params = self._rng.randn(n_gen_params) * 0.1

        # Discriminator: simple 1-hidden-layer MLP
        self._disc_hidden = max(16, n_features * 4)
        self._disc_W1 = self._rng.randn(n_features, self._disc_hidden) * 0.1
        self._disc_b1 = np.zeros(self._disc_hidden)
        self._disc_W2 = self._rng.randn(self._disc_hidden, 1) * 0.1
        self._disc_b2 = np.zeros(1)

        # Data normalisation
        self._data_mean: Optional[np.ndarray] = None
        self._data_std: Optional[np.ndarray] = None

        # Training history
        self._trained = False
        self._gen_losses: List[float] = []
        self._disc_losses: List[float] = []
        self._train_time: float = 0.0

    # ------------------------------------------------------------------
    # Quantum generator
    # ------------------------------------------------------------------

    def _quantum_generator(self, noise: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Parameterised quantum circuit as generator.

        Encode noise as rotations, apply variational layers (Ry + CNOT),
        measure expectations -> output features.

        Args:
            noise: (latent_dim,) latent vector
            params: flat array of circuit parameters

        Returns: (n_features,) generated sample
        """
        n = self.n_qubits
        dim = 2 ** n

        # Initialise |0...0>
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0 + 0j

        idx = 0

        # Encoding layer: Ry(noise_j * param_ij) on each qubit for each latent dim
        for j in range(self.latent_dim):
            for q in range(n):
                angle = float(noise[j]) * float(params[idx])
                state = self._apply_ry(state, n, q, angle)
                idx += 1

        # Variational layers
        for layer in range(self._n_gen_layers):
            # Single qubit rotations: Rx, Ry, Rz
            for q in range(n):
                state = self._apply_rx(state, n, q, float(params[idx]))
                idx += 1
                state = self._apply_ry(state, n, q, float(params[idx]))
                idx += 1
                state = self._apply_rz(state, n, q, float(params[idx]))
                idx += 1

            # Entangling CNOT chain
            for q in range(n - 1):
                state = self._apply_cnot(state, n, q, q + 1)
            if n > 1:
                state = self._apply_cnot(state, n, n - 1, 0)  # ring

        # Measure <Z_i> expectations for first n_features qubits
        probs = np.abs(state) ** 2
        expectations = np.zeros(min(n, self.n_features))
        for q in range(len(expectations)):
            for i in range(dim):
                bit = (i >> q) & 1
                expectations[q] += (1 - 2 * bit) * probs[i]

        # If n_features > n_qubits, fill remaining with combinations
        if self.n_features > n:
            output = np.zeros(self.n_features)
            output[:n] = expectations
            for f in range(n, self.n_features):
                # Linear combination of existing expectations
                output[f] = np.mean(expectations) * (1 + 0.1 * self._rng.randn())
        else:
            output = expectations[: self.n_features]

        return output

    def _classical_discriminator(self, sample: np.ndarray) -> float:
        """Simple feedforward discriminator.

        Returns: probability of sample being real data (0 to 1).
        """
        h = np.tanh(sample @ self._disc_W1 + self._disc_b1)
        logit = float((h @ self._disc_W2 + self._disc_b2)[0])
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50))))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        real_data: Any,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> "QuantumGAN":
        """Adversarial training.

        1. Train discriminator on real + generated
        2. Train generator to fool discriminator
        Track: gen_loss, disc_loss
        """
        t0 = time.monotonic()
        data = np.asarray(real_data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Handle NaN
        nan_mask = np.isnan(data)
        if nan_mask.any():
            col_means = np.nanmean(data, axis=0)
            for c in range(data.shape[1]):
                data[nan_mask[:, c], c] = col_means[c] if not np.isnan(col_means[c]) else 0.0

        # Adjust features
        if data.shape[1] != self.n_features:
            if data.shape[1] > self.n_features:
                data = data[:, : self.n_features]
            else:
                padded = np.zeros((data.shape[0], self.n_features))
                padded[:, : data.shape[1]] = data
                data = padded

        # Normalise
        self._data_mean = np.mean(data, axis=0)
        self._data_std = np.std(data, axis=0)
        self._data_std[self._data_std < 1e-10] = 1.0
        data_norm = (data - self._data_mean) / self._data_std

        self._gen_losses = []
        self._disc_losses = []

        disc_lr = self.learning_rate * 2.0
        gen_lr = self.learning_rate
        eps = 1e-8

        for epoch in range(max(1, epochs)):
            perm = self._rng.permutation(len(data_norm))
            epoch_disc_loss = 0.0
            epoch_gen_loss = 0.0
            n_batches = 0

            for start in range(0, len(data_norm), max(1, batch_size)):
                batch = data_norm[perm[start : start + batch_size]]
                if len(batch) == 0:
                    continue

                # ── Train discriminator ──
                d_loss = 0.0
                for sample in batch:
                    d_real = self._classical_discriminator(sample)
                    d_loss -= np.log(d_real + eps)

                # Generate fake samples
                fake_batch = []
                for _ in range(len(batch)):
                    noise = self._rng.randn(self.latent_dim)
                    fake = self._quantum_generator(noise, self._gen_params)
                    fake_batch.append(fake)
                    d_fake = self._classical_discriminator(fake)
                    d_loss -= np.log(1.0 - d_fake + eps)

                d_loss /= (2 * len(batch))

                # Discriminator gradient (numerical)
                self._update_discriminator(batch, np.array(fake_batch), disc_lr)

                # ── Train generator ──
                g_loss = 0.0
                param_grad = np.zeros_like(self._gen_params)
                shift = 0.01

                for _ in range(min(len(batch), 5)):  # sub-sample for speed
                    noise = self._rng.randn(self.latent_dim)
                    fake = self._quantum_generator(noise, self._gen_params)
                    d_fake = self._classical_discriminator(fake)
                    g_loss -= np.log(d_fake + eps)

                    # Parameter shift rule for gradient
                    for p in range(len(self._gen_params)):
                        params_plus = self._gen_params.copy()
                        params_plus[p] += shift
                        fake_plus = self._quantum_generator(noise, params_plus)
                        d_plus = self._classical_discriminator(fake_plus)

                        params_minus = self._gen_params.copy()
                        params_minus[p] -= shift
                        fake_minus = self._quantum_generator(noise, params_minus)
                        d_minus = self._classical_discriminator(fake_minus)

                        # Gradient of -log(D(G(z))) w.r.t. param
                        if d_plus > eps and d_minus > eps:
                            param_grad[p] += (-np.log(d_plus) + np.log(d_minus)) / (
                                2 * shift
                            )

                g_loss /= min(len(batch), 5)
                param_grad /= min(len(batch), 5)

                # Update generator
                self._gen_params -= gen_lr * param_grad

                epoch_disc_loss += d_loss
                epoch_gen_loss += g_loss
                n_batches += 1

            if n_batches > 0:
                self._disc_losses.append(epoch_disc_loss / n_batches)
                self._gen_losses.append(epoch_gen_loss / n_batches)

            if epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1:
                logger.debug(
                    "QGAN epoch %d/%d: disc_loss=%.4f gen_loss=%.4f",
                    epoch + 1,
                    epochs,
                    self._disc_losses[-1] if self._disc_losses else 0,
                    self._gen_losses[-1] if self._gen_losses else 0,
                )

        self._trained = True
        self._train_time = time.monotonic() - t0
        logger.info(
            "QuantumGAN trained: %d samples, %d features, %d epochs, time=%.2fs",
            len(data_norm),
            self.n_features,
            epochs,
            self._train_time,
        )
        return self

    def _update_discriminator(
        self,
        real_batch: np.ndarray,
        fake_batch: np.ndarray,
        lr: float,
    ) -> None:
        """Update discriminator weights via simple gradient step."""
        # Numerical gradient for small discriminator
        eps = 1e-5

        def disc_loss(W1, b1, W2, b2):
            loss = 0.0
            for s in real_batch:
                h = np.tanh(s @ W1 + b1)
                logit = float((h @ W2 + b2)[0])
                p = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
                loss -= np.log(p + 1e-8)
            for s in fake_batch:
                h = np.tanh(s @ W1 + b1)
                logit = float((h @ W2 + b2)[0])
                p = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
                loss -= np.log(1.0 - p + 1e-8)
            return loss / (len(real_batch) + len(fake_batch))

        # Gradient for W2 and b2 (cheaper — small)
        base_loss = disc_loss(
            self._disc_W1, self._disc_b1, self._disc_W2, self._disc_b2
        )

        # Analytical gradient for output layer
        for i in range(self._disc_hidden):
            self._disc_W2[i, 0] += eps
            l_plus = disc_loss(
                self._disc_W1, self._disc_b1, self._disc_W2, self._disc_b2
            )
            self._disc_W2[i, 0] -= 2 * eps
            l_minus = disc_loss(
                self._disc_W1, self._disc_b1, self._disc_W2, self._disc_b2
            )
            self._disc_W2[i, 0] += eps  # restore
            grad = (l_plus - l_minus) / (2 * eps)
            self._disc_W2[i, 0] -= lr * grad

        self._disc_b2[0] += eps
        l_plus = disc_loss(
            self._disc_W1, self._disc_b1, self._disc_W2, self._disc_b2
        )
        self._disc_b2[0] -= 2 * eps
        l_minus = disc_loss(
            self._disc_W1, self._disc_b1, self._disc_W2, self._disc_b2
        )
        self._disc_b2[0] += eps
        self._disc_b2[0] -= lr * (l_plus - l_minus) / (2 * eps)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, n_samples: int = 100) -> np.ndarray:
        """Generate synthetic market data samples.

        Output shape: (n_samples, n_features)
        Features: returns, volatility, volume, spread, momentum
        (or whatever the training data represented).
        """
        if not self._trained:
            raise RuntimeError("Must call train() before generate()")

        samples = np.zeros((n_samples, self.n_features))
        for i in range(n_samples):
            noise = self._rng.randn(self.latent_dim)
            raw = self._quantum_generator(noise, self._gen_params)
            samples[i] = raw

        # Denormalise
        if self._data_mean is not None and self._data_std is not None:
            samples = samples * self._data_std + self._data_mean

        return samples

    def augment_training_data(
        self, real_data: Any, augmentation_factor: float = 2.0
    ) -> np.ndarray:
        """Augment real data with synthetic samples for ML training.

        Useful for rare regime data augmentation (crisis scenarios).
        Combines real data with generated synthetic data.
        """
        data = np.asarray(real_data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if not self._trained:
            # Train on the real data first
            self.train(data, epochs=50)

        n_synthetic = int(len(data) * augmentation_factor)
        synthetic = self.generate(n_samples=n_synthetic)

        # Adjust synthetic to match real data shape
        if synthetic.shape[1] != data.shape[1]:
            if synthetic.shape[1] > data.shape[1]:
                synthetic = synthetic[:, : data.shape[1]]
            else:
                padded = np.zeros((synthetic.shape[0], data.shape[1]))
                padded[:, : synthetic.shape[1]] = synthetic
                synthetic = padded

        return np.vstack([data, synthetic])

    # ------------------------------------------------------------------
    # Quality evaluation
    # ------------------------------------------------------------------

    def evaluate_quality(
        self, real_data: Any, generated_data: Any
    ) -> Dict[str, Any]:
        """Assess generated data quality.

        Returns: {wasserstein_distance, correlation_preservation,
                  distribution_overlap, autocorrelation_match,
                  quality_score: 0-1}
        """
        real = np.asarray(real_data, dtype=np.float64)
        gen = np.asarray(generated_data, dtype=np.float64)

        if real.ndim == 1:
            real = real.reshape(-1, 1)
        if gen.ndim == 1:
            gen = gen.reshape(-1, 1)

        # Ensure same number of features
        n_feat = min(real.shape[1], gen.shape[1])
        real = real[:, :n_feat]
        gen = gen[:, :n_feat]

        results: Dict[str, Any] = {
            "n_real": real.shape[0],
            "n_generated": gen.shape[0],
            "n_features": n_feat,
        }

        # Wasserstein distance (1D, per feature, averaged)
        w_distances = []
        for f in range(n_feat):
            real_sorted = np.sort(real[:, f])
            gen_sorted = np.sort(gen[:, f])
            # Interpolate to same length
            n = max(len(real_sorted), len(gen_sorted))
            real_interp = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(real_sorted)),
                real_sorted,
            )
            gen_interp = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(gen_sorted)),
                gen_sorted,
            )
            w_dist = float(np.mean(np.abs(real_interp - gen_interp)))
            w_distances.append(w_dist)

        results["wasserstein_distance"] = round(float(np.mean(w_distances)), 6)

        # Correlation preservation
        if n_feat >= 2 and real.shape[0] >= 3 and gen.shape[0] >= 3:
            real_corr = np.corrcoef(real.T)
            gen_corr = np.corrcoef(gen.T)
            # Handle NaN in correlation matrices
            real_corr = np.nan_to_num(real_corr, nan=0.0)
            gen_corr = np.nan_to_num(gen_corr, nan=0.0)
            corr_diff = float(np.mean(np.abs(real_corr - gen_corr)))
            corr_preservation = max(0.0, 1.0 - corr_diff)
        else:
            corr_preservation = 1.0  # single feature, nothing to compare

        results["correlation_preservation"] = round(corr_preservation, 4)

        # Distribution overlap (per feature, using histogram intersection)
        overlaps = []
        for f in range(n_feat):
            r_min = min(float(np.min(real[:, f])), float(np.min(gen[:, f])))
            r_max = max(float(np.max(real[:, f])), float(np.max(gen[:, f])))
            if r_max - r_min < 1e-10:
                overlaps.append(1.0)
                continue
            bins = np.linspace(r_min, r_max, 30)
            h_real, _ = np.histogram(real[:, f], bins=bins, density=True)
            h_gen, _ = np.histogram(gen[:, f], bins=bins, density=True)
            # Normalize
            h_real = h_real / (np.sum(h_real) + 1e-10)
            h_gen = h_gen / (np.sum(h_gen) + 1e-10)
            overlap = float(np.sum(np.minimum(h_real, h_gen)))
            overlaps.append(overlap)

        results["distribution_overlap"] = round(float(np.mean(overlaps)), 4)

        # Autocorrelation match (for time series features)
        acf_matches = []
        for f in range(n_feat):
            if len(real[:, f]) > 5 and len(gen[:, f]) > 5:
                real_acf = self._simple_acf(real[:, f], nlags=5)
                gen_acf = self._simple_acf(gen[:, f], nlags=5)
                acf_diff = float(np.mean(np.abs(real_acf - gen_acf)))
                acf_matches.append(max(0.0, 1.0 - acf_diff))
            else:
                acf_matches.append(0.5)

        results["autocorrelation_match"] = round(float(np.mean(acf_matches)), 4)

        # Overall quality score (weighted average)
        w_score = max(0.0, 1.0 - results["wasserstein_distance"] * 10)
        quality_score = (
            0.3 * w_score
            + 0.3 * corr_preservation
            + 0.2 * results["distribution_overlap"]
            + 0.2 * results["autocorrelation_match"]
        )
        results["quality_score"] = round(float(np.clip(quality_score, 0.0, 1.0)), 4)

        return results

    @staticmethod
    def _simple_acf(x: np.ndarray, nlags: int = 5) -> np.ndarray:
        """Compute simple autocorrelation function."""
        x = x - np.mean(x)
        var = float(np.var(x))
        if var < 1e-12:
            return np.zeros(nlags)
        acf = np.zeros(nlags)
        for lag in range(nlags):
            if lag + 1 < len(x):
                acf[lag] = float(np.mean(x[: len(x) - lag - 1] * x[lag + 1 :])) / var
        return acf

    # ------------------------------------------------------------------
    # Gate primitives (statevector simulation)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_ry(state: np.ndarray, n: int, qubit: int, angle: float) -> np.ndarray:
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        gate = np.array([[c, -s], [s, c]], dtype=np.complex128)
        return QuantumGAN._apply_gate(state, n, qubit, gate)

    @staticmethod
    def _apply_rx(state: np.ndarray, n: int, qubit: int, angle: float) -> np.ndarray:
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        gate = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
        return QuantumGAN._apply_gate(state, n, qubit, gate)

    @staticmethod
    def _apply_rz(state: np.ndarray, n: int, qubit: int, angle: float) -> np.ndarray:
        gate = np.array(
            [[np.exp(-1j * angle / 2), 0], [0, np.exp(1j * angle / 2)]],
            dtype=np.complex128,
        )
        return QuantumGAN._apply_gate(state, n, qubit, gate)

    @staticmethod
    def _apply_gate(
        state: np.ndarray, n: int, qubit: int, gate: np.ndarray
    ) -> np.ndarray:
        """Apply 2x2 gate to a specific qubit."""
        shape = [2] * n
        psi = state.reshape(shape)
        psi = np.moveaxis(psi, qubit, -1)
        psi = np.einsum("ij,...j->...i", gate, psi)
        psi = np.moveaxis(psi, -1, qubit)
        return psi.reshape(2 ** n)

    @staticmethod
    def _apply_cnot(state: np.ndarray, n: int, control: int, target: int) -> np.ndarray:
        """Apply CNOT gate."""
        dim = 2 ** n
        new_state = np.zeros(dim, dtype=np.complex128)
        for i in range(dim):
            bits = [(i >> (n - 1 - q)) & 1 for q in range(n)]
            if bits[control] == 1:
                bits[target] ^= 1
                j = 0
                for q in range(n):
                    j = (j << 1) | bits[q]
                new_state[j] += state[i]
            else:
                new_state[i] += state[i]
        return new_state

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the QGAN state."""
        return {
            "n_features": self.n_features,
            "latent_dim": self.latent_dim,
            "n_qubits": self.n_qubits,
            "n_gen_params": len(self._gen_params),
            "trained": self._trained,
            "train_time_s": round(self._train_time, 3) if self._trained else None,
            "final_gen_loss": (
                round(self._gen_losses[-1], 4) if self._gen_losses else None
            ),
            "final_disc_loss": (
                round(self._disc_losses[-1], 4) if self._disc_losses else None
            ),
            "method": "quantum_inspired_gan",
        }
