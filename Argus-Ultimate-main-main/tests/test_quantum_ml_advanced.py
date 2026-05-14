"""
Tests for advanced quantum ML models:
- QuantumBoltzmannMachine (quantum_boltzmann.py)
- QuantumGAN (quantum_gan.py)
- QuantumPolicyNetwork (quantum_rl.py)

Run with: py -m pytest tests/test_quantum_ml_advanced.py -v
"""

from __future__ import annotations

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def returns_data(rng):
    """Synthetic log-return data (200 samples, 10 features)."""
    return rng.randn(200, 10) * 0.02


@pytest.fixture
def returns_1d(rng):
    """Synthetic 1D log-returns."""
    return rng.randn(300) * 0.02


@pytest.fixture
def market_data(rng):
    """Synthetic market data: (samples, 5) — returns, vol, volume, spread, momentum."""
    n = 150
    returns = rng.randn(n) * 0.02
    vol = np.abs(rng.randn(n)) * 0.05
    volume = np.exp(rng.randn(n) * 0.5 + 10)
    spread = np.abs(rng.randn(n)) * 0.001
    momentum = np.cumsum(returns)
    return np.column_stack([returns, vol, volume, spread, momentum])


# ═══════════════════════════════════════════════════════════════════════════
# Quantum Boltzmann Machine
# ═══════════════════════════════════════════════════════════════════════════


class TestQuantumBoltzmannMachine:
    """Tests for QuantumBoltzmannMachine."""

    def _make_qbm(self, **kwargs):
        from quantum.qml.quantum_boltzmann import QuantumBoltzmannMachine
        defaults = {"n_visible": 10, "n_hidden": 5, "learning_rate": 0.01, "seed": 42}
        defaults.update(kwargs)
        return QuantumBoltzmannMachine(**defaults)

    def test_init(self):
        qbm = self._make_qbm()
        assert qbm.n_visible == 10
        assert qbm.n_hidden == 5
        assert qbm.weights.shape == (10, 5)
        assert not qbm._fitted

    def test_init_validation(self):
        with pytest.raises(ValueError):
            self._make_qbm(n_visible=0)
        with pytest.raises(ValueError):
            self._make_qbm(n_hidden=-1)

    def test_fit_basic(self, returns_data):
        qbm = self._make_qbm()
        result = qbm.fit(returns_data, epochs=5, batch_size=32)
        assert result is qbm  # chaining
        assert qbm._fitted
        assert len(qbm._reconstruction_errors) == 5
        assert qbm._n_train_samples == 200

    def test_fit_1d_data(self, returns_1d):
        qbm = self._make_qbm(n_visible=1, n_hidden=3)
        qbm.fit(returns_1d, epochs=3)
        assert qbm._fitted

    def test_fit_convergence(self, returns_data):
        """Reconstruction error should generally decrease over training."""
        qbm = self._make_qbm()
        qbm.fit(returns_data, epochs=20, batch_size=32)
        errors = qbm._reconstruction_errors
        # First error should be higher than last (or at least not drastically worse)
        assert errors[-1] <= errors[0] * 2.0  # allow some tolerance

    def test_fit_nan_handling(self, rng):
        data = rng.randn(100, 10) * 0.02
        data[5, 3] = np.nan
        data[10, 7] = np.nan
        qbm = self._make_qbm()
        qbm.fit(data, epochs=3)
        assert qbm._fitted

    def test_generate_samples(self, returns_data):
        qbm = self._make_qbm()
        qbm.fit(returns_data, epochs=5)
        samples = qbm.generate_samples(n_samples=50, burn_in=10)
        assert samples.shape[0] == 50
        assert not np.any(np.isnan(samples))

    def test_generate_before_fit_raises(self):
        qbm = self._make_qbm()
        with pytest.raises(RuntimeError, match="fit"):
            qbm.generate_samples(10)

    def test_anomaly_score_fitted(self, returns_data):
        qbm = self._make_qbm()
        qbm.fit(returns_data, epochs=5)
        # Normal observation
        normal = returns_data[0]
        score_normal = qbm.anomaly_score(normal)
        assert 0.0 <= score_normal <= 1.0
        # Extreme observation
        extreme = np.ones(10) * 10.0
        score_extreme = qbm.anomaly_score(extreme)
        assert 0.0 <= score_extreme <= 1.0

    def test_anomaly_score_unfitted(self):
        qbm = self._make_qbm()
        score = qbm.anomaly_score(np.zeros(10))
        assert score == 0.5  # uncertain default

    def test_anomaly_score_nan(self, returns_data):
        qbm = self._make_qbm()
        qbm.fit(returns_data, epochs=3)
        obs = np.array([0.01, np.nan, 0.02, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        score = qbm.anomaly_score(obs)
        assert 0.0 <= score <= 1.0

    def test_estimate_distribution(self, returns_1d):
        qbm = self._make_qbm(n_visible=1, n_hidden=3)
        qbm.fit(returns_1d, epochs=5)
        dist = qbm.estimate_distribution(returns_1d)
        assert "mean" in dist
        assert "std" in dist
        assert "skewness" in dist
        assert "kurtosis" in dist
        assert "var_95" in dist
        assert "cvar_95" in dist
        assert "distribution_type" in dist
        assert dist["n_samples"] == 300

    def test_estimate_distribution_insufficient_data(self):
        qbm = self._make_qbm()
        dist = qbm.estimate_distribution([0.01])
        assert dist["distribution_type"] == "insufficient_data"

    def test_estimate_distribution_constant_data(self):
        qbm = self._make_qbm(n_visible=1, n_hidden=2)
        qbm.fit(np.ones(50), epochs=3)
        dist = qbm.estimate_distribution(np.ones(50))
        assert dist["std"] == 0.0

    def test_benchmark_vs_classical(self, returns_1d):
        qbm = self._make_qbm(n_visible=1, n_hidden=3)
        bench = qbm.benchmark_vs_classical(returns_1d)
        assert "best_method" in bench
        assert "honest_note" in bench
        assert "n_samples" in bench

    def test_summary(self, returns_data):
        qbm = self._make_qbm()
        s = qbm.summary()
        assert s["fitted"] is False
        qbm.fit(returns_data, epochs=3)
        s = qbm.summary()
        assert s["fitted"] is True
        assert s["method"] == "quantum_inspired_annealing_rbm"

    def test_free_energy(self, returns_data):
        qbm = self._make_qbm()
        qbm.fit(returns_data, epochs=3)
        v = qbm._normalise_data(returns_data[:5])
        fe = qbm._free_energy(v)
        assert fe.shape == (5,)
        assert not np.any(np.isnan(fe))

    def test_fit_wide_data_truncated(self, rng):
        """Data wider than n_visible gets truncated."""
        data = rng.randn(50, 20)
        qbm = self._make_qbm(n_visible=5, n_hidden=3)
        qbm.fit(data, epochs=2)
        assert qbm._fitted

    def test_fit_narrow_data_padded(self, rng):
        """Data narrower than n_visible gets padded."""
        data = rng.randn(50, 3)
        qbm = self._make_qbm(n_visible=10, n_hidden=5)
        qbm.fit(data, epochs=2)
        assert qbm._fitted


# ═══════════════════════════════════════════════════════════════════════════
# Quantum GAN
# ═══════════════════════════════════════════════════════════════════════════


class TestQuantumGAN:
    """Tests for QuantumGAN."""

    def _make_qgan(self, **kwargs):
        from quantum.qml.quantum_gan import QuantumGAN
        defaults = {"n_features": 5, "latent_dim": 3, "n_qubits": 3, "seed": 42}
        defaults.update(kwargs)
        return QuantumGAN(**defaults)

    def test_init(self):
        qgan = self._make_qgan()
        assert qgan.n_features == 5
        assert qgan.latent_dim == 3
        assert qgan.n_qubits == 3
        assert not qgan._trained

    def test_init_validation(self):
        with pytest.raises(ValueError):
            self._make_qgan(n_features=0)
        with pytest.raises(ValueError):
            self._make_qgan(latent_dim=0)
        with pytest.raises(ValueError):
            self._make_qgan(n_qubits=13)

    def test_train_basic(self, market_data):
        qgan = self._make_qgan()
        result = qgan.train(market_data, epochs=3, batch_size=32)
        assert result is qgan
        assert qgan._trained
        assert len(qgan._gen_losses) == 3
        assert len(qgan._disc_losses) == 3

    def test_train_1d(self, rng):
        data = rng.randn(100) * 0.02
        qgan = self._make_qgan(n_features=1, latent_dim=2, n_qubits=2)
        qgan.train(data, epochs=3)
        assert qgan._trained

    def test_train_nan_handling(self, rng):
        data = rng.randn(80, 5) * 0.02
        data[10, 2] = np.nan
        qgan = self._make_qgan()
        qgan.train(data, epochs=2)
        assert qgan._trained

    def test_generate(self, market_data):
        qgan = self._make_qgan()
        qgan.train(market_data, epochs=3)
        samples = qgan.generate(n_samples=50)
        assert samples.shape == (50, 5)
        assert not np.any(np.isnan(samples))

    def test_generate_before_train_raises(self):
        qgan = self._make_qgan()
        with pytest.raises(RuntimeError, match="train"):
            qgan.generate(10)

    def test_augment_training_data(self, market_data):
        qgan = self._make_qgan()
        qgan.train(market_data, epochs=3)
        augmented = qgan.augment_training_data(market_data, augmentation_factor=1.0)
        assert len(augmented) >= len(market_data)
        assert augmented.shape[1] == market_data.shape[1]

    def test_augment_auto_trains(self, market_data):
        """augment_training_data should auto-train if not already trained."""
        qgan = self._make_qgan()
        augmented = qgan.augment_training_data(market_data, augmentation_factor=0.5)
        assert qgan._trained
        assert len(augmented) > len(market_data)

    def test_evaluate_quality(self, market_data, rng):
        qgan = self._make_qgan()
        qgan.train(market_data, epochs=3)
        generated = qgan.generate(n_samples=100)
        quality = qgan.evaluate_quality(market_data, generated)
        assert "wasserstein_distance" in quality
        assert "correlation_preservation" in quality
        assert "distribution_overlap" in quality
        assert "autocorrelation_match" in quality
        assert "quality_score" in quality
        assert 0.0 <= quality["quality_score"] <= 1.0

    def test_evaluate_quality_1d(self, rng):
        real = rng.randn(100)
        gen = rng.randn(80)
        qgan = self._make_qgan(n_features=1)
        quality = qgan.evaluate_quality(real, gen)
        assert "quality_score" in quality

    def test_quantum_generator_produces_values(self):
        qgan = self._make_qgan(n_features=3, n_qubits=3)
        noise = np.array([0.5, -0.3, 1.0])
        output = qgan._quantum_generator(noise, qgan._gen_params)
        assert len(output) == 3
        assert not np.any(np.isnan(output))

    def test_discriminator_output_range(self, rng):
        qgan = self._make_qgan()
        sample = rng.randn(5)
        d = qgan._classical_discriminator(sample)
        assert 0.0 <= d <= 1.0

    def test_summary(self, market_data):
        qgan = self._make_qgan()
        s = qgan.summary()
        assert s["trained"] is False
        qgan.train(market_data, epochs=2)
        s = qgan.summary()
        assert s["trained"] is True
        assert s["method"] == "quantum_inspired_gan"

    def test_correlation_preservation(self, rng):
        """Generated data should have some correlation structure."""
        # Create correlated data
        n = 200
        x1 = rng.randn(n)
        x2 = x1 * 0.8 + rng.randn(n) * 0.2  # correlated with x1
        x3 = rng.randn(n)
        data = np.column_stack([x1, x2, x3])
        qgan = self._make_qgan(n_features=3, latent_dim=2, n_qubits=3)
        qgan.train(data, epochs=5)
        generated = qgan.generate(n_samples=200)
        quality = qgan.evaluate_quality(data, generated)
        # Just check it runs and produces valid metrics
        assert quality["correlation_preservation"] >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Quantum RL Policy Network
# ═══════════════════════════════════════════════════════════════════════════


class TestQuantumPolicyNetwork:
    """Tests for QuantumPolicyNetwork."""

    def _make_qpn(self, **kwargs):
        from quantum.qml.quantum_rl import QuantumPolicyNetwork
        defaults = {
            "n_state_features": 8,
            "n_actions": 3,
            "n_layers": 1,
            "n_qubits": 3,
            "learning_rate": 0.01,
            "seed": 42,
        }
        defaults.update(kwargs)
        return QuantumPolicyNetwork(**defaults)

    def test_init(self):
        qpn = self._make_qpn()
        assert qpn.n_state_features == 8
        assert qpn.n_actions == 3
        assert qpn.n_qubits == 3
        assert qpn._total_episodes == 0

    def test_init_validation(self):
        with pytest.raises(ValueError):
            self._make_qpn(n_state_features=0)
        with pytest.raises(ValueError):
            self._make_qpn(n_actions=1)
        with pytest.raises(ValueError):
            self._make_qpn(n_qubits=13)

    def test_forward_pass(self, rng):
        qpn = self._make_qpn()
        state = rng.randn(8)
        probs = qpn.forward(state)
        assert len(probs) == 3
        assert abs(np.sum(probs) - 1.0) < 1e-6
        assert all(p >= 0 for p in probs)

    def test_forward_short_state_padded(self, rng):
        qpn = self._make_qpn()
        state = rng.randn(3)  # shorter than n_state_features
        probs = qpn.forward(state)
        assert len(probs) == 3
        assert abs(np.sum(probs) - 1.0) < 1e-6

    def test_forward_nan_handling(self):
        qpn = self._make_qpn()
        state = np.array([0.1, np.nan, 0.3, np.nan, 0.0, 0.0, 0.0, 0.0])
        probs = qpn.forward(state)
        assert not np.any(np.isnan(probs))
        assert abs(np.sum(probs) - 1.0) < 1e-6

    def test_select_action(self, rng):
        qpn = self._make_qpn()
        state = rng.randn(8)
        action, prob, entropy = qpn.select_action(state)
        assert action in [0, 1, 2]
        assert 0.0 <= prob <= 1.0
        assert entropy >= 0.0

    def test_select_action_exploration(self):
        """With epsilon=1.0, actions should be random."""
        qpn = self._make_qpn(epsilon=1.0)
        state = np.zeros(8)
        actions = set()
        for _ in range(100):
            a, _, _ = qpn.select_action(state)
            actions.add(a)
        # Should see all 3 actions with high probability
        assert len(actions) >= 2

    def test_update_basic(self, rng):
        qpn = self._make_qpn()
        states = [rng.randn(8) for _ in range(5)]
        actions = [0, 1, 2, 1, 0]
        rewards = [0.1, -0.05, 0.2, 0.0, -0.1]
        result = qpn.update(states, actions, rewards)
        assert "loss" in result
        assert "avg_reward" in result
        assert "entropy" in result

    def test_update_empty(self):
        qpn = self._make_qpn()
        result = qpn.update([], [], [])
        assert result["loss"] == 0.0

    def test_update_changes_params(self, rng):
        qpn = self._make_qpn()
        params_before = qpn.params.copy()
        states = [rng.randn(8) for _ in range(3)]
        actions = [0, 1, 2]
        rewards = [1.0, -1.0, 0.5]
        qpn.update(states, actions, rewards)
        # At least some parameters should change
        assert not np.allclose(qpn.params, params_before)

    def test_train_episode(self, rng):
        qpn = self._make_qpn()
        step_count = [0]

        def env_step(action):
            step_count[0] += 1
            next_state = rng.randn(8)
            reward = rng.randn() * 0.1
            done = step_count[0] >= 10
            return next_state, reward, done

        result = qpn.train_episode(env_step, initial_state=rng.randn(8), max_steps=20)
        assert "total_reward" in result
        assert "steps" in result
        assert "actions_taken" in result
        assert result["steps"] == 10  # env said done after 10
        assert qpn._total_episodes == 1

    def test_train_episode_max_steps(self, rng):
        qpn = self._make_qpn()

        def env_step(action):
            return rng.randn(8), 0.01, False  # never done

        result = qpn.train_episode(env_step, initial_state=rng.randn(8), max_steps=5)
        assert result["steps"] == 5

    def test_get_trading_action(self):
        qpn = self._make_qpn()
        market_state = {
            "price": 60000.0,
            "returns_5": 0.01,
            "vol_10": 0.03,
            "rsi": 55.0,
            "macd": 0.001,
            "spread": 5.0,
            "volume": 1_000_000.0,
            "regime": "TRENDING",
        }
        result = qpn.get_trading_action(market_state)
        assert result["action"] in ["BUY", "HOLD", "SELL"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["entropy"] >= 0.0
        assert isinstance(result["exploration"], bool)
        assert "probabilities" in result
        assert len(result["probabilities"]) == 3

    def test_get_trading_action_unknown_regime(self):
        qpn = self._make_qpn()
        market_state = {"price": 50000.0, "regime": "CRISIS"}
        result = qpn.get_trading_action(market_state)
        assert result["action"] in ["BUY", "HOLD", "SELL"]

    def test_get_trading_action_missing_keys(self):
        """Should handle missing market state keys gracefully."""
        qpn = self._make_qpn()
        result = qpn.get_trading_action({})
        assert result["action"] in ["BUY", "HOLD", "SELL"]

    def test_summary(self, rng):
        qpn = self._make_qpn()
        s = qpn.summary()
        assert s["total_episodes"] == 0
        assert s["method"] == "quantum_policy_reinforce"

        # After training an episode
        def env_step(action):
            return rng.randn(8), 0.01, True

        qpn.train_episode(env_step, initial_state=rng.randn(8))
        s = qpn.summary()
        assert s["total_episodes"] == 1

    def test_multiple_episodes_reward_tracking(self, rng):
        qpn = self._make_qpn()
        for _ in range(3):
            step = [0]
            def env_step(action, _s=step):
                _s[0] += 1
                return rng.randn(8), rng.randn() * 0.1, _s[0] >= 5
            qpn.train_episode(env_step, initial_state=rng.randn(8))
        assert qpn._total_episodes == 3
        assert len(qpn._episode_rewards) == 3


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases across all models
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Cross-cutting edge case tests."""

    def test_qbm_single_sample(self):
        from quantum.qml.quantum_boltzmann import QuantumBoltzmannMachine
        qbm = QuantumBoltzmannMachine(n_visible=3, n_hidden=2, seed=42)
        # Single sample: should still fit (with padding)
        qbm.fit(np.array([[0.1, 0.2, 0.3]]), epochs=2)
        assert qbm._fitted

    def test_qbm_constant_data(self):
        from quantum.qml.quantum_boltzmann import QuantumBoltzmannMachine
        qbm = QuantumBoltzmannMachine(n_visible=5, n_hidden=3, seed=42)
        data = np.ones((50, 5)) * 0.5
        qbm.fit(data, epochs=3)
        assert qbm._fitted
        samples = qbm.generate_samples(10)
        assert not np.any(np.isnan(samples))

    def test_qgan_single_feature(self):
        from quantum.qml.quantum_gan import QuantumGAN
        qgan = QuantumGAN(n_features=1, latent_dim=1, n_qubits=2, seed=42)
        data = np.random.randn(50, 1) * 0.01
        qgan.train(data, epochs=2)
        samples = qgan.generate(20)
        assert samples.shape == (20, 1)

    def test_qrl_two_actions(self):
        from quantum.qml.quantum_rl import QuantumPolicyNetwork
        qpn = QuantumPolicyNetwork(n_state_features=4, n_actions=2, n_qubits=2, seed=42)
        probs = qpn.forward(np.array([0.1, 0.2, 0.3, 0.4]))
        assert len(probs) == 2
        assert abs(np.sum(probs) - 1.0) < 1e-6

    def test_qbm_anomaly_short_observation(self):
        from quantum.qml.quantum_boltzmann import QuantumBoltzmannMachine
        qbm = QuantumBoltzmannMachine(n_visible=10, n_hidden=5, seed=42)
        qbm.fit(np.random.randn(50, 10) * 0.01, epochs=3)
        # Observation shorter than n_visible -> should be padded
        score = qbm.anomaly_score(np.array([0.01, 0.02]))
        assert 0.0 <= score <= 1.0

    def test_qgan_evaluate_quality_constant(self):
        from quantum.qml.quantum_gan import QuantumGAN
        qgan = QuantumGAN(n_features=2, latent_dim=2, n_qubits=2, seed=42)
        real = np.ones((30, 2))
        gen = np.ones((30, 2))
        quality = qgan.evaluate_quality(real, gen)
        assert quality["wasserstein_distance"] == 0.0

    def test_qrl_forward_deterministic(self):
        from quantum.qml.quantum_rl import QuantumPolicyNetwork
        qpn = QuantumPolicyNetwork(n_state_features=4, n_actions=3, n_qubits=3, seed=42)
        state = np.array([0.1, 0.2, 0.3, 0.4])
        probs1 = qpn.forward(state)
        probs2 = qpn.forward(state)
        np.testing.assert_array_almost_equal(probs1, probs2)
