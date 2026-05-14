"""
core/jax_policy_network.py
==========================
Flax/Haiku policy network wired into JaxPPOTrainer.

Provides:
  - ActorCriticNetwork  : shared-trunk actor-critic in Flax (NNX)
  - PolicyInference     : JIT-compiled forward pass
  - NetworkFactory      : builds network + optimiser ready for JaxPPOTrainer

Falls back to a lightweight NumPy stub when Flax/optax are not installed
so the rest of the system keeps running.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger("argus.core.jax_policy_network")

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, vmap
    import flax.linen as nn
    import optax
    FLAX_AVAILABLE = True
    logger.info("Flax + optax detected — full neural policy enabled")
except ImportError:
    FLAX_AVAILABLE = False
    logger.warning(
        "Flax/optax not installed — using NumPy stub policy.\n"
        "Install: pip install flax optax"
    )
    jnp = np  # type: ignore


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PolicyConfig:
    obs_dim: int = 83          # LOB_DEPTH*4 + mid/spread/imbalance
    n_actions: int = 5
    hidden_dims: Tuple[int, ...] = (256, 256, 128)
    activation: str = "tanh"   # "tanh" | "relu" | "gelu"
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    init_log_std: float = -0.5


# ---------------------------------------------------------------------------
# Flax actor-critic network
# ---------------------------------------------------------------------------

if FLAX_AVAILABLE:
    class ActorCriticNetwork(nn.Module):
        """
        Shared-trunk actor-critic.

        Trunk  -> actor head  (logits over n_actions)
               -> critic head (scalar value estimate)
        """
        config: PolicyConfig

        @nn.compact
        def __call__(self, obs: Any, training: bool = False) -> Tuple[Any, Any]:
            act_fn = {
                "tanh": nn.tanh,
                "relu": nn.relu,
                "gelu": nn.gelu,
            }.get(self.config.activation, nn.tanh)

            x = obs.astype(jnp.float32)

            # Shared trunk
            for dim in self.config.hidden_dims:
                x = nn.Dense(dim)(x)
                x = nn.LayerNorm()(x)
                x = act_fn(x)

            # Actor head
            logits = nn.Dense(self.config.n_actions)(x)

            # Critic head
            value = nn.Dense(1)(x).squeeze(-1)

            return logits, value

else:
    # NumPy stub — same interface, random weights
    class ActorCriticNetwork:  # type: ignore
        def __init__(self, config: PolicyConfig) -> None:
            self.config = config
            self._W = np.random.randn(config.obs_dim, config.n_actions) * 0.01

        def __call__(self, obs: Any, training: bool = False) -> Tuple[Any, Any]:
            logits = obs @ self._W
            value = np.zeros(obs.shape[0] if obs.ndim > 1 else 1)
            return logits, value


# ---------------------------------------------------------------------------
# JIT-compiled inference wrapper
# ---------------------------------------------------------------------------

class PolicyInference:
    """
    Wraps ActorCriticNetwork with JIT-compiled forward pass.
    Handles param init, action sampling, and value extraction.
    """

    def __init__(self, network: Any, params: Any, config: PolicyConfig) -> None:
        self._net = network
        self._params = params
        self._cfg = config

        if FLAX_AVAILABLE:
            self._forward = jit(lambda p, x: network.apply(p, x))
        else:
            self._forward = lambda p, x: network(x)

    def action_and_value(
        self, obs: np.ndarray
    ) -> Tuple[int, float, float]:
        """
        Run forward pass and sample action.

        Returns
        -------
        action : int
        log_prob : float
        value : float
        """
        obs_arr = jnp.array(obs, dtype=jnp.float32)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr[None, :]  # add batch dim

        logits, value = self._forward(self._params, obs_arr)
        logits = np.array(logits[0])
        value = float(np.array(value[0]))

        # Softmax + categorical sample
        logits -= logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        action = int(np.random.choice(self._cfg.n_actions, p=probs))
        log_prob = float(np.log(probs[action] + 1e-8))

        return action, log_prob, value

    def update_params(self, new_params: Any) -> None:
        self._params = new_params

    @property
    def params(self) -> Any:
        return self._params


# ---------------------------------------------------------------------------
# PPO update step (Flax path)
# ---------------------------------------------------------------------------

class FlaxPPOUpdater:
    """
    JIT-compiled PPO clipped objective update.
    Replaces the stub in JaxPPOTrainer._ppo_update().
    """

    def __init__(
        self,
        network: Any,
        params: Any,
        config: PolicyConfig,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ) -> None:
        self._net = network
        self._params = params
        self._cfg = config
        self._clip_eps = clip_eps
        self._entropy_coef = entropy_coef
        self._value_coef = value_coef

        if FLAX_AVAILABLE:
            self._opt = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(config.learning_rate),
            )
            self._opt_state = self._opt.init(params)
            self._update_step = jit(self._ppo_loss_and_grad)
        else:
            self._opt = None
            self._opt_state = None
            self._update_step = self._numpy_stub_update

    def update(self, rollout: dict) -> float:
        """
        Run one PPO update epoch over rollout data.
        Returns mean loss.
        """
        if not FLAX_AVAILABLE:
            return 0.0

        obs = jnp.array(rollout["obs"])
        actions = jnp.array(rollout["actions"])
        old_log_probs = jnp.array(rollout["log_probs"])
        returns = jnp.array(rollout["returns"])
        old_values = jnp.array(rollout["values"])

        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        loss, grads = self._update_step(
            self._params, obs, actions, old_log_probs, returns, advantages
        )
        updates, self._opt_state = self._opt.update(grads, self._opt_state)
        self._params = optax.apply_updates(self._params, updates)
        return float(loss)

    def _ppo_loss_and_grad(self, params, obs, actions, old_log_probs, returns, advantages):
        def loss_fn(p):
            logits, values = self._net.apply(p, obs)

            # Log probs
            log_probs_all = jax.nn.log_softmax(logits)
            log_probs = log_probs_all[jnp.arange(len(actions)), actions]

            # PPO clipped ratio
            ratio = jnp.exp(log_probs - old_log_probs)
            clipped = jnp.clip(ratio, 1 - self._clip_eps, 1 + self._clip_eps)
            policy_loss = -jnp.minimum(ratio * advantages, clipped * advantages).mean()

            # Value loss
            value_loss = self._value_coef * ((values - returns) ** 2).mean()

            # Entropy bonus
            probs = jax.nn.softmax(logits)
            entropy = -(probs * jax.nn.log_softmax(logits)).sum(-1).mean()
            entropy_loss = -self._entropy_coef * entropy

            return policy_loss + value_loss + entropy_loss

        return jax.value_and_grad(loss_fn)(params)

    def _numpy_stub_update(self, *args) -> float:
        return 0.0

    @property
    def params(self) -> Any:
        return self._params


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class NetworkFactory:
    """
    Builds a wired (network, inference, updater) triple ready for
    JaxPPOTrainer injection.

    Usage
    -----
    inference, updater = NetworkFactory.build(config)
    trainer = JaxPPOTrainer(policy_network=inference, ...)
    # after each rollout:
    loss = updater.update(rollout)
    inference.update_params(updater.params)
    """

    @staticmethod
    def build(
        config: Optional[PolicyConfig] = None,
        seed: int = 42,
    ) -> Tuple[PolicyInference, FlaxPPOUpdater]:
        cfg = config or PolicyConfig()
        net = ActorCriticNetwork(cfg) if FLAX_AVAILABLE else ActorCriticNetwork(cfg)

        if FLAX_AVAILABLE:
            import jax
            key = jax.random.PRNGKey(seed)
            dummy_obs = jnp.zeros((1, cfg.obs_dim))
            params = net.init(key, dummy_obs)
            logger.info(
                "NetworkFactory: Flax ActorCritic initialised "
                "obs_dim=%d n_actions=%d hidden=%s",
                cfg.obs_dim, cfg.n_actions, cfg.hidden_dims,
            )
        else:
            params = None
            logger.warning("NetworkFactory: using NumPy stub params")

        inference = PolicyInference(net, params, cfg)
        updater = FlaxPPOUpdater(
            net, params, cfg,
            clip_eps=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=cfg.max_grad_norm,
        )
        return inference, updater
