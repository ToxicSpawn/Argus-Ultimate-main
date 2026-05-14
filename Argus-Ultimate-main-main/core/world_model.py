"""World model for ARGUS — transformer-inspired trading dynamics simulator.

Pure-numpy implementation of a simplified world model for model-based planning.

The world model learns three components from experience:

* ``StateEncoder``: maps raw observations (price, volume, regime) into a dense
  feature vector used as input to the dynamics models.
* ``TransitionModel``: linear mapping ``(state, action) -> next_state`` trained
  online with stochastic gradient descent.
* ``RewardModel``: linear mapping ``(state, action) -> scalar reward`` trained
  alongside the transition model.

``WorldModel`` exposes the public API that the rest of ARGUS consumes:

* ``encode_state(obs)`` — turn a raw observation dict into a numpy vector.
* ``predict_next(state, action)`` — single-step rollout returning
  ``(next_state, predicted_reward)``.
* ``rollout(state, actions_sequence)`` — multi-step imagined rollout used by
  MPC-style planners.
* ``update(state, action, actual_next, actual_reward)`` — online training step
  that nudges all learned parameters toward the observed transition.
* ``snapshot()`` — serialisable summary for monitoring / audit trail.

Uncertainty is estimated with a tiny ensemble of transition heads: the variance
across heads gives a rough epistemic uncertainty signal that downstream agents
can use to discount imagined trajectories.

Intended to be standalone. Depends only on stdlib + numpy.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WorldModelConfig:
    """Configuration for :class:`WorldModel`.

    All values are conservative defaults suitable for a ~8-dim state space and
    scalar actions. Override via constructor kwargs when instantiating.
    """

    state_dim: int = 8
    action_dim: int = 1
    ensemble_size: int = 3
    learning_rate: float = 1e-3
    reward_learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_update_norm: float = 1.0
    seed: int = 0


@dataclass
class RolloutResult:
    """Result of a multi-step imagined rollout.

    ``states`` has shape ``(horizon + 1, state_dim)`` — the first row is the
    initial state. ``rewards`` and ``uncertainty`` each have shape
    ``(horizon,)``.
    """

    states: np.ndarray
    rewards: np.ndarray
    uncertainty: np.ndarray
    total_reward: float
    mean_uncertainty: float


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class StateEncoder:
    """Encode a raw observation dict into a fixed-size feature vector.

    The encoder is deterministic — it does not learn. It standardises a small
    set of hand-picked features so that downstream linear models see inputs of
    roughly unit scale. Missing keys are imputed with zeros.
    """

    REGIME_LOOKUP = {
        "trending_up": 1.0,
        "trending_down": -1.0,
        "ranging": 0.0,
        "high_vol": 0.5,
        "crisis": -0.9,
    }

    def __init__(self, state_dim: int = 8) -> None:
        self.state_dim = int(state_dim)

    def encode(self, obs: Dict[str, Any]) -> np.ndarray:
        """Encode ``obs`` into a ``(state_dim,)`` numpy array.

        The canonical encoder produces up to 8 features; if ``state_dim`` is
        smaller the result is truncated, and if it is larger the extra slots
        are left as zeros (available for callers to fill with symbol-specific
        context before handing the vector on).
        """

        price = float(obs.get("price", 0.0))
        volume = float(obs.get("volume", 0.0))
        regime_label = str(obs.get("regime", "ranging")).lower()
        regime_conf = float(obs.get("regime_confidence", 0.5))
        volatility = float(obs.get("volatility", 0.0))
        spread = float(obs.get("spread", 0.0))
        position = float(obs.get("position", 0.0))
        pnl = float(obs.get("unrealized_pnl", 0.0))

        # Log-normalised price and volume keep the encoder numerically safe
        # across symbols whose raw magnitudes can vary by orders of magnitude.
        canonical = np.array(
            [
                math.log1p(abs(price)) * math.copysign(1.0, price or 1.0),
                math.log1p(abs(volume)),
                self.REGIME_LOOKUP.get(regime_label, 0.0),
                max(0.0, min(1.0, regime_conf)),
                math.tanh(volatility),
                math.tanh(spread * 100.0),
                math.tanh(position),
                math.tanh(pnl),
            ],
            dtype=np.float64,
        )
        features = np.zeros(self.state_dim, dtype=np.float64)
        n = min(self.state_dim, canonical.size)
        features[:n] = canonical[:n]
        return features


class TransitionModel:
    """Linear ``(state, action) -> next_state`` model with ensemble heads.

    Each head is an independent linear projection; the mean prediction is used
    as the point estimate and the variance across heads supplies an epistemic
    uncertainty signal. Parameters are updated with plain SGD.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        ensemble_size: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.ensemble_size = max(1, int(ensemble_size))
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        rng = rng or np.random.default_rng(0)
        in_dim = state_dim + action_dim
        # Initialise close to identity on the state portion so early rollouts
        # are stable before learning kicks in.
        self.heads: List[np.ndarray] = []
        for _ in range(self.ensemble_size):
            w = rng.normal(scale=0.05, size=(in_dim, state_dim))
            w[:state_dim, :state_dim] += np.eye(state_dim)
            self.heads.append(w)
        self.biases: List[np.ndarray] = [
            np.zeros(state_dim, dtype=np.float64) for _ in range(self.ensemble_size)
        ]

    def _input(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.concatenate([state, action])

    def predict(self, state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return ``(mean_next_state, uncertainty)``."""

        x = self._input(state, action)
        preds = np.stack([x @ w + b for w, b in zip(self.heads, self.biases)])
        mean = preds.mean(axis=0)
        uncertainty = float(preds.var(axis=0).mean()) if self.ensemble_size > 1 else 0.0
        return mean, uncertainty

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        target_next: np.ndarray,
        max_update_norm: float = 1.0,
    ) -> float:
        """Do one SGD step on every ensemble head. Return the mean loss."""

        x = self._input(state, action)
        total_loss = 0.0
        for idx in range(self.ensemble_size):
            w = self.heads[idx]
            b = self.biases[idx]
            pred = x @ w + b
            err = pred - target_next
            total_loss += float(0.5 * np.dot(err, err))
            grad_w = np.outer(x, err) + self.weight_decay * w
            grad_b = err
            # Gradient clipping keeps updates bounded when the environment is
            # noisy; this is especially important with just one sample per step.
            g_norm = float(np.linalg.norm(grad_w))
            if g_norm > max_update_norm:
                grad_w *= max_update_norm / max(g_norm, 1e-9)
            self.heads[idx] = w - self.learning_rate * grad_w
            self.biases[idx] = b - self.learning_rate * grad_b
        return total_loss / self.ensemble_size


class RewardModel:
    """Linear ``(state, action) -> scalar reward`` model trained by SGD."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        rng = rng or np.random.default_rng(1)
        self.weights = rng.normal(scale=0.05, size=state_dim + action_dim)
        self.bias = 0.0

    def _input(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.concatenate([state, action])

    def predict(self, state: np.ndarray, action: np.ndarray) -> float:
        x = self._input(state, action)
        return float(x @ self.weights + self.bias)

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        target_reward: float,
        max_update_norm: float = 1.0,
    ) -> float:
        x = self._input(state, action)
        pred = x @ self.weights + self.bias
        err = float(pred - target_reward)
        loss = 0.5 * err * err
        grad_w = err * x + self.weight_decay * self.weights
        g_norm = float(np.linalg.norm(grad_w))
        if g_norm > max_update_norm:
            grad_w *= max_update_norm / max(g_norm, 1e-9)
        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * err
        return float(loss)


# ---------------------------------------------------------------------------
# World model
# ---------------------------------------------------------------------------


class WorldModel:
    """World model coordinating the encoder, transition head, and reward head."""

    def __init__(self, config: Optional[WorldModelConfig] = None) -> None:
        self.config = config or WorldModelConfig()
        rng = np.random.default_rng(self.config.seed)
        self.encoder = StateEncoder(state_dim=self.config.state_dim)
        self.transition = TransitionModel(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            ensemble_size=self.config.ensemble_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            rng=rng,
        )
        self.reward = RewardModel(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            learning_rate=self.config.reward_learning_rate,
            weight_decay=self.config.weight_decay,
            rng=rng,
        )
        self.update_count = 0
        self.last_transition_loss = 0.0
        self.last_reward_loss = 0.0

    # -- API -------------------------------------------------------------

    def encode_state(self, obs: Dict[str, Any]) -> np.ndarray:
        """Encode ``obs`` using the shared :class:`StateEncoder`."""

        return self.encoder.encode(obs)

    def _coerce_action(self, action: Any) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(action, dtype=np.float64))
        if arr.size < self.config.action_dim:
            padded = np.zeros(self.config.action_dim, dtype=np.float64)
            padded[: arr.size] = arr
            return padded
        return arr[: self.config.action_dim]

    def predict_next(
        self, state: np.ndarray, action: Any
    ) -> Tuple[np.ndarray, float, float]:
        """Single-step prediction. Returns ``(next_state, reward, uncertainty)``."""

        a = self._coerce_action(action)
        next_state, uncertainty = self.transition.predict(state, a)
        predicted_reward = self.reward.predict(state, a)
        return next_state, predicted_reward, uncertainty

    def rollout(
        self, state: np.ndarray, actions_sequence: Sequence[Any]
    ) -> RolloutResult:
        """Imagined rollout over a sequence of actions."""

        horizon = len(actions_sequence)
        states = np.zeros((horizon + 1, self.config.state_dim), dtype=np.float64)
        rewards = np.zeros(horizon, dtype=np.float64)
        uncertainty = np.zeros(horizon, dtype=np.float64)
        states[0] = state
        current = state.copy()
        for step, action in enumerate(actions_sequence):
            next_state, r, u = self.predict_next(current, action)
            states[step + 1] = next_state
            rewards[step] = r
            uncertainty[step] = u
            current = next_state
        total_reward = float(rewards.sum())
        mean_u = float(uncertainty.mean()) if horizon > 0 else 0.0
        return RolloutResult(
            states=states,
            rewards=rewards,
            uncertainty=uncertainty,
            total_reward=total_reward,
            mean_uncertainty=mean_u,
        )

    def update(
        self,
        state: np.ndarray,
        action: Any,
        actual_next: np.ndarray,
        actual_reward: float,
    ) -> Dict[str, float]:
        """Train both dynamics and reward heads on one observed transition."""

        a = self._coerce_action(action)
        t_loss = self.transition.update(
            state, a, actual_next, max_update_norm=self.config.max_update_norm
        )
        r_loss = self.reward.update(
            state, a, float(actual_reward), max_update_norm=self.config.max_update_norm
        )
        self.update_count += 1
        self.last_transition_loss = t_loss
        self.last_reward_loss = r_loss
        return {"transition_loss": t_loss, "reward_loss": r_loss}

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary for monitoring."""

        return {
            "config": {
                "state_dim": self.config.state_dim,
                "action_dim": self.config.action_dim,
                "ensemble_size": self.config.ensemble_size,
            },
            "update_count": self.update_count,
            "last_transition_loss": float(self.last_transition_loss),
            "last_reward_loss": float(self.last_reward_loss),
            "transition_head_norm": float(
                np.mean([np.linalg.norm(w) for w in self.transition.heads])
            ),
            "reward_weight_norm": float(np.linalg.norm(self.reward.weights)),
        }
