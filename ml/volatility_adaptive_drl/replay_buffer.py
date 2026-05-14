"""Prioritized replay buffer with regime-balanced sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(slots=True)
class ReplayBufferConfig:
    capacity: int = 100_000
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_increment: float = 1e-4
    epsilon: float = 1e-5
    regime_balance_fraction: float = 0.5


@dataclass(slots=True)
class ReplayBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    weights: torch.Tensor
    indices: np.ndarray
    regimes: list[str]


class PrioritizedReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, config: ReplayBufferConfig | None = None) -> None:
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")
        self.config = config or ReplayBufferConfig()
        self.capacity = int(self.config.capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.priorities = np.ones(self.capacity, dtype=np.float32)
        self.regimes = np.full(self.capacity, "medium", dtype=object)
        self.position = 0
        self.size = 0
        self.beta = float(self.config.beta_start)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        regime: str,
    ) -> None:
        idx = self.position
        self.states[idx] = np.asarray(state, dtype=np.float32).reshape(self.state_dim)
        self.actions[idx] = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        self.rewards[idx, 0] = float(reward)
        self.next_states[idx] = np.asarray(next_state, dtype=np.float32).reshape(self.state_dim)
        self.dones[idx, 0] = float(done)
        self.regimes[idx] = str(regime)
        self.priorities[idx] = float(self.priorities[: max(self.size, 1)].max())
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        batch_size = min(int(batch_size), self.size)
        priority_indices, probabilities = self._priority_sample_indices(batch_size=batch_size)
        indices = self._blend_with_regime_balanced_indices(priority_indices=priority_indices, batch_size=batch_size)
        sample_probabilities = np.clip(probabilities[indices], self.config.epsilon, None)
        weights = (self.size * sample_probabilities) ** (-self.beta)
        weights /= max(float(weights.max()), 1e-8)
        self.beta = min(1.0, self.beta + self.config.beta_increment)
        return ReplayBatch(
            states=torch.as_tensor(self.states[indices], dtype=torch.float32, device=device),
            actions=torch.as_tensor(self.actions[indices], dtype=torch.float32, device=device),
            rewards=torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device),
            next_states=torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=device),
            dones=torch.as_tensor(self.dones[indices], dtype=torch.float32, device=device),
            weights=torch.as_tensor(weights[:, None], dtype=torch.float32, device=device),
            indices=indices,
            regimes=[str(self.regimes[idx]) for idx in indices],
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, error in zip(indices, np.asarray(td_errors, dtype=np.float32).reshape(-1), strict=False):
            self.priorities[int(idx)] = float(abs(error) + self.config.epsilon)

    def _priority_sample_indices(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        priorities = np.power(self.priorities[: self.size], self.config.alpha, dtype=np.float32)
        probabilities = priorities / max(float(priorities.sum()), 1e-8)
        indices = np.random.choice(self.size, size=batch_size, replace=False, p=probabilities)
        return indices.astype(np.int64), probabilities.astype(np.float32)

    def _blend_with_regime_balanced_indices(self, priority_indices: np.ndarray, batch_size: int) -> np.ndarray:
        balance_count = int(round(batch_size * self.config.regime_balance_fraction))
        if balance_count <= 0:
            return priority_indices
        regime_buckets: dict[str, list[int]] = {}
        for idx in range(self.size):
            regime_buckets.setdefault(str(self.regimes[idx]), []).append(idx)
        if len(regime_buckets) <= 1:
            return priority_indices
        balanced: list[int] = []
        per_regime = max(1, balance_count // len(regime_buckets))
        for regime_indices in regime_buckets.values():
            take = min(len(regime_indices), per_regime)
            balanced.extend(np.random.choice(regime_indices, size=take, replace=False).tolist())
        combined = list(dict.fromkeys(priority_indices.tolist() + balanced))
        if len(combined) < batch_size:
            for idx in priority_indices.tolist():
                if idx not in combined:
                    combined.append(idx)
                if len(combined) >= batch_size:
                    break
        return np.asarray(combined[:batch_size], dtype=np.int64)
