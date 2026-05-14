"""Soft Actor-Critic agent with twin critics and entropy tuning."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .replay_buffer import PrioritizedReplayBuffer, ReplayBatch, ReplayBufferConfig
from .volatility_adapter import VolatilityAdapter, VolatilityAdapterConfig

logger = logging.getLogger(__name__)

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass(slots=True)
class SACConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 128
    buffer_capacity: int = 100_000
    warmup_steps: int = 1_000
    target_entropy: float | None = None
    device: str = "cpu"


@dataclass(slots=True)
class SACUpdateMetrics:
    actor_loss: float
    critic_loss: float
    alpha_loss: float
    alpha: float
    q_value: float


class _PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(state)
        mean = self.mean_head(hidden)
        log_std = torch.clamp(self.log_std_head(hidden), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(state)
        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros((state.shape[0], 1), device=state.device)
            return action, log_prob
        std = log_std.exp()
        distribution = torch.distributions.Normal(mean, std)
        raw_action = distribution.rsample()
        squashed = torch.tanh(raw_action)
        correction = torch.log(1.0 - squashed.pow(2) + 1e-6)
        log_prob = distribution.log_prob(raw_action).sum(dim=-1, keepdim=True) - correction.sum(dim=-1, keepdim=True)
        return squashed, log_prob


class _QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([state, action], dim=-1))


class SACAgent:
    def __init__(
        self,
        config: SACConfig,
        replay_config: ReplayBufferConfig | None = None,
        adapter_config: VolatilityAdapterConfig | None = None,
    ) -> None:
        if config.state_dim <= 0 or config.action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")
        self.config = config
        self.device = torch.device(config.device)
        self.policy = _PolicyNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.q1 = _QNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.q2 = _QNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.target_q1 = _QNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.target_q2 = _QNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.actor_lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=config.critic_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=config.critic_lr)
        self.log_alpha = torch.tensor(np.log(0.2), dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.target_entropy = float(config.target_entropy if config.target_entropy is not None else -config.action_dim)
        replay_cfg = replay_config or ReplayBufferConfig(capacity=config.buffer_capacity)
        self.replay_buffer = PrioritizedReplayBuffer(config.state_dim, config.action_dim, replay_cfg)
        self.adapter = VolatilityAdapter(adapter_config)
        self.total_steps = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(
        self,
        state: np.ndarray,
        volatility: float = 0.0,
        regime: str = "medium",
        deterministic: bool = False,
    ) -> np.ndarray:
        state_tensor = self._state_tensor(state)
        with torch.no_grad():
            action, _ = self.policy.sample(state_tensor, deterministic=deterministic)
        adapted = self.adapter.adapt_position(action[0].cpu().numpy(), volatility=volatility, regime=regime)
        return adapted.astype(np.float32)

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        regime: str,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done, regime)
        self.total_steps += 1

    def update(self) -> SACUpdateMetrics | None:
        if len(self.replay_buffer) < max(self.config.batch_size, self.config.warmup_steps):
            return None
        batch = self.replay_buffer.sample(self.config.batch_size, self.device)
        return self._update_from_batch(batch)

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": asdict(self.config),
                "policy": self.policy.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "target_q1": self.target_q1.state_dict(),
                "target_q2": self.target_q2.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.target_q1.load_state_dict(checkpoint["target_q1"])
        self.target_q2.load_state_dict(checkpoint["target_q2"])
        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))

    def _update_from_batch(self, batch: ReplayBatch) -> SACUpdateMetrics:
        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample(batch.next_states)
            target_q1 = self.target_q1(batch.next_states, next_action)
            target_q2 = self.target_q2(batch.next_states, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            q_target = batch.rewards + (1.0 - batch.dones) * self.config.gamma * target_q

        current_q1 = self.q1(batch.states, batch.actions)
        current_q2 = self.q2(batch.states, batch.actions)
        q1_loss = (batch.weights * F.mse_loss(current_q1, q_target, reduction="none")).mean()
        q2_loss = (batch.weights * F.mse_loss(current_q2, q_target, reduction="none")).mean()
        critic_loss = q1_loss + q2_loss

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        critic_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        sampled_action, log_prob = self.policy.sample(batch.states)
        q1_pi = self.q1(batch.states, sampled_action)
        q2_pi = self.q2(batch.states, sampled_action)
        actor_loss = (self.alpha.detach() * log_prob - torch.min(q1_pi, q2_pi)).mean()

        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        td_errors = (q_target - current_q1).detach().abs().cpu().numpy().reshape(-1)
        self.replay_buffer.update_priorities(batch.indices, td_errors)
        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)

        return SACUpdateMetrics(
            actor_loss=float(actor_loss.item()),
            critic_loss=float(critic_loss.item()),
            alpha_loss=float(alpha_loss.item()),
            alpha=float(self.alpha.detach().item()),
            q_value=float(torch.min(q1_pi, q2_pi).mean().detach().item()),
        )

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
            target_param.data.mul_(1.0 - self.config.tau)
            target_param.data.add_(self.config.tau * source_param.data)

    def _state_tensor(self, state: np.ndarray) -> torch.Tensor:
        array = np.asarray(state, dtype=np.float32).reshape(1, self.config.state_dim)
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)
