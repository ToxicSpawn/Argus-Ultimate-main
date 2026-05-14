"""RLTuner — PPO online fine-tuner for ArgusAI.

Reward function:
  r = risk_adjusted_pnl - lambda_dd * max_drawdown_penalty

Safety gates:
  - No updates during CRISIS regime
  - No updates when drawdown > max_dd_gate
  - Update frequency controlled by update_every_n_steps

The tuner maintains a rolling experience buffer and runs a single PPO
clipped update step when triggered.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

CRISIS_REGIME_ID = 3


@dataclass
class Experience:
    state_repr: torch.Tensor   # (d_model,) backbone last-token
    action: int
    reward: float
    log_prob: float
    value: float
    done: bool
    regime_id: int


class ValueHead(nn.Module):
    """Critic value head for PPO."""

    def __init__(self, d_model: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RLTuner:
    """PPO online fine-tuner for ArgusAI.

    Args:
        model:             The ArgusAI model to tune.
        lr:                Learning rate for PPO update (default 3e-5).
        clip_ratio:        PPO clip epsilon (default 0.2).
        entropy_coef:      Entropy bonus coefficient (default 0.01).
        value_coef:        Value loss coefficient (default 0.5).
        gamma:             Discount factor (default 0.99).
        lambda_dd:         Drawdown penalty coefficient (default 2.0).
        max_dd_gate:       Maximum drawdown before updates are blocked (default 0.15).
        update_every:      Steps between PPO updates (default 64).
        buffer_size:       Experience buffer size (default 256).
        device:            Torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-5,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        gamma: float = 0.99,
        lambda_dd: float = 2.0,
        max_dd_gate: float = 0.15,
        update_every: int = 64,
        buffer_size: int = 256,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.gamma = gamma
        self.lambda_dd = lambda_dd
        self.max_dd_gate = max_dd_gate
        self.update_every = update_every
        self.device = torch.device(device)

        self.value_head = ValueHead(d_model=512).to(self.device)
        self.optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(self.value_head.parameters()),
            lr=lr,
            weight_decay=1e-4,
        )
        self.buffer: deque[Experience] = deque(maxlen=buffer_size)
        self._step_count = 0
        self._total_updates = 0
        self._last_loss: Optional[float] = None

    def push(
        self,
        state_repr: torch.Tensor,
        action: int,
        reward: float,
        log_prob: float,
        regime_id: int,
        done: bool = False,
        current_drawdown: float = 0.0,
    ) -> None:
        """Push one experience into the buffer and optionally trigger an update."""
        adjusted_reward = reward - self.lambda_dd * max(0.0, current_drawdown)
        value = float(self.value_head(state_repr.unsqueeze(0).to(self.device)).item())
        self.buffer.append(Experience(
            state_repr=state_repr.detach().cpu(),
            action=action,
            reward=adjusted_reward,
            log_prob=log_prob,
            value=value,
            done=done,
            regime_id=regime_id,
        ))
        self._step_count += 1
        if self._step_count % self.update_every == 0:
            self.update(current_drawdown=current_drawdown, regime_id=regime_id)

    def update(self, current_drawdown: float = 0.0, regime_id: int = 0) -> Optional[float]:
        """Run one PPO clipped update step.

        Returns the total loss float, or None if update was skipped.
        """
        if regime_id == CRISIS_REGIME_ID:
            logger.debug("RLTuner: skipping update — CRISIS regime")
            return None
        if current_drawdown > self.max_dd_gate:
            logger.debug("RLTuner: skipping update — drawdown=%.4f > gate=%.4f", current_drawdown, self.max_dd_gate)
            return None
        if len(self.buffer) < self.update_every:
            return None

        experiences = list(self.buffer)
        states = torch.stack([e.state_repr for e in experiences]).to(self.device)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([e.log_prob for e in experiences], dtype=torch.float32, device=self.device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32, device=self.device)
        old_values = torch.tensor([e.value for e in experiences], dtype=torch.float32, device=self.device)

        returns = self._compute_returns(rewards)
        advantages = (returns - old_values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        new_logits = self.model.direction_head(states).logits
        new_log_probs = F.log_softmax(new_logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
        entropy = -(F.softmax(new_logits, dim=-1) * F.log_softmax(new_logits, dim=-1)).sum(-1).mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        new_values = self.value_head(states).squeeze(1)
        value_loss = F.mse_loss(new_values, returns)

        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._total_updates += 1
        self._last_loss = float(total_loss.item())
        logger.debug(
            "RLTuner update #%d: policy_loss=%.4f value_loss=%.4f entropy=%.4f total=%.4f",
            self._total_updates, float(policy_loss), float(value_loss), float(entropy), self._last_loss,
        )
        return self._last_loss

    def _compute_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        returns = torch.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = float(rewards[t].item()) + self.gamma * running
            returns[t] = running
        return returns

    @property
    def total_updates(self) -> int:
        return self._total_updates

    @property
    def last_loss(self) -> Optional[float]:
        return self._last_loss
