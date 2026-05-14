"""Transformer-based regime-conditioned actor network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.distributions import Normal


@dataclass(slots=True)
class ActorConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    max_action: float = 1.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    regime_dim: int = 4
    entropy_bonus: float = 0.2


class TransformerActor(nn.Module):
    def __init__(self, config: ActorConfig) -> None:
        super().__init__()
        self.config = config
        self.state_proj = nn.Linear(config.state_dim, config.hidden_dim)
        self.regime_proj = nn.Linear(config.regime_dim, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.mean_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.log_std_head = nn.Linear(config.hidden_dim, config.action_dim)

    def forward(self, state_history: Tensor, regime_features: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        x = self.state_proj(state_history)
        if regime_features is not None:
            regime_emb = self.regime_proj(regime_features).unsqueeze(1)
            x = x + regime_emb
        encoded = self.encoder(x)
        pooled = self.norm(encoded[:, -1, :])
        mean = torch.tanh(self.mean_head(pooled)) * self.config.max_action
        log_std = self.log_std_head(pooled).clamp(self.config.log_std_min, self.config.log_std_max)
        return mean, log_std

    def distribution(self, state_history: Tensor, regime_features: Optional[Tensor] = None) -> Normal:
        mean, log_std = self.forward(state_history, regime_features)
        return Normal(mean, log_std.exp())

    def sample_action(
        self,
        state_history: Tensor,
        regime_features: Optional[Tensor] = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        dist = self.distribution(state_history, regime_features)
        raw_action = dist.mean if deterministic else dist.rsample()
        squashed = torch.tanh(raw_action) * self.config.max_action
        log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        log_prob = log_prob - torch.log(1 - torch.tanh(raw_action).pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return squashed, log_prob, entropy
