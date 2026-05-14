"""Transformer-based twin-critic implementation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class CriticConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    regime_dim: int = 4
    tau: float = 0.005


class _SingleCritic(nn.Module):
    def __init__(self, config: CriticConfig) -> None:
        super().__init__()
        self.state_proj = nn.Linear(config.state_dim, config.hidden_dim)
        self.action_proj = nn.Linear(config.action_dim, config.hidden_dim)
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
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, state_history: Tensor, action: Tensor, regime_features: Optional[Tensor]) -> Tensor:
        state_repr = self.state_proj(state_history)
        action_repr = self.action_proj(action).unsqueeze(1)
        x = state_repr + action_repr
        if regime_features is not None:
            x = x + self.regime_proj(regime_features).unsqueeze(1)
        encoded = self.encoder(x)
        return self.head(encoded[:, -1, :])


class TransformerCritic(nn.Module):
    def __init__(self, config: CriticConfig) -> None:
        super().__init__()
        self.config = config
        self.q1 = _SingleCritic(config)
        self.q2 = _SingleCritic(config)
        self.target_q1 = copy.deepcopy(self.q1)
        self.target_q2 = copy.deepcopy(self.q2)
        for module in (self.target_q1, self.target_q2):
            for param in module.parameters():
                param.requires_grad_(False)

    def forward(self, state_history: Tensor, action: Tensor, regime_features: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        return self.q1(state_history, action, regime_features), self.q2(state_history, action, regime_features)

    def target(self, state_history: Tensor, action: Tensor, regime_features: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        return self.target_q1(state_history, action, regime_features), self.target_q2(state_history, action, regime_features)

    def soft_update(self, tau: Optional[float] = None) -> None:
        tau = float(self.config.tau if tau is None else tau)
        with torch.no_grad():
            for online, target in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
                for source_param, target_param in zip(online.parameters(), target.parameters()):
                    target_param.data.mul_(1.0 - tau).add_(tau * source_param.data)
