"""Decision heads for Argus-AI.

Four specialised output heads:
  DirectionHead   — long / short / flat classification (3-class softmax)
  SizeHead        — position size as Beta distribution parameters (alpha, beta)
  TimingHead      — entry delay in ticks (regression, Huber loss)
  ConfidenceHead  — calibrated uncertainty via MC Dropout ensemble
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DIRECTION_CLASSES = 3  # 0=flat, 1=long, 2=short


class DirectionOutput(NamedTuple):
    logits: torch.Tensor       # (B, 3)
    probs: torch.Tensor        # (B, 3)
    action: torch.Tensor       # (B,) argmax


class SizeOutput(NamedTuple):
    alpha: torch.Tensor        # (B, 1) Beta alpha > 0
    beta: torch.Tensor         # (B, 1) Beta beta  > 0
    mean_size: torch.Tensor    # (B, 1) E[Beta] = alpha / (alpha + beta)


class TimingOutput(NamedTuple):
    delay_ticks: torch.Tensor  # (B, 1) predicted entry delay


class ConfidenceOutput(NamedTuple):
    mean: torch.Tensor         # (B, 1) mean confidence
    std: torch.Tensor          # (B, 1) uncertainty estimate
    raw_samples: torch.Tensor  # (B, mc_samples) MC dropout samples


class DirectionHead(nn.Module):
    """Long / short / flat classifier.

    Input: (B, d_model) last-token representation from backbone.
    """

    def __init__(self, d_model: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, DIRECTION_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> DirectionOutput:
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        action = logits.argmax(dim=-1)
        return DirectionOutput(logits=logits, probs=probs, action=action)


class SizeHead(nn.Module):
    """Position sizing via Beta distribution parameters.

    Outputs alpha and beta > 0 via softplus, so position size ~ Beta(alpha, beta).
    Mean position size = alpha / (alpha + beta), bounded in (0, 1).
    """

    def __init__(self, d_model: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> SizeOutput:
        raw = self.net(x)
        alpha = F.softplus(raw[:, 0:1]) + 1e-4
        beta = F.softplus(raw[:, 1:2]) + 1e-4
        mean_size = alpha / (alpha + beta)
        return SizeOutput(alpha=alpha, beta=beta, mean_size=mean_size)


class TimingHead(nn.Module):
    """Entry delay predictor (ticks until optimal entry).

    Outputs a non-negative delay via softplus. Trained with Huber loss.
    """

    def __init__(self, d_model: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> TimingOutput:
        raw = self.net(x)
        delay = F.softplus(raw)
        return TimingOutput(delay_ticks=delay)


class ConfidenceHead(nn.Module):
    """Calibrated uncertainty estimation via MC Dropout.

    Runs `mc_samples` stochastic forward passes with dropout enabled,
    returning mean and std of confidence scores.
    """

    def __init__(self, d_model: int = 512, dropout: float = 0.2, mc_samples: int = 20) -> None:
        super().__init__()
        self.mc_samples = mc_samples
        self.net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> ConfidenceOutput:
        was_training = self.training
        self.train()
        samples = torch.stack([self.net(x) for _ in range(self.mc_samples)], dim=1).squeeze(-1)
        if not was_training:
            self.eval()
        mean_conf = samples.mean(dim=1, keepdim=True)
        std_conf = samples.std(dim=1, keepdim=True)
        return ConfidenceOutput(mean=mean_conf, std=std_conf, raw_samples=samples)
