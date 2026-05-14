"""ArgusAITrainer — offline pre-training loop for ArgusAI.

Features:
  - Multi-task loss: direction CE + size MSE + timing Huber + confidence NLL
  - AdamW + cosine LR warmup schedule
  - Mixed-precision training (torch.cuda.amp)
  - Optuna integration hook for HPO
  - Checkpoint save/load with model state + optimizer state
  - Early stopping on validation Sharpe
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    total_steps: int = 50000
    batch_size: int = 64
    grad_clip: float = 1.0
    direction_weight: float = 1.0
    size_weight: float = 0.5
    timing_weight: float = 0.3
    confidence_weight: float = 0.2
    mixed_precision: bool = True
    checkpoint_dir: str = "checkpoints/argus_ai"
    save_every_n_steps: int = 5000
    eval_every_n_steps: int = 1000
    early_stop_patience: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainMetrics:
    step: int = 0
    train_loss: float = 0.0
    direction_loss: float = 0.0
    size_loss: float = 0.0
    timing_loss: float = 0.0
    confidence_loss: float = 0.0
    val_sharpe: float = 0.0
    lr: float = 0.0
    elapsed_s: float = 0.0
    history: List[Dict[str, float]] = field(default_factory=list)


class ArgusAITrainer:
    """Offline pre-trainer for ArgusAI.

    Args:
        model:   ArgusAI instance.
        config:  TrainerConfig.
    """

    def __init__(self, model: nn.Module, config: Optional[TrainerConfig] = None) -> None:
        self.model = model
        self.config = config or TrainerConfig()
        self.device = torch.device(self.config.device)
        self.model.to(self.device)

        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self._warmup_cosine)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.mixed_precision)
        self.metrics = TrainMetrics()
        self._best_val_sharpe = -float("inf")
        self._patience_counter = 0
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def _warmup_cosine(self, step: int) -> float:
        if step < self.config.warmup_steps:
            return step / max(1, self.config.warmup_steps)
        progress = (step - self.config.warmup_steps) / max(1, self.config.total_steps - self.config.warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def compute_loss(
        self,
        output: Any,
        direction_labels: torch.Tensor,
        size_targets: torch.Tensor,
        timing_targets: torch.Tensor,
        confidence_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-task training loss.

        Args:
            output:              ArgusAIOutput.
            direction_labels:    (B,) long — 0/1/2.
            size_targets:        (B, 1) float in [0, 1].
            timing_targets:      (B, 1) float ticks.
            confidence_targets:  (B, 1) float in [0, 1].

        Returns:
            (total_loss, component_dict)
        """
        c = self.config
        direction_loss = F.cross_entropy(output.direction_logits, direction_labels)
        size_loss = F.mse_loss(output.size_mean, size_targets)
        timing_loss = F.huber_loss(output.timing_delay, timing_targets)
        conf_loss = F.binary_cross_entropy(output.confidence_mean, confidence_targets)

        total = (
            c.direction_weight * direction_loss
            + c.size_weight * size_loss
            + c.timing_weight * timing_loss
            + c.confidence_weight * conf_loss
        )
        components = {
            "direction": float(direction_loss.item()),
            "size": float(size_loss.item()),
            "timing": float(timing_loss.item()),
            "confidence": float(conf_loss.item()),
            "total": float(total.item()),
        }
        return total, components

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
            regime_ids = batch["regime_ids"].to(self.device)
            output = self.model(
                regime_ids=regime_ids,
                lob=batch.get("lob", None) and batch["lob"].to(self.device),
                chart=batch.get("chart", None) and batch["chart"].to(self.device),
                sentiment=batch.get("sentiment", None) and batch["sentiment"].to(self.device),
                gnn=batch.get("gnn", None) and batch["gnn"].to(self.device),
                regime_vec=batch.get("regime_vec", None) and batch["regime_vec"].to(self.device),
            )
            loss, components = self.compute_loss(
                output=output,
                direction_labels=batch["direction_labels"].to(self.device),
                size_targets=batch["size_targets"].to(self.device),
                timing_targets=batch["timing_targets"].to(self.device),
                confidence_targets=batch["confidence_targets"].to(self.device),
            )

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.metrics.step += 1
        return components

    def save_checkpoint(self, tag: str = "latest") -> str:
        path = os.path.join(self.config.checkpoint_dir, f"argus_ai_{tag}.pt")
        torch.save({
            "step": self.metrics.step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_sharpe": self._best_val_sharpe,
            "config": self.config,
        }, path)
        logger.info("ArgusAI checkpoint saved → %s", path)
        return path

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self._best_val_sharpe = ckpt.get("best_val_sharpe", -float("inf"))
        self.metrics.step = ckpt["step"]
        logger.info("ArgusAI checkpoint loaded ← %s (step %d)", path, self.metrics.step)
        return self.metrics.step

    def optuna_objective(self, trial: Any, train_loader: DataLoader, val_fn: Callable) -> float:
        """Optuna integration — suggest hyperparams, train N steps, return val Sharpe."""
        self.config.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        self.config.direction_weight = trial.suggest_float("direction_weight", 0.5, 2.0)
        self.config.size_weight = trial.suggest_float("size_weight", 0.1, 1.0)
        self.config.timing_weight = trial.suggest_float("timing_weight", 0.1, 1.0)
        for step, batch in enumerate(train_loader):
            if step >= 500:
                break
            self.train_step(batch)
        return val_fn(self.model)
