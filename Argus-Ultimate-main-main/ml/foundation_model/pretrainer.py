"""Self-supervised pretraining utilities for the microstructure foundation model."""

# pyright: reportMissingImports=false

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .model import TradeFoundationModel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainingConfig:
    """Training settings for TradeFM-style self-supervision."""

    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 16
    epochs: int = 5
    mask_probability: float = 0.15
    next_event_weight: float = 1.0
    masked_event_weight: float = 0.5
    contrastive_weight: float = 0.25
    contrastive_temperature: float = 0.1
    max_grad_norm: float = 1.0
    mask_token_id: int = 0
    device: str = "cpu"
    log_every: int = 50
    min_sequence_length: int = 2


class FoundationModelPretrainer:
    """Combines next-event, masked-event, and contrastive order-flow losses."""

    def __init__(self, model: TradeFoundationModel, config: Optional[TrainingConfig] = None) -> None:
        self.model = model
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def collate_batch(self, batch: Iterable[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        items = list(batch)
        if not items:
            raise ValueError("batch is empty")

        sequences = [item["input_ids"] for item in items if isinstance(item.get("input_ids"), torch.Tensor)]
        if not sequences:
            raise ValueError("batch must contain tensor input_ids")

        padded = pad_sequence(sequences, batch_first=True, padding_value=self.config.mask_token_id)
        attention_mask = (padded != self.config.mask_token_id).long()
        asset_ids = [int(item.get("asset_ids", 0)) for item in items]
        return {
            "input_ids": padded,
            "attention_mask": attention_mask,
            "asset_ids": torch.tensor(asset_ids, dtype=torch.long),
        }

    def _move_batch(self, batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
        if "input_ids" not in result:
            raise ValueError("Batch must include input_ids")
        if result["input_ids"].ndim != 2:
            raise ValueError("input_ids must be rank-2 after collation")
        if result["input_ids"].size(1) < self.config.min_sequence_length:
            raise ValueError("input_ids sequence length is too short for next-event prediction")
        return result

    def mask_inputs(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = torch.full_like(input_ids, self.config.mask_probability, dtype=torch.float32)
        mask = torch.bernoulli(probabilities).bool()
        masked = input_ids.clone()
        masked[mask] = self.config.mask_token_id
        return masked, mask

    def next_event_loss(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), target_ids[:, 1:].reshape(-1))

    def masked_event_loss(self, masked_logits: torch.Tensor, target_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() == 0:
            return masked_logits.new_tensor(0.0)
        flat_logits = masked_logits.reshape(-1, masked_logits.size(-1))
        flat_targets = target_ids.reshape(-1)
        flat_mask = mask.reshape(-1)
        return F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])

    def contrastive_loss(self, representations: torch.Tensor, asset_ids: Optional[torch.Tensor]) -> torch.Tensor:
        if asset_ids is None or representations.size(0) < 2:
            return representations.new_tensor(0.0)
        normalized = F.normalize(representations, dim=-1)
        logits = torch.matmul(normalized, normalized.T) / max(self.config.contrastive_temperature, 1e-6)
        logits = logits - torch.eye(logits.size(0), device=logits.device) * 1e9
        positive_mask = asset_ids[:, None] == asset_ids[None, :]
        positive_mask.fill_diagonal_(False)
        valid_rows = positive_mask.any(dim=1)
        if not valid_rows.any():
            return representations.new_tensor(0.0)
        log_probs = F.log_softmax(logits, dim=-1)
        positive_log_probs = (log_probs * positive_mask.float()).sum(dim=-1) / positive_mask.float().sum(dim=-1).clamp_min(1.0)
        return -positive_log_probs[valid_rows].mean()

    def training_step(self, batch: Mapping[str, Any]) -> Dict[str, float]:
        batch_tensors = self._move_batch(batch)
        input_ids = batch_tensors["input_ids"]
        attention_mask = batch_tensors.get("attention_mask")
        asset_ids = batch_tensors.get("asset_ids")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        next_output = self.model(input_ids, attention_mask)
        next_loss = self.next_event_loss(next_output["logits"], input_ids)

        masked_input_ids, mask = self.mask_inputs(input_ids)
        masked_output = self.model(masked_input_ids, attention_mask)
        masked_loss = self.masked_event_loss(masked_output["logits"], input_ids, mask)

        pooled_repr = next_output["hidden_states"].mean(dim=1)
        contrastive = self.contrastive_loss(pooled_repr, asset_ids)

        total_loss = (
            self.config.next_event_weight * next_loss
            + self.config.masked_event_weight * masked_loss
            + self.config.contrastive_weight * contrastive
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": float(total_loss.detach().cpu().item()),
            "next_event_loss": float(next_loss.detach().cpu().item()),
            "masked_event_loss": float(masked_loss.detach().cpu().item()),
            "contrastive_loss": float(contrastive.detach().cpu().item()),
        }

    def fit(self, train_loader: Iterable[Mapping[str, Any]]) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []
        global_step = 0
        for epoch in range(self.config.epochs):
            for batch in train_loader:
                metrics = self.training_step(batch)
                metrics["epoch"] = float(epoch)
                history.append(metrics)
                global_step += 1
                if global_step % self.config.log_every == 0:
                    logger.info(
                        "Pretraining step=%d loss=%.4f next=%.4f masked=%.4f contrastive=%.4f",
                        global_step,
                        metrics["loss"],
                        metrics["next_event_loss"],
                        metrics["masked_event_loss"],
                        metrics["contrastive_loss"],
                    )
        return history

    @torch.no_grad()
    def evaluate_batch(self, batch: Mapping[str, Any]) -> Dict[str, float]:
        batch_tensors = self._move_batch(batch)
        input_ids = batch_tensors["input_ids"]
        attention_mask = batch_tensors.get("attention_mask")
        asset_ids = batch_tensors.get("asset_ids")

        self.model.eval()
        next_output = self.model(input_ids, attention_mask)
        next_loss = self.next_event_loss(next_output["logits"], input_ids)
        masked_input_ids, mask = self.mask_inputs(input_ids)
        masked_output = self.model(masked_input_ids, attention_mask)
        masked_loss = self.masked_event_loss(masked_output["logits"], input_ids, mask)
        pooled_repr = next_output["hidden_states"].mean(dim=1)
        contrastive = self.contrastive_loss(pooled_repr, asset_ids)
        total_loss = (
            self.config.next_event_weight * next_loss
            + self.config.masked_event_weight * masked_loss
            + self.config.contrastive_weight * contrastive
        )
        return {
            "loss": float(total_loss.detach().cpu().item()),
            "next_event_loss": float(next_loss.detach().cpu().item()),
            "masked_event_loss": float(masked_loss.detach().cpu().item()),
            "contrastive_loss": float(contrastive.detach().cpu().item()),
        }
