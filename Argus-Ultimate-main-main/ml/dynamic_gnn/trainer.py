"""Training pipeline for dynamic spatial-temporal GNN models."""

# pyright: reportMissingImports=false

from __future__ import annotations

import logging
import os
import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .dynamic_graph import GraphSnapshot
from .hybrid_model import HybridSpatialTemporalModel

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class TrainerConfig:
    epochs: int = 25
    learning_rate: float = 1e-3
    walk_forward_splits: int = 3
    early_stopping_patience: int = 5
    checkpoint_dir: str = "checkpoints/dynamic_gnn"
    sharpe_loss_weight: float = 0.2
    validation_fraction: float = 0.2
    sequence_length: int = 8


@dataclass(slots=True)
class TrainerResult:
    best_loss: float
    best_epoch: int
    history: dict[str, list[float]] = field(default_factory=dict)
    validation_scores: list[float] = field(default_factory=list)
    checkpoint_path: str | None = None


class DynamicGNNTrainer:
    def __init__(self, config: TrainerConfig | None = None) -> None:
        self.config = config or TrainerConfig()
        self.best_state: dict[str, Any] | None = None

    def train(
        self,
        model: HybridSpatialTemporalModel,
        graph_sequence: Sequence[GraphSnapshot],
        targets: np.ndarray,
        regimes: Sequence[str] | None = None,
    ) -> TrainerResult:
        sequences, y, sequence_regimes = self._make_sequences(graph_sequence, targets, regimes)
        if not sequences:
            raise ValueError("Not enough graph snapshots to create training sequences")

        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        history = {"train_loss": [], "val_loss": [], "sharpe_loss": []}
        best_loss = float("inf")
        best_epoch = 0
        patience = 0
        validation_scores = self.walk_forward_validate(model, sequences, y)
        checkpoint_path: str | None = None

        train_indices, val_indices = self._walk_forward_split(len(sequences))
        stratified_train = self._regime_stratified_indices(sequence_regimes, train_indices)

        for epoch in range(1, self.config.epochs + 1):
            train_loss = 0.0
            sharpe_loss = 0.0
            for idx in stratified_train:
                prediction = self._predict_single(model, sequences[idx])
                target = y[idx]
                loss_value, sharpe_component = self._loss(prediction, target)
                train_loss += loss_value
                sharpe_loss += sharpe_component

            val_loss = 0.0
            for idx in val_indices:
                prediction = self._predict_single(model, sequences[idx])
                target = y[idx]
                loss_value, _ = self._loss(prediction, target)
                val_loss += loss_value

            train_loss /= max(len(stratified_train), 1)
            sharpe_loss /= max(len(stratified_train), 1)
            val_loss /= max(len(val_indices), 1)
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))
            history["sharpe_loss"].append(float(sharpe_loss))

            logger.info(
                "DynamicGNN epoch %d/%d | train=%.6f | val=%.6f | sharpe=%.6f",
                epoch,
                self.config.epochs,
                train_loss,
                val_loss,
                sharpe_loss,
            )

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                patience = 0
                checkpoint_path = self._checkpoint(model, epoch, best_loss)
            else:
                patience += 1
                checkpoint_path = None
                if patience >= self.config.early_stopping_patience:
                    logger.info("Early stopping triggered at epoch %d", epoch)
                    break

        return TrainerResult(
            best_loss=float(best_loss),
            best_epoch=int(best_epoch),
            history=history,
            validation_scores=validation_scores,
            checkpoint_path=checkpoint_path,
        )

    def _make_sequences(
        self,
        graph_sequence: Sequence[GraphSnapshot],
        targets: np.ndarray,
        regimes: Sequence[str] | None = None,
    ) -> tuple[list[Sequence[GraphSnapshot]], np.ndarray, list[str]]:
        target_arr = np.asarray(targets, dtype=np.float32)
        if target_arr.ndim != 3:
            raise ValueError("targets must have shape (timesteps, nodes, output_dim)")
        if len(graph_sequence) != target_arr.shape[0]:
            raise ValueError("graph_sequence length must match targets timesteps")

        seq_len = max(2, self.config.sequence_length)
        sequences: list[Sequence[GraphSnapshot]] = []
        y: list[np.ndarray] = []
        regime_labels: list[str] = []
        for end_idx in range(seq_len, len(graph_sequence)):
            sequences.append(graph_sequence[end_idx - seq_len:end_idx])
            y.append(target_arr[end_idx])
            regime_labels.append(str(regimes[end_idx] if regimes is not None else "unknown"))
        return sequences, np.asarray(y, dtype=np.float32), regime_labels

    def _walk_forward_split(self, n_sequences: int) -> tuple[list[int], list[int]]:
        split_point = max(1, int(n_sequences * (1.0 - self.config.validation_fraction)))
        return list(range(split_point)), list(range(split_point, n_sequences))

    def walk_forward_validate(
        self,
        model: HybridSpatialTemporalModel,
        sequences: Sequence[Sequence[GraphSnapshot]],
        targets: np.ndarray,
    ) -> list[float]:
        n_sequences = len(sequences)
        if n_sequences <= 1:
            return []
        n_splits = max(1, min(self.config.walk_forward_splits, n_sequences - 1))
        split_edges = np.linspace(1, n_sequences, num=n_splits + 1, dtype=int)
        scores: list[float] = []
        for split_idx in range(1, len(split_edges)):
            start = split_edges[split_idx - 1]
            end = split_edges[split_idx]
            fold_losses = []
            for seq_idx in range(start, end):
                prediction = self._predict_single(model, sequences[seq_idx - 1])
                loss_value, _ = self._loss(prediction, targets[seq_idx - 1])
                fold_losses.append(loss_value)
            if fold_losses:
                scores.append(float(-np.mean(fold_losses)))
        return scores

    @staticmethod
    def _regime_stratified_indices(regimes: Sequence[str], candidate_indices: Sequence[int]) -> list[int]:
        buckets: dict[str, list[int]] = {}
        for idx in candidate_indices:
            buckets.setdefault(str(regimes[idx]), []).append(int(idx))
        merged: list[int] = []
        max_bucket = max((len(indices) for indices in buckets.values()), default=0)
        for offset in range(max_bucket):
            for indices in buckets.values():
                if offset < len(indices):
                    merged.append(indices[offset])
        return merged

    @staticmethod
    def _loss(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
        pred = np.asarray(prediction, dtype=np.float32)
        tgt = np.asarray(target, dtype=np.float32)
        mse = float(np.mean((pred - tgt) ** 2))
        portfolio_returns = pred[:, 0] * tgt[:, 0]
        sharpe = float(np.mean(portfolio_returns) / max(np.std(portfolio_returns), 1e-6))
        sharpe_loss = -sharpe
        return mse + 0.2 * sharpe_loss, sharpe_loss

    @staticmethod
    def mse_loss(prediction: np.ndarray, target: np.ndarray) -> float:
        pred = np.asarray(prediction, dtype=np.float32)
        tgt = np.asarray(target, dtype=np.float32)
        return float(np.mean((pred - tgt) ** 2))

    @staticmethod
    def sharpe_objective(prediction: np.ndarray, target: np.ndarray) -> float:
        pred = np.asarray(prediction, dtype=np.float32)
        tgt = np.asarray(target, dtype=np.float32)
        portfolio_returns = pred[:, 0] * tgt[:, 0]
        return float(np.mean(portfolio_returns) / max(np.std(portfolio_returns), 1e-6))

    @staticmethod
    def _predict_single(model: HybridSpatialTemporalModel, sequence: Sequence[GraphSnapshot]) -> np.ndarray:
        predictions, _ = model.forward(sequence, return_attention=False)
        return np.asarray(predictions, dtype=np.float32)

    def _checkpoint(self, model: HybridSpatialTemporalModel, epoch: int, loss_value: float) -> str:
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"dynamic_gnn_epoch_{epoch}.pkl")
        payload = {
            "epoch": int(epoch),
            "loss": float(loss_value),
            "model": model,
            "config": self.config,
        }
        with open(checkpoint_path, "wb") as handle:
            pickle.dump(payload, handle)
        self.best_state = payload
        return checkpoint_path
