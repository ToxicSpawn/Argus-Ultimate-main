"""
Multi-task learning framework for ARGUS trading models.

This module provides a shared-backbone learner with task-specific heads for
common trading tasks such as price direction, return prediction, volatility
forecasting, and regime classification. When PyTorch is available, the learner
uses a compact neural network backbone with optional uncertainty-based loss
balancing, gradient surgery, and task-specific fine-tuning. When PyTorch is not
available, it falls back to a NumPy implementation built on a shared latent
projection and per-task linear heads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_SUPPORTED_TASKS = {
    "price_direction": "classification",
    "return_prediction": "regression",
    "volatility_prediction": "regression",
    "regime_classification": "classification",
}

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


@dataclass
class TaskConfig:
    task_name: str
    task_type: str
    loss_weight: float
    output_dim: int

    def __post_init__(self) -> None:
        if self.task_type not in {"regression", "classification"}:
            raise ValueError(f"Unsupported task_type: {self.task_type}")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.loss_weight <= 0:
            raise ValueError("loss_weight must be positive")
        expected = _DEFAULT_SUPPORTED_TASKS.get(self.task_name)
        if expected is not None and expected != self.task_type:
            raise ValueError(
                f"Task '{self.task_name}' should use task_type '{expected}', got '{self.task_type}'"
            )


@dataclass
class MultiTaskConfig:
    tasks: List[TaskConfig] = field(default_factory=list)
    shared_hidden_dim: int = 128
    num_shared_layers: int = 3
    learning_rate: float = 0.001
    use_uncertainty_weighting: bool = True

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("MultiTaskConfig requires at least one task")
        if self.shared_hidden_dim <= 0:
            raise ValueError("shared_hidden_dim must be positive")
        if self.num_shared_layers <= 0:
            raise ValueError("num_shared_layers must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


if _TORCH_AVAILABLE:
    class _SharedBackbone(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, num_layers: int) -> None:
            super().__init__()
            layers: List[nn.Module] = []
            current_dim = input_dim
            for _ in range(num_layers):
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                ])
                current_dim = hidden_dim
            self.network = nn.Sequential(*layers)

        def forward(self, x: Any) -> Any:
            return self.network(x)


    class _TaskHead(nn.Module):
        def __init__(self, input_dim: int, output_dim: int) -> None:
            super().__init__()
            self.proj = nn.Linear(input_dim, output_dim)

        def forward(self, x: Any) -> Any:
            return self.proj(x)


    class _MultiTaskNetwork(nn.Module):
        def __init__(self, input_dim: int, config: MultiTaskConfig) -> None:
            super().__init__()
            self.backbone = _SharedBackbone(
                input_dim=input_dim,
                hidden_dim=config.shared_hidden_dim,
                num_layers=config.num_shared_layers,
            )
            self.heads = nn.ModuleDict(
                {
                    task.task_name: _TaskHead(config.shared_hidden_dim, task.output_dim)
                    for task in config.tasks
                }
            )
            self.log_vars = nn.ParameterDict(
                {
                    task.task_name: nn.Parameter(torch.zeros(1, dtype=torch.float32))
                    for task in config.tasks
                }
            )

        def forward(self, x: Any) -> Dict[str, Any]:
            shared = self.backbone(x)
            return {task_name: head(shared) for task_name, head in self.heads.items()}


class MultiTaskLearner:
    """Shared-backbone multi-task learner with optional PyTorch acceleration."""

    def __init__(
        self,
        config: Optional[MultiTaskConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config
        self.device = device or ("cuda" if _TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.model: Any = None
        self._task_map: Dict[str, TaskConfig] = {task.task_name: task for task in config.tasks} if config else {}
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
        self._shared_projection: Optional[np.ndarray] = None
        self._fallback_heads: Dict[str, Dict[str, np.ndarray]] = {}
        self._task_uncertainty: Dict[str, float] = {}
        self._task_learning_rates: Dict[str, float] = {}
        self._feature_importance_by_task: Dict[str, np.ndarray] = {}
        self._task_loss_history: Dict[str, List[float]] = {}
        self._training_epochs: int = 25
        self._batch_size: int = 64
        self._rng = np.random.default_rng(42)

    def build_model(self, input_dim: int, config: MultiTaskConfig) -> Any:
        self.config = config
        self._task_map = {task.task_name: task for task in config.tasks}
        self._task_uncertainty = {task.task_name: 0.0 for task in config.tasks}
        self._task_learning_rates = {
            task.task_name: self._derive_task_learning_rate(task)
            for task in config.tasks
        }
        self._task_loss_history = {task.task_name: [] for task in config.tasks}
        self._feature_importance_by_task = {
            task.task_name: np.zeros(int(input_dim), dtype=float)
            for task in config.tasks
        }

        if _TORCH_AVAILABLE:
            self.model = _MultiTaskNetwork(input_dim=input_dim, config=config).to(self.device)
            logger.info(
                "Built PyTorch multi-task model for %d tasks on %s",
                len(config.tasks),
                self.device,
            )
        else:
            hidden_dim = min(max(4, config.shared_hidden_dim), input_dim)
            self.model = {
                "backend": "numpy",
                "input_dim": int(input_dim),
                "hidden_dim": int(hidden_dim),
                "num_shared_layers": int(config.num_shared_layers),
                "tasks": list(self._task_map.keys()),
            }
            logger.info(
                "Built NumPy fallback multi-task model for %d tasks (torch unavailable)",
                len(config.tasks),
            )
        return self.model

    def train_shared_backbone(self, X: Any, task_data: Dict[str, np.ndarray]) -> dict:
        features = self._prepare_features(X, fit_stats=True)
        self._ensure_model(features.shape[1])
        targets = self._validate_task_data(task_data, len(features))

        if _TORCH_AVAILABLE:
            result = self._train_torch(features, targets)
        else:
            result = self._train_numpy(features, targets)

        self._update_feature_importance(features, targets)
        result["feature_importance"] = self._serialise_feature_importance()
        logger.info("Finished shared-backbone training for tasks: %s", sorted(targets))
        return result

    def predict_task(self, task_name: str, X: Any) -> np.ndarray:
        self._require_task(task_name)
        features = self._prepare_features(X, fit_stats=False)

        if _TORCH_AVAILABLE and self.model is not None:
            self.model.eval()
            with torch.no_grad():
                x_tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)
                outputs = self.model(x_tensor)[task_name]
            return self._format_predictions(task_name, outputs.detach().cpu().numpy())

        if task_name not in self._fallback_heads or self._shared_projection is None:
            raise RuntimeError(f"Task '{task_name}' has not been trained")

        hidden = self._numpy_backbone(features)
        head = self._fallback_heads[task_name]
        logits = hidden @ head["weights"] + head["bias"]
        return self._format_predictions(task_name, logits)

    def calculate_uncertainty_weighted_loss(self, losses: Dict[str, float]) -> float:
        if not losses:
            return 0.0

        total = 0.0
        for task_name, loss in losses.items():
            task = self._require_task(task_name)
            base_weight = task.loss_weight
            if self.config and self.config.use_uncertainty_weighting:
                log_var = float(self._task_uncertainty.get(task_name, 0.0))
                total += float(np.exp(-log_var) * loss * base_weight + log_var)
            else:
                total += float(base_weight * loss)
        return float(total)

    def evaluate_all_tasks(self, X_test: Any, y_test: Dict[str, np.ndarray]) -> Dict[str, float]:
        features = self._prepare_features(X_test, fit_stats=False)
        targets = self._validate_task_data(y_test, len(features))
        metrics: Dict[str, float] = {}

        for task_name, y_true in targets.items():
            preds = self.predict_task(task_name, X_test)
            task_cfg = self._require_task(task_name)

            if task_cfg.task_type == "classification":
                predicted_labels = self._classification_labels(preds, task_cfg.output_dim)
                true_labels = self._classification_labels(y_true, task_cfg.output_dim)
                metrics[task_name] = float(np.mean(predicted_labels == true_labels))
            else:
                y_true_reg = np.asarray(y_true, dtype=float).reshape(len(features), -1)
                y_pred_reg = np.asarray(preds, dtype=float).reshape(len(features), -1)
                metrics[task_name] = float(np.sqrt(np.mean((y_pred_reg - y_true_reg) ** 2)))

        return metrics

    def get_task_importance(self) -> Dict[str, float]:
        if not self._task_map:
            return {}

        raw_scores: Dict[str, float] = {}
        for task_name, task in self._task_map.items():
            uncertainty_penalty = float(np.exp(-self._task_uncertainty.get(task_name, 0.0)))
            history = self._task_loss_history.get(task_name, [])
            stability = 1.0 / (1.0 + float(np.mean(history[-5:]))) if history else 1.0
            raw_scores[task_name] = float(task.loss_weight * uncertainty_penalty * stability)

        total = sum(raw_scores.values()) or 1.0
        return {task_name: score / total for task_name, score in raw_scores.items()}

    def fine_tune_task(self, task_name: str, X: Any, y: Any, freeze_backbone: bool = True) -> dict:
        task_cfg = self._require_task(task_name)
        features = self._prepare_features(X, fit_stats=False)
        y_arr = self._prepare_target(task_name, y, len(features))

        if _TORCH_AVAILABLE and self.model is not None:
            result = self._fine_tune_torch_task(task_name, features, y_arr, freeze_backbone)
        else:
            result = self._fine_tune_numpy_task(task_name, features, y_arr)

        self._feature_importance_by_task[task_name] = self._compute_task_feature_importance(
            features,
            y_arr,
            task_cfg,
        )
        logger.info("Fine-tuned task '%s' (freeze_backbone=%s)", task_name, freeze_backbone)
        return result

    # ------------------------------------------------------------------
    # PyTorch implementation
    # ------------------------------------------------------------------

    def _train_torch(self, X: np.ndarray, task_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        n_samples = len(X)
        batch_size = min(self._batch_size, n_samples)
        last_epoch_losses: Dict[str, float] = {name: 0.0 for name in task_data}

        x_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_tensors = {
            task_name: torch.as_tensor(target, dtype=torch.float32, device=self.device)
            for task_name, target in task_data.items()
        }

        for epoch in range(self._training_epochs):
            permutation = torch.randperm(n_samples, device=self.device)
            epoch_losses = {name: [] for name in task_data}

            self.model.train()
            for start in range(0, n_samples, batch_size):
                idx = permutation[start:start + batch_size]
                batch_x = x_tensor[idx]
                batch_targets = {name: target[idx] for name, target in y_tensors.items()}

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                task_losses = {
                    task_name: self._torch_task_loss(task_name, outputs[task_name], batch_targets[task_name])
                    for task_name in batch_targets
                }

                weighted_loss = self._torch_weighted_loss(task_losses)
                weighted_loss.backward(retain_graph=True)
                self._apply_gradient_surgery(batch_x, batch_targets, task_losses)
                optimizer.step()

                for task_name, loss_value in task_losses.items():
                    scalar = float(loss_value.detach().cpu().item())
                    epoch_losses[task_name].append(scalar)

            for task_name, losses in epoch_losses.items():
                average_loss = float(np.mean(losses)) if losses else 0.0
                self._task_loss_history[task_name].append(average_loss)
                last_epoch_losses[task_name] = average_loss
                if self.config.use_uncertainty_weighting:
                    self._task_uncertainty[task_name] = float(
                        self.model.log_vars[task_name].detach().cpu().item()
                    )

            self._update_task_learning_rates(epoch)

        return {
            "backend": "torch",
            "epochs": self._training_epochs,
            "task_losses": last_epoch_losses,
            "weighted_loss": self.calculate_uncertainty_weighted_loss(last_epoch_losses),
            "task_learning_rates": dict(self._task_learning_rates),
            "feature_importance": self._serialise_feature_importance(),
        }

    def _torch_weighted_loss(self, task_losses: Dict[str, Any]) -> Any:
        total: Optional[Any] = None
        for task_name, loss in task_losses.items():
            task_cfg = self._require_task(task_name)
            if self.config.use_uncertainty_weighting:
                log_var = self.model.log_vars[task_name]
                weighted = torch.exp(-log_var) * loss * task_cfg.loss_weight + log_var
            else:
                weighted = loss * task_cfg.loss_weight
            total = weighted if total is None else total + weighted
        if total is None:
            raise RuntimeError("No task losses available for weighting")
        return total

    def _apply_gradient_surgery(
        self,
        batch_x: Any,
        batch_targets: Dict[str, Any],
        task_losses: Dict[str, Any],
    ) -> None:
        if self.model is None:
            return

        shared_params = [param for param in self.model.backbone.parameters() if param.requires_grad]
        if not shared_params:
            return

        task_grad_vectors: List[Any] = []
        for task_name in batch_targets:
            grads = torch.autograd.grad(
                task_losses[task_name],
                shared_params,
                retain_graph=True,
                allow_unused=True,
            )
            flat_grads = []
            for param, grad in zip(shared_params, grads):
                if grad is None:
                    flat_grads.append(torch.zeros_like(param).reshape(-1))
                else:
                    flat_grads.append(grad.reshape(-1))
            task_grad_vectors.append(torch.cat(flat_grads))

        if not task_grad_vectors:
            return

        projected: List[Any] = []
        for i, grad_i in enumerate(task_grad_vectors):
            adjusted = grad_i.clone()
            for j, grad_j in enumerate(task_grad_vectors):
                if i == j:
                    continue
                dot = torch.dot(adjusted, grad_j)
                denom = torch.dot(grad_j, grad_j) + 1e-12
                if dot < 0:
                    adjusted = adjusted - (dot / denom) * grad_j
            projected.append(adjusted)

        merged = torch.stack(projected, dim=0).mean(dim=0)
        cursor = 0
        for param in shared_params:
            size = param.numel()
            param.grad = merged[cursor:cursor + size].view_as(param).clone()
            cursor += size

    def _fine_tune_torch_task(
        self,
        task_name: str,
        X: np.ndarray,
        y: np.ndarray,
        freeze_backbone: bool,
    ) -> Dict[str, Any]:
        self.model.train()
        backbone_params = list(self.model.backbone.parameters())
        for param in backbone_params:
            param.requires_grad = not freeze_backbone

        params: List[Any] = list(self.model.heads[task_name].parameters())
        if not freeze_backbone:
            params.extend(backbone_params)
        if self.config.use_uncertainty_weighting:
            params.append(self.model.log_vars[task_name])

        lr = self._task_learning_rates.get(task_name, self.config.learning_rate)
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

        x_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        batch_size = min(self._batch_size, len(X))
        epochs = max(6, self._training_epochs // 3)
        final_loss = 0.0

        for _ in range(epochs):
            permutation = torch.randperm(len(X), device=self.device)
            epoch_losses: List[float] = []
            for start in range(0, len(X), batch_size):
                idx = permutation[start:start + batch_size]
                optimizer.zero_grad()
                outputs = self.model(x_tensor[idx])[task_name]
                loss = self._torch_task_loss(task_name, outputs, y_tensor[idx])
                if self.config.use_uncertainty_weighting:
                    log_var = self.model.log_vars[task_name]
                    weighted = torch.exp(-log_var) * loss * self._task_map[task_name].loss_weight + log_var
                else:
                    weighted = loss * self._task_map[task_name].loss_weight
                weighted.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))
            scheduler.step()
            final_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        self._task_learning_rates[task_name] = float(scheduler.get_last_lr()[0])
        self._task_loss_history.setdefault(task_name, []).append(final_loss)
        if self.config.use_uncertainty_weighting:
            self._task_uncertainty[task_name] = float(self.model.log_vars[task_name].detach().cpu().item())

        for param in backbone_params:
            param.requires_grad = True

        return {
            "backend": "torch",
            "task_name": task_name,
            "freeze_backbone": freeze_backbone,
            "epochs": epochs,
            "final_loss": final_loss,
            "learning_rate": self._task_learning_rates[task_name],
        }

    def _torch_task_loss(
        self,
        task_name: str,
        predictions: Any,
        targets: Any,
    ) -> Any:
        if F is None:
            raise RuntimeError("PyTorch functional API is unavailable")
        task_cfg = self._require_task(task_name)
        if task_cfg.task_type == "classification":
            if task_cfg.output_dim == 1:
                targets = targets.reshape(-1, 1)
                return F.binary_cross_entropy_with_logits(predictions, targets)
            labels = targets.argmax(dim=1) if targets.ndim > 1 and targets.shape[1] > 1 else targets.long().view(-1)
            return F.cross_entropy(predictions, labels)
        targets = targets.reshape(predictions.shape)
        return F.mse_loss(predictions, targets)

    # ------------------------------------------------------------------
    # NumPy fallback implementation
    # ------------------------------------------------------------------

    def _train_numpy(self, X: np.ndarray, task_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        hidden_dim = int(self.model["hidden_dim"])
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        projection = vt[:hidden_dim].T
        if projection.shape[1] < hidden_dim:
            projection = np.pad(projection, ((0, 0), (0, hidden_dim - projection.shape[1])))
        self._shared_projection = projection

        hidden = self._numpy_backbone(X)
        task_losses: Dict[str, float] = {}

        for task_name, target in task_data.items():
            task_cfg = self._require_task(task_name)
            if task_cfg.task_type == "classification":
                weights, bias = self._fit_numpy_classification_head(
                    task_name,
                    hidden,
                    target,
                    task_cfg.output_dim,
                )
                logits = hidden @ weights + bias
                probs = self._format_predictions(task_name, logits)
                loss = self._numpy_classification_loss(task_cfg, probs, target)
            else:
                weights, bias = self._fit_numpy_regression_head(hidden, target)
                preds = hidden @ weights + bias
                loss = float(np.mean((preds - target.reshape(len(X), -1)) ** 2))

            self._fallback_heads[task_name] = {"weights": weights, "bias": bias}
            task_losses[task_name] = float(loss)
            self._task_uncertainty[task_name] = float(np.log(max(loss, 1e-8)))
            self._task_loss_history.setdefault(task_name, []).append(float(loss))

        self._update_task_learning_rates(self._training_epochs - 1)
        return {
            "backend": "numpy",
            "epochs": 1,
            "task_losses": task_losses,
            "weighted_loss": self.calculate_uncertainty_weighted_loss(task_losses),
            "task_learning_rates": dict(self._task_learning_rates),
            "feature_importance": self._serialise_feature_importance(),
        }

    def _fine_tune_numpy_task(self, task_name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        if self._shared_projection is None:
            raise RuntimeError("Shared backbone must be trained before fine-tuning a task")

        task_cfg = self._require_task(task_name)
        hidden = self._numpy_backbone(X)
        if task_cfg.task_type == "classification":
            weights, bias = self._fit_numpy_classification_head(
                task_name,
                hidden,
                y,
                task_cfg.output_dim,
            )
            logits = hidden @ weights + bias
            preds = self._format_predictions(task_name, logits)
            final_loss = self._numpy_classification_loss(task_cfg, preds, y)
        else:
            weights, bias = self._fit_numpy_regression_head(hidden, y)
            preds = hidden @ weights + bias
            final_loss = float(np.mean((preds - y.reshape(len(X), -1)) ** 2))

        self._fallback_heads[task_name] = {"weights": weights, "bias": bias}
        self._task_uncertainty[task_name] = float(np.log(max(final_loss, 1e-8)))
        self._task_loss_history.setdefault(task_name, []).append(final_loss)
        self._task_learning_rates[task_name] *= 0.96

        return {
            "backend": "numpy",
            "task_name": task_name,
            "freeze_backbone": True,
            "epochs": 1,
            "final_loss": float(final_loss),
            "learning_rate": float(self._task_learning_rates[task_name]),
        }

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _ensure_model(self, input_dim: int) -> None:
        if self.config is None:
            raise RuntimeError("build_model or config initialisation is required before training")
        if self.model is None:
            self.build_model(input_dim=input_dim, config=self.config)

    def _prepare_features(self, X: Any, fit_stats: bool) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError("X must be a 2D array-like structure")

        if fit_stats or self._feature_mean is None or self._feature_std is None:
            self._feature_mean = np.mean(arr, axis=0)
            self._feature_std = np.std(arr, axis=0) + 1e-8

        return (arr - self._feature_mean) / self._feature_std

    def _prepare_target(self, task_name: str, y: Any, n_samples: int) -> np.ndarray:
        arr = np.asarray(y)
        if arr.shape[0] != n_samples:
            raise ValueError(
                f"Task '{task_name}' received {arr.shape[0]} targets for {n_samples} samples"
            )
        task_cfg = self._require_task(task_name)
        if task_cfg.task_type == "regression":
            arr = np.asarray(arr, dtype=float).reshape(n_samples, -1)
            if arr.shape[1] != task_cfg.output_dim:
                if task_cfg.output_dim == 1 and arr.shape[1] == 1:
                    return arr
                raise ValueError(
                    f"Task '{task_name}' expects output_dim={task_cfg.output_dim}, got {arr.shape[1]}"
                )
            return arr

        if task_cfg.output_dim == 1:
            arr = np.asarray(arr, dtype=float).reshape(n_samples, 1)
            return arr

        if arr.ndim == 1:
            labels = arr.astype(int)
            one_hot = np.zeros((n_samples, task_cfg.output_dim), dtype=float)
            clipped = np.clip(labels, 0, task_cfg.output_dim - 1)
            one_hot[np.arange(n_samples), clipped] = 1.0
            return one_hot

        arr = np.asarray(arr, dtype=float).reshape(n_samples, -1)
        if arr.shape[1] != task_cfg.output_dim:
            raise ValueError(
                f"Task '{task_name}' expects output_dim={task_cfg.output_dim}, got {arr.shape[1]}"
            )
        return arr

    def _validate_task_data(self, task_data: Dict[str, np.ndarray], n_samples: int) -> Dict[str, np.ndarray]:
        if not task_data:
            raise ValueError("task_data must not be empty")
        validated: Dict[str, np.ndarray] = {}
        for task_name, values in task_data.items():
            validated[task_name] = self._prepare_target(task_name, values, n_samples)
        return validated

    def _require_task(self, task_name: str) -> TaskConfig:
        if task_name not in self._task_map:
            raise KeyError(f"Unknown task '{task_name}'")
        return self._task_map[task_name]

    def _derive_task_learning_rate(self, task: TaskConfig) -> float:
        base = self.config.learning_rate if self.config else 0.001
        if task.task_type == "classification":
            return float(base * 0.8)
        return float(base * 1.1)

    def _update_task_learning_rates(self, epoch: int) -> None:
        for task_name, current_lr in list(self._task_learning_rates.items()):
            decay = 0.98 if epoch < max(1, self._training_epochs // 2) else 0.95
            self._task_learning_rates[task_name] = float(max(current_lr * decay, 1e-5))

    def _numpy_backbone(self, X: np.ndarray) -> np.ndarray:
        if self._shared_projection is None:
            raise RuntimeError("Shared backbone is not initialised")
        hidden = X @ self._shared_projection
        layers = max(1, self.config.num_shared_layers if self.config else 1)
        for _ in range(layers):
            hidden = np.tanh(hidden)
        return hidden

    def _fit_numpy_regression_head(self, hidden: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(target, dtype=float).reshape(len(hidden), -1)
        design = np.concatenate([hidden, np.ones((len(hidden), 1))], axis=1)
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
        weights = coeffs[:-1, :]
        bias = coeffs[-1:, :]
        return weights, bias

    def _fit_numpy_classification_head(
        self,
        task_name: str,
        hidden: np.ndarray,
        target: np.ndarray,
        output_dim: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if output_dim == 1:
            y = np.asarray(target, dtype=float).reshape(len(hidden), 1)
        else:
            y = self._prepare_target(task_name, target, len(hidden))
        y = np.clip(y, 1e-4, 1 - 1e-4) if output_dim == 1 else y
        logit_target = np.log(y / (1 - y)) if output_dim == 1 else y
        design = np.concatenate([hidden, np.ones((len(hidden), 1))], axis=1)
        coeffs, *_ = np.linalg.lstsq(design, logit_target, rcond=None)
        weights = coeffs[:-1, :]
        bias = coeffs[-1:, :]
        return weights, bias

    def _numpy_classification_loss(
        self,
        task_cfg: TaskConfig,
        predictions: np.ndarray,
        target: np.ndarray,
    ) -> float:
        y_true = self._prepare_target(task_cfg.task_name, target, len(predictions))
        if task_cfg.output_dim == 1:
            probs = np.clip(predictions.reshape(len(predictions), 1), 1e-7, 1 - 1e-7)
            return float(-np.mean(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs)))

        probs = np.clip(predictions, 1e-7, 1.0)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return float(-np.mean(np.sum(y_true * np.log(probs), axis=1)))

    def _format_predictions(self, task_name: str, raw: np.ndarray) -> np.ndarray:
        task_cfg = self._require_task(task_name)
        arr = np.asarray(raw, dtype=float).reshape(len(raw), -1)
        if task_cfg.task_type == "regression":
            return arr
        if task_cfg.output_dim == 1:
            return 1.0 / (1.0 + np.exp(-arr))
        shifted = arr - np.max(arr, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / (np.sum(exp_vals, axis=1, keepdims=True) + 1e-12)

    def _classification_labels(self, arr: Any, output_dim: int) -> np.ndarray:
        values = np.asarray(arr)
        if output_dim == 1:
            reshaped = values.reshape(len(values), -1)
            return (reshaped[:, 0] >= 0.5).astype(int)
        if values.ndim == 1:
            return values.astype(int)
        return np.argmax(values, axis=1)

    def _update_feature_importance(self, X: np.ndarray, task_data: Dict[str, np.ndarray]) -> None:
        for task_name, y in task_data.items():
            self._feature_importance_by_task[task_name] = self._compute_task_feature_importance(
                X,
                y,
                self._require_task(task_name),
            )

    def _compute_task_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_cfg: TaskConfig,
    ) -> np.ndarray:
        if _TORCH_AVAILABLE and self.model is not None:
            return self._gradient_feature_importance(task_cfg.task_name, X)

        y_arr = np.asarray(y, dtype=float).reshape(len(X), -1)
        scores: List[float] = []
        for column in range(X.shape[1]):
            feature = X[:, column]
            if np.std(feature) < 1e-10:
                scores.append(0.0)
                continue
            if task_cfg.task_type == "classification" and y_arr.shape[1] > 1:
                target_vec = np.argmax(y_arr, axis=1).astype(float)
            else:
                target_vec = y_arr[:, 0].astype(float)
            if np.std(target_vec) < 1e-10:
                scores.append(0.0)
            else:
                scores.append(abs(float(np.corrcoef(feature, target_vec)[0, 1])))
        importance = np.asarray(scores, dtype=float)
        total = np.sum(importance) or 1.0
        return importance / total

    def _gradient_feature_importance(self, task_name: str, X: np.ndarray) -> np.ndarray:
        if torch is None:
            raise RuntimeError("PyTorch is unavailable for gradient-based feature importance")
        self.model.eval()
        x_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        x_tensor.requires_grad_(True)
        outputs = self.model(x_tensor)[task_name]
        score = outputs.mean()
        self.model.zero_grad()
        score.backward()
        gradients = x_tensor.grad.detach().abs().mean(dim=0).cpu().numpy()
        total = float(np.sum(gradients)) or 1.0
        return gradients / total

    def _serialise_feature_importance(self) -> Dict[str, List[float]]:
        return {
            task_name: importance.astype(float).tolist()
            for task_name, importance in self._feature_importance_by_task.items()
        }


__all__ = [
    "TaskConfig",
    "MultiTaskConfig",
    "MultiTaskLearner",
]
