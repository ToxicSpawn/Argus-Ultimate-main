"""Temporal attention with linear-time selective memory for market graphs."""

# pyright: reportMissingImports=false, reportConstantRedefinition=false, reportOptionalMemberAccess=false, reportInvalidTypeForm=false

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


@dataclass(slots=True)
class TemporalAttentionConfig:
    input_dim: int
    hidden_dim: int = 64
    num_heads: int = 4
    memory_decay: float = 0.85
    dropout: float = 0.1
    max_memory_steps: int = 32

    def __post_init__(self) -> None:
        self.input_dim = int(self.input_dim)
        self.hidden_dim = int(self.hidden_dim)
        self.num_heads = max(1, int(self.num_heads))
        self.memory_decay = float(min(0.999, max(0.1, self.memory_decay)))
        self.dropout = float(min(0.9, max(0.0, self.dropout)))
        self.max_memory_steps = max(4, int(self.max_memory_steps))
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")


@dataclass(slots=True)
class SelectiveMemory:
    values: np.ndarray
    importance: np.ndarray
    timestamps: Optional[np.ndarray] = None

    def topk(self, k: int = 5) -> Dict[str, np.ndarray]:
        if self.importance.size == 0:
            return {"indices": np.zeros((0,), dtype=np.int64), "scores": np.zeros((0,), dtype=np.float32)}
        order = np.argsort(self.importance)[::-1][: max(1, int(k))]
        return {
            "indices": order.astype(np.int64),
            "scores": self.importance[order].astype(np.float32),
        }


if _TORCH_AVAILABLE:

    class TemporalAttentionEncoder(nn.Module):
        """Linear-time temporal encoder inspired by selective state-space memory."""

        def __init__(self, config: TemporalAttentionConfig) -> None:
            super().__init__()
            self.config = config
            self.head_dim = config.hidden_dim // config.num_heads
            self.in_proj = nn.Linear(config.input_dim, config.hidden_dim)
            self.value_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.gate_proj = nn.Linear(config.hidden_dim, config.num_heads)
            self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.dropout = nn.Dropout(config.dropout)
            self.last_attention_weights: Optional[np.ndarray] = None

        def forward(
            self,
            x: "torch.Tensor",
            mask: Optional["torch.Tensor"] = None,
            return_attention: bool = True,
        ) -> Tuple["torch.Tensor", Dict[str, np.ndarray]]:
            if x.dim() != 3:
                raise ValueError("x must have shape (batch, time, features)")
            batch_size, time_steps, _ = x.shape
            hidden = torch.tanh(self.in_proj(x))
            values = self.value_proj(hidden).view(batch_size, time_steps, self.config.num_heads, self.head_dim)
            gates = torch.sigmoid(self.gate_proj(hidden))
            if mask is not None:
                gates = gates * mask.unsqueeze(-1)

            state = torch.zeros(batch_size, self.config.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
            outputs = []
            attention_scores = []
            decay = self.config.memory_decay
            for step in range(time_steps):
                gate_t = gates[:, step].unsqueeze(-1)
                value_t = values[:, step]
                state = decay * state + gate_t * value_t
                norm = torch.clamp(torch.norm(state, dim=-1), min=1e-6)
                attention_scores.append(norm)
                outputs.append(state.reshape(batch_size, -1))

            stacked = torch.stack(outputs, dim=1)
            stacked = self.dropout(self.out_proj(stacked))
            scores = torch.stack(attention_scores, dim=1)
            weights = scores / torch.clamp(scores.sum(dim=1, keepdim=True), min=1e-6)
            attention_payload = {
                "weights": weights.detach().cpu().numpy().astype(np.float32),
                "head_importance": weights.mean(dim=1).detach().cpu().numpy().astype(np.float32),
                "memory_norm": scores.detach().cpu().numpy().astype(np.float32),
            }
            if return_attention:
                self.last_attention_weights = attention_payload["weights"]
            return stacked, attention_payload

        def selective_memory(self, x: np.ndarray) -> SelectiveMemory:
            outputs, attention = self.forward(torch.as_tensor(x, dtype=torch.float32), return_attention=True)
            del outputs
            weights = attention["weights"]
            if weights.ndim == 3:
                importance = weights.mean(axis=(0, 2))
            else:
                importance = weights.mean(axis=0)
            return SelectiveMemory(values=np.asarray(x, dtype=np.float32), importance=importance.astype(np.float32))

else:

    class TemporalAttentionEncoder:  # type: ignore[no-redef]
        def __init__(self, config: TemporalAttentionConfig) -> None:
            self.config = config
            rng = np.random.default_rng(42)
            self.w_in = rng.normal(0.0, 0.05, size=(config.input_dim, config.hidden_dim)).astype(np.float32)
            self.w_value = rng.normal(0.0, 0.05, size=(config.hidden_dim, config.hidden_dim)).astype(np.float32)
            self.w_gate = rng.normal(0.0, 0.05, size=(config.hidden_dim, config.num_heads)).astype(np.float32)
            self.w_out = rng.normal(0.0, 0.05, size=(config.hidden_dim, config.hidden_dim)).astype(np.float32)
            self.head_dim = config.hidden_dim // config.num_heads
            self.last_attention_weights: Optional[np.ndarray] = None

        def forward(
            self,
            x: np.ndarray,
            mask: Optional[np.ndarray] = None,
            return_attention: bool = True,
        ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
            array = np.asarray(x, dtype=np.float32)
            if array.ndim != 3:
                raise ValueError("x must have shape (batch, time, features)")
            hidden = np.tanh(array @ self.w_in)
            values = (hidden @ self.w_value).reshape(array.shape[0], array.shape[1], self.config.num_heads, self.head_dim)
            gates = 1.0 / (1.0 + np.exp(-(hidden @ self.w_gate)))
            if mask is not None:
                gates = gates * np.asarray(mask, dtype=np.float32)[..., None]

            state = np.zeros((array.shape[0], self.config.num_heads, self.head_dim), dtype=np.float32)
            outputs = []
            norms = []
            for step in range(array.shape[1]):
                gate_t = gates[:, step, :, None]
                state = self.config.memory_decay * state + gate_t * values[:, step]
                norms.append(np.linalg.norm(state, axis=-1))
                outputs.append(state.reshape(array.shape[0], -1))

            stacked = np.stack(outputs, axis=1) @ self.w_out
            norm_array = np.stack(norms, axis=1)
            weights = norm_array / np.clip(norm_array.sum(axis=1, keepdims=True), 1e-6, None)
            attention_payload = {
                "weights": weights.astype(np.float32),
                "head_importance": weights.mean(axis=1).astype(np.float32),
                "memory_norm": norm_array.astype(np.float32),
            }
            if return_attention:
                self.last_attention_weights = attention_payload["weights"]
            return stacked.astype(np.float32), attention_payload

        def selective_memory(self, x: np.ndarray) -> SelectiveMemory:
            _, attention = self.forward(x, return_attention=True)
            weights = attention["weights"]
            importance = weights.mean(axis=(0, 2)) if weights.ndim == 3 else weights.mean(axis=0)
            return SelectiveMemory(values=np.asarray(x, dtype=np.float32), importance=importance.astype(np.float32))
