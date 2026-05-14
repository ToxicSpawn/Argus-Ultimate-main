"""Decoder-only Transformer for microstructure foundation modelling."""

# pyright: reportMissingImports=false

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FoundationModelConfig:
    """Config for the TradeFM-style decoder-only transformer."""

    vocab_size: int = 1024
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    dropout: float = 0.1
    max_seq_len: int = 2048
    ffn_mult: int = 4
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5
    use_bias: bool = True

    def __post_init__(self) -> None:
        self.vocab_size = max(64, int(self.vocab_size))
        self.d_model = max(64, int(self.d_model))
        self.n_heads = max(1, int(self.n_heads))
        self.n_layers = max(1, int(self.n_layers))
        self.max_seq_len = max(32, int(self.max_seq_len))
        self.ffn_mult = max(2, int(self.ffn_mult))
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

    @classmethod
    def small_50m(cls, *, vocab_size: int = 1024, max_seq_len: int = 2048, dropout: float = 0.1) -> "FoundationModelConfig":
        """Return an iteration-friendly configuration near 50M parameters."""
        return cls(
            vocab_size=vocab_size,
            d_model=704,
            n_heads=8,
            n_layers=8,
            dropout=dropout,
            max_seq_len=max_seq_len,
            ffn_mult=4,
        )

    @classmethod
    def compact(cls, *, vocab_size: int = 1024, max_seq_len: int = 1024, dropout: float = 0.1) -> "FoundationModelConfig":
        """Return a fast local-iteration configuration."""
        return cls(
            vocab_size=vocab_size,
            d_model=256,
            n_heads=8,
            n_layers=4,
            dropout=dropout,
            max_seq_len=max_seq_len,
            ffn_mult=4,
        )

    def estimated_parameter_count(self) -> int:
        token_and_head = self.vocab_size * self.d_model * 2
        position = self.max_seq_len * self.d_model
        per_layer = (4 * self.d_model * self.d_model) + (2 * self.d_model * (self.d_model * self.ffn_mult))
        norms = self.n_layers * (4 * self.d_model) + self.d_model
        return int(token_and_head + position + self.n_layers * per_layer + norms)


class EventPositionalEncoding(nn.Module):
    """Learned positional embedding for event sequences."""

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        return self.embedding(positions).unsqueeze(0)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with explicit causal masking."""

    def __init__(self, config: FoundationModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(config.d_model, config.d_model * 3, bias=config.use_bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, dim = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def _reshape(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = _reshape(q)
        k = _reshape(k)
        v = _reshape(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        if attention_mask is not None:
            expanded = attention_mask[:, None, None, :].to(dtype=torch.bool)
            scores = scores.masked_fill(~expanded, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        output = self.out_proj(context)
        return output, attention_weights if return_attention else None


class DecoderBlock(nn.Module):
    """Pre-norm decoder block."""

    def __init__(self, config: FoundationModelConfig) -> None:
        super().__init__()
        inner_dim = config.d_model * config.ffn_mult
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = MultiHeadSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        activation = nn.GELU() if config.activation.lower() == "gelu" else nn.ReLU()
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, inner_dim, bias=config.use_bias),
            activation,
            nn.Dropout(config.dropout),
            nn.Linear(inner_dim, config.d_model, bias=config.use_bias),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_weights = self.attn(self.ln_1(x), causal_mask, attention_mask, return_attention=return_attention)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights


class TradeFoundationModel(nn.Module):
    """Decoder-only transformer for next-event and masked-event learning."""

    def __init__(self, config: Optional[FoundationModelConfig] = None) -> None:
        super().__init__()
        self.config = config or FoundationModelConfig()
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.position_embedding = EventPositionalEncoding(self.config.max_seq_len, self.config.d_model)
        self.dropout = nn.Dropout(self.config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(self.config) for _ in range(self.config.n_layers)])
        self.ln_f = nn.LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        logger.info("TradeFoundationModel initialised — %.2fM parameters", self.parameter_count() / 1e6)

    @property
    def n_parameters(self) -> int:
        return self.parameter_count()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=device))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, seq_len)")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.config.max_seq_len}")

        device = input_ids.device
        x = self.token_embedding(input_ids) + self.position_embedding(seq_len, device)
        x = self.dropout(x)
        causal_mask = self.build_causal_mask(seq_len, device)
        attention_maps: List[torch.Tensor] = []

        for block in self.blocks:
            x, attn = block(x, causal_mask, attention_mask, return_attention=return_attention)
            if attn is not None:
                attention_maps.append(attn)

        hidden_states = self.ln_f(x)
        logits = self.lm_head(hidden_states)
        output: Dict[str, torch.Tensor | List[torch.Tensor]] = {
            "logits": logits,
            "hidden_states": hidden_states,
        }
        if return_attention:
            output["attention_maps"] = attention_maps
        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids
        for _ in range(max_new_tokens):
            context = generated[:, -self.config.max_seq_len :]
            output = self.forward(context)
            logits_tensor = cast(torch.Tensor, output["logits"])
            logits = logits_tensor[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated
