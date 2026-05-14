"""ArgusBackbone — causal temporal transformer with RoPE positional encoding.

Architecture:
  - 512-dim model, 8 attention heads, 6 transformer layers
  - Causal mask (no future leakage)
  - Rotary Positional Encoding (RoPE) on Q/K projections
  - Pre-LayerNorm + GELU activations
  - Regime-conditioning via FiLM (Feature-wise Linear Modulation)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_REGIMES = 4  # RANGING, TRENDING, VOLATILE, CRISIS


class RotaryEmbedding(nn.Module):
    """Rotary Positional Encoding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 512) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cache", emb.sin()[None, None, :, :])

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        cos = self.cos_cache[:, :, :seq_len, :].to(q.dtype)
        sin = self.sin_cache[:, :, :seq_len, :].to(q.dtype)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for regime conditioning."""

    def __init__(self, d_model: int, regime_dim: int) -> None:
        super().__init__()
        self.gamma_proj = nn.Linear(regime_dim, d_model)
        self.beta_proj = nn.Linear(regime_dim, d_model)

    def forward(self, x: torch.Tensor, regime_emb: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_proj(regime_emb).unsqueeze(1)
        beta = self.beta_proj(regime_emb).unsqueeze(1)
        return gamma * x + beta


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.rope(q, k)

        attn = (q @ k.transpose(-2, -1)) / self.scale
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn = attn.masked_fill(causal_mask == 0, float("-inf"))
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block with FiLM regime conditioning."""

    def __init__(self, d_model: int, n_heads: int, regime_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.film = FiLMLayer(d_model, regime_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, regime_emb: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), mask))
        x = self.film(x, regime_emb)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class ArgusBackbone(nn.Module):
    """Causal temporal transformer backbone for Argus-AI.

    Args:
        d_model:     Model dimension (default 512).
        n_heads:     Attention heads (default 8).
        n_layers:    Transformer layers (default 6).
        input_dim:   Input feature dimension from ModalFusion output.
        regime_dim:  Regime embedding dimension (default 64).
        dropout:     Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        input_dim: int = 256,
        regime_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.regime_embed = nn.Embedding(NUM_REGIMES, regime_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, regime_dim, dropout) for _ in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        regime_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x:           (B, T, input_dim) fused feature tensor.
            regime_ids:  (B,) integer regime labels [0-3].
            mask:        Optional attention bias mask.

        Returns:
            (B, T, d_model) contextual representations.
        """
        x = self.input_proj(x)
        regime_emb = self.regime_embed(regime_ids)
        for block in self.blocks:
            x = block(x, regime_emb, mask)
        return self.ln_final(x)
