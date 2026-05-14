"""ModalFusion — fuses 5 market modalities into a single dense representation.

Modalities:
  1. LOB tensor        — limit order book (bid/ask levels, volume, imbalance)
  2. Chart CNN emb     — chart pattern embedding from ChartPatternCNN
  3. Sentiment vector  — FinBERT sentiment scores (pos/neg/neu + confidence)
  4. GNN asset flow    — cross-asset flow embedding from GNNAssetFlow
  5. Regime vector     — HMM + autoencoder regime probabilities

Missing modalities are gracefully zero-padded so the model degrades
gracefully when a data source is unavailable.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


MODALITY_DIMS: Dict[str, int] = {
    "lob": 128,
    "chart": 64,
    "sentiment": 32,
    "gnn": 64,
    "regime": 16,
}

FUSED_DIM = 256


class ModalEncoder(nn.Module):
    """Lightweight MLP encoder per modality."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossModalAttention(nn.Module):
    """Cross-modal attention to learn inter-modality dependencies."""

    def __init__(self, embed_dim: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, modalities: torch.Tensor) -> torch.Tensor:
        """Self-attention across modality tokens.

        Args:
            modalities: (B, num_modalities, embed_dim)

        Returns:
            (B, num_modalities, embed_dim) attended representations.
        """
        out, _ = self.attn(modalities, modalities, modalities)
        return self.ln(modalities + out)


class ModalFusion(nn.Module):
    """Fuse up to 5 market modalities into FUSED_DIM=256 representation.

    Args:
        per_modal_dim:  Shared projected dimension per modality (default 64).
        fused_dim:      Output dimension (default 256).
        n_cross_heads:  Heads for cross-modal attention (default 4).
        dropout:        Dropout probability.
    """

    def __init__(
        self,
        per_modal_dim: int = 64,
        fused_dim: int = FUSED_DIM,
        n_cross_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.per_modal_dim = per_modal_dim
        self.num_modalities = len(MODALITY_DIMS)

        self.encoders = nn.ModuleDict(
            {
                name: ModalEncoder(dim, per_modal_dim, dropout)
                for name, dim in MODALITY_DIMS.items()
            }
        )
        self.cross_attn = CrossModalAttention(per_modal_dim, n_cross_heads, dropout)
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.num_modalities * per_modal_dim, fused_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim * 2, fused_dim),
            nn.LayerNorm(fused_dim),
        )
        self._raw_dims = MODALITY_DIMS

    def forward(
        self,
        lob: Optional[torch.Tensor] = None,
        chart: Optional[torch.Tensor] = None,
        sentiment: Optional[torch.Tensor] = None,
        gnn: Optional[torch.Tensor] = None,
        regime: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse modalities. Missing modalities are zero-padded.

        All tensors should be (B, dim) or (B, T, dim).
        Returns (B, FUSED_DIM) or (B, T, FUSED_DIM).
        """
        inputs = {"lob": lob, "chart": chart, "sentiment": sentiment, "gnn": gnn, "regime": regime}
        batch_ref = next(v for v in inputs.values() if v is not None)
        B = batch_ref.shape[0]
        has_time = batch_ref.dim() == 3
        T = batch_ref.shape[1] if has_time else 1
        device = batch_ref.device
        dtype = batch_ref.dtype

        encoded = []
        for name, raw in inputs.items():
            raw_dim = self._raw_dims[name]
            if raw is None:
                placeholder = torch.zeros(B, T, raw_dim, device=device, dtype=dtype)
            else:
                placeholder = raw if raw.dim() == 3 else raw.unsqueeze(1).expand(B, T, -1)
            enc = self.encoders[name](placeholder)
            encoded.append(enc)

        stacked = torch.stack(encoded, dim=2)
        B_, T_, M, D = stacked.shape
        stacked_flat = stacked.reshape(B_ * T_, M, D)
        attended = self.cross_attn(stacked_flat).reshape(B_, T_, M * D)
        fused = self.fusion_proj(attended)
        return fused.squeeze(1) if not has_time else fused
