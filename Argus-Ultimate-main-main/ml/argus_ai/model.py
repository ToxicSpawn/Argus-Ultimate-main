"""ArgusAI — finance-specialised multimodal reasoning model.

Architecture overview:
  ┌─────────────────────────────────────────────────────────┐
  │  ModalFusion  (LOB + Chart + Sentiment + GNN + Regime)  │
  └───────────────────────┬─────────────────────────────────┘
                          │  (B, T, 256)
  ┌───────────────────────▼─────────────────────────────────┐
  │  ArgusBackbone  (6-layer causal transformer, 512d, RoPE) │
  └───────────────────────┬─────────────────────────────────┘
                          │  last token (B, 512)
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼                ▼
  DirectionHead     SizeHead         TimingHead    ConfidenceHead
  (flat/long/short) (Beta params)    (tick delay)  (MC Dropout)

Pipeline:
  1. ModalFusion encodes up to 5 modalities → (B, T, 256)
  2. ArgusBackbone contextualises the sequence → (B, T, 512)
  3. Last token is taken as the decision representation
  4. Four heads emit specialised outputs
  5. ChainOfThoughtReasoner interprets head outputs → CoTScratchpad
  6. RLTuner optionally online-updates the model
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ml.argus_ai.backbone import ArgusBackbone
from ml.argus_ai.fusion import ModalFusion
from ml.argus_ai.heads import DirectionHead, SizeHead, TimingHead, ConfidenceHead
from ml.argus_ai.cot_reasoner import ChainOfThoughtReasoner, CoTScratchpad

logger = logging.getLogger(__name__)


@dataclass
class ArgusAIOutput:
    direction_logits: torch.Tensor      # (B, 3)
    direction_probs: torch.Tensor       # (B, 3)
    direction_action: torch.Tensor      # (B,)
    size_alpha: torch.Tensor            # (B, 1)
    size_beta: torch.Tensor             # (B, 1)
    size_mean: torch.Tensor             # (B, 1)
    timing_delay: torch.Tensor          # (B, 1)
    confidence_mean: torch.Tensor       # (B, 1)
    confidence_std: torch.Tensor        # (B, 1)
    backbone_repr: torch.Tensor         # (B, 512) last-token repr
    cot: Optional[List[CoTScratchpad]] = None


class ArgusAI(nn.Module):
    """Argus-AI master model.

    Args:
        d_model:        Backbone model dimension (default 512).
        n_heads:        Backbone attention heads (default 8).
        n_layers:       Backbone transformer layers (default 6).
        regime_dim:     Regime embedding dimension (default 64).
        mc_samples:     MC Dropout samples for ConfidenceHead (default 20).
        dropout:        Global dropout rate.
        redis_client:   Optional Redis for CoT scratchpad publishing.
        cot_enabled:    Whether to run ChainOfThoughtReasoner.
        crisis_gate:    Whether to hard-gate all actions during CRISIS regime.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        regime_dim: int = 64,
        mc_samples: int = 20,
        dropout: float = 0.1,
        redis_client: Optional[Any] = None,
        cot_enabled: bool = True,
        crisis_gate: bool = True,
    ) -> None:
        super().__init__()
        self.cot_enabled = cot_enabled
        self.crisis_gate = crisis_gate

        self.fusion = ModalFusion(dropout=dropout)
        self.backbone = ArgusBackbone(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            input_dim=256,
            regime_dim=regime_dim,
            dropout=dropout,
        )
        self.direction_head = DirectionHead(d_model, dropout)
        self.size_head = SizeHead(d_model, dropout)
        self.timing_head = TimingHead(d_model, dropout)
        self.confidence_head = ConfidenceHead(d_model, dropout=min(dropout * 2, 0.3), mc_samples=mc_samples)

        self.cot = ChainOfThoughtReasoner(
            redis_client=redis_client,
            crisis_gate=crisis_gate,
        ) if cot_enabled else None

        self._n_params = sum(p.numel() for p in self.parameters())
        logger.info("ArgusAI initialised — %.2fM parameters", self._n_params / 1e6)

    @property
    def n_parameters(self) -> int:
        return self._n_params

    def forward(
        self,
        regime_ids: torch.Tensor,
        lob: Optional[torch.Tensor] = None,
        chart: Optional[torch.Tensor] = None,
        sentiment: Optional[torch.Tensor] = None,
        gnn: Optional[torch.Tensor] = None,
        regime_vec: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        symbols: Optional[List[str]] = None,
        volatility: Optional[List[float]] = None,
        spread: Optional[List[float]] = None,
        signal_quality: Optional[List[float]] = None,
        modality_mask: Optional[List[Dict[str, bool]]] = None,
    ) -> ArgusAIOutput:
        """Full forward pass.

        Args:
            regime_ids:     (B,) integer regime labels [0-3].
            lob:            (B, T, 128) or None.
            chart:          (B, T, 64)  or None.
            sentiment:      (B, 64) or None.  (static per bar)
            gnn:            (B, T, 64)  or None.
            regime_vec:     (B, 16)     or None.
            mask:           Optional attention bias.
            symbols:        List of symbol strings for CoT.
            volatility:     List of volatility floats for CoT.
            spread:         List of spread floats for CoT.
            signal_quality: List of signal quality scores for CoT.
            modality_mask:  Per-sample modality availability dicts.

        Returns:
            ArgusAIOutput dataclass.
        """
        fused = self.fusion(
            lob=lob,
            chart=chart,
            sentiment=sentiment,
            gnn=gnn,
            regime=regime_vec,
        )
        if fused.dim() == 2:
            fused = fused.unsqueeze(1)

        backbone_out = self.backbone(fused, regime_ids, mask)
        last_token = backbone_out[:, -1, :]

        dir_out = self.direction_head(last_token)
        size_out = self.size_head(last_token)
        timing_out = self.timing_head(last_token)
        conf_out = self.confidence_head(last_token)

        cot_results: Optional[List[CoTScratchpad]] = None
        if self.cot_enabled and self.cot is not None and symbols is not None:
            cot_results = []
            B = last_token.shape[0]
            for i in range(B):
                pad = self.cot.reason(
                    symbol=symbols[i] if i < len(symbols) else "UNKNOWN",
                    regime_id=int(regime_ids[i].item()),
                    volatility=volatility[i] if volatility else 0.01,
                    spread=spread[i] if spread else 0.001,
                    signal_quality=signal_quality[i] if signal_quality else 1.0,
                    direction_probs=dir_out.probs[i].detach().tolist(),
                    confidence_mean=float(conf_out.mean[i].item()),
                    confidence_std=float(conf_out.std[i].item()),
                    modality_mask=modality_mask[i] if modality_mask else None,
                )
                cot_results.append(pad)

        return ArgusAIOutput(
            direction_logits=dir_out.logits,
            direction_probs=dir_out.probs,
            direction_action=dir_out.action,
            size_alpha=size_out.alpha,
            size_beta=size_out.beta,
            size_mean=size_out.mean_size,
            timing_delay=timing_out.delay_ticks,
            confidence_mean=conf_out.mean,
            confidence_std=conf_out.std,
            backbone_repr=last_token,
            cot=cot_results,
        )
