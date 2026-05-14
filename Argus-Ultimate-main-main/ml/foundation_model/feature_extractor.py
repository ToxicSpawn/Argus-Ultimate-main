"""Representation extraction and downstream transfer helpers."""

# pyright: reportMissingImports=false

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional

import torch

from .model import TradeFoundationModel
from .tokenizer import UniversalMicrostructureTokenizer

logger = logging.getLogger(__name__)

try:
    from core.feature_store import FeatureStore
except Exception:  # pragma: no cover - optional dependency
    FeatureStore = None


@dataclass(slots=True)
class RepresentationOutput:
    """Extracted market-state representation."""

    embedding: torch.Tensor
    pooled_embedding: torch.Tensor
    attention_importance: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FoundationModelFeatureExtractor:
    """Extracts learned representations from the foundation model."""

    def __init__(
        self,
        model: TradeFoundationModel,
        tokenizer: Optional[UniversalMicrostructureTokenizer] = None,
        feature_store: Optional[Any] = None,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer or UniversalMicrostructureTokenizer()
        self.device = torch.device(device)
        self.feature_store = feature_store
        if self.feature_store is None and FeatureStore is not None:
            try:
                self.feature_store = FeatureStore(background=True)
            except TypeError:
                self.feature_store = FeatureStore()

    def _to_tensor(self, token_ids: Sequence[int]) -> torch.Tensor:
        return torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def get_market_state_embeddings(self, token_ids: Sequence[int]) -> RepresentationOutput:
        output = self.model(self._to_tensor(token_ids), return_attention=True)
        hidden_states = output["hidden_states"]
        pooled = hidden_states.mean(dim=1)
        attention_importance = self.feature_importance(output.get("attention_maps"))
        return RepresentationOutput(
            embedding=hidden_states.squeeze(0).cpu(),
            pooled_embedding=pooled.squeeze(0).cpu(),
            attention_importance=attention_importance.cpu() if attention_importance is not None else None,
        )

    def feature_importance(self, attention_maps: Optional[Sequence[torch.Tensor]]) -> Optional[torch.Tensor]:
        if not attention_maps:
            return None
        stacked = torch.stack([layer.mean(dim=1).mean(dim=0) for layer in attention_maps], dim=0)
        return stacked.mean(dim=0).mean(dim=0)

    def get_transfer_features(self, token_ids: Sequence[int]) -> Dict[str, torch.Tensor]:
        representation = self.get_market_state_embeddings(token_ids)
        return {
            "pooled_embedding": representation.pooled_embedding,
            "sequence_embedding": representation.embedding,
            "attention_importance": representation.attention_importance if representation.attention_importance is not None else torch.empty(0),
        }

    def extract_window(self, events: Sequence[Mapping[str, Any]], *, symbol: str = "UNKNOWN") -> RepresentationOutput:
        if not events:
            raise ValueError("events must be non-empty")
        return self.extract_from_events(events, symbol=symbol)

    def push_to_feature_store(self, symbol: str, representation: RepresentationOutput, *, ttl_s: float = 60.0) -> None:
        if self.feature_store is None:
            logger.debug("Feature store unavailable; skipping representation publish for %s", symbol)
            return
        try:
            self.feature_store.set(symbol, "tradefm_embedding", representation.pooled_embedding.tolist(), ttl_s=ttl_s)
            if representation.attention_importance is not None:
                self.feature_store.set(symbol, "tradefm_attention_importance", representation.attention_importance.tolist(), ttl_s=ttl_s)
        except Exception:
            logger.exception("Failed to publish foundation-model features for %s", symbol)

    def extract_from_events(self, events: Sequence[Mapping[str, Any]], *, symbol: str = "UNKNOWN") -> RepresentationOutput:
        token_ids = self.tokenizer.to_token_ids(events)
        representation = self.get_market_state_embeddings(token_ids)
        self.push_to_feature_store(symbol, representation)
        return representation
