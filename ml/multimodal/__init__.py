"""
Multi-modal fusion for ARGUS.

Combines multiple input modalities into a single feature vector via
cross-attention:
- Numeric market features (OHLCV, indicators, regime)
- Text embeddings (news headlines, LLM sentiment)
- Graph embeddings (from GCN/GAT on the asset correlation graph)

Used by ``ml/ensemble_signal_hub.py`` to produce a fused signal that
considers all three modalities jointly instead of averaging them.
"""

from .fusion_layer import MultiModalFusion, fuse_modalities

__all__ = ["MultiModalFusion", "fuse_modalities"]
