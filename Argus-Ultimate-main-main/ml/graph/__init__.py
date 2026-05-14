"""
Classical Graph Neural Networks for ARGUS.

- ``gcn``: Graph Convolutional Network (Kipf & Welling, 2017)
- ``gat``: Graph Attention Network (Veličković et al., 2018)

Both take an adjacency matrix (typically the asset correlation graph) and
node features, and produce updated node embeddings used as an additional
signal source in ``ml/ensemble_signal_hub.py``.

All implementations are pure-numpy with optional torch acceleration via the
existing ``TORCH_DEVICE`` pattern. Graceful degradation is preserved.
"""

from .gcn import GCN, gcn_forward
from .gat import GAT, gat_forward

__all__ = ["GCN", "gcn_forward", "GAT", "gat_forward"]
