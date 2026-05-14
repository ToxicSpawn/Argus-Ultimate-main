"""
Self-supervised learning (SSL) pretraining for ARGUS.

Currently ships:
- ``masked_return_prediction``: BERT-style masked return reconstruction
  for pretraining the transformer price predictor.

Design note
-----------
Per Plan-agent review, SimCLR/BYOL on returns was DROPPED — augmentation
design for returns is an open research problem. Masked return prediction
is the proven, robust SSL approach for time-series.
"""

from .masked_return_prediction import (
    MaskedReturnDataset,
    MaskedReturnPretrainer,
    mask_returns,
)

__all__ = [
    "MaskedReturnDataset",
    "MaskedReturnPretrainer",
    "mask_returns",
]
