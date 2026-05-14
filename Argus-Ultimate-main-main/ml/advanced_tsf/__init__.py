"""
Advanced time-series forecasting models.

Currently ships:
- ``itransformer``: Inverted Transformer (arXiv:2310.06625) — inverts the
  role of tokens and feature dimensions for long-horizon forecasting.

Design note per Plan-agent review
---------------------------------
Autoformer and Informer were DROPPED — iTransformer empirically beats both
on long-horizon forecasts. One strong option > three mediocre ones.
"""

from .itransformer import ITransformer, itransformer_forecast

__all__ = ["ITransformer", "itransformer_forecast"]
