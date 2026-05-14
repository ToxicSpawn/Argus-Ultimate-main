"""
ml.ensemble — superseded by ml.ensemble_signal_hub
===================================================

This module was previously a placeholder stub.  All ensemble functionality
in ARGUS is now provided by :mod:`ml.ensemble_signal_hub`, which aggregates
signals from FearGreed, LLM, Whale, News, Alpha, Volatility and Funding
sources.

For backward compatibility the main class is re-exported here so that
existing imports continue to work::

    from ml.ensemble import EnsembleSignalHub  # works
"""

from __future__ import annotations

from ml.ensemble_signal_hub import EnsembleSignalHub

# Alias for any legacy code that used 'EnsembleModel'
EnsembleModel = EnsembleSignalHub

__all__ = ["EnsembleSignalHub", "EnsembleModel"]
