"""
ml.feature_engineer — superseded by ml.feature_pipeline
========================================================

This module was previously a placeholder stub.  All feature engineering
in ARGUS is now provided by :mod:`ml.feature_pipeline`, which handles
PCA compression, feature selection, and pipeline transforms.

For backward compatibility the public functions are re-exported here so
that existing imports continue to work::

    from ml.feature_engineer import compress_features  # works
"""

from __future__ import annotations

from ml.feature_pipeline import compress_features, transform_for_model

# Alias for any legacy code referencing SPlusFeatureEngineer
SPlusFeatureEngineer = None  # was never implemented; import guard in __init__.py handles this

__all__ = ["compress_features", "transform_for_model"]
