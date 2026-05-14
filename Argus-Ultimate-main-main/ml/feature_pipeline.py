"""
Feature pipeline: compress features before classifier/regressor.
Uses PCA when sklearn is available; else identity/truncation. See docs IMPLEMENTABLE_NOW.md §3.5.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

# Optional PCA (lazy import so bot runs without sklearn)
_pca_fit: Any = None
_pca_latent_dim: Optional[int] = None


def _get_pca(latent_dim: int, fit_on: Optional[np.ndarray] = None):
    """Lazy PCA fit. When sklearn available, fit on fit_on and return transformer."""
    global _pca_fit, _pca_latent_dim
    try:
        from sklearn.decomposition import PCA
        if fit_on is not None and fit_on.shape[1] >= latent_dim:
            pca = PCA(n_components=min(latent_dim, fit_on.shape[1], fit_on.shape[0]))
            pca.fit(np.asarray(fit_on, dtype=float))
            _pca_fit = pca
            _pca_latent_dim = latent_dim
            return pca
        return _pca_fit if _pca_latent_dim == latent_dim else None
    except Exception as _e:
        logger.warning("feature_pipeline._get_pca: failed: %s", _e, exc_info=True)
        return None


def compress_features(
    features: Union[np.ndarray, List[float], Sequence[float]],
    latent_dim: Optional[int] = None,
    fit_pca_on: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compress feature vector to latent representation. When sklearn is available and
    fit_pca_on is provided, uses PCA; else identity or truncation.
    """
    arr = np.asarray(features, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if latent_dim is not None and arr.shape[1] > latent_dim:
        pca = _get_pca(latent_dim, fit_on=fit_pca_on if fit_pca_on is not None else arr)
        if pca is not None:
            return np.asarray(pca.transform(arr), dtype=float)
        out: np.ndarray = np.asarray(arr[:, :latent_dim].copy(), dtype=float)
        return out
    return np.asarray(arr.copy(), dtype=float)


def decompress_features(
    latent: Union[np.ndarray, List[float], Sequence[float]],
    original_dim: Optional[int] = None,
) -> np.ndarray:
    """
    Decompress latent to feature space. When PCA was used, inverse_transform;
    else pad with zeros or identity.
    """
    arr = np.asarray(latent, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if _pca_fit is not None and hasattr(_pca_fit, "inverse_transform"):
        return np.asarray(_pca_fit.inverse_transform(arr), dtype=float)
    if original_dim is not None and arr.shape[1] < original_dim:
        out = np.zeros((arr.shape[0], original_dim), dtype=float)
        out[:, : arr.shape[1]] = arr
        return out
    return np.asarray(arr.copy(), dtype=float)


def transform_for_model(
    features: Union[np.ndarray, List[float]],
    use_compression: bool = False,
    latent_dim: Optional[int] = None,
) -> np.ndarray:
    """
    Prepare feature vector for model input. When use_compression is True and
    a real autoencoder is loaded, returns compressed latent; else returns
    features as-is.
    """
    arr = np.asarray(features, dtype=float)
    if use_compression and latent_dim is not None:
        return compress_features(arr, latent_dim=latent_dim)
    return arr
