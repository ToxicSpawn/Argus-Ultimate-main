"""
Embedding model with graceful degradation.

Wraps ``sentence-transformers`` when installed; falls back to a NumPy-only
hashed-n-gram encoder when not. The fallback is much weaker but deterministic
and dependency-free — it produces a fixed-dim vector that roughly clusters
similar strings, good enough for cosine-similarity retrieval of past trade
setups.

API:
    model = EmbeddingModel(dim=128)
    vec = model.encode("BTC/USD trending up high vol")
    vecs = model.encode_batch(["setup A", "setup B"])
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Optional: try to import sentence-transformers
_ST_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _ST_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore


# ═════════════════════════════════════════════════════════════════════════════
# EmbeddingModel
# ═════════════════════════════════════════════════════════════════════════════


class EmbeddingModel:
    """
    Text embedding model with SentenceTransformer (if available) + NumPy
    hashed-n-gram fallback.

    Parameters
    ----------
    dim : int
        Embedding dimensionality. Used for the fallback; SentenceTransformer
        uses its native dim (384 for all-MiniLM-L6-v2).
    model_name : str, optional
        SentenceTransformer model name.
    """

    def __init__(
        self,
        dim: int = 128,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.dim = int(dim)
        self.model_name = model_name
        self._st_model: Optional[Any] = None
        self._backend: str = "hashed_ngram"

        if _ST_AVAILABLE:
            try:
                self._st_model = SentenceTransformer(model_name)
                self._backend = "sentence_transformer"
                self.dim = int(self._st_model.get_sentence_embedding_dimension())
                logger.info("EmbeddingModel: using sentence-transformers %s (dim=%d)",
                            model_name, self.dim)
            except Exception as exc:
                logger.debug("SentenceTransformer load failed: %s", exc)
                self._st_model = None
                self._backend = "hashed_ngram"

    @property
    def backend(self) -> str:
        return self._backend

    def encode(self, text: str) -> np.ndarray:
        """Encode a single string to a unit-norm vector."""
        if self._st_model is not None:
            try:
                vec = self._st_model.encode(text, normalize_embeddings=True)
                return np.asarray(vec, dtype=np.float32)
            except Exception:
                pass
        return self._hashed_ngram_encode(text)

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a batch; returns (n, dim) float32."""
        if self._st_model is not None:
            try:
                vecs = self._st_model.encode(list(texts), normalize_embeddings=True)
                return np.asarray(vecs, dtype=np.float32)
            except Exception:
                pass
        return np.stack([self._hashed_ngram_encode(t) for t in texts])

    def _hashed_ngram_encode(self, text: str, n: int = 3) -> np.ndarray:
        """
        NumPy-only fallback: hash character n-grams into a fixed-dim vector.

        Produces a coarse but deterministic embedding. Similar strings will
        share many n-grams → similar vectors after normalization.
        """
        vec = np.zeros(self.dim, dtype=np.float32)
        if not text:
            return vec
        text_lower = text.lower()
        # Character n-grams
        for i in range(max(0, len(text_lower) - n + 1)):
            gram = text_lower[i : i + n]
            h = int.from_bytes(
                hashlib.sha1(gram.encode("utf-8")).digest()[:4],
                "big",
            )
            idx = h % self.dim
            sign = 1.0 if (h >> 31) & 1 == 0 else -1.0
            vec[idx] += sign
        # Word tokens
        for word in text_lower.split():
            h = int.from_bytes(
                hashlib.sha1(word.encode("utf-8")).digest()[:4],
                "big",
            )
            idx = h % self.dim
            sign = 1.0 if (h >> 31) & 1 == 0 else -1.0
            vec[idx] += sign * 2.0
        # L2-normalize
        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            vec = vec / norm
        return vec


# ═════════════════════════════════════════════════════════════════════════════
# Module-level convenience
# ═════════════════════════════════════════════════════════════════════════════


_DEFAULT_MODEL: Optional[EmbeddingModel] = None


def _get_default() -> EmbeddingModel:
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = EmbeddingModel()
    return _DEFAULT_MODEL


def embed_text(text: str) -> np.ndarray:
    """Convenience wrapper around the default EmbeddingModel."""
    return _get_default().encode(text)


def embed_context(context: Dict[str, Any]) -> np.ndarray:
    """
    Embed a structured context dict (e.g. symbol, regime, volatility, side).

    Serializes the dict into a canonical text form before encoding.
    """
    # Sort keys for stability
    parts = []
    for key in sorted(context.keys()):
        value = context[key]
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        elif isinstance(value, (int, str)):
            parts.append(f"{key}={value}")
        elif isinstance(value, dict):
            parts.append(f"{key}=" + str(sorted(value.items())))
        else:
            parts.append(f"{key}={value}")
    text = " | ".join(parts)
    return _get_default().encode(text)
