"""
Vector store for ARGUS's RAG.

FAISS-backed (when available) with NumPy L2 fallback. Provides:
    store.add(id, vector, metadata)
    store.search(query_vector, k) → [(id, score, metadata), ...]
    store.save(path) / store.load(path)

Designed as a drop-in: if FAISS is installed, large stores are fast;
otherwise the NumPy fallback is O(n) per query but still functional for
n up to ~100k entries.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optional FAISS
_FAISS_AVAILABLE = False
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    faiss = None  # type: ignore


# ═════════════════════════════════════════════════════════════════════════════
# VectorStore
# ═════════════════════════════════════════════════════════════════════════════


class VectorStore:
    """
    Dense vector store with FAISS or NumPy backend.

    Parameters
    ----------
    dim : int
        Vector dimensionality.
    metric : str
        "l2" or "cosine" (cosine = inner product on normalized vectors).
    """

    def __init__(self, dim: int, metric: str = "cosine") -> None:
        self.dim = int(dim)
        self.metric = metric.lower()
        self._ids: List[str] = []
        self._vectors: List[np.ndarray] = []
        self._metadata: List[Dict[str, Any]] = []
        self._id_index: Dict[str, int] = {}

        self._faiss_index: Optional[Any] = None
        if _FAISS_AVAILABLE:
            try:
                if self.metric == "cosine":
                    self._faiss_index = faiss.IndexFlatIP(self.dim)
                else:
                    self._faiss_index = faiss.IndexFlatL2(self.dim)
                logger.info("VectorStore: using FAISS backend (dim=%d, metric=%s)",
                            self.dim, self.metric)
            except Exception as exc:
                logger.debug("FAISS init failed: %s", exc)
                self._faiss_index = None

    @property
    def backend(self) -> str:
        return "faiss" if self._faiss_index is not None else "numpy_l2"

    def __len__(self) -> int:
        return len(self._ids)

    # ── Mutation ────────────────────────────────────────────────────────────

    def add(
        self,
        id: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert a new vector. If id already exists, updates the entry."""
        vec = np.asarray(vector, dtype=np.float32).flatten()
        if vec.shape[0] != self.dim:
            raise ValueError(f"vector dim {vec.shape[0]} != store dim {self.dim}")

        # Normalize for cosine
        if self.metric == "cosine":
            norm = float(np.linalg.norm(vec))
            if norm > 1e-9:
                vec = vec / norm

        if id in self._id_index:
            # Update existing
            idx = self._id_index[id]
            self._vectors[idx] = vec
            self._metadata[idx] = metadata or {}
            if self._faiss_index is not None:
                # FAISS IndexFlat doesn't support update; rebuild
                self._rebuild_faiss()
        else:
            idx = len(self._ids)
            self._id_index[id] = idx
            self._ids.append(id)
            self._vectors.append(vec)
            self._metadata.append(metadata or {})
            if self._faiss_index is not None:
                self._faiss_index.add(vec.reshape(1, -1))

    def add_batch(
        self,
        ids: List[str],
        vectors: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Batch insert."""
        if metadatas is None:
            metadatas = [{} for _ in ids]
        for i, id in enumerate(ids):
            self.add(id, vectors[i], metadatas[i])

    def _rebuild_faiss(self) -> None:
        """Rebuild FAISS index from current vectors."""
        if not _FAISS_AVAILABLE or not self._vectors:
            return
        if self.metric == "cosine":
            self._faiss_index = faiss.IndexFlatIP(self.dim)
        else:
            self._faiss_index = faiss.IndexFlatL2(self.dim)
        mat = np.stack(self._vectors).astype(np.float32)
        self._faiss_index.add(mat)

    # ── Search ──────────────────────────────────────────────────────────────

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Return the top-k nearest neighbors.

        Returns list of (id, score, metadata). Score is higher-is-better
        for cosine (inner product), lower-is-better for L2.
        """
        if len(self._ids) == 0:
            return []

        q = np.asarray(query, dtype=np.float32).flatten()
        if q.shape[0] != self.dim:
            raise ValueError(f"query dim {q.shape[0]} != store dim {self.dim}")

        if self.metric == "cosine":
            norm = float(np.linalg.norm(q))
            if norm > 1e-9:
                q = q / norm

        k = min(k, len(self._ids))

        if self._faiss_index is not None:
            distances, indices = self._faiss_index.search(q.reshape(1, -1), k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self._ids):
                    continue
                # FAISS IndexFlatIP returns higher=better; IndexFlatL2 lower=better.
                # We expose "higher=better" uniformly by negating L2.
                score = float(dist)
                if self.metric == "l2":
                    score = -score
                results.append((self._ids[idx], score, self._metadata[idx]))
            return results

        # NumPy fallback
        mat = np.stack(self._vectors)  # (n, dim)
        if self.metric == "cosine":
            scores = mat @ q  # already-normalized vectors
        else:
            diffs = mat - q
            scores = -np.sum(diffs * diffs, axis=1)  # negative L2 so higher=better

        top_idx = np.argsort(scores)[::-1][:k]
        return [
            (self._ids[int(i)], float(scores[int(i)]), self._metadata[int(i)])
            for i in top_idx
        ]

    def get(self, id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Retrieve a stored entry by id."""
        if id not in self._id_index:
            return None
        idx = self._id_index[id]
        return self._vectors[idx].copy(), dict(self._metadata[idx])

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist the store to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "dim": self.dim,
            "metric": self.metric,
            "ids": self._ids,
            "vectors": [v.tolist() for v in self._vectors],
            "metadata": self._metadata,
        }
        with open(p, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """Load a vector store from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        store = cls(dim=state["dim"], metric=state["metric"])
        for id, vec_list, meta in zip(
            state["ids"], state["vectors"], state["metadata"]
        ):
            store.add(id, np.asarray(vec_list, dtype=np.float32), meta)
        return store
