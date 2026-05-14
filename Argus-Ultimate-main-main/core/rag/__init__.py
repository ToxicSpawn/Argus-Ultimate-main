"""
Retrieval-Augmented Generation (RAG) for ARGUS.

Provides a long-term searchable memory over past trades, drawdowns, and
market events. Every ARGUS module gets a "have I seen this setup before?"
query primitive.

- ``embedding_model``: SentenceTransformer with NumPy-only fallback
- ``vector_store``: FAISS-backed (optional) with NumPy L2 fallback
- ``trade_rag``: semantic retrieval over decision_journal + cross_session_memory
- ``event_rag``: retrieval over macro / news events
"""

from .embedding_model import EmbeddingModel, embed_text, embed_context
from .vector_store import VectorStore
from .trade_rag import TradeRAG
from .event_rag import EventRAG

__all__ = [
    "EmbeddingModel",
    "embed_text",
    "embed_context",
    "VectorStore",
    "TradeRAG",
    "EventRAG",
]
