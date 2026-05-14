"""Multimodal market state encoder."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MultimodalState:
    price_features: np.ndarray
    technical_features: np.ndarray
    sentiment_features: np.ndarray
    order_book_features: np.ndarray
    unified_state: np.ndarray
    metadata: dict[str, float] = field(default_factory=dict)


class MultimodalEncoder:
    def __init__(
        self,
        sequence_length: int = 32,
        price_feature_dim: int = 12,
        technical_feature_dim: int = 10,
        sentiment_feature_dim: int = 6,
        order_book_feature_dim: int = 8,
    ) -> None:
        self.sequence_length = int(max(4, sequence_length))
        self.price_feature_dim = price_feature_dim
        self.technical_feature_dim = technical_feature_dim
        self.sentiment_feature_dim = sentiment_feature_dim
        self.order_book_feature_dim = order_book_feature_dim

    @property
    def unified_dim(self) -> int:
        return (
            self.price_feature_dim
            + self.technical_feature_dim
            + self.sentiment_feature_dim
            + self.order_book_feature_dim
        )

    def encode(
        self,
        prices: Sequence[Sequence[float]] | np.ndarray,
        technical_indicators: Optional[Mapping[str, Sequence[float] | float]] = None,
        news_items: Optional[Sequence[Mapping[str, object] | str]] = None,
        order_book: Optional[Mapping[str, Sequence[float] | float]] = None,
    ) -> MultimodalState:
        price_features = self.encode_price_volume(prices)
        technical_features = self.encode_technical_indicators(technical_indicators)
        sentiment_features = self.encode_news_sentiment(news_items)
        order_book_features = self.encode_order_book(order_book)
        unified = np.concatenate(
            [price_features, technical_features, sentiment_features, order_book_features]
        ).astype(np.float32)
        return MultimodalState(
            price_features=price_features,
            technical_features=technical_features,
            sentiment_features=sentiment_features,
            order_book_features=order_book_features,
            unified_state=unified,
            metadata={"unified_dim": float(unified.shape[0])},
        )

    def encode_price_volume(self, prices: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        arr = np.asarray(prices, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[0] == 0:
            return np.zeros(self.price_feature_dim, dtype=np.float32)
        close = arr[:, 3] if arr.shape[1] >= 4 else arr[:, 0]
        volume = arr[:, 4] if arr.shape[1] >= 5 else np.ones_like(close)
        returns = np.diff(np.log(np.maximum(close, 1e-6)), prepend=np.log(max(close[0], 1e-6)))
        features = np.array(
            [
                close[-1],
                float(np.mean(close[-self.sequence_length :])),
                float(np.std(close[-self.sequence_length :])),
                float(np.min(close[-self.sequence_length :])),
                float(np.max(close[-self.sequence_length :])),
                float(returns[-1]),
                float(np.mean(returns[-self.sequence_length :])),
                float(np.std(returns[-self.sequence_length :])),
                float(volume[-1]),
                float(np.mean(volume[-self.sequence_length :])),
                float(np.std(volume[-self.sequence_length :])),
                float(np.corrcoef(close[-min(len(close), self.sequence_length) :], volume[-min(len(volume), self.sequence_length) :])[0, 1])
                if len(close) > 2
                else 0.0,
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)[: self.price_feature_dim]

    def encode_technical_indicators(
        self,
        technical_indicators: Optional[Mapping[str, Sequence[float] | float]],
    ) -> np.ndarray:
        if not technical_indicators:
            return np.zeros(self.technical_feature_dim, dtype=np.float32)
        ordered_keys = sorted(technical_indicators)[: self.technical_feature_dim]
        values = []
        for key in ordered_keys:
            raw = technical_indicators[key]
            if isinstance(raw, (list, tuple, np.ndarray)):
                seq = np.asarray(raw, dtype=np.float32)
                values.append(float(seq[-1]) if seq.size else 0.0)
            else:
                values.append(float(raw))
        padded = np.zeros(self.technical_feature_dim, dtype=np.float32)
        padded[: len(values)] = np.asarray(values, dtype=np.float32)
        return padded

    def encode_news_sentiment(
        self,
        news_items: Optional[Sequence[Mapping[str, object] | str]],
    ) -> np.ndarray:
        if not news_items:
            return np.zeros(self.sentiment_feature_dim, dtype=np.float32)
        sentiments = []
        lengths = []
        urgency = []
        keyword_hits = 0
        for item in news_items:
            if isinstance(item, str):
                text = item
                sentiment = self._heuristic_sentiment(text)
            else:
                text = str(item.get("headline") or item.get("text") or "")
                sentiment = float(item.get("sentiment", self._heuristic_sentiment(text)))
                urgency.append(float(item.get("urgency", 0.0) or 0.0))
            sentiments.append(sentiment)
            lengths.append(len(text.split()))
            keyword_hits += sum(text.lower().count(word) for word in ("etf", "fed", "hack", "liquidation", "upgrade"))
        features = np.array(
            [
                float(np.mean(sentiments)),
                float(np.std(sentiments)),
                float(np.max(sentiments)),
                float(np.min(sentiments)),
                float(np.mean(lengths) if lengths else 0.0),
                float(np.mean(urgency) if urgency else min(keyword_hits / max(len(news_items), 1), 5.0)),
            ],
            dtype=np.float32,
        )
        return features[: self.sentiment_feature_dim]

    def encode_order_book(
        self,
        order_book: Optional[Mapping[str, Sequence[float] | float]],
    ) -> np.ndarray:
        if not order_book:
            return np.zeros(self.order_book_feature_dim, dtype=np.float32)
        bids = np.asarray(order_book.get("bids", []), dtype=np.float32)
        asks = np.asarray(order_book.get("asks", []), dtype=np.float32)
        bid_size = float(np.sum(bids)) if bids.size else float(order_book.get("bid_size", 0.0) or 0.0)
        ask_size = float(np.sum(asks)) if asks.size else float(order_book.get("ask_size", 0.0) or 0.0)
        spread = float(order_book.get("spread", 0.0) or 0.0)
        imbalance = (bid_size - ask_size) / max(bid_size + ask_size, 1e-6)
        slope_bid = float(np.mean(np.diff(bids[:5]))) if bids.size > 1 else 0.0
        slope_ask = float(np.mean(np.diff(asks[:5]))) if asks.size > 1 else 0.0
        features = np.array(
            [
                bid_size,
                ask_size,
                imbalance,
                spread,
                float(order_book.get("microprice", 0.0) or 0.0),
                float(order_book.get("depth_ratio", 0.0) or 0.0),
                slope_bid,
                slope_ask,
            ],
            dtype=np.float32,
        )
        return features[: self.order_book_feature_dim]

    @staticmethod
    def _heuristic_sentiment(text: str) -> float:
        positive = ("beat", "approval", "strong", "surge", "upgrade", "bull")
        negative = ("miss", "hack", "lawsuit", "liquidation", "ban", "bear")
        text_lower = text.lower()
        score = sum(text_lower.count(word) for word in positive) - sum(text_lower.count(word) for word in negative)
        return float(np.tanh(score / 4.0))
