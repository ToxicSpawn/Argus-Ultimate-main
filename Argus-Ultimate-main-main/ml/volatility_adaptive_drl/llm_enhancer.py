"""LLM-style news summarisation with graceful fallbacks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMConfig:
    model_name: str = "heuristic-llm"
    max_headlines: int = 8
    sentiment_fallback_weight: float = 0.6
    narrative_max_chars: int = 320


class LLMEnhancer:
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        llm_callable: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.config = config or LLMConfig()
        self.llm_callable = llm_callable

    def summarize_recent_news(self, news_items: Sequence[Mapping[str, object] | str]) -> str:
        prompt = self._prompt(news_items)
        if self.llm_callable is not None:
            try:
                return self.llm_callable(prompt)[: self.config.narrative_max_chars]
            except Exception as exc:
                logger.warning("LLM summarisation failed, using fallback: %s", exc)
        return self._fallback_summary(news_items)

    def extract_key_events(self, news_items: Sequence[Mapping[str, object] | str]) -> list[str]:
        events: list[str] = []
        for item in news_items[: self.config.max_headlines]:
            text = item if isinstance(item, str) else str(item.get("headline") or item.get("text") or "")
            lowered = text.lower()
            for keyword in ("etf", "hack", "fed", "liquidation", "merger", "listing", "upgrade"):
                if keyword in lowered:
                    events.append(keyword)
        return sorted(set(events))

    def market_narrative(self, news_items: Sequence[Mapping[str, object] | str]) -> str:
        summary = self.summarize_recent_news(news_items)
        events = self.extract_key_events(news_items)
        suffix = f" Catalysts: {', '.join(events)}." if events else ""
        return (summary + suffix)[: self.config.narrative_max_chars]

    def fallback_sentiment_scores(self, news_items: Sequence[Mapping[str, object] | str]) -> np.ndarray:
        scores = []
        for item in news_items[: self.config.max_headlines]:
            text = item if isinstance(item, str) else str(item.get("headline") or item.get("text") or "")
            scores.append(self._score_text(text))
        if not scores:
            return np.zeros(3, dtype=np.float32)
        arr = np.asarray(scores, dtype=np.float32)
        return np.array([float(arr.mean()), float(arr.std()), float(arr.max())], dtype=np.float32)

    def _prompt(self, news_items: Sequence[Mapping[str, object] | str]) -> str:
        snippets = []
        for item in news_items[: self.config.max_headlines]:
            snippets.append(item if isinstance(item, str) else str(item.get("headline") or item.get("text") or ""))
        return "Summarise these market headlines with catalysts and risk signals:\n- " + "\n- ".join(snippets)

    def _fallback_summary(self, news_items: Sequence[Mapping[str, object] | str]) -> str:
        scores = self.fallback_sentiment_scores(news_items)
        polarity = "bullish" if scores[0] > 0.1 else "bearish" if scores[0] < -0.1 else "mixed"
        events = self.extract_key_events(news_items)
        catalysts = ", ".join(events[:4]) if events else "no dominant macro catalyst"
        return f"Market news tone is {polarity} with {catalysts} driving attention."

    @staticmethod
    def _score_text(text: str) -> float:
        text = text.lower()
        positive = sum(text.count(word) for word in ("approval", "surge", "adoption", "upgrade", "beat"))
        negative = sum(text.count(word) for word in ("hack", "ban", "liquidation", "miss", "downgrade"))
        return float(np.tanh((positive - negative) / 4.0))
