"""News and social sentiment scoring with optional FinBERT-style inputs.

This file is dependency-light by design: production can feed it model scores from
FinBERT or another NLP service, while offline tests use the built-in lexicon.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import statistics


POSITIVE = {"beat", "bullish", "upgrade", "adoption", "inflow", "partnership", "approval", "record", "growth", "rally"}
NEGATIVE = {"hack", "bearish", "downgrade", "lawsuit", "outflow", "ban", "liquidation", "fraud", "risk", "selloff"}


@dataclass
class SentimentResult:
    score: float
    label: str
    confidence: float
    drivers: list[str]


class NewsSentimentAnalyzer:
    def score_text(self, text: str, external_score: float | None = None) -> SentimentResult:
        if external_score is not None:
            score = max(min(external_score, 1.0), -1.0)
            return SentimentResult(score, self._label(score), abs(score), ["external_model"])
        words = re.findall(r"[a-zA-Z]+", text.lower())
        pos = [word for word in words if word in POSITIVE]
        neg = [word for word in words if word in NEGATIVE]
        raw = (len(pos) - len(neg)) / max(len(pos) + len(neg), 1)
        confidence = min((len(pos) + len(neg)) / 5, 1.0)
        return SentimentResult(float(raw), self._label(raw), float(confidence), pos + neg)

    def aggregate(self, articles: list[str], external_scores: list[float] | None = None) -> SentimentResult:
        results = [self.score_text(article, None if external_scores is None else external_scores[i]) for i, article in enumerate(articles)]
        if not results:
            return SentimentResult(0.0, "neutral", 0.0, [])
        score = float(statistics.mean(result.score for result in results))
        confidence = float(statistics.mean(result.confidence for result in results))
        drivers = [driver for result in results for driver in result.drivers[:3]]
        return SentimentResult(score, self._label(score), confidence, drivers[:10])

    @staticmethod
    def _label(score: float) -> str:
        if score > 0.2:
            return "bullish"
        if score < -0.2:
            return "bearish"
        return "neutral"


def _demo() -> None:
    analyzer = NewsSentimentAnalyzer()
    articles = ["Bitcoin ETF approval drives record inflow", "Exchange hack sparks liquidation risk"]
    print("News sentiment analyzer ready")
    print(analyzer.aggregate(articles))


if __name__ == "__main__":
    _demo()
