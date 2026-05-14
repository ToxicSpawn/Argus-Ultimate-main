"""
LLM-enhanced sentiment integration for Argus.

Combines optional FinBERT and VADER models with a deterministic rule-based
fallback so sentiment remains available even when NLP dependencies or external
data providers are missing.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, MutableSequence, Optional, Protocol, Sequence, Tuple

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline
except Exception:  # noqa: BLE001
    pipeline = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:  # noqa: BLE001
    SentimentIntensityAnalyzer = None


_VALID_SENTIMENTS = {"positive", "negative", "neutral"}
_VALID_SOURCES = {"news", "social", "earnings", "regulatory"}
_SOURCE_RELIABILITY = {
    "news": 0.95,
    "social": 0.65,
    "earnings": 0.9,
    "regulatory": 0.85,
}
_SOURCE_TIME_DECAY_HALF_LIFE_HOURS = {
    "news": 8.0,
    "social": 3.0,
    "earnings": 24.0,
    "regulatory": 36.0,
}
_POSITIVE_TERMS = {
    "beat", "beats", "growth", "bullish", "approval", "approved", "surge",
    "strong", "strength", "upgrade", "record", "expansion", "buyback",
    "partnership", "adoption", "profit", "profitable", "outperform", "rally",
    "breakout", "optimistic", "tailwind", "innovation", "accelerating",
}
_NEGATIVE_TERMS = {
    "miss", "missed", "bearish", "downgrade", "lawsuit", "ban", "banned",
    "probe", "investigation", "weak", "warning", "decline", "crash", "drop",
    "selloff", "sell-off", "fraud", "default", "hack", "breach", "risk",
    "uncertain", "headwind", "recession", "loss", "losses", "delisting",
}
_SOURCE_HINTS = {
    "social": ("tweet", "twitter", "x.com", "reddit", "r/", "wallstreetbets", "post:"),
    "earnings": ("earnings", "eps", "guidance", "revenue", "ebitda", "quarter", "q1", "q2", "q3", "q4"),
    "regulatory": ("sec", "cftc", "regulator", "regulatory", "compliance", "settlement", "filing", "doj", "fca"),
    "news": ("headline", "breaking", "reports", "reported", "according to"),
}


class SentimentContentProvider(Protocol):
    """Provider contract for news and social content sources."""

    def fetch_content(self, symbol: str, lookback_hours: int) -> Sequence[Any]:
        """Return content items for the requested symbol and lookback window."""


@dataclass(slots=True)
class SentimentScore:
    text: str
    sentiment: str
    confidence: float
    positive_score: float
    negative_score: float
    neutral_score: float
    source: str
    timestamp: datetime

    def __post_init__(self) -> None:
        self.text = str(self.text or "")
        self.sentiment = self.sentiment.lower().strip()
        if self.sentiment not in _VALID_SENTIMENTS:
            self.sentiment = "neutral"
        self.source = self.source.lower().strip() if self.source else "news"
        if self.source not in _VALID_SOURCES:
            self.source = "news"
        self.confidence = float(max(0.0, min(1.0, self.confidence)))
        self.positive_score = float(max(0.0, min(1.0, self.positive_score)))
        self.negative_score = float(max(0.0, min(1.0, self.negative_score)))
        self.neutral_score = float(max(0.0, min(1.0, self.neutral_score)))
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

    @property
    def signed_score(self) -> float:
        return float(max(-1.0, min(1.0, self.positive_score - self.negative_score)))


@dataclass(slots=True)
class _ProviderItem:
    text: str
    source: str
    timestamp: datetime
    symbols: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMEnsembleSentiment:
    """FinBERT-first ensemble sentiment engine with graceful fallbacks."""

    def __init__(
        self,
        *,
        finbert_model_name: str = "ProsusAI/finbert",
        enable_finbert: bool = True,
        enable_vader: bool = True,
        custom_rule_weight: float = 0.2,
        finbert_weight: float = 0.55,
        vader_weight: float = 0.25,
        source_reliability: Optional[Mapping[str, float]] = None,
        news_providers: Optional[Sequence[SentimentContentProvider]] = None,
        social_providers: Optional[Sequence[SentimentContentProvider]] = None,
        signal_pipeline: Optional[Any] = None,
    ) -> None:
        self.finbert_model_name = finbert_model_name
        self.enable_finbert = bool(enable_finbert)
        self.enable_vader = bool(enable_vader)
        self.signal_pipeline = signal_pipeline
        self.news_providers = list(news_providers or [])
        self.social_providers = list(social_providers or [])
        self._source_reliability = dict(_SOURCE_RELIABILITY)
        if source_reliability:
            self._source_reliability.update({str(k): float(v) for k, v in source_reliability.items()})

        weights = {
            "finbert": max(0.0, float(finbert_weight)),
            "vader": max(0.0, float(vader_weight)),
            "rules": max(0.0, float(custom_rule_weight)),
        }
        total = sum(weights.values()) or 1.0
        self._model_weights = {name: value / total for name, value in weights.items()}

        self._finbert = self._load_finbert_model() if self.enable_finbert else None
        self._vader = self._load_vader_model() if self.enable_vader else None
        self._rule_engine = self._load_rule_engine()

    def analyze_text(self, text: str) -> SentimentScore:
        """Analyze a single text item and return a normalized score."""
        normalized_text = str(text or "").strip()
        if not normalized_text:
            now = datetime.now(timezone.utc)
            return SentimentScore(
                text="",
                sentiment="neutral",
                confidence=0.0,
                positive_score=0.0,
                negative_score=0.0,
                neutral_score=1.0,
                source="news",
                timestamp=now,
            )

        source = self._infer_source(normalized_text)
        timestamp = datetime.now(timezone.utc)
        component_scores = self._collect_model_scores(normalized_text)
        positive, negative, neutral, confidence = self._blend_component_scores(component_scores)
        sentiment = self._resolve_sentiment_label(positive, negative, neutral)

        return SentimentScore(
            text=normalized_text,
            sentiment=sentiment,
            confidence=confidence,
            positive_score=positive,
            negative_score=negative,
            neutral_score=neutral,
            source=source,
            timestamp=timestamp,
        )

    def analyze_batch(self, texts: List[str]) -> List[SentimentScore]:
        """Analyze multiple texts in sequence."""
        return [self.analyze_text(text) for text in texts]

    def aggregate_sentiment(self, scores: List[SentimentScore]) -> dict:
        """Aggregate sentiment with source reliability and time decay."""
        if not scores:
            return self._empty_aggregate()

        now = datetime.now(timezone.utc)
        total_weight = 0.0
        signed_total = 0.0
        positive_total = 0.0
        negative_total = 0.0
        neutral_total = 0.0
        confidence_total = 0.0
        source_breakdown: Dict[str, Dict[str, float]] = {}

        for score in scores:
            weight = self._source_weight(score, now)
            total_weight += weight
            signed_value = score.signed_score
            signed_total += signed_value * weight
            positive_total += score.positive_score * weight
            negative_total += score.negative_score * weight
            neutral_total += score.neutral_score * weight
            confidence_total += score.confidence * weight

            bucket = source_breakdown.setdefault(score.source, {"count": 0.0, "weighted_sentiment": 0.0})
            bucket["count"] += 1.0
            bucket["weighted_sentiment"] += signed_value * weight

        if total_weight <= 0.0:
            return self._empty_aggregate()

        net_sentiment = signed_total / total_weight
        avg_positive = positive_total / total_weight
        avg_negative = negative_total / total_weight
        avg_neutral = neutral_total / total_weight
        avg_confidence = confidence_total / total_weight
        dominant = self._resolve_sentiment_label(avg_positive, avg_negative, avg_neutral)
        divergence = self._detect_divergence(scores)

        return {
            "sentiment": dominant,
            "net_sentiment": float(max(-1.0, min(1.0, net_sentiment))),
            "confidence": float(max(0.0, min(1.0, avg_confidence))),
            "positive_score": float(max(0.0, min(1.0, avg_positive))),
            "negative_score": float(max(0.0, min(1.0, avg_negative))),
            "neutral_score": float(max(0.0, min(1.0, avg_neutral))),
            "sample_size": len(scores),
            "source_breakdown": {
                source: {
                    "count": int(values["count"]),
                    "avg_weighted_sentiment": float(values["weighted_sentiment"] / max(values["count"], 1.0)),
                }
                for source, values in source_breakdown.items()
            },
            "divergence_detected": divergence,
        }

    def get_market_sentiment(self, symbol: str, lookback_hours: int) -> dict:
        """Fetch symbol-specific content, analyze it, and prepare pipeline output."""
        normalized_symbol = self._normalize_symbol(symbol)
        items = self._fetch_market_content(normalized_symbol, lookback_hours)
        scores = [self._score_provider_item(item) for item in items]
        aggregate = self.aggregate_sentiment(scores)
        momentum = self.calculate_sentiment_momentum(scores)
        shift = self.detect_sentiment_shift(scores, threshold=0.35)

        result = {
            "symbol": normalized_symbol,
            "lookback_hours": int(lookback_hours),
            "scores": scores,
            "aggregate": aggregate,
            "momentum": momentum,
            "sentiment_shift": shift,
            "pipeline_signal": self._build_signal_payload(normalized_symbol, aggregate, momentum, shift),
        }

        self._push_to_signal_pipeline(result)
        return result

    def calculate_sentiment_momentum(self, scores: List[SentimentScore]) -> float:
        """Measure change between older and newer sentiment windows."""
        if len(scores) < 2:
            return 0.0

        ordered = sorted(scores, key=lambda score: score.timestamp)
        midpoint = max(1, len(ordered) // 2)
        older = ordered[:midpoint]
        newer = ordered[midpoint:]
        if not newer:
            return 0.0

        older_avg = sum(score.signed_score for score in older) / max(len(older), 1)
        newer_avg = sum(score.signed_score for score in newer) / max(len(newer), 1)
        return float(max(-1.0, min(1.0, newer_avg - older_avg)))

    def detect_sentiment_shift(self, scores: List[SentimentScore], threshold: float) -> bool:
        """Detect abrupt changes in prevailing sentiment."""
        if len(scores) < 3:
            return False

        ordered = sorted(scores, key=lambda score: score.timestamp)
        recent = ordered[-3:]
        baseline = ordered[:-3] or ordered[:-1]
        baseline_avg = sum(score.signed_score for score in baseline) / max(len(baseline), 1)
        recent_avg = sum(score.signed_score for score in recent) / max(len(recent), 1)
        return abs(recent_avg - baseline_avg) >= float(threshold)

    def _load_finbert_model(self) -> Optional[Any]:
        if pipeline is None:
            logger.warning("FinBERT sentiment disabled: transformers dependency not available")
            return None
        try:
            return pipeline("text-classification", model=self.finbert_model_name, tokenizer=self.finbert_model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unable to initialize FinBERT model '%s': %s", self.finbert_model_name, exc)
            return None

    def _load_vader_model(self) -> Optional[Any]:
        if SentimentIntensityAnalyzer is None:
            logger.warning("VADER sentiment disabled: vaderSentiment dependency not available")
            return None
        try:
            return SentimentIntensityAnalyzer()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unable to initialize VADER sentiment analyzer: %s", exc)
            return None

    def _load_rule_engine(self) -> Optional[Any]:
        try:
            from ml.finbert_sentiment import CryptoSentimentAnalyzer

            return CryptoSentimentAnalyzer()
        except Exception as exc:  # noqa: BLE001
            logger.warning("CryptoSentimentAnalyzer fallback unavailable: %s", exc)
            return None

    def _collect_model_scores(self, text: str) -> Dict[str, Tuple[float, float, float, float]]:
        scores: Dict[str, Tuple[float, float, float, float]] = {}

        finbert_score = self._score_with_finbert(text)
        if finbert_score is not None:
            scores["finbert"] = finbert_score

        vader_score = self._score_with_vader(text)
        if vader_score is not None:
            scores["vader"] = vader_score

        scores["rules"] = self._score_with_rules(text)
        return scores

    def _blend_component_scores(self, components: Mapping[str, Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
        positive_total = 0.0
        negative_total = 0.0
        neutral_total = 0.0
        confidence_total = 0.0
        weight_total = 0.0

        for name, (positive, negative, neutral, confidence) in components.items():
            weight = self._model_weights.get(name, 0.0)
            if weight <= 0.0:
                continue
            positive_total += positive * weight
            negative_total += negative * weight
            neutral_total += neutral * weight
            confidence_total += confidence * weight
            weight_total += weight

        if weight_total <= 0.0:
            return 0.0, 0.0, 1.0, 0.0

        positive = positive_total / weight_total
        negative = negative_total / weight_total
        neutral = neutral_total / weight_total
        confidence = confidence_total / weight_total
        total = positive + negative + neutral
        if total > 0.0:
            positive /= total
            negative /= total
            neutral /= total
        return (
            float(max(0.0, min(1.0, positive))),
            float(max(0.0, min(1.0, negative))),
            float(max(0.0, min(1.0, neutral))),
            float(max(0.0, min(1.0, confidence))),
        )

    def _score_with_finbert(self, text: str) -> Optional[Tuple[float, float, float, float]]:
        if self._finbert is None:
            return None
        try:
            raw_result = self._finbert(text, truncation=True)
            result = raw_result[0] if isinstance(raw_result, list) else raw_result
            label = str(result.get("label", "neutral")).lower()
            score = float(result.get("score", 0.0))
            if "pos" in label:
                return score, max(0.0, 1.0 - score), 0.0, score
            if "neg" in label:
                return 0.0, score, max(0.0, 1.0 - score), score
            return max(0.0, (1.0 - score) * 0.5), max(0.0, (1.0 - score) * 0.5), score, score
        except Exception as exc:  # noqa: BLE001
            logger.warning("FinBERT inference failed: %s", exc)
            return None

    def _score_with_vader(self, text: str) -> Optional[Tuple[float, float, float, float]]:
        if self._vader is None:
            return None
        try:
            scores = self._vader.polarity_scores(text)
            compound = float(scores.get("compound", 0.0))
            return (
                float(scores.get("pos", 0.0)),
                float(scores.get("neg", 0.0)),
                float(scores.get("neu", 1.0)),
                float(min(1.0, abs(compound))),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("VADER inference failed: %s", exc)
            return None

    def _score_with_rules(self, text: str) -> Tuple[float, float, float, float]:
        if self._rule_engine is not None:
            try:
                result = self._rule_engine.analyze_text(text)
                sentiment_value = float(result.get("sentiment", 0.0) if isinstance(result, Mapping) else getattr(result, "sentiment", 0.0))
                confidence_value = float(result.get("confidence", 0.0) if isinstance(result, Mapping) else getattr(result, "confidence", 0.0))
                positive = max(0.0, sentiment_value)
                negative = max(0.0, -sentiment_value)
                neutral = max(0.0, 1.0 - (positive + negative))
                return positive, negative, neutral, float(max(0.0, min(1.0, confidence_value)))
            except Exception as exc:  # noqa: BLE001
                logger.warning("CryptoSentimentAnalyzer inference failed: %s", exc)

        tokens = re.findall(r"[A-Za-z][A-Za-z\-/]+", text.lower())
        if not tokens:
            return 0.0, 0.0, 1.0, 0.0

        positive_hits = sum(1 for token in tokens if token in _POSITIVE_TERMS)
        negative_hits = sum(1 for token in tokens if token in _NEGATIVE_TERMS)
        total_hits = positive_hits + negative_hits
        if total_hits == 0:
            return 0.0, 0.0, 1.0, 0.1

        positive = positive_hits / total_hits
        negative = negative_hits / total_hits
        neutral = max(0.0, 1.0 - min(1.0, positive + negative))
        confidence = min(1.0, total_hits / 6.0)
        return float(positive), float(negative), float(neutral), float(confidence)

    def _infer_source(self, text: str) -> str:
        lowered = text.lower()
        for source, hints in _SOURCE_HINTS.items():
            if any(hint in lowered for hint in hints):
                return source
        return "news"

    def _resolve_sentiment_label(self, positive: float, negative: float, neutral: float) -> str:
        if neutral >= positive and neutral >= negative:
            return "neutral"
        return "positive" if positive >= negative else "negative"

    def _normalize_symbol(self, symbol: str) -> str:
        normalized = str(symbol or "").upper().replace("/", "").replace("-", "")
        if normalized.endswith("USD") and len(normalized) > 3:
            return normalized[:-3]
        if normalized.endswith("USDT") and len(normalized) > 4:
            return normalized[:-4]
        return normalized

    def _extract_symbols(self, text: str) -> List[str]:
        matches = set()
        for cashtag in re.findall(r"\$([A-Za-z]{2,10})", text):
            matches.add(cashtag.upper())
        for token in re.findall(r"\b[A-Z]{2,10}\b", text):
            matches.add(token.upper())
        return sorted(matches)

    def _fetch_market_content(self, symbol: str, lookback_hours: int) -> List[_ProviderItem]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, int(lookback_hours)))
        items: List[_ProviderItem] = []

        for provider in self.news_providers:
            items.extend(self._normalize_provider_results(provider, symbol, lookback_hours, default_source="news"))
        for provider in self.social_providers:
            items.extend(self._normalize_provider_results(provider, symbol, lookback_hours, default_source="social"))

        filtered: List[_ProviderItem] = []
        for item in items:
            if item.timestamp < cutoff:
                continue
            item_symbols = item.symbols or self._extract_symbols(item.text)
            if item_symbols and symbol and symbol not in {self._normalize_symbol(value) for value in item_symbols}:
                continue
            filtered.append(item)
        return sorted(filtered, key=lambda item: item.timestamp)

    def _normalize_provider_results(
        self,
        provider: SentimentContentProvider,
        symbol: str,
        lookback_hours: int,
        *,
        default_source: str,
    ) -> List[_ProviderItem]:
        try:
            raw_items = provider.fetch_content(symbol, lookback_hours)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sentiment content provider fetch failed for %s: %s", provider.__class__.__name__, exc)
            return []

        normalized: List[_ProviderItem] = []
        for raw_item in raw_items:
            item = self._coerce_provider_item(raw_item, default_source=default_source)
            if item is not None:
                normalized.append(item)
        return normalized

    def _coerce_provider_item(self, raw_item: Any, *, default_source: str) -> Optional[_ProviderItem]:
        if isinstance(raw_item, _ProviderItem):
            return raw_item

        text = ""
        source = default_source
        timestamp = datetime.now(timezone.utc)
        symbols: List[str] = []
        metadata: Dict[str, Any] = {}

        if isinstance(raw_item, Mapping):
            text = str(raw_item.get("text") or raw_item.get("content") or raw_item.get("headline") or raw_item.get("title") or "")
            source = str(raw_item.get("source") or default_source)
            timestamp = self._coerce_datetime(raw_item.get("timestamp"))
            raw_symbols = raw_item.get("symbols", [])
            if isinstance(raw_symbols, str):
                raw_symbols = [raw_symbols]
            symbols = [self._normalize_symbol(value) for value in raw_symbols]
            metadata = dict(raw_item)
        else:
            text = str(getattr(raw_item, "text", "") or getattr(raw_item, "content", "") or getattr(raw_item, "headline", "") or getattr(raw_item, "title", ""))
            source = str(getattr(raw_item, "source", default_source) or default_source)
            timestamp = self._coerce_datetime(getattr(raw_item, "timestamp", None))
            raw_symbols = getattr(raw_item, "symbols", []) or []
            if isinstance(raw_symbols, str):
                raw_symbols = [raw_symbols]
            symbols = [self._normalize_symbol(value) for value in raw_symbols]

        text = text.strip()
        if not text:
            return None

        normalized_source = source.lower().strip()
        if normalized_source not in _VALID_SOURCES:
            normalized_source = default_source

        return _ProviderItem(
            text=text,
            source=normalized_source,
            timestamp=timestamp,
            symbols=symbols,
            metadata=metadata,
        )

    def _coerce_datetime(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str) and value.strip():
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                logger.debug("Unable to parse sentiment timestamp '%s'; using current UTC time", value)
        return datetime.now(timezone.utc)

    def _score_provider_item(self, item: _ProviderItem) -> SentimentScore:
        score = self.analyze_text(item.text)
        return SentimentScore(
            text=score.text,
            sentiment=score.sentiment,
            confidence=score.confidence,
            positive_score=score.positive_score,
            negative_score=score.negative_score,
            neutral_score=score.neutral_score,
            source=item.source,
            timestamp=item.timestamp,
        )

    def _source_weight(self, score: SentimentScore, now: datetime) -> float:
        age_seconds = max(0.0, (now - score.timestamp).total_seconds())
        age_hours = age_seconds / 3600.0
        base_reliability = self._source_reliability.get(score.source, 0.5)
        half_life = _SOURCE_TIME_DECAY_HALF_LIFE_HOURS.get(score.source, 6.0)
        decay = math.exp(-math.log(2.0) * age_hours / max(half_life, 1e-6))
        return float(base_reliability * decay * max(score.confidence, 0.05))

    def _detect_divergence(self, scores: Sequence[SentimentScore]) -> bool:
        if len(scores) < 2:
            return False
        per_source: Dict[str, List[float]] = {}
        for score in scores:
            per_source.setdefault(score.source, []).append(score.signed_score)

        if "news" in per_source and "social" in per_source:
            news_avg = sum(per_source["news"]) / max(len(per_source["news"]), 1)
            social_avg = sum(per_source["social"]) / max(len(per_source["social"]), 1)
            if news_avg * social_avg < 0 and abs(news_avg - social_avg) >= 0.4:
                return True

        source_avgs = [sum(values) / max(len(values), 1) for values in per_source.values() if values]
        return (max(source_avgs) - min(source_avgs)) >= 0.6 if len(source_avgs) >= 2 else False

    def _build_signal_payload(self, symbol: str, aggregate: Mapping[str, Any], momentum: float, shift: bool) -> Dict[str, Any]:
        net_sentiment = float(aggregate.get("net_sentiment", 0.0) or 0.0)
        confidence = float(aggregate.get("confidence", 0.0) or 0.0)
        directional_bias = "bullish" if net_sentiment > 0.1 else "bearish" if net_sentiment < -0.1 else "neutral"
        return {
            "symbol": symbol,
            "sentiment_bias": directional_bias,
            "sentiment_score": net_sentiment,
            "sentiment_confidence": confidence,
            "sentiment_momentum": momentum,
            "sentiment_shift": bool(shift),
            "divergence_detected": bool(aggregate.get("divergence_detected", False)),
            "signal_adjustment": float(max(-0.25, min(0.25, net_sentiment * confidence))),
        }

    def _push_to_signal_pipeline(self, result: Mapping[str, Any]) -> None:
        if self.signal_pipeline is None:
            return
        payload = result.get("pipeline_signal")
        if payload is None:
            return

        for method_name in ("ingest_sentiment_signal", "update_sentiment", "publish_sentiment"):
            method = getattr(self.signal_pipeline, method_name, None)
            if callable(method):
                try:
                    method(payload)
                    return
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Signal pipeline integration failed via %s: %s", method_name, exc)

        if isinstance(self.signal_pipeline, MutableSequence):
            self.signal_pipeline.append(payload)

    def _empty_aggregate(self) -> Dict[str, Any]:
        return {
            "sentiment": "neutral",
            "net_sentiment": 0.0,
            "confidence": 0.0,
            "positive_score": 0.0,
            "negative_score": 0.0,
            "neutral_score": 1.0,
            "sample_size": 0,
            "source_breakdown": {},
            "divergence_detected": False,
        }
