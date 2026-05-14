"""
LLM-powered research agent for institutional-grade market analysis.

Provides news aggregation, sentiment analysis, narrative extraction,
earnings analysis, and automated research report generation.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline
except Exception:  # noqa: BLE001
    pipeline = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:  # noqa: BLE01
    SentimentIntensityAnalyzer = None


# ─── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass(slots=True)
class NewsArticle:
    title: str
    content: str
    source: str
    published_at: datetime
    url: str
    symbols: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.title = str(self.title or "").strip()
        self.content = str(self.content or "").strip()
        self.source = str(self.source or "").strip()
        self.url = str(self.url or "").strip()
        if not self.symbols:
            self.symbols = []
        if self.published_at.tzinfo is None:
            self.published_at = self.published_at.replace(tzinfo=timezone.utc)

    def text_for_analysis(self) -> str:
        return f"{self.title}. {self.content}"

    def article_hash(self) -> str:
        content_key = f"{self.title}|{self.content[:200]}"
        return hashlib.sha256(content_key.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class SentimentResult:
    text: str
    sentiment: str
    confidence: float
    score: float
    key_phrases: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.text = str(self.text or "").strip()
        self.sentiment = str(self.sentiment or "neutral").lower().strip()
        if self.sentiment not in ("positive", "negative", "neutral"):
            self.sentiment = "neutral"
        self.confidence = float(max(0.0, min(1.0, self.confidence)))
        self.score = float(max(-1.0, min(1.0, self.score)))
        if not self.key_phrases:
            self.key_phrases = []


@dataclass(slots=True)
class SentimentAggregate:
    symbol: str
    sentiment: str
    net_score: float
    confidence: float
    sample_size: int
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    score_std: float
    time_range: Tuple[datetime, datetime]
    source_breakdown: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.net_score = float(max(-1.0, min(1.0, self.net_score)))
        self.confidence = float(max(0.0, min(1.0, self.confidence)))
        self.positive_ratio = float(max(0.0, min(1.0, self.positive_ratio)))
        self.negative_ratio = float(max(0.0, min(1.0, self.negative_ratio)))
        self.neutral_ratio = float(max(0.0, min(1.0, self.neutral_ratio)))
        self.score_std = float(max(0.0, self.score_std))


@dataclass(slots=True)
class SentimentShift:
    symbol: str
    previous_sentiment: str
    current_sentiment: str
    magnitude: float
    detected_at: datetime
    trigger_articles: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.magnitude = float(max(-1.0, min(1.0, self.magnitude)))
        if self.detected_at.tzinfo is None:
            self.detected_at = self.detected_at.replace(tzinfo=timezone.utc)


@dataclass(slots=True)
class Theme:
    name: str
    relevance: float
    article_count: int
    sentiment: str
    key_terms: List[str] = field(default_factory=list)
    articles: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = str(self.name or "").strip()
        self.relevance = float(max(0.0, min(1.0, self.relevance)))
        self.sentiment = str(self.sentiment or "neutral").lower().strip()
        if self.sentiment not in ("positive", "negative", "neutral"):
            self.sentiment = "neutral"


@dataclass(slots=True)
class Catalyst:
    event_type: str
    description: str
    impact: str
    probability: float
    timeframe: str
    related_symbols: List[str] = field(default_factory=list)
    source_article: str = ""

    def __post_init__(self) -> None:
        self.event_type = str(self.event_type or "").strip()
        self.description = str(self.description or "").strip()
        self.impact = str(self.impact or "").strip()
        self.probability = float(max(0.0, min(1.0, self.probability)))
        self.timeframe = str(self.timeframe or "").strip()


@dataclass(slots=True)
class NarrativeTimeline:
    symbol: str
    periods: List[NarrativePeriod] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.periods:
            self.periods = []


@dataclass(slots=True)
class NarrativePeriod:
    start_date: datetime
    end_date: datetime
    dominant_theme: str
    sentiment: str
    article_count: int
    key_events: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.start_date.tzinfo is None:
            self.start_date = self.start_date.replace(tzinfo=timezone.utc)
        if self.end_date.tzinfo is None:
            self.end_date = self.end_date.replace(tzinfo=timezone.utc)


@dataclass(slots=True)
class EarningsSummary:
    symbol: str
    period: str
    revenue: Optional[float]
    earnings_per_share: Optional[float]
    net_income: Optional[float]
    key_highlights: List[str] = field(default_factory=list)
    risks_mentioned: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.symbol = str(self.symbol or "").strip()
        self.period = str(self.period or "").strip()
        if not self.key_highlights:
            self.key_highlights = []
        if not self.risks_mentioned:
            self.risks_mentioned = []


@dataclass(slots=True)
class Guidance:
    symbol: str
    period: str
    revenue_guidance: Optional[Tuple[float, float]]
    eps_guidance: Optional[Tuple[float, float]]
    qualitative_guidance: List[str] = field(default_factory=list)
    tone: str = "neutral"

    def __post_init__(self) -> None:
        self.symbol = str(self.symbol or "").strip()
        self.period = str(self.period or "").strip()
        self.tone = str(self.tone or "neutral").lower().strip()
        if self.tone not in ("positive", "negative", "neutral"):
            self.tone = "neutral"
        if not self.qualitative_guidance:
            self.qualitative_guidance = []


@dataclass(slots=True)
class ComparisonResult:
    symbol: str
    period: str
    actual_revenue: Optional[float]
    estimated_revenue: Optional[float]
    revenue_surprise_pct: Optional[float]
    actual_eps: Optional[float]
    estimated_eps: Optional[float]
    eps_surprise_pct: Optional[float]
    beat_or_miss: str

    def __post_init__(self) -> None:
        self.symbol = str(self.symbol or "").strip()
        self.period = str(self.period or "").strip()
        self.beat_or_miss = str(self.beat_or_miss or "neutral").lower().strip()
        if self.beat_or_miss not in ("beat", "miss", "inline"):
            self.beat_or_miss = "inline"


@dataclass(slots=True)
class ResearchReport:
    symbol: str
    generated_at: datetime
    summary: str
    sentiment_analysis: SentimentAggregate
    themes: List[Theme] = field(default_factory=list)
    catalysts: List[Catalyst] = field(default_factory=list)
    technical_context: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    raw_news: List[NewsArticle] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.generated_at.tzinfo is None:
            self.generated_at = self.generated_at.replace(tzinfo=timezone.utc)


@dataclass(slots=True)
class DailyBriefing:
    date: datetime
    symbols: List[str]
    market_overview: str
    symbol_reports: Dict[str, ResearchReport] = field(default_factory=dict)
    top_themes: List[Theme] = field(default_factory=list)
    urgent_catalysts: List[Catalyst] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.date.tzinfo is None:
            self.date = self.date.replace(tzinfo=timezone.utc)


@dataclass(slots=True)
class SymbolAnalysis:
    symbol: str
    analyzed_at: datetime
    sentiment: SentimentAggregate
    themes: List[Theme]
    catalysts: List[Catalyst]
    narrative_timeline: Optional[NarrativeTimeline]
    earnings_summary: Optional[EarningsSummary]
    key_insights: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.analyzed_at.tzinfo is None:
            self.analyzed_at = self.analyzed_at.replace(tzinfo=timezone.utc)


@dataclass(slots=True)
class Alert:
    symbol: str
    alert_type: str
    message: str
    severity: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = str(self.symbol or "").strip()
        self.alert_type = str(self.alert_type or "").strip()
        self.message = str(self.message or "").strip()
        self.severity = str(self.severity or "info").lower().strip()
        if self.severity not in ("info", "warning", "critical"):
            self.severity = "info"
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)


@dataclass(slots=True)
class Insight:
    description: str
    confidence: float
    impact: str
    timeframe: str
    related_symbols: List[str] = field(default_factory=list)
    supporting_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.description = str(self.description or "").strip()
        self.confidence = float(max(0.0, min(1.0, self.confidence)))
        self.impact = str(self.impact or "").strip()
        self.timeframe = str(self.timeframe or "").strip()


# ─── Protocols ─────────────────────────────────────────────────────────────────


class NewsFetchFn(Protocol):
    async def __call__(self, symbols: List[str], hours: int) -> List[NewsArticle]: ...


# ─── NewsAggregator ────────────────────────────────────────────────────────────


class NewsAggregator:
    """Aggregates news from multiple sources with deduplication and filtering."""

    def __init__(self) -> None:
        self._sources: Dict[str, NewsFetchFn] = {}
        logger.info("NewsAggregator initialized")

    def add_source(self, name: str, fetch_fn: NewsFetchFn) -> None:
        if not name or not callable(fetch_fn):
            logger.warning("Invalid news source: name=%s, callable=%s", name, callable(fetch_fn))
            return
        self._sources[name] = fetch_fn
        logger.info("Added news source: %s", name)

    async def fetch_news(self, symbols: List[str], hours: int = 24) -> List[NewsArticle]:
        if not symbols:
            return []

        all_articles: List[NewsArticle] = []
        for source_name, fetch_fn in self._sources.items():
            try:
                articles = await fetch_fn(symbols, hours)
                if articles:
                    for article in articles:
                        if source_name not in article.source.lower():
                            article.source = f"{article.source} ({source_name})".strip(" ()")
                    all_articles.extend(articles)
                    logger.debug("Fetched %d articles from %s", len(articles), source_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to fetch news from %s: %s", source_name, exc)

        logger.info("Fetched %d total articles for %s", len(all_articles), symbols)
        return all_articles

    def deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        if not articles:
            return []

        seen_hashes: set[str] = set()
        unique: List[NewsArticle] = []
        for article in articles:
            h = article.article_hash()
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique.append(article)

        removed = len(articles) - len(unique)
        if removed > 0:
            logger.info("Deduplicated %d articles (%d removed)", len(articles), removed)
        return unique

    def filter_by_relevance(
        self,
        articles: List[NewsArticle],
        symbols: List[str],
        min_relevance: float = 0.5,
    ) -> List[NewsArticle]:
        if not articles or not symbols:
            return articles

        symbols_upper = {s.upper() for s in symbols}
        relevant: List[NewsArticle] = []
        for article in articles:
            article_symbols = {s.upper() for s in article.symbols}
            symbol_overlap = len(symbols_upper & article_symbols)
            if symbol_overlap == 0:
                text = f"{article.title} {article.content}".lower()
                symbol_mentions = sum(1 for s in symbols_upper if s.lower() in text)
                relevance = min(1.0, symbol_mentions / max(len(symbols), 1))
            else:
                relevance = min(1.0, symbol_overlap / max(len(symbols), 1) + 0.3)

            if relevance >= min_relevance:
                relevant.append(article)

        logger.info(
            "Filtered %d articles by relevance (min=%.2f), kept %d",
            len(articles),
            min_relevance,
            len(relevant),
        )
        return relevant


# ─── LLMSentimentAnalyzer ──────────────────────────────────────────────────────


class LLMSentimentAnalyzer:
    """Sentiment analysis with FinBERT, VADER, and rule-based fallback."""

    _POSITIVE_PHRASES = [
        "beat expectations", "exceeded guidance", "strong growth", "bullish outlook",
        "record revenue", "partnership announced", "product launch", "upgrade",
        "buyback program", "dividend increase", "market expansion", "innovation",
        "accelerating", "tailwind", "outperform", "rally", "breakout",
    ]
    _NEGATIVE_PHRASES = [
        "missed estimates", "lowered guidance", "weak demand", "bearish outlook",
        "revenue decline", "lawsuit filed", "regulatory probe", "downgrade",
        "layoffs", "restructuring", "headwind", "recession fears", "delisting",
        "fraud investigation", "data breach", "supply chain disruption",
    ]

    def __init__(
        self,
        *,
        enable_finbert: bool = True,
        enable_vader: bool = True,
        finbert_model_name: str = "ProsusAI/finbert",
    ) -> None:
        self._finbert = self._load_finbert(finbert_model_name) if enable_finbert else None
        self._vader = self._load_vader() if enable_vader else None
        logger.info(
            "LLMSentimentAnalyzer initialized (finbert=%s, vader=%s)",
            self._finbert is not None,
            self._vader is not None,
        )

    @staticmethod
    def _load_finbert(model_name: str) -> Optional[Any]:
        if pipeline is None:
            logger.warning("FinBERT unavailable: transformers not installed")
            return None
        try:
            return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FinBERT load failed (%s): %s", model_name, exc)
            return None

    @staticmethod
    def _load_vader() -> Optional[Any]:
        if SentimentIntensityAnalyzer is None:
            logger.warning("VADER unavailable: vaderSentiment not installed")
            return None
        try:
            return SentimentIntensityAnalyzer()
        except Exception as exc:  # noqa: BLE001
            logger.warning("VADER load failed: %s", exc)
            return None

    def analyze(self, article: NewsArticle) -> SentimentResult:
        text = article.text_for_analysis()
        if not text:
            return SentimentResult(
                text="", sentiment="neutral", confidence=0.0, score=0.0,
            )

        components = self._collect_scores(text)
        score, confidence = self._blend_scores(components)
        sentiment = self._label_from_score(score)
        phrases = self._extract_key_phrases(text)

        return SentimentResult(
            text=text[:500],
            sentiment=sentiment,
            confidence=confidence,
            score=score,
            key_phrases=phrases,
        )

    def batch_analyze(self, articles: List[NewsArticle]) -> List[SentimentResult]:
        return [self.analyze(article) for article in articles]

    def compute_aggregate_sentiment(
        self,
        symbol: str,
        articles: List[NewsArticle],
    ) -> SentimentAggregate:
        if not articles:
            return SentimentAggregate(
                symbol=symbol, sentiment="neutral", net_score=0.0,
                confidence=0.0, sample_size=0, positive_ratio=0.0,
                negative_ratio=0.0, neutral_ratio=1.0, score_std=0.0,
                time_range=(datetime.now(timezone.utc), datetime.now(timezone.utc)),
            )

        results = self.batch_analyze(articles)
        scores = [r.score for r in results]
        sentiments = [r.sentiment for r in results]

        n = len(scores)
        mean_score = sum(scores) / n
        variance = sum((s - mean_score) ** 2 for s in scores) / max(n - 1, 1)
        std = variance ** 0.5

        positive_count = sum(1 for s in sentiments if s == "positive")
        negative_count = sum(1 for s in sentiments if s == "negative")
        neutral_count = sum(1 for s in sentiments if s == "neutral")

        timestamps = [a.published_at for a in articles]
        min_ts = min(timestamps)
        max_ts = max(timestamps)

        source_counts: Dict[str, int] = defaultdict(int)
        for article in articles:
            source_counts[article.source] += 1

        return SentimentAggregate(
            symbol=symbol,
            sentiment=self._label_from_score(mean_score),
            net_score=mean_score,
            confidence=min(1.0, sum(r.confidence for r in results) / n),
            sample_size=n,
            positive_ratio=positive_count / n,
            negative_ratio=negative_count / n,
            neutral_ratio=neutral_count / n,
            score_std=std,
            time_range=(min_ts, max_ts),
            source_breakdown=dict(source_counts),
        )

    def detect_sentiment_shift(
        self,
        history: List[Tuple[datetime, SentimentAggregate]],
    ) -> Optional[SentimentShift]:
        if len(history) < 2:
            return None

        sorted_history = sorted(history, key=lambda x: x[0])
        current = sorted_history[-1][1]
        previous = sorted_history[-2][1]

        magnitude = current.net_score - previous.net_score
        if abs(magnitude) < 0.25:
            return None

        if current.sentiment != previous.sentiment or abs(magnitude) >= 0.4:
            return SentimentShift(
                symbol=current.symbol,
                previous_sentiment=previous.sentiment,
                current_sentiment=current.sentiment,
                magnitude=magnitude,
                detected_at=datetime.now(timezone.utc),
            )
        return None

    def _collect_scores(self, text: str) -> Dict[str, Tuple[float, float]]:
        scores: Dict[str, Tuple[float, float]] = {}
        finbert = self._score_finbert(text)
        if finbert is not None:
            scores["finbert"] = finbert
        vader = self._score_vader(text)
        if vader is not None:
            scores["vader"] = vader
        scores["rules"] = self._score_rules(text)
        return scores

    def _score_finbert(self, text: str) -> Optional[Tuple[float, float]]:
        if self._finbert is None:
            return None
        try:
            result = self._finbert(text[:512], truncation=True)
            item = result[0] if isinstance(result, list) else result
            label = str(item.get("label", "")).lower()
            score = float(item.get("score", 0.0))
            if "pos" in label:
                return score, score
            if "neg" in label:
                return -score, score
            return 0.0, score
        except Exception as exc:  # noqa: BLE001
            logger.warning("FinBERT inference failed: %s", exc)
            return None

    def _score_vader(self, text: str) -> Optional[Tuple[float, float]]:
        if self._vader is None:
            return None
        try:
            vs = self._vader.polarity_scores(text)
            compound = float(vs.get("compound", 0.0))
            confidence = min(1.0, abs(compound))
            return compound, confidence
        except Exception as exc:  # noqa: BLE001
            logger.warning("VADER inference failed: %s", exc)
            return None

    def _score_rules(self, text: str) -> Tuple[float, float]:
        lower = text.lower()
        pos_hits = sum(1 for p in self._POSITIVE_PHRASES if p in lower)
        neg_hits = sum(1 for n in self._NEGATIVE_PHRASES if n in lower)
        total = pos_hits + neg_hits
        if total == 0:
            return 0.0, 0.1
        score = (pos_hits - neg_hits) / total
        confidence = min(1.0, total / 4.0)
        return score, confidence

    def _blend_scores(self, components: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
        if not components:
            return 0.0, 0.0

        weights = {"finbert": 0.55, "vader": 0.25, "rules": 0.20}
        score_total = 0.0
        conf_total = 0.0
        w_total = 0.0

        for name, (score, conf) in components.items():
            w = weights.get(name, 0.0)
            if w <= 0:
                continue
            score_total += score * w
            conf_total += conf * w
            w_total += w

        if w_total <= 0:
            return 0.0, 0.0
        return score_total / w_total, conf_total / w_total

    @staticmethod
    def _label_from_score(score: float) -> str:
        if score > 0.1:
            return "positive"
        if score < -0.1:
            return "negative"
        return "neutral"

    @staticmethod
    def _extract_key_phrases(text: str) -> List[str]:
        phrases: List[str] = []
        sentences = re.split(r"[.!?]+", text)
        for sentence in sentences:
            sentence = sentence.strip()
            if 10 < len(sentence) < 150:
                score_words = sum(
                    1 for w in sentence.lower().split()
                    if w in {"beat", "surge", "growth", "strong", "record", "miss", "decline", "weak", "crash", "risk"}
                )
                if score_words >= 1:
                    phrases.append(sentence[:100])
        return phrases[:10]


# ─── MarketNarrativeExtractor ──────────────────────────────────────────────────


class MarketNarrativeExtractor:
    """Extracts themes, catalysts, and narratives from news articles."""

    _THEME_PATTERNS = {
        "AI/ML Adoption": ["artificial intelligence", "machine learning", "ai", "neural", "deep learning", "llm"],
        "Regulatory": ["sec", "regulation", "compliance", "enforcement", "fine", "penalty", "lawsuit"],
        "M&A Activity": ["merger", "acquisition", "acquire", "buyout", "takeover", "consolidation"],
        "Supply Chain": ["supply chain", "shortage", "disruption", "logistics", "semiconductor", "chip"],
        "Macro Economy": ["interest rate", "inflation", "fed", "recession", "gdp", "employment", "cpi"],
        "ESG": ["esg", "sustainability", "carbon", "emissions", "renewable", "green", "climate"],
        "Product Launch": ["launch", "release", "unveil", "announce", "new product", "beta"],
        "Earnings": ["earnings", "revenue", "eps", "guidance", "profit", "margin", "quarterly"],
    }

    _CATALYST_TYPES = {
        "earnings": ["earnings", "eps", "revenue", "guidance", "quarter"],
        "regulatory": ["sec filing", "approval", "ban", "regulation", "compliance"],
        "corporate_action": ["merger", "acquisition", "spinoff", "split", "buyback", "dividend"],
        "product": ["launch", "release", "update", "version", "beta", "ga"],
        "macro": ["fed", "rate", "cpi", "jobs", "gdp", "inflation"],
        "partnership": ["partnership", "collaboration", "deal", "agreement", "joint venture"],
    }

    def __init__(self, sentiment_analyzer: Optional[LLMSentimentAnalyzer] = None) -> None:
        self._sentiment_analyzer = sentiment_analyzer or LLMSentimentAnalyzer()
        logger.info("MarketNarrativeExtractor initialized")

    def extract_themes(self, articles: List[NewsArticle]) -> List[Theme]:
        if not articles:
            return []

        theme_articles: Dict[str, List[NewsArticle]] = defaultdict(list)
        for article in articles:
            text = f"{article.title} {article.content}".lower()
            for theme_name, keywords in self._THEME_PATTERNS.items():
                if any(kw in text for kw in keywords):
                    theme_articles[theme_name].append(article)

        themes: List[Theme] = []
        for theme_name, related_articles in theme_articles.items():
            sentiment_results = self._sentiment_analyzer.batch_analyze(related_articles)
            avg_score = sum(r.score for r in sentiment_results) / len(sentiment_results)
            key_terms = self._extract_theme_terms(related_articles, theme_name)
            themes.append(Theme(
                name=theme_name,
                relevance=min(1.0, len(related_articles) / max(len(articles), 1) * 2),
                article_count=len(related_articles),
                sentiment=self._sentiment_analyzer._label_from_score(avg_score),
                key_terms=key_terms,
                articles=[a.title[:80] for a in related_articles[:5]],
            ))

        themes.sort(key=lambda t: t.relevance, reverse=True)
        logger.info("Extracted %d themes from %d articles", len(themes), len(articles))
        return themes

    def identify_catalysts(self, articles: List[NewsArticle]) -> List[Catalyst]:
        if not articles:
            return []

        catalysts: List[Catalyst] = []
        for article in articles:
            text = f"{article.title} {article.content}".lower()
            for cat_type, keywords in self._CATALYST_TYPES.items():
                if any(kw in text for kw in keywords):
                    impact = self._assess_catalyst_impact(article, cat_type)
                    probability = self._assess_catalyst_probability(article, cat_type)
                    timeframe = self._infer_timeframe(article)
                    catalysts.append(Catalyst(
                        event_type=cat_type,
                        description=article.title[:200],
                        impact=impact,
                        probability=probability,
                        timeframe=timeframe,
                        related_symbols=article.symbols,
                        source_article=article.url,
                    ))
                    break

        catalysts.sort(key=lambda c: c.probability, reverse=True)
        logger.info("Identified %d catalysts from %d articles", len(catalysts), len(articles))
        return catalysts

    def generate_summary(self, symbol: str, articles: List[NewsArticle]) -> str:
        if not articles:
            return f"No recent news available for {symbol}."

        sorted_articles = sorted(articles, key=lambda a: a.published_at, reverse=True)
        recent = sorted_articles[:10]

        sentiment_results = self._sentiment_analyzer.batch_analyze(recent)
        avg_score = sum(r.score for r in sentiment_results) / len(sentiment_results)
        sentiment_label = self._sentiment_analyzer._label_from_score(avg_score)

        themes = self.extract_themes(recent)
        top_themes = themes[:3] if themes else []

        lines = [
            f"Market Summary for {symbol}",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            f"Overall Sentiment: {sentiment_label.upper()} (score: {avg_score:.2f})",
            f"Articles Analyzed: {len(recent)}",
            "",
        ]

        if top_themes:
            lines.append("Dominant Themes:")
            for theme in top_themes:
                lines.append(f"  - {theme.name} ({theme.article_count} articles, {theme.sentiment})")
            lines.append("")

        lines.append("Recent Headlines:")
        for article in recent[:5]:
            lines.append(f"  [{article.published_at.strftime('%Y-%m-%d')}] {article.title}")

        return "\n".join(lines)

    def track_narrative_evolution(
        self,
        symbol: str,
        days: int = 30,
    ) -> NarrativeTimeline:
        periods: List[NarrativePeriod] = []
        now = datetime.now(timezone.utc)
        period_length = timedelta(days=7)
        cutoff = now - timedelta(days=days)

        current_start = cutoff
        while current_start < now:
            current_end = min(current_start + period_length, now)
            period = NarrativePeriod(
                start_date=current_start,
                end_date=current_end,
                dominant_theme="No Data",
                sentiment="neutral",
                article_count=0,
            )
            periods.append(period)
            current_start = current_end

        return NarrativeTimeline(symbol=symbol, periods=periods)

    @staticmethod
    def _extract_theme_terms(articles: List[NewsArticle], theme_name: str) -> List[str]:
        text = " ".join(f"{a.title} {a.content}" for a in articles).lower()
        words = re.findall(r"\b[a-z]{4,}\b", text)
        stop_words = {"this", "that", "with", "from", "have", "been", "were", "will", "would", "could", "should"}
        freq: Dict[str, int] = defaultdict(int)
        for w in words:
            if w not in stop_words:
                freq[w] += 1
        return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]]

    @staticmethod
    def _assess_catalyst_impact(article: NewsArticle, cat_type: str) -> str:
        text = f"{article.title} {article.content}".lower()
        high_impact_words = {"major", "significant", "critical", "breaking", "urgent", "massive"}
        if any(w in text for w in high_impact_words):
            return "high"
        medium_impact_words = {"important", "notable", "key", "substantial", "considerable"}
        if any(w in text for w in medium_impact_words):
            return "medium"
        return "low"

    @staticmethod
    def _assess_catalyst_probability(article: NewsArticle, cat_type: str) -> float:
        text = f"{article.title} {article.content}".lower()
        confirmed_words = {"confirmed", "announced", "approved", "completed", "signed", "filed"}
        if any(w in text for w in confirmed_words):
            return 0.9
        probable_words = {"expected", "likely", "anticipated", "scheduled", "planned"}
        if any(w in text for w in probable_words):
            return 0.7
        speculative_words = {"rumored", "reported", "sources say", "may", "could", "might"}
        if any(w in text for w in speculative_words):
            return 0.4
        return 0.5

    @staticmethod
    def _infer_timeframe(article: NewsArticle) -> str:
        text = f"{article.title} {article.content}".lower()
        if any(w in text for w in {"today", "this morning", "just announced"}):
            return "immediate"
        if any(w in text for w in {"this week", "upcoming", "soon"}):
            return "short-term"
        if any(w in text for w in {"next quarter", "next month", "upcoming quarter"}):
            return "medium-term"
        if any(w in text for w in {"next year", "long-term", "future", "multi-year"}):
            return "long-term"
        return "unspecified"


# ─── EarningsAnalyzer ──────────────────────────────────────────────────────────


class EarningsAnalyzer:
    """Analyzes earnings call transcripts and compares to expectations."""

    _GUIDANCE_PATTERNS = [
        r"(?:expect|anticipate|project|forecast|guide)\s+(?:revenue|sales)\s+(?:of |to be )?[\$€£]?([\d,.]+)",
        r"(?:expect|anticipate|project|forecast|guide)\s+(?:eps|earnings per share)\s+(?:of |to be )?[\$€£]?([\d,.]+)",
        r"(?:revenue|sales)\s+(?:growth|increase)\s+(?:of |by )?(\d+)%",
        r"(?:margin|operating margin)\s+(?:of |to be |at )?(\d+)%",
    ]

    _RISK_KEYWORDS = [
        "risk", "uncertainty", "challenge", "headwind", "concern", "pressure",
        "volatility", "competitive", "regulatory", "litigation", "investigation",
        "impairment", "restructuring", "layoff", "disruption",
    ]

    _POSITIVE_KEYWORDS = [
        "strong", "growth", "exceeded", "beat", "record", "momentum",
        "accelerating", "expanding", "innovation", "opportunity", "tailwind",
        "market share", "margin expansion", "efficiency", "optimization",
    ]

    def __init__(self, sentiment_analyzer: Optional[LLMSentimentAnalyzer] = None) -> None:
        self._sentiment_analyzer = sentiment_analyzer or LLMSentimentAnalyzer()
        logger.info("EarningsAnalyzer initialized")

    def parse_earnings_call(self, transcript: str) -> EarningsSummary:
        if not transcript:
            return EarningsSummary(symbol="", period="", revenue=None, earnings_per_share=None, net_income=None)

        symbol = self._extract_symbol(transcript)
        period = self._extract_period(transcript)
        revenue = self._extract_metric(transcript, r"(?:revenue|total revenue|net revenue)\s+(?:of |was )?[\$€£]?([\d,.]+(?:\s+(?:million|billion|trillion))?)")
        eps = self._extract_metric(transcript, r"(?:eps|earnings per share|diluted eps)\s+(?:of |was )?[\$€£]?(-?[\d,.]+)")
        net_income = self._extract_metric(transcript, r"(?:net income|net earnings|net profit)\s+(?:of |was )?[\$€£]?(-?[\d,.]+(?:\s+(?:million|billion|trillion))?)")

        highlights = self._extract_highlights(transcript)
        risks = self._extract_risks(transcript)

        return EarningsSummary(
            symbol=symbol,
            period=period,
            revenue=revenue,
            earnings_per_share=eps,
            net_income=net_income,
            key_highlights=highlights,
            risks_mentioned=risks,
        )

    def extract_guidance(self, transcript: str) -> Guidance:
        if not transcript:
            return Guidance(symbol="", period="")

        symbol = self._extract_symbol(transcript)
        period = self._extract_guidance_period(transcript)
        revenue_guidance = self._extract_guidance_range(transcript, "revenue")
        eps_guidance = self._extract_guidance_range(transcript, "eps")
        qualitative = self._extract_qualitative_guidance(transcript)
        tone = self._assess_guidance_tone(transcript)

        return Guidance(
            symbol=symbol,
            period=period,
            revenue_guidance=revenue_guidance,
            eps_guidance=eps_guidance,
            qualitative_guidance=qualitative,
            tone=tone,
        )

    def compare_to_expectations(
        self,
        earnings: EarningsSummary,
        estimates: Dict[str, Any],
    ) -> ComparisonResult:
        actual_rev = earnings.revenue
        est_rev = estimates.get("revenue")
        rev_surprise = None
        if actual_rev is not None and est_rev is not None and est_rev != 0:
            rev_surprise = ((actual_rev - est_rev) / abs(est_rev)) * 100

        actual_eps = earnings.earnings_per_share
        est_eps = estimates.get("eps")
        eps_surprise = None
        if actual_eps is not None and est_eps is not None and est_eps != 0:
            eps_surprise = ((actual_eps - est_eps) / abs(est_eps)) * 100

        beat_miss = "inline"
        if rev_surprise is not None or eps_surprise is not None:
            avg_surprise = 0.0
            count = 0
            if rev_surprise is not None:
                avg_surprise += rev_surprise
                count += 1
            if eps_surprise is not None:
                avg_surprise += eps_surprise
                count += 1
            avg_surprise /= max(count, 1)
            if avg_surprise > 1.0:
                beat_miss = "beat"
            elif avg_surprise < -1.0:
                beat_miss = "miss"

        return ComparisonResult(
            symbol=earnings.symbol,
            period=earnings.period,
            actual_revenue=actual_rev,
            estimated_revenue=est_rev,
            revenue_surprise_pct=rev_surprise,
            actual_eps=actual_eps,
            estimated_eps=est_eps,
            eps_surprise_pct=eps_surprise,
            beat_or_miss=beat_miss,
        )

    def sentiment_of_earnings_call(self, transcript: str) -> SentimentResult:
        if not transcript:
            return SentimentResult(text="", sentiment="neutral", confidence=0.0, score=0.0)

        article = NewsArticle(
            title="Earnings Call Transcript",
            content=transcript[:2000],
            source="earnings",
            published_at=datetime.now(timezone.utc),
            url="",
        )
        return self._sentiment_analyzer.analyze(article)

    @staticmethod
    def _extract_symbol(transcript: str) -> str:
        match = re.search(r"(?:symbol|ticker|company)[:\s]+([A-Z]{1,5})", transcript, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return ""

    @staticmethod
    def _extract_period(transcript: str) -> str:
        match = re.search(r"(?:q[1-4]|quarter)\s+\w+\s+\d{4}", transcript, re.IGNORECASE)
        if match:
            return match.group(0)
        match = re.search(r"(?:fiscal year|fy)\s+\d{4}", transcript, re.IGNORECASE)
        if match:
            return match.group(0)
        return ""

    @staticmethod
    def _extract_metric(transcript: str, pattern: str) -> Optional[float]:
        match = re.search(pattern, transcript, re.IGNORECASE)
        if match:
            value_str = match.group(1).replace(",", "")
            multiplier = 1.0
            lower = value_str.lower()
            if "billion" in lower:
                multiplier = 1e9
            elif "million" in lower:
                multiplier = 1e6
            elif "trillion" in lower:
                multiplier = 1e12
            try:
                return float(re.sub(r"[^\d.\-]", "", value_str)) * multiplier
            except ValueError:
                pass
        return None

    def _extract_highlights(self, transcript: str) -> List[str]:
        sentences = re.split(r"[.!?]+", transcript)
        highlights = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20 or len(sentence) > 300:
                continue
            lower = sentence.lower()
            if any(kw in lower for kw in self._POSITIVE_KEYWORDS):
                highlights.append(sentence[:200])
            if len(highlights) >= 8:
                break
        return highlights

    def _extract_risks(self, transcript: str) -> List[str]:
        sentences = re.split(r"[.!?]+", transcript)
        risks = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20 or len(sentence) > 300:
                continue
            lower = sentence.lower()
            if any(kw in lower for kw in self._RISK_KEYWORDS):
                risks.append(sentence[:200])
            if len(risks) >= 8:
                break
        return risks

    @staticmethod
    def _extract_guidance_period(transcript: str) -> str:
        match = re.search(r"(?:next quarter|next year|full year|fiscal \d{4}|q[1-4] \d{4})", transcript, re.IGNORECASE)
        return match.group(0) if match else ""

    @staticmethod
    def _extract_guidance_range(transcript: str, metric: str) -> Optional[Tuple[float, float]]:
        pattern = rf"(?:{metric}|{metric.replace('eps', 'earnings per share')}).*?(?:between|range of|from)\s*[\$€£]?([\d,.]+)\s*(?:and|to|[-–])\s*[\$€£]?([\d,.]+)"
        match = re.search(pattern, transcript, re.IGNORECASE)
        if match:
            try:
                low = float(match.group(1).replace(",", ""))
                high = float(match.group(2).replace(",", ""))
                return (low, high)
            except ValueError:
                pass
        single_match = re.search(rf"(?:{metric}).*?[\$€£]?([\d,.]+)", transcript, re.IGNORECASE)
        if single_match:
            try:
                val = float(single_match.group(1).replace(",", ""))
                return (val * 0.95, val * 1.05)
            except ValueError:
                pass
        return None

    @staticmethod
    def _extract_qualitative_guidance(transcript: str) -> List[str]:
        guidance_phrases = [
            "we expect", "we anticipate", "we project", "we forecast",
            "we are guiding", "our outlook", "we see", "we believe",
            "management expects", "company expects",
        ]
        sentences = re.split(r"[.!?]+", transcript)
        qualitative = []
        for sentence in sentences:
            sentence = sentence.strip()
            if any(phrase in sentence.lower() for phrase in guidance_phrases):
                qualitative.append(sentence[:200])
            if len(qualitative) >= 5:
                break
        return qualitative

    @staticmethod
    def _assess_guidance_tone(transcript: str) -> str:
        lower = transcript.lower()
        positive_count = sum(1 for kw in ["strong", "growth", "confident", "optimistic", "positive", "momentum"] if kw in lower)
        negative_count = sum(1 for kw in ["cautious", "uncertain", "challenging", "conservative", "headwind", "pressure"] if kw in lower)
        if positive_count > negative_count + 1:
            return "positive"
        if negative_count > positive_count + 1:
            return "negative"
        return "neutral"


# ─── ResearchReportGenerator ───────────────────────────────────────────────────


class ResearchReportGenerator:
    """Generates comprehensive research reports and daily briefings."""

    def __init__(
        self,
        sentiment_analyzer: Optional[LLMSentimentAnalyzer] = None,
        narrative_extractor: Optional[MarketNarrativeExtractor] = None,
        earnings_analyzer: Optional[EarningsAnalyzer] = None,
    ) -> None:
        self._sentiment = sentiment_analyzer or LLMSentimentAnalyzer()
        self._narrative = narrative_extractor or MarketNarrativeExtractor(self._sentiment)
        self._earnings = earnings_analyzer or EarningsAnalyzer(self._sentiment)
        logger.info("ResearchReportGenerator initialized")

    def generate_report(
        self,
        symbol: str,
        news: List[NewsArticle],
        technical_data: Optional[Dict[str, Any]] = None,
    ) -> ResearchReport:
        sentiment_agg = self._sentiment.compute_aggregate_sentiment(symbol, news)
        themes = self._narrative.extract_themes(news)
        catalysts = self._narrative.identify_catalysts(news)
        summary_text = self._narrative.generate_summary(symbol, news)

        recommendations = self._generate_recommendations(sentiment_agg, themes, catalysts)
        risks = self._identify_risks(themes, catalysts, news)

        return ResearchReport(
            symbol=symbol,
            generated_at=datetime.now(timezone.utc),
            summary=summary_text,
            sentiment_analysis=sentiment_agg,
            themes=themes,
            catalysts=catalysts,
            technical_context=technical_data or {},
            recommendations=recommendations,
            risks=risks,
            raw_news=news[:20],
        )

    def generate_daily_briefing(self, symbols: List[str]) -> DailyBriefing:
        briefing = DailyBriefing(
            date=datetime.now(timezone.utc),
            symbols=symbols,
            market_overview="",
        )

        all_themes: List[Theme] = []
        all_catalysts: List[Catalyst] = []

        for symbol in symbols:
            try:
                report = self.generate_report(symbol, [])
                briefing.symbol_reports[symbol] = report
                all_themes.extend(report.themes)
                all_catalysts.extend(report.catalysts)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to generate report for %s: %s", symbol, exc)

        if all_themes:
            theme_counts: Dict[str, Tuple[int, float]] = defaultdict(lambda: (0, 0.0))
            for theme in all_themes:
                count, total_rel = theme_counts[theme.name]
                theme_counts[theme.name] = (count + 1, total_rel + theme.relevance)
            briefing.top_themes = sorted(
                [Theme(name=name, relevance=total / max(count, 1), article_count=count, sentiment="neutral")
                 for name, (count, total) in theme_counts.items()],
                key=lambda t: t.relevance,
                reverse=True,
            )[:10]

        briefing.urgent_catalysts = [
            c for c in all_catalysts
            if c.probability >= 0.7 and c.impact == "high"
        ][:10]

        briefing.market_overview = self._generate_market_overview(briefing)
        briefing.action_items = self._generate_action_items(briefing)

        logger.info("Generated daily briefing for %d symbols", len(symbols))
        return briefing

    def export_report(self, report: ResearchReport, format: str = "markdown") -> str:
        if format == "markdown":
            return self._export_markdown(report)
        if format == "json":
            return self._export_json(report)
        if format == "text":
            return self._export_text(report)
        logger.warning("Unknown export format: %s, defaulting to markdown", format)
        return self._export_markdown(report)

    def _generate_recommendations(
        self,
        sentiment: SentimentAggregate,
        themes: List[Theme],
        catalysts: List[Catalyst],
    ) -> List[str]:
        recommendations: List[str] = []
        if sentiment.net_score > 0.3 and sentiment.confidence > 0.6:
            recommendations.append("Strong positive sentiment suggests potential upside")
        elif sentiment.net_score < -0.3 and sentiment.confidence > 0.6:
            recommendations.append("Strong negative sentiment suggests potential downside risk")

        high_prob_catalysts = [c for c in catalysts if c.probability >= 0.8]
        if high_prob_catalysts:
            recommendations.append(f"Monitor {len(high_prob_catalysts)} high-probability catalyst(s)")

        risk_themes = [t for t in themes if t.sentiment == "negative"]
        if risk_themes:
            recommendations.append(f"Watch for negative developments in: {', '.join(t.name for t in risk_themes[:3])}")

        if not recommendations:
            recommendations.append("No strong directional signals; maintain current positioning")

        return recommendations

    @staticmethod
    def _identify_risks(
        themes: List[Theme],
        catalysts: List[Catalyst],
        news: List[NewsArticle],
    ) -> List[str]:
        risks: List[str] = []
        for theme in themes:
            if theme.sentiment == "negative":
                risks.append(f"Negative sentiment around {theme.name} theme")
        for catalyst in catalysts:
            if catalyst.impact == "high" and catalyst.probability < 0.5:
                risks.append(f"Uncertain outcome: {catalyst.description[:100]}")
        return risks[:10]

    @staticmethod
    def _generate_market_overview(briefing: DailyBriefing) -> str:
        total_reports = len(briefing.symbol_reports)
        positive_count = sum(
            1 for r in briefing.symbol_reports.values()
            if r.sentiment_analysis.sentiment == "positive"
        )
        negative_count = sum(
            1 for r in briefing.symbol_reports.values()
            if r.sentiment_analysis.sentiment == "negative"
        )

        lines = [
            f"Daily Market Briefing - {briefing.date.strftime('%Y-%m-%d')}",
            f"Symbols Covered: {total_reports}",
            f"Positive Sentiment: {positive_count}",
            f"Negative Sentiment: {negative_count}",
            f"Neutral Sentiment: {total_reports - positive_count - negative_count}",
            f"Urgent Catalysts: {len(briefing.urgent_catalysts)}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _generate_action_items(briefing: DailyBriefing) -> List[str]:
        items: List[str] = []
        for symbol, report in briefing.symbol_reports.items():
            if report.sentiment_analysis.sentiment == "positive" and report.sentiment_analysis.confidence > 0.7:
                items.append(f"Review long opportunities for {symbol}")
            elif report.sentiment_analysis.sentiment == "negative" and report.sentiment_analysis.confidence > 0.7:
                items.append(f"Review risk exposure for {symbol}")
        for catalyst in briefing.urgent_catalysts:
            items.append(f"Monitor catalyst: {catalyst.description[:80]}")
        return items[:15]

    @staticmethod
    def _export_markdown(report: ResearchReport) -> str:
        lines = [
            f"# Research Report: {report.symbol}",
            f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "## Summary",
            report.summary,
            "",
            "## Sentiment Analysis",
            f"- **Overall:** {report.sentiment_analysis.sentiment.upper()}",
            f"- **Net Score:** {report.sentiment_analysis.net_score:.3f}",
            f"- **Confidence:** {report.sentiment_analysis.confidence:.3f}",
            f"- **Sample Size:** {report.sentiment_analysis.sample_size}",
            "",
        ]

        if report.themes:
            lines.append("## Dominant Themes")
            for theme in report.themes[:5]:
                lines.append(f"- **{theme.name}** ({theme.article_count} articles, {theme.sentiment})")
            lines.append("")

        if report.catalysts:
            lines.append("## Key Catalysts")
            for catalyst in report.catalysts[:5]:
                lines.append(f"- [{catalyst.impact.upper()}] {catalyst.description} (probability: {catalyst.probability:.0%})")
            lines.append("")

        if report.recommendations:
            lines.append("## Recommendations")
            for rec in report.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        if report.risks:
            lines.append("## Risk Factors")
            for risk in report.risks:
                lines.append(f"- ⚠ {risk}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _export_json(report: ResearchReport) -> str:
        import json
        data = {
            "symbol": report.symbol,
            "generated_at": report.generated_at.isoformat(),
            "summary": report.summary,
            "sentiment": {
                "sentiment": report.sentiment_analysis.sentiment,
                "net_score": report.sentiment_analysis.net_score,
                "confidence": report.sentiment_analysis.confidence,
                "sample_size": report.sentiment_analysis.sample_size,
            },
            "themes": [
                {"name": t.name, "relevance": t.relevance, "sentiment": t.sentiment}
                for t in report.themes
            ],
            "catalysts": [
                {"type": c.event_type, "description": c.description, "impact": c.impact, "probability": c.probability}
                for c in report.catalysts
            ],
            "recommendations": report.recommendations,
            "risks": report.risks,
        }
        return json.dumps(data, indent=2, default=str)

    @staticmethod
    def _export_text(report: ResearchReport) -> str:
        lines = [
            f"RESEARCH REPORT: {report.symbol}",
            f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 60,
            "",
            "SUMMARY:",
            report.summary,
            "",
            f"SENTIMENT: {report.sentiment_analysis.sentiment.upper()} (score: {report.sentiment_analysis.net_score:.3f})",
            "",
        ]
        if report.recommendations:
            lines.append("RECOMMENDATIONS:")
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")
        return "\n".join(lines)


# ─── LLMResearchAgent ──────────────────────────────────────────────────────────


class LLMResearchAgent:
    """Top-level research agent coordinating analysis, monitoring, and insights."""

    def __init__(
        self,
        *,
        news_sources: Optional[Dict[str, NewsFetchFn]] = None,
        enable_finbert: bool = True,
        enable_vader: bool = True,
        monitoring_interval: int = 15,
    ) -> None:
        self._aggregator = NewsAggregator()
        if news_sources:
            for name, fetch_fn in news_sources.items():
                self._aggregator.add_source(name, fetch_fn)

        self._sentiment = LLMSentimentAnalyzer(
            enable_finbert=enable_finbert,
            enable_vader=enable_vader,
        )
        self._narrative = MarketNarrativeExtractor(self._sentiment)
        self._earnings = EarningsAnalyzer(self._sentiment)
        self._reporter = ResearchReportGenerator(self._sentiment, self._narrative, self._earnings)
        self._monitoring_interval = max(1, int(monitoring_interval))
        self._sentiment_history: Dict[str, List[Tuple[datetime, SentimentAggregate]]] = defaultdict(list)
        logger.info("LLMResearchAgent initialized (monitoring_interval=%dm)", self._monitoring_interval)

    async def analyze_symbol(self, symbol: str) -> SymbolAnalysis:
        articles = await self._aggregator.fetch_news([symbol], hours=24)
        articles = self._aggregator.deduplicate_articles(articles)
        articles = self._aggregator.filter_by_relevance(articles, [symbol], min_relevance=0.3)

        sentiment = self._sentiment.compute_aggregate_sentiment(symbol, articles)
        themes = self._narrative.extract_themes(articles)
        catalysts = self._narrative.identify_catalysts(articles)
        timeline = self._narrative.track_narrative_evolution(symbol, days=30)

        self._sentiment_history[symbol].append((datetime.now(timezone.utc), sentiment))
        if len(self._sentiment_history[symbol]) > 100:
            self._sentiment_history[symbol] = self._sentiment_history[symbol][-100:]

        insights = self._generate_insights(symbol, sentiment, themes, catalysts)
        risks = self._collect_risks(themes, catalysts)

        return SymbolAnalysis(
            symbol=symbol,
            analyzed_at=datetime.now(timezone.utc),
            sentiment=sentiment,
            themes=themes,
            catalysts=catalysts,
            narrative_timeline=timeline,
            earnings_summary=None,
            key_insights=insights,
            risk_factors=risks,
        )

    async def monitor_symbols(
        self,
        symbols: List[str],
        interval_minutes: Optional[int] = None,
    ) -> AsyncIterator[Alert]:
        interval = interval_minutes or self._monitoring_interval
        logger.info("Starting symbol monitoring for %s (interval=%dm)", symbols, interval)

        try:
            while True:
                for symbol in symbols:
                    try:
                        analysis = await self.analyze_symbol(symbol)
                        alerts = self._generate_alerts(analysis)
                        for alert in alerts:
                            yield alert
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Monitoring error for %s: %s", symbol, exc)
                        yield Alert(
                            symbol=symbol,
                            alert_type="monitoring_error",
                            message=str(exc),
                            severity="warning",
                            timestamp=datetime.now(timezone.utc),
                        )

                await asyncio.sleep(interval * 60)
        except asyncio.CancelledError:
            logger.info("Symbol monitoring cancelled for %s", symbols)
            raise

    def get_actionable_insights(self, analysis: SymbolAnalysis) -> List[Insight]:
        insights: List[Insight] = []

        if analysis.sentiment.net_score > 0.4 and analysis.sentiment.confidence > 0.6:
            insights.append(Insight(
                description=f"Strong bullish sentiment for {analysis.symbol} (score: {analysis.sentiment.net_score:.2f})",
                confidence=analysis.sentiment.confidence,
                impact="high",
                timeframe="short-term",
                related_symbols=[analysis.symbol],
                supporting_data={"sentiment_score": analysis.sentiment.net_score},
            ))
        elif analysis.sentiment.net_score < -0.4 and analysis.sentiment.confidence > 0.6:
            insights.append(Insight(
                description=f"Strong bearish sentiment for {analysis.symbol} (score: {analysis.sentiment.net_score:.2f})",
                confidence=analysis.sentiment.confidence,
                impact="high",
                timeframe="short-term",
                related_symbols=[analysis.symbol],
                supporting_data={"sentiment_score": analysis.sentiment.net_score},
            ))

        for catalyst in analysis.catalysts:
            if catalyst.probability >= 0.7 and catalyst.impact in ("high", "medium"):
                insights.append(Insight(
                    description=f"Catalyst alert: {catalyst.description}",
                    confidence=catalyst.probability,
                    impact=catalyst.impact,
                    timeframe=catalyst.timeframe,
                    related_symbols=catalyst.related_symbols or [analysis.symbol],
                    supporting_data={"catalyst_type": catalyst.event_type},
                ))

        for theme in analysis.themes:
            if theme.relevance > 0.6 and theme.article_count >= 3:
                insights.append(Insight(
                    description=f"Emerging theme: {theme.name} ({theme.article_count} articles)",
                    confidence=theme.relevance,
                    impact="medium" if theme.article_count < 5 else "high",
                    timeframe="medium-term",
                    related_symbols=[analysis.symbol],
                    supporting_data={"theme": theme.name, "sentiment": theme.sentiment},
                ))

        shift = self._sentiment.detect_sentiment_shift(self._sentiment_history.get(analysis.symbol, []))
        if shift:
            insights.append(Insight(
                description=f"Sentiment shift detected for {analysis.symbol}: {shift.previous_sentiment} -> {shift.current_sentiment}",
                confidence=min(1.0, abs(shift.magnitude)),
                impact="high",
                timeframe="immediate",
                related_symbols=[analysis.symbol],
                supporting_data={"shift_magnitude": shift.magnitude},
            ))

        return insights

    def _generate_insights(
        self,
        symbol: str,
        sentiment: SentimentAggregate,
        themes: List[Theme],
        catalysts: List[Catalyst],
    ) -> List[str]:
        insights: List[str] = []
        if sentiment.net_score > 0.3:
            insights.append(f"Positive sentiment prevailing (score: {sentiment.net_score:.2f})")
        elif sentiment.net_score < -0.3:
            insights.append(f"Negative sentiment prevailing (score: {sentiment.net_score:.2f})")

        if themes:
            top_theme = themes[0]
            insights.append(f"Dominant theme: {top_theme.name} ({top_theme.article_count} articles)")

        high_prob = [c for c in catalysts if c.probability >= 0.7]
        if high_prob:
            insights.append(f"{len(high_prob)} high-probability catalyst(s) identified")

        return insights

    @staticmethod
    def _collect_risks(themes: List[Theme], catalysts: List[Catalyst]) -> List[str]:
        risks: List[str] = []
        for theme in themes:
            if theme.sentiment == "negative":
                risks.append(f"Negative {theme.name} narrative")
        for catalyst in catalysts:
            if catalyst.impact == "high" and catalyst.probability < 0.5:
                risks.append(f"Uncertain high-impact event: {catalyst.event_type}")
        return risks[:10]

    def _generate_alerts(self, analysis: SymbolAnalysis) -> List[Alert]:
        alerts: List[Alert] = []

        if analysis.sentiment.net_score > 0.5 and analysis.sentiment.confidence > 0.7:
            alerts.append(Alert(
                symbol=analysis.symbol,
                alert_type="strong_positive_sentiment",
                message=f"Strong positive sentiment detected (score: {analysis.sentiment.net_score:.2f})",
                severity="info",
                timestamp=datetime.now(timezone.utc),
                metadata={"score": analysis.sentiment.net_score, "confidence": analysis.sentiment.confidence},
            ))
        elif analysis.sentiment.net_score < -0.5 and analysis.sentiment.confidence > 0.7:
            alerts.append(Alert(
                symbol=analysis.symbol,
                alert_type="strong_negative_sentiment",
                message=f"Strong negative sentiment detected (score: {analysis.sentiment.net_score:.2f})",
                severity="warning",
                timestamp=datetime.now(timezone.utc),
                metadata={"score": analysis.sentiment.net_score, "confidence": analysis.sentiment.confidence},
            ))

        for catalyst in analysis.catalysts:
            if catalyst.probability >= 0.8 and catalyst.impact == "high":
                alerts.append(Alert(
                    symbol=analysis.symbol,
                    alert_type="high_impact_catalyst",
                    message=f"High-impact catalyst: {catalyst.description[:100]}",
                    severity="critical",
                    timestamp=datetime.now(timezone.utc),
                    metadata={"catalyst_type": catalyst.event_type, "probability": catalyst.probability},
                ))

        shift = self._sentiment.detect_sentiment_shift(self._sentiment_history.get(analysis.symbol, []))
        if shift:
            alerts.append(Alert(
                symbol=analysis.symbol,
                alert_type="sentiment_shift",
                message=f"Sentiment shift: {shift.previous_sentiment} -> {shift.current_sentiment} (magnitude: {shift.magnitude:.2f})",
                severity="warning",
                timestamp=datetime.now(timezone.utc),
                metadata={"previous": shift.previous_sentiment, "current": shift.current_sentiment, "magnitude": shift.magnitude},
            ))

        return alerts
