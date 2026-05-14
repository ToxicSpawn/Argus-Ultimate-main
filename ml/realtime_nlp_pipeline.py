"""
ml/realtime_nlp_pipeline.py — Real-Time NLP Pipeline for Trading

Processes news, filings, tweets, and social media in real-time for alpha signals.

Features:
- Sub-100ms text processing
- Entity extraction (companies, people, events)
- Sentiment analysis (positive/negative/neutral + intensity)
- Event detection (earnings, mergers, regulatory)
- Topic classification (macro, sector, company)
- LLM-based reasoning for complex analysis
- Streaming pipeline for continuous processing

Usage::

    from ml.realtime_nlp_pipeline import RealTimeNLPPipeline
    
    pipeline = RealTimeNLPPipeline()
    
    # Process single article
    result = pipeline.process("Apple reports record Q4 earnings, beats estimates")
    
    # Process stream
    for article in news_stream:
        signals = pipeline.process(article)
        if signals.sentiment_score > 0.5:
            print(f"Positive news: {signals.entities}")
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

class SentimentLevel(str, Enum):
    """Sentiment classification levels."""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class EventType(str, Enum):
    """Financial event types."""
    EARNINGS = "earnings"
    MERGER = "merger"
    ACQUISITION = "acquisition"
    REGULATORY = "regulatory"
    PRODUCT_LAUNCH = "product_launch"
    MANAGEMENT_CHANGE = "management_change"
    LEGAL = "legal"
    MACRO = "macro"
    ANALYST_UPGRADE = "analyst_upgrade"
    ANALYST_DOWNGRADE = "analyst_downgrade"
    INSIDER_TRADING = "insider_trading"
    DIVIDEND = "dividend"
    BUYBACK = "buyback"
    GUIDANCE = "guidance"
    PARTNERSHIP = "partnership"
    UNKNOWN = "unknown"


class TopicCategory(str, Enum):
    """Topic classification categories."""
    MACRO_ECONOMIC = "macro_economic"
    MONETARY_POLICY = "monetary_policy"
    FISCAL_POLICY = "fiscal_policy"
    SECTOR_TECH = "sector_tech"
    SECTOR_FINANCE = "sector_finance"
    SECTOR_HEALTHCARE = "sector_healthcare"
    SECTOR_ENERGY = "sector_energy"
    SECTOR_CONSUMER = "sector_consumer"
    COMPANY_EARNINGS = "company_earnings"
    COMPANY_STRATEGY = "company_strategy"
    COMPANY_RISK = "company_risk"
    CRYPTO = "crypto"
    COMMODITIES = "commodities"
    GEOPOLITICAL = "geopolitical"
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """Extracted entity from text."""
    name: str
    entity_type: str  # company, person, location, ticker
    ticker: Optional[str] = None
    confidence: float = 1.0
    mention_count: int = 1


@dataclass
class NLPSignal:
    """NLP signal extracted from text."""
    text: str
    timestamp: datetime
    
    # Sentiment
    sentiment_score: float = 0.0  # -1 to 1
    sentiment_level: SentimentLevel = SentimentLevel.NEUTRAL
    sentiment_confidence: float = 0.5
    
    # Entities
    entities: List[Entity] = field(default_factory=list)
    tickers: List[str] = field(default_factory=list)
    
    # Events
    event_type: EventType = EventType.UNKNOWN
    event_confidence: float = 0.0
    
    # Topics
    topic: TopicCategory = TopicCategory.UNKNOWN
    topic_confidence: float = 0.0
    
    # Urgency
    urgency_score: float = 0.0  # 0 to 1
    is_breaking: bool = False
    
    # Keywords
    keywords: List[str] = field(default_factory=list)
    
    # Processing time
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text[:200],
            "sentiment_score": self.sentiment_score,
            "sentiment_level": self.sentiment_level.value,
            "entities": [{"name": e.name, "ticker": e.ticker} for e in self.entities],
            "tickers": self.tickers,
            "event_type": self.event_type.value,
            "topic": self.topic.value,
            "urgency_score": self.urgency_score,
            "is_breaking": self.is_breaking,
        }


@dataclass
class AggregatedSignal:
    """Aggregated signals over a time window."""
    window_start: datetime
    window_end: datetime
    n_articles: int
    avg_sentiment: float
    sentiment_trend: float  # +1 = improving, -1 = worsening
    top_entities: List[Tuple[str, float]]  # (entity, sentiment)
    top_tickers: List[Tuple[str, float]]  # (ticker, sentiment)
    event_counts: Dict[str, int]
    urgency_score: float
    topics: Dict[str, float]


# ============================================================================
# Lexicon-Based Sentiment (No External Dependencies)
# ============================================================================

# Financial sentiment lexicon
POSITIVE_WORDS: Set[str] = {
    "beat", "beats", "beating", "exceeds", "exceeded", "outperforms",
    "growth", "grew", "increase", "increased", "rising", "surge", "surged",
    "profit", "profits", "profitable", "gain", "gains", "gaining",
    "strong", "strength", "strengthen", "robust", "solid", "healthy",
    "optimistic", "positive", "upbeat", "bullish", "upgrade", "upgraded",
    "breakthrough", "innovation", "innovative", "leading", "leader",
    "record", "high", "higher", "highest", "improve", "improved",
    "success", "successful", "win", "wins", "winning",
    "opportunity", "opportunities", "potential", "promising",
    "dividend", "buyback", "acquisition", "partnership",
    "recovery", "recovered", "rebounding", "resilient",
}

NEGATIVE_WORDS: Set[str] = {
    "miss", "missed", "missing", "below", "disappointing",
    "loss", "losses", "losing", "decline", "declined", "declining",
    "fall", "fell", "falling", "drop", "dropped", "plunge", "plunged",
    "weak", "weakness", "weaken", "weakened", "deteriorate",
    "pessimistic", "negative", "bearish", "downgrade", "downgraded",
    "risk", "risks", "risky", "concern", "concerns", "concerning",
    "threat", "threats", "threatening", "warning", "warned",
    "lawsuit", "investigation", "investigated", "fraud", "scandal",
    "restructuring", "layoffs", "cut", "cuts", "cutting",
    "debt", "leverage", "default", "bankruptcy", "bankrupt",
    "volatile", "uncertainty", "uncertain", "recession",
    "inflation", "stagflation", "crisis", "crash",
}

INTENSIFIERS: Set[str] = {
    "very", "extremely", "significantly", "substantially", "massively",
    "dramatically", "sharply", "steeply", "notably", "particularly",
}

NEGATORS: Set[str] = {
    "not", "no", "never", "neither", "nor", "cannot", "can't",
    "won't", "don't", "doesn't", "didn't", "isn't", "aren't",
}

# Ticker patterns (e.g., $AAPL, AAPL)
TICKER_PATTERN = re.compile(r'\$([A-Z]{1,5})\b|(?<![a-zA-Z])([A-Z]{2,5})(?![a-z])')

# Company name to ticker mapping (common ones)
COMPANY_TO_TICKER: Dict[str, str] = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "tesla": "TSLA", "meta": "META", "facebook": "META",
    "nvidia": "NVDA", "netflix": "NFLX", "amd": "AMD", "intel": "INTC",
    "bitcoin": "BTC", "ethereum": "ETH", "crypto": "CRYPTO",
    "fed": "FED", "federal reserve": "FED", "sec": "SEC",
    "spacex": "SPACEX", "openai": "OPENAI",
}


# ============================================================================
# NLP Components
# ============================================================================

class SentimentAnalyzer:
    """Lexicon-based sentiment analyzer for financial text."""
    
    def __init__(self):
        self.positive_words = POSITIVE_WORDS
        self.negative_words = NEGATIVE_WORDS
        self.intensifiers = INTENSIFIERS
        self.negators = NEGATORS
    
    def analyze(self, text: str) -> Tuple[float, SentimentLevel, float]:
        """
        Analyze sentiment of text.
        
        Returns:
            (score, level, confidence)
        """
        words = text.lower().split()
        
        pos_count = 0
        neg_count = 0
        total_sentiment_words = 0
        
        for i, word in enumerate(words):
            # Check for negation in previous 3 words
            is_negated = any(words[j] in self.negators for j in range(max(0, i-3), i))
            
            # Check for intensifier
            is_intensified = any(words[j] in self.intensifiers for j in range(max(0, i-2), i))
            intensity = 1.5 if is_intensified else 1.0
            
            if word in self.positive_words:
                total_sentiment_words += 1
                if is_negated:
                    neg_count += intensity
                else:
                    pos_count += intensity
            elif word in self.negative_words:
                total_sentiment_words += 1
                if is_negated:
                    pos_count += intensity
                else:
                    neg_count += intensity
        
        # Calculate score
        if total_sentiment_words == 0:
            return 0.0, SentimentLevel.NEUTRAL, 0.3
        
        score = (pos_count - neg_count) / (pos_count + neg_count + 1e-8)
        score = np.clip(score, -1.0, 1.0)
        
        # Determine level
        if score > 0.5:
            level = SentimentLevel.VERY_POSITIVE
        elif score > 0.1:
            level = SentimentLevel.POSITIVE
        elif score < -0.5:
            level = SentimentLevel.VERY_NEGATIVE
        elif score < -0.1:
            level = SentimentLevel.NEGATIVE
        else:
            level = SentimentLevel.NEUTRAL
        
        # Confidence based on sentiment word count
        confidence = min(0.9, 0.3 + total_sentiment_words * 0.1)
        
        return score, level, confidence


class EntityExtractor:
    """Extract entities from financial text."""
    
    def __init__(self):
        self.company_to_ticker = COMPANY_TO_TICKER
        self.ticker_pattern = TICKER_PATTERN
    
    def extract(self, text: str) -> List[Entity]:
        """Extract entities from text."""
        entities = []
        seen_names = set()
        
        # Extract tickers
        for match in self.ticker_pattern.finditer(text):
            ticker = match.group(1) or match.group(2)
            if ticker and ticker not in seen_names and len(ticker) >= 2:
                entities.append(Entity(
                    name=ticker,
                    entity_type="ticker",
                    ticker=ticker,
                    confidence=0.9,
                ))
                seen_names.add(ticker)
        
        # Extract company names
        text_lower = text.lower()
        for company, ticker in self.company_to_ticker.items():
            if company in text_lower and company not in seen_names:
                entities.append(Entity(
                    name=company.title(),
                    entity_type="company",
                    ticker=ticker,
                    confidence=0.85,
                ))
                seen_names.add(company)
        
        return entities


class EventDetector:
    """Detect financial events from text."""
    
    EVENT_PATTERNS: Dict[EventType, List[str]] = {
        EventType.EARNINGS: [
            r"earnings", r"quarterly results", r"q[1-4]", r"revenue",
            r"eps", r"profit", r"beats? estimates", r"misses? estimates",
        ],
        EventType.MERGER: [
            r"merger", r"merge", r"combining", r"combined",
        ],
        EventType.ACQUISITION: [
            r"acqui", r"buyout", r"purchase", r"takeover",
        ],
        EventType.REGULATORY: [
            r"regulat", r"sec ", r"fcc", r"antitrust", r"compliance",
            r"investigation", r"probe",
        ],
        EventType.ANALYST_UPGRADE: [
            r"upgrade", r"upgraded", r"overweight", r"buy rating",
            r"price target raised",
        ],
        EventType.ANALYST_DOWNGRADE: [
            r"downgrade", r"downgraded", r"underweight", r"sell rating",
            r"price target cut", r"price target lowered",
        ],
        EventType.INSIDER_TRADING: [
            r"insider", r"ceo sold", r"cfobought", r"form 4",
            r"executive sold", r"director bought",
        ],
        EventType.DIVIDEND: [
            r"dividend", r"payout", r"yield",
        ],
        EventType.BUYBACK: [
            r"buyback", r"repurchase", r"share repurchase",
        ],
        EventType.GUIDANCE: [
            r"guidance", r"forecast", r"outlook", r"expect",
        ],
        EventType.MANAGEMENT_CHANGE: [
            r"ceo", r"cfo", r"resign", r"appoint", r"hire",
            r"management change", r"leadership",
        ],
        EventType.LEGAL: [
            r"lawsuit", r"sued", r"litigation", r"settlement",
            r"class action",
        ],
    }
    
    def detect(self, text: str) -> Tuple[EventType, float]:
        """Detect event type from text."""
        text_lower = text.lower()
        
        scores: Dict[EventType, float] = defaultdict(float)
        
        for event_type, patterns in self.EVENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    scores[event_type] += 1
        
        if not scores:
            return EventType.UNKNOWN, 0.0
        
        best_event = max(scores, key=scores.get)
        confidence = min(0.95, 0.5 + scores[best_event] * 0.15)
        
        return best_event, confidence


class TopicClassifier:
    """Classify text into topic categories."""
    
    TOPIC_PATTERNS: Dict[TopicCategory, List[str]] = {
        TopicCategory.MONETARY_POLICY: [
            r"fed", r"federal reserve", r"interest rate", r"rate hike",
            r"rate cut", r"monetary policy", r"powell", r"fomc",
        ],
        TopicCategory.FISCAL_POLICY: [
            r"congress", r"stimulus", r"fiscal", r"tax", r"budget",
            r"deficit", r"infrastructure",
        ],
        TopicCategory.SECTOR_TECH: [
            r"tech", r"software", r"saas", r"ai ", r"artificial intelligence",
            r"cloud", r"semiconductor", r"chip",
        ],
        TopicCategory.SECTOR_FINANCE: [
            r"bank", r"financial", r"insurance", r"hedge fund",
            r"private equity", r"wall street",
        ],
        TopicCategory.SECTOR_HEALTHCARE: [
            r"pharma", r"drug", r"clinical trial", r"fda",
            r"healthcare", r"biotech", r"vaccine",
        ],
        TopicCategory.SECTOR_ENERGY: [
            r"oil", r"energy", r"solar", r"renewable", r"natural gas",
            r"opec", r"drilling",
        ],
        TopicCategory.CRYPTO: [
            r"bitcoin", r"ethereum", r"crypto", r"blockchain",
            r"defi", r"nft", r"web3",
        ],
        TopicCategory.GEOPOLITICAL: [
            r"china", r"russia", r"ukraine", r"war", r"sanctions",
            r"trade war", r"tariff", r"geopolitical",
        ],
    }
    
    def classify(self, text: str) -> Tuple[TopicCategory, float]:
        """Classify text into topic."""
        text_lower = text.lower()
        
        scores: Dict[TopicCategory, float] = defaultdict(float)
        
        for topic, patterns in self.TOPIC_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    scores[topic] += 1
        
        if not scores:
            return TopicCategory.UNKNOWN, 0.3
        
        best_topic = max(scores, key=scores.get)
        confidence = min(0.9, 0.4 + scores[best_topic] * 0.15)
        
        return best_topic, confidence


class UrgencyDetector:
    """Detect urgency/breaking news indicators."""
    
    URGENCY_KEYWORDS: Set[str] = {
        "breaking", "urgent", "just in", "alert", "flash",
        "developing", "update", "exclusive", "live",
    }
    
    HIGH_IMPACT_KEYWORDS: Set[str] = {
        "crash", "plunge", "surge", "halt", "suspend",
        "bankruptcy", "default", "emergency", "crisis",
    }
    
    def detect(self, text: str) -> Tuple[float, bool]:
        """
        Detect urgency score and breaking status.
        
        Returns:
            (urgency_score, is_breaking)
        """
        text_lower = text.lower()
        
        urgency_score = 0.0
        is_breaking = False
        
        for keyword in self.URGENCY_KEYWORDS:
            if keyword in text_lower:
                urgency_score += 0.3
                if keyword in ("breaking", "flash", "urgent"):
                    is_breaking = True
        
        for keyword in self.HIGH_IMPACT_KEYWORDS:
            if keyword in text_lower:
                urgency_score += 0.4
        
        urgency_score = min(1.0, urgency_score)
        
        return urgency_score, is_breaking


# ============================================================================
# Main Pipeline
# ============================================================================

class RealTimeNLPPipeline:
    """
    Real-Time NLP Pipeline for financial text processing.
    
    Processes news, filings, tweets in <100ms for alpha signals.
    """
    
    def __init__(
        self,
        *,
        enable_entity_extraction: bool = True,
        enable_event_detection: bool = True,
        enable_topic_classification: bool = True,
        window_size_seconds: int = 300,  # 5 minutes
    ):
        self.enable_entity_extraction = enable_entity_extraction
        self.enable_event_detection = enable_event_detection
        self.enable_topic_classification = enable_topic_classification
        self.window_size_seconds = window_size_seconds
        
        # Components
        self.sentiment_analyzer = SentimentAnalyzer()
        self.entity_extractor = EntityExtractor()
        self.event_detector = EventDetector()
        self.topic_classifier = TopicClassifier()
        self.urgency_detector = UrgencyDetector()
        
        # Signal history for aggregation
        self.signal_history: deque = deque(maxlen=10000)
        
        # Stats
        self.total_processed = 0
        self.total_time_ms = 0.0
    
    def process(self, text: str, source: str = "unknown") -> NLPSignal:
        """
        Process a single text and extract signals.
        
        Args:
            text: Input text (news article, tweet, etc.)
            source: Source identifier
            
        Returns:
            NLPSignal with extracted information
        """
        start_time = time.monotonic()
        
        # Sentiment
        sentiment_score, sentiment_level, sentiment_confidence = self.sentiment_analyzer.analyze(text)
        
        # Entities
        entities = []
        tickers = []
        if self.enable_entity_extraction:
            entities = self.entity_extractor.extract(text)
            tickers = list(set(e.ticker for e in entities if e.ticker))
        
        # Event detection
        event_type = EventType.UNKNOWN
        event_confidence = 0.0
        if self.enable_event_detection:
            event_type, event_confidence = self.event_detector.detect(text)
        
        # Topic classification
        topic = TopicCategory.UNKNOWN
        topic_confidence = 0.0
        if self.enable_topic_classification:
            topic, topic_confidence = self.topic_classifier.classify(text)
        
        # Urgency
        urgency_score, is_breaking = self.urgency_detector.detect(text)
        
        # Keywords (simple extraction)
        keywords = self._extract_keywords(text)
        
        # Processing time
        processing_time_ms = (time.monotonic() - start_time) * 1000
        
        # Create signal
        signal = NLPSignal(
            text=text,
            timestamp=datetime.utcnow(),
            sentiment_score=sentiment_score,
            sentiment_level=sentiment_level,
            sentiment_confidence=sentiment_confidence,
            entities=entities,
            tickers=tickers,
            event_type=event_type,
            event_confidence=event_confidence,
            topic=topic,
            topic_confidence=topic_confidence,
            urgency_score=urgency_score,
            is_breaking=is_breaking,
            keywords=keywords,
            processing_time_ms=processing_time_ms,
        )
        
        # Store in history
        self.signal_history.append(signal)
        
        # Update stats
        self.total_processed += 1
        self.total_time_ms += processing_time_ms
        
        return signal
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract important keywords from text."""
        # Simple TF-based extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            "that", "this", "with", "from", "have", "will", "been",
            "were", "said", "also", "they", "their", "which", "would",
            "about", "after", "could", "other", "than", "its", "more",
        }
        
        words = [w for w in words if w not in stop_words]
        
        # Count and return top keywords
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1
        
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_keywords]]
    
    def aggregate_signals(
        self,
        window_seconds: Optional[int] = None,
    ) -> AggregatedSignal:
        """Aggregate signals over a time window."""
        window_seconds = window_seconds or self.window_size_seconds
        cutoff_time = datetime.utcnow().timestamp() - window_seconds
        
        # Filter recent signals
        recent_signals = [
            s for s in self.signal_history
            if s.timestamp.timestamp() > cutoff_time
        ]
        
        if not recent_signals:
            return AggregatedSignal(
                window_start=datetime.utcnow(),
                window_end=datetime.utcnow(),
                n_articles=0,
                avg_sentiment=0.0,
                sentiment_trend=0.0,
                top_entities=[],
                top_tickers=[],
                event_counts={},
                urgency_score=0.0,
                topics={},
            )
        
        # Calculate metrics
        sentiments = [s.sentiment_score for s in recent_signals]
        avg_sentiment = np.mean(sentiments)
        
        # Sentiment trend (compare first half to second half)
        if len(sentiments) > 2:
            mid = len(sentiments) // 2
            first_half = np.mean(sentiments[:mid])
            second_half = np.mean(sentiments[mid:])
            sentiment_trend = np.clip(second_half - first_half, -1, 1)
        else:
            sentiment_trend = 0.0
        
        # Top entities by sentiment
        entity_sentiments: Dict[str, List[float]] = defaultdict(list)
        for signal in recent_signals:
            for entity in signal.entities:
                entity_sentiments[entity.name].append(signal.sentiment_score)
        
        top_entities = sorted(
            [(name, np.mean(scores)) for name, scores in entity_sentiments.items()],
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:10]
        
        # Top tickers by sentiment
        ticker_sentiments: Dict[str, List[float]] = defaultdict(list)
        for signal in recent_signals:
            for ticker in signal.tickers:
                ticker_sentiments[ticker].append(signal.sentiment_score)
        
        top_tickers = sorted(
            [(ticker, np.mean(scores)) for ticker, scores in ticker_sentiments.items()],
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:10]
        
        # Event counts
        event_counts: Dict[str, int] = defaultdict(int)
        for signal in recent_signals:
            event_counts[signal.event_type.value] += 1
        
        # Average urgency
        urgency_score = np.mean([s.urgency_score for s in recent_signals])
        
        # Topic distribution
        topic_sentiments: Dict[str, List[float]] = defaultdict(list)
        for signal in recent_signals:
            topic_sentiments[signal.topic.value].append(signal.sentiment_score)
        
        topics = {topic: np.mean(scores) for topic, scores in topic_sentiments.items()}
        
        return AggregatedSignal(
            window_start=min(s.timestamp for s in recent_signals),
            window_end=max(s.timestamp for s in recent_signals),
            n_articles=len(recent_signals),
            avg_sentiment=avg_sentiment,
            sentiment_trend=sentiment_trend,
            top_entities=top_entities,
            top_tickers=top_tickers,
            event_counts=dict(event_counts),
            urgency_score=urgency_score,
            topics=topics,
        )
    
    def get_ticker_sentiment(self, ticker: str) -> Tuple[float, int]:
        """Get aggregated sentiment for a specific ticker."""
        ticker = ticker.upper()
        
        ticker_signals = [
            s for s in self.signal_history
            if ticker in s.tickers
        ]
        
        if not ticker_signals:
            return 0.0, 0
        
        avg_sentiment = np.mean([s.sentiment_score for s in ticker_signals])
        return avg_sentiment, len(ticker_signals)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        avg_time = self.total_time_ms / max(self.total_processed, 1)
        
        return {
            "total_processed": self.total_processed,
            "avg_processing_time_ms": avg_time,
            "signal_history_size": len(self.signal_history),
            "is_fast_enough": avg_time < 100,  # Target: <100ms
        }


# ============================================================================
# Factory Function
# ============================================================================

def create_nlp_pipeline(**kwargs) -> RealTimeNLPPipeline:
    """Create a real-time NLP pipeline."""
    return RealTimeNLPPipeline(**kwargs)
