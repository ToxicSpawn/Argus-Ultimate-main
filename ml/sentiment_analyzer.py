"""
Social & News Sentiment Analyzer
=================================
Analyzes sentiment from:
- Twitter/X (crypto influencers, breaking news)
- Reddit (r/cryptocurrency, r/bitcoin, r/ethfinance)
- Telegram (whale groups, signal channels)
- News sources (CoinDesk, The Block, Decrypt)
- On-chain governance (Snapshot, Tally)

Provides real-time sentiment signals for trading.
"""

import asyncio
import logging
import time
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class SentimentSource(Enum):
    """Sentiment data sources."""
    TWITTER = "twitter"
    REDDIT = "reddit"
    TELEGRAM = "telegram"
    NEWS = "news"
    DISCORD = "discord"
    GOVERNANCE = "governance"


class SentimentLevel(Enum):
    """Sentiment classification."""
    VERY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    VERY_BULLISH = 2


@dataclass
class SentimentData:
    """Single sentiment data point."""
    source: SentimentSource
    text: str
    sentiment_score: float  # -1 to 1
    sentiment_level: SentimentLevel
    confidence: float  # 0-1
    timestamp: float = field(default_factory=time.time)
    author: str = ""
    engagement: int = 0  # likes, retweets, upvotes
    symbols: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SentimentAggregate:
    """Aggregated sentiment for a symbol."""
    symbol: str
    timestamp: float
    overall_score: float  # -1 to 1
    overall_level: SentimentLevel
    volume: int  # Number of mentions
    source_breakdown: Dict[str, float] = field(default_factory=dict)
    trend: str = "stable"  # rising, falling, stable
    momentum: float = 0.0
    fear_greed_index: float = 50.0  # 0-100


class SentimentAnalyzer:
    """
    Sentiment Analyzer
    ==================
    Analyzes text sentiment using keyword matching and scoring.
    """
    
    # Bullish keywords (weighted)
    BULLISH_KEYWORDS = {
        # Strong bullish
        "moon": 0.8, "bullish": 0.7, "pump": 0.7, "breakout": 0.6,
        "rally": 0.7, "surge": 0.7, "soar": 0.8, "explode": 0.7,
        "accumulate": 0.6, "buy the dip": 0.7, "hodl": 0.5,
        "diamond hands": 0.6, "to the moon": 0.9, "100x": 0.8,
        "generational": 0.7, "life changing": 0.8,
        
        # Moderate bullish
        "up": 0.3, "higher": 0.3, "gain": 0.4, "positive": 0.4,
        "strong": 0.4, "support": 0.3, "bounce": 0.5, "recovery": 0.5,
        "adoption": 0.5, "partnership": 0.5, "upgrade": 0.4,
        "institutional": 0.5, "etf": 0.6, "approval": 0.6,
        
        # Technical bullish
        "golden cross": 0.7, "higher high": 0.5, "uptrend": 0.6,
        "bull flag": 0.6, "cup and handle": 0.5
    }
    
    # Bearish keywords (weighted)
    BEARISH_KEYWORDS = {
        # Strong bearish
        "crash": -0.9, "dump": -0.8, "bearish": -0.7, "rekt": -0.7,
        "liquidation": -0.8, "capitulation": -0.8, "plunge": -0.8,
        "collapse": -0.9, "rug pull": -0.9, "scam": -0.8,
        "dead cat bounce": -0.7, "bull trap": -0.7, "euphoria": -0.5,
        
        # Moderate bearish
        "down": -0.3, "lower": -0.3, "drop": -0.4, "decline": -0.4,
        "weak": -0.4, "resistance": -0.3, "sell": -0.4, "take profits": -0.3,
        "fear": -0.5, "panic": -0.6, "uncertainty": -0.4, "risk": -0.3,
        "ban": -0.7, "hack": -0.8, "exploit": -0.7, "vulnerability": -0.5,
        
        # Technical bearish
        "death cross": -0.7, "lower low": -0.5, "downtrend": -0.6,
        "head and shoulders": -0.5, "double top": -0.5
    }
    
    # Fear/Greed indicators
    FEAR_KEYWORDS = ["fear", "panic", "uncertainty", "doubt", "fud", "worried", "scared", "crash"]
    GREED_KEYWORDS = ["greed", "fomo", "euphoria", "moon", "100x", "lamborghini", "rich", "easy"]
    
    def __init__(self):
        self.history: Dict[str, List[SentimentData]] = defaultdict(list)
        self.aggregates: Dict[str, SentimentAggregate] = {}
        
    def analyze_text(self, text: str, source: SentimentSource, 
                     author: str = "", engagement: int = 0) -> SentimentData:
        """Analyze sentiment of a single text."""
        text_lower = text.lower()
        
        # Extract symbols mentioned
        symbols = self._extract_symbols(text)
        
        # Calculate sentiment score
        bullish_score = 0
        bearish_score = 0
        bullish_count = 0
        bearish_count = 0
        
        for keyword, weight in self.BULLISH_KEYWORDS.items():
            if keyword in text_lower:
                bullish_score += weight
                bullish_count += 1
        
        for keyword, weight in self.BEARISH_KEYWORDS.items():
            if keyword in text_lower:
                bearish_score += abs(weight)
                bearish_count += 1
        
        # Calculate final score (-1 to 1)
        if bullish_count + bearish_count == 0:
            score = 0
            confidence = 0.3  # Low confidence for neutral
        else:
            total = bullish_score + bearish_score
            if total > 0:
                score = (bullish_score - bearish_score) / total
            else:
                score = 0
            confidence = min(0.5 + (bullish_count + bearish_count) * 0.1, 0.95)
        
        # Determine sentiment level
        if score > 0.5:
            level = SentimentLevel.VERY_BULLISH
        elif score > 0.2:
            level = SentimentLevel.BULLISH
        elif score < -0.5:
            level = SentimentLevel.VERY_BEARISH
        elif score < -0.2:
            level = SentimentLevel.BEARISH
        else:
            level = SentimentLevel.NEUTRAL
        
        # Boost confidence with engagement
        if engagement > 1000:
            confidence = min(confidence + 0.1, 0.95)
        elif engagement > 100:
            confidence = min(confidence + 0.05, 0.95)
        
        data = SentimentData(
            source=source,
            text=text[:500],  # Truncate long texts
            sentiment_score=score,
            sentiment_level=level,
            confidence=confidence,
            author=author,
            engagement=engagement,
            symbols=symbols
        )
        
        # Store in history
        for symbol in symbols:
            self.history[symbol].append(data)
            # Keep only recent history
            if len(self.history[symbol]) > 1000:
                self.history[symbol] = self.history[symbol][-1000:]
        
        return data
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract crypto symbols from text."""
        # Common crypto symbols
        symbols = []
        text_upper = text.upper()
        
        # Known symbols
        known_symbols = [
            "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "DOT",
            "AVAX", "MATIC", "LINK", "UNI", "AAVE", "ATOM", "NEAR",
            "ARB", "OP", "APT", "SUI", "FIL", "ATOM", "NEAR", "XMR"
        ]
        
        for symbol in known_symbols:
            # Match symbol with word boundaries
            pattern = rf'\b{symbol}\b'
            if re.search(pattern, text_upper):
                symbols.append(symbol)
        
        # Also check for $SYMBOL pattern
        dollar_pattern = r'\$([A-Z]{2,5})\b'
        dollar_matches = re.findall(dollar_pattern, text_upper)
        symbols.extend(dollar_matches)
        
        return list(set(symbols))
    
    def calculate_fear_greed_index(self, symbol: str) -> float:
        """Calculate fear/greed index (0-100) for a symbol."""
        if symbol not in self.history or not self.history[symbol]:
            return 50.0
        
        recent = self.history[symbol][-100:]  # Last 100 data points
        
        fear_count = 0
        greed_count = 0
        
        for data in recent:
            text_lower = data.text.lower()
            for keyword in self.FEAR_KEYWORDS:
                if keyword in text_lower:
                    fear_count += 1
                    break
            for keyword in self.GREED_KEYWORDS:
                if keyword in text_lower:
                    greed_count += 1
                    break
        
        total = fear_count + greed_count
        if total == 0:
            return 50.0
        
        # Convert to 0-100 scale (0 = extreme fear, 100 = extreme greed)
        greed_ratio = greed_count / total
        return greed_ratio * 100
    
    def get_aggregate(self, symbol: str) -> Optional[SentimentAggregate]:
        """Get aggregated sentiment for a symbol."""
        if symbol not in self.history or not self.history[symbol]:
            return None
        
        recent = self.history[symbol][-500:]  # Last 500 data points
        
        # Calculate weighted average (more recent = higher weight)
        weights = np.exp(np.linspace(-1, 0, len(recent)))
        scores = [d.sentiment_score for d in recent]
        
        weighted_score = np.average(scores, weights=weights)
        
        # Source breakdown
        source_scores: Dict[str, List[float]] = defaultdict(list)
        for data in recent:
            source_scores[data.source.value].append(data.sentiment_score)
        
        source_breakdown = {
            source: np.mean(scores) for source, scores in source_scores.items()
        }
        
        # Calculate trend (compare recent vs older)
        if len(recent) >= 20:
            recent_avg = np.mean([d.sentiment_score for d in recent[-20:]])
            older_avg = np.mean([d.sentiment_score for d in recent[-40:-20]])
            momentum = recent_avg - older_avg
            
            if momentum > 0.1:
                trend = "rising"
            elif momentum < -0.1:
                trend = "falling"
            else:
                trend = "stable"
        else:
            momentum = 0
            trend = "stable"
        
        # Determine level
        if weighted_score > 0.5:
            level = SentimentLevel.VERY_BULLISH
        elif weighted_score > 0.2:
            level = SentimentLevel.BULLISH
        elif weighted_score < -0.5:
            level = SentimentLevel.VERY_BEARISH
        elif weighted_score < -0.2:
            level = SentimentLevel.BEARISH
        else:
            level = SentimentLevel.NEUTRAL
        
        aggregate = SentimentAggregate(
            symbol=symbol,
            timestamp=time.time(),
            overall_score=weighted_score,
            overall_level=level,
            volume=len(recent),
            source_breakdown=source_breakdown,
            trend=trend,
            momentum=momentum,
            fear_greed_index=self.calculate_fear_greed_index(symbol)
        )
        
        self.aggregates[symbol] = aggregate
        return aggregate
    
    def get_trading_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate trading signal from sentiment."""
        aggregate = self.get_aggregate(symbol)
        
        if not aggregate or aggregate.volume < 10:
            return {
                "signal": "neutral",
                "confidence": 0,
                "reason": "Insufficient sentiment data"
            }
        
        score = aggregate.overall_score
        momentum = aggregate.momentum
        fear_greed = aggregate.fear_greed_index
        
        # Contrarian signal at extremes
        if fear_greed > 80:  # Extreme greed = contrarian bearish
            signal = "short"
            confidence = 0.7
            reason = f"Extreme greed ({fear_greed:.0f}) - contrarian signal"
        elif fear_greed < 20:  # Extreme fear = contrarian bullish
            signal = "long"
            confidence = 0.7
            reason = f"Extreme fear ({fear_greed:.0f}) - contrarian signal"
        # Momentum-based signals
        elif score > 0.4 and momentum > 0.1:
            signal = "long"
            confidence = min(0.6 + abs(score) * 0.3, 0.9)
            reason = f"Bullish sentiment ({score:.2f}) with rising momentum"
        elif score < -0.4 and momentum < -0.1:
            signal = "short"
            confidence = min(0.6 + abs(score) * 0.3, 0.9)
            reason = f"Bearish sentiment ({score:.2f}) with falling momentum"
        else:
            signal = "neutral"
            confidence = 0.4
            reason = f"Neutral sentiment ({score:.2f})"
        
        return {
            "signal": signal,
            "confidence": confidence,
            "score": score,
            "momentum": momentum,
            "fear_greed_index": fear_greed,
            "volume": aggregate.volume,
            "trend": aggregate.trend,
            "reason": reason
        }


class TwitterSentimentCollector:
    """
    Twitter Sentiment Collector
    ===========================
    Collects and analyzes Twitter/X sentiment.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.analyzer = SentimentAnalyzer()
        
        # Influential accounts to track
        self.tracked_accounts = [
            "elonmusk", "saborchain", "CryptoCapo_", "GiganticRebirth",
            "Pentoshi1", "CryptoYoda", "TheCryptoDog", "CryptoCred",
            "PeterLBrandt", "woonomic", "zhusu", "balajis"
        ]
        
        # Keywords to track
        self.keywords = [
            "bitcoin", "ethereum", "crypto", "bullish", "bearish",
            "pump", "dump", "moon", "crash", "breakout"
        ]
    
    async def collect_tweets(self, query: str) -> List[SentimentData]:
        """Collect and analyze tweets."""
        # In production: use Twitter API v2
        # For now, return simulated data
        
        simulated_tweets = [
            {"text": "BTC looking bullish, breaking resistance! 🚀", "author": "trader1", "engagement": 500},
            {"text": "Market sentiment extremely fearful, time to accumulate", "author": "whale_watch", "engagement": 1200},
            {"text": "ETH gas fees dropping, bullish for adoption", "author": "defi_analyst", "engagement": 300}
        ]
        
        results = []
        for tweet in simulated_tweets:
            data = self.analyzer.analyze_text(
                tweet["text"],
                SentimentSource.TWITTER,
                tweet["author"],
                tweet["engagement"]
            )
            results.append(data)
        
        return results


class RedditSentimentCollector:
    """
    Reddit Sentiment Collector
    ==========================
    Collects sentiment from crypto subreddits.
    """
    
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.analyzer = SentimentAnalyzer()
        
        self.subreddits = [
            "cryptocurrency", "bitcoin", "ethereum", "defi",
            "altcoin", "SatoshiStreetBets", "CryptoMarkets"
        ]
    
    async def collect_posts(self, subreddit: str, limit: int = 100) -> List[SentimentData]:
        """Collect and analyze Reddit posts."""
        # In production: use Reddit API
        # For now, return simulated data
        
        simulated_posts = [
            {"text": "Just bought the dip on ETH, feeling bullish about Q2", "author": "hodler42", "engagement": 150},
            {"text": "BTC dominance rising, alt season might be over", "author": "analyst99", "engagement": 89},
            {"text": "DeFi protocols showing strong fundamentals despite market", "author": "defi_degen", "engagement": 234}
        ]
        
        results = []
        for post in simulated_posts:
            data = self.analyzer.analyze_text(
                post["text"],
                SentimentSource.REDDIT,
                post["author"],
                post["engagement"]
            )
            results.append(data)
        
        return results


class NewsSentimentCollector:
    """
    News Sentiment Collector
    ========================
    Collects sentiment from crypto news sources.
    """
    
    def __init__(self):
        self.analyzer = SentimentAnalyzer()
        
        self.news_sources = [
            "coindesk", "theblock", "decrypt", "cointelegraph",
            "bitcoinmagazine", "defiant", "reuters", "bloomberg"
        ]
    
    async def collect_news(self, keywords: List[str]) -> List[SentimentData]:
        """Collect and analyze news articles."""
        # In production: use news APIs, RSS feeds
        # For now, return simulated data
        
        simulated_news = [
            {"text": "SEC approves Bitcoin ETF application, major milestone for crypto", "source": "coindesk"},
            {"text": "Major exchange hack reported, $100M stolen", "source": "theblock"},
            {"text": "Institutional adoption accelerating, BlackRock increases BTC holdings", "source": "reuters"}
        ]
        
        results = []
        for news in simulated_news:
            data = self.analyzer.analyze_text(
                news["text"],
                SentimentSource.NEWS,
                news["source"],
                0  # News doesn't have engagement metric
            )
            results.append(data)
        
        return results


class SentimentAggregator:
    """
    Sentiment Aggregator
    ====================
    Aggregates sentiment from all sources.
    """
    
    def __init__(self):
        self.analyzer = SentimentAnalyzer()
        self.twitter = TwitterSentimentCollector()
        self.reddit = RedditSentimentCollector()
        self.news = NewsSentimentCollector()
        
        self.all_sentiment: Dict[str, List[SentimentData]] = defaultdict(list)
    
    async def collect_all(self, symbols: List[str]) -> Dict[str, SentimentAggregate]:
        """Collect sentiment from all sources for given symbols."""
        # Collect from all sources
        await self.twitter.collect_tweets("crypto")
        await self.reddit.collect_posts("cryptocurrency")
        await self.news.collect_news(["bitcoin", "ethereum"])
        
        # Get aggregates
        aggregates = {}
        for symbol in symbols:
            agg = self.analyzer.get_aggregate(symbol)
            if agg:
                aggregates[symbol] = agg
        
        return aggregates
    
    def get_market_sentiment(self) -> Dict[str, Any]:
        """Get overall market sentiment."""
        # Aggregate across major symbols
        major_symbols = ["BTC", "ETH", "SOL", "BNB", "XRP"]
        
        scores = []
        for symbol in major_symbols:
            agg = self.analyzer.get_aggregate(symbol)
            if agg:
                scores.append(agg.overall_score)
        
        if not scores:
            return {"score": 0, "level": "neutral", "fear_greed": 50}
        
        avg_score = np.mean(scores)
        fear_greed = self.analyzer.calculate_fear_greed_index("BTC")
        
        if avg_score > 0.3:
            level = "bullish"
        elif avg_score < -0.3:
            level = "bearish"
        else:
            level = "neutral"
        
        return {
            "score": avg_score,
            "level": level,
            "fear_greed_index": fear_greed,
            "symbols_analyzed": len(scores)
        }


# Export
__all__ = [
    "SentimentSource",
    "SentimentLevel",
    "SentimentData",
    "SentimentAggregate",
    "SentimentAnalyzer",
    "TwitterSentimentCollector",
    "RedditSentimentCollector",
    "NewsSentimentCollector",
    "SentimentAggregator"
]
