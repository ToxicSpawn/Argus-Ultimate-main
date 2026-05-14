"""
Real-Time Alternative Data Hub
==============================
Integrates satellite imagery, social sentiment, on-chain data, and other
alternative data sources for institutional-grade trading intelligence.

Data Sources:
- Satellite imagery (commodity tracking, facility monitoring)
- Social sentiment (Twitter/X, Reddit, Telegram, Discord)
- On-chain analytics (whale tracking, exchange flows, DeFi metrics)
- News feeds (real-time NLP processing)
- Economic indicators (macro data feeds)

Architecture:
- Async streaming with WebSocket connections
- Redis-backed caching for low-latency access
- Priority queuing for time-sensitive data
- Fallback mechanisms for API failures
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Types of alternative data sources."""
    SATELLITE = "satellite"
    SOCIAL_SENTIMENT = "social_sentiment"
    ON_CHAIN = "on_chain"
    NEWS = "news"
    ECONOMIC = "economic"
    WEATHER = "weather"
    SUPPLY_CHAIN = "supply_chain"
    REGULATORY = "regulatory"


class DataPriority(Enum):
    """Data priority levels for processing."""
    CRITICAL = 1    # Real-time market-moving events
    HIGH = 2        # Time-sensitive signals
    MEDIUM = 3      # Regular updates
    LOW = 4         # Historical/batch data


class SentimentScore(Enum):
    """Sentiment classification levels."""
    EXTREMELY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    EXTREMELY_BULLISH = 2


@dataclass
class AlternativeDataPoint:
    """Single alternative data observation."""
    source: DataSourceType
    timestamp: datetime
    symbol: Optional[str]
    data_type: str
    raw_data: Dict[str, Any]
    processed_data: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    priority: DataPriority = DataPriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.value,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "data_type": self.data_type,
            "confidence": self.confidence,
            "priority": self.priority.value,
            "processed_data": self.processed_data,
            "metadata": self.metadata
        }


@dataclass
class SentimentAggregate:
    """Aggregated sentiment across multiple sources."""
    symbol: str
    timestamp: datetime
    overall_score: float  # -1 to 1
    volume: int  # Number of mentions
    sources: Dict[str, float]  # Source -> score
    trending_topics: List[str]
    confidence: float
    velocity: float  # Rate of sentiment change
    
    def classification(self) -> SentimentScore:
        if self.overall_score < -0.6:
            return SentimentScore.EXTREMELY_BEARISH
        elif self.overall_score < -0.2:
            return SentimentScore.BEARISH
        elif self.overall_score < 0.2:
            return SentimentScore.NEUTRAL
        elif self.overall_score < 0.6:
            return SentimentScore.BULLISH
        else:
            return SentimentScore.EXTREMELY_BULLISH


@dataclass
class OnChainMetrics:
    """On-chain analytics for a blockchain/asset."""
    chain: str
    timestamp: datetime
    whale_transactions: List[Dict[str, Any]]
    exchange_netflow: float  # Positive = inflow, negative = outflow
    active_addresses: int
    transaction_volume: float
    gas_price_gwei: float
    defi_tvl_change: float
    smart_money_flows: List[Dict[str, Any]]
    exchange_reserves: Dict[str, float]  # Exchange -> balance
    confidence: float = 0.0
    
    @property
    def net_exchange_flow_signal(self) -> float:
        """Signal from exchange flows (negative = bullish, positive = bearish)."""
        if self.exchange_netflow > 1000:
            return -0.5  # Large inflow = potential selling pressure
        elif self.exchange_netflow < -1000:
            return 0.5   # Large outflow = accumulation
        return 0.0


@dataclass
class SatelliteObservation:
    """Satellite imagery analysis result."""
    location: str
    facility_type: str  # oil_storage, mining, shipping, etc.
    timestamp: datetime
    analysis_type: str
    metrics: Dict[str, float]
    change_from_baseline: float
    confidence: float
    imagery_url: Optional[str] = None
    
    def trading_signal(self) -> float:
        """Generate trading signal from satellite observation."""
        if self.facility_type == "oil_storage":
            # Increasing storage = bearish, decreasing = bullish
            return -self.change_from_baseline * 0.3
        elif self.facility_type == "mining":
            # More activity = potentially bullish for crypto
            return self.change_from_baseline * 0.2
        elif self.facility_type == "shipping":
            # More ships = more trade activity
            return self.change_from_baseline * 0.15
        return 0.0


class RealTimeDataHub:
    """
    Central hub for real-time alternative data integration.
    
    Manages multiple data sources, prioritizes processing, and provides
    unified access to alternative data signals.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Data storage
        self._data_buffer: Dict[str, List[AlternativeDataPoint]] = defaultdict(list)
        self._sentiment_cache: Dict[str, SentimentAggregate] = {}
        self._onchain_cache: Dict[str, OnChainMetrics] = {}
        self._satellite_cache: Dict[str, SatelliteObservation] = {}
        
        # Processing state
        self._active_sources: Set[DataSourceType] = set()
        self._subscribers: Dict[DataSourceType, List[Callable]] = defaultdict(list)
        self._processing_queue: asyncio.Queue = asyncio.Queue()
        
        # Statistics
        self._stats = {
            "total_received": 0,
            "total_processed": 0,
            "by_source": defaultdict(int),
            "errors": defaultdict(int),
            "last_update": None
        }
        
        # API configurations (would be loaded from config)
        self._api_keys = self._load_api_keys()
        
        logger.info("RealTimeDataHub initialized")
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from config or environment."""
        # In production, load from secure config
        return {
            "twitter_api": "",
            "reddit_api": "",
            "satellite_api": "",
            "onchain_api": "",
            "news_api": ""
        }
    
    async def start(self):
        """Start all data collection streams."""
        logger.info("Starting RealTimeDataHub...")
        
        tasks = [
            self._process_queue_loop(),
            self._sentiment_aggregation_loop(),
            self._onchain_monitoring_loop(),
            self._health_check_loop()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop(self):
        """Gracefully stop all data collection."""
        logger.info("Stopping RealTimeDataHub...")
        self._active_sources.clear()
    
    # =========================================================================
    # Social Sentiment Integration
    # =========================================================================
    
    async def collect_twitter_sentiment(
        self,
        symbols: List[str],
        keywords: Optional[List[str]] = None
    ) -> Dict[str, SentimentAggregate]:
        """
        Collect and analyze Twitter/X sentiment for given symbols.
        
        Uses:
        - Twitter API v2 (Academic Research access)
        - Cashtag and keyword tracking
        - Engagement-weighted scoring
        - Bot detection and filtering
        """
        results = {}
        
        for symbol in symbols:
            try:
                # Simulated Twitter sentiment collection
                # In production, use: tweepy, snscrape, or Twitter API
                
                mentions = await self._fetch_twitter_mentions(symbol, keywords)
                filtered = self._filter_bot_tweets(mentions)
                analyzed = self._analyze_tweet_sentiment(filtered)
                
                aggregate = SentimentAggregate(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    overall_score=analyzed["score"],
                    volume=analyzed["volume"],
                    sources={"twitter": analyzed["score"]},
                    trending_topics=analyzed["topics"],
                    confidence=analyzed["confidence"],
                    velocity=analyzed["velocity"]
                )
                
                self._sentiment_cache[f"twitter_{symbol}"] = aggregate
                results[symbol] = aggregate
                
                self._stats["by_source"]["twitter"] += 1
                
            except Exception as e:
                logger.error(f"Twitter sentiment error for {symbol}: {e}")
                self._stats["errors"]["twitter"] += 1
        
        return results
    
    async def _fetch_twitter_mentions(
        self,
        symbol: str,
        keywords: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Fetch recent Twitter mentions for a symbol."""
        # Placeholder - integrate with Twitter API v2
        # API endpoint: /2/tweets/search/recent
        # Query: ${symbol} OR $${symbol} lang:en
        
        # Simulated response for structure
        return [
            {
                "id": f"tweet_{i}",
                "text": f"${symbol} looking bullish after recent move",
                "created_at": datetime.now().isoformat(),
                "public_metrics": {
                    "like_count": np.random.randint(0, 1000),
                    "retweet_count": np.random.randint(0, 200),
                    "reply_count": np.random.randint(0, 50)
                },
                "author_id": f"user_{i}"
            }
            for i in range(np.random.randint(10, 100))
        ]
    
    def _filter_bot_tweets(self, tweets: List[Dict]) -> List[Dict]:
        """Filter out likely bot accounts."""
        filtered = []
        for tweet in tweets:
            # Simple bot detection heuristics
            metrics = tweet.get("public_metrics", {})
            
            # Skip accounts with suspicious patterns
            engagement_ratio = (
                metrics.get("like_count", 0) + 
                metrics.get("retweet_count", 0)
            )
            
            # In production, use ML-based bot detection
            if engagement_ratio >= 0:  # Placeholder
                filtered.append(tweet)
        
        return filtered
    
    def _analyze_tweet_sentiment(self, tweets: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment from filtered tweets."""
        if not tweets:
            return {
                "score": 0.0,
                "volume": 0,
                "topics": [],
                "confidence": 0.0,
                "velocity": 0.0
            }
        
        # Weighted sentiment based on engagement
        total_weight = 0
        weighted_sentiment = 0
        
        for tweet in tweets:
            metrics = tweet.get("public_metrics", {})
            weight = (
                1 + 
                metrics.get("like_count", 0) * 0.01 +
                metrics.get("retweet_count", 0) * 0.05
            )
            
            # Simple keyword-based sentiment (replace with FinBERT in production)
            text = tweet.get("text", "").lower()
            sentiment = self._simple_sentiment_score(text)
            
            weighted_sentiment += sentiment * weight
            total_weight += weight
        
        avg_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0
        
        # Extract trending topics
        topics = self._extract_topics(tweets)
        
        # Calculate velocity (rate of change)
        velocity = self._calculate_sentiment_velocity(avg_sentiment)
        
        return {
            "score": np.clip(avg_sentiment, -1, 1),
            "volume": len(tweets),
            "topics": topics[:5],
            "confidence": min(len(tweets) / 100, 1.0),
            "velocity": velocity
        }
    
    def _simple_sentiment_score(self, text: str) -> float:
        """Simple keyword-based sentiment scoring."""
        bullish_words = [
            "bullish", "moon", "pump", "breakout", "accumulate",
            "strong", "support", "rally", "surge", "buy"
        ]
        bearish_words = [
            "bearish", "dump", "crash", "sell", "resistance",
            "drop", "fall", "weak", "short", "liquidate"
        ]
        
        bullish_count = sum(1 for w in bullish_words if w in text)
        bearish_count = sum(1 for w in bearish_words if w in text)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        return (bullish_count - bearish_count) / total
    
    def _extract_topics(self, tweets: List[Dict]) -> List[str]:
        """Extract trending topics from tweets."""
        # Simple frequency-based extraction
        # In production, use LDA or BERTopic
        word_freq = defaultdict(int)
        
        for tweet in tweets:
            words = tweet.get("text", "").lower().split()
            for word in words:
                if len(word) > 3 and not word.startswith("http"):
                    word_freq[word] += 1
        
        # Return top topics
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:10]]
    
    def _calculate_sentiment_velocity(self, current_score: float) -> float:
        """Calculate rate of sentiment change."""
        # Store historical scores for velocity calculation
        if not hasattr(self, "_sentiment_history"):
            self._sentiment_history = []
        
        self._sentiment_history.append(current_score)
        if len(self._sentiment_history) > 10:
            self._sentiment_history = self._sentiment_history[-10:]
        
        if len(self._sentiment_history) < 2:
            return 0.0
        
        # Simple velocity: change over last 5 observations
        recent = self._sentiment_history[-5:]
        if len(recent) >= 2:
            return (recent[-1] - recent[0]) / len(recent)
        return 0.0
    
    async def collect_reddit_sentiment(
        self,
        symbol: str,
        subreddits: Optional[List[str]] = None
    ) -> SentimentAggregate:
        """
        Collect sentiment from Reddit (r/cryptocurrency, r/wallstreetbets, etc.).
        """
        subreddits = subreddits or ["cryptocurrency", "bitcoin", "wallstreetbets"]
        
        all_posts = []
        for sub in subreddits:
            posts = await self._fetch_reddit_posts(symbol, sub)
            all_posts.extend(posts)
        
        # Analyze sentiment
        analyzed = self._analyze_tweet_sentiment(all_posts)  # Reuse analysis
        
        aggregate = SentimentAggregate(
            symbol=symbol,
            timestamp=datetime.now(),
            overall_score=analyzed["score"],
            volume=analyzed["volume"],
            sources={"reddit": analyzed["score"]},
            trending_topics=analyzed["topics"],
            confidence=analyzed["confidence"] * 0.9,  # Slightly lower confidence
            velocity=analyzed["velocity"]
        )
        
        self._sentiment_cache[f"reddit_{symbol}"] = aggregate
        return aggregate
    
    async def _fetch_reddit_posts(
        self,
        symbol: str,
        subreddit: str
    ) -> List[Dict[str, Any]]:
        """Fetch Reddit posts mentioning the symbol."""
        # Placeholder for Reddit API integration
        # Use PRAW (Python Reddit API Wrapper) in production
        return [
            {
                "text": f"DD on {symbol}: The fundamentals are strong",
                "score": np.random.randint(0, 5000),
                "num_comments": np.random.randint(0, 200),
                "created_utc": time.time()
            }
            for _ in range(np.random.randint(5, 50))
        ]
    
    # =========================================================================
    # On-Chain Data Integration
    # =========================================================================
    
    async def collect_onchain_metrics(
        self,
        chain: str,
        token_address: Optional[str] = None
    ) -> OnChainMetrics:
        """
        Collect on-chain analytics for a blockchain.
        
        Metrics:
        - Whale transactions (>100k USD)
        - Exchange netflows
        - Active addresses
        - Transaction volume
        - Gas prices
        - DeFi TVL changes
        - Smart money flows
        """
        try:
            # Fetch various on-chain metrics
            whale_txs = await self._fetch_whale_transactions(chain, token_address)
            exchange_flow = await self._fetch_exchange_flows(chain)
            active_addrs = await self._fetch_active_addresses(chain)
            tx_volume = await self._fetch_transaction_volume(chain)
            gas_price = await self._fetch_gas_price(chain)
            defi_tvl = await self._fetch_defi_tvl(chain)
            smart_money = await self._fetch_smart_money_flows(chain)
            reserves = await self._fetch_exchange_reserves(chain)
            
            metrics = OnChainMetrics(
                chain=chain,
                timestamp=datetime.now(),
                whale_transactions=whale_txs,
                exchange_netflow=exchange_flow,
                active_addresses=active_addrs,
                transaction_volume=tx_volume,
                gas_price_gwei=gas_price,
                defi_tvl_change=defi_tvl,
                smart_money_flows=smart_money,
                exchange_reserves=reserves,
                confidence=0.85
            )
            
            self._onchain_cache[chain] = metrics
            self._stats["by_source"]["onchain"] += 1
            
            return metrics
            
        except Exception as e:
            logger.error(f"On-chain metrics error for {chain}: {e}")
            self._stats["errors"]["onchain"] += 1
            raise
    
    async def _fetch_whale_transactions(
        self,
        chain: str,
        token_address: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Fetch large transactions (whale movements)."""
        # Placeholder - integrate with Whale Alert, Glassnode, etc.
        return [
            {
                "hash": f"0x{hashlib.md5(str(i).encode()).hexdigest()}",
                "from": f"0x{''.join(['a']*40)}",
                "to": f"0x{''.join(['b']*40)}",
                "value_usd": np.random.uniform(100000, 10000000),
                "timestamp": datetime.now().isoformat(),
                "type": np.random.choice(["exchange_deposit", "exchange_withdrawal", "transfer"])
            }
            for i in range(np.random.randint(0, 20))
        ]
    
    async def _fetch_exchange_flows(self, chain: str) -> float:
        """Fetch net exchange flow (positive=inflow, negative=outflow)."""
        # Placeholder - integrate with CryptoQuant, Glassnode
        return np.random.uniform(-5000, 5000)  # BTC equivalent
    
    async def _fetch_active_addresses(self, chain: str) -> int:
        """Fetch number of active addresses."""
        base_values = {
            "ethereum": 450000,
            "bitcoin": 900000,
            "solana": 300000,
            "polygon": 200000
        }
        base = base_values.get(chain, 100000)
        return int(base * np.random.uniform(0.8, 1.2))
    
    async def _fetch_transaction_volume(self, chain: str) -> float:
        """Fetch transaction volume in USD."""
        base_values = {
            "ethereum": 5e9,
            "bitcoin": 10e9,
            "solana": 1e9,
            "polygon": 500e6
        }
        base = base_values.get(chain, 100e6)
        return base * np.random.uniform(0.7, 1.3)
    
    async def _fetch_gas_price(self, chain: str) -> float:
        """Fetch current gas price in Gwei."""
        if chain != "ethereum":
            return 0.0
        return np.random.uniform(10, 100)
    
    async def _fetch_defi_tvl(self, chain: str) -> float:
        """Fetch DeFi TVL change (percentage)."""
        return np.random.uniform(-5, 5)
    
    async def _fetch_smart_money_flows(
        self,
        chain: str
    ) -> List[Dict[str, Any]]:
        """Fetch smart money (whale/institution) wallet flows."""
        return [
            {
                "wallet": f"smart_{i}",
                "label": np.random.choice(["whale", "institution", "defi_protocol"]),
                "net_flow": np.random.uniform(-1000000, 1000000),
                "tokens": ["ETH", "BTC", "USDC"][:np.random.randint(1, 4)]
            }
            for i in range(np.random.randint(3, 10))
        ]
    
    async def _fetch_exchange_reserves(
        self,
        chain: str
    ) -> Dict[str, float]:
        """Fetch exchange reserves by exchange."""
        exchanges = ["binance", "coinbase", "kraken", "bybit", "okx"]
        return {
            exchange: np.random.uniform(10000, 500000)
            for exchange in exchanges
        }
    
    # =========================================================================
    # Satellite Imagery Integration
    # =========================================================================
    
    async def collect_satellite_data(
        self,
        facility_type: str,
        location: str
    ) -> SatelliteObservation:
        """
        Collect and analyze satellite imagery data.
        
        Facility types:
        - oil_storage: Track oil reserves (WTI correlation)
        - mining: Crypto mining facility activity
        - shipping: Port activity and trade flows
        - agriculture: Crop health and yield estimates
        """
        try:
            # Fetch satellite analysis
            analysis = await self._analyze_satellite_imagery(
                facility_type,
                location
            )
            
            observation = SatelliteObservation(
                location=location,
                facility_type=facility_type,
                timestamp=datetime.now(),
                analysis_type=analysis["type"],
                metrics=analysis["metrics"],
                change_from_baseline=analysis["change"],
                confidence=analysis["confidence"],
                imagery_url=analysis.get("imagery_url")
            )
            
            self._satellite_cache[f"{facility_type}_{location}"] = observation
            self._stats["by_source"]["satellite"] += 1
            
            return observation
            
        except Exception as e:
            logger.error(f"Satellite data error: {e}")
            self._stats["errors"]["satellite"] += 1
            raise
    
    async def _analyze_satellite_imagery(
        self,
        facility_type: str,
        location: str
    ) -> Dict[str, Any]:
        """Analyze satellite imagery for a facility."""
        # Placeholder - integrate with:
        # - Planet Labs
        # - Maxar
        # - Airbus Defence & Space
        # - Orbital Insight
        
        if facility_type == "oil_storage":
            return {
                "type": "oil_tank_analysis",
                "metrics": {
                    "tank_levels": np.random.uniform(0.3, 0.9),
                    "active_tanks": np.random.randint(50, 100),
                    "total_capacity_pct": np.random.uniform(60, 95)
                },
                "change": np.random.uniform(-0.1, 0.1),
                "confidence": 0.85
            }
        elif facility_type == "mining":
            return {
                "type": "mining_activity",
                "metrics": {
                    "facility_hashrate_estimate": np.random.uniform(0.7, 1.3),
                    "expansion_detected": np.random.choice([True, False]),
                    "truck_activity": np.random.randint(0, 50)
                },
                "change": np.random.uniform(-0.05, 0.15),
                "confidence": 0.75
            }
        elif facility_type == "shipping":
            return {
                "type": "port_activity",
                "metrics": {
                    "vessels_present": np.random.randint(10, 50),
                    "container_count": np.random.randint(1000, 10000),
                    "berth_occupancy": np.random.uniform(0.5, 1.0)
                },
                "change": np.random.uniform(-0.1, 0.1),
                "confidence": 0.80
            }
        
        return {
            "type": "general_analysis",
            "metrics": {},
            "change": 0.0,
            "confidence": 0.5
        }
    
    # =========================================================================
    # News & Economic Data
    # =========================================================================
    
    async def collect_news_sentiment(
        self,
        symbols: List[str],
        lookback_hours: int = 24
    ) -> Dict[str, Dict[str, Any]]:
        """
        Collect and analyze news sentiment.
        
        Sources:
        - Financial news APIs (Bloomberg, Reuters)
        - Crypto news (CoinDesk, The Block)
        - Press releases and filings
        """
        results = {}
        
        for symbol in symbols:
            articles = await self._fetch_news_articles(symbol, lookback_hours)
            analysis = self._analyze_news_sentiment(articles)
            results[symbol] = analysis
        
        return results
    
    async def _fetch_news_articles(
        self,
        symbol: str,
        lookback_hours: int
    ) -> List[Dict[str, Any]]:
        """Fetch recent news articles."""
        # Placeholder - integrate with NewsAPI, Alpha Vantage News
        return [
            {
                "title": f"{symbol} reaches new milestone",
                "description": f"Analysis of {symbol} recent performance...",
                "source": np.random.choice(["CoinDesk", "Bloomberg", "Reuters"]),
                "published_at": datetime.now().isoformat(),
                "url": f"https://example.com/news/{i}"
            }
            for i in range(np.random.randint(5, 30))
        ]
    
    def _analyze_news_sentiment(
        self,
        articles: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze sentiment from news articles."""
        if not articles:
            return {"score": 0.0, "volume": 0, "sources": {}}
        
        source_scores = defaultdict(list)
        
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            score = self._simple_sentiment_score(text.lower())
            source = article.get("source", "unknown")
            source_scores[source].append(score)
        
        # Calculate weighted average
        all_scores = [s for scores in source_scores.values() for s in scores]
        avg_score = np.mean(all_scores) if all_scores else 0.0
        
        return {
            "score": float(np.clip(avg_score, -1, 1)),
            "volume": len(articles),
            "sources": {
                source: float(np.mean(scores))
                for source, scores in source_scores.items()
            },
            "confidence": min(len(articles) / 20, 1.0)
        }
    
    async def collect_economic_indicators(self) -> Dict[str, float]:
        """Collect macro economic indicators."""
        # Placeholder - integrate with FRED, BLS, etc.
        return {
            "fed_funds_rate": 5.25,
            "cpi_yoy": 3.2,
            "gdp_growth": 2.1,
            "unemployment": 3.7,
            "vix": np.random.uniform(12, 30),
            "dxy": np.random.uniform(100, 105),
            "treasury_10y": np.random.uniform(4.0, 5.0)
        }
    
    # =========================================================================
    # Aggregation & Signal Generation
    # =========================================================================
    
    async def get_aggregated_sentiment(
        self,
        symbol: str
    ) -> SentimentAggregate:
        """Get aggregated sentiment across all sources."""
        all_scores = []
        all_sources = {}
        total_volume = 0
        
        # Collect from all cached sources
        for key, aggregate in self._sentiment_cache.items():
            if symbol.lower() in key.lower():
                all_scores.append(aggregate.overall_score)
                all_sources.update(aggregate.sources)
                total_volume += aggregate.volume
        
        if not all_scores:
            return SentimentAggregate(
                symbol=symbol,
                timestamp=datetime.now(),
                overall_score=0.0,
                volume=0,
                sources={},
                trending_topics=[],
                confidence=0.0,
                velocity=0.0
            )
        
        # Weighted average by confidence
        avg_score = np.mean(all_scores)
        
        return SentimentAggregate(
            symbol=symbol,
            timestamp=datetime.now(),
            overall_score=float(avg_score),
            volume=total_volume,
            sources=all_sources,
            trending_topics=[],
            confidence=min(total_volume / 500, 1.0),
            velocity=0.0
        )
    
    def get_combined_signal(
        self,
        symbol: str,
        chain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate combined trading signal from all alternative data sources.
        
        Returns:
        - Overall signal score (-1 to 1)
        - Confidence level
        - Contributing factors
        - Recommended action
        """
        signals = []
        weights = []
        factors = []
        
        # Sentiment signal
        sentiment_key = f"twitter_{symbol}"
        if sentiment_key in self._sentiment_cache:
            agg = self._sentiment_cache[sentiment_key]
            signals.append(agg.overall_score)
            weights.append(agg.confidence)
            factors.append({
                "source": "twitter_sentiment",
                "score": agg.overall_score,
                "confidence": agg.confidence,
                "volume": agg.volume
            })
        
        # On-chain signal
        if chain and chain in self._onchain_cache:
            metrics = self._onchain_cache[chain]
            onchain_signal = metrics.net_exchange_flow_signal
            signals.append(onchain_signal)
            weights.append(metrics.confidence)
            factors.append({
                "source": "onchain_exchange_flow",
                "score": onchain_signal,
                "confidence": metrics.confidence,
                "net_flow": metrics.exchange_netflow
            })
        
        # Satellite signal (if available)
        for key, obs in self._satellite_cache.items():
            sat_signal = obs.trading_signal()
            signals.append(sat_signal)
            weights.append(obs.confidence)
            factors.append({
                "source": f"satellite_{obs.facility_type}",
                "score": sat_signal,
                "confidence": obs.confidence
            })
        
        # Calculate weighted signal
        if not signals:
            return {
                "signal": 0.0,
                "confidence": 0.0,
                "action": "HOLD",
                "factors": []
            }
        
        weights = np.array(weights)
        signals = np.array(signals)
        
        if weights.sum() > 0:
            weighted_signal = np.average(signals, weights=weights)
            avg_confidence = weights.mean()
        else:
            weighted_signal = 0.0
            avg_confidence = 0.0
        
        # Determine action
        if weighted_signal > 0.3 and avg_confidence > 0.5:
            action = "BUY"
        elif weighted_signal < -0.3 and avg_confidence > 0.5:
            action = "SELL"
        else:
            action = "HOLD"
        
        return {
            "signal": float(weighted_signal),
            "confidence": float(avg_confidence),
            "action": action,
            "factors": factors,
            "timestamp": datetime.now().isoformat()
        }
    
    # =========================================================================
    # Background Processing Loops
    # =========================================================================
    
    async def _process_queue_loop(self):
        """Process incoming data from queue."""
        while True:
            try:
                data_point = await self._processing_queue.get()
                await self._process_data_point(data_point)
                self._stats["total_processed"] += 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
    
    async def _process_data_point(self, data_point: AlternativeDataPoint):
        """Process a single data point."""
        # Add to buffer
        key = f"{data_point.source.value}_{data_point.symbol}"
        self._data_buffer[key].append(data_point)
        
        # Trim buffer to last 1000 points
        if len(self._data_buffer[key]) > 1000:
            self._data_buffer[key] = self._data_buffer[key][-1000:]
        
        # Notify subscribers
        for callback in self._subscribers.get(data_point.source, []):
            try:
                await callback(data_point)
            except Exception as e:
                logger.error(f"Subscriber callback error: {e}")
    
    async def _sentiment_aggregation_loop(self):
        """Periodically aggregate sentiment data."""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute
                
                # Update sentiment aggregates
                for symbol in ["BTC", "ETH", "SOL"]:
                    await self.collect_twitter_sentiment([symbol])
                    await self.collect_reddit_sentiment(symbol)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sentiment aggregation error: {e}")
    
    async def _onchain_monitoring_loop(self):
        """Periodically fetch on-chain metrics."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                for chain in ["ethereum", "bitcoin"]:
                    await self.collect_onchain_metrics(chain)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"On-chain monitoring error: {e}")
    
    async def _health_check_loop(self):
        """Monitor data source health."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                self._stats["last_update"] = datetime.now().isoformat()
                
                # Log stats
                logger.info(
                    f"DataHub Stats: "
                    f"received={self._stats['total_received']}, "
                    f"processed={self._stats['total_processed']}, "
                    f"sources={dict(self._stats['by_source'])}"
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def subscribe(
        self,
        source: DataSourceType,
        callback: Callable[[AlternativeDataPoint], Any]
    ):
        """Subscribe to data updates from a source."""
        self._subscribers[source].append(callback)
        self._active_sources.add(source)
        logger.info(f"New subscriber for {source.value}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            **self._stats,
            "active_sources": [s.value for s in self._active_sources],
            "buffer_sizes": {
                k: len(v) for k, v in self._data_buffer.items()
            },
            "cached_sentiment": len(self._sentiment_cache),
            "cached_onchain": len(self._onchain_cache),
            "cached_satellite": len(self._satellite_cache)
        }
    
    def get_latest_data(
        self,
        source: DataSourceType,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[AlternativeDataPoint]:
        """Get latest data points from a source."""
        results = []
        
        for key, buffer in self._data_buffer.items():
            if source.value in key:
                if symbol is None or symbol.lower() in key.lower():
                    results.extend(buffer[-limit:])
        
        return sorted(results, key=lambda x: x.timestamp, reverse=True)[:limit]


# ============================================================================
# Convenience Functions
# ============================================================================

async def create_data_hub(
    config: Optional[Dict[str, Any]] = None
) -> RealTimeDataHub:
    """Create and initialize a RealTimeDataHub instance."""
    hub = RealTimeDataHub(config)
    return hub


async def quick_sentiment_check(
    symbol: str,
    hub: Optional[RealTimeDataHub] = None
) -> Dict[str, Any]:
    """Quick sentiment check for a symbol."""
    if hub is None:
        hub = await create_data_hub()
    
    # Collect from multiple sources
    twitter = await hub.collect_twitter_sentiment([symbol])
    reddit = await hub.collect_reddit_sentiment(symbol)
    
    # Get aggregated
    aggregated = await hub.get_aggregated_sentiment(symbol)
    
    return {
        "symbol": symbol,
        "twitter_score": twitter.get(symbol, SentimentAggregate(
            symbol=symbol,
            timestamp=datetime.now(),
            overall_score=0,
            volume=0,
            sources={},
            trending_topics=[],
            confidence=0,
            velocity=0
        )).overall_score,
        "reddit_score": reddit.overall_score,
        "aggregated_score": aggregated.overall_score,
        "total_volume": aggregated.volume,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Demo usage
    async def demo():
        hub = await create_data_hub()
        
        # Check BTC sentiment
        sentiment = await quick_sentiment_check("BTC", hub)
        print(f"BTC Sentiment: {sentiment}")
        
        # Get on-chain metrics
        onchain = await hub.collect_onchain_metrics("bitcoin")
        print(f"Bitcoin Exchange Netflow: {onchain.exchange_netflow}")
        
        # Get combined signal
        signal = hub.get_combined_signal("BTC", "bitcoin")
        print(f"Combined Signal: {signal}")
    
    asyncio.run(demo())
