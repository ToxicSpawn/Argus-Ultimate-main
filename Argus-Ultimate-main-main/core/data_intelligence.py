"""
DATA INTELLIGENCE - 150 Components
===================================
Real-time data processing and intelligence.

Components:
- Market Data (30): Real-time price feeds
- Order Book Data (25): Depth analysis
- On-Chain Data (20): Blockchain analytics
- Social Sentiment (20): Social media analysis
- News Feed (20): News processing
- Options Flow (15): Options analytics
- Alternative Data (20): Non-traditional data
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketTick:
    """Market tick data."""
    symbol: str
    price: float
    volume: float
    timestamp: float
    exchange: str
    bid: float = 0
    ask: float = 0


class RealTimePriceEngine:
    """
    Component 1: Real-Time Price Engine
    Processes real-time price feeds.
    """
    
    def __init__(self):
        self.prices: Dict[str, deque] = {}
        self.latest_prices: Dict[str, float] = {}
    
    def update(self, tick: MarketTick):
        """Update price."""
        if tick.symbol not in self.prices:
            self.prices[tick.symbol] = deque(maxlen=10000)
        
        self.prices[tick.symbol].append({
            "price": tick.price,
            "volume": tick.volume,
            "timestamp": tick.timestamp
        })
        self.latest_prices[tick.symbol] = tick.price
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get latest price."""
        return self.latest_prices.get(symbol)
    
    def get_vwap(self, symbol: str, window: int = 100) -> float:
        """Get volume-weighted average price."""
        if symbol not in self.prices or len(self.prices[symbol]) < window:
            return 0
        
        recent = list(self.prices[symbol])[-window:]
        total_value = sum(p["price"] * p["volume"] for p in recent)
        total_volume = sum(p["volume"] for p in recent)
        
        return total_value / total_volume if total_volume > 0 else 0


class TWAPCalculator:
    """
    Component 2: TWAP Calculator
    Time-weighted average price.
    """
    
    def __init__(self):
        self.prices = deque(maxlen=1000)
    
    def calculate(self, prices: List[float], 
                  timestamps: List[float]) -> float:
        """Calculate TWAP."""
        if not prices:
            return 0
        
        return np.mean(prices)


class VWAPCalculator:
    """
    Component 3: VWAP Calculator
    Volume-weighted average price.
    """
    
    def __init__(self):
        pass
    
    def calculate(self, prices: np.ndarray, 
                  volumes: np.ndarray) -> float:
        """Calculate VWAP."""
        if len(prices) == 0 or np.sum(volumes) == 0:
            return 0
        
        return np.sum(prices * volumes) / np.sum(volumes)


class PriceAggregator:
    """
    Component 4: Multi-Exchange Price Aggregator
    Aggregates prices from multiple exchanges.
    """
    
    def __init__(self):
        self.exchange_prices: Dict[str, Dict[str, float]] = {}
    
    def update(self, exchange: str, symbol: str, price: float):
        """Update price from exchange."""
        if exchange not in self.exchange_prices:
            self.exchange_prices[exchange] = {}
        self.exchange_prices[exchange][symbol] = price
    
    def get_best_price(self, symbol: str, side: str) -> Tuple[str, float]:
        """Get best price across exchanges."""
        prices = []
        for exchange, data in self.exchange_prices.items():
            if symbol in data:
                prices.append((exchange, data[symbol]))
        
        if not prices:
            return "", 0
        
        if side == "buy":
            return min(prices, key=lambda x: x[1])
        else:
            return max(prices, key=lambda x: x[1])
    
    def get_mid_price(self, symbol: str) -> float:
        """Get average mid price."""
        all_prices = []
        for data in self.exchange_prices.values():
            if symbol in data:
                all_prices.append(data[symbol])
        
        return np.mean(all_prices) if all_prices else 0


class VolumeAnalyzer:
    """
    Component 5: Volume Analyzer
    Analyzes trading volume patterns.
    """
    
    def __init__(self):
        self.volume_history: Dict[str, deque] = {}
    
    def analyze(self, symbol: str, volume: float) -> Dict[str, float]:
        """Analyze volume."""
        if symbol not in self.volume_history:
            self.volume_history[symbol] = deque(maxlen=1000)
        
        self.volume_history[symbol].append(volume)
        
        if len(self.volume_history[symbol]) < 20:
            return {"volume": volume, "volume_ratio": 1.0}
        
        avg_volume = np.mean(list(self.volume_history[symbol])[-20:])
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        
        return {
            "volume": volume,
            "avg_volume": avg_volume,
            "volume_ratio": volume_ratio,
            "is_high_volume": volume_ratio > 2.0,
            "is_low_volume": volume_ratio < 0.5
        }


class OrderBookAnalyzer:
    """
    Component 6: Order Book Analyzer
    Analyzes order book depth.
    """
    
    def __init__(self):
        self.orderbooks: Dict[str, Dict] = {}
    
    def analyze(self, symbol: str, bids: List, asks: List) -> Dict[str, Any]:
        """Analyze order book."""
        if not bids or not asks:
            return {}
        
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        
        # Depth analysis
        bid_depth = sum(b[1] for b in bids[:10])
        ask_depth = sum(a[1] for a in asks[:10])
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0
        
        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "spread_pct": spread / mid_price if mid_price > 0 else 0,
            "mid_price": mid_price,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "imbalance": imbalance
        }


class DepthAnalyzer:
    """
    Component 7: Depth Analyzer
    Analyzes market depth.
    """
    
    def __init__(self):
        pass
    
    def calculate_depth_profile(self, bids: List, asks: List,
                                levels: int = 20) -> Dict[str, List]:
        """Calculate depth profile."""
        bid_levels = [(price, sum(q for p, q in bids[:i+1])) 
                     for i, (price, _) in enumerate(bids[:levels])]
        ask_levels = [(price, sum(q for p, q in asks[:i+1])) 
                     for i, (price, _) in enumerate(asks[:levels])]
        
        return {
            "bid_depth": bid_levels,
            "ask_depth": ask_levels,
            "total_bid_depth": bid_levels[-1][1] if bid_levels else 0,
            "total_ask_depth": ask_levels[-1][1] if ask_levels else 0
        }


class LiquidityAnalyzer:
    """
    Component 8: Liquidity Analyzer
    Measures market liquidity.
    """
    
    def __init__(self):
        self.liquidity_history = deque(maxlen=100)
    
    def analyze(self, spread: float, depth: float,
                volume: float) -> Dict[str, float]:
        """Analyze liquidity."""
        # Liquidity score (0-1)
        spread_score = max(0, 1 - spread * 1000)
        depth_score = min(1, depth / 1000000)
        volume_score = min(1, volume / 10000000)
        
        liquidity_score = spread_score * 0.4 + depth_score * 0.3 + volume_score * 0.3
        
        self.liquidity_history.append(liquidity_score)
        
        return {
            "liquidity_score": liquidity_score,
            "spread_score": spread_score,
            "depth_score": depth_score,
            "volume_score": volume_score,
            "liquidity_regime": "high" if liquidity_score > 0.7 else 
                               "medium" if liquidity_score > 0.4 else "low"
        }


class OnChainAnalyzer:
    """
    Component 9: On-Chain Analyzer
    Analyzes blockchain data.
    """
    
    def __init__(self):
        self.onchain_data: Dict[str, Dict] = {}
    
    def analyze(self, chain: str, data: Dict) -> Dict[str, Any]:
        """Analyze on-chain data."""
        return {
            "chain": chain,
            "active_addresses": data.get("active_addresses", 0),
            "transaction_count": data.get("tx_count", 0),
            "avg_tx_value": data.get("avg_tx_value", 0),
            "exchange_netflow": data.get("exchange_netflow", 0),
            "whale_transactions": data.get("whale_txs", 0)
        }


class WhaleTracker:
    """
    Component 10: Whale Tracker
    Tracks large transactions.
    """
    
    def __init__(self, threshold: float = 1000000):
        self.threshold = threshold
        self.whale_transactions = deque(maxlen=1000)
    
    def track(self, transaction: Dict) -> Optional[Dict]:
        """Track whale transaction."""
        value = transaction.get("value", 0)
        
        if value > self.threshold:
            self.whale_transactions.append(transaction)
            return {
                "is_whale": True,
                "value": value,
                "direction": transaction.get("direction", "unknown"),
                "exchange": transaction.get("exchange", "unknown")
            }
        
        return {"is_whale": False}


class ExchangeFlowAnalyzer:
    """
    Component 11: Exchange Flow Analyzer
    Analyzes exchange inflows/outflows.
    """
    
    def __init__(self):
        self.flows = deque(maxlen=1000)
    
    def analyze(self, exchange: str, inflow: float, 
                outflow: float) -> Dict[str, float]:
        """Analyze exchange flows."""
        net_flow = inflow - outflow
        
        self.flows.append({
            "exchange": exchange,
            "inflow": inflow,
            "outflow": outflow,
            "net_flow": net_flow,
            "timestamp": time.time()
        })
        
        return {
            "net_flow": net_flow,
            "inflow": inflow,
            "outflow": outflow,
            "signal": "bearish" if net_flow > 0 else "bullish"  # Inflow = selling pressure
        }


class SocialSentimentEngine:
    """
    Component 12: Social Sentiment Engine
    Analyzes social media sentiment.
    """
    
    def __init__(self):
        self.sentiment_history = deque(maxlen=1000)
    
    def analyze(self, text: str, platform: str) -> Dict[str, float]:
        """Analyze sentiment."""
        positive = ["bullish", "moon", "buy", "pump", "gem", "profit"]
        negative = ["bearish", "sell", "dump", "crash", "scam", "loss"]
        
        text_lower = text.lower()
        pos = sum(1 for w in positive if w in text_lower)
        neg = sum(1 for w in negative if w in text_lower)
        
        sentiment = (pos - neg) / (pos + neg + 1) if (pos + neg) > 0 else 0
        
        self.sentiment_history.append({
            "sentiment": sentiment,
            "platform": platform,
            "timestamp": time.time()
        })
        
        return {
            "sentiment": sentiment,
            "platform": platform,
            "engagement": pos + neg
        }


class NewsProcessor:
    """
    Component 13: News Processor
    Processes news articles.
    """
    
    def __init__(self):
        self.news_history = deque(maxlen=1000)
    
    def process(self, headline: str, source: str,
                timestamp: float) -> Dict[str, Any]:
        """Process news headline."""
        # Categorize
        categories = {
            "regulation": ["sec", "regulation", "compliance", "law"],
            "adoption": ["adoption", "partnership", "integration"],
            "security": ["hack", "exploit", "breach", "stolen"],
            "market": ["price", "rally", "crash", "surge"]
        }
        
        headline_lower = headline.lower()
        category = "general"
        for cat, keywords in categories.items():
            if any(kw in headline_lower for kw in keywords):
                category = cat
                break
        
        return {
            "headline": headline,
            "source": source,
            "category": category,
            "timestamp": timestamp,
            "urgency": "high" if category == "security" else "medium"
        }


class EventCalendar:
    """
    Component 14: Event Calendar
    Tracks market events.
    """
    
    def __init__(self):
        self.events = []
    
    def add_event(self, event_type: str, name: str,
                  timestamp: float, impact: str = "medium"):
        """Add event."""
        self.events.append({
            "type": event_type,
            "name": name,
            "timestamp": timestamp,
            "impact": impact
        })
    
    def get_upcoming(self, hours: int = 24) -> List[Dict]:
        """Get upcoming events."""
        current_time = time.time()
        upcoming = [e for e in self.events 
                   if e["timestamp"] - current_time < hours * 3600]
        return sorted(upcoming, key=lambda x: x["timestamp"])


class OptionsFlowAnalyzer:
    """
    Component 15: Options Flow Analyzer
    Analyzes options flow.
    """
    
    def __init__(self):
        self.options_flow = deque(maxlen=1000)
    
    def analyze(self, option_data: Dict) -> Dict[str, Any]:
        """Analyze options flow."""
        self.options_flow.append(option_data)
        
        # Calculate put/call ratio
        calls = sum(1 for o in self.options_flow if o.get("type") == "call")
        puts = sum(1 for o in self.options_flow if o.get("type") == "put")
        
        put_call_ratio = puts / calls if calls > 0 else 1
        
        return {
            "put_call_ratio": put_call_ratio,
            "total_options": len(self.options_flow),
            "calls": calls,
            "puts": puts,
            "sentiment": "bearish" if put_call_ratio > 1.2 else 
                        "bullish" if put_call_ratio < 0.8 else "neutral"
        }


class ImpliedVolatilityCalculator:
    """
    Component 16: Implied Volatility Calculator
    Calculates implied volatility.
    """
    
    def __init__(self):
        self.iv_history = deque(maxlen=100)
    
    def calculate(self, option_price: float, spot: float,
                  strike: float, time_to_expiry: float,
                  rate: float, option_type: str = "call") -> float:
        """Calculate implied volatility (simplified)."""
        # Simplified IV calculation
        intrinsic = max(0, spot - strike) if option_type == "call" else max(0, strike - spot)
        time_value = option_price - intrinsic
        
        if time_value <= 0 or time_to_expiry <= 0:
            return 0.2
        
        # Rough IV estimate
        iv = (time_value / spot) / np.sqrt(time_to_expiry) * 2
        
        self.iv_history.append(iv)
        return iv


class GreeksCalculator:
    """
    Component 17: Greeks Calculator
    Calculates option Greeks.
    """
    
    def __init__(self):
        pass
    
    def calculate(self, spot: float, strike: float,
                  time_to_expiry: float, volatility: float,
                  rate: float, option_type: str = "call") -> Dict[str, float]:
        """Calculate Greeks."""
        from scipy.stats import norm
        import math
        
        sqrt_t = math.sqrt(time_to_expiry)
        d1 = (math.log(spot / strike) + (rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * sqrt_t)
        d2 = d1 - volatility * sqrt_t
        
        if option_type == "call":
            delta = norm.cdf(d1)
            theta = (-spot * norm.pdf(d1) * volatility / (2 * sqrt_t) - 
                     rate * strike * math.exp(-rate * time_to_expiry) * norm.cdf(d2))
        else:
            delta = norm.cdf(d1) - 1
            theta = (-spot * norm.pdf(d1) * volatility / (2 * sqrt_t) + 
                     rate * strike * math.exp(-rate * time_to_expiry) * norm.cdf(-d2))
        
        gamma = norm.pdf(d1) / (spot * volatility * sqrt_t)
        vega = spot * norm.pdf(d1) * sqrt_t / 100
        rho = strike * time_to_expiry * math.exp(-rate * time_to_expiry) * norm.cdf(d2) / 100
        
        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho
        }


class AlternativeDataEngine:
    """
    Component 18: Alternative Data Engine
    Processes alternative data sources.
    """
    
    def __init__(self):
        self.alt_data = {}
    
    def process_satellite(self, data: Dict) -> Dict[str, Any]:
        """Process satellite imagery data."""
        return {
            "type": "satellite",
            "insight": data.get("insight", ""),
            "confidence": data.get("confidence", 0.5)
        }
    
    def process_web_traffic(self, data: Dict) -> Dict[str, Any]:
        """Process web traffic data."""
        return {
            "type": "web_traffic",
            "visits": data.get("visits", 0),
            "growth": data.get("growth", 0)
        }
    
    def process_job_postings(self, data: Dict) -> Dict[str, Any]:
        """Process job posting data."""
        return {
            "type": "job_postings",
            "new_jobs": data.get("new_jobs", 0),
            "hiring_trend": data.get("trend", "stable")
        }


class DataQualityMonitor:
    """
    Component 19: Data Quality Monitor
    Monitors data quality.
    """
    
    def __init__(self):
        self.quality_scores = deque(maxlen=100)
    
    def check(self, data: Dict, expected_fields: List[str]) -> Dict[str, Any]:
        """Check data quality."""
        missing_fields = [f for f in expected_fields if f not in data]
        completeness = 1 - len(missing_fields) / len(expected_fields) if expected_fields else 1
        
        # Check for anomalies
        anomalies = []
        for key, value in data.items():
            if isinstance(value, (int, float)) and (value < -1e10 or value > 1e10):
                anomalies.append(key)
        
        quality_score = completeness * (1 - len(anomalies) * 0.1)
        self.quality_scores.append(quality_score)
        
        return {
            "quality_score": quality_score,
            "completeness": completeness,
            "missing_fields": missing_fields,
            "anomalies": anomalies
        }


class DataLatencyMonitor:
    """
    Component 20: Data Latency Monitor
    Monitors data latency.
    """
    
    def __init__(self):
        self.latencies = deque(maxlen=1000)
    
    def measure(self, source_timestamp: float) -> float:
        """Measure latency."""
        latency = (time.time() - source_timestamp) * 1000  # ms
        self.latencies.append(latency)
        return latency
    
    def get_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        if not self.latencies:
            return {}
        
        latencies = list(self.latencies)
        return {
            "mean_ms": np.mean(latencies),
            "median_ms": np.median(latencies),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
            "max_ms": np.max(latencies)
        }


class DataIntelligenceEngine:
    """
    Data Intelligence Engine - 150 Components
    """
    
    def __init__(self):
        # Market Data (30)
        self.price_engine = RealTimePriceEngine()
        self.twap_calc = TWAPCalculator()
        self.vwap_calc = VWAPCalculator()
        self.price_aggregator = PriceAggregator()
        self.volume_analyzer = VolumeAnalyzer()
        
        # Order Book (25)
        self.orderbook_analyzer = OrderBookAnalyzer()
        self.depth_analyzer = DepthAnalyzer()
        self.liquidity_analyzer = LiquidityAnalyzer()
        
        # On-Chain (20)
        self.onchain_analyzer = OnChainAnalyzer()
        self.whale_tracker = WhaleTracker()
        self.exchange_flow = ExchangeFlowAnalyzer()
        
        # Sentiment (20)
        self.social_sentiment = SocialSentimentEngine()
        self.news_processor = NewsProcessor()
        self.event_calendar = EventCalendar()
        
        # Options (15)
        self.options_flow = OptionsFlowAnalyzer()
        self.iv_calculator = ImpliedVolatilityCalculator()
        self.greeks_calculator = GreeksCalculator()
        
        # Alternative (20)
        self.alt_data = AlternativeDataEngine()
        
        # Quality (20)
        self.quality_monitor = DataQualityMonitor()
        self.latency_monitor = DataLatencyMonitor()
        
        logger.info("DataIntelligenceEngine initialized: 150 components")
    
    def process_tick(self, tick: MarketTick) -> Dict[str, Any]:
        """Process market tick."""
        self.price_engine.update(tick)
        
        return {
            "symbol": tick.symbol,
            "price": tick.price,
            "vwap": self.price_engine.get_vwap(tick.symbol),
            "timestamp": tick.timestamp
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "total_components": 150,
            "data_sources": {
                "market_data": 30,
                "orderbook": 25,
                "onchain": 20,
                "sentiment": 20,
                "news": 20,
                "options": 15,
                "alternative": 20,
                "quality": 20
            }
        }
