"""
Argus External Data Fetcher - Free APIs for Constant Improvement
================================================================

Fetches real-time data from multiple free APIs:
- Derivatives: Funding rates, open interest, liquidations, OI
- Sentiment: Fear & Greed, social sentiment, news
- Whale Alerts: Large transactions
- Macro Economic: CPI, GDP, interest rates, employment

All APIs are FREE and require NO API KEY unless noted.

Data is cached to respect rate limits and fetched at appropriate intervals.
"""

from __future__ import annotations

import logging
import time
import json
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import requests

logger = logging.getLogger(__name__)


@dataclass
class FundingRate:
    """Funding rate from exchange."""
    exchange: str
    symbol: str
    rate: float  # Annualized rate
    timestamp: float


@dataclass
class LiquidationEvent:
    """Liquidation event."""
    symbol: str
    side: str  # "long" or "short"
    value_usd: float
    timestamp: float


@dataclass
class WhaleAlert:
    """Large transaction alert."""
    symbol: str
    amount: float
    value_usd: float
    direction: str  # "in" (to exchange) or "out" (from exchange)
    timestamp: float


@dataclass
class MacroEvent:
    """Macroeconomic event."""
    country: str
    event: str
    actual: float
    forecast: float
    previous: float
    impact: str  # "low", "medium", "high"
    timestamp: float


@dataclass
class MarketSentiment:
    """Combined market sentiment data."""
    fear_greed_index: int  # 0-100
    fear_greed_label: str
    funding_rate_avg: float
    long_short_ratio: float
    sentiment_score: float  # -1 to 1
    timestamp: float


class FreeDataFetcher:
    """
    Fetches data from multiple free APIs.
    
    Rate limits respected via caching and间隔 fetching.
    """
    
    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'Argus/1.0',
            'Accept': 'application/json',
        })
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Cached data
        self._funding_rates: Dict[str, float] = {}
        self._fear_greed: Dict[str, Any] = {}
        self._liquidations: List[LiquidationEvent] = []
        self._whale_alerts: List[WhaleAlert] = []
        self._macro_events: List[MacroEvent] = []
        self._sentiment: Optional[MarketSentiment] = None
        self._open_interest: Dict[str, float] = {}
        
        # Timestamps for rate limiting
        self._last_funding_fetch: float = 0
        self._last_fear_greed_fetch: float = 0
        self._last_liquidation_fetch: float = 0
        self._last_whale_fetch: float = 0
        self._last_macro_fetch: float = 0
        self._last_oi_fetch: float = 0
        self._last_news_fetch: float = 0
        
        # Stats
        self.total_fetches = 0
        self.failed_fetches = 0
        self._fetch_times: deque = deque(maxlen=100)
        
        logger.info("FreeDataFetcher initialized")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # DERIVATIVES DATA
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def fetch_funding_rates(self, force: bool = False) -> Dict[str, float]:
        """
        Fetch funding rates from Loris Tools (free, no key).
        
        Returns dict of {symbol: rate} where rate is annualized.
        Updates every 0.5 seconds (market speed).
        """
        if not force and (time.time() - self._last_funding_fetch) < 0.5:
            return self._funding_rates
        
        try:
            start = time.perf_counter()
            response = self._session.get(
                "https://api.loris.tools/funding",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse funding rates (multiplied by 10000 in API)
            funding_data = data.get("funding_rates", {})
            for exchange, rates in funding_data.items():
                for symbol, raw_rate in rates.items():
                    # Convert from basis points to annualized rate
                    rate = (raw_rate / 10000) * 365 * 3  # 8-hour to annual
                    key = f"{symbol}_{exchange}"
                    self._funding_rates[key] = rate
            
            # Average funding rate across exchanges
            if self._funding_rates:
                avg_rate = sum(self._funding_rates.values()) / len(self._funding_rates)
                self._funding_rates["AVERAGE"] = avg_rate
            
            self._last_funding_fetch = time.perf_counter()
            self.total_fetches += 1
            self._fetch_times.append(time.perf_counter() - start)
            
            logger.debug(f"Fetched {len(self._funding_rates)} funding rates")
            
        except Exception as e:
            self.failed_fetches += 1
            logger.warning(f"Funding rates fetch failed: {e}")
        
        return self._funding_rates
    
    def fetch_liquidations(
        self,
        symbol: Optional[str] = None,
        min_value: float = 100000,
        period: str = "24h",
        force: bool = False,
    ) -> List[LiquidationEvent]:
        """
        Fetch liquidation data from Free Crypto News API (free, no key).
        """
        if not force and (time.time() - self._last_liquidation_fetch) < 0.5:
            return self._liquidations
        
        try:
            start = time.perf_counter()
            params = {"min_value": int(min_value), "period": period}
            if symbol:
                params["symbol"] = symbol
            
            response = self._session.get(
                "https://api.free-crypto-news.com/api/liquidations",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse liquidations
            self._liquidations = []
            for liq in data.get("liquidations", []):
                self._liquidations.append(LiquidationEvent(
                    symbol=liq.get("symbol", ""),
                    side=liq.get("side", ""),
                    value_usd=liq.get("value", 0),
                    timestamp=liq.get("timestamp", time.time()),
                ))
            
            self._last_liquidation_fetch = time.perf_counter()
            self.total_fetches += 1
            self._fetch_times.append(time.perf_counter() - start)
            
            logger.debug(f"Fetched {len(self._liquidations)} liquidation events")
            
        except Exception as e:
            self.failed_fetches += 1
            logger.debug(f"Liquidations fetch failed: {e}")
        
        return self._liquidations
    
    def fetch_open_interest(self, force: bool = False) -> Dict[str, float]:
        """
        Fetch open interest from Sharpe AI (free, no auth).
        """
        if not force and (time.time() - self._last_oi_fetch) < 0.5:
            return self._open_interest
        
        try:
            start = time.perf_counter()
            
            # Sharpe AI free endpoint
            response = self._session.get(
                "https://api.sharpe.ai/free/futures/oi",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse OI data
            for item in data if isinstance(data, list) else []:
                symbol = item.get("symbol", "")
                oi = item.get("open_interest", 0)
                if symbol:
                    self._open_interest[symbol] = oi
            
            self._last_oi_fetch = time.perf_counter()
            self.total_fetches += 1
            self._fetch_times.append(time.perf_counter() - start)
            
            logger.debug(f"Fetched {len(self._open_interest)} OI values")
            
        except Exception as e:
            self.failed_fetches += 1
            logger.debug(f"Open interest fetch failed: {e}")
        
        return self._open_interest
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # SENTIMENT DATA
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def fetch_fear_greed(self, force: bool = False) -> Dict[str, Any]:
        """
        Fetch Fear & Greed Index from Alternative.me (free, no key).
        
        Returns: {"value": 0-100, "label": "Extreme Fear"|"Fear"|"Neutral"|"Greed"|"Extreme Greed"}
        """
        if not force and (time.time() - self._last_fear_greed_fetch) < 0.5:
            return self._fear_greed
        
        try:
            start = time.perf_counter()
            response = self._session.get(
                "https://api.alternative.me/fng/?limit=1",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            latest = data.get("data", [{}])[0]
            self._fear_greed = {
                "value": int(latest.get("value", 50)),
                "label": latest.get("value_classification", "Neutral"),
                "timestamp": int(latest.get("timestamp", time.time())),
            }
            
            self._last_fear_greed_fetch = time.perf_counter()
            self.total_fetches += 1
            self._fetch_times.append(time.perf_counter() - start)
            
            logger.debug(f"Fear & Greed: {self._fear_greed['value']} ({self._fear_greed['label']})")
            
        except Exception as e:
            self.failed_fetches += 1
            logger.debug(f"Fear & Greed fetch failed: {e}")
        
        return self._fear_greed
    
    def fetch_whale_alerts(self, force: bool = False) -> List[WhaleAlert]:
        """
        Fetch whale alerts from Free Crypto News API (free, no key).
        """
        if not force and (time.time() - self._last_whale_fetch) < 0.5:
            return self._whale_alerts
        
        try:
            start = time.perf_counter()
            response = self._session.get(
                "https://api.free-crypto-news.com/api/whale-alerts",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            self._whale_alerts = []
            for alert in data.get("alerts", []):
                self._whale_alerts.append(WhaleAlert(
                    symbol=alert.get("symbol", ""),
                    amount=alert.get("amount", 0),
                    value_usd=alert.get("value_usd", 0),
                    direction=alert.get("direction", "unknown"),
                    timestamp=alert.get("timestamp", time.time()),
                ))
            
            self._last_whale_fetch = time.perf_counter()
            self.total_fetches += 1
            self._fetch_times.append(time.perf_counter() - start)
            
            logger.debug(f"Fetched {len(self._whale_alerts)} whale alerts")
            
        except Exception as e:
            self.failed_fetches += 1
            logger.debug(f"Whale alerts fetch failed: {e}")
        
        return self._whale_alerts
    
    def get_sentiment(self, force: bool = False) -> Optional[MarketSentiment]:
        """
        Combine all sentiment data into a single sentiment score.
        """
        # Fetch all data
        fear_greed = self.fetch_fear_greed(force)
        funding = self.fetch_funding_rates(force)
        
        if not fear_greed and not funding:
            return self._sentiment
        
        fg_value = fear_greed.get("value", 50)
        fg_label = fear_greed.get("label", "Neutral")
        avg_funding = funding.get("AVERAGE", 0.0)
        
        # Calculate sentiment score (-1 to 1)
        # Fear & Greed: 0=Extreme Fear (-1), 50=Neutral (0), 100=Extreme Greed (1)
        fg_score = (fg_value - 50) / 50
        
        # Funding: Negative = shorts paying (bullish), Positive = longs paying (bearish)
        funding_score = max(-1.0, min(1.0, -avg_funding * 100))
        
        # Weighted average
        sentiment_score = fg_score * 0.6 + funding_score * 0.4
        
        self._sentiment = MarketSentiment(
            fear_greed_index=fg_value,
            fear_greed_label=fg_label,
            funding_rate_avg=avg_funding,
            long_short_ratio=1.0,  # Will be updated if OI data available
            sentiment_score=sentiment_score,
            timestamp=time.time(),
        )
        
        return self._sentiment
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # MACRO ECONOMIC DATA
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def fetch_economic_calendar(self, force: bool = False) -> List[MacroEvent]:
        """
        Fetch economic calendar from Fin2Dev (free).
        
        CPI, GDP, NFP, interest rate decisions.
        """
        if not force and (time.time() - self._last_macro_fetch) < 0.5:
            return self._macro_events
        
        try:
            start = time.perf_counter()
            
            # Try World Bank API (completely free, no key)
            response = self._session.get(
                "https://api.worldbank.org/v2/country/US/indicator/FP.CPI.TOTL.ZG",
                params={"format": "json", "per_page": 5, "date": "2024:2025"},
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list) and len(data) > 1:
                for item in data[1][:5]:
                    if item.get("value") is not None:
                        self._macro_events.append(MacroEvent(
                            country="US",
                            event="CPI Inflation Rate",
                            actual=float(item["value"]),
                            forecast=0.0,  # World Bank doesn't provide forecasts
                            previous=0.0,
                            impact="high",
                            timestamp=time.time(),
                        ))
            
            self._last_macro_fetch = time.perf_counter()
            self.total_fetches += 1
            self._fetch_times.append(time.perf_counter() - start)
            
            logger.debug(f"Fetched {len(self._macro_events)} macro events")
            
        except Exception as e:
            self.failed_fetches += 1
            logger.debug(f"Macro data fetch failed: {e}")
        
        return self._macro_events
    
    def fetch_interest_rates(self) -> Dict[str, float]:
        """
        Fetch central bank interest rates from World Bank (free).
        """
        try:
            response = self._session.get(
                "https://api.worldbank.org/v2/country/US;EU;JP;GB/indicator/FR.INR.RISK",
                params={"format": "json", "per_page": 10, "date": "2024:2025"},
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            rates = {}
            if isinstance(data, list) and len(data) > 1:
                for item in data[1]:
                    country = item.get("country", {}).get("value", "")
                    value = item.get("value")
                    if value is not None:
                        rates[country] = float(value)
            
            return rates
            
        except Exception as e:
            logger.debug(f"Interest rates fetch failed: {e}")
            return {}
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # COMBINED DATA FOR TRADING
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def get_trading_signals(self) -> Dict[str, Any]:
        """
        Combine all data sources into trading signals.
        
        Returns dict with:
        - funding_signal: "bullish" | "bearish" | "neutral"
        - sentiment_signal: "bullish" | "bearish" | "neutral"
        - liquidation_pressure: "long_squeeze" | "short_squeeze" | "balanced"
        - macro_environment: "risk_on" | "risk_off" | "neutral"
        - combined_signal: overall recommendation
        - confidence: 0-1
        """
        funding = self.fetch_funding_rates()
        fear_greed = self.fetch_fear_greed()
        liquidations = self.fetch_liquidations()
        sentiment = self.get_sentiment()
        
        # Funding rate signal
        avg_funding = funding.get("AVERAGE", 0)
        if avg_funding > 0.05:  # >5% annualized = overleveraged longs
            funding_signal = "bearish"
            funding_detail = "Longs overleveraged - potential long squeeze"
        elif avg_funding < -0.02:  # Negative = shorts paying
            funding_signal = "bullish"
            funding_detail = "Shorts overleveraged - potential short squeeze"
        else:
            funding_signal = "neutral"
            funding_detail = "Funding rates normal"
        
        # Fear & Greed signal
        fg_value = fear_greed.get("value", 50)
        if fg_value < 25:
            fg_signal = "bullish"  # Extreme fear = buy opportunity
            fg_detail = "Extreme fear - contrarian buy signal"
        elif fg_value > 75:
            fg_signal = "bearish"  # Extreme greed = sell signal
            fg_detail = "Extreme greed - contrarian sell signal"
        else:
            fg_signal = "neutral"
            fg_detail = f"Fear & Greed at {fg_value}"
        
        # Liquidation pressure
        long_liqs = sum(l.value_usd for l in liquidations if l.side == "long")
        short_liqs = sum(l.value_usd for l in liquidations if l.side == "short")
        liq_ratio = long_liqs / max(short_liqs, 1)
        
        if liq_ratio > 2.0:
            liq_signal = "long_squeeze"
            liq_detail = "Heavy long liquidations - bearish pressure"
        elif liq_ratio < 0.5:
            liq_signal = "short_squeeze"
            liq_detail = "Heavy short liquidations - bullish pressure"
        else:
            liq_signal = "balanced"
            liq_detail = "Liquidations balanced"
        
        # Combined signal
        signals = [funding_signal, fg_signal]
        bullish_count = signals.count("bullish")
        bearish_count = signals.count("bearish")
        
        if bullish_count >= 2:
            combined = "BUY"
            confidence = 0.7 + (bullish_count * 0.05)
        elif bearish_count >= 2:
            combined = "SELL"
            confidence = 0.7 + (bearish_count * 0.05)
        else:
            combined = "HOLD"
            confidence = 0.5
        
        return {
            "funding_signal": funding_signal,
            "funding_detail": funding_detail,
            "funding_rate": avg_funding,
            "fear_greed_value": fg_value,
            "fear_greed_label": fear_greed.get("label", "Unknown"),
            "fear_greed_signal": fg_signal,
            "fear_greed_detail": fg_detail,
            "liquidation_signal": liq_signal,
            "liquidation_detail": liq_detail,
            "long_liquidations": long_liqs,
            "short_liquidations": short_liqs,
            "combined_signal": combined,
            "confidence": confidence,
            "sentiment_score": sentiment.sentiment_score if sentiment else 0.0,
            "timestamp": time.time(),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fetcher statistics."""
        return {
            "total_fetches": self.total_fetches,
            "failed_fetches": self.failed_fetches,
            "avg_fetch_time_ms": (sum(self._fetch_times) / len(self._fetch_times) * 1000) if self._fetch_times else 0,
            "funding_rates_cached": len(self._funding_rates),
            "fear_greed_cached": bool(self._fear_greed),
            "liquidations_cached": len(self._liquidations),
            "whale_alerts_cached": len(self._whale_alerts),
            "macro_events_cached": len(self._macro_events),
        }


# Singleton
_fetcher_instance: Optional[FreeDataFetcher] = None

def get_free_data_fetcher() -> FreeDataFetcher:
    """Get or create the free data fetcher singleton."""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = FreeDataFetcher()
    return _fetcher_instance
