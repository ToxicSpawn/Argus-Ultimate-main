"""Google Trends tracker for cryptocurrency market sentiment.

Tracks Google search trends for crypto keywords and generates
sentiment signals based on search interest changes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_KEYWORDS = ["bitcoin", "ethereum", "crypto", "blockchain", "defi"]


@dataclass
class TrendSignal:
    """Google Trends signal for a keyword.
    
    Attributes
    ----------
    keyword : str
        Search keyword
    score : int
        Trend score (0-100, where 50 is neutral)
    trend_direction : str
        Direction: "rising", "falling", "stable"
    change_pct : float
        Percentage change from previous period
    timeframe : str
        Timeframe string (e.g., "now 7-d", "today 3-m")
    timestamp : float
        Unix timestamp of the signal
    """
    keyword: str = ""
    score: int = 50
    trend_direction: str = "stable"
    change_pct: float = 0.0
    timeframe: str = "now 7-d"
    timestamp: float = field(default_factory=time.time)


class GoogleTrendsTracker:
    """Google Trends tracker for crypto market sentiment.
    
    Parameters
    ----------
    cache_ttl_s : float
        Cache time-to-live in seconds (default 3600)
    """
    
    def __init__(self, cache_ttl_s: float = 3600.0) -> None:
        self._cache_ttl_s = cache_ttl_s
        self._cache: Dict[str, TrendSignal] = {}
        self._pytrends: Optional[Any] = None
        
        # Try to initialize pytrends
        try:
            from pytrends.request import TrendReq
            self._pytrends = TrendReq()
            logger.info("GoogleTrendsTracker initialized with pytrends")
        except ImportError:
            logger.info("pytrends not installed, using neutral signals")
            self._pytrends = None
    
    def get_trend_score(
        self,
        keyword: str,
        timeframe: str = "now 7-d",
    ) -> TrendSignal:
        """Get trend score for a keyword.
        
        Returns cached signal if fresh, otherwise fetches new data.
        Returns neutral signal if pytrends is not installed.
        """
        cache_key = f"{keyword}:{timeframe}"
        
        # Check cache
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if time.time() - cached.timestamp < self._cache_ttl_s:
                return cached
        
        # Generate signal
        if self._pytrends is None:
            signal = TrendSignal(
                keyword=keyword,
                score=50,
                trend_direction="stable",
                change_pct=0.0,
                timeframe=timeframe,
            )
        else:
            signal = self._fetch_trend(keyword, timeframe)
        
        self._cache[cache_key] = signal
        return signal
    
    def _fetch_trend(self, keyword: str, timeframe: str) -> TrendSignal:
        """Fetch trend data from Google Trends."""
        try:
            self._pytrends.build_payload([keyword], timeframe=timeframe)
            data = self._pytrends.interest_over_time()
            
            if data.empty or keyword not in data.columns:
                return TrendSignal(keyword=keyword, timeframe=timeframe)
            
            values = data[keyword].values
            if len(values) < 2:
                return TrendSignal(keyword=keyword, score=int(values[0]) if len(values) > 0 else 50, timeframe=timeframe)
            
            current = values[-1]
            previous = values[-2]
            change_pct = ((current - previous) / max(previous, 1)) * 100
            
            if change_pct > 5:
                direction = "rising"
            elif change_pct < -5:
                direction = "falling"
            else:
                direction = "stable"
            
            return TrendSignal(
                keyword=keyword,
                score=int(current),
                trend_direction=direction,
                change_pct=change_pct,
                timeframe=timeframe,
            )
        except Exception as e:
            logger.warning("Failed to fetch trend for %s: %s", keyword, e)
            return TrendSignal(keyword=keyword, timeframe=timeframe)
    
    def get_multi_keyword_signal(
        self,
        keywords: Optional[List[str]] = None,
    ) -> float:
        """Get aggregate signal from multiple keywords.
        
        Returns score in [-1, 1] range.
        """
        if not keywords:
            keywords = DEFAULT_KEYWORDS
        
        if not keywords:
            return 0.0
        
        signals = [self.get_trend_score(kw) for kw in keywords]
        
        # Convert scores (0-100) to [-1, 1] range
        normalized = [(s.score - 50) / 50.0 for s in signals]
        return sum(normalized) / len(normalized)
    
    def is_euphoria(self, threshold: float = 80.0) -> bool:
        """Check if market is in euphoria state (high search interest)."""
        signal = self.get_trend_score("bitcoin")
        return signal.score >= threshold
    
    def is_capitulation(self, threshold: float = 20.0) -> bool:
        """Check if market is in capitulation state (low search interest)."""
        signal = self.get_trend_score("bitcoin")
        return signal.score <= threshold
    
    def clear_cache(self) -> None:
        """Clear all cached signals."""
        self._cache.clear()
