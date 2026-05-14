"""CryptoPanic news feed for sentiment analysis.

Fetches news headlines from CryptoPanic API and provides sentiment signals.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CRYPTOPANIC_API_URL = "https://cryptopanic.com/api/v1"


@dataclass
class NewsHeadline:
    """A single news headline from CryptoPanic.
    
    Attributes
    ----------
    title : str
        Headline text
    url : str
        Source URL
    source : str
        News source name
    published_at : float
        Unix timestamp when published
    currencies : list[str]
        Related currency symbols
    sentiment : str
        Sentiment label: "positive", "negative", "neutral"
    votes_positive : int
        Number of positive votes
    votes_negative : int
        Number of negative votes
    """
    title: str = ""
    url: str = ""
    source: str = ""
    published_at: float = field(default_factory=time.time)
    currencies: List[str] = field(default_factory=list)
    sentiment: str = "neutral"
    votes_positive: int = 0
    votes_negative: int = 0
    
    @property
    def sentiment_score(self) -> float:
        """Compute sentiment score in [-1, 1] range."""
        total = self.votes_positive + self.votes_negative
        if total == 0:
            return 0.0
        return (self.votes_positive - self.votes_negative) / total


class CryptoPanicFeed:
    """CryptoPanic news feed provider.
    
    Parameters
    ----------
    api_key : str
        CryptoPanic API key
    cache_ttl_s : float
        Cache time-to-live in seconds (default 300)
    """
    
    def __init__(
        self,
        api_key: str = "",
        cache_ttl_s: float = 300.0,
    ) -> None:
        self._api_key = api_key
        self._cache_ttl_s = cache_ttl_s
        self._cache: List[NewsHeadline] = []
        self._last_fetch_time: float = 0.0
    
    async def get_headlines(
        self,
        currencies: Optional[List[str]] = None,
        kind: str = "news",
    ) -> List[NewsHeadline]:
        """Fetch recent headlines.
        
        Returns cached headlines if fresh, otherwise fetches new data.
        Returns empty list if no API key is set.
        """
        if not self._api_key:
            return []
        
        now = time.time()
        
        # Return cached if fresh
        if (
            len(self._cache) > 0
            and (now - self._last_fetch_time) < self._cache_ttl_s
        ):
            return self._cache
        
        try:
            import aiohttp
            
            params = {
                "auth_token": self._api_key,
                "kind": kind,
            }
            if currencies:
                params["currencies"] = ",".join(currencies)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{CRYPTOPANIC_API_URL}/posts/",
                    params=params,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get("results", [])
                        self._cache = [
                            NewsHeadline(
                                title=r.get("title", ""),
                                url=r.get("url", ""),
                                source=r.get("source", {}).get("title", ""),
                                published_at=r.get("published_at", ""),
                                currencies=[
                                    c.get("code", "")
                                    for c in r.get("currencies", [])
                                ],
                                sentiment=self._parse_sentiment(r),
                                votes_positive=r.get("votes", {}).get("positive", 0),
                                votes_negative=r.get("votes", {}).get("negative", 0),
                            )
                            for r in results
                        ]
                        self._last_fetch_time = now
                        return self._cache
                    else:
                        logger.warning("CryptoPanic API returned status %d", resp.status)
                        return self._cache if self._cache else []
                        
        except Exception as e:
            logger.warning("Failed to fetch CryptoPanic headlines: %s", e)
            return self._cache if self._cache else []
    
    def _parse_sentiment(self, result: Dict[str, Any]) -> str:
        """Parse sentiment from API result."""
        votes = result.get("votes", {})
        pos = votes.get("positive", 0)
        neg = votes.get("negative", 0)
        
        if pos > neg * 2:
            return "positive"
        elif neg > pos * 2:
            return "negative"
        return "neutral"
    
    def get_aggregate_sentiment(
        self,
        headlines: List[NewsHeadline],
        symbol: Optional[str] = None,
    ) -> float:
        """Compute aggregate sentiment score for headlines.
        
        Returns score in [-1, 1] range.
        """
        if symbol:
            headlines = [h for h in headlines if symbol in h.currencies]
        
        if not headlines:
            return 0.0
        
        scores = [h.sentiment_score for h in headlines]
        return sum(scores) / len(scores)
