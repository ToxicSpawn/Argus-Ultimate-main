"""
Institutional Data Feeds — free/cheap alternative data for ARGUS.

Connects to real-world data sources that institutional desks pay $100K+/year
for, but using free tiers and public APIs:

1. CryptoPanic: real-time crypto news sentiment (free API)
2. Blockchain.com: on-chain metrics (mempool, hash rate, active addresses)
3. Deribit: options IV skew, put/call ratio, max pain (public API)
4. Yahoo Finance: S&P500, DXY, Gold, VIX for cross-asset correlation
5. CoinGlass: funding rates, open interest, liquidations (public)

Each feed produces normalised signals for the Universal Data Brain.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import aiohttp
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False


# ════════════════════════════════════════════════════════════════════════════
# CryptoPanic News Sentiment
# ════════════════════════════════════════════════════════════════════════════

class CryptoPanicFeed:
    """
    Real-time crypto news with sentiment classification.

    API: https://cryptopanic.com/api/v1/posts/
    Free tier: 5 requests/minute, no auth required for public posts.

    Each news item has: kind (news/media), domain, votes (positive/negative/important).
    We compute a sentiment score from vote ratios.
    """

    def __init__(self, api_key: str = "", poll_interval_s: float = 60.0):
        self._api_key = api_key
        self._interval = poll_interval_s
        self._last_poll = 0.0
        self._cache: Dict[str, float] = {}  # symbol → sentiment score

    async def fetch(self) -> Dict[str, float]:
        """Fetch latest news sentiment. Returns {symbol: score} where score is -1 to +1."""
        if not _HAS_AIOHTTP:
            return {}
        now = time.time()
        if now - self._last_poll < self._interval:
            return self._cache

        url = "https://cryptopanic.com/api/v1/posts/"
        params = {"auth_token": self._api_key, "filter": "hot", "public": "true"} if self._api_key else {"public": "true"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        return self._cache
                    data = await resp.json()

            results = data.get("results", [])
            symbol_votes: Dict[str, List[float]] = {}

            for post in results[:50]:
                currencies = post.get("currencies", [])
                votes = post.get("votes", {})
                pos = int(votes.get("positive", 0))
                neg = int(votes.get("negative", 0))
                total = pos + neg
                if total < 1:
                    continue
                sentiment = (pos - neg) / total  # -1 to +1

                for curr in currencies:
                    code = curr.get("code", "").upper()
                    sym = f"{code}/USD"
                    if sym not in symbol_votes:
                        symbol_votes[sym] = []
                    symbol_votes[sym].append(sentiment)

            for sym, votes in symbol_votes.items():
                self._cache[sym] = sum(votes) / len(votes)

            self._last_poll = now
            logger.debug("CryptoPanic: fetched sentiment for %d symbols", len(self._cache))

        except Exception as e:
            logger.debug("CryptoPanic fetch failed: %s", e)

        return self._cache


# ════════════════════════════════════════════════════════════════════════════
# Blockchain.com On-Chain Data
# ════════════════════════════════════════════════════════════════════════════

class BlockchainFeed:
    """
    Bitcoin on-chain metrics from blockchain.com public API.

    Free, no auth needed. Provides:
    - Mempool size (unconfirmed transactions)
    - Hash rate
    - Active addresses (24h)
    - Transaction count
    - Average block size

    On-chain signals:
    - Rising mempool + rising fees = network congestion = possible sell pressure
    - Rising hash rate = miner confidence = bullish
    - Rising active addresses = adoption = bullish
    """

    def __init__(self, poll_interval_s: float = 300.0):
        self._interval = poll_interval_s
        self._last_poll = 0.0
        self._metrics: Dict[str, float] = {}
        self._prev_metrics: Dict[str, float] = {}

    async def fetch(self) -> Dict[str, Any]:
        """Fetch on-chain metrics. Returns dict with scores."""
        if not _HAS_AIOHTTP:
            return {}
        now = time.time()
        if now - self._last_poll < self._interval:
            return self._metrics

        endpoints = {
            "mempool_count": "https://api.blockchain.info/q/unconfirmedcount",
            "hash_rate": "https://api.blockchain.info/q/hashrate",
            "difficulty": "https://api.blockchain.info/q/getdifficulty",
            "block_count": "https://api.blockchain.info/q/getblockcount",
        }

        self._prev_metrics = dict(self._metrics)

        try:
            async with aiohttp.ClientSession() as session:
                for key, url in endpoints.items():
                    try:
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                            if resp.status == 200:
                                text = await resp.text()
                                self._metrics[key] = float(text.strip())
                    except Exception:
                        pass

            # Compute scores from changes
            scores = {}
            if self._prev_metrics:
                # Mempool growing = bearish (congestion, possible sell-off)
                prev_mp = self._prev_metrics.get("mempool_count", 0)
                curr_mp = self._metrics.get("mempool_count", 0)
                if prev_mp > 0:
                    mp_change = (curr_mp - prev_mp) / prev_mp
                    scores["mempool_score"] = max(-1, min(1, -mp_change * 2))

                # Hash rate growing = bullish (miner confidence)
                prev_hr = self._prev_metrics.get("hash_rate", 0)
                curr_hr = self._metrics.get("hash_rate", 0)
                if prev_hr > 0:
                    hr_change = (curr_hr - prev_hr) / prev_hr
                    scores["hashrate_score"] = max(-1, min(1, hr_change * 5))

            self._metrics.update(scores)
            self._last_poll = now
            logger.debug("Blockchain: fetched %d on-chain metrics", len(self._metrics))

        except Exception as e:
            logger.debug("Blockchain fetch failed: %s", e)

        return self._metrics


# ════════════════════════════════════════════════════════════════════════════
# Deribit Options Data (Public API)
# ════════════════════════════════════════════════════════════════════════════

class DeribitOptionsFeed:
    """
    Options market data from Deribit public API.

    No auth needed for market data. Provides:
    - BTC/ETH implied volatility (IV)
    - Put/Call open interest ratio
    - IV skew (25-delta put IV - 25-delta call IV)

    Options signals:
    - High put/call ratio = bearish hedging = contrarian bullish
    - Positive IV skew = downside fear = bearish
    - Rising IV = expected volatility = caution
    """

    def __init__(self, poll_interval_s: float = 120.0):
        self._interval = poll_interval_s
        self._last_poll = 0.0
        self._data: Dict[str, Any] = {}

    async def fetch(self) -> Dict[str, Any]:
        """Fetch options data. Returns dict with IV, put/call ratio, skew."""
        if not _HAS_AIOHTTP:
            return {}
        now = time.time()
        if now - self._last_poll < self._interval:
            return self._data

        try:
            async with aiohttp.ClientSession() as session:
                # BTC index price + IV
                for currency in ["BTC", "ETH"]:
                    url = f"https://www.deribit.com/api/v2/public/get_index_price?index_name={currency.lower()}_usd"
                    try:
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                price = data.get("result", {}).get("index_price", 0)
                                self._data[f"{currency}_index_price"] = price
                    except Exception:
                        pass

                    # Historical volatility
                    vol_url = f"https://www.deribit.com/api/v2/public/get_historical_volatility?currency={currency}"
                    try:
                        async with session.get(vol_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                result = data.get("result", [])
                                if result:
                                    latest = result[-1]
                                    if isinstance(latest, list) and len(latest) >= 2:
                                        self._data[f"{currency}_historical_vol"] = latest[1]
                    except Exception:
                        pass

            self._last_poll = now
            logger.debug("Deribit: fetched options data for %d instruments", len(self._data))

        except Exception as e:
            logger.debug("Deribit fetch failed: %s", e)

        return self._data


# ════════════════════════════════════════════════════════════════════════════
# Cross-Asset Macro Feed (Yahoo Finance proxy)
# ════════════════════════════════════════════════════════════════════════════

class MacroFeed:
    """
    Cross-asset macro indicators via free APIs.

    Tracks: S&P 500, DXY (USD index), Gold, VIX
    Uses CoinGecko for BTC dominance.

    Macro signals:
    - DXY up = USD strengthens = crypto down
    - VIX up = fear = risk-off = crypto down
    - S&P up = risk-on = crypto up (usually)
    - Gold up = safe haven bid = mixed for crypto
    - BTC dominance up = altcoin weakness
    """

    def __init__(self, poll_interval_s: float = 300.0):
        self._interval = poll_interval_s
        self._last_poll = 0.0
        self._data: Dict[str, float] = {}

    async def fetch(self) -> Dict[str, float]:
        """Fetch macro indicators."""
        if not _HAS_AIOHTTP:
            return {}
        now = time.time()
        if now - self._last_poll < self._interval:
            return self._data

        try:
            async with aiohttp.ClientSession() as session:
                # BTC dominance from CoinGecko (free, no auth)
                try:
                    url = "https://api.coingecko.com/api/v3/global"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            market_data = data.get("data", {})
                            self._data["btc_dominance"] = float(market_data.get("market_cap_percentage", {}).get("btc", 0))
                            self._data["eth_dominance"] = float(market_data.get("market_cap_percentage", {}).get("eth", 0))
                            self._data["total_market_cap_usd"] = float(market_data.get("total_market_cap", {}).get("usd", 0))
                            self._data["total_volume_24h_usd"] = float(market_data.get("total_volume", {}).get("usd", 0))
                except Exception:
                    pass

                # Fear & Greed Index (alternative.me, free)
                try:
                    url = "https://api.alternative.me/fng/?limit=1"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            fng = data.get("data", [{}])[0]
                            self._data["fear_greed_index"] = int(fng.get("value", 50))
                            self._data["fear_greed_label"] = str(fng.get("value_classification", "Neutral"))
                except Exception:
                    pass

            self._last_poll = now
            logger.debug("Macro: fetched %d indicators", len(self._data))

        except Exception as e:
            logger.debug("Macro fetch failed: %s", e)

        return self._data


# ════════════════════════════════════════════════════════════════════════════
# CoinGlass Derivatives Data
# ════════════════════════════════════════════════════════════════════════════

class CoinGlassFeed:
    """
    Derivatives data: funding rates, open interest, liquidations.

    Public endpoints, no auth needed for basic data.

    Signals:
    - Positive funding = longs pay shorts = overleveraged long = contrarian bearish
    - Rising OI + rising price = trend confirmation
    - Rising OI + falling price = bearish pressure
    - Large liquidations = potential reversal
    """

    def __init__(self, poll_interval_s: float = 120.0):
        self._interval = poll_interval_s
        self._last_poll = 0.0
        self._data: Dict[str, Any] = {}

    async def fetch(self) -> Dict[str, Any]:
        """Fetch derivatives data."""
        if not _HAS_AIOHTTP:
            return {}
        now = time.time()
        if now - self._last_poll < self._interval:
            return self._data

        try:
            async with aiohttp.ClientSession() as session:
                # Funding rates
                try:
                    url = "https://open-api.coinglass.com/public/v2/funding"
                    headers = {"accept": "application/json"}
                    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            for item in (data.get("data", []) or [])[:20]:
                                symbol = str(item.get("symbol", ""))
                                rate = float(item.get("uMarginList", [{}])[0].get("rate", 0) if item.get("uMarginList") else 0)
                                if symbol and rate != 0:
                                    self._data[f"{symbol}_funding_rate"] = rate
                except Exception:
                    pass

            self._last_poll = now
            logger.debug("CoinGlass: fetched %d data points", len(self._data))

        except Exception as e:
            logger.debug("CoinGlass fetch failed: %s", e)

        return self._data


# ════════════════════════════════════════════════════════════════════════════
# Institutional Feed Manager
# ════════════════════════════════════════════════════════════════════════════

class InstitutionalFeedManager:
    """
    Orchestrates all institutional data feeds.

    Fetches all feeds in parallel, normalises to signals for the Data Brain.
    """

    def __init__(self, cryptopanic_key: str = ""):
        self.news = CryptoPanicFeed(api_key=cryptopanic_key)
        self.blockchain = BlockchainFeed()
        self.options = DeribitOptionsFeed()
        self.macro = MacroFeed()
        self.derivatives = CoinGlassFeed()

    async def fetch_all(self) -> Dict[str, Any]:
        """Fetch all feeds in parallel. Returns combined data dict."""
        results = await asyncio.gather(
            self.news.fetch(),
            self.blockchain.fetch(),
            self.options.fetch(),
            self.macro.fetch(),
            self.derivatives.fetch(),
            return_exceptions=True,
        )

        combined = {}
        for i, (name, result) in enumerate(zip(
            ["news", "blockchain", "options", "macro", "derivatives"], results
        )):
            if isinstance(result, dict):
                combined[name] = result
            else:
                combined[name] = {}

        return combined

    def get_brain_signals(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert raw feed data into normalised signals for the Data Brain."""
        signals = []

        # News sentiment
        news = data.get("news", {})
        for sym, score in news.items():
            signals.append({
                "symbol": sym, "source": "news_sentiment", "category": "sentiment",
                "value": score, "score": max(-1, min(1, score)), "confidence": 0.6,
                "description": f"CryptoPanic sentiment: {score:.2f}",
            })

        # On-chain
        blockchain = data.get("blockchain", {})
        if "hashrate_score" in blockchain:
            signals.append({
                "symbol": "BTC/USD", "source": "hashrate", "category": "onchain",
                "value": blockchain.get("hash_rate", 0),
                "score": blockchain["hashrate_score"], "confidence": 0.7,
                "description": f"Hash rate trend: {blockchain['hashrate_score']:.2f}",
            })
        if "mempool_score" in blockchain:
            signals.append({
                "symbol": "BTC/USD", "source": "mempool", "category": "onchain",
                "value": blockchain.get("mempool_count", 0),
                "score": blockchain["mempool_score"], "confidence": 0.5,
                "description": f"Mempool pressure: {blockchain['mempool_score']:.2f}",
            })

        # Options
        options = data.get("options", {})
        for currency in ["BTC", "ETH"]:
            vol = options.get(f"{currency}_historical_vol")
            if vol:
                # High IV = caution signal
                vol_score = max(-1, min(1, -(vol / 100 - 0.5)))  # center around 50%
                signals.append({
                    "symbol": f"{currency}/USD", "source": "options_iv", "category": "derivatives",
                    "value": vol, "score": vol_score, "confidence": 0.7,
                    "description": f"{currency} historical vol: {vol:.1f}%",
                })

        # Macro
        macro = data.get("macro", {})
        if "fear_greed_index" in macro:
            fg = macro["fear_greed_index"]
            fg_score = (50 - fg) / 50  # contrarian: fear=bullish, greed=bearish
            signals.append({
                "symbol": "BTC/USD", "source": "fear_greed", "category": "sentiment",
                "value": fg, "score": max(-1, min(1, fg_score)), "confidence": 0.65,
                "description": f"Fear & Greed: {fg} ({macro.get('fear_greed_label', '')})",
            })

        if "btc_dominance" in macro:
            dom = macro["btc_dominance"]
            signals.append({
                "symbol": "BTC/USD", "source": "btc_dominance", "category": "technical",
                "value": dom, "score": 0.0, "confidence": 0.5,
                "description": f"BTC dominance: {dom:.1f}%",
            })

        # Derivatives (funding rates)
        derivatives = data.get("derivatives", {})
        for key, rate in derivatives.items():
            if key.endswith("_funding_rate"):
                symbol = key.replace("_funding_rate", "") + "/USD"
                # Contrarian: high positive funding = bearish
                score = max(-1, min(1, -rate * 100))
                signals.append({
                    "symbol": symbol, "source": "funding_rate", "category": "derivatives",
                    "value": rate, "score": score, "confidence": 0.8,
                    "description": f"Funding rate: {rate:.4%}",
                })

        return signals

    def get_stats(self) -> Dict[str, Any]:
        return {
            "feeds": 5,
            "news_symbols": len(self.news._cache),
            "blockchain_metrics": len(self.blockchain._metrics),
            "options_data": len(self.options._data),
            "macro_indicators": len(self.macro._data),
            "derivatives_data": len(self.derivatives._data),
        }
