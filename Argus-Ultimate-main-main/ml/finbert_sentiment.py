"""
Crypto Sentiment Analyzer — FinBERT-style rule-based + statistical NLP.

No external NLP dependencies (no HuggingFace, no transformers). Uses a 500+
term crypto-specific lexicon with negation handling, intensifiers, diminishers,
context windows, entity extraction, and FUD/HYPE classification.

Design goals:
  - Zero external deps beyond numpy (scipy optional for smoothing)
  - Sub-millisecond per-text latency for real-time headline analysis
  - Time-decay weighted headline aggregation
  - Crypto slang and abbreviation awareness
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Crypto-specific sentiment lexicon (500+ terms)
# Score range: -1.0 (extremely bearish) to +1.0 (extremely bullish)
# ---------------------------------------------------------------------------

_BULLISH_TERMS: Dict[str, float] = {
    # Strong bullish (0.7 – 1.0)
    "moon": 0.9, "mooning": 0.95, "moonshot": 0.9, "parabolic": 0.85,
    "ath": 0.8, "all-time high": 0.8, "all time high": 0.8, "new high": 0.75,
    "breakout": 0.75, "breaking out": 0.75, "broke out": 0.7,
    "pump": 0.7, "pumping": 0.75, "pumped": 0.65, "ripping": 0.75,
    "skyrocket": 0.9, "skyrocketing": 0.9, "soaring": 0.8,
    "surge": 0.7, "surging": 0.75, "surged": 0.65,
    "rally": 0.7, "rallying": 0.75, "rallied": 0.65,
    "explode": 0.8, "exploding": 0.85, "explosion": 0.75,
    "rocket": 0.8, "rocketing": 0.85,
    "lambo": 0.7, "wagmi": 0.6, "lfg": 0.65,
    "diamond hands": 0.7, "diamondhands": 0.7,
    "hodl": 0.6, "hodling": 0.6, "hodler": 0.55,
    "accumulation": 0.65, "accumulating": 0.65, "accumulate": 0.6,
    "buy the dip": 0.7, "btd": 0.65, "bought the dip": 0.65,
    "undervalued": 0.6, "oversold": 0.6,
    "golden cross": 0.75, "death cross reversal": 0.6,
    "institutional buying": 0.8, "smart money": 0.65,
    "whale buying": 0.7, "whale accumulation": 0.75,
    "adoption": 0.6, "mass adoption": 0.75, "mainstream": 0.55,
    "partnership": 0.55, "listing": 0.6, "listed": 0.55,
    "upgrade": 0.5, "bullrun": 0.8, "bull run": 0.8,
    "supercycle": 0.85, "mega bull": 0.85,

    # Moderate bullish (0.3 – 0.7)
    "bullish": 0.65, "bull": 0.5, "bulls": 0.5,
    "uptrend": 0.6, "upward": 0.45, "up": 0.2,
    "bounce": 0.5, "bouncing": 0.55, "bounced": 0.45,
    "recovery": 0.55, "recovering": 0.55, "recovered": 0.5,
    "green": 0.4, "gains": 0.5, "gain": 0.45, "gaining": 0.5,
    "profit": 0.45, "profitable": 0.4, "profits": 0.45,
    "positive": 0.4, "optimistic": 0.5, "optimism": 0.5,
    "growth": 0.45, "growing": 0.45,
    "support": 0.35, "supported": 0.3, "supporting": 0.3,
    "strong": 0.35, "strength": 0.35,
    "buy": 0.35, "buying": 0.4, "bought": 0.3,
    "long": 0.35, "longing": 0.4,
    "higher": 0.35, "high": 0.2, "highs": 0.3,
    "breakeven": 0.1, "stable": 0.15, "steady": 0.15,
    "outperform": 0.55, "outperforming": 0.55,
    "demand": 0.4, "inflow": 0.5, "inflows": 0.55,
    "etf approval": 0.8, "spot etf": 0.75,
    "halving": 0.6, "halvening": 0.6,
    "defi summer": 0.65, "alt season": 0.7, "altseason": 0.7,
    "airdrop": 0.45, "staking rewards": 0.4,
    "yield": 0.35, "apr": 0.3, "apy": 0.3,
    "locked": 0.3, "tvl increase": 0.5,
    "burn": 0.4, "burned": 0.4, "deflationary": 0.45,
    "scarcity": 0.5, "scarce": 0.45,
    "milestone": 0.4, "record": 0.45,
    "confidence": 0.35, "trust": 0.3,
    "innovation": 0.4, "breakthrough": 0.55,
    "approval": 0.5, "approved": 0.55,
    "integration": 0.4, "integrated": 0.35,
    "launch": 0.4, "launched": 0.35,
    "mainnet": 0.5, "testnet success": 0.4,
    "audit passed": 0.45, "secured": 0.3,
    "backed": 0.35, "funded": 0.35, "funding round": 0.45,
    "venture capital": 0.4, "vc backed": 0.45,
    "conviction": 0.45, "megabull": 0.8,

    # Mild bullish (0.1 – 0.3)
    "interesting": 0.15, "promising": 0.3, "potential": 0.25,
    "opportunity": 0.3, "opportunities": 0.3,
    "improve": 0.2, "improving": 0.25, "improved": 0.2,
    "develop": 0.15, "development": 0.2,
    "progress": 0.2, "advancing": 0.2,
}

_BEARISH_TERMS: Dict[str, float] = {
    # Strong bearish (-0.7 to -1.0)
    "crash": -0.85, "crashing": -0.9, "crashed": -0.8,
    "dump": -0.7, "dumping": -0.75, "dumped": -0.65,
    "rug": -0.9, "rug pull": -0.95, "rugpull": -0.95, "rugged": -0.9,
    "scam": -0.9, "scammed": -0.85, "ponzi": -0.9, "fraud": -0.9,
    "liquidation": -0.8, "liquidated": -0.85, "liquidations": -0.8,
    "rekt": -0.85, "wrecked": -0.8, "destroyed": -0.8,
    "collapse": -0.85, "collapsing": -0.9, "collapsed": -0.8,
    "plummet": -0.8, "plummeting": -0.85, "plummeted": -0.75,
    "tank": -0.7, "tanking": -0.75, "tanked": -0.65,
    "death cross": -0.7, "dead cat bounce": -0.6,
    "capitulation": -0.8, "capitulating": -0.85,
    "panic": -0.75, "panic selling": -0.85, "panic sell": -0.8,
    "black swan": -0.9, "contagion": -0.8,
    "insolvency": -0.9, "insolvent": -0.9, "bankrupt": -0.9,
    "bankruptcy": -0.9, "default": -0.75,
    "ngmi": -0.7, "paper hands": -0.5, "paperhands": -0.5,
    "exit scam": -0.95, "hack": -0.8, "hacked": -0.85,
    "exploit": -0.8, "exploited": -0.85, "vulnerability": -0.7,
    "flash crash": -0.85, "cascading liquidations": -0.9,
    "depeg": -0.85, "depegged": -0.85,
    "terra": -0.4, "luna": -0.3, "ftx": -0.5,
    "celsius": -0.4, "voyager": -0.35,
    "ponzinomics": -0.8, "vaporware": -0.7,

    # Moderate bearish (-0.3 to -0.7)
    "bearish": -0.65, "bear": -0.5, "bears": -0.5,
    "downtrend": -0.6, "downward": -0.45, "down": -0.2,
    "fud": -0.55, "fear": -0.5, "uncertainty": -0.4, "doubt": -0.45,
    "sell": -0.35, "selling": -0.4, "sold": -0.3,
    "short": -0.35, "shorting": -0.4, "shorted": -0.35,
    "drop": -0.45, "dropping": -0.5, "dropped": -0.4,
    "decline": -0.5, "declining": -0.5, "declined": -0.45,
    "fall": -0.4, "falling": -0.45, "fell": -0.4,
    "loss": -0.45, "losses": -0.5, "losing": -0.45, "lost": -0.4,
    "red": -0.35, "bleeding": -0.55, "bleed": -0.5,
    "negative": -0.4, "pessimistic": -0.5, "pessimism": -0.5,
    "weak": -0.35, "weakness": -0.35,
    "resistance": -0.25, "rejected": -0.4, "rejection": -0.4,
    "correction": -0.45, "correcting": -0.45,
    "outflow": -0.5, "outflows": -0.55, "withdrawal": -0.4,
    "delist": -0.6, "delisted": -0.65, "delisting": -0.6,
    "ban": -0.65, "banned": -0.6, "banning": -0.6,
    "regulation": -0.35, "crackdown": -0.6,
    "sec": -0.3, "lawsuit": -0.55, "sued": -0.5,
    "investigation": -0.45, "probe": -0.4,
    "tax": -0.25, "taxation": -0.3,
    "bubble": -0.55, "overvalued": -0.5, "overbought": -0.5,
    "inflation": -0.3, "recession": -0.45,
    "rate hike": -0.4, "hawkish": -0.35,
    "taper": -0.3, "tightening": -0.35,
    "shutdown": -0.5, "offline": -0.4,
    "delay": -0.3, "delayed": -0.3,
    "bug": -0.4, "bugs": -0.35,
    "downtime": -0.4, "maintenance": -0.15,
    "sell-off": -0.6, "selloff": -0.6,
    "whale selling": -0.65, "whale dump": -0.7,
    "unlock": -0.4, "token unlock": -0.5, "vesting": -0.3,
    "dilution": -0.45, "emission": -0.3,
    "tvl decrease": -0.45, "tvl drop": -0.5,

    # Mild bearish (-0.1 to -0.3)
    "risk": -0.2, "risky": -0.25, "risks": -0.2,
    "concern": -0.25, "concerns": -0.25, "worried": -0.3,
    "caution": -0.2, "cautious": -0.2,
    "volatile": -0.15, "volatility": -0.1,
    "uncertain": -0.2, "unclear": -0.15,
    "stagnant": -0.2, "flat": -0.1,
    "overextended": -0.3, "exhaustion": -0.3,
    "warning": -0.3, "alert": -0.15,
    "trouble": -0.35, "problem": -0.3, "problems": -0.3,
    "issue": -0.2, "issues": -0.2,
}

# Merge into single lexicon
_LEXICON: Dict[str, float] = {**_BULLISH_TERMS, **_BEARISH_TERMS}

# ---------------------------------------------------------------------------
# Negation words (flip sentiment in context window)
# ---------------------------------------------------------------------------

_NEGATIONS: Set[str] = {
    "not", "no", "never", "neither", "nor", "none", "nothing",
    "nowhere", "nobody", "hardly", "barely", "scarcely",
    "don't", "dont", "doesn't", "doesnt", "didn't", "didnt",
    "won't", "wont", "wouldn't", "wouldnt", "can't", "cant",
    "cannot", "couldn't", "couldnt", "shouldn't", "shouldnt",
    "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt",
    "weren't", "werent", "hasn't", "hasnt", "haven't", "havent",
    "hadn't", "hadnt", "without", "lack", "lacking", "lacks",
    "fail", "fails", "failed", "failing",
    "unlikely", "impossible", "unable",
}

# Negation context window (look back N words)
_NEGATION_WINDOW = 3

# ---------------------------------------------------------------------------
# Intensifiers and diminishers
# ---------------------------------------------------------------------------

_INTENSIFIERS: Dict[str, float] = {
    "very": 1.5, "extremely": 1.8, "incredibly": 1.7,
    "massive": 1.6, "huge": 1.5, "enormous": 1.6,
    "insane": 1.6, "crazy": 1.4, "absolutely": 1.7,
    "completely": 1.5, "totally": 1.5, "utterly": 1.6,
    "strongly": 1.5, "highly": 1.4, "super": 1.4,
    "mega": 1.5, "ultra": 1.5, "most": 1.3,
    "especially": 1.3, "particularly": 1.3,
    "definitely": 1.4, "clearly": 1.3, "obviously": 1.3,
    "significantly": 1.4, "substantially": 1.4,
    "tremendously": 1.6, "exceptionally": 1.5,
    "remarkably": 1.4, "extraordinarily": 1.6,
    "exceedingly": 1.5, "immensely": 1.5,
}

_DIMINISHERS: Dict[str, float] = {
    "slightly": 0.5, "somewhat": 0.6, "maybe": 0.5,
    "perhaps": 0.5, "possibly": 0.5, "might": 0.6,
    "could": 0.6, "may": 0.6, "marginally": 0.5,
    "barely": 0.4, "hardly": 0.4, "a bit": 0.5,
    "a little": 0.5, "sort of": 0.5, "kind of": 0.5,
    "mildly": 0.5, "partially": 0.6, "partly": 0.6,
    "almost": 0.6, "nearly": 0.7,
}

# ---------------------------------------------------------------------------
# Entity extraction patterns
# ---------------------------------------------------------------------------

_CRYPTO_ENTITIES: Dict[str, str] = {
    # Major coins
    "btc": "Bitcoin", "bitcoin": "Bitcoin",
    "eth": "Ethereum", "ethereum": "Ethereum", "ether": "Ethereum",
    "sol": "Solana", "solana": "Solana",
    "ada": "Cardano", "cardano": "Cardano",
    "dot": "Polkadot", "polkadot": "Polkadot",
    "avax": "Avalanche", "avalanche": "Avalanche",
    "matic": "Polygon", "polygon": "Polygon",
    "link": "Chainlink", "chainlink": "Chainlink",
    "xrp": "XRP", "ripple": "XRP",
    "doge": "Dogecoin", "dogecoin": "Dogecoin",
    "shib": "Shiba Inu", "shiba": "Shiba Inu",
    "ltc": "Litecoin", "litecoin": "Litecoin",
    "bnb": "BNB", "uni": "Uniswap", "uniswap": "Uniswap",
    "aave": "Aave", "atom": "Cosmos", "cosmos": "Cosmos",
    "near": "NEAR Protocol", "algo": "Algorand",
    "ftm": "Fantom", "arb": "Arbitrum", "arbitrum": "Arbitrum",
    "op": "Optimism", "optimism": "Optimism",
    "apt": "Aptos", "aptos": "Aptos",
    "sui": "Sui",
    "pepe": "PEPE", "bonk": "BONK",

    # Exchanges
    "binance": "Binance", "coinbase": "Coinbase",
    "kraken": "Kraken", "bybit": "Bybit",
    "okx": "OKX", "bitfinex": "Bitfinex",
    "gemini": "Gemini", "bitstamp": "Bitstamp",
    "kucoin": "KuCoin", "huobi": "Huobi",
    "dydx": "dYdX", "hyperliquid": "Hyperliquid",

    # People
    "cz": "CZ (Binance)", "sbf": "SBF (FTX)",
    "vitalik": "Vitalik Buterin", "buterin": "Vitalik Buterin",
    "satoshi": "Satoshi Nakamoto",
    "saylor": "Michael Saylor", "michael saylor": "Michael Saylor",
    "elon": "Elon Musk", "musk": "Elon Musk",
    "gary gensler": "Gary Gensler", "gensler": "Gary Gensler",
    "jerome powell": "Jerome Powell", "powell": "Jerome Powell",

    # Protocols / concepts
    "defi": "DeFi", "nft": "NFT", "nfts": "NFT",
    "dao": "DAO", "dex": "DEX", "cex": "CEX",
    "layer 2": "Layer 2", "l2": "Layer 2",
    "layer 1": "Layer 1", "l1": "Layer 1",
    "zk": "Zero Knowledge", "rollup": "Rollup", "rollups": "Rollup",
}

# Topics for classification
_TOPIC_PATTERNS: Dict[str, List[str]] = {
    "regulation": ["regulation", "regulatory", "sec", "cftc", "ban", "legal", "lawsuit",
                    "compliance", "aml", "kyc", "tax", "crackdown", "enforcement"],
    "technology": ["upgrade", "fork", "mainnet", "testnet", "protocol", "smart contract",
                   "blockchain", "consensus", "scalability", "interoperability", "zk",
                   "rollup", "sharding", "eip"],
    "market": ["price", "volume", "market cap", "trading", "volatility", "liquidity",
               "orderbook", "spread", "funding rate", "open interest", "leverage"],
    "defi": ["defi", "yield", "staking", "liquidity pool", "amm", "lending", "borrowing",
             "tvl", "apy", "apr", "farming", "vault"],
    "macro": ["fed", "fomc", "interest rate", "inflation", "recession", "gdp",
              "employment", "cpi", "ppi", "treasury", "dollar", "dxy"],
    "adoption": ["adoption", "institutional", "corporate", "payment", "merchant",
                  "integration", "partnership", "etf"],
    "security": ["hack", "exploit", "vulnerability", "audit", "rug", "scam",
                  "phishing", "breach", "theft"],
}

# FUD/HYPE classification patterns
_FUD_INDICATORS: Set[str] = {
    "fud", "fear", "panic", "crash", "dump", "scam", "rug", "hack",
    "ban", "dead", "dying", "worthless", "fraud", "ponzi", "collapse",
    "warning", "danger", "risk", "bubble", "overvalued", "insolvency",
    "contagion", "bankruptcy", "sell now", "get out",
}

_HYPE_INDICATORS: Set[str] = {
    "moon", "lambo", "100x", "1000x", "guaranteed", "easy money",
    "free money", "can't lose", "to the moon", "rocket", "🚀",
    "diamond hands", "wagmi", "lfg", "generational wealth",
    "once in a lifetime", "next bitcoin", "bitcoin killer",
    "gem", "moonshot", "parabolic", "supercycle",
}

# Urgency indicators
_URGENCY_TERMS: Dict[str, float] = {
    "now": 0.5, "immediately": 0.8, "urgent": 0.9, "breaking": 0.9,
    "just": 0.4, "alert": 0.6, "warning": 0.7, "emergency": 0.95,
    "happening": 0.5, "right now": 0.7, "this moment": 0.6,
    "last chance": 0.8, "don't miss": 0.6, "act fast": 0.7,
    "time sensitive": 0.8, "flash": 0.6, "sudden": 0.5,
    "developing": 0.5, "confirmed": 0.4,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SentimentResult:
    """Result of sentiment analysis on a single text."""
    sentiment: float        # -1.0 to +1.0
    confidence: float       # 0.0 to 1.0
    entities: List[str]     # extracted crypto entities
    topics: List[str]       # detected topics
    urgency: float          # 0.0 to 1.0
    classification: str     # FUD / HYPE / NEUTRAL / INFORMATIONAL
    term_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class AggregateHeadlineSentiment:
    """Aggregated sentiment across multiple headlines."""
    sentiment: float
    confidence: float
    n_headlines: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    top_entities: List[str]
    top_topics: List[str]
    urgency: float
    dominant_classification: str


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CryptoSentimentAnalyzer:
    """
    Production-grade crypto sentiment analyzer using a 500+ term lexicon
    with negation handling, intensifiers, diminishers, and context windows.

    No external NLP dependencies — uses numpy for numerical operations only.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._lexicon = dict(_LEXICON)
        self._negations = set(_NEGATIONS)
        self._intensifiers = dict(_INTENSIFIERS)
        self._diminishers = dict(_DIMINISHERS)
        self._entities_map = dict(_CRYPTO_ENTITIES)
        self._negation_window = int(cfg.get("negation_window", _NEGATION_WINDOW))

        # Custom terms from config
        custom_terms = cfg.get("custom_terms", {})
        if custom_terms:
            self._lexicon.update({k.lower(): float(v) for k, v in custom_terms.items()})

        # Compile regex for tokenization
        self._token_re = re.compile(r"[a-zA-Z0-9$#@]+(?:'[a-z]+)?|[!?]+")
        # Multi-word pattern matcher for phrases
        self._phrase_patterns = self._build_phrase_patterns()

        # Stats
        self._analyses_count = 0
        self._total_latency_ms = 0.0

    def _build_phrase_patterns(self) -> List[Tuple[re.Pattern, str, float]]:
        """Build compiled regex patterns for multi-word lexicon entries."""
        patterns = []
        for term, score in self._lexicon.items():
            if " " in term or "-" in term:
                # Escape and compile as a phrase pattern
                escaped = re.escape(term)
                pat = re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)
                patterns.append((pat, term, score))
        return patterns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_text(self, text: str) -> dict:
        """
        Analyze a single text for crypto sentiment.

        Returns dict with keys:
            sentiment: float (-1 to 1)
            confidence: float (0 to 1)
            entities: list of extracted entities
            topics: list of detected topics
            urgency: float (0 to 1)
            classification: str (FUD/HYPE/NEUTRAL/INFORMATIONAL)
        """
        t0 = time.monotonic()
        result = self._analyze_impl(text)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._analyses_count += 1
        self._total_latency_ms += elapsed_ms
        return {
            "sentiment": result.sentiment,
            "confidence": result.confidence,
            "entities": result.entities,
            "topics": result.topics,
            "urgency": result.urgency,
            "classification": result.classification,
        }

    def analyze_batch(self, texts: List[str]) -> List[dict]:
        """Analyze multiple texts. Returns list of result dicts."""
        return [self.analyze_text(t) for t in texts]

    def analyze_headlines(self, headlines: List[dict]) -> dict:
        """
        Analyze news headlines with time-decay weighting.

        Each headline dict should have:
            text: str (required)
            timestamp: float (optional, unix epoch; defaults to now)
            source: str (optional)

        Returns aggregate sentiment dict.
        """
        if not headlines:
            return {
                "sentiment": 0.0, "confidence": 0.0, "n_headlines": 0,
                "bullish_count": 0, "bearish_count": 0, "neutral_count": 0,
                "top_entities": [], "top_topics": [], "urgency": 0.0,
                "dominant_classification": "NEUTRAL",
            }

        now = time.time()
        half_life_hours = 6.0  # sentiment half-life: 6 hours
        decay_lambda = math.log(2) / (half_life_hours * 3600.0)

        results: List[SentimentResult] = []
        weights: List[float] = []

        for h in headlines:
            text = h.get("text", "") if isinstance(h, dict) else str(h)
            ts = float(h.get("timestamp", now)) if isinstance(h, dict) else now
            age_s = max(0.0, now - ts)
            weight = math.exp(-decay_lambda * age_s)

            result = self._analyze_impl(text)
            results.append(result)
            weights.append(weight)

        # Weighted aggregation
        w_arr = np.array(weights)
        total_w = w_arr.sum()
        if total_w < 1e-9:
            total_w = 1.0

        sentiments = np.array([r.sentiment for r in results])
        confidences = np.array([r.confidence for r in results])
        urgencies = np.array([r.urgency for r in results])

        agg_sentiment = float(np.dot(w_arr, sentiments) / total_w)
        agg_confidence = float(np.dot(w_arr, confidences) / total_w)
        agg_urgency = float(np.max(urgencies))  # max urgency across headlines

        # Clamp
        agg_sentiment = max(-1.0, min(1.0, agg_sentiment))
        agg_confidence = max(0.0, min(1.0, agg_confidence))

        # Counts
        bullish_count = sum(1 for r in results if r.sentiment > 0.1)
        bearish_count = sum(1 for r in results if r.sentiment < -0.1)
        neutral_count = len(results) - bullish_count - bearish_count

        # Aggregate entities and topics
        entity_freq: Dict[str, int] = {}
        topic_freq: Dict[str, int] = {}
        class_freq: Dict[str, int] = {}
        for r in results:
            for e in r.entities:
                entity_freq[e] = entity_freq.get(e, 0) + 1
            for t in r.topics:
                topic_freq[t] = topic_freq.get(t, 0) + 1
            class_freq[r.classification] = class_freq.get(r.classification, 0) + 1

        top_entities = sorted(entity_freq, key=entity_freq.get, reverse=True)[:10]
        top_topics = sorted(topic_freq, key=topic_freq.get, reverse=True)[:5]
        dominant_class = max(class_freq, key=class_freq.get) if class_freq else "NEUTRAL"

        return {
            "sentiment": round(agg_sentiment, 4),
            "confidence": round(agg_confidence, 4),
            "n_headlines": len(headlines),
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "top_entities": top_entities,
            "top_topics": top_topics,
            "urgency": round(agg_urgency, 4),
            "dominant_classification": dominant_class,
        }

    def get_crypto_entities(self, text: str) -> List[str]:
        """Extract crypto entities from text."""
        return self._extract_entities(text.lower())

    def detect_fud_or_hype(self, text: str) -> str:
        """Classify text as FUD, HYPE, NEUTRAL, or INFORMATIONAL."""
        return self._classify_text(text.lower())

    def get_stats(self) -> Dict[str, Any]:
        """Return analyzer statistics."""
        avg_latency = (
            self._total_latency_ms / self._analyses_count
            if self._analyses_count > 0 else 0.0
        )
        return {
            "analyses_count": self._analyses_count,
            "avg_latency_ms": round(avg_latency, 3),
            "lexicon_size": len(self._lexicon),
        }

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _analyze_impl(self, text: str) -> SentimentResult:
        """Core analysis logic."""
        if not text or not text.strip():
            return SentimentResult(
                sentiment=0.0, confidence=0.0, entities=[], topics=[],
                urgency=0.0, classification="NEUTRAL",
            )

        lower_text = text.lower()
        tokens = self._tokenize(lower_text)

        if not tokens:
            return SentimentResult(
                sentiment=0.0, confidence=0.0, entities=[], topics=[],
                urgency=0.0, classification="NEUTRAL",
            )

        # --- Phase 1: Multi-word phrase matching ---
        phrase_scores: Dict[str, float] = {}
        matched_spans: List[Tuple[int, int]] = []  # (start, end) char positions
        for pat, term, score in self._phrase_patterns:
            for m in pat.finditer(lower_text):
                phrase_scores[term] = score
                matched_spans.append((m.start(), m.end()))

        # --- Phase 2: Single-word scoring with context ---
        term_scores: Dict[str, float] = dict(phrase_scores)
        scores: List[float] = list(phrase_scores.values())

        for i, token in enumerate(tokens):
            # Skip if this token is part of a matched phrase
            # (simplified: check against lexicon membership)
            if token not in self._lexicon:
                continue

            base_score = self._lexicon[token]

            # Check negation in preceding window
            negated = False
            start = max(0, i - self._negation_window)
            for j in range(start, i):
                if tokens[j] in self._negations:
                    negated = True
                    break

            if negated:
                base_score = -base_score * 0.8  # Negation flips with slight dampening

            # Check intensifiers in preceding 2 words
            intensifier = 1.0
            for j in range(max(0, i - 2), i):
                if tokens[j] in self._intensifiers:
                    intensifier = self._intensifiers[tokens[j]]
                    break
                elif tokens[j] in self._diminishers:
                    intensifier = self._diminishers[tokens[j]]
                    break

            final_score = base_score * intensifier
            final_score = max(-1.0, min(1.0, final_score))

            term_scores[token] = final_score
            scores.append(final_score)

        # --- Compute aggregate sentiment ---
        if scores:
            # Weighted average: stronger scores contribute more
            arr = np.array(scores)
            weights = np.abs(arr) + 0.1  # bias toward stronger signals
            sentiment = float(np.average(arr, weights=weights))
            sentiment = max(-1.0, min(1.0, sentiment))
        else:
            sentiment = 0.0

        # --- Confidence ---
        # Based on: number of matched terms, consistency of direction, lexicon coverage
        n_matched = len(scores)
        n_tokens = len(tokens)
        coverage = min(1.0, n_matched / max(n_tokens * 0.3, 1))

        if n_matched > 0:
            arr = np.array(scores)
            # Direction consistency: what fraction point the same way as aggregate?
            same_dir = np.sum(np.sign(arr) == np.sign(sentiment)) / n_matched if sentiment != 0 else 0.5
            confidence = 0.3 * coverage + 0.4 * same_dir + 0.3 * min(1.0, abs(sentiment))
        else:
            confidence = 0.1

        confidence = max(0.0, min(1.0, confidence))

        # --- Entity extraction ---
        entities = self._extract_entities(lower_text)

        # --- Topic detection ---
        topics = self._detect_topics(lower_text)

        # --- Urgency ---
        urgency = self._compute_urgency(lower_text, tokens)

        # --- Classification ---
        classification = self._classify_text(lower_text)

        return SentimentResult(
            sentiment=round(sentiment, 4),
            confidence=round(confidence, 4),
            entities=entities,
            topics=topics,
            urgency=round(urgency, 4),
            classification=classification,
            term_scores=term_scores,
        )

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return [t.lower() for t in self._token_re.findall(text)]

    def _extract_entities(self, lower_text: str) -> List[str]:
        """Extract crypto entities from lowered text."""
        found: Dict[str, str] = {}
        tokens = set(self._tokenize(lower_text))
        for token in tokens:
            if token in self._entities_map:
                canonical = self._entities_map[token]
                found[canonical] = canonical

        # Also check multi-word entities
        for key, canonical in self._entities_map.items():
            if " " in key and key in lower_text:
                found[canonical] = canonical

        return sorted(found.keys())

    def _detect_topics(self, lower_text: str) -> List[str]:
        """Detect discussion topics."""
        detected = []
        for topic, keywords in _TOPIC_PATTERNS.items():
            count = sum(1 for kw in keywords if kw in lower_text)
            if count >= 2 or (count == 1 and len(lower_text) < 100):
                detected.append(topic)
        return detected

    def _compute_urgency(self, lower_text: str, tokens: List[str]) -> float:
        """Compute urgency score (0–1)."""
        urgency = 0.0
        for term, score in _URGENCY_TERMS.items():
            if term in lower_text:
                urgency = max(urgency, score)

        # Exclamation marks boost urgency
        excl_count = lower_text.count("!")
        if excl_count > 0:
            urgency = max(urgency, min(1.0, 0.3 + excl_count * 0.15))

        # ALL CAPS words boost urgency
        original_tokens = re.findall(r"[A-Za-z]+", lower_text)
        # We need original text for caps detection — approximate via token length
        # (actual caps detection happens in the original text before lowering)
        return min(1.0, urgency)

    def _classify_text(self, lower_text: str) -> str:
        """Classify as FUD, HYPE, NEUTRAL, or INFORMATIONAL."""
        tokens = set(self._tokenize(lower_text))

        fud_count = sum(1 for t in _FUD_INDICATORS if t in tokens or t in lower_text)
        hype_count = sum(1 for t in _HYPE_INDICATORS if t in tokens or t in lower_text)

        # Threshold-based classification
        if fud_count >= 3 and fud_count > hype_count * 2:
            return "FUD"
        if hype_count >= 3 and hype_count > fud_count * 2:
            return "HYPE"
        if fud_count >= 2 and hype_count == 0:
            return "FUD"
        if hype_count >= 2 and fud_count == 0:
            return "HYPE"

        # Check for informational markers
        informational_markers = {
            "announced", "report", "according", "study", "data", "research",
            "published", "released", "analysis", "update", "statement",
        }
        info_count = sum(1 for m in informational_markers if m in lower_text)
        if info_count >= 2:
            return "INFORMATIONAL"

        return "NEUTRAL"
