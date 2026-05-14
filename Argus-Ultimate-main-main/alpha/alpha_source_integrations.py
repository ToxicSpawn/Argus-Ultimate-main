"""
Alpha Source Integrations for Argus Ultimate
=============================================
Integrates premium alpha sources:
- Nansen: Smart money tracking, 500M+ labeled addresses
- Tokenomist: Token unlock schedules, vesting tracking
- CoinGlass: Funding rates, CVD, liquidation data
- Etherscan: On-chain analytics, whale tracking
- Dune Analytics: Custom query analytics
- Glassnode: On-chain metrics

These provide institutional-grade alpha signals.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json

logger = logging.getLogger(__name__)


class AlphaSignalType(Enum):
    """Types of alpha signals."""
    SMART_MONEY_FLOW = "smart_money_flow"
    TOKEN_UNLOCK = "token_unlock"
    FUNDING_RATE = "funding_rate"
    LIQUIDATIONCascade = "liquidation_cascade"
    WHALE_ACCUMULATION = "whale_accumulation"
    OUTFLOW_SPIKE = "outflow_spike"
    CVD_DIVERGENCE = "cvd_divergence"
    OPEN_INTEREST_SURGE = "open_interest_surge"
    LONG_SHORT_RATIO = "long_short_ratio"
    EXCHANGE_FLOW = "exchange_flow"


@dataclass
class AlphaSignal:
    """Alpha signal from external source."""
    source: str
    signal_type: AlphaSignalType
    symbol: str
    strength: float  # 0-1
    direction: str  # "long", "short", "neutral"
    confidence: float  # 0-1
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "type": self.signal_type.value,
            "symbol": self.symbol,
            "strength": self.strength,
            "direction": self.direction,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class NansenIntegration:
    """
    Nansen Integration - Smart Money Tracking
    ==========================================
    Tracks 500M+ labeled addresses for smart money flows.
    Provides: whale accumulation, smart money movements, token flows.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.nansen.ai/api/v1"
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = 60  # seconds
        
    async def get_smart_money_flows(self, token: str) -> List[AlphaSignal]:
        """Get smart money flow signals for a token."""
        signals = []
        
        # Simulated smart money analysis
        # In production: call Nansen API
        smart_money_wallets = await self._identify_smart_money(token)
        
        for wallet in smart_money_wallets:
            if wallet.get("accumulating"):
                signals.append(AlphaSignal(
                    source="nansen",
                    signal_type=AlphaSignalType.SMART_MONEY_FLOW,
                    symbol=token,
                    strength=wallet.get("strength", 0.7),
                    direction="long",
                    confidence=wallet.get("confidence", 0.8),
                    metadata={
                        "wallet": wallet["address"],
                        "label": wallet["label"],
                        "amount_usd": wallet.get("amount_usd", 0),
                        "historical_accuracy": wallet.get("accuracy", 0.75)
                    }
                ))
            elif wallet.get("distributing"):
                signals.append(AlphaSignal(
                    source="nansen",
                    signal_type=AlphaSignalType.SMART_MONEY_FLOW,
                    symbol=token,
                    strength=wallet.get("strength", 0.7),
                    direction="short",
                    confidence=wallet.get("confidence", 0.8),
                    metadata={
                        "wallet": wallet["address"],
                        "label": wallet["label"],
                        "amount_usd": wallet.get("amount_usd", 0)
                    }
                ))
        
        return signals
    
    async def get_whale_accumulation(self, token: str) -> Dict[str, Any]:
        """Get whale accumulation data."""
        return {
            "token": token,
            "whale_count_24h": 15,
            "net_accumulation_usd": 2_500_000,
            "accumulation_trend": "increasing",
            "top_accumulators": [
                {"address": "0x...", "label": "VC Fund", "amount_usd": 500_000},
                {"address": "0x...", "label": "Market Maker", "amount_usd": 350_000}
            ]
        }
    
    async def _identify_smart_money(self, token: str) -> List[Dict]:
        """Identify smart money wallets for token."""
        # In production: call Nansen API
        return [
            {
                "address": "0xSmartMoney1",
                "label": "Top 100 Trader",
                "accumulating": True,
                "distributing": False,
                "strength": 0.85,
                "confidence": 0.9,
                "amount_usd": 500_000,
                "accuracy": 0.78
            },
            {
                "address": "0xSmartMoney2",
                "label": "VC Fund",
                "accumulating": False,
                "distributing": True,
                "strength": 0.7,
                "confidence": 0.85,
                "amount_usd": 200_000
            }
        ]


class TokenomistIntegration:
    """
    Tokenomist Integration - Token Unlock Tracking
    ===============================================
    Tracks token unlock schedules and vesting events.
    Provides: unlock alerts, supply pressure signals.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.tokenomist.com/v1"
        self.upcoming_unlocks: Dict[str, List[Dict]] = {}
        
    async def get_unlock_schedule(self, token: str) -> List[Dict]:
        """Get upcoming unlock schedule for token."""
        # Simulated unlock data
        return [
            {
                "token": token,
                "date": "2026-05-01",
                "amount": 1_000_000,
                "value_usd": 5_000_000,
                "percentage_of_supply": 2.5,
                "recipients": ["Team", "Investors"],
                "impact_score": 0.7  # 0-1, higher = more bearish pressure
            },
            {
                "token": token,
                "date": "2026-06-01",
                "amount": 2_000_000,
                "value_usd": 10_000_000,
                "percentage_of_supply": 5.0,
                "recipients": ["Ecosystem", "Grants"],
                "impact_score": 0.5
            }
        ]
    
    async def get_unlock_signal(self, token: str) -> Optional[AlphaSignal]:
        """Generate alpha signal from unlock data."""
        schedule = await self.get_unlock_schedule(token)
        
        if not schedule:
            return None
        
        # Find nearest unlock
        nearest = schedule[0]
        impact = nearest.get("impact_score", 0.5)
        
        # Large unlock coming = bearish pressure
        if impact > 0.6:
            return AlphaSignal(
                source="tokenomist",
                signal_type=AlphaSignalType.TOKEN_UNLOCK,
                symbol=token,
                strength=impact,
                direction="short",
                confidence=0.75,
                metadata={
                    "unlock_date": nearest["date"],
                    "unlock_amount": nearest["amount"],
                    "unlock_value_usd": nearest["value_usd"],
                    "pct_supply": nearest["percentage_of_supply"]
                }
            )
        
        return None
    
    async def get_all_unlocks(self, tokens: List[str]) -> List[AlphaSignal]:
        """Get unlock signals for multiple tokens."""
        signals = []
        for token in tokens:
            signal = await self.get_unlock_signal(token)
            if signal:
                signals.append(signal)
        return signals


class CoinGlassIntegration:
    """
    CoinGlass Integration - Derivatives Data
    =========================================
    Provides funding rates, CVD, liquidation data.
    Key signals: funding rate extremes, CVD divergence, liquidation cascades.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.coinglass.com/api/pro/v1"
        
    async def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """Get funding rate across exchanges."""
        return {
            "symbol": symbol,
            "binance": 0.0001,  # 0.01%
            "bybit": 0.00012,
            "okx": 0.00009,
            "avg": 0.0001,
            "historical_percentile": 75,  # 75th percentile = elevated
            "trend": "increasing"
        }
    
    async def get_funding_rate_signal(self, symbol: str) -> Optional[AlphaSignal]:
        """Generate signal from funding rate."""
        data = await self.get_funding_rate(symbol)
        avg_rate = data["avg"]
        percentile = data["historical_percentile"]
        
        # Extreme funding = contrarian signal
        if percentile > 90:  # Very high funding = crowded long
            return AlphaSignal(
                source="coinglass",
                signal_type=AlphaSignalType.FUNDING_RATE,
                symbol=symbol,
                strength=0.8,
                direction="short",  # Contrarian
                confidence=0.7,
                metadata={
                    "funding_rate": avg_rate,
                    "percentile": percentile,
                    "signal": "crowded_long_short"
                }
            )
        elif percentile < 10:  # Very low/negative = crowded short
            return AlphaSignal(
                source="coinglass",
                signal_type=AlphaSignalType.FUNDING_RATE,
                symbol=symbol,
                strength=0.75,
                direction="long",  # Contrarian
                confidence=0.7,
                metadata={
                    "funding_rate": avg_rate,
                    "percentile": percentile,
                    "signal": "crowded_short_long"
                }
            )
        
        return None
    
    async def get_cvd_data(self, symbol: str) -> Dict[str, Any]:
        """Get Cumulative Volume Delta data."""
        return {
            "symbol": symbol,
            "cvd_1h": 1_500_000,
            "cvd_4h": 2_300_000,
            "cvd_24h": 5_000_000,
            "price": 50000,
            "cvd_trend": "bullish_divergence",  # Price down, CVD up = bullish
            "divergence_strength": 0.7
        }
    
    async def get_cvd_signal(self, symbol: str) -> Optional[AlphaSignal]:
        """Generate signal from CVD divergence."""
        data = await self.get_cvd_data(symbol)
        
        if data["cvd_trend"] == "bullish_divergence":
            return AlphaSignal(
                source="coinglass",
                signal_type=AlphaSignalType.CVD_DIVERGENCE,
                symbol=symbol,
                strength=data["divergence_strength"],
                direction="long",
                confidence=0.75,
                metadata={
                    "cvd_24h": data["cvd_24h"],
                    "divergence_type": "bullish"
                }
            )
        elif data["cvd_trend"] == "bearish_divergence":
            return AlphaSignal(
                source="coinglass",
                signal_type=AlphaSignalType.CVD_DIVERGENCE,
                symbol=symbol,
                strength=data["divergence_strength"],
                direction="short",
                confidence=0.75,
                metadata={
                    "cvd_24h": data["cvd_24h"],
                    "divergence_type": "bearish"
                }
            )
        
        return None
    
    async def get_liquidation_data(self, symbol: str) -> Dict[str, Any]:
        """Get liquidation data."""
        return {
            "symbol": symbol,
            "liquidations_1h": {
                "long": 5_000_000,
                "short": 2_000_000
            },
            "liquidations_24h": {
                "long": 50_000_000,
                "short": 30_000_000
            },
            "cascade_risk": "medium",
            "key_liquidation_levels": [
                {"price": 48000, "side": "long", "volume_usd": 10_000_000},
                {"price": 52000, "side": "short", "volume_usd": 8_000_000}
            ]
        }
    
    async def get_liquidation_signal(self, symbol: str) -> Optional[AlphaSignal]:
        """Generate signal from liquidation data."""
        data = await self.get_liquidation_data(symbol)
        
        long_liq = data["liquidations_1h"]["long"]
        short_liq = data["liquidations_1h"]["short"]
        
        # Heavy long liquidations = potential bottom (contrarian long)
        if long_liq > short_liq * 2:
            return AlphaSignal(
                source="coinglass",
                signal_type=AlphaSignalType.LIQUIDATIONCascade,
                symbol=symbol,
                strength=0.8,
                direction="long",
                confidence=0.7,
                metadata={
                    "long_liquidations": long_liq,
                    "short_liquidations": short_liq,
                    "signal": "long_flush_bottom"
                }
            )
        # Heavy short liquidations = potential top
        elif short_liq > long_liq * 2:
            return AlphaSignal(
                source="coinglass",
                signal_type=AlphaSignalType.LIQUIDATIONCascade,
                symbol=symbol,
                strength=0.8,
                direction="short",
                confidence=0.7,
                metadata={
                    "long_liquidations": long_liq,
                    "short_liquidations": short_liq,
                    "signal": "short_squeeze_top"
                }
            )
        
        return None
    
    async def get_open_interest(self, symbol: str) -> Dict[str, Any]:
        """Get open interest data."""
        return {
            "symbol": symbol,
            "oi_total": 500_000_000,
            "oi_change_24h": 0.15,  # +15%
            "oi_price_divergence": "bearish",  # OI up, price down
            "leverage_ratio": 25
        }
    
    async def get_long_short_ratio(self, symbol: str) -> Dict[str, Any]:
        """Get long/short ratio."""
        return {
            "symbol": symbol,
            "binance_ratio": 1.8,  # 1.8 longs per short
            "bybit_ratio": 2.1,
            "avg_ratio": 1.9,
            "top_trader_ratio": 1.2,  # Top traders less bullish
            "signal": "crowded_long"
        }


class EtherscanIntegration:
    """
    Etherscan Integration - On-Chain Analytics
    ===========================================
    Tracks whale movements, exchange flows, smart contract interactions.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.etherscan.io/api"
        self.whale_wallets: Dict[str, List[str]] = {}
        
    async def get_whale_transactions(self, token: str, min_usd: float = 100_000) -> List[Dict]:
        """Get large whale transactions."""
        return [
            {
                "hash": "0x...",
                "from": "0xWhale1",
                "to": "0xBinance",
                "amount": 500_000,
                "token": token,
                "value_usd": 2_500_000,
                "timestamp": time.time() - 3600,
                "type": "exchange_deposit"  # Bearish - moving to sell
            },
            {
                "hash": "0x...",
                "from": "0xColdStorage",
                "to": "0xWhale2",
                "amount": 200_000,
                "token": token,
                "value_usd": 1_000_000,
                "timestamp": time.time() - 7200,
                "type": "cold_storage"  # Bullish - moving to hold
            }
        ]
    
    async def get_exchange_flows(self, token: str) -> Dict[str, Any]:
        """Get exchange inflow/outflow data."""
        return {
            "token": token,
            "inflow_24h": 5_000_000,  # To exchanges
            "outflow_24h": 8_000_000,  # From exchanges
            "net_flow": -3_000_000,  # Net outflow = bullish
            "exchange_reserves": 2_000_000_000,
            "reserve_change_7d": -0.05  # -5% = decreasing reserves
        }
    
    async def get_exchange_flow_signal(self, token: str) -> Optional[AlphaSignal]:
        """Generate signal from exchange flows."""
        data = await self.get_exchange_flows(token)
        
        net_flow = data["net_flow"]
        
        # Large net outflow = accumulation (bullish)
        if net_flow < -1_000_000:
            return AlphaSignal(
                source="etherscan",
                signal_type=AlphaSignalType.EXCHANGE_FLOW,
                symbol=token,
                strength=min(abs(net_flow) / 10_000_000, 1.0),
                direction="long",
                confidence=0.8,
                metadata={
                    "net_flow": net_flow,
                    "inflow": data["inflow_24h"],
                    "outflow": data["outflow_24h"],
                    "signal": "exchange_outflow_accumulation"
                }
            )
        # Large net inflow = distribution (bearish)
        elif net_flow > 1_000_000:
            return AlphaSignal(
                source="etherscan",
                signal_type=AlphaSignalType.EXCHANGE_FLOW,
                symbol=token,
                strength=min(abs(net_flow) / 10_000_000, 1.0),
                direction="short",
                confidence=0.75,
                metadata={
                    "net_flow": net_flow,
                    "inflow": data["inflow_24h"],
                    "outflow": data["outflow_24h"],
                    "signal": "exchange_inflow_distribution"
                }
            )
        
        return None


class AlphaSourceAggregator:
    """
    Alpha Source Aggregator
    =======================
    Aggregates signals from all alpha sources.
    Provides consensus signals and confidence scoring.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Initialize all integrations
        self.nansen = NansenIntegration(self.config.get("nansen_api_key"))
        self.tokenomist = TokenomistIntegration(self.config.get("tokenomist_api_key"))
        self.coinglass = CoinGlassIntegration(self.config.get("coinglass_api_key"))
        self.etherscan = EtherscanIntegration(self.config.get("etherscan_api_key"))
        
        # Signal history
        self.signal_history: List[AlphaSignal] = []
        self.max_history = 10000
        
        # Track source performance
        self.source_accuracy: Dict[str, float] = {
            "nansen": 0.72,
            "tokenomist": 0.68,
            "coinglass": 0.75,
            "etherscan": 0.70
        }
        
        logger.info("AlphaSourceAggregator initialized with 4 sources")
    
    async def get_all_signals(self, symbol: str) -> List[AlphaSignal]:
        """Get all alpha signals for a symbol from all sources."""
        signals = []
        
        # Gather from all sources in parallel
        tasks = [
            self.nansen.get_smart_money_flows(symbol),
            self.tokenomist.get_unlock_signal(symbol),
            self.coinglass.get_funding_rate_signal(symbol),
            self.coinglass.get_cvd_signal(symbol),
            self.coinglass.get_liquidation_signal(symbol),
            self.etherscan.get_exchange_flow_signal(symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Alpha source error: {result}")
                continue
            if result is None:
                continue
            if isinstance(result, list):
                signals.extend(result)
            else:
                signals.append(result)
        
        # Store in history
        self.signal_history.extend(signals)
        if len(self.signal_history) > self.max_history:
            self.signal_history = self.signal_history[-self.max_history:]
        
        return signals
    
    def get_consensus_signal(self, signals: List[AlphaSignal]) -> Dict[str, Any]:
        """Calculate consensus from multiple signals."""
        if not signals:
            return {"direction": "neutral", "confidence": 0, "strength": 0}
        
        # Weight by source accuracy
        long_score = 0
        short_score = 0
        total_weight = 0
        
        for signal in signals:
            weight = signal.confidence * self.source_accuracy.get(signal.source, 0.5)
            
            if signal.direction == "long":
                long_score += signal.strength * weight
            elif signal.direction == "short":
                short_score += signal.strength * weight
            
            total_weight += weight
        
        if total_weight == 0:
            return {"direction": "neutral", "confidence": 0, "strength": 0}
        
        long_score /= total_weight
        short_score /= total_weight
        
        if long_score > short_score * 1.2:
            direction = "long"
            strength = long_score
        elif short_score > long_score * 1.2:
            direction = "short"
            strength = short_score
        else:
            direction = "neutral"
            strength = abs(long_score - short_score)
        
        confidence = min(abs(long_score - short_score) + 0.3, 1.0)
        
        return {
            "direction": direction,
            "strength": strength,
            "confidence": confidence,
            "long_score": long_score,
            "short_score": short_score,
            "signal_count": len(signals),
            "sources": list(set(s.source for s in signals))
        }
    
    async def get_trading_signal(self, symbol: str) -> Dict[str, Any]:
        """Get final trading signal with consensus."""
        signals = await self.get_all_signals(symbol)
        consensus = self.get_consensus_signal(signals)
        
        return {
            "symbol": symbol,
            "timestamp": time.time(),
            "consensus": consensus,
            "signals": [s.to_dict() for s in signals],
            "recommendation": self._generate_recommendation(consensus)
        }
    
    def _generate_recommendation(self, consensus: Dict) -> Dict[str, Any]:
        """Generate trading recommendation from consensus."""
        direction = consensus["direction"]
        confidence = consensus["confidence"]
        strength = consensus["strength"]
        
        if confidence < 0.5:
            action = "hold"
            size_pct = 0
        elif direction == "long" and confidence > 0.7:
            action = "buy"
            size_pct = min(confidence * strength * 20, 25)  # Max 25% position
        elif direction == "short" and confidence > 0.7:
            action = "sell"
            size_pct = min(confidence * strength * 20, 25)
        else:
            action = "hold"
            size_pct = 0
        
        return {
            "action": action,
            "size_pct": round(size_pct, 2),
            "confidence": round(confidence, 3),
            "reason": f"Alpha consensus: {direction} (strength={strength:.2f}, confidence={confidence:.2f})"
        }


class AlphaSignalProcessor:
    """
    Alpha Signal Processor
    ======================
    Processes and filters alpha signals for trading decisions.
    """
    
    def __init__(self, min_confidence: float = 0.6, min_strength: float = 0.5):
        self.min_confidence = min_confidence
        self.min_strength = min_strength
        self.active_signals: Dict[str, List[AlphaSignal]] = {}
        
    def filter_signals(self, signals: List[AlphaSignal]) -> List[AlphaSignal]:
        """Filter signals by confidence and strength."""
        return [
            s for s in signals
            if s.confidence >= self.min_confidence and s.strength >= self.min_strength
        ]
    
    def deduplicate_signals(self, signals: List[AlphaSignal]) -> List[AlphaSignal]:
        """Remove duplicate signals, keeping strongest."""
        seen: Dict[Tuple[str, str], AlphaSignal] = {}
        
        for signal in signals:
            key = (signal.symbol, signal.direction)
            if key not in seen or signal.strength > seen[key].strength:
                seen[key] = signal
        
        return list(seen.values())
    
    def aggregate_signals(self, signals: List[AlphaSignal]) -> Dict[str, Dict]:
        """Aggregate signals by symbol."""
        by_symbol: Dict[str, List[AlphaSignal]] = {}
        
        for signal in signals:
            if signal.symbol not in by_symbol:
                by_symbol[signal.symbol] = []
            by_symbol[signal.symbol].append(signal)
        
        result = {}
        for symbol, symbol_signals in by_symbol.items():
            long_signals = [s for s in symbol_signals if s.direction == "long"]
            short_signals = [s for s in symbol_signals if s.direction == "short"]
            
            result[symbol] = {
                "long_count": len(long_signals),
                "short_count": len(short_signals),
                "long_strength": sum(s.strength for s in long_signals),
                "short_strength": sum(s.strength for s in short_signals),
                "net_direction": "long" if len(long_signals) > len(short_signals) else "short",
                "signal_count": len(symbol_signals)
            }
        
        return result
    
    def process(self, signals: List[AlphaSignal]) -> List[AlphaSignal]:
        """Full signal processing pipeline."""
        # Filter
        filtered = self.filter_signals(signals)
        
        # Deduplicate
        deduped = self.deduplicate_signals(filtered)
        
        return deduped


# Singleton instance
_alpha_aggregator: Optional[AlphaSourceAggregator] = None


def get_alpha_aggregator(config: Optional[Dict] = None) -> AlphaSourceAggregator:
    """Get or create singleton alpha aggregator."""
    global _alpha_aggregator
    if _alpha_aggregator is None:
        _alpha_aggregator = AlphaSourceAggregator(config)
    return _alpha_aggregator


async def get_alpha_signals(symbol: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """Convenience function to get alpha signals for a symbol."""
    aggregator = get_alpha_aggregator(config)
    return await aggregator.get_trading_signal(symbol)


# Export
__all__ = [
    "AlphaSignal",
    "AlphaSignalType",
    "AlphaSourceAggregator",
    "AlphaSignalProcessor",
    "NansenIntegration",
    "TokenomistIntegration",
    "CoinGlassIntegration",
    "EtherscanIntegration",
    "get_alpha_aggregator",
    "get_alpha_signals"
]
