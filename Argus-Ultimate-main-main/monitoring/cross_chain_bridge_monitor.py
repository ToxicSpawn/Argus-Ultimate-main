"""
Cross-Chain Bridge Monitor
===========================
Monitors whale movements across blockchains:
- Ethereum, BSC, Polygon, Arbitrum, Optimism, Avalanche
- Bridge transfers (LayerZero, Wormhole, Axelar, Stargate)
- Whale wallet tracking
- Large transfer alerts
- Flow analysis for alpha signals
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class Chain(Enum):
    """Supported blockchain networks."""
    ETHEREUM = "ethereum"
    BSC = "bsc"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    AVALANCHE = "avalanche"
    SOLANA = "solana"
    BASE = "base"


class BridgeType(Enum):
    """Bridge types."""
    LAYERZERO = "layerzero"
    WORMHOLE = "wormhole"
    AXELAR = "axelar"
    STARGATE = "stargate"
    MULTICHAIN = "multichain"
    CELER = "celer"
    SYNAPSE = "synapse"
    HOP = "hop"
    ORBITER = "orbiter"


@dataclass
class BridgeTransfer:
    """Cross-chain bridge transfer."""
    tx_hash: str
    source_chain: Chain
    destination_chain: Chain
    bridge: BridgeType
    token: str
    amount: float
    amount_usd: float
    sender: str
    recipient: str
    timestamp: float = field(default_factory=time.time)
    status: str = "pending"  # pending, completed, failed
    confirmations: int = 0


@dataclass
class WhaleWallet:
    """Tracked whale wallet."""
    address: str
    chain: Chain
    label: str  # "Exchange", "VC", "Foundation", etc.
    total_value_usd: float = 0.0
    last_activity: float = 0.0
    transfer_count: int = 0
    tags: List[str] = field(default_factory=list)


@dataclass
class WhaleAlert:
    """Whale movement alert."""
    alert_type: str  # "large_transfer", "exchange_deposit", "exchange_withdrawal"
    chain: Chain
    token: str
    amount: float
    amount_usd: float
    from_address: str
    to_address: str
    from_label: str
    to_label: str
    timestamp: float = field(default_factory=time.time)
    significance: str = "medium"  # low, medium, high, critical
    potential_impact: str = ""  # bullish, bearish, neutral


class BridgeMonitor:
    """
    Cross-Chain Bridge Monitor
    ===========================
    Monitors bridge transfers across chains.
    """
    
    def __init__(self):
        self.transfers: List[BridgeTransfer] = []
        self.bridge_stats: Dict[str, Dict[str, Any]] = {}
        
        # Known bridge contracts
        self.bridge_contracts: Dict[Chain, Dict[BridgeType, List[str]]] = {
            Chain.ETHEREUM: {
                BridgeType.LAYERZERO: ["0x66A71D67205499505F6b1E6fC37C76967C2f0B"],
                BridgeType.WORMHOLE: ["0x98f3c9e6E3fAce36baAAd55d718B37D8f9B8d23"],
                BridgeType.STARGATE: ["0x8731d54E9D02c286767d36a0b4C578605cc57655"],
                BridgeType.AXELAR: ["0x4F449524B89CBBD225A823701199834B4611B098"],
            },
            Chain.BSC: {
                BridgeType.LAYERZERO: ["0x66A71D67205499505F6b1E6fC37C76967C2f0B"],
                BridgeType.WORMHOLE: ["0xB6F6D86a8f9879A9c87f6437fc8d7B95B5329d55"],
            }
        }
        
        # Known whale wallets
        self.known_wallets: Dict[str, WhaleWallet] = {}
        self._init_known_wallets()
    
    def _init_known_wallets(self) -> None:
        """Initialize known whale wallets."""
        whales = [
            WhaleWallet("0x28C6c06298d514Db089934071355E5743bf21d60", Chain.ETHEREUM, "Binance", tags=["exchange"]),
            WhaleWallet("0x21a31Ee1afC51d94C2eFcCAa2092aD1028285549", Chain.ETHEREUM, "Binance", tags=["exchange"]),
            WhaleWallet("0xDFd5293D8e347dFe59E90eFd55b2956a1343963d", Chain.ETHEREUM, "Binance", tags=["exchange"]),
            WhaleWallet("0x56Eddb7aa87536c09CCc2793473599fD21A8b17F", Chain.ETHEREUM, "Gemini", tags=["exchange"]),
            WhaleWallet("0x0716a17FBAeE714f1E6aB0f9d59edbC5f09815C0", Chain.ETHEREUM, "Kraken", tags=["exchange"]),
            WhaleWallet("0x40B38765696e3d5d8d9d834D8AaD4bB6e418E489", Chain.ETHEREUM, "Robinhood", tags=["exchange"]),
        ]
        
        for wallet in whales:
            self.known_wallets[wallet.address.lower()] = wallet
    
    async def monitor_bridge(
        self,
        bridge: BridgeType,
        source_chain: Chain,
        dest_chain: Chain
    ) -> List[BridgeTransfer]:
        """Monitor a specific bridge for transfers."""
        logger.info(f"Monitoring {bridge.value}: {source_chain.value} -> {dest_chain.value}")
        
        # In production: query bridge contracts/subgraph
        # For now, return simulated data
        
        transfers = []
        return transfers
    
    def classify_transfer(self, transfer: BridgeTransfer) -> Dict[str, Any]:
        """Classify a bridge transfer's significance."""
        # Get wallet labels
        from_wallet = self.known_wallets.get(transfer.sender.lower())
        to_wallet = self.known_wallets.get(transfer.recipient.lower())
        
        from_label = from_wallet.label if from_wallet else "Unknown"
        to_label = to_wallet.label if to_wallet else "Unknown"
        
        # Determine significance
        if transfer.amount_usd > 10_000_000:
            significance = "critical"
        elif transfer.amount_usd > 1_000_000:
            significance = "high"
        elif transfer.amount_usd > 100_000:
            significance = "medium"
        else:
            significance = "low"
        
        # Determine potential impact
        impact = "neutral"
        
        # Exchange deposits often precede selling
        if to_wallet and "exchange" in to_wallet.tags:
            impact = "bearish"
        # Exchange withdrawals often indicate accumulation
        elif from_wallet and "exchange" in from_wallet.tags:
            impact = "bullish"
        # Large transfers to DeFi protocols
        elif transfer.destination_chain in [Chain.ARBITRUM, Chain.OPTIMISM]:
            impact = "bullish"  # Likely for DeFi activity
        
        return {
            "from_label": from_label,
            "to_label": to_label,
            "significance": significance,
            "potential_impact": impact,
            "is_whale": from_wallet is not None or to_wallet is not None
        }
    
    def create_alert(self, transfer: BridgeTransfer) -> WhaleAlert:
        """Create alert for significant transfer."""
        classification = self.classify_transfer(transfer)
        
        alert_type = "large_transfer"
        if classification["to_label"] in ["Binance", "Coinbase", "Kraken"]:
            alert_type = "exchange_deposit"
        elif classification["from_label"] in ["Binance", "Coinbase", "Kraken"]:
            alert_type = "exchange_withdrawal"
        
        return WhaleAlert(
            alert_type=alert_type,
            chain=transfer.source_chain,
            token=transfer.token,
            amount=transfer.amount,
            amount_usd=transfer.amount_usd,
            from_address=transfer.sender,
            to_address=transfer.recipient,
            from_label=classification["from_label"],
            to_label=classification["to_label"],
            significance=classification["significance"],
            potential_impact=classification["potential_impact"]
        )
    
    def get_flow_analysis(self, token: str, hours: int = 24) -> Dict[str, Any]:
        """Analyze cross-chain flows for a token."""
        cutoff = time.time() - hours * 3600
        
        recent_transfers = [
            t for t in self.transfers
            if t.token == token and t.timestamp > cutoff
        ]
        
        if not recent_transfers:
            return {"token": token, "transfers": 0, "flow": "neutral"}
        
        # Calculate net flow by chain
        chain_flows: Dict[str, float] = {}
        for transfer in recent_transfers:
            source = transfer.source_chain.value
            dest = transfer.destination_chain.value
            
            chain_flows[source] = chain_flows.get(source, 0) - transfer.amount_usd
            chain_flows[dest] = chain_flows.get(dest, 0) + transfer.amount_usd
        
        total_volume = sum(t.amount_usd for t in recent_transfers)
        
        # Determine overall flow direction
        eth_flow = chain_flows.get("ethereum", 0)
        if eth_flow > 1_000_000:
            flow_direction = "into_ethereum"
            signal = "bullish"
        elif eth_flow < -1_000_000:
            flow_direction = "out_of_ethereum"
            signal = "bearish"
        else:
            flow_direction = "balanced"
            signal = "neutral"
        
        return {
            "token": token,
            "period_hours": hours,
            "total_transfers": len(recent_transfers),
            "total_volume_usd": total_volume,
            "chain_flows": chain_flows,
            "net_ethereum_flow": eth_flow,
            "flow_direction": flow_direction,
            "signal": signal
        }


class WhaleTracker:
    """
    Whale Tracker
    =============
    Tracks and analyzes whale wallet activity.
    """
    
    def __init__(self):
        self.wallets: Dict[str, WhaleWallet] = {}
        self.activity_log: List[Dict[str, Any]] = []
        self.alerts: List[WhaleAlert] = []
        
        # Minimum thresholds
        self.min_alert_usd = 100_000
        self.min_whale_usd = 1_000_000
    
    def add_wallet(self, wallet: WhaleWallet) -> None:
        """Add wallet to tracking."""
        key = f"{wallet.chain.value}:{wallet.address.lower()}"
        self.wallets[key] = wallet
        logger.info(f"Tracking whale: {wallet.label} on {wallet.chain.value}")
    
    def update_activity(
        self,
        address: str,
        chain: Chain,
        activity_type: str,
        amount_usd: float,
        details: Dict[str, Any]
    ) -> None:
        """Update wallet activity."""
        key = f"{chain.value}:{address.lower()}"
        
        if key in self.wallets:
            wallet = self.wallets[key]
            wallet.last_activity = time.time()
            wallet.transfer_count += 1
            wallet.total_value_usd += amount_usd
        
        self.activity_log.append({
            "address": address,
            "chain": chain.value,
            "type": activity_type,
            "amount_usd": amount_usd,
            "details": details,
            "timestamp": time.time()
        })
    
    def get_active_whales(self, hours: int = 24) -> List[WhaleWallet]:
        """Get whales active in recent period."""
        cutoff = time.time() - hours * 3600
        return [
            w for w in self.wallets.values()
            if w.last_activity > cutoff
        ]
    
    def get_whale_summary(self) -> Dict[str, Any]:
        """Get summary of tracked whales."""
        total_wallets = len(self.wallets)
        active_24h = len(self.get_active_whales(24))
        active_7d = len(self.get_active_whales(168))
        
        by_chain: Dict[str, int] = {}
        for wallet in self.wallets.values():
            chain = wallet.chain.value
            by_chain[chain] = by_chain.get(chain, 0) + 1
        
        by_label: Dict[str, int] = {}
        for wallet in self.wallets.values():
            by_label[wallet.label] = by_label.get(wallet.label, 0) + 1
        
        return {
            "total_wallets": total_wallets,
            "active_24h": active_24h,
            "active_7d": active_7d,
            "by_chain": by_chain,
            "by_label": by_label,
            "total_alerts": len(self.alerts)
        }


class CrossChainAnalyzer:
    """
    Cross-Chain Analyzer
    ====================
    Analyzes cross-chain patterns for trading signals.
    """
    
    def __init__(self):
        self.bridge_monitor = BridgeMonitor()
        self.whale_tracker = WhaleTracker()
        
        self.signal_history: List[Dict[str, Any]] = []
    
    async def analyze_token_flows(self, token: str) -> Dict[str, Any]:
        """Analyze cross-chain flows for trading signal."""
        flow_analysis = self.bridge_monitor.get_flow_analysis(token, hours=24)
        
        # Generate signal based on flows
        signal = {
            "token": token,
            "timestamp": time.time(),
            "flow_analysis": flow_analysis
        }
        
        # Bullish signals
        if flow_analysis.get("signal") == "bullish":
            signal["direction"] = "long"
            signal["confidence"] = 0.6
            signal["reason"] = "Net inflow to Ethereum (accumulation)"
        
        # Bearish signals
        elif flow_analysis.get("signal") == "bearish":
            signal["direction"] = "short"
            signal["confidence"] = 0.6
            signal["reason"] = "Net outflow from Ethereum (distribution)"
        
        else:
            signal["direction"] = "neutral"
            signal["confidence"] = 0.3
            signal["reason"] = "Balanced cross-chain flows"
        
        self.signal_history.append(signal)
        return signal
    
    def get_market_signals(self) -> List[Dict[str, Any]]:
        """Get market-wide signals from cross-chain analysis."""
        signals = []
        
        # Analyze major tokens
        major_tokens = ["ETH", "BTC", "USDC", "USDT", "ARB", "OP"]
        
        for token in major_tokens:
            flow = self.bridge_monitor.get_flow_analysis(token, hours=24)
            if flow.get("total_transfers", 0) > 0:
                signals.append({
                    "token": token,
                    "signal": flow.get("signal", "neutral"),
                    "volume_usd": flow.get("total_volume_usd", 0),
                    "flow_direction": flow.get("flow_direction", "balanced")
                })
        
        return signals


# Export
__all__ = [
    "Chain",
    "BridgeType",
    "BridgeTransfer",
    "WhaleWallet",
    "WhaleAlert",
    "BridgeMonitor",
    "WhaleTracker",
    "CrossChainAnalyzer"
]
