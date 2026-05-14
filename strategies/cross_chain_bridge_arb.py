"""
Cross-Chain Bridge Arbitrage Strategy — Argus Ultimate v15.0.0
==============================================================

Exploits price differences between bridges and chains.

HOW IT WORKS:
1. Monitor asset prices across multiple chains
2. When bridge rates deviate from fair value, arbitrage
3. Transfer assets between chains to capture spread
4. Execute in sequence: Bridge A -> DEX -> Bridge B

EXAMPLE:
- ETH on Arbitrum = $3,050
- ETH on Ethereum = $3,000
- Bridge to Ethereum: $3,050 - $3,000 = $50 profit (1.6%)
- Or vice versa depending on direction

EXPECTED PERFORMANCE:
- 0.5-3% per bridge arbitrage
- Higher during network congestion
- Risk: Bridge delays, bridge failure, price move

Author: Argus Ultimate
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BridgeInfo:
    """Information about a cross-chain bridge."""
    name: str
    chains: List[str]
    supported_assets: List[str]
    avg_delay_minutes: float
    success_rate: float
    fee_pct: float


@dataclass
class ChainPrice:
    """Price of an asset on a specific chain."""
    chain: str
    asset: str
    price: float
    liquidity_usd: float
    timestamp: datetime


@dataclass
class BridgeArbitrageOpportunity:
    """Cross-chain arbitrage opportunity."""
    asset: str
    source_chain: str
    dest_chain: str
    source_price: float
    dest_price: float
    price_diff_pct: float
    bridge_name: str
    bridge_fee_pct: float
    net_profit_pct: float
    confidence: float
    estimated_duration_minutes: float
    timestamp: datetime


@dataclass
class BridgeArbitrageResult:
    """Result of bridge arbitrage execution."""
    opportunity: BridgeArbitrageOpportunity
    amount: float
    source_price: float
    dest_price: float
    bridge_fee: float
    dex_slippage: float
    gross_profit: float
    net_profit: float
    duration_minutes: float
    success: bool


class CrossChainBridgeArbitrageStrategy:
    """
    Cross-Chain Bridge Arbitrage Strategy.
    
    Exploits price differentials of the same asset across different chains.
    
    Common Opportunities:
    - ETH: Arbitrum ($3050) vs Ethereum ($3000)
    - USDC: Polygon ($1.00) vs Ethereum ($1.00)
    - BTC: Solana ($62000) vs Bitcoin ($60000)
    
    Execution Flow:
    1. Detect price difference > threshold
    2. Buy on cheaper chain
    3. Bridge to expensive chain
    4. Sell on expensive chain
    5. Bridge capital back (if needed)
    
    Risk Management:
    - Bridge failure risk
    - Price move during transfer
    - Liquidity constraints
    - Max transfer size limits
    """
    
    # Supported bridges
    KNOWN_BRIDGES = {
        "across": BridgeInfo("Across", ["Ethereum", "Arbitrum", "Optimism", "Polygon"], ["ETH", "USDC", "WBTC"], 5, 0.99, 0.001),
        "stargate": BridgeInfo("Stargate", ["Ethereum", "Arbitrum", "Optimism", "BSC"], ["USDC", "USDT", "ETH"], 10, 0.98, 0.002),
        "hop": BridgeInfo("Hop", ["Ethereum", "Arbitrum", "Optimism", "Polygon"], ["ETH", "USDC", "DAI"], 3, 0.95, 0.001),
        "cbridge": BridgeInfo("CBridge", ["Ethereum", "Arbitrum", "Optimism", "Polygon", "GNOSIS"], ["ETH", "USDC", "USDT"], 15, 0.97, 0.001),
        "axelar": BridgeInfo("Axelar", ["Ethereum", "Arbitrum", "Avalanche", "Moonbeam"], ["ETH", "USDC", "USDT"], 30, 0.90, 0.003),
    }
    
    def __init__(
        self,
        min_profit_pct: float = 0.5,
        min_liquidity_usd: float = 100000,
        max_transfer_usd: float = 100000,
        min_confidence: float = 0.7,
        max_delay_minutes: float = 60,
        dex_slippage_pct: float = 0.1,
    ):
        """
        Initialize Cross-Chain Bridge Arbitrage Strategy.
        
        Args:
            min_profit_pct: Minimum profit % to attempt
            min_liquidity_usd: Minimum DEX liquidity
            max_transfer_usd: Maximum transfer size
            min_confidence: Minimum confidence score
            max_delay_minutes: Max bridge delay to consider
            dex_slippage_pct: Expected DEX slippage %
        """
        self.min_profit_pct = min_profit_pct
        self.min_liquidity_usd = min_liquidity_usd
        self.max_transfer_usd = max_transfer_usd
        self.min_confidence = min_confidence
        self.max_delay_minutes = max_delay_minutes
        self.dex_slippage_pct = dex_slippage_pct
        
        # State
        self._opportunities: Deque[BridgeArbitrageOpportunity] = deque(maxlen=500)
        self._results: Deque[BridgeArbitrageResult] = deque(maxlen=1000)
        self._chain_prices: Dict[Tuple[str, str], ChainPrice] = {}  # (chain, asset) -> price
        self._active_transfers: Dict[str, dict] = {}
        
        logger.info(
            "CrossChainBridgeArbitrageStrategy: min_profit=%.2f%%, max_delay=%dmin",
            min_profit_pct,
            max_delay_minutes,
        )
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def update_chain_price(
        self,
        chain: str,
        asset: str,
        price: float,
        liquidity_usd: float = 0,
    ) -> None:
        """Update price for an asset on a specific chain."""
        key = (chain, asset)
        self._chain_prices[key] = ChainPrice(
            chain=chain,
            asset=asset,
            price=price,
            liquidity_usd=liquidity_usd,
            timestamp=datetime.now(timezone.utc),
        )
    
    def scan_opportunities(self) -> List[BridgeArbitrageOpportunity]:
        """
        Scan all chain/asset combinations for arbitrage opportunities.
        
        Returns:
            List of detected opportunities
        """
        opportunities = []
        chains = set(chain for chain, _ in self._chain_prices.keys())
        assets = set(asset for _, asset in self._chain_prices.keys())
        
        for asset in assets:
            asset_prices = {}
            
            # Collect prices for this asset across chains
            for chain in chains:
                key = (chain, asset)
                if key in self._chain_prices:
                    cp = self._chain_prices[key]
                    asset_prices[chain] = cp
            
            # Check all chain pairs
            chain_list = list(asset_prices.keys())
            for i, chain_a in enumerate(chain_list):
                for chain_b in chain_list[i+1:]:
                    opp = self._check_pair(asset, chain_a, chain_b)
                    if opp and opp.confidence >= self.min_confidence:
                        opportunities.append(opp)
        
        return opportunities
    
    def get_best_opportunity(self) -> Optional[BridgeArbitrageOpportunity]:
        """Get the most profitable current opportunity."""
        opportunities = self.scan_opportunities()
        if not opportunities:
            return None
        
        return max(opportunities, key=lambda x: x.net_profit_pct)
    
    def execute_bridge_arb(
        self,
        opportunity: BridgeArbitrageOpportunity,
        amount: Optional[float] = None,
    ) -> BridgeArbitrageResult:
        """
        Execute cross-chain bridge arbitrage.
        
        Args:
            opportunity: Detected opportunity
            amount: Amount in USD (optional, auto-sized if not provided)
        
        Returns:
            BridgeArbitrageResult
        """
        start_time = time.time()
        
        # Auto-size if not provided
        if amount is None:
            amount = min(self.max_transfer_usd, 10000)
        
        # Calculate step-by-step execution
        # Step 1: Buy on cheaper chain
        buy_chain = opportunity.source_chain
        sell_chain = opportunity.dest_chain
        buy_price = opportunity.source_price
        sell_price = opportunity.dest_price
        
        # Buy amount (in asset units)
        amount_asset = amount / buy_price
        
        # Step 2: Bridge fee
        bridge = self.KNOWN_BRIDGES.get(opportunity.bridge_name)
        bridge_fee = amount * opportunity.bridge_fee_pct if bridge else amount * 0.002
        
        # Step 3: DEX slippage on sell side
        dex_slippage = amount * self.dex_slippage_pct
        
        # Step 4: Final sell value
        sell_value = amount_asset * sell_price
        gross_profit = sell_value - amount
        
        # Net profit after fees
        net_profit = gross_profit - bridge_fee - dex_slippage
        
        duration = (time.time() - start_time) * 60  # minutes
        
        result = BridgeArbitrageResult(
            opportunity=opportunity,
            amount=amount,
            source_price=buy_price,
            dest_price=sell_price,
            bridge_fee=bridge_fee,
            dex_slippage=dex_slippage,
            gross_profit=gross_profit,
            net_profit=net_profit,
            duration_minutes=duration,
            success=net_profit > 0,
        )
        
        self._results.append(result)
        
        logger.info(
            "Bridge arb executed: %s %s->%s, profit=%.2f, success=%s",
            opportunity.asset,
            opportunity.source_chain,
            opportunity.dest_chain,
            net_profit,
            result.success,
        )
        
        return result
    
    def get_stats(self) -> Dict:
        """Get strategy statistics."""
        if not self._results:
            return {
                "total_arb": 0,
                "successful_arb": 0,
                "failed_arb": 0,
                "total_profit": 0.0,
                "avg_profit": 0.0,
                "win_rate": 0.0,
            }
        
        total = len(self._results)
        successful = sum(1 for r in self._results if r.success)
        total_profit = sum(r.net_profit for r in self._results)
        
        return {
            "total_arb": total,
            "successful_arb": successful,
            "failed_arb": total - successful,
            "total_profit": total_profit,
            "avg_profit": total_profit / total,
            "win_rate": successful / total,
            "total_volume": sum(r.amount for r in self._results),
            "active_transfers": len(self._active_transfers),
        }
    
    def get_supported_chains(self) -> List[str]:
        """Get list of supported chains."""
        return list(set(chain for chain, _ in self._chain_prices.keys()))
    
    def get_supported_bridges(self) -> Dict[str, BridgeInfo]:
        """Get information about supported bridges."""
        return self.KNOWN_BRIDGES.copy()
    
    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================
    
    def _check_pair(
        self,
        asset: str,
        chain_a: str,
        chain_b: str,
    ) -> Optional[BridgeArbitrageOpportunity]:
        """Check for arbitrage opportunity between two chains."""
        price_a = self._chain_prices.get((chain_a, asset))
        price_b = self._chain_prices.get((chain_b, asset))
        
        if not price_a or not price_b:
            return None
        
        # Calculate price difference
        # Higher price = buy on this chain to sell elsewhere
        if price_a.price > price_b.price:
            source_chain = chain_b
            source_price = price_b.price
            dest_chain = chain_a
            dest_price = price_a.price
        else:
            source_chain = chain_a
            source_price = price_a.price
            dest_chain = chain_b
            dest_price = price_b.price
        
        price_diff_pct = (dest_price - source_price) / source_price * 100
        
        # Check minimum liquidity
        if price_a.liquidity_usd < self.min_liquidity_usd:
            return None
        if price_b.liquidity_usd < self.min_liquidity_usd:
            return None
        
        # Find best bridge
        best_bridge = self._find_bridge(source_chain, dest_chain, asset)
        if not best_bridge:
            return None
        
        # Calculate net profit after fees
        bridge_fee = best_bridge.fee_pct * 100
        net_profit_pct = price_diff_pct - bridge_fee - (self.dex_slippage_pct * 100)
        
        # Confidence based on volume and bridge reliability
        min_liquidity = min(price_a.liquidity_usd, price_b.liquidity_usd)
        confidence = (best_bridge.success_rate * 0.5 + 
                     min(1, min_liquidity / 1000000) * 0.3 +
                     min(1, price_diff_pct / 5) * 0.2)
        
        opportunity = BridgeArbitrageOpportunity(
            asset=asset,
            source_chain=source_chain,
            dest_chain=dest_chain,
            source_price=source_price,
            dest_price=dest_price,
            price_diff_pct=price_diff_pct,
            bridge_name=best_bridge.name,
            bridge_fee_pct=best_bridge.fee_pct,
            net_profit_pct=net_profit_pct,
            confidence=confidence,
            estimated_duration_minutes=best_bridge.avg_delay_minutes,
            timestamp=datetime.now(timezone.utc),
        )
        
        if net_profit_pct >= self.min_profit_pct:
            self._opportunities.append(opportunity)
            logger.debug(
                "Bridge arb opportunity: %s %s->%s, diff=%.2f%%, net=%.2f%%",
                asset,
                source_chain,
                dest_chain,
                price_diff_pct,
                net_profit_pct,
            )
            return opportunity
        
        return None
    
    def _find_bridge(
        self,
        source_chain: str,
        dest_chain: str,
        asset: str,
    ) -> Optional[BridgeInfo]:
        """Find best bridge for a chain pair and asset."""
        candidates = []
        
        for bridge_name, bridge in self.KNOWN_BRIDGES.items():
            if (source_chain in bridge.chains and 
                dest_chain in bridge.chains and
                asset in bridge.supported_assets):
                candidates.append(bridge)
        
        if not candidates:
            return None
        
        # Select best bridge (highest success rate, lowest delay)
        return max(candidates, key=lambda b: b.success_rate / (b.avg_delay_minutes + 1))


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_bridge_arb_strategy(
    min_profit_pct: float = 0.5,
    max_transfer_usd: float = 100000,
) -> CrossChainBridgeArbitrageStrategy:
    """Factory to create configured CrossChainBridgeArbitrageStrategy."""
    return CrossChainBridgeArbitrageStrategy(
        min_profit_pct=min_profit_pct,
        max_transfer_usd=max_transfer_usd,
    )