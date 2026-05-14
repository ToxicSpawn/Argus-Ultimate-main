"""
NFT Analytics Module
=====================
Analyzes NFT markets for trading opportunities:
- Floor price tracking
- Volume analysis
- Whale activity
- Trait rarity scoring
- Collection momentum
- Sniping opportunities
- Flipping calculations

Supports: OpenSea, Blur, LooksRare, X2Y2
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class NFTMarketplace(Enum):
    """NFT marketplaces."""
    OPENSEA = "opensea"
    BLUR = "blur"
    LOOKSRARE = "looksRare"
    X2Y2 = "x2y2"
    SUDOOSWAP = "sudoswap"


class NFTCategory(Enum):
    """NFT categories."""
    PFP = "pfp"  # Profile pictures
    ART = "art"
    GAMING = "gaming"
    METAVERSE = "metaverse"
    MUSIC = "music"
    PHOTOGRAPHY = "photography"
    SPORTS = "sports"
    UTILITY = "utility"


@dataclass
class NFTCollection:
    """NFT collection data."""
    contract_address: str
    name: str
    category: NFTCategory
    total_supply: int
    floor_price_eth: float
    floor_price_24h_change: float
    volume_24h_eth: float
    volume_7d_eth: float
    owners: int
    listed_count: int
    listed_pct: float  # Percentage listed
    avg_price_eth: float
    market_cap_eth: float
    royalty_pct: float
    last_updated: float = field(default_factory=time.time)


@dataclass
class NFTTrait:
    """NFT trait."""
    trait_type: str
    value: str
    rarity_pct: float  # Percentage of NFTs with this trait
    floor_eth: float = 0.0  # Floor price for this trait


@dataclass
class NFTItem:
    """Individual NFT."""
    contract_address: str
    token_id: int
    owner: str
    traits: List[NFTTrait]
    last_sale_price: float = 0.0
    last_sale_time: float = 0.0
    current_listing_price: float = 0.0
    is_listed: bool = False
    rarity_rank: int = 0


@dataclass
class NFTSnipe:
    """NFT sniping opportunity."""
    collection: str
    token_id: int
    listing_price_eth: float
    floor_price_eth: float
    discount_pct: float
    estimated_profit_eth: float
    traits: List[NFTTrait]
    rarity_rank: int
    marketplace: NFTMarketplace
    listing_url: str = ""


class NFTCollectionAnalyzer:
    """
    NFT Collection Analyzer
    =======================
    Analyzes NFT collections for opportunities.
    """
    
    def __init__(self):
        self.collections: Dict[str, NFTCollection] = {}
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[float]] = {}
    
    def update_collection(self, collection: NFTCollection) -> None:
        """Update collection data."""
        self.collections[collection.contract_address] = collection
        
        # Update history
        if collection.contract_address not in self.price_history:
            self.price_history[collection.contract_address] = []
        if collection.contract_address not in self.volume_history:
            self.volume_history[collection.contract_address] = []
        
        self.price_history[collection.contract_address].append(collection.floor_price_eth)
        self.volume_history[collection.contract_address].append(collection.volume_24h_eth)
        
        # Keep only recent history
        if len(self.price_history[collection.contract_address]) > 1000:
            self.price_history[collection.contract_address] = self.price_history[collection.contract_address][-1000:]
    
    def calculate_momentum(self, contract_address: str) -> Dict[str, float]:
        """Calculate collection momentum."""
        if contract_address not in self.price_history:
            return {"momentum": 0, "trend": "neutral"}
        
        prices = self.price_history[contract_address]
        volumes = self.volume_history.get(contract_address, [])
        
        if len(prices) < 7:
            return {"momentum": 0, "trend": "neutral"}
        
        # Price momentum
        price_7d_ago = prices[-7] if len(prices) >= 7 else prices[0]
        price_now = prices[-1]
        price_momentum = (price_now - price_7d_ago) / price_7d_ago if price_7d_ago > 0 else 0
        
        # Volume momentum
        volume_momentum = 0
        if len(volumes) >= 7:
            vol_7d_ago = np.mean(volumes[-7:-1]) if len(volumes) > 1 else volumes[0]
            vol_now = volumes[-1]
            volume_momentum = (vol_now - vol_7d_ago) / vol_7d_ago if vol_7d_ago > 0 else 0
        
        # Combined momentum
        combined_momentum = price_momentum * 0.6 + volume_momentum * 0.4
        
        # Determine trend
        if combined_momentum > 0.1:
            trend = "bullish"
        elif combined_momentum < -0.1:
            trend = "bearish"
        else:
            trend = "neutral"
        
        return {
            "price_momentum": price_momentum,
            "volume_momentum": volume_momentum,
            "combined_momentum": combined_momentum,
            "trend": trend
        }
    
    def calculate_fair_value(self, contract_address: str) -> Optional[float]:
        """Calculate fair value estimate for collection."""
        if contract_address not in self.collections:
            return None
        
        collection = self.collections[contract_address]
        
        # Multiple valuation methods
        valuations = []
        
        # 1. Volume-weighted average
        if collection.volume_24h_eth > 0 and collection.listed_count > 0:
            valuations.append(collection.volume_24h_eth / collection.listed_count)
        
        # 2. Market cap based
        if collection.owners > 0:
            valuations.append(collection.market_cap_eth / collection.owners)
        
        # 3. Historical floor multiple
        if len(self.price_history.get(contract_address, [])) > 30:
            avg_floor = np.mean(self.price_history[contract_address][-30:])
            valuations.append(avg_floor)
        
        return np.mean(valuations) if valuations else collection.floor_price_eth
    
    def find_undervalued_collections(
        self,
        min_volume_eth: float = 10.0
    ) -> List[Dict[str, Any]]:
        """Find potentially undervalued collections."""
        undervalued = []
        
        for address, collection in self.collections.items():
            if collection.volume_24h_eth < min_volume_eth:
                continue
            
            fair_value = self.calculate_fair_value(address)
            if fair_value is None:
                continue
            
            discount = (fair_value - collection.floor_price_eth) / fair_value * 100
            
            if discount > 10:  # At least 10% discount
                momentum = self.calculate_momentum(address)
                
                undervalued.append({
                    "collection": collection.name,
                    "contract": address,
                    "floor_price": collection.floor_price_eth,
                    "fair_value": fair_value,
                    "discount_pct": discount,
                    "volume_24h": collection.volume_24h_eth,
                    "momentum": momentum["trend"],
                    "listed_pct": collection.listed_pct
                })
        
        return sorted(undervalued, key=lambda x: x["discount_pct"], reverse=True)


class NFTSniper:
    """
    NFT Sniper
    ==========
    Finds and executes NFT sniping opportunities.
    """
    
    def __init__(self, max_price_eth: float = 10.0):
        self.max_price_eth = max_price_eth
        self.snipe_history: List[Dict[str, Any]] = []
        self.min_discount_pct: float = 15.0  # Minimum 15% below floor
    
    def evaluate_snipe(
        self,
        nft: NFTItem,
        collection: NFTCollection,
        marketplace: NFTMarketplace
    ) -> Optional[NFTSnipe]:
        """Evaluate if an NFT is worth sniping."""
        listing_price = nft.current_listing_price
        
        if listing_price > self.max_price_eth:
            return None
        
        # Calculate discount from floor
        discount = (collection.floor_price_eth - listing_price) / collection.floor_price_eth * 100
        
        if discount < self.min_discount_pct:
            return None
        
        # Calculate trait premium
        trait_premium = self._calculate_trait_premium(nft.traits, collection)
        
        # Adjusted discount considering traits
        adjusted_value = collection.floor_price_eth * (1 + trait_premium)
        adjusted_discount = (adjusted_value - listing_price) / adjusted_value * 100
        
        # Estimate profit (accounting for fees and royalties)
        marketplace_fee = 0.025  # 2.5%
        royalty = collection.royalty_pct / 100
        
        sale_price = adjusted_value
        net_sale = sale_price * (1 - marketplace_fee - royalty)
        estimated_profit = net_sale - listing_price
        
        return NFTSnipe(
            collection=collection.name,
            token_id=nft.token_id,
            listing_price_eth=listing_price,
            floor_price_eth=collection.floor_price_eth,
            discount_pct=adjusted_discount,
            estimated_profit_eth=estimated_profit,
            traits=nft.traits,
            rarity_rank=nft.rarity_rank,
            marketplace=marketplace
        )
    
    def _calculate_trait_premium(self, traits: List[NFTTrait], collection: NFTCollection) -> float:
        """Calculate price premium based on rare traits."""
        if not traits:
            return 0.0
        
        premium = 0.0
        for trait in traits:
            # Rare traits (< 5% rarity) add value
            if trait.rarity_pct < 5:
                premium += 0.1  # 10% per rare trait
            elif trait.rarity_pct < 10:
                premium += 0.05  # 5% per uncommon trait
        
        return min(premium, 0.5)  # Cap at 50% premium
    
    def find_snipes(
        self,
        collection: NFTCollection,
        listings: List[NFTItem]
    ) -> List[NFTSnipe]:
        """Find sniping opportunities in a collection."""
        snipes = []
        
        for nft in listings:
            if not nft.is_listed:
                continue
            
            snipe = self.evaluate_snipe(nft, collection, NFTMarketplace.BLUR)
            if snipe:
                snipes.append(snipe)
        
        return sorted(snipes, key=lambda s: s.estimated_profit_eth, reverse=True)


class NFTFlipCalculator:
    """
    NFT Flip Calculator
    ===================
    Calculates profitability of NFT flips.
    """
    
    def __init__(self):
        self.marketplace_fees = {
            NFTMarketplace.OPENSEA: 0.025,
            NFTMarketplace.BLUR: 0.0,
            NFTMarketplace.LOOKSRARE: 0.02,
            NFTMarketplace.X2Y2: 0.005
        }
    
    def calculate_flip_profit(
        self,
        buy_price_eth: float,
        sell_price_eth: float,
        royalty_pct: float,
        marketplace: NFTMarketplace,
        gas_cost_eth: float = 0.01
    ) -> Dict[str, Any]:
        """Calculate flip profit."""
        # Buy costs
        marketplace_fee_buy = 0  # Usually no fee on buy
        
        # Sell costs
        marketplace_fee_sell = sell_price_eth * self.marketplace_fees.get(marketplace, 0.025)
        royalty_fee = sell_price_eth * (royalty_pct / 100)
        
        # Total costs
        total_buy_cost = buy_price_eth + gas_cost_eth
        total_sell_costs = marketplace_fee_sell + royalty_fee + gas_cost_eth
        
        # Profit
        gross_profit = sell_price_eth - buy_price_eth
        net_profit = sell_price_eth - total_buy_cost - total_sell_costs
        
        # ROI
        roi_pct = (net_profit / buy_price_eth * 100) if buy_price_eth > 0 else 0
        
        return {
            "buy_price": buy_price_eth,
            "sell_price": sell_price_eth,
            "gross_profit": gross_profit,
            "marketplace_fee": marketplace_fee_sell,
            "royalty_fee": royalty_fee,
            "gas_cost": gas_cost_eth * 2,
            "net_profit": net_profit,
            "roi_pct": roi_pct,
            "break_even_price": total_buy_cost + total_sell_costs
        }
    
    def find_optimal_sell_price(
        self,
        buy_price_eth: float,
        target_profit_pct: float,
        royalty_pct: float,
        marketplace: NFTMarketplace
    ) -> float:
        """Find optimal sell price for target profit."""
        marketplace_fee = self.marketplace_fees.get(marketplace, 0.025)
        gas_cost = 0.02  # Total gas for buy + sell
        
        # Solve: sell_price * (1 - fee - royalty) - buy_price - gas = target_profit
        total_fees = marketplace_fee + (royalty_pct / 100)
        target_profit = buy_price_eth * (target_profit_pct / 100)
        
        sell_price = (buy_price_eth + gas_cost + target_profit) / (1 - total_fees)
        
        return sell_price


class WhaleTrackerNFT:
    """
    NFT Whale Tracker
    =================
    Tracks NFT whale activity for alpha.
    """
    
    def __init__(self):
        self.whale_wallets: Dict[str, Dict[str, Any]] = {}
        self.whale_activity: List[Dict[str, Any]] = []
    
    def add_whale(self, address: str, label: str, total_value_eth: float) -> None:
        """Add whale wallet to tracking."""
        self.whale_wallets[address.lower()] = {
            "address": address,
            "label": label,
            "total_value_eth": total_value_eth,
            "last_activity": time.time()
        }
    
    def record_activity(
        self,
        whale_address: str,
        activity_type: str,  # "buy", "sell", "transfer"
        collection: str,
        token_id: int,
        price_eth: float
    ) -> None:
        """Record whale activity."""
        activity = {
            "whale": whale_address,
            "type": activity_type,
            "collection": collection,
            "token_id": token_id,
            "price_eth": price_eth,
            "timestamp": time.time()
        }
        
        self.whale_activity.append(activity)
        
        # Update whale last activity
        if whale_address.lower() in self.whale_wallets:
            self.whale_wallets[whale_address.lower()]["last_activity"] = time.time()
    
    def get_collection_whale_activity(
        self,
        collection: str,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get recent whale activity for a collection."""
        cutoff = time.time() - hours * 3600
        
        return [
            a for a in self.whale_activity
            if a["collection"] == collection
            and a["timestamp"] > cutoff
        ]
    
    def detect_whale_accumulation(self, collection: str) -> Dict[str, Any]:
        """Detect if whales are accumulating a collection."""
        recent_activity = self.get_collection_whale_activity(collection, hours=24)
        
        buys = [a for a in recent_activity if a["type"] == "buy"]
        sells = [a for a in recent_activity if a["type"] == "sell"]
        
        buy_volume = sum(a["price_eth"] for a in buys)
        sell_volume = sum(a["price_eth"] for a in sells)
        
        net_volume = buy_volume - sell_volume
        buy_sell_ratio = len(buys) / max(len(sells), 1)
        
        if net_volume > 10 and buy_sell_ratio > 2:
            signal = "accumulation"
            strength = min(net_volume / 100, 1.0)
        elif net_volume < -10 and buy_sell_ratio < 0.5:
            signal = "distribution"
            strength = min(abs(net_volume) / 100, 1.0)
        else:
            signal = "neutral"
            strength = 0
        
        return {
            "collection": collection,
            "signal": signal,
            "strength": strength,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "net_volume": net_volume,
            "buy_count": len(buys),
            "sell_count": len(sells),
            "buy_sell_ratio": buy_sell_ratio
        }


# Export
__all__ = [
    "NFTMarketplace",
    "NFTCategory",
    "NFTCollection",
    "NFTTrait",
    "NFTItem",
    "NFTSnipe",
    "NFTCollectionAnalyzer",
    "NFTSniper",
    "NFTFlipCalculator",
    "WhaleTrackerNFT"
]
