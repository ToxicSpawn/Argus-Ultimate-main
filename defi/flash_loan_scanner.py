"""
Argus Flash Loan Scanner
Version: 1.0.0

Scans for flash loan arbitrage opportunities across DEXes.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageOpportunity:
    """Flash loan arbitrage opportunity."""
    token: str
    buy_dex: str
    sell_dex: str
    buy_price: float
    sell_price: float
    amount: float
    gross_profit: float
    flash_loan_fee: float
    gas_cost: float
    net_profit: float
    profit_percentage: float
    chain: str
    timestamp: float
    confidence: float = 0.95


@dataclass
class DEXPrice:
    """Price from a DEX."""
    dex: str
    token: str
    price: float
    liquidity: float
    volume_24h: float
    timestamp: float


class FlashLoanScanner:
    """
    Scans for flash loan arbitrage opportunities.
    
    Monitors multiple DEXes across chains for price discrepancies.
    """
    
    VERSION = "1.0.0"
    
    # Supported chains
    CHAINS = {
        "ethereum": {
            "rpc_url": None,  # Set from config
            "dexes": ["uniswap_v2", "uniswap_v3", "sushiswap", "curve", "balancer"],
            "native_token": "WETH",
            "stablecoins": ["USDC", "USDT", "DAI"],
            "avg_gas_cost": 30.0  # USD
        },
        "arbitrum": {
            "rpc_url": None,
            "dexes": ["uniswap_v3", "sushiswap", "curve", "gmx"],
            "native_token": "WETH",
            "stablecoins": ["USDC", "USDT", "DAI"],
            "avg_gas_cost": 0.50  # USD
        },
        "polygon": {
            "rpc_url": None,
            "dexes": ["quickswap", "sushiswap", "curve", "balancer"],
            "native_token": "WMATIC",
            "stablecoins": ["USDC", "USDT", "DAI"],
            "avg_gas_cost": 0.01  # USD
        }
    }
    
    # Tokens to monitor
    MONITORED_TOKENS = [
        "WETH", "WBTC", "USDC", "USDT", "DAI",
        "stETH", "frxETH", "CRV", "CVX", "LINK",
        "UNI", "AAVE", "ARB", "MATIC"
    ]
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize scanner."""
        self.config = config or {}
        
        # Settings
        self.min_profit_usd = self.config.get("min_profit_usd", 50)
        self.min_profit_percent = self.config.get("min_profit_percent", 0.001)
        self.max_loan_usd = self.config.get("max_loan_usd", 1_000_000)
        self.scan_interval = self.config.get("scan_interval", 1.0)
        
        # Price cache
        self.prices: Dict[str, Dict[str, DEXPrice]] = {}  # token -> dex -> price
        self.last_scan: float = 0
        
        # Statistics
        self.scans_completed = 0
        self.opportunities_found = 0
        self.opportunities_executed = 0
        self.total_profit = 0.0
        self.opportunity_history: deque = deque(maxlen=1000)
        
        logger.info(f"FlashLoanScanner v{self.VERSION} initialized")
        logger.info(f"  Min profit: ${self.min_profit_usd}")
        logger.info(f"  Max loan: ${self.max_loan_usd:,.0f}")
    
    def scan_opportunities(self, chain: str = "ethereum") -> List[ArbitrageOpportunity]:
        """
        Scan for arbitrage opportunities on a chain.
        
        Returns list of profitable opportunities.
        """
        self.scans_completed += 1
        opportunities = []
        
        chain_config = self.CHAINS.get(chain)
        if not chain_config:
            return opportunities
        
        # Get prices from all DEXes (simulated - would query actual DEX contracts)
        prices = self._fetch_prices(chain)
        
        # Compare prices across DEXes
        for token in self.MONITORED_TOKENS:
            token_prices = prices.get(token, {})
            
            if len(token_prices) < 2:
                continue
            
            # Find price discrepancies
            dexes = list(token_prices.keys())
            
            for i, buy_dex in enumerate(dexes):
                for sell_dex in dexes[i+1:]:
                    buy_price = token_prices[buy_dex].price
                    sell_price = token_prices[sell_dex].price
                    
                    # Check both directions
                    for buy_d, sell_d, bp, sp in [
                        (buy_dex, sell_dex, buy_price, sell_price),
                        (sell_dex, buy_dex, sell_price, buy_price)
                    ]:
                        if sp > bp * (1 + self.min_profit_percent):
                            opp = self._calculate_opportunity(
                                token=token,
                                buy_dex=buy_d,
                                sell_dex=sell_d,
                                buy_price=bp,
                                sell_price=sp,
                                chain=chain,
                                gas_cost=chain_config["avg_gas_cost"]
                            )
                            
                            if opp and opp.net_profit >= self.min_profit_usd:
                                opportunities.append(opp)
                                self.opportunities_found += 1
                                self.opportunity_history.append(opp)
        
        # Sort by net profit
        opportunities.sort(key=lambda x: x.net_profit, reverse=True)
        
        return opportunities
    
    def _fetch_prices(self, chain: str) -> Dict[str, Dict[str, DEXPrice]]:
        """Fetch prices from all DEXes (simulated)."""
        prices: Dict[str, Dict[str, DEXPrice]] = {}
        
        chain_config = self.CHAINS.get(chain, {})
        dexes = chain_config.get("dexes", [])
        
        # Simulated price fetching
        # In production, would query actual DEX contracts via RPC
        base_prices = {
            "WETH": 3500.0,
            "WBTC": 65000.0,
            "USDC": 1.0,
            "USDT": 1.0,
            "DAI": 1.0,
            "stETH": 3495.0,
            "frxETH": 3498.0,
            "CRV": 0.50,
            "CVX": 3.50,
            "LINK": 15.0,
            "UNI": 7.50,
            "AAVE": 95.0,
            "ARB": 1.20,
            "MATIC": 0.70
        }
        
        for token in self.MONITORED_TOKENS:
            if token not in prices:
                prices[token] = {}
            
            base_price = base_prices.get(token, 1.0)
            
            for dex in dexes:
                # Add small random variance to simulate real market
                variance = np.random.uniform(-0.002, 0.002)  # ±0.2%
                price = base_price * (1 + variance)
                
                prices[token][dex] = DEXPrice(
                    dex=dex,
                    token=token,
                    price=price,
                    liquidity=np.random.uniform(1_000_000, 100_000_000),
                    volume_24h=np.random.uniform(100_000, 10_000_000),
                    timestamp=time.time()
                )
        
        return prices
    
    def _calculate_opportunity(self, token: str, buy_dex: str, sell_dex: str,
                               buy_price: float, sell_price: float,
                               chain: str, gas_cost: float) -> Optional[ArbitrageOpportunity]:
        """Calculate flash loan arbitrage opportunity."""
        
        # Calculate optimal loan amount
        price_diff = sell_price - buy_price
        profit_per_token = price_diff / buy_price
        
        # Optimal amount balances flash loan fee with profit
        # Flash loan fee is typically 0.05-0.09%
        flash_loan_fee_rate = 0.0009  # 0.09% (Aave V3)
        
        # Break-even amount = gas_cost / (profit_per_token - fee_rate)
        if profit_per_token <= flash_loan_fee_rate:
            return None
        
        optimal_amount_usd = gas_cost / (profit_per_token - flash_loan_fee_rate)
        optimal_amount_usd = min(optimal_amount_usd, self.max_loan_usd)
        
        if optimal_amount_usd < 1000:  # Minimum $1000 loan
            return None
        
        # Calculate profits
        gross_profit = optimal_amount_usd * profit_per_token
        flash_loan_fee = optimal_amount_usd * flash_loan_fee_rate
        net_profit = gross_profit - flash_loan_fee - gas_cost
        
        if net_profit <= 0:
            return None
        
        return ArbitrageOpportunity(
            token=token,
            buy_dex=buy_dex,
            sell_dex=sell_dex,
            buy_price=buy_price,
            sell_price=sell_price,
            amount=optimal_amount_usd,
            gross_profit=gross_profit,
            flash_loan_fee=flash_loan_fee,
            gas_cost=gas_cost,
            net_profit=net_profit,
            profit_percentage=profit_per_token,
            chain=chain,
            timestamp=time.time(),
            confidence=0.95
        )
    
    def verify_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """Verify opportunity still exists (re-check prices)."""
        # Re-fetch prices
        prices = self._fetch_prices(opportunity.chain)
        
        token_prices = prices.get(opportunity.token, {})
        buy_price = token_prices.get(opportunity.buy_dex)
        sell_price = token_prices.get(opportunity.sell_dex)
        
        if not buy_price or not sell_price:
            return False
        
        # Check if still profitable
        current_spread = (sell_price.price - buy_price.price) / buy_price.price
        required_spread = opportunity.profit_percentage * 0.8  # 80% of original
        
        return current_spread >= required_spread
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scanner statistics."""
        return {
            "version": self.VERSION,
            "scans_completed": self.scans_completed,
            "opportunities_found": self.opportunities_found,
            "opportunities_executed": self.opportunities_executed,
            "total_profit": self.total_profit,
            "min_profit_threshold": self.min_profit_usd,
            "max_loan_size": self.max_loan_usd
        }
    
    def get_recent_opportunities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent opportunities."""
        recent = list(self.opportunity_history)[-limit:]
        return [
            {
                "token": opp.token,
                "chain": opp.chain,
                "buy_dex": opp.buy_dex,
                "sell_dex": opp.sell_dex,
                "amount": opp.amount,
                "net_profit": opp.net_profit,
                "profit_percentage": opp.profit_percentage * 100
            }
            for opp in recent
        ]


# Global scanner instance
_scanner_instance: Optional[FlashLoanScanner] = None


def get_flash_loan_scanner(config: Dict[str, Any] = None) -> FlashLoanScanner:
    """Get or create global Flash Loan Scanner instance."""
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = FlashLoanScanner(config)
    return _scanner_instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    scanner = get_flash_loan_scanner({
        "min_profit_usd": 50,
        "max_loan_usd": 1_000_000
    })
    
    print("\n=== Flash Loan Scanner Test ===")
    
    # Scan for opportunities
    opportunities = scanner.scan_opportunities("ethereum")
    
    print(f"\nFound {len(opportunities)} opportunities:")
    for opp in opportunities[:5]:
        print(f"  {opp.token}: ${opp.net_profit:.2f} profit ({opp.profit_percentage*100:.3f}%)")
        print(f"    Buy: {opp.buy_dex} @ ${opp.buy_price:.2f}")
        print(f"    Sell: {opp.sell_dex} @ ${opp.sell_price:.2f}")
        print(f"    Loan: ${opp.amount:,.0f}")
    
    print(f"\nScanner Stats: {scanner.get_stats()}")
