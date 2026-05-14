"""
Cross-Exchange Arbitrage Strategy
=================================

Exploits price differences between exchanges.
Includes triangular, basis, and funding rate arbitrage.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal
from collections import defaultdict
import numpy as np

from core.unified_config import config

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity."""
    strategy: str  # 'simple', 'triangular', 'basis', 'funding'
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    size: float
    gross_profit: float
    fees: float
    net_profit: float
    profit_pct: float
    confidence: float
    timestamp: float


@dataclass
class ExchangePrice:
    """Price data from an exchange."""
    exchange: str
    symbol: str
    bid: float
    ask: float
    bid_volume: float
    ask_volume: float
    timestamp: float
    latency_ms: float


class CrossExchangeArbitrageStrategy:
    """
    Multi-exchange arbitrage strategy.
    
    Types of arbitrage:
    1. Simple: Buy low on one exchange, sell high on another
    2. Triangular: Cross-currency arbitrage (e.g., BTC-AUD → BTC-USD → USD-AUD)
    3. Basis: Spot vs perpetual futures arbitrage
    4. Funding: Funding rate arbitrage between perpetuals
    """
    
    def __init__(self, min_profit_bps: float = 10.0,
                 max_position_hold_seconds: float = 300.0):
        self.min_profit_bps = min_profit_bps  # 10 bps minimum
        self.max_hold_seconds = max_position_hold_seconds
        
        # Price feeds
        self.price_feeds: Dict[str, Dict[str, ExchangePrice]] = defaultdict(dict)
        self.last_update: Dict[str, float] = {}
        
        # Active arbitrages
        self.active_arbitrages: Dict[str, ArbitrageOpportunity] = {}
        
        # Performance tracking
        self.total_opportunities = 0
        self.executed_opportunities = 0
        self.total_profit = 0.0
        
        logger.info(f"Cross-Exchange Arbitrage initialized (min_profit={min_profit_bps}bps)")
    
    def update_price(self, exchange: str, symbol: str,
                    bid: float, ask: float,
                    bid_vol: float = 0, ask_vol: float = 0,
                    latency_ms: float = 0):
        """Update price from an exchange."""
        self.price_feeds[symbol][exchange] = ExchangePrice(
            exchange=exchange,
            symbol=symbol,
            bid=bid,
            ask=ask,
            bid_volume=bid_vol,
            ask_volume=ask_vol,
            timestamp=time.time(),
            latency_ms=latency_ms
        )
        self.last_update[f"{exchange}:{symbol}"] = time.time()
    
    async def scan_opportunities(self) -> List[ArbitrageOpportunity]:
        """
        Scan all symbols for arbitrage opportunities.
        
        Returns:
            List of viable opportunities
        """
        opportunities = []
        
        for symbol, exchange_prices in self.price_feeds.items():
            # Simple arbitrage
            simple_ops = self._find_simple_arbitrage(symbol, exchange_prices)
            opportunities.extend(simple_ops)
            
            # Triangular arbitrage (if AUD/USD rates available)
            if 'BTC-AUD' in symbol or 'BTC-USD' in symbol:
                triangular_ops = self._find_triangular_arbitrage(symbol, exchange_prices)
                opportunities.extend(triangular_ops)
        
        # Sort by profit
        opportunities.sort(key=lambda x: x.net_profit, reverse=True)
        
        # Update stats
        self.total_opportunities += len(opportunities)
        
        return opportunities
    
    def _find_simple_arbitrage(self, symbol: str,
                               exchange_prices: Dict[str, ExchangePrice]
                              ) -> List[ArbitrageOpportunity]:
        """Find simple buy-low-sell-high opportunities."""
        opportunities = []
        
        exchanges = list(exchange_prices.keys())
        
        for i, buy_ex in enumerate(exchanges):
            for sell_ex in exchanges[i+1:]:
                if buy_ex == sell_ex:
                    continue
                
                buy_price_data = exchange_prices[buy_ex]
                sell_price_data = exchange_prices[sell_ex]
                
                # Check if data is fresh (<5 seconds old)
                if time.time() - buy_price_data.timestamp > 5:
                    continue
                if time.time() - sell_price_data.timestamp > 5:
                    continue
                
                # Buy at ask on buy_ex, sell at bid on sell_ex
                buy_price = buy_price_data.ask
                sell_price = sell_price_data.bid
                
                # Calculate spread
                spread_bps = (sell_price - buy_price) / buy_price * 10000
                
                if spread_bps < self.min_profit_bps:
                    continue
                
                # Calculate fees
                buy_fee = self._get_fee(buy_ex, 'taker')
                sell_fee = self._get_fee(sell_ex, 'taker')
                
                # Calculate optimal size (limited by volume)
                size = min(buy_price_data.ask_volume, sell_price_data.bid_volume)
                size = min(size, 1000.0)  # Max $1000 per arb
                
                # Calculate profit
                gross_profit = (sell_price - buy_price) * size
                fees = (buy_price * size * buy_fee) + (sell_price * size * sell_fee)
                net_profit = gross_profit - fees
                
                if net_profit <= 0:
                    continue
                
                opportunity = ArbitrageOpportunity(
                    strategy='simple',
                    symbol=symbol,
                    buy_exchange=buy_ex,
                    sell_exchange=sell_ex,
                    buy_price=buy_price,
                    sell_price=sell_price,
                    size=size,
                    gross_profit=gross_profit,
                    fees=fees,
                    net_profit=net_profit,
                    profit_pct=spread_bps / 100,
                    confidence=min(1.0, spread_bps / 50.0),  # Higher spread = higher confidence
                    timestamp=time.time()
                )
                
                opportunities.append(opportunity)
        
        return opportunities
    
    def _find_triangular_arbitrage(self, symbol: str,
                                   exchange_prices: Dict[str, ExchangePrice]
                                  ) -> List[ArbitrageOpportunity]:
        """
        Find triangular arbitrage opportunities.
        
        Example: BTC-AUD → BTC-USDT → USDT-AUD
        """
        opportunities = []
        
        # This requires USD/AUD rate
        # Simplified implementation
        
        if 'AUD' in symbol:
            # Look for BTC-USD and USD-AUD conversion
            pass  # Complex implementation omitted for brevity
        
        return opportunities
    
    def _get_fee(self, exchange: str, side: str) -> float:
        """Get trading fee for exchange."""
        fee_structure = {
            'btcmarkets': {'maker': -0.0005, 'taker': 0.002},
            'bybit': {'maker': 0.001, 'taker': 0.001},
            'mexc': {'maker': 0.0, 'taker': 0.0005},
            'kraken': {'maker': 0.0016, 'taker': 0.0026},
        }
        
        exchange_fees = fee_structure.get(exchange, {'maker': 0.001, 'taker': 0.002})
        return exchange_fees.get(side, 0.002)
    
    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> bool:
        """
        Execute arbitrage opportunity.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Executing arbitrage: {opportunity.strategy} "
                   f"{opportunity.symbol} "
                   f"{opportunity.buy_exchange}→{opportunity.sell_exchange} "
                   f"Profit: ${opportunity.net_profit:.2f}")
        
        try:
            # Step 1: Buy on buy_exchange
            buy_order = await self._place_order(
                exchange=opportunity.buy_exchange,
                symbol=opportunity.symbol,
                side='buy',
                size=opportunity.size,
                price=opportunity.buy_price
            )
            
            if not buy_order:
                logger.error("Buy order failed")
                return False
            
            # Step 2: Sell on sell_exchange
            sell_order = await self._place_order(
                exchange=opportunity.sell_exchange,
                symbol=opportunity.symbol,
                side='sell',
                size=opportunity.size,
                price=opportunity.sell_price
            )
            
            if not sell_order:
                logger.error("Sell order failed - need to handle stuck position!")
                # TODO: Emergency position close
                return False
            
            # Track successful execution
            self.executed_opportunities += 1
            self.total_profit += opportunity.net_profit
            
            logger.info(f"Arbitrage executed successfully: +${opportunity.net_profit:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Arbitrage execution failed: {e}")
            return False
    
    async def _place_order(self, exchange: str, symbol: str,
                          side: str, size: float, price: float) -> Optional[Dict]:
        """Place order on exchange."""
        # This would integrate with exchange connectors
        # Simplified implementation
        
        logger.debug(f"Placing {side} order on {exchange}: {size} {symbol} @ {price}")
        
        # Simulate order
        return {
            'order_id': f"sim_{time.time()}",
            'status': 'filled',
            'filled_size': size,
            'avg_price': price
        }
    
    def get_performance_stats(self) -> Dict:
        """Get arbitrage performance statistics."""
        return {
            'total_opportunities': self.total_opportunities,
            'executed_opportunities': self.executed_opportunities,
            'execution_rate': (self.executed_opportunities / max(self.total_opportunities, 1)),
            'total_profit': self.total_profit,
            'avg_profit_per_trade': self.total_profit / max(self.executed_opportunities, 1),
            'monitored_symbols': len(self.price_feeds),
            'active_exchanges': len(set(ex for prices in self.price_feeds.values() for ex in prices.keys()))
        }


class BasisArbitrageStrategy:
    """
    Spot vs Perpetual Futures basis arbitrage.
    
    Strategy:
    - When basis (perp - spot) > funding rate: Short perp, long spot
    - When basis < funding rate: Long perp, short spot
    """
    
    def __init__(self, min_basis_bps: float = 15.0):
        self.min_basis_bps = min_basis_bps
        
        # Track funding rates
        self.funding_rates: Dict[str, Dict] = {}
        self.spot_prices: Dict[str, float] = {}
        self.perp_prices: Dict[str, float] = {}
        
        logger.info(f"Basis Arbitrage initialized (min_basis={min_basis_bps}bps)")
    
    def update_funding_rate(self, symbol: str, exchange: str,
                           funding_rate: float, next_funding_time: float):
        """Update funding rate."""
        if symbol not in self.funding_rates:
            self.funding_rates[symbol] = {}
        
        self.funding_rates[symbol][exchange] = {
            'rate': funding_rate,
            'next_funding': next_funding_time,
            'timestamp': time.time()
        }
    
    def update_prices(self, symbol: str, spot_price: float, perp_price: float):
        """Update spot and perpetual prices."""
        self.spot_prices[symbol] = spot_price
        self.perp_prices[symbol] = perp_price
    
    def scan_basis_opportunities(self) -> List[ArbitrageOpportunity]:
        """Scan for basis arbitrage opportunities."""
        opportunities = []
        
        for symbol in self.spot_prices:
            if symbol not in self.perp_prices:
                continue
            
            spot = self.spot_prices[symbol]
            perp = self.perp_prices[symbol]
            
            # Calculate basis
            basis = perp - spot
            basis_bps = basis / spot * 10000
            
            # Get funding rate
            funding_rate = 0.0
            if symbol in self.funding_rates:
                # Use most recent funding rate
                funding_data = self.funding_rates[symbol]
                if funding_data:
                    funding_rate = max(d['rate'] for d in funding_data.values())
            
            # Opportunity exists when |basis| > |funding|
            if abs(basis_bps) > self.min_basis_bps:
                if basis > 0:  # Perp premium
                    # Short perp, long spot
                    opp = ArbitrageOpportunity(
                        strategy='basis',
                        symbol=symbol,
                        buy_exchange='spot',
                        sell_exchange='perp',
                        buy_price=spot,
                        sell_price=perp,
                        size=100.0,  # Example size
                        gross_profit=basis * 100,
                        fees=0.0,  # Simplified
                        net_profit=basis * 100 - funding_rate * 100,
                        profit_pct=basis_bps / 100,
                        confidence=min(1.0, abs(basis_bps) / 50.0),
                        timestamp=time.time()
                    )
                    opportunities.append(opp)
                else:  # Spot premium
                    # Long perp, short spot
                    opp = ArbitrageOpportunity(
                        strategy='basis',
                        symbol=symbol,
                        buy_exchange='perp',
                        sell_exchange='spot',
                        buy_price=perp,
                        sell_price=spot,
                        size=100.0,
                        gross_profit=-basis * 100,
                        fees=0.0,
                        net_profit=-basis * 100 - funding_rate * 100,
                        profit_pct=-basis_bps / 100,
                        confidence=min(1.0, abs(basis_bps) / 50.0),
                        timestamp=time.time()
                    )
                    opportunities.append(opp)
        
        return opportunities


class FundingRateArbitrageStrategy:
    """
    Funding rate arbitrage between perpetual exchanges.
    
    Strategy:
    - Long on exchange with negative funding (getting paid)
    - Short on exchange with positive funding (paying less)
    """
    
    def __init__(self, min_funding_diff: float = 0.0001):  # 0.01%
        self.min_funding_diff = min_funding_diff
        self.funding_rates: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        logger.info(f"Funding Rate Arbitrage initialized")
    
    def update_funding(self, symbol: str, exchange: str, funding_rate: float):
        """Update funding rate for symbol on exchange."""
        self.funding_rates[symbol][exchange] = funding_rate
    
    def scan_funding_opportunities(self) -> List[ArbitrageOpportunity]:
        """Scan for funding arbitrage opportunities."""
        opportunities = []
        
        for symbol, exchange_funding in self.funding_rates.items():
            if len(exchange_funding) < 2:
                continue
            
            # Find max and min funding rates
            max_funding_ex = max(exchange_funding.items(), key=lambda x: x[1])
            min_funding_ex = min(exchange_funding.items(), key=lambda x: x[1])
            
            max_rate = max_funding_ex[1]
            min_rate = min_funding_ex[1]
            
            # Opportunity when difference is significant
            if (max_rate - min_rate) > self.min_funding_diff:
                # Short on high funding, long on low funding
                opp = ArbitrageOpportunity(
                    strategy='funding',
                    symbol=symbol,
                    buy_exchange=min_funding_ex[0],  # Long here (pay less/get paid)
                    sell_exchange=max_funding_ex[0],  # Short here (pay less)
                    buy_price=0,  # Not applicable for funding arb
                    sell_price=0,
                    size=100.0,
                    gross_profit=0,
                    fees=0,
                    net_profit=(max_rate - min_rate) * 100,  # Simplified
                    profit_pct=(max_rate - min_rate) * 100,
                    confidence=min(1.0, (max_rate - min_rate) / 0.001),
                    timestamp=time.time()
                )
                opportunities.append(opp)
        
        return opportunities


# Factory functions
def create_arbitrage_strategy(strategy_type: str = 'cross_exchange', **kwargs):
    """Create arbitrage strategy."""
    if strategy_type == 'cross_exchange':
        return CrossExchangeArbitrageStrategy(**kwargs)
    elif strategy_type == 'basis':
        return BasisArbitrageStrategy(**kwargs)
    elif strategy_type == 'funding':
        return FundingRateArbitrageStrategy(**kwargs)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


# Global instances
_arbitrage_strategies: Dict[str, Any] = {}


def get_arbitrage_strategy(name: str = 'cross_exchange') -> Any:
    """Get or create arbitrage strategy."""
    global _arbitrage_strategies
    if name not in _arbitrage_strategies:
        _arbitrage_strategies[name] = create_arbitrage_strategy(name)
    return _arbitrage_strategies[name]
