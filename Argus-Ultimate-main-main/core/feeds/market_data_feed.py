"""
Market Data Feed
================
Real-time market data from Bybit with enhanced features integration.

Provides:
- Real-time prices via WebSocket
- Funding rate history
- Order book snapshots
- Recent trades
- Enhanced features (funding rate signals, order book imbalance, volatility regime)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies
_BYBIT_AVAILABLE = False


def _ensure_bybit():
    global _BYBIT_AVAILABLE
    if not _BYBIT_AVAILABLE:
        try:
            from connectors.bybit.bybit_rest_client import BybitRestClient
            _BYBIT_AVAILABLE = True
            return BybitRestClient
        except ImportError:
            logger.warning("Bybit REST client not available")
            return None
    from connectors.bybit.bybit_rest_client import BybitRestClient
    return BybitRestClient


@dataclass
class MarketSnapshot:
    """Complete market data snapshot with enhanced features."""
    # Core price data
    symbol: str
    price: float
    bid: float
    ask: float
    spread: float
    timestamp: float
    
    # Price history
    prices: List[float] = field(default_factory=list)
    volumes: List[float] = field(default_factory=list)
    
    # Funding rate
    funding_rate: float = 0.0
    funding_rate_history: List[float] = field(default_factory=list)
    
    # Order book
    orderbook_bids: List[List[float]] = field(default_factory=list)
    orderbook_asks: List[List[float]] = field(default_factory=list)
    orderbook_imbalance: float = 0.0
    
    # Recent trades
    recent_trades: List[Dict[str, Any]] = field(default_factory=list)
    trade_flow_imbalance: float = 0.0
    
    # Enhanced features
    volatility_regime: str = "normal"  # low, normal, high, extreme
    volatility_score: float = 0.5
    
    # Cross-asset for correlation
    cross_asset_prices: Dict[str, List[float]] = field(default_factory=dict)


class MarketDataFeed:
    """Real-time market data feed from Bybit with enhanced features.
    
    Usage:
        feed = MarketDataFeed(symbol="BTCUSDT")
        await feed.start()
        
        # In trading loop
        snapshot = await feed.get_snapshot()
        # Use snapshot.price, snapshot.funding_rate, snapshot.orderbook_imbalance, etc.
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        price_history_size: int = 500,
        funding_history_size: int = 100,
        orderbook_levels: int = 25,
        update_interval: float = 1.0,  # seconds between REST updates
    ):
        self.symbol = symbol
        self.price_history_size = price_history_size
        self.funding_history_size = funding_history_size
        self.orderbook_levels = orderbook_levels
        self.update_interval = update_interval
        
        # Data stores
        self._prices: Deque[float] = deque(maxlen=price_history_size)
        self._volumes: Deque[float] = deque(maxlen=price_history_size)
        self._funding_rates: Deque[float] = deque(maxlen=funding_history_size)
        
        # Current state
        self._current_price: float = 0.0
        self._current_bid: float = 0.0
        self._current_ask: float = 0.0
        self._current_funding: float = 0.0
        self._orderbook_bids: List[List[float]] = []
        self._orderbook_asks: List[List[float]] = []
        self._recent_trades: List[Dict[str, Any]] = []
        
        # Cross-asset data (ETH for correlation)
        self._eth_prices: Deque[float] = deque(maxlen=price_history_size)
        
        # Client
        self._client = None
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._last_update: float = 0
        self._update_count: int = 0
        self._errors: int = 0
    
    async def start(self) -> bool:
        """Start the market data feed."""
        global _BYBIT_AVAILABLE
        
        BybitRestClient = _ensure_bybit()
        if BybitRestClient is None:
            logger.error("Cannot start market feed: Bybit REST client not available")
            return False
        
        try:
            self._client = BybitRestClient(testnet=False, category="linear")
            
            # Initial data fetch
            await self._fetch_all_data()
            
            # Start background update loop
            self._running = True
            self._update_task = asyncio.create_task(self._update_loop())
            
            logger.info(f"Market Data Feed started for {self.symbol}")
            logger.info(f"  Initial price: ${self._current_price:,.2f}")
            logger.info(f"  Funding rate: {self._current_funding * 100:.4f}%")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start market feed: {e}")
            self._errors += 1
            return False
    
    async def stop(self) -> None:
        """Stop the market data feed."""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.close()
        logger.info("Market Data Feed stopped")
    
    async def _update_loop(self) -> None:
        """Background loop to update market data."""
        while self._running:
            try:
                await self._fetch_all_data()
                self._update_count += 1
                self._last_update = time.time()
            except Exception as e:
                self._errors += 1
                logger.debug(f"Update error: {e}")
            
            await asyncio.sleep(self.update_interval)
    
    async def _fetch_all_data(self) -> None:
        """Fetch all market data from Bybit with timeout."""
        if not self._client:
            return
        
        # Fetch with timeout to prevent hanging
        try:
            await asyncio.wait_for(self._fetch_all_data_impl(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Market data fetch timeout - will retry next cycle")
        except Exception as e:
            logger.debug(f"Fetch all data error: {e}")
    
    async def _fetch_all_data_impl(self) -> None:
        """Internal implementation of data fetching."""
        # Fetch in parallel where possible
        try:
            # Ticker (price, bid, ask)
            ticker = await self._client.get_ticker(self.symbol)
            if ticker:
                self._current_price = float(ticker.last_price)
                self._current_bid = float(ticker.bid_price)
                self._current_ask = float(ticker.ask_price)
                self._prices.append(self._current_price)
                self._volumes.append(float(ticker.volume_24h))
        except Exception as e:
            logger.debug(f"Ticker fetch error: {e}")
        
        try:
            # Order book
            orderbook = await self._client.get_orderbook(
                self.symbol, 
                limit=self.orderbook_levels
            )
            if orderbook:
                self._orderbook_bids = orderbook.bids
                self._orderbook_asks = orderbook.asks
        except Exception as e:
            logger.debug(f"Orderbook fetch error: {e}")
        
        try:
            # Funding rate history
            funding_history = await self._client.get_funding_history(
                self.symbol,
                limit=self.funding_history_size
            )
            if funding_history:
                for record in funding_history:
                    rate = float(record.get("fundingRate", 0))
                    self._funding_rates.append(rate)
                if self._funding_rates:
                    self._current_funding = self._funding_rates[-1]
        except Exception as e:
            logger.debug(f"Funding rate fetch error: {e}")
        
        try:
            # Recent trades (optional - may not be available on all endpoints)
            trades = await self._client.get_recent_trades(self.symbol, limit=60)
            if trades:
                self._recent_trades = trades
        except Exception as e:
            logger.debug(f"Trades fetch error (optional): {e}")
    
    def get_snapshot(self) -> MarketSnapshot:
        """Get current market snapshot with all features."""
        prices_list = list(self._prices)
        volumes_list = list(self._volumes)
        
        # Calculate spread
        spread = self._current_ask - self._current_bid if self._current_bid and self._current_ask else 0.0
        
        # Calculate order book imbalance
        ob_imbalance = self._calculate_orderbook_imbalance()
        
        # Calculate trade flow imbalance
        tf_imbalance = self._calculate_trade_flow_imbalance()
        
        # Calculate volatility regime
        vol_regime, vol_score = self._calculate_volatility_regime(prices_list)
        
        return MarketSnapshot(
            symbol=self.symbol,
            price=self._current_price,
            bid=self._current_bid,
            ask=self._current_ask,
            spread=spread,
            timestamp=time.time(),
            prices=prices_list,
            volumes=volumes_list,
            funding_rate=self._current_funding,
            funding_rate_history=list(self._funding_rates),
            orderbook_bids=self._orderbook_bids,
            orderbook_asks=self._orderbook_asks,
            orderbook_imbalance=ob_imbalance,
            recent_trades=self._recent_trades,
            trade_flow_imbalance=tf_imbalance,
            volatility_regime=vol_regime,
            volatility_score=vol_score,
            cross_asset_prices={"ETH": list(self._eth_prices)},
        )
    
    def _calculate_orderbook_imbalance(self) -> float:
        """Calculate order book imbalance (-1 to +1).
        
        Positive = more bid volume (bullish pressure)
        Negative = more ask volume (bearish pressure)
        """
        if not self._orderbook_bids or not self._orderbook_asks:
            return 0.0
        
        try:
            bid_volume = sum(qty for _, qty in self._orderbook_bids[:10])
            ask_volume = sum(qty for _, qty in self._orderbook_asks[:10])
            
            if bid_volume + ask_volume == 0:
                return 0.0
            
            return (bid_volume - ask_volume) / (bid_volume + ask_volume)
        except Exception:
            return 0.0
    
    def _calculate_trade_flow_imbalance(self) -> float:
        """Calculate trade flow imbalance from recent trades."""
        if not self._recent_trades:
            return 0.0
        
        try:
            buy_volume = 0.0
            sell_volume = 0.0
            
            for trade in self._recent_trades[:30]:
                size = float(trade.get("size", 0))
                side = trade.get("side", "").lower()
                
                if side == "buy":
                    buy_volume += size
                elif side == "sell":
                    sell_volume += size
            
            if buy_volume + sell_volume == 0:
                return 0.0
            
            return (buy_volume - sell_volume) / (buy_volume + sell_volume)
        except Exception:
            return 0.0
    
    def _calculate_volatility_regime(self, prices: List[float]) -> Tuple[str, float]:
        """Calculate volatility regime based on recent price history.
        
        Returns:
            - regime: "low", "normal", "high", "extreme"
            - score: 0.0 (low vol) to 1.0 (extreme vol)
        """
        if len(prices) < 20:
            return "normal", 0.5
        
        try:
            # Calculate annualized volatility from recent returns
            recent = prices[-100:] if len(prices) >= 100 else prices
            returns = np.diff(recent) / recent[:-1]
            
            if len(returns) < 10:
                return "normal", 0.5
            
            # Annualized volatility (assuming 8760 hours per year for crypto)
            hourly_vol = np.std(returns)
            annualized_vol = hourly_vol * np.sqrt(8760)
            
            # Map to regime
            if annualized_vol < 0.3:  # < 30% annualized
                return "low", annualized_vol / 0.3
            elif annualized_vol < 0.6:  # 30-60%
                return "normal", 0.3 + (annualized_vol - 0.3) / 0.3 * 0.3
            elif annualized_vol < 1.0:  # 60-100%
                return "high", 0.6 + (annualized_vol - 0.6) / 0.4 * 0.3
            else:  # > 100%
                return "extreme", min(1.0, 0.9 + (annualized_vol - 1.0) * 0.1)
                
        except Exception:
            return "normal", 0.5
    
    def get_stats(self) -> Dict[str, Any]:
        """Get feed statistics."""
        return {
            "symbol": self.symbol,
            "running": self._running,
            "price_count": len(self._prices),
            "funding_count": len(self._funding_rates),
            "update_count": self._update_count,
            "errors": self._errors,
            "last_update": self._last_update,
            "current_price": self._current_price,
            "current_funding": self._current_funding,
        }
