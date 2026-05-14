"""Tests for Market Data Feed integration."""
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestMarketDataFeed:
    """Tests for MarketDataFeed class."""
    
    def test_market_snapshot_dataclass(self):
        """Test MarketSnapshot dataclass creation."""
        from core.feeds.market_data_feed import MarketSnapshot
        
        snapshot = MarketSnapshot(
            symbol="BTCUSDT",
            price=75000.0,
            bid=74999.0,
            ask=75001.0,
            spread=2.0,
            timestamp=1234567890.0,
        )
        
        assert snapshot.symbol == "BTCUSDT"
        assert snapshot.price == 75000.0
        assert snapshot.bid == 74999.0
        assert snapshot.ask == 75001.0
        assert snapshot.spread == 2.0
        assert snapshot.funding_rate == 0.0
        assert snapshot.orderbook_imbalance == 0.0
    
    def test_market_data_feed_creation(self):
        """Test MarketDataFeed creation with default parameters."""
        from core.feeds.market_data_feed import MarketDataFeed
        
        feed = MarketDataFeed(symbol="BTCUSDT")
        
        assert feed.symbol == "BTCUSDT"
        assert feed.price_history_size == 500
        assert feed.funding_history_size == 100
        assert feed.orderbook_levels == 25
        assert feed._current_price == 0.0
    
    def test_orderbook_imbalance_calculation(self):
        """Test order book imbalance calculation."""
        from core.feeds.market_data_feed import MarketDataFeed
        
        feed = MarketDataFeed(symbol="BTCUSDT")
        
        # Set up mock order book
        feed._orderbook_bids = [[75000, 10], [74999, 5], [74998, 3]]
        feed._orderbook_asks = [[75001, 8], [75002, 4], [75003, 2]]
        
        imbalance = feed._calculate_orderbook_imbalance()
        
        # (10+5+3) - (8+4+2) / (10+5+3 + 8+4+2)
        # = (18 - 14) / 32 = 4/32 = 0.125
        assert abs(imbalance - 0.125) < 0.001
    
    def test_volatility_regime_calculation(self):
        """Test volatility regime calculation."""
        import numpy as np
        from core.feeds.market_data_feed import MarketDataFeed
        
        feed = MarketDataFeed(symbol="BTCUSDT")
        
        # Create stable prices (low volatility)
        stable_prices = [75000 + np.random.randn() * 10 for _ in range(100)]
        regime, score = feed._calculate_volatility_regime(stable_prices)
        
        assert regime in ["low", "normal", "high", "extreme"]
        assert 0.0 <= score <= 1.0
    
    def test_stats_empty_feed(self):
        """Test stats on empty feed."""
        from core.feeds.market_data_feed import MarketDataFeed
        
        feed = MarketDataFeed(symbol="BTCUSDT")
        stats = feed.get_stats()
        
        assert stats["symbol"] == "BTCUSDT"
        assert stats["running"] == False
        assert stats["price_count"] == 0
        assert stats["update_count"] == 0


class TestBybitRESTExtensions:
    """Tests for extended Bybit REST client methods."""
    
    def test_funding_history_endpoint(self):
        """Test that get_funding_history method exists."""
        from connectors.bybit.bybit_rest_client import BybitRestClient
        
        client = BybitRestClient()
        assert hasattr(client, 'get_funding_history')
    
    def test_recent_trades_endpoint(self):
        """Test that get_recent_trades method exists."""
        from connectors.bybit.bybit_rest_client import BybitRestClient
        
        client = BybitRestClient()
        assert hasattr(client, 'get_recent_trades')


class TestMainIntegration:
    """Integration tests for main.py with market feed."""
    
    def test_market_feed_available_flag(self):
        """Test that MARKET_FEED_AVAILABLE flag is set."""
        import main
        
        assert hasattr(main, 'MARKET_FEED_AVAILABLE')
        assert main.MARKET_FEED_AVAILABLE == True
    
    def test_argus_has_market_feed_attribute(self):
        """Test that Argus class has market_feed attribute."""
        from main import Argus
        
        system = Argus(mode='paper', capital=1000)
        
        assert hasattr(system, 'market_feed')
        assert hasattr(system, 'market_snapshot')
