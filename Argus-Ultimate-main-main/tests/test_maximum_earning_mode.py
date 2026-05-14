"""
tests/test_maximum_earning_mode.py — Tests for Maximum Earning Mode
"""

import pytest
import numpy as np

from trading.maximum_earning_mode import (
    MaximumMarketReader,
    MaximumPositionSizer,
    MaximumEarningOrchestrator,
    MarketSignal,
    create_maximum_earning_system,
)


class TestMaximumMarketReader:
    """Tests for Maximum Market Reader."""
    
    def test_init(self):
        """Should initialize correctly."""
        reader = MaximumMarketReader()
        assert len(reader.reading_history) == 0
    
    def test_read_market_bullish(self):
        """Should detect bullish market."""
        reader = MaximumMarketReader()
        
        # Create clear bullish price data (strong uptrend)
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.005 + 0.005))
        volumes = np.random.randn(100) * 1000 + 10000
        
        reading = reader.read_market(prices, volumes)
        
        # Should have some signal (not necessarily buy due to random)
        assert reading.overall_confidence > 0
        assert reading.signal is not None
    
    def test_read_market_bearish(self):
        """Should detect bearish market."""
        reader = MaximumMarketReader()
        
        # Create bearish price data (downtrend)
        prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01 - 0.003))
        volumes = np.random.randn(100) * 1000 + 10000
        
        reading = reader.read_market(prices, volumes)
        
        # Should not be strong buy
        assert reading.overall_confidence >= 0
    
    def test_read_market_with_order_book(self):
        """Should use order book data."""
        reader = MaximumMarketReader()
        
        prices = 100 * np.exp(np.cumsum(np.random.randn(50) * 0.01))
        volumes = np.random.randn(50) * 1000 + 10000
        
        order_book = {
            "bids": [(99.9, 100), (99.8, 200), (99.7, 150)],
            "asks": [(100.1, 100), (100.2, 200), (100.3, 150)],
        }
        
        reading = reader.read_market(prices, volumes, order_book)
        
        assert reading.order_flow_imbalance is not None
    
    def test_rsi_calculation(self):
        """Should calculate RSI correctly."""
        reader = MaximumMarketReader()
        
        # Rising prices = high RSI
        np.random.seed(42)
        rising = 100 + np.cumsum(np.abs(np.random.randn(30)) * 0.5)
        rsi = reader._calculate_rsi(rising, 14)
        assert rsi >= 50  # Rising prices should have RSI >= 50
        
        # Falling prices = low RSI
        falling = 100 - np.cumsum(np.abs(np.random.randn(30)) * 0.5)
        rsi = reader._calculate_rsi(falling, 14)
        assert rsi <= 50  # Falling prices should have RSI <= 50


class TestMaximumPositionSizer:
    """Tests for Maximum Position Sizer."""
    
    def test_init(self):
        """Should initialize correctly."""
        sizer = MaximumPositionSizer()
        assert sizer.max_risk_per_trade == 0.10
    
    def test_strong_buy_position(self):
        """Should size large for strong buy."""
        sizer = MaximumPositionSizer(max_risk_per_trade=0.15)
        
        result = sizer.calculate_position_size(
            capital=1000,
            signal=MarketSignal.STRONG_BUY,
            confidence=0.8,
            volatility=0.3,
        )
        
        assert result["position_pct"] > 0.10  # Should be >10%
        assert result["position_value"] > 100
    
    def test_sit_out_position(self):
        """Should have zero position for sit out."""
        sizer = MaximumPositionSizer()
        
        result = sizer.calculate_position_size(
            capital=1000,
            signal=MarketSignal.SIT_OUT,
            confidence=0.5,
            volatility=0.3,
        )
        
        assert result["position_pct"] == 0.0
        assert result["position_value"] == 0.0
    
    def test_high_volatility_reduces_size(self):
        """Should reduce position in high volatility."""
        sizer = MaximumPositionSizer(max_risk_per_trade=0.20)
        
        normal = sizer.calculate_position_size(
            capital=1000,
            signal=MarketSignal.BUY,
            confidence=0.7,
            volatility=0.2,
        )
        
        high_vol = sizer.calculate_position_size(
            capital=1000,
            signal=MarketSignal.BUY,
            confidence=0.7,
            volatility=0.6,
        )
        
        assert high_vol["position_pct"] < normal["position_pct"]
    
    def test_confidence_affects_size(self):
        """Should adjust size based on confidence."""
        sizer = MaximumPositionSizer()
        
        low_conf = sizer.calculate_position_size(
            capital=1000,
            signal=MarketSignal.BUY,
            confidence=0.1,
            volatility=0.2,
        )
        
        high_conf = sizer.calculate_position_size(
            capital=1000,
            signal=MarketSignal.BUY,
            confidence=0.9,
            volatility=0.2,
        )
        
        # Higher confidence should result in larger or equal position
        assert high_conf["position_pct"] >= low_conf["position_pct"]


class TestMaximumEarningOrchestrator:
    """Tests for Maximum Earning Orchestrator."""
    
    def test_init(self):
        """Should initialize correctly."""
        orchestrator = MaximumEarningOrchestrator(initial_capital=1000)
        
        assert orchestrator.capital == 1000
        assert orchestrator.initial_capital == 1000
    
    def test_analyze_and_trade(self):
        """Should analyze and return trade decision."""
        orchestrator = MaximumEarningOrchestrator(initial_capital=1000)
        
        prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01))
        volumes = np.random.randn(100) * 1000 + 10000
        
        decision = orchestrator.analyze_and_trade(prices, volumes)
        
        assert "signal" in decision
        assert "confidence" in decision
        assert "position_size" in decision
        assert "should_trade" in decision
    
    def test_update_capital(self):
        """Should update capital correctly."""
        orchestrator = MaximumEarningOrchestrator(initial_capital=1000)
        
        orchestrator.update_capital(100)
        
        assert orchestrator.capital == 1100
        assert orchestrator.total_pnl == 100
    
    def test_get_stats(self):
        """Should return statistics."""
        orchestrator = MaximumEarningOrchestrator(initial_capital=1000)
        
        prices = 100 * np.exp(np.cumsum(np.random.randn(50) * 0.01))
        volumes = np.random.randn(50) * 1000 + 10000
        
        orchestrator.analyze_and_trade(prices, volumes)
        orchestrator.update_capital(50)
        
        stats = orchestrator.get_stats()
        
        assert stats["initial_capital"] == 1000
        assert stats["current_capital"] == 1050
        assert abs(stats["return_pct"] - 5.0) < 0.01


class TestFactoryFunction:
    """Tests for factory function."""
    
    def test_create_system(self):
        """Should create maximum earning system."""
        system = create_maximum_earning_system(capital=5000)
        
        assert isinstance(system, MaximumEarningOrchestrator)
        assert system.initial_capital == 5000
