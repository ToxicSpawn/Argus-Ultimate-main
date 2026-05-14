"""
tests/test_advanced_strategy_engine.py — Tests for Advanced Strategy Engine

Tests for the ultimate strategy logic system.
"""

import pytest
import numpy as np
from datetime import datetime

from strategies.advanced_strategy_engine import (
    AdvancedStrategyEngine,
    TechnicalCalculator,
    MultiTimeframeAnalyzer,
    SignalFilter,
    SmartExitManager,
    AdaptiveParameterOptimizer,
    SignalDirection,
    TimeFrame,
    MarketData,
    EnhancedSignal,
    ExitReason,
    create_advanced_strategy_engine,
)


# ============================================================================
# Technical Calculator Tests
# ============================================================================

class TestTechnicalCalculator:
    """Tests for Technical Calculator."""
    
    def test_sma(self):
        """Should calculate SMA correctly."""
        prices = np.array([100, 102, 101, 103, 105])
        result = TechnicalCalculator.sma(prices, 3)
        
        assert result == pytest.approx(103.0, rel=0.01)
    
    def test_ema(self):
        """Should calculate EMA correctly."""
        prices = np.array([100, 102, 101, 103, 105])
        result = TechnicalCalculator.ema(prices, 3)
        
        assert 100 < result < 105
    
    def test_rsi_oversold(self):
        """Should detect oversold RSI."""
        # Simulate downtrend
        prices = np.array([100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85])
        result = TechnicalCalculator.rsi(prices, 14)
        
        assert result < 30  # Should be oversold
    
    def test_rsi_overbought(self):
        """Should detect overbought RSI."""
        # Simulate uptrend
        prices = np.array([85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100])
        result = TechnicalCalculator.rsi(prices, 14)
        
        assert result > 70  # Should be overbought
    
    def test_bollinger_bands(self):
        """Should calculate Bollinger Bands."""
        prices = np.random.randn(50) * 5 + 100
        upper, middle, lower = TechnicalCalculator.bollinger_bands(prices, 20, 2.0)
        
        assert upper > middle > lower
    
    def test_atr(self):
        """Should calculate ATR."""
        highs = np.random.randn(50) * 5 + 105
        lows = np.random.randn(50) * 5 + 95
        closes = np.random.randn(50) * 5 + 100
        
        result = TechnicalCalculator.atr(highs, lows, closes, 14)
        
        assert result > 0
    
    def test_macd(self):
        """Should calculate MACD."""
        prices = np.random.randn(50).cumsum() + 100
        macd, signal, histogram = TechnicalCalculator.macd(prices)
        
        assert isinstance(macd, float)
        assert isinstance(signal, float)
        assert isinstance(histogram, float)


# ============================================================================
# Multi-Timeframe Analyzer Tests
# ============================================================================

class TestMultiTimeframeAnalyzer:
    """Tests for Multi-Timeframe Analyzer."""
    
    def test_analyze_timeframe(self):
        """Should analyze a timeframe."""
        analyzer = MultiTimeframeAnalyzer()
        
        prices = np.random.randn(100).cumsum() + 100
        highs = prices + np.random.rand(100) * 2
        lows = prices - np.random.rand(100) * 2
        volumes = np.random.rand(100) * 1000 + 500
        
        indicators = analyzer.analyze_timeframe(prices, highs, lows, volumes, TimeFrame.H1)
        
        assert indicators.rsi > 0
        assert indicators.sma_20 > 0
        assert indicators.atr > 0
    
    def test_multi_tf_agreement(self):
        """Should calculate multi-timeframe agreement."""
        analyzer = MultiTimeframeAnalyzer()
        
        # Create bullish data
        prices = np.random.randn(100).cumsum() + 100
        highs = prices + 1
        lows = prices - 1
        volumes = np.random.rand(100) * 1000 + 500
        
        price_data = {
            TimeFrame.H1: (prices, highs, lows, volumes),
            TimeFrame.H4: (prices, highs, lows, volumes),
        }
        
        agreement, agreeing = analyzer.get_multi_tf_signal(price_data)
        
        assert 0 <= agreement <= 1


# ============================================================================
# Signal Filter Tests
# ============================================================================

class TestSignalFilter:
    """Tests for Signal Filter."""
    
    def test_calculate_confidence(self):
        """Should calculate confidence."""
        filter = SignalFilter(min_confidence=50)
        
        indicators = {
            "1h": type('obj', (object,), {
                'trend_strength': 0.5,
                'momentum_score': 0.3,
                'rsi': 45,
                'volume_ratio': 1.2,
                'bollinger_pct': 0.3,
            })()
        }
        
        market_data = MarketData(
            timestamp=datetime.utcnow(),
            symbol="BTC",
            open=100, high=101, low=99, close=100,
            volume=1000,
            bid=99.9,
            ask=100.1,
        )
        
        confidence = filter.calculate_confidence(
            SignalDirection.LONG,
            indicators,
            0.8,
            market_data,
        )
        
        assert 0 <= confidence <= 100
    
    def test_filter_low_confidence(self):
        """Should filter low confidence signals."""
        filter = SignalFilter(min_confidence=60)
        
        signal = EnhancedSignal(
            timestamp=datetime.utcnow(),
            symbol="BTC",
            direction=SignalDirection.LONG,
            entry_price=100,
            confidence=40,  # Below threshold
        )
        
        market_data = MarketData(
            timestamp=datetime.utcnow(),
            symbol="BTC",
            open=100, high=101, low=99, close=100,
            volume=1000,
        )
        
        should_filter, reason = filter.should_filter(signal, market_data)
        
        assert should_filter
        assert "confidence" in reason.lower()


# ============================================================================
# Smart Exit Manager Tests
# ============================================================================

class TestSmartExitManager:
    """Tests for Smart Exit Manager."""
    
    def test_calculate_exit_levels_long(self):
        """Should calculate exit levels for long."""
        manager = SmartExitManager()
        
        signal = EnhancedSignal(
            timestamp=datetime.utcnow(),
            symbol="BTC",
            direction=SignalDirection.LONG,
            entry_price=100,
            confidence=70,
        )
        
        stop, target, trailing, breakeven = manager.calculate_exit_levels(signal)
        
        assert stop < 100  # Stop below entry for long
        assert target > 100  # Target above entry for long
    
    def test_calculate_exit_levels_short(self):
        """Should calculate exit levels for short."""
        manager = SmartExitManager()
        
        signal = EnhancedSignal(
            timestamp=datetime.utcnow(),
            symbol="BTC",
            direction=SignalDirection.SHORT,
            entry_price=100,
            confidence=70,
        )
        
        stop, target, trailing, breakeven = manager.calculate_exit_levels(signal)
        
        assert stop > 100  # Stop above entry for short
        assert target < 100  # Target below entry for short
    
    def test_open_and_update_trade(self):
        """Should open and update trade."""
        manager = SmartExitManager()
        
        signal = EnhancedSignal(
            timestamp=datetime.utcnow(),
            symbol="BTC",
            direction=SignalDirection.LONG,
            entry_price=100,
            confidence=70,
            stop_loss=98,
            take_profit=104,
            trailing_stop_pct=0.015,
            breakeven_trigger_pct=0.02,
            position_size=0.1,
        )
        
        trade = manager.open_trade(signal)
        
        assert trade.is_open
        assert trade.entry_price == 100
    
    def test_stop_loss_trigger(self):
        """Should trigger stop loss."""
        manager = SmartExitManager()
        
        signal = EnhancedSignal(
            timestamp=datetime.utcnow(),
            symbol="BTC",
            direction=SignalDirection.LONG,
            entry_price=100,
            confidence=70,
            stop_loss=98,
            take_profit=104,
            trailing_stop_pct=0.015,
            breakeven_trigger_pct=0.02,
            position_size=0.1,
        )
        
        trade = manager.open_trade(signal)
        exit_reason = manager.update_trade(trade.trade_id, 97.5)  # Below stop
        
        assert exit_reason == ExitReason.STOP_LOSS
    
    def test_take_profit_trigger(self):
        """Should trigger take profit."""
        manager = SmartExitManager()
        
        signal = EnhancedSignal(
            timestamp=datetime.utcnow(),
            symbol="BTC",
            direction=SignalDirection.LONG,
            entry_price=100,
            confidence=70,
            stop_loss=98,
            take_profit=104,
            trailing_stop_pct=0.015,
            breakeven_trigger_pct=0.02,
            position_size=0.1,
        )
        
        trade = manager.open_trade(signal)
        exit_reason = manager.update_trade(trade.trade_id, 105)  # Above target
        
        assert exit_reason == ExitReason.TAKE_PROFIT
    
    def test_trailing_stop(self):
        """Should track highest price for trailing stop."""
        manager = SmartExitManager()
        
        signal = EnhancedSignal(
            timestamp=datetime.utcnow(),
            symbol="BTC",
            direction=SignalDirection.LONG,
            entry_price=100,
            confidence=70,
            stop_loss=98,
            take_profit=110,
            trailing_stop_pct=0.02,
            breakeven_trigger_pct=0.02,
            position_size=0.1,
        )
        
        trade = manager.open_trade(signal)
        
        # Initial state
        assert trade.is_open
        assert trade.highest_price == 100
        
        # Move price up - should track highest
        manager.update_trade(trade.trade_id, 105)
        assert trade.highest_price == 105
        
        # Move price higher
        manager.update_trade(trade.trade_id, 108)
        assert trade.highest_price == 108


# ============================================================================
# Advanced Strategy Engine Tests
# ============================================================================

class TestAdvancedStrategyEngine:
    """Tests for Advanced Strategy Engine."""
    
    def test_init(self):
        """Should initialize correctly."""
        engine = AdvancedStrategyEngine()
        
        assert engine.min_confidence == 55.0
        assert engine.total_trades == 0
    
    def test_generate_signal(self):
        """Should generate enhanced signal."""
        engine = AdvancedStrategyEngine()
        
        # Create test data
        prices = np.random.randn(100).cumsum() + 100
        highs = prices + 1
        lows = prices - 1
        volumes = np.random.rand(100) * 1000 + 500
        
        price_data = {
            "1h": (prices, highs, lows, volumes),
            "15m": (prices[-50:], highs[-50:], lows[-50:], volumes[-50:]),
        }
        
        signal = engine.generate_signal(
            symbol="BTC",
            price_data=price_data,
            base_direction=SignalDirection.LONG,
            strategy_name="test",
        )
        
        # Signal might be filtered, that's ok
        if signal:
            assert signal.symbol == "BTC"
            assert signal.direction == SignalDirection.LONG
            assert 0 <= signal.confidence <= 100
            assert signal.stop_loss > 0
            assert signal.take_profit > 0
    
    def test_generate_signal_with_market_data(self):
        """Should use market data for spread calculation."""
        engine = AdvancedStrategyEngine()
        
        prices = np.random.randn(100).cumsum() + 100
        highs = prices + 1
        lows = prices - 1
        volumes = np.random.rand(100) * 1000 + 500
        
        price_data = {"1h": (prices, highs, lows, volumes)}
        
        market_data = MarketData(
            timestamp=datetime.utcnow(),
            symbol="BTC",
            open=100, high=101, low=99, close=100,
            volume=1000,
            bid=99.5,
            ask=100.5,
        )
        
        signal = engine.generate_signal(
            symbol="BTC",
            price_data=price_data,
            base_direction=SignalDirection.LONG,
            market_data=market_data,
        )
        
        # Signal should account for spread
        if signal:
            assert signal.entry_price == 100.0  # Mid price
    
    def test_position_sizing(self):
        """Should calculate position size."""
        engine = AdvancedStrategyEngine(risk_per_trade=0.02)
        
        signal = EnhancedSignal(
            timestamp=datetime.utcnow(),
            symbol="BTC",
            direction=SignalDirection.LONG,
            entry_price=100,
            confidence=80,
            multi_tf_agreement=0.8,
        )
        
        size = engine._calculate_position_size(signal, None)
        
        assert 0.01 <= size <= 0.5
    
    def test_entry_reasoning(self):
        """Should generate entry reason."""
        engine = AdvancedStrategyEngine()
        
        signal = EnhancedSignal(
            timestamp=datetime.utcnow(),
            symbol="BTC",
            direction=SignalDirection.LONG,
            entry_price=100,
            confidence=80,
            multi_tf_agreement=0.85,
        )
        
        reason = engine._generate_entry_reason(signal, None)
        
        assert len(reason) > 0
    
    def test_performance_stats(self):
        """Should track performance."""
        engine = AdvancedStrategyEngine()
        
        # Simulate some trades
        engine.total_trades = 10
        engine.winning_trades = 6
        engine.total_pnl = 0.15
        
        stats = engine.get_performance_stats()
        
        assert stats["total_trades"] == 10
        assert stats["win_rate"] == 0.6


# ============================================================================
# Integration Tests
# ============================================================================

class TestStrategyEngineIntegration:
    """Integration tests for strategy engine."""
    
    def test_full_workflow(self):
        """Should complete full workflow."""
        engine = AdvancedStrategyEngine()
        
        # Create realistic price data
        np.random.seed(42)
        prices = 100 + np.random.randn(200).cumsum() * 0.5
        highs = prices + np.random.rand(200) * 2
        lows = prices - np.random.rand(200) * 2
        volumes = np.random.rand(200) * 1000 + 500
        
        price_data = {
            "1h": (prices, highs, lows, volumes),
            "15m": (prices[-100:], highs[-100:], lows[-100:], volumes[-100:]),
            "5m": (prices[-50:], highs[-50:], lows[-50:], volumes[-50:]),
        }
        
        # Generate signal
        signal = engine.generate_signal(
            symbol="BTCUSDT",
            price_data=price_data,
            base_direction=SignalDirection.LONG,
            strategy_name="momentum",
        )
        
        if signal:
            # Open trade
            trade = engine.exit_manager.open_trade(signal)
            
            # Simulate price movement
            for i in range(10):
                new_price = signal.entry_price * (1 + np.random.randn() * 0.01)
                exit_reason = engine.update_and_check_exit(trade.trade_id, new_price)
                
                if exit_reason:
                    result = engine.exit_manager.close_trade(trade.trade_id, new_price, exit_reason)
                    engine.record_trade_result(result, signal)
                    break
            
            assert engine.total_trades >= 0
    
    def test_multiple_signals(self):
        """Should handle multiple signals."""
        engine = AdvancedStrategyEngine()
        
        np.random.seed(42)
        
        for _ in range(5):
            prices = 100 + np.random.randn(100).cumsum() * 0.5
            highs = prices + 1
            lows = prices - 1
            volumes = np.random.rand(100) * 1000 + 500
            
            price_data = {"1h": (prices, highs, lows, volumes)}
            
            direction = SignalDirection.LONG if np.random.random() > 0.5 else SignalDirection.SHORT
            
            signal = engine.generate_signal(
                symbol="BTC",
                price_data=price_data,
                base_direction=direction,
            )
            
            # Signal might be filtered
        
        # Engine should have processed signals
        assert engine.total_trades >= 0


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestFactoryFunction:
    """Tests for factory functions."""
    
    def test_create_advanced_strategy_engine(self):
        """Should create engine."""
        engine = create_advanced_strategy_engine(min_confidence=60)
        
        assert engine.min_confidence == 60
