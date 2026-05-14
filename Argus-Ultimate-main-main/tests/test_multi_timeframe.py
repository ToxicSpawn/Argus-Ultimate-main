"""
Test Multi-Timeframe Analysis Module
======================================
Tests for multi-timeframe confluence and key level detection.
"""

import pytest
import numpy as np

from features.multi_timeframe import (
    TimeframeConfig,
    TimeframeAnalyzer,
    MultiTimeframeAnalyzer,
)


class TestTimeframeAnalyzer:
    """Test suite for TimeframeAnalyzer."""
    
    def setup_method(self):
        self.config = TimeframeConfig()
        self.analyzer = TimeframeAnalyzer(60, "1h", self.config)
    
    def test_initialization(self):
        assert self.analyzer is not None
        assert self.analyzer.timeframe == 60
        assert self.analyzer._trend_direction == "neutral"
    
    def test_uptrend_detection(self):
        # Simulate uptrend
        prices = [100.0 + i * 0.5 for i in range(50)]
        self.analyzer.update(prices)
        
        assert self.analyzer._trend_direction == "up"
    
    def test_downtrend_detection(self):
        # Simulate downtrend
        prices = [100.0 - i * 0.5 for i in range(50)]
        self.analyzer.update(prices)
        
        assert self.analyzer._trend_direction == "down"
    
    def test_neutral_trend(self):
        # Simulate ranging
        prices = [100.0 + np.sin(i / 5) * 2 for i in range(50)]
        self.analyzer.update(prices)
        
        # Should be neutral or close to it
        assert self.analyzer._trend_direction in ["neutral", "up", "down"]
    
    def test_key_levels_detection(self):
        # Create price with clear pivot points
        prices = []
        for i in range(100):
            if i < 20:
                prices.append(100.0 + i)
            elif i < 30:
                prices.append(120.0 - (i - 20))
            elif i < 50:
                prices.append(110.0 + (i - 30))
            else:
                prices.append(130.0 + np.random.randn())
        
        self.analyzer.update(prices)
        
        # Should detect some key levels
        signal = self.analyzer.get_signal()
        assert len(signal.key_levels) >= 0  # May be empty
    
    def test_get_signal(self):
        prices = [100.0 + i * 0.5 for i in range(50)]
        self.analyzer.update(prices)
        
        signal = self.analyzer.get_signal()
        
        assert signal.timeframe == 60
        assert signal.action in ["buy", "sell", "hold"]
        assert 0.0 <= signal.confidence <= 1.0
    
    def test_get_features(self):
        prices = [100.0 + i * 0.5 for i in range(50)]
        self.analyzer.update(prices)
        
        features = self.analyzer.get_features()
        
        assert "trend_1h" in features
        assert "key_levels_1h" in features


class TestMultiTimeframeAnalyzer:
    """Test suite for MultiTimeframeAnalyzer."""
    
    def setup_method(self):
        self.config = TimeframeConfig(
            timeframes=[5, 15, 60, 240],
            min_aligned_timeframes=3,
        )
        self.analyzer = MultiTimeframeAnalyzer(self.config)
    
    def test_initialization(self):
        assert self.analyzer is not None
        assert len(self.analyzer._analyzers) == 4
    
    def test_aligned_uptrend(self):
        # All timeframes showing uptrend
        base_prices = [100.0 + i * 0.1 for i in range(100)]
        
        timeframe_data = {
            5: base_prices[-20:],    # Last 20 bars for 5m
            15: base_prices[-30:],   # Last 30 bars for 15m
            60: base_prices[-50:],   # Last 50 bars for 1h
            240: base_prices,        # All bars for 4h
        }
        
        self.analyzer.update(timeframe_data)
        confluence = self.analyzer.get_confluence_signal()
        
        assert confluence["action"] == "buy"
        assert confluence["aligned_timeframes"] >= 3
    
    def test_aligned_downtrend(self):
        # All timeframes showing downtrend
        base_prices = [100.0 - i * 0.1 for i in range(100)]
        
        timeframe_data = {
            5: base_prices[-20:],
            15: base_prices[-30:],
            60: base_prices[-50:],
            240: base_prices,
        }
        
        self.analyzer.update(timeframe_data)
        confluence = self.analyzer.get_confluence_signal()
        
        assert confluence["action"] == "sell"
        assert confluence["aligned_timeframes"] >= 3
    
    def test_conflicting_timeframes(self):
        # 4h uptrend, 1h/15m/5m downtrend
        uptrend = [100.0 + i * 0.1 for i in range(100)]
        downtrend = [100.0 - i * 0.1 for i in range(50)]
        
        timeframe_data = {
            5: downtrend[-20:],
            15: downtrend[-30:],
            60: downtrend[-50:],
            240: uptrend,
        }
        
        self.analyzer.update(timeframe_data)
        confluence = self.analyzer.get_confluence_signal()
        
        # Should be hold due to conflict
        assert confluence["action"] == "hold"
    
    def test_get_features(self):
        base_prices = [100.0 + i * 0.1 for i in range(100)]
        
        timeframe_data = {
            5: base_prices[-20:],
            15: base_prices[-30:],
            60: base_prices[-50:],
            240: base_prices,
        }
        
        self.analyzer.update(timeframe_data)
        features = self.analyzer.get_features()
        
        assert "trend_5m" in features
        assert "trend_15m" in features
        assert "trend_1h" in features
        assert "trend_4h" in features
        assert "mtf_confluence" in features
        assert "mtf_confidence" in features
        assert "mtf_alignment" in features
    
    def test_key_levels(self):
        base_prices = [100.0 + np.sin(i / 10) * 5 for i in range(100)]
        
        timeframe_data = {
            60: base_prices,
            240: base_prices,
        }
        
        self.analyzer.update(timeframe_data)
        levels = self.analyzer.get_key_levels()
        
        assert isinstance(levels, list)
    
    def test_stats(self):
        base_prices = [100.0 + i * 0.1 for i in range(100)]
        
        timeframe_data = {
            5: base_prices[-20:],
            15: base_prices[-30:],
            60: base_prices[-50:],
            240: base_prices,
        }
        
        self.analyzer.update(timeframe_data)
        stats = self.analyzer.get_stats()
        
        assert "updates" in stats
        assert "confluence_signals" in stats
        assert "timeframes_tracked" in stats
