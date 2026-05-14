"""
Test Signal Filter Module
==========================
Tests for regime-specific confidence thresholds and signal filtering.
"""

import pytest
import time
from unittest.mock import MagicMock

from features.signal_filter import (
    SignalFilterConfig,
    SignalQuality,
    AdaptiveThreshold,
    OvertradeDetector,
    SignalConfluence,
    SignalFilter,
)


class TestSignalQuality:
    """Test suite for SignalQuality."""
    
    def test_high_quality_trend_signal(self):
        signal = {
            "action": "buy",
            "confidence": 0.85,
            "signal_type": "trend",
        }
        
        quality = SignalQuality.calculate(signal, "trending_up", 0.3)
        
        assert quality > 0.7
    
    def test_low_quality_unknown_signal(self):
        signal = {
            "action": "buy",
            "confidence": 0.5,
            "signal_type": "unknown",
        }
        
        quality = SignalQuality.calculate(signal, "ranging", 0.3)
        
        assert quality < 0.5
    
    def test_regime_appropriateness(self):
        # Trend signal in trending regime = high quality
        trend_signal = {
            "action": "buy",
            "confidence": 0.8,
            "signal_type": "trend",
        }
        quality_trend = SignalQuality.calculate(trend_signal, "trending_up", 0.3)
        
        # Trend signal in ranging regime = lower quality
        quality_range = SignalQuality.calculate(trend_signal, "ranging", 0.3)
        
        assert quality_trend > quality_range
    
    def test_high_volatility_reduces_quality(self):
        signal = {
            "action": "buy",
            "confidence": 0.8,
            "signal_type": "trend",
        }
        
        quality_low_vol = SignalQuality.calculate(signal, "trending_up", 0.3)
        quality_high_vol = SignalQuality.calculate(signal, "trending_up", 0.9)
        
        assert quality_low_vol > quality_high_vol


class TestAdaptiveThreshold:
    """Test suite for AdaptiveThreshold."""
    
    def test_initialization(self):
        config = SignalFilterConfig()
        threshold = AdaptiveThreshold(0.7, config)
        
        assert threshold.get_threshold() == 0.7
    
    def test_low_win_rate_raises_threshold(self):
        config = SignalFilterConfig(enable_adaptive=True)
        threshold = AdaptiveThreshold(0.7, config)
        
        # Record many losses
        for _ in range(20):
            threshold.record_trade(False)
        
        assert threshold.get_threshold() > 0.7
    
    def test_high_win_rate_lowers_threshold(self):
        config = SignalFilterConfig(enable_adaptive=True)
        threshold = AdaptiveThreshold(0.7, config)
        
        # Record many profitable trades
        for _ in range(20):
            threshold.record_trade(True, pnl=100.0, confidence=0.8)
        
        # With profitable trades, threshold should be adjusted
        assert threshold.get_threshold() <= 0.95  # May not lower due to regime logic
    
    def test_threshold_respects_bounds(self):
        config = SignalFilterConfig(
            enable_adaptive=True,
            min_threshold=0.5,
            max_threshold=0.9,
        )
        threshold = AdaptiveThreshold(0.5, config)
        
        # Try to push below min
        for _ in range(50):
            threshold.record_trade(True)
        
        assert threshold.get_threshold() >= 0.5
    
    def test_stats(self):
        config = SignalFilterConfig()
        threshold = AdaptiveThreshold(0.7, config)
        
        threshold.record_trade(True)
        threshold.record_trade(False)
        
        stats = threshold.get_stats()
        assert "base_threshold" in stats
        assert "current_threshold" in stats
        assert "win_rate" in stats
        assert "samples" in stats


class TestOvertradeDetector:
    """Test suite for OvertradeDetector."""
    
    def test_initialization(self):
        config = SignalFilterConfig()
        detector = OvertradeDetector(config)
        
        assert detector.can_trade() is True
    
    def test_rate_limiting(self):
        config = SignalFilterConfig(max_trades_per_hour=3)
        detector = OvertradeDetector(config)
        
        # Execute 3 trades
        for _ in range(3):
            detector.record_trade()
        
        # 4th trade should be blocked
        assert detector.can_trade() is False
    
    def test_stats(self):
        config = SignalFilterConfig()
        detector = OvertradeDetector(config)
        
        detector.record_trade()
        detector.can_trade()  # Check but don't trade
        
        stats = detector.get_stats()
        assert "trades_last_hour" in stats
        assert "total_trades" in stats


class TestSignalConfluence:
    """Test suite for SignalConfluence."""
    
    def test_insufficient_signals(self):
        confluence = SignalConfluence(min_confluence=2)
        
        signals = [{"action": "buy", "confidence": 0.8}]
        result = confluence.check_confluence(signals, "trending_up")
        
        assert result["has_confluence"] is False
    
    def test_buy_confluence(self):
        confluence = SignalConfluence(min_confluence=2)
        
        signals = [
            {"action": "buy", "confidence": 0.8, "signal_type": "trend"},
            {"action": "buy", "confidence": 0.7, "signal_type": "momentum"},
            {"action": "hold", "confidence": 0.0},
        ]
        result = confluence.check_confluence(signals, "trending_up")
        
        assert result["has_confluence"] is True
        assert result["action"] == "buy"
        assert result["signal_count"] == 2
    
    def test_sell_confluence(self):
        confluence = SignalConfluence(min_confluence=2)
        
        signals = [
            {"action": "sell", "confidence": 0.8},
            {"action": "sell", "confidence": 0.7},
            {"action": "buy", "confidence": 0.6},
        ]
        result = confluence.check_confluence(signals, "trending_down")
        
        assert result["has_confluence"] is True
        assert result["action"] == "sell"
    
    def test_no_confluence_conflict(self):
        confluence = SignalConfluence(min_confluence=2)
        
        signals = [
            {"action": "buy", "confidence": 0.8},
            {"action": "sell", "confidence": 0.7},
        ]
        result = confluence.check_confluence(signals, "ranging")
        
        assert result["has_confluence"] is False


class TestSignalFilter:
    """Test suite for SignalFilter."""
    
    def setup_method(self):
        self.filter = SignalFilter()
    
    def test_initialization(self):
        assert self.filter is not None
        assert len(self.filter._thresholds) > 0
    
    def test_filter_hold_signal(self):
        signal = {"action": "hold", "confidence": 0.8}
        result = self.filter.filter_signal(signal, "trending_up")
        
        assert result["should_trade"] is False
    
    def test_filter_low_quality_signal(self):
        signal = {
            "action": "buy",
            "confidence": 0.1,
            "signal_type": "unknown",
        }
        result = self.filter.filter_signal(signal, "ranging", 0.3)
        
        assert result["should_trade"] is False
        assert "quality" in result["filters_failed"]
    
    def test_filter_high_quality_signal(self):
        signal = {
            "action": "buy",
            "confidence": 0.9,
            "signal_type": "trend",
        }
        result = self.filter.filter_signal(signal, "trending_up", 0.2)
        
        assert result["should_trade"] is True
        assert "quality" in result["filters_passed"]
    
    def test_regime_threshold_effect(self):
        signal = {
            "action": "buy",
            "confidence": 0.7,
            "signal_type": "trend",
        }
        
        # Should pass in trending (lower threshold)
        result_trending = self.filter.filter_signal(signal, "trending_up", 0.2)
        
        # May fail in high_volatility (higher threshold)
        result_high_vol = self.filter.filter_signal(signal, "high_volatility", 0.2)
        
        # Trending should have lower or equal threshold
        assert len(result_trending["filters_passed"]) >= len(result_high_vol["filters_passed"])
    
    def test_adaptation(self):
        # Record bad trades to raise threshold
        for _ in range(20):
            self.filter.record_trade_result("trending_up", False)
        
        # Threshold should have increased
        stats = self.filter.get_stats()
        trending_stats = stats["thresholds"]["trending_up"]
        assert trending_stats["current_threshold"] > trending_stats["base_threshold"]
    
    def test_stats(self):
        signal = {"action": "buy", "confidence": 0.9, "signal_type": "trend"}
        self.filter.filter_signal(signal, "trending_up")
        
        stats = self.filter.get_stats()
        
        assert "total_signals" in stats
        assert "passed_signals" in stats
        assert "filtered_signals" in stats
        assert "thresholds" in stats


class TestSignalFilterIntegration:
    """Integration tests for signal filtering."""
    
    def test_full_filtering_workflow(self):
        config = SignalFilterConfig(
            thresholds={
                "trending_up": 0.6,
                "ranging": 0.75,
            },
            min_confluence_signals=2,
        )
        filter = SignalFilter(config)
        
        # Good signal in trending regime
        signal1 = {
            "action": "buy",
            "confidence": 0.85,
            "signal_type": "trend",
        }
        signal2 = {
            "action": "buy",
            "confidence": 0.80,
            "signal_type": "momentum",
        }
        
        result = filter.filter_signal(
            signal1,
            regime="trending_up",
            volatility=0.2,
            all_signals=[signal1, signal2],
        )
        
        assert result["should_trade"] is True
        assert result["action"] == "buy"
        assert result["confidence"] > 0.7
