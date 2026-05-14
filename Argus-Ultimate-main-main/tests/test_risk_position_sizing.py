"""
Tests for risk management - position sizing module.
"""

from __future__ import annotations

import pytest
import numpy as np

from risk.position_sizing import (
    PositionSizer,
    SizingConfig,
    SizingMethod,
    kelly_position_size,
    volatility_adjusted_position_size,
)
from core.types import MarketRegime, RiskLevel, PositionSizeResult


class TestSizingConfig:
    """Tests for SizingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SizingConfig()

        assert config.fixed_risk_pct == 0.01
        assert config.max_position_pct == 0.10
        assert config.method == SizingMethod.DYNAMIC

    def test_custom_config(self):
        """Test custom configuration."""
        config = SizingConfig(
            fixed_risk_pct=0.02,
            max_position_pct=0.05,
            kelly_fraction=0.5,
        )

        assert config.fixed_risk_pct == 0.02
        assert config.max_position_pct == 0.05
        assert config.kelly_fraction == 0.5


class TestPositionSizer:
    """Tests for PositionSizer."""

    def test_sizer_creation(self):
        """Test creating position sizer."""
        sizer = PositionSizer()
        assert sizer is not None

    def test_fixed_fractional_sizing(self):
        """Test fixed fractional position sizing."""
        config = SizingConfig(
            method=SizingMethod.FIXED_FRACTIONAL,
            fixed_risk_pct=0.02,
        )
        sizer = PositionSizer(config)

        result = sizer.calculate_position_size(
            capital=10000.0,
            entry_price=50000.0,
            stop_loss=49000.0,  # 2% stop
        )

        assert isinstance(result, PositionSizeResult)
        assert result.quantity > 0
        assert "fixed_fractional" in result.method.lower()

    def test_volatility_adjusted_sizing(self):
        """Test volatility-adjusted position sizing."""
        config = SizingConfig(
            method=SizingMethod.VOLATILITY_ADJUSTED,
        )
        sizer = PositionSizer(config)

        result = sizer.calculate_position_size(
            capital=10000.0,
            entry_price=50000.0,
            stop_loss=49000.0,
            volatility=1500.0,  # ATR in price units
        )

        assert result.quantity > 0
        assert "volatility" in result.method.lower()

    def test_kelly_sizing(self):
        """Test Kelly criterion position sizing."""
        config = SizingConfig(
            method=SizingMethod.KELLY,
            kelly_fraction=0.25,
        )
        sizer = PositionSizer(config)

        # Record some trades to build history
        for _ in range(10):
            sizer.record_trade(100.0)  # wins
        for _ in range(5):
            sizer.record_trade(-80.0)  # losses

        result = sizer.calculate_position_size(
            capital=10000.0,
            entry_price=50000.0,
            stop_loss=49000.0,
        )

        assert result.quantity >= 0
        assert "kelly" in result.method.lower()

    def test_confidence_adjustment(self):
        """Test position size adjustment based on confidence."""
        # Use settings that won't hit position limits
        config = SizingConfig(
            max_position_pct=0.90,
            max_position_value_aud=500000.0,
        )
        sizer = PositionSizer(config)

        # High confidence
        result_high = sizer.calculate_position_size(
            capital=100000.0,
            entry_price=50000.0,
            stop_loss=49000.0,
            confidence=1.0,
        )

        # Low confidence
        result_low = sizer.calculate_position_size(
            capital=100000.0,
            entry_price=50000.0,
            stop_loss=49000.0,
            confidence=0.5,
        )

        # Lower confidence should result in smaller position
        assert result_low.quantity < result_high.quantity

    def test_regime_adjustment(self):
        """Test position size adjustment based on market regime."""
        sizer = PositionSizer()

        # Trending market (favorable)
        result_trend = sizer.calculate_position_size(
            capital=10000.0,
            entry_price=50000.0,
            stop_loss=49000.0,
            regime=MarketRegime.TREND_UP,
        )

        # High volatility (unfavorable)
        result_vol = sizer.calculate_position_size(
            capital=10000.0,
            entry_price=50000.0,
            stop_loss=49000.0,
            regime=MarketRegime.HIGH_VOL,
        )

        # High vol regime should result in smaller position
        assert result_vol.quantity <= result_trend.quantity

    def test_risk_level_adjustment(self):
        """Test position size adjustment based on risk level."""
        # Use settings that won't hit position limits
        config = SizingConfig(
            max_position_pct=0.90,
            max_position_value_aud=500000.0,
        )
        sizer = PositionSizer(config)

        # Low risk
        result_low = sizer.calculate_position_size(
            capital=100000.0,
            entry_price=50000.0,
            stop_loss=49000.0,
            risk_level=RiskLevel.LOW,
        )

        # High risk
        result_high = sizer.calculate_position_size(
            capital=100000.0,
            entry_price=50000.0,
            stop_loss=49000.0,
            risk_level=RiskLevel.HIGH,
        )

        # High risk level should result in smaller position
        assert result_high.quantity < result_low.quantity

    def test_max_position_limit(self):
        """Test that position size respects maximum limit."""
        config = SizingConfig(
            max_position_pct=0.10,  # 10% max
        )
        sizer = PositionSizer(config)

        result = sizer.calculate_position_size(
            capital=10000.0,
            entry_price=100.0,  # Low price would normally result in large position
            stop_loss=99.0,
        )

        # Position value should not exceed 10% of capital
        assert result.notional_aud <= 1000.0 * 1.01  # Allow small rounding

    def test_zero_stop_loss_distance(self):
        """Test handling of zero stop loss distance."""
        sizer = PositionSizer()

        result = sizer.calculate_position_size(
            capital=10000.0,
            entry_price=50000.0,
            stop_loss=50000.0,  # Same as entry
        )

        # Should handle gracefully (zero quantity)
        assert result.quantity == 0

    def test_position_result_fields(self):
        """Test that result contains all expected fields."""
        sizer = PositionSizer()

        result = sizer.calculate_position_size(
            capital=10000.0,
            entry_price=50000.0,
            stop_loss=49000.0,
        )

        assert hasattr(result, 'quantity')
        assert hasattr(result, 'notional_aud')
        assert hasattr(result, 'risk_amount')
        assert hasattr(result, 'risk_pct')
        assert hasattr(result, 'method')

    def test_consistency_across_calls(self):
        """Test that same inputs produce same outputs."""
        sizer = PositionSizer()

        result1 = sizer.calculate_position_size(
            capital=10000.0,
            entry_price=50000.0,
            stop_loss=49000.0,
            confidence=0.8,
        )

        result2 = sizer.calculate_position_size(
            capital=10000.0,
            entry_price=50000.0,
            stop_loss=49000.0,
            confidence=0.8,
        )

        assert result1.quantity == result2.quantity

    def test_record_trade(self):
        """Test recording trades for Kelly calculation."""
        sizer = PositionSizer()

        # Initially no trades
        assert sizer._total_trades == 0

        # Record some trades
        sizer.record_trade(100.0)  # win
        sizer.record_trade(-50.0)  # loss
        sizer.record_trade(75.0)   # win

        assert sizer._total_trades == 3
        assert sizer._winning_trades == 2
        assert sizer._total_win_pnl == 175.0
        assert sizer._total_loss_pnl == 50.0


class TestSizingMethod:
    """Tests for SizingMethod enum."""

    def test_sizing_methods_exist(self):
        """Test that expected sizing methods exist."""
        assert SizingMethod.FIXED_FRACTIONAL is not None
        assert SizingMethod.VOLATILITY_ADJUSTED is not None
        assert SizingMethod.KELLY is not None
        assert SizingMethod.DYNAMIC is not None


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_kelly_position_size(self):
        """Test Kelly position size calculation."""
        result = kelly_position_size(
            capital=10000.0,
            win_rate=0.55,
            win_loss_ratio=1.5,
            kelly_fraction=0.25,
        )

        assert result > 0
        assert result < 10000.0  # Less than full capital

    def test_kelly_position_size_negative_edge(self):
        """Test Kelly with no edge returns zero."""
        result = kelly_position_size(
            capital=10000.0,
            win_rate=0.30,  # Very low win rate
            win_loss_ratio=1.0,
            kelly_fraction=0.25,
        )

        assert result == 0.0

    def test_volatility_adjusted_position_size(self):
        """Test volatility-adjusted position size."""
        quantity = volatility_adjusted_position_size(
            capital=10000.0,
            risk_pct=0.02,
            entry_price=50000.0,
            stop_price=49000.0,
        )

        assert quantity > 0
        # Risk = quantity * (entry - stop) = 0.02 * 10000 = 200
        # quantity = 200 / 1000 = 0.2
        assert quantity == pytest.approx(0.2)

    def test_volatility_adjusted_with_atr(self):
        """Test volatility-adjusted with ATR."""
        quantity = volatility_adjusted_position_size(
            capital=10000.0,
            risk_pct=0.02,
            entry_price=50000.0,
            stop_price=49000.0,
            volatility=750.0,  # ATR
            atr_multiplier=2.0,
        )

        # Stop distance = max(750*2, 1000) = 1500
        # quantity = 200 / 1500 = 0.133...
        assert quantity > 0
        assert quantity < 0.2  # Less than without volatility
