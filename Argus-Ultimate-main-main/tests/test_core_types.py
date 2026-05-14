"""
Tests for core type definitions.
"""

from __future__ import annotations

import pytest
from datetime import datetime

from core.types import (
    Signal,
    SignalAction,
    MarketRegime,
    Position,
    Side,
    OrderRequest,
    OrderType,
    OrderResult,
    OrderStatus,
    Trade,
    Ticker,
    OHLCV,
    OrderBook,
    OrderBookLevel,
    RiskMetrics,
    RiskLevel,
    PositionSizeResult,
    RiskCheck,
    RegimeSnapshot,
    TradingConfig,
    ExecutionAlgo,
)


class TestSignal:
    """Tests for Signal dataclass."""

    def test_signal_creation(self):
        """Test basic signal creation."""
        signal = Signal(
            symbol="BTC/AUD",
            action=SignalAction.BUY,
            confidence=0.8,
            strength=0.7,
            entry_price=50000.0,
        )

        assert signal.symbol == "BTC/AUD"
        assert signal.action == SignalAction.BUY
        assert signal.confidence == 0.8
        assert signal.strength == 0.7
        assert signal.entry_price == 50000.0

    def test_signal_confidence_clamping(self):
        """Test that confidence is clamped to [0, 1]."""
        signal = Signal(
            symbol="BTC/AUD",
            action=SignalAction.BUY,
            confidence=1.5,  # Should be clamped to 1.0
            strength=0.5,
            entry_price=50000.0,
        )
        assert signal.confidence == 1.0

        signal2 = Signal(
            symbol="BTC/AUD",
            action=SignalAction.SELL,
            confidence=-0.5,  # Should be clamped to 0.0
            strength=0.5,
            entry_price=50000.0,
        )
        assert signal2.confidence == 0.0

    def test_signal_is_entry(self):
        """Test is_entry property."""
        buy_signal = Signal(
            symbol="BTC/AUD",
            action=SignalAction.BUY,
            confidence=0.8,
            strength=0.7,
            entry_price=50000.0,
        )
        assert buy_signal.is_entry is True

        hold_signal = Signal(
            symbol="BTC/AUD",
            action=SignalAction.HOLD,
            confidence=0.8,
            strength=0.7,
            entry_price=50000.0,
        )
        assert hold_signal.is_entry is False

    def test_signal_score(self):
        """Test score property."""
        signal = Signal(
            symbol="BTC/AUD",
            action=SignalAction.BUY,
            confidence=0.8,
            strength=0.5,
            entry_price=50000.0,
        )
        assert signal.score == pytest.approx(0.4)  # 0.8 * 0.5

    def test_signal_to_dict(self):
        """Test serialization to dict."""
        signal = Signal(
            symbol="BTC/AUD",
            action=SignalAction.BUY,
            confidence=0.8,
            strength=0.7,
            entry_price=50000.0,
            strategy_name="momentum",
        )
        d = signal.to_dict()

        assert d["symbol"] == "BTC/AUD"
        assert d["action"] == "BUY"
        assert d["confidence"] == 0.8
        assert d["strategy_name"] == "momentum"


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation(self):
        """Test basic position creation."""
        pos = Position(
            symbol="BTC/AUD",
            side=Side.BUY,
            quantity=0.1,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
            current_price=51000.0,
        )

        assert pos.symbol == "BTC/AUD"
        assert pos.side == Side.BUY
        assert pos.quantity == 0.1
        assert pos.entry_price == 50000.0

    def test_position_notional_value(self):
        """Test notional value calculation."""
        pos = Position(
            symbol="BTC/AUD",
            side=Side.BUY,
            quantity=0.1,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
            current_price=51000.0,
        )
        assert pos.notional_value == pytest.approx(5100.0)

    def test_position_unrealized_pnl_long(self):
        """Test unrealized P&L for long position."""
        pos = Position(
            symbol="BTC/AUD",
            side=Side.BUY,
            quantity=0.1,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
            current_price=51000.0,
        )
        # Long: (51000 - 50000) * 0.1 = 100
        assert pos.unrealized_pnl == pytest.approx(100.0)
        assert pos.is_profitable is True

    def test_position_unrealized_pnl_short(self):
        """Test unrealized P&L for short position."""
        pos = Position(
            symbol="BTC/AUD",
            side=Side.SELL,
            quantity=0.1,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
            current_price=49000.0,
        )
        # Short: (50000 - 49000) * 0.1 = 100
        assert pos.unrealized_pnl == pytest.approx(100.0)
        assert pos.is_profitable is True

    def test_position_stop_loss_trigger(self):
        """Test stop loss trigger detection."""
        pos = Position(
            symbol="BTC/AUD",
            side=Side.BUY,
            quantity=0.1,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
            current_price=49000.0,
            stop_loss=49500.0,
        )
        assert pos.should_stop_out() is True

        pos.current_price = 50000.0
        assert pos.should_stop_out() is False


class TestOrderBook:
    """Tests for OrderBook dataclass."""

    def test_order_book_mid_price(self):
        """Test mid-price calculation."""
        book = OrderBook(
            symbol="BTC/AUD",
            bids=[OrderBookLevel(price=49900.0, quantity=1.0)],
            asks=[OrderBookLevel(price=50100.0, quantity=1.0)],
            timestamp=datetime.utcnow(),
            exchange="kraken",
        )
        assert book.mid_price == pytest.approx(50000.0)

    def test_order_book_spread(self):
        """Test spread calculation."""
        book = OrderBook(
            symbol="BTC/AUD",
            bids=[OrderBookLevel(price=49900.0, quantity=1.0)],
            asks=[OrderBookLevel(price=50100.0, quantity=1.0)],
            timestamp=datetime.utcnow(),
            exchange="kraken",
        )
        assert book.spread == pytest.approx(200.0)

    def test_order_book_imbalance(self):
        """Test order book imbalance calculation."""
        book = OrderBook(
            symbol="BTC/AUD",
            bids=[
                OrderBookLevel(price=49900.0, quantity=10.0),
                OrderBookLevel(price=49800.0, quantity=5.0),
            ],
            asks=[
                OrderBookLevel(price=50100.0, quantity=3.0),
                OrderBookLevel(price=50200.0, quantity=2.0),
            ],
            timestamp=datetime.utcnow(),
            exchange="kraken",
        )
        # bid_depth = 15, ask_depth = 5, total = 20
        # imbalance = (15 - 5) / 20 = 0.5
        assert book.imbalance(levels=2) == pytest.approx(0.5)


class TestSideEnum:
    """Tests for Side enum."""

    def test_side_from_string(self):
        """Test parsing side from string."""
        assert Side.from_string("buy") == Side.BUY
        assert Side.from_string("BUY") == Side.BUY
        assert Side.from_string("long") == Side.BUY
        assert Side.from_string("sell") == Side.SELL
        assert Side.from_string("short") == Side.SELL

    def test_side_from_string_invalid(self):
        """Test invalid side string."""
        with pytest.raises(ValueError):
            Side.from_string("invalid")


class TestOHLCV:
    """Tests for OHLCV dataclass."""

    def test_ohlcv_properties(self):
        """Test OHLCV properties."""
        candle = OHLCV(
            timestamp=datetime.utcnow(),
            open=100.0,
            high=110.0,
            low=95.0,
            close=105.0,
            volume=1000.0,
        )

        assert candle.range == pytest.approx(15.0)  # high - low
        assert candle.body == pytest.approx(5.0)    # |close - open|
        assert candle.is_bullish is True            # close > open


class TestTicker:
    """Tests for Ticker dataclass."""

    def test_ticker_properties(self):
        """Test ticker properties."""
        ticker = Ticker(
            symbol="BTC/AUD",
            bid=49900.0,
            ask=50100.0,
            last=50000.0,
            volume_24h=1000.0,
            timestamp=datetime.utcnow(),
            exchange="kraken",
        )

        assert ticker.mid == pytest.approx(50000.0)
        assert ticker.spread == pytest.approx(200.0)
        # spread_bps = (200 / 50000) * 10000 = 40
        assert ticker.spread_bps == pytest.approx(40.0)


class TestTradingConfig:
    """Tests for TradingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TradingConfig()

        assert config.initial_capital_aud == 1000.0
        assert config.max_position_pct == 0.10
        assert config.max_daily_loss_pct == 0.05
        assert config.max_drawdown_pct == 0.15
        assert config.exchange == "kraken"
        assert config.dry_run is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = TradingConfig(
            initial_capital_aud=5000.0,
            max_position_pct=0.05,
            dry_run=False,
        )

        assert config.initial_capital_aud == 5000.0
        assert config.max_position_pct == 0.05
        assert config.dry_run is False
