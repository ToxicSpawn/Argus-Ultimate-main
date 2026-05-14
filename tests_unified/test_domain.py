"""
Tests for domain typed contracts: Signal, Order, Fill, BotState.
Covers Issue #21 acceptance criteria.
"""
from __future__ import annotations

import math
from decimal import Decimal

import pytest

from domain import Signal, Order, Fill, BotState


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

class TestSignal:
    def test_valid_signal(self) -> None:
        s = Signal(symbol="BTC/USD", side="buy", confidence=0.85,
                   strategy_id="rsi_reversal", timestamp=1_700_000_000.0)
        assert s.is_valid()

    def test_invalid_confidence_too_high(self) -> None:
        s = Signal(symbol="BTC/USD", side="buy", confidence=1.5,
                   strategy_id="x", timestamp=0.0)
        assert not s.is_valid()

    def test_invalid_empty_symbol(self) -> None:
        s = Signal(symbol="", side="sell", confidence=0.5,
                   strategy_id="x", timestamp=0.0)
        assert not s.is_valid()

    def test_frozen_immutable(self) -> None:
        s = Signal(symbol="ETH/USD", side="sell", confidence=0.7,
                   strategy_id="momentum", timestamp=0.0)
        with pytest.raises((AttributeError, TypeError)):
            s.confidence = 0.9  # type: ignore[misc]

    def test_optional_fields_have_defaults(self) -> None:
        s = Signal(symbol="SOL/USD", side="buy", confidence=0.6,
                   strategy_id="dca", timestamp=0.0)
        assert s.entry_price == 0.0
        assert s.reasoning == ""


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------

class TestOrder:
    def test_valid_market_order(self) -> None:
        o = Order(symbol="BTC/USD", side="buy",
                  quantity=Decimal("0.01"), order_type="market")
        o.validate()  # should not raise

    def test_valid_limit_order(self) -> None:
        o = Order(symbol="ETH/USD", side="sell",
                  quantity=Decimal("0.5"), order_type="limit",
                  price=Decimal("3200.00"))
        o.validate()

    def test_limit_order_missing_price_raises(self) -> None:
        o = Order(symbol="BTC/USD", side="buy",
                  quantity=Decimal("0.1"), order_type="limit")
        with pytest.raises(ValueError, match="price"):
            o.validate()

    def test_zero_quantity_raises(self) -> None:
        o = Order(symbol="BTC/USD", side="buy",
                  quantity=Decimal("0"), order_type="market")
        with pytest.raises(ValueError, match="quantity"):
            o.validate()

    def test_frozen_immutable(self) -> None:
        o = Order(symbol="BTC/USD", side="buy",
                  quantity=Decimal("1"), order_type="market")
        with pytest.raises((AttributeError, TypeError)):
            o.side = "sell"  # type: ignore[misc]

    def test_stop_limit_requires_both_prices(self) -> None:
        o = Order(symbol="BTC/USD", side="sell",
                  quantity=Decimal("0.01"), order_type="stop_limit",
                  price=Decimal("60000"))
        with pytest.raises(ValueError, match="stop_price"):
            o.validate()


# ---------------------------------------------------------------------------
# Fill
# ---------------------------------------------------------------------------

class TestFill:
    def _make_fill(self, side: str = "buy") -> Fill:
        return Fill(
            exchange_order_id="EX123",
            client_order_id="CLI456",
            symbol="BTC/USD",
            side=side,  # type: ignore[arg-type]
            filled_quantity=Decimal("0.01"),
            fill_price=Decimal("65000"),
            fee=Decimal("0.169"),
            fee_currency="USD",
            timestamp=1_700_000_000.0,
        )

    def test_notional_value(self) -> None:
        f = self._make_fill()
        assert f.notional_value == Decimal("650.00")

    def test_net_proceeds_buy_is_negative(self) -> None:
        f = self._make_fill(side="buy")
        assert f.net_proceeds < 0

    def test_net_proceeds_sell_is_positive(self) -> None:
        f = self._make_fill(side="sell")
        assert f.net_proceeds > 0

    def test_frozen_immutable(self) -> None:
        f = self._make_fill()
        with pytest.raises((AttributeError, TypeError)):
            f.fee = Decimal("0")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BotState
# ---------------------------------------------------------------------------

class TestBotState:
    def test_initial_win_rate_zero(self) -> None:
        state = BotState()
        assert state.win_rate == 0.0

    def test_win_rate_calculation(self) -> None:
        state = BotState(total_trades=10, winning_trades=7)
        assert math.isclose(state.win_rate, 0.7)

    def test_drawdown_pct_zero_when_no_peak(self) -> None:
        state = BotState()
        assert state.drawdown_pct == 0.0

    def test_drawdown_pct_calculation(self) -> None:
        state = BotState(
            equity_aud=Decimal("900"),
            peak_equity_aud=Decimal("1000"),
        )
        assert math.isclose(state.drawdown_pct, 0.10)

    def test_record_equity_updates_peak(self) -> None:
        state = BotState(equity_aud=Decimal("1000"), peak_equity_aud=Decimal("1000"))
        state.record_equity(Decimal("1100"))
        assert state.peak_equity_aud == Decimal("1100")

    def test_record_equity_appends_history(self) -> None:
        state = BotState()
        state.record_equity(Decimal("1000"))
        state.record_equity(Decimal("1050"))
        assert len(state.equity_history) == 2

    def test_equity_history_capped_at_500(self) -> None:
        state = BotState()
        for i in range(600):
            state.record_equity(Decimal(str(1000 + i)))
        assert len(state.equity_history) == 500
