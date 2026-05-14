"""Shared fixtures for tests_unified/."""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_exchange():
    ex = MagicMock()
    ex.id = "kraken"
    ex.fetch_ticker = AsyncMock(return_value={
        "symbol": "BTC/USDT", "last": 65000.0, "bid": 64990.0, "ask": 65010.0,
        "volume": 1234.5, "timestamp": 1700000000000,
    })
    ex.fetch_ohlcv = AsyncMock(return_value=[
        [1700000000000, 65000, 65100, 64900, 65050, 100.0],
    ])
    ex.create_order = AsyncMock(return_value={
        "id": "test-order-001", "status": "open", "symbol": "BTC/USDT",
        "side": "buy", "amount": 0.001, "price": 65000.0,
    })
    ex.cancel_order = AsyncMock(return_value={"id": "test-order-001", "status": "canceled"})
    ex.fetch_balance = AsyncMock(return_value={
        "USDT": {"free": 10000.0, "used": 0.0, "total": 10000.0},
        "BTC": {"free": 0.1, "used": 0.0, "total": 0.1},
    })
    return ex


@pytest.fixture
def mock_config():
    return {
        "exchange": {"id": "kraken", "paper": True, "rate_limit_rpm": 1200},
        "risk": {
            "max_position_pct": 0.05,
            "max_daily_loss": 0.02,
            "max_total_exposure": 0.8,
        },
        "execution": {"order_type": "limit", "slippage_tolerance": 0.001},
        "symbols": ["BTC/USDT", "ETH/USDT"],
    }


@pytest.fixture
def mock_db(tmp_path):
    return str(tmp_path / "test_argus.db")


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Batch 14 — M11: MockExchange and named exchange fixtures
# ---------------------------------------------------------------------------

class MockExchange:
    """Reusable fake ccxt exchange for unit and integration tests."""

    def __init__(self, exchange_id: str = "mock") -> None:
        self.id = exchange_id
        self.markets = {
            "BTC/USDT": {"symbol": "BTC/USDT", "base": "BTC", "quote": "USDT", "active": True},
            "ETH/USDT": {"symbol": "ETH/USDT", "base": "ETH", "quote": "USDT", "active": True},
        }

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 100, **kwargs):
        base_ts = 1_700_000_000_000
        return [
            [base_ts + i * 60_000, 65000.0 + i, 65100.0 + i, 64900.0 + i, 65050.0 + i, 100.0 + i]
            for i in range(min(limit, 100))
        ]

    async def fetch_ticker(self, symbol: str, **kwargs):
        return {
            "symbol": symbol,
            "last": 65000.0,
            "bid": 64990.0,
            "ask": 65010.0,
            "volume": 1234.5,
            "timestamp": 1_700_000_000_000,
        }

    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: float = None, **kwargs):
        return {
            "id": "mock-order-001",
            "status": "open",
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "amount": amount,
            "price": price or 65000.0,
            "filled": 0.0,
            "remaining": amount,
        }

    async def cancel_order(self, order_id: str, symbol: str = None, **kwargs):
        return {"id": order_id, "status": "canceled"}

    async def fetch_balance(self, **kwargs):
        return {
            "USDT": {"free": 10_000.0, "used": 0.0, "total": 10_000.0},
            "BTC":  {"free": 0.1,      "used": 0.0, "total": 0.1},
            "ETH":  {"free": 1.0,      "used": 0.0, "total": 1.0},
        }

    async def fetch_order_book(self, symbol: str, limit: int = 20, **kwargs):
        bids = [[65000.0 - i * 10, 0.5 + i * 0.1] for i in range(limit)]
        asks = [[65010.0 + i * 10, 0.5 + i * 0.1] for i in range(limit)]
        return {"symbol": symbol, "bids": bids, "asks": asks, "timestamp": 1_700_000_000_000}

    async def close(self):
        pass


@pytest.fixture
def mock_exchange_obj():
    """MockExchange instance (generic, id='mock')."""
    return MockExchange(exchange_id="mock")


@pytest.fixture
def mock_bybit():
    """Named Bybit mock exchange."""
    return MockExchange(exchange_id="bybit")


@pytest.fixture
def mock_kraken():
    """Named Kraken mock exchange."""
    return MockExchange(exchange_id="kraken")


@pytest.fixture
def paper_config():
    """Minimal config dict suitable for paper trading tests."""
    return {
        "mode": "paper",
        "paper_trading": {
            "enabled": True,
            "cycle_seconds": 1,
            "initial_capital": 10_000.0,
        },
        "exchange": {
            "id": "kraken",
            "paper": True,
            "sandbox": False,
            "rate_limit_rpm": 1200,
        },
        "risk": {
            "max_position_pct": 0.05,
            "max_daily_loss_pct": 0.02,
            "max_total_exposure": 0.80,
            "max_drawdown_pct": 15.0,
            "max_daily_loss_usd": 200.0,
        },
        "execution": {
            "order_type": "market",
            "slippage_tolerance": 0.001,
        },
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "logging": {"level": "WARNING"},
    }
