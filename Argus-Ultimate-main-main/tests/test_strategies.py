"""
Tests for strategies/strategy_library_impl.py

Covers implemented strategies and verifies placeholder strategies
return None without raising exceptions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int = 100, trend: float = 0.001) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame for testing."""
    rng = np.random.default_rng(42)
    prices = np.cumprod(1 + rng.normal(trend, 0.01, n))
    return pd.DataFrame({
        "open": prices * 0.999,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": rng.uniform(100, 1000, n),
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _signal_keys():
    return {"symbol", "action", "confidence", "price", "source"}


# ---------------------------------------------------------------------------
# TrendFollowingStrategy
# ---------------------------------------------------------------------------

class TestTrendFollowingStrategy:
    def test_returns_signal_or_none_with_sufficient_data(self):
        from strategies.strategy_library_impl import TrendFollowingStrategy
        s = TrendFollowingStrategy()
        result = s.analyze({"symbol": "BTC/USD", "price": 50000.0, "ohlcv_df": _make_ohlcv(100)})
        assert result is None or isinstance(result, dict)

    def test_returns_none_with_insufficient_data(self):
        from strategies.strategy_library_impl import TrendFollowingStrategy
        s = TrendFollowingStrategy()
        result = s.analyze({"symbol": "BTC/USD", "price": 50000.0, "ohlcv_df": _make_ohlcv(5)})
        assert result is None

    def test_signal_has_required_keys(self):
        from strategies.strategy_library_impl import TrendFollowingStrategy
        s = TrendFollowingStrategy()
        result = s.analyze({"symbol": "BTC/USD", "price": 50000.0, "ohlcv_df": _make_ohlcv(100)})
        if result is not None:
            assert _signal_keys().issubset(result.keys())

    def test_action_is_valid(self):
        from strategies.strategy_library_impl import TrendFollowingStrategy
        s = TrendFollowingStrategy()
        result = s.analyze({"symbol": "BTC/USD", "price": 50000.0, "ohlcv_df": _make_ohlcv(100)})
        if result is not None:
            assert result["action"] in {"BUY", "SELL", "HOLD"}

    def test_confidence_in_range(self):
        from strategies.strategy_library_impl import TrendFollowingStrategy
        s = TrendFollowingStrategy()
        result = s.analyze({"symbol": "BTC/USD", "price": 50000.0, "ohlcv_df": _make_ohlcv(100)})
        if result is not None:
            assert 0.0 <= result["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# MeanReversionStrategy
# ---------------------------------------------------------------------------

class TestMeanReversionStrategy:
    def test_returns_signal_or_none(self):
        from strategies.strategy_library_impl import MeanReversionStrategy
        s = MeanReversionStrategy()
        result = s.analyze({"symbol": "ETH/USD", "price": 3000.0, "ohlcv_df": _make_ohlcv(50)})
        assert result is None or isinstance(result, dict)

    def test_signal_keys_present(self):
        from strategies.strategy_library_impl import MeanReversionStrategy
        s = MeanReversionStrategy()
        result = s.analyze({"symbol": "ETH/USD", "price": 3000.0, "ohlcv_df": _make_ohlcv(50)})
        if result is not None:
            assert _signal_keys().issubset(result.keys())

    def test_no_crash_on_minimal_data(self):
        from strategies.strategy_library_impl import MeanReversionStrategy
        s = MeanReversionStrategy()
        result = s.analyze({"symbol": "ETH/USD", "price": 3000.0, "ohlcv_df": _make_ohlcv(3)})
        assert result is None or isinstance(result, dict)


# ---------------------------------------------------------------------------
# MomentumStrategy
# ---------------------------------------------------------------------------

class TestMomentumStrategy:
    def test_returns_signal_or_none(self):
        from strategies.strategy_library_impl import MomentumStrategy
        s = MomentumStrategy()
        result = s.analyze({"symbol": "BTC/USD", "price": 50000.0, "ohlcv_df": _make_ohlcv(100)})
        assert result is None or isinstance(result, dict)

    def test_no_crash_on_empty_data(self):
        from strategies.strategy_library_impl import MomentumStrategy
        s = MomentumStrategy()
        result = s.analyze({"symbol": "BTC/USD", "price": 50000.0})
        assert result is None or isinstance(result, dict)


# ---------------------------------------------------------------------------
# RegimeSwitchingStrategy
# ---------------------------------------------------------------------------

class TestRegimeSwitchingStrategy:
    def test_returns_signal_or_none(self):
        from strategies.strategy_library_impl import RegimeSwitchingStrategy
        s = RegimeSwitchingStrategy()
        result = s.analyze({"symbol": "BTC/USD", "price": 50000.0, "ohlcv_df": _make_ohlcv(60)})
        assert result is None or isinstance(result, dict)


# ---------------------------------------------------------------------------
# Placeholder strategies: must return None without raising
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("StratClass", [
    "ArbitrageStrategy",
    "FactorInvestingStrategy",
    "CrossExchangeArbStrategy",
    "QuantumPortfolioRotationEliteStrategy",
    "QuantumArbitrageEliteStrategy",
])
def test_placeholder_strategies_return_none(StratClass: str):
    """Placeholder strategies must return None (not raise) for any valid input."""
    import importlib
    mod = importlib.import_module("strategies.strategy_library_impl")
    cls = getattr(mod, StratClass)
    instance = cls()
    result = instance.analyze({"symbol": "BTC/USD", "price": 50000.0})
    assert result is None, f"{StratClass}.analyze() should return None, got {result!r}"


# ---------------------------------------------------------------------------
# Strategy base contract
# ---------------------------------------------------------------------------

def test_all_strategies_importable():
    """All strategy classes in strategy_library_impl must be importable."""
    from strategies.strategy_library_impl import (
        TrendFollowingStrategy, MeanReversionStrategy, MomentumStrategy,
        RegimeSwitchingStrategy, ArbitrageStrategy, FactorInvestingStrategy,
        CrossExchangeArbStrategy, HighFreqGridStrategy, CandlestickPatternStrategy,
    )
    for cls in [
        TrendFollowingStrategy, MeanReversionStrategy, MomentumStrategy,
        RegimeSwitchingStrategy, ArbitrageStrategy, FactorInvestingStrategy,
        CrossExchangeArbStrategy, HighFreqGridStrategy, CandlestickPatternStrategy,
    ]:
        instance = cls()
        assert hasattr(instance, "analyze"), f"{cls.__name__} missing .analyze()"
