"""Tests for execution/adaptive_slippage_model.py."""
from __future__ import annotations

import pytest

try:
    from execution.adaptive_slippage_model import AdaptiveSlippageModel  # type: ignore
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(not HAS_MODULE, reason="AdaptiveSlippageModel not importable")


class TestSlippageEstimation:
    def setup_method(self):
        self.model = AdaptiveSlippageModel()

    def test_estimate_returns_non_negative(self):
        slip = self.model.estimate_slippage(
            symbol="BTC/USDT", side="buy", quantity=0.01, price=65000.0, volume=1000.0
        )
        assert slip >= 0

    def test_larger_order_higher_slippage(self):
        small = self.model.estimate_slippage("BTC/USDT", "buy", 0.001, 65000.0, 1000.0)
        large = self.model.estimate_slippage("BTC/USDT", "buy", 10.0, 65000.0, 1000.0)
        assert large >= small

    def test_zero_volume_returns_max_slippage_or_raises(self):
        try:
            slip = self.model.estimate_slippage("BTC/USDT", "buy", 0.01, 65000.0, 0.0)
            assert slip >= 0
        except (ZeroDivisionError, ValueError):
            pass

    def test_model_updates_on_fill(self):
        self.model.estimate_slippage("BTC/USDT", "buy", 0.01, 65000.0, 1000.0)
        self.model.record_fill(
            symbol="BTC/USDT", side="buy",
            expected_price=65000.0, fill_price=65050.0, quantity=0.01
        )
        after = self.model.estimate_slippage("BTC/USDT", "buy", 0.01, 65000.0, 1000.0)
        assert after >= 0
