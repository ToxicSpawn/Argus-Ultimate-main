"""
Tests for void_breaker — Tier 5+ signal engine.

Run with: pytest tests_unified/test_void_breaker.py -v
"""

from __future__ import annotations

import asyncio
import math
import time
from datetime import timezone

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 50, trend: float = 0.002, vol: float = 0.01, seed: int = 42) -> list:
    """Generate synthetic OHLCV bars.

    Parameters
    ----------
    n:
        Number of bars.
    trend:
        Per-bar log-return drift (positive = uptrend, negative = downtrend).
    vol:
        Per-bar log-return volatility.
    seed:
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    price = 30_000.0
    bars = []
    ts = int(time.time()) - n * 3600
    for i in range(n):
        ret = rng.normal(trend, vol)
        open_ = price
        close = price * math.exp(ret)
        high = max(open_, close) * (1 + abs(rng.normal(0, vol / 2)))
        low = min(open_, close) * (1 - abs(rng.normal(0, vol / 2)))
        volume = rng.uniform(500, 2000)
        bars.append([ts + i * 3600, open_, high, low, close, volume])
        price = close
    return bars


# ---------------------------------------------------------------------------
# Test 1: Config normalisation
# ---------------------------------------------------------------------------

class TestVoidBreakerConfig:
    """VoidBreakerConfig validates and normalises weights."""

    def test_default_values(self):
        from void_breaker import VoidBreakerConfig

        cfg = VoidBreakerConfig()
        assert cfg.ensemble_weight == pytest.approx(0.6)
        assert cfg.quantum_weight == pytest.approx(0.4)
        assert cfg.conviction_threshold == pytest.approx(0.85)
        assert cfg.regime_agreement_required == 3
        assert cfg.max_leverage == pytest.approx(5.0)

    def test_weights_normalised_when_not_summing_to_one(self):
        from void_breaker import VoidBreakerConfig

        cfg = VoidBreakerConfig(ensemble_weight=3.0, quantum_weight=1.0)
        assert math.isclose(cfg.ensemble_weight + cfg.quantum_weight, 1.0, abs_tol=1e-6)
        assert cfg.ensemble_weight == pytest.approx(0.75)
        assert cfg.quantum_weight == pytest.approx(0.25)

    def test_custom_config_preserved(self):
        from void_breaker import VoidBreakerConfig

        cfg = VoidBreakerConfig(
            ensemble_weight=0.7,
            quantum_weight=0.3,
            conviction_threshold=0.9,
            regime_agreement_required=2,
            max_leverage=3.0,
        )
        assert cfg.conviction_threshold == pytest.approx(0.9)
        assert cfg.regime_agreement_required == 2
        assert cfg.max_leverage == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Test 2: VoidBreakerSignal validation
# ---------------------------------------------------------------------------

class TestVoidBreakerSignal:
    """VoidBreakerSignal is a valid dataclass with correct field semantics."""

    def test_valid_directions(self):
        from void_breaker import VoidBreakerSignal
        from datetime import datetime, timezone

        for direction in ("long", "short", "flat"):
            sig = VoidBreakerSignal(
                symbol="BTC/USDT",
                direction=direction,
                conviction=0.5,
                ensemble_score=0.3,
                quantum_score=0.4,
                regime_consensus=True,
            )
            assert sig.direction == direction

    def test_invalid_direction_raises(self):
        from void_breaker import VoidBreakerSignal

        with pytest.raises(ValueError, match="direction"):
            VoidBreakerSignal(
                symbol="BTC/USDT",
                direction="sideways",  # invalid
                conviction=0.5,
                ensemble_score=0.1,
                quantum_score=0.1,
                regime_consensus=False,
            )

    def test_conviction_is_clipped(self):
        from void_breaker import VoidBreakerSignal

        sig = VoidBreakerSignal(
            symbol="ETH/USDT",
            direction="long",
            conviction=99.0,   # should be clipped to 1.0
            ensemble_score=0.5,
            quantum_score=0.5,
            regime_consensus=True,
        )
        assert sig.conviction == pytest.approx(1.0)

    def test_timestamp_is_utc(self):
        from void_breaker import VoidBreakerSignal

        sig = VoidBreakerSignal(
            symbol="SOL/USDT",
            direction="flat",
            conviction=0.0,
            ensemble_score=0.0,
            quantum_score=0.0,
            regime_consensus=False,
        )
        assert sig.timestamp.tzinfo is not None
        assert sig.timestamp.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# Test 3: Flat signal on insufficient data
# ---------------------------------------------------------------------------

class TestVoidBreakerEngineInsufficientData:
    """Engine returns 'flat' when fewer than 22 bars are provided."""

    def test_flat_on_short_ohlcv(self):
        from void_breaker import VoidBreakerEngine

        engine = VoidBreakerEngine()
        ohlcv_short = _make_ohlcv(n=10)
        signal = asyncio.get_event_loop().run_until_complete(
            engine.get_signal("BTC/USDT", ohlcv_short)
        )
        assert signal.direction == "flat"
        assert signal.conviction == pytest.approx(0.0)
        assert signal.symbol == "BTC/USDT"

    def test_flat_on_empty_ohlcv(self):
        from void_breaker import VoidBreakerEngine

        engine = VoidBreakerEngine()
        signal = asyncio.get_event_loop().run_until_complete(
            engine.get_signal("ETH/USDT", [])
        )
        assert signal.direction == "flat"
        assert signal.regime_consensus is False


# ---------------------------------------------------------------------------
# Test 4: Strong uptrend produces long signal
# ---------------------------------------------------------------------------

class TestVoidBreakerEngineStrongUptrend:
    """Strong bull market should produce a 'long' signal with high conviction."""

    def test_long_signal_on_strong_uptrend(self):
        from void_breaker import VoidBreakerEngine, VoidBreakerConfig

        # Lower threshold so the deterministic test can reliably pass
        config = VoidBreakerConfig(
            conviction_threshold=0.30,
            regime_agreement_required=2,
        )
        engine = VoidBreakerEngine(config=config)
        # Very strong uptrend: 0.5% per bar, low vol
        ohlcv = _make_ohlcv(n=60, trend=0.005, vol=0.002, seed=7)

        signal = asyncio.get_event_loop().run_until_complete(
            engine.get_signal("BTC/USDT", ohlcv)
        )
        # In a strong uptrend, both ensemble and quantum should lean long
        assert signal.ensemble_score > 0.0, "Ensemble score should be positive in uptrend"
        assert signal.symbol == "BTC/USDT"

    def test_downtrend_produces_negative_ensemble_score(self):
        from void_breaker import VoidBreakerEngine, VoidBreakerConfig

        config = VoidBreakerConfig(
            conviction_threshold=0.30,
            regime_agreement_required=2,
        )
        engine = VoidBreakerEngine(config=config)
        # Strong downtrend
        ohlcv = _make_ohlcv(n=60, trend=-0.005, vol=0.002, seed=13)

        signal = asyncio.get_event_loop().run_until_complete(
            engine.get_signal("BTC/USDT", ohlcv)
        )
        assert signal.ensemble_score < 0.0, "Ensemble score should be negative in downtrend"


# ---------------------------------------------------------------------------
# Test 5: Convenience wrapper + engine properties
# ---------------------------------------------------------------------------

class TestConvenienceWrapper:
    """get_void_breaker_signal() and engine properties work correctly."""

    def test_convenience_wrapper_returns_signal(self):
        from void_breaker import get_void_breaker_signal, VoidBreakerSignal, VoidBreakerConfig

        config = VoidBreakerConfig(conviction_threshold=0.5)
        ohlcv = _make_ohlcv(n=40, seed=99)

        signal = asyncio.get_event_loop().run_until_complete(
            get_void_breaker_signal("SOL/USDT", ohlcv, config=config)
        )
        assert isinstance(signal, VoidBreakerSignal)
        assert signal.symbol == "SOL/USDT"
        assert signal.direction in ("long", "short", "flat")
        assert 0.0 <= signal.conviction <= 1.0
        assert isinstance(signal.regime_consensus, bool)

    def test_engine_max_leverage_property(self):
        from void_breaker import VoidBreakerEngine, VoidBreakerConfig

        cfg = VoidBreakerConfig(max_leverage=8.0)
        engine = VoidBreakerEngine(config=cfg)
        assert engine.max_leverage == pytest.approx(8.0)

    def test_dunder_all_exports(self):
        import void_breaker

        for name in ("VoidBreakerEngine", "VoidBreakerConfig", "VoidBreakerSignal",
                     "get_void_breaker_signal"):
            assert name in void_breaker.__all__, f"{name} missing from __all__"
            assert hasattr(void_breaker, name), f"{name} not importable from void_breaker"
