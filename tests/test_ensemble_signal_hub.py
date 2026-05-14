"""
Tests for ml.ensemble_signal_hub.

All external source dependencies are replaced with lightweight mocks so the
tests run without network access, API keys, or heavy ML packages.
"""

from __future__ import annotations

import time
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ml.ensemble_signal_hub import (
    CACHE_TTL,
    EnsembleSignal,
    EnsembleSignalHub,
    _neutral_signal,
    _SIZE_MAX,
    _SIZE_MIN,
    _SIZE_NEUTRAL,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_fg_reading(value: int = 50):
    """Build a minimal FearGreedReading-like object."""
    r = MagicMock()
    r.value = value
    r.classification = "Neutral"
    return r


def _make_whale_signal(direction: str = "NEUTRAL", strength: float = 0.5):
    ws = MagicMock()
    ws.direction = direction
    ws.strength = strength
    return ws


def _make_alpha_score(composite: float = 0.0, signal: str = "NEUTRAL"):
    a = MagicMock()
    a.composite = composite
    a.signal = signal
    a.confidence = 0.8
    return a


def _make_llm_signal(direction: str = "NEUTRAL", confidence: float = 0.6):
    ls = MagicMock()
    ls.direction = direction
    ls.confidence = confidence
    # as_numeric is a property on the real object; MagicMock needs explicit value
    ls.as_numeric = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}[direction]
    return ls


def _make_vol_forecast(regime: str = "NORMAL"):
    vf = MagicMock()
    vf.regime = regime
    vf.forecast_vol_1d = 40.0
    return vf


def _make_funding_prediction(predicted_rate_pct: float = 0.0):
    fp = MagicMock()
    fp.predicted_rate_pct = predicted_rate_pct
    fp.direction = "NEUTRAL"
    return fp


def _make_sentiment_score(aggregate_score: float = 0.0):
    ss = MagicMock()
    ss.aggregate_score = aggregate_score
    return ss


def _prices(n: int = 50, start: float = 40000.0) -> list:
    return [start + i * 10.0 for i in range(n)]


# ---------------------------------------------------------------------------
# 1. Neutral default when no sources are available
# ---------------------------------------------------------------------------


def test_neutral_default_no_sources():
    """Hub with all sources set to None returns neutral signal."""
    hub = EnsembleSignalHub(
        fear_greed=None,
        llm=None,
        whale=None,
        news=None,
        alpha=None,
        vol=None,
        funding=None,
    )
    # Disable all enabled flags too
    hub._enabled = {k: False for k in hub._enabled}

    sig = hub.update("BTC/USD", _prices())
    assert sig.composite == 0.0
    assert sig.size_multiplier == _SIZE_NEUTRAL
    assert sig.regime_bias == "NEUTRAL"


def test_get_last_returns_neutral_before_update():
    """get_last() returns neutral default when no cached value exists."""
    hub = EnsembleSignalHub()
    sig = hub.get_last("ETH/USD")
    assert isinstance(sig, EnsembleSignal)
    assert sig.composite == 0.0
    assert sig.regime_bias == "NEUTRAL"


# ---------------------------------------------------------------------------
# 2. Single source weighting
# ---------------------------------------------------------------------------


def test_single_alpha_source_only():
    """When only alpha is enabled, composite equals alpha composite."""
    fg_mock = MagicMock()
    alpha_mock = MagicMock()
    alpha_mock.update = MagicMock()
    alpha_mock.score = MagicMock(return_value=_make_alpha_score(composite=0.6))

    hub = EnsembleSignalHub(
        fear_greed=None,
        llm=None,
        whale=None,
        news=None,
        alpha=alpha_mock,
        vol=None,
        funding=None,
        config={"enabled": {
            "fear_greed": False,
            "llm": False,
            "whale": False,
            "news": False,
            "alpha": True,
            "vol_regime": False,
            "funding": False,
            "chain_metrics": False,
        }},
    )
    sig = hub.update("BTC/USD", _prices())
    # With single source, minimum-agreement dampening halves the signal
    # (need 2+ sources in same direction for full strength): 0.6 * 0.5 = 0.3
    assert abs(sig.composite - 0.3) < 1e-6


def test_single_fear_greed_bullish():
    """Extreme fear (value=5) should produce a positive composite."""
    fg_mock = MagicMock()

    async def _mock_get():
        return _make_fg_reading(value=5)

    fg_mock.get = _mock_get

    hub = EnsembleSignalHub(
        fear_greed=fg_mock,
        llm=None,
        whale=None,
        news=None,
        alpha=None,
        vol=None,
        funding=None,
        config={"enabled": {
            "fear_greed": True,
            "llm": False,
            "whale": False,
            "news": False,
            "alpha": False,
            "vol_regime": False,
            "funding": False,
        }},
    )
    sig = hub.update("BTC/USD", _prices())
    assert sig.composite > 0.0, "Extreme fear should be contrarian bullish"


def test_single_fear_greed_bearish():
    """Extreme greed (value=95) should produce a negative composite."""
    fg_mock = MagicMock()

    async def _mock_get():
        return _make_fg_reading(value=95)

    fg_mock.get = _mock_get

    hub = EnsembleSignalHub(
        fear_greed=fg_mock,
        llm=None, whale=None, news=None, alpha=None, vol=None, funding=None,
        config={"enabled": {
            "fear_greed": True, "llm": False, "whale": False,
            "news": False, "alpha": False, "vol_regime": False, "funding": False,
        }},
    )
    sig = hub.update("BTC/USD", _prices())
    assert sig.composite < 0.0, "Extreme greed should be contrarian bearish"


# ---------------------------------------------------------------------------
# 3. Composite calculation with multiple sources
# ---------------------------------------------------------------------------


def test_composite_weighted_average():
    """Composite is a proper weighted average of contributing sources."""
    # Use two sources: alpha=+1.0 (w=0.30), fear_greed=+1.0 (w=0.15)
    # expected composite = (1.0*0.30 + 1.0*0.15) / (0.30+0.15) = 1.0
    alpha_mock = MagicMock()
    alpha_mock.update = MagicMock()
    alpha_mock.score = MagicMock(return_value=_make_alpha_score(composite=1.0))

    fg_mock = MagicMock()

    async def _mock_get():
        return _make_fg_reading(value=0)  # extreme fear → +1.0

    fg_mock.get = _mock_get

    hub = EnsembleSignalHub(
        fear_greed=fg_mock,
        llm=None, whale=None, news=None,
        alpha=alpha_mock, vol=None, funding=None,
        config={"enabled": {
            "fear_greed": True, "llm": False, "whale": False,
            "news": False, "alpha": True, "vol_regime": False, "funding": False,
            "chain_metrics": False,
        }},
    )
    sig = hub.update("BTC/USD", _prices())
    assert sig.composite == pytest.approx(1.0, abs=0.01)


def test_composite_opposing_signals_cancel():
    """Alpha bullish + whale bearish with equal effective weight → near neutral."""
    alpha_mock = MagicMock()
    alpha_mock.update = MagicMock()
    # alpha weight = 0.30
    alpha_mock.score = MagicMock(return_value=_make_alpha_score(composite=1.0))

    whale_mock = MagicMock()
    # whale weight = 0.15; direction BEARISH strength=1.0 → -1.0
    whale_mock.get_signal = MagicMock(
        return_value=_make_whale_signal(direction="BEARISH", strength=1.0)
    )

    hub = EnsembleSignalHub(
        fear_greed=None, llm=None,
        whale=whale_mock, news=None,
        alpha=alpha_mock, vol=None, funding=None,
        config={
            "weights": {"alpha": 1.0, "whale": 1.0},
            "enabled": {
                "fear_greed": False, "llm": False, "whale": True,
                "news": False, "alpha": True, "vol_regime": False, "funding": False,
            },
        },
    )
    sig = hub.update("BTC/USD", _prices())
    # Equal weights, opposite signals → near 0
    assert abs(sig.composite) < 0.05


# ---------------------------------------------------------------------------
# 4. Size multiplier boundaries
# ---------------------------------------------------------------------------


def test_size_multiplier_min_clamp():
    """Size multiplier never goes below _SIZE_MIN even with extreme vol."""
    from ml.ensemble_signal_hub import EnsembleSignalHub as Hub
    hub = Hub()
    sm = hub._compute_size_multiplier(0.0, "EXTREME")
    assert sm >= _SIZE_MIN


def test_size_multiplier_max_clamp():
    """Size multiplier never exceeds _SIZE_MAX."""
    from ml.ensemble_signal_hub import EnsembleSignalHub as Hub
    hub = Hub()
    sm = hub._compute_size_multiplier(1.0, "LOW")
    assert sm <= _SIZE_MAX


def test_size_multiplier_neutral_when_zero_composite():
    """Zero composite + normal vol → neutral size multiplier."""
    from ml.ensemble_signal_hub import EnsembleSignalHub as Hub
    hub = Hub()
    sm = hub._compute_size_multiplier(0.0, "NORMAL")
    assert sm == pytest.approx(_SIZE_NEUTRAL, abs=0.01)


def test_size_multiplier_strong_signal_increases_size():
    """Strong signal (|composite|=1) should give size multiplier > neutral."""
    from ml.ensemble_signal_hub import EnsembleSignalHub as Hub
    hub = Hub()
    sm = hub._compute_size_multiplier(0.9, "NORMAL")
    assert sm > _SIZE_NEUTRAL


# ---------------------------------------------------------------------------
# 5. Cache TTL
# ---------------------------------------------------------------------------


def test_cache_ttl_returns_stale_result():
    """update() returns cached result within TTL without re-computing."""
    alpha_mock = MagicMock()
    alpha_mock.update = MagicMock()
    call_count = {"n": 0}

    def _score(sym):
        call_count["n"] += 1
        return _make_alpha_score(composite=0.5)

    alpha_mock.score = _score

    hub = EnsembleSignalHub(
        fear_greed=None, llm=None, whale=None, news=None,
        alpha=alpha_mock, vol=None, funding=None,
        config={
            "cache_ttl": 60,
            "enabled": {
                "fear_greed": False, "llm": False, "whale": False,
                "news": False, "alpha": True, "vol_regime": False, "funding": False,
            },
        },
    )
    hub.update("BTC/USD", _prices())
    hub.update("BTC/USD", _prices())  # second call — should use cache

    assert call_count["n"] == 1, "score() should only be called once within TTL"


def test_cache_ttl_expiry_recomputes():
    """After cache TTL expires, update() re-fetches source data."""
    alpha_mock = MagicMock()
    alpha_mock.update = MagicMock()
    call_count = {"n": 0}

    def _score(sym):
        call_count["n"] += 1
        return _make_alpha_score(composite=0.5)

    alpha_mock.score = _score

    hub = EnsembleSignalHub(
        fear_greed=None, llm=None, whale=None, news=None,
        alpha=alpha_mock, vol=None, funding=None,
        config={
            "cache_ttl": 1,  # 1-second TTL
            "enabled": {
                "fear_greed": False, "llm": False, "whale": False,
                "news": False, "alpha": True, "vol_regime": False, "funding": False,
            },
        },
    )
    hub.update("BTC/USD", _prices())
    # Manually expire cache
    hub._cache["BTC/USD"] = (time.time() - 2, hub._cache["BTC/USD"][1])
    hub.update("BTC/USD", _prices())  # should recompute

    assert call_count["n"] == 2, "score() should be called again after TTL expiry"


# ---------------------------------------------------------------------------
# 6. Vol regime reducing size
# ---------------------------------------------------------------------------


def test_vol_extreme_reduces_size_to_05():
    """EXTREME vol regime caps size multiplier at 0.5."""
    from ml.ensemble_signal_hub import EnsembleSignalHub as Hub
    hub = Hub()
    sm = hub._compute_size_multiplier(0.8, "EXTREME")
    assert sm == pytest.approx(0.5, abs=0.01)


def test_vol_elevated_reduces_size_to_07():
    """ELEVATED vol regime caps size multiplier at 0.7."""
    from ml.ensemble_signal_hub import EnsembleSignalHub as Hub
    hub = Hub()
    sm = hub._compute_size_multiplier(0.8, "ELEVATED")
    assert sm == pytest.approx(0.7, abs=0.01)


def test_vol_source_reduces_size_in_full_update():
    """When VolatilityForecaster returns EXTREME regime, size_multiplier is capped."""
    vol_mock = MagicMock()
    vol_mock.update = MagicMock()
    vol_mock.forecast = MagicMock(return_value=_make_vol_forecast(regime="EXTREME"))

    alpha_mock = MagicMock()
    alpha_mock.update = MagicMock()
    alpha_mock.score = MagicMock(return_value=_make_alpha_score(composite=0.9))

    hub = EnsembleSignalHub(
        fear_greed=None, llm=None, whale=None, news=None,
        alpha=alpha_mock, vol=vol_mock, funding=None,
        config={"enabled": {
            "fear_greed": False, "llm": False, "whale": False,
            "news": False, "alpha": True, "vol_regime": True, "funding": False,
        }},
    )
    sig = hub.update("BTC/USD", _prices())
    assert sig.size_multiplier <= 0.5 + 1e-6


# ---------------------------------------------------------------------------
# 7. Snapshot method
# ---------------------------------------------------------------------------


def test_snapshot_returns_dict():
    """snapshot() returns a dict with expected top-level keys."""
    hub = EnsembleSignalHub()
    snap = hub.snapshot()
    assert "sources" in snap
    assert "cache" in snap
    assert "last_source_values" in snap


def test_snapshot_sources_keys():
    """snapshot()['sources'] contains entries for all eight sources."""
    hub = EnsembleSignalHub()
    keys = set(hub.snapshot()["sources"].keys())
    expected = {"fear_greed", "llm", "whale", "news", "alpha", "vol_regime", "funding", "chain_metrics"}
    assert keys == expected


def test_snapshot_after_update_contains_cache_entry():
    """After update(), snapshot includes the cached entry."""
    alpha_mock = MagicMock()
    alpha_mock.update = MagicMock()
    alpha_mock.score = MagicMock(return_value=_make_alpha_score(composite=0.3))

    hub = EnsembleSignalHub(
        fear_greed=None, llm=None, whale=None, news=None,
        alpha=alpha_mock, vol=None, funding=None,
        config={"enabled": {
            "fear_greed": False, "llm": False, "whale": False,
            "news": False, "alpha": True, "vol_regime": False, "funding": False,
        }},
    )
    hub.update("SOL/USD", _prices())
    snap = hub.snapshot()
    assert "SOL/USD" in snap["cache"]
    assert "composite" in snap["cache"]["SOL/USD"]


# ---------------------------------------------------------------------------
# 8. Update with multiple mock sources
# ---------------------------------------------------------------------------


def test_update_with_all_sources_no_exception():
    """Full update with all sources mocked should complete without error."""
    fg_mock = MagicMock()

    async def _fg_get():
        return _make_fg_reading(value=30)

    fg_mock.get = _fg_get

    llm_mock = MagicMock()

    async def _llm_generate(**kwargs):
        return _make_llm_signal(direction="BULLISH")

    llm_mock.generate_signal = _llm_generate

    whale_mock = MagicMock()
    whale_mock.get_signal = MagicMock(
        return_value=_make_whale_signal(direction="BULLISH", strength=0.7)
    )

    news_mock = MagicMock()

    async def _news_get(sym):
        return _make_sentiment_score(aggregate_score=0.4)

    news_mock.get_signal = _news_get

    alpha_mock = MagicMock()
    alpha_mock.update = MagicMock()
    alpha_mock.score = MagicMock(return_value=_make_alpha_score(composite=0.5))

    vol_mock = MagicMock()
    vol_mock.update = MagicMock()
    vol_mock.forecast = MagicMock(return_value=_make_vol_forecast(regime="NORMAL"))

    funding_mock = MagicMock()
    funding_mock.predict = MagicMock(
        return_value=_make_funding_prediction(predicted_rate_pct=0.01)
    )

    hub = EnsembleSignalHub(
        fear_greed=fg_mock,
        llm=llm_mock,
        whale=whale_mock,
        news=news_mock,
        alpha=alpha_mock,
        vol=vol_mock,
        funding=funding_mock,
    )
    sig = hub.update("BTC/USD", _prices(), regime="BULL_TREND")

    assert isinstance(sig, EnsembleSignal)
    assert -1.0 <= sig.composite <= 1.0
    assert 0.0 <= sig.confidence <= 1.0
    assert _SIZE_MIN <= sig.size_multiplier <= _SIZE_MAX
    assert sig.regime_bias in ("BULLISH", "BEARISH", "NEUTRAL")


def test_update_failing_source_skipped_gracefully():
    """A source that raises an exception is skipped; other sources still contribute."""
    alpha_mock = MagicMock()
    alpha_mock.update = MagicMock()
    alpha_mock.score = MagicMock(return_value=_make_alpha_score(composite=0.6))

    whale_mock = MagicMock()
    whale_mock.get_signal = MagicMock(side_effect=RuntimeError("network failure"))

    hub = EnsembleSignalHub(
        fear_greed=None, llm=None,
        whale=whale_mock, news=None,
        alpha=alpha_mock, vol=None, funding=None,
        config={"enabled": {
            "fear_greed": False, "llm": False, "whale": True,
            "news": False, "alpha": True, "vol_regime": False, "funding": False,
        }},
    )
    sig = hub.update("BTC/USD", _prices())
    # Alpha should still contribute
    assert sig.composite != 0.0


# ---------------------------------------------------------------------------
# 9. Regime bias label
# ---------------------------------------------------------------------------


def test_regime_bias_bullish():
    from ml.ensemble_signal_hub import EnsembleSignalHub as Hub
    hub = Hub()
    assert hub._label(0.5) == "BULLISH"


def test_regime_bias_bearish():
    from ml.ensemble_signal_hub import EnsembleSignalHub as Hub
    hub = Hub()
    assert hub._label(-0.5) == "BEARISH"


def test_regime_bias_neutral():
    from ml.ensemble_signal_hub import EnsembleSignalHub as Hub
    hub = Hub()
    assert hub._label(0.1) == "NEUTRAL"
    assert hub._label(-0.1) == "NEUTRAL"


# ---------------------------------------------------------------------------
# 10. Config-driven weights and enabled flags
# ---------------------------------------------------------------------------


def test_config_overrides_weights():
    """Custom weights from config are applied correctly."""
    alpha_mock = MagicMock()
    alpha_mock.update = MagicMock()
    alpha_mock.score = MagicMock(return_value=_make_alpha_score(composite=1.0))

    hub = EnsembleSignalHub(
        fear_greed=None, llm=None, whale=None, news=None,
        alpha=alpha_mock, vol=None, funding=None,
        config={
            "weights": {"alpha": 0.99},
            "enabled": {
                "fear_greed": False, "llm": False, "whale": False,
                "news": False, "alpha": True, "vol_regime": False, "funding": False,
            },
        },
    )
    assert hub._weights["alpha"] == pytest.approx(0.99)


def test_config_disabled_source_not_called():
    """A disabled source should not be called even if instance is present."""
    whale_mock = MagicMock()
    whale_mock.get_signal = MagicMock(
        return_value=_make_whale_signal(direction="BEARISH")
    )

    hub = EnsembleSignalHub(
        fear_greed=None, llm=None,
        whale=whale_mock, news=None, alpha=None, vol=None, funding=None,
        config={"enabled": {
            "fear_greed": False, "llm": False, "whale": False,  # disabled
            "news": False, "alpha": False, "vol_regime": False, "funding": False,
        }},
    )
    hub.update("BTC/USD", _prices())
    whale_mock.get_signal.assert_not_called()
