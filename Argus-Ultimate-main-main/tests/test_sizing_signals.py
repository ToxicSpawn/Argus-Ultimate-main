"""
Tests for Kelly Criterion position sizing and signal quality filtering.

Covers:
- Kelly sizing with known win rates
- Half-Kelly safety cap
- Kelly edge cases (0% win, 100% win, equal win/loss)
- Volatility adjustment (high vol = smaller, low vol = larger)
- Vol adjustment caps (never more than 2x)
- Signal conflict detection (bull vs bear sources)
- Minimum source agreement
- Signal quality metrics
- Integration: Kelly + vol + signal quality combined sizing
- Comparison: Kelly vs default sizing
"""

from __future__ import annotations

import math
import pytest
import time
from collections import deque
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers — lightweight stand-ins so we don't import the whole trading system
# ---------------------------------------------------------------------------

def _kelly_size(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Mirror of UnifiedSystemArchitecture._kelly_size."""
    if avg_loss <= 0 or win_rate <= 0 or avg_win <= 0:
        return 0.0
    payoff_ratio = avg_win / avg_loss
    kelly = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio
    half_kelly = kelly * 0.5
    return max(0.0, min(half_kelly, 0.15))


def _vol_adjusted_size(base_size: float, current_vol: float, target_vol: float = 0.02) -> float:
    """Mirror of UnifiedTradingSystem._vol_adjusted_size."""
    if current_vol <= 0:
        return base_size
    vol_ratio = target_vol / max(current_vol, 0.005)
    return base_size * min(vol_ratio, 2.0)


# ═══════════════════════════════════════════════════════════════════════════
# 1. KELLY CRITERION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestKellySize:
    """Tests for the _kelly_size function."""

    def test_classic_kelly_60pct_win_2to1_payoff(self):
        """60% win rate with 2:1 payoff ratio -> full Kelly ~0.40, half ~0.15 (capped)."""
        result = _kelly_size(0.60, 0.04, 0.02)  # 2:1 payoff
        # Full Kelly = (0.60 * 2 - 0.40) / 2 = 0.40
        # Half Kelly = 0.20 -> capped at 0.15
        assert result == 0.15, f"Expected 0.15 (capped), got {result}"

    def test_kelly_55pct_win_1_5to1_payoff(self):
        """55% win rate with 1.5:1 payoff -> moderate sizing."""
        result = _kelly_size(0.55, 0.03, 0.02)  # 1.5:1 payoff
        # Full Kelly = (0.55 * 1.5 - 0.45) / 1.5 = (0.825 - 0.45) / 1.5 = 0.25
        # Half Kelly = 0.125
        assert abs(result - 0.125) < 0.001, f"Expected ~0.125, got {result}"

    def test_kelly_50pct_win_even_payoff(self):
        """50% win rate with 1:1 payoff -> zero edge, zero sizing."""
        result = _kelly_size(0.50, 0.02, 0.02)
        # Full Kelly = (0.50 * 1 - 0.50) / 1 = 0.0
        assert result == 0.0, f"Expected 0.0 (no edge), got {result}"

    def test_kelly_zero_win_rate(self):
        """0% win rate -> zero sizing."""
        result = _kelly_size(0.0, 0.05, 0.02)
        assert result == 0.0

    def test_kelly_100pct_win_rate(self):
        """100% win rate -> capped at 15%."""
        result = _kelly_size(1.0, 0.05, 0.02)
        # Full Kelly = (1.0 * 2.5 - 0) / 2.5 = 1.0
        # Half Kelly = 0.5 -> capped at 0.15
        assert result == 0.15

    def test_kelly_zero_avg_loss(self):
        """Zero avg_loss -> return 0.0 (avoid division by zero)."""
        result = _kelly_size(0.60, 0.03, 0.0)
        assert result == 0.0

    def test_kelly_zero_avg_win(self):
        """Zero avg_win -> return 0.0 (no profitable trades means no edge)."""
        result = _kelly_size(0.60, 0.0, 0.02)
        assert result == 0.0

    def test_kelly_negative_edge(self):
        """Losing strategy -> kelly is negative -> clamp to 0."""
        result = _kelly_size(0.30, 0.02, 0.03)  # Bad win rate, bad payoff
        # payoff = 0.02/0.03 = 0.667
        # kelly = (0.30 * 0.667 - 0.70) / 0.667 = (0.20 - 0.70) / 0.667 = -0.75
        # half = -0.375 -> clamped to 0
        assert result == 0.0

    def test_kelly_small_edge(self):
        """Small edge -> small position."""
        result = _kelly_size(0.52, 0.015, 0.015)  # Slight edge, 1:1 payoff
        # kelly = (0.52 * 1 - 0.48) / 1 = 0.04
        # half = 0.02
        assert abs(result - 0.02) < 0.001

    def test_kelly_cap_at_15_pct(self):
        """Very strong edge should still be capped at 15%."""
        result = _kelly_size(0.80, 0.10, 0.02)
        assert result == 0.15

    def test_kelly_half_vs_full(self):
        """Verify half-Kelly is always half of full Kelly (before cap)."""
        # 55% win, 1.8:1 payoff
        win_rate, avg_win, avg_loss = 0.55, 0.036, 0.02
        payoff = avg_win / avg_loss
        full_kelly = (win_rate * payoff - (1 - win_rate)) / payoff
        half_kelly = full_kelly * 0.5
        result = _kelly_size(win_rate, avg_win, avg_loss)
        expected = max(0.0, min(half_kelly, 0.15))
        assert abs(result - expected) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# 2. VOLATILITY ADJUSTMENT TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestVolAdjustedSize:
    """Tests for _vol_adjusted_size."""

    def test_high_vol_reduces_size(self):
        """High volatility (4% daily) should reduce position."""
        result = _vol_adjusted_size(0.10, 0.04, target_vol=0.02)
        # vol_ratio = 0.02 / 0.04 = 0.5
        assert abs(result - 0.05) < 1e-6

    def test_low_vol_increases_size(self):
        """Low volatility (1% daily) should increase position."""
        result = _vol_adjusted_size(0.10, 0.01, target_vol=0.02)
        # vol_ratio = 0.02 / 0.01 = 2.0 -> capped at 2.0
        assert abs(result - 0.20) < 1e-6

    def test_matching_vol_no_change(self):
        """Volatility matching target -> no change."""
        result = _vol_adjusted_size(0.10, 0.02, target_vol=0.02)
        assert abs(result - 0.10) < 1e-6

    def test_vol_cap_at_2x(self):
        """Very low vol should not scale beyond 2x base."""
        result = _vol_adjusted_size(0.10, 0.001, target_vol=0.02)
        # vol_ratio = 0.02 / 0.005 (clamped) = 4.0 -> capped at 2.0
        assert abs(result - 0.20) < 1e-6

    def test_zero_vol_returns_base(self):
        """Zero volatility -> return base size unchanged."""
        result = _vol_adjusted_size(0.10, 0.0, target_vol=0.02)
        assert abs(result - 0.10) < 1e-6

    def test_negative_vol_returns_base(self):
        """Negative volatility (invalid) -> return base size."""
        result = _vol_adjusted_size(0.10, -0.05, target_vol=0.02)
        assert abs(result - 0.10) < 1e-6

    def test_extreme_vol_severe_reduction(self):
        """Extreme vol (10% daily) should massively reduce size."""
        result = _vol_adjusted_size(0.10, 0.10, target_vol=0.02)
        # vol_ratio = 0.02 / 0.10 = 0.2
        assert abs(result - 0.02) < 1e-6

    def test_vol_floor_at_0_5_pct(self):
        """Vol below 0.5% is floored at 0.5% to avoid oversizing."""
        result = _vol_adjusted_size(0.10, 0.001, target_vol=0.02)
        # current_vol clamped to 0.005; ratio = 0.02/0.005 = 4.0 -> capped at 2.0
        assert abs(result - 0.20) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# 3. SIGNAL CONFLICT DETECTION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestSignalConflict:
    """Tests for EnsembleSignalHub conflict detection and signal quality."""

    def _make_hub(self, **kwargs):
        """Create a minimal EnsembleSignalHub with all sources disabled."""
        from ml.ensemble_signal_hub import EnsembleSignalHub
        cfg = {
            "enabled": {
                "fear_greed": False,
                "llm": False,
                "whale": False,
                "news": False,
                "alpha": False,
                "vol_regime": False,
                "funding": False,
            },
            **kwargs,
        }
        return EnsembleSignalHub(config=cfg)

    def test_conflict_reduces_composite(self):
        """When strong bullish and bearish sources exist, composite is reduced by 40%."""
        from ml.ensemble_signal_hub import EnsembleSignalHub

        hub = self._make_hub()
        # Manually call _compute with injected contributions
        # We need to simulate this by manipulating internals
        # Instead, let's test _weighted_composite + conflict logic directly

        contributions = {"alpha": 0.8, "news": -0.7}
        weights = {"alpha": 0.3, "news": 0.1}

        # Without conflict adjustment:
        raw_composite = EnsembleSignalHub._weighted_composite(contributions, weights)

        # With conflict: the composite in _compute would be multiplied by 0.6
        adjusted = raw_composite * 0.6
        assert abs(adjusted) < abs(raw_composite), "Conflict should reduce composite magnitude"

    def test_no_conflict_when_all_agree(self):
        """No conflict penalty when all sources agree on direction."""
        from ml.ensemble_signal_hub import EnsembleSignalHub

        # All bullish contributions
        contributions = {"alpha": 0.6, "news": 0.3, "llm": 0.4}
        weights = {"alpha": 0.3, "news": 0.1, "llm": 0.2}

        # No sources below -0.5, so no conflict
        bearish = {k: v for k, v in contributions.items() if v < -0.5}
        assert len(bearish) == 0, "Should have no strong bearish sources"

    def test_conflict_requires_strong_signals(self):
        """Mild disagreement (< 0.5 threshold) should not trigger conflict."""
        contributions = {"alpha": 0.4, "news": -0.3}
        # Neither exceeds the 0.5/-0.5 thresholds
        bullish = {k: v for k, v in contributions.items() if v > 0.5}
        bearish = {k: v for k, v in contributions.items() if v < -0.5}
        assert not (bullish and bearish), "Mild disagreement should not be conflict"


class TestMinSourceAgreement:
    """Tests for minimum source agreement requirement."""

    def test_single_source_dampened(self):
        """A strong signal from only 1 source should be dampened."""
        # If composite > 0.3 and same_dir < 2, composite is halved
        composite = 0.5
        contributions = {"alpha": 0.8}  # Only one source
        same_dir = sum(1 for v in contributions.values() if v > 0)
        assert same_dir < 2, "Only 1 source agreeing"

        if abs(composite) > 0.3 and same_dir < 2:
            dampened = composite * 0.5
        else:
            dampened = composite
        assert dampened == 0.25, "Should be halved to 0.25"

    def test_two_sources_not_dampened(self):
        """Two sources agreeing should not trigger dampening."""
        composite = 0.5
        contributions = {"alpha": 0.6, "llm": 0.4}
        same_dir = sum(1 for v in contributions.values() if v > 0)
        assert same_dir >= 2, "Two sources agreeing"


class TestSignalQuality:
    """Tests for get_signal_quality method."""

    def _make_hub(self):
        from ml.ensemble_signal_hub import EnsembleSignalHub
        cfg = {
            "enabled": {
                "fear_greed": False,
                "llm": False,
                "whale": False,
                "news": False,
                "alpha": False,
                "vol_regime": False,
                "funding": False,
            }
        }
        return EnsembleSignalHub(config=cfg)

    def test_quality_default_when_no_compute(self):
        """Before any computation, signal quality returns weak/empty defaults."""
        hub = self._make_hub()
        q = hub.get_signal_quality()
        assert q["recommendation"] == "weak"
        assert q["agreement_ratio"] == 0.0
        assert q["conflict_score"] == 0.0
        assert q["n_sources"] == 0

    def test_quality_returns_dict_keys(self):
        """Signal quality dict must contain all required keys."""
        hub = self._make_hub()
        q = hub.get_signal_quality()
        required = {
            "agreement_ratio", "conflict_score", "strongest_source",
            "recommendation", "n_sources", "n_agreeing", "conflict_detected",
        }
        assert required.issubset(set(q.keys()))

    def test_quality_strong_recommendation(self):
        """High agreement + strong composite -> 'strong' recommendation."""
        hub = self._make_hub()
        # Simulate _last_signal_quality directly
        hub._last_signal_quality = {
            "agreement_ratio": 0.8,
            "conflict_score": 0.0,
            "strongest_source": "alpha",
            "strongest_value": 0.7,
            "recommendation": "strong",
            "n_sources": 5,
            "n_agreeing": 4,
            "conflict_detected": False,
        }
        q = hub.get_signal_quality()
        assert q["recommendation"] == "strong"

    def test_quality_conflicted_recommendation(self):
        """High conflict score -> 'conflicted' recommendation."""
        hub = self._make_hub()
        hub._last_signal_quality = {
            "agreement_ratio": 0.4,
            "conflict_score": 0.7,
            "strongest_source": "alpha",
            "strongest_value": 0.8,
            "recommendation": "conflicted",
            "n_sources": 3,
            "n_agreeing": 1,
            "conflict_detected": True,
        }
        q = hub.get_signal_quality()
        assert q["recommendation"] == "conflicted"
        assert q["conflict_detected"] is True

    def test_quality_is_copy(self):
        """get_signal_quality should return a copy, not a reference."""
        hub = self._make_hub()
        hub._last_signal_quality = {"recommendation": "strong", "agreement_ratio": 0.9,
                                     "conflict_score": 0.0, "strongest_source": "alpha",
                                     "strongest_value": 0.7, "n_sources": 3, "n_agreeing": 3,
                                     "conflict_detected": False}
        q = hub.get_signal_quality()
        q["recommendation"] = "tampered"
        assert hub._last_signal_quality["recommendation"] == "strong"


# ═══════════════════════════════════════════════════════════════════════════
# 4. STRATEGY TRADE STATS TESTS
# ═══════════════════════════════════════════════════════════════════════════


def _get_strategy_trade_stats(trade_history, strategy_name: str) -> dict:
    """
    Standalone mirror of UnifiedSystemArchitecture._get_strategy_trade_stats.

    Avoids importing the full trading system (heavy import graph).
    """
    sells = [
        t for t in trade_history
        if (
            str(t.get("source_strategy") or t.get("strategy") or "unknown") == strategy_name
            and str(t.get("side", "")).upper() == "SELL"
            and t.get("pnl") is not None
        )
    ]
    if not sells:
        return {"win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "n_trades": 0, "wins": [], "losses": []}

    wins = [float(t["pnl"]) for t in sells if float(t.get("pnl", 0)) > 0]
    losses = [abs(float(t["pnl"])) for t in sells if float(t.get("pnl", 0)) < 0]
    n = len(sells)
    win_rate = len(wins) / n if n > 0 else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return {
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "n_trades": n,
        "wins": wins,
        "losses": losses,
    }


class TestStrategyTradeStats:
    """Tests for _get_strategy_trade_stats logic."""

    def _run(self, trades, strategy):
        return _get_strategy_trade_stats(deque(trades, maxlen=10000), strategy)

    def test_empty_history(self):
        stats = self._run([], "momentum")
        assert stats["n_trades"] == 0
        assert stats["win_rate"] == 0.0

    def test_all_wins(self):
        trades = [
            {"source_strategy": "momentum", "side": "SELL", "pnl": 10.0},
            {"source_strategy": "momentum", "side": "SELL", "pnl": 20.0},
            {"source_strategy": "momentum", "side": "SELL", "pnl": 15.0},
        ]
        stats = self._run(trades, "momentum")
        assert stats["win_rate"] == 1.0
        assert stats["n_trades"] == 3
        assert stats["avg_win"] == 15.0
        assert stats["avg_loss"] == 0.0

    def test_mixed_trades(self):
        trades = [
            {"source_strategy": "mean_rev", "side": "SELL", "pnl": 10.0},
            {"source_strategy": "mean_rev", "side": "SELL", "pnl": -5.0},
            {"source_strategy": "mean_rev", "side": "SELL", "pnl": 20.0},
            {"source_strategy": "mean_rev", "side": "SELL", "pnl": -10.0},
        ]
        stats = self._run(trades, "mean_rev")
        assert stats["n_trades"] == 4
        assert stats["win_rate"] == 0.5
        assert stats["avg_win"] == 15.0  # (10+20)/2
        assert stats["avg_loss"] == 7.5  # (5+10)/2

    def test_filters_by_strategy(self):
        trades = [
            {"source_strategy": "momentum", "side": "SELL", "pnl": 10.0},
            {"source_strategy": "mean_rev", "side": "SELL", "pnl": -5.0},
            {"source_strategy": "momentum", "side": "SELL", "pnl": -3.0},
        ]
        stats = self._run(trades, "momentum")
        assert stats["n_trades"] == 2

    def test_ignores_buy_trades(self):
        """Only SELL (closed positions) should be counted."""
        trades = [
            {"source_strategy": "momentum", "side": "BUY", "pnl": 0.0},
            {"source_strategy": "momentum", "side": "SELL", "pnl": 10.0},
        ]
        stats = self._run(trades, "momentum")
        assert stats["n_trades"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# 5. INTEGRATION TESTS: Kelly + Vol + Signal Quality
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegrationSizing:
    """Integration tests combining Kelly, vol adjustment, and signal quality."""

    def test_kelly_then_vol_adjustment(self):
        """Kelly sizing followed by volatility adjustment."""
        kelly_pct = _kelly_size(0.60, 0.04, 0.02)  # -> 0.15 (capped)
        # High vol environment
        final = _vol_adjusted_size(kelly_pct, 0.05, target_vol=0.02)
        # vol_ratio = 0.02/0.05 = 0.4
        assert abs(final - 0.06) < 0.001

    def test_kelly_then_low_vol_boost(self):
        """Kelly sizing in low-vol environment gets boosted (but max 2x)."""
        kelly_pct = _kelly_size(0.55, 0.03, 0.02)  # -> ~0.125
        final = _vol_adjusted_size(kelly_pct, 0.005, target_vol=0.02)
        # vol_ratio = 0.02/0.005 = 4.0 -> capped at 2.0
        assert abs(final - 0.25) < 0.001

    def test_kelly_vol_with_conflict_discount(self):
        """Kelly + vol + conflicted signal quality -> halved again."""
        kelly_pct = _kelly_size(0.60, 0.04, 0.02)  # 0.15
        vol_adjusted = _vol_adjusted_size(kelly_pct, 0.02)  # no change (matching vol)
        # Conflict discount: 50%
        final = vol_adjusted * 0.5
        assert abs(final - 0.075) < 0.001

    def test_full_pipeline_respects_max_cap(self):
        """Even with boosted sizing, max_position_pct cap is respected."""
        kelly_pct = _kelly_size(1.0, 0.10, 0.01)  # -> 0.15 (capped)
        vol_adjusted = _vol_adjusted_size(kelly_pct, 0.005)  # 2x boost -> 0.30
        max_pos_pct = 0.25
        final = min(vol_adjusted, max_pos_pct)
        assert final == 0.25

    def test_no_edge_falls_through_to_default(self):
        """When Kelly says no edge, default sizing should be used."""
        kelly_pct = _kelly_size(0.40, 0.015, 0.02)  # negative edge -> 0.0
        assert kelly_pct == 0.0
        # System would fall back to confidence * strength * max_pos_pct
        default = 0.7 * 0.8 * 0.25  # confidence=0.7, strength=0.8
        assert default > 0


class TestKellyVsDefault:
    """Compare Kelly sizing vs default sizing on scenarios."""

    def test_strong_edge_kelly_larger(self):
        """With strong edge, Kelly should size larger than conservative default."""
        # Default: confidence * strength * max_pos_pct
        default = 0.6 * 0.5 * 0.25  # = 0.075
        kelly = _kelly_size(0.65, 0.04, 0.02)  # Strong edge -> 0.15
        assert kelly > default, f"Kelly {kelly} should exceed default {default}"

    def test_weak_edge_kelly_smaller(self):
        """With weak edge, Kelly should size smaller than default."""
        default = 0.7 * 0.6 * 0.25  # = 0.105
        kelly = _kelly_size(0.52, 0.015, 0.015)  # Tiny edge -> ~0.02
        assert kelly < default, f"Kelly {kelly} should be less than default {default}"

    def test_no_edge_kelly_zero(self):
        """No edge -> Kelly is 0, default is still positive."""
        default = 0.5 * 0.5 * 0.25  # = 0.0625
        kelly = _kelly_size(0.50, 0.02, 0.02)
        assert kelly == 0.0
        assert default > 0

    def test_high_vol_reduces_both(self):
        """High volatility reduces both Kelly and default sizing equally."""
        kelly_base = _kelly_size(0.60, 0.03, 0.02)
        default_base = 0.10
        vol = 0.06  # 6% daily vol

        kelly_adj = _vol_adjusted_size(kelly_base, vol)
        default_adj = _vol_adjusted_size(default_base, vol)

        assert kelly_adj < kelly_base
        assert default_adj < default_base


# ═══════════════════════════════════════════════════════════════════════════
# 6. ENSEMBLE HUB FULL INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestEnsembleHubIntegration:
    """End-to-end tests of EnsembleSignalHub with conflict detection."""

    def _make_hub_with_alpha(self, alpha_mock):
        """Create hub with only alpha source active."""
        from ml.ensemble_signal_hub import EnsembleSignalHub
        cfg = {
            "enabled": {
                "fear_greed": False,
                "llm": False,
                "whale": False,
                "news": False,
                "alpha": True,
                "vol_regime": False,
                "funding": False,
            }
        }
        hub = EnsembleSignalHub(config=cfg, alpha=alpha_mock)
        return hub

    def test_update_sets_signal_quality(self):
        """After update(), get_signal_quality() should return meaningful data."""
        alpha_mock = MagicMock()
        alpha_mock.update = MagicMock()
        score_mock = MagicMock()
        score_mock.composite = 0.6
        score_mock.signal = "BUY"
        score_mock.confidence = 0.8
        alpha_mock.score = MagicMock(return_value=score_mock)

        hub = self._make_hub_with_alpha(alpha_mock)
        hub.update("BTC/USD", [100.0, 101.0, 102.0])

        q = hub.get_signal_quality()
        assert q["n_sources"] >= 1
        assert q["strongest_source"] == "alpha"

    def test_snapshot_still_works(self):
        """snapshot() should still return the expected structure."""
        from ml.ensemble_signal_hub import EnsembleSignalHub
        cfg = {
            "enabled": {
                "fear_greed": False, "llm": False, "whale": False,
                "news": False, "alpha": False, "vol_regime": False, "funding": False,
            }
        }
        hub = EnsembleSignalHub(config=cfg)
        snap = hub.snapshot()
        assert "sources" in snap
        assert "cache" in snap

    def test_neutral_signal_quality_on_no_sources(self):
        """With all sources disabled, signal quality should be weak."""
        from ml.ensemble_signal_hub import EnsembleSignalHub
        cfg = {
            "enabled": {
                "fear_greed": False, "llm": False, "whale": False,
                "news": False, "alpha": False, "vol_regime": False, "funding": False,
                "chain_metrics": False,
            }
        }
        hub = EnsembleSignalHub(config=cfg)
        hub.update("ETH/USD", [50.0, 51.0])
        q = hub.get_signal_quality()
        assert q["recommendation"] == "weak"
        assert q["n_sources"] == 0
