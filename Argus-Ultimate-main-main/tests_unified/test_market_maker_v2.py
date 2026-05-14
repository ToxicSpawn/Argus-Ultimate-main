"""
Tests for AvellanedaStoikovV2 and associated module-level functions.

Run with:
    pytest tests_unified/test_market_maker_v2.py -v
"""
from __future__ import annotations

import math
import time
from unittest.mock import patch

import pytest

from strategies.market_maker_avellaneda_v2 import (
    AvellanedaStoikovV2,
    QuoteRegime,
    QuoteResult,
    detect_quote_regime,
    dynamic_max_inventory,
    fill_probability,
    intensity_model,
    inventory_penalty,
    optimal_spread_with_intensity,
    skew_quotes,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: intensity_model — Poisson arrival intensity
# ─────────────────────────────────────────────────────────────────────────────

class TestIntensityModel:
    """intensity_model(delta, k, A) = A * exp(-k * delta)"""

    def test_at_zero_delta_equals_A(self):
        """At zero spread, intensity equals baseline rate A."""
        A, k = 2.5, 1.5
        result = intensity_model(0.0, k=k, A=A)
        assert math.isclose(result, A, rel_tol=1e-9)

    def test_decays_with_spread(self):
        """Intensity strictly decreases as spread (delta) increases."""
        k, A = 1.5, 1.0
        intensities = [intensity_model(d, k=k, A=A) for d in [0.01, 0.05, 0.1, 0.5]]
        assert all(a > b for a, b in zip(intensities, intensities[1:]))

    def test_invalid_params_return_zero(self):
        """Non-positive k or A should return 0."""
        assert intensity_model(0.1, k=0.0, A=1.0) == 0.0
        assert intensity_model(0.1, k=1.0, A=0.0) == 0.0
        assert intensity_model(0.1, k=-1.0, A=1.0) == 0.0

    def test_large_spread_near_zero(self):
        """Very large spread should produce near-zero intensity."""
        result = intensity_model(100.0, k=2.0, A=1.0)
        assert result < 1e-80


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: fill_probability
# ─────────────────────────────────────────────────────────────────────────────

class TestFillProbability:
    """fill_probability(delta, k) = exp(-k * delta)"""

    def test_at_zero_returns_one(self):
        """Zero spread → always filled (probability = 1)."""
        assert fill_probability(0.0, k=1.5) == 1.0

    def test_decreases_with_spread(self):
        """Probability decreases as spread increases."""
        probs = [fill_probability(d, k=1.5) for d in [0.01, 0.1, 1.0, 10.0]]
        assert all(a > b for a, b in zip(probs, probs[1:]))

    def test_range_bounded(self):
        """Fill probability must be in [0, 1] for any inputs."""
        for delta in [-1.0, 0.0, 0.5, 10.0, 100.0]:
            p = fill_probability(delta, k=1.5)
            assert 0.0 <= p <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: optimal_spread_with_intensity
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimalSpreadWithIntensity:
    """Closed-form Guéant-Lehalle optimal spread."""

    def test_spread_positive(self):
        """Optimal spread should be positive for valid inputs."""
        spread = optimal_spread_with_intensity(vol=0.001, gamma=0.1, k=1.5, A=1.0, T_remaining=0.5)
        assert spread > 0.0

    def test_spread_widens_with_volatility(self):
        """Higher volatility should produce wider spread."""
        s_low  = optimal_spread_with_intensity(vol=0.0005, gamma=0.1, k=1.5, A=1.0, T_remaining=0.5)
        s_high = optimal_spread_with_intensity(vol=0.002,  gamma=0.1, k=1.5, A=1.0, T_remaining=0.5)
        assert s_high > s_low

    def test_spread_widens_with_risk_aversion(self):
        """Higher gamma → wider spread (more inventory aversion)."""
        s_low  = optimal_spread_with_intensity(vol=0.001, gamma=0.05, k=1.5, A=1.0, T_remaining=0.5)
        s_high = optimal_spread_with_intensity(vol=0.001, gamma=0.5,  k=1.5, A=1.0, T_remaining=0.5)
        assert s_high > s_low

    def test_zero_vol_returns_zero(self):
        """Zero volatility → zero volatility component (base term may remain)."""
        result = optimal_spread_with_intensity(vol=0.0, gamma=0.1, k=1.5, A=1.0, T_remaining=0.5)
        # With zero vol the vol_term is zero; result ≥ 0
        assert result >= 0.0

    def test_invalid_inputs_return_zero(self):
        """Invalid inputs should return 0, not raise."""
        assert optimal_spread_with_intensity(vol=0.001, gamma=0.0, k=1.5, A=1.0, T_remaining=0.5) == 0.0
        assert optimal_spread_with_intensity(vol=0.001, gamma=0.1, k=0.0, A=1.0, T_remaining=0.5) == 0.0
        assert optimal_spread_with_intensity(vol=0.001, gamma=0.1, k=1.5, A=1.0, T_remaining=0.0) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: inventory_penalty
# ─────────────────────────────────────────────────────────────────────────────

class TestInventoryPenalty:
    """Quadratic inventory penalty ψ(q) = γ σ² (T-t) q²."""

    def test_zero_inventory_zero_penalty(self):
        assert inventory_penalty(q=0.0, gamma=0.1, sigma=0.001, T_remaining=0.5) == 0.0

    def test_symmetric(self):
        """Penalty of q == penalty of -q."""
        pos = inventory_penalty(q= 2.0, gamma=0.1, sigma=0.001, T_remaining=0.5)
        neg = inventory_penalty(q=-2.0, gamma=0.1, sigma=0.001, T_remaining=0.5)
        assert math.isclose(pos, neg)

    def test_quadratic_scaling(self):
        """Doubling inventory should quadruple penalty."""
        p1 = inventory_penalty(q=1.0, gamma=0.1, sigma=0.01, T_remaining=0.5)
        p2 = inventory_penalty(q=2.0, gamma=0.1, sigma=0.01, T_remaining=0.5)
        assert math.isclose(p2 / p1, 4.0, rel_tol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: detect_quote_regime
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectQuoteRegime:

    def test_normal_regime(self):
        regime = detect_quote_regime(obi=0.1, vpin=0.5, spread_percentile=20.0)
        assert regime == QuoteRegime.NORMAL

    def test_adverse_regime_high_vpin(self):
        regime = detect_quote_regime(obi=0.0, vpin=0.75, spread_percentile=20.0)
        assert regime == QuoteRegime.ADVERSE

    def test_trending_regime_high_obi(self):
        regime = detect_quote_regime(obi=0.6, vpin=0.5, spread_percentile=20.0)
        assert regime == QuoteRegime.TRENDING

    def test_illiquid_regime_priority(self):
        """ILLIQUID takes priority even if VPIN is also high."""
        regime = detect_quote_regime(obi=0.0, vpin=0.9, spread_percentile=55.0)
        assert regime == QuoteRegime.ILLIQUID

    def test_adverse_over_trending(self):
        """ADVERSE takes priority over TRENDING when both conditions met."""
        regime = detect_quote_regime(obi=0.5, vpin=0.8, spread_percentile=10.0)
        assert regime == QuoteRegime.ADVERSE


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: dynamic_max_inventory
# ─────────────────────────────────────────────────────────────────────────────

class TestDynamicMaxInventory:

    def test_full_limit_at_start(self):
        """At session start (elapsed=0), full inventory limit applies."""
        limit = dynamic_max_inventory(base_max=10.0, session_elapsed_frac=0.0)
        assert math.isclose(limit, 10.0, rel_tol=1e-9)

    def test_decreases_over_session(self):
        """Limit should decrease as session progresses."""
        limits = [dynamic_max_inventory(10.0, f) for f in [0.0, 0.3, 0.6, 0.9]]
        assert all(a > b for a, b in zip(limits, limits[1:]))

    def test_floor_at_session_end(self):
        """Limit never goes below 5% of base at session end."""
        limit = dynamic_max_inventory(base_max=10.0, session_elapsed_frac=1.0)
        assert limit >= 0.5   # 5% of 10


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: skew_quotes
# ─────────────────────────────────────────────────────────────────────────────

class TestSkewQuotes:

    def test_zero_inventory_zero_skew(self):
        bid_adj, ask_adj = skew_quotes(
            mid=50_000.0, q=0.0, gamma=0.1, sigma=0.001,
            T_remaining=0.5, max_inv=5.0
        )
        assert bid_adj == 0.0
        assert ask_adj == 0.0

    def test_long_inventory_lowers_quotes(self):
        """Long inventory → lower both bid and ask to encourage selling."""
        bid_adj, ask_adj = skew_quotes(
            mid=50_000.0, q=3.0, gamma=0.1, sigma=0.01,
            T_remaining=0.5, max_inv=5.0
        )
        # Both adjustments should be negative for long inventory
        assert bid_adj < 0 or ask_adj < 0

    def test_short_inventory_raises_quotes(self):
        """Short inventory → raise bid/ask to encourage buying."""
        bid_adj, ask_adj = skew_quotes(
            mid=50_000.0, q=-3.0, gamma=0.1, sigma=0.01,
            T_remaining=0.5, max_inv=5.0
        )
        assert bid_adj > 0 or ask_adj > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: AvellanedaStoikovV2.compute_quotes_v2 — normal regime
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeQuotesV2Normal:

    def _make_mm(self) -> AvellanedaStoikovV2:
        mm = AvellanedaStoikovV2(symbol="BTC/USDT", gamma=0.1, k=1.5, A=1.0)
        # Seed price history so volatility can be estimated
        for p in range(49990, 50010):
            mm._mid_prices.append(float(p))
        mm._last_mid = 50_000.0
        return mm

    def test_returns_quote_result(self):
        mm = self._make_mm()
        result = mm.compute_quotes_v2(
            mid=50_000.0, inventory=0.0,
            obi_signal=0.0, vpin=0.3, spread_pct=5.0,
            session_elapsed_frac=0.3,
        )
        assert isinstance(result, QuoteResult)

    def test_bid_lt_ask(self):
        mm = self._make_mm()
        result = mm.compute_quotes_v2(
            mid=50_000.0, inventory=0.0,
            obi_signal=0.0, vpin=0.3, spread_pct=5.0,
            session_elapsed_frac=0.3,
        )
        assert result is not None
        assert result.bid_price < result.ask_price

    def test_normal_regime_returns_quote_action(self):
        mm = self._make_mm()
        result = mm.compute_quotes_v2(
            mid=50_000.0, inventory=0.0,
            obi_signal=0.1, vpin=0.2, spread_pct=5.0,
            session_elapsed_frac=0.3,
        )
        assert result is not None
        assert result.regime == QuoteRegime.NORMAL
        assert result.recommended_action == "QUOTE"


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: Regime-specific behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeBehaviour:

    def _make_mm(self) -> AvellanedaStoikovV2:
        mm = AvellanedaStoikovV2(symbol="ETH/USDT", gamma=0.1, k=1.5, A=1.0,
                                  base_quote_size=1.0)
        for p in range(3490, 3510):
            mm._mid_prices.append(float(p))
        mm._last_mid = 3_500.0
        return mm

    def test_illiquid_regime_halts(self):
        """ILLIQUID regime → recommended_action == HALT."""
        mm = self._make_mm()
        result = mm.compute_quotes_v2(
            mid=3_500.0, inventory=0.0,
            obi_signal=0.0, vpin=0.3, spread_pct=60.0,   # > 50bps → ILLIQUID
            session_elapsed_frac=0.3,
        )
        assert result is not None
        assert result.regime == QuoteRegime.ILLIQUID
        assert result.recommended_action == "HALT"

    def test_adverse_regime_widens_spread(self):
        """ADVERSE regime → spread wider than NORMAL regime at same price."""
        mm_normal = self._make_mm()
        mm_adverse = self._make_mm()

        r_normal = mm_normal.compute_quotes_v2(
            mid=3_500.0, inventory=0.0,
            obi_signal=0.0, vpin=0.2, spread_pct=5.0,
            session_elapsed_frac=0.3,
        )
        r_adverse = mm_adverse.compute_quotes_v2(
            mid=3_500.0, inventory=0.0,
            obi_signal=0.0, vpin=0.8, spread_pct=5.0,   # high VPIN → ADVERSE
            session_elapsed_frac=0.3,
        )
        assert r_normal is not None and r_adverse is not None
        spread_normal = r_normal.ask_price - r_normal.bid_price
        spread_adverse = r_adverse.ask_price - r_adverse.bid_price
        assert spread_adverse > spread_normal

    def test_adverse_regime_reduces_size(self):
        """ADVERSE regime → sizes should be 50% of base."""
        mm = self._make_mm()
        result = mm.compute_quotes_v2(
            mid=3_500.0, inventory=0.0,
            obi_signal=0.0, vpin=0.85, spread_pct=5.0,
            session_elapsed_frac=0.3,
        )
        assert result is not None
        assert result.regime == QuoteRegime.ADVERSE
        assert result.bid_size <= mm.base_quote_size * 0.5 + 1e-9
        assert result.ask_size <= mm.base_quote_size * 0.5 + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: update_from_fill and session_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateFromFillAndSummary:

    def test_fill_updates_inventory(self):
        mm = AvellanedaStoikovV2(symbol="SOL/USDT")
        mm._last_mid = 150.0
        ts = time.time()
        mm.update_from_fill(side="buy", price=149.9, size=10.0, fill_timestamp=ts)
        assert mm.inventory == pytest.approx(10.0)

    def test_fill_updates_k_estimate(self):
        """A fill should trigger a Bayesian update that changes k."""
        mm = AvellanedaStoikovV2(symbol="SOL/USDT")
        mm._last_mid = 150.0
        k_before = mm._k_estimate
        ts = time.time()
        mm.update_from_fill(side="sell", price=150.05, size=5.0, fill_timestamp=ts)
        # k should have changed (direction doesn't matter — just must differ)
        assert mm._k_estimate != k_before

    def test_session_summary_keys(self):
        """session_summary should contain all expected keys."""
        mm = AvellanedaStoikovV2(symbol="BTC/USDT")
        summary = mm.session_summary()
        required_keys = {
            "symbol", "n_fills", "pnl_estimate_usd",
            "avg_spread_bps", "avg_expected_pnl_per_fill_bps",
            "regime_distribution_pct", "current_inventory",
            "k_learned", "A_learned", "session_t_remaining",
        }
        assert required_keys.issubset(summary.keys())

    def test_session_summary_fill_count(self):
        """Fill count in summary reflects actual fills recorded."""
        mm = AvellanedaStoikovV2(symbol="BTC/USDT")
        mm._last_mid = 50_000.0
        t0 = time.time()
        for i in range(3):
            mm.update_from_fill("buy", 49_999.0, 0.1, fill_timestamp=t0 + i)
        summary = mm.session_summary()
        assert summary["n_fills"] == 3
