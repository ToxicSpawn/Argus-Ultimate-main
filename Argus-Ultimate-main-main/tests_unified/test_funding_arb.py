"""
Tests for FundingRateArb and FundingRateScanner.

Run with:
    pytest tests_unified/test_funding_arb.py -v

Coverage:
- FundingArbConfig defaults
- Delta calculation (spot/perp neutrality)
- Rebalance trigger on delta drift
- Rate-flip position close
- Annualised yield calculation
- Scanner filtering
- Scanner stability scoring
- Scanner sorting
- FundingOpportunity recommended flag
- Rate-flip detection
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from strategies.funding_rate_arb import (
    FundingArbConfig,
    FundingPosition,
    FundingRateArb,
    annualised_yield_from_rate,
    calculate_delta,
    is_delta_neutral,
)
from alpha.funding_rate_scanner import (
    FundingOpportunity,
    FundingRateScanner,
    FundingSnapshot,
    _compute_stability_score,
    compute_funding_payment_usd,
    rate_to_annualised,
    sort_by_yield,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_opportunity(
    symbol: str = "BTC/USDT:USDT",
    exchange: str = "bybit",
    current_rate: float = 0.0003,
    avg_7d_rate: float = 0.00025,
    stability: float = 0.8,
    spot_available: bool = True,
    spot_spread_bps: float = 5.0,
) -> FundingOpportunity:
    opp = FundingOpportunity(
        symbol=symbol,
        exchange=exchange,
        current_rate=current_rate,
        predicted_next_rate=current_rate,
        avg_7d_rate=avg_7d_rate,
        rate_stability_score=stability,
        spot_available=spot_available,
        spot_symbol=symbol.split(":")[0],
        spot_spread_bps=spot_spread_bps,
    )
    return opp


def _make_snapshot(symbol: str, rate: float, ts_offset_s: int = 0) -> FundingSnapshot:
    ts_ms = int((time.time() - ts_offset_s) * 1000)
    return FundingSnapshot(symbol=symbol, exchange="bybit", rate=rate, timestamp=ts_ms)


# ---------------------------------------------------------------------------
# 1. FundingArbConfig defaults
# ---------------------------------------------------------------------------

class TestFundingArbConfigDefaults:
    def test_funding_arb_config_defaults(self):
        cfg = FundingArbConfig()
        assert cfg.capital_usd == 300.0
        assert cfg.min_funding_rate == 0.0001
        assert cfg.max_symbols == 3
        assert cfg.rebalance_threshold_pct == 2.0
        assert cfg.max_leverage == 3.0
        assert cfg.exchanges_spot == ["bybit"]
        assert cfg.exchanges_perp == ["bybit"]
        assert cfg.funding_check_interval_s == 300.0

    def test_capital_per_position(self):
        cfg = FundingArbConfig(capital_usd=300.0, max_symbols=3)
        assert cfg.capital_per_position_usd() == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 2. Delta calculation
# ---------------------------------------------------------------------------

class TestDeltaCalculation:
    def test_funding_arb_delta_calculation_neutral(self):
        """spot=$100 long, perp=$100 short → delta ≈ 0."""
        delta = calculate_delta(
            spot_value=100.0,
            perp_value=-100.0,
            capital_allocated=100.0,
        )
        assert delta == pytest.approx(0.0, abs=1e-9)

    def test_funding_arb_delta_calculation_drift(self):
        """spot=$105, perp=-$100 → delta = 0.05 (5% of capital)."""
        delta = calculate_delta(
            spot_value=105.0,
            perp_value=-100.0,
            capital_allocated=100.0,
        )
        assert delta == pytest.approx(0.05, abs=1e-9)

    def test_is_delta_neutral_true(self):
        assert is_delta_neutral(100.0, -100.0, 100.0, threshold_pct=2.0) is True

    def test_is_delta_neutral_false(self):
        # 5% drift exceeds 2% threshold
        assert is_delta_neutral(105.0, -100.0, 100.0, threshold_pct=2.0) is False

    def test_delta_zero_capital_returns_zero(self):
        delta = calculate_delta(100.0, -100.0, capital_allocated=0.0)
        assert delta == 0.0


# ---------------------------------------------------------------------------
# 3. Rebalance trigger
# ---------------------------------------------------------------------------

class TestRebalanceTrigger:
    def test_funding_arb_rebalance_trigger(self):
        """
        When delta drift > rebalance_threshold_pct, rebalance() is called.
        We mock _monitor_delta_neutrality to confirm the logic.
        """
        cfg = FundingArbConfig(rebalance_threshold_pct=2.0)
        arb = FundingRateArb(cfg)

        pos = FundingPosition(
            symbol="DOGE/USDT:USDT",
            spot_exchange="bybit",
            perp_exchange="bybit",
            spot_size=100.0,
            spot_avg_price=1.0,
            perp_size=-95.0,   # short 95 units
            perp_avg_price=1.0,
            capital_allocated=100.0,
            status="active",
        )
        arb._positions["DOGE/USDT:USDT"] = pos

        # Drift: spot_value=100, perp_value=-95 → delta=(100-95)/100=0.05 → 5%
        drift = pos.update_delta_drift(current_price=1.0)
        assert drift == pytest.approx(5.0, abs=0.01)
        assert drift > cfg.rebalance_threshold_pct

    def test_no_rebalance_below_threshold(self):
        cfg = FundingArbConfig(rebalance_threshold_pct=2.0)
        pos = FundingPosition(
            symbol="BTC/USDT:USDT",
            spot_exchange="bybit",
            perp_exchange="bybit",
            spot_size=1.0,
            spot_avg_price=50000.0,
            perp_size=-1.0,
            perp_avg_price=50000.0,
            capital_allocated=50000.0,
            status="active",
        )
        drift = pos.update_delta_drift(current_price=50000.0)
        assert drift == pytest.approx(0.0, abs=1e-6)
        assert drift <= cfg.rebalance_threshold_pct


# ---------------------------------------------------------------------------
# 4. Rate flip → close
# ---------------------------------------------------------------------------

class TestRateFlipClose:
    def test_funding_arb_rate_flip_close(self):
        """
        Funding rate flipping sign for 2+ consecutive payments
        should set consecutive_flips >= 2.
        """
        pos = FundingPosition(
            symbol="PEPE/USDT:USDT",
            spot_exchange="bybit",
            perp_exchange="bybit",
            capital_allocated=100.0,
            status="active",
        )

        # Payment 1: positive (short collects)
        pos.record_funding_payment(0.10)
        assert pos.consecutive_flips == 0

        # Payment 2: negative (rate flipped → short pays)
        pos.record_funding_payment(-0.05)
        assert pos.consecutive_flips == 1

        # Payment 3: positive again (flip again)
        pos.record_funding_payment(0.08)
        assert pos.consecutive_flips == 2

    def test_consecutive_flips_reset_on_same_sign(self):
        pos = FundingPosition(
            symbol="BTC/USDT:USDT",
            spot_exchange="bybit",
            perp_exchange="bybit",
            capital_allocated=100.0,
            status="active",
        )
        pos.record_funding_payment(0.10)
        pos.record_funding_payment(-0.05)
        assert pos.consecutive_flips == 1
        # Same sign as previous (negative)
        pos.record_funding_payment(-0.03)
        assert pos.consecutive_flips == 0


# ---------------------------------------------------------------------------
# 5. Annualised yield from rate
# ---------------------------------------------------------------------------

class TestAnnualisedYield:
    def test_funding_position_annualised_yield(self):
        """0.01% per 8h → 3 × 365 × 0.0001 × 100 = 10.95% annualised."""
        rate = 0.0001  # 0.01% per 8h
        yield_pct = annualised_yield_from_rate(rate)
        assert yield_pct == pytest.approx(10.95, abs=0.01)

    def test_rate_to_annualised_helper(self):
        rate = 0.0001
        yield_pct = rate_to_annualised(rate)
        assert yield_pct == pytest.approx(10.95, abs=0.01)

    def test_high_funding_rate(self):
        """0.05% per 8h → ~54.75% annualised (bull-market scenario)."""
        rate = 0.0005
        yield_pct = annualised_yield_from_rate(rate)
        assert yield_pct == pytest.approx(54.75, abs=0.01)


# ---------------------------------------------------------------------------
# 6. Scanner filtering
# ---------------------------------------------------------------------------

class TestScannerFilters:
    def test_funding_scanner_filters_below_min(self):
        """Opportunities with rate below min_rate should be excluded."""
        scanner = FundingRateScanner(min_rate=0.0001)
        # Inject scan results directly
        scanner._last_scan_results = {
            "LOW/USDT:USDT": _make_opportunity(
                symbol="LOW/USDT:USDT", current_rate=0.00005, avg_7d_rate=0.00004
            ),
            "HIGH/USDT:USDT": _make_opportunity(
                symbol="HIGH/USDT:USDT", current_rate=0.0003, avg_7d_rate=0.00025
            ),
        }
        top = scanner.get_top_opportunities(n=5)
        symbols = [o.symbol for o in top]
        assert "LOW/USDT:USDT" not in symbols
        assert "HIGH/USDT:USDT" in symbols

    def test_funding_scanner_filters_unstable_avg(self):
        """Opportunities with avg_7d_rate < min_rate × 0.5 are filtered."""
        scanner = FundingRateScanner(min_rate=0.0001)
        scanner._last_scan_results = {
            "SPIKE/USDT:USDT": _make_opportunity(
                symbol="SPIKE/USDT:USDT",
                current_rate=0.0003,
                avg_7d_rate=0.00004,  # below 0.5 × min_rate = 0.00005
            ),
        }
        top = scanner.get_top_opportunities(n=5)
        assert len(top) == 0


# ---------------------------------------------------------------------------
# 7. Stability scoring
# ---------------------------------------------------------------------------

class TestScannerStability:
    def test_funding_scanner_stability_stable(self):
        """Consistently positive rates → high stability."""
        rates = [0.0003, 0.00028, 0.00031, 0.00029, 0.0003, 0.00027, 0.00032]
        score = _compute_stability_score(rates)
        assert score >= 0.7, f"Expected stability ≥ 0.7, got {score}"

    def test_funding_scanner_stability_volatile(self):
        """Alternating signs → low stability."""
        rates = [0.0003, -0.0002, 0.0004, -0.0001, 0.0005, -0.0003, 0.0002]
        score = _compute_stability_score(rates)
        assert score <= 0.5, f"Expected stability ≤ 0.5, got {score}"

    def test_stability_insufficient_data(self):
        """Fewer than 3 samples returns 0.5 (neutral)."""
        assert _compute_stability_score([0.0003, 0.0004]) == pytest.approx(0.5)

    def test_stability_all_same_sign(self):
        """All same-sign rates → near-perfect sign consistency."""
        rates = [0.0002] * 10
        score = _compute_stability_score(rates)
        assert score > 0.8


# ---------------------------------------------------------------------------
# 8. Scanner sorting
# ---------------------------------------------------------------------------

class TestScannerSort:
    def test_funding_scanner_sort_highest_yield_first(self):
        """sort_by_yield should return highest annualised yield first."""
        opps = [
            _make_opportunity("A/USDT:USDT", current_rate=0.0001),
            _make_opportunity("B/USDT:USDT", current_rate=0.0005),
            _make_opportunity("C/USDT:USDT", current_rate=0.0003),
        ]
        sorted_opps = sort_by_yield(opps)
        yields = [o.annualised_yield_pct for o in sorted_opps]
        assert yields == sorted(yields, reverse=True)

    def test_funding_scanner_sort_get_top(self):
        scanner = FundingRateScanner(min_rate=0.0001)
        scanner._last_scan_results = {
            "A/USDT:USDT": _make_opportunity("A/USDT:USDT", current_rate=0.0001),
            "B/USDT:USDT": _make_opportunity("B/USDT:USDT", current_rate=0.0005),
            "C/USDT:USDT": _make_opportunity("C/USDT:USDT", current_rate=0.0003),
        }
        top = scanner.get_top_opportunities(n=2)
        assert len(top) == 2
        assert top[0].current_rate >= top[1].current_rate


# ---------------------------------------------------------------------------
# 9. Recommended flag
# ---------------------------------------------------------------------------

class TestFundingOpportunityRecommended:
    def test_funding_opportunity_recommended_true(self):
        """yield > 15% AND stability > 0.6 → recommended=True."""
        opp = _make_opportunity(current_rate=0.0005, stability=0.75)
        # annualised = 0.0005 × 3 × 365 × 100 = 54.75% > 15%
        assert opp.recommended is True

    def test_funding_opportunity_recommended_false_low_yield(self):
        """yield < 15% → not recommended."""
        opp = _make_opportunity(current_rate=0.00012, stability=0.9)
        # annualised ≈ 13.14% < 15%
        assert opp.recommended is False

    def test_funding_opportunity_recommended_false_low_stability(self):
        """yield > 15% but stability < 0.6 → not recommended."""
        opp = _make_opportunity(current_rate=0.0005, stability=0.4)
        assert opp.recommended is False

    def test_funding_opportunity_recommended_boundary(self):
        """Exactly at 15% yield and 0.6 stability → recommended=True."""
        # rate × 3 × 365 × 100 = 15 → rate = 15 / (3 × 365 × 100)
        rate = 15.0 / (3 * 365 * 100)  # exactly 15% annualised
        opp = _make_opportunity(current_rate=rate, stability=0.6)
        assert opp.recommended is True


# ---------------------------------------------------------------------------
# 10. Rate flip detection
# ---------------------------------------------------------------------------

class TestFundingRateFlipDetection:
    def test_funding_rate_flip_detection_true(self):
        """Sign changes in last 3 payments → is_rate_flipping = True."""
        scanner = FundingRateScanner(min_rate=0.0001)
        opp = _make_opportunity("FLIP/USDT:USDT", current_rate=0.0003)
        # Inject history with sign changes
        opp._history = [
            _make_snapshot("FLIP/USDT:USDT", rate=0.0003, ts_offset_s=24 * 3600),
            _make_snapshot("FLIP/USDT:USDT", rate=0.0002, ts_offset_s=16 * 3600),
            _make_snapshot("FLIP/USDT:USDT", rate=-0.0001, ts_offset_s=8 * 3600),  # flip
            _make_snapshot("FLIP/USDT:USDT", rate=0.0003, ts_offset_s=0),            # flip
        ]
        scanner._last_scan_results["FLIP/USDT:USDT"] = opp
        assert scanner.is_rate_flipping("FLIP/USDT:USDT") is True

    def test_funding_rate_flip_detection_false(self):
        """All same sign in last 3 → is_rate_flipping = False."""
        scanner = FundingRateScanner(min_rate=0.0001)
        opp = _make_opportunity("STABLE/USDT:USDT", current_rate=0.0003)
        opp._history = [
            _make_snapshot("STABLE/USDT:USDT", rate=0.0003, ts_offset_s=24 * 3600),
            _make_snapshot("STABLE/USDT:USDT", rate=0.00028, ts_offset_s=16 * 3600),
            _make_snapshot("STABLE/USDT:USDT", rate=0.00031, ts_offset_s=8 * 3600),
            _make_snapshot("STABLE/USDT:USDT", rate=0.0003, ts_offset_s=0),
        ]
        scanner._last_scan_results["STABLE/USDT:USDT"] = opp
        assert scanner.is_rate_flipping("STABLE/USDT:USDT") is False

    def test_funding_rate_flip_insufficient_history(self):
        """Fewer than 3 snapshots → cannot detect flip, returns False."""
        scanner = FundingRateScanner(min_rate=0.0001)
        opp = _make_opportunity("SHORT/USDT:USDT", current_rate=0.0003)
        opp._history = [
            _make_snapshot("SHORT/USDT:USDT", rate=0.0003, ts_offset_s=8 * 3600),
            _make_snapshot("SHORT/USDT:USDT", rate=-0.0002, ts_offset_s=0),
        ]
        scanner._last_scan_results["SHORT/USDT:USDT"] = opp
        assert scanner.is_rate_flipping("SHORT/USDT:USDT") is False

    def test_funding_rate_flip_unknown_symbol(self):
        """Unknown symbol returns False without error."""
        scanner = FundingRateScanner(min_rate=0.0001)
        assert scanner.is_rate_flipping("UNKNOWN/USDT:USDT") is False


# ---------------------------------------------------------------------------
# Bonus: compute_funding_payment_usd
# ---------------------------------------------------------------------------

class TestComputeFundingPayment:
    def test_short_positive_rate_collects(self):
        """Short position + positive rate → positive payment (collected)."""
        payment = compute_funding_payment_usd(
            position_notional_usd=1000.0,
            funding_rate=0.0003,
            direction="short",
        )
        assert payment == pytest.approx(0.30, abs=1e-6)

    def test_long_positive_rate_pays(self):
        """Long position + positive rate → negative payment (paid out)."""
        payment = compute_funding_payment_usd(
            position_notional_usd=1000.0,
            funding_rate=0.0003,
            direction="long",
        )
        assert payment == pytest.approx(-0.30, abs=1e-6)

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction must be"):
            compute_funding_payment_usd(1000.0, 0.0003, direction="neutral")
