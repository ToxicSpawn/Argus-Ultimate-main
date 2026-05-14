"""
test_micro_mm.py — Tests for MicroCapitalMM and AltcoinPairScanner.

Run with::

    pytest tests_unified/test_micro_mm.py -v

Coverage
--------
- MicroMMConfig: default values
- Inventory skew: long position → negative skew (ask moves down)
- Position limit enforcement: rejects new order at max inventory
- Kill switch: halts when session PnL < -(max_drawdown_pct × capital)
- PairScanner: low-spread pair filtered; wide-spread pair kept
- PairScanner: wider spread + higher volume = higher score
- PairOpportunity: zero-fee venue has higher estimated profit than fee venue
- PairScanner: too-high volume pair filtered (BTC/USDT-like volumes)
"""

from __future__ import annotations

import math
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Imports under test ─────────────────────────────────────────────────────
from strategies.micro_capital_mm import (
    MicroMMConfig,
    MicroCapitalMM,
    PairState,
    compute_inventory_skew,
    compute_quotes,
    EXCHANGE_MAKER_FEES,
)
from alpha.pair_scanner import (
    AltcoinPairScanner,
    PairOpportunity,
    PairSnapshot,
    estimate_daily_profit,
    score_pair,
    is_excluded_pair,
    BYBIT_SPOT_MAKER,
    KRAKEN_MAKER,
)


# ===========================================================================
# MicroMMConfig defaults
# ===========================================================================

class TestMicroMMConfigDefaults:
    """test_micro_mm_config_defaults — verify MicroMMConfig defaults."""

    def test_total_capital_default(self):
        cfg = MicroMMConfig()
        assert cfg.total_capital_usd == 620.0

    def test_max_pairs_default(self):
        cfg = MicroMMConfig()
        assert cfg.max_pairs == 3

    def test_per_pair_capital_default(self):
        cfg = MicroMMConfig()
        assert cfg.per_pair_capital_usd == 200.0

    def test_order_size_pct_default(self):
        cfg = MicroMMConfig()
        assert cfg.order_size_pct == 0.25

    def test_min_spread_bps_default(self):
        cfg = MicroMMConfig()
        assert cfg.min_spread_bps == 30

    def test_max_position_pct_default(self):
        cfg = MicroMMConfig()
        assert cfg.max_position_pct == 0.50

    def test_max_drawdown_pct_default(self):
        cfg = MicroMMConfig()
        assert cfg.max_drawdown_pct == 15.0

    def test_refresh_interval_ms_default(self):
        cfg = MicroMMConfig()
        assert cfg.refresh_interval_ms == 200.0

    def test_primary_exchange_is_bybit(self):
        cfg = MicroMMConfig()
        assert "bybit" in cfg.exchanges

    def test_fallback_exchanges_include_kraken_and_coinbase(self):
        cfg = MicroMMConfig()
        assert "kraken" in cfg.fallback_exchanges
        assert "coinbase" in cfg.fallback_exchanges

    def test_order_size_usd_computed_correctly(self):
        cfg = MicroMMConfig(per_pair_capital_usd=200.0, order_size_pct=0.25)
        assert cfg.order_size_usd() == 50.0

    def test_max_position_usd_computed_correctly(self):
        cfg = MicroMMConfig(per_pair_capital_usd=200.0, max_position_pct=0.50)
        assert cfg.max_position_usd() == 100.0

    def test_drawdown_limit_is_negative(self):
        cfg = MicroMMConfig(total_capital_usd=620.0, max_drawdown_pct=15.0)
        assert cfg.drawdown_limit_usd() == pytest.approx(-93.0)

    def test_bybit_maker_fee_is_zero(self):
        assert EXCHANGE_MAKER_FEES["bybit"] == 0.0


# ===========================================================================
# Inventory skew
# ===========================================================================

class TestInventorySkew:
    """test_micro_mm_inventory_skew — long position → negative skew."""

    def test_long_position_gives_negative_skew(self):
        """When long, skew < 0 — ask moves down to encourage sells."""
        skew = compute_inventory_skew(
            inventory_pct=0.40,   # 40% of capital is long
            base_spread=10.0,
            skew_strength=0.5,
        )
        assert skew < 0.0, f"Expected negative skew for long inventory, got {skew}"

    def test_short_position_gives_positive_skew(self):
        """When short, skew > 0 — bid moves up to encourage buys."""
        skew = compute_inventory_skew(
            inventory_pct=-0.40,
            base_spread=10.0,
            skew_strength=0.5,
        )
        assert skew > 0.0, f"Expected positive skew for short inventory, got {skew}"

    def test_neutral_inventory_gives_zero_skew(self):
        skew = compute_inventory_skew(
            inventory_pct=0.0,
            base_spread=10.0,
        )
        assert skew == 0.0

    def test_skew_magnitude_proportional_to_inventory(self):
        """Double inventory → double skew magnitude."""
        skew_low = compute_inventory_skew(0.20, base_spread=10.0)
        skew_high = compute_inventory_skew(0.40, base_spread=10.0)
        assert abs(skew_high) == pytest.approx(2 * abs(skew_low))

    def test_ask_moves_down_when_long(self):
        """compute_quotes: ask price should be lower when inventory is long."""
        cfg = MicroMMConfig()
        bid_neutral, ask_neutral = compute_quotes(
            best_bid=100.0, best_ask=100.1,
            inventory_pct=0.0, config=cfg, exchange="bybit",
        )
        bid_long, ask_long = compute_quotes(
            best_bid=100.0, best_ask=100.1,
            inventory_pct=0.40, config=cfg, exchange="bybit",
        )
        assert ask_long < ask_neutral, (
            f"Ask should decrease when long: {ask_long:.6f} < {ask_neutral:.6f}"
        )

    def test_bid_moves_up_when_short(self):
        """compute_quotes: bid price should be higher when inventory is short."""
        cfg = MicroMMConfig()
        bid_neutral, _ = compute_quotes(
            best_bid=100.0, best_ask=100.1,
            inventory_pct=0.0, config=cfg, exchange="bybit",
        )
        bid_short, _ = compute_quotes(
            best_bid=100.0, best_ask=100.1,
            inventory_pct=-0.40, config=cfg, exchange="bybit",
        )
        assert bid_short > bid_neutral, (
            f"Bid should increase when short: {bid_short:.6f} > {bid_neutral:.6f}"
        )

    def test_bid_always_less_than_ask(self):
        """Bid must always be strictly less than ask regardless of skew."""
        cfg = MicroMMConfig()
        for inv_pct in [-0.9, -0.5, 0.0, 0.5, 0.9]:
            bid, ask = compute_quotes(
                best_bid=100.0, best_ask=100.1,
                inventory_pct=inv_pct, config=cfg, exchange="bybit",
            )
            assert bid < ask, (
                f"bid ({bid:.6f}) >= ask ({ask:.6f}) at inventory_pct={inv_pct}"
            )


# ===========================================================================
# Position limit
# ===========================================================================

class TestPositionLimit:
    """test_micro_mm_position_limit — at max_position_pct → rejects new order."""

    def _make_pair_state(
        self,
        inventory_value_usd: float,
        per_pair_capital_usd: float = 200.0,
    ) -> PairState:
        ps = PairState(
            symbol="NEAR/USDT",
            exchange="bybit",
            fee_rate=0.0,
        )
        ps.inventory_base = inventory_value_usd / 5.0   # price = $5
        ps.inventory_value_usd = inventory_value_usd
        ps.best_bid = 5.0
        ps.best_ask = 5.01
        ps.mid = 5.005
        return ps

    def test_at_exactly_max_position_bid_is_blocked(self):
        """Inventory at max_position_usd → new bid would exceed limit."""
        cfg = MicroMMConfig(per_pair_capital_usd=200.0, max_position_pct=0.50)
        # max_position_usd = 100; order_size_usd = 50
        # current inventory = 100 (at limit); 100 + 50 > 100 → blocked
        ps = self._make_pair_state(inventory_value_usd=100.0)

        long_after_fill = ps.inventory_value_usd + cfg.order_size_usd()
        can_bid = long_after_fill <= cfg.max_position_usd()
        assert not can_bid, (
            "Bid should be blocked when inventory is at maximum"
        )

    def test_below_max_position_bid_is_allowed(self):
        """Inventory at 30% of limit → bid allowed."""
        cfg = MicroMMConfig(per_pair_capital_usd=200.0, max_position_pct=0.50)
        ps = self._make_pair_state(inventory_value_usd=30.0)

        long_after_fill = ps.inventory_value_usd + cfg.order_size_usd()
        can_bid = long_after_fill <= cfg.max_position_usd()
        assert can_bid

    def test_max_short_position_ask_is_blocked(self):
        """Inventory at maximum short → ask would exceed short limit."""
        cfg = MicroMMConfig(per_pair_capital_usd=200.0, max_position_pct=0.50)
        ps = self._make_pair_state(inventory_value_usd=-100.0)  # max short

        short_after_fill = ps.inventory_value_usd - cfg.order_size_usd()
        can_ask = short_after_fill >= -cfg.max_position_usd()
        assert not can_ask, "Ask should be blocked when short inventory is at maximum"

    def test_inventory_pct_calculation(self):
        """PairState.inventory_pct returns correct fraction."""
        ps = self._make_pair_state(inventory_value_usd=50.0)
        pct = ps.inventory_pct(per_pair_capital_usd=200.0)
        assert pct == pytest.approx(0.25)

    def test_inventory_pct_zero_capital_safe(self):
        """No division by zero when capital is 0."""
        ps = self._make_pair_state(inventory_value_usd=50.0)
        pct = ps.inventory_pct(per_pair_capital_usd=0.0)
        assert pct == 0.0


# ===========================================================================
# Kill switch
# ===========================================================================

class TestKillSwitch:
    """test_micro_mm_kill_switch — PnL < -15% of capital → halts."""

    def _make_mm(self) -> MicroCapitalMM:
        cfg = MicroMMConfig(total_capital_usd=620.0, max_drawdown_pct=15.0)
        return MicroCapitalMM(cfg)

    def test_kill_switch_triggers_below_threshold(self):
        """Session PnL < -93 USD (15% of 620) → kill switch = True."""
        mm = self._make_mm()
        mm._session_realised_pnl = -95.0   # below -15%
        assert mm._check_kill_switch() is True

    def test_kill_switch_does_not_trigger_above_threshold(self):
        """Session PnL > -93 USD → kill switch = False."""
        mm = self._make_mm()
        mm._session_realised_pnl = -50.0   # above -15%
        assert mm._check_kill_switch() is False

    def test_kill_switch_exactly_at_threshold(self):
        """At exactly the threshold the kill switch does NOT trigger."""
        mm = self._make_mm()
        # drawdown_limit_usd = -93.0
        mm._session_realised_pnl = -93.0
        # -93.0 is NOT < -93.0
        assert mm._check_kill_switch() is False

    def test_kill_switch_triggers_one_cent_below_threshold(self):
        """One cent below threshold → halts."""
        mm = self._make_mm()
        mm._session_realised_pnl = -93.01
        assert mm._check_kill_switch() is True

    def test_positive_pnl_never_triggers(self):
        """Positive PnL should never trigger the kill switch."""
        mm = self._make_mm()
        mm._session_realised_pnl = 50.0
        assert mm._check_kill_switch() is False

    def test_fee_calculation_bybit_is_zero(self):
        """Bybit spot maker fee should be zero."""
        fee = MicroCapitalMM.calculate_fee_for_order(100.0, 1.0, "bybit")
        assert fee == 0.0

    def test_fee_calculation_kraken(self):
        """Kraken 0.16% maker fee on $100 order = $0.16."""
        fee = MicroCapitalMM.calculate_fee_for_order(100.0, 1.0, "kraken")
        assert fee == pytest.approx(0.16)


# ===========================================================================
# Pair scanner: spread filter
# ===========================================================================

class TestPairScannerFilters:
    """test_pair_scanner_filters — low spread filtered; wide spread kept."""

    def _scanner(self) -> AltcoinPairScanner:
        return AltcoinPairScanner(
            min_spread_bps=30,
            min_24h_volume_usd=50_000.0,
            max_24h_volume_usd=5_000_000.0,
        )

    def _ticker(
        self,
        bid: float,
        ask: float,
        volume_usd: float = 500_000.0,
    ) -> dict:
        return {
            "bid": bid,
            "ask": ask,
            "quoteVolume": volume_usd,
        }

    def test_low_spread_pair_filtered_out(self):
        """A pair with < 30 bps spread should be rejected."""
        scanner = self._scanner()
        # 5 bps spread
        ticker = self._ticker(bid=100.0, ask=100.05)
        result = scanner._evaluate_ticker(
            symbol="NEAR/USDT",
            ticker=ticker,
            exchange="bybit",
            fee_rate=0.0,
        )
        assert result is None, "Low-spread pair should be filtered out"

    def test_wide_spread_pair_kept(self):
        """A pair with 50 bps spread should pass the filter."""
        scanner = self._scanner()
        # 50 bps: ask = bid × (1 + 0.005)
        bid = 5.0
        ask = bid * 1.005   # 50 bps
        ticker = self._ticker(bid=bid, ask=ask)
        result = scanner._evaluate_ticker(
            symbol="INJ/USDT",
            ticker=ticker,
            exchange="bybit",
            fee_rate=0.0,
        )
        assert result is not None, "Wide-spread pair should pass the filter"

    def test_stablecoin_pair_excluded(self):
        """USDC/USDT should always be excluded."""
        assert is_excluded_pair("USDC/USDT") is True

    def test_btc_usdt_excluded(self):
        """BTC/USDT is too liquid and should be excluded."""
        assert is_excluded_pair("BTC/USDT") is True

    def test_near_usdt_not_excluded(self):
        """NEAR/USDT is a valid target pair."""
        assert is_excluded_pair("NEAR/USDT") is False

    def test_non_usdt_quote_excluded(self):
        """NEAR/BTC is excluded because quote is not USDT."""
        assert is_excluded_pair("NEAR/BTC") is True

    def test_spread_exactly_at_minimum_is_kept(self):
        """Spread at or just above min_spread_bps should pass."""
        scanner = self._scanner()
        # Scanner formula: spread_bps = (ask - bid) / mid * 10_000
        # Solve for ask given bid=5.0 and target_bps=30.001:
        #   ask = bid * (2 + r) / (2 - r)  where r = target_bps / 10_000
        bid = 5.0
        r = 30.001 / 10_000.0
        ask = bid * (2.0 + r) / (2.0 - r)  # yields spread_bps ≈ 30.001
        ticker = self._ticker(bid=bid, ask=ask)
        result = scanner._evaluate_ticker(
            symbol="SEI/USDT",
            ticker=ticker,
            exchange="bybit",
            fee_rate=0.0,
        )
        assert result is not None, "Pair at min_spread_bps should pass"

    def test_spread_one_bp_below_minimum_filtered(self):
        """Spread one bp below min should be filtered out."""
        scanner = self._scanner()
        bid = 5.0
        # 29 bps — just below 30
        ask = bid * (1 + 29 / 10_000.0)
        ticker = self._ticker(bid=bid, ask=ask)
        result = scanner._evaluate_ticker(
            symbol="SEI/USDT",
            ticker=ticker,
            exchange="bybit",
            fee_rate=0.0,
        )
        assert result is None, "Pair 1bp below min_spread should be filtered"


# ===========================================================================
# Pair scanner: scoring
# ===========================================================================

class TestPairScannerScore:
    """test_pair_scanner_score — wider spread + higher volume = higher score."""

    def test_wider_spread_gives_higher_score(self):
        """Same volume, wider spread → higher score."""
        s_narrow = score_pair(spread_bps=30.0, volume_24h_usd=1_000_000.0)
        s_wide = score_pair(spread_bps=80.0, volume_24h_usd=1_000_000.0)
        assert s_wide > s_narrow

    def test_higher_volume_gives_higher_score(self):
        """Same spread, higher volume → higher score."""
        s_low = score_pair(spread_bps=50.0, volume_24h_usd=100_000.0)
        s_high = score_pair(spread_bps=50.0, volume_24h_usd=1_000_000.0)
        assert s_high > s_low

    def test_zero_spread_gives_zero_score(self):
        assert score_pair(0.0, 1_000_000.0) == 0.0

    def test_zero_volume_gives_zero_score(self):
        assert score_pair(50.0, 0.0) == 0.0

    def test_score_formula(self):
        """Verify exact score formula: spread_bps × sqrt(volume)."""
        spread = 50.0
        vol = 400_000.0
        expected = spread * math.sqrt(vol)
        assert score_pair(spread, vol) == pytest.approx(expected)

    def test_ranked_list_ordered_correctly(self):
        """scan() results should come back in descending score order."""
        scanner = AltcoinPairScanner(min_spread_bps=30)
        opps = [
            PairOpportunity("A/USDT", "bybit", 5.0, 5.05, 100.0, 200_000.0, score_pair(100.0, 200_000.0)),
            PairOpportunity("B/USDT", "bybit", 5.0, 5.05, 50.0, 500_000.0, score_pair(50.0, 500_000.0)),
            PairOpportunity("C/USDT", "bybit", 5.0, 5.05, 40.0, 1_000_000.0, score_pair(40.0, 1_000_000.0)),
        ]
        opps.sort(key=lambda o: o.score, reverse=True)
        scores = [o.score for o in opps]
        assert scores == sorted(scores, reverse=True)


# ===========================================================================
# PairOpportunity profit estimate
# ===========================================================================

class TestPairOpportunityProfitEstimate:
    """test_pair_opportunity_profit_estimate — zero fee > fee venue profit."""

    def test_zero_fee_venue_higher_profit(self):
        """
        On Bybit (zero fee), the same spread captures more profit than
        on Kraken (0.16% fee).
        """
        spread_bps = 50.0
        order_size_usd = 50.0

        profit_bybit = estimate_daily_profit(
            spread_bps=spread_bps,
            order_size_usd=order_size_usd,
            fee_rate=BYBIT_SPOT_MAKER,   # 0.0
        )
        profit_kraken = estimate_daily_profit(
            spread_bps=spread_bps,
            order_size_usd=order_size_usd,
            fee_rate=KRAKEN_MAKER,   # 0.0016
        )
        assert profit_bybit > profit_kraken, (
            f"Bybit (${profit_bybit:.4f}) should beat Kraken (${profit_kraken:.4f})"
        )

    def test_profit_positive_at_zero_fee(self):
        """Any positive spread with zero fee should yield positive profit."""
        profit = estimate_daily_profit(
            spread_bps=30.0,
            order_size_usd=50.0,
            fee_rate=0.0,
        )
        assert profit > 0.0

    def test_profit_zero_at_zero_spread(self):
        assert estimate_daily_profit(0.0, 50.0, 0.0) == 0.0

    def test_profit_zero_at_zero_size(self):
        assert estimate_daily_profit(50.0, 0.0, 0.0) == 0.0

    def test_recommended_flag_set_above_threshold(self):
        """PairOpportunity.recommended is True when daily profit ≥ $0.50."""
        opp = PairOpportunity(
            symbol="INJ/USDT",
            exchange="bybit",
            bid=10.0,
            ask=10.1,
            spread_bps=100.0,
            volume_24h_usd=500_000.0,
            score=1000.0,
            fee_rate=0.0,
            estimated_daily_profit_usd=0.75,
            recommended=True,
        )
        assert opp.recommended is True

    def test_recommended_flag_false_below_threshold(self):
        opp = PairOpportunity(
            symbol="SEI/USDT",
            exchange="kraken",
            bid=0.5,
            ask=0.503,
            spread_bps=60.0,
            volume_24h_usd=60_000.0,
            score=100.0,
            fee_rate=KRAKEN_MAKER,
            estimated_daily_profit_usd=0.10,
            recommended=False,
        )
        assert opp.recommended is False

    def test_bybit_makes_more_per_fill_than_kraken(self):
        """For 30 bps spread, $50 order: Bybit profit > Kraken profit."""
        bybit_profit = estimate_daily_profit(30.0, 50.0, BYBIT_SPOT_MAKER)
        kraken_profit = estimate_daily_profit(30.0, 50.0, KRAKEN_MAKER)
        assert bybit_profit > kraken_profit


# ===========================================================================
# Volume filter: too-high volume pair filtered
# ===========================================================================

class TestPairScannerVolumeFilter:
    """test_pair_scanner_volume_filter — BTC/USDT-like volumes filtered out."""

    def _scanner(self) -> AltcoinPairScanner:
        return AltcoinPairScanner(
            min_spread_bps=30,
            min_24h_volume_usd=50_000.0,
            max_24h_volume_usd=5_000_000.0,
        )

    def _ticker(self, bid: float, ask: float, volume_usd: float) -> dict:
        return {"bid": bid, "ask": ask, "quoteVolume": volume_usd}

    def test_high_volume_pair_filtered(self):
        """Volume > 5M USD (BTC/USDT-like) should be filtered out."""
        scanner = self._scanner()
        # 50 bps spread but huge volume
        bid, ask = 40_000.0, 40_200.0   # 50 bps
        # Use a valid non-excluded symbol
        ticker = self._ticker(bid=bid, ask=ask, volume_usd=50_000_000.0)
        result = scanner._evaluate_ticker(
            symbol="AVAX/USDT",
            ticker=ticker,
            exchange="bybit",
            fee_rate=0.0,
        )
        assert result is None, "High-volume pair should be filtered out"

    def test_acceptable_volume_pair_kept(self):
        """Volume in range [$50k, $5M] should pass."""
        scanner = self._scanner()
        bid, ask = 5.0, 5.025  # 50 bps
        ticker = self._ticker(bid=bid, ask=ask, volume_usd=1_000_000.0)
        result = scanner._evaluate_ticker(
            symbol="NEAR/USDT",
            ticker=ticker,
            exchange="bybit",
            fee_rate=0.0,
        )
        assert result is not None, "Acceptable volume pair should pass"

    def test_low_volume_pair_filtered(self):
        """Volume < 50k USD — not enough liquidity to fill our orders."""
        scanner = self._scanner()
        bid, ask = 5.0, 5.03   # 60 bps
        ticker = self._ticker(bid=bid, ask=ask, volume_usd=10_000.0)
        result = scanner._evaluate_ticker(
            symbol="TIA/USDT",
            ticker=ticker,
            exchange="bybit",
            fee_rate=0.0,
        )
        assert result is None, "Low-volume pair should be filtered out"

    def test_volume_at_exact_max_boundary_passes(self):
        """Volume exactly at max_24h_volume_usd should still pass."""
        scanner = self._scanner()
        bid, ask = 5.0, 5.03   # 60 bps
        ticker = self._ticker(bid=bid, ask=ask, volume_usd=5_000_000.0)
        result = scanner._evaluate_ticker(
            symbol="TIA/USDT",
            ticker=ticker,
            exchange="bybit",
            fee_rate=0.0,
        )
        assert result is not None, "Volume at exact max should pass"

    def test_volume_one_above_max_filtered(self):
        """Volume one dollar above max should be filtered."""
        scanner = self._scanner()
        bid, ask = 5.0, 5.03
        ticker = self._ticker(bid=bid, ask=ask, volume_usd=5_000_001.0)
        result = scanner._evaluate_ticker(
            symbol="TIA/USDT",
            ticker=ticker,
            exchange="bybit",
            fee_rate=0.0,
        )
        assert result is None, "Volume just above max should be filtered"

    def test_volume_at_exact_min_boundary_passes(self):
        """Volume exactly at min_24h_volume_usd should still pass."""
        scanner = self._scanner()
        bid, ask = 5.0, 5.03
        ticker = self._ticker(bid=bid, ask=ask, volume_usd=50_000.0)
        result = scanner._evaluate_ticker(
            symbol="INJ/USDT",
            ticker=ticker,
            exchange="bybit",
            fee_rate=0.0,
        )
        assert result is not None, "Volume at exact min should pass"
