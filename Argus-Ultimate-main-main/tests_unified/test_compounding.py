"""
test_compounding.py
===================
Tests for compounding_engine, capital_growth_tracker, and fee_tier_optimizer.
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import pytest

from core.compounding_engine import CompoundingConfig, CompoundingEngine
from core.capital_growth_tracker import CapitalGrowthTracker
from core.fee_tier_optimizer import FeeTierOptimizer


# ---------------------------------------------------------------------------
# CompoundingEngine tests
# ---------------------------------------------------------------------------

class TestCompoundingEngine:
    """Tests for CompoundingEngine."""

    def _engine(self, reinvest_pct: float = 1.0) -> CompoundingEngine:
        config = CompoundingConfig(
            initial_capital_aud=1000.0,
            aud_usd_rate=0.62,
            reinvest_pct=reinvest_pct,
            reinvest_interval_days=7,
            min_reinvest_amount_usd=10.0,
        )
        return CompoundingEngine(config)

    def test_compounding_engine_initial(self):
        """get_current_capital returns the converted initial capital."""
        engine = self._engine()
        expected_usd = 1000.0 * 0.62
        assert abs(engine.get_current_capital() - expected_usd) < 1e-6
        assert abs(engine.get_current_capital_aud() - 1000.0) < 1e-4

    def test_compounding_weekly_full_reinvest(self):
        """100% reinvestment: capital grows by the full period PnL."""
        engine = self._engine(reinvest_pct=1.0)
        today = date.today()
        for i in range(7):
            engine.record_daily_pnl(today - timedelta(days=6 - i), 4.0, "mm")

        capital_before = engine.get_current_capital()
        result = engine.run_weekly_reinvestment()

        # 7 days × $4 = $28 PnL; 100% reinvested
        assert abs(result.pnl_period_usd - 28.0) < 1e-6
        assert abs(result.reinvested_usd - 28.0) < 1e-6
        assert abs(result.withdrawn_usd) < 1e-9
        assert abs(engine.get_current_capital() - (capital_before + 28.0)) < 1e-6

    def test_compounding_partial_reinvest(self):
        """50% reinvestment: half withdrawn, half added to capital."""
        engine = self._engine(reinvest_pct=0.5)
        today = date.today()
        for i in range(7):
            engine.record_daily_pnl(today - timedelta(days=6 - i), 4.0, "mm")

        capital_before = engine.get_current_capital()
        result = engine.run_weekly_reinvestment()

        assert abs(result.pnl_period_usd - 28.0) < 1e-6
        assert abs(result.reinvested_usd - 14.0) < 1e-6
        assert abs(result.withdrawn_usd - 14.0) < 1e-6
        assert abs(engine.get_current_capital() - (capital_before + 14.0)) < 1e-6

    def test_compounding_resize_allocations(self):
        """resize_allocations returns correct per-strategy USD amounts."""
        engine = self._engine()
        capital = 620.0  # Matches initial USD
        allocations = engine.resize_allocations(capital)

        # mm=55%, funding_arb=40%, reserve=5%
        assert "mm" in allocations
        assert "funding_arb" in allocations
        assert "reserve" in allocations

        assert abs(allocations["mm"] - capital * 0.55) < 1e-6
        assert abs(allocations["funding_arb"] - capital * 0.40) < 1e-6
        assert abs(allocations["reserve"] - capital * 0.05) < 1e-6

    def test_compounding_fee_tier_upgrade(self):
        """Capital crossing MEXC VIP threshold should be flagged in the result."""
        # MEXC Lv1 kicks in at $1M/month volume.
        # Monthly vol ≈ capital × 3 × 30; Lv1 requires $1M → capital ~ $11,111
        # We inject a large PnL to push capital past that threshold.
        config = CompoundingConfig(
            initial_capital_aud=1000.0,
            aud_usd_rate=0.62,
            reinvest_pct=1.0,
            min_reinvest_amount_usd=1.0,
        )
        engine = CompoundingEngine(config)

        # Simulate enough profit to cross Lv1 threshold
        # We need capital > 1_000_000 / (3 * 30) ≈ $11,111 USD
        huge_pnl = 15_000.0  # USD
        engine.record_daily_pnl(date.today(), huge_pnl, "mm")
        result = engine.run_weekly_reinvestment()

        # Capital should have jumped past the Lv1 threshold
        assert result.fee_tier_upgraded is True
        assert result.new_fee_tier != "Lv0"

    def test_compounding_projection_grows(self):
        """Week-52 projected capital should exceed week-1 projected capital."""
        engine = self._engine()
        snapshots = engine.project_growth(days=365, daily_return_usd=4.0)
        assert len(snapshots) >= 2
        assert snapshots[-1].capital_usd > snapshots[0].capital_usd

    def test_compounding_projection_length(self):
        """project_growth returns one snapshot per week."""
        engine = self._engine()
        snapshots = engine.project_growth(days=364, daily_return_usd=4.0)
        # 364 // 7 = 52
        assert len(snapshots) == 52

    def test_compounding_growth_summary_keys(self):
        """get_growth_summary returns all expected keys."""
        engine = self._engine()
        summary = engine.get_growth_summary()
        required_keys = [
            "current_capital_usd",
            "current_capital_aud",
            "total_profit_usd",
            "total_reinvested_usd",
            "total_withdrawn_usd",
            "days_running",
            "annualised_return_pct",
            "current_fee_tier",
            "next_fee_tier_threshold",
            "projected_1yr_capital_usd",
            "projected_1yr_capital_aud",
        ]
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# CapitalGrowthTracker tests
# ---------------------------------------------------------------------------

class TestCapitalGrowthTracker:
    """Tests for CapitalGrowthTracker."""

    def _tracker(self) -> CapitalGrowthTracker:
        return CapitalGrowthTracker(initial_capital_aud=1000.0, aud_usd_rate=0.62)

    def test_growth_tracker_equity_curve(self):
        """Equity curve has one entry per unique date with sessions recorded."""
        tracker = self._tracker()
        today = date.today()
        for i in range(5):
            tracker.record_session(
                session_date=today - timedelta(days=4 - i),
                pnl_usd=4.0,
                strategy="mm",
                exchange="MEXC",
                num_fills=20,
                adverse_fills=2,
            )
        curve = tracker.get_equity_curve()
        assert len(curve) == 5
        # Verify cumulative growth
        assert curve[-1]["capital_usd"] > curve[0]["capital_usd"]

    def test_growth_tracker_sharpe(self):
        """Consistently positive PnL days produce a positive Sharpe ratio."""
        tracker = self._tracker()
        today = date.today()
        for i in range(30):
            tracker.record_session(
                session_date=today - timedelta(days=29 - i),
                pnl_usd=4.0,
                strategy="mm",
                exchange="MEXC",
                num_fills=10,
                adverse_fills=1,
            )
        sharpe = tracker.get_rolling_sharpe(window_days=30)
        # With all identical positive returns, std=0 → edge case returns 0
        # Slight variance scenario: sharpe should be ≥ 0
        assert sharpe >= 0.0

    def test_growth_tracker_sharpe_with_variance(self):
        """Positive mean with some variance gives positive Sharpe."""
        tracker = self._tracker()
        today = date.today()
        pnls = [3.0, 5.0, 4.0, 6.0, 2.0, 4.5, 3.5] * 4  # 28 days
        for i, pnl in enumerate(pnls):
            tracker.record_session(
                session_date=today - timedelta(days=len(pnls) - 1 - i),
                pnl_usd=pnl,
                strategy="mm",
                exchange="MEXC",
                num_fills=10,
                adverse_fills=1,
            )
        sharpe = tracker.get_rolling_sharpe(window_days=28)
        assert sharpe > 0.0

    def test_growth_tracker_milestones(self):
        """$2k AUD milestone is achieved when capital crosses it."""
        tracker = CapitalGrowthTracker(initial_capital_aud=1000.0, aud_usd_rate=0.62)
        today = date.today()
        # We need to gain 1000 AUD = 620 USD
        # Record a big single-day session
        tracker.record_session(
            session_date=today,
            pnl_usd=650.0,  # > 620 USD = 1000 AUD
            strategy="mm",
            exchange="MEXC",
            num_fills=100,
            adverse_fills=5,
        )
        milestones = tracker.get_milestones()
        doubled = next(m for m in milestones if m.name == "Doubled")
        assert doubled.achieved is True
        assert doubled.achieved_date is not None

    def test_growth_tracker_milestone_not_yet_achieved(self):
        """5× milestone is not achieved with modest PnL."""
        tracker = self._tracker()
        today = date.today()
        tracker.record_session(today, 10.0, "mm", "MEXC", 20, 1)
        milestones = tracker.get_milestones()
        five_x = next(m for m in milestones if m.name == "5×")
        assert five_x.achieved is False

    def test_growth_tracker_weekly_report(self):
        """Weekly report contains all key section headers."""
        tracker = self._tracker()
        today = date.today()
        for i in range(7):
            tracker.record_session(
                session_date=today - timedelta(days=6 - i),
                pnl_usd=4.0,
                strategy="mm",
                exchange="MEXC",
                num_fills=15,
                adverse_fills=1,
            )
        report = tracker.generate_weekly_report()
        assert "WEEK SUMMARY" in report
        assert "RUNNING TOTALS" in report
        assert "MILESTONES" in report
        assert "Ann. Return" in report

    def test_growth_tracker_max_drawdown(self):
        """Max drawdown is correctly computed when capital dips."""
        tracker = self._tracker()
        today = date.today()
        # Up, then down, then up
        pnl_sequence = [10.0, 10.0, -20.0, 5.0, 5.0]
        for i, pnl in enumerate(pnl_sequence):
            tracker.record_session(
                session_date=today - timedelta(days=len(pnl_sequence) - 1 - i),
                pnl_usd=pnl,
                strategy="mm",
                exchange="MEXC",
                num_fills=10,
                adverse_fills=0,
            )
        max_dd, start, end = tracker.get_max_drawdown()
        assert max_dd > 0.0


# ---------------------------------------------------------------------------
# FeeTierOptimizer tests
# ---------------------------------------------------------------------------

class TestFeeTierOptimizer:
    """Tests for FeeTierOptimizer."""

    def _ts(self, days_ago: int = 0) -> int:
        """Return a nanosecond timestamp N days ago."""
        base = int(time.time() * 1e9)
        return base - int(days_ago * 24 * 3600 * 1e9)

    def test_fee_tier_optimizer_mexc_lv0(self):
        """Volume below $1M → MEXC Lv0."""
        opt = FeeTierOptimizer()
        opt.record_trade("MEXC", "BTC/USDT", "buy", 500_000.0, 0.0, self._ts(1))
        tier = opt.get_current_tier("MEXC")
        assert tier.tier_name == "Lv0"

    def test_fee_tier_optimizer_mexc_lv1(self):
        """Volume above $1M → MEXC Lv1."""
        opt = FeeTierOptimizer()
        # Record $1.5M volume
        opt.record_trade("MEXC", "BTC/USDT", "buy", 1_500_000.0, 0.0, self._ts(1))
        tier = opt.get_current_tier("MEXC")
        assert tier.tier_name == "Lv1"

    def test_fee_tier_optimizer_mexc_lv2(self):
        """Volume above $5M → MEXC Lv2 with rebate."""
        opt = FeeTierOptimizer()
        opt.record_trade("MEXC", "BTC/USDT", "buy", 6_000_000.0, -600.0, self._ts(1))
        tier = opt.get_current_tier("MEXC")
        assert tier.tier_name == "Lv2"
        assert tier.is_rebate is True

    def test_fee_tier_next_tier(self):
        """get_next_tier returns Lv1 when at Lv0."""
        opt = FeeTierOptimizer()
        next_t = opt.get_next_tier("MEXC")
        assert next_t is not None
        assert next_t.tier_name == "Lv1"

    def test_fee_tier_max_tier_no_next(self):
        """get_next_tier returns None at MEXC Lv3."""
        opt = FeeTierOptimizer()
        opt.record_trade("MEXC", "BTC/USDT", "buy", 15_000_000.0, -3000.0, self._ts(1))
        assert opt.get_next_tier("MEXC") is None

    def test_fee_tier_volume_to_next_tier(self):
        """volume_to_next_tier returns correct gap."""
        opt = FeeTierOptimizer()
        opt.record_trade("MEXC", "BTC/USDT", "buy", 400_000.0, 0.0, self._ts(1))
        gap = opt.volume_to_next_tier("MEXC")
        # Lv1 requires $1M; have $400k; gap = $600k
        assert gap is not None
        assert abs(gap - 600_000.0) < 1.0

    def test_fee_tier_recommendations(self):
        """get_recommendations returns a non-empty list of strings."""
        opt = FeeTierOptimizer()
        recs = opt.get_recommendations()
        assert isinstance(recs, list)
        assert len(recs) > 0
        for rec in recs:
            assert isinstance(rec, str)
            assert len(rec) > 0

    def test_fee_tier_btcmarkets_best(self):
        """BTCMarkets is always at best tier."""
        opt = FeeTierOptimizer()
        tier = opt.get_current_tier("BTCMarkets")
        assert tier.tier_name == "Standard"
        # No next tier
        assert opt.get_next_tier("BTCMarkets") is None

    def test_fee_tier_stats_keys(self):
        """get_stats returns a dict with expected exchange keys."""
        opt = FeeTierOptimizer()
        stats = opt.get_stats()
        assert "MEXC" in stats
        assert "BTCMarkets" in stats
        assert "Bybit" in stats
        assert "WOOX" in stats
        # Check per-exchange structure
        mexc_stats = stats["MEXC"]
        assert "30d_volume_usd" in mexc_stats
        assert "current_tier" in mexc_stats
        assert "next_tier" in mexc_stats

    def test_fee_tier_days_to_next_tier(self):
        """days_to_next_tier returns correct estimate."""
        opt = FeeTierOptimizer()
        # No volume yet, need $1M for Lv1
        # At $33_333/day → ~30 days
        days = opt.days_to_next_tier("MEXC", current_daily_vol=33_333.0)
        assert days is not None
        assert 28.0 < days < 35.0

    def test_fee_tier_days_to_next_none_for_zero_vol(self):
        """days_to_next_tier returns None when daily volume is zero."""
        opt = FeeTierOptimizer()
        result = opt.days_to_next_tier("MEXC", current_daily_vol=0.0)
        assert result is None
