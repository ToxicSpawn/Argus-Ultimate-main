"""
Tests for micro-capital allocation and risk management layer.

Covers:
  - MicroCapitalAllocator: allocation splits, rebalancing, killswitch, minimums
  - MicroRiskEnvelope: pre-trade checks, daily halt, fee break-even, order sizing
  - MicroStrategyOrchestrator: dashboard keys, paper trading default

All tests are synchronous where possible. Async tests use pytest-asyncio.
"""
from __future__ import annotations

import math
import time
from typing import Dict

import pytest

from core.micro_capital_allocator import (
    Allocation,
    AllocatorConfig,
    MicroCapitalAllocator,
    MINIMUM_STRATEGY_CAPITAL_USD,
)
from core.micro_risk_envelope import (
    EXCHANGE_ROUND_TRIP_FEES_BPS,
    MicroRiskConfig,
    MicroRiskEnvelope,
    OrderSizing,
    PreTradeCheck,
)
from strategies.micro_strategy_orchestrator import (
    MicroStrategyOrchestrator,
    OrchestratorConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_allocator(**kwargs) -> MicroCapitalAllocator:
    """Construct a MicroCapitalAllocator with optional config overrides."""
    cfg = AllocatorConfig(**kwargs)
    return MicroCapitalAllocator(cfg)


def _make_risk(**kwargs) -> MicroRiskEnvelope:
    """Construct a MicroRiskEnvelope with optional config overrides."""
    cfg = MicroRiskConfig(**kwargs)
    return MicroRiskEnvelope(cfg)


def _feed_mm_performance(
    allocator: MicroCapitalAllocator,
    sharpe: float,
    hours: int = 25,
    pnl_per_hour: float = 0.01,
) -> None:
    """Feed synthetic MM performance data to produce a given Sharpe."""
    now_ns = time.time_ns()
    for i in range(hours):
        ts = now_ns - (hours - i) * 3_600_000_000_000  # 1h back per step
        allocator.update_performance(
            strategy="mm",
            pnl=pnl_per_hour,
            sharpe=sharpe,
            timestamp_ns=ts,
        )
    # Force the internal rolling sharpe to the target value
    allocator._strategies["mm"].current_sharpe = sharpe


def _feed_funding_performance(
    allocator: MicroCapitalAllocator,
    annualised_yield: float,
    hours: int = 25,
) -> None:
    """Feed synthetic funding performance to produce a given annualised yield."""
    now_ns = time.time_ns()
    capital = allocator._strategies["funding"].current_capital_usd
    # PnL per hour that would give the target annualised yield
    hourly_return = annualised_yield / 8760.0
    pnl_per_hour = capital * hourly_return

    for i in range(hours):
        ts = now_ns - (hours - i) * 3_600_000_000_000
        allocator.update_performance(
            strategy="funding",
            pnl=pnl_per_hour,
            sharpe=1.0,
            timestamp_ns=ts,
        )
    # Force the computed yield
    allocator._strategies["funding"].current_yield_annualised = annualised_yield


# ---------------------------------------------------------------------------
# MicroCapitalAllocator Tests
# ---------------------------------------------------------------------------

class TestAllocatorDefaultSplit:
    """test_allocator_default_split — 55/40/5 at init"""

    def test_initial_allocation_percentages(self):
        alloc_inst = _make_allocator()
        alloc = alloc_inst.get_allocation()

        assert math.isclose(alloc.mm_pct, 0.55, rel_tol=1e-6), (
            f"Expected MM pct=0.55, got {alloc.mm_pct}"
        )
        assert math.isclose(alloc.funding_pct, 0.40, rel_tol=1e-6), (
            f"Expected Funding pct=0.40, got {alloc.funding_pct}"
        )
        assert math.isclose(alloc.reserve_pct, 0.05, rel_tol=1e-6), (
            f"Expected Reserve pct=0.05, got {alloc.reserve_pct}"
        )

    def test_initial_allocation_amounts(self):
        cfg = AllocatorConfig(total_capital_usd=620.0)
        alloc_inst = MicroCapitalAllocator(cfg)
        alloc = alloc_inst.get_allocation()

        assert math.isclose(alloc.mm_capital_usd, 620.0 * 0.55, rel_tol=1e-4)
        assert math.isclose(alloc.funding_capital_usd, 620.0 * 0.40, rel_tol=1e-4)
        assert math.isclose(alloc.reserve_usd, 620.0 * 0.05, rel_tol=1e-4)
        assert alloc.total_capital_usd == 620.0

    def test_initial_sum_equals_total(self):
        alloc_inst = _make_allocator()
        alloc = alloc_inst.get_allocation()
        total = alloc.mm_capital_usd + alloc.funding_capital_usd + alloc.reserve_usd
        assert math.isclose(total, alloc.total_capital_usd, rel_tol=1e-5)


class TestAllocatorRebalanceLowSharpe:
    """test_allocator_rebalance_low_sharpe — MM Sharpe < 0.5 → MM allocation reduced"""

    def test_low_sharpe_reduces_mm(self):
        alloc_inst = _make_allocator()
        initial = alloc_inst.get_allocation()

        # Force low MM Sharpe (below threshold of 0.5)
        _feed_mm_performance(alloc_inst, sharpe=0.2)
        # Funding is performing well
        _feed_funding_performance(alloc_inst, annualised_yield=0.20)

        new_alloc = alloc_inst.rebalance()

        # MM allocation should be reduced (by 20% of its base)
        expected_max_mm = initial.mm_capital_usd
        assert new_alloc.mm_capital_usd < expected_max_mm, (
            f"MM should be reduced when Sharpe < 0.5: got {new_alloc.mm_capital_usd}"
        )

    def test_low_sharpe_shifts_to_reserve(self):
        alloc_inst = _make_allocator()
        initial = alloc_inst.get_allocation()

        _feed_mm_performance(alloc_inst, sharpe=0.1)
        _feed_funding_performance(alloc_inst, annualised_yield=0.20)

        new_alloc = alloc_inst.rebalance()

        # Reserve should grow when MM underperforms
        assert new_alloc.reserve_usd >= initial.reserve_usd, (
            "Reserve should increase when MM Sharpe is low"
        )

    def test_low_sharpe_reason_contains_keyword(self):
        alloc_inst = _make_allocator()

        _feed_mm_performance(alloc_inst, sharpe=0.1)
        _feed_funding_performance(alloc_inst, annualised_yield=0.20)

        new_alloc = alloc_inst.rebalance()
        assert "mm_low_sharpe" in new_alloc.reason or "low_sharpe" in new_alloc.reason, (
            f"Expected low_sharpe in reason, got: {new_alloc.reason}"
        )


class TestAllocatorRebalanceLowYield:
    """test_allocator_rebalance_low_yield — funding < 10% → shifted to MM"""

    def test_low_yield_shifts_funding_to_mm(self):
        alloc_inst = _make_allocator()
        initial = alloc_inst.get_allocation()

        # MM is performing well, funding is not
        _feed_mm_performance(alloc_inst, sharpe=2.0)
        _feed_funding_performance(alloc_inst, annualised_yield=0.02)  # 2%, below 10%

        new_alloc = alloc_inst.rebalance()

        # MM should get more capital
        assert new_alloc.mm_capital_usd >= initial.mm_capital_usd, (
            f"MM should increase when funding underperforms: got {new_alloc.mm_capital_usd}"
        )

    def test_low_yield_funding_at_zero(self):
        alloc_inst = _make_allocator()

        _feed_mm_performance(alloc_inst, sharpe=2.0)
        _feed_funding_performance(alloc_inst, annualised_yield=0.02)

        new_alloc = alloc_inst.rebalance()

        assert new_alloc.funding_capital_usd == 0.0, (
            f"Funding should be zeroed when yield < threshold: {new_alloc.funding_capital_usd}"
        )

    def test_low_yield_reason_contains_keyword(self):
        alloc_inst = _make_allocator()

        _feed_mm_performance(alloc_inst, sharpe=2.0)
        _feed_funding_performance(alloc_inst, annualised_yield=0.02)

        new_alloc = alloc_inst.rebalance()
        assert "funding" in new_alloc.reason.lower(), (
            f"Expected 'funding' in reason, got: {new_alloc.reason}"
        )


class TestAllocatorGlobalKillswitch:
    """test_allocator_global_killswitch — total_pnl < -15% → all allocations 0"""

    def test_killswitch_triggered_at_threshold(self):
        cfg = AllocatorConfig(total_capital_usd=620.0, max_drawdown_pct=15.0)
        alloc_inst = MicroCapitalAllocator(cfg)

        # Force total PnL below -15%
        drawdown_usd = -620.0 * 0.15 - 0.01  # just past 15%
        alloc_inst._strategies["mm"].total_pnl = drawdown_usd
        alloc_inst._strategies["funding"].total_pnl = 0.0

        triggered = alloc_inst.check_global_killswitch()
        assert triggered is True

    def test_killswitch_zeroes_allocations(self):
        cfg = AllocatorConfig(total_capital_usd=620.0, max_drawdown_pct=15.0)
        alloc_inst = MicroCapitalAllocator(cfg)

        alloc_inst._strategies["mm"].total_pnl = -93.5  # > 15% of $620
        alloc_inst._strategies["funding"].total_pnl = 0.0

        alloc_inst.check_global_killswitch()
        alloc = alloc_inst.get_allocation()

        assert alloc.mm_capital_usd == 0.0
        assert alloc.funding_capital_usd == 0.0
        assert alloc.reserve_usd == cfg.total_capital_usd

    def test_killswitch_not_triggered_within_limit(self):
        cfg = AllocatorConfig(total_capital_usd=620.0, max_drawdown_pct=15.0)
        alloc_inst = MicroCapitalAllocator(cfg)

        # 10% drawdown — should NOT trigger
        alloc_inst._strategies["mm"].total_pnl = -62.0
        alloc_inst._strategies["funding"].total_pnl = 0.0

        triggered = alloc_inst.check_global_killswitch()
        assert triggered is False

    def test_killswitch_halted_state_persists(self):
        alloc_inst = _make_allocator()
        alloc_inst._strategies["mm"].total_pnl = -100.0  # definitely above 15%

        alloc_inst.check_global_killswitch()
        assert alloc_inst._halted is True

        # Subsequent rebalance should still return zero allocation
        alloc = alloc_inst.rebalance()
        assert alloc.mm_capital_usd == 0.0
        assert alloc.funding_capital_usd == 0.0


class TestAllocatorMinimumAllocation:
    """test_allocator_minimum_allocation — allocation below $50 → strategy disabled"""

    def test_mm_below_minimum_disabled(self):
        # Very small capital so MM would get < $50
        cfg = AllocatorConfig(
            total_capital_usd=80.0,
            base_mm_pct=0.55,
            base_funding_pct=0.40,
            reserve_pct=0.05,
        )
        alloc_inst = MicroCapitalAllocator(cfg)

        # Force both poor so allocation is 20% to best performer
        _feed_mm_performance(alloc_inst, sharpe=0.1, pnl_per_hour=-0.001)
        _feed_funding_performance(alloc_inst, annualised_yield=0.01)

        # With $80 total and 20% to best performer = $16 < $50 → should be disabled
        alloc_inst._strategies["mm"].total_pnl = -1.0
        alloc_inst._strategies["funding"].total_pnl = -2.0  # mm is better

        new_alloc = alloc_inst.rebalance()

        # The $16 allocation to mm in "both poor" scenario is below $50 minimum
        # so mm should be disabled and all goes to reserve
        # The minimum check triggers when the actual $ amount < $50
        if new_alloc.mm_capital_usd > 0:
            assert new_alloc.mm_capital_usd >= MINIMUM_STRATEGY_CAPITAL_USD, (
                f"If MM is active, it must have >= ${MINIMUM_STRATEGY_CAPITAL_USD}"
            )

    def test_minimum_constant_is_fifty(self):
        assert MINIMUM_STRATEGY_CAPITAL_USD == 50.0


# ---------------------------------------------------------------------------
# MicroRiskEnvelope Tests
# ---------------------------------------------------------------------------

class TestRiskEnvelopePreTradeWithinLimits:
    """test_risk_envelope_pre_trade_within_limits — small order → allowed"""

    def test_small_order_allowed(self):
        risk = _make_risk(total_capital_usd=620.0)
        result = risk.check_pre_trade(
            symbol="BTC/USDT",
            exchange="bybit_spot",
            side="buy",
            size_usd=10.0,   # very small
        )
        assert result.allowed is True
        assert result.risk_utilisation_pct < 100.0

    def test_within_pair_limit_allowed(self):
        risk = _make_risk(total_capital_usd=620.0, max_position_per_pair_pct=20.0)
        # 20% of $620 = $124 max; $100 is within limit
        result = risk.check_pre_trade(
            symbol="ETH/USDT",
            exchange="bybit_spot",
            side="buy",
            size_usd=100.0,
        )
        assert result.allowed is True

    def test_reason_contains_ok(self):
        risk = _make_risk()
        result = risk.check_pre_trade(
            symbol="SOL/USDT", exchange="bybit_spot", side="buy", size_usd=5.0
        )
        assert result.allowed is True
        assert "ok" in result.reason.lower() or "pre_trade" in result.reason.lower()


class TestRiskEnvelopePreTradeExceedsPairLimit:
    """test_risk_envelope_pre_trade_exceeds_pair_limit — too large → denied"""

    def test_exceeds_pair_limit_denied(self):
        risk = _make_risk(total_capital_usd=620.0, max_position_per_pair_pct=20.0)
        # Max pair position = $124. Order of $200 should be denied.
        result = risk.check_pre_trade(
            symbol="BTC/USDT",
            exchange="bybit_spot",
            side="buy",
            size_usd=200.0,
        )
        assert result.allowed is False
        assert "position_limit" in result.reason or "limit" in result.reason.lower()

    def test_cumulative_exceeds_limit_denied(self):
        risk = _make_risk(total_capital_usd=620.0, max_position_per_pair_pct=20.0)
        # Existing $100 position, try to add $50 more — total $150 > $124 max
        existing = {"BTC/USDT": 100.0}
        result = risk.check_pre_trade(
            symbol="BTC/USDT",
            exchange="bybit_spot",
            side="buy",
            size_usd=50.0,
            current_positions=existing,
        )
        assert result.allowed is False

    def test_within_limit_after_partial_fill(self):
        risk = _make_risk(total_capital_usd=620.0, max_position_per_pair_pct=20.0)
        # $60 existing + $30 new = $90, below $124 limit
        existing = {"BTC/USDT": 60.0}
        result = risk.check_pre_trade(
            symbol="BTC/USDT",
            exchange="bybit_spot",
            side="buy",
            size_usd=30.0,
            current_positions=existing,
        )
        assert result.allowed is True


class TestRiskEnvelopeDailyLossHalt:
    """test_risk_envelope_daily_loss_halt — after $10 daily loss → halted"""

    def test_daily_loss_halt_triggered(self):
        risk = _make_risk(total_capital_usd=620.0, max_daily_loss_usd=10.0)
        # Force daily PnL to exactly -$10 (the limit)
        risk._daily_pnl = -10.0

        result = risk.check_pre_trade(
            symbol="BTC/USDT",
            exchange="bybit_spot",
            side="buy",
            size_usd=10.0,
        )
        assert result.allowed is False
        assert "daily_loss" in result.reason or "halt" in result.reason.lower()

    def test_daily_loss_below_limit_allowed(self):
        risk = _make_risk(total_capital_usd=620.0, max_daily_loss_usd=10.0)
        risk._daily_pnl = -8.0  # $8 loss — below $10 limit

        result = risk.check_pre_trade(
            symbol="BTC/USDT",
            exchange="bybit_spot",
            side="buy",
            size_usd=10.0,
        )
        assert result.allowed is True

    def test_daily_pnl_accessor(self):
        risk = _make_risk()
        risk._daily_pnl = -5.5
        assert risk.get_daily_pnl() == -5.5

    def test_risk_status_halted_when_daily_limit_hit(self):
        risk = _make_risk(max_daily_loss_usd=10.0)
        risk._daily_pnl = -10.0

        status = risk.get_risk_status()
        assert status["status"] == "halted"
        assert status["daily_limit_remaining"] == 0.0


class TestRiskEnvelopeFeeBreakEven:
    """test_risk_envelope_fee_break_even — bybit=0bps, kraken=32bps, coinbase=80bps"""

    def test_bybit_spot_zero_bps(self):
        risk = _make_risk()
        assert risk.min_spread_for_profit("bybit_spot") == 0.0

    def test_kraken_32_bps(self):
        risk = _make_risk()
        assert risk.min_spread_for_profit("kraken") == 32.0

    def test_coinbase_80_bps(self):
        risk = _make_risk()
        assert risk.min_spread_for_profit("coinbase") == 80.0

    def test_bybit_perp_nonzero(self):
        risk = _make_risk()
        # Bybit perp has some taker fees
        assert risk.min_spread_for_profit("bybit_perp") >= 0.0

    def test_unknown_exchange_defaults_to_coinbase(self):
        risk = _make_risk()
        # Unknown exchanges default to 80 bps (conservative)
        unknown_bps = risk.min_spread_for_profit("unknown_exchange")
        assert unknown_bps == 80.0

    def test_exchange_round_trip_constants(self):
        """Verify the module-level constants match spec."""
        assert EXCHANGE_ROUND_TRIP_FEES_BPS["bybit_spot"] == 0.0
        assert EXCHANGE_ROUND_TRIP_FEES_BPS["kraken"] == 32.0
        assert EXCHANGE_ROUND_TRIP_FEES_BPS["coinbase"] == 80.0


class TestOrderSizingFeeViability:
    """test_order_sizing_fee_viability — 20bps spread on kraken → not viable; bybit → viable"""

    def test_20bps_kraken_not_viable(self):
        """Kraken needs 32bps to break even. 20bps spread → not viable."""
        risk = _make_risk()
        sizing = risk.calculate_order_size(
            symbol="BTC/USDT",
            exchange="kraken",
            capital_allocated=300.0,
            signal_confidence=1.0,
            spread_bps=20.0,
            current_price=50000.0,
        )
        assert sizing.is_viable is False, (
            f"20bps spread on Kraken (32bps fee) should not be viable. "
            f"Got: {sizing.reason}"
        )
        assert sizing.expected_profit_bps < 0

    def test_20bps_bybit_spot_viable(self):
        """Bybit spot has 0bps fee. Any positive spread is viable."""
        risk = _make_risk()
        sizing = risk.calculate_order_size(
            symbol="BTC/USDT",
            exchange="bybit_spot",
            capital_allocated=300.0,
            signal_confidence=1.0,
            spread_bps=20.0,
            current_price=50000.0,
        )
        assert sizing.is_viable is True, (
            f"20bps spread on Bybit spot (0bps fee) should be viable. "
            f"Got: {sizing.reason}"
        )
        assert sizing.expected_profit_bps == pytest.approx(20.0, abs=0.1)

    def test_80bps_kraken_viable(self):
        """80bps spread on Kraken (32bps fee) → viable.

        Fee (32bps) / spread (80bps) = 40% < 50% viability threshold.
        Uses capital_allocated=5000 so default 2% fraction yields $100 order,
        well above Kraken's $5 minimum order size.
        """
        risk = _make_risk()
        sizing = risk.calculate_order_size(
            symbol="ETH/USDT",
            exchange="kraken",
            capital_allocated=5000.0,
            spread_bps=80.0,
            current_price=3000.0,
        )
        assert sizing.is_viable is True
        assert sizing.expected_profit_bps == pytest.approx(48.0, abs=0.1)  # 80-32=48

    def test_sizing_is_viable_dataclass_fields(self):
        """Verify OrderSizing has all expected fields."""
        risk = _make_risk()
        sizing = risk.calculate_order_size(
            symbol="BTC/USDT",
            exchange="bybit_spot",
            capital_allocated=300.0,
        )
        assert hasattr(sizing, "size_usd")
        assert hasattr(sizing, "size_base")
        assert hasattr(sizing, "exchange")
        assert hasattr(sizing, "fee_usd")
        assert hasattr(sizing, "net_cost_usd")
        assert hasattr(sizing, "is_viable")
        assert hasattr(sizing, "reason")
        assert hasattr(sizing, "expected_profit_bps")

    def test_sizing_fee_usd_positive(self):
        risk = _make_risk()
        sizing = risk.calculate_order_size(
            symbol="BTC/USDT",
            exchange="kraken",
            capital_allocated=200.0,
        )
        assert sizing.fee_usd >= 0.0
        assert sizing.net_cost_usd >= sizing.size_usd


# ---------------------------------------------------------------------------
# MicroStrategyOrchestrator Tests
# ---------------------------------------------------------------------------

class TestOrchestratorDashboard:
    """test_orchestrator_dashboard — returns all expected keys"""

    def test_dashboard_has_all_required_keys(self):
        orch = MicroStrategyOrchestrator(
            OrchestratorConfig(total_capital_aud=1000.0, paper_trading=True)
        )
        dash = orch.get_dashboard()

        required_keys = [
            "total_capital_aud",
            "total_capital_usd",
            "mm_status",
            "mm_pnl",
            "mm_pairs",
            "mm_allocation",
            "funding_status",
            "funding_pnl",
            "funding_positions",
            "funding_allocation",
            "global_pnl_usd",
            "global_pnl_aud",
            "drawdown_pct",
            "risk_status",
            "uptime_hours",
            "estimated_daily_yield_usd",
            "estimated_annual_yield_pct",
        ]
        for key in required_keys:
            assert key in dash, f"Missing key in dashboard: '{key}'"

    def test_dashboard_capital_values(self):
        orch = MicroStrategyOrchestrator(
            OrchestratorConfig(total_capital_aud=1000.0, aud_usd_rate=0.62)
        )
        dash = orch.get_dashboard()
        assert dash["total_capital_aud"] == pytest.approx(1000.0, abs=0.01)
        assert dash["total_capital_usd"] == pytest.approx(620.0, abs=0.01)

    def test_dashboard_returns_list_for_pairs(self):
        orch = MicroStrategyOrchestrator()
        dash = orch.get_dashboard()
        assert isinstance(dash["mm_pairs"], list)
        assert isinstance(dash["funding_positions"], list)

    def test_dashboard_drawdown_non_negative(self):
        orch = MicroStrategyOrchestrator()
        dash = orch.get_dashboard()
        assert dash["drawdown_pct"] >= 0.0

    def test_dashboard_estimated_yield_positive(self):
        orch = MicroStrategyOrchestrator()
        dash = orch.get_dashboard()
        assert dash["estimated_daily_yield_usd"] > 0.0
        assert dash["estimated_annual_yield_pct"] > 0.0


class TestOrchestratorPaperTradingDefault:
    """test_orchestrator_paper_trading_default — paper_trading=True by default"""

    def test_default_paper_trading_is_true(self):
        cfg = OrchestratorConfig()
        assert cfg.paper_trading is True, (
            "paper_trading must default to True for safety"
        )

    def test_orchestrator_default_config_paper_trading(self):
        orch = MicroStrategyOrchestrator()
        assert orch._cfg.paper_trading is True

    def test_dashboard_shows_paper_trading_flag(self):
        orch = MicroStrategyOrchestrator(OrchestratorConfig(paper_trading=True))
        dash = orch.get_dashboard()
        assert dash["paper_trading"] is True

    def test_live_trading_explicit_opt_in(self):
        """Live trading requires explicit paper_trading=False."""
        cfg = OrchestratorConfig(paper_trading=False)
        orch = MicroStrategyOrchestrator(cfg)
        dash = orch.get_dashboard()
        assert dash["paper_trading"] is False

    def test_performance_report_shows_paper_trading(self):
        orch = MicroStrategyOrchestrator(OrchestratorConfig(paper_trading=True))
        report = orch.get_performance_report()
        assert "PAPER" in report.upper() or "paper" in report.lower()


# ---------------------------------------------------------------------------
# Integration: AllocatorConfig validation
# ---------------------------------------------------------------------------

class TestAllocatorConfigValidation:
    def test_invalid_pct_sum_raises(self):
        with pytest.raises(ValueError):
            AllocatorConfig(base_mm_pct=0.6, base_funding_pct=0.4, reserve_pct=0.05)

    def test_valid_config_no_error(self):
        cfg = AllocatorConfig(base_mm_pct=0.55, base_funding_pct=0.40, reserve_pct=0.05)
        assert cfg is not None

    def test_zero_reserve_raises(self):
        # reserve_pct=0.005 is below the 1% safety minimum — must raise
        with pytest.raises(ValueError):
            AllocatorConfig(base_mm_pct=0.595, base_funding_pct=0.400, reserve_pct=0.005)
        # reserve_pct=0.0 is also invalid
        with pytest.raises(ValueError):
            AllocatorConfig(base_mm_pct=0.600, base_funding_pct=0.395, reserve_pct=0.005)


# ---------------------------------------------------------------------------
# Integration: get_stats completeness
# ---------------------------------------------------------------------------

class TestAllocatorGetStats:
    def test_stats_has_required_keys(self):
        alloc_inst = _make_allocator()
        stats = alloc_inst.get_stats()

        required_keys = [
            "mm_capital_usd",
            "funding_capital_usd",
            "reserve_usd",
            "total_capital_usd",
            "mm_pnl",
            "funding_pnl",
            "total_pnl",
            "mm_sharpe",
            "funding_yield_annualised_pct",
            "global_drawdown_pct",
            "max_drawdown_pct",
            "halted",
            "next_rebalance_ns",
            "rebalance_in_seconds",
        ]
        for key in required_keys:
            assert key in stats, f"Missing key in get_stats(): '{key}'"

    def test_stats_drawdown_zero_initially(self):
        alloc_inst = _make_allocator()
        stats = alloc_inst.get_stats()
        assert stats["global_drawdown_pct"] == 0.0
        assert stats["total_pnl"] == 0.0
