"""
Push 88 — Tests: BanditRouter
"""
from __future__ import annotations
import pytest
from strategies.bandit_router import BanditRouter, StrategySpec, StrategyStatus, AllocationRecord
from strategies.regime_consensus import MarketRegime

DEFAULT_SPECS = [
    StrategySpec(name="funding_rate_arb", category="arb"),
    StrategySpec(name="kalman_pairs",     category="mean_reversion"),
    StrategySpec(name="micro_capital_mm", category="scalping"),
    StrategySpec(name="momentum",          category="momentum"),
]

def make_router(capital: float = 10_000.0, **kwargs) -> BanditRouter:
    return BanditRouter(total_capital_usd=capital, strategies=list(DEFAULT_SPECS), ledger=None, **kwargs)


class TestBanditRouterInit:
    def test_invalid_capital_raises(self):
        with pytest.raises(ValueError):
            BanditRouter(total_capital_usd=0, strategies=DEFAULT_SPECS)

    def test_invalid_concentration_raises(self):
        with pytest.raises(ValueError):
            BanditRouter(total_capital_usd=1000, strategies=DEFAULT_SPECS, max_concentration=0.0)

    def test_registers_all_strategies(self):
        assert set(make_router().allocations().keys()) == {s.name for s in DEFAULT_SPECS}


class TestAllocations:
    def test_allocations_sum_to_capital(self):
        assert sum(make_router(capital=10_000.0).allocations().values()) == pytest.approx(10_000.0, rel=1e-4)

    def test_each_strategy_gets_positive_allocation(self):
        for v in make_router().allocations().values():
            assert v >= 0.0

    def test_concentration_cap_respected(self):
        r = make_router(max_concentration=0.40)
        for _ in range(200):
            r.record_outcome("funding_rate_arb", 100.0)
            r.record_outcome("momentum", -1.0)
        for name, usd in r.allocations().items():
            assert usd / 10_000.0 <= 0.40 + 1e-6

    def test_floor_applied(self):
        r = make_router(min_alloc_usd=100.0)
        for name, usd in r.allocations().items():
            if r._specs[name].status == StrategyStatus.ACTIVE:
                assert usd >= 99.99

    def test_capital_override(self):
        assert sum(make_router().allocations(capital_override=5_000.0).values()) == pytest.approx(5_000.0, rel=1e-4)


class TestRegimeAwareness:
    def test_range_regime_boosts_mean_reversion(self):
        r = make_router()
        assert r.allocations(regime=MarketRegime.RANGE)["kalman_pairs"] >= r.allocations(regime=MarketRegime.UNKNOWN)["kalman_pairs"] * 0.9

    def test_high_vol_disables_scalping(self):
        assert make_router().allocations(regime=MarketRegime.HIGH_VOL)["micro_capital_mm"] == pytest.approx(0.0, abs=1e-2)

    def test_trend_up_boosts_momentum(self):
        r = make_router()
        assert r.allocations(regime=MarketRegime.TREND_UP)["momentum"] >= r.allocations(regime=MarketRegime.UNKNOWN)["momentum"] * 0.9


class TestBanditLearning:
    def test_winner_gets_more_capital_over_time(self):
        r = make_router()
        for _ in range(150):
            r.record_outcome("funding_rate_arb", 50.0)
            r.record_outcome("momentum", -10.0)
        assert r.allocations()["funding_rate_arb"] > r.allocations()["momentum"]

    def test_unknown_strategy_is_ignored(self):
        r = make_router()
        r.record_outcome("ghost", 999.0)
        assert "ghost" not in r.allocations()


class TestKillSwitch:
    def test_strategy_suspended_on_poor_sharpe(self):
        r = make_router(sharpe_kill_threshold=100.0, kill_lookback_h=1.0)
        for _ in range(15):
            r.record_outcome("momentum", -100.0)
        assert r._specs["momentum"].status == StrategyStatus.SUSPENDED

    def test_suspended_strategy_gets_zero_allocation(self):
        r = make_router(sharpe_kill_threshold=100.0, kill_lookback_h=1.0)
        for _ in range(15):
            r.record_outcome("momentum", -100.0)
        assert r.allocations()["momentum"] == pytest.approx(0.0, abs=1e-6)

    def test_strategy_resumes_after_recovery(self):
        r = make_router(sharpe_kill_threshold=100.0, sharpe_resume_threshold=-999.0, kill_lookback_h=1.0)
        for _ in range(15):
            r.record_outcome("momentum", -100.0)
        assert r._specs["momentum"].status == StrategyStatus.SUSPENDED
        r.record_outcome("momentum", 1.0)
        assert r._specs["momentum"].status == StrategyStatus.ACTIVE

    def test_manual_disable(self):
        r = make_router()
        r.set_status("momentum", StrategyStatus.DISABLED)
        assert r.allocations()["momentum"] == pytest.approx(0.0, abs=1e-6)

    def test_set_status_unknown_strategy_raises(self):
        with pytest.raises(KeyError):
            make_router().set_status("ghost", StrategyStatus.DISABLED)


class TestSnapshot:
    def test_snapshot_returns_all_strategies(self):
        assert {r.strategy for r in make_router().snapshot()} == {s.name for s in DEFAULT_SPECS}

    def test_snapshot_sorted_by_allocation_desc(self):
        r = make_router()
        for _ in range(100):
            r.record_outcome("funding_rate_arb", 100.0)
        allocs = [rec.allocated_usd for rec in r.snapshot()]
        assert allocs == sorted(allocs, reverse=True)

    def test_snapshot_fields_present(self):
        for rec in make_router().snapshot():
            assert isinstance(rec, AllocationRecord)
            assert rec.allocated_usd >= 0.0


class TestCapitalManagement:
    def test_update_capital(self):
        r = make_router(capital=10_000.0)
        r.update_capital(20_000.0)
        assert sum(r.allocations().values()) == pytest.approx(20_000.0, rel=1e-4)

    def test_register_new_strategy(self):
        r = make_router()
        r.register(StrategySpec(name="new_strat", category="arb"))
        assert "new_strat" in r.allocations()

    def test_invalid_capital_update_raises(self):
        with pytest.raises(ValueError):
            make_router().update_capital(-1.0)
