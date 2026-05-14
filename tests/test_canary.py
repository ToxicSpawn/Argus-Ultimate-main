"""Unit tests for deploy.canary — Push 100 StatePersist."""
from __future__ import annotations

import pytest

from deploy.canary import CanaryConfig, CanaryController, CanaryPhase


@pytest.fixture()
def cfg() -> CanaryConfig:
    return CanaryConfig(
        strategy_name="momentum_v3",
        canary_version="v3.1",
        stable_version="v3.0",
        initial_weight=0.10,
        ramp_step=0.20,
        ramp_interval_s=0.0,    # immediate ramp in tests
        max_error_rate=0.02,
        max_latency_p99_ms=500.0,
        min_pnl_ratio=0.90,
        auto_ramp=True,
        pause_on_crisis=True,
    )


@pytest.fixture()
def ctrl(cfg) -> CanaryController:
    return CanaryController(cfg)


class TestCanaryLifecycle:
    def test_initial_phase_pending(self, ctrl):
        assert ctrl.phase == CanaryPhase.PENDING
        assert ctrl.weight == 0.0

    def test_start_sets_ramping(self, ctrl):
        ctrl.start()
        assert ctrl.phase == CanaryPhase.RAMPING
        assert ctrl.weight == pytest.approx(0.10)

    def test_tick_ramps_weight(self, ctrl):
        ctrl.start()
        ctrl.tick()
        assert ctrl.weight == pytest.approx(0.30)

    def test_rollback_on_high_error_rate(self, ctrl):
        ctrl.start()
        ctrl.tick(error_rate=0.05)
        assert ctrl.phase == CanaryPhase.ROLLED_BACK
        assert ctrl.weight == 0.0

    def test_rollback_on_high_latency(self, ctrl):
        ctrl.start()
        ctrl.tick(latency_p99_ms=600.0)
        assert ctrl.phase == CanaryPhase.ROLLED_BACK

    def test_rollback_on_pnl_ratio(self, ctrl):
        ctrl.start()
        ctrl.tick(pnl_ratio=0.80)
        assert ctrl.phase == CanaryPhase.ROLLED_BACK

    def test_pause_on_crisis(self, ctrl):
        ctrl.start()
        ctrl.tick(current_regime="CRISIS")
        assert ctrl.phase == CanaryPhase.PAUSED

    def test_resume_after_pause(self, ctrl):
        ctrl.start()
        ctrl.pause()
        ctrl.resume()
        assert ctrl.phase == CanaryPhase.RAMPING

    def test_reaches_full(self, ctrl):
        ctrl.start()
        for _ in range(10):          # 0.10 + 5*0.20 = 1.10 → capped at 1.0
            if ctrl.phase != CanaryPhase.RAMPING:
                break
            ctrl.tick()
        assert ctrl.phase == CanaryPhase.FULL
        assert ctrl.weight == pytest.approx(1.0)

    def test_route_to_canary(self, ctrl):
        ctrl.start()
        assert ctrl.route_to_canary(0.05) is True
        assert ctrl.route_to_canary(0.95) is False
