from __future__ import annotations

import unittest
from types import SimpleNamespace

from unified_trading_system import SystemState, UnifiedConfig, UnifiedSystemArchitecture


class _FakeMonitoring:
    async def update_metrics(self, _metrics):
        return None


class _FakeRiskManager:
    def update_capital(self, *_args, **_kwargs):
        return None

    def set_total_exposure(self, *_args, **_kwargs):
        return None

    def check_circuit_breaker(self):
        return False


class _FakeCapitalOptimizer:
    def update_capital(self, *_args, **_kwargs):
        return None

    async def optimize_signals(self, signals):
        return list(signals)


class _FakeExecutionEngine:
    def __init__(self):
        self.trade_ledger = SimpleNamespace(record_event=lambda **_kwargs: None)

    async def execute_signals(self, signals, correlation_id=None):
        _ = signals, correlation_id
        return []

    def get_last_stage_timings(self):
        return {
            "risk_gate_ms": 1.0,
            "execution_planning_ms": 2.0,
            "snapshot_persistence_ms": 3.0,
        }


class _FakeAIBrain:
    async def generate_trading_signals(self):
        return []

    def get_adaptation_status(self):
        return {}


class TestCycleLatencyInstrumentation(unittest.IsolatedAsyncioTestCase):
    async def test_cycle_stage_timing_is_recorded(self) -> None:
        cfg = UnifiedConfig()
        cfg.run_mode = "paper"
        cfg.emergency_shutdown_enabled = True
        cfg.emergency_shutdown_latency_spike_ms = 30_000.0
        cfg.runtime_safety_latency_grace_cycles = 2
        cfg.continuous_scan_enabled = False
        cfg.self_improvement_enabled = False
        cfg.hft_enabled = False
        cfg.paper_trading_peak_mode = False
        cfg.multi_language_enabled = False
        cfg.targets_enabled = False
        cfg.edge_cost_gate_enabled = False
        cfg.quant_fund_upgrades_enabled = False

        sys = UnifiedSystemArchitecture(cfg)
        sys.state = SystemState.RUNNING
        sys.monitoring = _FakeMonitoring()
        sys.unified_risk_manager = _FakeRiskManager()
        sys.capital_optimizer = _FakeCapitalOptimizer()
        sys.execution_engine = _FakeExecutionEngine()
        sys.ai_brain = _FakeAIBrain()

        await sys.run_trading_loop(cycle_seconds=0.0, max_cycles=1)

        stages = dict(getattr(sys, "_last_cycle_stage_timing_ms", {}) or {})
        expected_keys = {
            "market_data_ms",
            "feature_generation_ms",
            "strategy_evaluation_ms",
            "portfolio_targeting_ms",
            "liquidity_adjustment_ms",
            "risk_gate_ms",
            "execution_planning_ms",
            "snapshot_persistence_ms",
        }
        self.assertTrue(expected_keys.issubset(set(stages.keys())))
        self.assertGreaterEqual(float(stages.get("risk_gate_ms", 0.0) or 0.0), 0.0)

    async def test_latency_grace_cycles_defer_emergency(self) -> None:
        cfg = UnifiedConfig()
        cfg.run_mode = "paper"
        cfg.emergency_shutdown_enabled = True
        cfg.emergency_shutdown_latency_spike_ms = 30_000.0
        cfg.runtime_safety_latency_grace_cycles = 2

        sys = UnifiedSystemArchitecture(cfg)
        sys.state = SystemState.RUNNING
        sys._last_cycle_total_ms = 41_000.0

        sys._completed_cycles = 1
        self.assertFalse(sys._check_emergency_stop())

        sys._completed_cycles = 2
        self.assertTrue(sys._check_emergency_stop())


if __name__ == "__main__":
    unittest.main()
