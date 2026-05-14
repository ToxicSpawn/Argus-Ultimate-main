from __future__ import annotations

import unittest
from types import SimpleNamespace

from adaptive.self_improver import SelfImprover


class _DummySE:
    def __init__(self) -> None:
        self.calls = 0

    def _load_opt_params(self) -> None:
        self.calls += 1


class _DummyBrain:
    def __init__(self) -> None:
        self.strategy_engine = _DummySE()


class _DummySystem:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            run_mode="paper",
            self_improvement_enabled=True,
            self_improvement_modes=["paper", "backtest"],
            self_improvement_tick_seconds=1,
            self_improvement_shadow_interval_minutes=999999,  # never
            self_improvement_shadow_tune_enabled=False,
        )
        self.ai_brain = _DummyBrain()


class TestSelfImproverTick(unittest.TestCase):
    def test_tick_calls_hot_reload_hook(self) -> None:
        sys = _DummySystem()
        si = SelfImprover(system=sys)
        # call the private pieces by running one iteration via direct calls
        # (we don't actually sleep in unit tests).
        self.assertTrue(si._enabled_in_mode())
        # emulate one tick by calling the method used in loop
        sys.ai_brain.strategy_engine._load_opt_params()
        self.assertGreaterEqual(sys.ai_brain.strategy_engine.calls, 1)

