"""
tests/test_autonomous.py --- Tests for the autonomous intelligence layer.

Covers:
- AutonomousBrain (central decisions, memory, persistence)
- ReasoningEngine (chain building, explanations, log)
- AutoStrategyManager (evaluation, correlation, apply)
- AutoCapitalAllocator (Kelly, regime, concentration cap)
- AutoModelManager (staleness, drift, accuracy, retrain)
- AutoRiskAdjuster (drawdown, vol, streaks, time, events)
- SelfImprovementOrchestrator (full cycle, IQ, journal)

50+ tests total.
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportUndefinedVariable=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportPossiblyUnboundVariable=false, reportUninitializedInstanceVariable=false, reportArgumentType=false, reportOperatorIssue=false, reportIndexIssue=false, reportMissingTypeArgument=false, reportOptionalSubscript=false

import json
import os
import sqlite3
import tempfile
import time

import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from core.autonomous_brain import AutonomousBrain, AutonomousAction
from core.reasoning_engine import ReasoningEngine, ReasoningChain
from adaptive.auto_strategy_manager import AutoStrategyManager, StrategyAction
from adaptive.auto_capital_allocator import AutoCapitalAllocator
from adaptive.auto_model_manager import AutoModelManager, ModelAction
from adaptive.auto_risk_adjuster import AutoRiskAdjuster, RiskAssessment
from core.self_improvement_orchestrator import SelfImprovementOrchestrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """Return a temporary SQLite DB path."""
    return str(tmp_path / "test_decisions.db")


@pytest.fixture
def brain(tmp_db):
    return AutonomousBrain(db_path=tmp_db)


@pytest.fixture
def reasoning(tmp_path):
    return ReasoningEngine(db_path=str(tmp_path / "reasoning.db"))


@pytest.fixture
def strategy_mgr():
    return AutoStrategyManager()


@pytest.fixture
def capital_alloc():
    return AutoCapitalAllocator()


@pytest.fixture
def model_mgr():
    return AutoModelManager()


@pytest.fixture
def risk_adj():
    return AutoRiskAdjuster()


# ---------------------------------------------------------------------------
# AutonomousBrain tests
# ---------------------------------------------------------------------------

class TestAutonomousBrain:
    """Tests for the central decision engine."""

    def test_init_default(self, brain):
        assert brain.enabled is True
        assert brain.cycle_counter == 0

    def test_decide_empty_state(self, brain):
        actions = brain.decide({})
        assert isinstance(actions, list)

    def test_decide_deactivate_strategy(self, brain):
        state = {
            "strategies": {
                "bad_strat": {
                    "sharpe": -0.5,
                    "decay_mult": 0.2,
                    "is_active": True,
                    "trades_14d": 5,
                    "regime_match": True,
                }
            }
        }
        actions = brain.decide(state)
        deactivate = [a for a in actions if a.action_type == "deactivate_strategy"]
        assert len(deactivate) >= 1
        assert deactivate[0].target == "bad_strat"

    def test_decide_promote_strategy(self, brain):
        state = {
            "strategies": {
                "good_strat": {
                    "sharpe": 2.0,
                    "decay_mult": 0.9,
                    "is_active": True,
                    "trades_14d": 20,
                    "regime_match": True,
                }
            }
        }
        actions = brain.decide(state)
        promote = [a for a in actions if a.action_type == "activate_strategy"]
        assert len(promote) >= 1

    def test_decide_regime_mismatch(self, brain):
        state = {
            "regime": "mean_revert",
            "strategies": {
                "momentum": {
                    "sharpe": 0.5,
                    "decay_mult": 0.8,
                    "is_active": True,
                    "trades_14d": 10,
                    "regime_match": False,
                }
            }
        }
        actions = brain.decide(state)
        deactivate = [a for a in actions if a.action_type == "deactivate_strategy"]
        assert len(deactivate) >= 1

    def test_decide_dormant_strategy(self, brain):
        state = {
            "strategies": {
                "sleepy": {
                    "sharpe": 0.3,
                    "decay_mult": 0.8,
                    "is_active": True,
                    "trades_14d": 0,
                    "regime_match": True,
                }
            }
        }
        actions = brain.decide(state)
        dormant = [a for a in actions if "0 trades" in a.reason]
        assert len(dormant) >= 1

    def test_decide_pause_on_drawdown(self, brain):
        state = {"drawdown_pct": 20.0, "is_paused": False}
        actions = brain.decide(state)
        pause = [a for a in actions if a.action_type == "pause_trading"]
        assert len(pause) >= 1

    def test_decide_resume_after_recovery(self, brain):
        state = {
            "drawdown_pct": 5.0,
            "is_paused": True,
            "volatility": 0.5,
            "upcoming_events": [],
        }
        actions = brain.decide(state)
        resume = [a for a in actions if a.action_type == "resume_trading"]
        assert len(resume) >= 1

    def test_decide_retrain_model(self, brain):
        state = {
            "models": {
                "regime_clf": {
                    "last_train_ts": time.time() - 86400 * 20,  # 20 days ago
                    "accuracy": 0.5,
                    "peak_accuracy": 0.7,
                    "drift_score": 0.5,
                    "samples_since_train": 500,
                }
            }
        }
        actions = brain.decide(state)
        retrain = [a for a in actions if a.action_type == "retrain_model"]
        assert len(retrain) >= 1

    def test_decide_switch_venue(self, brain):
        state = {
            "venues": {
                "kraken": {"fill_rate": 0.99, "avg_slippage_bps": 2, "fee_bps": 8, "latency_ms": 50},
                "coinbase": {"fill_rate": 0.85, "avg_slippage_bps": 10, "fee_bps": 15, "latency_ms": 200},
            },
            "current_venue": "coinbase",
        }
        actions = brain.decide(state)
        switch = [a for a in actions if a.action_type == "switch_venue"]
        assert len(switch) >= 1
        assert switch[0].target == "kraken"

    def test_decide_risk_adjustment(self, brain):
        state = {
            "drawdown_pct": 10.0,
            "volatility": 1.0,
            "risk": {"loss_streak": 7, "win_streak": 0},
        }
        actions = brain.decide(state)
        adjust = [a for a in actions if a.action_type == "adjust_risk"]
        assert len(adjust) >= 1
        assert adjust[0].params["position_multiplier"] < 1.0

    def test_decide_confidence_filter(self, tmp_db):
        brain = AutonomousBrain(config={"autonomous_brain": {"min_confidence": 0.99}}, db_path=tmp_db)
        state = {"strategies": {"x": {"sharpe": 0.3, "decay_mult": 0.6, "is_active": True, "trades_14d": 1, "regime_match": True}}}
        actions = brain.decide(state)
        assert len(actions) == 0  # nothing reaches 0.99 confidence

    def test_persistence(self, brain, tmp_db):
        state = {"drawdown_pct": 20.0, "is_paused": False}
        brain.decide(state)
        conn = sqlite3.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        conn.close()
        assert count > 0

    def test_cross_session_memory(self, brain):
        brain.remember("test_key", {"value": 42})
        assert brain.recall("test_key") == {"value": 42}
        assert brain.recall("missing") is None

    def test_decision_history(self, brain):
        brain.decide({"drawdown_pct": 20.0, "is_paused": False})
        history = brain.get_decision_history(limit=10)
        assert len(history) > 0
        assert "action_type" in history[0]

    def test_disabled_brain(self, tmp_db):
        brain = AutonomousBrain(config={"autonomous_brain": {"enabled": False}}, db_path=tmp_db)
        actions = brain.decide({"drawdown_pct": 50.0})
        assert actions == []

    def test_action_max_cap(self, tmp_db):
        brain = AutonomousBrain(
            config={"autonomous_brain": {"max_actions_per_cycle": 2, "min_confidence": 0.0}},
            db_path=tmp_db,
        )
        state = {
            "drawdown_pct": 20.0,
            "is_paused": False,
            "volatility": 2.0,
            "risk": {"loss_streak": 10},
            "strategies": {
                "a": {"sharpe": -1, "decay_mult": 0.1, "is_active": True, "trades_14d": 0, "regime_match": False},
                "b": {"sharpe": -2, "decay_mult": 0.1, "is_active": True, "trades_14d": 0, "regime_match": False},
            },
        }
        actions = brain.decide(state)
        assert len(actions) <= 2

    def test_autonomous_action_dataclass(self):
        a = AutonomousAction(
            action_type="pause_trading", target="system",
            params={"dd": 15}, reason="test", confidence=0.9, priority=1,
        )
        d = a.to_dict()
        assert d["action_type"] == "pause_trading"
        assert d["confidence"] == 0.9

    def test_pause_on_volatility_spike(self, brain):
        state = {"volatility": 2.0, "is_paused": False}
        actions = brain.decide(state)
        pause = [a for a in actions if a.action_type == "pause_trading"]
        assert len(pause) >= 1

    def test_pause_before_macro_event(self, brain):
        state = {
            "is_paused": False,
            "upcoming_events": [{"name": "FOMC", "hours_until": 1.5}],
        }
        actions = brain.decide(state)
        pause = [a for a in actions if a.action_type == "pause_trading"]
        assert len(pause) >= 1


# ---------------------------------------------------------------------------
# ReasoningEngine tests
# ---------------------------------------------------------------------------

class TestReasoningEngine:
    """Tests for the explainable reasoning engine."""

    def test_build_chain_basic(self, reasoning):
        chain = reasoning.build_reasoning_chain({
            "decision_type": "deactivate_strategy",
            "target": "momentum_eth",
            "metrics": {"sharpe": -0.3, "decay_mult": 0.2},
            "regime": "mean_revert",
            "thresholds": {"sharpe": 0.0},
        })
        assert isinstance(chain, ReasoningChain)
        assert len(chain.premises) > 0
        assert chain.confidence > 0

    def test_explain_last_decision(self, reasoning):
        assert "No decisions" in reasoning.explain_last_decision()
        reasoning.build_reasoning_chain({"decision_type": "pause_trading", "target": "system"})
        explanation = reasoning.explain_last_decision()
        assert "Decision:" in explanation

    def test_get_decision_log(self, reasoning):
        reasoning.build_reasoning_chain({"decision_type": "adjust_risk", "target": "global"})
        log = reasoning.get_decision_log(lookback_hours=1)
        assert len(log) >= 1

    def test_chain_has_alternatives(self, reasoning):
        chain = reasoning.build_reasoning_chain({
            "decision_type": "pause_trading",
            "target": "system",
        })
        assert len(chain.alternatives) > 0

    def test_chain_to_explanation(self, reasoning):
        chain = reasoning.build_reasoning_chain({
            "decision_type": "retrain_model",
            "target": "regime_clf",
            "metrics": {"age_days": 15, "drift_score": 0.5, "accuracy": 0.4, "peak_accuracy": 0.7},
        })
        text = chain.to_explanation()
        assert "regime_clf" in text

    def test_log_size(self, reasoning):
        for i in range(5):
            reasoning.build_reasoning_chain({"decision_type": "adjust_risk", "target": f"t{i}"})
        assert reasoning.log_size == 5

    def test_chain_to_dict(self, reasoning):
        chain = reasoning.build_reasoning_chain({"decision_type": "switch_venue", "target": "kraken"})
        d = chain.to_dict()
        assert "premises" in d
        assert "evidence" in d
        assert "conclusion" in d


# ---------------------------------------------------------------------------
# AutoStrategyManager tests
# ---------------------------------------------------------------------------

class TestAutoStrategyManager:

    def test_evaluate_disable_on_negative_sharpe(self, strategy_mgr):
        metrics = {"strat_a": {"sharpe": 0.5, "sharpe_14d": -0.5, "sharpe_trend": -0.1, "current_weight": 0.2, "trades_7d": 10, "is_active": True, "strategy_type": "momentum"}}
        actions = strategy_mgr.evaluate_all_strategies(metrics)
        disable = [a for a in actions if a.action == "disable"]
        assert len(disable) == 1
        assert disable[0].strategy_name == "strat_a"

    def test_evaluate_promote_high_sharpe(self, strategy_mgr):
        metrics = {"strat_b": {"sharpe": 2.0, "sharpe_14d": 2.0, "sharpe_trend": 0.1, "current_weight": 0.1, "trades_7d": 15, "is_active": True, "strategy_type": "trend_following"}}
        actions = strategy_mgr.evaluate_all_strategies(metrics, regime="trending")
        keep = [a for a in actions if a.action == "keep" and a.new_weight > 0.1]
        assert len(keep) == 1

    def test_evaluate_regime_mismatch(self, strategy_mgr):
        metrics = {"momentum_strat": {"sharpe": 0.5, "sharpe_14d": 0.5, "sharpe_trend": 0.0, "current_weight": 0.2, "trades_7d": 10, "is_active": True, "strategy_type": "momentum"}}
        actions = strategy_mgr.evaluate_all_strategies(metrics, regime="mean_revert")
        reduce = [a for a in actions if a.action == "reduce"]
        assert len(reduce) == 1

    def test_evaluate_idle_strategy(self, strategy_mgr):
        metrics = {"idle_strat": {"sharpe": 0.3, "sharpe_14d": 0.3, "sharpe_trend": 0.0, "current_weight": 0.1, "trades_7d": 0, "is_active": True, "strategy_type": "grid"}}
        actions = strategy_mgr.evaluate_all_strategies(metrics, regime="calm")
        # Grid is compatible with calm, but idle -> reduce
        reduce = [a for a in actions if a.action == "reduce"]
        assert len(reduce) == 1

    def test_evaluate_new_backtest_passed(self, strategy_mgr):
        metrics = {"new_strat": {"sharpe": 1.0, "sharpe_14d": 1.0, "is_active": False, "backtest_passed": True}}
        actions = strategy_mgr.evaluate_all_strategies(metrics)
        enable = [a for a in actions if a.action == "enable"]
        assert len(enable) == 1
        assert enable[0].new_weight == 0.10

    def test_correlation_check(self, strategy_mgr):
        metrics = {
            "strat_a": {"sharpe": 1.5, "sharpe_14d": 1.5, "current_weight": 0.2, "trades_7d": 10, "is_active": True, "strategy_type": "trend_following"},
            "strat_b": {"sharpe": 0.5, "sharpe_14d": 0.5, "current_weight": 0.2, "trades_7d": 10, "is_active": True, "strategy_type": "trend_following"},
        }
        correlations = {("strat_a", "strat_b"): 0.9}
        actions = strategy_mgr.evaluate_all_strategies(metrics, correlations=correlations)
        disable = [a for a in actions if a.action == "disable" and "correlation" in a.reason]
        assert len(disable) == 1
        assert disable[0].strategy_name == "strat_b"

    def test_auto_apply_with_allocator(self, strategy_mgr):
        class MockAllocator:
            def __init__(self):
                self.weights = {}
                self.enabled = {}
            def set_weight(self, name, w):
                self.weights[name] = w
            def set_enabled(self, name, e):
                self.enabled[name] = e

        alloc = MockAllocator()
        actions = [
            StrategyAction(strategy_name="x", action="disable", reason="test", new_weight=0.0),
            StrategyAction(strategy_name="y", action="enable", reason="test", new_weight=0.1),
        ]
        applied = strategy_mgr.auto_apply(actions, allocator=alloc)
        assert applied == 2
        assert alloc.weights["x"] == 0.0
        assert alloc.enabled["y"] is True

    def test_health_report(self, strategy_mgr):
        metrics = {
            "healthy": {"sharpe": 1.0, "win_rate": 0.6, "trades_7d": 10, "current_weight": 0.3},
            "critical": {"sharpe": -1.0, "win_rate": 0.3, "trades_7d": 5, "current_weight": 0.1},
            "idle": {"sharpe": 0.5, "win_rate": 0.5, "trades_7d": 0, "current_weight": 0.1},
        }
        report = strategy_mgr.get_strategy_health_report(metrics)
        assert report["healthy"]["health"] == "healthy"
        assert report["critical"]["health"] == "critical"
        assert report["idle"]["health"] == "idle"


# ---------------------------------------------------------------------------
# AutoCapitalAllocator tests
# ---------------------------------------------------------------------------

class TestAutoCapitalAllocator:

    def test_optimize_basic(self, capital_alloc):
        strategies = {
            "a": {"win_rate": 0.6, "avg_win_pct": 2.0, "avg_loss_pct": 1.0, "sharpe": 1.0, "is_active": True},
            "b": {"win_rate": 0.55, "avg_win_pct": 1.5, "avg_loss_pct": 1.0, "sharpe": 0.8, "is_active": True},
        }
        result = capital_alloc.optimize(strategies, capital_aud=1000.0)
        assert len(result) >= 1
        assert all(v >= 50.0 for v in result.values())

    def test_optimize_concentration_cap(self, capital_alloc):
        strategies = {
            "dominant": {"win_rate": 0.8, "avg_win_pct": 5.0, "avg_loss_pct": 1.0, "sharpe": 3.0, "is_active": True},
            "weak": {"win_rate": 0.51, "avg_win_pct": 1.0, "avg_loss_pct": 1.0, "sharpe": 0.1, "is_active": True},
        }
        result = capital_alloc.optimize(strategies, capital_aud=1000.0)
        total = sum(result.values())
        for v in result.values():
            assert v / total <= 0.41  # allow small rounding

    def test_optimize_minimum_filter(self, capital_alloc):
        strategies = {
            "tiny": {"win_rate": 0.51, "avg_win_pct": 0.5, "avg_loss_pct": 1.0, "sharpe": 0.01, "is_active": True},
            "big": {"win_rate": 0.7, "avg_win_pct": 3.0, "avg_loss_pct": 1.0, "sharpe": 2.0, "is_active": True},
        }
        result = capital_alloc.optimize(strategies, capital_aud=100.0)
        # With only $100, tiny might not meet the $50 minimum
        for v in result.values():
            assert v >= 50.0

    def test_optimize_inactive_excluded(self, capital_alloc):
        strategies = {
            "active": {"win_rate": 0.6, "avg_win_pct": 2.0, "avg_loss_pct": 1.0, "sharpe": 1.0, "is_active": True},
            "inactive": {"win_rate": 0.6, "avg_win_pct": 2.0, "avg_loss_pct": 1.0, "sharpe": 1.0, "is_active": False},
        }
        result = capital_alloc.optimize(strategies, capital_aud=1000.0)
        assert "inactive" not in result

    def test_optimize_regime_boost(self, capital_alloc):
        strategies = {
            "suitable": {"win_rate": 0.6, "avg_win_pct": 2.0, "avg_loss_pct": 1.0, "sharpe": 1.0, "is_active": True, "regime_suitable": True},
            "unsuitable": {"win_rate": 0.6, "avg_win_pct": 2.0, "avg_loss_pct": 1.0, "sharpe": 1.0, "is_active": True, "regime_suitable": False},
        }
        result = capital_alloc.optimize(strategies, capital_aud=1000.0)
        if "suitable" in result and "unsuitable" in result:
            assert result["suitable"] > result["unsuitable"]

    def test_rebalance_check_no_drift(self, capital_alloc):
        capital_alloc._last_allocation = {"a": 500, "b": 500}
        assert capital_alloc.rebalance_check(
            current={"a": 500, "b": 500},
            optimal={"a": 500, "b": 500},
        ) is False

    def test_rebalance_check_drift(self, capital_alloc):
        assert capital_alloc.rebalance_check(
            current={"a": 800, "b": 200},
            optimal={"a": 500, "b": 500},
        ) is True

    def test_allocation_drift(self, capital_alloc):
        drift = capital_alloc.get_allocation_drift(
            current={"a": 600, "b": 400},
            optimal={"a": 500, "b": 500},
        )
        assert drift["a"] > 0  # overweight
        assert drift["b"] < 0  # underweight

    def test_optimize_empty(self, capital_alloc):
        assert capital_alloc.optimize({}, capital_aud=1000.0) == {}

    def test_optimize_zero_capital(self, capital_alloc):
        strategies = {"a": {"win_rate": 0.6, "avg_win_pct": 2.0, "avg_loss_pct": 1.0, "sharpe": 1.0, "is_active": True}}
        assert capital_alloc.optimize(strategies, capital_aud=0.0) == {}


# ---------------------------------------------------------------------------
# AutoModelManager tests
# ---------------------------------------------------------------------------

class TestAutoModelManager:

    def test_check_stale_model(self, model_mgr):
        registry = {
            "regime_clf": {
                "last_train_ts": time.time() - 86400 * 15,
                "accuracy": 0.65,
                "peak_accuracy": 0.70,
                "drift_score": 0.1,
                "samples_since_train": 500,
                "is_active": True,
            }
        }
        actions = model_mgr.check_all_models(registry)
        assert len(actions) == 1
        assert actions[0].action == "retrain"
        assert "stale" in actions[0].reason

    def test_check_drift(self, model_mgr):
        registry = {
            "vol_forecast": {
                "last_train_ts": time.time() - 86400 * 2,
                "accuracy": 0.6,
                "peak_accuracy": 0.6,
                "drift_score": 0.5,
                "samples_since_train": 200,
                "is_active": True,
            }
        }
        actions = model_mgr.check_all_models(registry)
        assert len(actions) == 1
        assert "drift" in actions[0].reason

    def test_check_accuracy_drop(self, model_mgr):
        registry = {
            "alpha_model": {
                "last_train_ts": time.time() - 86400 * 3,
                "accuracy": 0.55,
                "peak_accuracy": 0.75,
                "drift_score": 0.1,
                "samples_since_train": 200,
                "is_active": True,
            }
        }
        actions = model_mgr.check_all_models(registry)
        assert len(actions) == 1
        assert "accuracy" in actions[0].reason.lower()

    def test_check_disable_consistently_wrong(self, model_mgr):
        registry = {
            "bad_model": {
                "last_train_ts": time.time() - 86400 * 3,
                "accuracy": 0.40,
                "peak_accuracy": 0.70,
                "drift_score": 0.1,
                "samples_since_train": 200,
                "is_active": True,
            }
        }
        actions = model_mgr.check_all_models(registry)
        disable = [a for a in actions if a.action == "disable"]
        assert len(disable) == 1

    def test_check_new_data_available(self, model_mgr):
        registry = {
            "fresh_model": {
                "last_train_ts": time.time() - 86400 * 3,
                "accuracy": 0.65,
                "peak_accuracy": 0.65,
                "drift_score": 0.1,
                "samples_since_train": 1500,
                "is_active": True,
            }
        }
        actions = model_mgr.check_all_models(registry)
        assert len(actions) == 1
        assert "samples" in actions[0].reason

    def test_check_ok_model(self, model_mgr):
        registry = {
            "good_model": {
                "last_train_ts": time.time() - 86400 * 2,
                "accuracy": 0.70,
                "peak_accuracy": 0.72,
                "drift_score": 0.1,
                "samples_since_train": 200,
                "is_active": True,
            }
        }
        actions = model_mgr.check_all_models(registry)
        assert len(actions) == 0

    def test_auto_retrain_no_trainer(self, model_mgr):
        assert model_mgr.auto_retrain("x") is False

    def test_auto_retrain_with_trainer(self, model_mgr):
        class Trainer:
            def __init__(self):
                self.retrained = []
            def retrain(self, name):
                self.retrained.append(name)

        t = Trainer()
        assert model_mgr.auto_retrain("regime_clf", trainer=t) is True
        assert "regime_clf" in t.retrained

    def test_schedule_nightly(self, model_mgr):
        sched = model_mgr.schedule_nightly_check()
        assert "hour" in sched
        assert "callback" in sched

    def test_inactive_model_skipped(self, model_mgr):
        registry = {
            "disabled": {
                "last_train_ts": 0,
                "accuracy": 0.3,
                "is_active": False,
            }
        }
        actions = model_mgr.check_all_models(registry)
        assert len(actions) == 0


# ---------------------------------------------------------------------------
# AutoRiskAdjuster tests
# ---------------------------------------------------------------------------

class TestAutoRiskAdjuster:

    def test_normal_conditions(self, risk_adj):
        assessment = risk_adj.assess_risk_level({
            "drawdown_pct": 0.0,
            "volatility": 0.4,
            "win_streak": 0,
            "loss_streak": 0,
            "utc_hour": 12,
            "day_of_week": 2,
        })
        assert assessment.level == "normal"
        assert 0.95 <= assessment.position_multiplier <= 1.05

    def test_conservative_on_drawdown(self, risk_adj):
        assessment = risk_adj.assess_risk_level({"drawdown_pct": 20.0, "volatility": 0.5, "utc_hour": 12, "day_of_week": 2})
        assert assessment.level == "conservative"
        assert assessment.position_multiplier < 0.8

    def test_conservative_on_high_vol(self, risk_adj):
        assessment = risk_adj.assess_risk_level({"drawdown_pct": 0.0, "volatility": 1.2, "utc_hour": 12, "day_of_week": 2})
        assert assessment.position_multiplier < 0.8

    def test_aggressive_on_low_vol(self, risk_adj):
        assessment = risk_adj.assess_risk_level({
            "drawdown_pct": 0.0,
            "volatility": 0.1,
            "win_streak": 6,
            "utc_hour": 12,
            "day_of_week": 2,
        })
        assert assessment.position_multiplier > 1.0

    def test_loss_streak_reduces(self, risk_adj):
        assessment = risk_adj.assess_risk_level({
            "drawdown_pct": 0.0,
            "volatility": 0.4,
            "loss_streak": 8,
            "utc_hour": 12,
            "day_of_week": 2,
        })
        assert assessment.position_multiplier < 0.85

    def test_low_liquidity_hours(self, risk_adj):
        assessment = risk_adj.assess_risk_level({"utc_hour": 3, "day_of_week": 2})
        assert "low-liquidity" in assessment.reason

    def test_weekend_reduction(self, risk_adj):
        assessment = risk_adj.assess_risk_level({"utc_hour": 12, "day_of_week": 6})
        assert "weekend" in assessment.reason

    def test_macro_event_reduction(self, risk_adj):
        assessment = risk_adj.assess_risk_level({
            "utc_hour": 12,
            "day_of_week": 2,
            "upcoming_events": [{"name": "FOMC", "hours_until": 1.0}],
        })
        assert "FOMC" in assessment.reason
        assert assessment.position_multiplier < 0.8

    def test_apply_risk_level(self, risk_adj):
        class MockSystem:
            def __init__(self):
                self.mult = 1.0
                self.exp = 1.0
            def set_position_multiplier(self, m):
                self.mult = m
            def set_max_exposure(self, e):
                self.exp = e

        sys = MockSystem()
        assessment = RiskAssessment(level="conservative", position_multiplier=0.6, max_exposure_pct=0.5, reason="test")
        assert risk_adj.apply_risk_level(assessment, system=sys) is True
        assert sys.mult == 0.6
        assert sys.exp == 0.5

    def test_disabled_adjuster(self):
        adj = AutoRiskAdjuster(config={"enabled": False})
        assessment = adj.assess_risk_level({"drawdown_pct": 50.0})
        assert assessment.level == "normal"
        assert assessment.position_multiplier == 1.0


# ---------------------------------------------------------------------------
# SelfImprovementOrchestrator tests
# ---------------------------------------------------------------------------

class TestSelfImprovementOrchestrator:

    def _make_orchestrator(self, tmp_path, **kwargs):
        return SelfImprovementOrchestrator(
            config={"journal_path": str(tmp_path / "journal.jsonl"), "cycle_interval": 3},
            **kwargs,
        )

    def test_run_cycle_standalone(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        result = orch.run_improvement_cycle({})
        assert isinstance(result, ImprovementCycleResult)
        assert result.cycle_number == 1
        assert result.system_iq >= 0

    def test_run_cycle_with_components(self, tmp_path):
        brain = AutonomousBrain(db_path=str(tmp_path / "brain.db"))
        reasoning = ReasoningEngine(db_path=str(tmp_path / "reasoning.db"))
        strategy_mgr = AutoStrategyManager()
        model_mgr = AutoModelManager()
        risk_adj = AutoRiskAdjuster()

        orch = self._make_orchestrator(
            tmp_path,
            brain=brain, reasoning=reasoning,
            strategy_mgr=strategy_mgr, model_mgr=model_mgr,
            risk_adj=risk_adj,
        )

        state = {
            "regime": "trending",
            "drawdown_pct": 3.0,
            "volatility": 0.5,
            "strategies": {
                "mom": {"sharpe": 1.0, "sharpe_14d": 1.0, "win_rate": 0.6, "trades_14d": 10,
                         "trades_7d": 5, "is_active": True, "decay_mult": 0.8, "regime_match": True,
                         "strategy_type": "momentum", "current_weight": 0.2},
            },
            "models": {
                "regime_clf": {"last_train_ts": time.time() - 86400, "accuracy": 0.7,
                               "peak_accuracy": 0.72, "drift_score": 0.1, "samples_since_train": 100, "is_active": True},
            },
            "risk": {"win_streak": 2, "loss_streak": 0, "uptime_pct": 99.5, "error_rate": 0.01},
        }
        result = orch.run_improvement_cycle(state)
        assert result.risk_level in ("conservative", "normal", "aggressive")
        assert result.duration_seconds >= 0

    def test_should_run_interval(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        assert orch.should_run() is False  # cycle 1
        assert orch.should_run() is False  # cycle 2
        assert orch.should_run() is True   # cycle 3

    def test_get_summary_no_cycles(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        assert "No improvement" in orch.get_improvement_summary()

    def test_get_summary_after_cycle(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        orch.run_improvement_cycle({})
        summary = orch.get_improvement_summary()
        assert "Cycle #1" in summary
        assert "System IQ" in summary

    def test_system_iq_scoring(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        # Perfect conditions
        state = {
            "strategies": {"a": {"sharpe": 2.0, "win_rate": 0.65}},
            "risk": {"error_rate": 0.0, "uptime_pct": 100.0},
        }
        iq = orch.get_system_iq(state)
        assert iq > 0
        assert iq <= 100

    def test_journal_written(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        orch.run_improvement_cycle({})
        journal = tmp_path / "journal.jsonl"
        assert journal.exists()
        lines = journal.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["cycle_number"] == 1

    def test_results_property(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        orch.run_improvement_cycle({})
        orch.run_improvement_cycle({})
        assert len(orch.results) == 2
        assert orch.results[0]["cycle_number"] == 1

    def test_cycle_count(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        assert orch.cycle_count == 0
        orch.run_improvement_cycle({})
        assert orch.cycle_count == 1
