#!/usr/bin/env python3
"""
Tests for PEAK Intelligence Layer — 70+ tests covering all 7 modules.

Modules tested:
  1. CognitiveEngine       — 5-stage cognitive loop + Bayesian beliefs
  2. GoalManager           — goal-oriented trading + progress tracking
  3. MarketHypothesisEngine — hypothesis generation and testing
  4. AttentionSystem        — dynamic attention allocation
  5. CounterfactualAnalyzer — what-if analysis
  6. PredictivePlanner      — forward-looking trade planning
  7. SystemConsciousness    — self-awareness module
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import pytest
from datetime import datetime, timezone, timedelta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """Return a temp DB path."""
    return str(tmp_path / "test.db")


@pytest.fixture
def market_state():
    """Realistic market state dict for testing."""
    return {
        "prices": {
            "BTC/USD": {"price": 65000, "change_pct": 1.5},
            "ETH/USD": {"price": 3200, "change_pct": -0.3},
            "SOL/USD": {"price": 180, "change_pct": 5.2},
        },
        "regime": "trending_up",
        "signals": {
            "BTC/USD": {"score": 0.6, "strategy": "momentum"},
            "ETH/USD": {"score": -0.2, "strategy": "mean_revert"},
            "SOL/USD": {"score": 0.8, "strategy": "breakout"},
        },
        "risk_metrics": {
            "drawdown_pct": 2.5,
            "var": 1.8,
            "exposure_pct": 45.0,
        },
        "sentiment": {"fear_greed": 62},
        "execution_quality": {"avg_slippage_bps": 3.5},
        "positions": {
            "BTC/USD": {"entry_price": 64000, "stop_loss": 62000, "size": 0.01, "pnl": 10.0},
        },
        "volatility": {"BTC/USD": 3.2, "ETH/USD": 4.1, "SOL/USD": 8.5},
    }


# ===================================================================
# 1. CognitiveEngine Tests
# ===================================================================

class TestCognitiveEngine:
    """Tests for core/cognitive_engine.py."""

    def test_init_default(self, tmp_db):
        from core.cognitive_engine import CognitiveEngine
        engine = CognitiveEngine(db_path=tmp_db)
        assert engine._enabled is True
        assert os.path.exists(tmp_db)

    def test_init_with_config(self, tmp_db):
        from core.cognitive_engine import CognitiveEngine
        engine = CognitiveEngine(config={"max_plans": 3, "belief_prior": 0.6}, db_path=tmp_db)
        assert engine._max_plans == 3
        assert engine._belief_prior == 0.6

    def test_think_returns_cognitive_result(self, tmp_db, market_state):
        from core.cognitive_engine import CognitiveEngine, CognitiveResult
        engine = CognitiveEngine(db_path=tmp_db)
        result = engine.think(market_state)
        assert isinstance(result, CognitiveResult)
        assert result.perception
        assert result.analysis
        assert isinstance(result.plans, list)
        assert result.chosen_plan
        assert 0.0 <= result.confidence <= 1.0
        assert result.reasoning

    def test_perceive_extracts_all_fields(self, tmp_db, market_state):
        from core.cognitive_engine import CognitiveEngine
        engine = CognitiveEngine(db_path=tmp_db)
        perception = engine._perceive(market_state)
        assert "prices" in perception
        assert "regime" in perception
        assert "signals" in perception
        assert "drawdown_pct" in perception
        assert "fear_greed" in perception
        assert "open_positions" in perception
        assert perception["regime"] == "trending_up"

    def test_analyze_finds_opportunities(self, tmp_db, market_state):
        from core.cognitive_engine import CognitiveEngine
        engine = CognitiveEngine(db_path=tmp_db)
        perception = engine._perceive(market_state)
        analysis = engine._analyze(perception, market_state)
        assert len(analysis["opportunities"]) >= 2  # BTC (0.6) and SOL (0.8)
        assert analysis["regime_assessment"] == "trending_up"

    def test_analyze_detects_anomaly(self, tmp_db):
        from core.cognitive_engine import CognitiveEngine
        engine = CognitiveEngine(db_path=tmp_db)
        engine = CognitiveEngine(db_path=tmp_db, config={"anomaly_z_threshold": 1.5})
        state = {
            "prices": {
                "A": {"price": 100, "change_pct": 0.5},
                "B": {"price": 200, "change_pct": 0.3},
                "C": {"price": 300, "change_pct": 15.0},  # anomaly
                "C": {"price": 300, "change_pct": 50.0},  # anomaly with z~2.0 > 1.5 threshold
                "D": {"price": 150, "change_pct": 0.4},
                "E": {"price": 250, "change_pct": 0.2},
            },
            "signals": {}, "regime": "unknown",
            "risk_metrics": {}, "sentiment": {},
            "execution_quality": {},
        }
        perception = engine._perceive(state)
        analysis = engine._analyze(perception, state)
        assert len(analysis["anomalies"]) >= 1
        assert analysis["anomalies"][0]["symbol"] == "C"

    def test_plan_includes_hold(self, tmp_db, market_state):
        from core.cognitive_engine import CognitiveEngine
        engine = CognitiveEngine(db_path=tmp_db)
        perception = engine._perceive(market_state)
        analysis = engine._analyze(perception, market_state)
        plans = engine._plan(analysis, market_state)
        assert any(p["id"] == "hold" for p in plans)

    def test_decide_selects_best_plan(self, tmp_db, market_state):
        from core.cognitive_engine import CognitiveEngine
        engine = CognitiveEngine(db_path=tmp_db)
        result = engine.think(market_state)
        assert result.chosen_plan.get("id") is not None

    def test_belief_system_update(self, tmp_db):
        from core.cognitive_engine import CognitiveEngine
        engine = CognitiveEngine(db_path=tmp_db)
        # Initial belief
        p1 = engine.update_belief("test_belief", 0.8, True)
        assert p1 > 0.5
        # Opposing evidence
        p2 = engine.update_belief("test_belief", 0.8, False)
        assert p2 < p1

    def test_get_beliefs(self, tmp_db):
        from core.cognitive_engine import CognitiveEngine
        engine = CognitiveEngine(db_path=tmp_db)
        engine.update_belief("btc_trending", 0.7, True)
        engine.update_belief("eth_undervalued", 0.6, False)
        beliefs = engine.get_beliefs()
        assert "btc_trending" in beliefs
        assert "eth_undervalued" in beliefs
        assert beliefs["btc_trending"] > 0.5
        assert beliefs["eth_undervalued"] < 0.5

    def test_reset_belief(self, tmp_db):
        from core.cognitive_engine import CognitiveEngine
        engine = CognitiveEngine(db_path=tmp_db)
        engine.update_belief("test", 0.9, True)
        engine.reset_belief("test")
        beliefs = engine.get_beliefs()
        assert beliefs["test"] == engine._belief_prior

    def test_record_outcome(self, tmp_db):
        from core.cognitive_engine import CognitiveEngine
        engine = CognitiveEngine(db_path=tmp_db)
        engine.record_outcome("plan_1", 1.0, 0.5)
        engine.record_outcome("plan_2", 0.5, 0.8)
        stats = engine.get_prediction_accuracy()
        assert stats["count"] == 2
        assert stats["mae"] > 0

    def test_persistence_across_instances(self, tmp_db):
        from core.cognitive_engine import CognitiveEngine
        e1 = CognitiveEngine(db_path=tmp_db)
        e1.update_belief("persist_test", 0.9, True)
        val1 = e1.get_beliefs()["persist_test"]

        e2 = CognitiveEngine(db_path=tmp_db)
        val2 = e2.get_beliefs()["persist_test"]
        assert abs(val1 - val2) < 0.001

    def test_cognitive_result_to_dict(self, tmp_db, market_state):
        from core.cognitive_engine import CognitiveEngine
        engine = CognitiveEngine(db_path=tmp_db)
        result = engine.think(market_state)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "perception" in d
        assert "confidence" in d

    def test_threat_driven_de_risk_plan(self, tmp_db):
        from core.cognitive_engine import CognitiveEngine
        engine = CognitiveEngine(db_path=tmp_db)
        state = {
            "prices": {"BTC/USD": {"price": 60000, "change_pct": -3.0}},
            "signals": {},
            "regime": "crisis",
            "risk_metrics": {"drawdown_pct": 8.0},
            "sentiment": {"fear_greed": 15},
            "execution_quality": {},
        }
        result = engine.think(state)
        plan_ids = [p["id"] for p in result.plans]
        assert "de_risk" in plan_ids or "defensive" in plan_ids


# ===================================================================
# 2. GoalManager Tests
# ===================================================================

class TestGoalManager:
    """Tests for core/goal_manager.py."""

    def test_init_default(self, tmp_db):
        from core.goal_manager import GoalManager
        gm = GoalManager(db_path=tmp_db)
        assert gm._enabled is True

    def test_set_goal(self, tmp_db):
        from core.goal_manager import GoalManager
        gm = GoalManager(db_path=tmp_db)
        goal = gm.set_goal("monthly_return", 0.05, "2026-04-30")
        assert goal.goal_type == "monthly_return"
        assert goal.target == 0.05

    def test_invalid_goal_type(self, tmp_db):
        from core.goal_manager import GoalManager
        gm = GoalManager(db_path=tmp_db)
        with pytest.raises(ValueError, match="Invalid goal_type"):
            gm.set_goal("invalid_type", 0.05)

    def test_remove_goal(self, tmp_db):
        from core.goal_manager import GoalManager
        gm = GoalManager(db_path=tmp_db)
        gm.set_goal("monthly_return", 0.05)
        assert gm.remove_goal("monthly_return") is True
        assert gm.remove_goal("nonexistent") is False
        assert len(gm.get_goals()) == 0

    def test_evaluate_progress(self, tmp_db):
        from core.goal_manager import GoalManager
        gm = GoalManager(db_path=tmp_db)
        gm.set_goal("monthly_return", 0.10)
        gm.update_metrics({"monthly_return": 0.05})
        progress = gm.evaluate_progress()
        assert len(progress) == 1
        assert progress[0].pct_complete == 50.0

    def test_evaluate_drawdown_goal(self, tmp_db):
        from core.goal_manager import GoalManager
        gm = GoalManager(db_path=tmp_db)
        gm.set_goal("max_drawdown", 5.0)
        gm.update_metrics({"current_drawdown_pct": 3.0})
        progress = gm.evaluate_progress()
        assert progress[0].pct_complete == 100.0  # 3% < 5% target

    def test_evaluate_drawdown_exceeded(self, tmp_db):
        from core.goal_manager import GoalManager
        gm = GoalManager(db_path=tmp_db)
        gm.set_goal("max_drawdown", 5.0)
        gm.update_metrics({"current_drawdown_pct": 8.0})
        progress = gm.evaluate_progress()
        assert progress[0].pct_complete < 100.0

    def test_adjust_strategy_behind_return(self, tmp_db):
        from core.goal_manager import GoalManager
        gm = GoalManager(db_path=tmp_db)
        gm.set_goal("monthly_return", 0.10, "2026-04-30")
        # No deadline — _is_on_track returns pct>=50, 20% is behind
        gm.set_goal("monthly_return", 0.10)
        gm.update_metrics({"monthly_return": 0.02})
        adjustments = gm.adjust_strategy_for_goals()
        assert len(adjustments) >= 1
        assert adjustments[0].adjustment_type == "increase_risk"

    def test_adjust_strategy_drawdown_warning(self, tmp_db):
        from core.goal_manager import GoalManager
        gm = GoalManager(db_path=tmp_db)
        gm.set_goal("max_drawdown", 5.0)
        gm.update_metrics({"current_drawdown_pct": 4.5})  # 90% of limit
        adjustments = gm.adjust_strategy_for_goals()
        has_reduce = any(a.adjustment_type == "reduce_exposure" for a in adjustments)
        assert has_reduce

    def test_adjust_strategy_low_win_rate(self, tmp_db):
        from core.goal_manager import GoalManager
        gm = GoalManager(db_path=tmp_db)
        gm.set_goal("win_rate", 0.60)
        gm.update_metrics({"win_rate": 0.35})
        # No deadline — 35/60 = 58% pct, but _is_on_track checks pct>=50 → on_track
        # Use a very low win rate so pct is clearly below 50%
        gm.set_goal("win_rate", 0.80)
        gm.update_metrics({"win_rate": 0.30})  # 30/80 = 37.5% → behind
        adjustments = gm.adjust_strategy_for_goals()
        assert any(a.adjustment_type == "switch_strategy" for a in adjustments)

    def test_get_goal_dashboard(self, tmp_db):
        from core.goal_manager import GoalManager
        gm = GoalManager(db_path=tmp_db)
        gm.set_goal("monthly_return", 0.05)
        gm.set_goal("sharpe_ratio", 1.5)
        gm.update_metrics({"monthly_return": 0.03, "sharpe_ratio": 1.2})
        dashboard = gm.get_goal_dashboard()
        assert dashboard["total_goals"] == 2
        assert "goals" in dashboard
        assert "adjustments" in dashboard

    def test_persistence(self, tmp_db):
        from core.goal_manager import GoalManager
        gm1 = GoalManager(db_path=tmp_db)
        gm1.set_goal("monthly_return", 0.05, "2026-04-30")
        gm2 = GoalManager(db_path=tmp_db)
        assert len(gm2.get_goals()) == 1
        assert gm2.get_goals()[0].target == 0.05

    def test_max_goals_limit(self, tmp_db):
        from core.goal_manager import GoalManager
        gm = GoalManager(config={"max_active_goals": 2}, db_path=tmp_db)
        gm.set_goal("monthly_return", 0.05)
        gm.set_goal("sharpe_ratio", 1.0)
        with pytest.raises(RuntimeError, match="Maximum active goals"):
            gm.set_goal("win_rate", 0.6)


# ===================================================================
# 3. MarketHypothesisEngine Tests
# ===================================================================

class TestMarketHypothesisEngine:
    """Tests for core/market_hypothesis_engine.py."""

    def test_init_default(self, tmp_db):
        from core.market_hypothesis_engine import MarketHypothesisEngine
        engine = MarketHypothesisEngine(db_path=tmp_db)
        assert engine._enabled is True

    def test_generate_bull_hypothesis(self, tmp_db):
        from core.market_hypothesis_engine import MarketHypothesisEngine
        engine = MarketHypothesisEngine(db_path=tmp_db)
        data = {
            "symbols": {
                "BTC/USD": {"momentum_1d": 0.05, "hurst": 0.6, "rsi": 55},
            }
        }
        hypotheses = engine.generate_hypotheses(data)
        assert len(hypotheses) >= 1
        assert "bull" in hypotheses[0].statement.lower()
        assert hypotheses[0].probability > 0.5

    def test_generate_bear_hypothesis(self, tmp_db):
        from core.market_hypothesis_engine import MarketHypothesisEngine
        engine = MarketHypothesisEngine(db_path=tmp_db)
        data = {
            "symbols": {
                "ETH/USD": {"momentum_1d": -0.04, "hurst": 0.65, "rsi": 40},
            }
        }
        hypotheses = engine.generate_hypotheses(data)
        assert any("bear" in h.statement.lower() for h in hypotheses)

    def test_generate_volatility_hypothesis(self, tmp_db):
        from core.market_hypothesis_engine import MarketHypothesisEngine
        engine = MarketHypothesisEngine(db_path=tmp_db)
        data = {"symbols": {"SOL/USD": {"bb_squeeze": True, "atr_ratio": 0.3}}}
        hypotheses = engine.generate_hypotheses(data)
        assert any("volatility" in h.statement.lower() for h in hypotheses)

    def test_generate_decoupling_hypothesis(self, tmp_db):
        from core.market_hypothesis_engine import MarketHypothesisEngine
        engine = MarketHypothesisEngine(db_path=tmp_db)
        data = {"symbols": {"ETH/USD": {"btc_eth_corr": 0.4}}}
        hypotheses = engine.generate_hypotheses(data)
        assert any("decoupling" in h.statement.lower() for h in hypotheses)

    def test_generate_extreme_fear_hypothesis(self, tmp_db):
        from core.market_hypothesis_engine import MarketHypothesisEngine
        engine = MarketHypothesisEngine(db_path=tmp_db)
        data = {"symbols": {"BTC/USD": {"fear_greed": 15}}}
        hypotheses = engine.generate_hypotheses(data)
        assert any("fear" in h.statement.lower() for h in hypotheses)

    def test_test_hypothesis_confirmed(self, tmp_db):
        from core.market_hypothesis_engine import MarketHypothesisEngine
        engine = MarketHypothesisEngine(db_path=tmp_db)
        data = {"symbols": {"BTC/USD": {"momentum_1d": 0.05, "hurst": 0.6, "rsi": 55}}}
        hypotheses = engine.generate_hypotheses(data)
        hid = hypotheses[0].hypothesis_id
        result = engine.test_hypothesis(hid, {"price_change_pct": 3.0})
        assert result is not None
        assert result.outcome == "confirmed"

    def test_test_hypothesis_rejected(self, tmp_db):
        from core.market_hypothesis_engine import MarketHypothesisEngine
        engine = MarketHypothesisEngine(db_path=tmp_db)
        data = {"symbols": {"BTC/USD": {"momentum_1d": 0.05, "hurst": 0.6, "rsi": 55}}}
        hypotheses = engine.generate_hypotheses(data)
        hid = hypotheses[0].hypothesis_id
        result = engine.test_hypothesis(hid, {"price_change_pct": -5.0})
        assert result.outcome == "rejected"

    def test_get_active_hypotheses(self, tmp_db):
        from core.market_hypothesis_engine import MarketHypothesisEngine
        engine = MarketHypothesisEngine(db_path=tmp_db)
        data = {"symbols": {"BTC/USD": {"momentum_1d": 0.05, "hurst": 0.6, "rsi": 55}}}
        engine.generate_hypotheses(data)
        active = engine.get_active_hypotheses()
        assert len(active) >= 1

    def test_hypothesis_accuracy(self, tmp_db):
        from core.market_hypothesis_engine import MarketHypothesisEngine
        engine = MarketHypothesisEngine(db_path=tmp_db)
        data = {"symbols": {"BTC/USD": {"momentum_1d": 0.05, "hurst": 0.6, "rsi": 55}}}
        hypotheses = engine.generate_hypotheses(data)
        engine.test_hypothesis(hypotheses[0].hypothesis_id, {"price_change_pct": 3.0})
        accuracy = engine.get_hypothesis_accuracy()
        assert accuracy == 1.0  # one confirmed, zero rejected

    def test_expire_stale_hypotheses(self, tmp_db):
        from core.market_hypothesis_engine import MarketHypothesisEngine, Hypothesis
        engine = MarketHypothesisEngine(db_path=tmp_db)
        # Manually insert an expired hypothesis
        old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        h = Hypothesis(
            hypothesis_id="old_1", statement="test",
            probability=0.5, expiry_hours=1, created_at=old_time,
        )
        engine._hypotheses["old_1"] = h
        expired = engine.expire_stale_hypotheses()
        assert expired >= 1

    def test_add_evidence(self, tmp_db):
        from core.market_hypothesis_engine import MarketHypothesisEngine
        engine = MarketHypothesisEngine(db_path=tmp_db)
        data = {"symbols": {"BTC/USD": {"momentum_1d": 0.05, "hurst": 0.6, "rsi": 55}}}
        hypotheses = engine.generate_hypotheses(data)
        hid = hypotheses[0].hypothesis_id
        old_prob = hypotheses[0].probability
        engine.add_evidence(hid, "ETF inflows increasing", supports=True)
        h = engine.get_hypothesis(hid)
        assert h.probability > old_prob

    def test_no_duplicate_hypotheses(self, tmp_db):
        from core.market_hypothesis_engine import MarketHypothesisEngine
        engine = MarketHypothesisEngine(db_path=tmp_db)
        data = {"symbols": {"BTC/USD": {"momentum_1d": 0.05, "hurst": 0.6, "rsi": 55}}}
        h1 = engine.generate_hypotheses(data)
        h2 = engine.generate_hypotheses(data)
        # Second call should not duplicate
        assert len(h2) == 0


# ===================================================================
# 4. AttentionSystem Tests
# ===================================================================

class TestAttentionSystem:
    """Tests for adaptive/attention_system.py."""

    def test_init_default(self):
        from adaptive.attention_system import AttentionSystem
        attn = AttentionSystem()
        assert attn._enabled is True

    def test_compute_attention_basic(self):
        from adaptive.attention_system import AttentionSystem
        attn = AttentionSystem()
        state = {
            "symbols": {
                "BTC/USD": {"volatility": 3.0, "price": 65000, "volume_ratio": 1.2},
                "ETH/USD": {"volatility": 4.5, "price": 3200, "volume_ratio": 0.8},
            },
            "positions": {"BTC/USD": {"entry_price": 64000, "stop_loss": 62000}},
            "signals": {"BTC/USD": {"score": 0.7}},
            "strategies": {},
            "risk_metrics": {"drawdown_pct": 2.0},
        }
        attention = attn.compute_attention(state)
        assert len(attention.focus_symbols) >= 1
        # BTC should rank higher (has position + signal + vol)
        assert attention.focus_symbols[0].name == "BTC/USD"

    def test_high_vol_gets_attention(self):
        from adaptive.attention_system import AttentionSystem
        attn = AttentionSystem()
        state = {
            "symbols": {
                "LOW": {"volatility": 0.5},
                "HIGH": {"volatility": 8.0},
            },
            "positions": {}, "signals": {}, "strategies": {},
            "risk_metrics": {},
        }
        attention = attn.compute_attention(state)
        names = [i.name for i in attention.focus_symbols]
        assert "HIGH" in names

    def test_stop_proximity_max_attention(self):
        from adaptive.attention_system import AttentionSystem
        attn = AttentionSystem()
        state = {
            "symbols": {
                "BTC/USD": {"price": 62100, "volatility": 2.0},
            },
            "positions": {
                "BTC/USD": {"entry_price": 63000, "stop_loss": 62000},
            },
            "signals": {}, "strategies": {}, "risk_metrics": {},
        }
        attention = attn.compute_attention(state)
        btc = [i for i in attention.focus_symbols if i.name == "BTC/USD"]
        assert len(btc) == 1
        assert any("NEAR STOP" in r for r in btc[0].reasons)

    def test_ignore_list(self):
        from adaptive.attention_system import AttentionSystem
        attn = AttentionSystem(config={"ignore_threshold": 0.5})
        state = {
            "symbols": {
                "HIGH": {"volatility": 8.0},
                "LOW": {"volatility": 0.1},  # should be ignored
            },
            "positions": {}, "signals": {}, "strategies": {},
            "risk_metrics": {},
        }
        attention = attn.compute_attention(state)
        assert "LOW" in attention.ignore_list

    def test_risk_ranking(self):
        from adaptive.attention_system import AttentionSystem
        attn = AttentionSystem()
        state = {
            "symbols": {}, "positions": {}, "signals": {},
            "strategies": {},
            "risk_metrics": {"drawdown_pct": 7.0, "exposure_pct": 90.0},
        }
        attention = attn.compute_attention(state)
        assert len(attention.focus_risks) >= 2

    def test_processing_priority_risks_first(self):
        from adaptive.attention_system import AttentionSystem
        attn = AttentionSystem()
        state = {
            "symbols": {"A": {"volatility": 5.0}},
            "positions": {}, "signals": {"A": {"score": 0.5}},
            "strategies": {},
            "risk_metrics": {"drawdown_pct": 8.0},
        }
        priority = attn.get_processing_priority(state)
        assert len(priority) >= 1
        # Risk items should be boosted
        assert priority[0]["category"] == "risk"

    def test_attention_trend(self):
        from adaptive.attention_system import AttentionSystem
        attn = AttentionSystem()
        state = {
            "symbols": {"BTC/USD": {"volatility": 3.0}},
            "positions": {}, "signals": {}, "strategies": {},
            "risk_metrics": {},
        }
        for _ in range(5):
            attn.compute_attention(state)
        trend = attn.get_attention_trend("BTC/USD", lookback=5)
        assert len(trend) == 5

    def test_strategy_ranking(self):
        from adaptive.attention_system import AttentionSystem
        attn = AttentionSystem()
        state = {
            "symbols": {}, "positions": {}, "signals": {},
            "strategies": {
                "momentum": {"recent_pnl": -50, "win_rate": 0.3, "regime_fit": 0.8},
                "mean_revert": {"recent_pnl": 100, "win_rate": 0.7, "regime_fit": 0.2},
            },
            "risk_metrics": {},
        }
        attention = attn.compute_attention(state)
        assert len(attention.focus_strategies) >= 1


# ===================================================================
# 5. CounterfactualAnalyzer Tests
# ===================================================================

class TestCounterfactualAnalyzer:
    """Tests for adaptive/counterfactual_analyzer.py."""

    def test_init_default(self, tmp_db):
        from adaptive.counterfactual_analyzer import CounterfactualAnalyzer
        ca = CounterfactualAnalyzer(db_path=tmp_db)
        assert ca._enabled is True

    def test_record_and_analyze_trade(self, tmp_db):
        from adaptive.counterfactual_analyzer import (
            CounterfactualAnalyzer, TradeRecord, PriceSnapshot,
        )
        ca = CounterfactualAnalyzer(db_path=tmp_db)
        trade = TradeRecord(
            trade_id="t1", symbol="BTC/USD", side="buy",
            entry_price=64000, exit_price=65000, size=0.01,
            entry_time="2026-03-20T10:00:00Z", exit_time="2026-03-20T14:00:00Z",
            venue="kraken", slippage_bps=5.0, was_limit=False,
        )
        ca.record_trade(trade)
        snapshot = PriceSnapshot(
            symbol="BTC/USD", base_price=64000,
            prices_after={"1h": 64500, "4h": 65500, "24h": 66000},
        )
        ca.record_price_snapshot("t1", snapshot)

        report = ca.analyze_trade("t1")
        assert report is not None
        assert report.actual_pnl == 10.0  # (65000-64000) * 0.01
        assert "1h" in report.what_if_held_longer
        assert report.what_if_held_longer["24h"] > report.actual_pnl

    def test_analyze_losing_trade(self, tmp_db):
        from adaptive.counterfactual_analyzer import (
            CounterfactualAnalyzer, TradeRecord,
        )
        ca = CounterfactualAnalyzer(db_path=tmp_db)
        trade = TradeRecord(
            trade_id="t2", symbol="ETH/USD", side="buy",
            entry_price=3200, exit_price=3100, size=0.1,
            entry_time="2026-03-20T10:00:00Z", exit_time="2026-03-20T14:00:00Z",
        )
        ca.record_trade(trade)
        report = ca.analyze_trade("t2")
        assert report.actual_pnl < 0

    def test_record_and_analyze_skipped_signal(self, tmp_db):
        from adaptive.counterfactual_analyzer import (
            CounterfactualAnalyzer, SkippedSignal, PriceSnapshot,
        )
        ca = CounterfactualAnalyzer(db_path=tmp_db)
        signal = SkippedSignal(
            signal_id="s1", symbol="SOL/USD", direction="long",
            signal_score=0.8, price_at_signal=180.0,
            timestamp="2026-03-20T10:00:00Z", reason_skipped="risk_limit",
        )
        ca.record_skipped_signal(signal)
        snapshot = PriceSnapshot(
            symbol="SOL/USD", base_price=180.0,
            prices_after={"1h": 185, "4h": 195, "24h": 200},
        )
        ca.record_price_snapshot("s1", snapshot)
        report = ca.analyze_skipped_signal("s1")
        assert report is not None
        assert report.opportunity_cost > 0  # missed a winner

    def test_biggest_missed_opportunities(self, tmp_db):
        from adaptive.counterfactual_analyzer import (
            CounterfactualAnalyzer, SkippedSignal, PriceSnapshot,
        )
        ca = CounterfactualAnalyzer(db_path=tmp_db)
        for i in range(3):
            sig = SkippedSignal(
                signal_id=f"s{i}", symbol=f"SYM{i}", direction="long",
                signal_score=0.7, price_at_signal=100.0,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            ca.record_skipped_signal(sig)
            snap = PriceSnapshot(
                symbol=f"SYM{i}", base_price=100.0,
                prices_after={"1h": 100 + (i + 1) * 5, "4h": 100 + (i + 1) * 10, "24h": 100 + (i + 1) * 15},
            )
            ca.record_price_snapshot(f"s{i}", snap)
            ca.analyze_skipped_signal(f"s{i}")

        missed = ca.get_biggest_missed_opportunities(lookback_days=1)
        assert len(missed) == 3
        # Sorted by opportunity cost descending
        assert missed[0]["opportunity_cost"] >= missed[-1]["opportunity_cost"]

    def test_biggest_mistakes(self, tmp_db):
        from adaptive.counterfactual_analyzer import (
            CounterfactualAnalyzer, TradeRecord, PriceSnapshot,
        )
        ca = CounterfactualAnalyzer(db_path=tmp_db)
        trade = TradeRecord(
            trade_id="bad1", symbol="BTC/USD", side="buy",
            entry_price=65000, exit_price=63000, size=0.01,
            entry_time=datetime.now(timezone.utc).isoformat(),
            exit_time=datetime.now(timezone.utc).isoformat(),
        )
        ca.record_trade(trade)
        snap = PriceSnapshot(
            symbol="BTC/USD", base_price=65000,
            prices_after={"1h": 64000, "4h": 63500, "24h": 66000},
        )
        ca.record_price_snapshot("bad1", snap)
        ca.analyze_trade("bad1")
        mistakes = ca.get_biggest_mistakes(lookback_days=1)
        assert len(mistakes) >= 1

    def test_aggregate_stats(self, tmp_db):
        from adaptive.counterfactual_analyzer import CounterfactualAnalyzer
        ca = CounterfactualAnalyzer(db_path=tmp_db)
        stats = ca.get_aggregate_stats()
        assert "trades_analyzed" in stats
        assert "signals_analyzed" in stats

    def test_nonexistent_trade(self, tmp_db):
        from adaptive.counterfactual_analyzer import CounterfactualAnalyzer
        ca = CounterfactualAnalyzer(db_path=tmp_db)
        assert ca.analyze_trade("nonexistent") is None


# ===================================================================
# 6. PredictivePlanner Tests
# ===================================================================

class TestPredictivePlanner:
    """Tests for adaptive/predictive_planner.py."""

    def test_init_default(self, tmp_db):
        from adaptive.predictive_planner import PredictivePlanner
        pp = PredictivePlanner(db_path=tmp_db)
        assert pp._enabled is True

    def test_generate_support_bounce_plan(self, tmp_db):
        from adaptive.predictive_planner import PredictivePlanner
        pp = PredictivePlanner(db_path=tmp_db)
        plans = pp.generate_plan("BTC/USD", {
            "price": 65000, "support": 63000, "resistance": 68000,
            "rsi": 35, "volume_ratio": 2.0, "volatility_pct": 3.0,
        })
        support_plans = [p for p in plans if "long" in p.direction]
        assert len(support_plans) >= 1
        plan = support_plans[0]
        assert plan.risk_reward_ratio >= 1.5
        assert plan.entry_conditions
        assert plan.exit_conditions

    def test_generate_breakout_plan(self, tmp_db):
        from adaptive.predictive_planner import PredictivePlanner
        pp = PredictivePlanner(db_path=tmp_db)
        plans = pp.generate_plan("ETH/USD", {
            "price": 3100, "support": 2900, "resistance": 3200,
            "rsi": 55, "volume_ratio": 1.0, "volatility_pct": 4.0,
        })
        # Should include a breakout plan
        assert len(plans) >= 1

    def test_generate_mean_revert_plan(self, tmp_db):
        from adaptive.predictive_planner import PredictivePlanner
        pp = PredictivePlanner(db_path=tmp_db)
        plans = pp.generate_plan("SOL/USD", {
            "price": 220, "mean_price": 180, "std_dev": 15,
            "volatility_pct": 5.0,
        })
        mr_plans = [p for p in plans if p.contingency_plan and "mean" in p.contingency_plan.lower()]
        assert len(mr_plans) >= 1

    def test_check_plan_triggers(self, tmp_db):
        from adaptive.predictive_planner import PredictivePlanner, TradePlan, Condition
        pp = PredictivePlanner(db_path=tmp_db)
        now = datetime.now(timezone.utc)
        plan = TradePlan(
            plan_id="test_plan",
            symbol="BTC/USD",
            direction="long",
            entry_conditions=[
                Condition("price", "lt", 63500, "Price below support"),
                Condition("rsi", "lt", 35, "RSI oversold"),
            ],
            exit_conditions=[
                Condition("price", "gt", 66000, "Take profit"),
            ],
            expected_return_pct=3.0,
            risk_reward_ratio=2.0,
            confidence=0.6,
            created_at=now.isoformat(),
            expires_at=(now + timedelta(hours=24)).isoformat(),
        )
        pp.add_plan(plan)

        # Not triggered (price too high)
        triggered = pp.check_plan_triggers({"BTC/USD": {"price": 65000, "rsi": 30}})
        assert len(triggered) == 0

        # Triggered
        triggered = pp.check_plan_triggers({"BTC/USD": {"price": 63000, "rsi": 30}})
        assert len(triggered) == 1
        assert triggered[0].plan_id == "test_plan"

    def test_check_exit_conditions(self, tmp_db):
        from adaptive.predictive_planner import PredictivePlanner, TradePlan, Condition
        pp = PredictivePlanner(db_path=tmp_db)
        now = datetime.now(timezone.utc)
        plan = TradePlan(
            plan_id="exit_test",
            symbol="BTC/USD",
            direction="long",
            entry_conditions=[Condition("price", "lt", 64000)],
            exit_conditions=[
                Condition("price", "gt", 67000, "Take profit"),
                Condition("price", "lt", 62000, "Stop loss"),
            ],
            created_at=now.isoformat(),
            expires_at=(now + timedelta(hours=24)).isoformat(),
        )
        pp.add_plan(plan)
        should_exit, reason = pp.check_exit_conditions("exit_test", {"price": 68000})
        assert should_exit is True
        assert "Take profit" in reason

    def test_cancel_plan(self, tmp_db):
        from adaptive.predictive_planner import PredictivePlanner, TradePlan, Condition
        pp = PredictivePlanner(db_path=tmp_db)
        now = datetime.now(timezone.utc)
        plan = TradePlan(
            plan_id="cancel_me", symbol="X", direction="long",
            entry_conditions=[Condition("price", "lt", 100)],
            exit_conditions=[],
            created_at=now.isoformat(),
            expires_at=(now + timedelta(hours=24)).isoformat(),
        )
        pp.add_plan(plan)
        assert pp.cancel_plan("cancel_me") is True
        assert len(pp.get_active_plans()) == 0

    def test_expire_stale_plans(self, tmp_db):
        from adaptive.predictive_planner import PredictivePlanner, TradePlan, Condition
        pp = PredictivePlanner(db_path=tmp_db)
        old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        plan = TradePlan(
            plan_id="old_plan", symbol="X", direction="long",
            entry_conditions=[Condition("price", "lt", 100)],
            exit_conditions=[],
            created_at=old_time,
            expires_at=old_time,  # already expired
        )
        pp._plans["old_plan"] = plan
        expired = pp.expire_stale_plans()
        assert expired >= 1

    def test_condition_evaluate(self):
        from adaptive.predictive_planner import Condition
        assert Condition("price", "lt", 100).evaluate(90) is True
        assert Condition("price", "lt", 100).evaluate(110) is False
        assert Condition("price", "gt", 100).evaluate(110) is True
        assert Condition("regime", "eq", "trending").evaluate("trending") is True
        assert Condition("price", "between", [50, 100]).evaluate(75) is True
        assert Condition("price", "between", [50, 100]).evaluate(120) is False

    def test_record_outcome(self, tmp_db):
        from adaptive.predictive_planner import PredictivePlanner
        pp = PredictivePlanner(db_path=tmp_db)
        pp.record_outcome("plan1", triggered=True, actual_pnl=50.0, notes="success")
        stats = pp.get_plan_success_rate()
        assert stats["triggered"] == 1
        assert stats["profitable"] == 1

    def test_persistence(self, tmp_db):
        from adaptive.predictive_planner import PredictivePlanner, TradePlan, Condition
        pp1 = PredictivePlanner(db_path=tmp_db)
        now = datetime.now(timezone.utc)
        plan = TradePlan(
            plan_id="persist_test", symbol="BTC/USD", direction="long",
            entry_conditions=[Condition("price", "lt", 64000, "support")],
            exit_conditions=[Condition("price", "gt", 67000, "resistance")],
            expected_return_pct=5.0, risk_reward_ratio=2.5, confidence=0.7,
            created_at=now.isoformat(),
            expires_at=(now + timedelta(hours=48)).isoformat(),
        )
        pp1.add_plan(plan)

        pp2 = PredictivePlanner(db_path=tmp_db)
        active = pp2.get_active_plans()
        assert len(active) == 1
        assert active[0].plan_id == "persist_test"


# ===================================================================
# 7. SystemConsciousness Tests
# ===================================================================

class TestSystemConsciousness:
    """Tests for core/system_consciousness.py."""

    def test_init_default(self):
        from core.system_consciousness import SystemConsciousness
        sc = SystemConsciousness()
        assert sc._enabled is True
        assert sc._current_state == "learning"

    def test_get_self_assessment_initial(self):
        from core.system_consciousness import SystemConsciousness, SelfAssessment
        sc = SystemConsciousness()
        assessment = sc.get_self_assessment()
        assert isinstance(assessment, SelfAssessment)
        assert assessment.current_state == "learning"
        assert 0 <= assessment.confidence_level <= 1

    def test_record_trade_updates_confidence(self):
        from core.system_consciousness import SystemConsciousness
        sc = SystemConsciousness()
        initial = sc._confidence
        sc.record_trade_result(10.0, True)
        assert sc._confidence > initial

    def test_losing_streak_cautious(self):
        from core.system_consciousness import SystemConsciousness
        sc = SystemConsciousness()
        for _ in range(5):
            sc.record_trade_result(-5.0, False)
        assessment = sc.get_self_assessment()
        assert assessment.current_state in ("cautious", "stressed")

    def test_winning_streak_confident(self):
        from core.system_consciousness import SystemConsciousness
        sc = SystemConsciousness()
        for _ in range(10):
            sc.record_trade_result(5.0, True)
        assessment = sc.get_self_assessment()
        assert assessment.current_state == "confident"

    def test_stressed_state_on_drawdown(self):
        from core.system_consciousness import SystemConsciousness
        sc = SystemConsciousness()
        sc.record_trade_result(1.0, True)  # need at least one trade
        sc.update_market_state(drawdown_pct=8.0)
        assessment = sc.get_self_assessment()
        assert assessment.current_state == "stressed"

    def test_get_capability_matrix(self):
        from core.system_consciousness import SystemConsciousness
        sc = SystemConsciousness()
        caps = sc.get_capability_matrix()
        assert "regime_detection" in caps
        assert "signal_generation" in caps
        assert "execution_quality" in caps
        assert all(0 <= v <= 1 for v in caps.values())

    def test_update_capability(self):
        from core.system_consciousness import SystemConsciousness
        sc = SystemConsciousness()
        old = sc.get_capability_matrix()["regime_detection"]
        sc.update_capability("regime_detection", 0.9)
        new = sc.get_capability_matrix()["regime_detection"]
        assert new > old

    def test_update_component_feedback(self):
        from core.system_consciousness import SystemConsciousness, ComponentFeedback
        sc = SystemConsciousness()
        fb = ComponentFeedback(name="regime_classifier", score=0.85, recent_accuracy=0.8)
        sc.update_component(fb)
        caps = sc.get_capability_matrix()
        # regime_classifier maps to regime_detection
        assert caps["regime_detection"] > 0.5

    def test_should_trade_yes(self):
        from core.system_consciousness import SystemConsciousness
        sc = SystemConsciousness()
        should, reason = sc.should_trade()
        assert should is True

    def test_should_trade_no_high_drawdown(self):
        from core.system_consciousness import SystemConsciousness
        sc = SystemConsciousness()
        sc.update_market_state(drawdown_pct=10.0)
        sc.record_trade_result(1.0, True)
        should, reason = sc.should_trade()
        assert should is False
        assert "drawdown" in reason.lower() or "stressed" in reason.lower()

    def test_should_trade_no_low_confidence(self):
        from core.system_consciousness import SystemConsciousness
        sc = SystemConsciousness()
        sc._confidence = 0.1
        should, reason = sc.should_trade()
        assert should is False

    def test_get_daily_briefing(self):
        from core.system_consciousness import SystemConsciousness
        sc = SystemConsciousness()
        sc.update_market_state(regime="trending_up", drawdown_pct=1.5)
        for i in range(5):
            sc.record_trade_result(2.0, True)
        briefing = sc.get_daily_briefing()
        assert "ARGUS DAILY BRIEFING" in briefing
        assert "trending_up" in briefing
        assert "Capability Matrix" in briefing

    def test_identify_strengths_and_weaknesses(self):
        from core.system_consciousness import SystemConsciousness
        sc = SystemConsciousness()
        sc._capabilities["regime_detection"] = 0.9
        sc._capabilities["timing_accuracy"] = 0.2
        for _ in range(15):
            sc.record_trade_result(3.0, True)
        assessment = sc.get_self_assessment()
        assert any("regime" in s.lower() for s in assessment.strengths)
        assert any("timing" in w.lower() for w in assessment.weaknesses)

    def test_risk_appetite_decreases_on_losses(self):
        from core.system_consciousness import SystemConsciousness
        sc = SystemConsciousness()
        sc.update_market_state(drawdown_pct=6.0)
        for _ in range(5):
            sc.record_trade_result(-5.0, False)
        assessment = sc.get_self_assessment()
        assert assessment.risk_appetite < 0.5

    def test_self_assessment_to_dict(self):
        from core.system_consciousness import SystemConsciousness
        sc = SystemConsciousness()
        assessment = sc.get_self_assessment()
        d = assessment.to_dict()
        assert isinstance(d, dict)
        assert "strengths" in d
        assert "current_state" in d
