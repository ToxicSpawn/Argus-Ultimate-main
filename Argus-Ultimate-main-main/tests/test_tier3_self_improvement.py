#!/usr/bin/env python3
"""
Tests for Tier 3 Self-Improvement modules.

Covers:
  - strategies.strategy_generator  (StrategyGenerator)
  - execution.microstructure_adapter  (MicrostructureAdapter)
  - core.cross_session_memory  (CrossSessionMemory)
  - adaptive.learning_journal  (LearningJournal)

40+ tests total, 10+ per module.
"""

from __future__ import annotations

import json
import math
import os
import random
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Provide a temporary data directory for all modules."""
    return str(tmp_path)


# ===================================================================
# 1. StrategyGenerator Tests
# ===================================================================


from strategies.strategy_generator import (
    BacktestResult,
    StrategyGenerator,
    StrategyIdea,
)


def _make_ohlcv(n: int = 200, base_price: float = 100.0, seed: int = 42) -> list:
    """Generate synthetic OHLCV data for backtesting."""
    rng = random.Random(seed)
    bars = []
    price = base_price
    for _ in range(n):
        change = rng.gauss(0, 0.02)
        o = price
        c = price * (1 + change)
        h = max(o, c) * (1 + abs(rng.gauss(0, 0.005)))
        lo = min(o, c) * (1 - abs(rng.gauss(0, 0.005)))
        vol = rng.uniform(100, 1000)
        bars.append({"open": o, "high": h, "low": lo, "close": c, "volume": vol})
        price = c
    return bars


class TestStrategyGenerator:
    """Tests for StrategyGenerator."""

    def test_init(self, tmp_data_dir):
        """Generator initialises and creates data directory."""
        gen = StrategyGenerator(data_dir=tmp_data_dir)
        assert Path(tmp_data_dir).exists()

    def test_rule_based_generation_any_regime(self, tmp_data_dir):
        """Rule-based generation works with any market conditions."""
        gen = StrategyGenerator(data_dir=tmp_data_dir)
        idea = gen.generate_strategy_idea({"regime": "bull", "volatility": "high"})
        assert isinstance(idea, StrategyIdea)
        assert idea.name
        assert idea.entry_logic
        assert idea.exit_logic
        assert 0 < idea.confidence <= 1.0

    def test_rule_based_ranging_regime(self, tmp_data_dir):
        """Ranging regime selects RSI or BB templates."""
        gen = StrategyGenerator(data_dir=tmp_data_dir)
        # Run multiple times — at least one should match ranging
        names = set()
        for _ in range(50):
            idea = gen.generate_strategy_idea({"regime": "ranging"})
            names.add(idea.name)
        # rsi_oversold_bounce and bollinger_mean_reversion both prefer 'ranging'
        assert names & {"rsi_oversold_bounce", "bollinger_mean_reversion"}

    def test_high_volatility_adapts_risk(self, tmp_data_dir):
        """High volatility widens stop loss and reduces position size."""
        gen = StrategyGenerator(data_dir=tmp_data_dir)
        idea = gen.generate_strategy_idea({"regime": "ranging", "volatility": "high"})
        # position_size_pct should be scaled down (x0.7)
        assert idea.risk_params.get("position_size_pct", 999) < 12.0

    def test_low_volatility_adapts_risk(self, tmp_data_dir):
        """Low volatility tightens stop loss and increases position size."""
        gen = StrategyGenerator(data_dir=tmp_data_dir)
        idea = gen.generate_strategy_idea({"regime": "ranging", "volatility": "low"})
        # position_size_pct should be scaled up (x1.2)
        assert idea.risk_params.get("position_size_pct", 0) > 5.0

    def test_backtest_insufficient_data(self, tmp_data_dir):
        """Backtest with < 30 bars returns failed result."""
        gen = StrategyGenerator(data_dir=tmp_data_dir)
        idea = gen.generate_strategy_idea({})
        result = gen.backtest_idea(idea, _make_ohlcv(n=10))
        assert result.passed is False
        assert result.trade_count == 0

    def test_backtest_produces_result(self, tmp_data_dir):
        """Backtest on adequate data produces a BacktestResult with metrics."""
        gen = StrategyGenerator(data_dir=tmp_data_dir)
        idea = gen.generate_strategy_idea({"regime": "ranging"})
        result = gen.backtest_idea(idea, _make_ohlcv(n=500))
        assert isinstance(result, BacktestResult)
        assert result.idea_name == idea.name
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown_pct, float)

    def test_backtest_pass_criteria(self, tmp_data_dir):
        """Pass criteria check: sharpe > 0.5, dd < 20%, trades > 10."""
        gen = StrategyGenerator(data_dir=tmp_data_dir, min_sharpe=0.5, max_drawdown_pct=20.0, min_trade_count=10)
        idea = gen.generate_strategy_idea({})
        result = gen.backtest_idea(idea, _make_ohlcv(n=500))
        if result.passed:
            assert result.sharpe_ratio > 0.5
            assert result.max_drawdown_pct < 20.0
            assert result.trade_count >= 10

    def test_promote_and_retrieve(self, tmp_data_dir):
        """Promoted strategies persist and can be retrieved."""
        gen = StrategyGenerator(data_dir=tmp_data_dir)
        idea = StrategyIdea(
            name="test_strat",
            description="test",
            entry_logic="RSI(14) < 30",
            exit_logic="RSI(14) > 70",
            regime_preference="ranging",
            risk_params={"stop_loss_pct": 2.0},
            confidence=0.8,
        )
        result = BacktestResult(
            idea_name="test_strat",
            total_return_pct=12.5,
            sharpe_ratio=1.2,
            max_drawdown_pct=8.0,
            win_rate=0.6,
            trade_count=25,
            passed=True,
        )
        gen.promote_strategy(idea, result)
        promoted = gen.get_promoted_strategies()
        assert len(promoted) == 1
        assert promoted[0].name == "test_strat"

    def test_promote_multiple(self, tmp_data_dir):
        """Multiple promotions accumulate."""
        gen = StrategyGenerator(data_dir=tmp_data_dir)
        for i in range(3):
            idea = StrategyIdea(
                name=f"strat_{i}", description="t", entry_logic="x", exit_logic="y",
                regime_preference="any", risk_params={}, confidence=0.7,
            )
            result = BacktestResult(
                idea_name=f"strat_{i}", total_return_pct=5.0, sharpe_ratio=1.0,
                max_drawdown_pct=5.0, win_rate=0.5, trade_count=20, passed=True,
            )
            gen.promote_strategy(idea, result)
        assert len(gen.get_promoted_strategies()) == 3

    def test_llm_fallback_on_error(self, tmp_data_dir):
        """If LLM client raises, falls back to rule-based."""

        class BadLLM:
            def generate(self, prompt):
                raise RuntimeError("LLM unavailable")

        gen = StrategyGenerator(data_dir=tmp_data_dir)
        idea = gen.generate_strategy_idea({"regime": "bull"}, llm_client=BadLLM())
        assert isinstance(idea, StrategyIdea)

    def test_llm_valid_response(self, tmp_data_dir):
        """LLM client returning valid JSON produces a StrategyIdea."""

        class MockLLM:
            def generate(self, prompt):
                return json.dumps({
                    "name": "llm_momentum",
                    "description": "Momentum strategy from LLM",
                    "entry_logic": "RSI(14) > 50 AND MACD > 0",
                    "exit_logic": "RSI(14) < 40",
                    "regime_preference": "trending",
                    "risk_params": {"stop_loss_pct": 2.0, "take_profit_pct": 5.0, "position_size_pct": 10.0},
                    "confidence": 0.75,
                })

        gen = StrategyGenerator(data_dir=tmp_data_dir)
        idea = gen.generate_strategy_idea({}, llm_client=MockLLM())
        assert idea.name == "llm_momentum"
        assert idea.confidence == 0.75


# ===================================================================
# 2. MicrostructureAdapter Tests
# ===================================================================


from execution.microstructure_adapter import (
    ExecutionRecommendation,
    MicrostructureAdapter,
)


class TestMicrostructureAdapter:
    """Tests for MicrostructureAdapter."""

    def test_init_creates_db(self, tmp_data_dir):
        """Adapter creates SQLite database on init."""
        db = os.path.join(tmp_data_dir, "micro.db")
        adapter = MicrostructureAdapter(db_path=db)
        assert Path(db).exists()

    def test_record_execution(self, tmp_data_dir):
        """Recording an execution stores a row."""
        db = os.path.join(tmp_data_dir, "micro.db")
        adapter = MicrostructureAdapter(db_path=db)
        adapter.record_execution("BTC/AUD", "kraken", "limit", 14, 1.5, 80, 500.0)
        conn = sqlite3.connect(db)
        count = conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0]
        conn.close()
        assert count == 1

    def test_optimal_order_type_insufficient_data(self, tmp_data_dir):
        """With < 30 executions, defaults to 'limit'."""
        db = os.path.join(tmp_data_dir, "micro.db")
        adapter = MicrostructureAdapter(db_path=db)
        for i in range(10):
            adapter.record_execution("BTC/AUD", "kraken", "market", 10, 3.0, 50, 100)
        assert adapter.get_optimal_order_type("BTC/AUD", "kraken") == "limit"

    def test_optimal_order_type_with_data(self, tmp_data_dir):
        """With enough data, picks the order type with lower avg slippage."""
        db = os.path.join(tmp_data_dir, "micro.db")
        adapter = MicrostructureAdapter(db_path=db)
        # 35 limit orders with low slippage
        for i in range(35):
            adapter.record_execution("BTC/AUD", "kraken", "limit", 10, 1.0, 100, 200)
        # 35 market orders with high slippage
        for i in range(35):
            adapter.record_execution("BTC/AUD", "kraken", "market", 10, 8.0, 30, 200)
        assert adapter.get_optimal_order_type("BTC/AUD", "kraken") == "limit"

    def test_optimal_timing_empty(self, tmp_data_dir):
        """No data returns empty best/worst hours."""
        db = os.path.join(tmp_data_dir, "micro.db")
        adapter = MicrostructureAdapter(db_path=db)
        timing = adapter.get_optimal_timing("ETH/AUD")
        assert timing["best_hours"] == []
        assert timing["worst_hours"] == []

    def test_optimal_timing_with_data(self, tmp_data_dir):
        """Hours with lower slippage rank as best."""
        db = os.path.join(tmp_data_dir, "micro.db")
        adapter = MicrostructureAdapter(db_path=db)
        # Hour 2: low slippage
        for _ in range(35):
            adapter.record_execution("BTC/AUD", "kraken", "limit", 2, 0.5, 80, 200)
        # Hour 14: high slippage
        for _ in range(35):
            adapter.record_execution("BTC/AUD", "kraken", "limit", 14, 10.0, 80, 200)
        timing = adapter.get_optimal_timing("BTC/AUD")
        assert 2 in timing["best_hours"]
        assert 14 in timing["worst_hours"]

    def test_venue_preference_empty(self, tmp_data_dir):
        """No data returns empty venue list."""
        db = os.path.join(tmp_data_dir, "micro.db")
        adapter = MicrostructureAdapter(db_path=db)
        assert adapter.get_venue_preference("BTC/AUD") == []

    def test_venue_preference_ranking(self, tmp_data_dir):
        """Exchange with lower slippage + fill time ranks first."""
        db = os.path.join(tmp_data_dir, "micro.db")
        adapter = MicrostructureAdapter(db_path=db)
        # Kraken: good
        for _ in range(35):
            adapter.record_execution("BTC/AUD", "kraken", "limit", 10, 1.0, 50, 200)
        # Coinbase: bad
        for _ in range(35):
            adapter.record_execution("BTC/AUD", "coinbase", "limit", 10, 8.0, 300, 200)
        venues = adapter.get_venue_preference("BTC/AUD")
        assert len(venues) == 2
        assert venues[0] == "kraken"

    def test_execution_recommendation_default(self, tmp_data_dir):
        """Recommendation with no data uses conservative defaults."""
        db = os.path.join(tmp_data_dir, "micro.db")
        adapter = MicrostructureAdapter(db_path=db)
        rec = adapter.get_execution_recommendation("BTC/AUD", "kraken", 500.0)
        assert isinstance(rec, ExecutionRecommendation)
        assert rec.expected_slippage_bps == 5.0
        assert "Insufficient" in rec.timing_advice

    def test_execution_recommendation_split_large_order(self, tmp_data_dir):
        """Large orders should be split."""
        db = os.path.join(tmp_data_dir, "micro.db")
        adapter = MicrostructureAdapter(db_path=db)
        rec = adapter.get_execution_recommendation("BTC/AUD", "kraken", 6000.0)
        assert rec.split_count == 5

    def test_execution_recommendation_split_medium_order(self, tmp_data_dir):
        """Medium orders get moderate splitting."""
        db = os.path.join(tmp_data_dir, "micro.db")
        adapter = MicrostructureAdapter(db_path=db)
        rec = adapter.get_execution_recommendation("BTC/AUD", "kraken", 3000.0)
        assert rec.split_count == 3

    def test_execution_recommendation_no_split_small(self, tmp_data_dir):
        """Small orders should not be split."""
        db = os.path.join(tmp_data_dir, "micro.db")
        adapter = MicrostructureAdapter(db_path=db)
        rec = adapter.get_execution_recommendation("BTC/AUD", "kraken", 100.0)
        assert rec.split_count == 1

    def test_execution_recommendation_with_history(self, tmp_data_dir):
        """With enough history, recommendation uses actual slippage data."""
        db = os.path.join(tmp_data_dir, "micro.db")
        adapter = MicrostructureAdapter(db_path=db)
        for _ in range(35):
            adapter.record_execution("BTC/AUD", "kraken", "limit", 10, 2.0, 80, 200)
        rec = adapter.get_execution_recommendation("BTC/AUD", "kraken", 500.0)
        assert rec.expected_slippage_bps == 2.0
        assert rec.order_type == "limit"


# ===================================================================
# 3. CrossSessionMemory Tests
# ===================================================================


from core.cross_session_memory import CATEGORIES, CrossSessionMemory, Insight


class TestCrossSessionMemory:
    """Tests for CrossSessionMemory."""

    def test_init_creates_db(self, tmp_data_dir):
        """Memory creates database on init."""
        db = os.path.join(tmp_data_dir, "memory.db")
        mem = CrossSessionMemory(db_path=db)
        assert Path(db).exists()

    def test_record_insight(self, tmp_data_dir):
        """Recording an insight returns a positive row ID."""
        db = os.path.join(tmp_data_dir, "memory.db")
        mem = CrossSessionMemory(db_path=db)
        row_id = mem.record_insight("strategy_performance", "rsi_sharpe", 1.5)
        assert row_id > 0

    def test_invalid_category_raises(self, tmp_data_dir):
        """Recording with an invalid category raises ValueError."""
        db = os.path.join(tmp_data_dir, "memory.db")
        mem = CrossSessionMemory(db_path=db)
        with pytest.raises(ValueError, match="Unknown insight category"):
            mem.record_insight("invalid_category", "key", "value")

    def test_get_insights_by_category(self, tmp_data_dir):
        """Filtering by category returns only matching insights."""
        db = os.path.join(tmp_data_dir, "memory.db")
        mem = CrossSessionMemory(db_path=db)
        mem.record_insight("strategy_performance", "k1", "v1")
        mem.record_insight("risk_event", "k2", "v2")
        results = mem.get_insights(category="strategy_performance")
        assert len(results) == 1
        assert results[0].key == "k1"

    def test_get_insights_min_confidence(self, tmp_data_dir):
        """Low-confidence insights are filtered out."""
        db = os.path.join(tmp_data_dir, "memory.db")
        mem = CrossSessionMemory(db_path=db)
        mem.record_insight("market_pattern", "low", "val", confidence=0.2)
        mem.record_insight("market_pattern", "high", "val", confidence=0.9)
        results = mem.get_insights(min_confidence=0.5)
        assert len(results) == 1
        assert results[0].key == "high"

    def test_confirm_insight_boosts_confidence(self, tmp_data_dir):
        """Confirming an insight increments count and boosts confidence."""
        db = os.path.join(tmp_data_dir, "memory.db")
        mem = CrossSessionMemory(db_path=db)
        mem.record_insight("execution_quality", "slip_low", "1.2bps", confidence=0.6)
        mem.confirm_insight("execution_quality", "slip_low")
        results = mem.get_insights(category="execution_quality", min_confidence=0.0)
        assert results[0].times_confirmed == 1
        assert results[0].confidence == pytest.approx(0.65, abs=0.01)

    def test_confirm_nonexistent(self, tmp_data_dir):
        """Confirming a nonexistent insight returns False."""
        db = os.path.join(tmp_data_dir, "memory.db")
        mem = CrossSessionMemory(db_path=db)
        assert mem.confirm_insight("risk_event", "nope") is False

    def test_invalidate_insight(self, tmp_data_dir):
        """Invalidated insights are excluded from queries."""
        db = os.path.join(tmp_data_dir, "memory.db")
        mem = CrossSessionMemory(db_path=db)
        mem.record_insight("regime_transition", "bull_to_bear", {"duration": 14})
        mem.invalidate_insight("regime_transition", "bull_to_bear")
        results = mem.get_insights(category="regime_transition")
        assert len(results) == 0

    def test_startup_briefing_empty(self, tmp_data_dir):
        """Startup briefing on empty DB returns zeroed-out structure."""
        db = os.path.join(tmp_data_dir, "memory.db")
        mem = CrossSessionMemory(db_path=db)
        briefing = mem.get_startup_briefing()
        assert briefing["total_insights"] == 0
        assert briefing["best_strategies"] == []

    def test_startup_briefing_with_data(self, tmp_data_dir):
        """Startup briefing reflects recorded insights."""
        db = os.path.join(tmp_data_dir, "memory.db")
        mem = CrossSessionMemory(db_path=db)
        mem.record_insight("strategy_performance", "macd_cross", {"sharpe": 1.3}, confidence=0.9)
        mem.record_insight("risk_event", "circuit_breaker_trip", {"dd_pct": 12.5})
        mem.record_insight("model_drift", "regime_model_stale", {"accuracy": 0.52})
        briefing = mem.get_startup_briefing()
        assert briefing["total_insights"] == 3
        assert len(briefing["best_strategies"]) == 1
        assert len(briefing["risk_events"]) == 1
        assert len(briefing["model_drift_warnings"]) == 1

    def test_prune_stale(self, tmp_data_dir):
        """Pruning removes old unconfirmed insights."""
        db = os.path.join(tmp_data_dir, "memory.db")
        mem = CrossSessionMemory(db_path=db)
        # Insert an old insight directly
        old_ts = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        conn = sqlite3.connect(db)
        conn.execute(
            """INSERT INTO insights (category, key, value, confidence, source, timestamp, times_confirmed, invalidated)
               VALUES (?, ?, ?, ?, ?, ?, 0, 0)""",
            ("market_pattern", "old_pattern", '"stale"', 0.5, "test", old_ts),
        )
        conn.commit()
        conn.close()
        # Also add a recent one
        mem.record_insight("market_pattern", "fresh", "new")
        pruned = mem.prune_stale(max_age_days=90)
        assert pruned == 1
        # Fresh one remains
        results = mem.get_insights(category="market_pattern", min_confidence=0.0)
        assert len(results) == 1
        assert results[0].key == "fresh"

    def test_confidence_clamped(self, tmp_data_dir):
        """Confidence is clamped to [0, 1]."""
        db = os.path.join(tmp_data_dir, "memory.db")
        mem = CrossSessionMemory(db_path=db)
        mem.record_insight("strategy_performance", "high_conf", "val", confidence=5.0)
        results = mem.get_insights(min_confidence=0.0)
        assert results[0].confidence == 1.0


# ===================================================================
# 4. LearningJournal Tests
# ===================================================================


from adaptive.learning_journal import EVENT_TYPES, JournalEntry, LearningJournal


class TestLearningJournal:
    """Tests for LearningJournal."""

    def test_init_creates_db(self, tmp_data_dir):
        """Journal creates database on init."""
        db = os.path.join(tmp_data_dir, "journal.db")
        journal = LearningJournal(db_path=db)
        assert Path(db).exists()

    def test_record_event(self, tmp_data_dir):
        """Recording an event returns a positive row ID."""
        db = os.path.join(tmp_data_dir, "journal.db")
        journal = LearningJournal(db_path=db)
        entry_id = journal.record_event(
            "trade_win", "BTC long closed +3%",
            {"profit_pct": 3.0}, "Trend-following works in bull regime"
        )
        assert entry_id > 0

    def test_invalid_event_type_raises(self, tmp_data_dir):
        """Recording with invalid event type raises ValueError."""
        db = os.path.join(tmp_data_dir, "journal.db")
        journal = LearningJournal(db_path=db)
        with pytest.raises(ValueError, match="Unknown event type"):
            journal.record_event("invalid_type", "desc", {}, "lesson")

    def test_get_lessons_all(self, tmp_data_dir):
        """Get all lessons within lookback period."""
        db = os.path.join(tmp_data_dir, "journal.db")
        journal = LearningJournal(db_path=db)
        journal.record_event("trade_win", "win1", {}, "lesson1")
        journal.record_event("trade_loss", "loss1", {}, "lesson2")
        lessons = journal.get_lessons()
        assert len(lessons) == 2

    def test_get_lessons_by_type(self, tmp_data_dir):
        """Filtering by event type works."""
        db = os.path.join(tmp_data_dir, "journal.db")
        journal = LearningJournal(db_path=db)
        journal.record_event("trade_win", "win1", {}, "lesson1")
        journal.record_event("trade_loss", "loss1", {}, "lesson2")
        journal.record_event("trade_loss", "loss2", {}, "lesson3")
        losses = journal.get_lessons(event_type="trade_loss")
        assert len(losses) == 2

    def test_actionable_items(self, tmp_data_dir):
        """Actionable items returns unresolved entries."""
        db = os.path.join(tmp_data_dir, "journal.db")
        journal = LearningJournal(db_path=db)
        journal.record_event("risk_breach", "dd hit 10%", {"dd_pct": 10}, "Reduce size", actionable=True)
        journal.record_event("new_high", "equity ATH", {"equity": 1200}, "Keep going", actionable=False)
        items = journal.get_actionable_items()
        assert len(items) == 1
        assert items[0].event_type == "risk_breach"

    def test_mark_resolved(self, tmp_data_dir):
        """Marking as resolved removes from actionable list."""
        db = os.path.join(tmp_data_dir, "journal.db")
        journal = LearningJournal(db_path=db)
        eid = journal.record_event("strategy_decay", "rsi losing edge", {}, "Review params")
        assert len(journal.get_actionable_items()) == 1
        journal.mark_resolved(eid)
        assert len(journal.get_actionable_items()) == 0

    def test_mark_resolved_nonexistent(self, tmp_data_dir):
        """Marking nonexistent entry returns False."""
        db = os.path.join(tmp_data_dir, "journal.db")
        journal = LearningJournal(db_path=db)
        assert journal.mark_resolved(9999) is False

    def test_daily_summary_empty(self, tmp_data_dir):
        """Empty journal produces a clean summary."""
        db = os.path.join(tmp_data_dir, "journal.db")
        journal = LearningJournal(db_path=db)
        summary = journal.generate_daily_summary()
        assert "No journal entries" in summary

    def test_daily_summary_with_entries(self, tmp_data_dir):
        """Summary includes event counts and lessons."""
        db = os.path.join(tmp_data_dir, "journal.db")
        journal = LearningJournal(db_path=db)
        journal.record_event("trade_win", "BTC long +5%", {"pct": 5}, "Momentum is strong")
        journal.record_event("trade_loss", "ETH short -2%", {"pct": -2}, "Avoid counter-trend")
        journal.record_event("drawdown", "Portfolio -8%", {"dd": 8}, "Reduce exposure")
        summary = journal.generate_daily_summary()
        assert "3 events" in summary
        assert "TRADE_WIN" in summary
        assert "TRADE_LOSS" in summary
        assert "DRAWDOWN" in summary

    def test_all_event_types_valid(self, tmp_data_dir):
        """All defined event types are accepted."""
        db = os.path.join(tmp_data_dir, "journal.db")
        journal = LearningJournal(db_path=db)
        for etype in EVENT_TYPES:
            eid = journal.record_event(etype, f"test {etype}", {}, f"lesson for {etype}")
            assert eid > 0

    def test_metrics_preserved(self, tmp_data_dir):
        """Metrics dict round-trips through JSON correctly."""
        db = os.path.join(tmp_data_dir, "journal.db")
        journal = LearningJournal(db_path=db)
        metrics = {"profit_pct": 3.14, "regime": "bull", "symbols": ["BTC", "ETH"]}
        journal.record_event("trade_win", "test", metrics, "lesson")
        entries = journal.get_lessons()
        assert entries[0].metrics == metrics

    def test_journal_entry_fields(self, tmp_data_dir):
        """JournalEntry has all expected fields."""
        db = os.path.join(tmp_data_dir, "journal.db")
        journal = LearningJournal(db_path=db)
        journal.record_event("model_retrain", "regime model retrained", {"acc": 0.85}, "Accuracy improved")
        entry = journal.get_lessons()[0]
        assert isinstance(entry, JournalEntry)
        assert entry.event_type == "model_retrain"
        assert entry.actionable is True
        assert entry.resolved is False
        assert entry.timestamp
