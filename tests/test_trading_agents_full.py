from __future__ import annotations

from pathlib import Path

from core.event_bus import EventBus
from ml.trading_agents_full import DecisionMemory, MarketContext, TradingCoordinator
from ml.trading_agents_full.signal_synthesizer import SignalSynthesizer


def _context() -> MarketContext:
    return MarketContext(
        symbol="BTCUSDT",
        regime="trending_bull",
        market_data={
            "liquidity_score": 0.8,
            "realized_volatility": 0.35,
            "drawdown_risk": 0.2,
        },
        technical_indicators={
            "rsi": 62.0,
            "macd": 0.4,
            "trend_strength": 0.7,
            "breakout_score": 0.5,
            "volume_confirmation": 0.4,
        },
        fundamentals={
            "revenue_growth": 0.5,
            "earnings_growth": 0.4,
            "valuation_discount": 0.3,
            "free_cash_flow_margin": 0.2,
            "debt_to_equity": 0.15,
        },
        sentiment_data={
            "news_score": 0.45,
            "social_score": 0.35,
            "fear_greed_index": 62.0,
        },
        news_data={
            "event_impact": 0.4,
            "surprise_score": 0.25,
            "regulatory_risk": 0.1,
        },
        portfolio_state={
            "gross_exposure": 0.25,
            "symbol_exposure": 0.03,
            "value_at_risk": 0.02,
            "drawdown": 0.04,
        },
        risk_limits={
            "max_exposure": 1.0,
            "max_position": 0.1,
            "max_value_at_risk": 0.10,
            "max_drawdown": 0.15,
        },
        news_headlines=["ETF inflows accelerate", "On-chain demand improves"],
    )


def test_coordinator_runs_heuristic_cycle_and_records_memory(tmp_path: Path):
    memory = DecisionMemory(str(tmp_path / "memory.sqlite"))
    coordinator = TradingCoordinator(memory_store=memory)
    result = coordinator.run_cycle(_context(), publish=False)

    assert result.signal.action in {"buy", "sell", "hold"}
    assert 0.0 <= result.signal.confidence <= 1.0
    assert result.decision_record.symbol == "BTCUSDT"
    assert result.pattern_insight is not None
    assert coordinator.metrics.cycles_run == 1
    assert coordinator.metrics.heuristic_cycles == 1


def test_memory_store_similarity_and_pattern_learning(tmp_path: Path):
    memory = DecisionMemory(str(tmp_path / "memory.sqlite"))
    record = memory.store_decision(
        symbol="ETHUSDT",
        action="buy",
        confidence=0.7,
        regime="range_bound",
        net_score=0.22,
        reasoning=["test"],
        context_signature={"trend_strength": 0.2, "realized_volatility": 0.3},
    )
    memory.update_outcome(record.decision_id, 0.05)

    similar = memory.query_similar_situations(
        symbol="ETHUSDT",
        regime="range_bound",
        context_signature={"trend_strength": 0.22, "realized_volatility": 0.28},
    )
    insight = memory.learn_pattern_summary("ETHUSDT", "range_bound")

    assert similar
    assert similar[0].similarity > 0.9
    assert insight.sample_size == 1
    assert insight.preferred_action == "buy"


def test_synthesizer_blocks_trade_when_risk_limit_breached():
    synthesizer = SignalSynthesizer()
    analyses = TradingCoordinator().run_cycle(_context(), publish=False).analyses
    analyses["risk_manager"].risk_flags.append("risk_limit_breached")
    analyses["risk_manager"].score = -1.0
    analyses["risk_manager"].action = "sell"

    signal = synthesizer.synthesize(list(analyses.values()), regime="stress_bear", symbol="BTCUSDT")

    assert signal.action == "hold"
    assert "risk_limit_breached" in signal.risk_overrides


def test_signal_exposes_unified_adapter_payload():
    signal = TradingCoordinator().run_cycle(_context(), publish=False).signal
    unified_signal = signal.to_unified_signal()

    assert unified_signal is not None
    assert getattr(unified_signal, "symbol") == "BTCUSDT"
    assert getattr(unified_signal, "action") in {"BUY", "SELL", "HOLD"}


def test_coordinator_publishes_signal_generated_event(tmp_path: Path):
    memory = DecisionMemory(str(tmp_path / "memory.sqlite"))
    event_bus = EventBus()
    captured = []

    def _handler(event_type, data):
        captured.append((event_type, data))

    event_bus.subscribe("signal_generated", _handler)
    coordinator = TradingCoordinator(memory_store=memory, event_bus=event_bus)
    coordinator.run_cycle(_context(), publish=True)

    assert captured
    assert captured[0][0] == "signal_generated"
    assert captured[0][1]["symbol"] == "BTCUSDT"
