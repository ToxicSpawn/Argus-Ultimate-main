from __future__ import annotations

from datetime import date

import pytest

from intelligence import (
    BinanceVisionClient,
    EvidentlyDriftMonitor,
    ForecastAdvisor,
    HistoricalDataRequest,
    LocalVectorMemory,
    QuantStatsReporter,
    SmartMoneyInputs,
    SmartMoneyScorer,
    WalkForwardValidator,
    population_stability_index,
)


def test_quantstats_reporter_computes_fallback_summary(tmp_path):
    reporter = QuantStatsReporter()
    summary = reporter.summarize(
        [100.0, 105.0, 103.0, 110.0],
        trades=[{"pnl": 5.0}, {"pnl": -2.0}, {"pnl": 3.0}],
        output_html=tmp_path / "report.html",
    )
    assert summary.total_return == pytest.approx(0.10)
    assert summary.max_drawdown < 0
    assert summary.win_rate == pytest.approx(2 / 3)
    assert (tmp_path / "report.html").exists()


def test_binance_vision_urls_are_deterministic():
    client = BinanceVisionClient()
    url = client.monthly_url(HistoricalDataRequest(symbol="BTC/USDT", interval="1m", year=2024, month=2))
    assert url.endswith("/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-02.zip")
    daily = client.daily_url("ETH-USDT", date(2024, 1, 3), interval="5m")
    assert daily.endswith("/spot/daily/klines/ETHUSDT/5m/ETHUSDT-5m-2024-01-03.zip")


def test_walk_forward_validator_splits_and_summarizes():
    validator = WalkForwardValidator(train_size=4, test_size=2)
    data = list(range(10))

    def evaluator(train, test):
        return {"edge": sum(test) - sum(train) / len(train)}

    results = validator.evaluate(data, evaluator)
    assert len(results) == 3
    assert results[0].train_start == 0
    assert results[0].test_start == 4
    assert "avg_edge" in validator.summarize(results)


def test_population_stability_index_detects_shift():
    stable = population_stability_index([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    shifted = population_stability_index([1, 2, 3, 4, 5], [10, 11, 12, 13, 14])
    assert stable <= shifted
    monitor = EvidentlyDriftMonitor(threshold=0.1)
    result = monitor.compare({"returns": [1, 2, 3, 4, 5]}, {"returns": [10, 11, 12, 13, 14]})
    assert result.max_score >= 0
    assert result.backend in {"evidently", "psi_fallback"}


def test_forecast_advisor_outputs_advisory_signal():
    advisor = ForecastAdvisor()
    signal = advisor.advise("BTC/USD", [100, 101, 103, 106], horizon=2)
    assert signal.symbol == "BTC/USD"
    assert signal.forecast_price > 0
    assert 0 <= signal.confidence <= 1
    assert signal.model == "naive_momentum_volatility"


def test_smart_money_scorer_returns_bias_and_contributions():
    score = SmartMoneyScorer().score(
        SmartMoneyInputs(
            funding_rate=-0.0005,
            open_interest_change=0.2,
            whale_netflow=0.7,
            sentiment_score=0.4,
            macro_risk=0.1,
        )
    )
    assert -1 <= score.score <= 1
    assert 0 <= score.confidence <= 1
    assert score.bias in {"bullish", "bearish", "neutral"}
    assert "whale_netflow" in score.contributions


def test_local_vector_memory_adds_and_queries(tmp_path):
    memory = LocalVectorMemory(tmp_path / "memory.jsonl")
    first = memory.add("BTC funding turned negative while whales accumulated", {"symbol": "BTC"})
    memory.add("ETH volatility expanded after CPI release", {"symbol": "ETH"})
    results = memory.query("BTC whales funding", limit=1)
    assert results[0][0].id == first.id
    assert results[0][1] > 0
