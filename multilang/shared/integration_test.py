"""
Integration test suite for all 23 Argus language microservices.
Run: cd multilang/shared && pytest integration_test.py -v
Requires: all 23 containers running via docker compose.
"""
import json
import math
import os
import pathlib
import time

import pytest
import requests

BASE = os.getenv("ARGUS_MULTILANG_HOST", "http://127.0.0.1")
FIXTURES = pathlib.Path(__file__).parent / "test-fixtures"

SERVICES = {
    "rust": 8011, "cpp": 8012, "cuda": 8013, "go": 8014,
    "java": 8015, "scala": 8016, "kotlin": 8017, "swift": 8018,
    "csharp": 8019, "fsharp": 8020, "javascript": 8021, "typescript": 8022,
    "elixir": 8023, "erlang": 8024, "clojure": 8025, "haskell": 8026,
    "ruby": 8027, "r": 8028, "julia": 8029, "matlab": 8030,
    "crystal": 8031, "webassembly": 8032, "mojo": 8033,
}

ALL_TASK_TYPES = [
    "cycle_plan", "order_book_processing", "risk_calculation",
    "signal_score", "volatility_estimate", "regime_estimate",
    "slippage_estimate", "position_sizing", "drawdown_check",
    "correlation_estimate", "liquidity_score", "market_impact",
    "signal_filter", "confidence_calibration", "heartbeat",
    "var_estimate", "skew_estimate", "order_book_imbalance_series",
    "execution_quality_score", "regime_duration",
]

# Required output keys per task type
REQUIRED_KEYS = {
    "cycle_plan": ["cycle_boost", "language", "ok"],
    "order_book_processing": ["spread_bps", "imbalance", "mid", "language"],
    "risk_calculation": ["passed", "exposure_ratio", "max_ratio", "language"],
    "signal_score": ["score_delta", "language", "ok"],
    "volatility_estimate": ["volatility_annual_bps", "language", "ok"],
    "regime_estimate": ["regime", "confidence", "language", "ok"],
    "slippage_estimate": ["slippage_bps", "language", "ok"],
    "position_sizing": ["size_pct", "language", "ok"],
    "drawdown_check": ["passed", "language", "ok"],
    "correlation_estimate": ["correlation", "language", "ok"],
    "liquidity_score": ["liquidity_score", "language", "ok"],
    "market_impact": ["impact_bps", "language", "ok"],
    "signal_filter": ["accept", "language", "ok"],
    "confidence_calibration": ["calibrated_confidence", "language", "ok"],
    "heartbeat": ["ok", "language"],
    "var_estimate": ["var_pct", "cvar_pct", "language", "ok"],
    "skew_estimate": ["skew", "language", "ok"],
    "order_book_imbalance_series": ["imbalance_series", "language", "ok"],
    "execution_quality_score": ["score_0_1", "language", "ok"],
    "regime_duration": ["bars_in_regime", "regime_stable", "language", "ok"],
}

# Test data for each task type
def _load(name):
    with open(FIXTURES / name) as f:
        return json.load(f)


def _task_data(task_type):
    ob = _load("order_book_1.json")
    ret = _load("returns_100.json")
    ctx = _load("cycle_context.json")
    cal = _load("calibration_data.json")
    cor = _load("correlation_pair.json")
    fills = _load("fills_and_decisions.json")

    mapping = {
        "cycle_plan": ctx,
        "order_book_processing": ob,
        "risk_calculation": {"position_value": 300.0, "capital": 1000.0},
        "signal_score": {"confidence": 0.75, "score": 0.75},
        "volatility_estimate": {"returns": ret["returns"], "prices": ret["prices"]},
        "regime_estimate": {"prices": ret["prices"]},
        "slippage_estimate": {
            "order_book": ob, "side": "buy",
            "quantity": 0.5, "participation_rate": 0.01,
        },
        "position_sizing": {
            "capital": 1000.0, "volatility_bps": 15.0,
            "confidence": 0.7, "max_risk_pct": 0.02,
        },
        "drawdown_check": {
            "current_equity": 950.0, "peak_equity": 1000.0,
            "max_drawdown_pct": 0.12,
        },
        "correlation_estimate": cor,
        "liquidity_score": {"bids": ob["bids"], "asks": ob["asks"], "depth_levels": 5},
        "market_impact": {
            "side": "buy", "quantity": 10.0,
            "adv": 1000.0, "volatility": 0.02,
        },
        "signal_filter": {
            "confidence": 0.75, "regime": "trend", "volatility": 0.015,
        },
        "confidence_calibration": cal,
        "heartbeat": {"cycle_id": 42, "timestamp": time.time()},
        "var_estimate": {"returns": ret["returns"], "confidence_level": 0.95},
        "skew_estimate": {"returns": ret["returns"]},
        "order_book_imbalance_series": ob,
        "execution_quality_score": fills,
        "regime_duration": {"prices": ret["prices"], "regime_history": ["trend"] * 8},
    }
    return mapping[task_type]


def _url(lang, path=""):
    return f"{BASE}:{SERVICES[lang]}{path}"


def _available_services():
    """Return list of services that are actually running."""
    available = []
    for lang, port in SERVICES.items():
        try:
            r = requests.get(f"{BASE}:{port}/health", timeout=2)
            if r.status_code == 200:
                available.append(lang)
        except Exception:
            pass
    return available


# Discover which services are up (allows partial testing)
@pytest.fixture(scope="session")
def live_services():
    services = _available_services()
    if not services:
        pytest.skip("No multilang services are running")
    return services


# ---------------------------------------------------------------------------
# Protocol compliance tests
# ---------------------------------------------------------------------------

class TestProtocolCompliance:
    """Every running service must respond correctly on all endpoints."""

    def test_health(self, live_services):
        for lang in live_services:
            r = requests.get(_url(lang, "/health"), timeout=5)
            assert r.status_code == 200, f"{lang}: /health returned {r.status_code}"
            body = r.json()
            assert body.get("ok") or body.get("status") == "ok", f"{lang}: /health body: {body}"

    def test_ready(self, live_services):
        for lang in live_services:
            r = requests.get(_url(lang, "/ready"), timeout=5)
            assert r.status_code == 200, f"{lang}: /ready returned {r.status_code}"

    def test_metrics(self, live_services):
        for lang in live_services:
            r = requests.get(_url(lang, "/metrics"), timeout=5)
            assert r.status_code == 200, f"{lang}: /metrics returned {r.status_code}"
            body = r.json()
            assert "request_count" in body or "requests" in body, f"{lang}: /metrics missing counters"

    def test_capabilities(self, live_services):
        for lang in live_services:
            r = requests.get(_url(lang, "/capabilities"), timeout=5)
            assert r.status_code == 200, f"{lang}: /capabilities returned {r.status_code}"
            body = r.json()
            assert "task_types" in body, f"{lang}: /capabilities missing task_types"
            assert "language" in body, f"{lang}: /capabilities missing language"


# ---------------------------------------------------------------------------
# Task execution tests (per task type, across all services)
# ---------------------------------------------------------------------------

class TestTaskExecution:
    """Each service must handle all 20 task types with correct response shape."""

    @pytest.mark.parametrize("task_type", ALL_TASK_TYPES)
    def test_execute_returns_required_keys(self, live_services, task_type):
        data = _task_data(task_type)
        required = REQUIRED_KEYS[task_type]
        for lang in live_services:
            payload = {"task_type": task_type, "data": data, "timeout": 5.0}
            r = requests.post(_url(lang, "/execute"), json=payload, timeout=10)
            assert r.status_code == 200, f"{lang}/{task_type}: HTTP {r.status_code}"
            body = r.json()
            assert body.get("ok") is True, f"{lang}/{task_type}: not ok: {body}"
            result = body.get("result", body)
            for key in required:
                assert key in result, f"{lang}/{task_type}: missing key '{key}' in {list(result.keys())}"

    @pytest.mark.parametrize("task_type", ALL_TASK_TYPES)
    def test_execute_has_took_ms(self, live_services, task_type):
        data = _task_data(task_type)
        for lang in live_services:
            payload = {"task_type": task_type, "data": data, "timeout": 5.0}
            r = requests.post(_url(lang, "/execute"), json=payload, timeout=10)
            body = r.json()
            assert "took_ms" in body, f"{lang}/{task_type}: missing took_ms"
            assert isinstance(body["took_ms"], (int, float)), f"{lang}/{task_type}: took_ms not numeric"

    def test_language_field_matches_service(self, live_services):
        """The 'language' field in results must match the service identity."""
        data = _task_data("heartbeat")
        for lang in live_services:
            payload = {"task_type": "heartbeat", "data": data, "timeout": 5.0}
            r = requests.post(_url(lang, "/execute"), json=payload, timeout=10)
            body = r.json()
            result = body.get("result", body)
            assert result.get("language") == lang, \
                f"{lang}: language field is '{result.get('language')}' not '{lang}'"


# ---------------------------------------------------------------------------
# Numerical validation tests
# ---------------------------------------------------------------------------

class TestNumericalResults:
    """Validate that numeric outputs are within reasonable ranges."""

    def test_volatility_positive(self, live_services):
        data = _task_data("volatility_estimate")
        for lang in live_services:
            payload = {"task_type": "volatility_estimate", "data": data, "timeout": 5.0}
            r = requests.post(_url(lang, "/execute"), json=payload, timeout=10)
            result = r.json().get("result", r.json())
            vol = result.get("volatility_annual_bps", 0)
            assert vol > 0, f"{lang}: volatility should be positive, got {vol}"
            assert vol < 100000, f"{lang}: volatility unreasonably high: {vol}"

    def test_correlation_bounded(self, live_services):
        data = _task_data("correlation_estimate")
        for lang in live_services:
            payload = {"task_type": "correlation_estimate", "data": data, "timeout": 5.0}
            r = requests.post(_url(lang, "/execute"), json=payload, timeout=10)
            result = r.json().get("result", r.json())
            corr = result.get("correlation", 0)
            assert -1.0 <= corr <= 1.0, f"{lang}: correlation out of range: {corr}"

    def test_risk_pass_for_low_exposure(self, live_services):
        """30% exposure should pass for all languages (all have risk_max >= 0.40)."""
        data = {"position_value": 300.0, "capital": 1000.0}
        for lang in live_services:
            payload = {"task_type": "risk_calculation", "data": data, "timeout": 5.0}
            r = requests.post(_url(lang, "/execute"), json=payload, timeout=10)
            result = r.json().get("result", r.json())
            assert result.get("passed") is True, f"{lang}: 30% exposure should pass"

    def test_risk_fail_for_high_exposure(self, live_services):
        """90% exposure should fail for all languages."""
        data = {"position_value": 900.0, "capital": 1000.0}
        for lang in live_services:
            payload = {"task_type": "risk_calculation", "data": data, "timeout": 5.0}
            r = requests.post(_url(lang, "/execute"), json=payload, timeout=10)
            result = r.json().get("result", r.json())
            assert result.get("passed") is False, f"{lang}: 90% exposure should fail"

    def test_drawdown_pass_for_low_drawdown(self, live_services):
        """5% drawdown with 12% limit should pass."""
        data = {"current_equity": 950.0, "peak_equity": 1000.0, "max_drawdown_pct": 0.12}
        for lang in live_services:
            payload = {"task_type": "drawdown_check", "data": data, "timeout": 5.0}
            r = requests.post(_url(lang, "/execute"), json=payload, timeout=10)
            result = r.json().get("result", r.json())
            assert result.get("passed") is True, f"{lang}: 5% dd with 12% limit should pass"

    def test_var_positive(self, live_services):
        data = _task_data("var_estimate")
        for lang in live_services:
            payload = {"task_type": "var_estimate", "data": data, "timeout": 5.0}
            r = requests.post(_url(lang, "/execute"), json=payload, timeout=10)
            result = r.json().get("result", r.json())
            var_pct = result.get("var_pct", 0)
            assert var_pct >= 0, f"{lang}: VaR should be non-negative, got {var_pct}"

    def test_liquidity_score_bounded(self, live_services):
        data = _task_data("liquidity_score")
        for lang in live_services:
            payload = {"task_type": "liquidity_score", "data": data, "timeout": 5.0}
            r = requests.post(_url(lang, "/execute"), json=payload, timeout=10)
            result = r.json().get("result", r.json())
            score = result.get("liquidity_score", -1)
            assert 0.0 <= score <= 1.0, f"{lang}: liquidity score out of range: {score}"

    def test_execution_quality_bounded(self, live_services):
        data = _task_data("execution_quality_score")
        for lang in live_services:
            payload = {"task_type": "execution_quality_score", "data": data, "timeout": 5.0}
            r = requests.post(_url(lang, "/execute"), json=payload, timeout=10)
            result = r.json().get("result", r.json())
            score = result.get("score_0_1", -1)
            assert 0.0 <= score <= 1.0, f"{lang}: execution quality out of range: {score}"

    def test_cycle_boost_bounded(self, live_services):
        data = _task_data("cycle_plan")
        for lang in live_services:
            payload = {"task_type": "cycle_plan", "data": data, "timeout": 5.0}
            r = requests.post(_url(lang, "/execute"), json=payload, timeout=10)
            result = r.json().get("result", r.json())
            boost = result.get("cycle_boost", 999)
            assert -0.02 <= boost <= 0.02, f"{lang}: cycle boost out of range: {boost}"

    def test_regime_valid_value(self, live_services):
        data = _task_data("regime_estimate")
        for lang in live_services:
            payload = {"task_type": "regime_estimate", "data": data, "timeout": 5.0}
            r = requests.post(_url(lang, "/execute"), json=payload, timeout=10)
            result = r.json().get("result", r.json())
            regime = result.get("regime", "")
            assert regime in ("trend", "mean_revert", "high_vol"), \
                f"{lang}: invalid regime: {regime}"

    def test_confidence_calibration_bounded(self, live_services):
        data = _task_data("confidence_calibration")
        for lang in live_services:
            payload = {"task_type": "confidence_calibration", "data": data, "timeout": 5.0}
            r = requests.post(_url(lang, "/execute"), json=payload, timeout=10)
            result = r.json().get("result", r.json())
            cal = result.get("calibrated_confidence", -1)
            assert 0.0 <= cal <= 1.0, f"{lang}: calibrated confidence out of range: {cal}"


# ---------------------------------------------------------------------------
# Batch endpoint test
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch_returns_multiple_results(self, live_services):
        tasks = [
            {"task_type": "heartbeat", "data": {"cycle_id": 1}},
            {"task_type": "heartbeat", "data": {"cycle_id": 2}},
        ]
        for lang in live_services:
            r = requests.post(
                _url(lang, "/batch"),
                json={"tasks": tasks, "timeout": 5.0},
                timeout=10,
            )
            assert r.status_code == 200, f"{lang}: /batch returned {r.status_code}"
            body = r.json()
            results = body.get("results", body)
            assert isinstance(results, list), f"{lang}: /batch should return list"
            assert len(results) >= 2, f"{lang}: /batch returned {len(results)} results"


# ---------------------------------------------------------------------------
# Warm endpoint test
# ---------------------------------------------------------------------------

class TestWarm:
    def test_warm_returns_ok(self, live_services):
        for lang in live_services:
            r = requests.post(
                _url(lang, "/warm"),
                json={"portfolio_value_aud": 1000.0},
                timeout=10,
            )
            assert r.status_code == 200, f"{lang}: /warm returned {r.status_code}"
