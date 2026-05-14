"""
Argus multilang HTTP service. One instance per language (LANGUAGE env).
GET /health, GET /ready, GET /metrics, GET /capabilities, POST /execute, POST /batch, POST /warm.
Implements all 15 task types using protocol_logic.
"""
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from protocol_logic import execute, get_profile, LANGUAGE_PROFILES

app = FastAPI(title="Argus Multilang", version="2.0")
LANGUAGE = os.environ.get("LANGUAGE", "rust").lower().strip()

# Observability: request count and latency (in-memory)
_request_count = 0
_total_latency_ms = 0.0
_last_latency_ms = 0.0
_error_count = 0
_start_time = time.perf_counter()

TASK_TYPES = [
    "cycle_plan", "order_book_processing", "risk_calculation", "volatility_estimate", "signal_score",
    "regime_estimate", "slippage_estimate", "position_sizing", "drawdown_check", "correlation_estimate",
    "liquidity_score", "market_impact", "signal_filter", "confidence_calibration", "heartbeat",
    "var_estimate", "skew_estimate", "order_book_imbalance_series", "execution_quality_score", "regime_duration",
]


class ExecuteBody(BaseModel):
    task_type: str
    data: Dict[str, Any] = {}
    timeout: float = 1.0
    correlation_id: Optional[str] = None


class BatchItem(BaseModel):
    task_type: str
    data: Dict[str, Any] = {}


class BatchBody(BaseModel):
    tasks: List[BatchItem] = []
    timeout: float = 2.0
    correlation_id: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "language": LANGUAGE}


@app.get("/ready")
def ready():
    return {"ready": True, "language": LANGUAGE}


@app.get("/metrics")
def metrics():
    global _request_count, _total_latency_ms, _error_count, _start_time
    uptime_s = time.perf_counter() - _start_time
    return {
        "language": LANGUAGE,
        "request_count": _request_count,
        "total_latency_ms": round(_total_latency_ms, 2),
        "last_latency_ms": round(_last_latency_ms, 4),
        "error_count": _error_count,
        "uptime_s": round(uptime_s, 2),
    }


@app.get("/capabilities")
def capabilities():
    profile = get_profile(LANGUAGE)
    return {
        "task_types": TASK_TYPES,
        "language": LANGUAGE,
        "profile": profile,
    }


@app.post("/execute")
def run_execute(body: ExecuteBody):
    global _request_count, _total_latency_ms, _last_latency_ms, _error_count
    data = dict(body.data or {})
    if body.correlation_id:
        data["correlation_id"] = body.correlation_id
    t0 = time.perf_counter()
    try:
        result = execute(body.task_type, data, LANGUAGE)
        took_ms = (time.perf_counter() - t0) * 1000.0
        _request_count += 1
        _total_latency_ms += took_ms
        _last_latency_ms = took_ms
        out = {"ok": True, "result": result, "took_ms": round(took_ms, 4)}
        if body.correlation_id:
            out["correlation_id"] = body.correlation_id
        return out
    except Exception as e:
        took_ms = (time.perf_counter() - t0) * 1000.0
        _request_count += 1
        _error_count += 1
        _last_latency_ms = took_ms
        return {"ok": False, "error": str(e), "result": {}, "took_ms": round(took_ms, 4)}


@app.post("/batch")
def run_batch(body: BatchBody):
    global _request_count, _total_latency_ms, _last_latency_ms, _error_count
    results = []
    t0 = time.perf_counter()
    for item in body.tasks[:50]:  # cap at 50
        data = dict(item.data or {})
        if body.correlation_id:
            data["correlation_id"] = body.correlation_id
        try:
            result = execute(item.task_type, data, LANGUAGE)
            results.append({"ok": True, "result": result})
        except Exception as e:
            results.append({"ok": False, "error": str(e), "result": {}})
            _error_count += 1
    took_ms = (time.perf_counter() - t0) * 1000.0
    _request_count += 1
    _total_latency_ms += took_ms
    _last_latency_ms = took_ms
    out = {"ok": True, "results": results, "took_ms": round(took_ms, 4)}
    if body.correlation_id:
        out["correlation_id"] = body.correlation_id
    return out


@app.post("/warm")
def warm():
    return {"warmed": True, "language": LANGUAGE}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8011"))
    uvicorn.run(app, host="0.0.0.0", port=port)
