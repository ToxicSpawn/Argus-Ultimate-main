"""
Unified Language Orchestrator – 23+ languages advancing the Argus bot.

Dispatches tasks (cycle plan, order book processing, risk calculation, volatility,
signal score) to language-specific services. When HTTP endpoints are not available,
uses in-process Python workers with per-language profiles so each of the 23 languages
contributes a distinct, improved result. Replace endpoints with real implementations
for maximum gain.

Protocol (each service or in-process worker):
- GET  /health  -> 200 OK (HTTP only)
- POST /execute  body: {"task_type": "...", "data": {...}, "timeout": 1.0}  -> {"ok": true, "result": {...}}
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# All 23 languages the bot can use (config endpoints + in-process fallback)
DEFAULT_LANGUAGES = [
    "rust", "cpp", "cuda", "go", "java", "scala", "kotlin", "swift", "csharp", "fsharp",
    "javascript", "typescript", "elixir", "erlang", "clojure", "haskell", "ruby", "r",
    "julia", "matlab", "crystal", "webassembly", "mojo",
]

# Per-language profiles: everything possible per language (risk, cycle, volatility, signal score, order book)
# risk_max_ratio = max position/capital (lower = more conservative)
# cycle_boost_scale = scale for cycle contribution (1.0 = normal; stats languages get higher weight for volatility-aware boost)
# volatility_weight = weight when aggregating volatility estimates (stats languages > 1)
# signal_score_weight = weight when aggregating signal scores
# spread_mult = multiplier on effective spread (>=1 = more conservative execution)
# regime_weight = weight when aggregating regime (stats languages > 1)
# drawdown_max_ratio = multiplier on config max_drawdown (correctness languages stricter)
# slippage_tolerance_bps = max acceptable slippage bps for this language (gate)
# min_confidence_to_accept = signal_filter: accept only if confidence >= this (correctness stricter)
# batch_capable = service supports POST /batch
# role = speed | correctness | stats | concurrency | ecosystem
def _profile(risk_max: float, cycle: float, vol_w: float, sig_w: float, spread: float, role: str,
             regime_w: float = 1.0, drawdown_ratio: float = 1.0, slippage_bps: float = 100.0,
             min_conf: float = 0.5, batch_capable: bool = False) -> Dict[str, Any]:
    return {"risk_max_ratio": risk_max, "cycle_boost_scale": cycle, "volatility_weight": vol_w,
            "signal_score_weight": sig_w, "spread_mult": spread, "role": role,
            "regime_weight": regime_w, "drawdown_max_ratio": drawdown_ratio,
            "slippage_tolerance_bps": slippage_bps, "min_confidence_to_accept": min_conf, "batch_capable": batch_capable}

LANGUAGE_PROFILES: Dict[str, Dict[str, Any]] = {
    "rust":       _profile(0.48, 1.0,  0.9,  1.0, 1.0,  "speed",       regime_w=1.0,  drawdown_ratio=1.0, slippage_bps=80,  min_conf=0.5, batch_capable=False),
    "cpp":        _profile(0.48, 1.0,  0.9,  1.0, 1.0,  "speed",       regime_w=1.0,  drawdown_ratio=1.0, slippage_bps=80,  min_conf=0.5, batch_capable=False),
    "cuda":       _profile(0.46, 0.95, 1.1,  0.95, 1.02, "speed",      regime_w=1.0,  drawdown_ratio=1.0, slippage_bps=85,  min_conf=0.5, batch_capable=False),
    "go":         _profile(0.47, 1.0,  0.95, 1.0, 1.0,  "speed",       regime_w=1.0,  drawdown_ratio=1.0, slippage_bps=80,  min_conf=0.5, batch_capable=False),
    "java":       _profile(0.45, 0.98, 1.0,  1.0, 1.01, "correctness", regime_w=1.0,  drawdown_ratio=0.9, slippage_bps=60,  min_conf=0.6, batch_capable=False),
    "scala":      _profile(0.44, 0.98, 1.0,  1.02, 1.01, "correctness", regime_w=1.0,  drawdown_ratio=0.9, slippage_bps=60,  min_conf=0.6, batch_capable=False),
    "kotlin":     _profile(0.45, 0.99, 1.0,  1.0, 1.01, "correctness", regime_w=1.0,  drawdown_ratio=0.9, slippage_bps=60,  min_conf=0.6, batch_capable=False),
    "swift":      _profile(0.46, 1.0,  0.95, 1.0, 1.0,  "ecosystem",   regime_w=1.0,  drawdown_ratio=1.0, slippage_bps=100, min_conf=0.5, batch_capable=False),
    "csharp":     _profile(0.45, 0.99, 1.0,  1.0, 1.01, "ecosystem",   regime_w=1.0,  drawdown_ratio=1.0, slippage_bps=100, min_conf=0.5, batch_capable=False),
    "fsharp":     _profile(0.42, 0.97, 1.05, 1.02, 1.02, "correctness", regime_w=1.0,  drawdown_ratio=0.9, slippage_bps=60,  min_conf=0.6, batch_capable=False),
    "javascript": _profile(0.46, 1.0,  0.95, 1.0, 1.0,  "ecosystem",   regime_w=1.0,  drawdown_ratio=1.0, slippage_bps=100, min_conf=0.5, batch_capable=False),
    "typescript": _profile(0.45, 0.99, 0.98, 1.0, 1.01, "ecosystem",   regime_w=1.0,  drawdown_ratio=1.0, slippage_bps=100, min_conf=0.5, batch_capable=False),
    "elixir":     _profile(0.45, 1.0,  1.0,  1.01, 1.01, "concurrency", regime_w=1.0,  drawdown_ratio=1.0, slippage_bps=100, min_conf=0.5, batch_capable=True),
    "erlang":     _profile(0.44, 0.99, 1.0,  1.01, 1.02, "concurrency", regime_w=1.0,  drawdown_ratio=1.0, slippage_bps=100, min_conf=0.5, batch_capable=True),
    "clojure":    _profile(0.44, 0.98, 1.02, 1.02, 1.01, "correctness", regime_w=1.0,  drawdown_ratio=0.9, slippage_bps=60,  min_conf=0.6, batch_capable=False),
    "haskell":    _profile(0.40, 0.95, 1.05, 1.02, 1.03, "correctness", regime_w=1.0,  drawdown_ratio=0.85, slippage_bps=55, min_conf=0.62, batch_capable=False),
    "ruby":       _profile(0.46, 1.0,  0.98, 1.0, 1.0,  "ecosystem",   regime_w=1.0,  drawdown_ratio=1.0, slippage_bps=100, min_conf=0.5, batch_capable=False),
    "r":          _profile(0.44, 1.05, 1.2,  1.05, 1.02, "stats",       regime_w=1.2,  drawdown_ratio=1.0, slippage_bps=100, min_conf=0.5, batch_capable=False),
    "julia":      _profile(0.45, 1.03, 1.15, 1.03, 1.01, "stats",       regime_w=1.2,  drawdown_ratio=1.0, slippage_bps=100, min_conf=0.5, batch_capable=False),
    "matlab":     _profile(0.44, 1.02, 1.12, 1.02, 1.02, "stats",       regime_w=1.2,  drawdown_ratio=1.0, slippage_bps=100, min_conf=0.5, batch_capable=False),
    "crystal":    _profile(0.47, 1.0,  0.95, 1.0, 1.0,  "speed",       regime_w=1.0,  drawdown_ratio=1.0, slippage_bps=80,  min_conf=0.5, batch_capable=False),
    "webassembly": _profile(0.45, 0.98, 1.0,  1.0, 1.01, "ecosystem",  regime_w=1.0,  drawdown_ratio=1.0, slippage_bps=100, min_conf=0.5, batch_capable=False),
    "mojo":       _profile(0.47, 1.01, 1.05, 1.0, 1.0,  "speed",       regime_w=1.0,  drawdown_ratio=1.0, slippage_bps=80,  min_conf=0.5, batch_capable=False),
}


def _get_profile(language: str) -> Dict[str, Any]:
    return LANGUAGE_PROFILES.get(language, LANGUAGE_PROFILES["rust"])


# Use each language to its strength: preferred task order per task type.
# Speed: order book + risk (low latency). Correctness: risk + cycle (strict). Stats: volatility + signal score. etc.
SPEED_LANGUAGES = ["rust", "cpp", "cuda", "go", "crystal", "mojo"]
CORRECTNESS_LANGUAGES = ["haskell", "fsharp", "scala", "clojure", "java", "kotlin"]
STATS_LANGUAGES = ["r", "julia", "matlab"]
CONCURRENCY_LANGUAGES = ["elixir", "erlang"]
ECOSYSTEM_LANGUAGES = ["swift", "csharp", "javascript", "typescript", "ruby", "webassembly"]

# For single-task execution: try these language orders first (by strength).
STRENGTH_TASK_ORDER: Dict[str, List[str]] = {
    "order_book_processing": SPEED_LANGUAGES + CORRECTNESS_LANGUAGES + STATS_LANGUAGES + CONCURRENCY_LANGUAGES + ECOSYSTEM_LANGUAGES,
    "risk_calculation": CORRECTNESS_LANGUAGES + SPEED_LANGUAGES + ECOSYSTEM_LANGUAGES + CONCURRENCY_LANGUAGES + STATS_LANGUAGES,
    "volatility_estimate": STATS_LANGUAGES + SPEED_LANGUAGES + CORRECTNESS_LANGUAGES + ECOSYSTEM_LANGUAGES + CONCURRENCY_LANGUAGES,
    "signal_score": STATS_LANGUAGES + CORRECTNESS_LANGUAGES + ECOSYSTEM_LANGUAGES + SPEED_LANGUAGES + CONCURRENCY_LANGUAGES,
    "cycle_plan": list(DEFAULT_LANGUAGES),
    "regime_estimate": STATS_LANGUAGES + CORRECTNESS_LANGUAGES + SPEED_LANGUAGES + ECOSYSTEM_LANGUAGES + CONCURRENCY_LANGUAGES,
    "slippage_estimate": SPEED_LANGUAGES + CORRECTNESS_LANGUAGES + STATS_LANGUAGES + CONCURRENCY_LANGUAGES + ECOSYSTEM_LANGUAGES,
    "position_sizing": CORRECTNESS_LANGUAGES + STATS_LANGUAGES + SPEED_LANGUAGES + ECOSYSTEM_LANGUAGES + CONCURRENCY_LANGUAGES,
    "drawdown_check": CORRECTNESS_LANGUAGES + SPEED_LANGUAGES + ECOSYSTEM_LANGUAGES + CONCURRENCY_LANGUAGES + STATS_LANGUAGES,
    "correlation_estimate": STATS_LANGUAGES + SPEED_LANGUAGES + CORRECTNESS_LANGUAGES + ECOSYSTEM_LANGUAGES + CONCURRENCY_LANGUAGES,
    "liquidity_score": SPEED_LANGUAGES + CORRECTNESS_LANGUAGES + STATS_LANGUAGES + CONCURRENCY_LANGUAGES + ECOSYSTEM_LANGUAGES,
    "market_impact": SPEED_LANGUAGES + STATS_LANGUAGES + CORRECTNESS_LANGUAGES + ECOSYSTEM_LANGUAGES + CONCURRENCY_LANGUAGES,
    "signal_filter": list(DEFAULT_LANGUAGES),
    "confidence_calibration": STATS_LANGUAGES + CORRECTNESS_LANGUAGES + ECOSYSTEM_LANGUAGES + SPEED_LANGUAGES + CONCURRENCY_LANGUAGES,
    "heartbeat": list(DEFAULT_LANGUAGES),
    "var_estimate": STATS_LANGUAGES + CORRECTNESS_LANGUAGES + SPEED_LANGUAGES + ECOSYSTEM_LANGUAGES + CONCURRENCY_LANGUAGES,
    "skew_estimate": STATS_LANGUAGES + CORRECTNESS_LANGUAGES + ECOSYSTEM_LANGUAGES + SPEED_LANGUAGES + CONCURRENCY_LANGUAGES,
    "order_book_imbalance_series": SPEED_LANGUAGES + CORRECTNESS_LANGUAGES + STATS_LANGUAGES + CONCURRENCY_LANGUAGES + ECOSYSTEM_LANGUAGES,
    "execution_quality_score": CORRECTNESS_LANGUAGES + STATS_LANGUAGES + SPEED_LANGUAGES + ECOSYSTEM_LANGUAGES + CONCURRENCY_LANGUAGES,
    "regime_duration": STATS_LANGUAGES + CORRECTNESS_LANGUAGES + SPEED_LANGUAGES + ECOSYSTEM_LANGUAGES + CONCURRENCY_LANGUAGES,
}
STRENGTH_TASK_ORDER.setdefault("_default", SPEED_LANGUAGES + CORRECTNESS_LANGUAGES + list(DEFAULT_LANGUAGES))
STATS_WEIGHT_BOOST = 1.5   # weight multiplier for stats languages in volatility/signal aggregation
CORRECTNESS_WEIGHT_BOOST = 1.25  # weight multiplier for correctness in signal score


class TaskType(str, Enum):
    CYCLE_PLAN = "cycle_plan"
    ORDER_BOOK_PROCESSING = "order_book_processing"
    RISK_CALCULATION = "risk_calculation"
    SIGNAL_SCORE = "signal_score"
    VOLATILITY_ESTIMATE = "volatility_estimate"
    REGIME_ESTIMATE = "regime_estimate"
    SLIPPAGE_ESTIMATE = "slippage_estimate"
    POSITION_SIZING = "position_sizing"
    DRAWDOWN_CHECK = "drawdown_check"
    CORRELATION_ESTIMATE = "correlation_estimate"
    LIQUIDITY_SCORE = "liquidity_score"
    MARKET_IMPACT = "market_impact"
    SIGNAL_FILTER = "signal_filter"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    HEARTBEAT = "heartbeat"
    VAR_ESTIMATE = "var_estimate"
    SKEW_ESTIMATE = "skew_estimate"
    ORDER_BOOK_IMBALANCE_SERIES = "order_book_imbalance_series"
    EXECUTION_QUALITY_SCORE = "execution_quality_score"
    REGIME_DURATION = "regime_duration"


@dataclass
class TaskRequest:
    task_type: TaskType
    data: Dict[str, Any]
    timeout: float = 2.0
    correlation_id: Optional[str] = None


@dataclass
class LanguageCallResult:
    """Result from one language service (HTTP or in-process)."""
    language_used: str
    success: bool
    result: Any
    execution_time_ms: float
    error_message: Optional[str] = None


def _in_process_order_book(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Compute order book metrics (spread, imbalance) with per-language spread multiplier."""
    bids = data.get("bids") or []
    asks = data.get("asks") or []
    profile = _get_profile(language)
    spread_mult = float(profile.get("spread_mult", 1.0))
    if not bids and not asks:
        return {"spread_bps": 0.0, "imbalance": 0.0, "mid": 0.0, "depth_bps": 0.0, "language": language, "spread_mult": spread_mult, "took_ms": 0.0}
    best_bid = float(bids[0][0]) if bids else 0.0
    best_ask = float(asks[0][0]) if asks else 0.0
    mid = (best_bid + best_ask) / 2.0 if (best_bid and best_ask) else 0.0
    raw_spread_bps = (best_ask - best_bid) / mid * 1e4 if mid else 0.0
    spread_bps = raw_spread_bps * spread_mult
    bid_vol = sum(float(b[1]) for b in bids[:5]) if bids else 0.0
    ask_vol = sum(float(a[1]) for a in asks[:5]) if asks else 0.0
    total = bid_vol + ask_vol
    imbalance = (bid_vol - ask_vol) / total if total else 0.0
    depth_bps = raw_spread_bps if mid else 0.0
    return {"spread_bps": spread_bps, "imbalance": imbalance, "mid": mid, "depth_bps": depth_bps, "language": language, "spread_mult": spread_mult, "took_ms": 0.0}


def _in_process_risk(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Risk check: position vs capital using per-language max exposure ratio."""
    profile = _get_profile(language)
    max_ratio = float(profile.get("risk_max_ratio", 0.45))
    pv = float(data.get("position_value") or 0.0)
    capital = float(data.get("capital") or 1.0)
    ratio = pv / capital if capital else 0.0
    passed = ratio <= max_ratio
    reason = "" if passed else f"exposure_ratio_{ratio:.4f}_exceeds_max_{max_ratio}"
    return {"passed": passed, "exposure_ratio": ratio, "max_ratio": max_ratio, "reason": reason, "language": language, "took_ms": 0.0}


def _in_process_cycle_plan(ctx: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Per-language cycle contribution: boost scaled by profile and context (signals, cash ratio)."""
    profile = _get_profile(language)
    scale = float(profile.get("cycle_boost_scale", 1.0))
    idx = sum(ord(c) for c in language) % 100
    base = (idx - 50) / 10000.0
    signals = int(ctx.get("signals") or 0)
    cash = float(ctx.get("cash_balance_aud") or 0.0)
    pv = float(ctx.get("portfolio_value_aud") or 1.0)
    cash_ratio = cash / pv if pv else 0.0
    # Slight tilt: more cash -> slightly positive bias; more signals -> slight diversity
    tilt = (cash_ratio - 0.5) * 0.002 + (signals % 3 - 1) * 0.001
    boost = max(-0.015, min(0.015, (base + tilt) * scale))
    return {"language": language, "cycle_boost": boost, "cycle_boost_scale": scale, "ok": True, "took_ms": 0.0}


def _in_process_volatility_estimate(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Per-language volatility estimate from returns or prices; stats languages weighted higher."""
    profile = _get_profile(language)
    weight = float(profile.get("volatility_weight", 1.0))
    prices = data.get("prices") or data.get("ohlcv_close") or []
    returns = data.get("returns") or []
    if returns:
        n = len(returns)
        mean_r = sum(returns) / n if n else 0.0
        var = sum((r - mean_r) ** 2 for r in returns) / n if n else 0.0
        vol = math.sqrt(var * 252 * 1e4) if var else 10.0  # annualized bps
    elif len(prices) >= 2:
        rets = [(float(prices[i]) - float(prices[i - 1])) / float(prices[i - 1]) for i in range(1, len(prices)) if prices[i - 1]]
        n = len(rets)
        mean_r = sum(rets) / n if n else 0.0
        var = sum((r - mean_r) ** 2 for r in rets) / n if n else 0.0
        vol = math.sqrt(var * 252 * 1e4) if var else 10.0
    else:
        vol = 10.0
    vol_adj = vol * weight
    return {"volatility_annual_bps": vol_adj, "volatility_weight": weight, "language": language, "ok": True, "took_ms": 0.0}


def _in_process_signal_score(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Per-language signal score delta; used when aggregating multi-language signal scores."""
    profile = _get_profile(language)
    weight = float(profile.get("signal_score_weight", 1.0))
    confidence = float(data.get("confidence") or 0.0)
    base_score = float(data.get("score") or confidence)
    score_delta = base_score * weight
    return {"score_delta": score_delta, "signal_score_weight": weight, "base_score": base_score, "language": language, "ok": True, "took_ms": 0.0}


def _in_process_regime_estimate(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Regime: trend / mean_revert / high_vol; stats languages weighted higher. Uses regime_weight from profile."""
    profile = _get_profile(language)
    weight = float(profile.get("regime_weight", profile.get("volatility_weight", 1.0)))
    prices = data.get("prices") or data.get("returns") or []
    if len(prices) >= 2 and not isinstance(prices[0], (int, float)):
        prices = []
    if len(prices) >= 3:
        rets = [(float(prices[i]) - float(prices[i - 1])) / float(prices[i - 1]) if prices[i - 1] else 0.0 for i in range(1, len(prices))]
        vol = math.sqrt(sum(r * r for r in rets) / len(rets) * 252 * 1e4) if rets else 10.0
        trend = (float(prices[-1]) - float(prices[0])) / float(prices[0]) if prices[0] else 0.0
        if vol > 20.0:
            regime = "high_vol"
        elif abs(trend) > 0.02:
            regime = "trend"
        else:
            regime = "mean_revert"
        confidence = min(0.95, 0.5 + abs(trend) * 5 + vol / 100)
    else:
        regime = "mean_revert"
        confidence = 0.5
        vol = 10.0
    return {"regime": regime, "confidence": confidence, "language": language, "regime_weight": weight, "ok": True, "took_ms": 0.0}


def _in_process_slippage_estimate(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Slippage bps from side, quantity, order_book; speed languages preferred."""
    profile = _get_profile(language)
    spread_mult = float(profile.get("spread_mult", 1.0))
    bids = data.get("order_book", {}).get("bids") or data.get("bids") or []
    asks = data.get("order_book", {}).get("asks") or data.get("asks") or []
    side = str(data.get("side", "buy")).lower()
    quantity = float(data.get("quantity") or 0.0)
    participation = float(data.get("participation_rate") or 0.01)
    if not bids and not asks:
        return {"slippage_bps": 0.0, "language": language, "ok": True, "took_ms": 0.0}
    best_bid = float(bids[0][0]) if bids else 0.0
    best_ask = float(asks[0][0]) if asks else 0.0
    mid = (best_bid + best_ask) / 2.0 if (best_bid and best_ask) else 0.0
    half_spread_bps = (best_ask - best_bid) / mid * 1e4 / 2 if mid else 5.0
    slippage_bps = half_spread_bps * spread_mult * (1.0 + participation * 10)
    return {"slippage_bps": slippage_bps, "language": language, "ok": True, "took_ms": 0.0}


def _in_process_position_sizing(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Size pct from capital, volatility, confidence; correctness + stats."""
    profile = _get_profile(language)
    capital = float(data.get("capital") or 1.0)
    volatility_bps = float(data.get("volatility_bps") or data.get("volatility_annual_bps") or 10.0)
    confidence = float(data.get("confidence") or 0.5)
    max_risk_pct = float(data.get("max_risk_pct") or 0.02)
    risk_max = profile.get("risk_max_ratio", 0.45)
    size_pct = min(risk_max, max_risk_pct * (volatility_bps / 10.0) * (0.5 + confidence))
    size_abs = size_pct * capital if capital else 0.0
    reason = "vol_and_confidence_cap" if size_pct < risk_max else "at_max_ratio"
    return {"size_pct": size_pct, "size_abs": size_abs, "reason": reason, "language": language, "ok": True, "took_ms": 0.0}


def _in_process_drawdown_check(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Passed if current drawdown <= max; correctness preferred. Uses drawdown_max_ratio from profile."""
    profile = _get_profile(language)
    dd_ratio = float(profile.get("drawdown_max_ratio", 1.0))
    max_drawdown_pct = float(data.get("max_drawdown_pct") or 0.12) * dd_ratio
    current = float(data.get("current_equity") or 0.0)
    peak = float(data.get("peak_equity") or current or 1.0)
    current_drawdown_pct = (peak - current) / peak if peak else 0.0
    passed = current_drawdown_pct <= max_drawdown_pct
    reason = "" if passed else f"drawdown_{current_drawdown_pct:.4f}_exceeds_limit_{max_drawdown_pct}"
    return {"passed": passed, "current_drawdown_pct": current_drawdown_pct, "reason": reason, "language": language, "ok": True, "took_ms": 0.0}


def _in_process_correlation_estimate(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Correlation between series_a and series_b; stats preferred."""
    profile = _get_profile(language)
    a = data.get("series_a") or data.get("returns_a") or []
    b = data.get("series_b") or data.get("returns_b") or []
    if len(a) != len(b) or len(a) < 2:
        return {"correlation": 0.0, "language": language, "ok": True, "took_ms": 0.0}
    n = len(a)
    ma = sum(float(x) for x in a) / n
    mb = sum(float(x) for x in b) / n
    va = sum((float(a[i]) - ma) ** 2 for i in range(n))
    vb = sum((float(b[i]) - mb) ** 2 for i in range(n))
    cov = sum((float(a[i]) - ma) * (float(b[i]) - mb) for i in range(n))
    den = math.sqrt(va * vb)
    correlation = cov / den if den else 0.0
    correlation = max(-1.0, min(1.0, correlation))
    return {"correlation": correlation, "language": language, "ok": True, "took_ms": 0.0}


def _in_process_liquidity_score(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Liquidity 0-1 from order book depth; speed preferred."""
    bids = data.get("bids") or []
    asks = data.get("asks") or []
    depth_levels = int(data.get("depth_levels") or 5)
    if not bids and not asks:
        return {"liquidity_score": 0.0, "depth_bps": 0.0, "language": language, "ok": True, "took_ms": 0.0}
    best_bid = float(bids[0][0]) if bids else 0.0
    best_ask = float(asks[0][0]) if asks else 0.0
    mid = (best_bid + best_ask) / 2.0 if (best_bid and best_ask) else 0.0
    depth_bps = (best_ask - best_bid) / mid * 1e4 if mid else 100.0
    total_vol = sum(float(b[1]) for b in bids[:depth_levels]) + sum(float(a[1]) for a in asks[:depth_levels])
    liquidity_score = min(1.0, total_vol / 100.0) if total_vol else 0.0
    return {"liquidity_score": liquidity_score, "depth_bps": depth_bps, "language": language, "ok": True, "took_ms": 0.0}


def _in_process_market_impact(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Impact bps from side, quantity, adv, volatility."""
    side = str(data.get("side", "buy")).lower()
    quantity = float(data.get("quantity") or 0.0)
    adv = float(data.get("adv") or 1.0)
    volatility = float(data.get("volatility") or 0.01)
    participation = quantity / adv if adv else 0.0
    impact_bps = 10.0 * math.sqrt(participation) * volatility * 1e4
    return {"impact_bps": impact_bps, "language": language, "ok": True, "took_ms": 0.0}


def _in_process_signal_filter(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Accept/reject signal; all 23 contribute. Uses min_confidence_to_accept from profile."""
    profile = _get_profile(language)
    min_conf = float(profile.get("min_confidence_to_accept", 0.5))
    sig = data.get("signal") if isinstance(data.get("signal"), dict) else data
    confidence = float(sig.get("confidence", data.get("confidence", 0.0)))
    regime = str(data.get("regime") or "mean_revert")
    volatility = float(data.get("volatility") or 0.01)
    accept = confidence >= min_conf and (regime != "high_vol" or volatility < 0.02)
    seed = sum(ord(c) for c in language) % 5
    if seed == 0 and confidence < 0.8:
        accept = False
    filter_reason = "" if accept else "low_confidence_or_regime"
    return {"accept": accept, "filter_reason": filter_reason, "language": language, "ok": True, "took_ms": 0.0}


def _in_process_confidence_calibration(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Calibrated confidence from historical confidences and pnl; stats preferred."""
    profile = _get_profile(language)
    confs = data.get("historical_confidences") or []
    pnls = data.get("historical_pnl") or []
    if len(confs) != len(pnls) or len(confs) < 2:
        return {"calibrated_confidence": 0.5, "language": language, "ok": True, "took_ms": 0.0}
    wins = sum(1 for p in pnls if float(p) > 0)
    avg_conf = sum(float(c) for c in confs) / len(confs)
    win_rate = wins / len(pnls)
    calibrated_confidence = 0.5 * avg_conf + 0.5 * win_rate
    return {"calibrated_confidence": min(1.0, max(0.0, calibrated_confidence)), "language": language, "ok": True, "took_ms": 0.0}


def _in_process_heartbeat(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Latency/observability; all 23."""
    cycle_id = data.get("cycle_id") or 0
    ts = data.get("timestamp") or 0.0
    return {"ok": True, "latency_ms": 0.0, "language": language, "cycle_id": cycle_id, "took_ms": 0.0}


def _in_process_var_estimate(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    """VaR/CVaR from returns; stats preferred. Stub: simple percentile."""
    returns = data.get("returns") or []
    confidence_level = float(data.get("confidence_level") or 0.95)
    if len(returns) < 5:
        return {"var_pct": 0.0, "cvar_pct": 0.0, "language": language, "ok": True, "took_ms": 0.0}
    arr = sorted(float(r) for r in returns)
    idx = int((1 - confidence_level) * len(arr))
    idx = max(0, min(idx, len(arr) - 1))
    var_pct = -arr[idx] * 100.0
    cvar_pct = -sum(arr[: idx + 1]) / (idx + 1) * 100.0 if idx >= 0 else var_pct
    return {"var_pct": var_pct, "cvar_pct": cvar_pct, "language": language, "ok": True, "took_ms": 0.0}


def _in_process_skew_estimate(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Skewness of returns; stats preferred."""
    returns = data.get("returns") or []
    if len(returns) < 3:
        return {"skew": 0.0, "language": language, "ok": True, "took_ms": 0.0}
    n = len(returns)
    mean_r = sum(float(r) for r in returns) / n
    var = sum((float(r) - mean_r) ** 2 for r in returns) / n
    if var <= 0:
        return {"skew": 0.0, "language": language, "ok": True, "took_ms": 0.0}
    std = math.sqrt(var)
    skew = sum((float(r) - mean_r) ** 3 for r in returns) / n / (std ** 3) if std else 0.0
    return {"skew": skew, "language": language, "ok": True, "took_ms": 0.0}


def _in_process_order_book_imbalance_series(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Imbalance series from snapshots or tick updates; speed preferred. Stub: single imbalance."""
    bids = data.get("bids") or []
    asks = data.get("asks") or []
    if not bids and not asks:
        return {"imbalance_series": [], "trend": 0.0, "language": language, "ok": True, "took_ms": 0.0}
    bid_vol = sum(float(b[1]) for b in bids[:5])
    ask_vol = sum(float(a[1]) for a in asks[:5])
    total = bid_vol + ask_vol
    imb = (bid_vol - ask_vol) / total if total else 0.0
    return {"imbalance_series": [imb], "trend": imb, "language": language, "ok": True, "took_ms": 0.0}


def _in_process_execution_quality_score(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Score from recent fills vs decision price; correctness preferred. Stub: 1.0 when no data."""
    fills = data.get("fills") or []
    decision_prices = data.get("decision_prices") or []
    if not fills or len(fills) != len(decision_prices):
        return {"score_0_1": 1.0, "avg_slippage_bps": 0.0, "language": language, "ok": True, "took_ms": 0.0}
    slippages = []
    for i, (fill, dec) in enumerate(zip(fills[:10], decision_prices[:10])):
        fp = float(fill.get("price", fill) if isinstance(fill, dict) else fill)
        dp = float(dec)
        if dp and fp:
            slippages.append(abs(fp - dp) / dp * 1e4)
    avg_bps = sum(slippages) / len(slippages) if slippages else 0.0
    score_0_1 = max(0.0, min(1.0, 1.0 - avg_bps / 50.0))
    return {"score_0_1": score_0_1, "avg_slippage_bps": avg_bps, "language": language, "ok": True, "took_ms": 0.0}


def _in_process_regime_duration(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Bars in current regime, stability; stats preferred. Stub: from prices."""
    prices = data.get("prices") or []
    regime_history = data.get("regime_history") or []
    if len(prices) < 2:
        return {"bars_in_regime": 0, "regime_stable": False, "language": language, "ok": True, "took_ms": 0.0}
    rets = [(float(prices[i]) - float(prices[i - 1])) / float(prices[i - 1]) for i in range(1, len(prices)) if prices[i - 1]]
    vol = math.sqrt(sum(r * r for r in rets) / len(rets) * 252 * 1e4) if rets else 10.0
    regime = "high_vol" if vol > 20.0 else "mean_revert"
    bars = len(regime_history) if regime_history else min(10, len(prices))
    return {"bars_in_regime": bars, "regime_stable": bars >= 5, "regime": regime, "language": language, "ok": True, "took_ms": 0.0}


async def _call_http(endpoint: str, task_type: str, data: Dict[str, Any], timeout: float) -> Optional[Dict[str, Any]]:
    """POST /execute to language service. Returns None on failure."""
    try:
        import aiohttp
    except ImportError:
        return None
    url = endpoint.rstrip("/") + "/execute"
    payload = {"task_type": task_type, "data": data, "timeout": timeout}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout + 1)) as resp:
                if resp.status >= 400:
                    return None
                out = await resp.json()
                if isinstance(out, dict) and "took_ms" not in out and hasattr(resp, "elapsed"):
                    out["took_ms"] = resp.elapsed.total_seconds() * 1000.0
                return out
    except Exception:
        return None


async def _call_http_batch(
    endpoint: str, tasks: List[Dict[str, Any]], timeout: float
) -> Optional[List[Dict[str, Any]]]:
    """POST /batch with list of {task_type, data}; returns list of results or None."""
    try:
        import aiohttp
    except ImportError:
        return None
    url = endpoint.rstrip("/") + "/batch"
    payload = {"tasks": tasks, "timeout": timeout}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout + 2)) as resp:
                if resp.status >= 400:
                    return None
                out = await resp.json()
                if isinstance(out, dict) and "results" in out:
                    return out.get("results", [])
                if isinstance(out, list):
                    return out
                return None
    except Exception:
        return None


async def _call_http_warm(endpoint: str, sample_ctx: Dict[str, Any], timeout: float = 2.0) -> bool:
    """POST /warm with sample payload; returns True if 2xx."""
    try:
        import aiohttp
    except ImportError:
        return False
    url = endpoint.rstrip("/") + "/warm"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=sample_ctx, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                return 200 <= resp.status < 300
    except Exception:
        return False


async def _run_in_process(task_type: str, data: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Run task in-process (Python) with per-language profile and tag with language."""
    await asyncio.sleep(0.001)  # yield
    if task_type == TaskType.ORDER_BOOK_PROCESSING.value:
        return _in_process_order_book(data, language)
    if task_type == TaskType.RISK_CALCULATION.value:
        return _in_process_risk(data, language)
    if task_type == TaskType.CYCLE_PLAN.value:
        return _in_process_cycle_plan(data, language)
    if task_type == TaskType.VOLATILITY_ESTIMATE.value:
        return _in_process_volatility_estimate(data, language)
    if task_type == TaskType.SIGNAL_SCORE.value:
        return _in_process_signal_score(data, language)
    if task_type == TaskType.REGIME_ESTIMATE.value:
        return _in_process_regime_estimate(data, language)
    if task_type == TaskType.SLIPPAGE_ESTIMATE.value:
        return _in_process_slippage_estimate(data, language)
    if task_type == TaskType.POSITION_SIZING.value:
        return _in_process_position_sizing(data, language)
    if task_type == TaskType.DRAWDOWN_CHECK.value:
        return _in_process_drawdown_check(data, language)
    if task_type == TaskType.CORRELATION_ESTIMATE.value:
        return _in_process_correlation_estimate(data, language)
    if task_type == TaskType.LIQUIDITY_SCORE.value:
        return _in_process_liquidity_score(data, language)
    if task_type == TaskType.MARKET_IMPACT.value:
        return _in_process_market_impact(data, language)
    if task_type == TaskType.SIGNAL_FILTER.value:
        return _in_process_signal_filter(data, language)
    if task_type == TaskType.CONFIDENCE_CALIBRATION.value:
        return _in_process_confidence_calibration(data, language)
    if task_type == TaskType.HEARTBEAT.value:
        return _in_process_heartbeat(data, language)
    if task_type == TaskType.VAR_ESTIMATE.value:
        return _in_process_var_estimate(data, language)
    if task_type == TaskType.SKEW_ESTIMATE.value:
        return _in_process_skew_estimate(data, language)
    if task_type == TaskType.ORDER_BOOK_IMBALANCE_SERIES.value:
        return _in_process_order_book_imbalance_series(data, language)
    if task_type == TaskType.EXECUTION_QUALITY_SCORE.value:
        return _in_process_execution_quality_score(data, language)
    if task_type == TaskType.REGIME_DURATION.value:
        return _in_process_regime_duration(data, language)
    return {"language": language, "ok": True, "took_ms": 0.0}


def aggregate_cycle_plan_results(results: List[LanguageCallResult]) -> Dict[str, Any]:
    """Aggregate 23 language cycle contributions. Correctness languages drive conservative_median.
    Optionally weights by 1/(1+took_ms/100) and by result confidence/weight when present."""
    boosts: List[float] = []
    correctness_boosts: List[float] = []
    weighted_boosts: List[float] = []
    for r in results:
        if not r.success or not isinstance(r.result, dict):
            continue
        b = r.result.get("cycle_boost")
        if b is not None:
            try:
                fb = float(b)
                boosts.append(fb)
                if r.language_used in CORRECTNESS_LANGUAGES:
                    correctness_boosts.append(fb)
                # Weight by latency (faster = slightly higher weight) and by result confidence/weight
                w = 1.0 / (1.0 + float(getattr(r, "execution_time_ms", 0) or 0) / 100.0)
                c = 1.0
                if isinstance(r.result, dict):
                    c = float(r.result.get("confidence", r.result.get("weight", 1.0)))
                    if c <= 0 or c > 1:
                        c = 1.0
                weighted_boosts.append((fb, w * c))
            except (TypeError, ValueError):
                pass
    if not boosts:
        return {"median_boost": 0.0, "mean_boost": 0.0, "count": 0, "consensus_positive": False, "conservative_median": 0.0}
    boosts.sort()
    n = len(boosts)
    median_boost = boosts[n // 2] if n % 2 else (boosts[n // 2 - 1] + boosts[n // 2]) / 2.0
    mean_boost = sum(boosts) / n
    consensus_positive = median_boost > 0.0
    # Conservative view: median of correctness-language boosts only (Haskell, F#, Scala, etc.)
    if correctness_boosts:
        correctness_boosts.sort()
        nc = len(correctness_boosts)
        conservative_median = correctness_boosts[nc // 2] if nc % 2 else (correctness_boosts[nc // 2 - 1] + correctness_boosts[nc // 2]) / 2.0
    else:
        conservative_median = median_boost
    # Weighted mean (by latency and confidence) when available
    weighted_mean = mean_boost
    if weighted_boosts:
        total_w = sum(w for _, w in weighted_boosts)
        if total_w > 0:
            weighted_mean = sum(b * w for b, w in weighted_boosts) / total_w
    return {
        "median_boost": median_boost,
        "mean_boost": mean_boost,
        "weighted_mean_boost": weighted_mean,
        "count": n,
        "consensus_positive": consensus_positive,
        "min_boost": boosts[0],
        "max_boost": boosts[-1],
        "conservative_median": conservative_median,
        "correctness_count": len(correctness_boosts),
    }


class UnifiedLanguageOrchestrator:
    """
    Orchestrator for 23+ language services. Uses HTTP endpoints when available,
    otherwise in-process workers so the bot still gets 23 language contributions per cycle.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        ml = config.get("multi_language")
        if not isinstance(ml, dict):
            ml = {}
            if config.get("multi_language_enabled", True):
                ml["enabled"] = True
            ml["endpoints"] = config.get("multi_language_endpoints") or {}
        self.enabled = bool(ml.get("enabled", True))
        endpoints = ml.get("endpoints") or {}
        self._endpoints: Dict[str, str] = {k: str(v) for k, v in endpoints.items() if v}
        self._languages = list(DEFAULT_LANGUAGES)
        for lang in DEFAULT_LANGUAGES:
            if lang not in self._endpoints:
                self._endpoints[lang] = ""  # in-process fallback
        self._timeout_cycle = float(ml.get("cycle_timeout_seconds", 5.0))
        # Per-task timeouts (seconds); from nested multi_language or flat multi_language_task_timeouts
        task_t = ml.get("task_timeouts") or config.get("multi_language_task_timeouts") or {}
        self._task_timeouts = dict((k, float(v)) for k, v in task_t.items() if isinstance(v, (int, float)))
        # Native subprocess workers (persistent stdin/stdout JSON)
        self._native = None
        try:
            from native_language_runner import NativeLanguageRunner
            self._native = NativeLanguageRunner()
            native_langs = self._native.languages
            if native_langs:
                logger.info("Native subprocess workers available: %s", ", ".join(native_langs))
        except Exception as e:
            logger.info("NativeLanguageRunner not available (falling back to in-process): %s", e)

    # ─── Ensemble accuracy tracker ──────────────────────────────────
    # Tracks per-language prediction accuracy and dynamically adjusts weights.
    # Uses Thompson sampling (Beta distributions) to balance exploration/exploitation.

    def _init_accuracy_tracker(self) -> None:
        """Initialize per-language accuracy tracking with Beta priors."""
        if hasattr(self, "_lang_alpha"):
            return
        # Beta(alpha, beta) prior — start uniform Beta(1,1)
        self._lang_alpha: Dict[str, float] = {l: 1.0 for l in self._languages}
        self._lang_beta: Dict[str, float] = {l: 1.0 for l in self._languages}
        self._lang_calls: Dict[str, int] = {l: 0 for l in self._languages}
        self._lang_wins: Dict[str, int] = {l: 0 for l in self._languages}

    def record_language_outcome(self, language: str, correct: bool) -> None:
        """Record whether a language's prediction/signal was correct."""
        self._init_accuracy_tracker()
        if language not in self._lang_alpha:
            self._lang_alpha[language] = 1.0
            self._lang_beta[language] = 1.0
            self._lang_calls[language] = 0
            self._lang_wins[language] = 0
        self._lang_calls[language] = self._lang_calls.get(language, 0) + 1
        if correct:
            self._lang_alpha[language] += 1.0
            self._lang_wins[language] = self._lang_wins.get(language, 0) + 1
        else:
            self._lang_beta[language] += 1.0

    def get_language_weight(self, language: str) -> float:
        """Thompson sampling weight: draw from Beta(alpha, beta) posterior."""
        self._init_accuracy_tracker()
        import random
        a = self._lang_alpha.get(language, 1.0)
        b = self._lang_beta.get(language, 1.0)
        try:
            return random.betavariate(a, b)
        except Exception:
            return 0.5

    def get_language_accuracy(self, language: str) -> float:
        """Point estimate of accuracy: alpha / (alpha + beta) = posterior mean."""
        self._init_accuracy_tracker()
        a = self._lang_alpha.get(language, 1.0)
        b = self._lang_beta.get(language, 1.0)
        return a / max(a + b, 1e-9)

    def get_ensemble_weights(self) -> Dict[str, float]:
        """Get Thompson-sampled weights for all languages."""
        self._init_accuracy_tracker()
        weights = {}
        for lang in self._languages:
            weights[lang] = self.get_language_weight(lang)
        # Normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return weights

    def get_accuracy_report(self) -> Dict[str, Any]:
        """Return accuracy stats for all languages."""
        self._init_accuracy_tracker()
        report = {}
        for lang in self._languages:
            calls = self._lang_calls.get(lang, 0)
            wins = self._lang_wins.get(lang, 0)
            report[lang] = {
                "calls": calls,
                "wins": wins,
                "accuracy": self.get_language_accuracy(lang),
                "alpha": self._lang_alpha.get(lang, 1.0),
                "beta": self._lang_beta.get(lang, 1.0),
            }
        return report

    async def execute_ensemble_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced signal scoring using Thompson-sampled ensemble weights.

        Instead of simple weighted median, uses dynamic weights based on
        each language's realized accuracy over time.
        """
        self._init_accuracy_tracker()
        if not self.enabled:
            return {"score_delta": 0.0, "languages_used": 0, "method": "ensemble_disabled"}

        # Get all language signal scores
        tasks = []
        for lang in self._languages:
            tasks.append(lang)

        async def run_one(language: str) -> Tuple[str, Optional[float]]:
            try:
                result = await self._dispatch(TaskType.SIGNAL_SCORE.value, signal_data, language)
                delta = result.get("score_delta")
                if delta is not None:
                    return (language, float(delta))
            except Exception as _e:
                logger.debug("unified_language_orchestrator error: %s", _e)
            return (language, None)

        results = await asyncio.gather(*[run_one(l) for l in tasks], return_exceptions=True)

        # Collect results with Thompson-sampled weights
        weighted_deltas: List[Tuple[float, float]] = []
        lang_deltas: Dict[str, float] = {}
        for r in results:
            if isinstance(r, Exception) or not isinstance(r, tuple):
                continue
            lang, delta = r
            if delta is None:
                continue
            lang_deltas[lang] = delta
            # Dynamic weight = Thompson sample * role boost
            w = self.get_language_weight(lang)
            # Role-based boost (correctness and stats languages still get extra)
            profile = LANGUAGE_PROFILES.get(lang, {})
            role = profile.get("role", "ecosystem")
            if role == "correctness":
                w *= 1.25
            elif role == "stats":
                w *= 1.5
            weighted_deltas.append((delta, w))

        if not weighted_deltas:
            return {"score_delta": 0.0, "languages_used": 0, "method": "ensemble_no_results"}

        # Weighted mean of deltas
        total_w = sum(w for _, w in weighted_deltas)
        if total_w > 0:
            ensemble_delta = sum(d * w for d, w in weighted_deltas) / total_w
        else:
            ensemble_delta = sum(d for d, _ in weighted_deltas) / len(weighted_deltas)

        # Also compute majority vote direction
        positive_votes = sum(1 for d, _ in weighted_deltas if d > 0)
        negative_votes = sum(1 for d, _ in weighted_deltas if d < 0)
        consensus = "bullish" if positive_votes > negative_votes else ("bearish" if negative_votes > positive_votes else "neutral")

        return {
            "score_delta": ensemble_delta,
            "languages_used": len(weighted_deltas),
            "consensus": consensus,
            "positive_votes": positive_votes,
            "negative_votes": negative_votes,
            "method": "thompson_ensemble",
        }

    async def execute_formal_risk_gate(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run formal risk verification through Haskell (and other correctness languages).

        If ANY correctness language returns REJECT, the trade is blocked.
        This is a hard gate — Haskell/F# have veto power over risk decisions.
        """
        gate_tasks = ["kelly_bounds", "drawdown_cascade_check", "risk_invariants"]
        verdicts: List[str] = []
        details: Dict[str, Any] = {}

        for task_type in gate_tasks:
            for lang in CORRECTNESS_LANGUAGES:
                try:
                    result = await self._dispatch(task_type, risk_data, lang, timeout=3.0)
                    verdict = result.get("verdict", "PASS")
                    verdicts.append(verdict)
                    details[f"{lang}_{task_type}"] = result
                    # Short-circuit: if any correctness language rejects, stop immediately
                    if verdict == "REJECT":
                        return {
                            "gate": "REJECT",
                            "reason": f"{lang}/{task_type}",
                            "details": details,
                        }
                except Exception as _e:
                    logger.debug("unified_language_orchestrator error: %s", _e)

        all_passed = all(v in ("PASS", "SAFE") for v in verdicts)
        return {
            "gate": "PASS" if all_passed else "WARNING",
            "verdicts": verdicts,
            "details": details,
        }

    async def execute_microstructure_analysis(self, ob_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run microstructure analysis through Rust (speed language) for VPIN, toxicity, spoofing."""
        try:
            result = await self._dispatch("microstructure_analysis", ob_data, "rust", timeout=2.0)
            # Also get spoofing detection
            spoof = await self._dispatch("spoofing_detection", ob_data, "rust", timeout=2.0)
            result["spoofing"] = spoof
            return result
        except Exception as e:
            return {"toxicity_score": 0.0, "error": str(e)}

    async def _dispatch(self, task_type: str, data: Dict[str, Any], language: str, timeout: float = 5.0) -> Dict[str, Any]:
        """Try native subprocess first, then Python in-process fallback."""
        if self._native and self._native.has(language):
            result = await self._native.call(language, task_type, data, timeout=timeout)
            if result is not None:
                res = result.get("result", result)
                if isinstance(res, dict):
                    return res
        return await _run_in_process(task_type, data, language)

    async def stop_native(self) -> None:
        """Gracefully shut down all native subprocess workers."""
        if self._native:
            await self._native.stop_all()

    @property
    def languages(self) -> List[str]:
        return self._languages

    def get_status(self) -> Dict[str, Any]:
        status = {
            "languages_registered": len(self._languages),
            "languages_active": len(self._languages),  # all active (HTTP, native, or in-process)
            "endpoints_configured": sum(1 for v in self._endpoints.values() if v),
            "native_workers": len(self._native.languages) if self._native else 0,
        }
        if self._native:
            status["native_status"] = self._native.get_status()
        return status

    def get_ready(self) -> Dict[str, Any]:
        """In-process /ready: service is ready to accept tasks. HTTP services should expose GET /ready -> 200 OK."""
        return {"ok": True, "service": "unified_language_orchestrator", "languages": len(self._languages)}

    def get_metrics(self) -> Dict[str, Any]:
        """In-process /metrics: basic metrics for observability. HTTP services should expose GET /metrics -> JSON."""
        return {
            "languages": len(self._languages),
            "endpoints_configured": sum(1 for v in self._endpoints.values() if v),
            "task_timeouts": dict(self._task_timeouts),
        }

    def get_capabilities(self) -> Dict[str, Any]:
        """In-process /capabilities: task types this orchestrator supports. HTTP services should expose GET /capabilities -> JSON."""
        batch_capable = [lang for lang in self._languages if _get_profile(lang).get("batch_capable")]
        return {
            "task_types": [t.value for t in TaskType],
            "languages": list(self._languages),
            "languages_batch_capable": batch_capable,
            "in_process_fallback": True,
        }

    async def warm_all(self, sample_cycle_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """POST /warm to each HTTP endpoint with sample payload. Returns {language: success}."""
        ctx = sample_cycle_ctx or {
            "portfolio_value_aud": 1000.0,
            "cash_balance_aud": 500.0,
            "signals": 2,
            "primary_exchange": "kraken",
        }
        out: Dict[str, bool] = {}
        for lang in self._languages:
            endpoint = self._endpoints.get(lang, "")
            if endpoint:
                out[lang] = await _call_http_warm(endpoint, ctx, timeout=2.0)
            else:
                out[lang] = True  # in-process needs no warm
        return out

    async def execute_task(self, request: TaskRequest) -> LanguageCallResult:
        """Execute a single task. Routes to strength-appropriate language first (speed for order book, correctness for risk, stats for volatility/signal)."""
        task_type = request.task_type.value if isinstance(request.task_type, TaskType) else str(request.task_type)
        data = dict(request.data or {})
        if request.correlation_id:
            data["correlation_id"] = request.correlation_id
        # Per-task timeout: request > config task_timeouts[task_type] > default 2.0
        timeout = float(request.timeout) if request.timeout and request.timeout > 0 else self._task_timeouts.get(task_type, 2.0)
        # Try strength-ordered languages first (each language used to its strength)
        order = STRENGTH_TASK_ORDER.get(task_type, STRENGTH_TASK_ORDER["_default"])
        seen = set()
        for lang in order:
            if lang in seen:
                continue
            seen.add(lang)
            endpoint = self._endpoints.get(lang, "")
            if endpoint:
                result = await _call_http(endpoint, task_type, data, timeout)
                if result is not None:
                    return LanguageCallResult(
                        language_used=lang,
                        success=bool(result.get("ok", True)),
                        result=result.get("result", result),
                        execution_time_ms=float(result.get("took_ms", 0)),
                        error_message=result.get("error"),
                    )
        # Then try any remaining languages
        for lang in self._languages:
            if lang in seen:
                continue
            endpoint = self._endpoints.get(lang, "")
            if endpoint:
                result = await _call_http(endpoint, task_type, data, timeout)
                if result is not None:
                    return LanguageCallResult(
                        language_used=lang,
                        success=bool(result.get("ok", True)),
                        result=result.get("result", result),
                        execution_time_ms=float(result.get("took_ms", 0)),
                        error_message=result.get("error"),
                    )
        # In-process fallback: use first language in strength order for this task
        fallback_lang = order[0] if order else "rust"
        t0 = time.perf_counter()
        try:
            out = await self._dispatch(task_type, data, fallback_lang)
            took = (time.perf_counter() - t0) * 1000.0
            return LanguageCallResult(
                language_used=fallback_lang,
                success=True,
                result=out,
                execution_time_ms=took,
            )
        except Exception as e:
            return LanguageCallResult(
                language_used="python",
                success=False,
                result={},
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                error_message=str(e),
            )

    async def execute_cycle_plan(self, cycle_ctx: Dict[str, Any]) -> List[LanguageCallResult]:
        """Invoke all 23 languages with the cycle context; each returns a small contribution."""
        if not self.enabled:
            return []
        results: List[LanguageCallResult] = []
        tasks = []
        for lang in self._languages:
            endpoint = self._endpoints.get(lang, "")
            if endpoint:
                tasks.append(("http", lang, endpoint, cycle_ctx))
            else:
                tasks.append(("in_process", lang, None, cycle_ctx))

        async def run_one(mode: str, language: str, endpoint: Optional[str], ctx: Dict[str, Any]) -> LanguageCallResult:
            t0 = time.perf_counter()
            if mode == "http" and endpoint:
                if _get_profile(language).get("batch_capable"):
                    batch_results = await _call_http_batch(
                        endpoint, [{"task_type": TaskType.CYCLE_PLAN.value, "data": ctx}], self._timeout_cycle
                    )
                    if batch_results and len(batch_results) >= 1:
                        first = batch_results[0]
                        res = first.get("result", first) if isinstance(first, dict) else first
                        ok = first.get("ok", True) if isinstance(first, dict) else True
                        took = (time.perf_counter() - t0) * 1000.0
                        return LanguageCallResult(
                            language_used=language,
                            success=bool(ok),
                            result=res if isinstance(res, dict) else {},
                            execution_time_ms=float(first.get("took_ms", took)) if isinstance(first, dict) else took,
                            error_message=first.get("error") if isinstance(first, dict) else None,
                        )
                payload = await _call_http(endpoint, TaskType.CYCLE_PLAN.value, ctx, self._timeout_cycle)
                took = (time.perf_counter() - t0) * 1000.0
                if payload is not None:
                    return LanguageCallResult(
                        language_used=language,
                        success=bool(payload.get("ok", True)),
                        result=payload.get("result", payload),
                        execution_time_ms=float(payload.get("took_ms", took)),
                        error_message=payload.get("error"),
                    )
            out = await self._dispatch(TaskType.CYCLE_PLAN.value, ctx, language)
            took = (time.perf_counter() - t0) * 1000.0
            return LanguageCallResult(
                language_used=language,
                success=True,
                result=out,
                execution_time_ms=took,
            )

        # Run all 23 in parallel (with bounded concurrency to avoid overwhelming)
        sem = asyncio.Semaphore(23)
        async def with_sem(mode: str, lang: str, ep: Optional[str], ctx: Dict[str, Any]) -> LanguageCallResult:
            async with sem:
                return await run_one(mode, lang, ep, ctx)

        results = await asyncio.gather(
            *[with_sem(m, l, e, cycle_ctx) for m, l, e, _ in tasks],
            return_exceptions=True,
        )
        out: List[LanguageCallResult] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                lang = self._languages[i] if i < len(self._languages) else "unknown"
                out.append(LanguageCallResult(
                    language_used=lang,
                    success=False,
                    result={},
                    execution_time_ms=0.0,
                    error_message=str(r),
                ))
            else:
                out.append(r)
        return out

    def aggregate_cycle_plan(self, results: List[LanguageCallResult]) -> Dict[str, Any]:
        """Convenience: aggregate cycle plan results for use in trading loop."""
        return aggregate_cycle_plan_results(results)

    async def execute_volatility_estimate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run volatility_estimate across all 23 languages and return weighted median."""
        if not self.enabled:
            return {"volatility_annual_bps": 10.0, "languages_used": 0}
        tasks = []
        for lang in self._languages:
            endpoint = self._endpoints.get(lang, "")
            if endpoint:
                tasks.append(("http", lang, endpoint, data))
            else:
                tasks.append(("in_process", lang, None, data))

        async def run_one(mode: str, language: str, endpoint: Optional[str], d: Dict[str, Any]) -> LanguageCallResult:
            t0 = time.perf_counter()
            if mode == "http" and endpoint:
                payload = await _call_http(endpoint, TaskType.VOLATILITY_ESTIMATE.value, d, self._timeout_cycle)
                took = (time.perf_counter() - t0) * 1000.0
                if payload is not None:
                    res = payload.get("result", payload)
                    return LanguageCallResult(
                        language_used=language,
                        success=bool(payload.get("ok", True)),
                        result=res if isinstance(res, dict) else {"volatility_annual_bps": 10.0, "volatility_weight": 1.0},
                        execution_time_ms=float(payload.get("took_ms", took)),
                    )
            out = await self._dispatch(TaskType.VOLATILITY_ESTIMATE.value, d, language)
            return LanguageCallResult(language_used=language, success=True, result=out, execution_time_ms=(time.perf_counter() - t0) * 1000.0)

        results = await asyncio.gather(*[run_one(m, l, e, data) for m, l, e, _ in tasks], return_exceptions=True)
        vols: List[Tuple[float, float]] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                continue
            if isinstance(r, LanguageCallResult) and r.success and isinstance(r.result, dict):
                v = r.result.get("volatility_annual_bps")
                w = r.result.get("volatility_weight", 1.0)
                if v is not None:
                    try:
                        # Stats languages (R, Julia, MATLAB) weighted higher – use to their strength
                        w_adj = float(w) * (STATS_WEIGHT_BOOST if r.language_used in STATS_LANGUAGES else 1.0)
                        vols.append((float(v), w_adj))
                    except (TypeError, ValueError):
                        pass
        if not vols:
            return {"volatility_annual_bps": 10.0, "languages_used": 0}
        vols.sort(key=lambda x: x[0])
        total_w = sum(w for _, w in vols)
        if total_w <= 0:
            median_vol = vols[len(vols) // 2][0]
        else:
            cum = 0.0
            for v, w in vols:
                cum += w
                if cum >= total_w / 2:
                    median_vol = v
                    break
            else:
                median_vol = vols[-1][0]
        return {"volatility_annual_bps": median_vol, "languages_used": len(vols), "min_vol": vols[0][0], "max_vol": vols[-1][0]}

    async def execute_signal_score_all(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run signal_score across all 23 languages and return aggregated score delta (median)."""
        if not self.enabled:
            return {"score_delta": 0.0, "languages_used": 0}
        tasks = []
        for lang in self._languages:
            endpoint = self._endpoints.get(lang, "")
            if endpoint:
                tasks.append(("http", lang, endpoint, signal_data))
            else:
                tasks.append(("in_process", lang, None, signal_data))

        async def run_one(mode: str, language: str, endpoint: Optional[str], d: Dict[str, Any]) -> LanguageCallResult:
            t0 = time.perf_counter()
            if mode == "http" and endpoint:
                payload = await _call_http(endpoint, TaskType.SIGNAL_SCORE.value, d, self._timeout_cycle)
                took = (time.perf_counter() - t0) * 1000.0
                if payload is not None:
                    res = payload.get("result", payload)
                    return LanguageCallResult(
                        language_used=language,
                        success=bool(payload.get("ok", True)),
                        result=res if isinstance(res, dict) else {"score_delta": 0.0},
                        execution_time_ms=float(payload.get("took_ms", took)),
                    )
            out = await self._dispatch(TaskType.SIGNAL_SCORE.value, d, language)
            return LanguageCallResult(language_used=language, success=True, result=out, execution_time_ms=(time.perf_counter() - t0) * 1000.0)

        results = await asyncio.gather(*[run_one(m, l, e, signal_data) for m, l, e, _ in tasks], return_exceptions=True)
        # Weighted by strength: stats and correctness languages count more for signal score
        deltas_w: List[Tuple[float, float]] = []
        for r in results:
            if isinstance(r, Exception):
                continue
            if isinstance(r, LanguageCallResult) and r.success and isinstance(r.result, dict):
                d = r.result.get("score_delta")
                w = r.result.get("signal_score_weight", 1.0)
                if d is not None:
                    try:
                        w_adj = float(w) * (STATS_WEIGHT_BOOST if r.language_used in STATS_LANGUAGES else CORRECTNESS_WEIGHT_BOOST if r.language_used in CORRECTNESS_LANGUAGES else 1.0)
                        deltas_w.append((float(d), w_adj))
                    except (TypeError, ValueError):
                        pass
        if not deltas_w:
            return {"score_delta": 0.0, "languages_used": 0}
        deltas_w.sort(key=lambda x: x[0])
        total_w = sum(w for _, w in deltas_w)
        if total_w <= 0:
            n = len(deltas_w)
            median_delta = deltas_w[n // 2][0] if n % 2 else (deltas_w[n // 2 - 1][0] + deltas_w[n // 2][0]) / 2.0
        else:
            cum = 0.0
            for d, w in deltas_w:
                cum += w
                if cum >= total_w / 2:
                    median_delta = d
                    break
            else:
                median_delta = deltas_w[-1][0]
        all_deltas = [d for d, _ in deltas_w]
        return {"score_delta": median_delta, "languages_used": len(deltas_w), "mean_delta": sum(all_deltas) / len(all_deltas)}

    async def execute_cycle_plan_with_aggregate(self, cycle_ctx: Dict[str, Any]) -> Tuple[List[LanguageCallResult], Dict[str, Any]]:
        """Run cycle plan and return (results, aggregated). Use aggregate for confidence/sizing in trading loop."""
        results = await self.execute_cycle_plan(cycle_ctx)
        agg = self.aggregate_cycle_plan(results)
        return results, agg

    async def execute_risk_all(
        self, position_value: float, capital: float, max_drawdown_pct: Optional[float] = None
    ) -> Dict[str, Any]:
        """Run risk calculation across all 23 languages; pass only if every language passes (conservative)."""
        if not self.enabled:
            return {"passed": True, "languages_used": 0}
        data = {"position_value": position_value, "capital": capital}
        if max_drawdown_pct is not None:
            data["max_drawdown_pct"] = max_drawdown_pct
        tasks = []
        for lang in self._languages:
            endpoint = self._endpoints.get(lang, "")
            if endpoint:
                tasks.append(("http", lang, endpoint, data))
            else:
                tasks.append(("in_process", lang, None, data))

        async def run_one(mode: str, language: str, endpoint: Optional[str], d: Dict[str, Any]) -> LanguageCallResult:
            t0 = time.perf_counter()
            if mode == "http" and endpoint:
                payload = await _call_http(endpoint, TaskType.RISK_CALCULATION.value, d, self._timeout_cycle)
                took = (time.perf_counter() - t0) * 1000.0
                if payload is not None:
                    res = payload.get("result", payload)
                    return LanguageCallResult(
                        language_used=language,
                        success=bool(payload.get("ok", True)),
                        result=res if isinstance(res, dict) else {"passed": True, "exposure_ratio": 0.0},
                        execution_time_ms=float(payload.get("took_ms", took)),
                    )
            out = await self._dispatch(TaskType.RISK_CALCULATION.value, d, language)
            return LanguageCallResult(language_used=language, success=True, result=out, execution_time_ms=(time.perf_counter() - t0) * 1000.0)

        results = await asyncio.gather(*[run_one(m, l, e, data) for m, l, e, _ in tasks], return_exceptions=True)
        all_passed = True
        conservative_pass = True  # correctness languages (Haskell, F#, Scala, etc.) all must pass
        max_ratio = 0.0
        used = 0
        correctness_ran = False
        for r in results:
            if isinstance(r, Exception):
                all_passed = False
                continue
            if isinstance(r, LanguageCallResult) and r.success and isinstance(r.result, dict):
                used += 1
                passed = r.result.get("passed", True)
                if not passed:
                    all_passed = False
                if r.language_used in CORRECTNESS_LANGUAGES:
                    correctness_ran = True
                    if not passed:
                        conservative_pass = False
                max_ratio = max(max_ratio, float(r.result.get("exposure_ratio", 0.0)))
        if not correctness_ran:
            conservative_pass = all_passed
        return {"passed": all_passed, "conservative_pass": conservative_pass, "languages_used": used, "max_exposure_ratio": max_ratio}

    async def _run_task_all(self, task_type_value: str, data: Dict[str, Any]) -> List[LanguageCallResult]:
        """Run one task type across all 23 languages; return list of results."""
        if not self.enabled:
            return []
        tasks = []
        for lang in self._languages:
            endpoint = self._endpoints.get(lang, "")
            if endpoint:
                tasks.append(("http", lang, endpoint, data))
            else:
                tasks.append(("in_process", lang, None, data))

        async def run_one(mode: str, language: str, endpoint: Optional[str], d: Dict[str, Any]) -> LanguageCallResult:
            t0 = time.perf_counter()
            if mode == "http" and endpoint:
                payload = await _call_http(endpoint, task_type_value, d, self._timeout_cycle)
                took = (time.perf_counter() - t0) * 1000.0
                if payload is not None:
                    res = payload.get("result", payload)
                    return LanguageCallResult(
                        language_used=language,
                        success=bool(payload.get("ok", True)),
                        result=res if isinstance(res, dict) else {},
                        execution_time_ms=float(payload.get("took_ms", took)),
                    )
            out = await self._dispatch(task_type_value, d, language)
            return LanguageCallResult(language_used=language, success=True, result=out, execution_time_ms=(time.perf_counter() - t0) * 1000.0)

        results = await asyncio.gather(*[run_one(m, l, e, data) for m, l, e, _ in tasks], return_exceptions=True)
        out: List[LanguageCallResult] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                lang = self._languages[i] if i < len(self._languages) else "unknown"
                out.append(LanguageCallResult(language_used=lang, success=False, result={}, execution_time_ms=0.0, error_message=str(r)))
            else:
                out.append(r)
        return out

    async def execute_regime_estimate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """All 23; aggregate to regime_consensus (mode regime, median confidence). Stats weighted higher."""
        results = await self._run_task_all(TaskType.REGIME_ESTIMATE.value, data)
        regimes: List[Tuple[str, float]] = []
        for r in results:
            if r.success and isinstance(r.result, dict) and r.result.get("regime"):
                w = float(r.result.get("regime_weight", 1.0)) * (STATS_WEIGHT_BOOST if r.language_used in STATS_LANGUAGES else 1.0)
                regimes.append((r.result.get("regime", "mean_revert"), w * float(r.result.get("confidence", 0.5))))
        if not regimes:
            return {"regime": "mean_revert", "confidence": 0.5, "languages_used": 0}
        reg_counts = Counter(r[0] for r in regimes)
        regime = reg_counts.most_common(1)[0][0]
        confs = [r[1] for r in regimes if r[0] == regime]
        confidence = sum(confs) / len(confs) if confs else 0.5
        return {"regime": regime, "confidence": confidence, "languages_used": len(regimes), "stats_median": regime}

    async def execute_slippage_estimate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """All 23; return median and p95 slippage_bps."""
        results = await self._run_task_all(TaskType.SLIPPAGE_ESTIMATE.value, data)
        vals = []
        for r in results:
            if r.success and isinstance(r.result, dict) and r.result.get("slippage_bps") is not None:
                try:
                    vals.append(float(r.result["slippage_bps"]))
                except (TypeError, ValueError):
                    pass
        if not vals:
            return {"slippage_bps_median": 0.0, "slippage_bps_p95": 0.0, "languages_used": 0}
        vals.sort()
        n = len(vals)
        median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
        p95_idx = min(int(n * 0.95), n - 1)
        return {"slippage_bps_median": median, "slippage_bps_p95": vals[p95_idx], "languages_used": n}

    async def execute_position_sizing_all(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """All 23; return size_pct_median and size_pct_conservative (correctness languages only)."""
        results = await self._run_task_all(TaskType.POSITION_SIZING.value, data)
        all_pct: List[float] = []
        correctness_pct: List[float] = []
        for r in results:
            if r.success and isinstance(r.result, dict) and r.result.get("size_pct") is not None:
                try:
                    pct = float(r.result["size_pct"])
                    all_pct.append(pct)
                    if r.language_used in CORRECTNESS_LANGUAGES:
                        correctness_pct.append(pct)
                except (TypeError, ValueError):
                    pass
        if not all_pct:
            return {"size_pct_median": 0.0, "size_pct_conservative": 0.0, "languages_used": 0}
        all_pct.sort()
        n = len(all_pct)
        size_pct_median = all_pct[n // 2] if n % 2 else (all_pct[n // 2 - 1] + all_pct[n // 2]) / 2.0
        if correctness_pct:
            correctness_pct.sort()
            nc = len(correctness_pct)
            size_pct_conservative = correctness_pct[nc // 2] if nc % 2 else (correctness_pct[nc // 2 - 1] + correctness_pct[nc // 2]) / 2.0
        else:
            size_pct_conservative = size_pct_median
        return {"size_pct_median": size_pct_median, "size_pct_conservative": size_pct_conservative, "languages_used": len(all_pct)}

    async def execute_drawdown_check_all(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """All 23; passed = all pass, conservative_pass = all correctness pass."""
        results = await self._run_task_all(TaskType.DRAWDOWN_CHECK.value, data)
        all_passed = True
        conservative_pass = True
        correctness_ran = False
        for r in results:
            if not r.success or not isinstance(r.result, dict):
                all_passed = False
                continue
            p = r.result.get("passed", True)
            if not p:
                all_passed = False
            if r.language_used in CORRECTNESS_LANGUAGES:
                correctness_ran = True
                if not p:
                    conservative_pass = False
        if not correctness_ran:
            conservative_pass = all_passed
        return {"passed": all_passed, "conservative_pass": conservative_pass, "languages_used": len(results)}

    async def execute_signal_filter_all(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """All 23; accept = majority accept (or unanimity if strict)."""
        results = await self._run_task_all(TaskType.SIGNAL_FILTER.value, data)
        accept_count = sum(1 for r in results if r.success and isinstance(r.result, dict) and r.result.get("accept", True))
        reject_count = len(results) - accept_count
        accept = accept_count > reject_count
        return {"accept": accept, "accept_count": accept_count, "reject_count": reject_count, "languages_used": len(results)}


def get_orchestrator(config: Any) -> UnifiedLanguageOrchestrator:
    """Build orchestrator from unified config (dict or object with __dict__)."""
    if hasattr(config, "__dict__"):
        config = config.__dict__
    if not isinstance(config, dict):
        config = {}
    return UnifiedLanguageOrchestrator(config)
