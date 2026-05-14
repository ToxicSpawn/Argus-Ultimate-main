"""
Standalone protocol logic for multilang services. Mirrors unified_language_orchestrator
profiles and in-process handlers so each service is self-contained.
"""
import hashlib
import math
from typing import Any, Dict

LANGUAGE_PROFILES: Dict[str, Dict[str, Any]] = {
    "rust":       {"risk_max_ratio": 0.48, "cycle_boost_scale": 1.0,  "volatility_weight": 0.9,  "signal_score_weight": 1.0, "spread_mult": 1.0,  "role": "speed"},
    "cpp":        {"risk_max_ratio": 0.48, "cycle_boost_scale": 1.0,  "volatility_weight": 0.9,  "signal_score_weight": 1.0, "spread_mult": 1.0,  "role": "speed"},
    "cuda":       {"risk_max_ratio": 0.46, "cycle_boost_scale": 0.95, "volatility_weight": 1.1,  "signal_score_weight": 0.95, "spread_mult": 1.02, "role": "speed"},
    "go":         {"risk_max_ratio": 0.47, "cycle_boost_scale": 1.0,  "volatility_weight": 0.95, "signal_score_weight": 1.0, "spread_mult": 1.0,  "role": "speed"},
    "java":       {"risk_max_ratio": 0.45, "cycle_boost_scale": 0.98, "volatility_weight": 1.0,  "signal_score_weight": 1.0, "spread_mult": 1.01, "role": "correctness"},
    "scala":      {"risk_max_ratio": 0.44, "cycle_boost_scale": 0.98, "volatility_weight": 1.0,  "signal_score_weight": 1.02, "spread_mult": 1.01, "role": "correctness"},
    "kotlin":     {"risk_max_ratio": 0.45, "cycle_boost_scale": 0.99, "volatility_weight": 1.0,  "signal_score_weight": 1.0, "spread_mult": 1.01, "role": "correctness"},
    "swift":      {"risk_max_ratio": 0.46, "cycle_boost_scale": 1.0,  "volatility_weight": 0.95, "signal_score_weight": 1.0, "spread_mult": 1.0,  "role": "ecosystem"},
    "csharp":     {"risk_max_ratio": 0.45, "cycle_boost_scale": 0.99, "volatility_weight": 1.0,  "signal_score_weight": 1.0, "spread_mult": 1.01, "role": "ecosystem"},
    "fsharp":     {"risk_max_ratio": 0.42, "cycle_boost_scale": 0.97, "volatility_weight": 1.05, "signal_score_weight": 1.02, "spread_mult": 1.02, "role": "correctness"},
    "javascript": {"risk_max_ratio": 0.46, "cycle_boost_scale": 1.0,  "volatility_weight": 0.95, "signal_score_weight": 1.0, "spread_mult": 1.0,  "role": "ecosystem"},
    "typescript": {"risk_max_ratio": 0.45, "cycle_boost_scale": 0.99, "volatility_weight": 0.98, "signal_score_weight": 1.0, "spread_mult": 1.01, "role": "ecosystem"},
    "elixir":     {"risk_max_ratio": 0.45, "cycle_boost_scale": 1.0,  "volatility_weight": 1.0,  "signal_score_weight": 1.01, "spread_mult": 1.01, "role": "concurrency"},
    "erlang":     {"risk_max_ratio": 0.44, "cycle_boost_scale": 0.99, "volatility_weight": 1.0,  "signal_score_weight": 1.01, "spread_mult": 1.02, "role": "concurrency"},
    "clojure":    {"risk_max_ratio": 0.44, "cycle_boost_scale": 0.98, "volatility_weight": 1.02, "signal_score_weight": 1.02, "spread_mult": 1.01, "role": "correctness"},
    "haskell":    {"risk_max_ratio": 0.40, "cycle_boost_scale": 0.95, "volatility_weight": 1.05, "signal_score_weight": 1.02, "spread_mult": 1.03, "role": "correctness"},
    "ruby":       {"risk_max_ratio": 0.46, "cycle_boost_scale": 1.0,  "volatility_weight": 0.98, "signal_score_weight": 1.0, "spread_mult": 1.0,  "role": "ecosystem"},
    "r":          {"risk_max_ratio": 0.44, "cycle_boost_scale": 1.05, "volatility_weight": 1.2,  "signal_score_weight": 1.05, "spread_mult": 1.02, "role": "stats"},
    "julia":      {"risk_max_ratio": 0.45, "cycle_boost_scale": 1.03, "volatility_weight": 1.15, "signal_score_weight": 1.03, "spread_mult": 1.01, "role": "stats"},
    "matlab":     {"risk_max_ratio": 0.44, "cycle_boost_scale": 1.02, "volatility_weight": 1.12, "signal_score_weight": 1.02, "spread_mult": 1.02, "role": "stats"},
    "crystal":    {"risk_max_ratio": 0.47, "cycle_boost_scale": 1.0,  "volatility_weight": 0.95, "signal_score_weight": 1.0, "spread_mult": 1.0,  "role": "speed"},
    "webassembly": {"risk_max_ratio": 0.45, "cycle_boost_scale": 0.98, "volatility_weight": 1.0,  "signal_score_weight": 1.0, "spread_mult": 1.01, "role": "ecosystem"},
    "mojo":       {"risk_max_ratio": 0.47, "cycle_boost_scale": 1.01, "volatility_weight": 1.05, "signal_score_weight": 1.0, "spread_mult": 1.0,  "role": "speed"},
}


def get_profile(language: str) -> Dict[str, Any]:
    return LANGUAGE_PROFILES.get(language, LANGUAGE_PROFILES["rust"])


def handle_order_book(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    bids = data.get("bids") or []
    asks = data.get("asks") or []
    profile = get_profile(language)
    spread_mult = float(profile.get("spread_mult", 1.0))
    if not bids and not asks:
        return {"spread_bps": 0.0, "imbalance": 0.0, "mid": 0.0, "language": language, "spread_mult": spread_mult}
    best_bid = float(bids[0][0]) if bids else 0.0
    best_ask = float(asks[0][0]) if asks else 0.0
    mid = (best_bid + best_ask) / 2.0 if (best_bid and best_ask) else 0.0
    raw_spread_bps = (best_ask - best_bid) / mid * 1e4 if mid else 0.0
    spread_bps = raw_spread_bps * spread_mult
    bid_vol = sum(float(b[1]) for b in bids[:5]) if bids else 0.0
    ask_vol = sum(float(a[1]) for a in asks[:5]) if asks else 0.0
    total = bid_vol + ask_vol
    imbalance = (bid_vol - ask_vol) / total if total else 0.0
    return {"spread_bps": spread_bps, "imbalance": imbalance, "mid": mid, "language": language, "spread_mult": spread_mult}


def handle_risk(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    profile = get_profile(language)
    max_ratio = float(profile.get("risk_max_ratio", 0.45))
    pv = float(data.get("position_value") or 0.0)
    capital = float(data.get("capital") or 1.0)
    ratio = pv / capital if capital else 0.0
    passed = ratio <= max_ratio
    return {"passed": passed, "exposure_ratio": ratio, "max_ratio": max_ratio, "language": language}


def handle_cycle_plan(ctx: Dict[str, Any], language: str) -> Dict[str, Any]:
    profile = get_profile(language)
    scale = float(profile.get("cycle_boost_scale", 1.0))
    h = hashlib.sha256(str(sorted(ctx.items())).encode()).hexdigest()
    idx = sum(ord(c) for c in language) % 100
    base = ((int(h[:8], 16) % 200) - 100) / 10000.0 + (idx - 50) / 10000.0
    signals = int(ctx.get("signals") or 0)
    cash = float(ctx.get("cash_balance_aud") or 0.0)
    pv = float(ctx.get("portfolio_value_aud") or 1.0)
    cash_ratio = cash / pv if pv else 0.0
    tilt = (cash_ratio - 0.5) * 0.002 + (signals % 3 - 1) * 0.001
    boost = max(-0.015, min(0.015, (base + tilt) * scale))
    return {"language": language, "cycle_boost": boost, "cycle_boost_scale": scale, "ok": True}


def handle_volatility_estimate(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    profile = get_profile(language)
    weight = float(profile.get("volatility_weight", 1.0))
    prices = data.get("prices") or data.get("ohlcv_close") or []
    returns = data.get("returns") or []
    if returns:
        n = len(returns)
        mean_r = sum(returns) / n if n else 0.0
        var = sum((r - mean_r) ** 2 for r in returns) / n if n else 0.0
        vol = math.sqrt(var * 252 * 1e4) if var else 10.0
    elif len(prices) >= 2:
        rets = [(float(prices[i]) - float(prices[i - 1])) / float(prices[i - 1]) for i in range(1, len(prices)) if prices[i - 1]]
        n = len(rets)
        mean_r = sum(rets) / n if n else 0.0
        var = sum((r - mean_r) ** 2 for r in rets) / n if n else 0.0
        vol = math.sqrt(var * 252 * 1e4) if var else 10.0
    else:
        vol = 10.0
    seed = sum(ord(c) for c in language) % 7
    vol_adj = vol * (1.0 + (seed - 3) * 0.01) * weight
    return {"volatility_annual_bps": vol_adj, "volatility_weight": weight, "language": language, "ok": True}


def handle_signal_score(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    profile = get_profile(language)
    weight = float(profile.get("signal_score_weight", 1.0))
    confidence = float(data.get("confidence") or 0.0)
    base_score = float(data.get("score") or confidence)
    h = hashlib.sha256((language + str(sorted(data.items()))).encode()).hexdigest()
    delta = ((int(h[:6], 16) % 100) - 50) / 5000.0
    score_delta = delta * weight
    return {"score_delta": score_delta, "signal_score_weight": weight, "base_score": base_score, "language": language, "ok": True}


def handle_regime_estimate(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    prices = data.get("prices") or data.get("returns") or []
    if len(prices) >= 2 and not isinstance(prices[0], (int, float)):
        prices = []
    if len(prices) >= 3:
        rets = [(float(prices[i]) - float(prices[i - 1])) / float(prices[i - 1]) if prices[i - 1] else 0.0 for i in range(1, len(prices))]
        vol = math.sqrt(sum(r * r for r in rets) / len(rets) * 252 * 1e4) if rets else 10.0
        trend = (float(prices[-1]) - float(prices[0])) / float(prices[0]) if prices[0] else 0.0
        regime = "high_vol" if vol > 20.0 else ("trend" if abs(trend) > 0.02 else "mean_revert")
        confidence = min(0.95, 0.5 + abs(trend) * 5 + vol / 100)
    else:
        regime, confidence = "mean_revert", 0.5
    weight = float(get_profile(language).get("volatility_weight", 1.0))
    return {"regime": regime, "confidence": confidence, "language": language, "regime_weight": weight, "ok": True}


def handle_slippage_estimate(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    profile = get_profile(language)
    spread_mult = float(profile.get("spread_mult", 1.0))
    ob = data.get("order_book") or {}
    bids = ob.get("bids") or data.get("bids") or []
    asks = ob.get("asks") or data.get("asks") or []
    quantity = float(data.get("quantity") or 0.0)
    participation = float(data.get("participation_rate") or 0.01)
    if not bids and not asks:
        return {"slippage_bps": 0.0, "language": language, "ok": True}
    best_bid = float(bids[0][0]) if bids else 0.0
    best_ask = float(asks[0][0]) if asks else 0.0
    mid = (best_bid + best_ask) / 2.0 if (best_bid and best_ask) else 0.0
    half_spread_bps = (best_ask - best_bid) / mid * 1e4 / 2 if mid else 5.0
    slippage_bps = half_spread_bps * spread_mult * (1.0 + participation * 10)
    return {"slippage_bps": slippage_bps, "language": language, "ok": True}


def handle_position_sizing(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    profile = get_profile(language)
    capital = float(data.get("capital") or 1.0)
    volatility_bps = float(data.get("volatility_bps") or data.get("volatility_annual_bps") or 10.0)
    confidence = float(data.get("confidence") or 0.5)
    max_risk_pct = float(data.get("max_risk_pct") or 0.02)
    risk_max = profile.get("risk_max_ratio", 0.45)
    size_pct = min(risk_max, max_risk_pct * (volatility_bps / 10.0) * (0.5 + confidence))
    return {"size_pct": size_pct, "size_abs": size_pct * capital, "language": language, "ok": True}


def handle_drawdown_check(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    profile = get_profile(language)
    max_dd = float(profile.get("risk_max_ratio", 0.45))
    max_drawdown_pct = float(data.get("max_drawdown_pct") or 0.12)
    current = float(data.get("current_equity") or 0.0)
    peak = float(data.get("peak_equity") or current or 1.0)
    current_drawdown_pct = (peak - current) / peak if peak else 0.0
    passed = current_drawdown_pct <= max_drawdown_pct * max_dd
    return {"passed": passed, "current_drawdown_pct": current_drawdown_pct, "language": language, "ok": True}


def handle_correlation_estimate(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    a = data.get("series_a") or data.get("returns_a") or []
    b = data.get("series_b") or data.get("returns_b") or []
    if len(a) != len(b) or len(a) < 2:
        return {"correlation": 0.0, "language": language, "ok": True}
    n = len(a)
    ma = sum(float(x) for x in a) / n
    mb = sum(float(x) for x in b) / n
    va = sum((float(a[i]) - ma) ** 2 for i in range(n))
    vb = sum((float(b[i]) - mb) ** 2 for i in range(n))
    cov = sum((float(a[i]) - ma) * (float(b[i]) - mb) for i in range(n))
    den = math.sqrt(va * vb)
    correlation = max(-1.0, min(1.0, cov / den if den else 0.0))
    return {"correlation": correlation, "language": language, "ok": True}


def handle_liquidity_score(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    bids = data.get("bids") or []
    asks = data.get("asks") or []
    depth_levels = int(data.get("depth_levels") or 5)
    if not bids and not asks:
        return {"liquidity_score": 0.0, "depth_bps": 0.0, "language": language, "ok": True}
    best_bid = float(bids[0][0]) if bids else 0.0
    best_ask = float(asks[0][0]) if asks else 0.0
    mid = (best_bid + best_ask) / 2.0 if (best_bid and best_ask) else 0.0
    depth_bps = (best_ask - best_bid) / mid * 1e4 if mid else 100.0
    total_vol = sum(float(b[1]) for b in bids[:depth_levels]) + sum(float(a[1]) for a in asks[:depth_levels])
    liquidity_score = min(1.0, total_vol / 100.0) if total_vol else 0.0
    return {"liquidity_score": liquidity_score, "depth_bps": depth_bps, "language": language, "ok": True}


def handle_market_impact(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    quantity = float(data.get("quantity") or 0.0)
    adv = float(data.get("adv") or 1.0)
    volatility = float(data.get("volatility") or 0.01)
    participation = quantity / adv if adv else 0.0
    impact_bps = 10.0 * math.sqrt(participation) * volatility * 1e4
    return {"impact_bps": impact_bps, "language": language, "ok": True}


def handle_signal_filter(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    sig = data.get("signal") if isinstance(data.get("signal"), dict) else data
    confidence = float(sig.get("confidence", data.get("confidence", 0.0)))
    regime = str(data.get("regime") or "mean_revert")
    volatility = float(data.get("volatility") or 0.01)
    accept = confidence >= 0.5 and (regime != "high_vol" or volatility < 0.02)
    seed = sum(ord(c) for c in language) % 5
    if seed == 0 and confidence < 0.8:
        accept = False
    return {"accept": accept, "filter_reason": "" if accept else "low_confidence_or_regime", "language": language, "ok": True}


def handle_confidence_calibration(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    confs = data.get("historical_confidences") or []
    pnls = data.get("historical_pnl") or []
    if len(confs) != len(pnls) or len(confs) < 2:
        return {"calibrated_confidence": 0.5, "language": language, "ok": True}
    wins = sum(1 for p in pnls if float(p) > 0)
    avg_conf = sum(float(c) for c in confs) / len(confs)
    win_rate = wins / len(pnls)
    calibrated_confidence = min(1.0, max(0.0, 0.5 * avg_conf + 0.5 * win_rate))
    return {"calibrated_confidence": calibrated_confidence, "language": language, "ok": True}


def handle_heartbeat(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    return {"ok": True, "latency_ms": 0.0, "language": language, "cycle_id": data.get("cycle_id", 0)}


def handle_var_estimate(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    returns = data.get("returns") or []
    confidence_level = float(data.get("confidence_level") or 0.95)
    if len(returns) < 5:
        return {"var_pct": 0.0, "cvar_pct": 0.0, "language": language, "ok": True}
    arr = sorted(float(r) for r in returns)
    idx = max(0, min(int((1 - confidence_level) * len(arr)), len(arr) - 1))
    var_pct = -arr[idx] * 100.0
    cvar_pct = -sum(arr[: idx + 1]) / (idx + 1) * 100.0 if idx >= 0 else var_pct
    return {"var_pct": var_pct, "cvar_pct": cvar_pct, "language": language, "ok": True}


def handle_skew_estimate(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    returns = data.get("returns") or []
    if len(returns) < 3:
        return {"skew": 0.0, "language": language, "ok": True}
    n = len(returns)
    mean_r = sum(float(r) for r in returns) / n
    var = sum((float(r) - mean_r) ** 2 for r in returns) / n
    std = math.sqrt(var) if var else 0.0
    skew = sum((float(r) - mean_r) ** 3 for r in returns) / n / (std ** 3) if std else 0.0
    return {"skew": skew, "language": language, "ok": True}


def handle_order_book_imbalance_series(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    bids, asks = data.get("bids") or [], data.get("asks") or []
    if not bids and not asks:
        return {"imbalance_series": [], "trend": 0.0, "language": language, "ok": True}
    bid_vol = sum(float(b[1]) for b in bids[:5])
    ask_vol = sum(float(a[1]) for a in asks[:5])
    total = bid_vol + ask_vol
    imb = (bid_vol - ask_vol) / total if total else 0.0
    return {"imbalance_series": [imb], "trend": imb, "language": language, "ok": True}


def handle_execution_quality_score(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    fills = data.get("fills") or []
    decision_prices = data.get("decision_prices") or []
    if not fills or len(fills) != len(decision_prices):
        return {"score_0_1": 1.0, "avg_slippage_bps": 0.0, "language": language, "ok": True}
    slippages = []
    for f, d in zip(fills[:10], decision_prices[:10]):
        fp = float(f.get("price", f) if isinstance(f, dict) else f)
        dp = float(d)
        if dp and fp:
            slippages.append(abs(fp - dp) / dp * 1e4)
    avg_bps = sum(slippages) / len(slippages) if slippages else 0.0
    score_0_1 = max(0.0, min(1.0, 1.0 - avg_bps / 50.0))
    return {"score_0_1": score_0_1, "avg_slippage_bps": avg_bps, "language": language, "ok": True}


def handle_regime_duration(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    prices = data.get("prices") or []
    regime_history = data.get("regime_history") or []
    if len(prices) < 2:
        return {"bars_in_regime": 0, "regime_stable": False, "language": language, "ok": True}
    rets = [(float(prices[i]) - float(prices[i - 1])) / float(prices[i - 1]) for i in range(1, len(prices)) if prices[i - 1]]
    vol = math.sqrt(sum(r * r for r in rets) / len(rets) * 252 * 1e4) if rets else 10.0
    regime = "high_vol" if vol > 20.0 else "mean_revert"
    bars = len(regime_history) if regime_history else min(10, len(prices))
    return {"bars_in_regime": bars, "regime_stable": bars >= 5, "regime": regime, "language": language, "ok": True}


def execute(task_type: str, data: Dict[str, Any], language: str) -> Dict[str, Any]:
    if task_type == "order_book_processing":
        return handle_order_book(data, language)
    if task_type == "risk_calculation":
        return handle_risk(data, language)
    if task_type == "cycle_plan":
        return handle_cycle_plan(data, language)
    if task_type == "volatility_estimate":
        return handle_volatility_estimate(data, language)
    if task_type == "signal_score":
        return handle_signal_score(data, language)
    if task_type == "regime_estimate":
        return handle_regime_estimate(data, language)
    if task_type == "slippage_estimate":
        return handle_slippage_estimate(data, language)
    if task_type == "position_sizing":
        return handle_position_sizing(data, language)
    if task_type == "drawdown_check":
        return handle_drawdown_check(data, language)
    if task_type == "correlation_estimate":
        return handle_correlation_estimate(data, language)
    if task_type == "liquidity_score":
        return handle_liquidity_score(data, language)
    if task_type == "market_impact":
        return handle_market_impact(data, language)
    if task_type == "signal_filter":
        return handle_signal_filter(data, language)
    if task_type == "confidence_calibration":
        return handle_confidence_calibration(data, language)
    if task_type == "heartbeat":
        return handle_heartbeat(data, language)
    if task_type == "var_estimate":
        return handle_var_estimate(data, language)
    if task_type == "skew_estimate":
        return handle_skew_estimate(data, language)
    if task_type == "order_book_imbalance_series":
        return handle_order_book_imbalance_series(data, language)
    if task_type == "execution_quality_score":
        return handle_execution_quality_score(data, language)
    if task_type == "regime_duration":
        return handle_regime_duration(data, language)
    return {"language": language, "ok": True}
