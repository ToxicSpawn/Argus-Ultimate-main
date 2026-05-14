// rust_worker.rs — Argus Rust stdin/stdout worker
// Reads one JSON line from stdin, processes the task, writes one JSON result
// line to stdout, then loops. Never exits until stdin is closed.
//
// Dependencies (Cargo.toml):
//   serde = { version = "1", features = ["derive"] }
//   serde_json = "1"
//
// Build: rustc --edition 2021 rust_worker.rs  (or use cargo)
// Run:   echo '{"task_type":"heartbeat","data":{}}' | ./rust_worker

use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Language profile constants — Rust
// ---------------------------------------------------------------------------

const LANG: &str = "rust";
const RISK_MAX: f64 = 0.48;
const CYCLE_SCALE: f64 = 1.0;
const VOL_WEIGHT: f64 = 0.9;
const SIG_WEIGHT: f64 = 1.0;
const SPREAD_MULT: f64 = 1.0;
const _ROLE: &str = "speed";
const MIN_CONF_TO_ACCEPT: f64 = 0.5;

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct Request {
    task_type: String,
    #[serde(default)]
    data: Value,
}

#[derive(Serialize)]
struct Response {
    ok: bool,
    result: Value,
    took_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn get_f64(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().unwrap_or(0.0),
        _ => 0.0,
    }
}

fn get_f64_or(v: &Value, fallback: f64) -> f64 {
    match v {
        Value::Null => fallback,
        Value::Number(n) => {
            let f = n.as_f64().unwrap_or(0.0);
            if f == 0.0 { fallback } else { f }
        }
        _ => fallback,
    }
}

fn to_float_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(arr) => arr.iter().map(get_f64).collect(),
        _ => vec![],
    }
}

fn to_pair_vec(v: &Value) -> Vec<[f64; 2]> {
    match v {
        Value::Array(arr) => arr
            .iter()
            .filter_map(|e| {
                if let Value::Array(row) = e {
                    if row.len() >= 2 {
                        return Some([get_f64(&row[0]), get_f64(&row[1])]);
                    }
                }
                None
            })
            .collect(),
        _ => vec![],
    }
}

fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

// FNV-1a 32-bit hash
fn fnv1a(s: &str) -> u32 {
    let mut h: u32 = 2166136261;
    for b in s.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(16777619);
    }
    h
}

// SHA-256 → first 8 bytes as u64
fn sha256_int(s: &str) -> u64 {
    // Manual SHA-256 is complex; use a seeded FNV chain as a deterministic
    // substitute that produces the same statistical spread for hashing purposes.
    // For strict SHA256 parity, link with ring or sha2 crate.
    let h1 = fnv1a(s) as u64;
    let h2 = fnv1a(&format!("{}__seed2", s)) as u64;
    (h1 << 32) | h2
}

// Sorted JSON string of the data Value for hashing
fn sorted_key_string(data: &Value) -> String {
    if let Value::Object(map) = data {
        let mut keys: Vec<&String> = map.keys().collect();
        keys.sort();
        let mut buf = String::with_capacity(256);
        for k in keys {
            buf.push_str(k);
            buf.push(':');
            match &map[k] {
                Value::String(s) => buf.push_str(s),
                Value::Number(n) => buf.push_str(&n.to_string()),
                Value::Bool(b) => buf.push(if *b { '1' } else { '0' }),
                _ => buf.push('?'),
            }
            buf.push(',');
        }
        buf
    } else {
        String::new()
    }
}

fn welford_vol(returns: &[f64]) -> f64 {
    let n = returns.len();
    if n == 0 {
        return 10.0;
    }
    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    for (i, &r) in returns.iter().enumerate() {
        let delta = r - mean;
        mean += delta / (i + 1) as f64;
        m2 += delta * (r - mean);
    }
    let variance = m2 / n as f64;
    if variance <= 0.0 {
        return 10.0;
    }
    (variance * 252.0).sqrt() * 1e4
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 || b.len() != n {
        return 0.0;
    }
    let ma = a.iter().sum::<f64>() / n as f64;
    let mb = b.iter().sum::<f64>() / n as f64;
    let (mut va, mut vb, mut cov) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        va += da * da;
        vb += db * db;
        cov += da * db;
    }
    let den = (va * vb).sqrt();
    if den == 0.0 { 0.0 } else { clamp(cov / den, -1.0, 1.0) }
}

fn ob_bids(data: &Value) -> Vec<[f64; 2]> {
    let bids = to_pair_vec(&data["bids"]);
    if !bids.is_empty() {
        return bids;
    }
    if let Value::Object(ob) = &data["order_book"] {
        if let Some(b) = ob.get("bids") {
            return to_pair_vec(b);
        }
    }
    bids
}

fn ob_asks(data: &Value) -> Vec<[f64; 2]> {
    let asks = to_pair_vec(&data["asks"]);
    if !asks.is_empty() {
        return asks;
    }
    if let Value::Object(ob) = &data["order_book"] {
        if let Some(a) = ob.get("asks") {
            return to_pair_vec(a);
        }
    }
    asks
}

fn top5_vols(levels: &[[f64; 2]]) -> f64 {
    levels.iter().take(5).map(|l| l[1]).sum()
}

// ---------------------------------------------------------------------------
// Task handlers
// ---------------------------------------------------------------------------

// 1. cycle_plan
fn handle_cycle_plan(data: &Value) -> Value {
    let mut sorted_data = data.clone();
    let json_str = if let Value::Object(ref mut map) = sorted_data {
        let mut keys: Vec<String> = map.keys().cloned().collect();
        keys.sort();
        let mut ordered = serde_json::Map::new();
        for k in &keys {
            ordered.insert(k.clone(), map[k].clone());
        }
        serde_json::to_string(&Value::Object(ordered)).unwrap_or_default()
    } else {
        String::new()
    };
    let hash_int = sha256_int(&json_str);
    let base = (hash_int % 200) as f64 - 100.0;
    let base = base / 10000.0;

    let pv = get_f64_or(&data["portfolio_value_aud"], 1.0);
    let cash = get_f64(&data["cash_balance_aud"]);
    let signals = get_f64(&data["signals"]) as i64;
    let cash_ratio = if pv != 0.0 { cash / pv } else { 0.0 };
    let tilt = (cash_ratio - 0.5) * 0.002 + (signals % 3 - 1) as f64 * 0.001;
    let boost = clamp((base + tilt) * CYCLE_SCALE, -0.015, 0.015);

    json!({
        "language": LANG,
        "cycle_boost": boost,
        "ok": true
    })
}

// 2. order_book_processing
fn handle_order_book_processing(data: &Value) -> Value {
    let bids = ob_bids(data);
    let asks = ob_asks(data);

    let best_bid = bids.first().map(|l| l[0]).unwrap_or(0.0);
    let best_ask = asks.first().map(|l| l[0]).unwrap_or(0.0);
    let mid = if best_bid > 0.0 && best_ask > 0.0 { (best_bid + best_ask) / 2.0 } else { 0.0 };
    let spread_bps = if mid > 0.0 { (best_ask - best_bid) / mid * 1e4 * SPREAD_MULT } else { 0.0 };

    let bid_vol = top5_vols(&bids);
    let ask_vol = top5_vols(&asks);
    let total = bid_vol + ask_vol;
    let imbalance = if total > 0.0 { (bid_vol - ask_vol) / total } else { 0.0 };

    // ── VPIN (Volume-Synchronized Probability of Informed Trading) ──
    // Classify each level's volume as buy/sell using tick rule proxy:
    // levels above mid → sell pressure, below mid → buy pressure
    let (mut buy_vol, mut sell_vol) = (0.0f64, 0.0f64);
    for level in &bids {
        buy_vol += level[1]; // bid volume = buy interest
    }
    for level in &asks {
        sell_vol += level[1]; // ask volume = sell interest
    }
    let total_classified = buy_vol + sell_vol;
    let vpin = if total_classified > 0.0 {
        (buy_vol - sell_vol).abs() / total_classified
    } else {
        0.0
    };

    // ── Order Flow Toxicity Score (0 = safe, 1 = toxic) ──
    // Combines VPIN with spread widening and depth asymmetry
    let depth_ratio = if ask_vol > 0.0 { bid_vol / ask_vol } else { 1.0 };
    let depth_asym = (depth_ratio - 1.0).abs().min(2.0) / 2.0; // 0-1
    let spread_score = (spread_bps / 50.0).min(1.0); // wide spread = toxic
    let toxicity = clamp(vpin * 0.4 + depth_asym * 0.3 + spread_score * 0.3, 0.0, 1.0);

    // ── Spoofing Detection (heuristic) ──
    // Large volume at distant levels with thin near-levels suggests spoofing
    let near_bid_vol: f64 = bids.iter().take(3).map(|l| l[1]).sum();
    let far_bid_vol: f64 = bids.iter().skip(3).map(|l| l[1]).sum();
    let near_ask_vol: f64 = asks.iter().take(3).map(|l| l[1]).sum();
    let far_ask_vol: f64 = asks.iter().skip(3).map(|l| l[1]).sum();
    let bid_spoof_ratio = if near_bid_vol > 0.0 { far_bid_vol / near_bid_vol } else { 0.0 };
    let ask_spoof_ratio = if near_ask_vol > 0.0 { far_ask_vol / near_ask_vol } else { 0.0 };
    // Spoof flag if far volume is >5x near volume on either side
    let spoof_detected = bid_spoof_ratio > 5.0 || ask_spoof_ratio > 5.0;

    // ── Iceberg Detection (hidden liquidity proxy) ──
    // Uniformly small displayed sizes across many levels suggests icebergs
    let bid_sizes: Vec<f64> = bids.iter().take(10).map(|l| l[1]).collect();
    let iceberg_score = if bid_sizes.len() >= 5 {
        let mean_sz = bid_sizes.iter().sum::<f64>() / bid_sizes.len() as f64;
        let var = bid_sizes.iter().map(|s| (s - mean_sz).powi(2)).sum::<f64>() / bid_sizes.len() as f64;
        let cv = if mean_sz > 0.0 { var.sqrt() / mean_sz } else { 1.0 };
        // Low coefficient of variation = suspiciously uniform = possible icebergs
        clamp(1.0 - cv.min(1.0), 0.0, 1.0)
    } else {
        0.0
    };

    // ── Kyle's Lambda (price impact coefficient) ──
    // lambda = spread / (2 * depth) — higher means more price impact per unit traded
    let total_depth = bid_vol + ask_vol;
    let kyles_lambda = if total_depth > 0.0 && mid > 0.0 {
        (best_ask - best_bid) / (2.0 * total_depth)
    } else {
        0.0
    };

    json!({
        "spread_bps": spread_bps,
        "imbalance": imbalance,
        "mid": mid,
        "vpin": vpin,
        "toxicity": toxicity,
        "spoof_detected": spoof_detected,
        "bid_spoof_ratio": bid_spoof_ratio,
        "ask_spoof_ratio": ask_spoof_ratio,
        "iceberg_score": iceberg_score,
        "kyles_lambda": kyles_lambda,
        "depth_ratio": depth_ratio,
        "language": LANG
    })
}

// 3. risk_calculation
fn handle_risk_calculation(data: &Value) -> Value {
    let pv = get_f64(&data["position_value"]);
    let capital = get_f64_or(&data["capital"], 1.0);
    let ratio = if capital != 0.0 { pv / capital } else { 0.0 };
    let passed = ratio <= RISK_MAX;

    json!({
        "passed": passed,
        "exposure_ratio": ratio,
        "max_ratio": RISK_MAX,
        "language": LANG
    })
}

// 4. volatility_estimate
fn handle_volatility_estimate(data: &Value) -> Value {
    let returns = to_float_vec(&data["returns"]);
    let prices = if data["prices"].is_array() {
        to_float_vec(&data["prices"])
    } else {
        to_float_vec(&data["ohlcv_close"])
    };

    let vol = if !returns.is_empty() {
        welford_vol(&returns)
    } else if prices.len() >= 2 {
        let rets: Vec<f64> = prices
            .windows(2)
            .filter(|w| w[0] != 0.0)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        welford_vol(&rets)
    } else {
        10.0
    };

    json!({
        "volatility_annual_bps": vol * VOL_WEIGHT,
        "volatility_weight": VOL_WEIGHT,
        "language": LANG,
        "ok": true
    })
}

// 5. signal_score
fn handle_signal_score(data: &Value) -> Value {
    let key = format!("{}{}", LANG, sorted_key_string(data));
    let h = fnv1a(&key);
    let delta = (h % 100) as f64 - 50.0;
    let score_delta = delta / 5000.0 * SIG_WEIGHT;

    json!({
        "score_delta": score_delta,
        "signal_score_weight": SIG_WEIGHT,
        "language": LANG,
        "ok": true
    })
}

// 6. regime_estimate
fn handle_regime_estimate(data: &Value) -> Value {
    let prices = if data["prices"].is_array() {
        to_float_vec(&data["prices"])
    } else {
        to_float_vec(&data["returns"])
    };

    let (regime, confidence) = if prices.len() >= 3 {
        let rets: Vec<f64> = prices
            .windows(2)
            .filter(|w| w[0] != 0.0)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        let vol = welford_vol(&rets);
        let trend = if prices[0] != 0.0 {
            (prices[prices.len() - 1] - prices[0]) / prices[0]
        } else {
            0.0
        };
        let regime = if vol > 20.0 {
            "high_vol"
        } else if trend.abs() > 0.02 {
            "trend"
        } else {
            "mean_revert"
        };
        let conf = (0.5_f64 + trend.abs() * 5.0 + vol / 100.0).min(0.95);
        (regime, conf)
    } else {
        ("mean_revert", 0.5)
    };

    json!({
        "regime": regime,
        "confidence": confidence,
        "regime_weight": 1.0,
        "language": LANG,
        "ok": true
    })
}

// 7. slippage_estimate
fn handle_slippage_estimate(data: &Value) -> Value {
    let bids = ob_bids(data);
    let asks = ob_asks(data);
    let participation = get_f64_or(&data["participation_rate"], 0.01);

    let best_bid = bids.first().map(|l| l[0]).unwrap_or(0.0);
    let best_ask = asks.first().map(|l| l[0]).unwrap_or(0.0);
    let mid = if best_bid > 0.0 && best_ask > 0.0 { (best_bid + best_ask) / 2.0 } else { 0.0 };

    let half_spread_bps = if mid > 0.0 {
        (best_ask - best_bid) / mid * 1e4 / 2.0
    } else {
        5.0
    };
    let slippage_bps = half_spread_bps * SPREAD_MULT * (1.0 + participation * 10.0);

    json!({
        "slippage_bps": slippage_bps,
        "language": LANG,
        "ok": true
    })
}

// 8. position_sizing
fn handle_position_sizing(data: &Value) -> Value {
    let capital = get_f64_or(&data["capital"], 1.0);
    let vol_bps = {
        let v = get_f64(&data["volatility_bps"]);
        if v != 0.0 { v } else { get_f64_or(&data["volatility_annual_bps"], 10.0) }
    };
    let confidence = get_f64_or(&data["confidence"], 0.5);
    let max_risk_pct = get_f64_or(&data["max_risk_pct"], 0.02);

    let size_pct = (max_risk_pct * (vol_bps / 10.0) * (0.5 + confidence)).min(RISK_MAX);

    json!({
        "size_pct": size_pct,
        "size_abs": size_pct * capital,
        "language": LANG,
        "ok": true
    })
}

// 9. drawdown_check
fn handle_drawdown_check(data: &Value) -> Value {
    let current = get_f64(&data["current_equity"]);
    let peak = get_f64_or(&data["peak_equity"], current.max(1.0));
    let max_dd = get_f64_or(&data["max_drawdown_pct"], 0.12);

    let dd = if peak > 0.0 { (peak - current) / peak } else { 0.0 };
    let passed = dd <= max_dd * RISK_MAX;

    json!({
        "passed": passed,
        "current_drawdown_pct": dd,
        "language": LANG,
        "ok": true
    })
}

// 10. correlation_estimate
fn handle_correlation_estimate(data: &Value) -> Value {
    let a = if data["series_a"].is_array() {
        to_float_vec(&data["series_a"])
    } else {
        to_float_vec(&data["returns_a"])
    };
    let b = if data["series_b"].is_array() {
        to_float_vec(&data["series_b"])
    } else {
        to_float_vec(&data["returns_b"])
    };

    json!({
        "correlation": pearson(&a, &b),
        "language": LANG,
        "ok": true
    })
}

// 11. liquidity_score
fn handle_liquidity_score(data: &Value) -> Value {
    let bids = ob_bids(data);
    let asks = ob_asks(data);

    let best_bid = bids.first().map(|l| l[0]).unwrap_or(0.0);
    let best_ask = asks.first().map(|l| l[0]).unwrap_or(0.0);
    let mid = if best_bid > 0.0 && best_ask > 0.0 { (best_bid + best_ask) / 2.0 } else { 0.0 };
    let depth_bps = if mid > 0.0 { (best_ask - best_bid) / mid * 1e4 } else { 100.0 };

    let total_vol = top5_vols(&bids) + top5_vols(&asks);
    let score = (total_vol / 100.0).min(1.0);

    json!({
        "liquidity_score": score,
        "depth_bps": depth_bps,
        "language": LANG,
        "ok": true
    })
}

// 12. market_impact
fn handle_market_impact(data: &Value) -> Value {
    let quantity = get_f64(&data["quantity"]);
    let adv = get_f64_or(&data["adv"], 1.0);
    let volatility = get_f64_or(&data["volatility"], 0.01);
    let participation = if adv > 0.0 { quantity / adv } else { 0.0 };
    let impact_bps = 10.0 * participation.sqrt() * volatility * 1e4;

    json!({
        "impact_bps": impact_bps,
        "language": LANG,
        "ok": true
    })
}

// 13. signal_filter
fn handle_signal_filter(data: &Value) -> Value {
    let confidence = {
        let c = get_f64(&data["confidence"]);
        if c != 0.0 {
            c
        } else if let Value::Object(sig) = &data["signal"] {
            sig.get("confidence").map(get_f64).unwrap_or(0.0)
        } else {
            0.0
        }
    };
    let regime = data["regime"].as_str().unwrap_or("mean_revert");
    let volatility = get_f64(&data["volatility"]);

    let accept = confidence >= MIN_CONF_TO_ACCEPT && !(regime == "high_vol" && volatility >= 0.02);
    let filter_reason = if accept { "" } else { "low_confidence_or_regime" };

    json!({
        "accept": accept,
        "filter_reason": filter_reason,
        "language": LANG,
        "ok": true
    })
}

// 14. confidence_calibration
fn handle_confidence_calibration(data: &Value) -> Value {
    let confs = to_float_vec(&data["historical_confidences"]);
    let pnls = to_float_vec(&data["historical_pnl"]);

    if confs.is_empty() || confs.len() != pnls.len() {
        return json!({ "calibrated_confidence": 0.5, "language": LANG, "ok": true });
    }
    let wins = pnls.iter().filter(|&&p| p > 0.0).count();
    let avg_conf = confs.iter().sum::<f64>() / confs.len() as f64;
    let win_rate = wins as f64 / pnls.len() as f64;
    let calibrated = clamp(avg_conf * 0.5 + win_rate * 0.5, 0.0, 1.0);

    json!({
        "calibrated_confidence": calibrated,
        "language": LANG,
        "ok": true
    })
}

// 15. heartbeat
fn handle_heartbeat(data: &Value) -> Value {
    let cycle_id = &data["cycle_id"];
    json!({
        "ok": true,
        "latency_ms": 0.0,
        "language": LANG,
        "cycle_id": cycle_id
    })
}

// 16. var_estimate
fn handle_var_estimate(data: &Value) -> Value {
    let returns = to_float_vec(&data["returns"]);
    let confidence_level = get_f64_or(&data["confidence_level"], 0.95);

    if returns.len() < 5 {
        return json!({ "var_pct": 0.0, "cvar_pct": 0.0, "language": LANG, "ok": true });
    }

    let mut sorted = returns.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let idx = ((1.0 - confidence_level) * sorted.len() as f64) as usize;
    let idx = idx.min(sorted.len() - 1);
    let var_pct = -sorted[idx] * 100.0;

    let cnt = idx + 1;
    let cvar_pct = -sorted[..cnt].iter().sum::<f64>() / cnt as f64 * 100.0;

    json!({
        "var_pct": var_pct,
        "cvar_pct": cvar_pct,
        "language": LANG,
        "ok": true
    })
}

// 17. skew_estimate
fn handle_skew_estimate(data: &Value) -> Value {
    let returns = to_float_vec(&data["returns"]);
    if returns.len() < 3 {
        return json!({ "skew": 0.0, "language": LANG, "ok": true });
    }
    let n = returns.len() as f64;
    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    for (i, &r) in returns.iter().enumerate() {
        let delta = r - mean;
        mean += delta / (i + 1) as f64;
        m2 += delta * (r - mean);
    }
    let variance = m2 / n;
    let std = variance.sqrt();
    if std == 0.0 {
        return json!({ "skew": 0.0, "language": LANG, "ok": true });
    }
    let m3: f64 = returns.iter().map(|&r| { let d = (r - mean) / std; d * d * d }).sum::<f64>() / n;

    json!({
        "skew": m3,
        "language": LANG,
        "ok": true
    })
}

// 18. order_book_imbalance_series
fn handle_order_book_imbalance_series(data: &Value) -> Value {
    let bids = ob_bids(data);
    let asks = ob_asks(data);

    let bid_vol = top5_vols(&bids);
    let ask_vol = top5_vols(&asks);
    let total = bid_vol + ask_vol;
    let imb = if total > 0.0 { (bid_vol - ask_vol) / total } else { 0.0 };

    json!({
        "imbalance_series": [imb],
        "trend": imb,
        "language": LANG,
        "ok": true
    })
}

// 19. execution_quality_score
fn handle_execution_quality_score(data: &Value) -> Value {
    let fills = match &data["fills"] {
        Value::Array(a) => a.clone(),
        _ => vec![],
    };
    let decisions = match &data["decision_prices"] {
        Value::Array(a) => a.clone(),
        _ => vec![],
    };

    if fills.is_empty() || fills.len() != decisions.len() {
        return json!({ "score_0_1": 1.0, "avg_slippage_bps": 0.0, "language": LANG, "ok": true });
    }

    let mut total_slip = 0.0f64;
    let mut count = 0usize;
    for (f, d) in fills.iter().zip(decisions.iter()) {
        let fp = match f {
            Value::Object(m) => m.get("price").map(get_f64).unwrap_or(0.0),
            _ => get_f64(f),
        };
        let dp = get_f64(d);
        if fp != 0.0 && dp != 0.0 {
            total_slip += (fp - dp).abs() / dp * 1e4;
            count += 1;
        }
    }
    let avg_bps = if count > 0 { total_slip / count as f64 } else { 0.0 };
    let score = clamp(1.0 - avg_bps / 50.0, 0.0, 1.0);

    json!({
        "score_0_1": score,
        "avg_slippage_bps": avg_bps,
        "language": LANG,
        "ok": true
    })
}

// 20. regime_duration
fn handle_regime_duration(data: &Value) -> Value {
    let prices = to_float_vec(&data["prices"]);
    let history_len = match &data["regime_history"] {
        Value::Array(a) => a.len(),
        _ => 0,
    };

    let regime = if prices.len() >= 2 {
        let rets: Vec<f64> = prices
            .windows(2)
            .filter(|w| w[0] != 0.0)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        let vol = welford_vol(&rets);
        if vol > 20.0 { "high_vol" } else { "mean_revert" }
    } else {
        "mean_revert"
    };

    let bars = if history_len > 0 { history_len } else { prices.len().min(10) };

    json!({
        "bars_in_regime": bars,
        "regime_stable": bars >= 5,
        "regime": regime,
        "language": LANG,
        "ok": true
    })
}

// 21. vpin — Volume-Synchronized Probability of Informed Trading
fn handle_vpin(data: &Value) -> Value {
    let trade_volumes = to_float_vec(&data["trade_volumes"]);
    let bucket_size = get_f64_or(&data["bucket_size"], 50.0);

    if trade_volumes.is_empty() || bucket_size <= 0.0 {
        return json!({
            "vpin": 0.0,
            "toxicity": "low",
            "n_buckets": 0,
            "language": LANG
        });
    }

    // Group trades into volume buckets
    let mut buckets: Vec<f64> = Vec::new();
    let mut bucket_buy: f64 = 0.0;
    let mut bucket_sell: f64 = 0.0;
    let mut bucket_vol: f64 = 0.0;

    for &vol in &trade_volumes {
        let abs_vol = vol.abs();
        if vol >= 0.0 {
            bucket_buy += abs_vol;
        } else {
            bucket_sell += abs_vol;
        }
        bucket_vol += abs_vol;

        while bucket_vol >= bucket_size {
            // fraction of this trade that fills the current bucket
            let overflow = bucket_vol - bucket_size;
            // ratio of order imbalance for this bucket
            let imbalance = (bucket_buy - bucket_sell).abs() / bucket_size;
            buckets.push(clamp(imbalance, 0.0, 1.0));

            // Start new bucket with overflow
            if overflow > 0.0 {
                // The overflow retains the direction of the last trade
                if vol >= 0.0 {
                    bucket_buy = overflow;
                    bucket_sell = 0.0;
                } else {
                    bucket_buy = 0.0;
                    bucket_sell = overflow;
                }
                bucket_vol = overflow;
            } else {
                bucket_buy = 0.0;
                bucket_sell = 0.0;
                bucket_vol = 0.0;
            }
        }
    }

    let n_buckets = buckets.len();

    // VPIN = rolling mean of last 50 bucket ratios
    let window = 50.min(n_buckets);
    let vpin = if window > 0 {
        let start = n_buckets - window;
        buckets[start..].iter().sum::<f64>() / window as f64
    } else {
        0.0
    };
    let vpin = clamp(vpin, 0.0, 1.0);

    let toxicity = if vpin < 0.3 {
        "low"
    } else if vpin <= 0.6 {
        "medium"
    } else {
        "high"
    };

    json!({
        "vpin": vpin,
        "toxicity": toxicity,
        "n_buckets": n_buckets,
        "language": LANG
    })
}

// 22. microstructure_analysis — Combined microstructure analysis
fn handle_microstructure_analysis(data: &Value) -> Value {
    let bids = ob_bids(data);
    let asks = ob_asks(data);

    let best_bid = bids.first().map(|l| l[0]).unwrap_or(0.0);
    let best_ask = asks.first().map(|l| l[0]).unwrap_or(0.0);
    let mid = if best_bid > 0.0 && best_ask > 0.0 {
        (best_bid + best_ask) / 2.0
    } else {
        0.0
    };
    let spread_bps = if mid > 0.0 {
        (best_ask - best_bid) / mid * 1e4
    } else {
        0.0
    };

    // --- kyle_lambda: price impact coefficient ---
    // Use recent_trades to compute |delta_price| / |net_order_flow| averaged over chunks
    let kyle_lambda = {
        let trades = match &data["recent_trades"] {
            Value::Array(arr) => arr.clone(),
            _ => vec![],
        };
        if trades.len() >= 2 {
            let chunk_size = 5.max(1);
            let mut lambdas: Vec<f64> = Vec::new();
            let chunks: Vec<&[Value]> = trades.chunks(chunk_size).collect();
            for chunk in &chunks {
                if chunk.len() < 2 {
                    continue;
                }
                let first_price = chunk.first().map(|t| get_f64(&t["price"])).unwrap_or(0.0);
                let last_price = chunk.last().map(|t| get_f64(&t["price"])).unwrap_or(0.0);
                let delta_price = (last_price - first_price).abs();
                let net_flow: f64 = chunk.iter().map(|t| {
                    let vol = get_f64(&t["volume"]);
                    let side = t["side"].as_str().unwrap_or("buy");
                    if side == "sell" { -vol } else { vol }
                }).sum();
                let abs_flow = net_flow.abs();
                if abs_flow > 0.0 {
                    lambdas.push(delta_price / abs_flow);
                }
            }
            if !lambdas.is_empty() {
                lambdas.iter().sum::<f64>() / lambdas.len() as f64
            } else {
                0.0
            }
        } else {
            0.0
        }
    };

    // --- depth_imbalance: weighted bid/ask depth ratio at top 10 levels ---
    let depth_imbalance = {
        let n = 10.min(bids.len()).min(asks.len());
        if n > 0 {
            let mut weighted_bid = 0.0f64;
            let mut weighted_ask = 0.0f64;
            for i in 0..n {
                let w = 1.0 / (i as f64 + 1.0);
                weighted_bid += bids[i][1] * w;
                weighted_ask += asks[i][1] * w;
            }
            let total = weighted_bid + weighted_ask;
            if total > 0.0 {
                weighted_bid / total
            } else {
                0.5
            }
        } else {
            0.5
        }
    };

    // --- spread_volatility: std dev of spread over snapshots ---
    let spread_volatility_bps = {
        let spread_history = to_float_vec(&data["spread_history"]);
        if spread_history.len() >= 2 {
            let n = spread_history.len() as f64;
            let mean = spread_history.iter().sum::<f64>() / n;
            let variance = spread_history.iter().map(|&s| (s - mean) * (s - mean)).sum::<f64>() / n;
            variance.sqrt()
        } else {
            0.0
        }
    };

    // --- hidden_liquidity_ratio: detect potential icebergs ---
    let hidden_liquidity_ratio = {
        let all_levels: Vec<f64> = bids.iter().chain(asks.iter()).map(|l| l[1]).collect();
        if all_levels.len() >= 2 {
            let avg_vol = all_levels.iter().sum::<f64>() / all_levels.len() as f64;
            let threshold = avg_vol * 3.0;
            let iceberg_count = all_levels.iter().filter(|&&v| v > threshold).count();
            iceberg_count as f64 / all_levels.len() as f64
        } else {
            0.0
        }
    };

    // --- toxicity_score composite: 0.4*norm_kyle + 0.3*(1-depth_balance) + 0.3*spread_vol_norm ---
    let norm_kyle = clamp(kyle_lambda / (kyle_lambda + 1.0), 0.0, 1.0);
    // depth_imbalance is bid_weight / total; perfectly balanced = 0.5
    // deviation from 0.5 indicates imbalance
    let depth_balance = 1.0 - (depth_imbalance - 0.5).abs() * 2.0;
    let spread_vol_normalized = clamp(spread_volatility_bps / (spread_volatility_bps + 10.0), 0.0, 1.0);

    let toxicity_score = clamp(
        0.4 * norm_kyle + 0.3 * (1.0 - depth_balance) + 0.3 * spread_vol_normalized,
        0.0,
        1.0,
    );

    json!({
        "kyle_lambda": kyle_lambda,
        "depth_imbalance": depth_imbalance,
        "spread_volatility_bps": spread_volatility_bps,
        "hidden_liquidity_ratio": hidden_liquidity_ratio,
        "toxicity_score": toxicity_score,
        "language": LANG
    })
}

// 23. spoofing_detection — Detect potential spoofing patterns
fn handle_spoofing_detection(data: &Value) -> Value {
    // Support array of snapshots or single snapshot
    let snapshots = match &data["order_book_snapshots"] {
        Value::Array(arr) if !arr.is_empty() => arr.clone(),
        _ => {
            // Treat current bids/asks as a single snapshot
            vec![data.clone()]
        }
    };

    // Analyse the last snapshot
    let snap = snapshots.last().unwrap();
    let bids = ob_bids(snap);
    let asks = ob_asks(snap);

    // Compute volume asymmetry for one side:
    // ratio = avg(top 3 volume) / avg(levels 4-10 volume)
    let side_asymmetry = |levels: &[[f64; 2]]| -> f64 {
        let top3: Vec<f64> = levels.iter().take(3).map(|l| l[1]).collect();
        let rest: Vec<f64> = levels.iter().skip(3).take(7).map(|l| l[1]).collect();
        if top3.is_empty() || rest.is_empty() {
            return 1.0;
        }
        let avg_top = top3.iter().sum::<f64>() / top3.len() as f64;
        let avg_rest = rest.iter().sum::<f64>() / rest.len() as f64;
        if avg_rest > 0.0 {
            avg_top / avg_rest
        } else {
            1.0
        }
    };

    let bid_asym = side_asymmetry(&bids);
    let ask_asym = side_asymmetry(&asks);

    // Spoofing threshold: top 3 levels have >5x the volume of levels 4-10
    let spoof_threshold = 5.0;
    let bid_spoof = bid_asym > spoof_threshold;
    let ask_spoof = ask_asym > spoof_threshold;

    let side = if bid_spoof && ask_spoof {
        // Both sides — pick the more extreme one
        if bid_asym >= ask_asym { "bid" } else { "ask" }
    } else if bid_spoof {
        "bid"
    } else if ask_spoof {
        "ask"
    } else {
        "none"
    };

    // Spoof score: normalized measure of how extreme the asymmetry is
    let max_asym = bid_asym.max(ask_asym);
    let spoof_score = if max_asym <= 1.0 {
        0.0
    } else {
        // Map asymmetry into 0..1 using a sigmoid-like scaling
        // At 5x -> ~0.5, at 10x -> ~0.75, at 20x -> ~0.875
        clamp(1.0 - 1.0 / (max_asym / spoof_threshold), 0.0, 1.0)
    };

    json!({
        "spoof_score": spoof_score,
        "side": side,
        "volume_asymmetry_bid": bid_asym,
        "volume_asymmetry_ask": ask_asym,
        "language": LANG
    })
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

fn dispatch(task_type: &str, data: &Value) -> Value {
    match task_type {
        "cycle_plan"                  => handle_cycle_plan(data),
        "order_book_processing"       => handle_order_book_processing(data),
        "risk_calculation"            => handle_risk_calculation(data),
        "volatility_estimate"         => handle_volatility_estimate(data),
        "signal_score"                => handle_signal_score(data),
        "regime_estimate"             => handle_regime_estimate(data),
        "slippage_estimate"           => handle_slippage_estimate(data),
        "position_sizing"             => handle_position_sizing(data),
        "drawdown_check"              => handle_drawdown_check(data),
        "correlation_estimate"        => handle_correlation_estimate(data),
        "liquidity_score"             => handle_liquidity_score(data),
        "market_impact"               => handle_market_impact(data),
        "signal_filter"               => handle_signal_filter(data),
        "confidence_calibration"      => handle_confidence_calibration(data),
        "heartbeat"                   => handle_heartbeat(data),
        "var_estimate"                => handle_var_estimate(data),
        "skew_estimate"               => handle_skew_estimate(data),
        "order_book_imbalance_series" => handle_order_book_imbalance_series(data),
        "execution_quality_score"     => handle_execution_quality_score(data),
        "regime_duration"             => handle_regime_duration(data),
        "vpin"                        => handle_vpin(data),
        "microstructure_analysis"     => handle_microstructure_analysis(data),
        "spoofing_detection"          => handle_spoofing_detection(data),
        _ => json!({ "error": format!("unknown task_type: {}", task_type) }),
    }
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let start = Instant::now();

        let resp: Response = match serde_json::from_str::<Request>(&line) {
            Err(e) => Response {
                ok: false,
                result: json!({}),
                took_ms: 0.0,
                error: Some(format!("json parse error: {}", e)),
            },
            Ok(req) => {
                let data = if req.data.is_null() { json!({}) } else { req.data };
                let result = std::panic::catch_unwind(|| dispatch(&req.task_type, &data));
                let took_ms = start.elapsed().as_secs_f64() * 1000.0;
                match result {
                    Ok(r) => {
                        if r.get("error").is_some() && r.get("ok").is_none() {
                            Response {
                                ok: false,
                                result: json!({}),
                                took_ms,
                                error: r.get("error").and_then(|v| v.as_str()).map(String::from),
                            }
                        } else {
                            Response { ok: true, result: r, took_ms, error: None }
                        }
                    }
                    Err(e) => Response {
                        ok: false,
                        result: json!({}),
                        took_ms,
                        error: Some(format!("panic: {:?}", e)),
                    },
                }
            }
        };

        let json_out = serde_json::to_string(&resp).unwrap_or_else(|_| {
            r#"{"ok":false,"result":{},"took_ms":0.0,"error":"serialization error"}"#.to_string()
        });
        writeln!(out, "{}", json_out).ok();
        out.flush().ok();
    }
}
