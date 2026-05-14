use serde_json::{json, Value};
use std::error::Error;

pub fn dispatch(task_type: &str, data: &Value) -> Result<Value, String> {
    match task_type {
        "order_book" => execute_order_book(data),
        "risk" => execute_risk(data),
        "volatility" => execute_volatility(data),
        "regime" => execute_regime(data),
        "var_estimate" => execute_var_estimate(data),
        "cycle_plan" => execute_cycle_plan(data),
        _ => Err(format!("Unsupported task_type: {}", task_type)),
    }
}

fn execute_order_book(data: &Value) -> Result<Value, String> {
    let spread_bps: f64 = data["spread_bps"].as_f64().unwrap_or(10.0);
    let imbalance: f64 = data["imbalance"].as_f64().unwrap_or(0.0);
    let depth_ratio: f64 = data["depth_ratio"].as_f64().unwrap_or(1.0);

    // Simple order book simulation
    let spread = spread_bps / 10000.0;
    let toxicity = (imbalance.abs() * 0.5 + (1.0 - depth_ratio) * 0.5).min(1.0);

    Ok(json!({
        "spread_bps": spread_bps,
        "imbalance": imbalance,
        "depth_ratio": depth_ratio,
        "toxicity": toxicity,
        "vpin": imbalance.abs(),
        "spoof_detected": toxicity > 0.7,
        "language": "rust",
    }))
}

fn execute_risk(data: &Value) -> Result<Value, String> {
    let portfolio_value: f64 = data["portfolio_value_aud"].as_f64().unwrap_or(1000.0);
    let cash_balance: f64 = data["cash_balance_aud"].as_f64().unwrap_or(500.0);
    let max_position_ratio: f64 = data["max_position_ratio"].as_f64().unwrap_or(0.5);

    let position_size = cash_balance * max_position_ratio;
    let risk_ratio = position_size / portfolio_value;

    Ok(json!({
        "risk_ratio": risk_ratio,
        "max_position_size_aud": position_size,
        "pass": risk_ratio <= 0.48, // Rust profile risk_max_ratio
        "language": "rust",
    }))
}

fn execute_volatility(data: &Value) -> Result<Value, String> {
    let returns: Vec<f64> = data["returns"]
        .as_array()
        .ok_or("returns must be an array")?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0))
        .collect();

    if returns.is_empty() {
        return Err("returns array is empty".into());
    }

    let sum: f64 = returns.iter().sum();
    let mean = sum / returns.len() as f64;
    let variance: f64 = returns.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    let volatility = variance.sqrt();

    Ok(json!({
        "volatility": volatility,
        "mean_return": mean,
        "language": "rust",
    }))
}

fn execute_regime(data: &Value) -> Result<Value, String> {
    let returns: Vec<f64> = data["returns"]
        .as_array()
        .ok_or("returns must be an array")?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0))
        .collect();

    if returns.is_empty() {
        return Err("returns array is empty".into());
    }

    let sum: f64 = returns.iter().sum();
    let mean = sum / returns.len() as f64;

    let regime = if mean > 0.001 {
        "bull"
    } else if mean < -0.001 {
        "bear"
    } else {
        "sideways"
    };

    Ok(json!({
        "regime": regime,
        "regime_confidence": 0.8,
        "language": "rust",
    }))
}

fn execute_var_estimate(data: &Value) -> Result<Value, String> {
    let returns: Vec<f64> = data["returns"]
        .as_array()
        .ok_or("returns must be an array")?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0))
        .collect();

    if returns.is_empty() {
        return Err("returns array is empty".into());
    }

    let sorted = {
        let mut s = returns.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s
    };

    let n = sorted.len();
    let var_95_index = (n as f64 * 0.05) as usize;
    let var_95 = sorted[var_95_index];

    Ok(json!({
        "var_95": var_95,
        "cvar_95": var_95 * 1.5,
        "language": "rust",
    }))
}

fn execute_cycle_plan(data: &Value) -> Result<Value, String> {
    let portfolio_value: f64 = data["portfolio_value_aud"].as_f64().unwrap_or(1000.0);
    let cash_balance: f64 = data["cash_balance_aud"].as_f64().unwrap_or(500.0);
    let signals: i32 = data["signals"].as_i64().unwrap_or(2) as i32;
    let primary_exchange: &str = data["primary_exchange"].as_str().unwrap_or("kraken");

    let cash_ratio = cash_balance / portfolio_value;
    let boost = (cash_ratio * 2.0 + signals as f64 * 0.5).min(3.0);

    Ok(json!({
        "cycle_boost": boost,
        "cash_ratio": cash_ratio,
        "signals": signals,
        "primary_exchange": primary_exchange,
        "language": "rust",
    }))
}
