/// ARGUS Rust LOB hot-path engine — Order Book Worker
///
/// Protocol: reads JSON lines from stdin, writes JSON lines to stdout (persistent loop).
///
/// Task types:
///   lob_update       — apply a price-level delta, return alpha signals
///   lob_snapshot     — return full book state
///   compute_signals  — return all alpha signals without updating book
///   benchmark        — run 1M update/compute cycles, return ns/op
///
/// Also delegates legacy commands to original rust_engine logic:
///   correlation_matrix, portfolio_var, kelly_fraction, signal_zscore
///
/// Build:
///   cargo build --release
///   # binary at target/release/rust_engine(.exe)
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// ─── LOB Data Structures ─────────────────────────────────────────────────────

/// A single price level in the limit order book.
#[derive(Debug, Clone)]
struct PriceLevel {
    price: f64,
    size: f64,
    count: u32,
}

/// Level-2 order book for one symbol.
/// bids sorted descending (highest bid first).
/// asks sorted ascending (lowest ask first).
struct L2Book {
    bids: Vec<PriceLevel>,
    asks: Vec<PriceLevel>,
    last_update_ns: u64,
    symbol: String,
}

impl L2Book {
    fn new(symbol: &str) -> Self {
        L2Book {
            bids: Vec::with_capacity(64),
            asks: Vec::with_capacity(64),
            last_update_ns: 0,
            symbol: symbol.to_string(),
        }
    }

    /// Binary-search insert / update / delete on one side.
    /// For bids: sorted desc. For asks: sorted asc.
    /// size == 0.0 → delete the level.
    fn update_level(&mut self, side: &str, price: f64, size: f64, count: u32) {
        let levels = if side == "bid" { &mut self.bids } else { &mut self.asks };
        // binary search for existing level
        let pos = if side == "bid" {
            // descending: highest price first
            levels.binary_search_by(|pl| {
                pl.price.partial_cmp(&price).unwrap_or(std::cmp::Ordering::Equal).reverse()
            })
        } else {
            // ascending: lowest price first
            levels.binary_search_by(|pl| {
                pl.price.partial_cmp(&price).unwrap_or(std::cmp::Ordering::Equal)
            })
        };

        match pos {
            Ok(idx) => {
                if size == 0.0 {
                    levels.remove(idx);
                } else {
                    levels[idx].size = size;
                    levels[idx].count = count;
                }
            }
            Err(idx) => {
                if size > 0.0 {
                    levels.insert(idx, PriceLevel { price, size, count });
                }
                // size == 0 and not found → no-op
            }
        }
        self.last_update_ns = now_ns();
    }

    fn best_bid(&self) -> f64 {
        self.bids.first().map(|l| l.price).unwrap_or(0.0)
    }

    fn best_ask(&self) -> f64 {
        self.asks.first().map(|l| l.price).unwrap_or(0.0)
    }

    fn mid(&self) -> f64 {
        let bb = self.best_bid();
        let ba = self.best_ask();
        if bb > 0.0 && ba > 0.0 {
            (bb + ba) * 0.5
        } else {
            0.0
        }
    }

    fn spread_bps(&self) -> f64 {
        let m = self.mid();
        if m == 0.0 {
            return 0.0;
        }
        (self.best_ask() - self.best_bid()) / m * 1e4
    }

    /// Order Book Imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty)
    /// over the top `levels` price levels on each side.
    fn imbalance(&self, levels: usize) -> f64 {
        let bid_qty: f64 = self.bids.iter().take(levels).map(|l| l.size).sum();
        let ask_qty: f64 = self.asks.iter().take(levels).map(|l| l.size).sum();
        let total = bid_qty + ask_qty;
        if total == 0.0 {
            0.0
        } else {
            (bid_qty - ask_qty) / total
        }
    }

    /// Volume-weighted mid using `levels` levels on each side.
    fn weighted_mid(&self, levels: usize) -> f64 {
        let bid_vwap: f64 = self.bids.iter().take(levels)
            .map(|l| l.price * l.size).sum();
        let ask_vwap: f64 = self.asks.iter().take(levels)
            .map(|l| l.price * l.size).sum();
        let bid_vol: f64 = self.bids.iter().take(levels).map(|l| l.size).sum();
        let ask_vol: f64 = self.asks.iter().take(levels).map(|l| l.size).sum();
        let total_vol = bid_vol + ask_vol;
        if total_vol == 0.0 {
            return self.mid();
        }
        (bid_vwap + ask_vwap) / total_vol
    }

    /// Glosten-Milgrom microprice:
    ///   microprice = ask * bid_vol_weight + bid * ask_vol_weight
    /// where weights are best-level volumes.
    fn microprice(&self) -> f64 {
        let bb = self.best_bid();
        let ba = self.best_ask();
        if bb == 0.0 || ba == 0.0 {
            return self.mid();
        }
        let bvol = self.bids.first().map(|l| l.size).unwrap_or(0.0);
        let avol = self.asks.first().map(|l| l.size).unwrap_or(0.0);
        let total = bvol + avol;
        if total == 0.0 {
            return self.mid();
        }
        // microprice = bid * (avol/total) + ask * (bvol/total)
        bb * (avol / total) + ba * (bvol / total)
    }

    /// Exponentially decay-weighted OBI — levels closer to mid have higher weight.
    fn book_pressure(&self, levels: usize) -> f64 {
        let decay = 0.7_f64;
        let mut bid_w = 0.0_f64;
        let mut ask_w = 0.0_f64;
        let mut w = 1.0_f64;
        let n = levels.min(self.bids.len().max(self.asks.len()));
        for i in 0..n {
            let bvol = self.bids.get(i).map(|l| l.size).unwrap_or(0.0);
            let avol = self.asks.get(i).map(|l| l.size).unwrap_or(0.0);
            bid_w += bvol * w;
            ask_w += avol * w;
            w *= decay;
        }
        let total = bid_w + ask_w;
        if total == 0.0 {
            0.0
        } else {
            (bid_w - ask_w) / total
        }
    }

    /// Compute all signals at once.
    fn signals(&self) -> LOBSignals {
        LOBSignals {
            obi: self.imbalance(5),
            weighted_mid: self.weighted_mid(5),
            microprice: self.microprice(),
            spread_bps: self.spread_bps(),
            pressure: self.book_pressure(10),
            timestamp_ns: self.last_update_ns,
        }
    }

    /// Serialise to Value for snapshot.
    fn snapshot_value(&self) -> Value {
        let bids: Vec<Value> = self.bids.iter().map(|l| {
            json!({ "price": l.price, "size": l.size, "count": l.count })
        }).collect();
        let asks: Vec<Value> = self.asks.iter().map(|l| {
            json!({ "price": l.price, "size": l.size, "count": l.count })
        }).collect();
        json!({
            "symbol": self.symbol,
            "bids": bids,
            "asks": asks,
            "best_bid": self.best_bid(),
            "best_ask": self.best_ask(),
            "mid": self.mid(),
            "spread_bps": self.spread_bps(),
            "last_update_ns": self.last_update_ns
        })
    }
}

#[derive(Debug)]
struct LOBSignals {
    obi: f64,
    weighted_mid: f64,
    microprice: f64,
    spread_bps: f64,
    pressure: f64,
    timestamp_ns: u64,
}

impl LOBSignals {
    fn to_value(&self) -> Value {
        json!({
            "obi": self.obi,
            "weighted_mid": self.weighted_mid,
            "microprice": self.microprice,
            "spread_bps": self.spread_bps,
            "pressure": self.pressure,
            "timestamp_ns": self.timestamp_ns
        })
    }
}

// ─── Nanosecond clock ─────────────────────────────────────────────────────────

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

// ─── Global book store ────────────────────────────────────────────────────────

type BookStore = HashMap<String, L2Book>;

fn get_or_create<'a>(books: &'a mut BookStore, symbol: &str) -> &'a mut L2Book {
    if !books.contains_key(symbol) {
        books.insert(symbol.to_string(), L2Book::new(symbol));
    }
    books.get_mut(symbol).unwrap()
}

// ─── Request / Response ───────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Request {
    /// New-style LOB tasks use task_type; legacy commands use "command"
    #[serde(default)]
    task_type: String,
    /// Legacy field (rust_engine protocol)
    #[serde(default)]
    command: String,
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

// ─── LOB task handlers ───────────────────────────────────────────────────────

/// `lob_update` — apply a delta to the book and return signals.
///
/// data fields:
///   symbol: string           (default "BTC-USD")
///   side:   "bid" | "ask"
///   price:  f64
///   size:   f64              (0 = delete level)
///   count:  u32 (optional)
fn handle_lob_update(books: &mut BookStore, data: &Value) -> Value {
    let symbol = data["symbol"].as_str().unwrap_or("BTC-USD");
    let side = data["side"].as_str().unwrap_or("bid");
    let price = data["price"].as_f64().unwrap_or(0.0);
    let size = data["size"].as_f64().unwrap_or(0.0);
    let count = data["count"].as_u64().unwrap_or(1) as u32;

    if price <= 0.0 {
        return json!({ "error": "price must be > 0" });
    }

    let book = get_or_create(books, symbol);
    book.update_level(side, price, size, count);
    let sigs = book.signals();
    sigs.to_value()
}

/// `lob_snapshot` — return the full book.
fn handle_lob_snapshot(books: &mut BookStore, data: &Value) -> Value {
    let symbol = data["symbol"].as_str().unwrap_or("BTC-USD");
    let book = get_or_create(books, symbol);
    book.snapshot_value()
}

/// `compute_signals` — return signals without updating book.
fn handle_compute_signals(books: &mut BookStore, data: &Value) -> Value {
    let symbol = data["symbol"].as_str().unwrap_or("BTC-USD");
    let book = get_or_create(books, symbol);
    book.signals().to_value()
}

/// `benchmark` — 1M update+compute cycles, returns ns/op.
fn handle_benchmark() -> Value {
    let mut book = L2Book::new("BENCH");

    // Pre-seed with a small book
    let base_bid = 45000.0_f64;
    let base_ask = 45001.0_f64;
    for i in 0_u32..20 {
        let offset = i as f64 * 0.5;
        book.update_level("bid", base_bid - offset, 1.0 + offset * 0.1, 1);
        book.update_level("ask", base_ask + offset, 1.0 + offset * 0.1, 1);
    }

    let iterations: u64 = 1_000_000;
    let start = Instant::now();

    for i in 0..iterations {
        // Alternate updates so the compiler can't optimise away
        let side = if i % 2 == 0 { "bid" } else { "ask" };
        let price = if i % 2 == 0 {
            base_bid - (i % 10) as f64 * 0.5
        } else {
            base_ask + (i % 10) as f64 * 0.5
        };
        let size = 1.0 + (i % 5) as f64;
        book.update_level(side, price, size, 1);
        let _sigs = book.signals();
    }

    let elapsed_ns = start.elapsed().as_nanos() as f64;
    let ns_per_op = elapsed_ns / iterations as f64;
    let ops_per_sec = 1e9 / ns_per_op;

    json!({
        "iterations": iterations,
        "elapsed_ns": elapsed_ns as u64,
        "ns_per_op": ns_per_op,
        "ops_per_sec": ops_per_sec,
        "final_mid": book.mid(),
        "final_spread_bps": book.spread_bps()
    })
}

// ─── Legacy rust_engine command handlers ─────────────────────────────────────

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    v.iter().sum::<f64>() / v.len() as f64
}

fn std_dev(v: &[f64]) -> f64 {
    if v.len() < 2 { return 0.0; }
    let m = mean(v);
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
    var.sqrt()
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 { return 0.0; }
    let ma = mean(&a[..n]);
    let mb = mean(&b[..n]);
    let (mut cov, mut va, mut vb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    let denom = (va * vb).sqrt();
    if denom < 1e-15 { 0.0 } else { cov / denom }
}

fn cmd_correlation_matrix(data: &Value) -> Value {
    let series = match data.get("series") {
        Some(Value::Array(arr)) => arr,
        _ => return json!({ "error": "missing 'series' array of arrays" }),
    };
    let vecs: Vec<Vec<f64>> = series.iter().map(|s| {
        s.as_array().unwrap_or(&vec![]).iter()
            .filter_map(|v| v.as_f64())
            .collect()
    }).collect();
    let n = vecs.len();
    let mut matrix = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        matrix[i][i] = 1.0;
        for j in (i + 1)..n {
            let r = pearson(&vecs[i], &vecs[j]);
            matrix[i][j] = r;
            matrix[j][i] = r;
        }
    }
    json!({ "matrix": matrix })
}

fn cmd_portfolio_var(data: &Value) -> Value {
    let returns: Vec<f64> = match data.get("returns") {
        Some(Value::Array(arr)) => arr.iter().filter_map(|v| v.as_f64()).collect(),
        _ => return json!({ "error": "missing 'returns' array" }),
    };
    let weights: Vec<f64> = match data.get("weights") {
        Some(Value::Array(arr)) => arr.iter().filter_map(|v| v.as_f64()).collect(),
        _ => return json!({ "error": "missing 'weights' array" }),
    };
    let confidence = data.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.95);
    let n_assets = weights.len();
    if n_assets == 0 { return json!({ "error": "weights empty" }); }
    let n_periods = returns.len() / n_assets;
    if n_periods < 2 || returns.len() % n_assets != 0 {
        return json!({ "error": "returns length not divisible by weights length" });
    }
    let mut portfolio_returns: Vec<f64> = Vec::with_capacity(n_periods);
    for t in 0..n_periods {
        let mut pr = 0.0_f64;
        for a in 0..n_assets { pr += returns[t * n_assets + a] * weights[a]; }
        portfolio_returns.push(pr);
    }
    portfolio_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((1.0 - confidence) * n_periods as f64).floor() as usize;
    let var = -portfolio_returns[idx.min(n_periods - 1)];
    json!({ "var": var, "confidence": confidence, "n_periods": n_periods })
}

fn cmd_kelly_fraction(data: &Value) -> Value {
    let win_rate = data.get("win_rate").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let avg_win = data.get("avg_win").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let avg_loss = data.get("avg_loss").and_then(|v| v.as_f64()).unwrap_or(0.0);
    if avg_loss.abs() < 1e-15 {
        return json!({ "kelly": 0.0 });
    }
    let b = avg_win / avg_loss.abs();
    let kelly = win_rate - (1.0 - win_rate) / b;
    let kelly_clamped = kelly.max(0.0).min(1.0);
    json!({ "kelly": kelly_clamped, "kelly_raw": kelly, "win_rate": win_rate, "payoff_ratio": b })
}

fn cmd_signal_zscore(data: &Value) -> Value {
    let values: Vec<f64> = match data.get("values") {
        Some(Value::Array(arr)) => arr.iter().filter_map(|v| v.as_f64()).collect(),
        _ => return json!({ "error": "missing 'values' array" }),
    };
    if values.is_empty() { return json!({ "zscores": [] }); }
    let m = mean(&values);
    let s = std_dev(&values);
    let zscores: Vec<f64> = if s < 1e-15 {
        vec![0.0; values.len()]
    } else {
        values.iter().map(|v| (v - m) / s).collect()
    };
    json!({ "zscores": zscores, "mean": m, "std": s })
}

// ─── Dispatcher ───────────────────────────────────────────────────────────────

fn dispatch(books: &mut BookStore, task_type: &str, command: &str, data: &Value) -> Value {
    // Prefer task_type (LOB protocol) over command (legacy protocol)
    let key = if !task_type.is_empty() { task_type } else { command };
    match key {
        "lob_update"       => handle_lob_update(books, data),
        "lob_snapshot"     => handle_lob_snapshot(books, data),
        "compute_signals"  => handle_compute_signals(books, data),
        "benchmark"        => handle_benchmark(),
        // Legacy commands
        "correlation_matrix" => cmd_correlation_matrix(data),
        "portfolio_var"      => cmd_portfolio_var(data),
        "kelly_fraction"     => cmd_kelly_fraction(data),
        "signal_zscore"      => cmd_signal_zscore(data),
        _ => json!({ "error": format!("unknown task_type/command: {}", key) }),
    }
}

// ─── Main loop ────────────────────────────────────────────────────────────────

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    // Global book store — one L2Book per symbol
    let mut books: BookStore = HashMap::new();

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
                let data = if req.data.is_null() { json!({}) } else { req.data.clone() };
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    dispatch(&mut books, &req.task_type, &req.command, &data)
                }));
                let took_ms = start.elapsed().as_secs_f64() * 1000.0;
                match result {
                    Ok(r) => {
                        if r.get("error").is_some() {
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
