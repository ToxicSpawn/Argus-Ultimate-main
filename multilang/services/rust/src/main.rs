mod profile;
mod tasks;

use actix_web::{web, App, HttpServer, HttpResponse};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::profile::Profile;
use crate::tasks::dispatch;

// ---------------------------------------------------------------------------
// Shared application state
// ---------------------------------------------------------------------------
pub struct AppState {
    pub request_count: AtomicU64,
    pub total_latency_ns: AtomicU64,
    pub error_count: AtomicU64,
    pub start_time: Instant,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            request_count: AtomicU64::new(0),
            total_latency_ns: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }
}

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------
#[derive(Deserialize)]
pub struct ExecuteRequest {
    pub task_type: String,
    #[serde(default)]
    pub data: serde_json::Value,
    #[serde(default)]
    pub timeout: Option<f64>,
    #[serde(default)]
    pub correlation_id: Option<String>,
}

#[derive(Serialize)]
pub struct ExecuteResponse {
    pub ok: bool,
    pub result: serde_json::Value,
    pub took_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
}

#[derive(Deserialize)]
pub struct BatchRequest {
    pub tasks: Vec<ExecuteRequest>,
    #[serde(default)]
    pub timeout: Option<f64>,
}

#[derive(Serialize)]
pub struct BatchResponse {
    pub results: Vec<ExecuteResponse>,
    pub took_ms: f64,
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------
async fn health() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({
        "status": "healthy",
        "language": Profile::LANGUAGE,
        "role": Profile::ROLE,
        "timestamp": now_epoch_ms(),
    }))
}

async fn ready() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({
        "ready": true,
        "language": Profile::LANGUAGE,
    }))
}

async fn metrics(state: web::Data<AppState>) -> HttpResponse {
    let reqs = state.request_count.load(Ordering::Relaxed);
    let lat = state.total_latency_ns.load(Ordering::Relaxed);
    let errs = state.error_count.load(Ordering::Relaxed);
    let uptime = state.start_time.elapsed().as_secs_f64();
    let avg_ms = if reqs > 0 {
        (lat as f64 / reqs as f64) / 1_000_000.0
    } else {
        0.0
    };

    HttpResponse::Ok().json(serde_json::json!({
        "request_count": reqs,
        "error_count": errs,
        "avg_latency_ms": avg_ms,
        "uptime_seconds": uptime,
        "language": Profile::LANGUAGE,
        "role": Profile::ROLE,
    }))
}

async fn capabilities() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({
        "language": Profile::LANGUAGE,
        "role": Profile::ROLE,
        "task_types": [
            "order_book", "risk", "volatility", "regime", "var_estimate",
            "correlation", "market_impact", "slippage", "position_sizing", "drawdown",
            "liquidity", "signal_score", "signal_filter", "confidence_cal", "cycle_plan",
            "heartbeat", "skew", "imbalance_series", "execution_quality", "regime_duration"
        ],
        "profile": {
            "risk_max_ratio": Profile::RISK_MAX_RATIO,
            "cycle_boost_scale": Profile::CYCLE_BOOST_SCALE,
            "volatility_weight": Profile::VOLATILITY_WEIGHT,
            "signal_score_weight": Profile::SIGNAL_SCORE_WEIGHT,
            "spread_mult": Profile::SPREAD_MULT,
            "regime_weight": Profile::REGIME_WEIGHT,
            "drawdown_max_ratio": Profile::DRAWDOWN_MAX_RATIO,
            "slippage_tolerance_bps": Profile::SLIPPAGE_TOLERANCE_BPS,
            "min_confidence_to_accept": Profile::MIN_CONFIDENCE_TO_ACCEPT,
        }
    }))
}

async fn execute(
    state: web::Data<AppState>,
    body: web::Json<ExecuteRequest>,
) -> HttpResponse {
    let start = Instant::now();
    state.request_count.fetch_add(1, Ordering::Relaxed);

    let req = body.into_inner();
    let correlation_id = req.correlation_id.clone();

    match dispatch(&req.task_type, &req.data) {
        Ok(result) => {
            let took_ms = start.elapsed().as_secs_f64() * 1000.0;
            state
                .total_latency_ns
                .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            HttpResponse::Ok().json(ExecuteResponse {
                ok: true,
                result,
                took_ms,
                error: None,
                correlation_id,
            })
        }
        Err(e) => {
            let took_ms = start.elapsed().as_secs_f64() * 1000.0;
            state.error_count.fetch_add(1, Ordering::Relaxed);
            state
                .total_latency_ns
                .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            HttpResponse::Ok().json(ExecuteResponse {
                ok: false,
                result: serde_json::json!(null),
                took_ms,
                error: Some(e),
                correlation_id,
            })
        }
    }
}

async fn batch(
    state: web::Data<AppState>,
    body: web::Json<BatchRequest>,
) -> HttpResponse {
    let start = Instant::now();
    let req = body.into_inner();
    let mut results = Vec::with_capacity(req.tasks.len());

    for task in &req.tasks {
        state.request_count.fetch_add(1, Ordering::Relaxed);
        let t0 = Instant::now();
        let correlation_id = task.correlation_id.clone();
        match dispatch(&task.task_type, &task.data) {
            Ok(result) => {
                let took = t0.elapsed().as_secs_f64() * 1000.0;
                state
                    .total_latency_ns
                    .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                results.push(ExecuteResponse {
                    ok: true,
                    result,
                    took_ms: took,
                    error: None,
                    correlation_id,
                });
            }
            Err(e) => {
                let took = t0.elapsed().as_secs_f64() * 1000.0;
                state.error_count.fetch_add(1, Ordering::Relaxed);
                state
                    .total_latency_ns
                    .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                results.push(ExecuteResponse {
                    ok: false,
                    result: serde_json::json!(null),
                    took_ms: took,
                    error: Some(e),
                    correlation_id,
                });
            }
        }
    }

    let took_ms = start.elapsed().as_secs_f64() * 1000.0;
    HttpResponse::Ok().json(BatchResponse { results, took_ms })
}

async fn warm() -> HttpResponse {
    // Pre-warm: nothing special needed for Rust — the binary is already compiled.
    HttpResponse::Ok().json(serde_json::json!({
        "ok": true,
        "language": Profile::LANGUAGE,
        "message": "warm-up complete",
    }))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
fn now_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8011);

    let state = web::Data::new(AppState::new());

    println!(
        "Argus Rust service starting on 0.0.0.0:{} (role={})",
        port,
        Profile::ROLE
    );

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .route("/health", web::get().to(health))
            .route("/ready", web::get().to(ready))
            .route("/metrics", web::get().to(metrics))
            .route("/capabilities", web::get().to(capabilities))
            .route("/execute", web::post().to(execute))
            .route("/batch", web::post().to(batch))
            .route("/warm", web::post().to(warm))
    })
    .bind(format!("0.0.0.0:{}", port))?
    .run()
    .await
}
