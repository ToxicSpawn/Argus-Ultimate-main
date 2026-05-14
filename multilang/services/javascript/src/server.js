/**
 * Argus JavaScript multilang HTTP service.
 * Fastify server exposing the full Argus protocol:
 *   GET  /health, /ready, /metrics, /capabilities
 *   POST /execute, /batch, /warm
 *
 * Set PORT env to override default 8021.
 * All 20 task types are handled via dynamic import of src/tasks/*.js.
 */

import Fastify from "fastify";
import PROFILE from "./profile.js";

// Dynamic task imports (all 20 handlers)
import cyclePlan from "./tasks/cycle_plan.js";
import orderBookProcessing from "./tasks/order_book_processing.js";
import riskCalculation from "./tasks/risk_calculation.js";
import volatilityEstimate from "./tasks/volatility_estimate.js";
import signalScore from "./tasks/signal_score.js";
import regimeEstimate from "./tasks/regime_estimate.js";
import slippageEstimate from "./tasks/slippage_estimate.js";
import positionSizing from "./tasks/position_sizing.js";
import drawdownCheck from "./tasks/drawdown_check.js";
import correlationEstimate from "./tasks/correlation_estimate.js";
import liquidityScore from "./tasks/liquidity_score.js";
import marketImpact from "./tasks/market_impact.js";
import signalFilter from "./tasks/signal_filter.js";
import confidenceCalibration from "./tasks/confidence_calibration.js";
import heartbeat from "./tasks/heartbeat.js";
import varEstimate from "./tasks/var_estimate.js";
import skewEstimate from "./tasks/skew_estimate.js";
import orderBookImbalanceSeries from "./tasks/order_book_imbalance_series.js";
import executionQualityScore from "./tasks/execution_quality_score.js";
import regimeDuration from "./tasks/regime_duration.js";

const LANGUAGE = "javascript";

const TASK_TYPES = [
  "cycle_plan",
  "order_book_processing",
  "risk_calculation",
  "volatility_estimate",
  "signal_score",
  "regime_estimate",
  "slippage_estimate",
  "position_sizing",
  "drawdown_check",
  "correlation_estimate",
  "liquidity_score",
  "market_impact",
  "signal_filter",
  "confidence_calibration",
  "heartbeat",
  "var_estimate",
  "skew_estimate",
  "order_book_imbalance_series",
  "execution_quality_score",
  "regime_duration",
];

/** Map task_type string to handler function */
const HANDLERS = {
  cycle_plan: cyclePlan,
  order_book_processing: orderBookProcessing,
  risk_calculation: riskCalculation,
  volatility_estimate: volatilityEstimate,
  signal_score: signalScore,
  regime_estimate: regimeEstimate,
  slippage_estimate: slippageEstimate,
  position_sizing: positionSizing,
  drawdown_check: drawdownCheck,
  correlation_estimate: correlationEstimate,
  liquidity_score: liquidityScore,
  market_impact: marketImpact,
  signal_filter: signalFilter,
  confidence_calibration: confidenceCalibration,
  heartbeat: heartbeat,
  var_estimate: varEstimate,
  skew_estimate: skewEstimate,
  order_book_imbalance_series: orderBookImbalanceSeries,
  execution_quality_score: executionQualityScore,
  regime_duration: regimeDuration,
};

// Observability counters
let requestCount = 0;
let totalLatencyMs = 0.0;
let lastLatencyMs = 0.0;
let errorCount = 0;
const startTime = performance.now();

/**
 * Dispatch a single task and return result with timing.
 * @param {string} taskType
 * @param {object} data
 * @returns {{ result: object, tookMs: number, ok: boolean, error?: string }}
 */
function dispatch(taskType, data) {
  const handler = HANDLERS[taskType];
  const t0 = performance.now();
  try {
    const result = handler ? handler(data) : { language: LANGUAGE, ok: true };
    const tookMs = performance.now() - t0;
    return { result, tookMs, ok: true };
  } catch (err) {
    const tookMs = performance.now() - t0;
    return { result: {}, tookMs, ok: false, error: String(err?.message || err) };
  }
}

const fastify = Fastify({ logger: false });

// ── GET /health ────────────────────────────────────────────────────────────────
fastify.get("/health", async () => {
  return { status: "ok", language: LANGUAGE };
});

// ── GET /ready ─────────────────────────────────────────────────────────────────
fastify.get("/ready", async () => {
  return { ready: true, language: LANGUAGE };
});

// ── GET /metrics ───────────────────────────────────────────────────────────────
fastify.get("/metrics", async () => {
  const uptimeS = (performance.now() - startTime) / 1000.0;
  return {
    language: LANGUAGE,
    request_count: requestCount,
    total_latency_ms: Math.round(totalLatencyMs * 100) / 100,
    last_latency_ms: Math.round(lastLatencyMs * 10000) / 10000,
    error_count: errorCount,
    uptime_s: Math.round(uptimeS * 100) / 100,
  };
});

// ── GET /capabilities ──────────────────────────────────────────────────────────
fastify.get("/capabilities", async () => {
  return {
    task_types: TASK_TYPES,
    language: LANGUAGE,
    profile: PROFILE,
  };
});

// ── POST /execute ──────────────────────────────────────────────────────────────
fastify.post("/execute", async (req) => {
  const body = req.body || {};
  const taskType = String(body.task_type || "");
  const data = body.data && typeof body.data === "object" ? { ...body.data } : {};
  const correlationId = body.correlation_id || null;
  if (correlationId) data.correlation_id = correlationId;

  const { result, tookMs, ok, error } = dispatch(taskType, data);
  requestCount++;
  totalLatencyMs += tookMs;
  lastLatencyMs = tookMs;
  if (!ok) errorCount++;

  const out = {
    ok,
    result,
    took_ms: Math.round(tookMs * 10000) / 10000,
  };
  if (!ok) out.error = error;
  if (correlationId) out.correlation_id = correlationId;
  return out;
});

// ── POST /batch ────────────────────────────────────────────────────────────────
fastify.post("/batch", async (req) => {
  const body = req.body || {};
  const tasks = Array.isArray(body.tasks) ? body.tasks.slice(0, 50) : [];
  const correlationId = body.correlation_id || null;

  const t0 = performance.now();
  const results = new Array(tasks.length);

  for (let i = 0; i < tasks.length; i++) {
    const item = tasks[i];
    const taskType = String(item.task_type || "");
    const data = item.data && typeof item.data === "object" ? { ...item.data } : {};
    if (correlationId) data.correlation_id = correlationId;

    const { result, ok, error } = dispatch(taskType, data);
    if (!ok) errorCount++;
    results[i] = ok ? { ok: true, result } : { ok: false, error, result: {} };
  }

  const tookMs = performance.now() - t0;
  requestCount++;
  totalLatencyMs += tookMs;
  lastLatencyMs = tookMs;

  const out = {
    ok: true,
    results,
    took_ms: Math.round(tookMs * 10000) / 10000,
  };
  if (correlationId) out.correlation_id = correlationId;
  return out;
});

// ── POST /warm ─────────────────────────────────────────────────────────────────
fastify.post("/warm", async () => {
  return { warmed: true, language: LANGUAGE };
});

// ── Start ──────────────────────────────────────────────────────────────────────
const PORT = parseInt(process.env.PORT || "8021", 10);

try {
  await fastify.listen({ port: PORT, host: "0.0.0.0" });
  console.log(`argus-javascript-service listening on port ${PORT}`);
} catch (err) {
  console.error(err);
  process.exit(1);
}
