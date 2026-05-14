/**
 * Argus TypeScript multilang HTTP service.
 * Fastify + TypeScript — strict types, all 20 task handlers.
 * Run: npm run build && npm start
 * Default port: 8022 (override with PORT env).
 */

import Fastify, { FastifyInstance } from "fastify";
import PROFILE from "./profile.js";
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

const LANGUAGE = "typescript";

const TASK_TYPES = [
  "cycle_plan", "order_book_processing", "risk_calculation", "volatility_estimate",
  "signal_score", "regime_estimate", "slippage_estimate", "position_sizing",
  "drawdown_check", "correlation_estimate", "liquidity_score", "market_impact",
  "signal_filter", "confidence_calibration", "heartbeat", "var_estimate",
  "skew_estimate", "order_book_imbalance_series", "execution_quality_score", "regime_duration",
] as const;

type Handler = (data: Record<string, unknown>) => Record<string, unknown>;

const HANDLERS: Record<string, Handler> = {
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

// Observability
let requestCount = 0;
let totalLatencyMs = 0.0;
let lastLatencyMs = 0.0;
let errorCount = 0;
const startTime = performance.now();

function dispatch(taskType: string, data: Record<string, unknown>): { result: Record<string, unknown>; tookMs: number; ok: boolean; error?: string } {
  const handler = HANDLERS[taskType];
  const t0 = performance.now();
  try {
    const result = handler ? handler(data) : { language: LANGUAGE, ok: true };
    return { result, tookMs: performance.now() - t0, ok: true };
  } catch (err) {
    return { result: {}, tookMs: performance.now() - t0, ok: false, error: String(err instanceof Error ? err.message : err) };
  }
}

const fastify: FastifyInstance = Fastify({ logger: false });

fastify.get("/health", async () => ({ status: "ok", language: LANGUAGE }));
fastify.get("/ready", async () => ({ ready: true, language: LANGUAGE }));
fastify.get("/metrics", async () => ({
  language: LANGUAGE,
  request_count: requestCount,
  total_latency_ms: Math.round(totalLatencyMs * 100) / 100,
  last_latency_ms: Math.round(lastLatencyMs * 10000) / 10000,
  error_count: errorCount,
  uptime_s: Math.round((performance.now() - startTime) / 10) / 100,
}));
fastify.get("/capabilities", async () => ({ task_types: TASK_TYPES, language: LANGUAGE, profile: PROFILE }));

fastify.post("/execute", async (req) => {
  const body = req.body as Record<string, unknown>;
  const taskType = String(body["task_type"] ?? "");
  const data: Record<string, unknown> = { ...(body["data"] as Record<string, unknown> ?? {}) };
  const correlationId = (body["correlation_id"] as string | undefined) ?? null;
  if (correlationId) data["correlation_id"] = correlationId;

  const { result, tookMs, ok, error } = dispatch(taskType, data);
  requestCount++; totalLatencyMs += tookMs; lastLatencyMs = tookMs;
  if (!ok) errorCount++;

  const out: Record<string, unknown> = { ok, result, took_ms: Math.round(tookMs * 10000) / 10000 };
  if (!ok) out["error"] = error;
  if (correlationId) out["correlation_id"] = correlationId;
  return out;
});

fastify.post("/batch", async (req) => {
  const body = req.body as Record<string, unknown>;
  const tasks = ((body["tasks"] as unknown[]) ?? []).slice(0, 50) as Record<string, unknown>[];
  const correlationId = (body["correlation_id"] as string | undefined) ?? null;

  const t0 = performance.now();
  const results = tasks.map((item) => {
    const taskType = String(item["task_type"] ?? "");
    const data: Record<string, unknown> = { ...(item["data"] as Record<string, unknown> ?? {}) };
    if (correlationId) data["correlation_id"] = correlationId;
    const { result, ok, error } = dispatch(taskType, data);
    if (!ok) errorCount++;
    return ok ? { ok: true, result } : { ok: false, error, result: {} };
  });

  const tookMs = performance.now() - t0;
  requestCount++; totalLatencyMs += tookMs; lastLatencyMs = tookMs;

  const out: Record<string, unknown> = { ok: true, results, took_ms: Math.round(tookMs * 10000) / 10000 };
  if (correlationId) out["correlation_id"] = correlationId;
  return out;
});

fastify.post("/warm", async () => ({ warmed: true, language: LANGUAGE }));

const PORT = parseInt(process.env["PORT"] ?? "8022", 10);
await fastify.listen({ port: PORT, host: "0.0.0.0" });
console.log(`argus-typescript-service listening on port ${PORT}`);
