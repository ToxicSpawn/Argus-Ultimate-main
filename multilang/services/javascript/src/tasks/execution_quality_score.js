/**
 * execution_quality_score – Fill-vs-decision slippage quality metric.
 *
 * Scores 0–1 based on average slippage in bps vs a 50 bps ceiling.
 *
 * V8 optimisation: capped loop (max 10 fills), monomorphic shape.
 */

const LANGUAGE = "javascript";
const MAX_FILLS = 10;
const BPS_CEILING = 50.0;

/**
 * @param {object} data
 * @returns {object}
 */
export default function executionQualityScore(data) {
  const fills = data.fills || [];
  const decisionPrices = data.decision_prices || [];

  if (fills.length === 0 || fills.length !== decisionPrices.length) {
    return { score_0_1: 1.0, avg_slippage_bps: 0.0, language: LANGUAGE, ok: true };
  }

  const count = Math.min(fills.length, MAX_FILLS);
  let totalBps = 0.0;
  let used = 0;

  for (let i = 0; i < count; i++) {
    const f = fills[i];
    const fp = typeof f === "object" && f !== null ? +(f.price != null ? f.price : f) : +f;
    const dp = +decisionPrices[i];
    if (dp > 0 && fp > 0) {
      totalBps += Math.abs(fp - dp) / dp * 1e4;
      used++;
    }
  }

  const avgBps = used > 0 ? totalBps / used : 0.0;
  const score = Math.max(0.0, Math.min(1.0, 1.0 - avgBps / BPS_CEILING));

  return {
    score_0_1: score,
    avg_slippage_bps: avgBps,
    language: LANGUAGE,
    ok: true,
  };
}
