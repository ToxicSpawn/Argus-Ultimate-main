/**
 * regime_estimate – Vol threshold + trend magnitude classification.
 *
 * Classifies the current market regime as "high_vol", "trend" or "mean_revert"
 * using realised volatility and price trend magnitude.
 *
 * V8 optimisation: no intermediate arrays, single pass for vol computation.
 */

import PROFILE from "../profile.js";

const LANGUAGE = "javascript";

/**
 * @param {object} data
 * @returns {object}
 */
export default function regimeEstimate(data) {
  const prices = data.prices || data.returns || [];
  const len = prices.length;

  let regime = "mean_revert";
  let confidence = 0.5;

  if (len >= 3) {
    // Compute realised vol and trend in a single pass
    let sumSqRet = 0.0;
    let retCount = 0;
    for (let i = 1; i < len; i++) {
      const prev = +prices[i - 1];
      if (prev === 0) continue;
      const r = (+prices[i] - prev) / prev;
      sumSqRet += r * r;
      retCount++;
    }

    const vol = retCount > 0 ? Math.sqrt((sumSqRet / retCount) * 252 * 1e4) : 10.0;

    const first = +prices[0];
    const last = +prices[len - 1];
    const trend = first !== 0 ? (last - first) / first : 0.0;

    if (vol > 20.0) {
      regime = "high_vol";
    } else if (Math.abs(trend) > 0.02) {
      regime = "trend";
    }
    // else stays "mean_revert"

    confidence = Math.min(0.95, 0.5 + Math.abs(trend) * 5 + vol / 100);
  }

  return {
    regime: regime,
    confidence: confidence,
    language: LANGUAGE,
    regime_weight: PROFILE.regime_weight,
    ok: true,
  };
}
