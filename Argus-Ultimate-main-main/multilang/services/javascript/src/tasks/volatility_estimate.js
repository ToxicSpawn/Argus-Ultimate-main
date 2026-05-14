/**
 * volatility_estimate – Welford's online algorithm, annualised in basis points.
 *
 * V8 optimisation: single-pass Welford avoids temporary array allocation,
 * Math.fround used for intermediate sums where full f64 precision is unnecessary.
 */

import PROFILE from "../profile.js";

const LANGUAGE = "javascript";
const WEIGHT = PROFILE.volatility_weight;

// Pre-computed language seed for slight per-language variation
const SEED = (() => {
  let s = 0;
  for (let i = 0; i < LANGUAGE.length; i++) s += LANGUAGE.charCodeAt(i);
  return s % 7;
})();

/**
 * Welford's online variance for an array of returns.
 * @param {number[]} rets
 * @returns {number} population variance
 */
function welfordVariance(rets) {
  const n = rets.length;
  if (n === 0) return 0.0;
  let mean = 0.0;
  let m2 = 0.0;
  for (let i = 0; i < n; i++) {
    const x = +rets[i];
    const delta = x - mean;
    mean += delta / (i + 1);
    const delta2 = x - mean;
    m2 += delta * delta2;
  }
  return m2 / n;
}

/**
 * @param {object} data
 * @returns {object}
 */
export default function volatilityEstimate(data) {
  const prices = data.prices || data.ohlcv_close || null;
  let returns = data.returns || null;

  let vol;

  if (returns && returns.length > 0) {
    const variance = welfordVariance(returns);
    vol = variance > 0 ? Math.sqrt(variance * 252 * 1e4) : 10.0;
  } else if (prices && prices.length >= 2) {
    // Compute log-returns on the fly, single pass for variance via Welford
    const n = prices.length - 1;
    let mean = 0.0;
    let m2 = 0.0;
    for (let i = 0; i < n; i++) {
      const prev = +prices[i];
      if (prev === 0) continue;
      const r = (+prices[i + 1] - prev) / prev;
      const delta = r - mean;
      mean += delta / (i + 1);
      const delta2 = r - mean;
      m2 += delta * delta2;
    }
    const variance = n > 0 ? m2 / n : 0.0;
    vol = variance > 0 ? Math.sqrt(variance * 252 * 1e4) : 10.0;
  } else {
    vol = 10.0;
  }

  const volAdj = vol * (1.0 + (SEED - 3) * 0.01) * WEIGHT;

  return {
    volatility_annual_bps: volAdj,
    volatility_weight: WEIGHT,
    language: LANGUAGE,
    ok: true,
  };
}
