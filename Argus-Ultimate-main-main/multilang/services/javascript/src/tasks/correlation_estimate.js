/**
 * correlation_estimate – Numerically stable Pearson correlation.
 *
 * Uses a single-pass two-variable Welford algorithm for numerical stability,
 * avoiding catastrophic cancellation with large values.
 *
 * V8 optimisation: single loop, no temporary arrays, always same shape result.
 */

const LANGUAGE = "javascript";

/**
 * @param {object} data
 * @returns {object}
 */
export default function correlationEstimate(data) {
  const a = data.series_a || data.returns_a || [];
  const b = data.series_b || data.returns_b || [];
  const n = a.length;

  if (n !== b.length || n < 2) {
    return { correlation: 0.0, language: LANGUAGE, ok: true };
  }

  // Single-pass numerically stable Pearson
  let meanA = 0.0;
  let meanB = 0.0;
  let c = 0.0;    // co-moment
  let varA = 0.0; // sum of squared deviations A
  let varB = 0.0; // sum of squared deviations B

  for (let i = 0; i < n; i++) {
    const ai = +a[i];
    const bi = +b[i];
    const k = i + 1;
    const dA = ai - meanA;
    const dB = bi - meanB;
    meanA += dA / k;
    meanB += dB / k;
    const dA2 = ai - meanA;
    const dB2 = bi - meanB;
    varA += dA * dA2;
    varB += dB * dB2;
    c += dA * dB2;
  }

  const denom = Math.sqrt(varA * varB);
  let correlation = denom !== 0 ? c / denom : 0.0;

  // Clamp to [-1, 1] for floating-point safety
  if (correlation > 1.0) correlation = 1.0;
  if (correlation < -1.0) correlation = -1.0;

  return {
    correlation: correlation,
    language: LANGUAGE,
    ok: true,
  };
}
