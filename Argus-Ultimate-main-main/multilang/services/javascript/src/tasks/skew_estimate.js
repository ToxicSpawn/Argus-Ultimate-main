/**
 * skew_estimate – Fisher's moment coefficient of skewness.
 *
 * Two-pass: mean then central moments. Single accumulated variance reused.
 *
 * V8 optimisation: monomorphic numeric loop, no temporary arrays.
 */

const LANGUAGE = "javascript";

/**
 * @param {object} data
 * @returns {object}
 */
export default function skewEstimate(data) {
  const returns = data.returns || [];
  const n = returns.length;

  if (n < 3) {
    return { skew: 0.0, language: LANGUAGE, ok: true };
  }

  // Pass 1: mean
  let mean = 0.0;
  for (let i = 0; i < n; i++) mean += +returns[i];
  mean /= n;

  // Pass 2: variance and third central moment
  let sumVar = 0.0;
  let sumCube = 0.0;
  for (let i = 0; i < n; i++) {
    const d = +returns[i] - mean;
    const d2 = d * d;
    sumVar += d2;
    sumCube += d2 * d;
  }
  const variance = sumVar / n;
  const std = variance > 0 ? Math.sqrt(variance) : 0.0;
  const skew = std > 0 ? (sumCube / n) / (std * std * std) : 0.0;

  return {
    skew: skew,
    language: LANGUAGE,
    ok: true,
  };
}
