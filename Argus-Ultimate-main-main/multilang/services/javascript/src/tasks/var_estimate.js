/**
 * var_estimate – Historical VaR and CVaR at a given confidence level.
 *
 * Sort-based historical simulation. Returns percentage loss figures.
 *
 * V8 optimisation: numeric sort, single-pass CVaR accumulation.
 */

const LANGUAGE = "javascript";

/**
 * @param {object} data
 * @returns {object}
 */
export default function varEstimate(data) {
  const returns = data.returns || [];
  const confidenceLevel = +(data.confidence_level || 0.95);

  if (returns.length < 5) {
    return { var_pct: 0.0, cvar_pct: 0.0, language: LANGUAGE, ok: true };
  }

  // Copy and sort ascending
  const arr = new Array(returns.length);
  for (let i = 0; i < returns.length; i++) arr[i] = +returns[i];
  arr.sort((a, b) => a - b);

  const idx = Math.max(0, Math.min(Math.floor((1 - confidenceLevel) * arr.length), arr.length - 1));
  const varPct = -arr[idx] * 100.0;

  let cvarSum = 0.0;
  const cvarCount = idx + 1;
  for (let i = 0; i <= idx; i++) cvarSum += arr[i];
  const cvarPct = cvarCount > 0 ? -(cvarSum / cvarCount) * 100.0 : varPct;

  return {
    var_pct: varPct,
    cvar_pct: cvarPct,
    language: LANGUAGE,
    ok: true,
  };
}
