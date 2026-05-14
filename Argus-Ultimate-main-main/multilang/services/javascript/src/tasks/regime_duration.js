/**
 * regime_duration – Bars-in-current-regime and stability flag.
 *
 * Derives regime from price volatility; uses regime_history length if provided.
 *
 * V8 optimisation: single-pass returns computation, monomorphic result.
 */

const LANGUAGE = "javascript";

/**
 * @param {object} data
 * @returns {object}
 */
export default function regimeDuration(data) {
  const prices = data.prices || [];
  const regimeHistory = data.regime_history || [];

  if (prices.length < 2) {
    return { bars_in_regime: 0, regime_stable: false, language: LANGUAGE, ok: true };
  }

  const n = prices.length - 1;
  let sumSq = 0.0;
  for (let i = 0; i < n; i++) {
    const prev = +prices[i];
    if (prev === 0) continue;
    const r = (+prices[i + 1] - prev) / prev;
    sumSq += r * r;
  }
  const vol = n > 0 ? Math.sqrt(sumSq / n * 252 * 1e4) : 10.0;
  const regime = vol > 20.0 ? "high_vol" : "mean_revert";

  const bars = regimeHistory.length > 0 ? regimeHistory.length : Math.min(10, prices.length);

  return {
    bars_in_regime: bars,
    regime_stable: bars >= 5,
    regime: regime,
    language: LANGUAGE,
    ok: true,
  };
}
