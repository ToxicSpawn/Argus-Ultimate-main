/**
 * signal_filter – Regime and confidence gate for signal acceptance.
 *
 * V8 optimisation: monomorphic branches, pre-computed language seed.
 */

const LANGUAGE = "javascript";

// Per-language seed: tighter filter for seed === 0
const SEED = (() => {
  let s = 0;
  for (let i = 0; i < LANGUAGE.length; i++) s += LANGUAGE.charCodeAt(i);
  return s % 5;
})();

/**
 * @param {object} data
 * @returns {object}
 */
export default function signalFilter(data) {
  const sig = data.signal && typeof data.signal === "object" ? data.signal : data;
  const confidence = +(sig.confidence != null ? sig.confidence : (data.confidence || 0.0));
  const regime = String(data.regime || "mean_revert");
  const volatility = +(data.volatility || 0.01);

  let accept = confidence >= 0.5 && (regime !== "high_vol" || volatility < 0.02);
  if (SEED === 0 && confidence < 0.8) accept = false;

  return {
    accept: accept,
    filter_reason: accept ? "" : "low_confidence_or_regime",
    language: LANGUAGE,
    ok: true,
  };
}
