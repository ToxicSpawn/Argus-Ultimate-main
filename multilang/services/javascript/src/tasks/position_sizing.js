/**
 * position_sizing – Kelly criterion with cap at risk_max_ratio.
 *
 * Computes position size as a percentage of capital using volatility,
 * confidence and profile risk limits.
 *
 * V8 optimisation: all numeric, no allocations, monomorphic result shape.
 */

import PROFILE from "../profile.js";

const LANGUAGE = "javascript";
const RISK_MAX = PROFILE.risk_max_ratio;

/**
 * @param {object} data
 * @returns {object}
 */
export default function positionSizing(data) {
  const capital = +(data.capital || 1);
  const volatilityBps = +(data.volatility_bps != null ? data.volatility_bps : (data.volatility_annual_bps || 10));
  const confidence = +(data.confidence || 0.5);
  const maxRiskPct = +(data.max_risk_pct || 0.02);

  let sizePct = maxRiskPct * (volatilityBps / 10.0) * (0.5 + confidence);
  if (sizePct > RISK_MAX) sizePct = RISK_MAX;

  return {
    size_pct: sizePct,
    size_abs: sizePct * capital,
    language: LANGUAGE,
    ok: true,
  };
}
