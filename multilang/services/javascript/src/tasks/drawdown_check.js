/**
 * drawdown_check – (peak - current) / peak check.
 *
 * Verifies current drawdown stays within the allowed maximum threshold,
 * scaled by the profile's risk_max_ratio.
 *
 * V8 optimisation: pure numeric, no branching polymorphism.
 */

import PROFILE from "../profile.js";

const LANGUAGE = "javascript";
const MAX_DD_RATIO = PROFILE.risk_max_ratio;

/**
 * @param {object} data
 * @returns {object}
 */
export default function drawdownCheck(data) {
  const maxDrawdownPct = +(data.max_drawdown_pct || 0.12);
  const current = +(data.current_equity || 0);
  const peak = +(data.peak_equity || current || 1);

  const currentDrawdownPct = peak !== 0 ? (peak - current) / peak : 0.0;
  const passed = currentDrawdownPct <= maxDrawdownPct * MAX_DD_RATIO;

  return {
    passed: passed,
    current_drawdown_pct: currentDrawdownPct,
    language: LANGUAGE,
    ok: true,
  };
}
