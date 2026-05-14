/**
 * risk_calculation – position_value/capital vs risk_max_ratio threshold.
 *
 * V8 optimisation: simple monomorphic comparison, no allocations.
 */

import PROFILE from "../profile.js";

const LANGUAGE = "javascript";
const MAX_RATIO = PROFILE.risk_max_ratio;

/**
 * @param {object} data
 * @returns {object}
 */
export default function riskCalculation(data) {
  const positionValue = +(data.position_value || 0);
  const capital = +(data.capital || 1);
  const ratio = capital !== 0 ? positionValue / capital : 0.0;
  const passed = ratio <= MAX_RATIO;

  return {
    passed: passed,
    exposure_ratio: ratio,
    max_ratio: MAX_RATIO,
    language: LANGUAGE,
  };
}
