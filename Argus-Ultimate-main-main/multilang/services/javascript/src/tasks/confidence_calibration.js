/**
 * confidence_calibration – Win-rate-blended confidence calibration.
 *
 * Blends historical average confidence with empirical win rate (50/50).
 *
 * V8 optimisation: single-pass loops, typed numeric accumulation.
 */

const LANGUAGE = "javascript";

/**
 * @param {object} data
 * @returns {object}
 */
export default function confidenceCalibration(data) {
  const confs = data.historical_confidences || [];
  const pnls = data.historical_pnl || [];

  if (confs.length !== pnls.length || confs.length < 2) {
    return { calibrated_confidence: 0.5, language: LANGUAGE, ok: true };
  }

  const n = confs.length;
  let sumConf = 0.0;
  let wins = 0;
  for (let i = 0; i < n; i++) {
    sumConf += +confs[i];
    if (+pnls[i] > 0) wins++;
  }
  const avgConf = sumConf / n;
  const winRate = wins / n;
  const calibrated = Math.min(1.0, Math.max(0.0, 0.5 * avgConf + 0.5 * winRate));

  return {
    calibrated_confidence: calibrated,
    language: LANGUAGE,
    ok: true,
  };
}
