/**
 * Argus JavaScript language profile constants.
 * These values are used across all task handlers to apply language-specific
 * behaviour consistent with the unified orchestrator profiles.
 */

/** @type {Readonly<{risk_max_ratio:number, cycle_boost_scale:number, volatility_weight:number, signal_score_weight:number, spread_mult:number, role:string, regime_weight:number, drawdown_max_ratio:number, slippage_tolerance_bps:number, min_confidence_to_accept:number}>} */
const PROFILE = Object.freeze({
  risk_max_ratio: 0.46,
  cycle_boost_scale: 1.0,
  volatility_weight: 0.95,
  signal_score_weight: 1.0,
  spread_mult: 1.0,
  role: "ecosystem",
  regime_weight: 1.0,
  drawdown_max_ratio: 1.0,
  slippage_tolerance_bps: 100,
  min_confidence_to_accept: 0.5,
});

export default PROFILE;
