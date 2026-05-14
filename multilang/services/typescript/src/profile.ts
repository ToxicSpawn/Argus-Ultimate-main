/**
 * Argus TypeScript language profile constants.
 * TypeScript role: "ecosystem" — strict null checks, typed boundaries.
 */

export interface LanguageProfile {
  readonly risk_max_ratio: number;
  readonly cycle_boost_scale: number;
  readonly volatility_weight: number;
  readonly signal_score_weight: number;
  readonly spread_mult: number;
  readonly role: string;
  readonly regime_weight: number;
  readonly drawdown_max_ratio: number;
  readonly slippage_tolerance_bps: number;
  readonly min_confidence_to_accept: number;
}

const PROFILE: LanguageProfile = Object.freeze({
  risk_max_ratio: 0.45,
  cycle_boost_scale: 0.99,
  volatility_weight: 0.98,
  signal_score_weight: 1.0,
  spread_mult: 1.01,
  role: "ecosystem",
  regime_weight: 1.0,
  drawdown_max_ratio: 1.0,
  slippage_tolerance_bps: 100,
  min_confidence_to_accept: 0.5,
});

export default PROFILE;
