/// Rust language profile constants for the Argus trading system.
/// Language: rust, Role: speed
pub struct Profile;

impl Profile {
    pub const RISK_MAX_RATIO: f64 = 0.48;
    pub const CYCLE_BOOST_SCALE: f64 = 1.0;
    pub const VOLATILITY_WEIGHT: f64 = 0.9;
    pub const SIGNAL_SCORE_WEIGHT: f64 = 1.0;
    pub const SPREAD_MULT: f64 = 1.0;
    pub const ROLE: &'static str = "speed";
    pub const REGIME_WEIGHT: f64 = 1.0;
    pub const DRAWDOWN_MAX_RATIO: f64 = 1.0;
    pub const SLIPPAGE_TOLERANCE_BPS: f64 = 80.0;
    pub const MIN_CONFIDENCE_TO_ACCEPT: f64 = 0.5;
    pub const LANGUAGE: &'static str = "rust";
}
