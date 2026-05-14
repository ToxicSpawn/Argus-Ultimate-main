#!/usr/bin/env python3
"""
config/unified_config.py
========================
UnifiedConfig dataclass — extracted from unified_trading_system.py.

This module owns the single authoritative copy of UnifiedConfig.
unified_trading_system.py imports from here; all other modules
should also import from here rather than from unified_trading_system.

Backward-compat re-export is provided via config/__init__.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


@dataclass
class UnifiedConfig:
    """Unified configuration for all three systems"""
    config_version: int = 1  # From YAML; increment for breaking config migrations
    # Capital settings
    starting_capital_aud: float = 1000.0
    currency: str = "AUD"
    # FX (used when trading USD-quoted pairs with AUD base capital)
    aud_to_usd: float = 0.65  # USD per 1 AUD (configurable)

    # Exchange settings
    primary_exchange: str = "kraken"
    secondary_exchange: str = "coinbase_advanced"
    supported_exchanges: List[str] = field(default_factory=lambda: ["kraken", "coinbase_advanced"])

    # Position sizing (optimized for $1K AUD max earnings)
    min_position_size_aud: float = 10.0
    max_position_size_aud: float = 250.0
    max_position_pct: float = 0.25
    max_total_exposure_pct: float = 0.98
    max_concurrent_positions: int = 5

    # Risk management
    max_daily_loss_pct: float = 0.10
    max_drawdown_pct: float = 0.25
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.08
    use_volatility_adjusted_limits: bool = False
    realized_vol_pct: float = 0.0
    # PR-14 portfolio guardrails
    portfolio_var_limit_pct: float = 5.0
    portfolio_cvar_limit_pct: float = 8.0
    portfolio_var_confidence: float = 0.95
    portfolio_var_lookback_trades: int = 50
    cluster_drawdown_brake_pct: float = 0.0
    target_cluster_cap_pct: float = 0.40
    risk_cluster_map: Dict[str, str] = field(default_factory=dict)
    portfolio_vol_target_pct: float = 2.0
    portfolio_liquidity_spread_ref_bps: float = 20.0
    portfolio_exposure_min_scale: float = 0.30
    # Portfolio Target Engine v1
    targets_enabled: bool = False
    target_convergence_alpha: float = 1.0
    target_rebalance_min_delta_pct: float = 0.02
    target_score_confidence_weight: float = 1.0
    target_score_net_edge_weight: float = 1.0
    target_regime_boost_enabled: bool = True
    # Strategy Evaluation Engine v1
    strategy_evaluation_enabled: bool = True
    strategy_evaluation_persist_interval_cycles: int = 10
    strategy_evaluation_min_trades_for_ranking: int = 5
    strategy_evaluation_use_regime_scoped_metrics: bool = True
    strategy_evaluation_sharpe_like_min_trades: int = 5
    strategy_evaluation_max_metrics_history_points: int = 500
    strategy_evaluation_halt_on_error: bool = False
    strategy_evaluation_db_path: str = "data/strategy_metrics.db"
    # Self-Optimizing Meta Engine v1
    self_optimizing_meta_enabled: bool = True
    self_optimizing_meta_advisory_only: bool = False
    self_optimizing_meta_update_interval_cycles: int = 10
    self_optimizing_meta_min_trades_for_reweighting: int = 5
    self_optimizing_meta_alpha: float = 0.2
    self_optimizing_meta_max_weight_change_per_update: float = 0.10
    self_optimizing_meta_min_weight_per_strategy: float = 0.05
    self_optimizing_meta_max_weight_per_strategy: float = 0.45
    self_optimizing_meta_baseline_weight_mode: str = "equal"
    self_optimizing_meta_score_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "expectancy": 1.0,
            "sharpe_like": 1.0,
            "profit_factor": 0.75,
            "drawdown_penalty": 1.0,
            "fee_penalty": 0.5,
            "slippage_penalty": 0.5,
        }
    )
    self_optimizing_meta_regime_multipliers: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "trend": {"momentum": 1.15, "breakout": 1.10},
            "range": {"mean_reversion": 1.15},
        }
    )
    self_optimizing_meta_db_path: str = "data/meta_weights.db"
    # Liquidity-Aware Risk Engine v1
    liquidity_risk_enabled: bool = True
    liquidity_risk_depth_fraction_limit: float = 0.04
    liquidity_risk_thin_spread_threshold_bps: float = 6.0
    liquidity_risk_danger_spread_threshold_bps: float = 12.0
    liquidity_risk_min_depth_threshold: float = 0.5
    liquidity_risk_slippage_threshold_bps: float = 10.0
    liquidity_risk_min_liquidity_score: float = 0.2
    liquidity_risk_score_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "depth": 1.0,
            "spread": 1.0,
            "fill_ratio": 0.75,
        }
    )
    # Champion / Challenger Promotion Engine v1
    champion_challenger_enabled: bool = True
    champion_challenger_advisory_only: bool = True
    champion_challenger_min_trades_for_promotion: int = 10
    champion_challenger_max_drawdown_pct_for_promotion: float = 0.12
    champion_challenger_require_expectancy_improvement: bool = True
    champion_challenger_require_profit_factor_improvement: bool = False
    champion_challenger_require_sharpe_like_improvement: bool = True
    champion_challenger_persist_interval_cycles: int = 10
    champion_challenger_db_path: str = "data/champion_challenger.db"
    champion_challenger_artifacts_dir: str = "deploy/promotions"
    champion_challenger_promotion_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "net_pnl": 1.0,
            "expectancy": 1.0,
            "profit_factor": 0.75,
            "sharpe_like": 1.0,
            "drawdown_penalty": 1.25,
            "fee_penalty": 0.5,
        }
    )
    # Market Microstructure Engine v1
    market_microstructure_enabled: bool = True
    market_microstructure_rolling_window: int = 20
    market_microstructure_vacuum_spread_jump_bps: float = 4.0
    market_microstructure_vacuum_depth_drop_ratio: float = 0.5
    market_microstructure_high_adverse_selection_threshold: float = 0.7
    market_microstructure_use_in_execution_alpha: bool = True
    market_microstructure_use_in_liquidity_risk: bool = True
    # Recon-Required Recovery Engine v1
    recon_recovery_enabled: bool = True
    recon_recovery_stale_threshold_seconds: float = 60.0
    recon_recovery_base_retry_delay_seconds: float = 5.0
    recon_recovery_max_retries: int = 5
    recon_recovery_halt_on_retry_exhausted: bool = True
    # System Health Metrics
    system_health_metrics_enabled: bool = True
    system_health_metrics_snapshot_interval_cycles: int = 10
    # Execution Alpha Engine v2
    execution_alpha_enabled: bool = True
    execution_alpha_maker_spread_threshold_bps: float = 2.0
    execution_alpha_min_fill_probability: float = 0.35
    execution_alpha_slice_threshold_pct: float = 0.03
    execution_alpha_maker_fallback_seconds: float = 8.0
    execution_alpha_telemetry_window: int = 200

    # Runtime soak safety controls
    runtime_safety_latency_grace_cycles: int = 2
    live_safe_disable_pinnacle_ai_brain: bool = False

    # Market data runtime controls
    market_data_ohlcv_cache_seconds: float = 30.0
    market_data_ohlcv_poll_interval_seconds: float = 30.0
    market_data_ohlcv_retry_attempts: int = 2
    # Emergency shutdown extra conditions
    emergency_shutdown_enabled: bool = True
    emergency_shutdown_latency_spike_ms: Optional[float] = None
    emergency_shutdown_flash_crash_pct: Optional[float] = None
    emergency_shutdown_network_fail: bool = False
    emergency_shutdown_arb_spread_bps: Optional[float] = None

    # Commission and slippage awareness
    kraken_maker_fee: float = 0.0016
    kraken_taker_fee: float = 0.0026
    coinbase_maker_fee: float = 0.005
    coinbase_taker_fee: float = 0.005
    slippage_pct: float = 0.001

    # Execution engine settings
    order_type: str = "market"
    retry_attempts: int = 3
    retry_delay_seconds: float = 5.0
    max_slippage_pct: float = 0.01
    signal_cooldown_bars: int = 0
    vwap_large_order_threshold_aud: float = 80.0
    use_twap_for_large_orders: bool = False
    order_fill_timeout_seconds: float = 0.0
    max_spread_bps: float = 0.0
    use_is_gate: bool = False
    max_avg_is_bps: float = 0.0
    portfolio_weight_method: str = "hrp"
    multi_venue_enabled: bool = True
    multi_venue_min_notional_aud: float = 200.0
    twap_min_notional_usd: float = 250.0
    twap_duration_minutes: float = 5.0
    use_venue_routing_by_spread: bool = False
    dca_levels_pct: Optional[List[float]] = None
    use_correlation_aware_sizing: bool = False
    max_correlated_exposure: float = 0.6
    correlation_matrix: Optional[Dict[Any, float]] = None
    # Data – tick store + data lake
    persist_tick_store: bool = True
    use_lake_read: bool = True
    persist_to_lake: bool = True
    persist_to_tick_store: bool = True
    lake_path: str = "data/lake"

    # Trading pairs
    trading_pairs: List[str] = field(default_factory=lambda: [
        "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD",
        "BTC/EUR", "ETH/EUR"
    ])

    # AI settings
    ai_enabled: bool = True
    consciousness_enabled: bool = True
    swarm_intelligence_enabled: bool = True
    num_ai_agents: int = 50
    min_signal_confidence: float = 0.75
    live_min_signal_confidence: Optional[float] = None
    max_concurrent_signals: int = 2

    # Monitoring
    prometheus_enabled: bool = True
    grafana_enabled: bool = True
    prometheus_port: int = 9090
    grafana_port: int = 3000

    # Multi-language system
    multi_language_enabled: bool = True
    multi_language_endpoints: Dict[str, str] = field(default_factory=dict)
    use_cycle_aggregate_boost: bool = True
    use_conservative_cycle_boost: bool = False
    use_weighted_mean_boost: bool = False
    use_risk_all: bool = False
    use_conservative_risk: bool = False
    use_regime_estimate: bool = False
    use_slippage_estimate: bool = False
    use_drawdown_check: bool = False
    use_position_sizing_gate: bool = False
    use_signal_filter_gate: bool = False
    max_slippage_bps: float = 100.0
    multi_language_task_timeouts: Optional[Dict[str, float]] = None
    multi_language_warm_on_start: bool = False

    # Runtime mode safety interlock
    run_mode: str = "paper"
    node_role: str = "single-node"
    command_bus_enabled: bool = False
    command_bus_db_path: str = "data/command_bus.db"
    command_bus_queue: str = "default"
    command_bus_hmac_key_env: str = "ARGUS_COMMAND_HMAC_KEY"
    command_bus_instruction_ttl_seconds: float = 5.0
    command_bus_require_signature: bool = True
    command_bus_max_batch: int = 64
    command_bus_max_notional_aud: float = 0.0
    # OMEGA-01 execution mesh
    execution_mesh_enabled: bool = False
    execution_mesh_max_lanes: int = 8
    execution_mesh_max_queue_per_lane: int = 128
    execution_mesh_batch_size: int = 8
    execution_mesh_parallel_lanes: bool = True
    execution_mesh_halt_on_lane_error: bool = True
    execution_mesh_symbols: List[str] = field(default_factory=list)
    live_disabled_strategies: List[str] = field(default_factory=list)
    # Edge gate
    live_require_paper_edge: bool = False
    live_min_trades_paper: int = 20
    live_min_win_rate_pct: float = 45.0

    # Backtest/paper realism
    backtest: Dict[str, Any] = field(default_factory=dict)

    # Evolution
    evolution_load_evolved: bool = False
    evolution_params_path: str = "data/evolved_params.json"
    evolution_continuous_enabled: bool = True
    evolution_interval_hours: float = 24.0
    evolution_fitness_days: int = 7
    evolution_generations: int = 5
    evolution_population_size: int = 12
    evolution_auto_apply: bool = True
    evolution_realtime_interval_minutes: float = 60.0
    evolution_realtime_fitness_days: float = 1.0
    evolution_use_live_feed: bool = True
    evolution_realtime_generations: int = 3
    evolution_realtime_population_size: int = 8
    evolution_min_bars_for_live_feed: int = 20
    evolution_run_with_market: bool = True
    evolution_trigger_on_trade: bool = False
    evolution_after_n_trades: int = 5
    evolution_dry_run: bool = False
    evolution_debounce_minutes: float = 15.0
    evolution_allow_apply_live: bool = False
    evolution_seed: Optional[int] = None
    evolution_min_trades: int = 1
    evolution_ga_mutation_prob: float = 0.2
    evolution_ga_mutation_sigma: float = 0.15
    evolution_ga_crossover_prob: float = 0.7
    evolution_early_stop_generations: int = 0
    evolution_early_stop_threshold: float = 0.001
    evolution_fitness_cache_size: int = 0
    evolution_parallel_fitness_workers: int = 0
    evolution_walk_forward_train_ratio: Optional[float] = None
    evolution_negative_return_penalty_weight: float = 0.0
    evolution_use_composite_fitness: bool = False
    evolution_backup_before_apply: bool = True
    evolution_version_history_size: int = 0
    evolution_multi_timeframes: Optional[List[str]] = None
    evolution_multi_timeframe_weights: Optional[List[float]] = None
    evolution_overfit_penalty_weight: float = 0.0
    evolution_volatility_penalty_weight: float = 0.0
    evolution_composite_calmar_weight: float = 0.0
    evolution_strategy_whitelist_override: Optional[List[str]] = None
    evolution_allocator_decay_after_apply: Optional[float] = None
    use_evolution_strategy_reward: bool = False

    # Strategy pack
    strategies_enabled: List[str] = field(default_factory=lambda: ["hunter", "farmer", "shadow"])
    strategies_max_extra_signals: int = 2
    strategy_whitelist: List[str] = field(default_factory=list)
    use_regime_lstm_boost: bool = False
    use_volatility_regime_scale: bool = False
    volatility_regime_high_threshold: float = 0.02
    use_funding_rate_filter: bool = False
    funding_rates_url: Optional[str] = None
    funding_rate_skip_long_threshold: float = 0.0001
    regime_filter_enabled: bool = False
    regime_filter_trend_strategies: List[str] = field(default_factory=lambda: ["trend_following", "quantum_trend_following_elite"])
    regime_filter_mr_strategies: List[str] = field(default_factory=lambda: ["mean_reversion", "quantum_mean_reversion_elite"])

    # Strategy library
    strategy_library_enabled: bool = True
    strategy_library_modes: List[str] = field(default_factory=lambda: ["paper", "backtest"])
    strategy_library_strategies_enabled: List[str] = field(
        default_factory=lambda: [
            "momentum", "mean_reversion", "trend_following", "pairs_trading",
            "market_making", "arbitrage", "candlestick_patterns", "high_freq_grid",
            "regime_switching", "stat_arb", "factor_investing", "cross_exchange_arb",
            "absolute_tier", "akashic_tier", "apeiron_tier", "chronos_tier",
            "omega_tier", "paradox_tier", "singularity_tier", "source_tier",
            "thanatos_tier", "void_tier",
            "quantum_momentum_elite", "quantum_mean_reversion_elite",
            "quantum_trend_following_elite", "quantum_breakout_elite",
            "quantum_portfolio_rotation_elite", "quantum_arbitrage_elite",
        ]
    )

    # Quantum / quant-fund optional features
    quantum_features_enabled: bool = True
    quantum_features_modes: List[str] = field(default_factory=lambda: ["paper", "backtest"])
    quantum_consciousness_enabled: bool = True
    quantum_method: str = "quantum_approximate"
    quantum_strength: float = 1.0

    quant_fund_upgrades_enabled: bool = True
    quant_fund_upgrades_modes: List[str] = field(default_factory=lambda: ["paper", "backtest"])
    quant_fund_risk_engine_enabled: bool = True

    # Adaptive behavior
    adaptive_enabled: bool = True
    adaptive_minutes_per_bar: float = 60.0
    adaptive_tuner_alpha: float = 0.15
    adaptive_min_trades_before_bias: int = 3

    # StrategyEngine tunables
    se_buy_rsi: float = 35.0
    se_sell_rsi: float = 65.0
    se_buy_bb: float = 0.30
    se_sell_bb: float = 0.70
    se_trend_rsi_buy: float = 55.0
    se_trend_rsi_sell: float = 45.0

    # Offline optimization
    optimized_params_load: bool = False
    optimized_params_path: str = "data/optimized_params.json"
    optimized_params_timeframe: str = ""
    optimized_params: Optional[Dict[str, Any]] = None
    optimized_params_by_timeframe: Optional[Dict[str, Any]] = None

    # Strategy allocator
    strategy_allocator_enabled: bool = True
    strategy_allocator_modes: List[str] = field(default_factory=lambda: ["paper", "backtest"])
    strategy_allocator_timeframe: str = ""
    strategy_allocator_persist_path: str = "data/strategy_allocator_stats.json"
    strategy_allocator_min_trades_before_bias: int = 5
    strategy_allocator_exploration_c: float = 1.2
    strategy_allocator_ema_alpha: float = 0.15
    strategy_allocator_max_total_signals: int = 5
    strategy_allocator_max_per_strategy: int = 2

    # Edge-vs-cost gate
    edge_cost_gate_enabled: bool = True
    edge_cost_gate_modes: List[str] = field(default_factory=lambda: ["paper", "backtest"])
    edge_cost_gate_buffer_mult: float = 1.25
    edge_cost_gate_min_edge_pct: float = 0.0
    edge_cost_gate_live_buffer_mult: Optional[float] = None
    edge_cost_gate_live_min_edge_pct: Optional[float] = None
    edge_cost_gate_fee_mult: float = 2.0
    edge_cost_gate_slippage_mult: float = 2.0

    # Continuous best-trade scanner (peak mode)
    continuous_scan_enabled: bool = True
    continuous_scan_interval_seconds: float = 10.0
    continuous_scan_top_n: int = 5
    continuous_scan_use_cached_best: bool = True
    continuous_scan_max_age_seconds: float = 30.0
    continuous_scan_parallel_sources: bool = True
    continuous_scan_use_liquidity_boost: bool = True
    continuous_scan_liquidity_spread_pct_cap: float = 0.05
    continuous_scan_diversity_max_per_symbol: int = 2
    continuous_scan_diversity_max_per_strategy: int = 2
    continuous_scan_adaptive_interval_enabled: bool = True
    continuous_scan_min_interval_seconds: float = 5.0
    continuous_scan_max_interval_seconds: float = 30.0
    continuous_scan_max_symbols_per_scan: int = 25
    signal_multi_timeframe_enabled: bool = False
    signal_primary_timeframe: str = "1h"
    signal_entry_timeframe: str = "15m"
    external_alpha_enabled: bool = False
    external_alpha_url: str = ""
    external_alpha_timeout_seconds: float = 5.0
    strategy_plugin_modules: List[str] = field(default_factory=list)
    dynamic_universe_enabled: bool = False
    dynamic_universe_interval_cycles: int = 0
    dynamic_universe_top_n: int = 15
    paper_trading_peak_mode: bool = True
    paper_trading_overrides: Dict[str, Any] = field(default_factory=dict)
    paper_simulates_live: bool = False

    # Self-improvement loop
    self_improvement_enabled: bool = True
    self_improvement_modes: List[str] = field(default_factory=lambda: ["paper", "backtest"])
    self_improvement_tick_seconds: int = 1
    self_improvement_shadow_interval_minutes: int = 240
    self_improvement_interval_minutes: int = 240
    self_improvement_shadow_tune_enabled: bool = True
    self_improvement_shadow_tune_days_total: int = 30
    self_improvement_shadow_tune_train_days: int = 10
    self_improvement_shadow_tune_test_days: int = 5
    self_improvement_shadow_tune_evals: int = 6
    self_improvement_shadow_tune_warmup: int = 50
    self_improvement_shadow_tune_timeframe: str = "1h"
    self_improvement_shadow_tune_top: int = 10
    self_improvement_apply_on_improvement_only: bool = True
    self_improvement_min_delta_return_pct: float = 0.10
    self_improvement_max_drawdown_pct: float = 2.0
    self_improvement_min_trades: int = 3
    self_improvement_state_path: str = "data/self_improvement_state.json"
    self_improvement_try_quantum_on_off: bool = True
    self_improvement_apply_quantum_choice: bool = True
    self_improvement_validation_timeframes: List[str] = field(default_factory=lambda: ["1h", "15m"])
    self_improvement_promotion_min_delta_score: float = 0.10
    self_improvement_promotion_require_all_timeframes: bool = True

    # Adaptive universe selection
    adaptive_universe_enabled: bool = True
    adaptive_universe_modes: List[str] = field(default_factory=lambda: ["paper", "backtest", "live"])
    adaptive_universe_top_n: int = 10
    adaptive_universe_max_active: int = 5
    adaptive_universe_min_hold_cycles: int = 20
    adaptive_universe_state_path: str = "data/adaptive_universe.json"

    # Latency optimization
    fast_mode: bool = False
    latency_ws_order_preference: bool = True
    latency_tick_to_trade_threshold_ms: float = 500.0
    latency_ping_interval_s: float = 30.0
    latency_connection_pool_enabled: bool = True
    latency_fire_and_forget_enabled: bool = False

    # Emergency shutdown
    emergency_stop_enabled: bool = True
    max_consecutive_losses: int = 5
    max_error_rate: float = 0.10
    auto_reduce_after_n_losses: int = 0
    auto_reduce_factor: float = 0.6

    # ------------------------------------------------------------------ #
    #  YAML loaders                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_unified_yaml(cls, path: str = "unified_config.yaml") -> "UnifiedConfig":
        """
        Load the runtime configuration from `unified_config.yaml`.

        Best-effort mapper — the YAML may contain more keys than this
        dataclass models; extra keys are silently ignored.
        """
        try:
            p = Path(path)
            if not p.exists():
                return cls()
            with open(p, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f) or {}
            return cls.from_unified_yaml_dict(y)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e} (using defaults)")
            return cls()

    @classmethod
    def from_unified_yaml_dict(cls, y: Dict[str, Any]) -> "UnifiedConfig":
        def _parse_correlation_matrix(raw: Any) -> Optional[Dict[Tuple[str, str], float]]:
            """Convert YAML correlation_matrix to dict (sym_a, sym_b) -> corr."""
            if raw is None:
                return None
            if isinstance(raw, dict):
                out: Dict[Tuple[str, str], float] = {}
                for k, v in raw.items():
                    try:
                        if isinstance(k, (list, tuple)) and len(k) >= 2:
                            key = (str(k[0]), str(k[1]))
                        elif isinstance(k, str) and "|" in k:
                            a, b = k.split("|", 1)
                            key = (a.strip(), b.strip())
                        else:
                            continue
                        out[key] = float(v)
                    except (TypeError, ValueError):
                        continue
                return out if out else None
            return None

        capital = y.get("capital", {}) or {}
        risk = y.get("risk", {}) or {}
        es = risk.get("emergency_shutdown") or {}
        exchanges = y.get("exchanges", {}) or {}
        execution = y.get("execution", {}) or {}
        execution_engine = y.get("execution_engine", {}) or {}
        execution_alpha_engine = y.get("execution_alpha_engine", {}) or {}
        execution_alpha_legacy = y.get("execution_alpha", {}) or {}
        data_cfg = y.get("data", {}) or {}
        ai = y.get("ai_brain", {}) or {}
        fx = y.get("fx", {}) or {}
        monitoring = y.get("monitoring", {}) or {}
        multi_language = y.get("multi_language", {}) or {}
        runtime = y.get("runtime", {}) or {}
        command_bus_cfg = (runtime.get("command_bus", {}) or {}) if isinstance(runtime, dict) else {}
        execution_mesh_cfg = (runtime.get("execution_mesh", {}) or {}) if isinstance(runtime, dict) else {}
        strategies = y.get("strategies", {}) or {}
        strategy_library = y.get("strategy_library", {}) or {}
        quantum_features = y.get("quantum_features", {}) or {}
        quant_fund_upgrades = y.get("quant_fund_upgrades", {}) or {}
        adaptive = y.get("adaptive", {}) or {}
        optimization = y.get("optimization", {}) or {}
        strategy_engine = y.get("strategy_engine", {}) or {}
        strategy_allocator = y.get("strategy_allocator", {}) or {}
        edge_gate = y.get("edge_cost_gate", {}) or {}
        portfolio_target_engine = y.get("portfolio_target_engine", {}) or {}
        liquidity_risk_engine = y.get("liquidity_risk_engine", {}) or {}
        strategy_evaluation_engine = y.get("strategy_evaluation_engine", {}) or {}
        self_optimizing_meta_engine = y.get("self_optimizing_meta_engine", {}) or {}
        champion_challenger = y.get("champion_challenger", {}) or {}
        market_microstructure_engine = y.get("market_microstructure_engine", {}) or {}
        recon_recovery_engine = y.get("recon_recovery_engine", {}) or {}
        system_health_metrics = y.get("system_health_metrics", {}) or {}
        runtime_safety = y.get("runtime_safety", {}) or {}
        market_data = y.get("market_data", {}) or {}
        continuous_scan = y.get("continuous_scan", {}) or {}
        paper_trading = y.get("paper_trading", {}) or {}
        evolution = y.get("evolution", {}) or {}
        self_impr = y.get("self_improvement", {}) or {}
        adaptive_universe = y.get("adaptive_universe", {}) or {}
        prom = (monitoring.get("prometheus", {}) or {}) if isinstance(monitoring, dict) else {}
        graf = (monitoring.get("grafana", {}) or {}) if isinstance(monitoring, dict) else {}

        # Backward compatible: accept legacy `coinbase_pro`
        secondary = str(exchanges.get("secondary", "coinbase_advanced"))
        supported = list(exchanges.get("supported", ["kraken", "coinbase_advanced"]))
        supported = ["coinbase_advanced" if s == "coinbase_pro" else s for s in supported]
        if secondary == "coinbase_pro":
            secondary = "coinbase_advanced"

        kraken = exchanges.get("kraken", {}) or {}
        coinbase = exchanges.get("coinbase_advanced", {}) or exchanges.get("coinbase_pro", {}) or {}
        cc_weights_cfg = (
            champion_challenger.get("promotion_weights", {})
            if isinstance(champion_challenger, dict)
            else {}
        ) or {}
        cc_weights = {
            "net_pnl": float(cc_weights_cfg.get("net_pnl", 1.0) or 1.0),
            "expectancy": float(cc_weights_cfg.get("expectancy", 1.0) or 1.0),
            "profit_factor": float(cc_weights_cfg.get("profit_factor", 0.75) or 0.75),
            "sharpe_like": float(cc_weights_cfg.get("sharpe_like", 1.0) or 1.0),
            "drawdown_penalty": float(cc_weights_cfg.get("drawdown_penalty", 1.25) or 1.25),
            "fee_penalty": float(cc_weights_cfg.get("fee_penalty", 0.5) or 0.5),
        }

        return cls(
            config_version=int(y.get("config_version", 1) or 1),
            # Capital
            starting_capital_aud=float(capital.get("starting_capital_aud", 1000.0)),
            currency=str(capital.get("currency", "AUD")),
            aud_to_usd=float(fx.get("aud_to_usd", 0.65)),
            # Exchanges
            primary_exchange=str(exchanges.get("primary", "kraken")),
            secondary_exchange=secondary,
            supported_exchanges=supported,
            # Position sizing
            min_position_size_aud=float(capital.get("min_position_size_aud", 10.0)),
            max_position_size_aud=float(capital.get("max_position_size_aud", 120.0)),
            max_position_pct=float(capital.get("max_position_pct", 0.12)),
            max_total_exposure_pct=float(capital.get("max_total_exposure_pct", 0.50)),
            max_concurrent_positions=int(capital.get("max_concurrent_positions", 4)),
            # Risk
            max_daily_loss_pct=float(risk.get("max_daily_loss_pct", 0.05)),
            max_drawdown_pct=float(risk.get("max_drawdown_pct", 0.15)),
            stop_loss_pct=float(risk.get("stop_loss_pct", 0.02)),
            take_profit_pct=float(risk.get("take_profit_pct", 0.05)),
            max_consecutive_losses=int(risk.get("max_consecutive_losses", 5)),
            max_error_rate=float(risk.get("max_error_rate", 0.10)),
            auto_reduce_after_n_losses=int(risk.get("auto_reduce_after_n_losses", 0) or 0),
            auto_reduce_factor=float(risk.get("auto_reduce_factor", 0.6) or 0.6),
            use_volatility_adjusted_limits=bool(risk.get("use_volatility_adjusted_limits", False)),
            realized_vol_pct=float(risk.get("realized_vol_pct", 0.0) or 0.0),
            portfolio_var_limit_pct=float(risk.get("portfolio_var_limit_pct", 0.0) or 0.0),
            portfolio_cvar_limit_pct=float(risk.get("portfolio_cvar_limit_pct", 0.0) or 0.0),
            portfolio_var_confidence=float(risk.get("portfolio_var_confidence", 0.95) or 0.95),
            portfolio_var_lookback_trades=int(risk.get("portfolio_var_lookback_trades", 50) or 50),
            cluster_drawdown_brake_pct=float(risk.get("cluster_drawdown_brake_pct", 0.0) or 0.0),
            target_cluster_cap_pct=float(risk.get("target_cluster_cap_pct", 0.40) or 0.40),
            risk_cluster_map=dict(risk.get("risk_cluster_map", {}) or {}),
            portfolio_vol_target_pct=float(risk.get("portfolio_vol_target_pct", 2.0) or 2.0),
            portfolio_liquidity_spread_ref_bps=float(risk.get("portfolio_liquidity_spread_ref_bps", 20.0) or 20.0),
            portfolio_exposure_min_scale=float(risk.get("portfolio_exposure_min_scale", 0.30) or 0.30),
            emergency_shutdown_enabled=bool(es.get("enabled", False)),
            emergency_shutdown_latency_spike_ms=float(es["latency_spike_ms"]) if es.get("latency_spike_ms") is not None else None,
            emergency_shutdown_flash_crash_pct=float(es["flash_crash_pct"]) if es.get("flash_crash_pct") is not None else None,
            emergency_shutdown_network_fail=bool(es.get("network_fail", False)),
            emergency_shutdown_arb_spread_bps=float(es["arb_spread_bps"]) if es.get("arb_spread_bps") is not None else None,
            # Fees/slippage
            kraken_maker_fee=float(kraken.get("maker_fee", 0.0016)),
            kraken_taker_fee=float(kraken.get("taker_fee", 0.0026)),
            coinbase_maker_fee=float(coinbase.get("maker_fee", 0.005)),
            coinbase_taker_fee=float(coinbase.get("taker_fee", 0.005)),
            slippage_pct=float(execution.get("slippage_pct", 0.001)),
            order_type=str(execution_engine.get("order_type", "market")),
            retry_attempts=int(execution_engine.get("retry_attempts", 3)),
            retry_delay_seconds=float(execution_engine.get("retry_delay_seconds", 5)),
            max_slippage_pct=float(execution_engine.get("max_slippage_pct", 0.01)),
            signal_cooldown_bars=int(execution.get("signal_cooldown_bars", 0) or 0),
            vwap_large_order_threshold_aud=float(execution_engine.get("vwap_large_order_threshold_aud", 80.0)),
            use_twap_for_large_orders=bool(execution_engine.get("use_twap_for_large_orders", False)),
            order_fill_timeout_seconds=float(execution_engine.get("order_fill_timeout_seconds", 0) or 0),
            max_spread_bps=float(execution_engine.get("max_spread_bps", 0) or 0),
            use_is_gate=bool(execution_engine.get("use_is_gate", False)),
            max_avg_is_bps=float(execution_engine.get("max_avg_is_bps", 0) or 0),
            portfolio_weight_method=str(execution_engine.get("portfolio_weight_method", "hrp") or "hrp"),
            multi_venue_enabled=bool(execution_engine.get("multi_venue_enabled", True)),
            multi_venue_min_notional_aud=float(execution_engine.get("multi_venue_min_notional_aud", 200.0)),
            twap_min_notional_usd=float(execution_engine.get("twap_min_notional_usd", 250.0) or 250.0),
            twap_duration_minutes=float(execution_engine.get("twap_duration_minutes", 5.0) or 5.0),
            use_venue_routing_by_spread=bool(execution_engine.get("use_venue_routing_by_spread", False)),
            dca_levels_pct=list(execution_engine.get("dca_levels_pct") or []) if isinstance(execution_engine.get("dca_levels_pct"), (list, tuple)) else None,
            use_correlation_aware_sizing=bool(execution_engine.get("use_correlation_aware_sizing", False)),
            max_correlated_exposure=float(execution_engine.get("max_correlated_exposure", 0.6) or 0.6),
            correlation_matrix=_parse_correlation_matrix(execution_engine.get("correlation_matrix")),
            persist_tick_store=bool(data_cfg.get("persist_tick_store", True)),
            use_lake_read=bool(data_cfg.get("use_lake_read", True)),
            persist_to_lake=bool(data_cfg.get("persist_to_lake", True)),
            persist_to_tick_store=bool(data_cfg.get("persist_to_tick_store", True)),
            lake_path=str(data_cfg.get("lake_path", "data/lake") or "data/lake"),
            # Pairs
            trading_pairs=list(y.get("trading_pairs", cls().trading_pairs)),
            # AI
            ai_enabled=bool(ai.get("enabled", True)),
            consciousness_enabled=bool(ai.get("consciousness_enabled", True)),
            swarm_intelligence_enabled=bool(ai.get("swarm_intelligence_enabled", True)),
            num_ai_agents=int(ai.get("num_ai_agents", cls().num_ai_agents)),
            min_signal_confidence=float(ai.get("min_signal_confidence", 0.75)),
            live_min_signal_confidence=float(ai["live_min_signal_confidence"]) if ai.get("live_min_signal_confidence") is not None else None,
            max_concurrent_signals=int(ai.get("max_concurrent_signals", 2)),
            # Monitoring
            prometheus_enabled=bool(prom.get("enabled", True)),
            grafana_enabled=bool(graf.get("enabled", True)),
            prometheus_port=int(prom.get("port", cls().prometheus_port)),
            grafana_port=int(graf.get("port", cls().grafana_port)),
            # Multi-language
            multi_language_enabled=bool(multi_language.get("enabled", True)),
            multi_language_endpoints=dict(multi_language.get("endpoints", {}) or {}),
            use_cycle_aggregate_boost=bool(multi_language.get("use_cycle_aggregate_boost", True)),
            use_conservative_cycle_boost=bool(multi_language.get("use_conservative_cycle_boost", False)),
            use_weighted_mean_boost=bool(multi_language.get("use_weighted_mean_boost", False)),
            use_risk_all=bool(multi_language.get("use_risk_all", False)),
            use_conservative_risk=bool(multi_language.get("use_conservative_risk", False)),
            use_regime_estimate=bool(multi_language.get("use_regime_estimate", False)),
            use_slippage_estimate=bool(multi_language.get("use_slippage_estimate", False)),
            use_drawdown_check=bool(multi_language.get("use_drawdown_check", False)),
            use_position_sizing_gate=bool(multi_language.get("use_position_sizing_gate", False)),
            use_signal_filter_gate=bool(multi_language.get("use_signal_filter_gate", False)),
            max_slippage_bps=float(multi_language.get("max_slippage_bps", 100.0) or 100.0),
            multi_language_task_timeouts=(
                dict((k, float(v)) for k, v in (multi_language.get("task_timeouts") or {}).items() if isinstance(v, (int, float)))
                if multi_language.get("task_timeouts") else None
            ),
            multi_language_warm_on_start=bool(multi_language.get("warm_on_start", False)),
            # Runtime
            run_mode=str(runtime.get("mode", "paper") or "paper"),
            node_role=str(runtime.get("node_role", "single-node") or "single-node"),
            command_bus_enabled=bool(command_bus_cfg.get("enabled", False)),
            command_bus_db_path=str(command_bus_cfg.get("db_path", "data/command_bus.db") or "data/command_bus.db"),
            command_bus_queue=str(command_bus_cfg.get("queue", "default") or "default"),
            command_bus_hmac_key_env=str(command_bus_cfg.get("hmac_key_env", "ARGUS_COMMAND_HMAC_KEY") or "ARGUS_COMMAND_HMAC_KEY"),
            command_bus_instruction_ttl_seconds=float(command_bus_cfg.get("instruction_ttl_seconds", 5.0) or 5.0),
            command_bus_require_signature=bool(command_bus_cfg.get("require_signature", True)),
            command_bus_max_batch=int(command_bus_cfg.get("max_batch", 64) or 64),
            command_bus_max_notional_aud=float(command_bus_cfg.get("max_notional_aud", 0.0) or 0.0),
            execution_mesh_enabled=bool(execution_mesh_cfg.get("enabled", False)),
            execution_mesh_max_lanes=int(execution_mesh_cfg.get("max_lanes", 8) or 8),
            execution_mesh_max_queue_per_lane=int(execution_mesh_cfg.get("max_queue_per_lane", 128) or 128),
            execution_mesh_batch_size=int(execution_mesh_cfg.get("batch_size", 8) or 8),
            execution_mesh_parallel_lanes=bool(execution_mesh_cfg.get("parallel_lanes", True)),
            execution_mesh_halt_on_lane_error=bool(execution_mesh_cfg.get("halt_on_lane_error", True)),
            execution_mesh_symbols=list(execution_mesh_cfg.get("symbols", []) or []),
            live_disabled_strategies=list(runtime.get("live_disabled_strategies", []) or []),
            live_require_paper_edge=bool(runtime.get("live_require_paper_edge", False)),
            live_min_trades_paper=int(runtime.get("live_min_trades_paper", 20) or 20),
            live_min_win_rate_pct=float(runtime.get("live_min_win_rate_pct", 45) or 45),
            # Evolution
            evolution_load_evolved=bool(evolution.get("load_evolved", False)),
            evolution_params_path=str(evolution.get("params_path", "data/evolved_params.json") or "data/evolved_params.json"),
            evolution_continuous_enabled=bool(evolution.get("continuous_enabled", True)),
            evolution_interval_hours=float(evolution.get("interval_hours", 24.0) or 24.0),
            evolution_fitness_days=int(evolution.get("fitness_days", 7) or 7),
            evolution_generations=int(evolution.get("generations", 5) or 5),
            evolution_population_size=int(evolution.get("population_size", 12) or 12),
            evolution_auto_apply=bool(evolution.get("auto_apply", True)),
            evolution_realtime_interval_minutes=float(evolution.get("realtime_interval_minutes", 60.0) or 60.0),
            evolution_realtime_fitness_days=float(evolution.get("realtime_fitness_days", 1.0) or 1.0),
            evolution_use_live_feed=bool(evolution.get("use_live_feed", True)),
            evolution_realtime_generations=int(evolution.get("realtime_generations", 3) or 3),
            evolution_realtime_population_size=int(evolution.get("realtime_population_size", 8) or 8),
            evolution_min_bars_for_live_feed=int(evolution.get("min_bars_for_live_feed", 20) or 20),
            evolution_run_with_market=bool(evolution.get("run_with_market", True)),
            evolution_trigger_on_trade=bool(evolution.get("trigger_on_trade", False)),
            evolution_after_n_trades=int(evolution.get("after_n_trades", 5) or 5),
            evolution_dry_run=bool(evolution.get("dry_run", False)),
            evolution_debounce_minutes=float(evolution.get("debounce_minutes", 15.0) or 15.0),
            evolution_allow_apply_live=bool(evolution.get("allow_apply_live", False)),
            evolution_seed=int(evolution["seed"]) if evolution.get("seed") is not None else None,
            evolution_min_trades=int(evolution.get("min_trades", 1) or 1),
            evolution_ga_mutation_prob=float((evolution.get("ga") or {}).get("mutation_prob", 0.2) or 0.2),
            evolution_ga_mutation_sigma=float((evolution.get("ga") or {}).get("mutation_sigma", 0.15) or 0.15),
            evolution_ga_crossover_prob=float((evolution.get("ga") or {}).get("crossover_prob", 0.7) or 0.7),
            evolution_early_stop_generations=int(evolution.get("early_stop_generations", 0) or 0),
            evolution_early_stop_threshold=float(evolution.get("early_stop_threshold", 0.001) or 0.001),
            evolution_fitness_cache_size=int(evolution.get("fitness_cache_size", 0) or 0),
            evolution_parallel_fitness_workers=int(evolution.get("parallel_fitness_workers", 0) or 0),
            evolution_walk_forward_train_ratio=float(evolution["walk_forward_train_ratio"]) if evolution.get("walk_forward_train_ratio") is not None else None,
            evolution_negative_return_penalty_weight=float(evolution.get("negative_return_penalty_weight", 0.0) or 0.0),
            evolution_use_composite_fitness=bool(evolution.get("use_composite_fitness", False)),
            evolution_backup_before_apply=bool(evolution.get("backup_before_apply", True)),
            evolution_version_history_size=int(evolution.get("version_history_size", 0) or 0),
            evolution_multi_timeframes=list(evolution.get("multi_timeframes") or []) if evolution.get("multi_timeframes") is not None else None,
            evolution_multi_timeframe_weights=list(evolution.get("multi_timeframe_weights") or []) if evolution.get("multi_timeframe_weights") is not None else None,
            evolution_overfit_penalty_weight=float(evolution.get("overfit_penalty_weight", 0) or 0),
            evolution_volatility_penalty_weight=float(evolution.get("volatility_penalty_weight", 0) or 0),
            evolution_composite_calmar_weight=float(evolution.get("composite_calmar_weight", 0) or 0),
            evolution_strategy_whitelist_override=list(evolution.get("strategy_whitelist_override") or []) if evolution.get("strategy_whitelist_override") is not None else None,
            evolution_allocator_decay_after_apply=float(evolution.get("allocator_decay_after_apply")) if evolution.get("allocator_decay_after_apply") is not None else None,
            use_evolution_strategy_reward=bool(evolution.get("use_evolution_strategy_reward", False)),
            # Strategies
            strategies_enabled=list(strategies.get("enabled", ["hunter", "farmer", "shadow"]) or ["hunter", "farmer", "shadow"]),
            strategies_max_extra_signals=int(strategies.get("max_extra_signals", 2) or 2),
            strategy_whitelist=list(strategies.get("strategy_whitelist", []) or []),
            use_regime_lstm_boost=bool(strategies.get("use_regime_lstm_boost", False)),
            use_volatility_regime_scale=bool(strategies.get("use_volatility_regime_scale", False)),
            volatility_regime_high_threshold=float(strategies.get("volatility_regime_high_threshold", 0.02) or 0.02),
            use_funding_rate_filter=bool(strategies.get("use_funding_rate_filter", False)),
            funding_rates_url=str(strategies.get("funding_rates_url") or "") or None,
            funding_rate_skip_long_threshold=float(strategies.get("funding_rate_skip_long_threshold", 0.0001) or 0.0001),
            regime_filter_enabled=bool(strategies.get("regime_filter_enabled", False)),
            regime_filter_trend_strategies=list(strategies.get("regime_filter_trend_strategies", []) or []),
            regime_filter_mr_strategies=list(strategies.get("regime_filter_mr_strategies", []) or []),
            # Strategy library
            strategy_library_enabled=bool(strategy_library.get("enabled", True)),
            strategy_library_modes=list(strategy_library.get("modes", ["paper", "backtest"]) or ["paper", "backtest"]),
            strategy_library_strategies_enabled=list(
                strategy_library.get("enabled_strategies", cls().strategy_library_strategies_enabled)
                or cls().strategy_library_strategies_enabled
            ),
            # Quantum features
            quantum_features_enabled=bool(quantum_features.get("enabled", True)),
            quantum_features_modes=list(quantum_features.get("modes", ["paper", "backtest"]) or ["paper", "backtest"]),
            quantum_consciousness_enabled=bool((quantum_features.get("quantum_consciousness", {}) or {}).get("enabled", True)),
            quantum_method=str((quantum_features.get("quantum_optimization", {}) or {}).get("method", "quantum_approximate") or "quantum_approximate"),
            quantum_strength=float((quantum_features.get("quantum_optimization", {}) or {}).get("strength", 1.0) or 1.0),
            # Quant fund upgrades
            quant_fund_upgrades_enabled=bool(quant_fund_upgrades.get("enabled", True)),
            quant_fund_upgrades_modes=list(quant_fund_upgrades.get("modes", ["paper", "backtest"]) or ["paper", "backtest"]),
            quant_fund_risk_engine_enabled=bool((quant_fund_upgrades.get("risk_engine", {}) or {}).get("enabled", True)),
            # Adaptive
            adaptive_enabled=bool(adaptive.get("enabled", True)),
            adaptive_minutes_per_bar=float(adaptive.get("minutes_per_bar", 60.0) or 60.0),
            adaptive_tuner_alpha=float(adaptive.get("tuner_alpha", 0.15) or 0.15),
            adaptive_min_trades_before_bias=int(adaptive.get("min_trades_before_bias", 3) or 3),
            # StrategyEngine tunables
            se_buy_rsi=float(strategy_engine.get("se_buy_rsi", cls().se_buy_rsi) or cls().se_buy_rsi),
            se_sell_rsi=float(strategy_engine.get("se_sell_rsi", cls().se_sell_rsi) or cls().se_sell_rsi),
            se_buy_bb=float(strategy_engine.get("se_buy_bb", cls().se_buy_bb) or cls().se_buy_bb),
            se_sell_bb=float(strategy_engine.get("se_sell_bb", cls().se_sell_bb) or cls().se_sell_bb),
            se_trend_rsi_buy=float(strategy_engine.get("se_trend_rsi_buy", cls().se_trend_rsi_buy) or cls().se_trend_rsi_buy),
            se_trend_rsi_sell=float(strategy_engine.get("se_trend_rsi_sell", cls().se_trend_rsi_sell) or cls().se_trend_rsi_sell),
            # Offline optimization
            optimized_params_load=bool(optimization.get("optimized_params_load", False)),
            optimized_params_path=str(optimization.get("optimized_params_path", "data/optimized_params.json") or "data/optimized_params.json"),
            optimized_params_timeframe=str(optimization.get("optimized_params_timeframe", "") or ""),
            # Strategy allocator
            strategy_allocator_enabled=bool(strategy_allocator.get("enabled", True)),
            strategy_allocator_modes=list(strategy_allocator.get("modes", ["paper", "backtest"]) or ["paper", "backtest"]),
            strategy_allocator_timeframe=str(strategy_allocator.get("timeframe", "") or ""),
            strategy_allocator_persist_path=str(strategy_allocator.get("persist_path", "data/strategy_allocator_stats.json") or "data/strategy_allocator_stats.json"),
            strategy_allocator_min_trades_before_bias=int(strategy_allocator.get("min_trades_before_bias", 5) or 5),
            strategy_allocator_exploration_c=float(strategy_allocator.get("exploration_c", 1.2) or 1.2),
            strategy_allocator_ema_alpha=float(strategy_allocator.get("ema_alpha", 0.15) or 0.15),
            strategy_allocator_max_total_signals=int(strategy_allocator.get("max_total_signals", 5) or 5),
            strategy_allocator_max_per_strategy=int(strategy_allocator.get("max_per_strategy", 2) or 2),
            # Edge/cost gate
            edge_cost_gate_enabled=bool(edge_gate.get("enabled", True)),
            edge_cost_gate_modes=list(edge_gate.get("modes", ["paper", "backtest"]) or ["paper", "backtest"]),
            edge_cost_gate_buffer_mult=float(edge_gate.get("buffer_mult", 1.25) or 1.25),
            edge_cost_gate_min_edge_pct=float(edge_gate.get("min_edge_pct", 0.0) or 0.0),
            edge_cost_gate_live_buffer_mult=float(edge_gate["live_buffer_mult"]) if edge_gate.get("live_buffer_mult") is not None else None,
            edge_cost_gate_live_min_edge_pct=float(edge_gate["live_min_edge_pct"]) if edge_gate.get("live_min_edge_pct") is not None else None,
            edge_cost_gate_fee_mult=float(edge_gate.get("fee_mult", 2.0) or 2.0),
            edge_cost_gate_slippage_mult=float(edge_gate.get("slippage_mult", 2.0) or 2.0),
            # Portfolio target engine
            targets_enabled=bool(portfolio_target_engine.get("enabled", False)),
            target_convergence_alpha=float(portfolio_target_engine.get("convergence_alpha", 1.0) or 1.0),
            target_rebalance_min_delta_pct=float(portfolio_target_engine.get("rebalance_min_delta_pct", 0.02) or 0.02),
            target_score_confidence_weight=float(portfolio_target_engine.get("score_confidence_weight", 1.0) or 1.0),
            target_score_net_edge_weight=float(portfolio_target_engine.get("score_net_edge_weight", 1.0) or 1.0),
            target_regime_boost_enabled=bool(portfolio_target_engine.get("regime_boost_enabled", True)),
            # Liquidity risk engine
            liquidity_risk_enabled=bool(liquidity_risk_engine.get("enabled", True)),
            liquidity_risk_depth_fraction_limit=float(liquidity_risk_engine.get("depth_fraction_limit", 0.04) or 0.04),
            liquidity_risk_thin_spread_threshold_bps=float(liquidity_risk_engine.get("thin_spread_threshold_bps", 6.0) or 6.0),
            liquidity_risk_danger_spread_threshold_bps=float(liquidity_risk_engine.get("danger_spread_threshold_bps", 12.0) or 12.0),
            liquidity_risk_min_depth_threshold=float(liquidity_risk_engine.get("min_depth_threshold", 0.5) or 0.5),
            liquidity_risk_slippage_threshold_bps=float(liquidity_risk_engine.get("slippage_threshold_bps", 10.0) or 10.0),
            liquidity_risk_min_liquidity_score=float(liquidity_risk_engine.get("min_liquidity_score", 0.2) or 0.2),
            liquidity_risk_score_weights={
                "depth": float(((liquidity_risk_engine.get("score_weights", {}) or {}).get("depth", 1.0) or 1.0)),
                "spread": float(((liquidity_risk_engine.get("score_weights", {}) or {}).get("spread", 1.0) or 1.0)),
                "fill_ratio": float(((liquidity_risk_engine.get("score_weights", {}) or {}).get("fill_ratio", 0.75) or 0.75)),
            },
            # Strategy Evaluation Engine v1
            strategy_evaluation_enabled=bool(strategy_evaluation_engine.get("enabled", True)),
            strategy_evaluation_persist_interval_cycles=int(strategy_evaluation_engine.get("persist_interval_cycles", 10) or 10),
            strategy_evaluation_min_trades_for_ranking=int(strategy_evaluation_engine.get("min_trades_for_ranking", 5) or 5),
            strategy_evaluation_use_regime_scoped_metrics=bool(strategy_evaluation_engine.get("use_regime_scoped_metrics", True)),
            strategy_evaluation_sharpe_like_min_trades=int(strategy_evaluation_engine.get("sharpe_like_min_trades", 5) or 5),
            strategy_evaluation_max_metrics_history_points=int(strategy_evaluation_engine.get("max_metrics_history_points", 500) or 500),
            strategy_evaluation_halt_on_error=bool(strategy_evaluation_engine.get("halt_on_error", False)),
            strategy_evaluation_db_path=str(strategy_evaluation_engine.get("db_path", "data/strategy_metrics.db") or "data/strategy_metrics.db"),
            # Self-Optimizing Meta Engine v1
            self_optimizing_meta_enabled=bool(self_optimizing_meta_engine.get("enabled", True)),
            self_optimizing_meta_advisory_only=bool(self_optimizing_meta_engine.get("advisory_only", False)),
            self_optimizing_meta_update_interval_cycles=int(self_optimizing_meta_engine.get("update_interval_cycles", 10) or 10),
            self_optimizing_meta_min_trades_for_reweighting=int(self_optimizing_meta_engine.get("min_trades_for_reweighting", 5) or 5),
            self_optimizing_meta_alpha=float(self_optimizing_meta_engine.get("meta_alpha", 0.2) or 0.2),
            self_optimizing_meta_max_weight_change_per_update=float(self_optimizing_meta_engine.get("max_weight_change_per_update", 0.10) or 0.10),
            self_optimizing_meta_min_weight_per_strategy=float(self_optimizing_meta_engine.get("min_weight_per_strategy", 0.05) or 0.05),
            self_optimizing_meta_max_weight_per_strategy=float(self_optimizing_meta_engine.get("max_weight_per_strategy", 0.45) or 0.45),
            self_optimizing_meta_baseline_weight_mode=str(self_optimizing_meta_engine.get("baseline_weight_mode", "equal") or "equal"),
            self_optimizing_meta_score_weights={
                "expectancy": float(((self_optimizing_meta_engine.get("score_weights", {}) or {}).get("expectancy", 1.0) or 1.0)),
                "sharpe_like": float(((self_optimizing_meta_engine.get("score_weights", {}) or {}).get("sharpe_like", 1.0) or 1.0)),
                "profit_factor": float(((self_optimizing_meta_engine.get("score_weights", {}) or {}).get("profit_factor", 0.75) or 0.75)),
                "drawdown_penalty": float(((self_optimizing_meta_engine.get("score_weights", {}) or {}).get("drawdown_penalty", 1.0) or 1.0)),
                "fee_penalty": float(((self_optimizing_meta_engine.get("score_weights", {}) or {}).get("fee_penalty", 0.5) or 0.5)),
                "slippage_penalty": float(((self_optimizing_meta_engine.get("score_weights", {}) or {}).get("slippage_penalty", 0.5) or 0.5)),
            },
            self_optimizing_meta_regime_multipliers={
                str(_k): {str(_sk): float(_sv) for _sk, _sv in dict(_v or {}).items()}
                for _k, _v in dict(self_optimizing_meta_engine.get("regime_multipliers", {}) or {}).items()
            },
            self_optimizing_meta_db_path=str(self_optimizing_meta_engine.get("db_path", "data/meta_weights.db") or "data/meta_weights.db"),
            # Champion / Challenger
            champion_challenger_enabled=bool(champion_challenger.get("enabled", True)),
            champion_challenger_advisory_only=bool(champion_challenger.get("advisory_only", True)),
            champion_challenger_min_trades_for_promotion=int(champion_challenger.get("min_trades_for_promotion", 10) or 10),
            champion_challenger_max_drawdown_pct_for_promotion=float(champion_challenger.get("max_drawdown_pct_for_promotion", 0.12) or 0.12),
            champion_challenger_require_expectancy_improvement=bool(champion_challenger.get("require_expectancy_improvement", True)),
            champion_challenger_require_profit_factor_improvement=bool(champion_challenger.get("require_profit_factor_improvement", False)),
            champion_challenger_require_sharpe_like_improvement=bool(champion_challenger.get("require_sharpe_like_improvement", True)),
            champion_challenger_persist_interval_cycles=int(champion_challenger.get("persist_interval_cycles", 10) or 10),
            champion_challenger_db_path=str(champion_challenger.get("db_path", "data/champion_challenger.db") or "data/champion_challenger.db"),
            champion_challenger_artifacts_dir=str(champion_challenger.get("artifacts_dir", "deploy/promotions") or "deploy/promotions"),
            champion_challenger_promotion_weights=dict(cc_weights),
            # Market Microstructure Engine v1
            market_microstructure_enabled=bool(market_microstructure_engine.get("enabled", True)),
            market_microstructure_rolling_window=int(market_microstructure_engine.get("rolling_window", 20) or 20),
            market_microstructure_vacuum_spread_jump_bps=float(market_microstructure_engine.get("vacuum_spread_jump_bps", 4.0) or 4.0),
            market_microstructure_vacuum_depth_drop_ratio=float(market_microstructure_engine.get("vacuum_depth_drop_ratio", 0.5) or 0.5),
            market_microstructure_high_adverse_selection_threshold=float(market_microstructure_engine.get("high_adverse_selection_threshold", 0.7) or 0.7),
            market_microstructure_use_in_execution_alpha=bool(market_microstructure_engine.get("use_in_execution_alpha", True)),
            market_microstructure_use_in_liquidity_risk=bool(market_microstructure_engine.get("use_in_liquidity_risk", True)),
            # Recon-Required Recovery Engine v1
            recon_recovery_enabled=bool(recon_recovery_engine.get("enabled", True)),
            recon_recovery_stale_threshold_seconds=float(recon_recovery_engine.get("stale_threshold_seconds", 60.0) or 60.0),
            recon_recovery_base_retry_delay_seconds=float(recon_recovery_engine.get("base_retry_delay_seconds", 5.0) or 5.0),
            recon_recovery_max_retries=int(recon_recovery_engine.get("max_retries", 5) or 5),
            recon_recovery_halt_on_retry_exhausted=bool(recon_recovery_engine.get("halt_on_retry_exhausted", True)),
            # System Health Metrics
            system_health_metrics_enabled=bool(system_health_metrics.get("enabled", True)),
            system_health_metrics_snapshot_interval_cycles=int(system_health_metrics.get("snapshot_interval_cycles", 10) or 10),
            # Execution Alpha Engine v2
            execution_alpha_enabled=bool(execution_alpha_engine.get("enabled", True)),
            execution_alpha_maker_spread_threshold_bps=float(
                execution_alpha_engine.get("maker_spread_threshold_bps", execution_alpha_legacy.get("maker_spread_threshold_bps", 2.0)) or 2.0
            ),
            execution_alpha_min_fill_probability=float(
                execution_alpha_engine.get("min_fill_probability", execution_alpha_legacy.get("min_fill_probability", 0.35)) or 0.35
            ),
            execution_alpha_slice_threshold_pct=float(
                execution_alpha_engine.get("slice_threshold_pct", execution_alpha_legacy.get("slice_threshold_pct", 0.03)) or 0.03
            ),
            execution_alpha_maker_fallback_seconds=float(
                execution_alpha_engine.get("maker_fallback_seconds", execution_alpha_legacy.get("maker_timeout_seconds", 8.0)) or 8.0
            ),
            execution_alpha_telemetry_window=int(
                execution_alpha_engine.get("telemetry_window", execution_alpha_legacy.get("telemetry_window", 200)) or 200
            ),
            runtime_safety_latency_grace_cycles=int(runtime_safety.get("latency_grace_cycles", 2) or 2),
            live_safe_disable_pinnacle_ai_brain=bool(runtime_safety.get("live_safe_disable_pinnacle_ai_brain", False)),
            market_data_ohlcv_cache_seconds=float(market_data.get("ohlcv_cache_seconds", 30.0) or 30.0),
            market_data_ohlcv_poll_interval_seconds=float(market_data.get("ohlcv_poll_interval_seconds", 30.0) or 30.0),
            market_data_ohlcv_retry_attempts=int(market_data.get("ohlcv_retry_attempts", 2) or 2),
            # Continuous best-trade scanner
            continuous_scan_enabled=bool(continuous_scan.get("enabled", True)),
            continuous_scan_interval_seconds=float(continuous_scan.get("interval_seconds", 10.0) or 10.0),
            continuous_scan_top_n=int(continuous_scan.get("top_n", 5) or 5),
            continuous_scan_use_cached_best=bool(continuous_scan.get("use_cached_best", True)),
            continuous_scan_max_age_seconds=float(continuous_scan.get("max_age_seconds", 30.0) or 30.0),
            continuous_scan_parallel_sources=bool(continuous_scan.get("parallel_sources", True)),
            continuous_scan_use_liquidity_boost=bool(continuous_scan.get("use_liquidity_boost", True)),
            continuous_scan_liquidity_spread_pct_cap=float(continuous_scan.get("liquidity_spread_pct_cap", 0.05) or 0.05),
            continuous_scan_diversity_max_per_symbol=int(continuous_scan.get("diversity_max_per_symbol", 2) or 2),
            continuous_scan_diversity_max_per_strategy=int(continuous_scan.get("diversity_max_per_strategy", 2) or 2),
            continuous_scan_adaptive_interval_enabled=bool(continuous_scan.get("adaptive_interval_enabled", True)),
            continuous_scan_min_interval_seconds=float(continuous_scan.get("min_interval_seconds", 5.0) or 5.0),
            continuous_scan_max_interval_seconds=float(continuous_scan.get("max_interval_seconds", 30.0) or 30.0),
            continuous_scan_max_symbols_per_scan=int(continuous_scan.get("max_symbols_per_scan", 25) or 25),
            signal_multi_timeframe_enabled=bool(continuous_scan.get("signal_multi_timeframe_enabled", False)),
            signal_primary_timeframe=str(continuous_scan.get("signal_primary_timeframe", "1h") or "1h"),
            signal_entry_timeframe=str(continuous_scan.get("signal_entry_timeframe", "15m") or "15m"),
            external_alpha_enabled=bool(continuous_scan.get("external_alpha_enabled", False)),
            external_alpha_url=str(continuous_scan.get("external_alpha_url", "") or ""),
            external_alpha_timeout_seconds=float(continuous_scan.get("external_alpha_timeout_seconds", 5.0) or 5.0),
            strategy_plugin_modules=list(continuous_scan.get("strategy_plugin_modules") or []),
            dynamic_universe_enabled=bool(continuous_scan.get("dynamic_universe_enabled", False)),
            dynamic_universe_interval_cycles=int(continuous_scan.get("dynamic_universe_interval_cycles", 0) or 0),
            dynamic_universe_top_n=int(continuous_scan.get("dynamic_universe_top_n", 15) or 15),
            # Paper trading
            paper_trading_peak_mode=bool(paper_trading.get("peak_mode", True)),
            paper_trading_overrides=dict(
                (k, v) for k, v in (paper_trading or {}).items()
                if k not in ("peak_mode", "simulate_live") and isinstance(v, (bool, int, float, str, list, dict, type(None)))
            ),
            paper_simulates_live=bool(paper_trading.get("simulate_live", False)),
            # Self improvement
            self_improvement_enabled=bool(self_impr.get("enabled", True)),
            self_improvement_modes=list(self_impr.get("modes", ["paper", "backtest"]) or ["paper", "backtest"]),
            self_improvement_tick_seconds=int(self_impr.get("tick_seconds", 1) or 1),
            self_improvement_shadow_interval_minutes=int(self_impr.get("shadow_interval_minutes", self_impr.get("interval_minutes", 240)) or 240),
            self_improvement_interval_minutes=int(self_impr.get("interval_minutes", 240) or 240),
            self_improvement_shadow_tune_enabled=bool(self_impr.get("shadow_tune_enabled", True)),
            self_improvement_shadow_tune_days_total=int(self_impr.get("shadow_tune_days_total", 30) or 30),
            self_improvement_shadow_tune_train_days=int(self_impr.get("shadow_tune_train_days", 10) or 10),
            self_improvement_shadow_tune_test_days=int(self_impr.get("shadow_tune_test_days", 5) or 5),
            self_improvement_shadow_tune_evals=int(self_impr.get("shadow_tune_evals", 6) or 6),
            self_improvement_shadow_tune_warmup=int(self_impr.get("shadow_tune_warmup", 50) or 50),
            self_improvement_shadow_tune_timeframe=str(self_impr.get("shadow_tune_timeframe", "1h") or "1h"),
            self_improvement_shadow_tune_top=int(self_impr.get("shadow_tune_top", 10) or 10),
            self_improvement_apply_on_improvement_only=bool(self_impr.get("apply_on_improvement_only", True)),
            self_improvement_min_delta_return_pct=float(self_impr.get("min_delta_return_pct", 0.10) or 0.10),
            self_improvement_max_drawdown_pct=float(self_impr.get("max_drawdown_pct", 2.0) or 2.0),
            self_improvement_min_trades=int(self_impr.get("min_trades", 3) or 3),
            self_improvement_state_path=str(self_impr.get("state_path", "data/self_improvement_state.json") or "data/self_improvement_state.json"),
            self_improvement_try_quantum_on_off=bool(self_impr.get("try_quantum_on_off", True)),
            self_improvement_apply_quantum_choice=bool(self_impr.get("apply_quantum_choice", True)),
            self_improvement_validation_timeframes=list(self_impr.get("validation_timeframes", ["1h", "15m"]) or ["1h", "15m"]),
            self_improvement_promotion_min_delta_score=float(self_impr.get("promotion_min_delta_score", 0.10) or 0.10),
            self_improvement_promotion_require_all_timeframes=bool(self_impr.get("promotion_require_all_timeframes", True)),
            # Adaptive universe selection
            adaptive_universe_enabled=bool(adaptive_universe.get("enabled", True)),
            adaptive_universe_modes=list(adaptive_universe.get("modes", ["paper", "backtest", "live"]) or ["paper", "backtest", "live"]),
            adaptive_universe_top_n=int(adaptive_universe.get("top_n", 10) or 10),
            adaptive_universe_max_active=int(adaptive_universe.get("max_active", 5) or 5),
            adaptive_universe_min_hold_cycles=int(adaptive_universe.get("min_hold_cycles", 20) or 20),
            adaptive_universe_state_path=str(adaptive_universe.get("state_path", "data/adaptive_universe.json") or "data/adaptive_universe.json"),
            # Backtest/paper realism
            backtest=dict(y.get("backtest", {}) or {}),
        )
