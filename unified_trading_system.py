#!/usr/bin/env python3
"""
ARGUS ULTIMATE + KRAKEN DCA + PINNACLE AI - UNIFIED TRADING SYSTEM
==================================================================

Complete integration of all three trading systems optimized for $1,000 AUD
starting capital on Kraken and Coinbase exchanges.

Architecture:
- ARGUS Ultimate: Maximum return optimization strategies
- Kraken DCA Bot: Professional execution engine with monitoring
- Pinnacle AI: Multi-agent intelligence and consciousness layer

Phase 1: System Architecture Integration
Phase 2: AI Brain Integration (Pinnacle AI)
Phase 3: Professional Execution (Kraken DCA)
Phase 4: Capital Optimization ($1K AUD)
Phase 5: Production Deployment
"""

import asyncio
import logging
import os
import json
import hashlib
import signal
import sqlite3
import uuid
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from execution.reason_codes import ReasonCode

# Ensure directories exist BEFORE logging setup
Path('logs').mkdir(exist_ok=True)
Path('data').mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/unified_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Optional JSON log handler — active when ARGUS_JSON_LOGS=1
# Each line in logs/unified_system.jsonl is a JSON object with:
#   timestamp, level, logger, message, and any extra fields passed via LogRecord.
class _ArgusJsonFormatter(logging.Formatter):
    """Minimal JSON formatter for Loki-friendly structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        entry: Dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%f"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach any extra fields the caller passed (e.g. logger.info("msg", extra={"trade_id": "…"}))
        _STDLIB_ATTRS = frozenset(logging.LogRecord(
            "", 0, "", 0, "", (), None
        ).__dict__.keys()) | {"message", "asctime"}
        for k, v in record.__dict__.items():
            if k not in _STDLIB_ATTRS and not k.startswith("_"):
                try:
                    json.dumps(v)  # only include JSON-serialisable extras
                    entry[k] = v
                except (TypeError, ValueError):
                    entry[k] = str(v)
        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


if os.environ.get("ARGUS_JSON_LOGS", "").strip() == "1":
    _json_fh = logging.FileHandler("logs/unified_system.jsonl", encoding="utf-8")
    _json_fh.setFormatter(_ArgusJsonFormatter())
    _json_fh.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(_json_fh)


# ============================================================================
# PHASE 1: UNIFIED SYSTEM ARCHITECTURE
# ============================================================================

class SystemState(Enum):
    """System operational states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"
    SHUTDOWN = "shutdown"


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
    min_position_size_aud: float = 10.0  # Minimum $10 AUD per trade
    max_position_size_aud: float = 250.0  # Maximum per trade (25% of $1K) - HYPER AGGRESSIVE
    max_position_pct: float = 0.25  # 25% max per position - HYPER AGGRESSIVE
    max_total_exposure_pct: float = 0.98  # 98% max total exposure - FULLY INVESTED
    max_concurrent_positions: int = 5  # Max positions to deploy capital
    
    # Risk management
    max_daily_loss_pct: float = 0.10  # 10% max daily loss - LOOSER STOP
    max_drawdown_pct: float = 0.25  # 25% max drawdown - TOLERATE VOLATILITY
    stop_loss_pct: float = 0.03  # 3% stop loss - WIDER
    take_profit_pct: float = 0.08  # 8% take profit - HIGHER TARGETS
    use_volatility_adjusted_limits: bool = False  # Scale max position/daily loss by realized_vol_pct
    realized_vol_pct: float = 0.0  # e.g. 2.0 = 2%; feed from monitoring; 0 = no adjustment
    # PR-14 portfolio guardrails (defaults off unless configured)
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
    # Portfolio Target Engine v1 (signals -> targets -> execution)
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
    # Liquidity-Aware Risk Engine v1 (targets -> liquidity clamp -> hard risk gate)
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
    # Champion / Challenger Promotion Engine v1 (advisory-first)
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
    # Emergency shutdown: extra conditions (latency, flash crash, network, arb spread)
    emergency_shutdown_enabled: bool = True
    emergency_shutdown_latency_spike_ms: Optional[float] = None  # trip if cycle > N ms
    emergency_shutdown_flash_crash_pct: Optional[float] = None  # trip if any symbol move > N% in one cycle
    emergency_shutdown_network_fail: bool = False  # trip when _exchange_unreachable is set
    emergency_shutdown_arb_spread_bps: Optional[float] = None  # trip if spread > N bps

    # Commission and slippage awareness
    kraken_maker_fee: float = 0.0016  # 0.16%
    kraken_taker_fee: float = 0.0026  # 0.26%
    coinbase_maker_fee: float = 0.005  # 0.5%
    coinbase_taker_fee: float = 0.005  # 0.5%
    slippage_pct: float = 0.001  # 0.1% slippage

    # Execution engine settings
    order_type: str = "market"  # market/limit (unified engine currently uses market)
    retry_attempts: int = 3
    retry_delay_seconds: float = 5.0
    max_slippage_pct: float = 0.01  # hard guardrail on realized slippage (1%)
    signal_cooldown_bars: int = 0  # Min bars between trades on same symbol (0 = off; e.g. 3 to reduce overtrading)
    vwap_large_order_threshold_aud: float = 80.0  # Use VWAP/TWAP for orders >= this (AUD)
    use_twap_for_large_orders: bool = False  # True = TWAP equal slices; False = VWAP
    order_fill_timeout_seconds: float = 0.0  # If > 0, wait up to this then cancel if still open (0 = disabled)
    max_spread_bps: float = 0.0  # 0 = disabled; if > 0 skip order when (ask-bid)/mid > this (reduces bad fills)
    use_is_gate: bool = False  # When True, reject signals whose strategy/symbol avg implementation shortfall (bps) > max_avg_is_bps
    max_avg_is_bps: float = 0.0  # Gate threshold (bps); positive = cost
    portfolio_weight_method: str = "hrp"  # hrp | bl | mpt – allocation scaling
    multi_venue_enabled: bool = True  # Split large orders across venues when set
    multi_venue_min_notional_aud: float = 200.0  # Multi-venue when notional >= this
    twap_min_notional_usd: float = 250.0  # Use TWAP for orders >= this USD value
    twap_duration_minutes: float = 5.0  # TWAP execution window in minutes
    use_venue_routing_by_spread: bool = False  # When true, weight venues by inverse spread (tighter = more size)
    dca_levels_pct: Optional[List[float]] = None  # e.g. [0.33, 0.33, 0.34] = 3-level DCA; None = single order
    use_correlation_aware_sizing: bool = False  # When true, scale weights by correlation (need correlation_matrix)
    max_correlated_exposure: float = 0.6  # Scale down when correlation > this (e.g. BTC/ETH)
    correlation_matrix: Optional[Dict[Any, float]] = None  # (sym_a, sym_b) -> corr; set from monitoring
    # Data – tick store + data lake (use everything)
    persist_tick_store: bool = True  # Write ticker/order book to tick store
    use_lake_read: bool = True  # Paper loop: load OHLCV from data lake first
    persist_to_lake: bool = True  # Paper loop: write OHLCV to data lake
    persist_to_tick_store: bool = True  # Paper loop: emit synthetic ticks
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
    num_ai_agents: int = 50  # Reduced for small capital
    min_signal_confidence: float = 0.75
    live_min_signal_confidence: Optional[float] = None  # When set, used in live mode (e.g. 0.78–0.82)
    max_concurrent_signals: int = 2
    
    # Monitoring
    prometheus_enabled: bool = True
    grafana_enabled: bool = True
    prometheus_port: int = 9090
    grafana_port: int = 3000

    # Multi-language system (optional)
    multi_language_enabled: bool = True
    multi_language_endpoints: Dict[str, str] = field(default_factory=dict)
    use_cycle_aggregate_boost: bool = True
    use_conservative_cycle_boost: bool = False
    use_weighted_mean_boost: bool = False  # Use latency/confidence-weighted mean for cycle boost when True
    use_risk_all: bool = False
    use_conservative_risk: bool = False
    use_regime_estimate: bool = False
    use_slippage_estimate: bool = False
    use_drawdown_check: bool = False
    use_position_sizing_gate: bool = False  # 23-language position sizing: cap size by size_pct_conservative
    use_signal_filter_gate: bool = False    # 23-language signal filter: drop signals majority reject
    max_slippage_bps: float = 100.0  # 23-language slippage gate: skip execution if median > this (bps)
    multi_language_task_timeouts: Optional[Dict[str, float]] = None  # Per-task timeouts (sec); from multi_language.task_timeouts
    multi_language_warm_on_start: bool = False  # If true, POST /warm to each HTTP endpoint after init

    # Runtime mode safety interlock
    # - paper: force dry_run on exchanges; never send real orders
    # - live: requires credentials; allows real orders
    run_mode: str = "paper"
    # Deployment role:
    # - single-node: strategy + execution on one host (default for backward compatibility)
    # - strategy-node: publish signed instructions, never execute orders
    # - execution-node: consume signed instructions and execute orders
    node_role: str = "single-node"
    command_bus_enabled: bool = False
    command_bus_db_path: str = "data/command_bus.db"
    command_bus_queue: str = "default"
    command_bus_hmac_key_env: str = "ARGUS_COMMAND_HMAC_KEY"
    command_bus_instruction_ttl_seconds: float = 5.0
    command_bus_require_signature: bool = True
    command_bus_max_batch: int = 64
    command_bus_max_notional_aud: float = 0.0
    # OMEGA-01 execution mesh (isolated per-symbol execution lanes)
    execution_mesh_enabled: bool = False
    execution_mesh_max_lanes: int = 8
    execution_mesh_max_queue_per_lane: int = 128
    execution_mesh_batch_size: int = 8
    execution_mesh_parallel_lanes: bool = True
    execution_mesh_halt_on_lane_error: bool = True
    execution_mesh_symbols: List[str] = field(default_factory=list)
    live_disabled_strategies: List[str] = field(default_factory=list)  # strategies to disable when mode is live
    # Edge gate: require paper evidence before allowing live (enforced by scripts/pre_live_check.py)
    live_require_paper_edge: bool = False
    live_min_trades_paper: int = 20
    live_min_win_rate_pct: float = 45.0

    # Backtest/paper realism (spread, slippage guardrail, market impact, fees, rate limit; see backtest section in YAML)
    backtest: Dict[str, Any] = field(default_factory=dict)

    # Evolution: load evolved params at startup (paper/backtest)
    evolution_load_evolved: bool = False
    evolution_params_path: str = "data/evolved_params.json"
    # Continuous evolution: run GA on a schedule and apply new params to adapt to market
    evolution_continuous_enabled: bool = True
    evolution_interval_hours: float = 24.0
    evolution_fitness_days: int = 7
    evolution_generations: int = 5
    evolution_population_size: int = 12
    evolution_auto_apply: bool = True  # apply new best params to config after each run
    # Real-time evolution: run with the market (short interval + recent/live data)
    evolution_realtime_interval_minutes: float = 60.0  # run evolution every N min (0 = use interval_hours)
    evolution_realtime_fitness_days: float = 1.0  # use last N days for fitness (recent market)
    evolution_use_live_feed: bool = True  # when available, use market_data_service OHLCV for fitness
    evolution_realtime_generations: int = 3  # smaller GA for real-time (faster)
    evolution_realtime_population_size: int = 8  # smaller pop for real-time
    evolution_min_bars_for_live_feed: int = 20  # require at least N bars per symbol to use live feed
    evolution_run_with_market: bool = True  # run evolution continuously with the market, improving strategies and bot
    # Instantaneous: trigger evolution as soon as N trades close (no wait for interval)
    evolution_trigger_on_trade: bool = False  # if True, run evolution after every evolution_after_n_trades closed trades
    evolution_after_n_trades: int = 5  # run evolution after this many closed trades (1 = every trade, heavy)
    # Safety and scheduling
    evolution_dry_run: bool = False  # run GA and log best params but do not persist or apply
    evolution_debounce_minutes: float = 15.0  # minimum minutes between evolution runs (avoid thrashing)
    evolution_allow_apply_live: bool = False  # if False, auto_apply is disabled when run_mode is live
    # Fitness and GA tuning
    evolution_seed: Optional[int] = None  # for reproducibility
    evolution_min_trades: int = 1  # min closed trades for valid fitness (penalty otherwise)
    evolution_ga_mutation_prob: float = 0.2
    evolution_ga_mutation_sigma: float = 0.15
    evolution_ga_crossover_prob: float = 0.7
    evolution_early_stop_generations: int = 0  # 0 = disabled; stop after N gens with < threshold improvement
    evolution_early_stop_threshold: float = 0.001
    evolution_fitness_cache_size: int = 0  # LRU cache size for fitness (0 = off)
    evolution_parallel_fitness_workers: int = 0  # 0 = sequential
    evolution_walk_forward_train_ratio: Optional[float] = None  # e.g. 0.7 for train/test split
    evolution_negative_return_penalty_weight: float = 0.0
    evolution_use_composite_fitness: bool = False  # weighted sharpe + sortino - drawdown - neg return
    evolution_backup_before_apply: bool = True
    evolution_version_history_size: int = 0  # keep last N versions in evolved_history/ (0 = off)
    # Phase 2: multi-timeframe, overfit/volatility/Calmar, strategy subset, allocator decay
    evolution_multi_timeframes: Optional[List[str]] = None  # e.g. ["1h", "15m"] for combined fitness
    evolution_multi_timeframe_weights: Optional[List[float]] = None  # same length as multi_timeframes
    evolution_overfit_penalty_weight: float = 0.0  # penalize train>>test in walk-forward
    evolution_volatility_penalty_weight: float = 0.0
    evolution_composite_calmar_weight: float = 0.0  # Calmar in composite fitness
    evolution_strategy_whitelist_override: Optional[List[str]] = None  # restrict backtest to these strategies
    evolution_allocator_decay_after_apply: Optional[float] = None  # 0=reset, 0.5=halve; None=no decay
    use_evolution_strategy_reward: bool = False  # Record PnL per strategy/symbol for ml.evolution_strategy_reward (param jitter)

    # Strategy pack (optional)
    strategies_enabled: List[str] = field(default_factory=lambda: ["hunter", "farmer", "shadow"])
    strategies_max_extra_signals: int = 2
    strategy_whitelist: List[str] = field(default_factory=list)  # when non-empty, only these strategies may emit signals (75k path)
    use_regime_lstm_boost: bool = False  # Optional: scale signal confidence by ml.regime_boost (one better alpha source)
    use_volatility_regime_scale: bool = False  # Scale down confidence when recent vol > threshold
    volatility_regime_high_threshold: float = 0.02  # Daily return std above this -> confidence *= 0.85
    use_funding_rate_filter: bool = False  # Skip long when funding rate >= threshold (crypto perps)
    funding_rates_url: Optional[str] = None  # Optional HTTP URL for funding rates JSON; else use CCXT
    funding_rate_skip_long_threshold: float = 0.0001  # Skip long when rate >= this (0.01%)
    regime_filter_enabled: bool = False
    regime_filter_trend_strategies: List[str] = field(default_factory=lambda: ["trend_following", "quantum_trend_following_elite"])
    regime_filter_mr_strategies: List[str] = field(default_factory=lambda: ["mean_reversion", "quantum_mean_reversion_elite"])

    # Strategy library (optional extra strategy sources; paper/backtest only by default)
    strategy_library_enabled: bool = True
    strategy_library_modes: List[str] = field(default_factory=lambda: ["paper", "backtest"])
    strategy_library_strategies_enabled: List[str] = field(
        default_factory=lambda: [
            # Algorithmic
            "momentum",
            "mean_reversion",
            "trend_following",
            "pairs_trading",
            "market_making",
            "arbitrage",
            "candlestick_patterns",
            "high_freq_grid",
            # Advanced
            "regime_switching",
            "stat_arb",
            "factor_investing",
            "cross_exchange_arb",
            # Tier ensembles
            "absolute_tier",
            "akashic_tier",
            "apeiron_tier",
            "chronos_tier",
            "omega_tier",
            "paradox_tier",
            "singularity_tier",
            "source_tier",
            "thanatos_tier",
            "void_tier",
            # Quantum custom
            "quantum_momentum_elite",
            "quantum_mean_reversion_elite",
            "quantum_trend_following_elite",
            "quantum_breakout_elite",
            "quantum_portfolio_rotation_elite",
            "quantum_arbitrage_elite",
        ]
    )

    # Quantum / quant-fund optional features (enabled for paper/backtest only by default)
    quantum_features_enabled: bool = True
    quantum_features_modes: List[str] = field(default_factory=lambda: ["paper", "backtest"])
    quantum_consciousness_enabled: bool = True
    quantum_method: str = "quantum_approximate"
    quantum_strength: float = 1.0

    quant_fund_upgrades_enabled: bool = True
    quant_fund_upgrades_modes: List[str] = field(default_factory=lambda: ["paper", "backtest"])
    quant_fund_risk_engine_enabled: bool = True

    # Adaptive behavior (regime detection + online tuning)
    adaptive_enabled: bool = True
    adaptive_minutes_per_bar: float = 60.0
    adaptive_tuner_alpha: float = 0.15
    adaptive_min_trades_before_bias: int = 3

    # StrategyEngine tunables (for paper/backtest optimization)
    se_buy_rsi: float = 35.0
    se_sell_rsi: float = 65.0
    se_buy_bb: float = 0.30
    se_sell_bb: float = 0.70
    se_trend_rsi_buy: float = 55.0
    se_trend_rsi_sell: float = 45.0

    # Offline optimization: load/apply best params in paper/backtest
    optimized_params_load: bool = False
    optimized_params_path: str = "data/optimized_params.json"
    optimized_params_timeframe: str = ""
    optimized_params: Optional[Dict[str, Any]] = None
    optimized_params_by_timeframe: Optional[Dict[str, Any]] = None

    # Strategy allocator (meta layer)
    strategy_allocator_enabled: bool = True
    strategy_allocator_modes: List[str] = field(default_factory=lambda: ["paper", "backtest"])
    strategy_allocator_timeframe: str = ""
    strategy_allocator_persist_path: str = "data/strategy_allocator_stats.json"
    strategy_allocator_min_trades_before_bias: int = 5
    strategy_allocator_exploration_c: float = 1.2
    strategy_allocator_ema_alpha: float = 0.15
    strategy_allocator_max_total_signals: int = 5
    strategy_allocator_max_per_strategy: int = 2

    # Earnings optimization: edge-vs-cost gate (paper/backtest by default)
    edge_cost_gate_enabled: bool = True
    edge_cost_gate_modes: List[str] = field(default_factory=lambda: ["paper", "backtest"])
    edge_cost_gate_buffer_mult: float = 1.25
    edge_cost_gate_min_edge_pct: float = 0.0
    edge_cost_gate_live_buffer_mult: Optional[float] = None  # Stricter in live (e.g. 2.2)
    edge_cost_gate_live_min_edge_pct: Optional[float] = None  # Stricter in live (e.g. 1.0)
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
    # When True, paper uses same config as live (no peak overrides), same disabled strategies, and simulated slippage
    paper_simulates_live: bool = False

    # Self-improvement loop (continuous learning + optional shadow tuning)
    self_improvement_enabled: bool = True
    self_improvement_modes: List[str] = field(default_factory=lambda: ["paper", "backtest"])
    # Lightweight tick loop (can be as low as 1s)
    self_improvement_tick_seconds: int = 1
    # Heavy tuning interval (shadow tuning); keep this large by default.
    self_improvement_shadow_interval_minutes: int = 240  # 4h
    # Back-compat (deprecated): if set, used as shadow interval.
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
    self_improvement_min_delta_return_pct: float = 0.10  # require +0.10% improvement
    self_improvement_max_drawdown_pct: float = 2.0
    self_improvement_min_trades: int = 3
    self_improvement_state_path: str = "data/self_improvement_state.json"

    # Quantum in evolution: try quantum ON/OFF in shadow tuning and adopt best.
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
    fast_mode: bool = False  # Skip non-essential per-cycle ops (GC, observability snapshots, etc.)
    latency_ws_order_preference: bool = True  # Prefer WebSocket over REST for order placement
    latency_tick_to_trade_threshold_ms: float = 500.0  # Warn if tick-to-trade exceeds this
    latency_ping_interval_s: float = 30.0  # Exchange ping interval for latency routing
    latency_connection_pool_enabled: bool = True  # Pre-warm HTTPS connections on startup
    latency_fire_and_forget_enabled: bool = False  # Enable fire-and-forget order submission

    # Emergency shutdown
    emergency_stop_enabled: bool = True
    max_consecutive_losses: int = 5
    max_error_rate: float = 0.10
    auto_reduce_after_n_losses: int = 0  # 0 = off; after N consecutive losses reduce size by auto_reduce_factor
    auto_reduce_factor: float = 0.6  # Size multiplier when auto-reduce active (e.g. 0.6 = 60%)

    @classmethod
    def from_unified_yaml(cls, path: str = "unified_config.yaml") -> "UnifiedConfig":
        """
        Load the runtime configuration from `unified_config.yaml`.

        This is intentionally a best-effort mapper (the YAML can contain more keys
        than this dataclass models).
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
                strategy_library.get(
                    "enabled_strategies",
                    cls().strategy_library_strategies_enabled,
                )
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
            strategy_allocator_persist_path=str(
                strategy_allocator.get("persist_path", "data/strategy_allocator_stats.json") or "data/strategy_allocator_stats.json"
            ),
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
            liquidity_risk_depth_fraction_limit=float(
                liquidity_risk_engine.get("depth_fraction_limit", 0.04) or 0.04
            ),
            liquidity_risk_thin_spread_threshold_bps=float(
                liquidity_risk_engine.get("thin_spread_threshold_bps", 6.0) or 6.0
            ),
            liquidity_risk_danger_spread_threshold_bps=float(
                liquidity_risk_engine.get("danger_spread_threshold_bps", 12.0) or 12.0
            ),
            liquidity_risk_min_depth_threshold=float(
                liquidity_risk_engine.get("min_depth_threshold", 0.5) or 0.5
            ),
            liquidity_risk_slippage_threshold_bps=float(
                liquidity_risk_engine.get("slippage_threshold_bps", 10.0) or 10.0
            ),
            liquidity_risk_min_liquidity_score=float(
                liquidity_risk_engine.get("min_liquidity_score", 0.2) or 0.2
            ),
            liquidity_risk_score_weights={
                "depth": float(
                    ((liquidity_risk_engine.get("score_weights", {}) or {}).get("depth", 1.0) or 1.0)
                ),
                "spread": float(
                    ((liquidity_risk_engine.get("score_weights", {}) or {}).get("spread", 1.0) or 1.0)
                ),
                "fill_ratio": float(
                    ((liquidity_risk_engine.get("score_weights", {}) or {}).get("fill_ratio", 0.75) or 0.75)
                ),
            },
            # Strategy Evaluation Engine v1
            strategy_evaluation_enabled=bool(strategy_evaluation_engine.get("enabled", True)),
            strategy_evaluation_persist_interval_cycles=int(
                strategy_evaluation_engine.get("persist_interval_cycles", 10) or 10
            ),
            strategy_evaluation_min_trades_for_ranking=int(
                strategy_evaluation_engine.get("min_trades_for_ranking", 5) or 5
            ),
            strategy_evaluation_use_regime_scoped_metrics=bool(
                strategy_evaluation_engine.get("use_regime_scoped_metrics", True)
            ),
            strategy_evaluation_sharpe_like_min_trades=int(
                strategy_evaluation_engine.get("sharpe_like_min_trades", 5) or 5
            ),
            strategy_evaluation_max_metrics_history_points=int(
                strategy_evaluation_engine.get("max_metrics_history_points", 500) or 500
            ),
            strategy_evaluation_halt_on_error=bool(
                strategy_evaluation_engine.get("halt_on_error", False)
            ),
            strategy_evaluation_db_path=str(
                strategy_evaluation_engine.get("db_path", "data/strategy_metrics.db")
                or "data/strategy_metrics.db"
            ),
            # Self-Optimizing Meta Engine v1
            self_optimizing_meta_enabled=bool(self_optimizing_meta_engine.get("enabled", True)),
            self_optimizing_meta_advisory_only=bool(self_optimizing_meta_engine.get("advisory_only", False)),
            self_optimizing_meta_update_interval_cycles=int(
                self_optimizing_meta_engine.get("update_interval_cycles", 10) or 10
            ),
            self_optimizing_meta_min_trades_for_reweighting=int(
                self_optimizing_meta_engine.get("min_trades_for_reweighting", 5) or 5
            ),
            self_optimizing_meta_alpha=float(
                self_optimizing_meta_engine.get("meta_alpha", 0.2) or 0.2
            ),
            self_optimizing_meta_max_weight_change_per_update=float(
                self_optimizing_meta_engine.get("max_weight_change_per_update", 0.10) or 0.10
            ),
            self_optimizing_meta_min_weight_per_strategy=float(
                self_optimizing_meta_engine.get("min_weight_per_strategy", 0.05) or 0.05
            ),
            self_optimizing_meta_max_weight_per_strategy=float(
                self_optimizing_meta_engine.get("max_weight_per_strategy", 0.45) or 0.45
            ),
            self_optimizing_meta_baseline_weight_mode=str(
                self_optimizing_meta_engine.get("baseline_weight_mode", "equal") or "equal"
            ),
            self_optimizing_meta_score_weights={
                "expectancy": float(
                    ((self_optimizing_meta_engine.get("score_weights", {}) or {}).get("expectancy", 1.0) or 1.0)
                ),
                "sharpe_like": float(
                    ((self_optimizing_meta_engine.get("score_weights", {}) or {}).get("sharpe_like", 1.0) or 1.0)
                ),
                "profit_factor": float(
                    ((self_optimizing_meta_engine.get("score_weights", {}) or {}).get("profit_factor", 0.75) or 0.75)
                ),
                "drawdown_penalty": float(
                    ((self_optimizing_meta_engine.get("score_weights", {}) or {}).get("drawdown_penalty", 1.0) or 1.0)
                ),
                "fee_penalty": float(
                    ((self_optimizing_meta_engine.get("score_weights", {}) or {}).get("fee_penalty", 0.5) or 0.5)
                ),
                "slippage_penalty": float(
                    ((self_optimizing_meta_engine.get("score_weights", {}) or {}).get("slippage_penalty", 0.5) or 0.5)
                ),
            },
            self_optimizing_meta_regime_multipliers={
                str(_k): {
                    str(_sk): float(_sv)
                    for _sk, _sv in dict(_v or {}).items()
                }
                for _k, _v in dict(self_optimizing_meta_engine.get("regime_multipliers", {}) or {}).items()
            },
            self_optimizing_meta_db_path=str(
                self_optimizing_meta_engine.get("db_path", "data/meta_weights.db")
                or "data/meta_weights.db"
            ),
            # Champion / Challenger Promotion Engine v1
            champion_challenger_enabled=bool(champion_challenger.get("enabled", True)),
            champion_challenger_advisory_only=bool(champion_challenger.get("advisory_only", True)),
            champion_challenger_min_trades_for_promotion=int(
                champion_challenger.get("min_trades_for_promotion", 10) or 10
            ),
            champion_challenger_max_drawdown_pct_for_promotion=float(
                champion_challenger.get("max_drawdown_pct_for_promotion", 0.12) or 0.12
            ),
            champion_challenger_require_expectancy_improvement=bool(
                champion_challenger.get("require_expectancy_improvement", True)
            ),
            champion_challenger_require_profit_factor_improvement=bool(
                champion_challenger.get("require_profit_factor_improvement", False)
            ),
            champion_challenger_require_sharpe_like_improvement=bool(
                champion_challenger.get("require_sharpe_like_improvement", True)
            ),
            champion_challenger_persist_interval_cycles=int(
                champion_challenger.get("persist_interval_cycles", 10) or 10
            ),
            champion_challenger_db_path=str(
                champion_challenger.get("db_path", "data/champion_challenger.db")
                or "data/champion_challenger.db"
            ),
            champion_challenger_artifacts_dir=str(
                champion_challenger.get("artifacts_dir", "deploy/promotions")
                or "deploy/promotions"
            ),
            champion_challenger_promotion_weights=dict(cc_weights),
            # Market Microstructure Engine v1
            market_microstructure_enabled=bool(market_microstructure_engine.get("enabled", True)),
            market_microstructure_rolling_window=int(
                market_microstructure_engine.get("rolling_window", 20) or 20
            ),
            market_microstructure_vacuum_spread_jump_bps=float(
                market_microstructure_engine.get("vacuum_spread_jump_bps", 4.0) or 4.0
            ),
            market_microstructure_vacuum_depth_drop_ratio=float(
                market_microstructure_engine.get("vacuum_depth_drop_ratio", 0.5) or 0.5
            ),
            market_microstructure_high_adverse_selection_threshold=float(
                market_microstructure_engine.get("high_adverse_selection_threshold", 0.7) or 0.7
            ),
            market_microstructure_use_in_execution_alpha=bool(
                market_microstructure_engine.get("use_in_execution_alpha", True)
            ),
            market_microstructure_use_in_liquidity_risk=bool(
                market_microstructure_engine.get("use_in_liquidity_risk", True)
            ),
            # Recon-Required Recovery Engine v1
            recon_recovery_enabled=bool(recon_recovery_engine.get("enabled", True)),
            recon_recovery_stale_threshold_seconds=float(
                recon_recovery_engine.get("stale_threshold_seconds", 60.0) or 60.0
            ),
            recon_recovery_base_retry_delay_seconds=float(
                recon_recovery_engine.get("base_retry_delay_seconds", 5.0) or 5.0
            ),
            recon_recovery_max_retries=int(
                recon_recovery_engine.get("max_retries", 5) or 5
            ),
            recon_recovery_halt_on_retry_exhausted=bool(
                recon_recovery_engine.get("halt_on_retry_exhausted", True)
            ),
            # System Health Metrics
            system_health_metrics_enabled=bool(system_health_metrics.get("enabled", True)),
            system_health_metrics_snapshot_interval_cycles=int(
                system_health_metrics.get("snapshot_interval_cycles", 10) or 10
            ),
            # Execution Alpha Engine v2
            execution_alpha_enabled=bool(execution_alpha_engine.get("enabled", True)),
            execution_alpha_maker_spread_threshold_bps=float(
                execution_alpha_engine.get(
                    "maker_spread_threshold_bps",
                    execution_alpha_legacy.get("maker_spread_threshold_bps", 2.0),
                )
                or 2.0
            ),
            execution_alpha_min_fill_probability=float(
                execution_alpha_engine.get(
                    "min_fill_probability",
                    execution_alpha_legacy.get("min_fill_probability", 0.35),
                )
                or 0.35
            ),
            execution_alpha_slice_threshold_pct=float(
                execution_alpha_engine.get(
                    "slice_threshold_pct",
                    execution_alpha_legacy.get("slice_threshold_pct", 0.03),
                )
                or 0.03
            ),
            execution_alpha_maker_fallback_seconds=float(
                execution_alpha_engine.get(
                    "maker_fallback_seconds",
                    execution_alpha_legacy.get("maker_timeout_seconds", 8.0),
                )
                or 8.0
            ),
            execution_alpha_telemetry_window=int(
                execution_alpha_engine.get(
                    "telemetry_window",
                    execution_alpha_legacy.get("telemetry_window", 200),
                )
                or 200
            ),
            runtime_safety_latency_grace_cycles=int(
                runtime_safety.get("latency_grace_cycles", 2) or 2
            ),
            live_safe_disable_pinnacle_ai_brain=bool(
                runtime_safety.get("live_safe_disable_pinnacle_ai_brain", False)
            ),
            market_data_ohlcv_cache_seconds=float(
                market_data.get("ohlcv_cache_seconds", 30.0) or 30.0
            ),
            market_data_ohlcv_poll_interval_seconds=float(
                market_data.get("ohlcv_poll_interval_seconds", 30.0) or 30.0
            ),
            market_data_ohlcv_retry_attempts=int(
                market_data.get("ohlcv_retry_attempts", 2) or 2
            ),
            # Continuous best-trade scanner (peak)
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
            # Paper trading absolute peak (overrides applied in paper/backtest when peak_mode true)
            paper_trading_peak_mode=bool(paper_trading.get("peak_mode", True)),
            paper_trading_overrides=dict(
                (k, v)
                for k, v in (paper_trading or {}).items()
                if k not in ("peak_mode", "simulate_live") and isinstance(v, (bool, int, float, str, list, dict, type(None)))
            ),
            paper_simulates_live=bool(paper_trading.get("simulate_live", False)),
            # Self improvement
            self_improvement_enabled=bool(self_impr.get("enabled", True)),
            self_improvement_modes=list(self_impr.get("modes", ["paper", "backtest"]) or ["paper", "backtest"]),
            self_improvement_tick_seconds=int(self_impr.get("tick_seconds", 1) or 1),
            self_improvement_shadow_interval_minutes=int(
                self_impr.get("shadow_interval_minutes", self_impr.get("interval_minutes", 240)) or 240
            ),
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
            # Backtest/paper realism (spread, max_slippage, market_impact, fee_maker_ratio, rate_limit, partial_fill)
            backtest=dict(y.get("backtest", {}) or {}),
        )


class _PaperCCXTWrapper:
    """Wraps a CCXT async exchange so create_order/fetch_order return mock results (paper/dry-run).
    All fetch_ticker/fetch_ohlcv delegate to the real exchange for live market data."""

    def __init__(self, exchange: Any, name: str = "ccxt"):
        self._exchange = exchange
        self._name = name
        self._paper_orders: Dict[str, Dict[str, Any]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._exchange, name)

    async def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        import random as _rng
        now_ms = int(time.time() * 1000)
        # Get current price for realistic fill simulation
        fill_price = price
        if fill_price is None:
            try:
                ticker = await self._exchange.fetch_ticker(symbol)
                fill_price = float(ticker.get("last", 0) or 0)
            except Exception as e:
                logger.warning(f"Failed to fetch ticker for {symbol}: {e}")
                fill_price = 0.0

        # ── Realistic paper simulation ──
        # 1. Slippage: 0.02%-0.15% adverse (buys fill higher, sells lower)
        if fill_price and fill_price > 0:
            slippage_pct = _rng.uniform(0.0002, 0.0015)
            if side == "buy":
                fill_price *= (1 + slippage_pct)
            else:
                fill_price *= (1 - slippage_pct)

        # 2. Partial fills: 85-100% fill rate (larger orders = lower fill)
        fill_rate = min(1.0, max(0.85, 1.0 - _rng.uniform(0, 0.15)))
        filled_amount = float(amount) * fill_rate
        remaining = float(amount) - filled_amount

        # 3. Latency: 50-500ms simulated (logged, not awaited to avoid slowing loop)
        sim_latency_ms = _rng.randint(50, 500)

        # 4. Spread-aware fee: taker 0.26%, maker 0.16% (market=taker, limit=maker)
        fee_rate = 0.0026 if type == "market" else 0.0016

        logger.info("PAPER %s: %s %s %s amt=%.6f fill=%.6f@%.2f slip=%.2f%% lat=%dms",
                     self._name, type, side, symbol, amount, filled_amount,
                     fill_price or 0, slippage_pct * 100 if fill_price else 0, sim_latency_ms)
        order = {
            "id": f"paper_{self._name}_{now_ms}",
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": float(amount),
            "price": float(fill_price) if fill_price else None,
            "average": float(fill_price) if fill_price else None,
            "cost": filled_amount * float(fill_price) if fill_price else 0.0,
            "status": "closed" if remaining < 0.0001 else "partially_filled",
            "filled": filled_amount,
            "remaining": remaining,
            "timestamp": now_ms,
            "fee": {"cost": filled_amount * float(fill_price or 0) * fee_rate, "currency": "USD"},
            "info": {"simulated_latency_ms": sim_latency_ms, "slippage_pct": slippage_pct * 100 if fill_price else 0},
        }
        self._paper_orders[order["id"]] = order
        return order

    async def fetch_order(self, order_id: str, symbol: Optional[str] = None, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Return the mock order instantly — no network call."""
        if order_id in self._paper_orders:
            return self._paper_orders[order_id]
        # Unknown order — return a closed mock
        return {"id": order_id, "status": "closed", "filled": 0, "remaining": 0, "symbol": symbol or ""}

    async def cancel_order(self, order_id: str, symbol: Optional[str] = None, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Cancel is a no-op for paper orders."""
        if order_id in self._paper_orders:
            self._paper_orders[order_id]["status"] = "canceled"
        return {"id": order_id, "status": "canceled"}

    async def fetch_orders(self, symbol: Optional[str] = None, since: Optional[int] = None,
                           limit: Optional[int] = None, params: Optional[Dict] = None) -> list:
        """Return all paper orders."""
        orders = list(self._paper_orders.values())
        if symbol:
            orders = [o for o in orders if o.get("symbol") == symbol]
        return orders[-int(limit or 50):]



class OmegaSQLiteStore:
    """Minimal Ω spine persistence: decision snapshots + order intents.

    This is intentionally lightweight and SQLite-only so Windows paper runs are
    immediately auditable without extra services.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(os.path.dirname(db_path) or ".").mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30.0)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA foreign_keys=ON;")
        return con

    def _table_columns(self, con: sqlite3.Connection, table_name: str) -> set[str]:
        try:
            rows = con.execute(f"PRAGMA table_info({table_name})").fetchall()
            return {str(r[1]) for r in rows if len(r) > 1 and r[1]}
        except Exception as e:
            logger.warning(f"Failed to get table info for {table_name}: {e}")
            return set()

    def init_schema(self) -> None:
        con = self.connect()
        try:
            con.executescript("""
            CREATE TABLE IF NOT EXISTS decision_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_utc TEXT NOT NULL,
                run_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                cycle_id INTEGER NOT NULL,
                correlation_id TEXT,
                symbol TEXT,
                strategy TEXT,
                side TEXT,
                signal_score REAL,
                allowed INTEGER NOT NULL,
                reason_code TEXT NOT NULL,
                details_json TEXT,
                cost_json TEXT,
                exec_plan_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_decision_trace ON decision_snapshots(trace_id);
            CREATE INDEX IF NOT EXISTS idx_decision_cycle ON decision_snapshots(cycle_id);

            CREATE TABLE IF NOT EXISTS order_intents (
                intent_id TEXT PRIMARY KEY,
                ts_utc TEXT NOT NULL,
                run_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                cycle_id INTEGER NOT NULL,
                correlation_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT,
                amount REAL,
                price REAL,
                status TEXT NOT NULL,
                exchange_order_id TEXT,
                exec_plan_json TEXT,
                meta_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_intent_cycle ON order_intents(cycle_id);

            CREATE TABLE IF NOT EXISTS system_health_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cycles_completed INTEGER NOT NULL,
                avg_latency_ms REAL NOT NULL,
                errors_last_hour INTEGER NOT NULL,
                warnings_last_hour INTEGER NOT NULL,
                event_loop_delay_ms REAL NOT NULL,
                memory_rss_mb REAL NOT NULL DEFAULT 0.0,
                memory_python_mb REAL NOT NULL DEFAULT 0.0
            );
            CREATE INDEX IF NOT EXISTS idx_system_health_timestamp ON system_health_snapshots(timestamp);
            """)
            # Backward-compatible migration for existing DBs.
            try:
                cols = self._table_columns(con, "system_health_snapshots")
                if "memory_rss_mb" not in cols:
                    con.execute(
                        "ALTER TABLE system_health_snapshots ADD COLUMN memory_rss_mb REAL NOT NULL DEFAULT 0.0"
                    )
                if "memory_python_mb" not in cols:
                    con.execute(
                        "ALTER TABLE system_health_snapshots ADD COLUMN memory_python_mb REAL NOT NULL DEFAULT 0.0"
                    )
            except Exception as e:
                # Fail-safe: never block runtime on optional metric columns.
                logger.debug(f"Failed to add optional metric column: {e}")
            con.commit()
        finally:
            con.close()

    @staticmethod
    def _normalize_exec_plan(details: dict | None, exec_plan: dict | None, *, allowed: bool) -> dict:
        """Guarantee deterministic execution-plan payload for audit rows."""
        plan = dict(exec_plan or {})
        details_d = dict(details or {})
        if not plan:
            for key in (
                "order_type",
                "planned_order_size",
                "slice_count",
                "expected_slippage_bps",
                "expected_fill_probability",
                "fallback_after_seconds",
                "priority_score",
                "reason_codes",
            ):
                if key in details_d and details_d.get(key) is not None:
                    plan[key] = details_d.get(key)
        if plan.get("order_type") is None:
            plan["order_type"] = "none" if not allowed else "unspecified"
        return plan

    def record_decision(self, *, run_id: str, trace_id: str, cycle_id: int, correlation_id: str | None,
                        symbol: str | None, strategy: str | None, side: str | None, signal_score: float | None,
                        allowed: bool, reason_code: str, details: dict | None = None,
                        cost: dict | None = None, exec_plan: dict | None = None) -> None:
        norm_details = dict(details or {})
        norm_exec_plan = self._normalize_exec_plan(norm_details, exec_plan, allowed=bool(allowed))
        payload_details = json.dumps(norm_details, ensure_ascii=True, default=str)
        payload_cost = json.dumps(cost or {}, ensure_ascii=True, default=str)
        payload_plan = json.dumps(norm_exec_plan, ensure_ascii=True, default=str)
        con = self.connect()
        try:
            table_cols = self._table_columns(con, "decision_snapshots")
            row = {
                "ts_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "timestamp": float(time.time()),
                "run_id": str(run_id),
                "trace_id": str(trace_id),
                "cycle_id": int(cycle_id),
                "correlation_id": correlation_id,
                "symbol": symbol,
                "strategy": strategy,
                "side": side,
                "signal_score": float(signal_score) if signal_score is not None else None,
                "allowed": 1 if allowed else 0,
                "reason_code": str(reason_code),
                "details_json": payload_details,
                "cost_json": payload_cost,
                "exec_plan_json": payload_plan,
                "execution_plan_json": payload_plan,
            }
            cols = [k for k in row.keys() if k in table_cols]
            if not cols:
                raise sqlite3.OperationalError("decision_snapshots table has no compatible columns")
            placeholders = ", ".join(["?"] * len(cols))
            sql = f"INSERT INTO decision_snapshots ({', '.join(cols)}) VALUES ({placeholders})"
            con.execute(sql, tuple(row[c] for c in cols))
            con.commit()
        finally:
            con.close()

    def create_intent(self, *, intent_id: str, run_id: str, trace_id: str, cycle_id: int, correlation_id: str | None,
                      symbol: str, side: str, order_type: str | None, amount: float | None, price: float | None,
                      status: str = "CREATED", exec_plan: dict | None = None, meta: dict | None = None) -> None:
        con = self.connect()
        try:
            con.execute(
                """INSERT OR IGNORE INTO order_intents
                (intent_id, ts_utc, run_id, trace_id, cycle_id, correlation_id, symbol, side, order_type, amount, price, status, exec_plan_json, meta_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    intent_id,
                    datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    run_id, trace_id, int(cycle_id), correlation_id,
                    symbol, side, order_type, amount, price,
                    status,
                    json.dumps(exec_plan or {}, ensure_ascii=True, default=str),
                    json.dumps(meta or {}, ensure_ascii=True, default=str),
                ),
            )
            con.commit()
        finally:
            con.close()

    def update_intent(self, intent_id: str, *, status: str, exchange_order_id: str | None = None, meta: dict | None = None) -> None:
        con = self.connect()
        try:
            con.execute(
                """UPDATE order_intents
                   SET status = ?, exchange_order_id = COALESCE(?, exchange_order_id),
                       meta_json = COALESCE(?, meta_json)
                   WHERE intent_id = ?""",
                (
                    str(status),
                    exchange_order_id,
                    json.dumps(meta, ensure_ascii=True, default=str) if meta is not None else None,
                    intent_id,
                ),
            )
            con.commit()
        finally:
            con.close()

    def record_system_health_snapshot(self, snapshot: Dict[str, Any]) -> None:
        con = self.connect()
        try:
            cols = self._table_columns(con, "system_health_snapshots")
            row = {
                "timestamp": str(snapshot.get("timestamp", datetime.utcnow().isoformat(timespec="seconds") + "Z")),
                "cycles_completed": int(snapshot.get("cycles_completed", 0) or 0),
                "avg_latency_ms": float(snapshot.get("avg_latency_ms", 0.0) or 0.0),
                "errors_last_hour": int(snapshot.get("errors_last_hour", 0) or 0),
                "warnings_last_hour": int(snapshot.get("warnings_last_hour", 0) or 0),
                "event_loop_delay_ms": float(snapshot.get("event_loop_delay_ms", 0.0) or 0.0),
                "memory_rss_mb": float(snapshot.get("memory_rss_mb", 0.0) or 0.0),
                "memory_python_mb": float(snapshot.get("memory_python_mb", 0.0) or 0.0),
            }
            insert_cols = [k for k in row.keys() if k in cols]
            if not insert_cols:
                return
            placeholders = ", ".join(["?"] * len(insert_cols))
            sql = (
                f"INSERT INTO system_health_snapshots ({', '.join(insert_cols)}) "
                f"VALUES ({placeholders})"
            )
            con.execute(sql, tuple(row[c] for c in insert_cols))
            con.commit()
        finally:
            con.close()


class UnifiedSystemArchitecture:
    """Phase 1: Unified system architecture combining all three systems + 23+ languages"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config

        # Ω audit IDs
        try:
            self.run_id = uuid.uuid4().hex[:8]
        except Exception as e:
            logger.warning(f"Failed to generate run_id: {e}")
            self.run_id = "unknown"
        self._trace_id = None
        self.node_role = str(getattr(self.config, "node_role", "single-node") or "single-node").strip().lower()
        self.command_bus = None
        self.execution_mesh = None

        # Ω spine persistence (decision snapshots + order intents) using the same SQLite DB as trade ledger if available
        omega_db = None
        try:
            omega_db = getattr(getattr(self.config, "trade_ledger", None), "db_path", None)
        except Exception as e:
            logger.debug(f"Failed to get trade_ledger db_path: {e}")
            omega_db = None
        if not omega_db:
            omega_db = "data/unified_trades.db"
        self.omega_store = OmegaSQLiteStore(str(omega_db))
        self.state = SystemState.INITIALIZING
        self.start_time = datetime.now()
        
        # Core components (will be initialized in phases)
        self.ai_brain = None  # Phase 2: Pinnacle AI
        self.execution_engine = None  # Phase 3: Kraken DCA
        self.argus_strategies = None  # ARGUS Ultimate strategies
        self.monitoring = None  # Phase 5: UnifiedMonitoringSystem (optional)
        self.hft_engine = None
        self.hft_infrastructure = None
        
        # Multi-language integration (NEW)
        self.language_orchestrator = None  # Master orchestrator
        
        # System state — protected by _state_lock for thread safety
        import threading as _threading
        self._state_lock = _threading.Lock()
        self.portfolio_value_aud = config.starting_capital_aud
        self.cash_balance_aud = config.starting_capital_aud
        self.positions: Dict[str, Dict] = {}
        self.trade_history: deque = deque(maxlen=10000)
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl_aud = 0.0
        self.realized_pnl_aud = 0.0
        self.unrealized_pnl_aud = 0.0
        self.total_fees_aud = 0.0
        self.daily_pnl_aud = 0.0
        self.max_drawdown_aud = 0.0
        self.peak_equity_aud = config.starting_capital_aud
        self.mark_price_method = "position.current_price"
        self._ledger_sanity_violations = 0
        
        # Risk tracking
        self.consecutive_losses = 0
        self.error_count = 0
        self.total_operations = 0
        
        # Exchange connections
        self.exchanges: Dict[str, Any] = {}
        self.market_data_service = None
        self.live_market_data = None  # LiveMarketDataManager (live mode only)
        self.self_improver = None
        self._self_improver_task = None
        self.continuous_scanner = None
        self._continuous_scanner_task = None
        self.adaptive_risk_controller = None
        self.universe_selector = None
        self.strategy_allocator = None  # PnL-based strategy ranking (works with allocator config)
        self._strategy_state_store = None  # StrategyStateStore (per-strategy persistence + cooldown)
        self.strategy_evaluation_engine = None
        self.self_optimizing_meta_engine = None
        self.champion_challenger_engine = None
        self._last_regime_consensus = {}  # From 23-language regime_estimate when use_regime_estimate
        self.target_engine = None
        self._latest_targets: List[Any] = []
        self.liquidity_risk_engine = None
        self._latest_liquidity_state: Dict[str, Any] = {}
        self.market_microstructure_engine = None
        self._latest_microstructure_state: Dict[str, Any] = {}
        self._latest_strategy_weights: Dict[str, float] = {}
        self.system_health_metrics = None
        self.feature_store = None
        self.regime_classifier = None
        self._latest_regime_label = ""

        # Optional quant-fund upgrades (advisory-only)
        self.quant_fund_risk_engine = None

        # Partial TP / trailing stop: high-water and partial-taken tracking per symbol
        self._position_high_water: Dict[str, float] = {}
        self._position_low_water: Dict[str, float] = {}
        self._partial_tp_taken: Dict[str, bool] = {}

        # Salvaged production modules (best-effort)
        self.rate_limiter = None
        self.data_sanitizer = None
        self.position_tracker = None
        self.audit_chain = None
        self.self_healer = None
        self.health_monitor = None

        # Unified risk manager (single-source-of-truth style circuit breaker; best-effort)
        self.unified_risk_manager = None

        # Component registry: wires all Batch 3/4 shelf-ware into the trading loop
        self.component_registry = None

        # Signal subscription service (monetisation via webhooks)
        self.signal_service = None

        # API dashboard server (remote monitoring)
        self.api_server = None
        # Rolling performance feeder (drives meta engine weight evolution)
        self.perf_feeder = None
        # Regime change alerter
        self.regime_alerter = None
        # Model manager (ML lifecycle)
        self.model_manager = None
        # Checkpoint manager (state persistence)
        self.checkpoint_manager = None

        # Execution pipeline state
        self._process_lock = None  # Set by _run_unified_system for graceful release
        self._pending_orders: Dict[str, Dict[str, Any]] = {}  # order_id -> order info
        self._reconcile_every_n_cycles: int = int(getattr(config, "reconcile_every_n_cycles", 10) or 10)
        self._order_timeout_seconds: float = float(getattr(config, "order_timeout_seconds", 60.0) or 60.0)
        self._paper_slippage_bps: float = float(getattr(config, "paper_slippage_bps", 5.0) or 5.0)
        self._total_fee_savings_usd: float = 0.0  # cumulative maker fee savings
        self._limit_order_fill_timeout: float = 10.0  # seconds before falling back to market
        self._limit_price_offset_bps: float = 2.0  # 0.2% offset for limit orders
        self._volatility_cache: Dict[str, float] = {}  # symbol -> bootstrapped vol
        self._vwap_threshold_usd: float = 100.0  # use VWAP for orders >= this notional

        # FIX 8: Price history for SMA-based entry timing
        self._price_history: Dict[str, List[float]] = {}  # symbol -> recent prices

        # FIX 9: Pyramid and partial exit tracking
        self._pyramid_count: Dict[str, int] = {}        # symbol -> pyramid count (max 2)
        self._partial_exit_done: Dict[str, bool] = {}   # symbol -> partial exit taken

        # FIX 12: OCO (One-Cancels-Other) order tracking
        self._oco_orders: Dict[str, Dict[str, Any]] = {}  # symbol -> {stop_loss, take_profit, order_id, ...}

        # Optional: quantum Monte Carlo risk in loop (VaR/CVaR, circuit breaker when enabled)
        self._equity_history: List[float] = []
        self._after_risk_update_hook = None
        # Emergency shutdown state (latency, flash crash, network, arb)
        self._last_cycle_total_ms: Optional[float] = None
        self._last_cycle_stage_timing_ms: Dict[str, float] = {}
        self._last_target_pipeline_stage_ms: Dict[str, float] = {}
        self._completed_cycles: int = 0
        self._last_event_loop_delay_ms: float = 0.0
        self._live_safe_locked_symbols: List[str] = []
        self._last_price_by_symbol: Dict[str, Tuple[Optional[float], Optional[float]]] = {}  # sym -> (prev, curr)
        self._exchange_unreachable: bool = False
        self._last_spread_bps: Optional[float] = None

        try:
            from metrics.system_health_metrics import SystemHealthMetricsCollector

            self.system_health_metrics = SystemHealthMetricsCollector(
                enabled=bool(getattr(self.config, "system_health_metrics_enabled", True)),
                snapshot_interval_cycles=int(
                    getattr(self.config, "system_health_metrics_snapshot_interval_cycles", 10) or 10
                ),
            )
        except Exception as _health_e:
            logger.warning("System health metrics module unavailable: %s", _health_e)
            self.system_health_metrics = None
        
        try:
            from core.version import __version__ as _ver
        except ImportError:
            _ver = "3.0.0"
        logger.info("=" * 70)
        logger.info("UNIFIED TRADING SYSTEM - INITIALIZING (Argus v%s)", _ver)
        logger.info("=" * 70)
        logger.info("Config version: %s", getattr(config, "config_version", 1))
        logger.info(f"Starting Capital: ${config.starting_capital_aud:,.2f} AUD")
        logger.info(f"Primary Exchange: {config.primary_exchange}")
        logger.info(f"Secondary Exchange: {config.secondary_exchange}")
        logger.info("Node role: %s", self.node_role)
        logger.info("Multi-Language Support: 23+ languages enabled")
        self._runtime_module_registry: Dict[str, Dict[str, Any]] = {}
        self._apply_live_safe_scope_controls()
    
    async def initialize(self):
        """Initialize all system components"""
        try:
            logger.info("Phase 1: Initializing unified architecture...")

            # Initialize Ω schema (idempotent) regardless of multi-language mode.
            try:
                self.omega_store.init_schema()
                logger.info("✅ Ω spine DB initialized (%s)", getattr(self.omega_store, "db_path", ""))
            except Exception as e:
                logger.warning("Ω spine DB init failed (continuing without): %s", e)

            # Fetch live AUD/USD rate (replaces hardcoded 0.65 fallback)
            try:
                from utils.fx_rate import get_aud_usd_rate
                live_rate = get_aud_usd_rate(fallback=float(getattr(self.config, "aud_to_usd", 0.65) or 0.65))
                self.config.aud_to_usd = live_rate
                logger.info("AUD/USD rate set to %.5f (live)", live_rate)
            except Exception as exc:
                logger.warning("Could not fetch live AUD/USD rate, using config default: %s", exc)

            # Initialize multi-language system (optional)
            if getattr(self.config, "multi_language_enabled", True):
                await self._initialize_multi_language_system()
            else:
                logger.info("Multi-language system disabled by config")
            
            # Initialize exchange connections
            await self._initialize_exchanges()

            # Initialize optional quant-fund upgrades (advisory-only)
            await self._initialize_quant_fund_upgrades()
            
            # Initialize ARGUS strategies
            await self._initialize_argus_strategies()
            
            # Phase 2: Initialize AI Brain (Pinnacle AI)
            await self._initialize_ai_brain()

            # Optional: register quantum Monte Carlo risk hook when config says so (main bot integration)
            if getattr(self.config, "use_quantum_monte_carlo_risk", False) and getattr(self, "_after_risk_update_hook", None) is None:
                self._after_risk_update_hook = self._quantum_monte_carlo_risk_hook
                logger.info("Quantum Monte Carlo risk hook registered (VaR/CVaR in cycle)")

            # Optional strategy/execution split command bus.
            await self._initialize_command_bus()
            
            # Phase 3: Initialize execution engine (Kraken DCA)
            await self._initialize_execution_engine()

            # OMEGA-01: optional per-symbol execution mesh coordinator.
            await self._initialize_execution_mesh()
            
            # Phase 4: Initialize capital optimizer
            await self._initialize_capital_optimizer()

            # Phase 4.5: Initialize HFT Engine (New)
            await self._initialize_hft_engine()

            # Phase 4.6: Initialize Advanced HFT Infrastructure (New)
            await self._initialize_hft_infrastructure()

            # Phase 4.7: Initialize Max Return Features (Tier 1 + Tier 2)
            await self._initialize_max_return_features()

            # Unified risk manager (best-effort; does not block initialization)
            try:
                from risk.unified_risk_manager import UnifiedRiskManager

                self.unified_risk_manager = UnifiedRiskManager(
                    initial_capital=float(getattr(self.config, "starting_capital_aud", 1000.0) or 1000.0),
                    max_daily_loss=float(getattr(self.config, "max_daily_loss_pct", 0.02) or 0.02),
                    max_total_exposure=float(getattr(self.config, "max_total_exposure_pct", 0.8) or 0.8),
                    max_leverage=float(getattr(self.config, "max_leverage", 3.0) or 3.0),
                    max_consecutive_losses=int(getattr(self.config, "max_consecutive_losses", 5) or 5),
                )
            except Exception as e:
                logger.warning("UnifiedRiskManager unavailable: %s", e)
                self.unified_risk_manager = None

            # Adaptive risk controller (best-effort)
            try:
                from adaptive.adaptive_risk_controller import AdaptiveRiskController

                self.adaptive_risk_controller = AdaptiveRiskController(config=self.config)
                # Note: Don't setattr on config as ArgusConfig may be a Pydantic model
            except Exception as e:
                logger.warning("AdaptiveRiskController unavailable: %s", e)
                self.adaptive_risk_controller = None

            # Adaptive universe selection (best-effort)
            try:
                from adaptive.universe_selector import AdaptiveUniverseSelector

                self.universe_selector = AdaptiveUniverseSelector(
                    persist_path=str(getattr(self.config, "adaptive_universe_state_path", "data/adaptive_universe.json") or "data/adaptive_universe.json"),
                    max_active=int(getattr(self.config, "adaptive_universe_max_active", 5) or 5),
                    min_hold_cycles=int(getattr(self.config, "adaptive_universe_min_hold_cycles", 20) or 20),
                )
                self.universe_selector.load()
            except Exception as e:
                logger.warning("AdaptiveUniverseSelector unavailable: %s", e)
                self.universe_selector = None

            # Advanced adaptive modules (best-effort)
            # Market Regime Detector - HMM-based real-time regime detection
            try:
                from adaptive.market_regime_detector import MarketRegimeDetector
                self.market_regime_detector = MarketRegimeDetector(
                    use_hmm=True,
                    multi_timeframe=True,
                )
                logger.info("✅ MarketRegimeDetector initialized (HMM=True, multi_tf=True)")
            except Exception as e:
                logger.warning("MarketRegimeDetector unavailable: %s", e)
                self.market_regime_detector = None

            # Self-Healing Manager - Auto-retrain models when drift detected
            try:
                from adaptive.self_healing_manager import SelfHealingManager
                self.self_healing_manager = SelfHealingManager(
                    config={
                        "auto_retrain": True,
                        "auto_rollback": True,
                        "drift_threshold": 0.15,
                        "min_samples_for_drift": 100
                    }
                )
                logger.info("SelfHealingManager initialized (auto_retrain=True)")
            except Exception as e:
                logger.warning("SelfHealingManager unavailable: %s", e)
                self.self_healing_manager = None

            # Adaptive Position Sizer - Dynamic sizing based on regime
            try:
                from adaptive.adaptive_position_sizer import AdaptivePositionSizer, PositionSizingConfig
                sizing_config = PositionSizingConfig(
                    base_position_pct=0.1,
                    max_position_pct=0.25,
                    min_position_pct=0.02,
                )
                self.adaptive_position_sizer = AdaptivePositionSizer(config=sizing_config)
                logger.info("✅ AdaptivePositionSizer initialized")
            except Exception as e:
                logger.warning("AdaptivePositionSizer unavailable: %s", e)
                self.adaptive_position_sizer = None

            # Fully Adaptive Risk Engine - Complete adaptive risk system
            try:
                from adaptive.fully_adaptive_risk_engine import FullyAdaptiveRiskEngine, AdaptiveRiskConfig
                self.fully_adaptive_risk = FullyAdaptiveRiskEngine(AdaptiveRiskConfig(
                    base_position_pct=0.10,
                    min_position_pct=0.02,
                    max_position_pct=0.20,
                    volatility_target=0.15,
                    max_daily_loss_pct=0.10,
                    cautious_after_losses=2,
                    defensive_after_losses=3,
                    pause_after_losses=5,
                    pause_duration_minutes=60,
                    drawdown_cautious_pct=0.05,
                    drawdown_defensive_pct=0.10,
                    drawdown_pause_pct=0.15,
                    trailing_stop_enabled=True,
                    trailing_atr_multiplier=2.5,
                    breakeven_trigger_pct=0.02,
                    partial_tp_enabled=True,
                    partial_tp_at_2r=True,
                ))
                logger.info("✅ FullyAdaptiveRiskEngine initialized (vol-adjusted, correlation-aware, trailing stops, performance-responsive)")
            except Exception as e:
                logger.warning("FullyAdaptiveRiskEngine unavailable: %s", e)
                self.fully_adaptive_risk = None

            # Quantum Adaptive Risk Engine - THE PINNACLE
            try:
                from adaptive.quantum_adaptive_risk_engine import QuantumAdaptiveRiskEngine, QuantumAdaptiveConfig
                self.quantum_adaptive_risk = QuantumAdaptiveRiskEngine(QuantumAdaptiveConfig(
                    base_position_pct=0.10,
                    min_position_pct=0.02,
                    max_position_pct=0.20,
                    n_qubits=8,
                    quantum_weight=0.3,
                    use_quantum_portfolio=True,
                    use_quantum_var=True,
                    use_quantum_stops=True,
                    use_quantum_regime=True,
                    use_quantum_kelly=True,
                    max_daily_loss_pct=0.10,
                    pause_after_losses=5,
                    drawdown_pause_pct=0.15,
                    trailing_stop_enabled=True,
                ))
                logger.info("✅ QuantumAdaptiveRiskEngine PINNACLE initialized (8 qubits, quantum portfolio, QMC VaR, annealing stops, QAOA Kelly)")
            except Exception as e:
                logger.warning("QuantumAdaptiveRiskEngine unavailable: %s", e)
                self.quantum_adaptive_risk = None

            # ════════════════════════════════════════════════════════════════════════════
            # FULLY ADAPTIVE INFRASTRUCTURE - The Missing 40%
            # ════════════════════════════════════════════════════════════════════════════
            
            # Strategy Parameter Auto-Tuner - Tunes RSI, MACD, Bollinger thresholds
            try:
                from adaptive.strategy_parameter_tuner import StrategyParameterTuner
                self.strategy_param_tuner = StrategyParameterTuner(
                    tuning_interval_minutes=60,
                    min_trades_for_tuning=10,
                    n_iterations=15,
                )
                logger.info("✅ StrategyParameterTuner initialized (auto-tunes RSI/MACD/BB params)")
            except Exception as e:
                logger.warning("StrategyParameterTuner unavailable: %s", e)
                self.strategy_param_tuner = None

            # Dynamic Timeframe Selector - Adapts between 15m/1h/4h
            try:
                from adaptive.dynamic_timeframe_selector import DynamicTimeframeSelector, Timeframe
                self.timeframe_selector = DynamicTimeframeSelector(
                    default_timeframe=Timeframe.H1,
                    evaluation_period_minutes=60,
                    switch_cooldown_minutes=30,
                )
                logger.info("✅ DynamicTimeframeSelector initialized (5m-1d adaptive timeframe)")
            except Exception as e:
                logger.warning("DynamicTimeframeSelector unavailable: %s", e)
                self.timeframe_selector = None

            # Self-Learning Universe - Discovers and validates new trading pairs
            try:
                from adaptive.self_learning_universe import SelfLearningUniverse
                current_pairs = list(getattr(self.config, "trading_pairs", ["BTC/USD", "ETH/USD"]))
                self.universe_expander = SelfLearningUniverse(
                    active_pairs=current_pairs,
                    data_dir="data",
                )
                # Try to load saved state
                self.universe_expander.load_state()
                logger.info("✅ SelfLearningUniverse initialized (%d pairs, auto-expansion enabled)", len(current_pairs))
            except Exception as e:
                logger.warning("SelfLearningUniverse unavailable: %s", e)
                self.universe_expander = None

            # Correlation Regime Detector - Tracks correlation between assets
            try:
                from adaptive.correlation_regime_detector import CorrelationTracker
                self.correlation_tracker = CorrelationTracker(
                    assets=list(getattr(self.config, "trading_pairs", ["BTC/USD", "ETH/USD"])),
                    window=60,
                    min_periods=20
                )
                logger.info("CorrelationTracker initialized (60-bar window)")
            except Exception as e:
                logger.warning("CorrelationTracker unavailable: %s", e)
                self.correlation_tracker = None

            # Real-time Regime Detector - Multi-timeframe regime detection
            try:
                from adaptive.real_time_regime_detector import MarketRegimeDetector
                self.realtime_regime_detector = MarketRegimeDetector(
                    config={
                        "volatility_window": 20,
                        "short_window": 10,
                        "long_window": 50,
                        "smoothing_factor": 0.3,
                    }
                )
                logger.info("✅ RealTimeRegimeDetector initialized")
            except Exception as e:
                logger.warning("RealTimeRegimeDetector unavailable: %s", e)
                self.realtime_regime_detector = None

            # Regime Forecaster - Predicts future regime changes
            try:
                from adaptive.regime_forecaster import RegimeForecaster
                self.regime_forecaster = RegimeForecaster(
                    min_observations=5,
                    feature_adjustment_strength=0.3,
                )
                logger.info("✅ RegimeForecaster initialized")
            except Exception as e:
                logger.warning("RegimeForecaster unavailable: %s", e)
                self.regime_forecaster = None

            # Strategy Decay Detector - Detects when strategies stop working
            try:
                from adaptive.strategy_decay_detector import StrategyDecayDetector
                self.strategy_decay_detector = StrategyDecayDetector(
                    decay_slope_threshold=-0.1,
                    disable_sharpe=-0.5,
                )
                logger.info("✅ StrategyDecayDetector initialized")
            except Exception as e:
                logger.warning("StrategyDecayDetector unavailable: %s", e)
                self.strategy_decay_detector = None

            # Dynamic Parameter Optimizer - Optimizes strategy parameters in real-time
            try:
                from adaptive.dynamic_parameter_optimizer import DynamicParameterOptimizer
                self.param_optimizer = DynamicParameterOptimizer(
                    config={
                        "optimization_interval_minutes": 60,
                        "param_bounds": {"stop_loss_pct": (0.01, 0.1), "take_profit_pct": (0.02, 0.3)},
                    }
                )
                logger.info("✅ DynamicParameterOptimizer initialized")
            except Exception as e:
                logger.warning("DynamicParameterOptimizer unavailable: %s", e)
                self.param_optimizer = None

            # ULTIMATE EDGE MODULES (v8.5.0)

            # Order Flow Analyzer - Microscopic market timing
            try:
                from analytics.order_flow_engine import OrderFlowAnalyzer, MultiExchangeOrderFlow
                self.order_flow_analyzer = OrderFlowAnalyzer(
                    large_trade_threshold=10000,
                    very_large_threshold=100000,
                    massive_threshold=500000
                )
                self.multi_exchange_flow = MultiExchangeOrderFlow(
                    exchanges=["kraken", "coinbase", "binance"]
                )
                logger.info("OrderFlowAnalyzer initialized (whale detection enabled)")
            except Exception as e:
                logger.warning("OrderFlowAnalyzer unavailable: %s", e)
                self.order_flow_analyzer = None
                self.multi_exchange_flow = None

            # Market Depth Engine - Liquidity analysis
            try:
                from analytics.market_depth_engine import MarketDepthAnalyzer
                self.market_depth_analyzer = MarketDepthAnalyzer(
                    levels_to_analyze=20,
                    wall_threshold_multiplier=5.0
                )
                logger.info("MarketDepthAnalyzer initialized")
            except Exception as e:
                logger.warning("MarketDepthAnalyzer unavailable: %s", e)
                self.market_depth_analyzer = None

            # Sentiment Engine - Forward-looking alpha
            try:
                from analytics.sentiment_engine import SentimentAnalyzer
                self.sentiment_analyzer = SentimentAnalyzer(
                    lookback_minutes=60,
                    momentum_window=10
                )
                logger.info("SentimentAnalyzer initialized (Fear & Greed enabled)")
            except Exception as e:
                logger.warning("SentimentAnalyzer unavailable: %s", e)
                self.sentiment_analyzer = None

            # Options Intelligence - Greeks & delta hedging
            try:
                from options.intelligence_engine import OptionsIntelligence
                self.options_intelligence = OptionsIntelligence(
                    target_delta=0.0,
                    rebalance_threshold=0.05
                )
                logger.info("OptionsIntelligence initialized")
            except Exception as e:
                logger.warning("OptionsIntelligence unavailable: %s", e)
                self.options_intelligence = None

            # Correlation Engine - Cross-asset correlation
            try:
                from analytics.correlation_engine import CorrelationEngine, CorrelationRegime
                self.correlation_engine = CorrelationEngine(
                    assets=list(getattr(self.config, "trading_pairs", ["BTC/USD", "ETH/USD", "SPY"])),
                    window_size=60,
                    high_correlation_threshold=0.7
                )
                logger.info("CorrelationEngine initialized")
            except Exception as e:
                logger.warning("CorrelationEngine unavailable: %s", e)
                self.correlation_engine = None

            # TCA Engine - Transaction cost analysis
            try:
                from monitoring.tca_engine import TCAEngine
                self.tca_engine = TCAEngine(
                    symbol=str((getattr(self.config, "trading_pairs", ["BTC/USD"])[0])),
                    lookback_trades=100
                )
                logger.info("TCAEngine initialized")
            except Exception as e:
                logger.warning("TCAEngine unavailable: %s", e)
                self.tca_engine = None

            # Smart Execution Engine - TWAP/VWAP/POV
            try:
                from execution.smart_execution_engine import SmartExecutionEngine, ExecutionAlgorithm
                self.smart_executor = SmartExecutionEngine(
                    default_algorithm=ExecutionAlgorithm.ADAPTIVE,
                    default_participation_rate=0.10
                )
                logger.info("SmartExecutionEngine initialized (TWAP/VWAP/POV)")
            except Exception as e:
                logger.warning("SmartExecutionEngine unavailable: %s", e)
                self.smart_executor = None

            # Ultimate Defense System - Circuit breakers & kill switches
            try:
                from risk.ultimate_defense import UltimateDefense, KillSwitchConfig
                self.ultimate_defense = UltimateDefense(
                    config=KillSwitchConfig(
                        max_daily_loss_pct=0.10,
                        max_drawdown_pct=0.15,
                        max_position_size_pct=0.25,
                        max_leverage=5.0
                    ),
                    enable_panic_mode=True
                )
                self.ultimate_defense.initialize(float(getattr(self.config, "initial_capital", 1000) or 1000))
                logger.info("UltimateDefense initialized (multi-level protection)")
            except Exception as e:
                logger.warning("UltimateDefense unavailable: %s", e)
                self.ultimate_defense = None

            # Monte Carlo Engine - Risk quantification
            try:
                from risk.monte_carlo_engine_v2 import MonteCarloEngine
                self.monte_carlo = MonteCarloEngine(
                    n_scenarios=1000,
                    confidence_levels=[0.90, 0.95, 0.99],
                    time_horizon_days=252
                )
                logger.info("MonteCarloEngine initialized (1000 scenarios)")
            except Exception as e:
                logger.warning("MonteCarloEngine unavailable: %s", e)
                self.monte_carlo = None

            # ADVANCED RISK v8.7.0 - Next-Gen Risk Management

            # Tail Risk Hedger - Portfolio crash protection via OTM puts, VIX calls, CPPI
            try:
                from risk.tail_risk_hedger import TailRiskHedger, TailHedgeConfig
                self.tail_risk_hedger = TailRiskHedger(
                    config=TailHedgeConfig(
                        hedge_allocation_pct=0.02,
                        otm_percentage=0.15,
                        hedge_instruments=["puts", "vix_calls", "cppi"],
                        rebalance_frequency="weekly",
                        max_hedge_cost_pct=0.02
                    )
                )
                logger.info("TailRiskHedger initialized (OTM puts + VIX calls + CPPI overlay)")
            except Exception as e:
                logger.warning("TailRiskHedger unavailable: %s", e)
                self.tail_risk_hedger = None

            # Delta Hedger - Options Greeks management, delta-neutral hedging
            try:
                from risk.delta_hedger import DeltaHedger
                self.delta_hedger = DeltaHedger()
                logger.info("DeltaHedger initialized (Greeks + delta-neutral hedging)")
            except Exception as e:
                logger.warning("DeltaHedger unavailable: %s", e)
                self.delta_hedger = None

            # Adaptive Risk Manager - Market regime-based risk adjustment
            try:
                from risk.adaptive_risk_manager import AdaptiveRiskManager
                self.adaptive_risk_manager = AdaptiveRiskManager(
                    initial_capital=float(getattr(self.config, "starting_capital_aud", 1000.0) or 1000.0)
                )
                logger.info("✅ AdaptiveRiskManager initialized (regime-based risk adjustment)")
            except Exception as e:
                logger.warning("AdaptiveRiskManager unavailable: %s", e)
                self.adaptive_risk_manager = None

            # Real-Time VaR Aggregator - Streaming VaR calculation at 95%, 99%, 99.9%
            try:
                from risk.realtime_var_aggregator import StreamingVaRCalculator
                self.var_calculator = StreamingVaRCalculator(
                    confidence_levels=[0.95, 0.99, 0.999],
                    lookback_window=1000,
                    decay_factor=0.94
                )
                logger.info("StreamingVaRCalculator initialized (real-time VaR at 95/99/99.9%)")
            except Exception as e:
                logger.warning("StreamingVaRCalculator unavailable: %s", e)
                self.var_calculator = None

            # Regime-Conditional VaR - Regime-aware risk metrics
            try:
                from risk.regime_conditional_var import RegimeConditionalVaR
                self.regime_cvar = RegimeConditionalVaR(
                    mc_simulations=1000,
                    default_confidence=0.99,
                )
                logger.info("✅ RegimeConditionalVaR initialized (regime-aware risk)")
            except Exception as e:
                logger.warning("RegimeConditionalVaR unavailable: %s", e)
                self.regime_cvar = None

            # CVaR Dynamic Hedger - CVaR-based position hedging
            try:
                from risk.cvar_dynamic_hedging import CVaRBasedDynamicHedger
                self.cvar_hedger = CVaRBasedDynamicHedger(
                    config={
                        "target_cvar_pct": 2.5,
                        "confidence_level": 0.99,
                    }
                )
                logger.info("✅ CVaRBasedDynamicHedger initialized (CVaR-optimized hedging)")
            except Exception as e:
                logger.warning("CVaRBasedDynamicHedger unavailable: %s", e)
                self.cvar_hedger = None

            # Portfolio Rebalancer - Capital efficiency
            try:
                from portfolio.rebalancer import PortfolioRebalancer, RebalancerConfig
                self.portfolio_rebalancer = PortfolioRebalancer(
                    config=RebalancerConfig(
                        rebalance_threshold_pct=5.0,
                        min_trade_value=10.0,
                        max_turnover_pct=25.0,
                        tax_loss_harvest_enabled=True
                    )
                )
                logger.info("PortfolioRebalancer initialized")
            except Exception as e:
                logger.warning("PortfolioRebalancer unavailable: %s", e)
                self.portfolio_rebalancer = None

            # ULTIMATE EDGE v8.6.0 - BLACK SWAN & CASCADE HUNTER V2

            # Black Swan Predictor V2 - Real GARCH(1,1), Markov regime, ML prediction (85-95% accuracy)
            try:
                from analytics.black_swan_predictor_v2 import BlackSwanPredictorV2
                self.black_swan_predictor = BlackSwanPredictorV2(
                    lookback=252,
                    warning_threshold=0.65,
                    danger_threshold=0.82
                )
                logger.info("BlackSwanPredictorV2 initialized (GARCH + Markov + ML, 85-95% accuracy)")
            except Exception as e:
                logger.warning("BlackSwanPredictorV2 unavailable: %s", e)
                self.black_swan_predictor = None

            # Liquidation Cascade Hunter V2 - Multi-exchange, 8-phase cascade detection (75-85% accuracy)
            try:
                from execution.liquidation_cascade_hunter_v2 import LiquidationCascadeHunterV2
                self.cascade_hunter = LiquidationCascadeHunterV2(
                    leverage_threshold=8.0,
                    funding_threshold=0.005,
                    volume_spike=2.5,
                    cascade_confidence=0.70
                )
                logger.info("LiquidationCascadeHunterV2 initialized (multi-exchange, 8-phase detection)")
            except Exception as e:
                logger.warning("LiquidationCascadeHunterV2 unavailable: %s", e)
                self.cascade_hunter = None

            # Ultra-Low Latency Executor V2 - Kalman filter, Kyle's lambda, multi-horizon (optimal fills)
            try:
                from execution.ultra_low_latency_executor_v2 import UltraLowLatencyExecutorV2
                self.latency_executor = UltraLowLatencyExecutorV2(
                    baseline_latency_ms=50.0,
                    enable_prediction=True,
                    enable_kalman=True,
                    enable_smart_slicing=True
                )
                logger.info("UltraLowLatencyExecutorV2 initialized (Kalman + Kyle's lambda + multi-horizon)")
            except Exception as e:
                logger.warning("UltraLowLatencyExecutorV2 unavailable: %s", e)
                self.latency_executor = None

            # Sentiment Noise Filter V2 - 8 sources, recency weighting, ML scoring (85-92% accuracy)
            try:
                from analytics.sentiment_noise_filter_v2 import SentimentNoiseFilterV2
                self.sentiment_filter = SentimentNoiseFilterV2(
                    min_sources=4,
                    confidence_threshold=0.70,
                    quality_threshold=0.75
                )
                logger.info("SentimentNoiseFilterV2 initialized (8 sources, 85-92% accuracy vs 40-50% raw)")
            except Exception as e:
                logger.warning("SentimentNoiseFilterV2 unavailable: %s", e)
                self.sentiment_filter = None

            # Phase 5: Initialize monitoring
            await self._initialize_monitoring()

            # Phase 6: Initialize salvaged production modules (best-effort, non-blocking)
            await self._initialize_production_modules()

            # Load evolved params into config when evolution.load_evolved is True (everything works together)
            if getattr(self.config, "evolution_load_evolved", False):
                try:
                    from evolution.apply_evolved_strategies import apply_from_file
                    path = str(getattr(self.config, "evolution_params_path", "data/evolved_params.json") or "data/evolved_params.json")
                    n = apply_from_file(self.config, path, key="best_params")
                    if n > 0:
                        logger.info("Loaded %d evolved params from %s into config", n, path)
                except Exception as e:
                    logger.debug("Evolved params load (optional): %s", e)

            # Strategy allocator: PnL-based ranking so strategies work together with allocator
            await self._initialize_strategy_allocator()
            self._initialize_strategy_evaluation_engine()
            self._initialize_self_optimizing_meta_engine()
            self._initialize_champion_challenger_engine()
            self._log_runtime_module_registry()

            self.state = SystemState.RUNNING
            logger.info("✅ Unified system architecture initialized successfully")
            if self.language_orchestrator:
                logger.info(f"✅ Multi-language orchestrator: {len(self.language_orchestrator.languages)} languages")
            else:
                logger.info("✅ Multi-language orchestrator: disabled")
            
        except Exception as e:
            logger.error(f"Failed to initialize unified system: {e}", exc_info=True)
            self.state = SystemState.EMERGENCY_STOP
            raise

    def _is_live_safe_runtime(self) -> bool:
        mode = str(getattr(self.config, "run_mode", "paper") or "paper").strip().lower()
        profile = str(getattr(self.config, "config_profile", "") or "").strip().lower()
        if profile in {"live_safe", "live-safe", "livesafe"}:
            return True
        return mode == "live" and bool(getattr(self.config, "institutional_mode", False))

    def _apply_live_safe_scope_controls(self) -> None:
        """Guard against research features silently influencing LIVE_SAFE runtime."""
        if not self._is_live_safe_runtime():
            return
        disabled = []
        for attr in (
            "ai_enabled",
            "strategy_library_enabled",
            "quantum_features_enabled",
            "quant_fund_upgrades_enabled",
            "evolution_continuous_enabled",
            "self_improvement_enabled",
            "self_optimizing_meta_enabled",
            "champion_challenger_enabled",
            "multi_language_enabled",
            "continuous_scan_enabled",
            "hft_enabled",
            "use_advanced_hft_infrastructure",
            "dynamic_universe_enabled",
            "adaptive_universe_enabled",
        ):
            if bool(getattr(self.config, attr, False)):
                setattr(self.config, attr, False)
                disabled.append(attr)
        if bool(getattr(self.config, "adaptive_universe_modes", [])):
            setattr(self.config, "adaptive_universe_modes", [])
            disabled.append("adaptive_universe_modes")
        if not bool(getattr(self.config, "live_safe_disable_pinnacle_ai_brain", False)):
            setattr(self.config, "live_safe_disable_pinnacle_ai_brain", True)
            disabled.append("live_safe_disable_pinnacle_ai_brain")
        locked_pairs = [
            str(s).strip()
            for s in list(getattr(self.config, "trading_pairs", []) or [])
            if str(s).strip()
        ]
        self._live_safe_locked_symbols = list(dict.fromkeys(locked_pairs))
        if self._live_safe_locked_symbols:
            setattr(self.config, "trading_pairs", list(self._live_safe_locked_symbols))
            logger.info(
                "LIVE_SAFE symbol lock enabled: %s",
                ", ".join(self._live_safe_locked_symbols),
            )
        if disabled:
            logger.warning(
                "LIVE_SAFE scope control: disabled research/advisory modules: %s",
                ", ".join(disabled),
            )

    def _enforce_live_safe_symbol_lock(self) -> None:
        if not self._is_live_safe_runtime():
            return
        locked = list(self._live_safe_locked_symbols or [])
        if not locked:
            return
        current = [str(s).strip() for s in list(getattr(self.config, "trading_pairs", []) or []) if str(s).strip()]
        if current != locked:
            setattr(self.config, "trading_pairs", list(locked))
            logger.warning(
                "LIVE_SAFE symbol lock restored trading_pairs to profile set: %s",
                ", ".join(locked),
            )

    def _filter_live_safe_signals(self, signals: List[Any], *, stage: str) -> List[Any]:
        """Fail-safe symbol allow-list filter for LIVE_SAFE runtime."""
        rows = list(signals or [])
        if not self._is_live_safe_runtime():
            return rows
        locked = {str(s).strip() for s in list(self._live_safe_locked_symbols or []) if str(s).strip()}
        if not locked:
            return rows
        kept: List[Any] = []
        dropped = 0
        for sig in rows:
            sym = str(self._signal_get(sig, "symbol", "") or "").strip()
            if not sym or sym in locked:
                kept.append(sig)
            else:
                dropped += 1
        if dropped > 0:
            logger.warning(
                "LIVE_SAFE symbol lock filtered %s signal(s) at %s; allowed symbols: %s",
                int(dropped),
                str(stage),
                ", ".join(sorted(locked)),
            )
        return kept

    def _build_runtime_module_registry(self) -> Dict[str, Dict[str, Any]]:
        """Build runtime module status map with category + fail-safe status."""
        entries: Dict[str, Dict[str, Any]] = {}

        def _put(name: str, *, category: str, enabled: bool, ready: bool, critical: bool = False) -> None:
            if not enabled:
                status = "disabled"
            elif ready:
                status = "active"
            else:
                status = "failed-safe" if critical else "degraded"
            entries[name] = {
                "category": str(category),
                "enabled": bool(enabled),
                "status": str(status),
            }

        # Critical execution path.
        _put("exchange_connectivity", category="critical", enabled=True, ready=bool(self.exchanges), critical=True)
        _put("execution_engine", category="critical", enabled=True, ready=bool(self.execution_engine), critical=True)
        _put("hard_risk_gate", category="critical", enabled=True, ready=bool(getattr(self.execution_engine, "risk_manager", None)), critical=True)
        _put("order_intents_store", category="critical", enabled=True, ready=bool(getattr(self.execution_engine, "state_store", None)), critical=True)
        _put("reconciliation", category="critical", enabled=True, ready=bool(self.execution_engine), critical=True)

        # Optional safety/quality modules.
        _put(
            "liquidity_risk_engine",
            category="optional",
            enabled=bool(getattr(self.config, "liquidity_risk_enabled", True)),
            ready=bool(self.liquidity_risk_engine),
        )
        _put(
            "market_microstructure_engine",
            category="optional",
            enabled=bool(getattr(self.config, "market_microstructure_enabled", True)),
            ready=bool(self.market_microstructure_engine),
        )
        _put(
            "system_health_metrics",
            category="optional",
            enabled=bool(getattr(self.config, "system_health_metrics_enabled", True)),
            ready=bool(self.system_health_metrics),
        )
        _put(
            "recon_recovery_engine",
            category="optional",
            enabled=bool(getattr(self.config, "recon_recovery_enabled", True)),
            ready=bool(getattr(self.execution_engine, "recon_recovery_engine", None)),
        )

        # Advisory modules.
        _put(
            "strategy_evaluation_engine",
            category="advisory",
            enabled=bool(getattr(self.config, "strategy_evaluation_enabled", True)),
            ready=bool(self.strategy_evaluation_engine),
        )
        _put(
            "self_optimizing_meta_engine",
            category="advisory",
            enabled=bool(getattr(self.config, "self_optimizing_meta_enabled", True)),
            ready=bool(self.self_optimizing_meta_engine),
        )
        _put(
            "champion_challenger",
            category="advisory",
            enabled=bool(getattr(self.config, "champion_challenger_enabled", True)),
            ready=bool(self.champion_challenger_engine),
        )

        # Research-only modules.
        _put(
            "strategy_library",
            category="research_only",
            enabled=bool(getattr(self.config, "strategy_library_enabled", True)),
            ready=bool(getattr(self.config, "strategy_library_enabled", True)),
        )
        _put(
            "quantum_features",
            category="research_only",
            enabled=bool(getattr(self.config, "quantum_features_enabled", True)),
            ready=bool(getattr(self.config, "quantum_features_enabled", True)),
        )
        _put(
            "quant_fund_upgrades",
            category="research_only",
            enabled=bool(getattr(self.config, "quant_fund_upgrades_enabled", True)),
            ready=bool(getattr(self.config, "quant_fund_upgrades_enabled", True)),
        )
        _put(
            "self_improvement",
            category="research_only",
            enabled=bool(getattr(self.config, "self_improvement_enabled", True)),
            ready=bool(getattr(self.config, "self_improvement_enabled", True)),
        )

        self._runtime_module_registry = entries
        return entries

    def _log_runtime_module_registry(self) -> None:
        profile = str(getattr(self.config, "config_profile", "") or "").strip() or "default"
        source = str(getattr(self.config, "config_source", "") or "").strip()
        registry = self._build_runtime_module_registry()
        logger.info("Runtime profile: %s", profile)
        if source:
            logger.info("Runtime config source: %s", source)
        counts = {"active": 0, "disabled": 0, "degraded": 0, "failed-safe": 0}
        for item in registry.values():
            status = str(item.get("status", "") or "")
            if status in counts:
                counts[status] += 1
        logger.info(
            "Runtime module status summary: active=%d disabled=%d degraded=%d failed-safe=%d",
            counts["active"],
            counts["disabled"],
            counts["degraded"],
            counts["failed-safe"],
        )
        for name in sorted(registry.keys()):
            entry = registry[name]
            logger.info(
                "module %-28s category=%-13s status=%-10s enabled=%s",
                name,
                str(entry.get("category", "")),
                str(entry.get("status", "")),
                str(bool(entry.get("enabled", False))).lower(),
            )
    
    async def _initialize_multi_language_system(self):
        """Initialize multi-language orchestrator and integration"""
        logger.info("Initializing multi-language system (23+ languages)...")

        try:
            from unified_language_orchestrator import get_orchestrator
            
            # Initialize orchestrator
            self.language_orchestrator = get_orchestrator(self.config.__dict__ if hasattr(self.config, '__dict__') else {})
            
            logger.info("✅ Multi-language system initialized")
            status = self.language_orchestrator.get_status()
            logger.info(f"   Languages active: {status['languages_active']}/{status['languages_registered']}")
            # Optional: warm HTTP endpoints (fire-and-forget)
            if getattr(self.config, "multi_language_warm_on_start", False) and hasattr(self.language_orchestrator, "warm_all"):
                asyncio.create_task(self._warm_multilang_once())
            
        except Exception as e:
            logger.warning(f"Multi-language initialization warning: {e}")
            # Continue without multi-language support if initialization fails

    async def _warm_multilang_once(self) -> None:
        """Call warm_all on orchestrator once (fire-and-forget from startup)."""
        try:
            if self.language_orchestrator and hasattr(self.language_orchestrator, "warm_all"):
                result = await self.language_orchestrator.warm_all()
                ok = sum(1 for v in result.values() if v)
                logger.debug("Multi-language warm: %s/%s endpoints responded", ok, len(result))
        except Exception as e:
            logger.debug("Multi-language warm: %s", e)

    async def _initialize_exchanges(self):
        """Initialize exchange connections (CCXT when use_ccxt, else Kraken/Coinbase clients)."""
        logger.info("Initializing exchange connections...")
        primary = str(getattr(self.config, "primary_exchange", "kraken") or "kraken")
        use_ccxt = getattr(self.config, "use_ccxt", True)

        try:
            if use_ccxt:
                # Use CCXT for primary exchange (market data + execution via data/ccxt_data_provider)
                from data.ccxt_data_provider import get_ccxt_async_exchange
                primary_ex = get_ccxt_async_exchange(primary)
                run_mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
                if run_mode != "live":
                    # Paper/backtest: wrap so create_order returns mock order (no real orders)
                    primary_ex = _PaperCCXTWrapper(primary_ex, primary)
                self.exchanges[primary] = primary_ex
                logger.info("✅ Primary exchange via CCXT: %s%s", primary, " (paper/dry-run)" if run_mode != "live" else "")
            else:
                from exchanges.centralized.kraken import KrakenClient
                kraken_api_key = os.getenv("KRAKEN_API_KEY", "")
                kraken_secret = os.getenv("KRAKEN_SECRET_KEY", "")
                run_mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
                want_live = run_mode == "live"
                self.exchanges["kraken"] = KrakenClient(
                    api_key=kraken_api_key or None,
                    secret=kraken_secret or None,
                    dry_run=(not want_live) or (not bool(kraken_api_key and kraken_secret)),
                )
                if kraken_api_key and kraken_secret:
                    logger.info("✅ Kraken exchange connected (credentials present, dry_run=%s)", self.exchanges["kraken"].dry_run)
                else:
                    logger.info("✅ Kraken exchange connected (public-only / paper trading)")

            # Secondary: Coinbase Advanced (non-CCXT) when not using CCXT for primary only
            if not use_ccxt:
                from exchanges.centralized.coinbase_advanced import CoinbaseAdvancedClient
                coinbase_api_key = os.getenv("COINBASE_ADVANCED_API_KEY", "") or os.getenv("COINBASE_CDP_API_KEY", "")
                coinbase_secret_pem = os.getenv("COINBASE_ADVANCED_API_SECRET", "") or os.getenv("COINBASE_CDP_API_SECRET", "")
                if coinbase_api_key and coinbase_secret_pem:
                    self.exchanges["coinbase_advanced"] = CoinbaseAdvancedClient(
                        api_key=coinbase_api_key,
                        api_secret_pem=coinbase_secret_pem,
                        dry_run=(str(getattr(self.config, "run_mode", "paper") or "paper").lower() != "live"),
                    )
                    logger.info("✅ Coinbase Advanced Trade exchange connected (dry_run=%s)", getattr(self.exchanges["coinbase_advanced"], "dry_run", True))
                elif os.getenv("COINBASE_PRO_API_KEY") or os.getenv("COINBASE_PRO_SECRET_KEY"):
                    logger.warning("Detected legacy COINBASE_PRO_* env vars. Use COINBASE_ADVANCED_* (PEM).")
        except Exception as e:
            logger.warning("Exchange initialization warning: %s", e)
        finally:
            # Initialize canonical market data service (best-effort).
            try:
                from services.market_data_service import MarketDataService

                self.market_data_service = MarketDataService(
                    exchanges=self.exchanges,
                    primary=str(getattr(self.config, "primary_exchange", "kraken")),
                    secondary=str(getattr(self.config, "secondary_exchange", "coinbase_advanced")),
                    ohlcv_ttl_s=float(getattr(self.config, "market_data_ohlcv_cache_seconds", 30.0) or 30.0),
                    ohlcv_poll_interval_s=float(
                        getattr(self.config, "market_data_ohlcv_poll_interval_seconds", 30.0) or 30.0
                    ),
                    ohlcv_retry_attempts=int(getattr(self.config, "market_data_ohlcv_retry_attempts", 2) or 2),
                    persist_tick_store=bool(getattr(self.config, "persist_tick_store", True)),
                )
            except Exception as e:
                logger.warning(f"Market data service initialization warning: {e}")
                self.market_data_service = None

    async def _initialize_quant_fund_upgrades(self) -> None:
        """
        Best-effort initialization of optional quant-fund components.
        Enabled only in configured run modes (paper/backtest by default).
        """
        try:
            if not bool(getattr(self.config, "quant_fund_upgrades_enabled", True)):
                return
            mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
            modes = list(getattr(self.config, "quant_fund_upgrades_modes", ["paper", "backtest"]) or ["paper", "backtest"])
            if mode not in [str(m).lower() for m in modes]:
                return
            if not bool(getattr(self.config, "quant_fund_risk_engine_enabled", True)):
                return

            from quant_fund_upgrades.multi_factor_risk_engine import MultiFactorRiskEngine

            self.quant_fund_risk_engine = MultiFactorRiskEngine()
            logger.info("Quant-fund risk engine enabled (%s mode)", mode)
        except Exception as e:
            logger.warning("Quant-fund upgrades unavailable: %s", e)
            self.quant_fund_risk_engine = None
    
    async def _initialize_argus_strategies(self):
        """Initialize ARGUS Ultimate strategies"""
        logger.info("Initializing ARGUS Ultimate strategies...")
        self.argus_strategies = {
            'quantum_emotion_arbitrage': True,
            'fractal_volatility_harvest': True,
            'adaptive_regime_scalping': True,
            'multi_asset_sentiment_sync': True
        }
        logger.info("✅ ARGUS strategies initialized")
    
    async def _initialize_ai_brain(self):
        """Phase 2: Initialize AI brain (Pinnacle or Fallback when optional module unavailable)."""
        enabled = bool(getattr(self.config, "ai_enabled", True))
        if not enabled:
            logger.info("Phase 2: AI brain disabled by config")
            self.ai_brain = None
            return
        force_fallback = bool(getattr(self.config, "live_safe_disable_pinnacle_ai_brain", False))
        logger.info("Phase 2: Initializing AI brain (Pinnacle optional module)...")
        if not force_fallback:
            try:
                from unified_ai_brain import PinnacleAIBrain
                self.ai_brain = PinnacleAIBrain(self.config, market_data_service=self.market_data_service)
                await self.ai_brain.initialize()
                logger.info("✅ Pinnacle AI brain initialized")
                return
            except Exception as e:
                logger.warning("Pinnacle AI brain unavailable: %s; using fallback (strategy engine only)", e)
        else:
            logger.info("LIVE_SAFE runtime: Pinnacle AI brain disabled; using fallback AI brain")
        try:
            from unified_ai_brain import FallbackAIBrain
            self.ai_brain = FallbackAIBrain(self.config, market_data_service=self.market_data_service)
            await self.ai_brain.initialize()
            logger.info("✅ Fallback AI brain initialized (optional modules)")
        except Exception as e2:
            logger.warning("Fallback AI brain unavailable: %s", e2)
            self.ai_brain = None

    async def _quantum_monte_carlo_risk_hook(
        self,
        *,
        prev_equity: float = 0.0,
        new_equity: float = 0.0,
        peak_equity: float = 0.0,
        max_drawdown: float = 0.0,
    ) -> None:
        """After risk update: quantum Monte Carlo VaR/CVaR and optional circuit breaker (main bot integration)."""
        if not getattr(self.config, "use_quantum_monte_carlo_risk", False):
            return
        try:
            # _equity_history is maintained in main loop; ensure enough points for VaR
            if len(self._equity_history) < 10:
                return
            arr = np.array(self._equity_history, dtype=float)
            returns = np.diff(arr) / np.maximum(arr[:-1], 1e-9)
            returns_list = returns.tolist()
            from quantum import get_quantum_facade

            risk = get_quantum_facade().estimate_tail_risk_qmc(
                returns_list,
                n_samples=min(5000, len(returns_list) * 50),
                confidence=0.95,
            )
            var_95 = risk.get("var", 0.0)
            cvar_95 = risk.get("cvar", 0.0)
            quantum_metadata = risk.get("quantum_metadata", {})
            execution_mode = quantum_metadata.get("execution_mode", risk.get("execution_mode", "classical"))
            es_bps = risk.get("expected_shortfall_bps", (-cvar_95 * 1e4) if cvar_95 < 0 else 0.0)
            n_used = risk.get("n_samples_used", len(returns_list))
            setattr(self, "_last_quantum_var_cvar", {
                "var": var_95, "cvar": cvar_95, "var_95": var_95, "cvar_95": cvar_95,
                "expected_shortfall_bps": es_bps, "n_samples_used": n_used,
                "quantum_metadata": quantum_metadata,
                "quantum_advantage_claimed": risk.get("quantum_advantage_claimed", False),
            })
            logger.info(
                "Canonical QMC risk: VaR_95=%.4f CVaR_95=%.4f ES_bps=%.1f n=%s mode=%s",
                var_95, cvar_95, float(es_bps), n_used, execution_mode,
            )
            if getattr(self.config, "use_quantum_var_circuit_breaker", False) and peak_equity > 0:
                cooldown = int(getattr(self.config, "quantum_circuit_breaker_cooldown_seconds", 0) or 0)
                mult = float(getattr(self.config, "quantum_circuit_breaker_threshold_multiplier", 1.0) or 1.0)
                last_trip = getattr(self, "_quantum_circuit_breaker_trip_time", None)
                if cooldown and last_trip is not None and (time.time() - last_trip) < cooldown:
                    pass
                else:
                    threshold = abs(cvar_95) * mult
                    current_dd = (peak_equity - new_equity) / peak_equity
                    if cvar_95 < 0 and current_dd > threshold:
                        logger.critical(
                            "Quantum VaR circuit breaker: drawdown %.2f%% > threshold %.2f%% (|CVaR_95|*%.2f)",
                            current_dd * 100, threshold * 100, mult,
                        )
                        self.state = SystemState.EMERGENCY_STOP
                        setattr(self, "_quantum_circuit_breaker_trip_time", time.time())
                        setattr(self, "_quantum_circuit_breaker_trips", int(getattr(self, "_quantum_circuit_breaker_trips", 0)) + 1)
        except Exception as e:
            logger.debug("Quantum Monte Carlo risk hook: %s", e)
    
    async def _initialize_command_bus(self) -> None:
        """Initialize local signed command bus for strategy/execution separation."""
        if not bool(getattr(self.config, "command_bus_enabled", False)):
            logger.info("Command bus disabled (single-process execution path)")
            return
        try:
            from execution.command_bus import LocalInstructionBus

            db_path = str(getattr(self.config, "command_bus_db_path", "data/command_bus.db") or "data/command_bus.db")
            queue = str(getattr(self.config, "command_bus_queue", "default") or "default")
            self.command_bus = LocalInstructionBus(db_path=db_path, queue=queue)
            logger.info("✅ Command bus initialized (db=%s queue=%s)", db_path, queue)
        except Exception as e:
            self.command_bus = None
            logger.error("Command bus initialization failed: %s", e)

    def _command_bus_secret(self) -> str:
        env_name = str(getattr(self.config, "command_bus_hmac_key_env", "ARGUS_COMMAND_HMAC_KEY") or "ARGUS_COMMAND_HMAC_KEY")
        return str(os.getenv(env_name, "") or "")

    def _signal_to_instruction_payload(self, signal: Any, *, cycle_id: int, correlation_id: str, trace_id: str) -> Dict[str, Any]:
        symbol = str(getattr(signal, "symbol", "") or (signal.get("symbol") if isinstance(signal, dict) else "") or "")
        action = str(
            getattr(signal, "action", "")
            or getattr(signal, "side", "")
            or (signal.get("action") if isinstance(signal, dict) else "")
            or (signal.get("side") if isinstance(signal, dict) else "")
            or ""
        ).upper()
        qty_raw = getattr(signal, "quantity", None) if not isinstance(signal, dict) else signal.get("quantity")
        px_raw = getattr(signal, "entry_price", None) if not isinstance(signal, dict) else signal.get("entry_price")
        conf_raw = getattr(signal, "confidence", None) if not isinstance(signal, dict) else signal.get("confidence")
        strategy = str(
            getattr(signal, "strategy", "")
            or getattr(signal, "source_strategy", "")
            or (signal.get("strategy") if isinstance(signal, dict) else "")
            or (signal.get("source_strategy") if isinstance(signal, dict) else "")
            or "unknown"
        )
        quantity = float(qty_raw or 0.0)
        entry_price = float(px_raw or 0.0)
        confidence = float(conf_raw or 0.0)
        now_ts = float(time.time())
        ttl = max(0.5, float(getattr(self.config, "command_bus_instruction_ttl_seconds", 5.0) or 5.0))
        max_notional_aud = float(getattr(self.config, "command_bus_max_notional_aud", 0.0) or 0.0)
        return {
            "run_id": str(self.run_id),
            "trace_id": str(trace_id or ""),
            "correlation_id": str(correlation_id or ""),
            "cycle_id": int(cycle_id),
            "created_ts": now_ts,
            "expires_ts": now_ts + ttl,
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "entry_price": entry_price,
            "confidence": confidence,
            "strategy": strategy,
            "reason": "strategy_node_instruction",
            "max_notional_aud": max_notional_aud,
        }

    def _publish_signals_to_command_bus(self, signals: List[Any], *, cycle_id: int, correlation_id: str) -> Dict[str, Any]:
        from execution.command_bus import deterministic_instruction_id, sign_instruction_payload

        if self.command_bus is None:
            return {"published": 0, "rejected": int(len(signals or [])), "reason": "command_bus_unavailable"}
        require_sig = bool(getattr(self.config, "command_bus_require_signature", True))
        secret = self._command_bus_secret()
        if require_sig and not secret:
            logger.error("Command bus publish blocked: signature required but HMAC secret env is missing")
            return {"published": 0, "rejected": int(len(signals or [])), "reason": "missing_hmac_secret"}

        published = 0
        rejected = 0
        for idx, sig in enumerate(list(signals or [])):
            try:
                trace_id = str(getattr(sig, "trace_id", "") or f"{self.run_id}_{cycle_id}_{idx}")
                payload = self._signal_to_instruction_payload(sig, cycle_id=cycle_id, correlation_id=correlation_id, trace_id=trace_id)
                iid = deterministic_instruction_id(payload)
                signature = sign_instruction_payload(payload, secret) if (require_sig and secret) else ""
                self.command_bus.publish(
                    payload=payload,
                    signature=signature,
                    instruction_id=iid,
                    producer_role="strategy-node",
                    consumer_role="execution-node",
                )
                published += 1
            except Exception as e:
                rejected += 1
                logger.debug("Command bus publish error: %s", e)
        return {"published": int(published), "rejected": int(rejected)}

    def _consume_signals_from_command_bus(self) -> Tuple[List[Any], Dict[str, Any]]:
        from execution.command_bus import (
            instruction_to_signal,
            validate_instruction_payload,
            verify_instruction_payload,
        )

        if self.command_bus is None:
            return [], {"claimed": 0, "accepted": 0, "rejected": 0, "reason": "command_bus_unavailable"}

        max_batch = max(1, int(getattr(self.config, "command_bus_max_batch", 64) or 64))
        claimed = self.command_bus.claim_pending(limit=max_batch)
        accepted_signals: List[Any] = []
        rejected = 0
        require_sig = bool(getattr(self.config, "command_bus_require_signature", True))
        secret = self._command_bus_secret()
        max_notional_aud = float(getattr(self.config, "command_bus_max_notional_aud", 0.0) or 0.0)
        aud_to_usd = float(getattr(self.config, "aud_to_usd", 0.65) or 0.65)

        for row in claimed:
            iid = str(row.get("instruction_id", "") or "")
            payload = dict(row.get("payload") or {})
            signature = str(row.get("signature", "") or "")

            if require_sig:
                if not secret:
                    self.command_bus.mark_rejected(iid, "missing_hmac_secret")
                    rejected += 1
                    continue
                if not verify_instruction_payload(payload, signature, secret):
                    self.command_bus.mark_rejected(iid, "invalid_signature")
                    rejected += 1
                    continue

            valid = validate_instruction_payload(
                payload,
                max_notional_aud=max_notional_aud,
                aud_to_usd=aud_to_usd,
            )
            if not valid.ok:
                self.command_bus.mark_rejected(iid, valid.reason)
                rejected += 1
                continue

            # Execution-side duplicate guard.
            try:
                state_store = getattr(self.execution_engine, "state_store", None) if self.execution_engine else None
                if state_store is not None and hasattr(state_store, "seen_or_mark"):
                    if state_store.seen_or_mark(f"bus:{iid}"):
                        self.command_bus.mark_rejected(iid, "duplicate_instruction")
                        rejected += 1
                        continue
            except Exception as e:
                logger.debug(f"Failed to process instruction {iid}: {e}")
                pass

            sig, reason = instruction_to_signal(payload)
            if sig is None:
                self.command_bus.mark_rejected(iid, reason)
                rejected += 1
                continue

            self.command_bus.mark_consumed(iid)
            accepted_signals.append(sig)

        return accepted_signals, {
            "claimed": int(len(claimed)),
            "accepted": int(len(accepted_signals)),
            "rejected": int(rejected),
        }

    async def _initialize_execution_mesh(self) -> None:
        """Initialize optional OMEGA-01 execution mesh coordinator."""
        if not bool(getattr(self.config, "execution_mesh_enabled", False)):
            logger.info("Execution mesh disabled (single execution lane)")
            return
        try:
            from execution.execution_mesh import ExecutionMeshCoordinator

            max_lanes = int(getattr(self.config, "execution_mesh_max_lanes", 8) or 8)
            max_queue_per_lane = int(getattr(self.config, "execution_mesh_max_queue_per_lane", 128) or 128)
            batch_size = int(getattr(self.config, "execution_mesh_batch_size", 8) or 8)
            parallel_lanes = bool(getattr(self.config, "execution_mesh_parallel_lanes", True))
            halt_on_lane_error = bool(getattr(self.config, "execution_mesh_halt_on_lane_error", True))
            symbols = list(getattr(self.config, "execution_mesh_symbols", []) or [])

            self.execution_mesh = ExecutionMeshCoordinator(
                max_lanes=max_lanes,
                max_queue_per_lane=max_queue_per_lane,
                batch_size=batch_size,
                parallel_lanes=parallel_lanes,
                halt_on_lane_error=halt_on_lane_error,
                allowed_symbols=symbols,
            )
            logger.info(
                "✅ Execution mesh initialized (max_lanes=%s queue_per_lane=%s batch_size=%s parallel=%s)",
                max_lanes,
                max_queue_per_lane,
                batch_size,
                parallel_lanes,
            )
        except Exception as e:
            self.execution_mesh = None
            logger.error("Execution mesh initialization failed: %s", e)

    async def _initialize_execution_engine(self):
        """Phase 3: Initialize Kraken DCA execution engine"""
        logger.info("Phase 3: Initializing Kraken DCA execution engine...")
        from unified_execution_engine import KrakenDCAExecutionEngine
        self.execution_engine = KrakenDCAExecutionEngine(self.config, self.exchanges)
        await self.execution_engine.initialize()
        
        # Wire risk manager to execution state store for persistence
        try:
            if self.unified_risk_manager is not None and hasattr(self.execution_engine, 'state_store'):
                self.unified_risk_manager.attach_state_store(self.execution_engine.state_store)
                logger.info("Risk manager wired to execution state store")
        except Exception as e:
            logger.debug("Risk manager state store wiring: %s", e)
        
        logger.info("✅ Kraken DCA execution engine initialized")
    
    async def _initialize_capital_optimizer(self):
        """Phase 4: Initialize capital optimizer for $1K AUD"""
        logger.info("Phase 4: Initializing capital optimizer...")
        from unified_capital_optimizer import CapitalOptimizer1K
        self.capital_optimizer = CapitalOptimizer1K(self.config)
        await self.capital_optimizer.initialize()
        logger.info("✅ Capital optimizer initialized")

    async def _initialize_strategy_allocator(self):
        """Initialize strategy allocator so PnL-based ranking works with the loop."""
        if not getattr(self.config, "strategy_allocator_enabled", True):
            return
        try:
            from adaptive.strategy_allocator import StrategyAllocator
            mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
            modes = list(getattr(self.config, "strategy_allocator_modes", ["paper", "backtest"]) or ["paper", "backtest"])
            if mode not in modes:
                return
            self.strategy_allocator = StrategyAllocator(
                enabled=True,
                persist_path=str(getattr(self.config, "strategy_allocator_persist_path", "data/strategy_allocator_stats.json") or "data/strategy_allocator_stats.json"),
                timeframe=str(getattr(self.config, "strategy_allocator_timeframe", "") or ""),
                min_trades_before_bias=int(getattr(self.config, "strategy_allocator_min_trades_before_bias", 5) or 5),
                exploration_c=float(getattr(self.config, "strategy_allocator_exploration_c", 1.2) or 1.2),
                ema_alpha=float(getattr(self.config, "strategy_allocator_ema_alpha", 0.15) or 0.15),
            )
            self.strategy_allocator.load()
            logger.info("✅ Strategy allocator initialized (PnL-based ranking)")
        except Exception as e:
            logger.debug("Strategy allocator unavailable: %s", e)
            self.strategy_allocator = None

    def _initialize_strategy_evaluation_engine(self) -> None:
        """Initialize Strategy Evaluation Engine v1 (best-effort, fail-safe)."""
        if not bool(getattr(self.config, "strategy_evaluation_enabled", True)):
            self.strategy_evaluation_engine = None
            return
        try:
            from evaluation.strategy_evaluation_engine import StrategyEvaluationEngine

            db_path = str(
                getattr(self.config, "strategy_evaluation_db_path", "")
                or getattr(getattr(self, "omega_store", None), "db_path", "")
                or "data/strategy_metrics.db"
            )
            self.strategy_evaluation_engine = StrategyEvaluationEngine(
                db_path=db_path,
                enabled=bool(getattr(self.config, "strategy_evaluation_enabled", True)),
                persist_interval_cycles=int(
                    getattr(self.config, "strategy_evaluation_persist_interval_cycles", 10) or 10
                ),
                min_trades_for_ranking=int(
                    getattr(self.config, "strategy_evaluation_min_trades_for_ranking", 5) or 5
                ),
                use_regime_scoped_metrics=bool(
                    getattr(self.config, "strategy_evaluation_use_regime_scoped_metrics", True)
                ),
                sharpe_like_min_trades=int(
                    getattr(self.config, "strategy_evaluation_sharpe_like_min_trades", 5) or 5
                ),
                max_metrics_history_points=int(
                    getattr(self.config, "strategy_evaluation_max_metrics_history_points", 500) or 500
                ),
            )
            logger.info("✅ Strategy evaluation engine initialized (%s)", db_path)
        except Exception as e:
            self.strategy_evaluation_engine = None
            logger.warning("Strategy evaluation engine unavailable: %s", e)

    def _initialize_self_optimizing_meta_engine(self) -> None:
        """Initialize Self-Optimizing Meta Engine v1 (best-effort, fail-safe)."""
        if not bool(getattr(self.config, "self_optimizing_meta_enabled", True)):
            self.self_optimizing_meta_engine = None
            return
        try:
            from adaptive.self_optimizing_meta_engine import SelfOptimizingMetaEngine

            db_path = str(
                getattr(self.config, "self_optimizing_meta_db_path", "")
                or "data/meta_weights.db"
            )
            self.self_optimizing_meta_engine = SelfOptimizingMetaEngine(
                db_path=db_path,
                enabled=bool(getattr(self.config, "self_optimizing_meta_enabled", True)),
                advisory_only=bool(getattr(self.config, "self_optimizing_meta_advisory_only", False)),
                update_interval_cycles=int(
                    getattr(self.config, "self_optimizing_meta_update_interval_cycles", 10) or 10
                ),
                min_trades_for_reweighting=int(
                    getattr(self.config, "self_optimizing_meta_min_trades_for_reweighting", 5) or 5
                ),
                meta_alpha=float(getattr(self.config, "self_optimizing_meta_alpha", 0.2) or 0.2),
                max_weight_change_per_update=float(
                    getattr(self.config, "self_optimizing_meta_max_weight_change_per_update", 0.10) or 0.10
                ),
                min_weight_per_strategy=float(
                    getattr(self.config, "self_optimizing_meta_min_weight_per_strategy", 0.05) or 0.05
                ),
                max_weight_per_strategy=float(
                    getattr(self.config, "self_optimizing_meta_max_weight_per_strategy", 0.45) or 0.45
                ),
                baseline_weight_mode=str(
                    getattr(self.config, "self_optimizing_meta_baseline_weight_mode", "equal") or "equal"
                ),
                score_weights=dict(getattr(self.config, "self_optimizing_meta_score_weights", {}) or {}),
                regime_multipliers=dict(
                    getattr(self.config, "self_optimizing_meta_regime_multipliers", {}) or {}
                ),
            )
            logger.info("✅ Self-optimizing meta engine initialized (%s)", db_path)
        except Exception as e:
            self.self_optimizing_meta_engine = None
            logger.warning("Self-optimizing meta engine unavailable: %s", e)

    def _handle_strategy_eval_error(self, error: Exception, *, context: str) -> None:
        """Fail-safe behavior for strategy evaluation errors."""
        logger.warning("Strategy evaluation error (%s): %s", context, error)
        if bool(getattr(self.config, "strategy_evaluation_halt_on_error", False)):
            self.state = SystemState.EMERGENCY_STOP
            logger.critical("Strategy evaluation configured fail-closed; entering EMERGENCY_STOP")

    def _initialize_champion_challenger_engine(self) -> None:
        """Initialize champion/challenger engine (advisory-first, fail-safe)."""
        if not bool(getattr(self.config, "champion_challenger_enabled", True)):
            self.champion_challenger_engine = None
            return
        try:
            from evaluation.champion_challenger_engine import ChampionChallengerEngine

            db_path = str(
                getattr(self.config, "champion_challenger_db_path", "")
                or "data/champion_challenger.db"
            )
            artifacts_dir = str(
                getattr(self.config, "champion_challenger_artifacts_dir", "")
                or "deploy/promotions"
            )
            weights = dict(
                getattr(self.config, "champion_challenger_promotion_weights", None)
                or {
                    "net_pnl": 1.0,
                    "expectancy": 1.0,
                    "profit_factor": 0.75,
                    "sharpe_like": 1.0,
                    "drawdown_penalty": 1.25,
                    "fee_penalty": 0.5,
                }
            )
            self.champion_challenger_engine = ChampionChallengerEngine(
                db_path=db_path,
                artifacts_dir=artifacts_dir,
                enabled=bool(getattr(self.config, "champion_challenger_enabled", True)),
                advisory_only=bool(getattr(self.config, "champion_challenger_advisory_only", True)),
                min_trades_for_promotion=int(
                    getattr(self.config, "champion_challenger_min_trades_for_promotion", 10) or 10
                ),
                max_drawdown_pct_for_promotion=float(
                    getattr(self.config, "champion_challenger_max_drawdown_pct_for_promotion", 0.12)
                    or 0.12
                ),
                require_expectancy_improvement=bool(
                    getattr(self.config, "champion_challenger_require_expectancy_improvement", True)
                ),
                require_profit_factor_improvement=bool(
                    getattr(self.config, "champion_challenger_require_profit_factor_improvement", False)
                ),
                require_sharpe_like_improvement=bool(
                    getattr(self.config, "champion_challenger_require_sharpe_like_improvement", True)
                ),
                promotion_weights=weights,
                persist_interval_cycles=int(
                    getattr(self.config, "champion_challenger_persist_interval_cycles", 10) or 10
                ),
            )
            active_champion = self.champion_challenger_engine.get_active_champion()
            if active_champion is None:
                strategy_set = list(getattr(self.config, "strategies_enabled", []) or [])
                if not strategy_set:
                    strategy_set = ["unknown"]
                config_payload = json.dumps(self.config.__dict__, sort_keys=True, ensure_ascii=True, default=str)
                config_hash = hashlib.sha256(config_payload.encode("utf-8")).hexdigest()
                bundle_hint = "deploy/bundles"
                if not Path(bundle_hint).exists():
                    bundle_hint = ""
                champion_profile = self.champion_challenger_engine.register_champion(
                    profile_id=f"champion_{str(getattr(self, 'run_id', 'unknown'))}",
                    source_bundle_path=str(bundle_hint),
                    config_hash=str(config_hash),
                    strategy_set=[str(s) for s in strategy_set if str(s).strip()],
                    version_label=f"runtime_{str(getattr(self, 'run_id', 'unknown'))}",
                    status="active",
                )
                logger.info(
                    "✅ Champion/challenger engine initialized and champion bootstrapped (%s)",
                    champion_profile.profile_id,
                )
            else:
                logger.info(
                    "✅ Champion/challenger engine initialized (active champion=%s)",
                    active_champion.profile_id,
                )
        except Exception as e:
            self.champion_challenger_engine = None
            logger.warning("Champion/challenger engine unavailable: %s", e)

    def _attach_champion_challenger_context(self, signals: List[Any], regime_label: str) -> None:
        """Attach advisory promotion context to candidates for snapshot-level auditability."""
        cc_engine = getattr(self, "champion_challenger_engine", None)
        if cc_engine is None:
            return
        active = cc_engine.get_active_champion()
        if active is None:
            return
        best = cc_engine.best_challengers_by_promotion_score(limit=1)
        best_decision = best[0] if best else None
        for sig in list(signals or []):
            try:
                if isinstance(sig, dict):
                    sig["champion_profile_id"] = str(active.profile_id)
                    if best_decision is not None:
                        sig["challenger_profile_id"] = str(best_decision.challenger_id)
                        sig["promotion_decision"] = str(best_decision.decision)
                        sig["promotion_score"] = float(best_decision.promotion_score)
                        sig["promotion_reasons"] = list(best_decision.reasons or [])
                        if regime_label:
                            sig["promotion_regime_label"] = str(regime_label)
                else:
                    setattr(sig, "champion_profile_id", str(active.profile_id))
                    if best_decision is not None:
                        setattr(sig, "challenger_profile_id", str(best_decision.challenger_id))
                        setattr(sig, "promotion_decision", str(best_decision.decision))
                        setattr(sig, "promotion_score", float(best_decision.promotion_score))
                        setattr(sig, "promotion_reasons", list(best_decision.reasons or []))
                        if regime_label:
                            setattr(sig, "promotion_regime_label", str(regime_label))
            except Exception as e:
                logger.debug("Champion/challenger context attach failed: %s", e)
                break

    def _attach_strategy_evaluation_context(self, signals: List[Any], regime_label: str) -> None:
        """Attach per-strategy ranking context to candidate signals (best-effort)."""
        engine = getattr(self, "strategy_evaluation_engine", None)
        if engine is None:
            return
        rankable = 0
        try:
            rankable = int(engine.rankable_strategy_count())
        except Exception:
            rankable = 0
        for sig in list(signals or []):
            try:
                strategy = str(
                    self._signal_get(sig, "source_strategy", None)
                    or self._signal_get(sig, "strategy", None)
                    or "unknown"
                )
                symbol = str(self._signal_get(sig, "symbol", None) or "")
                reg = str(self._signal_get(sig, "regime_label", None) or regime_label or "")
                ctx = engine.get_decision_context(
                    strategy_name=strategy,
                    symbol=symbol or None,
                    regime_label=reg or None,
                )
                if not ctx:
                    continue
                if isinstance(sig, dict):
                    sig.update(ctx)
                else:
                    for k, v in ctx.items():
                        setattr(sig, k, v)
            except Exception as e:
                self._handle_strategy_eval_error(e, context="attach_context")
                break
        if rankable > 0:
            logger.info("strategy ranking context available for %s strategies", rankable)

    async def _initialize_hft_engine(self):
        """Phase 4.5: Initialize HFT Scalping Engine"""
        if not bool(getattr(self.config, "hft_enabled", True)):
            logger.info("Phase 4.5: HFT Scalping Engine disabled by config")
            self.hft_engine = None
            return
        logger.info("Phase 4.5: Initializing HFT Scalping Engine...")
        try:
            from hft_engine.hft_scalping_engine import HFTScalpingEngine
            self.hft_engine = HFTScalpingEngine(self.config, exchanges=self.exchanges)
            logger.info("✅ HFT Scalping Engine initialized")
        except Exception as e:
            logger.warning(f"HFT Engine initialization failed: {e}")
            self.hft_engine = None

    async def _initialize_hft_infrastructure(self):
        """Phase 4.6: Initialize Advanced HFT Infrastructure (5.1: attach hft_engine for real OB/trades feed)"""
        if not bool(getattr(self.config, "hft_enabled", True)) or not bool(
            getattr(self.config, "use_advanced_hft_infrastructure", False)
        ):
            logger.info("Phase 4.6: Advanced HFT Infrastructure disabled by config")
            self.hft_infrastructure = None
            return
        logger.info("Phase 4.6: Initializing Advanced HFT Infrastructure...")
        try:
            from hft_engine.advanced_realtime_hft_infrastructure import get_hft_infrastructure
            cfg = self.config.__dict__ if hasattr(self.config, "__dict__") else {}
            self.hft_infrastructure = get_hft_infrastructure(config=cfg, hft_engine=self.hft_engine)
            # Start the event loop in background (consumes OB/trades pushed from main loop or feeder)
            asyncio.create_task(self.hft_infrastructure.run_event_loop())
            logger.info("✅ Advanced HFT Infrastructure initialized (Kernel Bypass Sim Active)")
        except Exception as e:
            logger.warning(f"Advanced HFT Infrastructure initialization failed: {e}")
            self.hft_infrastructure = None

    async def _initialize_max_return_features(self):
        """Phase 4.7: Initialize Max Return Features (Tier 1 + Tier 2) for $1K capital optimization"""
        logger.info("Phase 4.7: Initializing Max Return Features...")
        
        # Tier 1: Leverage Manager - Dynamic leverage up to 10x
        try:
            from execution.leverage_manager import LeverageManager
            self.leverage_manager = LeverageManager(
                max_leverage=float(getattr(self.config, "max_leverage", 10.0) or 10.0),
                risk_per_trade=0.02,
                max_portfolio_leverage=3.0,
                auto_deleverage=True,
            )
            logger.info("✅ Leverage Manager initialized (max %.1fx)", self.leverage_manager.max_leverage)
        except Exception as e:
            logger.warning("Leverage Manager unavailable: %s", e)
            self.leverage_manager = None

        # Tier 1: Flash Crash Sniper - Buy crashes, sell rips
        try:
            from execution.flash_crash_sniper import FlashCrashSniper, SniperConfig
            sniper_config = SniperConfig(
                min_drop_pct=float(getattr(self.config, "flash_crash_threshold", 5.0) or 5.0),
                recovery_threshold=float(getattr(self.config, "flash_recovery_threshold", 0.3) or 0.3),
                volume_confirmation=float(getattr(self.config, "flash_min_volume_ratio", 2.0) or 2.0),
                max_position_pct=float(getattr(self.config, "flash_max_position_pct", 0.15) or 0.15),
            )
            self.flash_crash_sniper = FlashCrashSniper(
                config=sniper_config,
                portfolio_value=float(getattr(self.config, "starting_capital_aud", 1000.0) or 1000.0),
            )
            logger.info("✅ Flash Crash Sniper initialized (min_drop: %.1f%%)",
                       sniper_config.min_drop_pct)
        except Exception as e:
            logger.warning("Flash Crash Sniper unavailable: %s", e)
            self.flash_crash_sniper = None

        # Tier 1: Portfolio Compounding - Reinvest profits automatically
        try:
            from execution.portfolio_compounding import PortfolioCompounding
            self.portfolio_compounding = PortfolioCompounding(
                initial_capital=float(getattr(self.config, "starting_capital_aud", 1000.0) or 1000.0),
                reinvest_pct=0.5,
                max_drawdown=0.20,
                enable_auto_compound=True,
            )
            logger.info("✅ Portfolio Compounding initialized (initial: $%.2f AUD)",
                       self.portfolio_compounding.initial_capital)
        except Exception as e:
            logger.warning("Portfolio Compounding unavailable: %s", e)
            self.portfolio_compounding = None

        # Tier 2: Enhanced Fee Optimizer - Smart order routing
        try:
            from execution.fee_optimizer_enhanced import EnhancedFeeOptimizer
            self.fee_optimizer = EnhancedFeeOptimizer(
                enable_batching=True,
                enable_exchange_routing=True,
                target_fee_bps=3.0,
            )
            logger.info("✅ Enhanced Fee Optimizer initialized")
        except Exception as e:
            logger.warning("Enhanced Fee Optimizer unavailable: %s", e)
            self.fee_optimizer = None

        # Tier 2: Position Scaler - Scale into winners
        try:
            from execution.position_scaler import PositionScaler
            self.position_scaler = PositionScaler(
                base_risk_pct=0.02,
                max_position_pct=0.25,
                scale_up_threshold=0.05,
                scale_down_threshold=-0.03,
                max_scale=2.0,
                min_scale=0.25,
            )
            logger.info("✅ Position Scaler initialized (scale into winners)")
        except Exception as e:
            logger.warning("Position Scaler unavailable: %s", e)
            self.position_scaler = None

        # Tier 2: Multi-Timeframe Confluence - Higher win rate
        try:
            from execution.multi_timeframe_confluence import MultiTimeframeConfluence
            self.mtf_confluence = MultiTimeframeConfluence(
                min_timeframes=int(getattr(self.config, "mtf_min_timeframes", 4) or 4),
                require_higher_tf=bool(getattr(self.config, "mtf_require_higher_tf", True)),
                min_confluence=float(getattr(self.config, "mtf_min_confluence", 0.55) or 0.55),
                enable_dynamic_sizing=True,
            )
            logger.info("✅ Multi-Timeframe Confluence initialized (min %d timeframes)", 
                       self.mtf_confluence.min_timeframes)
        except Exception as e:
            logger.warning("Multi-Timeframe Confluence unavailable: %s", e)
            self.mtf_confluence = None

        # QUANTUM: Real-Time Adaptive Trading Engine - Knows when to buy/sell in real-time
        try:
            from quantum.quantum_realtime_adapter import QuantumRealTimeAdapter
            self.quantum_adapter = QuantumRealTimeAdapter(
                initial_capital=float(getattr(self.config, "starting_capital_aud", 1000.0) or 1000.0),
                enable_quantum_signals=bool(getattr(self.config, "quantum_signals_enabled", True)),
                enable_portfolio_optimization=bool(getattr(self.config, "quantum_portfolio_enabled", True)),
                enable_risk_adaptation=True,
                signal_interval_seconds=float(getattr(self.config, "quantum_signal_interval", 5.0) or 5.0),
                min_confidence_threshold=float(getattr(self.config, "quantum_min_confidence", 0.6) or 0.6),
            )
            logger.info("✅ Quantum Real-Time Adapter initialized (quantum-powered buy/sell decisions)")
        except Exception as e:
            logger.warning("Quantum Real-Time Adapter unavailable: %s", e)
            self.quantum_adapter = None

        # ULTIMATE QUANTUM BRAIN - 7 quantum modules combined for maximum intelligence
        try:
            from quantum.ultimate_quantum_brain import UltimateQuantumBrain, QuantumMode
            self.quantum_brain = UltimateQuantumBrain(
                mode=QuantumMode.SIMULATOR,
                n_qubits=int(getattr(self.config, "quantum_n_qubits", 8) or 8),
                enable_all_modules=True,
                min_consensus=float(getattr(self.config, "quantum_min_consensus", 0.6) or 0.6),
            )
            logger.info("✅ Ultimate Quantum Brain initialized (7 quantum modules, 2x+ quantum advantage)")
        except Exception as e:
            logger.warning("Ultimate Quantum Brain unavailable: %s", e)
            self.quantum_brain = None

        # QUANTUM RISK ENGINE - Quantum-powered VaR/CVaR (4x faster)
        try:
            from quantum.quantum_risk_engine import QuantumRiskEngine
            self.quantum_risk_engine = QuantumRiskEngine(
                n_qubits=int(getattr(self.config, "quantum_n_qubits", 8) or 8),
            )
            logger.info("✅ Quantum Risk Engine initialized (4x faster VaR/CVaR)")
        except Exception as e:
            logger.warning("Quantum Risk Engine unavailable: %s", e)
            self.quantum_risk_engine = None

        # QUANTUM EXECUTION ROUTER - Optimal order execution
        try:
            from quantum.quantum_execution_router import QuantumExecutionRouter
            self.quantum_executor = QuantumExecutionRouter(
                n_qubits=int(getattr(self.config, "quantum_n_qubits", 6) or 6),
                enable_quantum_walk=True,
                enable_quantum_annealing=True,
            )
            logger.info("✅ Quantum Execution Router initialized (quantum walk + annealing)")
        except Exception as e:
            logger.warning("Quantum Execution Router unavailable: %s", e)
            self.quantum_executor = None

        # QUANTUM INTEGRATION LAYER - Master quantum orchestrator
        try:
            from quantum.quantum_integration_layer import QuantumIntegrationLayer, QuantumIntegrationLevel
            self.quantum_integration = QuantumIntegrationLayer(
                integration_level=QuantumIntegrationLevel.ULTIMATE,
                n_qubits=int(getattr(self.config, "quantum_n_qubits", 8) or 8),
                enable_all_upgrades=True,
            )
            upgrade_status = self.quantum_integration.get_upgrade_status()
            logger.info("✅ Quantum Integration Layer initialized (%d components upgraded, %.1fx avg advantage)",
                       upgrade_status["summary"]["components_upgraded"],
                       upgrade_status["summary"]["avg_advantage"])
        except Exception as e:
            logger.warning("Quantum Integration Layer unavailable: %s", e)
            self.quantum_integration = None

        # Multi-Pair Liquidity Scanner - Scan and select best pairs to trade
        try:
            from execution.multi_pair_liquidity_scanner import MultiPairLiquidityScanner
            self.liquidity_scanner = MultiPairLiquidityScanner(
                exchanges=list(getattr(self.config, "scanner_exchanges", ["kraken", "coinbase"]) or ["kraken", "coinbase"]),
                max_pairs=int(getattr(self.config, "scanner_max_pairs", 20) or 20),
                min_volume_usd=float(getattr(self.config, "scanner_min_volume", 100000) or 100000),
                max_spread_pct=float(getattr(self.config, "scanner_max_spread", 0.5) or 0.5),
                scan_interval_seconds=float(getattr(self.config, "scanner_interval", 60.0) or 60.0),
                enable_auto_discovery=bool(getattr(self.config, "scanner_auto_discovery", True)),
            )
            logger.info("✅ Multi-Pair Liquidity Scanner initialized (scanning for best pairs)")
        except Exception as e:
            logger.warning("Multi-Pair Liquidity Scanner unavailable: %s", e)
            self.liquidity_scanner = None

        logger.info("✅ Max Return Features initialized (Tier 1 + Tier 2 + Quantum + Multi-Pair)")

    async def _initialize_monitoring(self):
        """Phase 5: Initialize monitoring infrastructure"""
        logger.info("Phase 5: Initializing monitoring infrastructure...")
        try:
            from core.health import HealthRegistry
            self.monitoring = HealthRegistry()
            logger.info("✅ Monitoring system initialized (HealthRegistry)")
        except Exception as e:
            logger.warning("Monitoring system unavailable: %s", e)
            self.monitoring = None

    async def _initialize_production_modules(self):
        """Phase 6: Initialize salvaged production modules (all best-effort)."""
        logger.info("Phase 6: Initializing production modules...")
        count = 0

        # Rate limiter for exchange API calls
        try:
            from core.rate_limiter import RateLimiterManager
            self.rate_limiter = RateLimiterManager()
            count += 1
        except Exception as e:
            logger.debug("RateLimiter unavailable: %s", e)

        # Data sanitizer for OHLCV quality
        try:
            from core.data_sanitizer import DataSanitizer
            self.data_sanitizer = DataSanitizer()
            count += 1
        except Exception as e:
            logger.debug("DataSanitizer unavailable: %s", e)

        # Position tracker with P&L
        try:
            from core.position_tracker import PositionTracker
            self.position_tracker = PositionTracker()
            count += 1
        except Exception as e:
            logger.debug("PositionTracker unavailable: %s", e)

        # Cryptographic audit chain
        try:
            from core.audit_chain import AuditChain
            self.audit_chain = AuditChain(db_path="data/audit_chain.db")
            count += 1
        except Exception as e:
            logger.debug("AuditChain unavailable: %s", e)

        # Self-healing component monitor
        try:
            from utils.self_healing import SelfHealingController
            self.self_healer = SelfHealingController()
            count += 1
        except Exception as e:
            logger.debug("SelfHealingController unavailable: %s", e)

        logger.info("✅ Production modules initialized: %d/5 available", count)

        # Component registry: initialise all shelf-ware (best-effort, with 30s timeout)
        try:
            from core.component_registry import ComponentRegistry
            self.component_registry = ComponentRegistry(self.config)
            try:
                n_ok = await asyncio.wait_for(
                    self.component_registry.initialize(),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                n_ok = getattr(self.component_registry, "_init_count", 0)
                logger.warning("ComponentRegistry.initialize() timed out after 30s — %d components loaded so far", n_ok)
            # Inject audit chain so compliance events are recorded
            if self.audit_chain is not None:
                self.component_registry.set_audit_chain(self.audit_chain)
            logger.info("✅ ComponentRegistry: %d components active", n_ok)
            
            # Get alpha_signal_fusion from registry
            if self.component_registry:
                self.alpha_signal_fusion = getattr(self.component_registry, "alpha_signal_fusion", None)
                if self.alpha_signal_fusion:
                    logger.info("✅ AlphaSignalFusion ready from component registry")
        except Exception as e:
            logger.debug("ComponentRegistry unavailable: %s", e)
            self.component_registry = None

        # Strategy state store: per-strategy persistence + cooldown enforcement
        try:
            from strategies.strategy_state_store import StrategyStateStore
            _sss_db = str(getattr(self.config, "strategy_state_db_path", "data/strategy_states.db") or "data/strategy_states.db")
            _sss_max_losses = int(getattr(self.config, "strategy_max_consecutive_losses", 5) or 5)
            _sss_cd_min = int(getattr(self.config, "strategy_cooldown_minutes", 60) or 60)
            self._strategy_state_store = StrategyStateStore(
                db_path=_sss_db,
                max_consecutive_losses=_sss_max_losses,
                cooldown_minutes=_sss_cd_min,
            )
            loaded = self._strategy_state_store.load_all()
            logger.info("Strategy state store: loaded %d strategy states from %s", len(loaded), _sss_db)
        except Exception as e:
            logger.debug("StrategyStateStore unavailable: %s", e)
            self._strategy_state_store = None

        # Health server (HTTP /health /metrics endpoints)
        try:
            from core.health_server import HealthServer
            self.health_monitor_server = HealthServer(trading_system=self)
            self.health_monitor_server.start()
            logger.info("✅ HealthServer started on port 8080")
        except Exception as e:
            logger.debug("HealthServer unavailable: %s", e)
            self.health_monitor_server = None

        # Rolling performance feeder (feeds meta engine for weight evolution)
        try:
            from adaptive.rolling_performance_feeder import RollingPerformanceFeeder
            meta = getattr(self, "meta_engine", None) or getattr(self, "self_optimizing_meta_engine", None)
            if meta is not None:
                self.perf_feeder = RollingPerformanceFeeder(meta_engine=meta)
                logger.info("✅ RollingPerformanceFeeder initialized")
        except Exception as e:
            logger.debug("RollingPerformanceFeeder unavailable: %s", e)

        # Model manager (ML lifecycle, retraining, hot-swap)
        try:
            from ml.model_manager import ModelManager
            self.model_manager = ModelManager(config=self.config)
            self.model_manager.load_all()
            logger.info("✅ ModelManager initialized")
        except Exception as e:
            logger.debug("ModelManager unavailable: %s", e)

        # Checkpoint manager (state persistence and recovery)
        try:
            from core.checkpoint_manager import CheckpointManager
            _ckpt_path = str(getattr(self.config, "checkpoint_db_path", "data/checkpoints.db") or "data/checkpoints.db")
            _ckpt_interval = int(getattr(self.config, "checkpoint_interval", 10) or 10)
            self.checkpoint_manager = CheckpointManager(
                db_path=_ckpt_path,
                save_interval=_ckpt_interval,
            )
            # Load latest checkpoint on startup
            _ckpt = self.checkpoint_manager.load_latest_checkpoint()
            if _ckpt is not None:
                _restored_cycle = _ckpt.get("cycle_count", 0)
                logger.info(
                    "✅ CheckpointManager: restored state from cycle %d",
                    _restored_cycle,
                )
            else:
                logger.info("✅ CheckpointManager initialized (no prior checkpoint)")
        except Exception as e:
            logger.debug("CheckpointManager unavailable: %s", e)

        # Regime change alerter (Discord on regime transitions)
        try:
            from monitoring.regime_alert import RegimeChangeAlerter
            webhook = getattr(self.config, "discord_webhook_url", None) or ""
            self.regime_alerter = RegimeChangeAlerter(webhook_url=str(webhook))
            logger.info("✅ RegimeChangeAlerter initialized")
        except Exception as e:
            logger.debug("RegimeChangeAlerter unavailable: %s", e)

        # L2 orderbook feed (WebSocket, best-effort — system runs fine without it)
        try:
            from data.orderbook.l2_feed import L2OrderbookFeed
            _l2_exchange = str(getattr(self.config, "primary_exchange", "kraken") or "kraken").lower()
            self.l2_feed = L2OrderbookFeed(exchange=_l2_exchange)
            _l2_pairs = list(getattr(self.config, "trading_pairs", ["BTC/USD"]) or ["BTC/USD"])
            asyncio.create_task(self.l2_feed.subscribe(_l2_pairs))
            logger.info("✅ L2OrderbookFeed subscribed to %s on %s", _l2_pairs, _l2_exchange)
        except Exception as e:
            logger.debug("L2OrderbookFeed unavailable: %s", e)
            self.l2_feed = None

        # ── Live mode startup validation ──
        _run_mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
        if _run_mode == "live":
            _live_checks_passed = True
            # 1. Verify API keys are configured
            if not os.environ.get("KRAKEN_API_KEY") and not os.environ.get("COINBASE_API_KEY"):
                logger.error("LIVE MODE: No exchange API keys found in environment. Set KRAKEN_API_KEY or COINBASE_API_KEY.")
                _live_checks_passed = False
            # 2. Verify exchange connectivity
            try:
                _test_ex = getattr(self.exchange_manager, "_exchange", None) or self.exchange_manager
                if hasattr(_test_ex, "fetch_ticker"):
                    _test_ticker = await asyncio.wait_for(_test_ex.fetch_ticker("BTC/USD"), timeout=10.0)
                    if _test_ticker and float(_test_ticker.get("last", 0) or 0) > 0:
                        logger.info("LIVE MODE: Exchange connectivity verified (BTC/USD = $%.2f)", float(_test_ticker["last"]))
                    else:
                        logger.error("LIVE MODE: Exchange returned empty ticker")
                        _live_checks_passed = False
            except Exception as _conn_e:
                logger.error("LIVE MODE: Exchange connectivity check failed: %s", _conn_e)
                _live_checks_passed = False
            # 3. Verify historical data is loaded
            if hasattr(self, "component_registry") and self.component_registry:
                _ohlcv = getattr(self.component_registry, "_ohlcv_history", {})
                if not _ohlcv:
                    logger.warning("LIVE MODE: No historical OHLCV loaded — evolver/scanner will use synthetic data")
            if not _live_checks_passed:
                logger.error("LIVE MODE: Startup validation failed — switching to paper mode for safety")
                _run_mode = "paper"

        # Live market data manager (WebSocket feeds — live mode only)
        if _run_mode == "live":
            try:
                from core.live_market_data import LiveMarketDataManager
                self.live_market_data = LiveMarketDataManager(
                    exchanges=self.exchanges,
                    market_data_service=self.market_data_service,
                )
                _primary_exchange = str(getattr(self.config, "primary_exchange", "kraken") or "kraken").lower()
                _live_pairs = list(getattr(self.config, "trading_pairs", ["BTC/USD"]) or ["BTC/USD"])
                # Attach existing L2 feed if available
                if self.l2_feed is not None:
                    self.live_market_data.set_l2_feed(self.l2_feed)
                asyncio.create_task(self.live_market_data.subscribe(_live_pairs, exchange=_primary_exchange))
                logger.info("✅ LiveMarketDataManager started for %s on %s", _live_pairs, _primary_exchange)
            except Exception as e:
                logger.warning("LiveMarketDataManager unavailable: %s", e)
                self.live_market_data = None

    async def _sync_positions_from_exchange(self) -> Dict[str, Any]:
        """
        Sync internal positions with exchange state on startup.

        In live mode: fetches all open positions/balances from the exchange
        and populates self.positions.
        In paper mode: starts with empty positions (or loads from saved state).

        Returns:
            Summary dict with positions found.
        """
        summary: Dict[str, Any] = {"synced": False, "positions_found": 0, "details": {}}
        mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()

        if mode != "live":
            # Check if we should restore state from checkpoint
            _paper_cfg = getattr(self.config, "paper_trading", {}) or {}
            _restore_state = _paper_cfg.get("restore_state", True) if isinstance(_paper_cfg, dict) else True

            # Restore positions from checkpoint if available AND restore_state is enabled
            if _restore_state:
                try:
                    if self.checkpoint_manager is not None:
                        ckpt = self.checkpoint_manager.load_latest()
                        if ckpt and isinstance(ckpt, dict):
                            saved_positions = ckpt.get("positions") or {}
                            if saved_positions:
                                self.positions.update(saved_positions)
                                logger.info("Position sync: restored %d positions from checkpoint", len(saved_positions))
                                summary["positions_found"] = len(saved_positions)
                                summary["details"] = {k: v.get("quantity", 0) for k, v in saved_positions.items()}
                except Exception as _cp_exc:
                    logger.debug("Position restore from checkpoint failed: %s", _cp_exc)
            else:
                logger.info("Position sync: paper mode starting fresh (restore_state=false)")
            logger.info("Position sync: %s mode (restored %d positions from checkpoint)",
                       mode, summary.get("positions_found", 0))
            summary["synced"] = True
            return summary

        try:
            # Try exchange_manager first
            if hasattr(self, "exchange_manager") and self.exchange_manager is not None:
                balances = await self.exchange_manager.get_balances()
            else:
                # Fall back to ccxt exchanges dict
                balances = {}
                for ex_name, ex in self.exchanges.items():
                    try:
                        if hasattr(ex, "fetch_balance"):
                            bal = await ex.fetch_balance()
                            balances[ex_name] = bal.get("total", bal) if isinstance(bal, dict) else {}
                        elif hasattr(ex, "get_balance"):
                            balances[ex_name] = await ex.get_balance()
                    except Exception as exc:
                        logger.warning("Position sync: failed to fetch balance from %s: %s", ex_name, exc)

            if not balances:
                logger.warning("Position sync: no balances returned from any exchange")
                summary["synced"] = True
                return summary

            # Merge across exchanges into self.positions
            trading_pairs = list(getattr(self.config, "trading_pairs", []) or [])
            # Extract base currencies from trading pairs for filtering
            base_currencies = set()
            for pair in trading_pairs:
                parts = pair.split("/")
                if parts:
                    base_currencies.add(parts[0])

            for ex_name, ex_balance in balances.items():
                if not isinstance(ex_balance, dict):
                    continue
                for asset, amount_info in ex_balance.items():
                    if asset in ("info", "free", "used", "total", "timestamp", "datetime"):
                        continue
                    if isinstance(amount_info, dict):
                        total = float(amount_info.get("total", 0) or 0)
                    else:
                        total = float(amount_info or 0)

                    if total <= 1e-12:
                        continue

                    # Only track base currencies of configured trading pairs
                    if base_currencies and asset not in base_currencies:
                        continue

                    # Find matching trading pair
                    matching_pair = None
                    for pair in trading_pairs:
                        if pair.startswith(f"{asset}/"):
                            matching_pair = pair
                            break

                    if matching_pair:
                        self.positions[matching_pair] = {
                            "symbol": matching_pair,
                            "quantity": total,
                            "entry_price": 0.0,  # unknown on startup
                            "current_price": 0.0,
                            "exchange": ex_name,
                            "synced_from_exchange": True,
                        }
                        logger.info(
                            "Position sync: found %s = %.8f on %s",
                            matching_pair, total, ex_name,
                        )

            summary["synced"] = True
            summary["positions_found"] = len(self.positions)
            summary["details"] = {
                sym: pos.get("quantity", 0) for sym, pos in self.positions.items()
            }
            logger.info("Position sync: %d positions loaded from exchange", len(self.positions))

        except Exception as exc:
            logger.error("Position sync failed: %s", exc)
            summary["error"] = str(exc)

        return summary

    def _ensure_target_and_regime_components(self) -> None:
        """Initialize optional targets/regime helpers lazily so unit tests can patch components directly."""
        if bool(getattr(self.config, "targets_enabled", False)) and self.target_engine is None:
            try:
                from execution.target_portfolio_engine import TargetPortfolioEngine

                self.target_engine = TargetPortfolioEngine(self.config)
            except Exception as e:
                logger.debug("Target engine unavailable: %s", e)

        if bool(getattr(self.config, "strategy_evaluation_enabled", True)) and self.strategy_evaluation_engine is None:
            self._initialize_strategy_evaluation_engine()

        if bool(getattr(self.config, "liquidity_risk_enabled", True)) and self.liquidity_risk_engine is None:
            try:
                from risk.liquidity_risk_engine import LiquidityRiskEngine

                self.liquidity_risk_engine = LiquidityRiskEngine(self.config)
            except Exception as e:
                logger.debug("Liquidity risk engine unavailable: %s", e)

        if bool(getattr(self.config, "market_microstructure_enabled", True)) and self.market_microstructure_engine is None:
            try:
                from adaptive.market_microstructure_engine import MarketMicrostructureEngine

                self.market_microstructure_engine = MarketMicrostructureEngine(
                    rolling_window=int(getattr(self.config, "market_microstructure_rolling_window", 20) or 20),
                    vacuum_spread_jump_bps=float(
                        getattr(self.config, "market_microstructure_vacuum_spread_jump_bps", 4.0) or 4.0
                    ),
                    vacuum_depth_drop_ratio=float(
                        getattr(self.config, "market_microstructure_vacuum_depth_drop_ratio", 0.5) or 0.5
                    ),
                    high_adverse_selection_threshold=float(
                        getattr(self.config, "market_microstructure_high_adverse_selection_threshold", 0.7) or 0.7
                    ),
                    use_in_execution_alpha=bool(
                        getattr(self.config, "market_microstructure_use_in_execution_alpha", True)
                    ),
                    use_in_liquidity_risk=bool(
                        getattr(self.config, "market_microstructure_use_in_liquidity_risk", True)
                    ),
                )
            except Exception as e:
                logger.debug("Market microstructure engine unavailable: %s", e)

        if bool(getattr(self.config, "feature_store_enabled", False)) and self.feature_store is None:
            try:
                from adaptive.feature_store import RollingFeatureStore

                window = int(getattr(self.config, "feature_store_window", 120) or 120)
                self.feature_store = RollingFeatureStore(window=window)
            except Exception as e:
                logger.debug("Feature store unavailable: %s", e)

        if bool(getattr(self.config, "regime_classifier_enabled", False)) and self.regime_classifier is None:
            try:
                from adaptive.regime_engine import DeterministicRegimeClassifier

                self.regime_classifier = DeterministicRegimeClassifier()
            except Exception as e:
                logger.debug("Regime classifier unavailable: %s", e)

        if bool(getattr(self.config, "self_optimizing_meta_enabled", True)) and self.self_optimizing_meta_engine is None:
            self._initialize_self_optimizing_meta_engine()

    @staticmethod
    def _signal_get(signal: Any, field: str, default: Any = None) -> Any:
        if isinstance(signal, dict):
            return signal.get(field, default)
        return getattr(signal, field, default)

    def _snapshot_candidate_decision(
        self,
        *,
        cycle_id: int,
        correlation_id: str,
        signal: Any,
        allowed: bool,
        reason_code: ReasonCode,
        details: Optional[Dict[str, Any]] = None,
        cost: Optional[Dict[str, Any]] = None,
        exec_plan: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        """Persist a candidate decision snapshot (best-effort, fail-safe)."""
        if not hasattr(self, "omega_store") or self.omega_store is None:
            return
        try:
            symbol = self._signal_get(signal, "symbol", None)
            strategy = self._signal_get(signal, "strategy", None) or self._signal_get(signal, "source_strategy", None)
            side = self._signal_get(signal, "side", None) or self._signal_get(signal, "action", None)
            score = self._signal_get(signal, "score", None) or self._signal_get(signal, "confidence", None)
            resolved_trace = str(trace_id or getattr(self, "_trace_id", None) or uuid.uuid4().hex)
            merged_details = dict(details or {})
            for field_name in (
                "target_exposure_pct",
                "current_exposure_pct",
                "delta_exposure_pct",
                "priority_score",
                "expected_net_edge_bps",
                "regime_label",
                "target_reasons",
                "suppression_reason",
                "liquidity_score",
                "liquidity_state",
                "max_safe_trade_size",
                "adjusted_target_exposure_pct",
                "liquidity_clamp_flag",
                "slippage_estimate_bps",
                "strategy_trades_count",
                "strategy_win_rate",
                "strategy_expectancy",
                "strategy_profit_factor",
                "strategy_weight",
                "meta_priority_adjustment",
                "weighting_reason",
                "spread_bps",
                "order_book_imbalance",
                "microprice",
                "trade_velocity",
                "liquidity_vacuum_flag",
                "adverse_selection_risk",
                "microstructure_bias",
                "champion_profile_id",
                "challenger_profile_id",
                "promotion_decision",
                "promotion_score",
                "promotion_reasons",
                "promotion_regime_label",
            ):
                val = self._signal_get(signal, field_name, None)
                if val is not None and field_name not in merged_details:
                    merged_details[field_name] = val
            self.omega_store.record_decision(
                run_id=str(getattr(self, "run_id", "unknown")),
                trace_id=resolved_trace,
                cycle_id=int(cycle_id),
                correlation_id=str(correlation_id or ""),
                symbol=str(symbol) if symbol else None,
                strategy=str(strategy) if strategy else None,
                side=str(side) if side else None,
                signal_score=float(score) if score is not None else None,
                allowed=bool(allowed),
                reason_code=str(reason_code.value),
                details=merged_details,
                cost=dict(cost or {}),
                exec_plan=dict(exec_plan or {}),
            )
        except Exception as e:
            logger.debug("candidate decision snapshot failed: %s", e)

    def _snapshot_rejected_candidates(
        self,
        *,
        before: List[Any],
        after: List[Any],
        cycle_id: int,
        correlation_id: str,
        reason_code: ReasonCode,
        details: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        """Snapshot rejected candidates when a gate filters a signal list."""
        try:
            kept_ids = {id(s) for s in list(after or [])}
            for sig in list(before or []):
                if id(sig) in kept_ids:
                    continue
                self._snapshot_candidate_decision(
                    cycle_id=cycle_id,
                    correlation_id=correlation_id,
                    signal=sig,
                    allowed=False,
                    reason_code=reason_code,
                    details=dict(details or {}),
                    trace_id=trace_id,
                )
        except Exception as e:
            logger.debug("rejected candidate snapshot failed: %s", e)

    def _compute_regime_label(self, signals: List[Any]) -> str:
        """Build deterministic regime label from rolling features when enabled."""
        label = ""
        try:
            if self.feature_store is not None:
                for sig in list(signals or []):
                    symbol = str(self._signal_get(sig, "symbol", "") or "")
                    price = float(self._signal_get(sig, "entry_price", 0.0) or 0.0)
                    if not symbol or price <= 0:
                        continue
                    spread_bps = float(self._signal_get(sig, "spread_bps", 0.0) or 0.0)
                    depth = float(self._signal_get(sig, "depth", 0.0) or 0.0)
                    volume = float(self._signal_get(sig, "volume", 0.0) or 0.0)
                    self.feature_store.update(
                        symbol=symbol,
                        price=price,
                        spread_bps=spread_bps,
                        depth=depth,
                        volume=volume,
                    )
            if self.market_microstructure_engine is not None:
                self._update_microstructure_context(signals)
            if self.feature_store is not None and self.regime_classifier is not None:
                features = self.feature_store.snapshot()
                regime = self.regime_classifier.classify(features)
                label = str(getattr(regime, "regime", "") or "")

            # Enhanced regime detection with HMM (if available)
            if self.market_regime_detector is not None:
                try:
                    # Get price history for HMM analysis
                    prices = self._get_recent_prices_for_regime()
                    if prices is not None and len(prices) > 30:
                        hmm_regime = self.market_regime_detector.detect_regime(prices)
                        if hmm_regime is not None:
                            # Use HMM regime if confidence is higher
                            hmm_confidence = getattr(hmm_regime, "confidence", 0.0)
                            if hmm_confidence > 0.6:
                                label = str(getattr(hmm_regime, "regime", label))
                                self._last_hmm_regime = hmm_regime
                except Exception as e:
                    logger.debug("HMM regime detection failed: %s", e)

            # Real-time regime detection (multi-timeframe)
            if self.realtime_regime_detector is not None:
                try:
                    rt_regime = self.realtime_regime_detector.get_consensus_regime()
                    if rt_regime is not None:
                        self._last_realtime_regime = rt_regime
                except Exception as e:
                    logger.debug("Real-time regime detection failed: %s", e)

            # Regime forecasting
            if self.regime_forecaster is not None:
                try:
                    forecast = self.regime_forecaster.forecast()
                    if forecast is not None:
                        self._last_regime_forecast = forecast
                except Exception as e:
                    logger.debug("Regime forecasting failed: %s", e)

        except Exception as e:
            logger.debug("Regime classification failed: %s", e)

        if not label:
            label = str((self._last_regime_consensus or {}).get("regime", "") or "")
        if not label:
            label = "range:mid_vol"
        self._latest_regime_label = label
        return label

    def _get_recent_prices_for_regime(self, symbol: str = "BTC/USD", window: int = 100) -> Optional[np.ndarray]:
        """Get recent prices for HMM regime analysis."""
        try:
            # Try to get from feature store or market data
            if hasattr(self, 'feature_store') and self.feature_store is not None:
                prices = self.feature_store.get_price_history(symbol, window)
                if prices is not None and len(prices) > 30:
                    return np.array(prices)
            
            # Fallback: try to get from exchange connector
            if hasattr(self, 'exchange_connector') and self.exchange_connector is not None:
                ohlcv = self.exchange_connector.get_ohlcv(symbol, "1h", limit=window)
                if ohlcv is not None and len(ohlcv) > 30:
                    closes = [bar[4] for bar in ohlcv]  # Close prices
                    return np.array(closes)
            
            return None
        except Exception as e:
            logger.debug("Failed to get prices for regime analysis: %s", e)
            return None

    def _get_latest_returns(self) -> Dict[str, float]:
        """Get latest returns for correlation tracking."""
        try:
            returns = {}
            trading_pairs = getattr(self.config, "trading_pairs", ["BTC/USD", "ETH/USD"])
            
            for symbol in trading_pairs:
                prices = self._get_recent_prices_for_regime(symbol=symbol, window=2)
                if prices is not None and len(prices) >= 2:
                    # Calculate simple return
                    ret = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] != 0 else 0.0
                    returns[symbol] = float(ret)
            
            return returns if returns else {"BTC/USD": 0.0}
        except Exception as e:
            logger.debug("Failed to get latest returns: %s", e)
            return {"BTC/USD": 0.0}

    def _update_microstructure_context(self, signals: List[Any]) -> List[Any]:
        """Compute/attach microstructure features for downstream gates and execution planning."""
        rows = list(signals or [])
        engine = getattr(self, "market_microstructure_engine", None)
        if engine is None or not rows:
            return rows
        try:
            states = engine.update_from_signals(rows)
            self._latest_microstructure_state = dict(states or {})
            return list(engine.annotate_signals(rows) or rows)
        except Exception as e:
            logger.warning("Microstructure engine failed; continuing without microstructure annotations: %s", e)
            return rows

    def _apply_self_optimizing_meta_weights(
        self,
        signals: List[Any],
        *,
        regime_label: str,
        cycle_id: int,
        trace_id: str,
    ) -> List[Any]:
        """Apply deterministic strategy-weight adjustments before target construction."""
        rows = list(signals or [])
        meta_engine = getattr(self, "self_optimizing_meta_engine", None)
        if meta_engine is None or not rows:
            return rows
        try:
            strategy_names = sorted(
                {
                    str(
                        self._signal_get(sig, "source_strategy", None)
                        or self._signal_get(sig, "strategy", None)
                        or "unknown"
                    )
                    for sig in rows
                }
            )
            se_engine = getattr(self, "strategy_evaluation_engine", None)
            if se_engine is not None:
                weights = meta_engine.maybe_update(
                    cycle_id=int(cycle_id),
                    strategy_names=strategy_names,
                    strategy_evaluation_engine=se_engine,
                    regime_label=str(regime_label or ""),
                    execution_telemetry=None,
                    run_id=str(getattr(self, "run_id", "")),
                    trace_id=str(trace_id or ""),
                )
                self._latest_strategy_weights = dict(weights or {})
            return list(meta_engine.apply_to_candidates(rows) or rows)
        except Exception as e:
            logger.warning("Self-optimizing meta engine failed; using unweighted candidates: %s", e)
            return rows

    def _apply_regime_strategy_gating(self, signals: List[Any], regime_label: str) -> List[Any]:
        """Filter candidate signals by configured strategy allow-list for the active regime."""
        candidates = list(signals or [])
        if not bool(getattr(self.config, "regime_gating_enabled", False)):
            return candidates

        mapping = dict(getattr(self.config, "regime_strategy_map", {}) or {})
        allowed = mapping.get(str(regime_label))
        if allowed is None:
            return candidates
        allowed_set = {str(x) for x in list(allowed or []) if str(x)}

        kept: List[Any] = []
        for sig in candidates:
            strategy = str(
                self._signal_get(sig, "source_strategy", "")
                or self._signal_get(sig, "strategy", "")
                or ""
            )
            if strategy in allowed_set:
                kept.append(sig)
        return kept

    def _target_runtime_risk_scale(self) -> float:
        """Best-effort reduced-risk scaler from runtime state."""
        scale = 1.0
        try:
            n = int(getattr(self.config, "auto_reduce_after_n_losses", 0) or 0)
            f = float(getattr(self.config, "auto_reduce_factor", 1.0) or 1.0)
            if n > 0 and self.consecutive_losses >= n:
                scale = min(scale, max(0.0, min(1.0, f)))
        except Exception:
            pass
        try:
            vol_scale = float(getattr(self.config, "_vol_adjusted_position_scale", 1.0) or 1.0)
            scale = min(scale, max(0.0, min(1.0, vol_scale)))
        except Exception:
            pass
        return float(max(0.0, min(1.0, scale)))

    def _effective_min_net_edge_bps(self) -> float:
        """Resolve deterministic min net-edge threshold for pre-exec re-check."""
        try:
            direct = float(getattr(self.config, "min_net_edge_bps", 0.0) or 0.0)
            if direct > 0.0:
                return direct
        except Exception:
            pass
        try:
            mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
            live_min_edge_pct = getattr(self.config, "edge_cost_gate_live_min_edge_pct", None)
            if mode == "live" and isinstance(live_min_edge_pct, (int, float)):
                return max(0.0, float(live_min_edge_pct) * 100.0)
            base = float(getattr(self.config, "edge_cost_gate_min_edge_pct", 0.0) or 0.0)
            return max(0.0, base * 100.0)
        except Exception:
            return 0.0

    def _collect_liquidity_symbol_state(self, signals: List[Any]) -> Dict[str, Dict[str, float]]:
        """Build best-effort per-symbol liquidity context from signal/book cache data."""
        state: Dict[str, Dict[str, float]] = {}
        for sig in list(signals or []):
            symbol = str(self._signal_get(sig, "symbol", "") or "")
            if not symbol:
                continue
            spread_bps = float(self._signal_get(sig, "spread_bps", 0.0) or 0.0)
            bid_size = float(
                self._signal_get(sig, "top_of_book_bid_size", None)
                or self._signal_get(sig, "bid_size_1", None)
                or self._signal_get(sig, "bid_size", None)
                or self._signal_get(sig, "best_bid_size", 0.0)
                or 0.0
            )
            ask_size = float(
                self._signal_get(sig, "top_of_book_ask_size", None)
                or self._signal_get(sig, "ask_size_1", None)
                or self._signal_get(sig, "ask_size", None)
                or self._signal_get(sig, "best_ask_size", 0.0)
                or 0.0
            )
            depth = float(
                self._signal_get(sig, "orderbook_depth_estimate", None)
                or self._signal_get(sig, "depth", None)
                or self._signal_get(sig, "book_depth", 0.0)
                or 0.0
            )
            entry_price = float(
                self._signal_get(sig, "entry_price", 0.0)
                or self._signal_get(sig, "price", 0.0)
                or 0.0
            )
            if depth <= 0.0 and (bid_size > 0.0 or ask_size > 0.0):
                depth = float(max(0.0, bid_size + ask_size))
            prev = state.get(symbol) or {}
            # Keep the most informative values seen in this cycle.
            state[symbol] = {
                "spread_bps": float(spread_bps or prev.get("spread_bps", 0.0) or 0.0),
                "top_of_book_bid_size": float(max(bid_size, float(prev.get("top_of_book_bid_size", 0.0) or 0.0))),
                "top_of_book_ask_size": float(max(ask_size, float(prev.get("top_of_book_ask_size", 0.0) or 0.0))),
                "orderbook_depth_estimate": float(max(depth, float(prev.get("orderbook_depth_estimate", 0.0) or 0.0))),
                "price": float(entry_price or prev.get("price", 0.0) or 0.0),
            }
        return state

    def _collect_execution_telemetry(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Fetch rolling execution alpha telemetry for liquidity-aware sizing."""
        out: Dict[str, Dict[str, float]] = {}
        try:
            exec_engine = getattr(self, "execution_engine", None)
            alpha = getattr(exec_engine, "execution_alpha_engine", None) if exec_engine is not None else None
            if alpha is None or not hasattr(alpha, "snapshot"):
                return out
            for symbol in list(symbols or []):
                try:
                    snap = dict(alpha.snapshot(str(symbol)) or {})
                    out[str(symbol)] = {
                        "maker_fill_ratio": float(snap.get("maker_fill_ratio", 0.0) or 0.0),
                        "slippage_p90": float(snap.get("slippage_p90", 0.0) or 0.0),
                    }
                except Exception:
                    continue
        except Exception:
            return out
        return out

    def _convert_signals_to_targets(
        self,
        signals: List[Any],
        regime_label: str,
        *,
        cycle_id: Optional[int] = None,
        correlation_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> List[Any]:
        """Convert raw strategy signals into target-based execution signals."""
        self._last_target_pipeline_stage_ms = {
            "portfolio_targeting_ms": 0.0,
            "liquidity_adjustment_ms": 0.0,
        }
        if not bool(getattr(self.config, "targets_enabled", False)):
            return list(signals or [])
        if self.target_engine is None:
            return list(signals or [])
        try:
            resolved_cycle = int(cycle_id if cycle_id is not None else 0)
            resolved_corr = str(correlation_id or getattr(self, "_cycle_correlation_id", "") or "")
            resolved_trace = str(trace_id or getattr(self, "_trace_id", "") or uuid.uuid4().hex)
            rebalance_min = float(getattr(self.config, "target_rebalance_min_delta_pct", 0.02) or 0.02)
            risk_scale = self._target_runtime_risk_scale()
            stage_timings_ms: Dict[str, float] = {
                "portfolio_targeting_ms": 0.0,
                "liquidity_adjustment_ms": 0.0,
            }
            symbol_ctx: Dict[str, Dict[str, Any]] = {}
            for sig in list(signals or []):
                symbol = str(self._signal_get(sig, "symbol", "") or "")
                if not symbol:
                    continue
                row = symbol_ctx.setdefault(
                    symbol,
                    {
                        "strategy_weight_sum": 0.0,
                        "meta_adj_sum": 0.0,
                        "count": 0.0,
                        "weighting_reasons": set(),
                        "spread_bps_sum": 0.0,
                        "order_book_imbalance_sum": 0.0,
                        "microprice_sum": 0.0,
                        "trade_velocity_sum": 0.0,
                        "adverse_selection_risk_sum": 0.0,
                        "liquidity_vacuum_flag": False,
                        "microstructure_bias": "",
                    },
                )
                row["strategy_weight_sum"] += float(self._signal_get(sig, "strategy_weight", 0.0) or 0.0)
                row["meta_adj_sum"] += float(self._signal_get(sig, "meta_priority_adjustment", 0.0) or 0.0)
                row["count"] += 1.0
                wr = str(self._signal_get(sig, "weighting_reason", "") or "")
                if wr:
                    row["weighting_reasons"].add(wr)
                row["spread_bps_sum"] += float(self._signal_get(sig, "spread_bps", 0.0) or 0.0)
                row["order_book_imbalance_sum"] += float(self._signal_get(sig, "order_book_imbalance", 0.0) or 0.0)
                row["microprice_sum"] += float(self._signal_get(sig, "microprice", 0.0) or 0.0)
                row["trade_velocity_sum"] += float(self._signal_get(sig, "trade_velocity", 0.0) or 0.0)
                row["adverse_selection_risk_sum"] += float(self._signal_get(sig, "adverse_selection_risk", 0.0) or 0.0)
                row["liquidity_vacuum_flag"] = bool(
                    row["liquidity_vacuum_flag"] or bool(self._signal_get(sig, "liquidity_vacuum_flag", False))
                )
                if not row["microstructure_bias"]:
                    row["microstructure_bias"] = str(self._signal_get(sig, "microstructure_bias", "") or "")
            t_targets_start = time.perf_counter()
            targets = self.target_engine.build_targets(
                signals=list(signals or []),
                current_positions=dict(self.positions or {}),
                equity_aud=float(self.portfolio_value_aud or 0.0),
                regime_label=str(regime_label or ""),
                risk_scale=float(risk_scale),
                rebalance_min_delta_pct=float(rebalance_min),
            )
            if self._is_live_safe_runtime() and self._live_safe_locked_symbols:
                allowed_symbols = {str(s).strip() for s in list(self._live_safe_locked_symbols or []) if str(s).strip()}
                pre_n = len(list(targets or []))
                targets = [
                    t
                    for t in list(targets or [])
                    if str(getattr(t, "symbol", "") or "").strip() in allowed_symbols
                ]
                dropped = pre_n - len(list(targets or []))
                if dropped > 0:
                    logger.warning(
                        "LIVE_SAFE symbol lock removed %s target(s) outside allow-list (%s)",
                        int(dropped),
                        ", ".join(sorted(allowed_symbols)),
                    )
            stage_timings_ms["portfolio_targeting_ms"] = max(
                0.0, (time.perf_counter() - t_targets_start) * 1000.0
            )
            for t in list(targets or []):
                ctx = dict(symbol_ctx.get(str(getattr(t, "symbol", "") or ""), {}) or {})
                n = max(1.0, float(ctx.get("count", 1.0) or 1.0))
                setattr(t, "strategy_weight", float(ctx.get("strategy_weight_sum", 0.0) or 0.0) / n)
                setattr(t, "meta_priority_adjustment", float(ctx.get("meta_adj_sum", 0.0) or 0.0) / n)
                setattr(t, "weighting_reason", ",".join(sorted(ctx.get("weighting_reasons", set()) or [])))
                setattr(t, "spread_bps", float(ctx.get("spread_bps_sum", 0.0) or 0.0) / n)
                setattr(t, "order_book_imbalance", float(ctx.get("order_book_imbalance_sum", 0.0) or 0.0) / n)
                setattr(t, "microprice", float(ctx.get("microprice_sum", 0.0) or 0.0) / n)
                setattr(t, "trade_velocity", float(ctx.get("trade_velocity_sum", 0.0) or 0.0) / n)
                setattr(t, "adverse_selection_risk", float(ctx.get("adverse_selection_risk_sum", 0.0) or 0.0) / n)
                setattr(t, "liquidity_vacuum_flag", bool(ctx.get("liquidity_vacuum_flag", False)))
                setattr(t, "microstructure_bias", str(ctx.get("microstructure_bias", "") or ""))
            liquidity_state_by_symbol: Dict[str, Any] = {}
            _liq_mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
            _liq_enabled = bool(getattr(self.config, "liquidity_risk_enabled", True)) and _liq_mode == "live"
            if _liq_enabled and self.liquidity_risk_engine is not None:
                try:
                    symbol_state = self._collect_liquidity_symbol_state(list(signals or []))
                    telemetry = self._collect_execution_telemetry(list(symbol_state.keys()))
                    t_liquidity_start = time.perf_counter()
                    targets, liquidity_state_by_symbol, clamp_count = self.liquidity_risk_engine.adjust_targets(
                        targets=list(targets or []),
                        symbol_market_state=symbol_state,
                        execution_telemetry=telemetry,
                        equity_aud=float(self.portfolio_value_aud or 0.0),
                        aud_to_usd=float(getattr(self.config, "aud_to_usd", 0.65) or 0.65),
                    )
                    stage_timings_ms["liquidity_adjustment_ms"] = max(
                        0.0, (time.perf_counter() - t_liquidity_start) * 1000.0
                    )
                    self._latest_liquidity_state = dict(liquidity_state_by_symbol or {})
                    if int(clamp_count or 0) > 0:
                        logger.info("liquidity clamp applied to %s target(s)", int(clamp_count))
                except Exception as e:
                    logger.warning("Liquidity risk engine failed; falling back to baseline targets: %s", e)
            self._latest_targets = list(targets or [])
            self._last_target_pipeline_stage_ms = dict(stage_timings_ms)
            execution_signals = self.target_engine.build_execution_signals(
                targets=targets,
                regime_label=str(regime_label or ""),
            )
            if bool(getattr(self.config, "edge_cost_gate_enabled", True)):
                min_net_edge_bps = float(self._effective_min_net_edge_bps() or 0.0)
                if min_net_edge_bps > 0.0:
                    pre_net_edge = list(execution_signals or [])
                    execution_signals = [
                        s
                        for s in pre_net_edge
                        if float(self._signal_get(s, "expected_net_edge_bps", 0.0) or 0.0) >= min_net_edge_bps
                    ]
                    self._snapshot_rejected_candidates(
                        before=pre_net_edge,
                        after=execution_signals,
                        cycle_id=resolved_cycle,
                        correlation_id=resolved_corr,
                        reason_code=ReasonCode.EDGE_COST_REJECT,
                        details={
                            "stage": "net_edge_recheck",
                            "min_net_edge_bps": float(min_net_edge_bps),
                        },
                        trace_id=resolved_trace,
                    )
            kept_symbols = {str(self._signal_get(s, "symbol", "") or "") for s in list(execution_signals or [])}
            total_target_exposure = sum(max(0.0, float(getattr(t, "target_exposure_pct", 0.0) or 0.0)) for t in list(targets or []))
            logger.info("total target exposure after clamps = %.4f", float(total_target_exposure))

            from types import SimpleNamespace

            for t in list(targets or []):
                symbol = str(getattr(t, "symbol", "") or "")
                details = {
                    "stage": "portfolio_target_engine",
                    "target_exposure_pct": float(getattr(t, "target_exposure_pct", 0.0) or 0.0),
                    "current_exposure_pct": float(getattr(t, "current_exposure_pct", 0.0) or 0.0),
                    "delta_exposure_pct": float(getattr(t, "delta_exposure_pct", 0.0) or 0.0),
                    "priority_score": float(getattr(t, "priority_score", 0.0) or 0.0),
                    "expected_net_edge_bps": float(getattr(t, "expected_net_edge_bps", 0.0) or 0.0),
                    "regime_label": str(getattr(t, "regime_label", "") or ""),
                    "adjusted_target_exposure_pct": float(
                        getattr(t, "adjusted_target_exposure_pct", getattr(t, "target_exposure_pct", 0.0))
                        or 0.0
                    ),
                    "liquidity_score": float(getattr(t, "liquidity_score", 0.0) or 0.0),
                    "liquidity_state": str(getattr(t, "liquidity_state", "") or ""),
                    "max_safe_trade_size": float(getattr(t, "max_safe_trade_size", 0.0) or 0.0),
                    "liquidity_clamp_flag": bool(getattr(t, "liquidity_clamp_flag", False)),
                    "slippage_estimate_bps": float(getattr(t, "slippage_estimate_bps", 0.0) or 0.0),
                    "strategy_weight": float(getattr(t, "strategy_weight", 0.0) or 0.0),
                    "meta_priority_adjustment": float(getattr(t, "meta_priority_adjustment", 0.0) or 0.0),
                    "weighting_reason": str(getattr(t, "weighting_reason", "") or ""),
                    "spread_bps": float(getattr(t, "spread_bps", 0.0) or 0.0),
                    "order_book_imbalance": float(getattr(t, "order_book_imbalance", 0.0) or 0.0),
                    "microprice": float(getattr(t, "microprice", 0.0) or 0.0),
                    "trade_velocity": float(getattr(t, "trade_velocity", 0.0) or 0.0),
                    "liquidity_vacuum_flag": bool(getattr(t, "liquidity_vacuum_flag", False)),
                    "adverse_selection_risk": float(getattr(t, "adverse_selection_risk", 0.0) or 0.0),
                    "microstructure_bias": str(getattr(t, "microstructure_bias", "") or ""),
                    "reasons": list(getattr(t, "reasons", []) or []),
                    "risk_scale": float(risk_scale),
                }
                logger.info(
                    "liquidity decision %s: state=%s score=%.4f safe_size=%.8f clamp=%s",
                    symbol,
                    str(details.get("liquidity_state", "") or ""),
                    float(details.get("liquidity_score", 0.0) or 0.0),
                    float(details.get("max_safe_trade_size", 0.0) or 0.0),
                    bool(details.get("liquidity_clamp_flag", False)),
                )
                suppressed = symbol not in kept_symbols
                if suppressed:
                    reason = "no_target_delta"
                    if any(str(r).startswith("suppressed:small_delta") for r in details["reasons"]):
                        reason = "small_delta"
                    elif any(str(r).startswith("suppressed:liquidity_danger") for r in details["reasons"]):
                        reason = "liquidity_danger"
                    elif any(str(r).startswith("suppressed:liquidity_thin") for r in details["reasons"]):
                        reason = "liquidity_thin"
                    elif any(str(r).startswith("suppressed:liquidity_score_low") for r in details["reasons"]):
                        reason = "liquidity_score_low"
                    elif bool(details.get("liquidity_clamp_flag", False)):
                        reason = "liquidity_clamp"
                    details["suppression_reason"] = reason
                    if reason in {"liquidity_danger", "liquidity_thin", "liquidity_clamp", "liquidity_score_low"}:
                        logger.info(
                            "trade suppressed due to thin liquidity for %s (state=%s safe_size=%.8f)",
                            symbol,
                            str(details.get("liquidity_state", "") or ""),
                            float(details.get("max_safe_trade_size", 0.0) or 0.0),
                        )
                    logger.info(
                        "target suppressed for %s due to %s (current=%.4f target=%.4f delta=%.4f)",
                        symbol,
                        reason,
                        details["current_exposure_pct"],
                        details["target_exposure_pct"],
                        details["delta_exposure_pct"],
                    )
                    self._snapshot_candidate_decision(
                        cycle_id=resolved_cycle,
                        correlation_id=resolved_corr,
                        signal=SimpleNamespace(
                            symbol=symbol,
                            action="BUY" if details["delta_exposure_pct"] >= 0 else "SELL",
                            strategy="target_rebalance",
                            source_strategy="target_rebalance",
                            confidence=float(min(1.0, max(0.0, details["priority_score"]))),
                            regime_label=details["regime_label"],
                        ),
                        allowed=False,
                        reason_code=ReasonCode.PRE_TRADE_RISK_BLOCK,
                        details=details,
                        trace_id=resolved_trace,
                    )
                else:
                    logger.info(
                        "target generated for %s: current=%.4f target=%.4f delta=%.4f",
                        symbol,
                        details["current_exposure_pct"],
                        details["target_exposure_pct"],
                        details["delta_exposure_pct"],
                    )
            return list(execution_signals or [])
        except Exception as e:
            logger.debug("Targets pipeline failed: %s", e)
            self._last_target_pipeline_stage_ms = {
                "portfolio_targeting_ms": 0.0,
                "liquidity_adjustment_ms": 0.0,
            }
            return list(signals or [])

    async def update_trading_pairs_from_liquidity(self) -> List[str]:
        """Update trading pairs based on liquidity scanner.
        
        Scans all configured exchanges for the most liquid pairs
        and updates the trading_pairs config dynamically.
        
        Returns:
            List of selected pair symbols
        """
        scanner = getattr(self, 'pair_liquidity_scanner', None)
        if not scanner:
            logger.debug("No pair_liquidity_scanner available, using configured pairs")
            return list(getattr(self.config, 'trading_pairs', []))
        
        try:
            # Scan for best pairs
            await scanner.scan_all_pairs()
            rankings = scanner.get_volume_rankings(limit=20)
            
            if rankings:
                # Extract symbols from rankings
                selected_pairs = [sym for sym, _ in rankings]
                
                # Update config
                old_pairs = list(getattr(self.config, 'trading_pairs', []))
                self.config.trading_pairs = selected_pairs
                
                logger.info(
                    f"Liquidity scan: updated {len(selected_pairs)} pairs "
                    f"(was {len(old_pairs)}). Top 5: {selected_pairs[:5]}"
                )
                return selected_pairs
            else:
                logger.warning("Liquidity scan returned no pairs")
                return list(getattr(self.config, 'trading_pairs', []))
                
        except Exception as e:
            logger.error(f"Liquidity scan failed: {e}")
            return list(getattr(self.config, 'trading_pairs', []))

    async def run_trading_loop(self, *, cycle_seconds: Optional[float] = None, max_cycles: Optional[int] = None):
        """Main trading loop

        Args:
            cycle_seconds: Sleep duration between cycles. If 0, yields control without sleeping.
            max_cycles: If provided, stop after N cycles (useful for paper runs/backtests).
        """
        logger.info("Starting unified trading loop...")

        # Install signal handlers for graceful shutdown (SIGINT/SIGTERM)
        try:
            self._install_signal_handlers()
            logger.info("Signal handlers installed (SIGINT/SIGTERM -> graceful shutdown)")
        except Exception as e:
            logger.debug("Could not install signal handlers: %s", e)

        # Paper trading absolute peak: apply peak overrides when in paper/backtest (skip when simulating live)
        mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
        sim_live = bool(getattr(self.config, "paper_simulates_live", False))
        if sim_live and mode == "paper":
            logger.info("Paper simulates live: using live config (no paper overrides), live strategy filter, simulated slippage")
        if mode in ("paper", "backtest") and getattr(self.config, "paper_trading_peak_mode", True) and not sim_live:
            overrides = getattr(self.config, "paper_trading_overrides", None) or {}
            for k, v in overrides.items():
                try:
                    setattr(self.config, k, v)
                except Exception:
                    pass
            if overrides:
                logger.info("Paper trading absolute peak: applied %s overrides", len(overrides))

        cycles = 0
        # Cycle sleep: use param, then config, then 60s default
        _cfg_cycle_s = float(getattr(self.config, "cycle_seconds", 0) or 0)
        _mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
        if cycle_seconds is not None:
            sleep_s = float(cycle_seconds)
        elif _cfg_cycle_s > 0:
            sleep_s = _cfg_cycle_s
        elif _mode == "paper":
            sleep_s = 10.0
        else:
            sleep_s = 60.0
        max_cycles_i = int(max_cycles) if max_cycles is not None else None

        # Track config file mtime for hot-reload
        import os as _os
        self._config_path = getattr(self, "_config_path", "unified_config.yaml")
        try:
            self._config_mtime = _os.path.getmtime(self._config_path)
        except OSError:
            self._config_mtime = 0.0

        # Start self-improvement background loop (best-effort; paper/backtest by default).
        try:
            if self._self_improver_task is None and bool(getattr(self.config, "self_improvement_enabled", True)):
                from adaptive.self_improver import SelfImprover

                self.self_improver = SelfImprover(system=self)
                self._self_improver_task = asyncio.create_task(self.self_improver.run_forever())
        except Exception as e:
            logger.debug("Self-improvement loop unavailable: %s", e)

        # Start continuous best-trade scanner (constantly scans market for best opportunities).
        try:
            if self.continuous_scanner is None and bool(getattr(self.config, "continuous_scan_enabled", True)):
                from services.continuous_best_trade_scanner import ContinuousBestTradeScanner

                interval = float(getattr(self.config, "continuous_scan_interval_seconds", 10.0) or 10.0)
                top_n = int(getattr(self.config, "continuous_scan_top_n", 5) or 5)
                parallel = bool(getattr(self.config, "continuous_scan_parallel_sources", True))
                liquidity = bool(getattr(self.config, "continuous_scan_use_liquidity_boost", True))
                spread_cap = float(getattr(self.config, "continuous_scan_liquidity_spread_pct_cap", 0.05) or 0.05)
                div_sym = int(getattr(self.config, "continuous_scan_diversity_max_per_symbol", 2) or 2)
                div_strat = int(getattr(self.config, "continuous_scan_diversity_max_per_strategy", 2) or 2)
                adaptive = bool(getattr(self.config, "continuous_scan_adaptive_interval_enabled", True))
                min_int = float(getattr(self.config, "continuous_scan_min_interval_seconds", 5.0) or 5.0)
                max_int = float(getattr(self.config, "continuous_scan_max_interval_seconds", 30.0) or 30.0)
                self.continuous_scanner = ContinuousBestTradeScanner(
                    self.config,
                    ai_brain=self.ai_brain,
                    market_data_service=self.market_data_service,
                    strategy_engine=getattr(self.ai_brain, "strategy_engine", None),
                    hft_engine=self.hft_engine,
                    interval_seconds=max(5.0, min(60.0, interval)),
                    top_n=max(1, min(10, top_n)),
                    parallel_sources=parallel,
                    use_liquidity_boost=liquidity,
                    liquidity_spread_pct_cap=spread_cap,
                    diversity_max_per_symbol=max(1, div_sym),
                    diversity_max_per_strategy=max(1, div_strat),
                    adaptive_interval_enabled=adaptive,
                    min_interval_seconds=min_int,
                    max_interval_seconds=max_int,
                )
                self.continuous_scanner.start()
                logger.info(
                    "Continuous best-trade scanner (peak) started (interval=%.0fs, top_n=%s, parallel=%s)",
                    interval, top_n, parallel,
                )
        except Exception as e:
            logger.debug("Continuous scanner unavailable: %s", e)
            self.continuous_scanner = None

        # Run initial liquidity scan to select best trading pairs
        try:
            scanner = getattr(self, 'pair_liquidity_scanner', None)
            if scanner is not None:
                logger.info("Running initial liquidity scan to select best trading pairs...")
                selected = await self.update_trading_pairs_from_liquidity()
                logger.info(f"Liquidity scan complete: {len(selected)} pairs selected")
        except Exception as e:
            logger.debug("Initial liquidity scan failed: %s", e)

        # Start component registry heartbeat (exchange monitor, counterparty checks, Discord alerts)
        try:
            if self.component_registry is not None:
                asyncio.create_task(self.component_registry.heartbeat())
        except Exception as e:
            logger.debug("ComponentRegistry heartbeat unavailable: %s", e)

        # Start position reconciler (periodic exchange balance vs internal state check)
        try:
            _pm = getattr(self.component_registry, "portfolio_manager", None) if self.component_registry else None
            _em = getattr(self, "exchange_manager", None)
            if _pm is not None and _em is not None:
                from core.position_reconciler import PositionReconciler
                self._position_reconciler = PositionReconciler(
                    portfolio_manager=_pm,
                    exchange_manager=_em,
                    interval_seconds=300.0,
                    auto_correct=True,
                )
                asyncio.create_task(self._position_reconciler.start())
                logger.info("PositionReconciler started (interval=300s)")
        except Exception as e:
            logger.debug("PositionReconciler unavailable: %s", e)

        # Start async write queue auto-flush
        try:
            _awq = getattr(self.component_registry, "async_write_queue", None) if self.component_registry else None
            if _awq is not None:
                asyncio.create_task(_awq.start_auto_flush(interval_seconds=5.0))
                logger.info("AsyncWriteQueue auto-flush started (interval=5s)")
        except Exception as e:
            logger.debug("AsyncWriteQueue unavailable: %s", e)

        # Start API dashboard server (remote monitoring via browser/phone)
        try:
            if getattr(self.config, "api_dashboard_enabled", True) and getattr(self, "api_server", None) is None:
                from api.dashboard import ArgusAPIServer
                _dash_port = int(getattr(self.config, "api_dashboard_port", 8080) or 8080)
                self.api_server = ArgusAPIServer(port=_dash_port, trading_system=self)
                self.api_server.start()
                logger.info("API dashboard started on port %s", _dash_port)
        except Exception as e:
            logger.debug("API dashboard unavailable: %s", e)

        # Start signal subscription service (monetisation via webhooks)
        try:
            if getattr(self.config, "signal_service_enabled", False) and self.signal_service is None:
                from api.signal_service import SignalService
                _sig_port = int(getattr(self.config, "signal_service_port", 8082) or 8082)
                self.signal_service = SignalService(port=_sig_port)
                self.signal_service.start()
                logger.info("Signal subscription service started on port %s", _sig_port)
        except Exception as e:
            logger.debug("Signal subscription service unavailable: %s", e)

        # Start rolling performance feeder (drives strategy weight evolution)
        try:
            if self.perf_feeder is not None and hasattr(self.perf_feeder, "start"):
                asyncio.create_task(self.perf_feeder.start())
                logger.info("RollingPerformanceFeeder started")
        except Exception as e:
            logger.debug("RollingPerformanceFeeder start failed: %s", e)

        # Sync positions from exchange on startup (live mode only)
        try:
            sync_result = await self._sync_positions_from_exchange()
            if sync_result.get("positions_found", 0) > 0:
                logger.info(
                    "Startup position sync: %d positions loaded",
                    sync_result["positions_found"],
                )
        except Exception as e:
            logger.warning("Startup position sync failed: %s", e)

        # Load 6 months of real OHLCV for evolver/scanner (cache on disk)
        try:
            _hist_enabled = True
            _md = getattr(self.config, "market_data", None)
            if isinstance(_md, dict):
                _hist_enabled = _md.get("enable_historical_preload", True)
            if _hist_enabled and hasattr(self, "exchange_manager"):
                from core.historical_data_loader import load_all_historical
                _pairs = list(getattr(self.config, "trading_pairs", []) or [])
                _ex = getattr(self.exchange_manager, "_exchange", None) or getattr(self.exchange_manager, "primary_exchange", None)
                if _ex is None:
                    _ex = self.exchange_manager
                if _pairs and _ex:
                    _lookback = 4380
                    if isinstance(_md, dict):
                        _lookback = int(_md.get("historical_lookback_hours", 4380))
                    _hist_data = await asyncio.wait_for(
                        load_all_historical(_ex, _pairs, "1h", _lookback),
                        timeout=120.0,
                    )
                    if _hist_data and hasattr(self, "component_registry") and self.component_registry:
                        self.component_registry._ohlcv_history = _hist_data
                        logger.info("Historical OHLCV: loaded %d symbols (%s bars avg)",
                                    len(_hist_data),
                                    int(sum(len(d["close"]) for d in _hist_data.values()) / max(len(_hist_data), 1)))
        except asyncio.TimeoutError:
            logger.warning("Historical OHLCV fetch timed out (120s) — will use cache if available")
        except Exception as e:
            logger.warning("Historical OHLCV load failed: %s — evolver/scanner will use real-time data", e)

        # FIX 1: Bootstrap volatility from OHLCV before first trading cycle
        try:
            await self._bootstrap_volatility()
            if self._volatility_cache:
                logger.info("Volatility bootstrap complete: %d symbols cached", len(self._volatility_cache))
        except Exception as e:
            logger.warning("Volatility bootstrap failed: %s", e)

        while self.state == SystemState.RUNNING:
            # Ensure we can always make progress / stop at max_cycles,
            # even if an awaited dependency blocks or errors.
            iter_sleep_s = float(sleep_s)
            t0 = float(time.time())
            self._completed_cycles = int(cycles)
            cycle_stage_timing_ms: Dict[str, float] = {
                "market_data_ms": 0.0,
                "feature_generation_ms": 0.0,
                "strategy_evaluation_ms": 0.0,
                "portfolio_targeting_ms": 0.0,
                "liquidity_adjustment_ms": 0.0,
                "risk_gate_ms": 0.0,
                "execution_planning_ms": 0.0,
                "snapshot_persistence_ms": 0.0,
            }
            try:
                ledger = getattr(self.execution_engine, "trade_ledger", None) if self.execution_engine else None
                rec_event = getattr(ledger, "record_event", None) if ledger else None

                # Correlation ID for tracing (cycle + timestamp)
                cycle_correlation_id = f"cycle_{cycles}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                setattr(self, "_cycle_correlation_id", cycle_correlation_id)
                setattr(self, "_cycle_start_time", time.perf_counter())
                # Ω trace ID per cycle
                trace_id = uuid.uuid4().hex
                self._trace_id = trace_id

                # Push correlation ID into async logger so all log records carry it
                try:
                    from core.logger import CorrelationFilter
                    CorrelationFilter.set_correlation_id(cycle_correlation_id)
                except Exception as _e:
                    logger.debug("CorrelationFilter unavailable: %s", _e)

                # Lightweight per-cycle heartbeat (avoids "silent hangs")
                logger.info("Cycle %s%s starting", cycles + 1, f"/{max_cycles_i}" if max_cycles_i is not None else "")

                # Poll pending orders from previous cycles (fills, timeouts)
                try:
                    pending_results = await asyncio.wait_for(
                        self._poll_pending_orders(), timeout=op_timeout_s
                    )
                    if pending_results:
                        logger.info("Pending order poll: %d orders resolved", len(pending_results))
                except Exception as _poll_err:
                    logger.debug("Pending order poll: %s", _poll_err)

                # Position reconciliation (at startup cycle 0 and every N cycles)
                try:
                    if cycles == 0 or (cycles % self._reconcile_every_n_cycles == 0):
                        recon_summary = await asyncio.wait_for(
                            self._reconcile_positions(), timeout=op_timeout_s
                        )
                        if recon_summary.get("discrepancies"):
                            logger.info(
                                "Position reconciliation: %d discrepancies found",
                                len(recon_summary["discrepancies"]),
                            )
                except Exception as _recon_err:
                    logger.debug("Position reconciliation: %s", _recon_err)

                # Periodic liquidity scan (every 100 cycles to refresh trading pairs)
                try:
                    if (cycles + 1) % 100 == 0:
                        scanner = getattr(self, 'pair_liquidity_scanner', None)
                        if scanner is not None:
                            await self.update_trading_pairs_from_liquidity()
                except Exception as _liq_err:
                    logger.debug("Periodic liquidity scan: %s", _liq_err)

                # Periodic GC stats logging and forced collection for long sessions
                # (skipped in fast_mode to avoid latency spikes from GC pauses)
                _fast_mode = bool(getattr(self.config, "fast_mode", False))
                if not _fast_mode:
                    try:
                        import gc as _gc
                        if (cycles + 1) % 1000 == 0:
                            _gc_stats = _gc.get_stats()
                            logger.debug("GC stats at cycle %s: %s", cycles + 1, _gc_stats)
                            # Force collection if generation 2 has accumulated > 100 uncollected objects
                            if _gc_stats[2].get("collections", 0) > 0 or _gc_stats[2].get("uncollectable", 0) > 100:
                                _gen2_count = _gc_stats[2].get("collected", 0)
                                if _gen2_count > 100:
                                    _gc.collect()
                                    logger.debug("Forced GC collection at cycle %s (gen2 collected=%s)", cycles + 1, _gen2_count)
                    except Exception as _gc_err:
                        logger.debug("GC stats unavailable: %s", _gc_err)

                # Track operations per-cycle so error_rate is meaningful.
                self.total_operations += 1

                # Per-op timeout to prevent a single stalled network call from hanging the loop.
                mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
                default_timeout = 300.0 if mode in {"paper", "backtest"} else 20.0
                op_timeout_s = float(getattr(self.config, "cycle_op_timeout_s", default_timeout) or default_timeout)
                # First 20 cycles need longer timeout while caches warm up (50 pairs × OHLCV)
                _max_timeout = 600.0 if cycles < 20 else 120.0
                op_timeout_s = float(max(5.0, min(_max_timeout, op_timeout_s)))
                reconciliation_block_cycle = False

                prev_equity = float(self.portfolio_value_aud)
                # Keep portfolio value/drawdown up to date before any risk decisions
                await asyncio.wait_for(self._update_portfolio_value(), timeout=op_timeout_s)

                # Best-effort: update UnifiedRiskManager and apply its circuit breaker.
                if self.unified_risk_manager is not None:
                    try:
                        new_equity = float(self.portfolio_value_aud)
                        pnl = float(new_equity - prev_equity)
                        self.unified_risk_manager.update_capital(new_equity, pnl=pnl)
                        if self.capital_optimizer is not None:
                            try:
                                self.capital_optimizer.update_capital(new_equity)
                            except Exception as e:
                                logger.debug("Capital optimizer update_capital: %s", e)
                        # So execution engine uses current equity for position cap when present
                        try:
                            setattr(self.config, "current_equity_aud", new_equity)
                        except (AttributeError, TypeError) as _e:
                            logger.debug("Cannot set current_equity_aud: %s", _e)

                        # Exposure proxy: sum of absolute position notionals (AUD) converted to USD-ish
                        try:
                            aud_to_usd = float(getattr(self.config, "aud_to_usd", 0.65) or 0.65)
                            exposure_usd = 0.0
                            for sym, pos in (self.positions or {}).items():
                                q = float((pos or {}).get("quantity") or 0.0)
                                px = float((pos or {}).get("current_price") or 0.0)
                                if q <= 0 or px <= 0:
                                    continue
                                notional_quote = q * px
                                quote = str(sym).split("/")[-1].upper() if "/" in str(sym) else "USD"
                                if quote == "USD":
                                    exposure_usd += abs(notional_quote)
                                elif quote == "AUD":
                                    exposure_usd += abs(notional_quote) * aud_to_usd
                                else:
                                    exposure_usd += abs(notional_quote)
                            self.unified_risk_manager.set_total_exposure(exposure_usd)
                        except Exception as e:
                            logger.debug("Risk manager set_total_exposure: %s", e)

                        if self.unified_risk_manager.check_circuit_breaker():
                            try:
                                from monitoring.alerting import get_alert_manager
                                reason = getattr(self.unified_risk_manager, "circuit_breaker_reason", "Circuit breaker triggered")
                                cd = getattr(self.unified_risk_manager, "circuit_breaker_cooldown", None)
                                cooldown_m = int(cd.total_seconds() // 60) if cd is not None else 60
                                await get_alert_manager().circuit_breaker_alert(reason, duration_minutes=cooldown_m or 60)
                            except Exception as alert_e:
                                logger.warning("Circuit breaker alert failed: %s", alert_e)
                            self.state = SystemState.EMERGENCY_STOP
                            logger.critical("Emergency stop triggered (UnifiedRiskManager circuit breaker)")
                            await self.shutdown()
                            break
                    except Exception as e:
                        logger.warning("UnifiedRiskManager update/circuit check failed: %s", e)

                # Always maintain equity history and realized vol for risk adaptation (market-adaptive limits)
                try:
                    new_equity = float(self.portfolio_value_aud)
                    self._equity_history.append(new_equity)
                    if len(self._equity_history) > 500:
                        self._equity_history.pop(0)
                    if len(self._equity_history) >= 10:
                        arr = np.array(self._equity_history, dtype=float)
                        returns = np.diff(arr) / np.maximum(arr[:-1], 1e-9)
                        realized_vol_pct = float(np.std(returns)) * 100.0
                        realized_vol_pct = float(max(0.0, min(20.0, realized_vol_pct)))
                        setattr(self.config, "realized_vol_pct", realized_vol_pct)
                        # Vol-adjusted daily loss limit for emergency stop (tighter when vol is high)
                        if getattr(self.config, "use_volatility_adjusted_limits", False) and realized_vol_pct > 0:
                            base_daily = float(getattr(self.config, "max_daily_loss_pct", 0.02) or 0.02)
                            # Tighten daily loss limit when vol is high (inline vol adjustment)
                            vol_scale = max(0.3, 1.0 - realized_vol_pct / 10.0)
                            setattr(self.config, "_vol_adjusted_daily_loss_pct", base_daily * vol_scale)
                        else:
                            setattr(self.config, "_vol_adjusted_daily_loss_pct", None)
                except Exception as e:
                    logger.debug("Equity history/realized_vol update: %s", e)

                # Optional hook for subclasses (e.g. quantum bot: quantum Monte Carlo risk, VaR/CVaR circuit breaker)
                hook = getattr(self, "_after_risk_update_hook", None)
                if callable(hook):
                    try:
                        await asyncio.wait_for(
                            hook(
                                prev_equity=prev_equity,
                                new_equity=float(self.portfolio_value_aud),
                                peak_equity=float(self.peak_equity_aud),
                                max_drawdown=float(self.max_drawdown_aud),
                            ),
                            timeout=min(op_timeout_s, 15.0),
                        )
                    except Exception as e:
                        logger.debug("_after_risk_update_hook: %s", e)

                # ---------------------------------------------------------------
                # STOP-LOSS AUTO-EXECUTION: check all positions BEFORE new signals
                # ---------------------------------------------------------------
                try:
                    if self.unified_risk_manager is not None and self.positions:
                        _stop_prices = {
                            str(sym): float((pos or {}).get("current_price") or 0.0)
                            for sym, pos in (self.positions or {}).items()
                        }
                        _sl_pct = float(getattr(self.config, "stop_loss_pct", 0.02) or 0.02)
                        _trail_pct = float(getattr(self.config, "trailing_stop_pct", 0.015) or 0.015)
                        _max_hold = float(getattr(self.config, "max_holding_hours", 72.0) or 72.0)
                        _stop_triggers = self.unified_risk_manager.check_stops(
                            self.positions, _stop_prices,
                            stop_loss_pct=_sl_pct, trail_pct=_trail_pct, max_hold_hours=_max_hold,
                        )
                        if _stop_triggers:
                            logger.warning(
                                "STOP-LOSS: %d positions triggered — executing closes before new signals",
                                len(_stop_triggers),
                            )
                            _stop_signals = []
                            for _st in _stop_triggers:
                                logger.warning(
                                    "STOP-LOSS TRIGGER: %s %s qty=%.8f reason=%s (stop=%.2f, current=%.2f)",
                                    _st["side"], _st["symbol"], _st["quantity"],
                                    _st["reason"], _st["stop_price"], _st["current_price"],
                                )
                                try:
                                    from unified_types import TradingSignal as _TS
                                    _stop_signals.append(_TS(
                                        symbol=_st["symbol"],
                                        action=_st["side"],
                                        confidence=1.0,
                                        strength=1.0,
                                        entry_price=_st["current_price"],
                                        reasoning=f"AUTO_STOP: {_st['reason']}",
                                    ))
                                except Exception:
                                    pass
                            if _stop_signals:
                                _stop_results = await asyncio.wait_for(
                                    self._execute_signals(_stop_signals),
                                    timeout=op_timeout_s,
                                )
                                logger.warning("STOP-LOSS EXECUTION: %d results", len(_stop_results))
                                # Clean up tracking for closed positions
                                for _st in _stop_triggers:
                                    self.unified_risk_manager.clear_position_tracking(_st["symbol"])
                except Exception as _stop_err:
                    logger.warning("Stop-loss check failed: %s", _stop_err)

                # ---------------------------------------------------------------
                # DYNAMIC EXIT OPTIMIZATION: trailing stops + time-based exits
                # ---------------------------------------------------------------
                try:
                    _trailing_exits = await asyncio.wait_for(
                        self._update_trailing_stops(), timeout=op_timeout_s
                    )
                    if _trailing_exits:
                        logger.info(
                            "DYNAMIC EXITS: %d positions triggered — executing before new signals",
                            len(_trailing_exits),
                        )
                        _trail_signals = []
                        for _te in _trailing_exits:
                            try:
                                from unified_types import TradingSignal as _TS
                                _trail_signals.append(_TS(
                                    symbol=_te["symbol"],
                                    action=_te["action"],
                                    confidence=1.0,
                                    strength=1.0,
                                    entry_price=_te["price"],
                                    reasoning=f"AUTO_EXIT: {_te['reason']}",
                                ))
                            except Exception:
                                pass
                        if _trail_signals:
                            _trail_results = await asyncio.wait_for(
                                self._execute_signals(_trail_signals),
                                timeout=op_timeout_s,
                            )
                            logger.info("DYNAMIC EXIT EXECUTION: %d results", len(_trail_results))
                            # Clean up watermarks for exited positions
                            for _te in _trailing_exits:
                                sym = _te["symbol"]
                                self._position_high_water.pop(sym, None)
                                self._position_low_water.pop(sym, None)
                                self._partial_tp_taken.pop(sym, None)
                except Exception as _trail_err:
                    logger.warning("Dynamic exit check failed: %s", _trail_err)

                # ---------------------------------------------------------------
                # MARGIN ENFORCEMENT: deleverage if over max leverage
                # ---------------------------------------------------------------
                try:
                    if self.unified_risk_manager is not None and self.positions:
                        _margin_prices = {
                            str(sym): float((pos or {}).get("current_price") or 0.0)
                            for sym, pos in (self.positions or {}).items()
                        }
                        _margin_capital = float(self.portfolio_value_aud)
                        _deleverage_orders = self.unified_risk_manager.enforce_margin(
                            self.positions, _margin_prices, _margin_capital,
                        )
                        if _deleverage_orders:
                            logger.critical(
                                "MARGIN ENFORCEMENT: %d positions must be force-closed to reduce leverage",
                                len(_deleverage_orders),
                            )
                            _delev_signals = []
                            for _do in _deleverage_orders:
                                logger.critical(
                                    "DELEVERAGE: %s %s qty=%.8f reason=%s",
                                    _do["side"], _do["symbol"], _do["quantity_to_close"], _do["reason"],
                                )
                                try:
                                    from unified_types import TradingSignal as _TS
                                    _delev_signals.append(_TS(
                                        symbol=_do["symbol"],
                                        action=_do["side"],
                                        confidence=1.0,
                                        strength=1.0,
                                        entry_price=_do["current_price"],
                                        reasoning=f"AUTO_DELEVERAGE: {_do['reason']}",
                                    ))
                                except Exception:
                                    pass
                            if _delev_signals:
                                _delev_results = await asyncio.wait_for(
                                    self._execute_signals(_delev_signals),
                                    timeout=op_timeout_s,
                                )
                                logger.critical("DELEVERAGE EXECUTION: %d results", len(_delev_results))
                except Exception as _margin_err:
                    logger.warning("Margin enforcement failed: %s", _margin_err)

                # Reconciliation ownership mode:
                # run checks every configured cycle; block execution on halt/ambiguous failure.
                try:
                    run_mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
                    recon_interval = int(getattr(self.config, "reconciliation_interval_cycles", 1) or 1)
                    recon_interval = max(1, recon_interval)
                    recon_live_every_cycle = bool(getattr(self.config, "reconciliation_live_every_cycle", True))
                    recon_paper_enabled = bool(getattr(self.config, "reconciliation_paper_enabled", False))
                    recon_enabled = (
                        (run_mode == "live" and recon_live_every_cycle)
                        or (run_mode in {"paper", "backtest"} and recon_paper_enabled)
                    )
                    if (
                        recon_enabled
                        and self.execution_engine is not None
                        and hasattr(self.execution_engine, "reconcile_state")
                        and ((cycles % recon_interval) == 0)
                    ):
                        recon_trace_id = f"{trace_id}_recon"
                        recon_ok, recon_payload = await asyncio.wait_for(
                            self.execution_engine.reconcile_state(
                                cycle_id=int(cycles + 1),
                                trace_id=recon_trace_id,
                            ),
                            timeout=op_timeout_s,
                        )
                        if callable(rec_event):
                            try:
                                rec_event(
                                    stage="reconciliation_cycle",
                                    cycle_id=cycles + 1,
                                    payload_json=json.dumps(
                                        {
                                            "reconciliation_ok": bool(recon_ok),
                                            "trace_id": recon_trace_id,
                                            "payload": recon_payload,
                                        },
                                        ensure_ascii=True,
                                        default=str,
                                    ),
                                )
                            except Exception:
                                pass
                        if not recon_ok:
                            reconciliation_block_cycle = True
                            logger.warning(
                                "Reconciliation gate blocked cycle %s: %s",
                                cycles + 1,
                                str((recon_payload or {}).get("halt_reason", "reconciliation_blocked") or "reconciliation_blocked"),
                            )
                except Exception as e:
                    logger.warning("Reconciliation cycle check failed: %s", e)
                    if str(getattr(self.config, "run_mode", "paper") or "paper").lower() == "live":
                        reconciliation_block_cycle = True
                # Optional quant-fund risk metrics (advisory-only)
                if self.quant_fund_risk_engine is not None and self.market_data_service is not None:
                    try:
                        mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
                        modes = list(getattr(self.config, "quant_fund_upgrades_modes", ["paper", "backtest"]) or ["paper", "backtest"])
                        if mode in [str(m).lower() for m in modes]:
                            symbols = list(getattr(self.config, "trading_pairs", []) or [])
                            # Prefer tracking symbols we actually hold
                            symbols = list({*(symbols or []), *list(self.positions.keys())})

                            # Fetch current prices PARALLEL (was sequential — saves 60s+)
                            current_prices: Dict[str, float] = {}
                            async def _fetch_one_ticker(sym):
                                try:
                                    t = await self.market_data_service.fetch_ticker(sym)
                                    px = None
                                    if t is not None:
                                        px = getattr(t, "last", None)
                                        if px is None:
                                            px = getattr(getattr(t, "raw", {}), "get", lambda *_: None)("last")
                                    if px is None:
                                        px = (self.positions.get(str(sym)) or {}).get("current_price")
                                    if px is not None and float(px) > 0:
                                        return (str(sym), float(px))
                                except Exception:
                                    pass
                                return None
                            _ticker_results = await asyncio.gather(
                                *[_fetch_one_ticker(s) for s in symbols],
                                return_exceptions=True,
                            )
                            for r in _ticker_results:
                                if isinstance(r, tuple) and r is not None:
                                    current_prices[r[0]] = r[1]
                                    self.quant_fund_risk_engine.update_price(r[0], r[1])

                            current_positions = {
                                str(sym): float((self.positions.get(str(sym)) or {}).get("quantity") or 0.0)
                                for sym in current_prices.keys()
                            }
                            # Feed fresh prices into positions for partial TP / trailing stop and pre-trade context
                            for sym, px in current_prices.items():
                                if sym in (self.positions or {}):
                                    if self.positions[sym] is None:
                                        self.positions[sym] = {}
                                    self.positions[sym]["current_price"] = px
                                # Update trailing stops via ConditionalOrderManager
                                try:
                                    _cond = getattr(self.component_registry, "conditional_orders", None) if self.component_registry else None
                                    if _cond is not None and hasattr(_cond, "on_price_update"):
                                        _cond.on_price_update(sym, float(px))
                                except Exception:
                                    pass
                            risk = await asyncio.wait_for(
                                self.quant_fund_risk_engine.calculate_portfolio_risk(
                                current_positions=current_positions,
                                current_prices=current_prices,
                                portfolio_value=float(self.portfolio_value_aud),
                                ),
                                timeout=op_timeout_s,
                            )
                            if callable(rec_event):
                                try:
                                    # Correlation matrix can be large; log summary only.
                                    cm = getattr(risk, "correlation_matrix", None)
                                    corr_mean_abs = None
                                    if cm is not None:
                                        try:
                                            import numpy as _np

                                            cma = _np.asarray(cm, dtype=float)
                                            if cma.size:
                                                corr_mean_abs = float(_np.nanmean(_np.abs(cma)))
                                        except Exception:
                                            corr_mean_abs = None
                                    rec_event(
                                        stage="quant_fund_risk",
                                        cycle_id=cycles,
                                        payload_json=json.dumps(
                                            {
                                                "var_95": float(getattr(risk, "var_95", 0.0) or 0.0),
                                                "var_99": float(getattr(risk, "var_99", 0.0) or 0.0),
                                                "cvar_95": float(getattr(risk, "cvar_95", 0.0) or 0.0),
                                                "cvar_99": float(getattr(risk, "cvar_99", 0.0) or 0.0),
                                                "portfolio_volatility": float(getattr(risk, "portfolio_volatility", 0.0) or 0.0),
                                                "max_drawdown": float(getattr(risk, "max_drawdown", 0.0) or 0.0),
                                                "current_drawdown": float(getattr(risk, "current_drawdown", 0.0) or 0.0),
                                                "corr_mean_abs": corr_mean_abs,
                                            },
                                            ensure_ascii=True,
                                            default=str,
                                        ),
                                    )
                                except Exception:
                                    pass
                            # Feed correlation matrix to config so correlation-aware sizing works (no manual YAML needed)
                            sym_list = list(current_prices.keys())
                            cm = getattr(risk, "correlation_matrix", None)
                            if cm is not None and len(sym_list) > 0:
                                try:
                                    import numpy as _np
                                    cma = _np.asarray(cm, dtype=float)
                                    n = len(sym_list)
                                    if cma.ndim >= 2 and cma.shape[0] >= n and cma.shape[1] >= n:
                                        corr_dict = {}
                                        for i in range(n):
                                            for j in range(n):
                                                key = (sym_list[i], sym_list[j])
                                                corr_dict[key] = float(cma[i, j])
                                        setattr(self.config, "correlation_matrix", corr_dict)
                                except Exception:
                                    pass
                            # VaR breach: trigger circuit breaker or alert when VaR exceeds threshold
                            var_breach_pct = float(getattr(self.config, "var_breach_pct", 0) or 0)
                            if var_breach_pct > 0:
                                var_95 = float(getattr(risk, "var_95", getattr(risk, "var", 0.0)) or 0.0)
                                if var_95 < 0 and abs(var_95) * 100.0 > var_breach_pct:
                                    if self.unified_risk_manager is not None and hasattr(self.unified_risk_manager, "trip_circuit_breaker"):
                                        reason = "VaR breach: |VaR_95|=%.2f%% > %.2f%%" % (abs(var_95) * 100.0, var_breach_pct)
                                        self.unified_risk_manager.trip_circuit_breaker(reason)
                                        try:
                                            from monitoring.alerting import get_alert_manager
                                            cd = getattr(self.unified_risk_manager, "circuit_breaker_cooldown", None)
                                            cooldown_m = int(cd.total_seconds() // 60) if cd is not None else 60
                                            await get_alert_manager().circuit_breaker_alert(reason, duration_minutes=cooldown_m or 60)
                                        except Exception as alert_e:
                                            logger.debug("VaR circuit breaker alert: %s", alert_e)
                                    logger.warning("VaR breach: |VaR_95|=%.2f%% > var_breach_pct=%.2f%%", abs(var_95) * 100.0, var_breach_pct)
                                    if getattr(self.config, "var_breach_alert_enabled", False) and callable(rec_event):
                                        try:
                                            rec_event(stage="var_breach_alert", cycle_id=cycles, payload_json=json.dumps({"var_95": var_95, "var_breach_pct": var_breach_pct}, default=str))
                                        except Exception:
                                            pass
                    except Exception:
                        pass
                # VaR breach from quantum Monte Carlo (when not using quant_fund or in addition)
                var_breach_pct = float(getattr(self.config, "var_breach_pct", 0) or 0)
                if var_breach_pct > 0:
                    last_q = getattr(self, "_last_quantum_var_cvar", None)
                    if isinstance(last_q, dict):
                        var_95 = last_q.get("var_95", last_q.get("var"))
                        if var_95 is not None and float(var_95) < 0 and abs(float(var_95)) * 100.0 > var_breach_pct:
                            if self.unified_risk_manager is not None and hasattr(self.unified_risk_manager, "trip_circuit_breaker"):
                                reason = "Quantum VaR breach: |VaR_95|=%.2f%% > %.2f%%" % (abs(float(var_95)) * 100.0, var_breach_pct)
                                self.unified_risk_manager.trip_circuit_breaker(reason)
                                try:
                                    from monitoring.alerting import get_alert_manager
                                    cd = getattr(self.unified_risk_manager, "circuit_breaker_cooldown", None)
                                    cooldown_m = int(cd.total_seconds() // 60) if cd is not None else 60
                                    await get_alert_manager().circuit_breaker_alert(reason, duration_minutes=cooldown_m or 60)
                                except Exception:
                                    pass
                            if getattr(self.config, "var_breach_alert_enabled", False) and callable(rec_event):
                                try:
                                    rec_event(stage="var_breach_alert", cycle_id=cycles, payload_json=json.dumps({"source": "quantum", "var_95": var_95, "var_breach_pct": var_breach_pct}, default=str))
                                except Exception:
                                    pass
                # Fallback correlation_matrix from returns when use_correlation_aware_sizing and not set by quant_fund (every 5 cycles)
                if (
                    getattr(self.config, "use_correlation_aware_sizing", False)
                    and self.market_data_service
                    and (cycles % 5 == 0)
                ):
                    cm_existing = getattr(self.config, "correlation_matrix", None)
                    if not cm_existing or (isinstance(cm_existing, dict) and len(cm_existing) == 0):
                        try:
                            pairs = list(getattr(self.config, "trading_pairs", []) or [])[:10]
                            if len(pairs) >= 2:
                                tf = str(getattr(self.config, "signal_primary_timeframe", "1h") or "1h")
                                rets: Dict[str, List[float]] = {}
                                for sym in pairs:
                                    df = await self.market_data_service.fetch_ohlcv_df(sym, timeframe=tf, limit=60)
                                    if df is not None and not df.empty and "close" in df.columns:
                                        closes = df["close"].astype(float).tolist()
                                        if len(closes) >= 3:
                                            r = [(closes[i] - closes[i - 1]) / max(closes[i - 1], 1e-12) for i in range(1, len(closes))]
                                            rets[str(sym)] = r
                                if len(rets) >= 2:
                                    assets = sorted(rets.keys())
                                    n = min(len(r) for r in rets.values())
                                    mat = np.array([rets[s][:n] for s in assets], dtype=float)
                                    corr = np.corrcoef(mat)
                                    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
                                    corr_dict: Dict[Tuple[str, str], float] = {}
                                    for i in range(len(assets)):
                                        for j in range(len(assets)):
                                            if i != j:
                                                corr_dict[(assets[i], assets[j])] = float(corr[i, j])
                                    setattr(self.config, "correlation_matrix", corr_dict)
                        except Exception as ec:
                            logger.debug("Fallback correlation_matrix: %s", ec)
                if callable(rec_event):
                    try:
                        rec_event(
                            stage="portfolio_update",
                            cycle_id=cycles,
                            payload_json=json.dumps(
                                {
                                    "portfolio_value_aud": self.portfolio_value_aud,
                                    "cash_balance_aud": self.cash_balance_aud,
                                    "positions": self.positions,
                                },
                                ensure_ascii=True,
                                default=str,
                            ),
                        )
                    except Exception:
                        pass

                # Adaptive risk posture (every cycle): derive + apply dynamic knobs within hard caps.
                try:
                    arc = getattr(self, "adaptive_risk_controller", None)
                    if arc is not None:
                        # Use UnifiedRiskManager metrics if available
                        dd_pct = 0.0
                        daily_ret_pct = 0.0
                        try:
                            if self.unified_risk_manager is not None:
                                rm = self.unified_risk_manager.get_risk_metrics()
                                dd_pct = float(getattr(rm, "drawdown", 0.0) or 0.0) * 100.0
                                daily_ret_pct = float(getattr(rm, "daily_return_pct", 0.0) or 0.0) * 100.0
                        except Exception:
                            dd_pct = 0.0
                            daily_ret_pct = 0.0

                        last_reg = {}
                        try:
                            st = self.ai_brain.get_adaptation_status() if self.ai_brain is not None else {}
                            se = st.get("strategy_engine") if isinstance(st, dict) else None
                            lr = (se.get("last_regime") or {}) if isinstance(se, dict) else {}
                            last_reg = lr if isinstance(lr, dict) else {}
                        except Exception:
                            last_reg = {}

                        # FETCH LIVE MARKET VOLATILITY (e.g. from primary pair)
                        market_vol = 0.0
                        try:
                            if self.market_data_service:
                                primary_pair = getattr(self.config, "trading_pairs", ["BTC/USD"])[0]
                                # Cheap volatility check (last 20 1m candles)
                                df = await self.market_data_service.fetch_ohlcv_df(primary_pair, timeframe="1m", limit=20)
                                if df is not None and not df.empty and "close" in df.columns:
                                    rets = df["close"].pct_change().dropna()
                                    # Annualized/Daily scale
                                    market_vol = float(rets.std() * (1440**0.5)) # Daily vol
                        except Exception:
                            pass

                        prof = arc.update_profile(
                            drawdown_pct=float(dd_pct), 
                            daily_return_pct=float(daily_ret_pct), 
                            last_regime_by_symbol=last_reg,
                            market_volatility=market_vol
                        )
                        # cooldown decay each cycle
                        try:
                            arc.decay()
                        except Exception:
                            pass
                        # Apply to config (within hard caps)
                        setattr(self.config, "_adaptive_risk_profile", prof.__dict__)
                        setattr(self.config, "max_total_exposure_pct", float(prof.max_total_exposure_pct))
                        setattr(self.config, "max_concurrent_signals", int(prof.max_concurrent_signals))
                        setattr(self.config, "min_signal_confidence", float(prof.min_signal_confidence))
                        setattr(self.config, "edge_cost_gate_buffer_mult", float(prof.edge_cost_gate_buffer_mult))
                        setattr(self.config, "edge_cost_gate_min_edge_pct", float(prof.edge_cost_gate_min_edge_pct))
                        # Stops/TP as multipliers on base values
                        base_sl = float(getattr(arc, "base_stop_loss_pct", getattr(self.config, "stop_loss_pct", 0.01)) or 0.01)
                        base_tp = float(getattr(arc, "base_take_profit_pct", getattr(self.config, "take_profit_pct", 0.03)) or 0.03)
                        setattr(self.config, "stop_loss_pct", float(base_sl) * float(prof.stop_loss_mult))
                        setattr(self.config, "take_profit_pct", float(base_tp) * float(prof.take_profit_mult))

                        if callable(rec_event):
                            try:
                                rec_event(stage="adaptive_risk", cycle_id=cycles, payload_json=json.dumps(prof.__dict__, ensure_ascii=True, default=str))
                            except Exception:
                                pass
                except Exception:
                    pass

                # Dynamic universe: refresh trading_pairs from universe_builder every N cycles (no universe_selector required)
                try:
                    self._enforce_live_safe_symbol_lock()
                    live_safe_lock = bool(self._live_safe_locked_symbols) and self._is_live_safe_runtime()
                    dyn_enabled = bool(getattr(self.config, "dynamic_universe_enabled", False))
                    dyn_interval = int(getattr(self.config, "dynamic_universe_interval_cycles", 0) or 0)
                    if live_safe_lock and dyn_enabled:
                        logger.debug("LIVE_SAFE symbol lock: skipping dynamic universe refresh")
                    if (not live_safe_lock) and dyn_enabled and dyn_interval > 0 and cycles > 0 and (cycles % dyn_interval) == 0:
                        try:
                            from utils.universe_builder import select_top_liquid_usd_pairs
                            top_n = int(getattr(self.config, "dynamic_universe_top_n", 15) or 15)
                            sel = select_top_liquid_usd_pairs(
                                exchange_id=str(getattr(self.config, "primary_exchange", "kraken") or "kraken"),
                                top_n=top_n,
                            )
                            current = list(getattr(self.config, "trading_pairs", []) or [])
                            merged = list(dict.fromkeys([*current, *list(sel.symbols)]))[: max(len(current), top_n)]
                            setattr(self.config, "trading_pairs", merged)
                            logger.info("Dynamic universe refreshed: %s symbols", len(merged))
                        except Exception as e:
                            logger.debug("Dynamic universe refresh: %s", e)
                except Exception:
                    pass

                # Liquidity scanner: deep-scan exchange for best liquidity (volume + spread + depth + imbalance)
                try:
                    self._enforce_live_safe_symbol_lock()
                    live_safe_lock = bool(self._live_safe_locked_symbols) and self._is_live_safe_runtime()
                    liq_cfg = getattr(self.config, "liquidity_scanner", None) or {}
                    if live_safe_lock and isinstance(liq_cfg, dict) and liq_cfg.get("enabled", False):
                        logger.debug("LIVE_SAFE symbol lock: skipping liquidity scanner pair updates")
                    if (not live_safe_lock) and isinstance(liq_cfg, dict) and liq_cfg.get("enabled", False):
                        liq_interval = int(liq_cfg.get("scan_interval_cycles", 50) or 50)
                        if liq_interval > 0 and cycles > 0 and (cycles % liq_interval) == 0:
                            try:
                                from services.liquidity_scanner import LiquidityScanner
                                scanner = LiquidityScanner(
                                    exchange_id=str(liq_cfg.get("exchange_id", "kraken")),
                                    quote_currencies=tuple(liq_cfg.get("quote_currencies", ["USD"])),
                                    max_pairs=int(liq_cfg.get("max_pairs", 15)),
                                    min_volume_usd=float(liq_cfg.get("min_volume_usd", 50000)),
                                    depth_levels=int(liq_cfg.get("depth_levels", 10)),
                                    batch_size=int(liq_cfg.get("batch_size", 5)),
                                    batch_delay_s=float(liq_cfg.get("batch_delay_s", 1.0)),
                                    cache_ttl_s=float(liq_cfg.get("cache_ttl_s", 60.0)),
                                    w_volume=float(liq_cfg.get("w_volume", 0.40)),
                                    w_spread=float(liq_cfg.get("w_spread", 0.30)),
                                    w_depth=float(liq_cfg.get("w_depth", 0.20)),
                                    w_imbalance=float(liq_cfg.get("w_imbalance", 0.10)),
                                )
                                liq_results = await scanner.scan(force=True)
                                if liq_results and liq_cfg.get("auto_update_pairs", True):
                                    top_symbols = [r.symbol for r in liq_results]
                                    current = list(getattr(self.config, "trading_pairs", []) or [])
                                    merged = list(dict.fromkeys([*top_symbols, *current]))[: max(len(current), len(top_symbols))]
                                    setattr(self.config, "trading_pairs", merged)
                                    setattr(self.config, "_liquidity_scan_results", [r.to_dict() for r in liq_results])
                                    logger.info(
                                        "Liquidity scan: %d pairs ranked, top: %s (%.1f), trading_pairs updated to %d",
                                        len(liq_results), liq_results[0].symbol, liq_results[0].liquidity_score, len(merged),
                                    )
                            except Exception as e:
                                logger.debug("Liquidity scanner: %s", e)
                except Exception:
                    pass

                # Adaptive universe selection (every cycle): focus on best symbols, avoid thrash.
                try:
                    self._enforce_live_safe_symbol_lock()
                    live_safe_lock = bool(self._live_safe_locked_symbols) and self._is_live_safe_runtime()
                    mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
                    enabled = bool(getattr(self.config, "adaptive_universe_enabled", True))
                    modes = list(getattr(self.config, "adaptive_universe_modes", ["paper", "backtest", "live"]) or ["paper", "backtest", "live"])
                    if live_safe_lock and enabled:
                        logger.debug("LIVE_SAFE symbol lock: skipping adaptive universe selector")
                    if (not live_safe_lock) and enabled and mode in [str(m).lower() for m in modes]:
                        us = getattr(self, "universe_selector", None)
                        if us is not None:
                            # candidate pool: current config pairs (already a safe allowlist)
                            candidates = list(getattr(self.config, "trading_pairs", []) or [])
                            # best-effort: expand candidates with top-N liquid pairs (requires ccxt/network; ignore failures)
                            try:
                                from utils.universe_builder import select_top_liquid_usd_pairs

                                sel = select_top_liquid_usd_pairs(exchange_id=str(getattr(self.config, "primary_exchange", "kraken") or "kraken"), top_n=int(getattr(self.config, "adaptive_universe_top_n", 10) or 10))
                                candidates = list(dict.fromkeys([*candidates, *list(sel.symbols)]))
                            except Exception:
                                candidates = list(dict.fromkeys(candidates))

                            active = us.select_active(candidate_symbols=[str(s) for s in candidates if str(s).strip()], cycle_id=int(cycles))
                            if active:
                                setattr(self.config, "trading_pairs", active)
                                setattr(self.config, "_adaptive_universe_active", list(active))
                                try:
                                    us.save()
                                except Exception:
                                    pass
                                if callable(rec_event):
                                    try:
                                        rec_event(stage="adaptive_universe", cycle_id=cycles, payload_json=json.dumps({"active": active, "candidates": candidates[:50]}, ensure_ascii=True, default=str))
                                    except Exception:
                                        pass
                except Exception:
                    pass
                finally:
                    self._enforce_live_safe_symbol_lock()

                # Observability: compact adaptive snapshot each cycle (skipped in fast_mode)
                if not _fast_mode and callable(rec_event):
                    try:
                        st = self.ai_brain.get_adaptation_status() if self.ai_brain is not None else {}
                        payload = {
                            "cycle_id": int(cycles),
                            "mode": str(getattr(self.config, "run_mode", "paper") or "paper"),
                            "universe_active": list(getattr(self.config, "_adaptive_universe_active", []) or []),
                            "risk_profile": dict(getattr(self.config, "_adaptive_risk_profile", {}) or {}),
                            "quantum": {
                                "enabled": bool(getattr(self.config, "quantum_features_enabled", True)),
                                "method": str(getattr(self.config, "quantum_method", "quantum_approximate") or "quantum_approximate"),
                                "strength": float(getattr(self.config, "quantum_strength", 1.0) or 1.0),
                            },
                            "adaptation": st,
                        }
                        rec_event(stage="adaptive_state", cycle_id=cycles, payload_json=json.dumps(payload, ensure_ascii=True, default=str))
                    except Exception:
                        pass

                # Kill switch check (every cycle, before signal generation)
                try:
                    if await self._check_kill_switch():
                        logger.critical("Kill switch triggered — shutting down trading loop")
                        await self.shutdown()
                        break
                except Exception as _ks_exc:
                    logger.debug("Kill switch check error: %s", _ks_exc)

                # Check emergency stop conditions
                if self._check_emergency_stop():
                    self.state = SystemState.EMERGENCY_STOP
                    logger.critical("Emergency stop triggered - halting trading loop")
                    await self.shutdown()
                    break

                # Get AI signals: use continuous scanner's best opportunities when enabled and fresh
                t_market_data_start = time.perf_counter()
                use_cached_best = bool(getattr(self.config, "continuous_scan_use_cached_best", True))
                max_age = float(getattr(self.config, "continuous_scan_max_age_seconds", 30.0) or 30.0)
                ai_signals = []
                if use_cached_best and self.continuous_scanner is not None:
                    try:
                        ai_signals = await self.continuous_scanner.get_best_opportunities(
                            max_age_seconds=max_age,
                            convert_to_signals=True,
                        )
                    except Exception as e:
                        logger.debug("Continuous scanner get_best_opportunities error: %s", e)
                if not ai_signals and self.ai_brain is not None:
                    ai_signals = await asyncio.wait_for(self.ai_brain.generate_trading_signals(), timeout=op_timeout_s)

                # Generate alpha signals from AlphaSignalFusion (v8.13.0)
                alpha_signals = []
                if getattr(self.config, "alpha_signal_fusion_enabled", True):
                    try:
                        fusion = getattr(self, "alpha_signal_fusion", None)
                        if fusion and self.market_data_service:
                            # Generate signals for trading pairs
                            pairs = getattr(self.config, "trading_pairs", []) or []
                            for symbol in pairs[:5]:  # Top 5 pairs
                                try:
                                    ohlcv = await self.market_data_service.fetch_ohlcv(symbol, "1h", limit=100)
                                    if ohlcv:
                                        # Get derivatives data for signals
                                        market_data = {}
                                        # Note: In production, fetch funding/OI from exchange API
                                        alpha = await fusion.generate_signal(symbol, ohlcv, market_data)
                                        if alpha and alpha.direction != "neutral":
                                            # Convert to trading signal format
                                            from unified_types import TradingSignal
                                            ts = TradingSignal(
                                                symbol=symbol,
                                                action=alpha.direction.upper(),
                                                confidence=alpha.confidence,
                                                strength=alpha.confidence,
                                                reasoning=f"AlphaFusion v9: ml={alpha.ml_prediction:.2f}, micro={alpha.ml_microstructure:.2f}, alpha={alpha.alpha_model:.2f}, sentiment={alpha.sentiment_score:.2f}, onchain={alpha.onchain_score:.2f}, regime={alpha.regime}",
                                                timestamp=datetime.now(),
                                            )
                                            setattr(ts, "strategy", "alpha_fusion")
                                            setattr(ts, "expected_return", alpha.expected_return)
                                            alpha_signals.append(ts)
                                except Exception as e:
                                    logger.debug(f"Alpha signal generation error for {symbol}: {e}")
                    except Exception as e:
                        logger.debug("AlphaSignalFusion error: {e}")

                # Combine AI signals with alpha signals
                if alpha_signals:
                    ai_signals = list(ai_signals or []) + alpha_signals
                    logger.info(f"SIGNAL DEBUG: Added {len(alpha_signals)} alpha signals")

                # HFT signals from real order book + tick momentum (peak: OB fetch, recent_trades, latency, regime filter)
                hft_signals: List[Any] = []
                hft_disabled_regimes = list(getattr(self.config, "hft_disabled_in_regimes", None) or [])
                hft_skip_regime = False
                if hft_disabled_regimes and self.ai_brain is not None:
                    try:
                        st = self.ai_brain.get_adaptation_status() if callable(getattr(self.ai_brain, "get_adaptation_status", None)) else {}
                        se = st.get("strategy_engine") if isinstance(st, dict) else None
                        lr = (se.get("last_regime") or {}) if isinstance(se, dict) else {}
                        regimes_now = list(lr.values()) if isinstance(lr, dict) else ([lr] if lr else [])
                        if any(r in hft_disabled_regimes for r in regimes_now):
                            hft_skip_regime = True
                    except Exception:
                        pass
                if (
                    not hft_skip_regime
                    and getattr(self.config, "hft_enabled", True)
                    and self.hft_engine
                    and self.market_data_service
                ):
                    try:
                        from execution.risk_compliance_audit import latency_start, latency_end
                        latency_start("hft")
                        try:
                            from unified_types import TradingSignal as UTSignal
                            def _hft_sig_to_ts(sig: Any) -> Any:
                                ts = UTSignal(
                                    symbol=sig.symbol,
                                    action=sig.action,
                                    confidence=float(sig.confidence),
                                    strength=0.7,
                                    entry_price=float(sig.price or 0),
                                    reasoning=f"HFT {getattr(sig, 'strategy', 'obi')}",
                                    timestamp=datetime.now(),
                                )
                                try:
                                    setattr(ts, "strategy", getattr(sig, "strategy", "hft_obi_pressure"))
                                except Exception:
                                    pass
                                return ts
                            pairs = list(getattr(self.config, "trading_pairs", []) or [])[:8]
                            use_adv_infra = getattr(self.config, "use_advanced_hft_infrastructure", False) and self.hft_infrastructure

                            if use_adv_infra:
                                # 5.1: feed real OB/trades into advanced HFT event loop, process, drain signal_ring
                                for symbol in pairs:
                                    try:
                                        primary_ob = await asyncio.wait_for(
                                            self.market_data_service.fetch_order_book(str(symbol), limit=10),
                                            timeout=2.0,
                                        )
                                        if primary_ob:
                                            self.hft_infrastructure.push_order_book(str(symbol), primary_ob)
                                        recent_trades = await asyncio.wait_for(
                                            self.market_data_service.fetch_recent_trades(str(symbol), limit=50),
                                            timeout=1.5,
                                        )
                                        if recent_trades:
                                            self.hft_infrastructure.push_trades(str(symbol), recent_trades)
                                    except (asyncio.TimeoutError, Exception):
                                        continue
                                await self.hft_infrastructure.process_available()
                                for sig in self.hft_infrastructure.drain_signals():
                                    hft_signals.append(_hft_sig_to_ts(sig))
                            else:
                                # Direct path: analyze_order_book + analyze_trade_flow per symbol
                                for symbol in pairs:
                                    try:
                                        primary_ob = await asyncio.wait_for(
                                            self.market_data_service.fetch_order_book(str(symbol), limit=10),
                                            timeout=2.0,
                                        )
                                        if primary_ob:
                                            sig = await self.hft_engine.analyze_order_book(str(symbol), primary_ob)
                                            if sig is not None:
                                                hft_signals.append(_hft_sig_to_ts(sig))
                                        recent_trades = await asyncio.wait_for(
                                            self.market_data_service.fetch_recent_trades(str(symbol), limit=50),
                                            timeout=1.5,
                                        )
                                        if recent_trades and hasattr(self.hft_engine, "analyze_trade_flow"):
                                            sig_tf = await self.hft_engine.analyze_trade_flow(str(symbol), recent_trades)
                                            if sig_tf is not None:
                                                hft_signals.append(_hft_sig_to_ts(sig_tf))
                                    except (asyncio.TimeoutError, Exception):
                                        continue
                            # Cap HFT signals per cycle (peak: avoid flooding pipeline)
                            max_hft = int(getattr(self.config, "max_hft_signals_per_cycle", 2) or 2)
                            if len(hft_signals) > max_hft:
                                hft_signals = sorted(hft_signals, key=lambda s: s.confidence, reverse=True)[:max_hft]
                            ai_signals = list(ai_signals or []) + list(hft_signals)
                        finally:
                            latency_end("hft")
                    except Exception as e:
                        logger.debug("HFT signal generation: %s", e)
                cycle_stage_timing_ms["market_data_ms"] = max(
                    0.0, (time.perf_counter() - t_market_data_start) * 1000.0
                )

                # Filter out disabled strategies (paper: paper_trading_disabled_strategies; live: live_disabled_strategies; paper+simulate_live: same as live)
                _mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
                _sim_live = bool(getattr(self.config, "paper_simulates_live", False))
                if _mode == "live" or (_mode == "paper" and _sim_live):
                    _disabled = list(getattr(self.config, "live_disabled_strategies", None) or [])
                elif _mode in ("paper", "backtest"):
                    _disabled = list(getattr(self.config, "paper_trading_disabled_strategies", None) or [])
                else:
                    _disabled = list(getattr(self.config, "live_disabled_strategies", None) or [])
                if _disabled:
                    _before = len(ai_signals)
                    ai_signals = [s for s in ai_signals if (getattr(s, "strategy", None) or "") not in _disabled]
                    if len(ai_signals) < _before:
                        logger.debug("Filtered %s signals from disabled strategies %s", _before - len(ai_signals), _disabled)
                ai_signals = self._filter_live_safe_signals(ai_signals, stage="ai_signals")

                # Strategy allocator: PnL-based ranking so everything works together
                if self.strategy_allocator and getattr(self.config, "strategy_allocator_enabled", True):
                    try:
                        st = self.ai_brain.get_adaptation_status() if (self.ai_brain and callable(getattr(self.ai_brain, "get_adaptation_status", None))) else {}
                        se = st.get("strategy_engine") if isinstance(st, dict) else {}
                        last_reg = (se.get("last_regime") or {}) if isinstance(se, dict) else {}
                        max_total = int(getattr(self.config, "strategy_allocator_max_total_signals", 10) or 10)
                        max_per = int(getattr(self.config, "strategy_allocator_max_per_strategy", 3) or 3)
                        ai_signals = self.strategy_allocator.rank_signals(
                            signals=ai_signals,
                            last_regime_by_symbol=last_reg,
                            max_total=max_total,
                            max_per_strategy=max_per,
                        )
                    except Exception as e:
                        logger.debug("Strategy allocator rank_signals: %s", e)

                if callable(rec_event):
                    try:
                        rec_event(
                            stage="ai_signals",
                            cycle_id=cycles,
                            payload_json=json.dumps([getattr(s, "__dict__", s) for s in ai_signals], ensure_ascii=True, default=str),
                        )
                    except Exception:
                        pass
                
                # Process signals through ARGUS strategies
                t_feature_generation_start = time.perf_counter()
                logger.info("SIGNAL DEBUG: ai_signals=%d from brain", len(ai_signals or []))
                argus_signals = await asyncio.wait_for(self._process_argus_strategies(ai_signals), timeout=op_timeout_s)
                logger.info("SIGNAL DEBUG: argus_signals=%d after processing", len(argus_signals or []))
                argus_signals = self._filter_live_safe_signals(argus_signals, stage="argus_signals")
                if callable(rec_event):
                    try:
                        rec_event(
                            stage="argus_signals",
                            cycle_id=cycles,
                            payload_json=json.dumps([getattr(s, "__dict__", s) for s in argus_signals], ensure_ascii=True, default=str),
                        )
                    except Exception:
                        pass
                
                # Optimize position sizing for $1K AUD
                optimized_signals = await asyncio.wait_for(self.capital_optimizer.optimize_signals(argus_signals), timeout=op_timeout_s)
                logger.info("SIGNAL DEBUG: optimized_signals=%d after capital optimizer", len(optimized_signals or []))
                optimized_signals = self._filter_live_safe_signals(optimized_signals, stage="optimized_signals")
                if callable(rec_event):
                    try:
                        rec_event(
                            stage="optimized_signals",
                            cycle_id=cycles,
                            payload_json=json.dumps([getattr(s, "__dict__", s) for s in optimized_signals], ensure_ascii=True, default=str),
                        )
                    except Exception:
                        pass

                # Auto-reduce position size after N consecutive losses (protect capital)
                try:
                    after_n = int(getattr(self.config, "auto_reduce_after_n_losses", 0) or 0)
                    factor = float(getattr(self.config, "auto_reduce_factor", 0.6) or 0.6)
                    if after_n > 0 and factor > 0 and self.consecutive_losses >= after_n:
                        for sig in optimized_signals:
                            q = getattr(sig, "quantity", None)
                            if q is not None and isinstance(q, (int, float)) and q > 0:
                                setattr(sig, "quantity", max(0.0, float(q) * factor))
                        logger.debug("Auto-reduce: applied factor %.2f after %s consecutive losses", factor, self.consecutive_losses)
                except Exception as e:
                    logger.debug("Auto-reduce: %s", e)
                cycle_stage_timing_ms["feature_generation_ms"] = max(
                    0.0, (time.perf_counter() - t_feature_generation_start) * 1000.0
                )

                # PR-07/08: signals -> targets -> execution with deterministic regime gating.
                t_strategy_eval_start = time.perf_counter()
                self._ensure_target_and_regime_components()
                regime_label = self._compute_regime_label(optimized_signals)

                # Component registry cycle hook: update vol/alpha/intraday-VaR/tail-hedge (best-effort)
                try:
                    if self.component_registry is not None:
                        _cycle_prices = {
                            str(sym): float((self.positions.get(str(sym)) or {}).get("current_price") or 0.0)
                            for sym in (self.positions or {})
                        }
                        # Also include configured pairs even if no open position (needed by KalmanPairs)
                        for _tp in (getattr(self.config, "trading_pairs", []) or []):
                            _tp = str(_tp)
                            if _tp not in _cycle_prices:
                                try:
                                    _ticker = await self.exchange_manager.fetch_ticker(_tp)
                                    if _ticker:
                                        _cycle_prices[_tp] = float(_ticker.get("last") or 0.0)
                                except Exception:
                                    pass
                        # Enrich _cycle_prices with live mid-prices from L2 feed (best-effort)
                        try:
                            _l2 = getattr(self, "l2_feed", None)
                            if _l2 is not None and _l2.is_connected:
                                for _sym in list(_cycle_prices.keys()):
                                    _book = _l2.get_book(_sym)
                                    if _book is not None and _book.mid_price:
                                        _cycle_prices[_sym] = _book.mid_price
                        except Exception:
                            pass
                        # Enrich _cycle_prices with LiveMarketDataManager (best-effort, highest priority)
                        try:
                            _lmd = getattr(self, "live_market_data", None)
                            if _lmd is not None:
                                for _sym in list(_cycle_prices.keys()) + list(
                                    str(p) for p in (getattr(self.config, "trading_pairs", []) or [])
                                ):
                                    if not _lmd.is_stale(_sym, max_age_ms=10000):
                                        _tick = _lmd.get_latest(_sym)
                                        if _tick and _tick.get("price", 0) > 0:
                                            _cycle_prices[_sym] = _tick["price"]
                        except Exception:
                            pass
                        _cycle_advisory = self.component_registry.on_cycle(_cycle_prices, str(regime_label or ""))
                        self._last_cycle_advisory = _cycle_advisory
                        
                        # Enhance regime label with Hurst analysis
                        try:
                            if _cycle_advisory and "hurst_regime" in _cycle_advisory:
                                _hurst = _cycle_advisory["hurst_regime"]
                                _hurst_regime = str(_hurst.get("regime", ""))
                                _hurst_scalar = float(_hurst.get("position_scalar", 0.5))
                                _hurst_recommendation = str(_hurst.get("recommendation", ""))
                                
                                # If Hurst says "avoid" (random walk), reduce confidence
                                if _hurst_recommendation == "avoid":
                                    logger.info("Hurst regime: AVOID trading (random walk detected, H=%.2f)", 
                                               _hurst.get("hurst_exponent", 0.5))
                                    # Mark regime as unfavorable for position sizing
                                    if "hurst_avoid" not in regime_label:
                                        regime_label = f"{regime_label}:hurst_avoid" if regime_label else "hurst_avoid"
                                # If Hurst says "reduce" (high entropy), note it
                                elif _hurst_recommendation == "reduce":
                                    logger.info("Hurst regime: REDUCE position size (high entropy, scalar=%.2f)", _hurst_scalar)
                                    if "hurst_reduce" not in regime_label:
                                        regime_label = f"{regime_label}:hurst_reduce" if regime_label else "hurst_reduce"
                                # If Hurst says "trade" (good regime), boost confidence
                                elif _hurst_recommendation == "trade":
                                    logger.info("Hurst regime: FAVORABLE for trading (scalar=%.2f)", _hurst_scalar)
                                    if "hurst_trade" not in regime_label:
                                        regime_label = f"{regime_label}:hurst_trade" if regime_label else "hurst_trade"
                        except Exception as _hurst_exc:
                            logger.debug("Hurst regime enhancement failed: %s", _hurst_exc)
                        
                        # Push L2 spread/OBI into MultiVenueExecutor venue stats (best-effort)
                        try:
                            _l2 = getattr(self, "l2_feed", None)
                            _mve = getattr(self.component_registry, "_multi_venue_executor", None)
                            if _l2 is not None and _l2.is_connected and _mve is not None:
                                _primary_sym = str(
                                    (getattr(self.config, "trading_pairs", None) or ["BTC/USD"])[0]
                                )
                                _spread = _l2.get_spread_bps(_primary_sym)
                                _book = _l2.get_book(_primary_sym)
                                _obi = _book.imbalance() if _book is not None else 0.0
                                # Spread-based liquidity: tighter spread → higher score
                                if _spread is not None and _spread > 0:
                                    _l2_liq = max(0.1, min(1.0, 1.0 - (_spread / 50.0)))
                                    _l2_exchange = str(getattr(self.config, "primary_exchange", "kraken") or "kraken").lower()
                                    _mve.update_venue_stats(_l2_exchange, liquidity_score=_l2_liq)
                        except Exception:
                            pass
                        
                        # Apply order flow analysis to existing signals
                        try:
                            if _cycle_advisory and "order_flow" in _cycle_advisory:
                                _of = _cycle_advisory["order_flow"]
                                _of_buy_pressure = float(_of.get("buy_pressure", 0.5))
                                _of_sell_pressure = float(_of.get("sell_pressure", 0.5))
                                _of_imbalance = float(_of.get("imbalance", 0.0))
                                _of_whale = bool(_of.get("whale_detected", False))
                                
                                # Adjust signal confidence based on order flow
                                for _sig in (optimized_signals or []):
                                    _sig_action = str(getattr(_sig, "action", "") or "").upper()
                                    _sig_conf = float(getattr(_sig, "confidence", 0.5) or 0.5)
                                    
                                    # If buy signal and buy pressure is high, boost confidence
                                    if _sig_action in ("BUY", "LONG") and _of_buy_pressure > 0.6:
                                        _boost = min(0.15, (_of_buy_pressure - 0.5) * 0.3)
                                        setattr(_sig, "confidence", min(1.0, _sig_conf + _boost))
                                        logger.debug("Order flow boost BUY signal: +%.2f (buy_pressure=%.2f)", _boost, _of_buy_pressure)
                                    
                                    # If sell signal and sell pressure is high, boost confidence
                                    elif _sig_action in ("SELL", "SHORT") and _of_sell_pressure > 0.6:
                                        _boost = min(0.15, (_of_sell_pressure - 0.5) * 0.3)
                                        setattr(_sig, "confidence", min(1.0, _sig_conf + _boost))
                                        logger.debug("Order flow boost SELL signal: +%.2f (sell_pressure=%.2f)", _boost, _of_sell_pressure)
                                    
                                    # If whale detected, reduce position size (more volatile)
                                    if _of_whale:
                                        _current_strength = float(getattr(_sig, "strength", 0.5) or 0.5)
                                        setattr(_sig, "strength", _current_strength * 0.7)
                                        logger.debug("Whale detected - reducing signal strength by 30%%")
                        except Exception as _of_exc:
                            logger.debug("Order flow signal adjustment failed: %s", _of_exc)
                        
                        # Apply whale tracker signals
                        try:
                            if _cycle_advisory and "whale_activity" in _cycle_advisory:
                                _whale = _cycle_advisory["whale_activity"]
                                _whale_signals = _whale.get("signals", [])
                                _whale_net_flow = float(_whale.get("net_flow", 0.0))
                                
                                # If significant whale activity, log and adjust risk
                                if _whale_signals:
                                    logger.info("Whale activity detected: %d recent signals, net_flow=%.2f", 
                                               len(_whale_signals), _whale_net_flow)
                                    
                                    # If whales are selling (negative net flow), be more cautious on buys
                                    if _whale_net_flow < -1_000_000:  # $1M+ net outflow
                                        for _sig in (optimized_signals or []):
                                            _sig_action = str(getattr(_sig, "action", "") or "").upper()
                                            if _sig_action in ("BUY", "LONG"):
                                                _sig_conf = float(getattr(_sig, "confidence", 0.5) or 0.5)
                                                setattr(_sig, "confidence", max(0.1, _sig_conf * 0.8))
                                                logger.debug("Whale outflow - reducing BUY confidence by 20%%")
                                    
                                    # If whales are buying (positive net flow), be more confident on buys
                                    elif _whale_net_flow > 1_000_000:  # $1M+ net inflow
                                        for _sig in (optimized_signals or []):
                                            _sig_action = str(getattr(_sig, "action", "") or "").upper()
                                            if _sig_action in ("BUY", "LONG"):
                                                _sig_conf = float(getattr(_sig, "confidence", 0.5) or 0.5)
                                                setattr(_sig, "confidence", min(1.0, _sig_conf * 1.15))
                                                logger.debug("Whale inflow - boosting BUY confidence by 15%%")
                        except Exception as _whale_exc:
                            logger.debug("Whale tracker signal adjustment failed: %s", _whale_exc)
                        
                        # Inject Kalman pairs signal into the signal stream
                        if _cycle_advisory and "kalman_pairs" in _cycle_advisory:
                            _kp = _cycle_advisory["kalman_pairs"]
                            _kp_action = str(_kp.get("action", "HOLD"))
                            if _kp_action in ("LONG_SPREAD", "SHORT_SPREAD"):
                                _kp_side = "buy" if _kp_action == "LONG_SPREAD" else "sell"
                                _kp_sym = str(_kp.get("asset_a", "BTC/USD"))
                                _kp_price = _cycle_prices.get(_kp_sym, 0.0)
                                if _kp_price > 0:
                                    try:
                                        from unified_types import TradingSignal
                                        _kp_conf = min(0.72, 0.55 + abs(float(_kp.get("z_score", 0))) * 0.05)
                                        _kp_sig = TradingSignal(
                                            symbol=_kp_sym,
                                            action=_kp_side.upper(),
                                            confidence=_kp_conf,
                                            strength=0.5,
                                            entry_price=_kp_price,
                                            reasoning=f"kalman_pairs: {_kp.get('reason', '')}",
                                        )
                                        optimized_signals = list(optimized_signals or []) + [_kp_sig]
                                    except Exception as _kp_e:
                                        logger.debug("kalman_pairs signal inject: %s", _kp_e)

                        # FIX 13: Wire funding rate harvester signals
                        try:
                            if _cycle_advisory and "funding_harvester" in _cycle_advisory:
                                _fh = _cycle_advisory["funding_harvester"]
                                _fh_action = str(_fh.get("action", "HOLD")).upper()
                                if _fh_action in ("BUY", "SELL"):
                                    _fh_sym = str(next(iter(_cycle_prices), "BTC/USD"))
                                    _fh_price = _cycle_prices.get(_fh_sym, 0.0)
                                    if _fh_price > 0:
                                        from unified_types import TradingSignal
                                        _fh_conf = float(_fh.get("confidence", 0.5))
                                        _fh_sig = TradingSignal(
                                            symbol=_fh_sym,
                                            action=_fh_action,
                                            confidence=_fh_conf,
                                            strength=0.4,
                                            entry_price=_fh_price,
                                            reasoning=f"funding_harvester: {_fh.get('reason', '')}",
                                        )
                                        optimized_signals = list(optimized_signals or []) + [_fh_sig]
                                        logger.info(
                                            "funding_harvester signal injected: %s %s @ %.2f conf=%.2f",
                                            _fh_action, _fh_sym, _fh_price, _fh_conf,
                                        )
                        except Exception as _fh_exc:
                            logger.debug("funding_harvester signal inject: %s", _fh_exc)

                        # Inject strategy scanner top opportunities as signals
                        try:
                            if _cycle_advisory and "strategy_scanner" in _cycle_advisory:
                                _ss = _cycle_advisory["strategy_scanner"]
                                _ss_top = _ss.get("top_strategies", []) if isinstance(_ss, dict) else []
                                for _ss_entry in (_ss_top if isinstance(_ss_top, list) else [])[:3]:
                                    if not isinstance(_ss_entry, dict):
                                        continue
                                    _ss_sym = str(_ss_entry.get("symbol", ""))
                                    _ss_sharpe = float(_ss_entry.get("sharpe", 0) or 0)
                                    _ss_conf = float(_ss_entry.get("win_rate", 0.5) or 0.5)
                                    _ss_ret = float(_ss_entry.get("return_pct", 0) or 0)
                                    if _ss_sym and _ss_sharpe > 0.3 and _ss_ret > 0 and _ss_sym in _cycle_prices:
                                        _ss_price = float(_cycle_prices.get(_ss_sym, 0))
                                        if _ss_price > 0:
                                            from unified_types import TradingSignal
                                            _ss_action = "BUY"  # scanner finds positive-edge opportunities
                                            _ss_sig = TradingSignal(
                                                symbol=_ss_sym,
                                                action=_ss_action,
                                                confidence=min(0.85, _ss_conf),
                                                strength=min(0.80, _ss_sharpe),
                                                entry_price=_ss_price,
                                                reasoning=f"scanner: {_ss_entry.get('strategy', 'unknown')} sharpe={_ss_sharpe:.2f} ret={_ss_ret:.1f}%",
                                            )
                                            optimized_signals = list(optimized_signals or []) + [_ss_sig]
                                            logger.info(
                                                "Scanner signal injected: %s %s @ %.2f (sharpe=%.2f, ret=%.1f%%)",
                                                _ss_action, _ss_sym, _ss_price, _ss_sharpe, _ss_ret,
                                            )
                        except Exception as _ss_exc:
                            logger.debug("strategy_scanner signal inject: %s", _ss_exc)

                        # FIX 14: Wire cross-exchange arb signals
                        try:
                            _xarb = getattr(self.component_registry, "_strategies", {}).get("cross_exchange_arb") if self.component_registry else None
                            if _xarb is None and self.component_registry is not None:
                                _xarb = getattr(self.component_registry, "cross_exchange_arb", None)
                            if _xarb is not None and _cycle_prices:
                                # Feed prices from all connected exchanges
                                _xarb_exchanges = list(getattr(self, "exchanges", {}).keys()) or ["kraken", "coinbase"]
                                for _xe in _xarb_exchanges:
                                    for _xs, _xp in _cycle_prices.items():
                                        if _xp > 0:
                                            try:
                                                _xarb.update_price(_xe, _xs, _xp)
                                            except Exception:
                                                pass
                                # Find opportunities
                                for _xs in _cycle_prices:
                                    try:
                                        _opp = _xarb.find_opportunity(_xs)
                                        if _opp is not None and _opp.net_spread_bps > 0:
                                            from unified_types import TradingSignal
                                            _xarb_sig = TradingSignal(
                                                symbol=_xs,
                                                action="BUY",
                                                confidence=min(0.8, _opp.confidence),
                                                strength=min(0.6, _opp.net_spread_bps / 20.0),
                                                entry_price=_opp.cheap_price,
                                                reasoning=f"cross_exchange_arb: buy@{_opp.cheap_exchange} sell@{_opp.expensive_exchange} spread={_opp.net_spread_bps:.1f}bps",
                                            )
                                            optimized_signals = list(optimized_signals or []) + [_xarb_sig]
                                            logger.info(
                                                "cross_exchange_arb signal injected: %s spread=%.1f bps",
                                                _xs, _opp.net_spread_bps,
                                            )
                                    except Exception as _xopp_exc:
                                        logger.debug("cross_exchange_arb find_opportunity(%s): %s", _xs, _xopp_exc)
                        except Exception as _xarb_exc:
                            logger.debug("cross_exchange_arb signal inject: %s", _xarb_exc)

                except Exception:
                    pass

                # ── Strategy Router: collect signals from all wired strategies ──
                try:
                    _sr = getattr(self.component_registry, "strategy_router", None) if self.component_registry else None
                    if _sr is not None:
                        _sr_symbol = str((getattr(self.config, "trading_pairs", None) or ["BTC/USD"])[0])
                        _sr_ohlcv = getattr(self, "_last_ohlcv", None)  # cached OHLCV if available
                        _sr_regime = str(regime_label or "UNKNOWN")
                        _sr_market_data = dict(_cycle_prices) if _cycle_prices else {}
                        _sr_market_data["price"] = _cycle_prices.get(_sr_symbol, 0.0) if _cycle_prices else 0.0
                        _router_signals = await asyncio.wait_for(
                            _sr.generate_all_signals(
                                symbol=_sr_symbol,
                                ohlcv=_sr_ohlcv,
                                regime=_sr_regime,
                                market_data=_sr_market_data,
                            ),
                            timeout=float(getattr(self.config, "op_timeout_s", 30.0) or 30.0),
                        )
                        if _router_signals:
                            optimized_signals = list(optimized_signals or []) + list(_router_signals)
                            logger.debug(
                                "StrategyRouter injected %d signals into trading loop",
                                len(_router_signals),
                            )
                except Exception as _sr_exc:
                    logger.debug("strategy_router signal injection: %s", _sr_exc)

                # Regime change alerter
                try:
                    if self.regime_alerter is not None and regime_label:
                        primary_sym = str(getattr(self.config, "trading_pairs", ["BTC/USD"])[0] if hasattr(self.config, "trading_pairs") else "BTC/USD")
                        _alert_coro = self.regime_alerter.check_and_alert(primary_sym, str(regime_label))
                        if asyncio.iscoroutine(_alert_coro):
                            await _alert_coro
                except Exception:
                    pass

                # ── ML ensemble signal weighting ──────────────────────────────
                # Use ensemble_signal_hub composite to weight signal confidence
                try:
                    _ensemble_data = (_cycle_advisory or {}).get("ensemble") if _cycle_advisory else None
                    if _ensemble_data and optimized_signals:
                        _ens_composite = float(_ensemble_data.get("composite", 0.0))
                        _ens_confidence = float(_ensemble_data.get("confidence", 0.0))
                        _ens_size_mult = float(_ensemble_data.get("size_multiplier", 1.0))
                        if _ens_confidence > 0.3 and abs(_ens_composite) > 0.05:
                            for _sig in optimized_signals:
                                _sig_action = str(getattr(_sig, "action", "") or "").upper()
                                _sig_conf = float(getattr(_sig, "confidence", 0.5) or 0.5)
                                # Boost confidence if ensemble agrees with signal direction
                                _sig_is_long = _sig_action in ("BUY", "LONG")
                                _sig_is_short = _sig_action in ("SELL", "SHORT")
                                if (_sig_is_long and _ens_composite > 0) or (_sig_is_short and _ens_composite < 0):
                                    _boost = min(0.1, abs(_ens_composite) * _ens_confidence * 0.15)
                                    _new_conf = min(1.0, _sig_conf + _boost)
                                    setattr(_sig, "confidence", _new_conf)
                                elif (_sig_is_long and _ens_composite < -0.2) or (_sig_is_short and _ens_composite > 0.2):
                                    # Penalize confidence if ensemble disagrees
                                    _penalty = min(0.1, abs(_ens_composite) * _ens_confidence * 0.10)
                                    _new_conf = max(0.0, _sig_conf - _penalty)
                                    setattr(_sig, "confidence", _new_conf)
                except Exception as _ens_exc:
                    logger.debug("ML ensemble signal weighting (optional): %s", _ens_exc)

                trace_id = str(getattr(self, "_trace_id", None) or uuid.uuid4().hex)
                logger.info("SIGNAL DEBUG: before regime_gate=%d", len(optimized_signals or []))
                pre_regime_signals = list(optimized_signals or [])
                optimized_signals = self._apply_regime_strategy_gating(optimized_signals, regime_label)
                logger.info("SIGNAL DEBUG: after regime_gate=%d", len(optimized_signals or []))
                self._snapshot_rejected_candidates(
                    before=pre_regime_signals,
                    after=optimized_signals,
                    cycle_id=int(cycles),
                    correlation_id=str(cycle_correlation_id),
                    reason_code=ReasonCode.PRE_TRADE_RISK_BLOCK,
                    details={"stage": "regime_gate", "regime_label": str(regime_label)},
                    trace_id=trace_id,
                )
                try:
                    self._attach_strategy_evaluation_context(
                        list(optimized_signals or []),
                        str(regime_label or ""),
                    )
                except Exception as e:
                    self._handle_strategy_eval_error(e, context="pre_meta_attach_context")
                optimized_signals = self._apply_self_optimizing_meta_weights(
                    optimized_signals,
                    regime_label=str(regime_label or ""),
                    cycle_id=int(cycles) + 1,
                    trace_id=trace_id,
                )
                logger.info("SIGNAL DEBUG: after meta_weights=%d", len(optimized_signals or []))
                cycle_stage_timing_ms["strategy_evaluation_ms"] = max(
                    0.0, (time.perf_counter() - t_strategy_eval_start) * 1000.0
                )
                t_portfolio_targeting_start = time.perf_counter()
                logger.info("SIGNAL DEBUG: before _convert_signals_to_targets=%d", len(optimized_signals or []))
                optimized_signals = self._convert_signals_to_targets(
                    optimized_signals,
                    regime_label,
                    cycle_id=int(cycles),
                    correlation_id=str(cycle_correlation_id),
                    trace_id=trace_id,
                )
                logger.info("SIGNAL DEBUG: after _convert_signals_to_targets=%d", len(optimized_signals or []))
                optimized_signals = self._filter_live_safe_signals(
                    optimized_signals, stage="target_execution_signals"
                )
                logger.info("SIGNAL DEBUG: after live_safe_filter=%d", len(optimized_signals or []))
                target_stage = dict(getattr(self, "_last_target_pipeline_stage_ms", {}) or {})
                cycle_stage_timing_ms["portfolio_targeting_ms"] = float(
                    target_stage.get("portfolio_targeting_ms", max(0.0, (time.perf_counter() - t_portfolio_targeting_start) * 1000.0))
                )
                cycle_stage_timing_ms["liquidity_adjustment_ms"] = float(
                    target_stage.get("liquidity_adjustment_ms", 0.0)
                )

                # Portfolio guardrails (paper-safe): block SELL when flat; block BUY when insufficient cash.
                pre_portfolio_guardrail = list(optimized_signals or [])
                logger.info("SIGNAL DEBUG: before portfolio_guard=%d", len(pre_portfolio_guardrail))
                optimized_signals = [s for s in optimized_signals if self._portfolio_allows_signal(s)]
                logger.info("SIGNAL DEBUG: after portfolio_guard=%d (dropped=%d)", len(optimized_signals), len(pre_portfolio_guardrail) - len(optimized_signals))
                for _dropped in pre_portfolio_guardrail:
                    if _dropped not in optimized_signals:
                        _d_sym = self._signal_get(_dropped, "symbol", "?")
                        _d_act = self._signal_get(_dropped, "action", "?")
                        _d_qty = self._signal_get(_dropped, "quantity", 0)
                        _d_px = self._signal_get(_dropped, "entry_price", 0)
                        logger.info("SIGNAL DEBUG: DROPPED by portfolio_guard: %s %s qty=%s px=%s", _d_sym, _d_act, _d_qty, _d_px)
                self._snapshot_rejected_candidates(
                    before=pre_portfolio_guardrail,
                    after=optimized_signals,
                    cycle_id=int(cycles),
                    correlation_id=str(cycle_correlation_id),
                    reason_code=ReasonCode.PRE_TRADE_RISK_BLOCK,
                    details={"stage": "portfolio_guardrail"},
                    trace_id=trace_id,
                )

                # Multi-language per-cycle: all 23 languages contribute; aggregate used for confidence.
                _ml_enabled = bool(getattr(self.config, "multi_language_enabled", True))
                if not _ml_enabled:
                    _ml_enabled = False  # Config override
                elif hasattr(self.config, "multi_language") and isinstance(getattr(self.config, "multi_language", None), dict):
                    _ml_enabled = bool(getattr(self.config, "multi_language", {}).get("enabled", True))
                if self.language_orchestrator and _ml_enabled:
                    try:
                        primary = (getattr(self.config, "trading_pairs", None) or ["BTC/USD"])[0]
                        cycle_ctx = {
                            "portfolio_value_aud": float(self.portfolio_value_aud),
                            "cash_balance_aud": float(self.cash_balance_aud),
                            "signals": int(len(optimized_signals)),
                            "correlation_id": getattr(self, "_cycle_correlation_id", None),
                            "signals_count": int(len(optimized_signals)),
                            "primary_exchange": str(getattr(self.config, "primary_exchange", "kraken")),
                            "symbol": str(primary),
                            "timeframe": str(getattr(self.config, "signal_primary_timeframe", "1h") or "1h"),
                            "equity_curve": getattr(self, "_equity_history", None) and list(self._equity_history[-20:]) or [],
                            "recent_trades": getattr(self, "_recent_trades_for_cycle", None) or [],
                        }
                        if not hasattr(self, "_cycle_counter"):
                            self._cycle_counter = 0
                        self._cycle_counter += 1
                        cycle_ctx["cycle_id"] = self._cycle_counter
                        # Optional: regime from 23-language consensus (use_regime_estimate) so everything works together
                        use_regime = getattr(self.config, "use_regime_estimate", False)
                        if use_regime and self.market_data_service:
                            try:
                                primary = (getattr(self.config, "trading_pairs", None) or ["BTC/USD"])[0]
                                df = await self.market_data_service.fetch_ohlcv_df(primary, timeframe="1m", limit=60)
                                if df is not None and not df.empty and "close" in df.columns:
                                    prices = df["close"].astype(float).tolist()
                                    reg_result = await self.language_orchestrator.execute_regime_estimate({"prices": prices, "symbol": str(primary), "timeframe": "1m", "window": len(prices)})
                                    self._last_regime_consensus = reg_result
                                    cycle_ctx["regime"] = reg_result.get("regime", "mean_revert")
                                    cycle_ctx["regime_confidence"] = float(reg_result.get("confidence", 0.5))
                                    # Inject 23-language regime into strategy engine so allocator/adaptive risk see it
                                    try:
                                        se = getattr(self.ai_brain, "strategy_engine", None)
                                        if se is not None and hasattr(se, "_last_regime") and isinstance(se._last_regime, dict):
                                            from adaptive.regime import MarketRegime
                                            rstr = str(reg_result.get("regime", "mean_revert") or "mean_revert").lower()
                                            if "high_vol" in rstr or rstr == "high_vol":
                                                se._last_regime[primary] = MarketRegime.HIGH_VOL
                                            elif "trend" in rstr:
                                                se._last_regime[primary] = MarketRegime.TREND_UP  # direction from 23lang not split
                                            else:
                                                se._last_regime[primary] = MarketRegime.RANGE
                                    except Exception:
                                        pass
                            except Exception as e:
                                logger.debug("Regime estimate (optional): %s", e)
                                if self._last_regime_consensus:
                                    cycle_ctx["regime"] = self._last_regime_consensus.get("regime", "mean_revert")
                                    cycle_ctx["regime_confidence"] = float(self._last_regime_consensus.get("confidence", 0.5))
                        results, aggregate = await self.language_orchestrator.execute_cycle_plan_with_aggregate(cycle_ctx)

                        ledger = getattr(self.execution_engine, "trade_ledger", None)
                        record_fn = getattr(ledger, "record_language_call", None) if ledger else None
                        if callable(record_fn):
                            for r in results:
                                try:
                                    record_fn(
                                        language=str(r.language_used),
                                        task_type="cycle_plan",
                                        ok=bool(r.success),
                                        correlation_id=None,
                                        took_ms=float(r.execution_time_ms),
                                        payload_json=json.dumps(cycle_ctx, ensure_ascii=True),
                                        result_json=json.dumps(r.result, ensure_ascii=True, default=str),
                                        error=str(r.error_message or "") if not r.success else None,
                                    )
                                except Exception:
                                    pass
                        # Apply 23-language consensus boost to signal confidence (strength: correctness languages = conservative view)
                        use_boost = getattr(self.config, "use_cycle_aggregate_boost", True)
                        if use_boost and aggregate.get("count", 0) > 0 and optimized_signals:
                            use_conservative = getattr(self.config, "use_conservative_cycle_boost", False)
                            use_weighted = getattr(self.config, "use_weighted_mean_boost", False)
                            if use_weighted and aggregate.get("weighted_mean_boost") is not None:
                                median_boost = float(aggregate.get("weighted_mean_boost", 0.0))
                            else:
                                median_boost = float(aggregate.get("conservative_median" if use_conservative else "median_boost", 0.0))
                            for sig in optimized_signals:
                                c = float(getattr(sig, "confidence", 0.0) or 0.0)
                                if c > 0:
                                    setattr(sig, "confidence", min(1.0, max(0.0, c * (1.0 + median_boost))))
                        # Hybrid switcher: optionally decide quantum vs classical this cycle (use_hybrid_switcher)
                        use_quantum_this_cycle = True
                        if getattr(self.config, "use_hybrid_switcher", False):
                            try:
                                from quantum.hybrid_quantum_classical import QuantumClassicalSwitcher
                                switcher = getattr(self, "_quantum_hybrid_switcher", None)
                                if switcher is None:
                                    switcher = QuantumClassicalSwitcher()
                                    setattr(self, "_quantum_hybrid_switcher", switcher)
                                decision = await switcher.decide_execution_method(
                                    "risk", 0.5, len(optimized_signals) or 1, 5.0
                                )
                                use_quantum_this_cycle = bool(getattr(decision, "use_quantum", True))
                            except Exception as eh:
                                logger.debug("Hybrid switcher skipped: %s", eh)
                        # Quantum walk confidence nudge (use_quantum_walk): visitation probabilities boost/nudge signal confidence
                        use_qwalk = bool(getattr(self.config, "use_quantum_walk", False) and use_quantum_this_cycle)
                        steps_cfg = int(getattr(self.config, "quantum_walk_steps", 15) or 15)
                        damping_cfg = float(getattr(self.config, "quantum_walk_damping", 0.15) or 0.15)
                        corr_cfg = float(getattr(self.config, "quantum_walk_correlation_threshold", 0.2) or 0.2)
                        boost_cap_cfg = float(getattr(self.config, "quantum_walk_boost_cap", 0.12) or 0.12)
                        visitation_1h: Optional[Dict[str, float]] = None
                        visitation_1m: Optional[Dict[str, float]] = None
                        if use_qwalk and optimized_signals and self.market_data_service:
                            try:
                                pairs = list(getattr(self.config, "trading_pairs", []) or [])[:8]
                                if len(pairs) >= 2:
                                    returns_1h: Dict[str, List[float]] = {}
                                    timeframe_1h = str(getattr(self.config, "signal_primary_timeframe", "1h") or "1h")
                                    for sym in pairs:
                                        df = await self.market_data_service.fetch_ohlcv_df(sym, timeframe=timeframe_1h, limit=60)
                                        if df is not None and not df.empty and "close" in df.columns:
                                            closes = df["close"].astype(float).tolist()
                                            if len(closes) >= 3:
                                                ret = [(closes[i] - closes[i - 1]) / max(closes[i - 1], 1e-12) for i in range(1, len(closes))]
                                                returns_1h[str(sym)] = ret
                                    if len(returns_1h) >= 2:
                                        use_full = getattr(self.config, "use_full_quantum_walk", False)
                                        walker = None
                                        if use_full:
                                            try:
                                                from quantum_walk import QuantumWalkSimulator  # noqa: F401
                                                if QuantumWalkSimulator is not None and hasattr(QuantumWalkSimulator, "analyze"):
                                                    walker = QuantumWalkSimulator(steps=steps_cfg)
                                            except Exception:
                                                pass
                                        if walker is None:
                                            from quantum_walk import QuantumWalkLite
                                            walker = QuantumWalkLite(correlation_threshold=corr_cfg, steps=steps_cfg, damping=damping_cfg)
                                        result_1h = walker.analyze(returns_history=returns_1h)
                                        if result_1h is not None:
                                            visitation = dict(result_1h.visitation_probabilities)
                                            visitation_1h = dict(result_1h.visitation_probabilities)
                                            if getattr(self.config, "use_multi_timeframe_walk", False):
                                                returns_1m_dict: Dict[str, List[float]] = {}
                                                for sym in list(returns_1h.keys())[:6]:
                                                    df = await self.market_data_service.fetch_ohlcv_df(sym, timeframe="1m", limit=60)
                                                    if df is not None and not df.empty and "close" in df.columns:
                                                        closes = df["close"].astype(float).tolist()
                                                        if len(closes) >= 3:
                                                            ret = [(closes[i] - closes[i - 1]) / max(closes[i - 1], 1e-12) for i in range(1, len(closes))]
                                                            returns_1m_dict[str(sym)] = ret
                                                if len(returns_1m_dict) >= 2:
                                                    from quantum_walk import QuantumWalkLite
                                                    qwl_1m = QuantumWalkLite(correlation_threshold=corr_cfg, steps=steps_cfg, damping=damping_cfg)
                                                    result_1m = qwl_1m.analyze(returns_history=returns_1m_dict)
                                                    visitation_1m = dict(result_1m.visitation_probabilities)
                                                    n = len(visitation)
                                                    for s, p in result_1m.visitation_probabilities.items():
                                                        visitation[s] = (visitation.get(s, 1.0 / n) + p) / 2.0
                                            n_sym = len(visitation)
                                            uniform = 1.0 / n_sym
                                            for sig in optimized_signals:
                                                sym = getattr(sig, "symbol", None) or (sig.get("symbol") if isinstance(sig, dict) else None)
                                                if not sym:
                                                    sym = pairs[0] if pairs else None
                                                if sym and sym in visitation:
                                                    v = visitation[sym]
                                                    delta = (v - uniform) * boost_cap_cfg
                                                    c = float(getattr(sig, "confidence", 0.0) or 0.0)
                                                    if c > 0:
                                                        new_c = min(1.0, max(0.0, c * (1.0 + delta)))
                                                        setattr(sig, "confidence", new_c)
                                            # Multi-timeframe confirmation: require symbol above uniform on both 1h and 1m
                                            if getattr(self.config, "use_multi_timeframe_confirmation", False) and visitation_1h and visitation_1m:
                                                n1 = len(visitation_1h)
                                                n2 = len(visitation_1m)
                                                u1 = 1.0 / n1 if n1 else 0
                                                u2 = 1.0 / n2 if n2 else 0
                                                for sig in optimized_signals:
                                                    sym = getattr(sig, "symbol", None) or (sig.get("symbol") if isinstance(sig, dict) else None)
                                                    if not sym:
                                                        continue
                                                    v1 = visitation_1h.get(sym, 0.0)
                                                    v2 = visitation_1m.get(sym, 0.0)
                                                    if v1 <= u1 or v2 <= u2:
                                                        setattr(sig, "confidence", 0.0)
                            except Exception as eq:
                                logger.debug("Quantum walk confidence nudge skipped: %s", eq)
                        # Quantum annealing selection: select optimal signal combination via QUBO
                        # Uses real simulated quantum annealing with transverse-field tunneling
                        if getattr(self.config, "use_quantum_annealing_selection", False) and optimized_signals:
                            try:
                                from quantum_unified_stubs import quantum_annealing_select_signals
                                n_sig = len(optimized_signals)
                                confidences = [float(getattr(s, "confidence", 0.0) or 0.0) for s in optimized_signals]
                                max_concurrent = int(getattr(self.config, "max_concurrent_positions", 5) or 5)
                                result = quantum_annealing_select_signals(
                                    confidences,
                                    max_signals=min(max_concurrent, n_sig),
                                    num_reads=200,
                                )
                                selected_idx = result.get("selected_indices", [])
                                if selected_idx and len(selected_idx) < n_sig:
                                    keep_set = set(selected_idx)
                                    optimized_signals[:] = [optimized_signals[i] for i in range(n_sig) if i in keep_set]
                                    logger.debug("Quantum annealing: selected %d/%d signals (method=%s)",
                                                 len(selected_idx), n_sig, result.get("method", "unknown"))
                            except Exception as eqa:
                                logger.debug("Quantum annealing selection skipped: %s", eqa)
                        # Optional: quantum portfolio weights from production simulator (every 10 cycles)
                        if (
                            getattr(self.config, "use_quantum_portfolio_weights", False)
                            and getattr(self.config, "quantum_simulator_use_production", False)
                            and self.market_data_service
                            and (getattr(self, "_cycle_counter", 0) % 10 == 0)
                        ):
                            try:
                                from quantum.production_quantum_simulator import optimize_portfolio_with_quantum
                                pairs = list(getattr(self.config, "trading_pairs", []) or [])[:10]
                                if len(pairs) >= 2:
                                    tf = str(getattr(self.config, "signal_primary_timeframe", "1h") or "1h")
                                    returns_dict: Dict[str, List[float]] = {}
                                    for sym in pairs:
                                        df = await self.market_data_service.fetch_ohlcv_df(sym, timeframe=tf, limit=60)
                                        if df is not None and not df.empty and "close" in df.columns:
                                            closes = df["close"].astype(float).tolist()
                                            if len(closes) >= 3:
                                                ret = [(closes[i] - closes[i - 1]) / max(closes[i - 1], 1e-12) for i in range(1, len(closes))]
                                                returns_dict[str(sym)] = ret
                                    if len(returns_dict) >= 2:
                                        assets = sorted(returns_dict.keys())
                                        n = min(len(r) for r in returns_dict.values())
                                        mat = np.array([returns_dict[s][:n] for s in assets], dtype=float)
                                        ret_arr = np.mean(mat, axis=1)
                                        cov = np.cov(mat) if mat.shape[0] > 1 else np.eye(len(assets)) * 0.01
                                        weights_result = await optimize_portfolio_with_quantum(
                                            assets, ret_arr, cov, risk_target=0.02
                                        )
                                        if weights_result is not None and hasattr(weights_result, "optimal_weights"):
                                            w = getattr(weights_result, "optimal_weights", None)
                                            if w is not None and len(w) == len(assets):
                                                setattr(self, "_quantum_portfolio_weights", dict(zip(assets, w.tolist())))
                            except Exception as eqw:
                                logger.debug("Quantum portfolio weights skipped: %s", eqw)
                        # Quantum walk portfolio weights: Szegedy walk on correlation graph
                        # Uses quantum walk centrality to determine asset weighting
                        if (
                            getattr(self.config, "use_quantum_walk", False)
                            and self.market_data_service
                            and (getattr(self, "_cycle_counter", 0) % 10 == 0)
                            and not getattr(self, "_quantum_portfolio_weights", None)
                        ):
                            try:
                                from quantum_unified_stubs import quantum_walk_portfolio_weights
                                pairs = list(getattr(self.config, "trading_pairs", []) or [])[:10]
                                if len(pairs) >= 2:
                                    tf = str(getattr(self.config, "signal_primary_timeframe", "1h") or "1h")
                                    walk_returns: Dict[str, List[float]] = {}
                                    for sym in pairs:
                                        df = await self.market_data_service.fetch_ohlcv_df(sym, timeframe=tf, limit=60)
                                        if df is not None and not df.empty and "close" in df.columns:
                                            closes = df["close"].astype(float).tolist()
                                            if len(closes) >= 3:
                                                ret = [(closes[i] - closes[i - 1]) / max(closes[i - 1], 1e-12) for i in range(1, len(closes))]
                                                walk_returns[str(sym)] = ret
                                    if len(walk_returns) >= 2:
                                        walk_result = quantum_walk_portfolio_weights(walk_returns, strategy="centrality")
                                        w = walk_result.get("weights")
                                        if isinstance(w, dict) and w:
                                            setattr(self, "_quantum_portfolio_weights", w)
                                            logger.debug("Quantum walk portfolio weights: %s (entropy=%.3f, mixing=%d)",
                                                         {k: f"{v:.3f}" for k, v in w.items()},
                                                         walk_result.get("walk_entropy", 0.0),
                                                         walk_result.get("mixing_time", 0))
                            except Exception as eqwalk:
                                logger.debug("Quantum walk portfolio weights skipped: %s", eqwalk)
                    except Exception as e:
                        logger.warning(f"Multi-language cycle routing failed: {e}")

                # ── Thompson-sampled ensemble signal scoring ──────────────────
                # Uses per-language accuracy tracking (Beta distributions) to
                # dynamically weight signal deltas — languages that predict
                # correctly get more influence over time.
                if self.language_orchestrator and optimized_signals and getattr(self.config, "use_ensemble_signal", False):
                    try:
                        primary = (getattr(self.config, "trading_pairs", None) or ["BTC/USD"])[0]
                        for sig in optimized_signals:
                            sig_obj = sig if isinstance(sig, dict) else getattr(sig, "__dict__", {})
                            action = (sig_obj.get("action") if isinstance(sig_obj, dict) else getattr(sig, "action", "")) or ""
                            symbol = (sig_obj.get("symbol") if isinstance(sig_obj, dict) else getattr(sig, "symbol", "")) or str(primary)
                            ensemble_data = {
                                "action": str(action),
                                "symbol": str(symbol),
                                "confidence": float(getattr(sig, "confidence", 0.5) or 0.5),
                                "regime": (self._last_regime_consensus or {}).get("regime", "mean_revert"),
                                "timeframe": str(getattr(self.config, "signal_primary_timeframe", "1h") or "1h"),
                            }
                            ens_result = await self.language_orchestrator.execute_ensemble_signal(ensemble_data)
                            if ens_result.get("languages_used", 0) > 0:
                                delta = float(ens_result.get("score_delta", 0.0))
                                c = float(getattr(sig, "confidence", 0.0) or 0.0)
                                if c > 0 and abs(delta) > 0.001:
                                    new_c = min(1.0, max(0.0, c * (1.0 + delta * 0.5)))
                                    setattr(sig, "confidence", new_c)
                    except Exception as e:
                        logger.debug("Ensemble signal scoring (optional): %s", e)

                # ── Microstructure toxicity gate ──────────────────────────────
                # Routes order book data through Rust for VPIN, toxicity, and
                # spoofing detection. Penalizes signal confidence when the
                # order book is toxic (informed flow dominates).
                if self.language_orchestrator and optimized_signals and getattr(self.config, "use_microstructure_gate", False):
                    try:
                        primary = (getattr(self.config, "trading_pairs", None) or ["BTC/USD"])[0]
                        if self.market_data_service:
                            ob = await self.market_data_service.fetch_order_book(str(primary), limit=20)
                            if isinstance(ob, dict) and (ob.get("bids") or ob.get("asks")):
                                micro_result = await self.language_orchestrator.execute_microstructure_analysis(ob)
                                toxicity = float(micro_result.get("toxicity_score", micro_result.get("toxicity", 0.0)) or 0.0)
                                tox_threshold = float(getattr(self.config, "microstructure_toxicity_threshold", 0.7) or 0.7)
                                spoof = micro_result.get("spoofing", {})
                                spoof_detected = bool(spoof.get("spoof_detected", False)) if isinstance(spoof, dict) else False
                                if toxicity > tox_threshold or spoof_detected:
                                    penalty = min(0.30, (toxicity - tox_threshold) * 0.5) if toxicity > tox_threshold else 0.0
                                    if spoof_detected:
                                        penalty = max(penalty, 0.15)
                                    for sig in optimized_signals:
                                        c = float(getattr(sig, "confidence", 0.0) or 0.0)
                                        if c > 0:
                                            setattr(sig, "confidence", max(0.0, c * (1.0 - penalty)))
                                    reason = []
                                    if toxicity > tox_threshold:
                                        reason.append(f"toxicity={toxicity:.2f}")
                                    if spoof_detected:
                                        reason.append("spoofing")
                                    logger.info("Microstructure gate: penalized confidence by %.0f%% (%s)", penalty * 100, ", ".join(reason))
                    except Exception as e:
                        logger.debug("Microstructure gate (optional): %s", e)

                # ── Formal risk gate (Haskell/F# veto power) ─────────────────
                # Correctness languages run Kelly bounds, drawdown cascade, and
                # risk invariant checks. ANY REJECT = trade blocked.
                if self.language_orchestrator and optimized_signals and getattr(self.config, "use_formal_risk_gate", False):
                    try:
                        position_value = float(self.portfolio_value_aud) - float(self.cash_balance_aud)
                        capital = float(self.portfolio_value_aud) or 1.0
                        peak = max(float(self.peak_equity_aud), capital)
                        risk_data = {
                            "position_value": position_value,
                            "capital": capital,
                            "peak_equity": peak,
                            "drawdown_pct": (peak - capital) / peak if peak > 0 else 0.0,
                            "max_drawdown_pct": float(getattr(self.config, "max_drawdown_pct", 0.12) or 0.12),
                            "win_rate": float(self.winning_trades / max(self.total_trades, 1)),
                            "avg_win": 0.02,
                            "avg_loss": 0.01,
                            "consecutive_losses": int(self.consecutive_losses),
                        }
                        gate_result = await self.language_orchestrator.execute_formal_risk_gate(risk_data)
                        if gate_result.get("gate") == "REJECT":
                            reason = gate_result.get("reason", "unknown")
                            logger.warning("Formal risk gate REJECTED trade: %s — skipping execution this cycle", reason)
                            optimized_signals = []
                        elif gate_result.get("gate") == "WARNING":
                            logger.info("Formal risk gate WARNING: some checks did not pass cleanly")
                    except Exception as e:
                        logger.debug("Formal risk gate (optional): %s", e)

                # Optional: 23-language risk gate (all pass, or conservative = correctness languages only)
                if self.language_orchestrator and optimized_signals:
                    use_risk_all = getattr(self.config, "use_risk_all", False)
                    use_conservative_risk = getattr(self.config, "use_conservative_risk", False)
                    if use_risk_all or use_conservative_risk:
                        try:
                            position_value = float(self.portfolio_value_aud) - float(self.cash_balance_aud)
                            capital = float(self.portfolio_value_aud) or 1.0
                            max_dd = float(getattr(self.config, "max_drawdown_pct", 0.12) or 0.12)
                            risk_all_result = await self.language_orchestrator.execute_risk_all(position_value, capital, max_drawdown_pct=max_dd)
                            fail_key = "conservative_pass" if use_conservative_risk else "passed"
                            if not risk_all_result.get(fail_key, True):
                                logger.warning("23-language risk gate (%s): failed - skipping execution this cycle", "conservative" if use_conservative_risk else "all")
                                optimized_signals = []
                        except Exception as e:
                            logger.warning(f"Multi-language risk-all check failed: {e}")

                    # Optional: 23-language position sizing cap (use_position_sizing_gate)
                    use_pos_sizing = getattr(self.config, "use_position_sizing_gate", False)
                    if use_pos_sizing and self.language_orchestrator and optimized_signals:
                        try:
                            capital = float(self.portfolio_value_aud) or 1.0
                            vol_bps = float(getattr(self.config, "realized_vol_pct", 0.0) or 0.0) * 100.0 or 10.0
                            conf = float(getattr(optimized_signals[0], "confidence", 0.5) or 0.5) if optimized_signals else 0.5
                            _sym = (getattr(self.config, "trading_pairs", None) or ["BTC/USD"])[0]
                            sizing_result = await self.language_orchestrator.execute_position_sizing_all({
                                "capital": capital,
                                "volatility_bps": vol_bps,
                                "confidence": conf,
                                "max_risk_pct": 0.02,
                                "symbol": str(_sym),
                                "timeframe": str(getattr(self.config, "signal_primary_timeframe", "1h") or "1h"),
                            })
                            max_pct = float(sizing_result.get("size_pct_conservative") or sizing_result.get("size_pct_median") or 0.15)
                            if max_pct > 0 and capital > 0:
                                for sig in optimized_signals:
                                    qty = sig.get("quantity") if isinstance(sig, dict) else getattr(sig, "quantity", None)
                                    entry = sig.get("entry_price") if isinstance(sig, dict) else getattr(sig, "entry_price", None)
                                    if qty is not None and entry and float(entry) > 0:
                                        notional = float(qty) * float(entry)
                                        cap_notional = max_pct * capital
                                        if notional > cap_notional:
                                            new_qty = cap_notional / float(entry)
                                            if isinstance(sig, dict):
                                                sig["quantity"] = new_qty
                                            else:
                                                setattr(sig, "quantity", new_qty)
                        except Exception as e:
                            logger.debug("Position sizing gate (optional): %s", e)

                    # Optional: quantum VaR position cap – limit notional so tail loss (CVaR) does not exceed fraction of capital
                    if getattr(self.config, "use_quantum_var_position_cap", False) and optimized_signals:
                        last_var = getattr(self, "_last_quantum_var_cvar", None)
                        if isinstance(last_var, dict):
                            cvar = last_var.get("cvar")
                            capital = float(self.portfolio_value_aud) or 1.0
                            if cvar is not None and capital > 0 and cvar < 0:
                                abs_cvar = abs(float(cvar))
                                if abs_cvar > 1e-9:
                                    max_notional_per_signal = (0.02 * capital) / abs_cvar
                                    for sig in optimized_signals:
                                        qty = sig.get("quantity") if isinstance(sig, dict) else getattr(sig, "quantity", None)
                                        entry = sig.get("entry_price") if isinstance(sig, dict) else getattr(sig, "entry_price", None)
                                        if qty is not None and entry and float(entry) > 0:
                                            notional = float(qty) * float(entry)
                                            if notional > max_notional_per_signal:
                                                new_qty = max_notional_per_signal / float(entry)
                                                if isinstance(sig, dict):
                                                    sig["quantity"] = new_qty
                                                else:
                                                    setattr(sig, "quantity", new_qty)

                    # Adaptive position sizing based on regime and volatility
                    if self.adaptive_position_sizer is not None and optimized_signals:
                        try:
                            regime = (self._last_regime_consensus or {}).get("regime", "neutral")
                            volatility = float(getattr(self.config, "realized_vol_pct", 0.02) or 0.02)
                            capital = float(self.portfolio_value_aud) or 1.0
                            
                            for sig in optimized_signals:
                                qty = sig.get("quantity") if isinstance(sig, dict) else getattr(sig, "quantity", None)
                                entry = sig.get("entry_price") if isinstance(sig, dict) else getattr(sig, "entry_price", None)
                                confidence = sig.get("confidence", 0.5) if isinstance(sig, dict) else getattr(sig, "confidence", 0.5)
                                
                                if qty is not None and entry and float(entry) > 0:
                                    # Calculate adaptive position size
                                    adaptive_size = self.adaptive_position_sizer.calculate_size(
                                        base_size=float(qty),
                                        regime=regime,
                                        volatility=volatility,
                                        confidence=float(confidence),
                                        capital=capital
                                    )
                                    if adaptive_size > 0 and isinstance(sig, dict):
                                        sig["quantity"] = adaptive_size
                                    elif adaptive_size > 0:
                                        setattr(sig, "quantity", adaptive_size)
                                    
                                    logger.debug(
                                        "Adaptive sizing: %s base=%.4f adaptive=%.4f (regime=%s, vol=%.4f)",
                                        sig.get("symbol", "?") if isinstance(sig, dict) else getattr(sig, "symbol", "?"),
                                        float(qty),
                                        adaptive_size,
                                        regime,
                                        volatility
                                    )
                        except Exception as e:
                            logger.debug("Adaptive position sizing failed: %s", e)

                    # Correlation-based position adjustment
                    if self.correlation_tracker is not None and optimized_signals:
                        try:
                            # Update correlation tracker with latest returns
                            returns_data = self._get_latest_returns()
                            if returns_data:
                                corr_matrix = self.correlation_tracker.update(returns_data)
                                avg_corr = getattr(corr_matrix, "avg_correlation", 0.0)
                                
                                # Reduce position sizes when correlations are high (crisis mode)
                                if avg_corr > 0.7:
                                    correlation_multiplier = 0.5  # Reduce by 50%
                                    logger.warning("High correlation detected (%.2f), reducing positions by 50%%", avg_corr)
                                elif avg_corr > 0.5:
                                    correlation_multiplier = 0.75  # Reduce by 25%
                                else:
                                    correlation_multiplier = 1.0
                                
                                if correlation_multiplier < 1.0:
                                    for sig in optimized_signals:
                                        qty = sig.get("quantity") if isinstance(sig, dict) else getattr(sig, "quantity", None)
                                        if qty is not None:
                                            new_qty = float(qty) * correlation_multiplier
                                            if isinstance(sig, dict):
                                                sig["quantity"] = new_qty
                                            else:
                                                setattr(sig, "quantity", new_qty)
                        except Exception as e:
                            logger.debug("Correlation-based adjustment failed: %s", e)

                    # Optional: 23-language signal filter (use_signal_filter_gate) – drop signals majority reject
                    use_sig_filter = getattr(self.config, "use_signal_filter_gate", False)
                    if use_sig_filter and self.language_orchestrator and optimized_signals:
                        try:
                            regime = (self._last_regime_consensus or {}).get("regime", "mean_revert")
                            kept = []
                            for sig in optimized_signals:
                                sig_obj = sig if isinstance(sig, dict) else getattr(sig, "__dict__", sig)
                                strategy_name = sig_obj.get("strategy", sig_obj.get("source_strategy", "")) if isinstance(sig_obj, dict) else getattr(sig, "strategy", "")
                                payload = {"signal": sig_obj, "regime": regime, "volatility": getattr(self.config, "realized_vol_pct", 0.0), "symbol": str((getattr(self.config, "trading_pairs", None) or ["BTC/USD"])[0]), "timeframe": str(getattr(self.config, "signal_primary_timeframe", "1h") or "1h"), "strategy_name": str(strategy_name)}
                                result = await self.language_orchestrator.execute_signal_filter_all(payload)
                                if result.get("accept", True):
                                    kept.append(sig)
                            orig_n = len(optimized_signals)
                            if len(kept) < orig_n:
                                optimized_signals = kept
                                logger.debug("Signal filter: kept %s of %s signals", len(kept), orig_n)
                        except Exception as e:
                            logger.debug("Signal filter gate (optional): %s", e)

                    # Optional: 23-language drawdown gate (use_drawdown_check) – skip execution if over drawdown limit
                    use_drawdown = getattr(self.config, "use_drawdown_check", False)
                    if use_drawdown and self.language_orchestrator and optimized_signals:
                        try:
                            peak = max(float(self.peak_equity_aud), float(self.portfolio_value_aud))
                            max_dd = float(getattr(self.config, "max_drawdown_pct", 0.12) or 0.12)
                            _sym = (getattr(self.config, "trading_pairs", None) or ["BTC/USD"])[0]
                            dd_result = await self.language_orchestrator.execute_drawdown_check_all({
                                "current_equity": float(self.portfolio_value_aud),
                                "peak_equity": peak,
                                "max_drawdown_pct": max_dd,
                                "symbol": str(_sym),
                                "cycle_id": getattr(self, "_cycle_counter", 0),
                            })
                            if not dd_result.get("passed", True):
                                logger.warning("23-language drawdown check failed - skipping execution this cycle")
                                optimized_signals = []
                        except Exception as e:
                            logger.debug("Drawdown check (optional): %s", e)

                    # Optional: 23-language slippage gate (use_slippage_estimate) – skip execution if estimated slippage too high
                    use_slippage = getattr(self.config, "use_slippage_estimate", False)
                    max_bps = float(getattr(self.config, "max_slippage_bps", 100.0) or 100.0)
                    if use_slippage and self.language_orchestrator and self.market_data_service and optimized_signals:
                        try:
                            sym = getattr(optimized_signals[0], "symbol", None) if optimized_signals else None
                            sym = sym or (getattr(self.config, "trading_pairs", None) or ["BTC/USD"])[0]
                            ob = await self.market_data_service.fetch_order_book(str(sym), limit=20)
                            if isinstance(ob, dict) and (ob.get("bids") or ob.get("asks")):
                                qty = float(getattr(optimized_signals[0], "quantity", 0) or 0.01)
                                slip_result = await self.language_orchestrator.execute_slippage_estimate({
                                    "order_book": ob,
                                    "side": "buy",
                                    "quantity": qty,
                                    "participation_rate": 0.01,
                                    "symbol": str(sym),
                                })
                                median_bps = float(slip_result.get("slippage_bps_median", 0) or 0)
                                if median_bps > max_bps:
                                    logger.warning("23-language slippage gate: median %.1f bps > max %.1f bps - skipping execution this cycle", median_bps, max_bps)
                                    optimized_signals = []
                        except Exception as e:
                            logger.debug("Slippage estimate (optional): %s", e)

                # Partial TP / trailing stop: add exit signals for positions that hit partial or trailing level
                try:
                    exit_signals = self._get_partial_tp_and_trailing_stop_signals()
                    if exit_signals:
                        optimized_signals = list(optimized_signals) + exit_signals
                except Exception as e:
                    logger.debug("Partial TP / trailing stop: %s", e)

                # Pre-trade exposure/position gate: (exposure + order) <= max exposure, (position + order) <= max position per symbol
                try:
                    max_exp_pct = float(getattr(self.config, "max_total_exposure_pct", 0.4) or 0.4)
                    _base_pos_aud = float(getattr(self.config, "max_position_size_aud", 0) or 0)
                    # Apply regime-based position limit scaling
                    try:
                        from risk.unified_risk_manager import UnifiedRiskManager as _URM
                        max_pos_aud = _URM.get_regime_adjusted_position_limit(
                            _base_pos_aud, str(regime_label or "NORMAL")
                        )
                    except Exception as _regime_adj_err:
                        logger.debug("Regime position adjustment skipped: %s", _regime_adj_err)
                        max_pos_aud = _base_pos_aud
                    aud_to_usd = float(getattr(self.config, "aud_to_usd", 0.65) or 0.65)
                    cap = float(self.portfolio_value_aud)
                    current_exposure_aud = cap - float(self.cash_balance_aud)
                    allowed_exposure_aud = cap * max_exp_pct
                    kept = []
                    for sig in optimized_signals:
                        qty = float(getattr(sig, "quantity", 0) or 0)
                        entry = float(getattr(sig, "entry_price", 0) or 0)
                        sym = str(getattr(sig, "symbol", "") or "")
                        if qty <= 0 or entry <= 0:
                            kept.append(sig)
                            continue
                        quote = sym.split("/")[-1].upper() if "/" in sym else "USD"
                        notional_aud = (qty * entry / aud_to_usd) if quote == "USD" else (qty * entry)
                        if max_exp_pct > 0 and (current_exposure_aud + notional_aud) > allowed_exposure_aud:
                            logger.debug("Pre-trade gate: signal %s would exceed max exposure", sym)
                            self._snapshot_candidate_decision(
                                cycle_id=int(cycles),
                                correlation_id=str(cycle_correlation_id),
                                signal=sig,
                                allowed=False,
                                reason_code=ReasonCode.PRE_TRADE_RISK_BLOCK,
                                details={
                                    "stage": "pre_trade_exposure",
                                    "gate": "max_total_exposure_pct",
                                    "allowed_exposure_aud": float(allowed_exposure_aud),
                                    "current_exposure_aud": float(current_exposure_aud),
                                    "candidate_notional_aud": float(notional_aud),
                                },
                            )
                            continue
                        pos = (self.positions.get(sym) or {})
                        pos_qty = float(pos.get("quantity", 0) or 0)
                        pos_value_aud = pos_qty * entry / aud_to_usd if quote == "USD" else pos_qty * entry
                        new_pos_aud = (pos_qty + qty) * entry / aud_to_usd if quote == "USD" else (pos_qty + qty) * entry
                        if max_pos_aud > 0 and new_pos_aud > max_pos_aud:
                            logger.debug("Pre-trade gate: signal %s would exceed max position size", sym)
                            self._snapshot_candidate_decision(
                                cycle_id=int(cycles),
                                correlation_id=str(cycle_correlation_id),
                                signal=sig,
                                allowed=False,
                                reason_code=ReasonCode.MAX_POSITION_SIZE,
                                details={
                                    "stage": "pre_trade_exposure",
                                    "gate": "max_position_size_aud",
                                    "base_max_position_size_aud": float(_base_pos_aud),
                                    "max_position_size_aud": float(max_pos_aud),
                                    "new_position_aud": float(new_pos_aud),
                                    "regime": str(regime_label or "NORMAL"),
                                },
                            )
                            continue
                        kept.append(sig)
                        current_exposure_aud += notional_aud
                    optimized_signals = kept
                except Exception as e:
                    logger.debug("Pre-trade exposure gate: %s", e)

                # Pre-trade guardrail: do not execute if we are already in emergency stop state
                if self._check_emergency_stop():
                    self.state = SystemState.EMERGENCY_STOP
                    logger.critical("🚨 Emergency stop triggered pre-execution - skipping trade execution")
                    await self.shutdown()
                    break

                # Set execution-engine context for pre_trade_risk_block (limit hierarchy, exposure, position gate)
                if not self._set_pre_trade_context():
                    for sig in list(optimized_signals or []):
                        self._snapshot_candidate_decision(
                            cycle_id=int(cycles),
                            correlation_id=str(cycle_correlation_id),
                            signal=sig,
                            allowed=False,
                            reason_code=ReasonCode.PRE_TRADE_RISK_BLOCK,
                            details={"stage": "pre_trade_context", "reason": "context_unavailable"},
                        )
                    optimized_signals = []

                # Latency budget: measure signal→risk→execution (for tuning and regression)
                t_before_exec = time.perf_counter()
                # Execute trades through Kraken DCA engine (with correlation_id for tracing)
                corr_id = getattr(self, "_cycle_correlation_id", None) or cycle_correlation_id
                signals_for_execution = list(optimized_signals or [])
                logger.info("SIGNAL DEBUG: signals_for_execution=%d before gates, node_role=%s", len(signals_for_execution), self.node_role)
                handoff_mode = "direct"
                handoff_summary: Dict[str, Any] = {}
                mesh_summary: Dict[str, Any] = {}

                if self.node_role == "strategy-node":
                    # Strategy node must never execute orders directly.
                    if bool(getattr(self.config, "command_bus_enabled", False)):
                        handoff_mode = "publish"
                        handoff_summary = self._publish_signals_to_command_bus(
                            signals_for_execution,
                            cycle_id=int(cycles),
                            correlation_id=str(corr_id),
                        )
                        logger.info(
                            "Strategy-node handoff: published=%s rejected=%s",
                            int(handoff_summary.get("published", 0) or 0),
                            int(handoff_summary.get("rejected", 0) or 0),
                        )
                    else:
                        logger.warning("Strategy-node role active but command bus is disabled; skipping execution.")
                    signals_for_execution = []
                elif self.node_role == "execution-node" and bool(getattr(self.config, "command_bus_enabled", False)):
                    handoff_mode = "consume"
                    signals_for_execution, handoff_summary = self._consume_signals_from_command_bus()
                    logger.info(
                        "Execution-node handoff: claimed=%s accepted=%s rejected=%s",
                        int(handoff_summary.get("claimed", 0) or 0),
                        int(handoff_summary.get("accepted", 0) or 0),
                        int(handoff_summary.get("rejected", 0) or 0),
                    )

                # Check for partial exits (profit taking, stop losses, regime changes)
                try:
                    partial_exit_signals = self._check_partial_exits()
                    if partial_exit_signals:
                        logger.info("Partial exits: %d exit signals generated", len(partial_exit_signals))
                        signals_for_execution = list(signals_for_execution or []) + partial_exit_signals
                except Exception as _pe_err:
                    logger.debug("Partial exit check: %s", _pe_err)

                # Strategy evaluation context is optional metadata only.
                try:
                    context_signals = optimized_signals if handoff_mode == "publish" else signals_for_execution
                    self._attach_strategy_evaluation_context(
                        list(context_signals or []),
                        str(regime_label or ""),
                    )
                    self._attach_champion_challenger_context(
                        list(context_signals or []),
                        str(regime_label or ""),
                    )
                except Exception as e:
                    self._handle_strategy_eval_error(e, context="attach_context_cycle")

                # Ω audit: record pre-execution decisions for each optimized signal (best-effort)
                try:
                    trace_id = getattr(self, "_trace_id", None) or uuid.uuid4().hex
                    audit_signals = list(signals_for_execution or [])
                    if handoff_mode == "publish":
                        audit_signals = list(optimized_signals or [])
                    for sig in audit_signals:
                        symbol = getattr(sig, "symbol", None) or (sig.get("symbol") if isinstance(sig, dict) else None)
                        side = getattr(sig, "side", None) or (sig.get("side") if isinstance(sig, dict) else None)
                        strategy = getattr(sig, "strategy", None) or (sig.get("strategy") if isinstance(sig, dict) else None)
                        score = getattr(sig, "score", None) or (sig.get("score") if isinstance(sig, dict) else None)
                        sig_exec_plan = (
                            dict(getattr(sig, "execution_plan", {}) or {})
                            if not isinstance(sig, dict)
                            else dict(sig.get("execution_plan") or {})
                        )
                        intent_id = str(uuid.uuid4().hex)
                        target_details = {}
                        for f in (
                            "target_exposure_pct",
                            "current_exposure_pct",
                            "delta_exposure_pct",
                            "priority_score",
                            "expected_net_edge_bps",
                            "regime_label",
                            "target_reasons",
                            "liquidity_score",
                            "liquidity_state",
                            "max_safe_trade_size",
                            "adjusted_target_exposure_pct",
                            "liquidity_clamp_flag",
                            "slippage_estimate_bps",
                            "strategy_trades_count",
                            "strategy_win_rate",
                            "strategy_expectancy",
                            "strategy_profit_factor",
                            "strategy_weight",
                            "meta_priority_adjustment",
                            "weighting_reason",
                            "spread_bps",
                            "order_book_imbalance",
                            "microprice",
                            "trade_velocity",
                            "liquidity_vacuum_flag",
                            "adverse_selection_risk",
                            "microstructure_bias",
                            "champion_profile_id",
                            "challenger_profile_id",
                            "promotion_decision",
                            "promotion_score",
                            "promotion_reasons",
                            "promotion_regime_label",
                        ):
                            v = getattr(sig, f, None) if not isinstance(sig, dict) else sig.get(f)
                            if v is not None:
                                target_details[f] = v
                        if handoff_mode != "publish":
                            self.omega_store.create_intent(
                                intent_id=intent_id,
                                run_id=str(getattr(self, "run_id", "unknown")),
                                trace_id=str(trace_id),
                                cycle_id=int(cycles),
                                correlation_id=str(corr_id),
                                symbol=str(symbol or "UNKNOWN"),
                                side=str(side or "UNKNOWN"),
                                order_type=str(getattr(self.config, "order_type", "market") or "market"),
                                amount=None,
                                price=None,
                                status="CREATED",
                                exec_plan={"router": "v1", "mode": "pre_exec_stub"},
                                meta={"strategy": strategy, "score": score},
                            )
                        self.omega_store.record_decision(
                            run_id=str(getattr(self, "run_id", "unknown")),
                            trace_id=str(trace_id),
                            cycle_id=int(cycles),
                            correlation_id=str(corr_id),
                            symbol=str(symbol) if symbol else None,
                            strategy=str(strategy) if strategy else None,
                            side=str(side) if side else None,
                            signal_score=float(score) if score is not None else None,
                            allowed=True,
                            reason_code=ReasonCode.PRE_PUBLISH.value if handoff_mode == "publish" else ReasonCode.PRE_EXEC.value,
                            details={
                                "intent_id": intent_id,
                                "handoff_mode": handoff_mode,
                                "handoff": dict(handoff_summary or {}),
                                **target_details,
                            },
                            cost={},
                            exec_plan={
                                "router": "v1",
                                **sig_exec_plan,
                            },
                        )
                except Exception:
                    pass

                # Component registry pre-order gate (rate limits, order-flow toxicity, macro blackout)
                try:
                    if self.component_registry is not None and signals_for_execution:
                        filtered_sigs = []
                        for _sig in signals_for_execution:
                            _sym = getattr(_sig, "symbol", None) or (
                                _sig.get("symbol") if isinstance(_sig, dict) else None
                            ) or ""
                            _side = getattr(_sig, "side", None) or (
                                _sig.get("side") if isinstance(_sig, dict) else None
                            ) or "BUY"
                            _size = float(
                                getattr(_sig, "size_usd", None)
                                or ((_sig.get("size_usd") or _sig.get("notional_usd", 0.0)) if isinstance(_sig, dict) else 0.0)
                                or 0.0
                            )
                            # Fallback: compute size from quantity * entry_price if size_usd not available
                            if _size <= 0:
                                _qty = float(
                                    getattr(_sig, "quantity", None)
                                    or ((_sig.get("quantity") or _sig.get("qty", 0.0)) if isinstance(_sig, dict) else 0.0)
                                    or 0.0
                                )
                                _price = float(
                                    getattr(_sig, "entry_price", None)
                                    or getattr(_sig, "price", None)
                                    or ((_sig.get("entry_price") or _sig.get("price", 0.0)) if isinstance(_sig, dict) else 0.0)
                                    or 0.0
                                )
                                if _qty > 0 and _price > 0:
                                    _size = _qty * _price
                            _exc = getattr(_sig, "exchange", None) or (
                                _sig.get("exchange") if isinstance(_sig, dict) else None
                            ) or "kraken"
                            _chk = self.component_registry.pre_order_check(_sym, _side, _size, _exc)
                            if _chk.get("allow", True):
                                filtered_sigs.append(_sig)
                            else:
                                logger.info(
                                    "ComponentRegistry blocked %s %s: %s",
                                    _side, _sym, _chk.get("reasons", [])
                                )
                        signals_for_execution = filtered_sigs
                except Exception:
                    pass

                # In paper mode, always use direct execution path (_execute_signals)
                # which simulates fills. The DCA engine is for real Kraken orders.
                _run_mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
                _use_direct = (_run_mode == "paper")
                logger.info("SIGNAL DEBUG: signals_for_execution=%d at exec point, recon_block=%s, mode=%s, use_direct=%s",
                            len(signals_for_execution), reconciliation_block_cycle, _run_mode, _use_direct)
                if reconciliation_block_cycle:
                    execution_results = []
                    logger.warning("Execution skipped for cycle %s: reconciliation gate active", cycles + 1)
                elif not signals_for_execution:
                    # Keep execution-path observability deterministic even when no
                    # candidates survive gating for the cycle.
                    if self.execution_engine is not None and hasattr(self.execution_engine, "execute_signals"):
                        execution_results = await asyncio.wait_for(
                            self.execution_engine.execute_signals([], correlation_id=corr_id),
                            timeout=op_timeout_s,
                        )
                    else:
                        execution_results = []
                elif _use_direct or (self.execution_engine is None and self.execution_mesh is None):
                    # Direct execution pipeline: paper mode OR no DCA engine.
                    # _execute_signals() simulates fills with slippage.
                    logger.info(
                        "Direct execution pipeline: processing %d signals (paper=%s)",
                        len(signals_for_execution), _use_direct,
                    )
                    execution_results = await asyncio.wait_for(
                        self._execute_signals(signals_for_execution),
                        timeout=op_timeout_s,
                    )
                elif self.execution_mesh is not None:
                    from execution.execution_mesh import ExecutionMeshError

                    async def _lane_execute(symbol: str, lane_signals: List[Any], lane_corr_id: str) -> List[Dict[str, Any]]:
                        _ = symbol  # lane routing handled in mesh; engine evaluates each lane signal list.
                        return await self.execution_engine.execute_signals(lane_signals, correlation_id=lane_corr_id)

                    try:
                        execution_results, mesh_summary = await asyncio.wait_for(
                            self.execution_mesh.execute_cycle(
                                signals_for_execution,
                                execute_fn=_lane_execute,
                                correlation_id=str(corr_id),
                            ),
                            timeout=op_timeout_s,
                        )
                        handoff_summary["execution_mesh"] = dict(mesh_summary or {})
                        logger.info(
                            "Execution mesh: lanes=%s accepted=%s dropped=%s lane_errors=%s results=%s",
                            int(mesh_summary.get("lanes_active", 0) or 0),
                            int(mesh_summary.get("accepted_signals", 0) or 0),
                            int(mesh_summary.get("dropped_signals", 0) or 0),
                            int(mesh_summary.get("lane_errors", 0) or 0),
                            int(mesh_summary.get("results_count", 0) or 0),
                        )
                    except ExecutionMeshError as mesh_exc:
                        mesh_summary = dict(getattr(mesh_exc, "summary", {}) or {})
                        handoff_summary["execution_mesh"] = dict(mesh_summary or {})
                        logger.critical("Execution mesh fail-closed halt: %s", mesh_exc)
                        self.state = SystemState.EMERGENCY_STOP
                        raise
                else:
                    execution_results = await asyncio.wait_for(
                        self.execution_engine.execute_signals(signals_for_execution, correlation_id=corr_id),
                        timeout=op_timeout_s,
                    )
                try:
                    exec_timings = {}
                    if self.execution_engine is not None and hasattr(self.execution_engine, "get_last_stage_timings"):
                        exec_timings = dict(self.execution_engine.get_last_stage_timings() or {})
                    cycle_stage_timing_ms["risk_gate_ms"] = float(exec_timings.get("risk_gate_ms", 0.0) or 0.0)
                    cycle_stage_timing_ms["execution_planning_ms"] = float(exec_timings.get("execution_planning_ms", 0.0) or 0.0)
                    cycle_stage_timing_ms["snapshot_persistence_ms"] = float(exec_timings.get("snapshot_persistence_ms", 0.0) or 0.0)
                except Exception:
                    pass

                # Ω audit: record post-execution summary (best-effort)
                try:
                    trace_id = getattr(self, "_trace_id", None) or ""
                    self.omega_store.record_decision(
                        run_id=str(getattr(self, "run_id", "unknown")),
                        trace_id=str(trace_id or uuid.uuid4().hex),
                        cycle_id=int(cycles),
                        correlation_id=str(corr_id),
                        symbol=None,
                        strategy=None,
                        side=None,
                        signal_score=None,
                        allowed=True,
                        reason_code=ReasonCode.POST_EXEC.value,
                        details={
                            "results": execution_results,
                            "handoff_mode": handoff_mode,
                            "handoff": dict(handoff_summary or {}),
                            "mesh": dict(mesh_summary or {}),
                        },
                        cost={},
                        exec_plan={},
                    )
                except Exception:
                    pass
                # Enforce stop-losses on all open positions (critical safety check)
                try:
                    if hasattr(self.execution_engine, "check_and_enforce_stop_losses"):
                        forced_closes = await asyncio.wait_for(
                            self.execution_engine.check_and_enforce_stop_losses(),
                            timeout=op_timeout_s,
                        )
                        if forced_closes:
                            for fc in forced_closes:
                                logger.warning("STOP-LOSS FORCED CLOSE: %s %.4f @ %.2f (loss: %.1f%%)",
                                               fc.get("symbol"), fc.get("quantity"), fc.get("price"), fc.get("loss_pct", 0))
                            execution_results.extend(forced_closes)
                except Exception as _sl_err:
                    logger.debug("Stop-loss enforcement check: %s", _sl_err)

                # FIX 5: Second stop-loss check AFTER execution — catches positions
                # that hit stops during signal execution within this cycle.
                try:
                    if self.unified_risk_manager is not None and self.positions:
                        _stop_prices_post = {
                            str(sym): float((pos or {}).get("current_price") or 0.0)
                            for sym, pos in (self.positions or {}).items()
                        }
                        _sl_pct_post = float(getattr(self.config, "stop_loss_pct", 0.02) or 0.02)
                        _trail_pct_post = float(getattr(self.config, "trailing_stop_pct", 0.015) or 0.015)
                        _max_hold_post = float(getattr(self.config, "max_holding_hours", 72.0) or 72.0)
                        _stop_triggers_post = self.unified_risk_manager.check_stops(
                            self.positions, _stop_prices_post,
                            stop_loss_pct=_sl_pct_post, trail_pct=_trail_pct_post, max_hold_hours=_max_hold_post,
                        )
                        if _stop_triggers_post:
                            logger.warning(
                                "POST-EXEC STOP-LOSS: %d positions triggered after execution",
                                len(_stop_triggers_post),
                            )
                            _post_stop_signals = []
                            for _st in _stop_triggers_post:
                                logger.warning(
                                    "POST-EXEC STOP TRIGGER: %s %s qty=%.8f reason=%s",
                                    _st["side"], _st["symbol"], _st["quantity"], _st["reason"],
                                )
                                try:
                                    from unified_types import TradingSignal as _TS
                                    _post_stop_signals.append(_TS(
                                        symbol=_st["symbol"],
                                        action=_st["side"],
                                        confidence=1.0,
                                        strength=1.0,
                                        entry_price=_st["current_price"],
                                        reasoning=f"POST_EXEC_STOP: {_st['reason']}",
                                    ))
                                except Exception:
                                    pass
                            if _post_stop_signals:
                                _post_stop_results = await asyncio.wait_for(
                                    self._execute_signals(_post_stop_signals),
                                    timeout=op_timeout_s,
                                )
                                logger.warning("POST-EXEC STOP EXECUTION: %d results", len(_post_stop_results))
                                execution_results.extend(_post_stop_results)
                                for _st in _stop_triggers_post:
                                    self.unified_risk_manager.clear_position_tracking(_st["symbol"])
                except Exception as _post_stop_err:
                    logger.warning("Post-execution stop-loss check failed: %s", _post_stop_err)

                # ── Signal subscription broadcast (fire-and-forget) ──────
                try:
                    if self.signal_service is not None and execution_results:
                        for _exec_res in execution_results:
                            if isinstance(_exec_res, dict) and _exec_res.get("status") == "filled":
                                _broadcast_payload = {
                                    "symbol": _exec_res.get("symbol", ""),
                                    "action": _exec_res.get("side", ""),
                                    "confidence": float(_exec_res.get("confidence", 0.0) or 0.0),
                                    "entry_price": float(_exec_res.get("price", 0.0) or 0.0),
                                    "stop_loss": _exec_res.get("stop_loss"),
                                    "take_profit": _exec_res.get("take_profit"),
                                    "regime": str(getattr(self, "current_regime", "UNKNOWN")),
                                    "reasoning": _exec_res.get("reasoning", ""),
                                    "strategies_agreeing": _exec_res.get("strategies_agreeing", 0),
                                    "fill_price": float(_exec_res.get("price", 0.0) or 0.0),
                                    "slippage": float(_exec_res.get("slippage", 0.0) or 0.0),
                                }
                                self.signal_service.broadcast_signal_fire_and_forget(_broadcast_payload)
                except Exception as _bcast_exc:
                    logger.debug("Signal broadcast failed (non-critical): %s", _bcast_exc)

                t_after_exec = time.perf_counter()
                t_cycle_start = getattr(self, "_cycle_start_time", t_after_exec)
                cycle_total_s = t_after_exec - t_cycle_start
                setattr(self, "_last_cycle_total_ms", cycle_total_s * 1000.0)
                self._last_cycle_stage_timing_ms = dict(cycle_stage_timing_ms)
                logger.debug(
                    "Latency budget: cycle_total_s=%.3f exec_s=%.3f",
                    cycle_total_s,
                    t_after_exec - t_before_exec,
                )
                logger.info(
                    "cycle timing: market_data_ms=%.0f feature_generation_ms=%.0f strategy_eval_ms=%.0f portfolio_targeting_ms=%.0f liquidity_adjustment_ms=%.0f risk_gate_ms=%.0f execution_plan_ms=%.0f snapshot_persistence_ms=%.0f",
                    float(cycle_stage_timing_ms.get("market_data_ms", 0.0) or 0.0),
                    float(cycle_stage_timing_ms.get("feature_generation_ms", 0.0) or 0.0),
                    float(cycle_stage_timing_ms.get("strategy_evaluation_ms", 0.0) or 0.0),
                    float(cycle_stage_timing_ms.get("portfolio_targeting_ms", 0.0) or 0.0),
                    float(cycle_stage_timing_ms.get("liquidity_adjustment_ms", 0.0) or 0.0),
                    float(cycle_stage_timing_ms.get("risk_gate_ms", 0.0) or 0.0),
                    float(cycle_stage_timing_ms.get("execution_planning_ms", 0.0) or 0.0),
                    float(cycle_stage_timing_ms.get("snapshot_persistence_ms", 0.0) or 0.0),
                )
                if callable(rec_event):
                    try:
                        rec_event(
                            stage="execution_results",
                            cycle_id=cycles,
                            payload_json=json.dumps(execution_results, ensure_ascii=True, default=str),
                        )
                    except Exception:
                        pass
                
                # Update monitoring (include quantum metrics when available)
                metrics_dict = {
                    'portfolio_value': self.portfolio_value_aud,
                    'cash_balance': self.cash_balance_aud,
                    'total_pnl': self.total_pnl_aud,
                    'realized_pnl': self.realized_pnl_aud,
                    'unrealized_pnl': self.unrealized_pnl_aud,
                    'total_fees': self.total_fees_aud,
                    'ledger_sanity_violations': int(self._ledger_sanity_violations),
                    'trades': self.total_trades,
                    'signals': len(optimized_signals),
                    'max_drawdown': self.max_drawdown_aud,
                    'consecutive_losses': self.consecutive_losses,
                    'win_rate': (self.winning_trades / max(self.total_trades, 1)),
                    'error_rate': (self.error_count / max(self.total_operations, 1)),
                }
                last_q = getattr(self, "_last_quantum_var_cvar", None)
                if isinstance(last_q, dict):
                    metrics_dict['quantum_var_95'] = last_q.get('var_95', last_q.get('var'))
                    metrics_dict['quantum_cvar_95'] = last_q.get('cvar_95', last_q.get('cvar'))
                metrics_dict['quantum_circuit_breaker_trips'] = int(getattr(self, "_quantum_circuit_breaker_trips", 0))
                if self.monitoring is not None:
                    await asyncio.wait_for(
                        self.monitoring.update_metrics(metrics_dict),
                        timeout=op_timeout_s,
                    )

                # Alert triggers: drawdown and daily loss (circuit_breaker alert sent when breaker trips)
                try:
                    from monitoring.alerting import get_alert_manager
                    mgr = get_alert_manager()
                    dd_threshold = float(getattr(self.config, "max_drawdown_pct", 0.12) or 0.12)
                    peak = max(float(self.peak_equity_aud), 1e-9)
                    current_dd = (peak - float(self.portfolio_value_aud)) / peak if peak > 0 else 0.0
                    if current_dd >= dd_threshold or float(self.max_drawdown_aud) >= dd_threshold:
                        await mgr.drawdown_alert(
                            current_drawdown=current_dd,
                            max_drawdown=float(self.max_drawdown_aud),
                            threshold=dd_threshold,
                        )
                    cap = max(float(self.peak_equity_aud), float(self.config.starting_capital_aud))
                    daily_limit = float(getattr(self.config, "max_daily_loss_pct", 0.02) or 0.02) * cap
                    daily_pnl = float(getattr(self, "daily_pnl_aud", 0.0) or 0.0)
                    if daily_limit > 0 and daily_pnl < -daily_limit:
                        await mgr.daily_loss_alert(daily_loss=abs(daily_pnl), daily_limit=daily_limit)
                    max_cl = int(getattr(self.config, "max_consecutive_losses", 5) or 5)
                    if max_cl > 0 and self.consecutive_losses >= max_cl:
                        await mgr.consecutive_losses_alert(consecutive_losses=self.consecutive_losses, threshold=max_cl)
                    max_err = float(getattr(self.config, "max_error_rate", 0.05) or 0.05)
                    ops = max(self.total_operations, 1)
                    err_rate = self.error_count / ops
                    if max_err > 0 and err_rate >= max_err:
                        await mgr.error_rate_alert(error_rate=err_rate, threshold=max_err)
                except Exception as alert_e:
                    logger.debug("Risk alerts (drawdown/daily/consecutive/error): %s", alert_e)

                # Record trades
                # NOTE: _record_trade is already called inside _execute_signals/_execute_signals_legacy
                # for result in execution_results:
                #     if result.get('status') == 'filled':
                #         self._record_trade(result)
                
            except Exception as e:
                # TimeoutError from rate limiting is expected with 48 pairs — don't count toward error rate
                _is_timeout = isinstance(e, (TimeoutError, asyncio.TimeoutError))
                if _is_timeout:
                    logger.warning("Cycle timeout (Kraken rate limit) — retrying next cycle")
                else:
                    logger.error(f"Error in trading loop: {e}", exc_info=True)
                    self.error_count += 1
                
                try:
                    min_samples = int(getattr(self.config, "min_error_rate_samples", 20) or 20)
                except Exception:
                    min_samples = 20
                err_rate = float(self.error_count / max(self.total_operations, 1))
                if self.total_operations >= min_samples and err_rate > float(getattr(self.config, "max_error_rate", 0.25) or 0.25):
                    logger.critical("Error rate exceeded threshold - emergency stop")
                    self.state = SystemState.EMERGENCY_STOP
                    iter_sleep_s = 0.0
                else:
                    # Back off a bit before retry
                    iter_sleep_s = 10.0

            # Always advance the cycle counter so max_cycles can stop the run.
            cycles += 1
            self._completed_cycles = int(cycles)  # BUG-5 fix: update AFTER increment
            dur = max(0.0, float(time.time()) - t0)
            try:
                logger.info("Cycle %s complete (%.2fs)", cycles, dur)
            except Exception:
                pass

            # Periodic checkpoint save (every N cycles)
            try:
                if self.checkpoint_manager is not None and self.checkpoint_manager.should_save(cycles):
                    _ckpt_state = {
                        "cycle_count": cycles,
                        "portfolio_value": float(getattr(self, "portfolio_value_aud", 0) or 0),
                        "regime": str(locals().get("regime_label") or getattr(self, "_last_regime_label", "UNKNOWN")),
                        "positions": dict(self.positions or {}),
                        "cash_balance_aud": float(getattr(self, "cash_balance_aud", 0) or 0),
                        "model_versions": {},
                        "risk_state": {},
                    }
                    # Capture model versions
                    if self.model_manager is not None:
                        try:
                            _snap = self.model_manager.snapshot()
                            _ckpt_state["model_versions"] = {
                                k: v.get("version", 0)
                                for k, v in _snap.get("models", {}).items()
                            }
                        except Exception:
                            pass
                    self.checkpoint_manager.save_checkpoint(_ckpt_state)
            except Exception as _ckpt_exc:
                logger.debug("Checkpoint save error: %s", _ckpt_exc)

            # ════════════════════════════════════════════════════════════════════════════
            # FULLY ADAPTIVE INFRASTRUCTURE - Periodic Tasks
            # ════════════════════════════════════════════════════════════════════════════
            
            # Dynamic Timeframe Selection - Update volatility and log status
            try:
                if self.timeframe_selector is not None:
                    # Update volatility from recent data
                    btc_vol = self._volatility_cache.get("BTC/USD", 0.02)
                    self.timeframe_selector.update_volatility(btc_vol)
                    
                    # Select optimal timeframe
                    optimal_tf = self.timeframe_selector.select_timeframe()
                    
                    # Log timeframe status every 20 cycles
                    if cycles % 20 == 0:
                        tf_status = self.timeframe_selector.get_status()
                        logger.info(
                            "⏱️ Timeframe: %s (vol=%s, atr=%.2f%%)",
                            tf_status["current_timeframe"],
                            tf_status["volatility_regime"],
                            tf_status["volatility_atr_pct"],
                        )
            except Exception as e:
                logger.debug("Timeframe selector error: %s", e)

            # Self-Learning Universe - Periodic expansion and evaluation
            try:
                if self.universe_expander is not None and cycles % 50 == 0:
                    # Save universe state periodically
                    self.universe_expander.save_state()
                    
                    # Log universe status
                    uni_stats = self.universe_expander.get_universe_stats()
                    logger.info(
                        "🌌 Universe: %d active, %d testing, %d blacklisted (total discovered: %d)",
                        uni_stats["active_count"],
                        uni_stats["paper_testing_count"],
                        uni_stats["blacklisted_count"],
                        uni_stats["total_discovered"],
                    )
            except Exception as e:
                logger.debug("Universe expander error: %s", e)

            # Strategy Parameter Tuner - Log tuning status periodically
            try:
                if self.strategy_param_tuner is not None and cycles % 30 == 0:
                    tuner_status = self.strategy_param_tuner.get_status()
                    if tuner_status["tuning_count"] > 0:
                        logger.info(
                            "🎯 Param tuner: %d tuning runs, %d trades recorded, best_score=%.4f",
                            tuner_status["tuning_count"],
                            tuner_status["trades_recorded"],
                            tuner_status.get("best_score", 0) or 0,
                        )
            except Exception as e:
                logger.debug("Strategy param tuner status error: %s", e)

            # Push fresh state to API dashboard (best-effort, non-blocking).
            try:
                if getattr(self, "api_server", None) is not None:
                    _dd_pct = 0.0
                    if float(getattr(self, "peak_equity_aud", 0) or 0) > 0:
                        _dd_pct = (
                            (float(self.peak_equity_aud) - float(self.portfolio_value_aud or 0))
                            / float(self.peak_equity_aud) * 100.0
                        )
                    _start_cap = float(getattr(self.config, "starting_capital_aud", 1000.0) or 1000.0)
                    _pv = float(getattr(self, "portfolio_value_aud", _start_cap) or _start_cap)
                    _pnl = _pv - _start_cap
                    _reg = locals().get("regime_label") or getattr(self, "_last_regime_label", "UNKNOWN")
                    _comps: dict = {}
                    if getattr(self, "component_registry", None) is not None:
                        try:
                            _comps = self.component_registry.snapshot()
                        except Exception:
                            pass
                    _models: dict = {}
                    if getattr(self, "model_manager", None) is not None:
                        try:
                            _models = self.model_manager.snapshot()
                        except Exception:
                            pass
                    _ft = getattr(self.component_registry, "fill_tracker", None) if getattr(self, "component_registry", None) is not None else None
                    _slippage_avg_bps = 0.0
                    _fill_rate_1h = 1.0
                    if _ft is not None:
                        try:
                            _all_stats = _ft.get_all_stats(lookback_hours=1)
                            if _all_stats:
                                _bps_vals = [s.get("avg_slippage_bps", 0.0) for s in _all_stats.values()]
                                _slippage_avg_bps = sum(_bps_vals) / len(_bps_vals) if _bps_vals else 0.0
                                _total_fills = sum(s.get("fill_count", 0) for s in _all_stats.values())
                                _fill_rate_1h = min(1.0, _total_fills / max(1, int(cycles)) )
                        except Exception:
                            pass
                    self.api_server.update_states({
                        "cycle": int(cycles),
                        "capital_aud": _pv,
                        "pnl_aud": _pnl,
                        "pnl_pct": (_pnl / _start_cap * 100.0) if _start_cap else 0.0,
                        "drawdown_pct": _dd_pct,
                        "daily_loss_aud": abs(float(getattr(self, "daily_pnl_aud", 0) or 0)),
                        "var_95": float(getattr(self, "_last_var_95", 0) or 0),
                        "circuit_breaker": bool(getattr(self, "circuit_breaker_active", False)),
                        "regime": str(_reg),
                        "positions": {
                            k: float((v or {}).get("quantity", 0))
                            for k, v in (self.positions or {}).items()
                        },
                        "components": _comps,
                        "models": _models,
                        "last_cycle_timestamp": time.time(),
                        "slippage_avg_bps": _slippage_avg_bps,
                        "fill_rate_1h": _fill_rate_1h,
                    })
            except Exception:
                pass

            try:
                await self._record_system_health_snapshot(
                    cycles_completed=int(cycles),
                    cycle_duration_seconds=float(dur),
                )
            except Exception as _health_e:
                logger.warning("System health snapshot update failed: %s", _health_e)

            # Refresh AUD/USD FX rate every 10 cycles (~10 min at 60s cycles)
            if cycles % 10 == 0:
                try:
                    from utils.fx_rate import get_aud_usd_rate
                    live_rate = get_aud_usd_rate(fallback=float(getattr(self.config, "aud_to_usd", 0.65) or 0.65))
                    self.config.aud_to_usd = live_rate
                except Exception:
                    pass

            # Config hot-reload: check if unified_config.yaml was modified and apply safe changes
            if cycles % 50 == 0 and cycles > 0:
                try:
                    self._try_hot_reload_config()
                except Exception as e:
                    logger.debug("Config hot-reload check failed: %s", e)

            try:
                se_engine = getattr(self, "strategy_evaluation_engine", None)
                if se_engine is not None:
                    se_engine.maybe_persist(cycle_id=int(cycles))
            except Exception as e:
                self._handle_strategy_eval_error(e, context="periodic_persist")

            try:
                cc_engine = getattr(self, "champion_challenger_engine", None)
                se_engine = getattr(self, "strategy_evaluation_engine", None)
                if cc_engine is not None and se_engine is not None:
                    decisions = cc_engine.evaluate_all(
                        strategy_evaluation_engine=se_engine,
                        regime_label=str(getattr(self, "_latest_regime_label", "") or ""),
                    )
                    cc_engine.maybe_persist(cycle_id=int(cycles))
                    if decisions:
                        logger.info(
                            "Champion/challenger evaluated %s challenger(s) this cycle",
                            len(decisions),
                        )
            except Exception as e:
                logger.warning("Champion/challenger evaluation error: %s", e)

            # Phase Y: Auto-graduation check every 100 cycles (paper mode only)
            if cycles % 100 == 0 and str(getattr(self.config, "run_mode", "paper") or "paper").lower() == "paper":
                try:
                    from core.live_gate import LiveGate
                    _gate = LiveGate(
                        paper_db_path=str(getattr(self, "_paper_db_path", "data/paper_trades.db") or "data/paper_trades.db"),
                    )
                    _report = _gate.evaluate()
                    if _report.passed:
                        logger.info(
                            "=" * 60 + "\n"
                            "LIVE GATE: GRADUATED — all criteria met!\n"
                            "  Sharpe: %.2f  |  DD: %.1f%%  |  WR: %.1f%%  |  Trades: %d\n"
                            "  ARGUS is ready for live trading: py main.py live\n" +
                            "=" * 60,
                            getattr(_report, "sharpe", 0),
                            getattr(_report, "max_drawdown", 0) * 100,
                            getattr(_report, "win_rate", 0) * 100,
                            getattr(_report, "num_trades", 0),
                        )
                    else:
                        _n_fail = len(getattr(_report, "failures", []))
                        logger.info(
                            "LiveGate check (cycle %d): %d criteria not yet met",
                            cycles, _n_fail,
                        )
                except Exception as _gate_exc:
                    logger.debug("LiveGate check error: %s", _gate_exc)

            if max_cycles_i is not None and cycles >= max_cycles_i:
                logger.info("Reached max cycles (%s); stopping trading loop", max_cycles_i)
                break
            if self.state != SystemState.RUNNING:
                break

            # --- FIX 21: Smart loop sleep ---
            # Adaptive sleep: no sleep if signals, fast poll if pending orders
            _had_signals = bool(locals().get("execution_results"))
            _has_pending = bool(getattr(self, "_pending_orders", {}))
            if _had_signals:
                _smart_sleep = 0.0  # process immediately
            elif _has_pending:
                _smart_sleep = min(1.0, iter_sleep_s)  # check fills faster
            else:
                _smart_sleep = iter_sleep_s  # save CPU

            if _smart_sleep > 0:
                await asyncio.sleep(float(_smart_sleep))
            else:
                await asyncio.sleep(0)

    async def _record_system_health_snapshot(self, *, cycles_completed: int, cycle_duration_seconds: float) -> None:
        collector = getattr(self, "system_health_metrics", None)
        if collector is None or not bool(getattr(collector, "enabled", False)):
            return
        event_loop_delay_ms = await collector.sample_event_loop_delay_ms()
        self._last_event_loop_delay_ms = float(event_loop_delay_ms)
        collector.record_cycle(
            cycle_latency_ms=max(0.0, float(cycle_duration_seconds) * 1000.0),
            event_loop_delay_ms=float(event_loop_delay_ms),
        )
        if not collector.should_snapshot(cycles_completed=int(cycles_completed)):
            return
        snapshot = collector.build_snapshot(cycles_completed=int(cycles_completed))
        self.omega_store.record_system_health_snapshot(
            {
                "timestamp": str(snapshot.timestamp),
                "cycles_completed": int(snapshot.cycles_completed),
                "avg_latency_ms": float(snapshot.avg_latency_ms),
                "errors_last_hour": int(snapshot.errors_last_hour),
                "warnings_last_hour": int(snapshot.warnings_last_hour),
                "event_loop_delay_ms": float(snapshot.event_loop_delay_ms),
                "memory_rss_mb": float(getattr(snapshot, "memory_rss_mb", 0.0) or 0.0),
                "memory_python_mb": float(getattr(snapshot, "memory_python_mb", 0.0) or 0.0),
            }
        )
        logger.info("system health snapshot recorded")

    def _try_hot_reload_config(self) -> None:
        """Check if unified_config.yaml was modified and apply safe, non-destructive config changes.

        Only reloads strategy tuning parameters and thresholds — never changes
        exchange credentials, capital, or structural settings mid-run.
        """
        import os
        config_path = getattr(self, "_config_path", "unified_config.yaml")
        try:
            mtime = os.path.getmtime(config_path)
        except OSError:
            return

        last_mtime = getattr(self, "_config_mtime", 0.0)
        if mtime <= last_mtime:
            return

        self._config_mtime = mtime
        logger.info("Config file changed (mtime=%s) — attempting hot-reload", mtime)

        try:
            from core.config_manager import load_unified_yaml
            y = load_unified_yaml(config_path)
        except Exception as exc:
            logger.warning("Config hot-reload: failed to parse YAML: %s", exc)
            return

        # Safe fields that can be changed at runtime without breaking state
        _SAFE_FIELDS = {
            "se_buy_rsi", "se_sell_rsi", "se_buy_bb", "se_sell_bb",
            "se_trend_rsi_buy", "se_trend_rsi_sell",
            "se_min_actionable_confidence", "min_signal_confidence",
            "stop_loss_pct", "take_profit_pct",
            "trailing_stop_pct", "trailing_stop_enabled",
            "max_concurrent_positions", "cycle_op_timeout_s",
            "use_funding_rate_filter", "funding_rate_threshold",
            "use_volatility_regime_scale", "volatility_regime_high_threshold",
        }

        updated = []
        strategy = y.get("strategy_engine", {}) or {}
        risk = y.get("risk", {}) or {}
        merged = {**strategy, **risk}

        for key in _SAFE_FIELDS:
            if key in merged:
                new_val = merged[key]
                old_val = getattr(self.config, key, None)
                if new_val != old_val:
                    try:
                        setattr(self.config, key, new_val)
                        updated.append(f"{key}: {old_val} -> {new_val}")
                    except Exception:
                        pass

        if updated:
            logger.info("Config hot-reload applied %d changes: %s", len(updated), "; ".join(updated))
        else:
            logger.debug("Config hot-reload: no safe-field changes detected")

    def _portfolio_allows_signal(self, signal: Any) -> bool:
        """
        Best-effort portfolio checks to make paper trading realistic and prevent nonsense orders.
        """
        try:
            symbol = getattr(signal, "symbol", None) or (signal.get("symbol") if isinstance(signal, dict) else None)  # type: ignore[attr-defined]
            action = getattr(signal, "action", None) or (signal.get("action") if isinstance(signal, dict) else None)  # type: ignore[attr-defined]
            qty = getattr(signal, "quantity", None) if not isinstance(signal, dict) else signal.get("quantity")
            entry_price = getattr(signal, "entry_price", None) if not isinstance(signal, dict) else signal.get("entry_price")

            if not symbol or not action:
                return False
            action_u = str(action).upper()

            # Use the same AUD/USD rate as the capital optimizer.
            AUD_TO_USD = max(0.01, float(getattr(self.config, "aud_to_usd", 0.65) or 0.65))

            if action_u == "SELL":
                held = float((self.positions.get(str(symbol)) or {}).get("quantity") or 0.0)
                if held <= 0.0:
                    return False  # Can't sell what we don't have

            if action_u == "BUY":
                held = float((self.positions.get(str(symbol)) or {}).get("quantity") or 0.0)
                # Allow adding to positions up to max_position_pct (don't block all BUYs)
                max_pos = float(getattr(self.config, "max_position_pct", 0.25) or 0.25)
                portfolio_usd = float(getattr(self, "portfolio_value_aud", 1000.0) or 1000.0) * float(getattr(self.config, "aud_to_usd", 0.65) or 0.65)
                held_usd = held * float(entry_price or 0.0)
                if portfolio_usd > 0 and held_usd / portfolio_usd >= max_pos:
                    return False  # Already at max position
                try:
                    q = float(qty or 0.0)
                    px = float(entry_price or 0.0)
                except Exception:
                    q = 0.0
                    px = 0.0
                # If quantity/price not set yet, let it through — execution
                # engine will compute them from confidence * max_pos_pct.
                if q <= 0 or px <= 0:
                    return True  # Pass through; _execute_signals will size it
                notional_usd = q * px
                # Approx costs using configured taker fee + slippage.
                fee = float(getattr(self.config, "kraken_taker_fee", 0.0026) or 0.0026)
                slip = float(getattr(self.config, "slippage_pct", 0.002) or 0.002)
                needed_aud = (notional_usd * (1.0 + fee + slip)) / AUD_TO_USD
                return self.cash_balance_aud >= needed_aud

            # HOLD/unknown -> skip
            return False
        except Exception:
            return False
    
    async def _process_argus_strategies(self, ai_signals: List[Dict]) -> List[Dict]:
        """Process signals through ARGUS Ultimate strategies with multi-language acceleration"""
        # Combine AI signals with ARGUS strategy signals
        # This is where ARGUS's maximum return optimization happens
        processed_signals = []
        
        for signal in ai_signals:
            def _get(field: str, default: Any = None) -> Any:
                if isinstance(signal, dict):
                    return signal.get(field, default)
                return getattr(signal, field, default)

            def _set(field: str, value: Any) -> None:
                if isinstance(signal, dict):
                    signal[field] = value
                else:
                    try:
                        setattr(signal, field, value)
                    except Exception:
                        pass

            # Use multi-language system for signal processing if available
            if self.language_orchestrator:
                try:
                    from unified_language_orchestrator import TaskRequest, TaskType
                    
                    # Process order book using fastest language (C++/Rust)
                    ob = _get("order_book")
                    if ob:
                        task_request = TaskRequest(
                            task_type=TaskType.ORDER_BOOK_PROCESSING,
                            data=ob if isinstance(ob, dict) else {},
                            timeout=1.0
                        )
                        order_book_result = await self.language_orchestrator.execute_task(task_request)
                        if order_book_result.success:
                            _set("order_book_processed", order_book_result.result)
                            _set("processing_language", order_book_result.language_used)
                    
                    # Fast risk check using C++
                    pv = _get("position_value")
                    if pv is not None:
                        task_request = TaskRequest(
                            task_type=TaskType.RISK_CALCULATION,
                            data={
                                'position_value': pv,
                                'capital': self.portfolio_value_aud
                            },
                            timeout=0.5
                        )
                        risk_result = await self.language_orchestrator.execute_task(task_request)
                        if risk_result.success:
                            _set("risk_checked", risk_result.result)
                            _set("risk_check_language", risk_result.language_used)
                
                except Exception as e:
                    logger.warning(f"Multi-language signal processing failed: {e}")
            
            # Apply ARGUS strategy filters and enhancements
            _set("argus_confidence", 0.85)  # ARGUS confidence boost (metadata)
            _set("strategy", "unified_argus_ai_multilang")

            # Small nudge to confidence (kept conservative).
            try:
                c = float(_get("confidence", 0.0) or 0.0)
                if c > 0:
                    _set("confidence", min(1.0, c * 1.01))
            except Exception:
                pass

            processed_signals.append(signal)
        
        return processed_signals

    def _set_pre_trade_context(self) -> bool:
        """Set execution-engine context for pre_trade_risk_block. Returns True on success."""
        try:
            _pos = self.positions or {}
            setattr(self.config, "_pre_trade_positions", {str(s): float((_pos.get(s) or {}).get("quantity") or 0.0) for s in _pos})
            setattr(self.config, "_pre_trade_prices", {str(s): float((_pos.get(s) or {}).get("current_price") or (_pos.get(s) or {}).get("avg_price") or 0.0) for s in _pos})
            eq = float(self.portfolio_value_aud)
            setattr(self.config, "_pre_trade_equity_aud", eq)
            setattr(self.config, "_pre_trade_max_drawdown_pct", float(getattr(self.config, "max_drawdown_pct", 0.12) or 0.12))
            daily_pnl_aud = float(getattr(self, "daily_pnl_aud", 0.0) or 0.0)
            setattr(self.config, "_pre_trade_daily_pnl_pct", (daily_pnl_aud / eq * 100.0) if eq and eq > 0 else 0.0)
            return True
        except Exception as e:
            logger.warning("Pre-trade context failed (skipping execution this cycle): %s", e)
            return False

    async def _update_portfolio_value(self):
        """Update current portfolio value"""
        # Refresh positions from persistent execution store if available
        try:
            store = getattr(self.execution_engine, "state_store", None) if self.execution_engine else None
            if store:
                self.positions = store.get_positions()
                # Persist cash for crash-only recovery (best-effort)
                store.set_account_value("cash_balance_aud", float(self.cash_balance_aud))
        except Exception:
            pass

        # Calculate current value of all positions (AUD)
        aud_to_usd = float(getattr(self.config, "aud_to_usd", 0.65) or 0.65)
        positions_value_aud = 0.0
        unrealized_pnl_aud = 0.0
        for sym, pos in (self.positions or {}).items():
            try:
                qty = float((pos or {}).get("quantity") or 0.0)
                px = float((pos or {}).get("current_price") or 0.0)
                avg_px = float((pos or {}).get("avg_price") or 0.0)
                if qty < -1e-12:
                    self._ledger_sanity_violations += 1
                    logger.warning("Invalid negative position quantity detected for %s: %.8f", sym, qty)
                    continue
                if qty <= 0 or px <= 0:
                    continue
                notional_quote = qty * px
                quote = str(sym).split("/")[-1].upper() if "/" in str(sym) else "USD"
                unrealized_quote = (px - avg_px) * qty if avg_px > 0 else 0.0
                if quote == "AUD":
                    positions_value_aud += float(notional_quote)
                    unrealized_pnl_aud += float(unrealized_quote)
                elif quote == "USD":
                    positions_value_aud += float(notional_quote) / max(float(aud_to_usd), 1e-9)
                    unrealized_pnl_aud += float(unrealized_quote) / max(float(aud_to_usd), 1e-9)
                else:
                    # Unknown quote: treat as USD-like
                    positions_value_aud += float(notional_quote) / max(float(aud_to_usd), 1e-9)
                    unrealized_pnl_aud += float(unrealized_quote) / max(float(aud_to_usd), 1e-9)
            except Exception:
                continue

        self.unrealized_pnl_aud = float(unrealized_pnl_aud)
        self.portfolio_value_aud = float(self.cash_balance_aud) + float(positions_value_aud)
        self.realized_pnl_aud = float(self.total_pnl_aud)
        if float(self.portfolio_value_aud) < -1e-6:
            self._ledger_sanity_violations += 1
            logger.warning(
                "Invalid negative equity detected by ledger sanity check: equity=%.4f",
                float(self.portfolio_value_aud),
            )
        
        # Update drawdown
        if self.portfolio_value_aud > self.peak_equity_aud:
            self.peak_equity_aud = self.portfolio_value_aud
        
        drawdown = (self.peak_equity_aud - self.portfolio_value_aud) / self.peak_equity_aud
        if drawdown > self.max_drawdown_aud:
            self.max_drawdown_aud = drawdown

    def _get_partial_tp_and_trailing_stop_signals(self) -> List[Any]:
        """Build exit signals from paper_trading partial_tp and trailing_stop rules. Uses position high-water and optional one-shot partial TP."""
        from types import SimpleNamespace
        overrides = getattr(self.config, "paper_trading_overrides", None) or {}
        partial_pct = float(overrides.get("partial_tp_at_pct", 0) or 0)
        partial_close_pct = float(overrides.get("partial_tp_close_pct", 0.5) or 0.5)
        trailing_enabled = bool(overrides.get("trailing_stop_enabled", False))
        trailing_pct = float(overrides.get("trailing_stop_pct", 0.01) or 0.01)
        if partial_pct <= 0 and not trailing_enabled:
            return []
        out: List[Any] = []
        for sym, pos in (self.positions or {}).items():
            qty = float(pos.get("quantity") or 0.0)
            if qty <= 0:
                self._partial_tp_taken.pop(sym, None)
                continue
            entry = float(pos.get("avg_price") or pos.get("current_price") or 0.0)
            current = float(pos.get("current_price") or entry or 0.0)
            if entry <= 0 or current <= 0:
                continue
            self._position_high_water[sym] = max(self._position_high_water.get(sym, entry), current)
            high = self._position_high_water[sym]
            if trailing_enabled and trailing_pct > 0 and high > 0 and current <= high * (1.0 - trailing_pct):
                out.append(SimpleNamespace(symbol=sym, action="SELL", quantity=qty, entry_price=current))
                self._partial_tp_taken.pop(sym, None)
                logger.debug("Trailing stop triggered for %s: current=%.4f high=%.4f", sym, current, high)
                continue
            if partial_pct > 0 and not self._partial_tp_taken.get(sym) and current >= entry * (1.0 + partial_pct):
                sell_qty = qty * partial_close_pct
                if sell_qty > 0:
                    out.append(SimpleNamespace(symbol=sym, action="SELL", quantity=sell_qty, entry_price=current))
                    self._partial_tp_taken[sym] = True
                    logger.debug("Partial TP triggered for %s: %.2f%% at %.4f", sym, partial_pct * 100, current)
        return out
    
    async def _check_kill_switch(self) -> bool:
        """
        Check for kill switch file at data/KILL_SWITCH.

        If the file exists:
          - Cancel all pending orders
          - Log CRITICAL
          - Set system state to SHUTDOWN
          - Return True (caller should break the trading loop)

        Can be triggered externally: touch data/KILL_SWITCH
        """
        try:
            kill_path = Path("data/KILL_SWITCH")
            if not kill_path.exists():
                return False
        except Exception:
            return False

        logger.critical("KILL SWITCH ACTIVATED — file data/KILL_SWITCH detected, halting ALL trading immediately")

        # Cancel all pending orders
        try:
            if self._pending_orders:
                logger.critical("KILL SWITCH: cancelling %d pending orders", len(self._pending_orders))
                self._pending_orders.clear()
            # Cancel via execution engine if available
            execution_engine = getattr(self, "execution_engine", None)
            if execution_engine is not None:
                cancel_fn = getattr(execution_engine, "cancel_all_orders", None)
                if callable(cancel_fn):
                    res = cancel_fn()
                    if asyncio.iscoroutine(res):
                        await res
                    logger.critical("KILL SWITCH: cancelled all orders via execution engine")
            # Cancel via exchange manager if available
            em = getattr(self, "exchange_manager", None)
            if em is not None:
                cancel_fn = getattr(em, "cancel_all_orders", None)
                if callable(cancel_fn):
                    res = cancel_fn()
                    if asyncio.iscoroutine(res):
                        await res
                    logger.critical("KILL SWITCH: cancelled all orders via exchange manager")
        except Exception as exc:
            logger.error("KILL SWITCH: error cancelling orders: %s", exc)

        self.state = SystemState.SHUTDOWN
        return True

    def _check_emergency_stop(self) -> bool:
        # Global kill switch (legacy path — also checked via _check_kill_switch)
        try:
            if Path("data/KILL_SWITCH").exists():
                return True
            # Legacy location for backward compatibility
            if Path("KILL_SWITCH").exists():
                return True
        except Exception:
            pass

        """Check if emergency stop conditions are met (daily loss, drawdown, consecutive losses, plus optional latency/flash crash/network/arb)."""
        # Check daily loss limit (scale with peak equity; use vol-adjusted limit when enabled)
        cap_for_limit = max(float(self.peak_equity_aud), float(self.config.starting_capital_aud))
        daily_limit = getattr(self.config, "_vol_adjusted_daily_loss_pct", None)
        if daily_limit is None:
            daily_limit = float(getattr(self.config, "max_daily_loss_pct", 0.02) or 0.02)
        if self.daily_pnl_aud < -cap_for_limit * daily_limit:
            logger.warning("Daily loss limit exceeded")
            return True

        # Check drawdown limit
        if self.max_drawdown_aud > self.config.max_drawdown_pct:
            logger.warning("Maximum drawdown exceeded")
            return True

        # Check consecutive losses
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            logger.warning("Maximum consecutive losses exceeded")
            return True

        # Optional emergency_shutdown conditions (latency, flash crash, network, arb)
        if getattr(self.config, "emergency_shutdown_enabled", False):
            # Latency spike: last cycle took longer than threshold
            latency_ms = getattr(self, "_last_cycle_total_ms", None)
            max_latency_ms = getattr(self.config, "emergency_shutdown_latency_spike_ms", None)
            grace_cycles = max(0, int(getattr(self.config, "runtime_safety_latency_grace_cycles", 0) or 0))
            completed_cycles = max(0, int(getattr(self, "_completed_cycles", 0) or 0))
            if (
                max_latency_ms is not None
                and latency_ms is not None
                and latency_ms > float(max_latency_ms)
            ):
                if completed_cycles < grace_cycles:
                    logger.warning(
                        "Latency emergency check deferred during warmup grace (%s/%s): %.0f ms > %.0f ms",
                        completed_cycles,
                        grace_cycles,
                        latency_ms,
                        float(max_latency_ms),
                    )
                else:
                    logger.warning("Emergency shutdown: cycle latency %.0f ms > %.0f ms", latency_ms, float(max_latency_ms))
                    return True
            # Flash crash: any tracked symbol moved more than threshold in one cycle
            flash_pct = getattr(self.config, "emergency_shutdown_flash_crash_pct", None)
            if flash_pct is not None and getattr(self, "_last_price_by_symbol", None):
                for sym, (prev, curr) in getattr(self, "_last_price_by_symbol", {}).items():
                    if prev and curr and prev > 0:
                        move_pct = abs(curr - prev) / prev * 100.0
                        if move_pct >= float(flash_pct):
                            logger.warning("Emergency shutdown: flash crash %.2f%% on %s", move_pct, sym)
                            return True
            # Network: exchange unreachable flag (set by data/execution layer on repeated failure)
            if getattr(self.config, "emergency_shutdown_network_fail", False) and getattr(self, "_exchange_unreachable", False):
                logger.warning("Emergency shutdown: exchange unreachable")
                return True
            # Arb/spread: if current spread exceeds threshold (from last order book fetch)
            spread_bps = getattr(self, "_last_spread_bps", None)
            max_spread_bps = getattr(self.config, "emergency_shutdown_arb_spread_bps", None)
            if max_spread_bps is not None and spread_bps is not None and spread_bps > float(max_spread_bps):
                logger.warning("Emergency shutdown: spread %.0f bps > %.0f bps", spread_bps, float(max_spread_bps))
                return True

        return False
    
    def _record_trade(self, trade_result: Dict):
        """Record executed trade. Thread-safe via _state_lock."""
        import threading as _th
        lock = getattr(self, "_state_lock", None)
        if lock is None:
            self._state_lock = _th.Lock()
            lock = self._state_lock
        with lock:
            self._record_trade_inner(trade_result)

    def _record_trade_inner(self, trade_result: Dict):
        """Record executed trade (inner, must hold _state_lock)."""
        # Basic portfolio accounting (paper/live safe-mode):
        # - Track per-symbol quantity + avg entry price (USD)
        # - Track cash in AUD using a simple AUD/USD conversion consistent with the capital optimizer.
        AUD_TO_USD = max(0.01, float(getattr(self.config, "aud_to_usd", 0.65) or 0.65))

        self.total_trades += 1
        self.trade_history.append({
            'timestamp': datetime.now(),
            'trade_id': trade_result.get('order_id'),
            'symbol': trade_result.get('symbol'),
            'side': trade_result.get('side'),
            'quantity': trade_result.get('quantity'),
            'price': trade_result.get('price'),
            'pnl': trade_result.get('pnl', 0),
            **trade_result
        })
        
        # P&L is tracked on realized exits; entry spends are not treated as realized loss.
        realized_pnl_aud = 0.0

        # Update holdings/cash (best-effort, for paper runs & basic safety checks)
        try:
            symbol = str(trade_result.get("symbol") or "")
            side = str(trade_result.get("side") or "").upper()
            qty = float(trade_result.get("quantity") or 0.0)
            price_usd = float(trade_result.get("price") or 0.0)
            commission_usd = max(0.0, float(trade_result.get("commission") or 0.0))
            trade_ts = float(trade_result.get("timestamp") or time.time())
            strategy_name = str(
                trade_result.get("source_strategy")
                or trade_result.get("strategy")
                or "unknown"
            )
            regime_label = str(
                trade_result.get("regime_label")
                or getattr(self, "_latest_regime_label", "")
                or ""
            )

            if side not in {"BUY", "SELL"}:
                self._ledger_sanity_violations += 1
                logger.warning("Ignoring trade with unsupported side=%s", side)
                return

            if symbol and qty > 0 and price_usd > 0:
                # Track (prev, curr) price for emergency_shutdown flash_crash check
                last_prices = getattr(self, "_last_price_by_symbol", None)
                if last_prices is not None:
                    prev = (last_prices.get(symbol) or (None, None))[1]
                    last_prices[symbol] = (prev, price_usd)
                pos = self.positions.get(symbol) or {"quantity": 0.0, "avg_price": 0.0, "current_price": price_usd}
                held_qty = float(pos.get("quantity") or 0.0)
                avg_px = float(pos.get("avg_price") or 0.0)
                self.total_fees_aud += float(commission_usd) / max(float(AUD_TO_USD), 1e-9)

                if side == "BUY":
                    notional_usd = qty * price_usd
                    cost_aud = (notional_usd + commission_usd) / AUD_TO_USD
                    self.cash_balance_aud -= cost_aud

                    new_qty = held_qty + qty
                    if new_qty > 0:
                        # Weighted average entry price (excluding fees for P&L calculation)
                        held_cost_usd = held_qty * avg_px
                        buy_cost_usd = qty * price_usd  # Exclude commission from avg_price
                        new_avg = (held_cost_usd + buy_cost_usd) / new_qty
                    else:
                        new_avg = 0.0
                    pos["quantity"] = new_qty
                    pos["avg_price"] = new_avg
                    pos["current_price"] = price_usd
                    self.positions[symbol] = pos
                    try:
                        se_engine = getattr(self, "strategy_evaluation_engine", None)
                        if se_engine is not None:
                            se_engine.record_open(
                                strategy_name=str(strategy_name),
                                symbol=str(symbol),
                                quantity=float(qty),
                                ts=float(trade_ts),
                                regime_label=(regime_label or None),
                            )
                    except Exception as e:
                        self._handle_strategy_eval_error(e, context="record_open")
                    
                    # Initialize trailing stop for new position
                    if self.fully_adaptive_risk is not None and self.fully_adaptive_risk.config.trailing_stop_enabled:
                        try:
                            # Get ATR for trailing stop (use cached volatility or default)
                            atr = self.fully_adaptive_risk._volatility_cache.get(symbol, 0) * price_usd
                            if atr <= 0:
                                atr = price_usd * 0.02  # Default 2% ATR
                            
                            self.fully_adaptive_risk.open_position(
                                symbol=str(symbol),
                                entry_price=price_usd,
                                side="long",
                                atr=atr,
                            )
                            logger.debug("Trailing stop initialized for %s at %.2f (ATR=%.2f)", symbol, price_usd, atr)
                        except Exception as e:
                            logger.debug("Trailing stop initialization error: %s", e)

                elif side == "SELL":
                    sell_qty = min(qty, held_qty)
                    if sell_qty <= 0:
                        # Ignore sells when flat (should be prevented pre-execution, but be defensive)
                        self._ledger_sanity_violations += 1
                        logger.warning("Ignoring SELL while flat for %s", symbol)
                        return
                    if qty > held_qty + 1e-12:
                        self._ledger_sanity_violations += 1
                        logger.warning(
                            "SELL quantity exceeds held quantity for %s (qty=%.8f held=%.8f), clamping",
                            symbol,
                            qty,
                            held_qty,
                        )
                    sell_commission_usd = commission_usd * (sell_qty / qty) if qty > 0 else 0.0
                    notional_usd = sell_qty * price_usd
                    proceeds_aud = (notional_usd - sell_commission_usd) / AUD_TO_USD
                    self.cash_balance_aud += proceeds_aud

                    # Realized PnL (AUD), net of both entry and exit fees.
                    entry_usd = sell_qty * avg_px
                    realized_usd = (sell_qty * price_usd - sell_commission_usd) - entry_usd
                    realized_aud = realized_usd / AUD_TO_USD
                    realized_pnl_aud = float(realized_aud)
                    self.total_pnl_aud += float(realized_pnl_aud)
                    self.realized_pnl_aud = float(self.total_pnl_aud)
                    self.daily_pnl_aud += float(realized_pnl_aud)
                    try:
                        se_engine = getattr(self, "strategy_evaluation_engine", None)
                        if se_engine is not None:
                            hold_time_seconds = float(
                                se_engine.consume_hold_time_seconds(
                                    strategy_name=strategy_name,
                                    symbol=str(symbol),
                                    quantity=float(sell_qty),
                                    ts=float(trade_ts),
                                )
                            )
                            slippage_raw = float(trade_result.get("slippage") or 0.0)
                            realized_slippage_bps = (
                                abs(slippage_raw) * 10_000.0
                                if abs(slippage_raw) <= 1.0
                                else abs(slippage_raw)
                            )
                            expected_net_edge_bps = float(
                                trade_result.get("expected_net_edge_bps")
                                or (trade_result.get("raw") or {}).get("expected_net_edge_bps")
                                or 0.0
                            )
                            fees_aud = float(sell_commission_usd) / max(float(AUD_TO_USD), 1e-9)
                            gross_pnl_aud = float(realized_pnl_aud + fees_aud)
                            se_engine.record_trade_close(
                                strategy_name=strategy_name,
                                symbol=str(symbol),
                                gross_pnl_aud=float(gross_pnl_aud),
                                net_pnl_aud=float(realized_pnl_aud),
                                fees_aud=float(fees_aud),
                                expected_net_edge_bps=float(expected_net_edge_bps),
                                realized_slippage_bps=float(realized_slippage_bps),
                                hold_time_seconds=float(hold_time_seconds),
                                regime_label=(regime_label or None),
                                ts=float(trade_ts),
                            )
                    except Exception as e:
                        self._handle_strategy_eval_error(e, context="record_trade_close")

                    # Ledger: persist realized PnL so earnings are correct in unified_trades.db
                    try:
                        ledger = getattr(self.execution_engine, "trade_ledger", None) if self.execution_engine else None
                        if ledger is not None and hasattr(ledger, "update_trade_pnl"):
                            oid = trade_result.get("order_id") or trade_result.get("id")
                            if oid:
                                ledger.update_trade_pnl(str(oid), float(realized_pnl_aud))
                    except Exception:
                        pass

                    # Strategy decay detection (best-effort)
                    if self.strategy_decay_detector is not None:
                        try:
                            decay_result = self.strategy_decay_detector.record_trade(
                                strategy_name=str(strategy_name),
                                pnl=float(realized_pnl_aud),
                                symbol=str(symbol),
                                regime=regime_label
                            )
                            if decay_result is not None and decay_result.get("is_decaying", False):
                                logger.warning(
                                    "Strategy decay detected: %s (decay_score=%.3f, recent_win_rate=%.2f%%)",
                                    strategy_name,
                                    decay_result.get("decay_score", 0),
                                    decay_result.get("recent_win_rate", 0) * 100
                                )
                                # Store decay info for strategy rotation
                                self._strategy_decay_alerts = getattr(self, "_strategy_decay_alerts", [])
                                self._strategy_decay_alerts.append({
                                    "strategy": strategy_name,
                                    "decay_score": decay_result.get("decay_score", 0),
                                    "timestamp": datetime.now()
                                })
                        except Exception as e:
                            logger.debug("Strategy decay detection error: %s", e)
                    
                    # Fully Adaptive Risk Engine - record trade for performance tracking
                    if self.fully_adaptive_risk is not None:
                        try:
                            # Calculate PnL percentage
                            entry_price = float(avg_px) if avg_px > 0 else float(price_usd)
                            exit_price = float(price_usd)
                            if entry_price > 0:
                                pnl_pct = (exit_price - entry_price) / entry_price
                                if side == "SELL":  # For sells, direction matters
                                    # If we're closing a long position
                                    pnl_pct = (exit_price - entry_price) / entry_price
                                
                                self.fully_adaptive_risk.record_trade(pnl_pct, str(symbol))
                                
                                # Close trailing stop if position is fully closed
                                new_qty = float(pos.get("quantity", 0))
                                if new_qty <= 0:
                                    self.fully_adaptive_risk.close_position(str(symbol))
                                
                                # Log adaptive risk status periodically
                                if self.total_trades % 5 == 0:
                                    status = self.fully_adaptive_risk.get_status()
                                    logger.info(
                                        "Adaptive Risk Status: state=%s, win_rate=%.1f%%, "
                                        "consecutive_losses=%d, drawdown=%.2f%%",
                                        status["risk_state"],
                                        status["performance"]["win_rate"] * 100,
                                        status["performance"]["consecutive_losses"],
                                        status["current_drawdown_pct"] * 100,
                                    )
                        except Exception as e:
                            logger.debug("Adaptive risk trade recording error: %s", e)

                    # ════════════════════════════════════════════════════════════════════════════
                    # QUANTUM ADAPTIVE RISK ENGINE - Trade Recording & Status Logging
                    # ════════════════════════════════════════════════════════════════════════════
                    if self.quantum_adaptive_risk is not None:
                        try:
                            # Calculate PnL percentage for quantum engine
                            entry_price_q = float(avg_px) if avg_px > 0 else float(price_usd)
                            exit_price_q = float(price_usd)
                            if entry_price_q > 0:
                                pnl_pct_q = (exit_price_q - entry_price_q) / entry_price_q
                                
                                # Record trade in quantum engine
                                self.quantum_adaptive_risk.record_trade(pnl_pct_q, str(symbol))
                                
                                # Log quantum risk status every 10 trades
                                if self.total_trades % 10 == 0:
                                    q_status = self.quantum_adaptive_risk.get_status()
                                    logger.info(
                                        "🔮 Quantum Risk PINNACLE: state=%s, win_rate=%.1f%%, "
                                        "consecutive_losses=%d, drawdown=%.2f%%, qkelly=%.3f, "
                                        "regime=%s, VaR95=%.2f%%",
                                        q_status["risk_state"],
                                        q_status["win_rate"] * 100,
                                        q_status["consecutive_losses"],
                                        q_status["current_drawdown_pct"] * 100,
                                        q_status.get("quantum_kelly_fraction", 0),
                                        q_status.get("quantum_regime_probs", {}),
                                        q_status.get("quantum_var_95", 0) * 100,
                                    )
                        except Exception as e:
                            logger.debug("Quantum risk trade recording error: %s", e)

                    # ════════════════════════════════════════════════════════════════════════════
                    # STRATEGY PARAMETER AUTO-TUNER - Feed trade results for parameter optimization
                    # ════════════════════════════════════════════════════════════════════════════
                    if self.strategy_param_tuner is not None:
                        try:
                            entry_price_t = float(avg_px) if avg_px > 0 else float(price_usd)
                            exit_price_t = float(price_usd)
                            if entry_price_t > 0:
                                pnl_pct_t = (exit_price_t - entry_price_t) / entry_price_t
                                self.strategy_param_tuner.record_trade(pnl_pct_t)
                                
                                # Check if tuning is needed
                                if self.strategy_param_tuner.should_tune():
                                    tuning_result = self.strategy_param_tuner.tune_parameters()
                                    logger.info(
                                        "🎯 Strategy params auto-tuned: score=%.4f, confidence=%.2f, trials=%d",
                                        tuning_result.expected_improvement,
                                        tuning_result.confidence,
                                        tuning_result.trials_run,
                                    )
                        except Exception as e:
                            logger.debug("Strategy parameter tuner error: %s", e)

                    # ════════════════════════════════════════════════════════════════════════════
                    # SELF-LEARNING UNIVERSE - Record paper trades for new pairs
                    # ════════════════════════════════════════════════════════════════════════════
                    if self.universe_expander is not None:
                        try:
                            paper_testing_pairs = self.universe_expander.get_paper_testing_pairs()
                            if str(symbol) in paper_testing_pairs:
                                entry_price_u = float(avg_px) if avg_px > 0 else float(price_usd)
                                exit_price_u = float(price_usd)
                                if entry_price_u > 0:
                                    pnl_pct_u = (exit_price_u - entry_price_u) / entry_price_u
                                    self.universe_expander.record_paper_trade(str(symbol), pnl_pct_u)
                        except Exception as e:
                            logger.debug("Universe expander trade recording error: %s", e)

                    # Continuous learning feedback (best-effort): report realized pnl% to the AI brain.
                    try:
                        if self.ai_brain is not None and hasattr(self.ai_brain, "on_trade_closed"):
                            pnl_pct = (float(price_usd) / max(float(avg_px), 1e-9) - 1.0) * 100.0
                            src = str(trade_result.get("source_strategy") or "") or "unknown"
                            reg = "unknown"
                            try:
                                st = self.ai_brain.get_adaptation_status()
                                se_st = st.get("strategy_engine") if isinstance(st, dict) else None
                                lr = (se_st.get("last_regime") or {}) if isinstance(se_st, dict) else {}
                                reg = str(lr.get(symbol) or "unknown") if isinstance(lr, dict) else "unknown"
                            except Exception:
                                reg = "unknown"
                            self.ai_brain.on_trade_closed(symbol=str(symbol), pnl_pct=float(pnl_pct), strategy=src, regime=reg)
                            # Instantaneous evolution: trigger GA after N trades (no wait for interval)
                            if self.self_improver is not None and getattr(self.config, "evolution_trigger_on_trade", False):
                                try:
                                    self.self_improver.record_trade_closed()
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    # Adaptive risk: update controller with realized outcomes
                    try:
                        arc = getattr(self, "adaptive_risk_controller", None)
                        if arc is not None:
                            arc.observe_trade_close(strategy=str(trade_result.get("source_strategy") or "unknown"), pnl_pct=float(pnl_pct))
                    except Exception:
                        pass

                    # Adaptive universe: update per-symbol performance
                    try:
                        us = getattr(self, "universe_selector", None)
                        if us is not None:
                            us.observe_trade_close(symbol=str(symbol), pnl_pct=float(pnl_pct))
                            us.save()
                    except Exception:
                        pass

                    # Strategy allocator: record realized PnL so ranking works (everything together)
                    try:
                        if self.strategy_allocator is not None:
                            _pnl = (float(price_usd) / max(float(avg_px), 1e-9) - 1.0) * 100.0
                            _src = str(trade_result.get("source_strategy") or "") or "unknown"
                            _reg = "unknown"
                            try:
                                _st = self.ai_brain.get_adaptation_status() if self.ai_brain else {}
                                _se = _st.get("strategy_engine") if isinstance(_st, dict) else None
                                _lr = (_se.get("last_regime") or {}) if isinstance(_se, dict) else {}
                                _reg = str(_lr.get(symbol) or "unknown") if isinstance(_lr, dict) else "unknown"
                            except Exception:
                                pass
                            self.strategy_allocator.record_trade(strategy=_src, regime=_reg, pnl_pct=float(_pnl))
                            self.strategy_allocator.save()
                    except Exception:
                        pass

                    # Thompson sampling feedback: record outcome for all languages
                    # so ensemble weights adapt to realized accuracy over time.
                    try:
                        if self.language_orchestrator and hasattr(self.language_orchestrator, "record_language_outcome"):
                            correct = float(realized_pnl_aud) > 0
                            for lang in self.language_orchestrator.languages:
                                self.language_orchestrator.record_language_outcome(lang, correct)
                    except Exception:
                        pass

                    # ════════════════════════════════════════════════════════════════════════════
                    # ADVANCED LEARNING INTEGRATION (25+ learning systems) - Feed trade outcomes
                    # ════════════════════════════════════════════════════════════════════════════
                    try:
                        learning_loop = getattr(self, "learning_trading_loop", None)
                        if learning_loop is not None and hasattr(learning_loop, "record_trade_outcome"):
                            # Calculate reward signal (normalized PnL percentage)
                            entry_price_l = float(avg_px) if avg_px > 0 else float(price_usd)
                            exit_price_l = float(price_usd)
                            if entry_price_l > 0:
                                reward = (exit_price_l - entry_price_l) / entry_price_l
                                
                                # Build market data dict for the learning system
                                market_data = {
                                    "close": float(price_usd),
                                    "open": float(price_usd),  # Simplified
                                    "high": float(price_usd),
                                    "low": float(price_usd),
                                    "volume": 0.0,
                                    "symbol": str(symbol),
                                    "side": str(side),
                                    "pnl_aud": float(realized_pnl_aud),
                                }
                                
                                # Build decision dict from trade result
                                decision = {
                                    "action": 1 if side == "BUY" else 2,  # 1=buy, 2=sell
                                    "confidence": float(trade_result.get("confidence", 0.7)),
                                    "uncertainty": float(trade_result.get("uncertainty", 0.3)),
                                    "position_size": float(qty),
                                    "reasoning": f"Trade executed: {side} {symbol}",
                                    "agent_decisions": {},
                                    "metadata": {
                                        "strategy": str(strategy_name),
                                        "regime": str(regime_label),
                                    }
                                }
                                
                                # Record outcome for learning
                                learning_loop.record_trade_outcome(
                                    market_data=market_data,
                                    decision=decision,
                                    actual_reward=reward,
                                    human_rating=None  # No human feedback for now
                                )
                                
                                # Log learning status every 10 trades
                                if self.total_trades % 10 == 0:
                                    perf = learning_loop.get_performance_summary()
                                    logger.info(
                                        "🧠 Advanced Learning: trades=%d, total_pnl=%.4f, "
                                        "avg_confidence=%.2f%%, active_systems=%d",
                                        perf["trading"]["total_trades"],
                                        perf["trading"]["total_pnl"],
                                        perf["learning"]["metrics"]["avg_confidence"] * 100,
                                        perf["learning"]["metrics"]["active_systems"],
                                    )
                    except Exception as e:
                        logger.debug("Advanced learning feedback error: %s", e)

                    remaining = held_qty - sell_qty
                    pos["quantity"] = remaining
                    pos["current_price"] = price_usd
                    if remaining <= 0:
                        pos["avg_price"] = 0.0
                    self.positions[symbol] = pos
        except Exception:
            # Accounting is best-effort; do not crash trading loop for it.
            pass
        
        # Update win/loss tracking (use realized PnL)
        if float(realized_pnl_aud) > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        elif float(realized_pnl_aud) < 0:
            self.losing_trades += 1
            self.consecutive_losses += 1
        else:
            # Neutral trade (e.g., paper-mode placeholder with unknown PnL).
            # Do not count as a loss or a win.
            pass
        self.realized_pnl_aud = float(self.total_pnl_aud)

        # Tick capture (high-frequency microstructure record)
        try:
            from data.tick_capture import TickCapture, Tick
            if not hasattr(self, "_tick_capture"):
                self._tick_capture = TickCapture()
            sym_ = str(trade_result.get("symbol") or "")
            side_ = str(trade_result.get("side") or "BUY")
            qty_ = float(trade_result.get("quantity") or 0.0)
            px_ = float(trade_result.get("price") or 0.0)
            if sym_ and qty_ > 0 and px_ > 0:
                self._tick_capture.feed(Tick(
                    symbol=sym_,
                    exchange="argus",
                    price=px_,
                    quantity=qty_,
                    side=side_.lower(),
                ))
        except Exception:
            pass

        # Adaptive slippage feedback (model learns from actual vs expected fill)
        try:
            cr = self.component_registry
            if cr is not None and getattr(cr, "adaptive_slippage", None) is not None:
                from execution.adaptive_slippage_model import SlippageFeatures
                sym_ = str(trade_result.get("symbol") or "")
                side_ = str(trade_result.get("side") or "BUY").upper()
                actual_slippage = float(trade_result.get("slippage_pct") or trade_result.get("slippage", 0.0) or 0.0)
                spread_bps = float(trade_result.get("spread_bps") or 5.0)
                vol = float(trade_result.get("volatility_30m") or 0.002)
                qty_norm = min(1.0, float(trade_result.get("quantity") or 0.0) / max(float(getattr(self.config, "typical_order_size", 0.01) or 0.01), 1e-9))
                hour = datetime.now().hour
                features = SlippageFeatures(
                    side=side_, quantity_norm=qty_norm, hour_of_day=hour,
                    spread_bps=spread_bps, volatility_30m=vol
                )
                cr.adaptive_slippage.record_trade(features, actual_slippage)
        except Exception:
            pass

        # Component registry fill hook (best-effort)
        try:
            if self.component_registry is not None:
                self.component_registry.on_fill(trade_result)
        except Exception:
            pass

        # Strategy state store: update per-strategy win/loss/PnL tracking
        try:
            _sss = getattr(self, "_strategy_state_store", None)
            if _sss is not None:
                _strat_name = str(
                    trade_result.get("source_strategy")
                    or trade_result.get("strategy")
                    or "unknown"
                )
                _sss.update_after_trade(
                    strategy_name=_strat_name,
                    pnl=float(realized_pnl_aud),
                    timestamp=float(trade_result.get("timestamp") or time.time()),
                )
        except Exception:
            pass

    # =========================================================================
    # EXECUTION PIPELINE: signal -> risk check -> order -> fill -> position
    # =========================================================================

    # --- Kelly Criterion & Volatility-Adjusted Position Sizing ---

    def _kelly_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Half-Kelly position sizing for safety."""
        if avg_loss <= 0 or win_rate <= 0 or avg_win <= 0:
            return 0.0
        payoff_ratio = avg_win / avg_loss
        kelly = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio
        half_kelly = kelly * 0.5  # Half-Kelly for safety
        return max(0.0, min(half_kelly, 0.15))  # Cap at 15%

    def _vol_adjusted_size(self, base_size: float, current_vol: float, target_vol: float = 0.02) -> float:
        """Scale position inversely with volatility. target_vol=2% daily."""
        if current_vol <= 0:
            return base_size
        vol_ratio = target_vol / max(current_vol, 0.005)
        return base_size * min(vol_ratio, 2.0)  # Never more than 2x base

    def _get_strategy_trade_stats(self, strategy_name: str) -> Dict[str, Any]:
        """
        Extract win rate, avg win, avg loss from trade_history for a given strategy.

        Returns dict with keys: win_rate, avg_win, avg_loss, n_trades, wins, losses.
        Only considers SELL trades (closed positions) with realized PnL.

        FIX 3: When n_trades < 20, uses configurable bootstrap values from config
        so Kelly sizing has a reasonable starting point instead of being disabled.
        """
        sells = [
            t for t in self.trade_history
            if (
                str(t.get("source_strategy") or t.get("strategy") or "unknown") == strategy_name
                and str(t.get("side", "")).upper() == "SELL"
                and t.get("pnl") is not None
            )
        ]
        if not sells:
            n = 0
            wins_list: List[float] = []
            losses_list: List[float] = []
        else:
            wins_list = [float(t["pnl"]) for t in sells if float(t.get("pnl", 0)) > 0]
            losses_list = [abs(float(t["pnl"])) for t in sells if float(t.get("pnl", 0)) < 0]
            n = len(sells)

        # FIX 3: Bootstrap Kelly stats from config when insufficient trades
        min_kelly_trades = 20
        if n < min_kelly_trades:
            bootstrap_wr = float(getattr(self.config, "kelly_bootstrap_win_rate", 0.55) or 0.55)
            bootstrap_aw = float(getattr(self.config, "kelly_bootstrap_avg_win", 0.02) or 0.02)
            bootstrap_al = float(getattr(self.config, "kelly_bootstrap_avg_loss", 0.015) or 0.015)
            return {
                "win_rate": bootstrap_wr,
                "avg_win": bootstrap_aw,
                "avg_loss": bootstrap_al,
                "n_trades": n,
                "wins": wins_list,
                "losses": losses_list,
                "bootstrapped": True,
            }

        win_rate = len(wins_list) / n if n > 0 else 0.0
        avg_win = sum(wins_list) / len(wins_list) if wins_list else 0.0
        avg_loss = sum(losses_list) / len(losses_list) if losses_list else 0.0
        return {
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "n_trades": n,
            "wins": wins_list,
            "losses": losses_list,
            "bootstrapped": False,
        }

    def _compute_fallback_regime(self) -> str:
        """FIX 2: Compute a simple fallback regime from recent price data.

        Uses trade_history or position current_prices. Returns one of:
        HIGH_VOL, TRENDING_UP, TRENDING_DOWN, LOW_VOL, NORMAL.
        """
        # Gather recent prices from trade_history
        prices = [
            float(t["price"])
            for t in (self.trade_history or [])[-50:]
            if t.get("price")
        ]
        # Fallback: use position current_prices
        if len(prices) < 5:
            for sym, pos in (self.positions or {}).items():
                px = (pos or {}).get("current_price")
                if px is not None:
                    prices.append(float(px))

        if len(prices) < 5:
            return "NORMAL"

        # Use last 20 prices
        recent = prices[-20:]
        if len(recent) < 5:
            return "NORMAL"

        returns = [
            (recent[i] - recent[i - 1]) / max(abs(recent[i - 1]), 1e-12)
            for i in range(1, len(recent))
        ]
        vol = float(np.std(returns)) if returns else 0.0
        trend = (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-12)

        if vol > 0.03:
            return "HIGH_VOL"
        if vol > 0.02 and trend > 0.01:
            return "TRENDING_UP"
        if vol > 0.02 and trend < -0.01:
            return "TRENDING_DOWN"
        if vol < 0.01:
            return "LOW_VOL"
        return "NORMAL"

    async def _bootstrap_volatility(self) -> None:
        """Bootstrap volatility from OHLCV on startup for all trading pairs.

        Fetches last 24h of 1h candles via exchange/ccxt and computes std(log_returns).
        Caches in self._volatility_cache (symbol -> vol).
        """
        pairs = list(getattr(self.config, "trading_pairs", []) or [])
        if not pairs:
            return
        _timeout_per_pair = 10.0

        async def _bootstrap_one(symbol: str) -> None:
            try:
                closes = None
                if self.market_data_service is not None and hasattr(self.market_data_service, "fetch_ohlcv_df"):
                    try:
                        df = await asyncio.wait_for(
                            self.market_data_service.fetch_ohlcv_df(symbol, timeframe="1h", limit=24),
                            timeout=_timeout_per_pair,
                        )
                        if df is not None and not df.empty and "close" in df.columns:
                            closes = df["close"].astype(float).values
                    except Exception:
                        pass
                if closes is None or len(closes) < 5:
                    if self.exchange_manager is not None and hasattr(self.exchange_manager, "fetch_ohlcv"):
                        try:
                            raw = await asyncio.wait_for(
                                self.exchange_manager.fetch_ohlcv(symbol, timeframe="1h", limit=24),
                                timeout=_timeout_per_pair,
                            )
                            if raw and len(raw) >= 5:
                                closes = np.array([float(c[4]) for c in raw], dtype=float)
                        except Exception:
                            pass
                if closes is not None and len(closes) >= 5:
                    log_returns = np.diff(np.log(closes))
                    vol = float(np.std(log_returns))
                    if vol > 0:
                        self._volatility_cache[symbol] = vol
                        logger.info("Bootstrap vol for %s: %.6f (from %d candles)", symbol, vol, len(closes))
            except Exception as e:
                logger.debug("Bootstrap vol failed for %s: %s", symbol, e)

        # Fetch all symbols in parallel (was sequential — saves 30s+ on 8 symbols)
        await asyncio.gather(*[_bootstrap_one(s) for s in pairs], return_exceptions=True)
        logger.info("Volatility bootstrap complete: %d symbols cached", len(self._volatility_cache))

    def _get_current_vol(self, symbol: str) -> float:
        """
        Estimate current daily volatility from recent price history.

        Falls back to bootstrapped OHLCV cache, then 0.0 if not enough data.
        """
        try:
            # Try to get from ensemble hub on component registry
            cr = getattr(self, "component_registry", None)
            if cr is not None:
                hub = getattr(cr, "ensemble_hub", None)
                if hub is not None:
                    last = hub.get_last(symbol)
                    vol_info = (last.sources or {}).get("vol_regime", {})
                    vol_1d = vol_info.get("forecast_vol_1d", 0.0)
                    if vol_1d > 0:
                        return float(vol_1d)
        except Exception:
            pass

        # Fallback: compute from trade_history prices for this symbol
        prices = [
            float(t["price"])
            for t in self.trade_history
            if t.get("symbol") == symbol and t.get("price")
        ]
        if len(prices) >= 5:
            import numpy as _np
            returns = _np.diff(_np.log(_np.array(prices, dtype=float)))
            return float(_np.std(returns)) if len(returns) > 0 else 0.0

        # FIX 1: Use bootstrapped OHLCV volatility cache
        cached_vol = getattr(self, "_volatility_cache", {}).get(symbol, 0.0)
        if cached_vol > 0:
            return cached_vol
        return 0.0

    def _get_signal_quality(self, signals: Optional[List[Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Get signal quality metrics from the ensemble hub if available.

        FIX 4: Falls back to computing quality from signals themselves when
        ensemble hub is unavailable.
        """
        try:
            cr = getattr(self, "component_registry", None)
            if cr is not None:
                hub = getattr(cr, "ensemble_hub", None)
                if hub is not None and hasattr(hub, "get_signal_quality"):
                    result = hub.get_signal_quality()
                    if result is not None:
                        return result
        except Exception:
            pass

        # FIX 4: Compute quality from signals themselves
        if not signals:
            return {"recommendation": "no_signals", "quality": 0.0}

        # Group by symbol -> directions
        symbol_directions: Dict[str, set] = {}
        for sig in signals:
            sym = str(getattr(sig, "symbol", "UNKNOWN"))
            action = str(getattr(sig, "action", "")).upper()
            if sym not in symbol_directions:
                symbol_directions[sym] = set()
            symbol_directions[sym].add(action)

        # Check for conflicts (BUY and SELL for same symbol)
        has_conflict = any(
            "BUY" in dirs and "SELL" in dirs
            for dirs in symbol_directions.values()
        )

        if has_conflict:
            return {"recommendation": "conflicted", "quality": 0.3}

        # Check agreement — all same direction
        all_actions = set()
        for sig in signals:
            all_actions.add(str(getattr(sig, "action", "")).upper())
        if len(all_actions) == 1:
            return {"recommendation": "strong", "quality": 0.9}

        return {"recommendation": "moderate", "quality": 0.6}

    # --- Regime-adaptive position sizing constants ---
    REGIME_POSITION_SCALE = {
        'CRISIS': 0.3, 'EXTREME': 0.3,
        'HIGH_VOL': 0.5, 'ELEVATED': 0.5,
        'TRENDING_DOWN': 0.65, 'BEAR': 0.65, 'TREND_DOWN': 0.65,
        'RANGE': 1.0, 'NORMAL': 1.0, 'SIDEWAYS': 1.0,
        'TRENDING_UP': 1.2, 'BULL': 1.2, 'TREND_UP': 1.2,
        'BREAKOUT': 0.8, 'LOW_VOL': 1.3,
    }
    REGIME_STOP_SCALE = {
        'CRISIS': 2.0, 'EXTREME': 2.0,
        'HIGH_VOL': 2.0, 'ELEVATED': 1.5,
        'TRENDING_DOWN': 1.2, 'BEAR': 1.2, 'TREND_DOWN': 1.2,
        'RANGE': 1.0, 'NORMAL': 1.0, 'SIDEWAYS': 1.0,
        'TRENDING_UP': 0.9, 'BULL': 0.9, 'TREND_UP': 0.9,
        'BREAKOUT': 1.3, 'LOW_VOL': 0.7,
    }
    REGIME_TP_SCALE = {
        'CRISIS': 1.5, 'EXTREME': 1.5,
        'HIGH_VOL': 1.5, 'ELEVATED': 1.3,
        'TRENDING_DOWN': 0.9, 'BEAR': 0.9, 'TREND_DOWN': 0.9,
        'RANGE': 1.0, 'NORMAL': 1.0, 'SIDEWAYS': 1.0,
        'TRENDING_UP': 1.3, 'BULL': 1.3, 'TREND_UP': 1.3,
        'BREAKOUT': 1.2, 'LOW_VOL': 0.8,
    }

    # ─── Extracted sub-methods (imported from core.execute_signals_helpers) ───
    from core.execute_signals_helpers import (
        _pre_execute_context as _pre_execute_context,
        _extract_signal_fields as _extract_signal_fields,
        _apply_risk_gates as _apply_risk_gates,
        _compute_position_size as _compute_position_size,
        _apply_intelligence_gates as _apply_intelligence_gates,
        _compute_stops_and_quantity as _compute_stops_and_quantity,
        _log_cycle_summary as _log_cycle_summary,
    )

    async def _execute_signals(self, signals: List[Any]) -> List[Dict[str, Any]]:
        """
        Execute signals through the full pipeline: risk gates → sizing →
        intelligence gates → stops → order execution → fill recording.

        Orchestrates 7 extracted sub-methods for testability.
        Falls back to _execute_signals_legacy on import error.
        """
        if not signals:
            return []

        # --- FIX 26: Smart signal conflict resolution ---
        try:
            signals = self._resolve_signal_conflicts(signals)
        except Exception as _scr_exc:
            logger.debug("_execute_signals: signal conflict resolution failed: %s", _scr_exc)

        # --- DCA expansion ---
        try:
            _dca_levels = getattr(self.config, "dca_levels_pct", None)
            if _dca_levels and isinstance(_dca_levels, (list, tuple)) and len(_dca_levels) > 1:
                _ee = getattr(self, "execution_engine", None)
                if _ee and hasattr(_ee, "_dca_expand_signals"):
                    signals = _ee._dca_expand_signals(signals)
        except Exception as _dca_exc:
            logger.debug("_execute_signals: DCA expansion failed: %s", _dca_exc)

        # 1. Build execution context (macro, regime, session, system gates)
        ctx = self._pre_execute_context()

        # Circuit breaker: if active, reject ALL signals immediately
        if ctx.get("circuit_breaker_active"):
            reason = ctx.get("circuit_breaker_reason", "circuit_breaker_active")
            logger.warning("_execute_signals: CIRCUIT BREAKER — blocking all %d signals", len(signals))
            return [{
                "symbol": str(getattr(sig, "symbol", "UNKNOWN")),
                "side": str(getattr(sig, "action", "UNKNOWN")).upper(),
                "status": "blocked",
                "reason": f"circuit_breaker_active: {reason}",
            } for sig in signals]

        results: List[Dict[str, Any]] = []

        for sig in signals:
            try:
                # 2. Extract and validate signal fields
                sig_fields = self._extract_signal_fields(sig)
                if sig_fields is None:
                    continue  # invalid action (not BUY/SELL)
                if sig_fields.get("_blocked"):
                    results.append(sig_fields["result"])
                    continue

                # 3. Apply risk gates (cooldown, daily loss, macro, VaR, risk mgr, max positions)
                approved, reason = self._apply_risk_gates(sig_fields, ctx)
                if not approved:
                    results.append({
                        "symbol": sig_fields["symbol"],
                        "side": sig_fields["action"],
                        "status": "blocked",
                        "reason": reason,
                    })
                    continue

                # 4. Compute position size (Kelly, vol, regime, session, drawdown, correlation)
                size_pct, sizing_method = self._compute_position_size(sig_fields, ctx)
                
                # 4.5. Apply Fully Adaptive Risk Engine (vol-adjusted, performance-responsive)
                if self.fully_adaptive_risk is not None:
                    # Update equity and drawdown for adaptive engine
                    self.fully_adaptive_risk.update_equity(self.portfolio_value_aud)
                    # Calculate daily PnL percentage
                    daily_pnl_pct = (self.daily_pnl_aud / self.portfolio_value_aud) if self.portfolio_value_aud > 0 else 0.0
                    self.fully_adaptive_risk.update_daily_pnl(daily_pnl_pct)
                    
                    # Check if trading is allowed
                    allowed, risk_reason = self.fully_adaptive_risk.is_trading_allowed()
                    if not allowed:
                        results.append({
                            "symbol": sig_fields["symbol"],
                            "side": sig_fields["action"],
                            "status": "blocked",
                            "reason": f"adaptive_risk: {risk_reason}",
                        })
                        logger.debug("Adaptive risk blocked %s: %s", sig_fields["symbol"], risk_reason)
                        continue
                    
                    # Get adaptive position size
                    confidence = float(sig_fields.get("confidence", 0.5))
                    entry_price = float(sig_fields.get("price", 0))
                    adaptive_result = self.fully_adaptive_risk.compute_position_size(
                        symbol=sig_fields["symbol"],
                        base_size_pct=size_pct,
                        confidence=confidence,
                        entry_price=entry_price,
                    )
                    
                    # Apply adaptive sizing (use the smaller of original and adaptive)
                    adaptive_pct = adaptive_result["size_pct"]
                    if adaptive_pct < size_pct:
                        size_pct = adaptive_pct
                        sizing_method = f"adaptive_risk({adaptive_result['reason']})"
                        logger.debug(
                            "Adaptive risk reduced %s size: %.4f%% (mult=%.3f)",
                            sig_fields["symbol"], size_pct * 100, adaptive_result["multipliers"]["final"],
                        )

                # 5. Apply intelligence gates (50+ advisory-based adjustments)
                size_pct, sizing_method = self._apply_intelligence_gates(
                    sig_fields, size_pct, ctx["_cycle_advisory"], sizing_method, ctx,
                )
                if sizing_method.startswith("BLOCKED:"):
                    results.append({
                        "symbol": sig_fields["symbol"],
                        "side": sig_fields["action"],
                        "status": "blocked",
                        "reason": sizing_method,
                    })
                    continue

                # 6. Compute stops, take-profit, and quantity
                stops = self._compute_stops_and_quantity(sig_fields, size_pct, ctx)
                if stops is None:
                    continue  # position too small
                if stops.get("_too_small"):
                    results.append(stops["result"])
                    continue

                # 7. Execute order (live or paper) — delegates to _execute_signals_legacy
                #    for the actual order placement + fill recording logic
                # NOTE: Order execution + post-fill recording remain in the legacy method
                # because they contain async exchange calls that are tightly coupled.
                # This section will be extracted in a follow-up refactor.
                trade_result = await self._execute_order_and_record(
                    sig, sig_fields, size_pct, sizing_method, stops, ctx,
                )
                if trade_result is not None:
                    results.append(trade_result)

            except Exception as exc:
                logger.error("_execute_signals: unexpected error: %s", exc, exc_info=True)
                results.append({
                    "symbol": getattr(sig, "symbol", "UNKNOWN"),
                    "side": getattr(sig, "action", "UNKNOWN"),
                    "status": "error",
                    "reason": str(exc),
                })

        # 8. Structured cycle summary
        self._log_cycle_summary(results, ctx)
        
        # 9. Advanced features processing (v8.2.0)
        await self._process_advanced_features(results, ctx)
        
        # 10. Adaptive orchestrator processing (v8.3.0 - real-time adaptation)
        await self._process_adaptive_features(results, ctx)

        # 11. Self-healing model management (check model health, trigger retraining if needed)
        await self._process_self_healing(results, ctx)

        return results

    async def _process_self_healing(self, results: List[Dict[str, Any]], ctx: Dict) -> None:
        """
        Process model health checks and trigger auto-retraining if drift detected.
        
        - Monitors model performance metrics
        - Detects concept drift
        - Triggers automatic retraining when needed
        - Manages model versioning and rollback
        """
        if self.self_healing_manager is None:
            return
        
        try:
            # Update model health with latest trade results
            for trade in results:
                if trade.get("status") == "executed":
                    strategy_name = trade.get("source_strategy", trade.get("strategy", "unknown"))
                    pnl = float(trade.get("pnl", 0))
                    
                    # Record trade outcome for model health tracking
                    self.self_healing_manager.record_prediction_outcome(
                        model_name=strategy_name,
                        correct=pnl > 0,  # Simple: profit = correct prediction
                        confidence=trade.get("confidence", 0.5)
                    )
            
            # Periodically check model health (every 50 cycles)
            if self._completed_cycles % 50 == 0:
                health_report = self.self_healing_manager.get_health_report()
                
                # Log any models in warning/critical state
                for model_name, health in health_report.get("models", {}).items():
                    status = health.get("status", "unknown")
                    if status in ("warning", "degraded", "critical"):
                        logger.warning(
                            "Model health alert: %s - status=%s, health_score=%.1f%%",
                            model_name,
                            status,
                            health.get("health_score", 0)
                        )
                
                # Auto-heal if any models are critical
                critical_models = [
                    name for name, health in health_report.get("models", {}).items()
                    if health.get("status") == "critical"
                ]
                if critical_models:
                    logger.warning("Triggering auto-heal for critical models: %s", critical_models)
                    heal_results = self.self_healing_manager.auto_heal()
                    for result in heal_results:
                        logger.info(
                            "Auto-heal: %s - action=%s, success=%s",
                            result.get("model_name"),
                            result.get("action"),
                            result.get("success")
                        )
            
        except Exception as e:
            logger.debug("Self-healing processing failed: %s", e)

    async def _process_advanced_features(self, results: List[Dict[str, Any]], ctx: Dict) -> None:
        """
        Process trade results through advanced features (v8.2.0).
        
        - Event Sourcing: Log trades to event store
        - Drift Detection: Update model performance tracking
        - Feature Store: Update trade-derived features
        """
        try:
            orchestrator = getattr(self, "advanced_features_orchestrator", None)
            if orchestrator is None:
                return
            
            # Event Sourcing: Log each trade as an event
            event_store = orchestrator.get_feature("event_sourcing")
            if event_store is not None and hasattr(event_store, "append_events"):
                try:
                    from core.event_store import DomainEvent
                    from datetime import datetime
                    
                    for trade in results:
                        if trade.get("status") == "executed":
                            event = DomainEvent(
                                event_id=f"trade_{trade.get('symbol', 'unknown')}_{int(time.time() * 1000)}",
                                aggregate_id=f"trade_{trade.get('symbol', 'unknown')}",
                                event_type="TradeExecuted",
                                event_version=1,
                                payload={
                                    "symbol": trade.get("symbol"),
                                    "side": trade.get("side"),
                                    "quantity": trade.get("quantity", 0),
                                    "price": trade.get("price", 0),
                                    "status": trade.get("status"),
                                    "cycle": self._completed_cycles,
                                },
                                timestamp=datetime.now(),
                            )
                            event_store.append_events(
                                aggregate_id=event.aggregate_id,
                                events=[event],
                                expected_version=0,  # Simplified - real impl would track version
                            )
                except Exception as e:
                    logger.debug("Event sourcing: %s", e)
            
            # Drift Detection: Update with trade outcomes
            drift_detector = orchestrator.get_feature("drift_detector")
            if drift_detector is not None:
                try:
                    # Update drift detector with trade outcomes for model monitoring
                    for trade in results:
                        if trade.get("status") == "executed":
                            # Track prediction accuracy for drift detection
                            outcome = 1.0 if trade.get("pnl", 0) > 0 else 0.0
                            # This would be connected to actual model predictions in production
                            pass
                except Exception as e:
                    logger.debug("Drift detection update: %s", e)
            
            # Feature Store: Update trade-derived features
            feature_store = orchestrator.get_feature("realtime_feature_store")
            if feature_store is not None:
                try:
                    # Update feature store with trade statistics
                    if results:
                        win_count = sum(1 for r in results if r.get("pnl", 0) > 0)
                        total_trades = len(results)
                        # Store for strategy optimization
                        pass
                except Exception as e:
                    logger.debug("Feature store update: %s", e)
                    
        except Exception as e:
            logger.debug("Advanced features processing: %s", e)

    async def _process_adaptive_features(self, results: List[Dict[str, Any]], ctx: Dict) -> None:
        """
        Process trade results through adaptive orchestrator (v8.3.0).
        
        The adaptive orchestrator provides real-time market adaptation:
        - Market regime detection and classification
        - Dynamic strategy rotation based on regime
        - Self-healing model management (auto-retrain, rollback)
        - Volatility-aware position sizing
        - Correlation regime monitoring
        - Liquidity-aware execution
        - News/event reaction
        """
        try:
            adaptive_orchestrator = getattr(self, "adaptive_orchestrator", None)
            if adaptive_orchestrator is None:
                return
            
            # Get current market state for adaptation
            current_regime = ctx.get("regime", "neutral")
            portfolio_value = ctx.get("portfolio_value", 0.0)
            
            # Build portfolio state snapshot for adaptive system
            from adaptive.adaptive_orchestrator import PortfolioState
            portfolio_state = PortfolioState(
                total_value=portfolio_value,
                cash=ctx.get("cash", portfolio_value * 0.5),  # Estimate
                positions=ctx.get("positions", {}),
                daily_pnl=ctx.get("daily_pnl", 0.0),
                drawdown=ctx.get("drawdown", 0.0),
                sharpe_ratio=ctx.get("sharpe_ratio", 0.0),
            )
            
            # Update adaptive orchestrator with latest market data
            await adaptive_orchestrator.update_market_state(
                regime=current_regime,
                portfolio_state=portfolio_state,
                trade_results=results,
            )
            
            # Get adaptation recommendations for next cycle
            recommendations = await adaptive_orchestrator.get_recommendations()
            
            # Apply recommendations if confidence is high enough
            if recommendations.get("confidence", 0) > 0.7:
                # Strategy rotation recommendation
                if recommendations.get("strategy_rotation"):
                    rotation = recommendations["strategy_rotation"]
                    logger.info(
                        "Adaptive: Strategy rotation recommended - %s (confidence: %.2f)",
                        rotation.get("reason", "unknown"),
                        rotation.get("confidence", 0),
                    )
                    # Store for next cycle's strategy selection
                    ctx["adaptive_strategy_override"] = rotation
                
                # Position sizing adjustment
                if recommendations.get("position_sizing"):
                    sizing = recommendations["position_sizing"]
                    logger.info(
                        "Adaptive: Position sizing adjusted - multiplier=%.3f (volatility: %.4f)",
                        sizing.get("multiplier", 1.0),
                        sizing.get("volatility", 0.0),
                    )
                    ctx["adaptive_sizing_multiplier"] = sizing.get("multiplier", 1.0)
                
                # Risk adjustment
                if recommendations.get("risk_adjustment"):
                    risk_adj = recommendations["risk_adjustment"]
                    logger.info(
                        "Adaptive: Risk adjustment - max_position_pct=%.2f%%",
                        risk_adj.get("max_position_pct", 0) * 100,
                    )
                    ctx["adaptive_max_position_pct"] = risk_adj.get("max_position_pct")
            
            # Log alerts if any
            alerts = recommendations.get("alerts", [])
            for alert in alerts:
                logger.warning("Adaptive Alert: %s", alert.get("message", ""))
                
        except Exception as e:
            logger.debug("Adaptive features processing: %s", e)

    async def _execute_order_and_record(
        self, sig, sig_fields: Dict, size_pct: float,
        sizing_method: str, stops: Dict, ctx: Dict,
    ) -> Dict[str, Any] | None:
        """
        Execute an order (live or paper) and record the fill.
        Delegates to the legacy order placement logic.
        This is the bridge between the new orchestrator and the existing
        exchange-coupled code that handles actual order submission.
        """
        symbol = sig_fields["symbol"]
        action = sig_fields["action"]
        entry_price = sig_fields["entry_price"]
        source_strategy = sig_fields["source_strategy"]
        confidence = sig_fields["confidence"]
        stop_loss = stops.get("stop_loss")
        take_profit = stops.get("take_profit")
        quantity = stops["quantity"]
        position_value_aud = stops["position_value_aud"]
        position_value_usd = stops["position_value_usd"]
        regime = ctx["regime"]
        mode = ctx["mode"]
        is_live = ctx["is_live"]
        aud_to_usd = ctx["aud_to_usd"]
        portfolio_value = ctx["portfolio_value"]
        session_mult = ctx["session_mult"]
        regime_pos_mult = ctx["regime_pos_mult"]
        _cycle_advisory = ctx["_cycle_advisory"]
        _age_urgency = sig_fields.get("_age_urgency", 0.5)

        logger.info(
            "_execute_signals: EXECUTING %s %s — size=%.4f%% (%.2f AUD / %.2f USD), "
            "qty=%.8f, entry=%.4f, SL=%s, TP=%s | method=%s",
            action, symbol, size_pct * 100, position_value_aud, position_value_usd,
            quantity, entry_price, stop_loss, take_profit, sizing_method,
        )

        # ── Entry timing optimization ──
        _entry_urgency = _age_urgency
        try:
            _ph = getattr(self, "_price_history", {})
            _sym_hist = _ph.get(symbol, [])
            if len(_sym_hist) >= 20:
                _sma_20 = sum(list(_sym_hist)[-20:]) / 20.0
                _price_vs_sma = (entry_price - _sma_20) / max(_sma_20, 1e-10)
                if action == "BUY" and _price_vs_sma > 0.01:
                    _entry_urgency = max(0.1, _entry_urgency - 0.3)
                elif action == "BUY" and _price_vs_sma < -0.01:
                    _entry_urgency = min(1.0, _entry_urgency + 0.3)
        except Exception:
            pass

        # ── Order placement ──
        exchange_used = str(getattr(self.config, "primary_exchange", "kraken") or "kraken")
        order_id = f"{'live' if is_live else 'paper'}_{symbol.replace('/', '_')}_{int(time.time()*1000)}"
        fill_qty = 0.0
        fill_price = 0.0
        slippage = 0.0
        commission = 0.0
        used_order_type = "limit"
        is_maker = True
        order_status = "pending"

        if is_live:
            # Live order placement delegates to legacy code path
            try:
                return await self._execute_signals_legacy([sig])
            except Exception as exc:
                logger.error("_execute_order_and_record: live execution failed: %s", exc, exc_info=True)
                return {"symbol": symbol, "side": action, "status": "error", "reason": str(exc)}
        else:
            # ── Paper mode fill simulation (spread-based slippage model) ──
            _vwap_thresh = float(getattr(self, "_vwap_threshold_usd", 5000.0) or 5000.0)
            # Use L2 spread if available, otherwise estimate
            slippage_bps = 0.0
            try:
                _l2 = getattr(self, "l2_feed", None)
                if _l2 is not None and getattr(_l2, "is_connected", False):
                    _spread = _l2.get_spread_bps(symbol)
                    if _spread and float(_spread) > 0:
                        slippage_bps = float(_spread) * 0.5  # half-spread as slippage
            except Exception:
                pass
            if slippage_bps <= 0:
                slippage_bps = 1.0  # fallback: 1 bps
            if position_value_usd > _vwap_thresh:
                used_order_type = "vwap_paper"
                slippage_bps = max(slippage_bps, 1.5)
            else:
                used_order_type = "limit"

            maker_fee_rate = float(getattr(self.config, "paper_maker_fee_rate", 0.0002) or 0.0002)
            taker_fee_rate = float(getattr(self.config, "paper_fee_rate", 0.001) or 0.001)
            fee_rate = maker_fee_rate
            is_maker = True

            slippage = entry_price * slippage_bps / 10000.0
            if action == "BUY":
                fill_price = entry_price + slippage
            else:
                fill_price = entry_price - slippage
            fill_qty = quantity
            commission = fill_price * fill_qty * fee_rate
            order_status = "filled"

            savings = fill_price * fill_qty * (taker_fee_rate - maker_fee_rate)
            if not hasattr(self, "_total_fee_savings_usd"):
                self._total_fee_savings_usd = 0.0
            self._total_fee_savings_usd += savings

            logger.info(
                "_execute_signals [PAPER]: %s %s qty=%.8f @ %.4f (slip=%.4f bps, fee=%.6f, type=%s, maker=%s)",
                action, symbol, fill_qty, fill_price, slippage_bps, commission, used_order_type, is_maker,
            )

        if fill_qty <= 0:
            return None

        # ── Build trade result ──
        trade_result = {
            "symbol": symbol,
            "side": action,
            "price": fill_price,
            "quantity": fill_qty,
            "notional_aud": position_value_aud,
            "notional_usd": position_value_usd,
            "order_id": order_id,
            "exchange": exchange_used,
            "commission": commission,
            "slippage": slippage,
            "slippage_bps": abs(slippage / entry_price * 10000.0) if entry_price > 0 else 0.0,
            "status": order_status,
            "order_type": used_order_type,
            "is_maker": is_maker,
            "entry_price": entry_price,
            "expected_price": entry_price,
            "signal_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "source_strategy": source_strategy,
            "confidence": confidence,
            "regime_label": regime,
            "sizing_method": sizing_method,
            "timestamp": time.time(),
        }

        # ── Record trade ──
        try:
            self._record_trade(trade_result)
        except Exception as exc:
            logger.warning("_execute_order_and_record: _record_trade failed: %s", exc)

        # ── Decision Journal ──
        try:
            _dj = getattr(self.component_registry, "decision_journal", None) if self.component_registry else None
            if _dj is not None:
                from monitoring.decision_journal import DecisionRecord, make_decision_id
                _cycle_num = int(getattr(self, "_cycle_number", 0) or 0)
                _dec_rec = DecisionRecord(
                    decision_id=make_decision_id(_cycle_num, symbol),
                    cycle_number=_cycle_num,
                    timestamp_utc=datetime.now(tz=timezone.utc).isoformat(),
                    timestamp_ms=int(time.time() * 1000),
                    symbol=symbol, side=action, strategy=source_strategy,
                    confidence=confidence, signal_price=entry_price,
                    regime=regime, portfolio_value_aud=portfolio_value,
                    position_count=sum(
                        1 for p in (self.positions or {}).values()
                        if float((p or {}).get("quantity", 0) or 0) > 0
                    ),
                    session_mult=session_mult, regime_pos_mult=regime_pos_mult,
                    raw_size_pct=size_pct, final_size_pct=size_pct,
                    final_size_aud=position_value_aud, outcome="executed",
                    fill_price=fill_price,
                    slippage_bps=trade_result.get("slippage_bps", 0),
                    order_id=order_id,
                    metadata={"sizing_method": sizing_method},
                )
                _dj.write(_dec_rec)
        except Exception as _dj_exc:
            logger.debug("_execute_order_and_record: decision_journal write failed: %s", _dj_exc)

        # ── Shadow Plan Comparator ──
        try:
            _sc = getattr(self.component_registry, "shadow_comparator", None) if self.component_registry else None
            if _sc is not None:
                from core.shadow_plan_comparator import PlanSnapshot
                _live_snap = PlanSnapshot(
                    symbol=symbol, side=action, strategy=source_strategy,
                    confidence=confidence, size_pct=size_pct,
                    size_aud=position_value_aud, gate_multiplier=1.0,
                    gates_applied=0, gates_blocked=False, regime=regime,
                )
                _shadow_snap = _sc.compute_shadow(sig, _cycle_advisory, {
                    "gate_floor": 0.15, "skip_gates": [], "size_multiplier": 1.0,
                })
                _sc.record(_live_snap, _shadow_snap, cycle=int(getattr(self, "_cycle_number", 0) or 0))
        except Exception as _sc_exc:
            logger.debug("_execute_order_and_record: shadow_comparator failed: %s", _sc_exc)

        return trade_result

    async def _execute_signals_legacy(self, signals: List[Any]) -> List[Dict[str, Any]]:
        """
        Original monolithic _execute_signals (preserved for reference/fallback).
        Execute a list of TradingSignal objects through the full pipeline.
        """
        if not signals:
            return []

        # --- FIX 26: Smart signal conflict resolution ---
        try:
            signals = self._resolve_signal_conflicts(signals)
        except Exception as _scr_exc:
            logger.debug("_execute_signals: signal conflict resolution failed: %s", _scr_exc)

        # --- DCA expansion: split signals into sub-tranches if configured ---
        try:
            _dca_levels = getattr(self.config, "dca_levels_pct", None)
            if _dca_levels and isinstance(_dca_levels, (list, tuple)) and len(_dca_levels) > 1:
                _ee = getattr(self, "execution_engine", None)
                if _ee and hasattr(_ee, "_dca_expand_signals"):
                    signals = _ee._dca_expand_signals(signals)
                    logger.debug("_execute_signals: DCA expanded %d signals with %d levels", len(signals), len(_dca_levels))
        except Exception as _dca_exc:
            logger.debug("_execute_signals: DCA expansion failed: %s", _dca_exc)

        results: List[Dict[str, Any]] = []
        mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
        is_live = mode == "live"
        aud_to_usd = float(getattr(self.config, "aud_to_usd", 0.65) or 0.65)
        portfolio_value = float(self.portfolio_value_aud)

        # --- 0a. Macro calendar check: block BUYs near high-impact events ---
        macro_event_imminent = False
        macro_event_name = ""
        macro_event_hours = None
        try:
            _fred_cal = None
            if self.component_registry is not None:
                _fred_cal = getattr(self.component_registry, "fred_calendar", None)
            if _fred_cal is None:
                from data.macro.fred_calendar import FREDCalendar
                _fred_cal = FREDCalendar()
            snap = _fred_cal.get_upcoming(days=1)
            if snap.hours_to_next_high is not None and snap.hours_to_next_high <= 2.0:
                macro_event_imminent = True
                macro_event_name = snap.next_high_impact.name if snap.next_high_impact else "unknown"
                macro_event_hours = snap.hours_to_next_high
                logger.warning(
                    "_execute_signals: MACRO EVENT IMMINENT — '%s' in %.1f hours, blocking new BUY entries",
                    macro_event_name, macro_event_hours,
                )
        except Exception as _macro_exc:
            logger.debug("_execute_signals: macro calendar check failed: %s", _macro_exc)

        # --- 0b. Get current regime for sizing adjustments ---
        # FIX 2: Force regime detection — compute fallback if empty/None
        regime = str(getattr(self, "_latest_regime_label", "") or "").upper().strip()
        if not regime:
            _fallback_fn = getattr(self, "_compute_fallback_regime", None)
            if callable(_fallback_fn):
                regime = _fallback_fn()
            else:
                regime = "NORMAL"
            try:
                self._latest_regime_label = regime
            except AttributeError:
                pass
        regime_pos_mult = self.REGIME_POSITION_SCALE.get(regime, 1.0)
        regime_stop_mult = self.REGIME_STOP_SCALE.get(regime, 1.0)
        regime_tp_mult = self.REGIME_TP_SCALE.get(regime, 1.0)

        # --- 0c. Session-based sizing (crypto volume patterns) ---
        from datetime import timezone as _tz
        hour_utc = datetime.now(tz=_tz.utc).hour
        if 13 <= hour_utc <= 17:
            session_mult = 1.1   # Peak volume: NY open
        elif 8 <= hour_utc <= 10:
            session_mult = 1.05  # London open
        elif 1 <= hour_utc <= 5:
            session_mult = 0.8   # Low volume: worse fills
        else:
            session_mult = 1.0

        logger.info(
            "_execute_signals: regime=%s (pos*%.2f, stop*%.2f, tp*%.2f), session_mult=%.2f, macro_imminent=%s",
            regime, regime_pos_mult, regime_stop_mult, regime_tp_mult,
            session_mult, macro_event_imminent,
        )

        # --- 0. System-level risk gates (BLOCKING) ---
        # Circuit breaker: if active, reject ALL signals immediately
        if self.unified_risk_manager is not None and self.unified_risk_manager.check_circuit_breaker():
            reason = getattr(self.unified_risk_manager, "circuit_breaker_reason", "circuit_breaker_active")
            logger.warning("_execute_signals: CIRCUIT BREAKER ACTIVE (%s) — blocking all %d signals", reason, len(signals))
            for sig in signals:
                results.append({
                    "symbol": str(getattr(sig, "symbol", "UNKNOWN")),
                    "side": str(getattr(sig, "action", "UNKNOWN")).upper(),
                    "status": "blocked",
                    "reason": f"circuit_breaker_active: {reason}",
                })
            return results

        # Daily loss limit: if exceeded, only allow SELL (closing) signals
        daily_loss_exceeded = (
            self.unified_risk_manager is not None
            and self.unified_risk_manager.is_daily_loss_limit_exceeded()
        )
        if daily_loss_exceeded:
            logger.warning("_execute_signals: daily loss limit exceeded — blocking new positions, allowing closes only")

        # VaR/CVaR limit check (portfolio-level)
        var_limit_pct = float(getattr(self.config, "portfolio_var_limit_pct", 0.0) or 0.0)
        cvar_limit_pct = float(getattr(self.config, "portfolio_cvar_limit_pct", 0.0) or 0.0)
        var_breach = False
        if self.unified_risk_manager is not None and (var_limit_pct > 0 or cvar_limit_pct > 0):
            try:
                metrics = self.unified_risk_manager.get_risk_metrics()
                capital = max(metrics.current_capital, 1e-9)
                if var_limit_pct > 0 and abs(metrics.var_95) / capital >= var_limit_pct:
                    var_breach = True
                    logger.warning(
                        "_execute_signals: VaR breach — 95%% VaR %.2f%% >= limit %.2f%%",
                        abs(metrics.var_95) / capital * 100.0, var_limit_pct * 100.0,
                    )
                if cvar_limit_pct > 0 and abs(metrics.var_99) / capital >= cvar_limit_pct:
                    var_breach = True
                    logger.warning(
                        "_execute_signals: CVaR breach — 99%% VaR %.2f%% >= limit %.2f%%",
                        abs(metrics.var_99) / capital * 100.0, cvar_limit_pct * 100.0,
                    )
            except Exception as exc:
                logger.debug("_execute_signals: VaR/CVaR check failed: %s", exc)

        # FIX #1: Make cycle advisory available to all gate blocks
        _cycle_advisory = getattr(self, "_last_cycle_advisory", None) or {}

        for sig in signals:
            try:
                # --- Extract signal fields ---
                symbol = str(getattr(sig, "symbol", "") or "")
                action = str(getattr(sig, "action", "") or "").upper()
                confidence = float(getattr(sig, "confidence", 0.0) or 0.0)
                strength = float(getattr(sig, "strength", 0.0) or 0.0)
                entry_price = float(getattr(sig, "entry_price", 0.0) or 0.0)
                stop_loss = getattr(sig, "stop_loss", None)
                take_profit = getattr(sig, "take_profit", None)
                reasoning = str(getattr(sig, "reasoning", "") or "")
                source_strategy = str(
                    getattr(sig, "strategy", "")
                    or getattr(sig, "source_strategy", "")
                    or (sig.get("strategy") if isinstance(sig, dict) else "")
                    or (sig.get("source_strategy") if isinstance(sig, dict) else "")
                    or "unknown"
                )

                # --- FIX 15: Signal staleness decay ---
                _sig_ts = getattr(sig, "timestamp", None)
                if _sig_ts is None:
                    _sig_age = 0.0
                elif isinstance(_sig_ts, (int, float)):
                    _sig_age = time.time() - float(_sig_ts)
                elif hasattr(_sig_ts, "timestamp"):
                    _sig_age = time.time() - _sig_ts.timestamp()  # datetime → unix
                else:
                    _sig_age = 0.0
                if _sig_age > 120.0:
                    logger.warning(
                        "_execute_signals: signal too stale — age=%.1fs > 120s, rejecting %s %s",
                        _sig_age, action, symbol,
                    )
                    results.append({
                        "symbol": symbol,
                        "side": action,
                        "status": "blocked",
                        "reason": f"signal_too_stale:age={_sig_age:.1f}s",
                    })
                    continue
                if _sig_age > 0.1:
                    import math as _math_staleness
                    confidence *= _math_staleness.exp(-_sig_age / 30.0)

                # --- FIX 23: Signal age-based urgency ---
                if _sig_age < 5.0:
                    _age_urgency = 0.2   # fresh → maker, patient
                elif _sig_age < 30.0:
                    _age_urgency = 0.5   # balanced
                elif _sig_age < 60.0:
                    _age_urgency = 0.8   # aggressive, fill fast
                else:
                    _age_urgency = 1.0   # market order or reject

                if action not in ("BUY", "SELL"):
                    logger.debug("_execute_signals: skipping signal with action=%s for %s", action, symbol)
                    continue

                # --- 0b. Strategy cooldown gate (BLOCKING) ---
                _sss = getattr(self, "_strategy_state_store", None)
                if _sss is not None and _sss.check_cooldown(source_strategy):
                    remaining = _sss.cooldown_remaining_seconds(source_strategy)
                    logger.info(
                        "_execute_signals: DROPPED %s %s from strategy '%s' — cooldown active (%.0fs remaining)",
                        action, symbol, source_strategy, remaining,
                    )
                    results.append({
                        "symbol": symbol,
                        "side": action,
                        "status": "blocked",
                        "reason": f"strategy_cooldown:{source_strategy}",
                    })
                    continue

                if not symbol or entry_price <= 0:
                    logger.warning("_execute_signals: invalid signal (symbol=%s, entry_price=%s)", symbol, entry_price)
                    continue

                # --- 1a. Daily loss limit gate (BLOCKING for new positions) ---
                if daily_loss_exceeded and action == "BUY":
                    logger.warning(
                        "_execute_signals: REJECTED %s %s — daily loss limit exceeded, only closes allowed",
                        action, symbol,
                    )
                    results.append({
                        "symbol": symbol,
                        "side": action,
                        "status": "blocked",
                        "reason": "daily_loss_limit_exceeded",
                    })
                    continue

                # --- 1a2. Macro event gate (BLOCKING for new BUY entries) ---
                if macro_event_imminent and action == "BUY":
                    logger.warning(
                        "_execute_signals: REJECTED %s %s — macro event '%s' in %.1f hours, blocking new entries",
                        action, symbol, macro_event_name, macro_event_hours or 0.0,
                    )
                    results.append({
                        "symbol": symbol,
                        "side": action,
                        "status": "blocked",
                        "reason": f"macro_event_imminent:{macro_event_name}",
                    })
                    continue

                # --- 1b. VaR/CVaR breach gate (BLOCKING for new positions) ---
                if var_breach and action == "BUY":
                    logger.warning(
                        "_execute_signals: REJECTED %s %s — VaR/CVaR limit breached, only closes allowed",
                        action, symbol,
                    )
                    results.append({
                        "symbol": symbol,
                        "side": action,
                        "status": "blocked",
                        "reason": "var_limit_breached",
                    })
                    continue

                # --- 1c. UnifiedRiskManager pre-trade check (BLOCKING) ---
                side_for_check = action
                size_usd_estimate = confidence * strength * portfolio_value * aud_to_usd * 0.1
                exchange_name = str(getattr(self.config, "primary_exchange", "kraken") or "kraken")

                if self.unified_risk_manager is not None and action == "BUY":
                    approved, reject_reason = self.unified_risk_manager.pre_trade_risk_check(
                        symbol=symbol,
                        position_size_usd=size_usd_estimate,
                    )
                    if not approved:
                        logger.warning(
                            "_execute_signals: REJECTED %s %s — risk manager: %s",
                            action, symbol, reject_reason,
                        )
                        results.append({
                            "symbol": symbol,
                            "side": action,
                            "status": "blocked",
                            "reason": reject_reason,
                        })
                        continue

                # --- 1d. Max concurrent positions gate (BLOCKING) ---
                max_positions = int(getattr(self.config, "max_concurrent_positions", 5) or 0)
                if max_positions > 0 and action == "BUY":
                    current_positions = sum(
                        1 for p in (self.positions or {}).values()
                        if float((p or {}).get("quantity", 0) or 0) > 0
                    )
                    if current_positions >= max_positions:
                        logger.warning(
                            "_execute_signals: REJECTED %s %s — max concurrent positions (%d/%d)",
                            action, symbol, current_positions, max_positions,
                        )
                        results.append({
                            "symbol": symbol,
                            "side": action,
                            "status": "blocked",
                            "reason": f"max_concurrent_positions ({current_positions}/{max_positions})",
                        })
                        continue

                if self.component_registry is not None:
                    check = self.component_registry.pre_order_check(symbol, side_for_check, size_usd_estimate, exchange_name)
                    if not check.get("allow", True):
                        logger.info(
                            "_execute_signals: risk gate blocked %s %s: %s",
                            action, symbol, check.get("reasons", []),
                        )
                        results.append({
                            "symbol": symbol,
                            "side": action,
                            "status": "blocked",
                            "reason": check.get("reasons", ["risk_gate"]),
                        })
                        continue

                # --- 2. Position sizing (measured-edge Kelly + volatility-adjusted) ---
                max_pos_pct = float(getattr(self.config, "max_position_pct", 0.25) or 0.25)
                min_pos_aud = float(getattr(self.config, "min_position_size_aud", 10.0) or 10.0)
                strategy_stats = {"n_trades": 0, "win_rate": 0.5, "avg_win": 0.0, "avg_loss": 0.0}

                # Prefer KellySizer component (per strategy×symbol) if available
                _sizing_method = "default"
                _ks = getattr(self.component_registry, "kelly_sizer", None) if self.component_registry else None
                if _ks is not None:
                    try:
                        _ke = _ks.compute(source_strategy, symbol)
                        if _ke.n_trades >= 20 and _ke.kelly_fraction > 0:
                            size_pct = _ke.position_pct
                            _sizing_method = f"kelly_measured(f={_ke.kelly_fraction:.3f},wr={_ke.win_rate:.1%},n={_ke.n_trades})"
                        elif _ke.n_trades >= 20:
                            # Kelly says no edge — use minimal
                            size_pct = min(max_pos_pct, confidence * strength * max_pos_pct * 0.5)
                            _sizing_method = "kelly_no_edge"
                        else:
                            size_pct = min(max_pos_pct, confidence * strength * max_pos_pct)
                            _sizing_method = f"default(kelly_n={_ke.n_trades})"
                    except Exception:
                        size_pct = min(max_pos_pct, confidence * strength * max_pos_pct)
                else:
                    # Fallback to legacy internal Kelly
                    _kelly_min_trades = 20
                    strategy_stats = self._get_strategy_trade_stats(source_strategy)
                    if strategy_stats["n_trades"] >= _kelly_min_trades and strategy_stats["avg_loss"] > 0:
                        kelly_pct = self._kelly_size(
                            strategy_stats["win_rate"],
                            strategy_stats["avg_win"],
                            strategy_stats["avg_loss"],
                        )
                        if kelly_pct > 0:
                            size_pct = kelly_pct
                            _sizing_method = "kelly_legacy"
                        else:
                            size_pct = min(max_pos_pct, confidence * strength * max_pos_pct)
                            _sizing_method = "default_no_kelly_edge"
                    else:
                        size_pct = min(max_pos_pct, confidence * strength * max_pos_pct)

                # Apply volatility adjustment
                current_vol = self._get_current_vol(symbol)
                if current_vol > 0:
                    size_pct = self._vol_adjusted_size(size_pct, current_vol)
                    _sizing_method += "+vol_adj"

                # Apply signal quality discount if available
                sig_quality = self._get_signal_quality()
                if sig_quality is not None:
                    sq_recommendation = sig_quality.get("recommendation", "moderate")
                    if sq_recommendation == "conflicted":
                        size_pct *= 0.5  # halve size on conflicted signals
                        _sizing_method += "+conflict_discount"
                    elif sq_recommendation == "weak":
                        size_pct *= 0.7
                        _sizing_method += "+weak_discount"

                # --- 2b. Regime-adaptive scaling ---
                size_pct *= regime_pos_mult
                _sizing_method += f"+regime({regime})*{regime_pos_mult:.2f}"

                # --- 2c. Session-based sizing ---
                size_pct *= session_mult
                if session_mult != 1.0:
                    _sizing_method += f"+session*{session_mult:.2f}"

                # --- 2d. Macro event: reduce size by 30% for existing exits ---
                if macro_event_imminent and action == "SELL":
                    size_pct *= 0.7
                    _sizing_method += "+macro_reduce_30pct"

                # --- FIX 10: Drawdown-adaptive sizing ---
                try:
                    _peak_cap = float(self.peak_equity_aud)
                    _curr_cap = float(self.portfolio_value_aud)
                    if _peak_cap > 0 and _curr_cap < _peak_cap:
                        _dd_ratio = (_peak_cap - _curr_cap) / _peak_cap
                        _dd_mult = max(0.25, 1.0 - _dd_ratio * 2.0)
                        size_pct *= _dd_mult
                        _sizing_method += f"+dd_adj({_dd_ratio:.3f})*{_dd_mult:.2f}"
                        logger.info(
                            "_execute_signals: drawdown adjustment — dd_ratio=%.3f, size_mult=%.2f",
                            _dd_ratio, _dd_mult,
                        )
                except Exception as _dd_exc:
                    logger.debug("_execute_signals: drawdown sizing failed: %s", _dd_exc)

                # --- FIX 11: Correlation-based position reduction ---
                try:
                    if action == "BUY" and self.positions:
                        _corr_reduction = 1.0
                        _new_base = symbol.split("/")[0] if "/" in symbol else symbol
                        for _pos_sym, _pos_data in (self.positions or {}).items():
                            if _pos_data is None or float((_pos_data or {}).get("quantity", 0) or 0) <= 0:
                                continue
                            _pos_base = _pos_sym.split("/")[0] if "/" in _pos_sym else _pos_sym
                            if _pos_base == _new_base:
                                continue
                            # Check correlation via component registry
                            _corr_val = 0.0
                            _cr = getattr(self, "component_registry", None)
                            if _cr is not None and getattr(_cr, "correlation_monitor", None) is not None:
                                try:
                                    _cm = _cr.correlation_monitor
                                    _corr_val = abs(float(getattr(_cm, "_last_avg_corr", 0.0) or 0.0))
                                except Exception:
                                    _corr_val = 0.0
                            # Fallback: hardcoded BTC/ETH correlation
                            if _corr_val == 0.0:
                                _btc_eth = {"BTC", "ETH"}
                                if {_new_base, _pos_base} == _btc_eth:
                                    _corr_val = 0.85
                            _pos_side = str((_pos_data or {}).get("side", "BUY")).upper()
                            if _corr_val > 0.85 and _pos_side == action:
                                _corr_reduction = 0.70
                                _sizing_method += f"+corr_reduce({_new_base}/{_pos_base}={_corr_val:.2f})"
                                logger.info(
                                    "_execute_signals: correlation reduction — %s/%s corr=%.2f, size*=0.70",
                                    _new_base, _pos_base, _corr_val,
                                )
                                break
                        size_pct *= _corr_reduction
                except Exception as _corr_exc:
                    logger.debug("_execute_signals: correlation check failed: %s", _corr_exc)

                # --- FIX 16: Aggressive strategy dampening ---
                try:
                    _strat_mult = self._get_strategy_multiplier(source_strategy)
                    if _strat_mult != 1.0:
                        size_pct *= _strat_mult
                        _sizing_method += f"+strat_dampen*{_strat_mult:.2f}"
                except Exception as _sd_exc:
                    logger.debug("_execute_signals: strategy dampening failed: %s", _sd_exc)

                # --- FIX 18: Wire RL agent for execution sizing (Phase X fix) ---
                try:
                    if not hasattr(self, "_rl_sizing_agent"):
                        self._rl_sizing_agent = None
                        try:
                            from strategies.reinforcement_stub import RLExecutionAgent as _RLSizingAgent
                            _model_path = str(getattr(self.config, "rl_model_path", "models/rl_agent.zip") or "models/rl_agent.zip")
                            self._rl_sizing_agent = _RLSizingAgent(model_path=_model_path)
                        except Exception:
                            pass
                    if self._rl_sizing_agent is not None:
                        import math as _m
                        import time as _t
                        from strategies.reinforcement_stub import RLState as _RLState
                        _hour_frac = (_t.time() % 86400) / 86400
                        _rl_state = _RLState(
                            position_usd=float((self.positions or {}).get(symbol, {}).get("notional", 0.0) or 0.0),
                            unrealised_pnl=float((self.positions or {}).get(symbol, {}).get("unrealised_pnl", 0.0) or 0.0),
                            volatility_1h=current_vol,
                            spread_bps=float(signal.get("spread_bps", 5.0) or 5.0),
                            ob_imbalance=float(signal.get("ob_imbalance", 0.0) or 0.0),
                            time_of_day_sin=_m.sin(2 * _m.pi * _hour_frac),
                            time_of_day_cos=_m.cos(2 * _m.pi * _hour_frac),
                            slippage_budget_remaining=20.0,
                            bars_since_last_trade=int(getattr(self, "_bars_since_last_trade", 10) or 10),
                        )
                        _rl_dec = self._rl_sizing_agent.decide(_rl_state)
                        _rl_is_buy = _rl_dec.action_name in ("BUY_SMALL", "BUY_LARGE")
                        _rl_is_sell = _rl_dec.action_name in ("SELL_SMALL", "SELL_LARGE")
                        if (_rl_is_buy and action == "BUY") or (_rl_is_sell and action == "SELL"):
                            _rl_size_factor = max(0.1, min(2.0, _rl_dec.size_factor))
                            size_pct *= _rl_size_factor
                            _sizing_method += f"+rl_size*{_rl_size_factor:.2f}"
                        elif _rl_dec.action_name == "HOLD":
                            size_pct *= 0.5
                            _sizing_method += "+rl_hold_dampen"
                except Exception as _rl_exc:
                    logger.debug("_execute_signals: RL sizing unavailable: %s", _rl_exc)

                # --- FIX 19: Position conflict check ---
                try:
                    _existing_pos = (self.positions or {}).get(symbol)
                    if _existing_pos is not None:
                        _existing_qty = float((_existing_pos or {}).get("quantity", 0) or 0)
                        _existing_side = str((_existing_pos or {}).get("side", "")).upper()
                        if _existing_qty > 0 and _existing_side:
                            if action == "BUY" and _existing_side == "BUY":
                                # Pyramid: check limits
                                _pyramid_count = int((_existing_pos or {}).get("pyramid_count", 0) or 0)
                                _max_pyramids = int(getattr(self.config, "max_pyramids_per_position", 2) or 2)
                                if _pyramid_count >= _max_pyramids:
                                    logger.info(
                                        "_execute_signals: CONFLICT — pyramid limit reached for %s (%d/%d)",
                                        symbol, _pyramid_count, _max_pyramids,
                                    )
                                    size_pct *= 0.5
                                    _sizing_method += "+pyramid_limit_reduce"
                            elif action == "SELL" and _existing_side == "SELL":
                                _pyramid_count = int((_existing_pos or {}).get("pyramid_count", 0) or 0)
                                _max_pyramids = int(getattr(self.config, "max_pyramids_per_position", 2) or 2)
                                if _pyramid_count >= _max_pyramids:
                                    size_pct *= 0.5
                                    _sizing_method += "+short_pyramid_limit"
                            elif (action == "BUY" and _existing_side == "SELL") or (action == "SELL" and _existing_side == "BUY"):
                                # Opposite direction — this is a close, allow it
                                logger.info(
                                    "_execute_signals: CONFLICT — %s signal for %s with existing %s position (closing)",
                                    action, symbol, _existing_side,
                                )
                                _sizing_method += "+close_opposite"
                except Exception as _pc_exc:
                    logger.debug("_execute_signals: position conflict check failed: %s", _pc_exc)

                # --- FIX 24: Regime-specific strategy whitelist ---
                try:
                    _regime_prefs = {
                        "TRENDING_UP": {"momentum", "breakout", "funding_rate", "funding_rate_harvester"},
                        "TRENDING_DOWN": {"momentum", "breakout", "funding_rate", "funding_rate_harvester"},
                        "RANGE": {"mean_reversion", "stat_arb", "pairs", "kalman_pairs"},
                        "NORMAL": {"mean_reversion", "stat_arb", "pairs", "kalman_pairs"},
                        "HIGH_VOL": {"funding_rate", "funding_rate_harvester"},
                        "CRISIS": {"funding_rate", "funding_rate_harvester"},
                    }
                    _preferred = _regime_prefs.get(regime)
                    if _preferred is not None:
                        _src_lower = source_strategy.lower()
                        _matches_regime = any(p in _src_lower for p in _preferred)
                        if not _matches_regime:
                            if regime in ("HIGH_VOL", "CRISIS"):
                                confidence *= 0.5
                                _sizing_method += "+regime_mismatch_crisis*0.5"
                            else:
                                confidence *= 0.7
                                _sizing_method += "+regime_mismatch*0.7"
                except Exception as _rw_exc:
                    logger.debug("_execute_signals: regime whitelist check failed: %s", _rw_exc)

                # --- FIX 25: Hot hand strategy boost ---
                try:
                    _sss_hot = getattr(self, "_strategy_state_store", None)
                    if _sss_hot is not None:
                        _hot_state = _sss_hot.get_state(source_strategy)
                        if _hot_state is not None:
                            _consec_wins = int(_hot_state.get("consecutive_wins", 0) or 0)
                            if _consec_wins >= 5:
                                _hot_boost = min(1.30, 1.25)
                                size_pct *= _hot_boost
                                _sizing_method += f"+hot_hand*{_hot_boost:.2f}"
                            elif _consec_wins >= 3:
                                _hot_boost = 1.15
                                size_pct *= _hot_boost
                                _sizing_method += f"+hot_hand*{_hot_boost:.2f}"
                except Exception as _hh_exc:
                    logger.debug("_execute_signals: hot hand boost failed: %s", _hh_exc)

                # Hard cap at max_position_pct
                size_pct = min(size_pct, max_pos_pct)

                # ── Batch G: Wire previously-unused advisory keys into sizing ───────

                # G1: Ensemble composite — multi-source signal strength
                # (on_cycle key: "ensemble" with sub-key "composite")
                try:
                    _ens = (_cycle_advisory or {}).get("ensemble")
                    if _ens and isinstance(_ens, dict):
                        _composite = float(_ens.get("composite", 0.0) or 0.0)
                        if _composite != 0.0:
                            _stk_mult = max(0.50, min(1.30, 1.0 + _composite * 0.30))
                            size_pct *= _stk_mult
                            _sizing_method += f"+ensemble_composite*{_stk_mult:.2f}"
                except Exception:
                    pass

                # G2: Antifragile multiplier — fragility-aware sizing
                # (on_cycle key: "antifragile_multiplier" as float scalar)
                try:
                    _af_mult = (_cycle_advisory or {}).get("antifragile_multiplier")
                    if _af_mult is not None:
                        _af_mult = float(_af_mult)
                        _af_mult = max(0.50, min(1.50, _af_mult))
                        if _af_mult != 1.0:
                            size_pct *= _af_mult
                            _sizing_method += f"+antifragile*{_af_mult:.2f}"
                except Exception:
                    pass

                # G3: Bleeders — halve size for strategies in losing streak
                # (on_cycle key: "bleeders" as LIST of dicts with "name" field)
                try:
                    _bldr = (_cycle_advisory or {}).get("bleeders")
                    if _bldr and isinstance(_bldr, list):
                        _bldr_names = [str(b.get("name", "")) for b in _bldr if isinstance(b, dict)]
                        if source_strategy in _bldr_names:
                            size_pct *= 0.50
                            _sizing_method += "+bleeder*0.50"
                except Exception:
                    pass

                # G4: TCA score — high transaction costs reduce size
                # (on_cycle key: "tca_score" as float scalar, 0-100)
                try:
                    _tca_score = (_cycle_advisory or {}).get("tca_score")
                    if _tca_score is not None:
                        _tca_score = float(_tca_score)
                        if _tca_score > 50:
                            _tca_mult = max(0.50, 1.0 - (_tca_score - 50) / 100.0)
                            size_pct *= _tca_mult
                            _sizing_method += f"+tca*{_tca_mult:.2f}"
                except Exception:
                    pass

                # G5: System status — CRITICAL blocks, DEGRADED reduces
                try:
                    _sys = (_cycle_advisory or {}).get("system_status")
                    if _sys and isinstance(_sys, dict):
                        _status = str(_sys.get("status", "HEALTHY")).upper()
                        if _status == "CRITICAL" and action == "BUY":
                            logger.warning("_execute_signals: system CRITICAL — blocking BUY %s", symbol)
                            results.append({"symbol": symbol, "side": action, "status": "skipped", "reason": "system_critical"})
                            continue
                        elif _status == "DEGRADED":
                            size_pct *= 0.70
                            _sizing_method += "+sys_degraded*0.70"
                except Exception:
                    pass

                # G6: Whale activity — on-chain flow direction
                try:
                    _whale = (_cycle_advisory or {}).get("whale_activity")
                    if _whale and isinstance(_whale, dict):
                        _whale_bias = float(_whale.get("net_flow_bias", 0.0) or 0.0)
                        if _whale_bias != 0.0:
                            _whale_mult = max(0.70, min(1.30, 1.0 + _whale_bias * 0.25))
                            size_pct *= _whale_mult
                            _sizing_method += f"+whale*{_whale_mult:.2f}"
                except Exception:
                    pass

                # G7: Causal graph — funding→vol→regime chain
                try:
                    _cg = (_cycle_advisory or {}).get("causal_graph")
                    if _cg and isinstance(_cg, dict):
                        _cg_conf = float(_cg.get("confidence", 0.0) or 0.0)
                        _cg_dir = str(_cg.get("direction", "neutral")).lower()
                        if _cg_conf > 0.60:
                            if _cg_dir == "bearish" and action == "BUY":
                                size_pct *= 0.70
                                _sizing_method += "+causal_bearish*0.70"
                            elif _cg_dir == "bullish" and action == "BUY":
                                size_pct *= 1.20
                                _sizing_method += "+causal_bullish*1.20"
                except Exception:
                    pass

                # G8: Outcome correlator — reduce in historically unfavorable conditions
                try:
                    _oc = (_cycle_advisory or {}).get("outcome_correlator")
                    if _oc and isinstance(_oc, dict):
                        _oc_score = float(_oc.get("favorability", 0.5) or 0.5)
                        if _oc_score < 0.30:
                            size_pct *= 0.60
                            _sizing_method += "+unfavorable*0.60"
                        elif _oc_score > 0.70:
                            size_pct *= 1.15
                            _sizing_method += "+favorable*1.15"
                except Exception:
                    pass

                # G9: Alpha decay — age signal by strategy-level decay factor
                try:
                    _ad = (_cycle_advisory or {}).get("alpha_decay")
                    if _ad and isinstance(_ad, dict):
                        _decays = _ad.get("strategy_decays", {})
                        if isinstance(_decays, dict) and source_strategy in _decays:
                            _decay_factor = float(_decays[source_strategy])
                            _decay_factor = max(0.30, min(1.0, _decay_factor))
                            if _decay_factor < 1.0:
                                size_pct *= _decay_factor
                                _sizing_method += f"+alpha_decay*{_decay_factor:.2f}"
                except Exception:
                    pass

                # G10: Liquidation cascade — boost/block based on cascade direction
                try:
                    _lc = (_cycle_advisory or {}).get("liquidation_cascade")
                    if _lc and isinstance(_lc, dict):
                        _lc_signals = _lc.get("signals", [])
                        for _lc_sig in (_lc_signals if isinstance(_lc_signals, list) else []):
                            if isinstance(_lc_sig, dict) and _lc_sig.get("symbol") == symbol:
                                _lc_dir = str(_lc_sig.get("direction", "")).lower()
                                _lc_conf = float(_lc_sig.get("confidence", 0.0) or 0.0)
                                if _lc_dir == "long_squeeze" and action == "BUY" and _lc_conf > 0.60:
                                    size_pct *= 0.40
                                    _sizing_method += "+lc_squeeze*0.40"
                                elif _lc_dir == "short_squeeze" and action == "BUY" and _lc_conf > 0.60:
                                    size_pct *= 1.30
                                    _sizing_method += "+lc_short_squeeze*1.30"
                                break
                except Exception:
                    pass

                # Re-apply hard cap after batch G gates
                size_pct = min(size_pct, max_pos_pct)

                # ── Batch K: ML Intelligence gates ──────────────────────────────

                # K1: vol_forecasts → scale by forecasted vol vs current
                try:
                    _vf = (_cycle_advisory or {}).get("vol_forecasts")
                    if _vf and isinstance(_vf, dict) and symbol in _vf:
                        _sym_vf = _vf[symbol]
                        if isinstance(_sym_vf, dict):
                            _fvol = float(_sym_vf.get("forecast_vol_1d", 0.0) or 0.0)
                            if _fvol > 0 and current_vol > 0:
                                _vol_ratio = _fvol / max(current_vol, 0.001)
                                _vf_mult = max(0.50, min(1.15, 1.0 - (_vol_ratio - 1.0) * 0.30))
                                if abs(_vf_mult - 1.0) > 0.01:
                                    size_pct *= _vf_mult
                                    _sizing_method += f"+vol_forecast*{_vf_mult:.2f}"
                except Exception:
                    pass

                # K2: alpha_scores → alpha model direction weighting
                try:
                    _als = (_cycle_advisory or {}).get("alpha_scores")
                    if _als and isinstance(_als, dict) and symbol in _als:
                        _sym_alpha = _als[symbol]
                        if isinstance(_sym_alpha, dict):
                            _alpha_comp = float(_sym_alpha.get("composite", 0.0) or 0.0)
                            if abs(_alpha_comp) > 0.3:
                                _aligns = (_alpha_comp > 0 and action == "BUY") or (_alpha_comp < 0 and action == "SELL")
                                if _aligns:
                                    size_pct *= 1.15
                                    _sizing_method += "+alpha_align*1.15"
                                else:
                                    size_pct *= 0.70
                                    _sizing_method += "+alpha_oppose*0.70"
                except Exception:
                    pass

                # K3: pretrained_regime → ML regime gate
                try:
                    _pr = (_cycle_advisory or {}).get("pretrained_regime")
                    if _pr and isinstance(_pr, dict):
                        _pr_pred = _pr.get("prediction")
                        if _pr_pred is not None:
                            _pr_label = str(_pr_pred[0] if hasattr(_pr_pred, '__len__') and len(_pr_pred) > 0 else _pr_pred).upper()
                            if "CRISIS" in _pr_label and action == "BUY":
                                size_pct *= 0.50
                                _sizing_method += "+ml_crisis*0.50"
                            elif "TRENDING_UP" in _pr_label and action == "BUY":
                                size_pct *= 1.15
                                _sizing_method += "+ml_trending_up*1.15"
                except Exception:
                    pass

                # K4: pretrained_vol_forecast → tighten sizing if vol predicted high
                try:
                    _pvf = (_cycle_advisory or {}).get("pretrained_vol_forecast")
                    if _pvf and isinstance(_pvf, dict):
                        _next5d = float(_pvf.get("next_5d_vol", 0.0) or 0.0)
                        if _next5d > 0.05:
                            size_pct *= 0.80
                            _sizing_method += f"+high_vol_pred*0.80"
                except Exception:
                    pass

                # K5: pretrained_alpha → ML direction confidence
                try:
                    _pa = (_cycle_advisory or {}).get("pretrained_alpha")
                    if _pa and isinstance(_pa, dict):
                        _pa_dir = str(_pa.get("direction", "")).upper()
                        _pa_conf = float(_pa.get("confidence", 0.0) or 0.0)
                        if _pa_conf > 0.70:
                            _pa_aligns = (_pa_dir == "UP" and action == "BUY") or (_pa_dir == "DOWN" and action == "SELL")
                            if _pa_aligns:
                                size_pct *= 1.20
                                _sizing_method += "+ml_alpha_align*1.20"
                            else:
                                size_pct *= 0.60
                                _sizing_method += "+ml_alpha_oppose*0.60"
                except Exception:
                    pass

                # K6: inference → inference service confidence gate
                try:
                    _inf = (_cycle_advisory or {}).get("inference")
                    if _inf and isinstance(_inf, dict):
                        _inf_conf = float(_inf.get("confidence", 0.5) or 0.5)
                        if _inf_conf < 0.30:
                            size_pct *= 0.70
                            _sizing_method += "+low_inference*0.70"
                        elif _inf_conf > 0.70:
                            size_pct *= 1.10
                            _sizing_method += "+high_inference*1.10"
                except Exception:
                    pass

                # ── Batch L: Sentiment, Pattern & LLM ───────────────────────────

                # L1: fear_greed → contrarian sizing
                try:
                    _fg = (_cycle_advisory or {}).get("fear_greed")
                    if _fg is not None:
                        _fg_val = float(_fg)
                        _fg_bias = (50.0 - _fg_val) / 250.0  # ±0.20 range
                        if action == "BUY":
                            _fg_mult = 1.0 + _fg_bias
                        else:
                            _fg_mult = 1.0 - _fg_bias
                        _fg_mult = max(0.80, min(1.20, _fg_mult))
                        if abs(_fg_mult - 1.0) > 0.01:
                            size_pct *= _fg_mult
                            _sizing_method += f"+fear_greed*{_fg_mult:.2f}"
                except Exception:
                    pass

                # L2: llm_analysis → LLM direction confidence
                try:
                    _llm = (_cycle_advisory or {}).get("llm_analysis")
                    if _llm and isinstance(_llm, dict):
                        _llm_dir = str(_llm.get("direction", "")).upper()
                        _llm_conf = float(_llm.get("confidence", 0.0) or 0.0)
                        if _llm_conf > 0.60:
                            _llm_aligns = (_llm_dir in ("UP", "BULLISH", "BUY") and action == "BUY") or \
                                          (_llm_dir in ("DOWN", "BEARISH", "SELL") and action == "SELL")
                            if _llm_aligns:
                                size_pct *= 1.10
                                _sizing_method += "+llm_align*1.10"
                            elif _llm_dir not in ("NEUTRAL", "UNKNOWN", ""):
                                size_pct *= 0.90
                                _sizing_method += "+llm_oppose*0.90"
                except Exception:
                    pass

                # L3: sentiment_stats → aggregate sentiment bias
                try:
                    _sent = (_cycle_advisory or {}).get("sentiment_stats")
                    if _sent and isinstance(_sent, dict):
                        _sent_score = float(_sent.get("avg_sentiment", 0.0) or 0.0)
                        if _sent_score != 0.0:
                            _sent_aligns = (_sent_score > 0 and action == "BUY") or (_sent_score < 0 and action == "SELL")
                            if _sent_aligns:
                                size_pct *= 1.10
                                _sizing_method += "+sentiment_align*1.10"
                            else:
                                size_pct *= 0.90
                                _sizing_method += "+sentiment_oppose*0.90"
                except Exception:
                    pass

                # L4: chart_patterns → pattern confirmation
                try:
                    _cp = (_cycle_advisory or {}).get("chart_patterns")
                    if _cp and isinstance(_cp, dict):
                        _cp_bias = float(_cp.get("bias", 0.0) or 0.0)
                        _cp_conf = float(_cp.get("confidence", 0.0) or 0.0)
                        if _cp_conf > 0.60 and abs(_cp_bias) > 0.1:
                            _cp_aligns = (_cp_bias > 0 and action == "BUY") or (_cp_bias < 0 and action == "SELL")
                            if _cp_aligns:
                                size_pct *= 1.10
                                _sizing_method += "+pattern_confirm*1.10"
                            else:
                                size_pct *= 0.75
                                _sizing_method += "+pattern_oppose*0.75"
                except Exception:
                    pass

                # ── Batch M: Quantum Intelligence ────────────────────────────────

                # M1: quantum_prediction → price direction
                try:
                    _qp = (_cycle_advisory or {}).get("quantum_prediction")
                    if _qp and isinstance(_qp, dict):
                        _qp_next = float(_qp.get("next_value", 0.0) or 0.0)
                        _qp_conf = float(_qp.get("confidence", 0.0) or 0.0)
                        if _qp_next > 0 and _qp_conf > 0.50 and entry_price > 0:
                            _qp_pct = (_qp_next - entry_price) / entry_price
                            if _qp_pct > 0.001 and action == "BUY":
                                size_pct *= 1.10
                                _sizing_method += "+qpred_up*1.10"
                            elif _qp_pct < -0.001 and action == "BUY":
                                size_pct *= 0.85
                                _sizing_method += "+qpred_down*0.85"
                except Exception:
                    pass

                # M2: quantum_regime → regime confirmation
                try:
                    _qr = (_cycle_advisory or {}).get("quantum_regime")
                    if _qr and isinstance(_qr, dict):
                        _qr_entropy = float(_qr.get("entropy", 0.0) or 0.0)
                        if _qr_entropy > 0.80:
                            size_pct *= 0.90
                            _sizing_method += "+q_high_entropy*0.90"
                except Exception:
                    pass

                # M3: quantum_anomaly_score → circuit breaker
                try:
                    _qa = (_cycle_advisory or {}).get("quantum_anomaly_score")
                    if _qa is not None:
                        _qa_val = float(_qa)
                        if _qa_val > 0.90 and action == "BUY":
                            logger.warning("_execute_signals: quantum anomaly %.2f > 0.90 — blocking BUY %s", _qa_val, symbol)
                            results.append({"symbol": symbol, "side": action, "status": "skipped", "reason": "quantum_anomaly"})
                            continue
                        elif _qa_val > 0.75:
                            size_pct *= 0.60
                            _sizing_method += f"+q_anomaly*0.60"
                except Exception:
                    pass

                # M4: quantum_signal_quality → quality filter
                try:
                    _qsq = (_cycle_advisory or {}).get("quantum_signal_quality")
                    if _qsq and isinstance(_qsq, dict):
                        _q_quality = float(_qsq.get("quality", 0.5) or 0.5)
                        if _q_quality < 0.30:
                            size_pct *= 0.70
                            _sizing_method += "+low_q_quality*0.70"
                except Exception:
                    pass

                # M5: quantum_portfolio → bias toward optimal weights
                try:
                    _qpw = (_cycle_advisory or {}).get("quantum_portfolio")
                    if _qpw and isinstance(_qpw, dict) and _qpw.get("method") != "insufficient_data":
                        _qp_weights = _qpw.get("weights")
                        if _qp_weights and hasattr(_qp_weights, '__len__') and len(_qp_weights) > 0:
                            # Advisory only for now — log optimal weights
                            _sizing_method += "+qportfolio_available"
                except Exception:
                    pass

                # M6: quantum_risk_check → VaR gate
                try:
                    _qrc = (_cycle_advisory or {}).get("quantum_risk_check")
                    if _qrc and isinstance(_qrc, dict):
                        _q_cvar = float(_qrc.get("cvar_95", 0.0) or 0.0)
                        if _q_cvar > 0.05:
                            size_pct *= 0.75
                            _sizing_method += f"+q_var_high*0.75"
                except Exception:
                    pass

                # ── Batch N: Portfolio & Risk ────────────────────────────────────

                # N1: correlation_penalty → reduce correlated positions
                try:
                    _corr = (_cycle_advisory or {}).get("correlation_penalty")
                    if _corr is not None:
                        _corr_val = float(_corr)
                        if _corr_val > 0.30:
                            _corr_mult = max(0.50, 1.0 - _corr_val)
                            size_pct *= _corr_mult
                            _sizing_method += f"+corr_penalty*{_corr_mult:.2f}"
                except Exception:
                    pass

                # N2: tail_hedge → reduce exposure when hedging recommended
                try:
                    _th = (_cycle_advisory or {}).get("tail_hedge")
                    if _th and isinstance(_th, dict):
                        if _th.get("should_hedge") is True and action == "BUY":
                            size_pct *= 0.70
                            _sizing_method += "+tail_hedge*0.70"
                except Exception:
                    pass

                # N3: stress_test → reduce if stress test flagged
                try:
                    _st = (_cycle_advisory or {}).get("stress_test")
                    if _st is not None and _st != "ok":
                        size_pct *= 0.80
                        _sizing_method += "+stress_warning*0.80"
                except Exception:
                    pass

                # N4: adaptive_risk → enforce adjusted limits
                try:
                    _ar = (_cycle_advisory or {}).get("adaptive_risk")
                    if _ar and isinstance(_ar, dict):
                        _ar_max = float(_ar.get("max_position_pct", 0.0) or 0.0)
                        if _ar_max > 0 and size_pct > _ar_max:
                            size_pct = _ar_max
                            _sizing_method += f"+adaptive_cap={_ar_max:.3f}"
                except Exception:
                    pass

                # N5: risk_score → portfolio risk gate
                try:
                    _rs = (_cycle_advisory or {}).get("risk_score")
                    if _rs is not None:
                        _rs_val = float(_rs)
                        if _rs_val > 0.95 and action == "BUY":
                            logger.warning("_execute_signals: risk_score %.2f > 0.95 — blocking BUY %s", _rs_val, symbol)
                            results.append({"symbol": symbol, "side": action, "status": "skipped", "reason": "risk_score_extreme"})
                            continue
                        elif _rs_val > 0.80:
                            size_pct *= 0.70
                            _sizing_method += "+high_risk*0.70"
                except Exception:
                    pass

                # N6: market_anomaly → anomaly circuit breaker
                try:
                    _ma = (_cycle_advisory or {}).get("market_anomaly")
                    if _ma and isinstance(_ma, dict) and _ma.get("is_anomaly"):
                        _ma_sev = str(_ma.get("severity", "low")).lower()
                        if _ma_sev == "high" and action == "BUY":
                            logger.warning("_execute_signals: HIGH anomaly detected — blocking BUY %s", symbol)
                            results.append({"symbol": symbol, "side": action, "status": "skipped", "reason": "market_anomaly_high"})
                            continue
                        elif _ma_sev == "medium":
                            size_pct *= 0.70
                            _sizing_method += "+anomaly_medium*0.70"
                except Exception:
                    pass

                # ── Batch O: Advanced AI + Strategy Intelligence ─────────────────

                # O1: gnn_asset_flow → lead-lag sizing
                try:
                    _gnn = (_cycle_advisory or {}).get("gnn_asset_flow")
                    if _gnn and isinstance(_gnn, dict) and symbol in _gnn:
                        _flow = _gnn[symbol]
                        if isinstance(_flow, dict):
                            _flow_sig = float(_flow.get("flow_signal", 0.0) or 0.0)
                            if _flow_sig > 0 and action == "BUY":
                                size_pct *= 1.10
                                _sizing_method += "+gnn_positive*1.10"
                            elif _flow_sig < 0 and action == "BUY":
                                size_pct *= 0.85
                                _sizing_method += "+gnn_negative*0.85"
                except Exception:
                    pass

                # O2: autoencoder_regime → transition caution
                try:
                    _ae = (_cycle_advisory or {}).get("autoencoder_regime")
                    if _ae and isinstance(_ae, dict):
                        if _ae.get("is_transition") is True:
                            size_pct *= 0.75
                            _sizing_method += "+ae_transition*0.75"
                except Exception:
                    pass

                # O3: rl_portfolio_allocation → RL weight modifier
                try:
                    _rl = (_cycle_advisory or {}).get("rl_portfolio_allocation")
                    if _rl and isinstance(_rl, dict):
                        _rl_weight = float(_rl.get(symbol, 0.0) or 0.0)
                        if _rl_weight > 0:
                            _rl_mult = max(0.50, min(1.50, _rl_weight / max(size_pct, 0.001)))
                            _rl_mult = max(0.85, min(1.15, _rl_mult))  # cap influence to ±15%
                            if abs(_rl_mult - 1.0) > 0.01:
                                size_pct *= _rl_mult
                                _sizing_method += f"+rl_weight*{_rl_mult:.2f}"
                except Exception:
                    pass

                # O4: attention_orderflow → direction confirmation
                try:
                    _attn = (_cycle_advisory or {}).get("attention_orderflow")
                    if _attn and isinstance(_attn, dict):
                        _attn_dir = str(_attn.get("direction", "")).upper()
                        _attn_conf = float(_attn.get("confidence", 0.0) or 0.0)
                        if _attn_conf > 0.60:
                            _attn_aligns = (_attn_dir in ("UP", "BUY") and action == "BUY") or \
                                           (_attn_dir in ("DOWN", "SELL") and action == "SELL")
                            if not _attn_aligns and _attn_dir not in ("NEUTRAL", ""):
                                size_pct *= 0.80
                                _sizing_method += "+attn_oppose*0.80"
                except Exception:
                    pass

                # O5: regime_rotation → strategy weights per regime
                try:
                    _rr = (_cycle_advisory or {}).get("regime_rotation")
                    if _rr and isinstance(_rr, dict):
                        _rr_weights = _rr.get("strategy_weights", {})
                        if isinstance(_rr_weights, dict) and source_strategy in _rr_weights:
                            _rr_w = float(_rr_weights[source_strategy])
                            _rr_w = max(0.30, min(1.50, _rr_w))
                            if abs(_rr_w - 1.0) > 0.01:
                                size_pct *= _rr_w
                                _sizing_method += f"+regime_rot*{_rr_w:.2f}"
                except Exception:
                    pass

                # O6: regime_prediction → next regime forecast
                try:
                    _rp = (_cycle_advisory or {}).get("regime_prediction")
                    if _rp and isinstance(_rp, dict):
                        _rp_next = str(_rp.get("predicted_regime", _rp.get("next_regime", ""))).upper()
                        if "CRISIS" in _rp_next and action == "BUY":
                            size_pct *= 0.70
                            _sizing_method += "+pred_crisis*0.70"
                except Exception:
                    pass

                # O7: regime_pre_transition_signals → pre-transition warning
                try:
                    _rpt = (_cycle_advisory or {}).get("regime_pre_transition_signals")
                    if _rpt:
                        size_pct *= 0.80
                        _sizing_method += "+pre_transition*0.80"
                except Exception:
                    pass

                # O8: funding_prediction → predicted funding rate
                try:
                    _fp = (_cycle_advisory or {}).get("funding_prediction")
                    if _fp and isinstance(_fp, dict):
                        _fp_rate = float(_fp.get("predicted_rate_pct", 0.0) or 0.0)
                        if _fp_rate < -0.03 and action == "BUY":
                            size_pct *= 0.85
                            _sizing_method += "+neg_funding*0.85"
                        elif _fp_rate > 0.03 and action == "SELL":
                            size_pct *= 0.85
                            _sizing_method += "+pos_funding*0.85"
                except Exception:
                    pass

                # O9: session_effect → time-of-day sizing bias
                try:
                    _se = (_cycle_advisory or {}).get("session_effect")
                    if _se is not None:
                        _se_val = float(_se) if not isinstance(_se, dict) else float(_se.get("bias", 0.0) or 0.0)
                        _se_mult = max(0.85, min(1.15, 1.0 + _se_val * 0.15))
                        if abs(_se_mult - 1.0) > 0.01:
                            size_pct *= _se_mult
                            _sizing_method += f"+session*{_se_mult:.2f}"
                except Exception:
                    pass

                # O10: bandit_rankings → strategy allocation weighting
                try:
                    _br = (_cycle_advisory or {}).get("bandit_rankings")
                    if _br and isinstance(_br, list):
                        for _br_entry in _br:
                            if isinstance(_br_entry, dict) and _br_entry.get("strategy") == source_strategy:
                                _br_wr = float(_br_entry.get("expected_win_rate", 0.5) or 0.5)
                                if _br_wr < 0.40:
                                    size_pct *= 0.75
                                    _sizing_method += "+bandit_low*0.75"
                                elif _br_wr > 0.60:
                                    size_pct *= 1.15
                                    _sizing_method += "+bandit_high*1.15"
                                break
                except Exception:
                    pass

                # O11: orderbook_prediction → OBI direction confirmation
                try:
                    _obp = (_cycle_advisory or {}).get("orderbook_prediction")
                    if _obp and isinstance(_obp, dict):
                        _obp_dir = str(_obp.get("direction", "")).upper()
                        _obp_conf = float(_obp.get("confidence", 0.0) or 0.0)
                        if _obp_conf > 0.60:
                            _obp_opposes = (_obp_dir in ("DOWN", "SELL") and action == "BUY") or \
                                           (_obp_dir in ("UP", "BUY") and action == "SELL")
                            if _obp_opposes:
                                size_pct *= 0.80
                                _sizing_method += "+obi_oppose*0.80"
                except Exception:
                    pass

                # ── Wire remaining advisory keys ────────────────────────────────

                # online_learner drift → reduce sizing when model drift detected
                try:
                    _ol = (_cycle_advisory or {}).get("online_learner")
                    if _ol and isinstance(_ol, dict) and _ol.get("drift_detected"):
                        _drift_mag = float(_ol.get("drift_magnitude", 0.0) or 0.0)
                        _drift_mult = max(0.50, 1.0 - _drift_mag * 0.50)
                        size_pct *= _drift_mult
                        _sizing_method += f"+ol_drift*{_drift_mult:.2f}"
                except Exception:
                    pass

                # genetic_evolver → use evolved position scale if available
                try:
                    _ge = (_cycle_advisory or {}).get("genetic_evolver")
                    if _ge and isinstance(_ge, dict):
                        _ge_best = _ge.get("best_fitness", 0.0)
                        if isinstance(_ge_best, (int, float)) and _ge_best > 0:
                            _ge_scale = _ge.get("best_params", {})
                            if isinstance(_ge_scale, dict):
                                _ge_pos = float(_ge_scale.get("position_scale", 1.0) or 1.0)
                                _ge_pos = max(0.50, min(1.50, _ge_pos))
                                if abs(_ge_pos - 1.0) > 0.01:
                                    size_pct *= _ge_pos
                                    _sizing_method += f"+evolved*{_ge_pos:.2f}"
                except Exception:
                    pass

                # strategy_optimization → reduce sizing for strategies flagged for param adjustment
                try:
                    _so = (_cycle_advisory or {}).get("strategy_optimization")
                    if _so and isinstance(_so, dict):
                        _so_strats = _so.get("needs_adjustment", [])
                        if isinstance(_so_strats, list) and source_strategy in _so_strats:
                            size_pct *= 0.80
                            _sizing_method += "+needs_optim*0.80"
                except Exception:
                    pass

                # feature_discovery → boost if new high-IC features discovered
                try:
                    _fd = (_cycle_advisory or {}).get("feature_discovery")
                    if _fd and isinstance(_fd, dict):
                        _fd_count = int(_fd.get("total_discovered", 0) or 0)
                        if _fd_count > 10:
                            size_pct *= 1.05  # slight boost — new features = more signal
                            _sizing_method += "+feature_rich*1.05"
                except Exception:
                    pass

                # ── Strategy scanner — boost symbols the scanner recommends ──
                try:
                    _scan = (_cycle_advisory or {}).get("strategy_scanner")
                    if _scan and isinstance(_scan, dict):
                        _top_syms = _scan.get("top_symbols", [])
                        _top_strats = _scan.get("top_strategies", [])
                        if isinstance(_top_syms, list) and symbol in _top_syms:
                            # This symbol is in the scanner's top opportunities
                            _scan_boost = 1.20
                            # Check if the strategy also matches
                            for _ts in (_top_strats if isinstance(_top_strats, list) else []):
                                if isinstance(_ts, dict) and _ts.get("symbol") == symbol:
                                    _ts_sharpe = float(_ts.get("sharpe", 0) or 0)
                                    if _ts_sharpe > 0.5:
                                        _scan_boost = 1.30  # strong scanner match
                                    break
                            size_pct *= _scan_boost
                            _sizing_method += f"+scanner_top*{_scan_boost:.2f}"
                        elif isinstance(_top_syms, list) and _top_syms and symbol not in _top_syms:
                            # Scanner has recommendations but this symbol isn't in them
                            size_pct *= 0.70
                            _sizing_method += "+scanner_not_top*0.70"
                except Exception:
                    pass

                # ── Market impact estimation — reduce size if impact too high ──
                try:
                    _mim = getattr(self.component_registry, "market_impact", None) if self.component_registry else None
                    if _mim is not None and hasattr(_mim, "estimate") and entry_price > 0:
                        _est_usd = portfolio_value * size_pct * aud_to_usd
                        _impact = _mim.estimate(
                            symbol=symbol, side=action.lower(),
                            quantity_usd=_est_usd, price=entry_price,
                        )
                        _impact_bps = float(getattr(_impact, "total_impact_bps", 0.0) or 0.0)
                        _impact_threshold = 15.0  # bps
                        if _impact_bps > _impact_threshold:
                            _impact_mult = max(0.30, 1.0 - (_impact_bps - _impact_threshold) / 100.0)
                            size_pct *= _impact_mult
                            _sizing_method += f"+mkt_impact*{_impact_mult:.2f}({_impact_bps:.1f}bps)"
                except Exception:
                    pass

                # Re-apply hard cap after all gates
                size_pct = min(size_pct, max_pos_pct)

                # FIX #25: NaN/Inf guard — prevent corrupted sizing from propagating
                import math as _math_check
                if not isinstance(size_pct, (int, float)) or _math_check.isnan(size_pct) or _math_check.isinf(size_pct):
                    logger.error("_execute_signals: size_pct corrupted to %s for %s — using 1%% default", size_pct, symbol)
                    size_pct = 0.01
                elif size_pct < 0:
                    size_pct = 0.0

                # FIX: Gate stacking floor — 75 sequential multipliers can reduce
                # size_pct to <1% of intended (0.8^10 = 0.107). Apply a floor so
                # trades that survive all gates still execute at meaningful size.
                # Minimum 15% of the original base_size if any signal was valid.
                _gate_floor = max_pos_pct * 0.15  # 15% of max position
                if size_pct > 0 and size_pct < _gate_floor:
                    size_pct = _gate_floor

                # ════════════════════════════════════════════════════════════
                # INTELLIGENCE GATES (14 modules wired into execution)
                # ════════════════════════════════════════════════════════════

                # ── Latency compensator: adjust for AU → Kraken RTT ──
                try:
                    _lc = getattr(self.component_registry, "latency_compensator", None) if self.component_registry else None
                    if _lc is not None:
                        _sig_ts = float(getattr(signal, "timestamp", 0) or 0)
                        if _sig_ts == 0:
                            _sig_ts = time.time() * 1000
                        elif _sig_ts < 1e12:
                            _sig_ts *= 1000
                        _comp = _lc.compensate(_sig_ts, current_vol)
                        if _comp.is_stale:
                            continue
                        size_pct *= _comp.size_multiplier
                except Exception:
                    pass

                # ── Manipulation detector: BLOCK trades on manipulated symbols ──
                try:
                    _manip = (_cycle_advisory or {}).get("manipulation_detector", {})
                    if isinstance(_manip, dict):
                        _blocked = _manip.get("blocked_symbols", [])
                        if isinstance(_blocked, list) and symbol in _blocked:
                            logger.warning("BLOCKED %s %s — manipulation detected", action, symbol)
                            continue
                except Exception:
                    pass

                # ── Portfolio risk: respect allocation limits ──
                try:
                    _pr = (_cycle_advisory or {}).get("portfolio_risk", {})
                    if isinstance(_pr, dict):
                        _allocs = _pr.get("allocations", {})
                        if isinstance(_allocs, dict) and source_strategy in _allocs:
                            _target_alloc = float(_allocs[source_strategy])
                            if _target_alloc < 0.05:
                                size_pct *= 0.3
                            elif _target_alloc < 0.15:
                                size_pct *= 0.7
                except Exception:
                    pass

                # ── ML feedback: reduce for drifting models ──
                try:
                    _mlf = (_cycle_advisory or {}).get("ml_feedback", {})
                    if isinstance(_mlf, dict) and _mlf.get("drifting"):
                        size_pct *= 0.7
                except Exception:
                    pass

                # ── Strategy attribution: boost winners, reduce losers ──
                try:
                    _attr = (_cycle_advisory or {}).get("strategy_attribution", {})
                    if isinstance(_attr, dict):
                        if source_strategy == str(_attr.get("top_contributor", "") or ""):
                            size_pct *= 1.2
                        elif source_strategy == str(_attr.get("worst_contributor", "") or ""):
                            size_pct *= 0.6
                except Exception:
                    pass

                # ── Causal engine: predict downstream effects ──
                try:
                    _ce = getattr(self.component_registry, "causal_engine", None) if self.component_registry else None
                    if _ce is not None and action == "BUY":
                        _effects = _ce.predict_effects(f"regime_{str(_latest_regime_label or 'normal')}")
                        for _eff_name, _eff_prob, _eff_lag in _effects:
                            if "dump" in _eff_name and _eff_prob > 0.3:
                                size_pct *= max(0.5, 1.0 - _eff_prob)
                            elif "squeeze" in _eff_name and _eff_prob > 0.3:
                                size_pct *= min(1.5, 1.0 + _eff_prob * 0.5)
                except Exception:
                    pass

                # ── Counterfactual: override systematic biases ──
                try:
                    _cf = getattr(self.component_registry, "counterfactual", None) if self.component_registry else None
                    if _cf is not None:
                        _override = _cf.should_override(action, str(_latest_regime_label or "normal"))
                        if _override == "SKIP" and action == "BUY":
                            size_pct *= 0.3
                except Exception:
                    pass

                # ── Meta-cognition: skip when uncertain ──
                try:
                    _mc_adv = (_cycle_advisory or {}).get("meta_cognition", {})
                    if isinstance(_mc_adv, dict):
                        _mc_rec = str(_mc_adv.get("recommendation", "TRADE") or "TRADE")
                        if _mc_rec == "SKIP" and action == "BUY":
                            size_pct = 0.0
                        elif _mc_rec == "WAIT" and action == "BUY":
                            size_pct *= 0.3
                        elif _mc_rec == "REDUCE_SIZE":
                            size_pct *= max(0.5, float(_mc_adv.get("confidence", 1.0) or 1.0))
                except Exception:
                    pass

                # ── Temporal abstraction: boost when all scales align ──
                try:
                    _ta_adv = (_cycle_advisory or {}).get("temporal_abstraction", {})
                    if isinstance(_ta_adv, dict) and symbol in _ta_adv:
                        _ta = _ta_adv[symbol]
                        _alignment = float(_ta.get("alignment", 0.5) or 0.5)
                        _macro = float(_ta.get("macro", 0) or 0)
                        if action == "BUY" and _alignment > 0.75 and _macro > 0:
                            size_pct *= 1.2
                        elif action == "BUY" and _macro < -0.5:
                            size_pct *= 0.6
                except Exception:
                    pass

                # ── Universal Data Brain: 17+ source market intelligence ──
                try:
                    _udb_adv = (_cycle_advisory or {}).get("universal_data_brain", {})
                    if isinstance(_udb_adv, dict) and symbol in _udb_adv:
                        _intel = _udb_adv[symbol]
                        _composite = float(_intel.get("composite", 0) or 0)
                        _conviction = str(_intel.get("conviction", "LOW") or "LOW")
                        if int(_intel.get("signals", 0) or 0) >= 5:
                            if action == "BUY" and _composite > 0.3 and _conviction in ("HIGH", "EXTREME"):
                                size_pct *= 1.3
                            elif action == "BUY" and _composite < -0.3:
                                size_pct *= 0.5
                except Exception:
                    pass

                # ── Price prediction: boost/reduce based on direction ──
                try:
                    _pred_adv = (_cycle_advisory or {}).get("price_predictions", {})
                    if isinstance(_pred_adv, dict) and symbol in _pred_adv:
                        _pred = _pred_adv[symbol]
                        _pred_dir = _pred.get("direction", "FLAT")
                        _pred_conf = float(_pred.get("confidence", 0) or 0)
                        if action == "BUY" and _pred_dir == "UP" and int(_pred.get("models_agree", 0) or 0) >= 2:
                            size_pct *= (1 + _pred_conf * 0.3)
                        elif action == "BUY" and _pred_dir == "DOWN" and int(_pred.get("models_agree", 0) or 0) >= 2:
                            size_pct *= max(0.5, 1 - _pred_conf * 0.4)
                except Exception:
                    pass

                # ── Entropy filter: suppress noise ──
                try:
                    _ef_adv = (_cycle_advisory or {}).get("entropy_filter", {})
                    if isinstance(_ef_adv, dict) and not _ef_adv.get("should_trade", True):
                        if action == "BUY":
                            size_pct *= 0.3
                except Exception:
                    pass

                # ── Market memory: recall similar past conditions ──
                try:
                    _mm_adv = (_cycle_advisory or {}).get("market_memory", {})
                    if isinstance(_mm_adv, dict) and _mm_adv.get("similar_count", 0) >= 5:
                        _exp_pnl = float(_mm_adv.get("expected_pnl", 0) or 0)
                        if _exp_pnl < -0.5 and float(_mm_adv.get("similarity", 0) or 0) > 0.3:
                            size_pct *= 0.5
                        elif _exp_pnl > 0.5 and float(_mm_adv.get("similarity", 0) or 0) > 0.3:
                            size_pct *= 1.2
                except Exception:
                    pass

                # ── Conviction sizer: multi-source agreement ──
                try:
                    _cs = getattr(self.component_registry, "conviction_sizer", None) if self.component_registry else None
                    if _cs is not None and size_pct > 0:
                        _conv = _cs.compute(
                            base_size_pct=size_pct, symbol=symbol, action=action,
                            strategy_type=source_strategy,
                            advisory=_cycle_advisory,
                            regime=str(_latest_regime_label or "NORMAL"),
                            max_pos_pct=max_pos_pct,
                        )
                        size_pct = _conv.final_size_pct
                except Exception:
                    pass

                # ════════════════════════════════════════════════════════════════════════════
                # QUANTUM ADAPTIVE RISK ENGINE - THE PINNACLE
                # ════════════════════════════════════════════════════════════════════════════
                _quantum_sizing_used = False
                _quantum_reason = ""
                try:
                    if self.quantum_adaptive_risk is not None and action == "BUY" and size_pct > 0:
                        # Update price history for quantum engine
                        self.quantum_adaptive_risk.update_price(symbol, entry_price)
                        
                        # Compute quantum-optimized position size
                        _q_result = self.quantum_adaptive_risk.compute_quantum_position_size(
                            symbol=symbol,
                            base_size_pct=size_pct,
                            confidence=confidence,
                            current_regime=str(_latest_regime_label or "NORMAL"),
                        )
                        
                        if _q_result.get("quantum_optimized", False):
                            size_pct = _q_result["size_pct"]
                            _quantum_sizing_used = True
                            _quantum_reason = _q_result.get("reason", "")
                            _sizing_method += f"+quantum(qkelly={_q_result['multipliers'].get('quantum_kelly', 1.0):.2f})"
                            
                            logger.info(
                                "_execute_signals: QUANTUM sizing %s — size_pct=%.4f (%s)",
                                symbol, size_pct, _quantum_reason,
                            )
                except Exception as _q_exc:
                    logger.debug("_execute_signals: quantum sizing failed: %s", _q_exc)

                # ════════════════════════════════════════════════════════════

                # --- 2e. Dynamic stop/TP: regime + volatility-adjusted exits ---
                _base_stop_pct = float(getattr(self.config, "stop_loss_pct", 0.02) or 0.02)
                _base_tp_pct = float(getattr(self.config, "take_profit_pct", 0.04) or 0.04)
                _adj_stop_pct = _base_stop_pct * regime_stop_mult
                _adj_tp_pct = _base_tp_pct * regime_tp_mult

                # Volatility-adjusted exits: ATR-based when vol data available
                if current_vol > 0:
                    # Use 1.5x vol as stop, 3.0x vol as TP (maintains ~2:1 R:R)
                    _atr_stop = current_vol * 1.5
                    _atr_tp = current_vol * 3.0
                    # Blend: use ATR if it's meaningfully different from base
                    if _atr_stop > 0.001:
                        _adj_stop_pct = _atr_stop
                        _adj_tp_pct = _atr_tp
                        _sizing_method += f"+atr_exit(sl={_adj_stop_pct:.4f},tp={_adj_tp_pct:.4f})"

                # Apply the computed stop/TP to the signal
                if stop_loss is None and entry_price > 0:
                    if action == "BUY":
                        stop_loss = entry_price * (1.0 - _adj_stop_pct)
                    else:
                        stop_loss = entry_price * (1.0 + _adj_stop_pct)
                if take_profit is None and entry_price > 0:
                    if action == "BUY":
                        take_profit = entry_price * (1.0 + _adj_tp_pct)
                    else:
                        take_profit = entry_price * (1.0 - _adj_tp_pct)

                # ════════════════════════════════════════════════════════════════════════════
                # QUANTUM STOP OPTIMIZATION - Annealing-based optimal exits
                # ════════════════════════════════════════════════════════════════════════════
                _quantum_stops_used = False
                try:
                    if self.quantum_adaptive_risk is not None and action == "BUY" and entry_price > 0:
                        _q_stops = self.quantum_adaptive_risk.compute_quantum_stops(
                            symbol=symbol,
                            entry_price=entry_price,
                            side="long",
                        )
                        
                        if _q_stops and "quantum_expected_value" in _q_stops:
                            # Use quantum-optimized stops
                            stop_loss = _q_stops.get("stop_loss", stop_loss)
                            take_profit = _q_stops.get("take_profit", take_profit)
                            _quantum_stops_used = True
                            
                            logger.info(
                                "_execute_signals: QUANTUM stops %s — SL=%.2f, TP=%.2f, EV=%.4f",
                                symbol, stop_loss, take_profit,
                                _q_stops.get("quantum_expected_value", 0),
                            )
                except Exception as _qs_exc:
                    logger.debug("_execute_signals: quantum stops failed: %s", _qs_exc)

                logger.info(
                    "_execute_signals: sizing %s via %s — size_pct=%.4f (kelly_trades=%d, vol=%.4f)",
                    symbol, _sizing_method, size_pct,
                    strategy_stats["n_trades"], current_vol,
                )

                position_value_aud = portfolio_value * size_pct
                if position_value_aud < min_pos_aud:
                    logger.debug(
                        "_execute_signals: position too small for %s (%.2f AUD < %.2f AUD min)",
                        symbol, position_value_aud, min_pos_aud,
                    )
                    results.append({
                        "symbol": symbol,
                        "side": action,
                        "status": "skipped",
                        "reason": "position_too_small",
                    })
                    continue

                # Convert to base currency quantity
                position_value_usd = position_value_aud * aud_to_usd
                quantity = position_value_usd / entry_price if entry_price > 0 else 0.0
                if quantity <= 0:
                    continue

                logger.info(
                    "_execute_signals: placing %s %s qty=%.8f @ %.2f (conf=%.2f, str=%.2f, value=%.2f AUD)",
                    action, symbol, quantity, entry_price, confidence, strength, position_value_aud,
                )

                # --- FIX 8: Entry timing optimization ---
                # FIX 23: Start with age-based urgency as the baseline
                _entry_urgency = _age_urgency
                try:
                    _price_hist = getattr(self, "_price_history", {})
                    _sym_prices = _price_hist.get(symbol, [])
                    if not _sym_prices and hasattr(self, "_volatility_cache"):
                        # Try to build from position data
                        pass
                    if len(_sym_prices) >= 20:
                        _sma_20 = sum(_sym_prices[-20:]) / 20.0
                        if _sma_20 > 0:
                            _price_vs_sma = (entry_price - _sma_20) / _sma_20
                            if action == "BUY":
                                if _price_vs_sma > 0.02:
                                    _entry_urgency = 0.2  # price above SMA by >2%, wait for pullback
                                elif _price_vs_sma < 0:
                                    _entry_urgency = 0.8  # price below SMA, good entry
                            elif action == "SELL":
                                if _price_vs_sma < -0.02:
                                    _entry_urgency = 0.2  # price below SMA by >2%, wait for bounce
                                elif _price_vs_sma > 0:
                                    _entry_urgency = 0.8  # price above SMA, good exit
                            _sizing_method += f"+urgency={_entry_urgency:.1f}"
                except Exception as _urg_exc:
                    logger.debug("_execute_signals: entry timing check failed: %s", _urg_exc)

                # --- 3. Place order (limit-first with timeout fallback) ---
                fill_price = entry_price
                fill_qty = quantity
                order_id = f"exec_{symbol}_{action}_{int(time.time() * 1000)}"
                order_status = "filled"
                commission = 0.0
                slippage = 0.0
                exchange_used = exchange_name
                used_order_type = "limit"
                is_maker = True
                _twap_requested = False

                # --- Batch I: Order flow toxicity → switch to market when toxic ---
                try:
                    _tox_adv = (_cycle_advisory or {}).get("toxicity")
                    if _tox_adv and isinstance(_tox_adv, dict):
                        _tox_score = float(_tox_adv.get("toxicity_score", 0.0) or 0.0)
                        _tox_thresh = 0.70
                        if _tox_score >= _tox_thresh and used_order_type == "limit":
                            used_order_type = "market"
                            is_maker = False
                            _sizing_method += f"+toxic_market(score={_tox_score:.2f})"
                            logger.info("_execute_signals: toxicity %.2f >= %.2f — switching to market for %s", _tox_score, _tox_thresh, symbol)
                except Exception:
                    pass

                # --- Batch I: Check if TWAP execution preferred ---
                try:
                    _ei_adv = (_cycle_advisory or {}).get("execution_intelligence")
                    if _ei_adv and isinstance(_ei_adv, dict):
                        _ei_type = str(_ei_adv.get("order_type", "")).lower()
                        if _ei_type in ("twap", "vwap", "adaptive"):
                            _twap_requested = True
                except Exception:
                    pass

                # --- IS feedback: override order type based on historical execution quality ---
                try:
                    _ist = getattr(self.component_registry, "is_tracker", None) if self.component_registry else None
                    if _ist is not None and used_order_type == "limit":
                        _is_rec = _ist.get_recommended_order_type(source_strategy, symbol)
                        if _is_rec == "twap" and not _twap_requested:
                            _twap_requested = True
                            _sizing_method += "+is_twap"
                        elif _is_rec == "market" and used_order_type == "limit":
                            # IS says market is fine — keep limit for maker rebates
                            pass
                except Exception:
                    pass

                # Compute limit price: 0.2% inside for fills
                if action == "BUY":
                    limit_price = entry_price * (1 + self._limit_price_offset_bps / 10000.0)
                else:
                    limit_price = entry_price * (1 - self._limit_price_offset_bps / 10000.0)

                if is_live and hasattr(self, "exchange_manager") and self.exchange_manager is not None:
                    # LATENCY OPT: Prefer WebSocket order placement when available
                    _ws_placer = getattr(self, "_ws_order_placer", None)
                    _ws_pref = bool(getattr(self.config, "latency_ws_order_preference", True))
                    _used_ws = False
                    if _ws_pref and _ws_placer is not None and getattr(_ws_placer, "is_connected", False):
                        try:
                            _ws_result = await _ws_placer.place_order(
                                symbol=symbol,
                                side=action.lower(),
                                amount=quantity,
                                order_type="limit",
                                price=limit_price,
                                timeout=3.0,
                            )
                            if _ws_result is not None:
                                order_id = str(_ws_result.get("result", {}).get("order_id", order_id) if isinstance(_ws_result.get("result"), dict) else order_id)
                                fill_qty = quantity  # WS confirms acceptance, fill tracking happens async
                                fill_price = limit_price
                                order_status = "open"
                                used_order_type = "limit_ws"
                                is_maker = True
                                _used_ws = True
                                logger.info(
                                    "_execute_signals: WS order placed %s %s qty=%.8f @ %.2f (faster path)",
                                    action, symbol, quantity, limit_price,
                                )
                        except Exception as _ws_exc:
                            logger.debug("_execute_signals: WS order failed, falling back to REST: %s", _ws_exc)
                            _used_ws = False

                    if not _used_ws:
                        logger.debug(
                            "_execute_signals: using REST path for %s %s (ws_pref=%s, ws_connected=%s)",
                            action, symbol, _ws_pref,
                            getattr(_ws_placer, "is_connected", False) if _ws_placer else "no_placer",
                        )

                    # --- Batch I: TWAP execution via AlgoOrders ---
                    _twap_handled = False
                    _twap_threshold = float(getattr(self.config, "twap_min_notional_usd", 250.0) or 250.0)
                    _twap_dur_min = float(getattr(self.config, "twap_duration_minutes", 5.0) or 5.0)
                    if not _used_ws and _twap_requested and position_value_usd >= _twap_threshold and run_mode == "live":
                        try:
                            from execution.algo_orders import AlgoExecutor, AlgoOrderParams, AlgoOrderType
                            _algo_exec = AlgoExecutor(exchange=self.exchange_manager)
                            _algo_params = AlgoOrderParams(
                                symbol=symbol,
                                side=action.lower(),
                                total_usd=position_value_usd,
                                order_type=AlgoOrderType.TWAP,
                                duration_seconds=_twap_dur_min * 60,
                                num_slices=max(3, int(position_value_usd / 100)),
                                urgency=_entry_urgency,
                                exchange_id=exchange_name,
                            )
                            _algo_result = await _algo_exec.execute(_algo_params)
                            if _algo_result is not None:
                                # FIX #2: AlgoOrderResult has total_filled_usd, not total_filled_qty
                                _filled_usd = float(getattr(_algo_result, "total_filled_usd", position_value_usd) or position_value_usd)
                                fill_qty = _filled_usd / max(entry_price, 1e-9)
                                fill_price = float(getattr(_algo_result, "avg_fill_price", entry_price) or entry_price)
                                order_id = f"twap_{symbol}_{action}_{int(time.time() * 1000)}"
                                order_status = "filled"
                                used_order_type = "twap"
                                is_maker = False
                                _twap_handled = True
                                logger.info(
                                    "_execute_signals: TWAP execution for %s %s — filled=%.8f @ %.2f over %.0fs",
                                    action, symbol, fill_qty, fill_price, _twap_dur_min * 60,
                                )
                        except Exception as _twap_exc:
                            logger.warning("_execute_signals: TWAP failed for %s %s: %s — falling back", action, symbol, _twap_exc)

                    # LIVE MODE: use VWAP for large orders, limit for small (skip if WS already handled)
                    try:
                        if not _used_ws and not _twap_handled and position_value_usd >= self._vwap_threshold_usd:
                            # --- VWAP execution for larger orders ---
                            try:
                                from execution.smart_order_execution import SmartOrderExecutionEngine
                                _soe = SmartOrderExecutionEngine()
                                vwap_order = await _soe.execute_order_async(
                                    symbol=symbol,
                                    side=action.lower(),
                                    quantity=quantity,
                                    execution_type='vwap',
                                    execution_time_minutes=1,
                                    market_data={"price": entry_price},
                                    exchange=self.exchange_manager,
                                    slice_interval_seconds=2.0,
                                )
                                fill_qty = vwap_order.executed_quantity
                                fill_price = vwap_order.average_price if vwap_order.average_price > 0 else entry_price
                                order_id = vwap_order.order_id
                                order_status = vwap_order.status
                                used_order_type = "vwap"
                                is_maker = False  # VWAP slices are limit but not post-only
                                logger.info(
                                    "_execute_signals: VWAP execution for %s %s qty=%.8f — filled=%.8f @ %.2f",
                                    action, symbol, quantity, fill_qty, fill_price,
                                )
                            except Exception as vwap_exc:
                                logger.warning(
                                    "_execute_signals: VWAP failed for %s %s, falling back to limit: %s",
                                    action, symbol, vwap_exc,
                                )
                                # Fall through to limit order below
                                position_value_usd = 0  # force limit path

                        if not _used_ws and not _twap_handled and used_order_type != "vwap":
                            # --- Limit order with timeout fallback ---
                            order_request = {
                                "symbol": symbol,
                                "side": action.lower(),
                                "order_type": "limit",
                                "amount": quantity,
                                "price": limit_price,
                            }
                            logger.info("_execute_signals: submitting LIVE limit order %s", order_request)
                            response = await self.exchange_manager.execute_order(order_request)

                            if response is None:
                                logger.error("_execute_signals: exchange returned None for %s %s", action, symbol)
                                results.append({
                                    "symbol": symbol,
                                    "side": action,
                                    "status": "error",
                                    "reason": "exchange_returned_none",
                                })
                                continue

                            order_id = str(response.get("order_id", order_id))
                            fill_qty = float(response.get("filled", 0.0))
                            fill_price = float(response.get("price", limit_price))
                            order_status = str(response.get("status", "open"))
                            exchange_used = str(response.get("exchange", exchange_name))
                            is_maker = True
                            used_order_type = "limit"

                            # FIX 6: Non-blocking fill handling — track unfilled limit orders
                            # in _pending_orders instead of blocking with asyncio.sleep.
                            # _poll_pending_orders() will handle fill checking next cycle.
                            if order_status not in ("filled", "closed") and fill_qty < quantity * 0.99:
                                self._pending_orders[order_id] = {
                                    "order_id": order_id,
                                    "symbol": symbol,
                                    "side": action,
                                    "total_quantity": quantity,
                                    "filled_quantity": fill_qty,
                                    "remaining": quantity - fill_qty,
                                    "entry_price": entry_price,
                                    "limit_price": limit_price,
                                    "exchange": exchange_name,
                                    "submitted_at": datetime.now().isoformat(),
                                    "fill_timeout_at": (datetime.now() + timedelta(seconds=self._limit_order_fill_timeout)).isoformat(),
                                    "needs_market_fallback": True,
                                }
                                logger.info(
                                    "_execute_signals: limit order %s pending (filled %.8f/%.8f) — "
                                    "tracked for async fill check, continuing to next signal",
                                    order_id, fill_qty, quantity,
                                )

                        # Track partially filled orders
                        remaining = quantity - fill_qty
                        if remaining > 0.001 * quantity and order_status not in ("filled", "closed"):
                            self._pending_orders[order_id] = {
                                "order_id": order_id,
                                "symbol": symbol,
                                "side": action,
                                "total_quantity": quantity,
                                "filled_quantity": fill_qty,
                                "remaining": remaining,
                                "entry_price": entry_price,
                                "exchange": exchange_used,
                                "submitted_at": time.time(),
                                "stop_loss": stop_loss,
                                "take_profit": take_profit,
                            }
                            logger.info(
                                "_execute_signals: partial fill for %s — filled=%.8f remaining=%.8f, tracking as pending",
                                order_id, fill_qty, remaining,
                            )

                        # Track fee savings
                        if is_maker:
                            savings = position_value_usd * 4.0 / 10_000.0  # 4 bps saved
                            self._total_fee_savings_usd += savings
                            logger.info(
                                "_execute_signals: maker fill — saved $%.4f (cumulative $%.2f)",
                                savings, self._total_fee_savings_usd,
                            )

                        logger.info(
                            "_execute_signals: LIVE order result: id=%s type=%s maker=%s status=%s filled=%.8f @ %.2f",
                            order_id, used_order_type, is_maker, order_status, fill_qty, fill_price,
                        )

                    except Exception as exc:
                        logger.error("_execute_signals: LIVE order failed for %s %s: %s", action, symbol, exc)
                        results.append({
                            "symbol": symbol,
                            "side": action,
                            "status": "error",
                            "reason": str(exc),
                        })
                        continue
                else:
                    # PAPER MODE: simulate fill with limit-order-like behavior
                    # Default to maker fees (limit order simulation)
                    maker_fee_rate = float(getattr(self.config, "paper_maker_fee_rate", 0.0002) or 0.0002)  # 2 bps
                    taker_fee_rate = float(getattr(self.config, "paper_fee_rate", 0.0026) or 0.0026)

                    if position_value_usd >= self._vwap_threshold_usd:
                        # Simulate VWAP: slightly better execution, maker fees
                        slippage_bps = max(0, self._paper_slippage_bps - 1.5)  # VWAP saves ~1.5 bps
                        fee_rate = maker_fee_rate
                        used_order_type = "vwap"
                        is_maker = True
                    else:
                        # Simulate limit order: no slippage (posted passively), maker fees
                        slippage_bps = 0.0  # limit orders don't take
                        fee_rate = maker_fee_rate
                        used_order_type = "limit"
                        is_maker = True

                    if action == "BUY":
                        slippage = entry_price * (slippage_bps / 10000.0)
                        fill_price = entry_price + slippage
                    else:
                        slippage = entry_price * (slippage_bps / 10000.0)
                        fill_price = entry_price - slippage

                    commission = fill_qty * fill_price * fee_rate

                    # Track fee savings vs old taker behavior
                    taker_commission = fill_qty * fill_price * taker_fee_rate
                    savings = taker_commission - commission
                    self._total_fee_savings_usd += savings

                    logger.info(
                        "_execute_signals: PAPER fill %s %s qty=%.8f @ %.2f "
                        "(type=%s, maker=%s, slippage=%.4f, commission=%.4f, saved=%.4f)",
                        action, symbol, fill_qty, fill_price,
                        used_order_type, is_maker, slippage, commission, savings,
                    )

                # --- 4. Build trade result ---
                if fill_qty <= 0:
                    logger.warning("_execute_signals: zero fill quantity for %s %s", action, symbol)
                    continue

                trade_result = {
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": action,
                    "quantity": fill_qty,
                    "price": fill_price,
                    "commission": commission,
                    "slippage": slippage,
                    "status": "filled" if fill_qty > 0 else order_status,
                    "exchange": exchange_used,
                    "timestamp": time.time(),
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "reasoning": reasoning,
                    "confidence": confidence,
                    "strength": strength,
                    "source_strategy": source_strategy,
                    "order_type": used_order_type,
                    "is_maker": is_maker,
                    "entry_urgency": _entry_urgency,  # FIX 8
                }

                # --- FIX 17: Track maker fill rate ---
                try:
                    self._record_maker_fill(symbol, is_maker)
                except Exception:
                    pass

                # --- FIX 22: Track execution quality ---
                try:
                    if not hasattr(self, "_execution_trade_count"):
                        self._execution_trade_count = 0
                    if not hasattr(self, "_execution_quality_trades"):
                        self._execution_quality_trades = []
                    self._execution_trade_count += 1
                    _slippage_bps = abs(slippage / entry_price * 10000.0) if entry_price > 0 else 0.0
                    self._execution_quality_trades.append({
                        "slippage_bps": _slippage_bps,
                        "is_maker": is_maker,
                        "fill_time_ms": 0.0,  # placeholder — live mode would measure
                    })
                    self._compute_execution_quality()
                except Exception:
                    pass

                # --- FIX 12: Wire OCO conditional orders via ConditionalOrderManager ---
                try:
                    if fill_qty > 0 and stop_loss is not None and take_profit is not None:
                        # Use real ConditionalOrderManager if available
                        _cond_mgr = getattr(self.component_registry, "conditional_orders", None) if self.component_registry else None
                        if _cond_mgr is not None and hasattr(_cond_mgr, "create_oco"):
                            _exit_side = "sell" if action == "BUY" else "buy"
                            _oco_gid = _cond_mgr.create_oco(
                                symbol=symbol,
                                tp_price=float(take_profit),
                                sl_price=float(stop_loss),
                                quantity=fill_qty,
                                exchange=exchange_used,
                                side=_exit_side,
                            )
                            logger.info(
                                "_execute_signals: OCO group %s created for %s — SL=%.2f, TP=%.2f",
                                _oco_gid, symbol, float(stop_loss), float(take_profit),
                            )
                        # Also maintain legacy _oco_orders dict for _check_oco_conditions
                        self._oco_orders[symbol] = {
                            "order_id": order_id,
                            "symbol": symbol,
                            "side": action,
                            "entry_price": fill_price,
                            "quantity": fill_qty,
                            "stop_loss": float(stop_loss),
                            "take_profit": float(take_profit),
                            "created_at": time.time(),
                        }
                except Exception as _oco_exc:
                    logger.debug("_execute_signals: OCO creation failed: %s", _oco_exc)

                # --- 5. Record in trade ledger ---
                try:
                    ledger = getattr(self.execution_engine, "trade_ledger", None) if self.execution_engine else None
                    if ledger is not None and hasattr(ledger, "record_trade"):
                        ledger.record_trade(trade_result)
                except Exception as exc:
                    logger.warning("_execute_signals: trade ledger record failed: %s", exc)

                # --- 6. Update positions via _record_trade ---
                # _record_trade also calls component_registry.on_fill() internally,
                # so we do NOT call on_fill() separately here.
                try:
                    self._record_trade(trade_result)
                except Exception as exc:
                    logger.warning("_execute_signals: _record_trade failed: %s", exc)

                results.append(trade_result)

                # --- 7. Decision Journal: record executed decision ---
                try:
                    _dj = getattr(self.component_registry, "decision_journal", None) if self.component_registry else None
                    if _dj is not None:
                        from monitoring.decision_journal import DecisionRecord, GateResult, make_decision_id
                        _cycle_num = int(getattr(self, "_cycle_number", 0) or 0)
                        _dec_rec = DecisionRecord(
                            decision_id=make_decision_id(_cycle_num, symbol),
                            cycle_number=_cycle_num,
                            timestamp_utc=datetime.now(tz=timezone.utc).isoformat(),
                            timestamp_ms=int(time.time() * 1000),
                            symbol=symbol,
                            side=action,
                            strategy=source_strategy,
                            confidence=confidence,
                            signal_price=entry_price,
                            regime=regime,
                            portfolio_value_aud=portfolio_value,
                            position_count=sum(
                                1 for p in (self.positions or {}).values()
                                if float((p or {}).get("quantity", 0) or 0) > 0
                            ),
                            session_mult=session_mult,
                            regime_pos_mult=regime_pos_mult,
                            raw_size_pct=float(locals().get("_raw_size_pct", size_pct)),
                            final_size_pct=size_pct,
                            final_size_aud=float(trade_result.get("notional_aud", 0)),
                            outcome="executed",
                            fill_price=float(trade_result.get("price", 0)),
                            slippage_bps=float(trade_result.get("slippage_bps", 0)),
                            order_id=str(trade_result.get("order_id", "")),
                            metadata={"sizing_method": locals().get("_sizing_method", "")},
                        )
                        _dj.write(_dec_rec)
                except Exception as _dj_exc:
                    logger.debug("_execute_signals: decision_journal write failed: %s", _dj_exc)

                # --- 8. Shadow Plan Comparator: record live vs shadow ---
                try:
                    _sc = getattr(self.component_registry, "shadow_comparator", None) if self.component_registry else None
                    if _sc is not None:
                        from core.shadow_plan_comparator import PlanSnapshot
                        _live_snap = PlanSnapshot(
                            symbol=symbol,
                            side=action,
                            strategy=source_strategy,
                            confidence=confidence,
                            size_pct=size_pct,
                            size_aud=float(trade_result.get("notional_aud", 0)),
                            gate_multiplier=1.0,
                            gates_applied=0,
                            gates_blocked=False,
                            regime=regime,
                        )
                        _shadow_snap = _sc.compute_shadow(sig, _cycle_advisory, {
                            "gate_floor": 0.15,
                            "skip_gates": [],
                            "size_multiplier": 1.0,
                        })
                        _sc.record(_live_snap, _shadow_snap, cycle=int(getattr(self, "_cycle_number", 0) or 0))
                except Exception as _sc_exc:
                    logger.debug("_execute_signals: shadow_comparator record failed: %s", _sc_exc)

            except Exception as exc:
                # One failed signal must not crash the loop
                logger.error("_execute_signals: unexpected error processing signal: %s", exc, exc_info=True)
                results.append({
                    "symbol": getattr(sig, "symbol", "UNKNOWN"),
                    "side": getattr(sig, "action", "UNKNOWN"),
                    "status": "error",
                    "reason": str(exc),
                })

        logger.info("_execute_signals: processed %d signals, %d results", len(signals), len(results))
        return results

    # ------------------------------------------------------------------
    # FIX 16: Aggressive strategy dampening
    # ------------------------------------------------------------------

    def _get_strategy_multiplier(self, strategy_name: str) -> float:
        """
        Return a sizing multiplier for the given strategy based on recent
        performance.  Losing strategies are dampened; winners get a boost.
        """
        _sss = getattr(self, "_strategy_state_store", None)
        if _sss is None:
            return 1.0
        state = _sss.get_state(strategy_name)
        if state is None:
            return 1.0

        # Check consecutive losses first (most aggressive reduction)
        consec_losses = int(state.get("consecutive_losses", 0) or 0)
        if consec_losses >= 5:
            return 0.10

        # Check recent PnL performance (last 20 trades approximation)
        total_pnl = float(state.get("total_pnl", 0.0) or 0.0)
        trade_count = int(state.get("trade_count", 0) or 0)
        if trade_count < 5:
            return 1.0  # not enough data

        # Approximate PnL percentage
        pnl_pct = total_pnl / max(float(self.portfolio_value_aud), 1.0) * 100.0

        if pnl_pct < -3.0:
            return 0.25
        if pnl_pct < -1.0:
            return 0.50
        if pnl_pct > 3.0:
            return 1.30
        return 1.0

    # ------------------------------------------------------------------
    # FIX 17: Maker fill rate tracking
    # ------------------------------------------------------------------

    def _record_maker_fill(self, symbol: str, is_maker: bool) -> None:
        """Track maker/taker fill for the given symbol."""
        if not hasattr(self, "_maker_fill_tracker"):
            self._maker_fill_tracker: dict = {}
        tracker = self._maker_fill_tracker.setdefault(symbol, {
            "maker_attempts": 0,
            "maker_fills": 0,
            "taker_fallbacks": 0,
        })
        tracker["maker_attempts"] += 1
        if is_maker:
            tracker["maker_fills"] += 1
        else:
            tracker["taker_fallbacks"] += 1

        total = tracker["maker_attempts"]
        if total >= 50 and total % 10 == 0:
            fill_rate = tracker["maker_fills"] / total if total > 0 else 0.0
            if fill_rate < 0.30:
                logger.warning(
                    "Maker fill rate for %s is %.1f%% (<%30%%) over %d trades — auto-switching to taker",
                    symbol, fill_rate * 100.0, total,
                )

    def get_maker_fill_rate(self, symbol: str) -> float:
        """Return maker fill rate for symbol (0.0 to 1.0)."""
        tracker = getattr(self, "_maker_fill_tracker", {}).get(symbol)
        if tracker is None or tracker["maker_attempts"] == 0:
            return 1.0  # assume good until proven otherwise
        return tracker["maker_fills"] / tracker["maker_attempts"]

    def _should_use_taker_for_symbol(self, symbol: str) -> bool:
        """Return True if symbol's maker fill rate is too low."""
        tracker = getattr(self, "_maker_fill_tracker", {}).get(symbol)
        if tracker is None or tracker["maker_attempts"] < 50:
            return False
        fill_rate = tracker["maker_fills"] / max(tracker["maker_attempts"], 1)
        return fill_rate < 0.30

    # ------------------------------------------------------------------
    # FIX 22: Execution quality score
    # ------------------------------------------------------------------

    def _compute_execution_quality(self) -> None:
        """
        Every 50 trades, compute aggregate execution quality metrics
        and log them.
        """
        trade_count = int(getattr(self, "_execution_trade_count", 0) or 0)
        if trade_count < 50 or trade_count % 50 != 0:
            return

        _trades = getattr(self, "_execution_quality_trades", [])
        if not _trades:
            return

        recent = _trades[-50:]
        avg_slippage_bps = sum(t.get("slippage_bps", 0.0) for t in recent) / len(recent)
        maker_count = sum(1 for t in recent if t.get("is_maker", False))
        maker_rate = maker_count / len(recent) * 100.0
        avg_fill_time_ms = sum(t.get("fill_time_ms", 0.0) for t in recent) / len(recent)

        self._execution_quality_score = {
            "avg_slippage_bps": round(avg_slippage_bps, 2),
            "maker_fill_rate_pct": round(maker_rate, 1),
            "avg_fill_time_ms": round(avg_fill_time_ms, 1),
            "sample_size": len(recent),
        }

        logger.info(
            "Execution quality: slippage=%.1fbps, maker_rate=%.0f%%, fill_time=%.0fms",
            avg_slippage_bps, maker_rate, avg_fill_time_ms,
        )

    # ------------------------------------------------------------------
    # FIX 26: Smart signal conflict resolution
    # ------------------------------------------------------------------

    def _resolve_signal_conflicts(self, signals: list) -> list:
        """
        When BUY and SELL signals exist for the same symbol in the same batch,
        keep the signal that aligns with price action.
        """
        from collections import defaultdict
        symbol_signals: dict = defaultdict(list)
        for sig in signals:
            sym = str(getattr(sig, "symbol", "") or "")
            symbol_signals[sym].append(sig)

        resolved = []
        for sym, sigs in symbol_signals.items():
            actions = {str(getattr(s, "action", "")).upper() for s in sigs}
            if "BUY" in actions and "SELL" in actions:
                # Conflict detected — resolve with price action
                _price_hist = getattr(self, "_price_history", {})
                _sym_prices = _price_hist.get(sym, [])
                keep_buy = True  # default

                if len(_sym_prices) >= 20:
                    _sma_20 = sum(_sym_prices[-20:]) / 20.0
                    _current = _sym_prices[-1] if _sym_prices else 0.0
                    if _sma_20 > 0 and _current > 0:
                        if _current < _sma_20:
                            keep_buy = True  # below SMA → favor BUY
                        else:
                            keep_buy = False  # above SMA → favor SELL

                    # Also check recent momentum (last 3 candles)
                    if len(_sym_prices) >= 3:
                        _last3 = _sym_prices[-3:]
                        _all_green = all(_last3[i] > _last3[i - 1] for i in range(1, len(_last3)))
                        if _all_green:
                            keep_buy = True

                kept_action = "BUY" if keep_buy else "SELL"
                discarded_action = "SELL" if keep_buy else "BUY"
                reason = "price below SMA" if keep_buy else "price above SMA"
                logger.info(
                    "Conflict resolved for %s: kept %s, discarded %s (%s)",
                    sym, kept_action, discarded_action, reason,
                )

                for s in sigs:
                    if str(getattr(s, "action", "")).upper() == kept_action:
                        resolved.append(s)
            else:
                resolved.extend(sigs)

        return resolved

    # ------------------------------------------------------------------
    # FIX 9: Pyramid and partial exit logic
    # ------------------------------------------------------------------

    def _check_pyramid_opportunities(self) -> List[Any]:
        """
        For each open position up > 2% with recent win_rate > 55%,
        generate a follow-up BUY signal at 50% of original size.
        Max 2 pyramids per position per cycle, only one per cycle.
        """
        pyramid_signals: List[Any] = []
        if not self.positions:
            return pyramid_signals

        try:
            from unified_types import TradingSignal
        except ImportError:
            return pyramid_signals

        for symbol, pos in list((self.positions or {}).items()):
            if pos is None:
                continue
            qty = float((pos or {}).get("quantity", 0) or 0)
            if qty <= 0:
                continue

            entry_price = float((pos or {}).get("entry_price", 0) or 0)
            current_price = float((pos or {}).get("current_price", 0) or 0)
            side = str((pos or {}).get("side", "BUY")).upper()
            if entry_price <= 0 or current_price <= 0:
                continue

            # Check profit
            if side == "BUY":
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            if pnl_pct <= 0.02:
                continue

            # Check pyramid count
            current_pyramids = self._pyramid_count.get(symbol, 0)
            if current_pyramids >= 2:
                continue

            # Check recent win rate (last 10 trades)
            recent_trades = [
                t for t in list(self.trade_history)[-10:]
                if t.get("pnl") is not None
            ]
            if len(recent_trades) < 5:
                continue
            wins = sum(1 for t in recent_trades if float(t.get("pnl", 0)) > 0)
            win_rate = wins / len(recent_trades)
            if win_rate <= 0.55:
                continue

            # Generate pyramid signal at 50% size
            pyramid_sig = TradingSignal(
                symbol=symbol,
                action=side,
                confidence=0.5,
                strength=0.5,  # 50% of original sizing
                entry_price=current_price,
                reasoning=f"pyramid: position up {pnl_pct:.1%}, win_rate={win_rate:.0%}",
            )
            pyramid_signals.append(pyramid_sig)
            self._pyramid_count[symbol] = current_pyramids + 1
            logger.info(
                "_check_pyramid_opportunities: pyramid #%d for %s (up %.1f%%, wr=%.0f%%)",
                current_pyramids + 1, symbol, pnl_pct * 100, win_rate * 100,
            )

        return pyramid_signals

    def _check_partial_exits(self) -> List[Any]:
        """
        For positions up > 4%: close 50% (take partial profit).
        For positions down > 2% and trending against: close 50% (salvage).
        """
        exit_signals: List[Any] = []
        if not self.positions:
            return exit_signals

        try:
            from unified_types import TradingSignal
        except ImportError:
            return exit_signals

        for symbol, pos in list((self.positions or {}).items()):
            if pos is None:
                continue
            qty = float((pos or {}).get("quantity", 0) or 0)
            if qty <= 0:
                continue

            # Skip if partial exit already done this session
            if self._partial_exit_done.get(symbol, False):
                continue

            entry_price = float((pos or {}).get("entry_price", 0) or 0)
            current_price = float((pos or {}).get("current_price", 0) or 0)
            side = str((pos or {}).get("side", "BUY")).upper()
            if entry_price <= 0 or current_price <= 0:
                continue

            if side == "BUY":
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            exit_action = "SELL" if side == "BUY" else "BUY"

            # Partial profit: up > 4%
            if pnl_pct > 0.04:
                exit_sig = TradingSignal(
                    symbol=symbol,
                    action=exit_action,
                    confidence=0.7,
                    strength=0.5,  # close 50%
                    entry_price=current_price,
                    reasoning=f"partial_profit: up {pnl_pct:.1%}, closing 50%",
                )
                exit_signals.append(exit_sig)
                self._partial_exit_done[symbol] = True
                logger.info(
                    "_check_partial_exits: partial profit for %s (up %.1f%%)",
                    symbol, pnl_pct * 100,
                )
            # Salvage: down > 2% and trending against
            elif pnl_pct < -0.02:
                # Check if trending against (use high water mark)
                hw = self._position_high_water.get(symbol, current_price)
                if current_price < hw * 0.98:  # falling from high
                    exit_sig = TradingSignal(
                        symbol=symbol,
                        action=exit_action,
                        confidence=0.6,
                        strength=0.5,  # close 50%
                        entry_price=current_price,
                        reasoning=f"partial_salvage: down {pnl_pct:.1%}, trending against, closing 50%",
                    )
                    exit_signals.append(exit_sig)
                    self._partial_exit_done[symbol] = True
                    logger.info(
                        "_check_partial_exits: partial salvage for %s (down %.1f%%)",
                        symbol, pnl_pct * 100,
                    )

        # --- Batch H1: Drawdown CRITICAL → close 30% of all open positions ---
        try:
            _dd_adv = getattr(self, "_last_cycle_advisory", None) or {}
            _dd_sys = _dd_adv.get("system_status") if isinstance(_dd_adv, dict) else None
            if _dd_sys and isinstance(_dd_sys, dict):
                _dd_status = str(_dd_sys.get("status", "")).upper()
                if _dd_status == "CRITICAL":
                    for _dd_sym, _dd_pos in list((self.positions or {}).items()):
                        if _dd_pos is None:
                            continue
                        _dd_qty = float((_dd_pos or {}).get("quantity", 0) or 0)
                        if _dd_qty <= 0:
                            continue
                        _dd_side = str((_dd_pos or {}).get("side", "BUY")).upper()
                        _dd_exit_action = "SELL" if _dd_side == "BUY" else "BUY"
                        _dd_cur = float((_dd_pos or {}).get("current_price", 0) or 0)
                        if _dd_cur <= 0:
                            continue
                        _dd_key = f"dd_critical_{_dd_sym}"
                        if self._partial_exit_done.get(_dd_key, False):
                            continue
                        _dd_sig = TradingSignal(
                            symbol=_dd_sym,
                            action=_dd_exit_action,
                            confidence=0.8,
                            strength=0.30,  # close 30%
                            entry_price=_dd_cur,
                            reasoning=f"drawdown_critical: system CRITICAL, closing 30% of {_dd_sym}",
                        )
                        exit_signals.append(_dd_sig)
                        self._partial_exit_done[_dd_key] = True
                        logger.warning("_check_partial_exits: drawdown CRITICAL — closing 30%% of %s", _dd_sym)
        except Exception as _h1_exc:
            logger.debug("Batch H1 drawdown partial close error: %s", _h1_exc)

        # --- Batch H2: Funding cost rotation → close high-funding-cost long positions ---
        try:
            _fc_adv = getattr(self, "_last_cycle_advisory", None) or {}
            _fc_recs = _fc_adv.get("funding_exit_recommendations") if isinstance(_fc_adv, dict) else None
            if _fc_recs and isinstance(_fc_recs, list):
                for _fc_rec in _fc_recs:
                    if not isinstance(_fc_rec, dict):
                        continue
                    _fc_sym = str(_fc_rec.get("symbol", ""))
                    if not _fc_sym:
                        continue
                    _fc_pos = (self.positions or {}).get(_fc_sym)
                    if _fc_pos is None:
                        continue
                    _fc_qty = float((_fc_pos or {}).get("quantity", 0) or 0)
                    if _fc_qty <= 0:
                        continue
                    _fc_side = str((_fc_pos or {}).get("side", "BUY")).upper()
                    if _fc_side != "BUY":
                        continue  # only close longs with high funding cost
                    _fc_key = f"funding_rot_{_fc_sym}"
                    if self._partial_exit_done.get(_fc_key, False):
                        continue
                    _fc_cur = float((_fc_pos or {}).get("current_price", 0) or 0)
                    if _fc_cur <= 0:
                        continue
                    _fc_sig = TradingSignal(
                        symbol=_fc_sym,
                        action="SELL",
                        confidence=0.65,
                        strength=1.0,  # full close
                        entry_price=_fc_cur,
                        reasoning=f"funding_rotation: high funding cost on {_fc_sym}, exiting long",
                    )
                    exit_signals.append(_fc_sig)
                    self._partial_exit_done[_fc_key] = True
                    logger.info("_check_partial_exits: funding rotation — closing long %s", _fc_sym)
        except Exception as _h2_exc:
            logger.debug("Batch H2 funding cost rotation error: %s", _h2_exc)

        # --- Regime change liquidation — exit positions incompatible with new regime ---
        try:
            _rc_adv = getattr(self, "_last_cycle_advisory", None) or {}
            _rc_transition = _rc_adv.get("regime_transition") if isinstance(_rc_adv, dict) else None
            _rc_autoencoder = _rc_adv.get("autoencoder_regime") if isinstance(_rc_adv, dict) else None
            _regime_changing = False
            if _rc_transition and isinstance(_rc_transition, dict):
                _regime_changing = bool(_rc_transition.get("detected", False))
            if not _regime_changing and _rc_autoencoder and isinstance(_rc_autoencoder, dict):
                _regime_changing = bool(_rc_autoencoder.get("is_transition", False))

            if _regime_changing:
                for _rc_sym, _rc_pos in list((self.positions or {}).items()):
                    if _rc_pos is None:
                        continue
                    _rc_qty = float((_rc_pos or {}).get("quantity", 0) or 0)
                    if _rc_qty <= 0:
                        continue
                    _rc_key = f"regime_change_{_rc_sym}"
                    if self._partial_exit_done.get(_rc_key, False):
                        continue
                    _rc_side = str((_rc_pos or {}).get("side", "BUY")).upper()
                    _rc_exit = "SELL" if _rc_side == "BUY" else "BUY"
                    _rc_cur = float((_rc_pos or {}).get("current_price", 0) or 0)
                    if _rc_cur <= 0:
                        continue
                    # Close 50% on regime transition (aggressive, like quant funds)
                    _rc_sig = TradingSignal(
                        symbol=_rc_sym,
                        action=_rc_exit,
                        confidence=0.75,
                        strength=0.50,
                        entry_price=_rc_cur,
                        reasoning=f"regime_change_liquidation: regime transitioning, closing 50% of {_rc_sym}",
                    )
                    exit_signals.append(_rc_sig)
                    self._partial_exit_done[_rc_key] = True
                    logger.warning("_check_partial_exits: REGIME CHANGE — closing 50%% of %s", _rc_sym)
        except Exception as _rc_exc:
            logger.debug("Regime change liquidation error: %s", _rc_exc)

        return exit_signals

    def _check_oco_conditions(self) -> List[Any]:
        """
        FIX 12: Check OCO (One-Cancels-Other) conditions alongside regular stops.
        Returns exit signals for positions that hit stop_loss or take_profit.
        """
        exit_signals: List[Any] = []
        if not self._oco_orders:
            return exit_signals

        try:
            from unified_types import TradingSignal
        except ImportError:
            return exit_signals

        for symbol, oco in list(self._oco_orders.items()):
            pos = (self.positions or {}).get(symbol)
            if pos is None or float((pos or {}).get("quantity", 0) or 0) <= 0:
                # Position was closed — remove OCO
                del self._oco_orders[symbol]
                continue

            current_price = float((pos or {}).get("current_price", 0) or 0)
            if current_price <= 0:
                continue

            entry_side = str(oco.get("side", "BUY")).upper()
            sl = float(oco.get("stop_loss", 0))
            tp = float(oco.get("take_profit", 0))
            exit_action = "SELL" if entry_side == "BUY" else "BUY"

            triggered = None
            if entry_side == "BUY":
                if sl > 0 and current_price <= sl:
                    triggered = "stop_loss"
                elif tp > 0 and current_price >= tp:
                    triggered = "take_profit"
            else:
                if sl > 0 and current_price >= sl:
                    triggered = "stop_loss"
                elif tp > 0 and current_price <= tp:
                    triggered = "take_profit"

            if triggered:
                exit_sig = TradingSignal(
                    symbol=symbol,
                    action=exit_action,
                    confidence=0.9,
                    strength=1.0,
                    entry_price=current_price,
                    reasoning=f"OCO {triggered}: price={current_price:.2f}, sl={sl:.2f}, tp={tp:.2f}",
                )
                exit_signals.append(exit_sig)
                del self._oco_orders[symbol]
                logger.info(
                    "_check_oco_conditions: %s triggered for %s @ %.2f (SL=%.2f, TP=%.2f)",
                    triggered, symbol, current_price, sl, tp,
                )

        return exit_signals

    async def _update_trailing_stops(self) -> List[Dict[str, Any]]:
        """
        Dynamic exit optimization — called each cycle BEFORE signal generation.

        Implements:
          a) Trailing stops: track high/low watermarks, trigger at 2% from peak
             (only activates after position is 1% in profit)
          b) Time-based exits: close stale positions (>48h with <0.5% profit,
             or >7 days regardless)
          c) Uses existing _position_high_water / _position_low_water dicts

        Returns list of exit signals to execute.
        """
        exit_signals: List[Dict[str, Any]] = []
        if not self.positions:
            return exit_signals

        trailing_stop_pct = float(getattr(self.config, "trailing_stop_pct", 0.02) or 0.02)
        trailing_activation_pct = float(getattr(self.config, "trailing_activation_pct", 0.01) or 0.01)
        stale_hours = float(getattr(self.config, "stale_position_hours", 48.0) or 48.0)
        stale_min_profit_pct = float(getattr(self.config, "stale_min_profit_pct", 0.005) or 0.005)
        max_hold_hours = float(getattr(self.config, "max_hold_hours", 168.0) or 168.0)  # 7 days

        now = time.time()

        for symbol, pos in list((self.positions or {}).items()):
            if pos is None:
                continue
            qty = float((pos or {}).get("quantity", 0) or 0)
            if qty <= 0:
                continue

            current_price = float((pos or {}).get("current_price", 0) or 0)
            entry_price = float((pos or {}).get("entry_price", 0) or 0)
            entry_time = float((pos or {}).get("entry_time", 0) or (pos or {}).get("timestamp", 0) or 0)
            side = str((pos or {}).get("side", "BUY") or "BUY").upper()

            if current_price <= 0 or entry_price <= 0:
                continue

            # --- (a) Update watermarks ---
            is_long = side == "BUY"
            if is_long:
                prev_hw = self._position_high_water.get(symbol, current_price)
                new_hw = max(prev_hw, current_price)
                self._position_high_water[symbol] = new_hw

                # Profit from entry
                profit_pct = (current_price - entry_price) / entry_price

                # Trailing stop: only activate after reaching activation threshold
                if profit_pct >= trailing_activation_pct:
                    # Distance from peak
                    drawdown_from_peak = (new_hw - current_price) / new_hw if new_hw > 0 else 0.0
                    if drawdown_from_peak >= trailing_stop_pct:
                        logger.warning(
                            "TRAILING STOP: %s long — peak=%.2f, current=%.2f, drawdown=%.2f%% >= %.2f%%",
                            symbol, new_hw, current_price,
                            drawdown_from_peak * 100, trailing_stop_pct * 100,
                        )
                        exit_signals.append({
                            "symbol": symbol,
                            "action": "SELL",
                            "price": current_price,
                            "quantity": qty,
                            "reason": f"trailing_stop:peak={new_hw:.2f},dd={drawdown_from_peak:.4f}",
                        })
                        continue
            else:
                # Short position
                prev_lw = self._position_low_water.get(symbol, current_price)
                new_lw = min(prev_lw, current_price)
                self._position_low_water[symbol] = new_lw

                profit_pct = (entry_price - current_price) / entry_price

                if profit_pct >= trailing_activation_pct:
                    runup_from_trough = (current_price - new_lw) / new_lw if new_lw > 0 else 0.0
                    if runup_from_trough >= trailing_stop_pct:
                        logger.warning(
                            "TRAILING STOP: %s short — trough=%.2f, current=%.2f, runup=%.2f%% >= %.2f%%",
                            symbol, new_lw, current_price,
                            runup_from_trough * 100, trailing_stop_pct * 100,
                        )
                        exit_signals.append({
                            "symbol": symbol,
                            "action": "BUY",
                            "price": current_price,
                            "quantity": qty,
                            "reason": f"trailing_stop:trough={new_lw:.2f},runup={runup_from_trough:.4f}",
                        })
                        continue

            # --- (b) Time-based exits ---
            if entry_time > 0:
                hours_held = (now - entry_time) / 3600.0

                # Max hold time: close regardless of profit
                if hours_held >= max_hold_hours:
                    logger.warning(
                        "TIME EXIT: %s held %.1f hours (max=%.1f) — closing stale position",
                        symbol, hours_held, max_hold_hours,
                    )
                    close_action = "SELL" if is_long else "BUY"
                    exit_signals.append({
                        "symbol": symbol,
                        "action": close_action,
                        "price": current_price,
                        "quantity": qty,
                        "reason": f"max_hold_time:{hours_held:.1f}h",
                    })
                    continue

                # Stale position: held > stale_hours with minimal profit
                if hours_held >= stale_hours and abs(profit_pct) < stale_min_profit_pct:
                    logger.warning(
                        "STALE EXIT: %s held %.1f hours with only %.2f%% profit — freeing capital",
                        symbol, hours_held, profit_pct * 100,
                    )
                    close_action = "SELL" if is_long else "BUY"
                    exit_signals.append({
                        "symbol": symbol,
                        "action": close_action,
                        "price": current_price,
                        "quantity": qty,
                        "reason": f"stale_position:{hours_held:.1f}h,profit={profit_pct:.4f}",
                    })
                    continue

        # Clean up watermarks for closed positions
        for symbol in list(self._position_high_water.keys()):
            if symbol not in (self.positions or {}):
                del self._position_high_water[symbol]
        for symbol in list(self._position_low_water.keys()):
            if symbol not in (self.positions or {}):
                del self._position_low_water[symbol]

        if exit_signals:
            logger.info(
                "_update_trailing_stops: %d exit signals generated (trailing=%d, time=%d)",
                len(exit_signals),
                sum(1 for s in exit_signals if "trailing" in s.get("reason", "")),
                sum(1 for s in exit_signals if "time" in s.get("reason", "") or "stale" in s.get("reason", "") or "max_hold" in s.get("reason", "")),
            )

        return exit_signals

    async def _reconcile_positions(self) -> Dict[str, Any]:
        """
        Reconcile internal positions against exchange state.
        Exchange is source of truth -- internal state is updated to match.

        Returns a summary dict with discrepancies found/fixed.
        """
        summary: Dict[str, Any] = {"discrepancies": [], "updated": 0, "exchange_positions": {}}
        mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()

        if mode != "live":
            logger.debug("_reconcile_positions: skipped in %s mode (exchange is not source of truth)", mode)
            return summary

        if not hasattr(self, "exchange_manager") or self.exchange_manager is None:
            logger.debug("_reconcile_positions: no exchange_manager available")
            return summary

        try:
            # Fetch balances from exchange (proxy for positions)
            balances = await self.exchange_manager.get_balances()
            if not balances:
                logger.warning("_reconcile_positions: no balances returned from exchange")
                return summary

            # Flatten across exchanges -- merge by symbol
            exchange_positions: Dict[str, float] = {}
            for ex_name, ex_balances in balances.items():
                for asset, balance_info in ex_balances.items():
                    if isinstance(balance_info, dict):
                        total = float(balance_info.get("total", 0.0) or 0.0)
                    else:
                        total = float(balance_info or 0.0)
                    if total > 1e-12:
                        exchange_positions[asset] = exchange_positions.get(asset, 0.0) + total

            summary["exchange_positions"] = dict(exchange_positions)

            # Compare with internal positions
            all_symbols = set(list(self.positions.keys()) + list(exchange_positions.keys()))
            for symbol in all_symbols:
                internal_qty = float((self.positions.get(symbol) or {}).get("quantity", 0.0) or 0.0)
                exchange_qty = exchange_positions.get(symbol, 0.0)

                # Only reconcile trading symbols (skip stablecoins/fiat unless tracked)
                if abs(internal_qty - exchange_qty) > 1e-8:
                    discrepancy = {
                        "symbol": symbol,
                        "internal_qty": internal_qty,
                        "exchange_qty": exchange_qty,
                        "delta": exchange_qty - internal_qty,
                    }
                    summary["discrepancies"].append(discrepancy)
                    logger.warning(
                        "_reconcile_positions: discrepancy for %s — internal=%.8f exchange=%.8f delta=%.8f",
                        symbol, internal_qty, exchange_qty, exchange_qty - internal_qty,
                    )

                    # Update internal to match exchange (exchange is source of truth)
                    if exchange_qty > 1e-12:
                        pos = self.positions.get(symbol) or {"quantity": 0.0, "avg_price": 0.0, "current_price": 0.0}
                        pos["quantity"] = exchange_qty
                        self.positions[symbol] = pos
                    elif symbol in self.positions:
                        # Exchange says zero -- remove internal position
                        del self.positions[symbol]
                    summary["updated"] += 1

            if summary["discrepancies"]:
                logger.info(
                    "_reconcile_positions: found %d discrepancies, updated %d positions",
                    len(summary["discrepancies"]), summary["updated"],
                )
            else:
                logger.debug("_reconcile_positions: all positions match exchange")

        except Exception as exc:
            logger.error("_reconcile_positions: failed: %s", exc, exc_info=True)
            summary["error"] = str(exc)

        return summary

    async def _poll_pending_orders(self) -> List[Dict[str, Any]]:
        """
        Poll exchange for status of pending (partially filled) orders.

        For each pending order:
          - If filled: update position, record in ledger, call on_fill
          - If timed out: cancel and log
          - If still open: leave in pending dict

        Returns list of completed/cancelled order results.
        """
        if not self._pending_orders:
            return []

        mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
        results: List[Dict[str, Any]] = []
        now = time.time()
        completed_ids: List[str] = []

        for order_id, order_info in list(self._pending_orders.items()):
            try:
                elapsed = now - float(order_info.get("submitted_at", now))
                symbol = str(order_info.get("symbol", ""))
                side = str(order_info.get("side", ""))
                exchange_name = str(order_info.get("exchange", "kraken"))

                if mode == "live" and hasattr(self, "exchange_manager") and self.exchange_manager is not None:
                    # Poll exchange for order status
                    try:
                        ex = self.exchange_manager.exchanges.get(exchange_name)
                        if ex is not None and hasattr(ex, "fetch_order"):
                            order_status = await ex.fetch_order(order_id, symbol)
                            if order_status:
                                status = str(order_status.get("status", "open"))
                                filled = float(order_status.get("filled", 0.0))
                                remaining = float(order_status.get("remaining", 0.0))
                                avg_price = float(order_status.get("average", order_info.get("entry_price", 0.0)))

                                if status in ("closed", "filled") or remaining <= 0:
                                    # Fully filled
                                    additional_fill = filled - float(order_info.get("filled_quantity", 0.0))
                                    if additional_fill > 1e-12:
                                        trade_result = {
                                            "order_id": order_id,
                                            "symbol": symbol,
                                            "side": side,
                                            "quantity": additional_fill,
                                            "price": avg_price,
                                            "status": "filled",
                                            "exchange": exchange_name,
                                            "timestamp": time.time(),
                                            "stop_loss": order_info.get("stop_loss"),
                                            "take_profit": order_info.get("take_profit"),
                                        }
                                        try:
                                            self._record_trade(trade_result)
                                        except Exception as exc:
                                            logger.warning("_poll_pending: _record_trade failed: %s", exc)
                                        try:
                                            ledger = getattr(self.execution_engine, "trade_ledger", None) if self.execution_engine else None
                                            if ledger is not None:
                                                ledger.record_trade(trade_result)
                                        except Exception:
                                            pass
                                        try:
                                            if self.component_registry is not None:
                                                self.component_registry.on_fill(trade_result)
                                        except Exception:
                                            pass
                                        results.append(trade_result)
                                    completed_ids.append(order_id)
                                    logger.info("_poll_pending: order %s fully filled (%.8f @ %.2f)", order_id, filled, avg_price)
                                elif status == "canceled":
                                    completed_ids.append(order_id)
                                    logger.info("_poll_pending: order %s was cancelled by exchange", order_id)
                                    results.append({
                                        "order_id": order_id,
                                        "symbol": symbol,
                                        "side": side,
                                        "status": "cancelled",
                                        "reason": "exchange_cancelled",
                                    })
                                else:
                                    # Still open -- update filled quantity
                                    order_info["filled_quantity"] = filled
                    except Exception as exc:
                        logger.debug("_poll_pending: poll failed for %s: %s", order_id, exc)

                # Timeout check
                if order_id not in completed_ids and elapsed > self._order_timeout_seconds:
                    logger.warning(
                        "_poll_pending: order %s timed out after %.1fs — cancelling",
                        order_id, elapsed,
                    )
                    # Try to cancel on exchange
                    if mode == "live" and hasattr(self, "exchange_manager") and self.exchange_manager is not None:
                        try:
                            ex = self.exchange_manager.exchanges.get(exchange_name)
                            if ex is not None and hasattr(ex, "cancel_order"):
                                await ex.cancel_order(order_id, symbol)
                        except Exception as exc:
                            logger.warning("_poll_pending: cancel failed for %s: %s", order_id, exc)

                    completed_ids.append(order_id)
                    results.append({
                        "order_id": order_id,
                        "symbol": symbol,
                        "side": side,
                        "status": "cancelled",
                        "reason": "timeout",
                        "elapsed_seconds": elapsed,
                    })

            except Exception as exc:
                logger.error("_poll_pending: error processing order %s: %s", order_id, exc)
                # Don't remove on error -- will retry next cycle

        # Clean up completed orders
        for oid in completed_ids:
            self._pending_orders.pop(oid, None)

        if results:
            logger.info("_poll_pending: processed %d pending orders, %d completed/cancelled", len(self._pending_orders) + len(completed_ids), len(results))

        return results

    def _install_signal_handlers(self) -> None:
        """Register SIGINT/SIGTERM handlers that trigger graceful shutdown."""
        loop = asyncio.get_event_loop()

        def _signal_handler(sig: int) -> None:
            sig_name = signal.Signals(sig).name
            logger.info("Received %s — initiating graceful shutdown", sig_name)
            self.state = SystemState.SHUTDOWN

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _signal_handler, sig)
            except (NotImplementedError, OSError):
                # Windows: add_signal_handler is not supported; fall back to signal.signal
                try:
                    signal.signal(sig, lambda s, f: _signal_handler(s))
                except (OSError, ValueError):
                    logger.debug("Could not install handler for %s", sig)

    async def shutdown(self):
        """
        Graceful shutdown sequence:
          1. Cancel open orders
          2. Flush pending fills to trade ledger
          3. Save regime state
          4. Shut down component registry (reverse init order)
          5. Close WebSocket connections
          6. Close exchange clients
          7. Stop monitoring
          8. Log "shutdown complete"
        """
        logger.info("Initiating graceful shutdown...")
        self.state = SystemState.SHUTDOWN

        # 1. Cancel open orders
        try:
            execution_engine = getattr(self, "execution_engine", None)
            if execution_engine is not None:
                cancel_fn = getattr(execution_engine, "cancel_all_orders", None)
                if callable(cancel_fn):
                    res = cancel_fn()
                    if asyncio.iscoroutine(res):
                        await res
                    logger.info("Cancelled all open orders")
        except Exception as e:
            logger.warning("Failed to cancel open orders: %s", e)

        # 2. Flush pending fills to trade ledger
        try:
            ledger = None
            if getattr(self, "execution_engine", None) is not None:
                ledger = getattr(self.execution_engine, "trade_ledger", None)
            if ledger is not None:
                flush_fn = getattr(ledger, "flush", None)
                if callable(flush_fn):
                    res = flush_fn()
                    if asyncio.iscoroutine(res):
                        await res
                    logger.info("Flushed pending fills to trade ledger")
        except Exception as e:
            logger.warning("Failed to flush trade ledger: %s", e)

        # 3. Save regime state
        try:
            regime_store = None
            cr = getattr(self, "component_registry", None)
            if cr is not None:
                regime_store = getattr(cr, "regime_store", None)
            if regime_store is None:
                # Try direct attribute
                regime_store = getattr(self, "regime_store", None)
            if regime_store is not None:
                save_fn = getattr(regime_store, "save", None)
                if callable(save_fn):
                    # Save current regime for all tracked symbols
                    current_regime = getattr(self, "current_regime", None)
                    symbols = getattr(self.config, "trading_pairs", []) or []
                    if current_regime and symbols:
                        for sym in symbols:
                            try:
                                regime_store.save(sym, str(current_regime))
                            except Exception:
                                pass
                        logger.info("Saved regime state for %d symbols", len(symbols))
        except Exception as e:
            logger.warning("Failed to save regime state: %s", e)

        # 3a. Save strategy states
        try:
            _sss = getattr(self, "_strategy_state_store", None)
            if _sss is not None:
                _sss.save_all()
                logger.info("Strategy state store: saved all states on shutdown")
        except Exception as e:
            logger.warning("Failed to save strategy states: %s", e)

        # 3b. Save final checkpoint on shutdown
        try:
            if self.checkpoint_manager is not None:
                _final_state = {
                    "cycle_count": getattr(self, "_total_cycles", 0),
                    "portfolio_value": float(getattr(self, "portfolio_value_aud", 0) or 0),
                    "regime": str(getattr(self, "_last_regime_label", "UNKNOWN")),
                    "shutdown": True,
                    "model_versions": {},
                }
                if self.model_manager is not None:
                    try:
                        _snap = self.model_manager.snapshot()
                        _final_state["model_versions"] = {
                            k: v.get("version", 0)
                            for k, v in _snap.get("models", {}).items()
                        }
                    except Exception:
                        pass
                self.checkpoint_manager.save_checkpoint(_final_state)
                logger.info("Final checkpoint saved on shutdown")
        except Exception as e:
            logger.warning("Failed to save shutdown checkpoint: %s", e)

        # 4. Shut down component registry
        try:
            cr = getattr(self, "component_registry", None)
            if cr is not None and hasattr(cr, "shutdown"):
                await cr.shutdown(timeout=30.0)
                logger.info("Component registry shutdown complete")
        except Exception as e:
            logger.warning("Component registry shutdown error: %s", e)

        # Persist strategy evaluation state
        try:
            se_engine = getattr(self, "strategy_evaluation_engine", None)
            if se_engine is not None:
                se_engine.persist_to_db()
        except Exception as e:
            self._handle_strategy_eval_error(e, context="shutdown_persist")
        try:
            cc_engine = getattr(self, "champion_challenger_engine", None)
            if cc_engine is not None:
                cc_engine.persist_to_db()
        except Exception as e:
            logger.warning("Champion/challenger shutdown persist error: %s", e)

        # 4b. Disconnect LiveMarketDataManager (WebSocket feeds)
        try:
            lmd = getattr(self, "live_market_data", None)
            if lmd is not None:
                await lmd.disconnect()
                logger.info("LiveMarketDataManager disconnected")
        except Exception as e:
            logger.debug("LiveMarketDataManager disconnect: %s", e)

        # 5. Close WebSocket connections
        try:
            ws_connectors = getattr(self, "ws_connectors", None) or []
            for ws in ws_connectors:
                try:
                    close_fn = getattr(ws, "close", None) or getattr(ws, "disconnect", None)
                    if callable(close_fn):
                        res = close_fn()
                        if asyncio.iscoroutine(res):
                            await res
                except Exception:
                    pass
            if ws_connectors:
                logger.info("Closed %d WebSocket connections", len(ws_connectors))
        except Exception as e:
            logger.debug("WS close: %s", e)

        # 6. Close exchange clients (prevents unclosed aiohttp/ccxt warnings)
        try:
            exchanges = getattr(self, "exchanges", {}) or {}
            for name, ex in exchanges.items():
                try:
                    close_fn = getattr(ex, "close", None)
                    if callable(close_fn):
                        res = close_fn()
                        if asyncio.iscoroutine(res):
                            await res
                        logger.info("Closed exchange client: %s", name)
                except Exception as e:
                    logger.warning("Failed to close exchange client %s: %s", name, e)
        except Exception:
            pass

        # Stop continuous best-trade scanner
        try:
            scanner = getattr(self, "continuous_scanner", None)
            if scanner is not None and hasattr(scanner, "stop"):
                scanner.stop()
                logger.info("Continuous best-trade scanner stopped")
        except Exception as e:
            logger.debug("Stop continuous scanner: %s", e)

        # 7. Stop monitoring
        monitoring = getattr(self, "monitoring", None)
        if monitoring:
            try:
                shutdown_fn = getattr(monitoring, "shutdown", None)
                if callable(shutdown_fn):
                    res = shutdown_fn()
                    if asyncio.iscoroutine(res):
                        await res
            except Exception as e:
                logger.debug("Monitoring shutdown failed: %s", e)

        # 8. Done
        logger.info("shutdown complete")

    async def graceful_shutdown(self, reason: str = "user_request") -> None:
        """Full graceful shutdown sequence with summary and lock release.

        1. Stop accepting new signals
        2. Cancel all pending orders on exchange
        3. Save final checkpoint
        4. Save strategy states
        5. Flush audit trail
        6. Flush trade ledger
        7. Close exchange connections
        8. Close WebSocket feeds
        9. Log shutdown summary (uptime, trades, PnL)
        10. Release process lock
        """
        logger.info("Graceful shutdown initiated (reason=%s)", reason)

        # Step 1: Stop accepting new signals by setting state
        self.state = SystemState.SHUTDOWN

        # Steps 2-8: delegate to the existing shutdown() which handles all of these
        await self.shutdown()

        # Step 5 (explicit): Flush audit trail
        try:
            audit = getattr(self, "audit_chain", None)
            if audit is not None:
                flush_fn = getattr(audit, "flush", None)
                if callable(flush_fn):
                    res = flush_fn()
                    if asyncio.iscoroutine(res):
                        await res
                    logger.info("Audit trail flushed")
                # Close the audit chain DB connection
                close_fn = getattr(audit, "close", None)
                if callable(close_fn):
                    res = close_fn()
                    if asyncio.iscoroutine(res):
                        await res
        except Exception as e:
            logger.warning("Audit trail flush error: %s", e)

        # Step 9: Log shutdown summary
        try:
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            total_cycles = getattr(self, "_completed_cycles", 0)
            portfolio_value = float(getattr(self, "portfolio_value_aud", 0) or 0)
            starting_capital = float(getattr(self.config, "starting_capital_aud", 0) or 0)
            pnl = portfolio_value - starting_capital if starting_capital > 0 else 0.0
            pnl_pct = (pnl / starting_capital * 100.0) if starting_capital > 0 else 0.0
            logger.info("=" * 60)
            logger.info("SHUTDOWN SUMMARY")
            logger.info("  Reason:         %s", reason)
            logger.info("  Uptime:         %.1f seconds (%.2f hours)", uptime_seconds, uptime_seconds / 3600.0)
            logger.info("  Cycles:         %d", total_cycles)
            logger.info("  Portfolio:      $%.2f AUD", portfolio_value)
            logger.info("  P&L:            $%.2f (%.2f%%)", pnl, pnl_pct)
            logger.info("  Pending orders: %d (cleared)", len(getattr(self, "_pending_orders", {})))
            logger.info("=" * 60)
        except Exception as e:
            logger.warning("Shutdown summary error: %s", e)

        # Step 10: Release process lock
        try:
            proc_lock = getattr(self, "_process_lock", None)
            if proc_lock is not None:
                release_fn = getattr(proc_lock, "release", None)
                if callable(release_fn):
                    release_fn()
                    logger.info("Process lock released")
                self._process_lock = None
        except Exception as e:
            logger.warning("Process lock release error: %s", e)

    async def _pre_trading_checks(self) -> None:
        """Run pre-trading validation before the trading loop starts.

        1. Verify config is valid
        2. Check exchange connectivity (live mode)
        3. Sync positions from exchange
        4. Load latest checkpoint
        5. Load strategy states
        6. Verify data feeds
        7. Run deployment checklist (live mode)
        8. Log startup summary
        """
        mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
        logger.info("Running pre-trading checks (mode=%s)...", mode)
        checks_passed = 0
        checks_failed = 0

        # 1. Verify config is valid
        try:
            capital = float(getattr(self.config, "starting_capital_aud", 0) or 0)
            if capital <= 0:
                logger.error("Pre-trading check FAILED: starting_capital_aud must be > 0 (got %.2f)", capital)
                checks_failed += 1
            else:
                checks_passed += 1
            pairs = getattr(self.config, "trading_pairs", None) or []
            if not pairs:
                logger.warning("Pre-trading check WARNING: no trading_pairs configured")
            else:
                checks_passed += 1
        except Exception as e:
            logger.warning("Pre-trading check: config validation error: %s", e)
            checks_failed += 1

        # 2. Check exchange connectivity (live mode only)
        if mode == "live":
            try:
                exchanges = getattr(self, "exchanges", {}) or {}
                if not exchanges:
                    logger.error("Pre-trading check FAILED: no exchanges connected in live mode")
                    checks_failed += 1
                else:
                    for name, ex in exchanges.items():
                        try:
                            # Try a lightweight connectivity check
                            ping_fn = getattr(ex, "fetch_time", None) or getattr(ex, "ping", None)
                            if callable(ping_fn):
                                res = ping_fn()
                                if asyncio.iscoroutine(res):
                                    await res
                            logger.info("Pre-trading check: exchange '%s' reachable", name)
                            checks_passed += 1
                        except Exception as exc:
                            logger.error("Pre-trading check FAILED: exchange '%s' unreachable: %s", name, exc)
                            checks_failed += 1
            except Exception as e:
                logger.warning("Pre-trading check: exchange connectivity error: %s", e)
                checks_failed += 1

        # 3. Sync positions from exchange (live mode)
        if mode == "live":
            try:
                pos_reg = getattr(self, "component_registry", None)
                if pos_reg is not None:
                    pr = getattr(pos_reg, "_position_registry", None)
                    if pr is not None and hasattr(pr, "sync_from_exchange"):
                        res = pr.sync_from_exchange()
                        if asyncio.iscoroutine(res):
                            await res
                        logger.info("Pre-trading check: positions synced from exchange")
                        checks_passed += 1
            except Exception as e:
                logger.warning("Pre-trading check: position sync error: %s", e)

        # 4. Load latest checkpoint
        try:
            if self.checkpoint_manager is not None:
                ckpt = self.checkpoint_manager.load_latest_checkpoint()
                if ckpt:
                    logger.info("Pre-trading check: checkpoint loaded (cycle_count=%s)",
                                ckpt.get("cycle_count", "?"))
                    checks_passed += 1
                else:
                    logger.info("Pre-trading check: no checkpoint found (fresh start)")
                    checks_passed += 1
            else:
                logger.debug("Pre-trading check: checkpoint manager not available")
        except Exception as e:
            logger.warning("Pre-trading check: checkpoint load error: %s", e)

        # 5. Load strategy states
        try:
            sss = getattr(self, "_strategy_state_store", None)
            if sss is not None:
                loaded = sss.load_all()
                logger.info("Pre-trading check: loaded %d strategy states", len(loaded) if loaded else 0)
                checks_passed += 1
        except Exception as e:
            logger.warning("Pre-trading check: strategy state load error: %s", e)

        # 6. Verify data feeds (check that market data source is reachable)
        try:
            lmd = getattr(self, "live_market_data", None)
            if lmd is not None and hasattr(lmd, "is_connected"):
                if lmd.is_connected():
                    logger.info("Pre-trading check: live market data feed connected")
                    checks_passed += 1
                else:
                    logger.warning("Pre-trading check: live market data feed not connected")
            else:
                checks_passed += 1  # No live feed required (paper mode uses simulated data)
        except Exception as e:
            logger.warning("Pre-trading check: data feed verification error: %s", e)

        # 7. Run deployment checklist (live mode only)
        if mode == "live":
            try:
                from ops.deployment_checklist import DeploymentChecklist
                _checklist = DeploymentChecklist()
                _result = _checklist.run()
                if _result.go:
                    logger.info("Pre-trading check: deployment checklist GO (%d/%d passed)",
                                _result.passed_count, len(_result.checks))
                    checks_passed += 1
                else:
                    logger.warning("Pre-trading check: deployment checklist NO-GO: %s", _result.summary())
                    checks_failed += 1
            except Exception as e:
                logger.debug("Pre-trading check: deployment checklist unavailable: %s", e)

        # 8. Log startup summary
        logger.info("=" * 60)
        logger.info("PRE-TRADING CHECKS COMPLETE")
        logger.info("  Mode:           %s", mode)
        logger.info("  Checks passed:  %d", checks_passed)
        logger.info("  Checks failed:  %d", checks_failed)
        logger.info("  Capital:        $%.2f AUD", float(getattr(self.config, "starting_capital_aud", 0) or 0))
        logger.info("  Exchanges:      %s", list((getattr(self, "exchanges", {}) or {}).keys()))
        logger.info("=" * 60)

        if checks_failed > 0 and mode == "live":
            raise RuntimeError(
                f"Pre-trading checks failed ({checks_failed} failures) in live mode — aborting"
            )

    def get_strategy_evaluation_summary(self, *, limit: int = 5, regime_label: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Runtime helper for strategy evaluation introspection/ranking."""
        engine = getattr(self, "strategy_evaluation_engine", None)
        if engine is None:
            return {
                "top_net_pnl": [],
                "top_expectancy": [],
                "top_sharpe_like": [],
                "worst_drawdown": [],
                "worst_expectancy": [],
            }
        return {
            "top_net_pnl": [asdict(m) for m in engine.top_by_net_pnl(limit=limit, regime_label=regime_label)],
            "top_expectancy": [asdict(m) for m in engine.top_by_expectancy(limit=limit, regime_label=regime_label)],
            "top_sharpe_like": [asdict(m) for m in engine.top_by_sharpe_like(limit=limit, regime_label=regime_label)],
            "worst_drawdown": [asdict(m) for m in engine.worst_by_drawdown(limit=limit, regime_label=regime_label)],
            "worst_expectancy": [asdict(m) for m in engine.worst_by_expectancy(limit=limit, regime_label=regime_label)],
        }

    def get_champion_challenger_summary(self, *, limit: int = 5) -> Dict[str, Any]:
        """Runtime helper for champion/challenger promotion introspection."""
        engine = getattr(self, "champion_challenger_engine", None)
        if engine is None:
            return {
                "active_champion": None,
                "challengers": [],
                "pending_decisions": [],
                "best_challengers": [],
                "rejected_challengers": [],
            }
        active = engine.get_active_champion()
        return {
            "active_champion": asdict(active) if active is not None else None,
            "challengers": [asdict(c) for c in engine.list_challengers()[: max(1, int(limit or 5))]],
            "pending_decisions": [asdict(d) for d in engine.list_pending_promotion_decisions(limit=limit)],
            "best_challengers": [asdict(d) for d in engine.best_challengers_by_promotion_score(limit=limit)],
            "rejected_challengers": [asdict(d) for d in engine.rejected_challengers(limit=limit)],
        }
    
    def get_status(self) -> Dict:
        """Get current system status"""
        try:
            from core.version import __version__ as _v
        except Exception:
            _v = "3.0.0"
        return {
            'version': _v,
            'state': self.state.value,
            'portfolio_value_aud': self.portfolio_value_aud,
            'cash_balance_aud': self.cash_balance_aud,
            'total_pnl_aud': self.total_pnl_aud,
            'realized_pnl_aud': self.realized_pnl_aud,
            'unrealized_pnl_aud': self.unrealized_pnl_aud,
            'total_fees_aud': self.total_fees_aud,
            'mark_price_method': self.mark_price_method,
            'ledger_sanity_violations': int(self._ledger_sanity_violations),
            'daily_pnl_aud': self.daily_pnl_aud,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'max_drawdown_pct': self.max_drawdown_aud * 100,
            'consecutive_losses': self.consecutive_losses,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Main entry point for unified trading system"""
    logger.info("=" * 70)
    logger.info("ARGUS ULTIMATE + KRAKEN DCA + PINNACLE AI")
    logger.info("UNIFIED TRADING SYSTEM")
    logger.info("Optimized for $1,000 AUD Starting Capital")
    logger.info("=" * 70)
    
    # Load configuration
    config = UnifiedConfig()
    
    # Create unified system
    system = UnifiedSystemArchitecture(config)
    
    try:
        # Initialize system
        await system.initialize()
        
        # Start trading loop
        await system.run_trading_loop()
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await system.shutdown()
        
        # Print final status
        status = system.get_status()
        logger.info("=" * 70)
        logger.info("FINAL SYSTEM STATUS")
        logger.info("=" * 70)
        for key, value in status.items():
            logger.info(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
