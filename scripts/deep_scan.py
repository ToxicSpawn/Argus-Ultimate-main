#!/usr/bin/env python
"""ARGUS Deep Error Scanner — checks every possible error type."""
import ast, os, re, sys, sqlite3
from pathlib import Path

# Ensure we run from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 70)
print("ARGUS ULTIMATE -- DEEP ERROR SCAN")
print("=" * 70)

total_issues = 0

# 1. SYNTAX
syntax_errs = []
py_count = 0
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", ".claude", "venv")]
    for f in files:
        if not f.endswith(".py"):
            continue
        py_count += 1
        path = os.path.join(root, f)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                ast.parse(fh.read(), filename=path)
        except SyntaxError as e:
            syntax_errs.append(f"{path}:{e.lineno}: {e.msg}")
print(f"\n1. SYNTAX: {len(syntax_errs)} errors in {py_count} files")
for e in syntax_errs:
    print(f"   {e}")
if not syntax_errs:
    print("   ALL CLEAN")
total_issues += len(syntax_errs)

# 2. IMPORTS
import_errs = []
modules = [
    "unified_trading_system", "unified_execution_engine", "unified_capital_optimizer",
    "core.component_registry", "core.config_manager", "core.cross_session_memory",
    "core.autonomous_brain", "core.cognitive_engine", "core.goal_manager",
    "core.system_consciousness", "core.reasoning_engine", "core.self_improvement_orchestrator",
    "core.event_bus", "core.event_sourcing", "core.feature_store",
    "core.multi_agent_coordinator", "core.service_circuit_breaker",
    "strategies.unified.strategy_engine", "strategies.grid_trader", "strategies.bb_squeeze",
    "strategies.seasonal_patterns", "strategies.volume_profile", "strategies.strategy_generator",
    "risk.unified_risk_manager", "risk.kelly_position_sizer", "risk.dynamic_drawdown_controller",
    "risk.black_swan_detector", "risk.contagion_model", "risk.risk_parity_allocator",
    "risk.factor_model", "risk.hierarchical_risk_parity", "risk.regime_conditional_var",
    "risk.anti_gaming_layer", "risk.correlation_regime_learner", "risk.liquidity_risk_engine",
    "adaptive.regime_strategy_router", "adaptive.strategy_decay_detector",
    "adaptive.dynamic_cycle_timer", "adaptive.learning_journal", "adaptive.attention_system",
    "adaptive.predictive_planner", "adaptive.counterfactual_analyzer", "adaptive.self_debugger",
    "adaptive.auto_strategy_manager", "adaptive.auto_capital_allocator",
    "adaptive.auto_model_manager", "adaptive.auto_risk_adjuster", "adaptive.regime_forecaster",
    "adaptive.self_optimizing_meta_engine",
    "ml.model_manager", "ml.confidence_calibrator", "ml.ensemble_voter",
    "ml.signal_quality_scorer", "ml.optuna_tuner", "ml.trade_outcome_labeler",
    "ml.feature_drift_detector", "ml.transformer_price_predictor", "ml.multi_horizon_fusion",
    "ml.causal_discovery", "ml.meta_learner", "ml.triple_barrier_labeler",
    "ml.volatility_estimators", "ml.onnx_model_server", "ml.synthetic_data_generator",
    "execution.recon_recovery_engine", "execution.fill_probability_model",
    "execution.partial_fill_manager", "execution.adaptive_twap",
    "execution.cross_venue_executor", "execution.fee_optimizer",
    "execution.smart_order_router_v2", "execution.time_of_day_optimizer",
    "execution.microstructure_adapter", "execution.dead_mans_switch",
    "monitoring.alerting", "monitoring.opportunity_cost_tracker",
    "monitoring.attribution_engine", "monitoring.strategy_similarity_detector",
    "monitoring.drawdown_dna", "monitoring.streak_analyzer",
    "data.orderbook.l2_feed", "data.orderbook.order_flow_imbalance",
    "data.tick_engine", "data.liquidation_heatmap", "data.funding_rate_predictor",
    "data.onchain.chain_metrics", "data.onchain.stablecoin_flow_tracker",
    "data.onchain.exchange_reserve_monitor",
    "data.alternative.google_trends_tracker", "data.sentiment.reddit_sentiment",
    "data.defi.tvl_tracker", "data.mining.hash_rate_tracker",
    "data.market.stablecoin_dominance", "data.derivatives.options_sentiment",
    "data.etf.bitcoin_etf_flows",
    "evolution.neuroevolution", "evolution.strategy_breeder",
    "research.regime_changepoint", "research.fractal_analyzer",
    "research.entropy_signal_quality", "research.optimal_stopping",
    "research.lead_lag_discovery", "research.hurst_regime",
    "backtesting.continuous_backtester", "backtesting.ab_test_framework",
    "compliance.tax_loss_harvester", "compliance.eofy_harvester",
    "ops.watchdog", "ops.auto_restart", "ops.log_anomaly_detector",
    "ops.config_auto_tuner", "ops.health_monitor_daemon", "ops.canary_deployer",
    "ops.dead_trade_detector", "services.market_data_service",
]
for m in modules:
    try:
        __import__(m)
    except Exception as e:
        import_errs.append(f"{m}: {type(e).__name__}: {str(e)[:80]}")
print(f"\n2. IMPORTS: {len(import_errs)} errors of {len(modules)} modules")
for e in import_errs:
    print(f"   {e}")
if not import_errs:
    print(f"   ALL {len(modules)} MODULES IMPORT CLEAN")
total_issues += len(import_errs)

# 3. CONFIG
print()
config_issues = []
try:
    from core.config_manager import load_unified_trading_config
    cfg = load_unified_trading_config("unified_config.yaml")
    checks = {
        "starting_capital_aud": (cfg.starting_capital_aud, lambda v: v > 0),
        "max_total_exposure_pct": (cfg.max_total_exposure_pct, lambda v: 0 < v <= 1.0),
        "max_position_pct": (cfg.max_position_pct, lambda v: 0 < v <= 0.5),
        "take_profit_pct": (cfg.take_profit_pct, lambda v: 0 < v <= 0.20),
        "stop_loss_pct": (cfg.stop_loss_pct, lambda v: 0 < v <= 0.10),
        "max_consecutive_losses": (cfg.max_consecutive_losses, lambda v: v >= 3),
    }
    for name, (val, check) in checks.items():
        if not check(val):
            config_issues.append(f"{name}={val}")
except Exception as e:
    config_issues.append(f"LOAD FAILED: {e}")
print(f"3. CONFIG: {len(config_issues)} issues")
for i in config_issues:
    print(f"   {i}")
if not config_issues:
    print("   ALL VALUES IN RANGE")
total_issues += len(config_issues)

# 4. MODELS
print()
model_issues = []
model_count = 0
for f in os.listdir("models"):
    path = f"models/{f}"
    if os.path.isfile(path):
        model_count += 1
        if os.path.getsize(path) == 0:
            model_issues.append(f"EMPTY: {f}")
print(f"4. MODELS: {len(model_issues)} issues of {model_count} files")
for i in model_issues:
    print(f"   {i}")
if not model_issues:
    print("   ALL NON-EMPTY")
total_issues += len(model_issues)

# 5. DATABASES
print()
db_issues = []
dbs = []
for root, dirs, files in os.walk("data"):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in files:
        if f.endswith(".db"):
            dbs.append(os.path.join(root, f))
for db in dbs:
    try:
        conn = sqlite3.connect(db)
        result = conn.execute("PRAGMA integrity_check").fetchone()
        conn.close()
        if result[0] != "ok":
            db_issues.append(f"{db}: integrity={result[0]}")
    except Exception as e:
        db_issues.append(f"{db}: {e}")
print(f"5. DATABASES: {len(db_issues)} corrupt of {len(dbs)} checked")
for i in db_issues:
    print(f"   {i}")
if not db_issues:
    print(f"   ALL {len(dbs)} DATABASES HEALTHY")
total_issues += len(db_issues)

# 6. STATE
print()
state_issues = []
for f in ["KILL_SWITCH", "kill_switch"]:
    if os.path.exists(f):
        state_issues.append(f"{f} exists")
print(f"6. STATE: {len(state_issues)} issues")
if not state_issues:
    print("   CLEAN")
total_issues += len(state_issues)

# 7. ENV
print()
env_issues = []
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                val = val.strip().strip('"').strip("'")
                if not val or val in ("changeme", "xxx", "your_key_here"):
                    env_issues.append(f"{key.strip()}: placeholder")
    print(f"7. ENV: {len(env_issues)} issues")
    if not env_issues:
        print("   ALL KEYS SET")
else:
    env_issues.append(".env missing")
    print("7. ENV: MISSING")
total_issues += len(env_issues)

# 8. HISTORICAL DATA
print()
if os.path.exists("data/historical_ohlcv.db"):
    conn = sqlite3.connect("data/historical_ohlcv.db")
    count = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
    conn.close()
    print(f"8. HISTORICAL DATA: {count:,} candles")
else:
    print("8. HISTORICAL DATA: MISSING")
    total_issues += 1

# 9. DISK
print()
import shutil
total_disk, used, free = shutil.disk_usage(".")
print(f"9. DISK: {free / 1e9:.1f} GB free")
if free < 1e9:
    print("   WARNING: less than 1GB free")
    total_issues += 1

# SUMMARY
print()
print("=" * 70)
print(f"TOTAL ISSUES: {total_issues}")
if total_issues == 0:
    print("RESULT: ZERO ERRORS -- SYSTEM IS COMPLETELY CLEAN")
else:
    print(f"RESULT: {total_issues} ISSUES NEED ATTENTION")
print("=" * 70)
