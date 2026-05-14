# ARGUS Ultimate

Production crypto algorithmic trading system. Python 3.12+, async.

## Quick Start
- Entry: `py main.py paper` (paper trading), `py main.py live` (live)
- Config: `unified_config.yaml` (200+ fields), profiles in `config/profiles/`
- Tests: `py -m pytest tests/ -q`

## Architecture
- `main.py` → CLI entrypoint
- `unified_trading_system.py` → Master coordinator (10K lines, god object — refactor planned)
- `core/component_registry.py` → 70+ component initialization and lifecycle
- `strategies/unified/strategy_engine.py` → Signal generation
- `execution/` → Order execution (algo_orders, smart_order, multi_venue)
- `risk/` → Risk management (VaR, drawdown, circuit breakers)
- `ml/` → ML models (regime, alpha, ensemble, online learning)
- `monitoring/` → Audit trail, trade ledger, alerts
- `core/config_manager.py` → YAML config loading + validation

## Advanced Features (v8.2.0)
New modules for institutional-grade trading capabilities:

### Execution & TCA
- `monitoring/tca_enhanced.py` → Transaction Cost Analysis engine
- `execution/pov_executor.py` → Percentage of Volume execution algorithm

### Risk Management
- `risk/tail_risk_hedger.py` → Tail risk hedging with VaR/CVaR
- `risk/stress_tester_enhanced.py` → Multi-scenario stress testing
- `risk/delta_hedger.py` → Options delta hedging module

### ML & AI
- `ml/drift_detector.py` → Concept drift detection and alerts
- `ml/llm_sentiment_enhanced.py` → LLM-based sentiment analysis (FinBERT)
- `ml/uncertainty_quantifier.py` → Bayesian uncertainty quantification
- `ml/ab_testing.py` → A/B testing framework for strategies
- `ml/gnn_trainer.py` → Graph Neural Network training for cross-asset analysis
- `ml/multi_task_learner.py` → Multi-task learning with shared backbones
- `ml/causal_inference.py` → DAG-based causal inference
- `ml/diffusion_generator.py` → Diffusion models for synthetic data generation
- `ml/feature_store_realtime.py` → Real-time feature store with streaming

### Portfolio Optimization
- `portfolio/black_litterman_optimizer.py` → Black-Litterman portfolio optimization
- `quantum/quantum_optimizer.py` → Quantum-inspired portfolio optimization

### Infrastructure
- `core/event_store.py` → Event sourcing audit trail
- `core/cqrs_handler.py` → CQRS pattern (Command Query Responsibility Segregation)
- `core/tracing_enhanced.py` → Distributed tracing with OpenTelemetry
- `core/advanced_features_orchestrator.py` → Orchestrator for all advanced features

### Compliance
- `compliance/mifid2_compliance.py` → MiFID II regulatory reporting

## Key Patterns
- All components are optional — try/except with graceful degradation
- Advisory dict: `on_cycle()` returns dict of advisory keys consumed in `_execute_signals()`
- `_last_cycle_advisory` persists advisory between cycles for `_check_partial_exits()`
- `_state_lock` (threading.Lock) protects positions, portfolio_value, daily_pnl
- Lazy imports inside functions to avoid circular imports
- Advanced features use `AdvancedFeaturesOrchestrator` for coordinated initialization

## Testing
- `py -m pytest tests/ -q` — run all tests
- `py -m pytest tests/test_advanced_features.py -v` — test new advanced features
- Tests use `unittest.TestCase` + `unittest.mock.MagicMock`
- ComponentRegistry requires `config=MagicMock()` in tests

## Config
- `unified_config.yaml` is the single source of truth
- Profile overlays: `ARGUS_CONFIG_PROFILE=r740` activates `config/profiles/r740.yaml`
- All config keys registered in `core/config_manager.py` `_KNOWN_TOP_LEVEL_KEYS`
- Advanced features configured under `advanced_features:` section (21 feature flags)

## Conventions
- Use `py` launcher (not `python` or `python3`) on this machine
- Logger: `logger = logging.getLogger(__name__)` in every module
- Compliance modules (AUSTRAC, ATO CGT, MiFID II) log at WARNING, not DEBUG
- AUD is base capital currency; USD is quote currency for most pairs

## Optional Dependencies
Some advanced features require additional packages:
- `torch` — GNN training, multi-task learning, diffusion models
- `scipy` — Statistical tests, optimization
- `qiskit` — Quantum portfolio optimization (optional, falls back to classical)
- `transformers` — LLM sentiment analysis (FinBERT)
- `opentelemetry-api` — Distributed tracing

Install all optional dependencies: `pip install torch scipy qiskit transformers opentelemetry-api`
