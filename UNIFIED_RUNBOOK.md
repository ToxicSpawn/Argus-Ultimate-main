## Unified Runbook (Canonical Production Path)

This repository contains multiple historical/experimental trading stacks. The **only supported production path** is the **Unified System**.

**Production deployment (R730 + desktop + 2× switches):** See [docs/DEPLOYMENT_R730_AND_DESKTOP.md](docs/DEPLOYMENT_R730_AND_DESKTOP.md) and [docs/R730_SETUP.md](docs/R730_SETUP.md).

### Canonical commands

- **Paper trading (safe)**

```bash
python main.py paper --capital 1000 --cycles 20 --cycle-seconds 60
```

- **Fast paper smoke (no wall-clock delay)**

```bash
python main.py paper --capital 1000 --cycles 50 --cycle-seconds 0 --no-multilang
```

- **Live trading (real money)**

```bash
python main.py live --capital 1000
```

Live trading requires a manual confirmation prompt and valid API keys.

- **Validation (dependency-free)**

```bash
python main.py validate
```

This performs strict unified config validation (including profile overlays).  
Set `ARGUS_VALIDATE_RUN_TESTS=1` to additionally run the stdlib `unittest` suite.

- **Health check (config, deps, ledger)**

```bash
python scripts/health_check.py --config unified_config.yaml
```

- **Health check with paper smoke (1-day paper loop; pass if no crash and equity ≥ 90% of start)**

```bash
python scripts/health_check.py --config unified_config.yaml --paper-smoke
```

Use `--paper-days N` to run N days for the paper smoke test.

### Observability — Grafana + Prometheus + Loki (Push 49)

**Start the full observability stack:**

```bash
docker compose up -d prometheus grafana loki
```

**Start everything (bot + observability):**

```bash
docker compose up -d
```

**Access Grafana:**
- URL: http://localhost:3000
- Default credentials: `admin` / `argus_admin` (override with `GRAFANA_ADMIN_PASSWORD` env var)
- Dashboards auto-load on first start from `grafana/provisioning/dashboards/`
- Datasources auto-configured: Prometheus (`http://prometheus:9090`) + Loki (`http://loki:3100`)

**Available dashboards:**

| Dashboard | UID | Description |
|-----------|-----|-------------|
| Argus Trading | `argus` | OFI, VPIN, MTF, PnL, equity, drawdown, trades |
| Argus — HMM Regime | `argus-regime` | HMM label, state probs (bull/sideways/bear), scalar |

**Prometheus metrics endpoint:**
- Bot exposes metrics on port `8001` (override: `ARGUS_METRICS_PORT`)
- Prometheus scrapes `argus:8001/metrics` every 15s (configured in `prometheus.yml`)

**Key metrics for regime monitoring (Push 46):**

| Metric | Type | Description |
|--------|------|-------------|
| `argus_regime_label` | Gauge | HMM regime: bull=1, sideways=0, bear=-1 |
| `argus_regime_scalar` | Gauge | Position-sizing scalar (1.3/1.0/0.6) |
| `argus_regime_bull_prob` | Gauge | HMM bull state probability |
| `argus_regime_sideways_prob` | Gauge | HMM sideways state probability |
| `argus_regime_bear_prob` | Gauge | HMM bear state probability |

**Loki log ingestion:**
- Loki listens on `http://loki:3100`
- Configure your log shipper (Promtail/alloy) to push Argus logs to Loki
- Loki data persists in Docker volume `loki_data`

**Reset Grafana data (wipe dashboards/prefs):**

```bash
docker compose down -v && docker compose up -d
```

### Linux Infra Hardening (Execution Island)

```bash
# Plan only (no changes)
sudo ./ops/linux/apply_execution_island.sh --iface enp3s0f0 --os-cpus 0-3 --exec-cpus 4-15 --cpu-isolation 4-15

# Apply deterministic host profile
sudo ./ops/linux/apply_execution_island.sh --apply --iface enp3s0f0 --os-cpus 0-3 --exec-cpus 4-15 --cpu-isolation 4-15

# Verify host tuning
python scripts/infra_verify_host.py --iface enp3s0f0 --output reports/infra/verification_latest.json

# Fail-closed preflight gate
python scripts/infra_preflight.py --report reports/infra/verification_latest.json --output reports/infra/infra_preflight_latest.json --max-clock-offset-us 250

# Latency/jitter report
python scripts/infra_latency_report.py --db data/unified_trades.db --output-dir reports/infra

# Replayable audit bus export/replay
python scripts/export_audit_bus.py --db data/unified_trades.db --output logs/audit_bus_latest.jsonl --limit 5000
python scripts/replay_audit_bus.py --input logs/audit_bus_latest.jsonl --speed 0

# Prebuild bundle for upcoming R740 host
python scripts/r740_prepare_bundle.py --manifest docs/hardware/R740_PREBUILD_MANIFEST.yaml --output-root deploy/r740_bundle
python scripts/r740_bundle_check.py --bundle-root deploy/r740_bundle --output reports/infra/r740_bundle_check_latest.json

# One-command prebuild suite (validate + bundle build + bundle check)
python scripts/r740_prebuild_suite.py --config unified_config.yaml --manifest docs/hardware/R740_PREBUILD_MANIFEST.yaml --bundle-root deploy/r740_bundle --suite-output reports/infra/r740_prebuild_suite_latest.json

# Day-0 acceptance (after hardware is assembled)
python scripts/r740_capture_host_facts.py --output reports/infra/r740_host_facts_latest.json
python scripts/r740_hardware_acceptance.py --spec docs/hardware/R740_ACCEPTANCE_SPEC.yaml --facts reports/infra/r740_host_facts_latest.json --output reports/infra/r740_acceptance_latest.json
```

### Configuration

- **Primary config**: `unified_config.yaml`
- Key sections used by the unified runtime:
  - **capital**: `starting_capital_aud`, position sizing bounds
  - **fx**: `aud_to_usd` (USD per 1 AUD) used for sizing/accounting on USD-quoted pairs
  - **risk**: daily loss / drawdown / consecutive loss thresholds
  - **exchanges**: `primary`, `secondary`, fee tiers
  - **ai_brain**: `min_signal_confidence`, `max_concurrent_signals`, agent counts
  - **execution_engine**: `order_type`, `retry_attempts`, `retry_delay_seconds`, `max_slippage_pct`
  - **monitoring**: Prometheus/Grafana toggles and ports

### Credentials (environment variables)

- **Kraken (required for live)**
  - `KRAKEN_API_KEY`
  - `KRAKEN_SECRET_KEY`

- **Coinbase Advanced Trade (optional, enables secondary venue in live)**
  - `COINBASE_ADVANCED_API_KEY`
  - `COINBASE_ADVANCED_API_SECRET` (PEM private key string)

Notes:
- In **paper mode**, Kraken runs in `dry_run` (orders are simulated) but can still use public market data.
- Coinbase Advanced market data endpoints are authenticated in this build; market data is expected to come from Kraken public endpoints by default.

### Market data and signals

- **Market data** is fetched via `services/market_data_service.py` (ticker/OHLCV/order book best-effort).
- **Signals** are generated by:
  - `strategies/unified/strategy_engine.py` (indicator-driven, conservative), and
  - `unified_ai_brain.py` (agent consensus + optional scoring).

### Quantum (what is actually used)

The Unified System uses **quantum-inspired** (dependency-light) components by default:

- **`quantum_consciousness/quantum_inspired_optimizer.py`**: reweights signal confidence/strength
  using a mean-variance style optimizer when OHLCV covariance is available (paper/backtest only).
- **`quant_fund_upgrades/multi_factor_risk_engine.py`**: optional additional risk metrics (paper/backtest only).
- **`quantum_walk/QuantumWalkLite`**: import-safe correlation-network “walk” heuristic (no SciPy/NetworkX/Qiskit required).

Additionally, the repo now includes an **in-repo, dependency-light circuit simulator** used by the
experimental `quantum/production_quantum_simulator.py` path:

- **`quantum_simulator.py`**: statevector simulator (≤20 qubits) + MPS backend (≤100 qubits) + optional noise hooks.
- **Config**: `unified_config.yaml` → `quantum_simulator:` (applied as env vars by `core/config_manager.py`).
- **Scoreboard**: compare trading results with quantum toggles:

```bash
python scripts/quantum_scoreboard.py --config unified_config.yaml --days 30 --timeframe 1h --csv-dir data/ohlcv
```

The large `quantum/` tree contains many experimental modules and optional vendor integrations; it is
kept import-safe via `quantum/__init__.py` but is not required for the canonical run path.

### Execution, persistence, monitoring

- **Execution engine**: `unified_execution_engine.py` (retries/backoff, slippage guardrail, best-effort reconciliation)
- **Trade ledger (SQLite)**: `data/unified_trades.db` via `monitoring/trade_ledger.py`
  - Records: symbol, side, exchange, price, size, status, commission, slippage, pnl, raw payload
- **Monitoring**: `unified_monitoring.py`
  - Prometheus metrics if `prometheus_client` is installed; otherwise monitoring is limited to logs.

### Legacy / experimental stacks (not production)

These exist in the repo but are **not** part of the canonical production path:
- `enhanced_trading_launcher.py` (legacy; previously imported by `main.py`, now removed)
- `core/argus.py` and `core/ARGUS/core/*` (legacy backtest/sim stack)
- `quantum/*` and `tests/quantum/*` (experimental; some parts are import-safe, but not canonical)
