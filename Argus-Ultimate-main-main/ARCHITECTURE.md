# Argus Ultimate — Architecture Reference

> **Version:** 8.17.0 | **Push:** 81 | **Status:** Production-ready

This document is the **single source of truth** for how Argus is structured, what the canonical entrypoints are, and what legacy files exist for historical reference only.

---

## Canonical Entrypoint Hierarchy

```
scripts/start.py          ← PRODUCTION entrypoint (Docker, cloud, Hetzner)
run_paper.py              ← paper trading (quick local dev)
run_ultimate.py           ← full-stack mode (legacy compat wrapper)
main.py                   ← CLI entrypoint (argparse, all flags)
```

**For Docker:** `docker-compose up -d` → calls `scripts/start.py` via CMD.
**For local dev:** `python run_paper.py` or `python main.py --mode paper`.
**Do not use** `argus_bot.py`, `run_godmode.py`, `run_peak.py`, or `start_paper.py` — these are legacy and will be removed in v9.

---

## Layer Map (Push 70–81)

```
┌─────────────────────────────────────────────────────────┐
│                   scripts/start.py                      │  Production entrypoint
│                   main.py (CLI)                         │  Dev entrypoint
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│               core/system.py (ArgusSystem)              │  Push 80 — Orchestrator
│  ┌──────────────────────────────────────────────────┐   │
│  │  StrategyRegistry                                │   │  Push 70
│  │    ├── MomentumStrategy                          │   │  Push 71
│  │    ├── MeanReversionStrategy                     │   │  Push 72
│  │    └── MLEnsembleStrategy                        │   │  Push 73
│  ├── AsyncSignalBus                                  │   │  Push 74
│  ├── BacktestEngine + WalkForwardEngine              │   │  Push 75–76
│  ├── MonteCarloEngine + ExecutionEngine              │   │  Push 77
│  ├── RiskManager + PositionSizer + MarginWatcher     │   │  Push 78
│  ├── FastAPI dashboard + Prometheus + WebSocket      │   │  Push 79
│  └── Docker Compose full-stack                       │   │  Push 80
│  └── [THIS FILE] Consolidation + CI hardening        │   │  Push 81
└──────────────────────────────────────────────────────┘
```

---

## Directory Reference

| Directory | Status | Purpose |
|---|---|---|
| `core/` | ✅ **Active** | Canonical system layers (push 70–81) |
| `strategies/` | ✅ **Active** | Strategy modules loaded by StrategyRegistry |
| `execution/` | ✅ **Active** | Order management, adapters |
| `risk/` | ✅ **Active** | Risk manager, position sizer |
| `backtest/` | ✅ **Active** | Backtester, walk-forward, Monte Carlo |
| `monitoring/` | ✅ **Active** | Prometheus, Grafana dashboards |
| `api/` | ✅ **Active** | FastAPI routes, WebSocket feeds |
| `connectors/` | ✅ **Active** | Exchange connectors (Binance, Kraken, paper) |
| `ml/` | ✅ **Active** | ML ensemble models, feature engineering |
| `hft/` | ✅ **Active** | HFT engine, order book, latency tools |
| `quantum/` | ✅ **Active** | Quantum simulation layer |
| `config/` | ✅ **Active** | YAML config, env overrides |
| `tests/` | ✅ **Active** | pytest suites (unit + integration) |
| `scripts/` | ✅ **Active** | Production start, deploy helpers |
| `docker/` | ✅ **Active** | Dockerfile variants, compose configs |
| `docs/` | ✅ **Active** | Extended documentation |
| `grafana/` | ✅ **Active** | Grafana provisioning |
| `infra/` | ✅ **Active** | Cloud infra (Hetzner, fly.io, Vultr) |
| `argus/` | ✅ **Active** | Core package init |
| `argus_live/` | ⚠️ **Legacy** | Pre-push-70 live trading (superseded by `core/`) |
| `argus_omega/` | ⚠️ **Legacy** | Experimental omega layer (archived) |
| `unified_trading_system.py` | ⚠️ **Legacy** | 640KB monolith — superseded by `core/system.py` |
| `unified_execution_engine.py` | ⚠️ **Legacy** | 200KB monolith — superseded by `execution/` |
| `run_godmode.py` | ⚠️ **Legacy** | Use `scripts/start.py` instead |
| `run_peak.py` | ⚠️ **Legacy** | Use `scripts/start.py` instead |
| `argus_bot.py` | ⚠️ **Legacy** | Superseded by `core/system.py` |
| `void_breaker/` | ⚠️ **Legacy** | Experimental — not integrated |
| `quant_fund_upgrades/` | ⚠️ **Legacy** | Prototype research, not production |
| `generated_strategies/` | ⚠️ **Legacy** | Auto-generated, not version-controlled logic |
| `marketplace/` | 🔬 **Planned** | Strategy marketplace (push 85+) |
| `evolution/` | 🔬 **Research** | Genetic algorithm strategy evolution |
| `web3_defi/` | 🔬 **Research** | DeFi/on-chain integrations (future) |

---

## Configuration Priority (highest → lowest)

```
ARGUS_* env vars
  └─ config/config.yml          (user overrides)
       └─ config/default_config.yml  (shipped defaults)
```

All thresholds, strategy params, exchange settings, API keys, and log levels are
controlled through this hierarchy. **Never hardcode secrets** — use `ARGUS_*` env vars
or a `.env` file (see `.env.example`).

---

## Key Files

| File | Size | Purpose |
|---|---|---|
| `core/system.py` | ~8KB | `ArgusSystem` orchestrator — start here |
| `scripts/start.py` | ~3KB | Production entrypoint |
| `config/default_config.yml` | ~4KB | Fully documented defaults |
| `docker-compose.yml` | ~3KB | Full stack (bot + prometheus + grafana + redis) |
| `Dockerfile` | ~1.6KB | Multi-stage prod image |
| `.github/workflows/ci.yml` | ~6KB | CI pipeline (lint → type → security → test → docker) |
| `UNIFIED_RUNBOOK.md` | ~8KB | Ops runbook (deploy, rollback, incident response) |
| `CHANGELOG.md` | ~3KB | Version history |

---

## Quick Start

```bash
# Full stack (recommended)
docker-compose up -d
open http://localhost:3000   # Grafana
curl http://localhost:8080/health

# Paper trading (local dev)
cp .env.example .env         # fill in your keys
python run_paper.py

# Run tests
pytest tests/ -q --tb=short

# Lint + type check
ruff check core/
mypy core/ --ignore-missing-imports
```

---

## Roadmap (Push 82+)

| Push | Planned Feature |
|---|---|
| 82 | Live Binance WebSocket feed + real order book |
| 83 | Strategy performance dashboard (Grafana panels) |
| 84 | Alerting rules (Prometheus → PagerDuty/Telegram) |
| 85 | Kubernetes Helm chart (Hetzner k3s) |
| 86 | Multi-symbol portfolio rebalancer |
| 87 | RL agent (PPO) integrated into signal bus |
| 88 | Backtest report PDF export |
| 89 | Strategy marketplace v1 |
| 90 | v9.0.0 — legacy cleanup, full consolidation |
