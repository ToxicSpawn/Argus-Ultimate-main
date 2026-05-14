# Legacy Files — Deprecation Notice

> These files are kept for historical reference and backward compatibility.
> **Do not build new features on top of them.**
> They will be archived/removed in **v9.0.0 (Push 90)**.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the canonical layer map and active modules.

---

## Deprecated Entrypoints

| File | Replacement | Remove in |
|---|---|---|
| `argus_bot.py` | `core/system.py` + `scripts/start.py` | v9.0 |
| `run_godmode.py` | `scripts/start.py` | v9.0 |
| `run_peak.py` | `scripts/start.py` | v9.0 |
| `start_paper.py` | `run_paper.py` | v9.0 |
| `run_paper_argus.py` | `run_paper.py` | v9.0 |
| `main_legacy.py` | `main.py` | v9.0 |

## Deprecated Monoliths

| File | Size | Replacement | Remove in |
|---|---|---|---|
| `unified_trading_system.py` | 640KB | `core/system.py` + all `core/` layers | v9.0 |
| `unified_execution_engine.py` | 200KB | `execution/` package | v9.0 |
| `unified_language_orchestrator.py` | 74KB | `core/` + `connectors/` | v9.0 |
| `unified_capital_optimizer.py` | 15KB | `core/risk/` + `portfolio/` | v9.0 |
| `unified_ai_brain.py` | 8.7KB | `ml/` package | v9.0 |
| `native_language_runner.py` | 22KB | `connectors/` + `multilang/` | v9.0 |

## Deprecated Directories

| Directory | Status | Notes |
|---|---|---|
| `argus_live/` | Legacy | Pre-push-70 live stack; superseded by `core/` |
| `argus_omega/` | Legacy | Experimental layer; not integrated |
| `void_breaker/` | Legacy | Prototype; no production integration |
| `quant_fund_upgrades/` | Legacy | Research prototypes; evaluated per-push |
| `generated_strategies/` | Legacy | Auto-generated; not manually maintained |
| `.auto_fixer_learning.json` | Legacy | Auto-fixer training data; not runtime |
| `.adaptive_fix_patterns.json` | Legacy | Auto-fixer training data; not runtime |

---

## Migration Guide

### Old way (pre-push-70)
```python
# DON'T DO THIS
from unified_trading_system import UnifiedTradingSystem
sys = UnifiedTradingSystem(config)
sys.run()
```

### New way (push 80+)
```python
# DO THIS
from core.system import ArgusSystem
system = ArgusSystem.paper("BTCUSDT")   # or from_config(config_dict)
await system.start()
await system.tick("BTCUSDT", price)
await system.stop()
```

### Docker (recommended)
```bash
docker-compose up -d
# That's it. Everything wires up automatically.
```
