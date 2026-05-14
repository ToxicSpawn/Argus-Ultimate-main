# Argus Polyglot Activation Plan

Argus is Python-authoritative for ML training, orchestration, risk state, and
order lifecycle. Other languages should accelerate bounded tasks and return
plain JSON/numeric outputs.

## Active layers

- **Python**: ML training, PyTorch/sklearn/Optuna, live coordinator.
- **Rust**: LOB hot path, correlation/VaR/Kelly/z-score, execution sidecar.
- **C/C++**: fast math and L2 order-book kernels.
- **CUDA/C++**: optional Monte Carlo, batch EMA, correlation, signal scoring.
- **Julia**: portfolio optimization / risk parity / Kelly research kernels.
- **R**: statistics, VaR/CVaR, skew, confidence calibration via optional bridge.
- **SQL**: Timescale/Postgres views for win-rate, execution quality, language latency.
- **TypeScript/JavaScript**: dashboard/control-plane services and mesh tasks.
- **Go**: order-router and reliable collector/daemon pattern.

## Safe integration order

1. Keep Python as the final trading authority.
2. Use `core.polyglot_engine.PolyglotEngine` for optional native/fallback kernels.
3. Use `unified_language_orchestrator.py` for HTTP mesh tasks and consensus.
4. Add analytics through SQL views before changing live write paths.
5. Only offload live execution as helper calculations; never mutate positions or orders outside Python.

## Verification

Run:

```powershell
py scripts/polyglot_health_check.py
```

The script writes `data/polyglot_health_report.json` and confirms fallbacks work
when native runtimes are unavailable.
