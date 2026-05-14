# Alternative Run Paths

The **canonical production path** is `main.py paper` / `main.py live` (unified system). These **alternative run paths** use a different execution pipeline (ExecutionPipeline + rich position sizing from `risk.position_sizing`) and optional evolution or peak strategies.

---

## Commands (from main.py)

| Command | Script | What it does |
|--------|--------|--------------|
| **peak** | run_peak.py | Peak performance: best pairs, multi-factor Peak Alpha strategy, aggressive sizing, 10s cycles. |
| **ultimate** | run_ultimate.py | Ultimate mode: adapted strategy library (momentum, mean reversion, tiers, ultimate), execution pipeline, position sizer. |
| **evolution** | run_ultimate_evolution.py | Parameter evolution (godmode_evolution_v2). |
| **godmode** | run_godmode.py | Godmode evolution + adapted strategies + execution pipeline. |

**Usage (via main.py):**

```bash
python main.py peak --capital 1000
python main.py ultimate --capital 1000
python main.py evolution --capital 1000
python main.py godmode --capital 1000
```

Optional: `--config unified_config.yaml` (passed through when you use main.py).

**Or run scripts directly:**

```bash
python run_peak.py --capital 1000
python run_ultimate.py --capital 1000
python run_ultimate_evolution.py
python run_godmode.py --capital 1000
```

---

## Differences vs unified path

| Aspect | Unified (paper/live) | Alternative (peak/ultimate/godmode) |
|--------|----------------------|-------------------------------------|
| **Entry** | main.py paper / live | main.py peak / ultimate / evolution / godmode (or run_*.py) |
| **Execution** | KrakenDCAExecutionEngine (unified_execution_engine) | execution.pipeline.ExecutionPipeline |
| **Position sizing** | CapitalOptimizer1K (unified_capital_optimizer) | risk.position_sizing.PositionSizer (Kelly, volatility, regime, etc.) |
| **Signals** | AI brain + strategy engine + HFT + (now) strategy library | Adapted strategies from advanced_adapter + evolution/peak logic |
| **Config** | unified_config.yaml | Script-specific (click options + config where supported) |

Use the **unified path** for the single production bot (paper/live). Use **alternative paths** when you want the legacy pipeline, richer position sizing, or evolution runs.
