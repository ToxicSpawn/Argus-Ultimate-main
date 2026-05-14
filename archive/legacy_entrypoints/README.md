# Archived Legacy Entry Points

These files were moved here during the unified entry point cleanup (May 2026).
All of them were thin stubs or redirects to `main.py`. **Do not use them.**
The canonical entry point is `main.py` at the repo root.

```
python main.py paper    # paper trading
python main.py live     # live trading (requires soak gate + credentials)
python main.py hybrid   # hybrid mode
```

## What each file was

| File | Size | Was |
|------|------|-----|
| `argus_bot.py` | 143B | Earliest standalone bot stub — redirected to main |
| `argus_max_adaptation.py` | 154B | Max adaptation experiment stub |
| `argus_omega_v2.py` | 148B | Omega v2 iteration stub |
| `argus_quantum_adaptive.py` | 156B | Quantum adaptive stub |
| `argus_ultimate.py` | 148B | Prior named version stub |
| `argus_ultimate_integration.py` | 195B | Integration test stub |
| `launch_ultimate.py` | 165B | Launch wrapper stub |
| `launch_quantum_maximum.py` | 219B | Quantum-max mode launcher stub |
| `launch_quantum_production.py` | 228B | Production quantum launcher stub |
| `main_adaptive.py` | 147B | Adaptive-focused variant stub |
| `main_legacy.py` | 386B | Explicit legacy redirect (now superseded) |
| `native_language_runner.py` | 188B | Rust/C++ dispatcher stub |
| `paper_adaptive.py` | 158B | Paper adaptive stub |
| `paper_all_skills.py` | 166B | Paper all-skills stub |
| `paper_enhanced_v2.py` | 161B | Paper enhanced v2 stub |
| `paper_final.py` | 155B | Paper final stub |
| `paper_kraken.py` | 154B | Paper Kraken stub |
| `paper_real_data.py` | 159B | Paper real data stub |
| `paper_ultimate_v3.py` | 167B | Paper ultimate v3 stub |
| `paper_unified_learning.py` | 184B | Paper unified learning stub |
| `run_demo.py` | 146B | Demo mode stub |
| `run_evolution.py` | 161B | Evolution runner stub |
| `run_maximum_evolution.py` | 185B | Max evolution stub |
| `run_optimize.py` | 158B | Optimizer stub |
| `run_paper.py` | 149B | Paper runner stub |
| `run_paper_argus.py` | 159B | Paper argus stub |
| `run_paper_test.py` | 158B | Paper test stub |
| `run_pinnacle.py` | 158B | Pinnacle runner stub |
| `run_quantum_evolution.py` | 185B | Quantum evolution stub |
| `run_ultimate_evolution.py` | 188B | Ultimate evolution stub |
| `run_validation_backtest.py` | 191B | Validation backtest stub |
| `start_paper.py` | 155B | Paper start stub |
| `train_pinnacle.py` | 164B | Pinnacle trainer stub |

## Still-active non-legacy runners (kept at root)

| File | Purpose |
|------|--------|
| `run_peak.py` | Peak mode — has substantial independent logic (23KB) |
| `run_ultimate.py` | Ultimate runner — has substantial independent logic (60KB) |
| `run_godmode.py` | Godmode runner — has substantial independent logic (14KB) |
| `pinnacle_engine.py` | Pinnacle engine module (302B redirect to core engine) |
