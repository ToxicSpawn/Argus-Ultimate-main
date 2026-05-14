# Paper Trading Scripts

All paper-trading runners live here. Run from the **repo root**.

| Script | Purpose |
|--------|---------|
| `paper_adaptive.py` | Adaptive regime-aware paper run |
| `paper_all_skills.py` | All strategies enabled (stress test) |
| `paper_enhanced_v2.py` | Enhanced v2 paper run |
| `paper_final.py` | Final consolidated paper runner |
| `paper_kraken.py` | Kraken-specific paper trading |
| `paper_real_data.py` | Paper run using live market data |
| `paper_ultimate_v3.py` | Ultimate v3 paper runner |
| `paper_unified_learning.py` | Unified learning loop paper run |

## Quick start

```bash
# Recommended starting point
python scripts/runners/run_paper.py

# Or target a specific flavour
python paper_trading/paper_kraken.py
```

See [`ENTRYPOINTS.md`](../ENTRYPOINTS.md) for the full entrypoint reference.
