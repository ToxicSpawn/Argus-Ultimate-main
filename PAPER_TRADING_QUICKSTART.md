# ARGUS Paper Trading Quickstart

Use this when `run_paper.py` fails or the full runtime is not wired cleanly yet.

## Fast start

From the repo root:

```bash
py -3 -m pip install pandas requests click rich
py -3 run_paper_argus.py --capital 1000 --symbols BTC/AUD --symbols ETH/AUD --cycles 20 --cycle-seconds 15
```

## What it does

- fetches real 1-minute Kraken public OHLC data
- runs your existing strategies:
  - momentum
  - mean_reversion
  - breakout
  - scalping
- simulates entries/exits in paper mode
- writes artifacts to `artifacts/paper/`

## Files you should see

- `artifacts/paper/journal.jsonl`
- `artifacts/paper/ledger.jsonl`
- `artifacts/paper/proving_review.json`
- `artifacts/paper/control_surface.json`
- `artifacts/paper/replay_audit.json`
- `artifacts/paper/regression_summary.json`

## If you want it to keep running

```bash
py -3 run_paper_argus.py --capital 1000 --symbols BTC/AUD --symbols ETH/AUD --cycles 0 --cycle-seconds 15
```

Stop with `Ctrl+C`.

## Important note

This runner is designed to get you **actually paper trading now** even if the larger execution runtime still has broken imports or incomplete wiring.
