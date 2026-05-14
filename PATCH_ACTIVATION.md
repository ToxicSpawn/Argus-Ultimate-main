# Patch Activation Guide

The Dynamic Kelly + Regime patch is **ready but needs a one-line import** added to your
main entry point to become active.

## Step 1 — Activate the monkey-patch

At the very top of `main.py` (or `run_ultimate.py`), before any engine instantiation:

```python
import core.execution_engine_live_patch  # noqa: F401 — activates Dynamic Kelly + Regime bootstrap
```

That single import:
- Replaces `_calculate_quantity` with `DynamicKellySizer` (rolling empirical Kelly per regime)
- Wraps `initialize()` to call `bootstrap_regime_manager(config)` at startup
- Registers `_record_fill_pnl` so the Kelly window learns from each fill

## Step 2 — Feed the regime each cycle

In your main trading loop (wherever you fetch OHLCV prices each cycle):

```python
from core.regime_loop_hook import tick as regime_tick

# prices_dict: {"BTC/USDT": close_series, "ETH/USDT": close_series, ...}
await regime_tick(prices_dict, primary="BTC/USDT")
```

This calls `CrossAssetRegimeDetector → RegimeConsensusWeighter → AdaptiveATRStops` automatically
and caches the result (TTL: 60s by default, configurable via `regime_cache_ttl_s` in config).

## Step 3 — Record PnL after fills

After each confirmed fill in `execute_signals`, add:

```python
self._record_fill_pnl(rd)   # rd = the execution result dict
```

This is already wired in `execution_engine_live_patch.py` — the method is attached to the class
automatically.  You only need to call it at the right point if you customise the fill loop.

## Step 4 — Run WF + Optuna calibration

```bash
python scripts/run_wf_optuna.py \
    --symbol BTC/USDT \
    --timeframe 1d \
    --n-trials 200 \
    --train-days 365 \
    --test-days 90 \
    --output results/wf_optuna_btc.json
```

Update `kelly_window`, `kelly_max_fraction`, `regime_lookback` in `unified_config.yaml` with the
output `best_params`.

## Config knobs (all optional — defaults work out of the box)

| Key | Default | Effect |
|-----|---------|--------|
| `dynamic_kelly_sizing` | `true` | Enable Dynamic Kelly |
| `kelly_window` | `50` | Rolling trade window for Kelly estimation |
| `kelly_max_fraction` | `0.25` | Max fraction of capital per trade |
| `kelly_min_fraction` | `0.01` | Floor fraction |
| `regime_cache_ttl_s` | `60` | Seconds before regime is re-detected |
| `regime_lookback` | `60` | Candles for regime detection |
| `regime_vol_threshold` | `0.025` | Vol threshold for regime classification |
| `regime_trend_threshold` | `0.015` | Trend threshold for regime classification |
