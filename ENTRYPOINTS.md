# Argus Ultimate — Entry Points

> **Last updated:** May 2026 — post-cleanup canonical map.

## The One Entry Point You Need

```bash
python main.py paper    # paper trading (default — safe)
python main.py live     # live trading  (requires soak gate + credentials)
python main.py hybrid   # hybrid mode
```

`main.py` is the **sole canonical entry point**. All subsystems (AI brain, strategy router,
quantum sim, 23-language mesh, HFT engine, evolution, monitoring) are wired through it.

---

## Still-Active Specialised Runners

These files have substantial independent logic and are not mere stubs:

| File | Purpose | When to use |
|------|---------|-------------|
| [`run_peak.py`](run_peak.py) | Peak mode with extended scanner logic (23KB) | When peak-mode-specific overrides are needed |
| [`run_ultimate.py`](run_ultimate.py) | Ultimate runner with embedded strategy logic (60KB) | Legacy ultimate mode |
| [`run_godmode.py`](run_godmode.py) | Godmode runner with aggressive risk settings (14KB) | Dev/testing only |
| [`pinnacle_engine.py`](pinnacle_engine.py) | Pinnacle engine module redirect | Called by main.py internally |

---

## Config Entry Points

The runtime configuration is now split into canonical sub-files:

| File | What it owns |
|------|--------------|
| [`config/constitution/profile.yaml`](config/constitution/profile.yaml) | Mode, version, timezone, hardware template |
| [`config/runtime/safety.yaml`](config/runtime/safety.yaml) | Soak gate, reconciliation, circuit breakers, emergency shutdown |
| [`config/exchange/venues.yaml`](config/exchange/venues.yaml) | Exchange fees, API key env vars, trading pairs, execution routing |
| [`config/strategy/router.yaml`](config/strategy/router.yaml) | Strategy router, whitelist, evaluation engine, champion/challenger |
| [`unified_config.yaml`](unified_config.yaml) | **Legacy** — do not add new policy here. Retained for backward compat only. |

---

## Archived Legacy Entry Points

All the old stub files (143–228 byte redirects) have been moved to
[`archive/legacy_entrypoints/`](archive/legacy_entrypoints/) with a README
explaining what each one was. They are **not deleted** — they exist for
historical reference only.

---

## Quick Checklist Before Going Live

1. Set `profile: "live"` in `config/constitution/profile.yaml`
2. Set `mode: "live"` in `config/runtime/safety.yaml`
3. Confirm `reports/soak_gate_latest.json` exists and passes all gates
4. Set env vars: `KRAKEN_API_KEY`, `KRAKEN_SECRET` (and `BINANCE_*` if using Binance)
5. Set `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` for drawdown alerts
6. Run `python main.py live`
