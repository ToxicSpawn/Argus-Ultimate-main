# 23-Language System – Advancing the Bot

The Argus bot uses **23 coding languages** to advance signal processing, risk checks, cycle planning, volatility, and signal scoring. The **Unified Language Orchestrator** (`unified_language_orchestrator.py`) coordinates them, with **per-language profiles** and **aggregation** so every language improves the system.

## How it works

1. **Cycle plan** – Every trading cycle, the orchestrator calls all **23 languages** with the current context (portfolio value, cash, signal count, exchange). Each language returns a `cycle_boost` (scaled by its profile). Results are aggregated (median/mean); the **median boost is applied to signal confidence** before execution when `use_cycle_aggregate_boost: true`.

2. **Order book processing** – When processing signals, the bot sends order book data to the fastest available language (Rust/C++/Go preferred) for spread and imbalance. Each language can use a **spread_mult** from its profile (conservative languages use ≥1). If no HTTP service is up, an in-process worker runs with that language’s profile.

3. **Risk calculation** – Position vs capital checks use each language’s **risk_max_ratio** (e.g. Haskell 0.40, Rust 0.48). Optional **execute_risk_all** runs all 23 and passes only if every language passes; enable with `use_risk_all: true`.

4. **Volatility estimate** – `execute_volatility_estimate(data)` runs all 23 with prices/returns; returns **weighted median** volatility (stats languages have higher weight).

5. **Signal score** – `execute_signal_score_all(signal_data)` runs all 23; returns **median score_delta** for use in confidence.

## Languages (23)

Rust, C++, CUDA, Go, Java, Scala, Kotlin, Swift, C#, F#, JavaScript, TypeScript, Elixir, Erlang, Clojure, Haskell, Ruby, R, Julia, MATLAB, Crystal, WebAssembly, Mojo.

## Modes

- **In-process (default)** – No Docker or HTTP services required. The orchestrator runs **23 in-process workers** (Python) and tags each result with the corresponding language name. The bot still gets 23 parallel contributions per cycle and all logic/ledger paths run. This is how the bot “uses” 23 languages out of the box.

- **HTTP mesh (optional)** – When you run the multi-language Docker stack, set `multi_language.endpoints` in `unified_config.yaml` to each service URL (e.g. `http://argus-rust:8011`). The orchestrator will POST tasks to those services. Replace in-process workers with real Rust/C++/Go/etc. implementations for lower latency or custom logic.

## Config (`unified_config.yaml`)

```yaml
multi_language:
  enabled: true
  use_cycle_aggregate_boost: true   # apply median of 23 boosts to signal confidence
  use_risk_all: false               # if true, skip execution when any language fails risk
  endpoints:
    rust: "http://argus-rust:8011"
    cpp: "http://argus-cpp:8012"
    # ... (23 entries; leave empty or omit for in-process)
```

If an endpoint is missing or the service is down, that language falls back to in-process (with full per-language profile logic).

## Disabling

- In config: `multi_language.enabled: false`.
- Or run paper with: `py main.py paper --no-multilang` (disables orchestrator init).

## Per-language profiles

Each language has a profile in `unified_language_orchestrator.py` (`LANGUAGE_PROFILES`): `risk_max_ratio`, `cycle_boost_scale`, `volatility_weight`, `signal_score_weight`, `spread_mult`, `role`. In-process logic uses these so every language contributes differently (e.g. Haskell more conservative, R/Julia higher volatility weight). Real HTTP services should implement the same behavior; see **`multilang/PROTOCOL.md`** for request/response shapes for all five task types.

## Adding real implementations

To run actual code in each language:

1. Implement a small HTTP service per language that exposes:
   - `GET /health` → 200 OK
   - `POST /execute` → body `{"task_type": "cycle_plan"|"order_book_processing"|"risk_calculation"|"volatility_estimate"|"signal_score", "data": {...}, "timeout": 1.0}` → `{"ok": true, "result": {...}, "took_ms": 0.1}`

2. See **`multilang/PROTOCOL.md`** for exact `data` and `result` shapes per task type.

3. Use `docker-compose.multi-lang.yml` (if present) or your own stack to start all 23 services.

4. Point `multi_language.endpoints` in config to those services (e.g. `http://localhost:8011` for Rust).

The orchestrator will then use HTTP for every language that is reachable and fall back to in-process (with full profile logic) when a request fails or an endpoint is not set.
