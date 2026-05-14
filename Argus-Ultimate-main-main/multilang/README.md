# Multi-Language Mesh (23 Languages)

This folder holds the **protocol**, **reference service**, and **Docker setup** for all 23 language services. The orchestrator is **`unified_language_orchestrator.py`** at the repo root.

## Contents

- **PROTOCOL.md** – HTTP API every language service must implement: `GET /health`, `POST /execute` with `task_type` and `data`. Covers all five task types.
- **service/** – Single FastAPI app that implements the protocol; each container runs with `LANGUAGE=rust`, `LANGUAGE=cpp`, etc., so all 23 are runnable from one codebase.
- **Dockerfile** – Builds the service image (Python 3.11, FastAPI, uvicorn).
- **docker-compose.yml** – Starts all 23 containers (argus-rust:8011 through argus-mojo:8033).

## Run all 23 services

From repo root:

```bash
docker compose -f multilang/docker-compose.yml build
docker compose -f multilang/docker-compose.yml up -d
```

- **Bot on host**: Point `unified_config.yaml` at localhost so the bot can reach the mesh:

```yaml
multi_language:
  enabled: true
  endpoints:
    rust: "http://127.0.0.1:8011"
    cpp: "http://127.0.0.1:8012"
    cuda: "http://127.0.0.1:8013"
    go: "http://127.0.0.1:8014"
    java: "http://127.0.0.1:8015"
    scala: "http://127.0.0.1:8016"
    kotlin: "http://127.0.0.1:8017"
    swift: "http://127.0.0.1:8018"
    csharp: "http://127.0.0.1:8019"
    fsharp: "http://127.0.0.1:8020"
    javascript: "http://127.0.0.1:8021"
    typescript: "http://127.0.0.1:8022"
    elixir: "http://127.0.0.1:8023"
    erlang: "http://127.0.0.1:8024"
    clojure: "http://127.0.0.1:8025"
    haskell: "http://127.0.0.1:8026"
    ruby: "http://127.0.0.1:8027"
    r: "http://127.0.0.1:8028"
    julia: "http://127.0.0.1:8029"
    matlab: "http://127.0.0.1:8030"
    crystal: "http://127.0.0.1:8031"
    webassembly: "http://127.0.0.1:8032"
    mojo: "http://127.0.0.1:8033"
```

- **Bot in Docker (same compose)**: Use hostnames `http://argus-rust:8011`, `http://argus-cpp:8012`, etc. (as in the default config).

Quick health check:

```bash
curl -s http://127.0.0.1:8011/health
curl -s -X POST http://127.0.0.1:8011/execute -H "Content-Type: application/json" -d '{"task_type":"cycle_plan","data":{"portfolio_value_aud":1000,"cash_balance_aud":500,"signals":2,"primary_exchange":"kraken"},"timeout":1}'
```

## Improvements (all 23 languages)

The orchestrator now gives each language **everything possible** in-process:

1. **Per-language profiles** (`LANGUAGE_PROFILES`) – Each of the 23 has:
   - `risk_max_ratio` (e.g. Haskell 0.40, Rust 0.48)
   - `cycle_boost_scale`, `volatility_weight`, `signal_score_weight`, `spread_mult`, `role`

2. **Richer in-process logic**
   - **Order book**: spread and imbalance with language-specific `spread_mult`.
   - **Risk**: pass/fail using that language’s `risk_max_ratio`.
   - **Cycle plan**: boost scaled by profile and context (signals, cash ratio).
   - **Volatility estimate**: from prices/returns; stats languages weighted higher.
   - **Signal score**: per-language score delta with weight.

3. **Aggregation**
   - **Cycle plan**: `aggregate_cycle_plan_results()` → median/mean boost, consensus; used to adjust signal confidence before execution.
   - **Volatility**: `execute_volatility_estimate()` runs all 23, returns weighted median.
   - **Signal score**: `execute_signal_score_all()` runs all 23, returns median score delta.
   - **Risk**: `execute_risk_all()` runs all 23; pass only if every language passes (optional gate in trading loop).

4. **Config** (`unified_config.yaml` → `multi_language`)
   - `use_cycle_aggregate_boost: true` – apply median cycle boost to confidence.
   - `use_risk_all: false` – set true to skip execution when any language fails risk.

## Using each language to its strength

The orchestrator routes tasks by strength (see **`docs/LANGUAGE_STRENGTHS.md`**):

- **Order book / risk (single call)** → tries **Speed** languages first (Rust, C++, Go, CUDA, Crystal, Mojo), then **Correctness** (Haskell, F#, Scala, …) for risk.
- **Volatility / signal score** → **Stats** languages (R, Julia, MATLAB) get higher weight in aggregation.
- **Cycle plan** → all 23 run; **Correctness** languages drive `conservative_median`; config can use it for a stricter boost.
- **Risk all** → optional gate on **Correctness** only (`conservative_pass`).

So native implementations in Rust/C++/Go will be preferred for order book and risk; R/Julia/MATLAB for volatility; Haskell/F#/Scala for strict risk and conservative cycle view.

## Replacing with a native implementation

The provided service is one Python app run as 23 containers. You can replace any language with a **native** server (e.g. real Rust, Go, C++):

1. Implement HTTP server with `GET /health` and `POST /execute` (see **PROTOCOL.md**).
2. For each `task_type`, parse `data` and return the `result` shape from PROTOCOL.md.
3. Use the language’s profile (see `LANGUAGE_PROFILES` in `unified_language_orchestrator.py` or `service/protocol_logic.py`) for `risk_max_ratio`, `spread_mult`, etc.
4. Run your service on the same port (e.g. Rust on 8011) and either stop the `argus-rust` container or point the config to your service URL.

When an endpoint is missing or down, the orchestrator falls back to the in-process worker for that language.
