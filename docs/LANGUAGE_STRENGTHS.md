# Using Each Coding Language to Its Strengths

The orchestrator routes work so each of the 23 languages is used where it is strongest.

---

## Strength groups

| Strength | Languages | Best for |
|----------|-----------|----------|
| **Speed** | Rust, C++, CUDA, Go, Crystal, Mojo | Order book processing, risk (low latency); single-task execution tries these first for order book and risk. |
| **Correctness** | Haskell, F#, Scala, Clojure, Java, Kotlin | Risk (strict limits), cycle plan (conservative view); risk single-task tries these first; cycle aggregate exposes `conservative_median` from these only. |
| **Stats** | R, Julia, MATLAB | Volatility estimate, signal score; weighted higher in volatility and signal-score aggregation. |
| **Concurrency** | Elixir, Erlang | Cycle plan, coordination; full 23 still run, these add diversity. |
| **Ecosystem** | Swift, C#, JavaScript, TypeScript, Ruby, WebAssembly | Cycle plan, signal score, APIs/dashboards; same protocol across runtimes. |

---

## How it’s used

1. **Single-task (order book, risk, volatility, signal score)**  
   `execute_task` tries languages in a **strength order** for that task:
   - **Order book** → Speed first (Rust, C++, Go, …), then others.
   - **Risk** → Correctness first (Haskell, F#, Scala, …), then Speed, then others.
   - **Volatility** → Stats first (R, Julia, MATLAB), then others.
   - **Signal score** → Stats then Correctness, then others.

2. **Cycle plan (all 23)**  
   All 23 run every cycle. Aggregation adds:
   - **median_boost** – median of all 23 (used for confidence).
   - **conservative_median** – median of Correctness languages only (stricter).  
   Config: `use_conservative_cycle_boost: true` uses `conservative_median` for the cycle boost applied to confidence.

3. **Volatility estimate (all 23)**  
   Stats languages (R, Julia, MATLAB) get a **1.5× weight** in the weighted median so their strength (numerics/stats) drives the result more.

4. **Signal score (all 23)**  
   Stats get **1.5×** and Correctness **1.25×** weight in the weighted median.

5. **Risk all (all 23)**  
   - **passed** – true only if every language passes.
   - **conservative_pass** – true only if every **Correctness** language passes.  
   Config: `use_risk_all: true` gates on `passed`; `use_conservative_risk: true` gates on `conservative_pass` (correctness languages as the strict gate).

---

## Config (`unified_config.yaml` → `multi_language`)

```yaml
multi_language:
  enabled: true
  use_cycle_aggregate_boost: true    # use median (or conservative) boost on confidence
  use_conservative_cycle_boost: false  # true = use correctness-language median only
  use_risk_all: false                # true = skip execution if any language fails risk
  use_conservative_risk: false       # true = skip if any correctness language fails risk
  endpoints: { ... }
```

---

## Summary

- **Speed** → order book and risk (first choice for single-task).
- **Correctness** → risk and cycle (first for risk; conservative cycle median and risk gate).
- **Stats** → volatility and signal score (higher weight in aggregation).
- **Concurrency / Ecosystem** → cycle plan and signal score (diversity and integration).

Real HTTP services in each language should implement the same logic; the orchestrator already routes and weights by these strengths.
