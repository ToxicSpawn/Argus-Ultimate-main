# Coding Languages in Argus – What They Are and How Each Is Used

The bot uses **23 coding languages** through the Unified Language Orchestrator. Each has a **role** (speed, correctness, stats, concurrency, ecosystem) and is used for specific **tasks** and **aggregations**.

---

## The 23 languages

| # | Language    | Role         | Risk (max ratio) | How it’s used |
|---|-------------|--------------|------------------|----------------|
| 1 | **Rust**    | speed        | 48%              | Order book, risk, slippage, liquidity, market impact (first choice for low-latency). Cycle plan every cycle. |
| 2 | **C++**     | speed        | 48%              | Same as Rust: hot path (order book, risk, slippage). Cycle plan every cycle. |
| 3 | **CUDA**    | speed        | 46%              | Order book, risk; can run GPU batch volatility/regime. Cycle plan every cycle. |
| 4 | **Go**      | speed        | 47%              | Order book, risk, slippage, liquidity. Cycle plan every cycle. |
| 5 | **Java**    | correctness  | 45%              | Risk first (strict), then cycle plan; conservative_median and conservative_pass. |
| 6 | **Scala**   | correctness  | 44%              | Same as Java; higher signal_score_weight. Conservative cycle and risk gate. |
| 7 | **Kotlin**  | correctness  | 45%              | Risk, cycle plan, signal score. Conservative view. |
| 8 | **Swift**   | ecosystem    | 46%              | Cycle plan, signal score; same protocol for iOS/macOS tools. |
| 9 | **C#**      | ecosystem    | 45%              | Cycle plan, risk, signal score; .NET integration. |
|10 | **F#**      | correctness  | 42%              | Strictest correctness language: risk, cycle (conservative_median), signal score. |
|11 | **JavaScript** | ecosystem | 46%              | Cycle plan, signal score; Node/browser, dashboards. |
|12 | **TypeScript** | ecosystem | 45%              | Same as JS with types; APIs and front-end. |
|13 | **Elixir**  | concurrency  | 45%              | Cycle plan, many concurrent tasks; diversity in aggregation. |
|14 | **Erlang**  | concurrency  | 44%              | Same as Elixir; fault-tolerant, concurrent cycle contributions. |
|15 | **Clojure** | correctness  | 44%              | Risk, cycle plan, signal score; data-oriented, conservative. |
|16 | **Haskell** | correctness | 40%              | Strictest risk (40% max); conservative_median and conservative_pass. |
|17 | **Ruby**    | ecosystem    | 46%              | Cycle plan, signal score; scripting and tooling. |
|18 | **R**       | stats        | 44%              | Volatility, regime, correlation, signal score, calibration (1.5× weight in stats aggregation). |
|19 | **Julia**   | stats        | 45%              | Volatility, regime, correlation, position sizing (1.5× weight). |
|20 | **MATLAB**  | stats        | 44%              | Volatility, regime, correlation (1.5× weight); research numerics. |
|21 | **Crystal** | speed       | 47%              | Order book, risk, slippage; Ruby-like, native speed. |
|22 | **WebAssembly** | ecosystem | 45%              | Cycle plan, signal score; portable, sandboxed (browser/server). |
|23 | **Mojo**    | speed        | 47%              | Order book, risk, numerics; Python-like, high performance. |

---

## How each is used (by role)

### Speed (Rust, C++, CUDA, Go, Crystal, Mojo)
- **Tried first for:** order_book_processing, risk_calculation, slippage_estimate, liquidity_score, market_impact.
- **Used in:** every cycle_plan; single-task calls prefer these when an HTTP service is available.
- **Goal:** low latency, minimal GC; real implementations in these languages give the biggest latency gain.

### Correctness (Haskell, F#, Scala, Clojure, Java, Kotlin)
- **Tried first for:** risk_calculation, position_sizing, drawdown_check.
- **Used in:** cycle_plan → **conservative_median** (median of only these 6); **conservative_pass** in risk_all (all 6 must pass).
- **Goal:** strict limits (e.g. Haskell 40% max exposure), fewer bugs; use as the “strict” vote.

### Stats (R, Julia, MATLAB)
- **Tried first for:** volatility_estimate, regime_estimate, correlation_estimate, confidence_calibration.
- **Used in:** volatility and signal_score aggregation with **1.5× weight**; regime_consensus.
- **Goal:** numerics, time series, calibration; drive volatility and regime when all 23 run.

### Concurrency (Elixir, Erlang)
- **Used in:** cycle_plan every cycle; signal_filter, heartbeat; diversity in aggregation.
- **Goal:** many small concurrent contributions; can run aggregation or coordination in real implementations.

### Ecosystem (Swift, C#, JavaScript, TypeScript, Ruby, WebAssembly)
- **Used in:** cycle_plan, signal_score; same protocol in Node, browser, .NET, iOS, Ruby.
- **Goal:** reuse same logic in dashboards, APIs, mobile; one protocol, many runtimes.

---

## Tasks each language can run (15 task types)

All 23 can run every task; the **order** in which they’re tried (single-task) or **weight** (aggregation) depends on role.

| Task                  | Who’s tried first / weighted higher      |
|-----------------------|-------------------------------------------|
| cycle_plan            | All 23 every cycle; correctness → conservative_median. |
| order_book_processing | Speed (Rust, C++, Go, …).                 |
| risk_calculation      | Correctness, then Speed.                  |
| volatility_estimate   | Stats (R, Julia, MATLAB) 1.5× weight.     |
| signal_score         | Stats 1.5×, Correctness 1.25×.           |
| regime_estimate       | Stats first.                              |
| slippage_estimate     | Speed first.                              |
| position_sizing       | Correctness, then Stats.                  |
| drawdown_check        | Correctness first.                        |
| correlation_estimate  | Stats first.                              |
| liquidity_score       | Speed first.                              |
| market_impact         | Speed, then Stats.                        |
| signal_filter         | All 23 (majority vote).                   |
| confidence_calibration| Stats, then Correctness.                  |
| heartbeat             | All 23 (observability).                   |

---

## Profile knobs (per language)

Each language has a **profile** that shapes its output and weight:

- **risk_max_ratio** – Max position/capital (e.g. Haskell 0.40, Rust 0.48). Lower = more conservative.
- **cycle_boost_scale** – Scale for cycle_boost (e.g. R/Julia 1.02–1.05). Stats languages slightly higher.
- **volatility_weight** – Weight in volatility aggregation (stats > 1).
- **signal_score_weight** – Weight in signal score aggregation.
- **spread_mult** – Multiplier on effective spread (≥ 1 = more conservative execution).
- **role** – speed | correctness | stats | concurrency | ecosystem (drives task order and aggregation).

---

## Summary

- **23 languages:** Rust, C++, CUDA, Go, Java, Scala, Kotlin, Swift, C#, F#, JavaScript, TypeScript, Elixir, Erlang, Clojure, Haskell, Ruby, R, Julia, MATLAB, Crystal, WebAssembly, Mojo.
- **Each is used** in cycle_plan every cycle and in the 15 task types; **single-task** calls try languages in a **strength order** (speed for order book/slippage, correctness for risk/drawdown, stats for volatility/regime).
- **Aggregation** uses all 23 for median_boost and consensus; **correctness** languages drive conservative_median and conservative_pass; **stats** languages get higher weight in volatility and signal score.

For config and options, see **`docs/LANGUAGE_STRENGTHS.md`** and **`multilang/PROTOCOL.md`**.
