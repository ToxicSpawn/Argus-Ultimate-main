# Everything That Can Be Added to All 23 Coding Languages

A single checklist of **every** capability, task type, field, and extension that can be added to the 23-language mesh so each language can contribute more.

---

## 1. Current (already implemented)

| Area | What each of the 23 has |
|------|--------------------------|
| **Task types** | `cycle_plan`, `order_book_processing`, `risk_calculation`, `volatility_estimate`, `signal_score` |
| **Profile fields** | `risk_max_ratio`, `cycle_boost_scale`, `volatility_weight`, `signal_score_weight`, `spread_mult`, `role` |
| **API** | `GET /health`, `POST /execute` with `task_type`, `data`, `timeout` |
| **Cycle plan data** | `portfolio_value_aud`, `cash_balance_aud`, `signals`, `primary_exchange` |
| **Cycle plan result** | `language`, `cycle_boost`, `cycle_boost_scale`, `ok` |
| **Order book data** | `bids`, `asks` (arrays of [price, size]) |
| **Order book result** | `spread_bps`, `imbalance`, `mid`, `language`, `spread_mult` |
| **Risk data** | `position_value`, `capital` |
| **Risk result** | `passed`, `exposure_ratio`, `max_ratio`, `language` |
| **Volatility data** | `prices` or `ohlcv_close` or `returns` |
| **Volatility result** | `volatility_annual_bps`, `volatility_weight`, `language`, `ok` |
| **Signal score data** | `confidence`, `score`, other signal fields |
| **Signal score result** | `score_delta`, `signal_score_weight`, `base_score`, `language`, `ok` |
| **Strength routing** | Speed / Correctness / Stats / Concurrency / Ecosystem; task-order and aggregation weights |
| **Aggregation** | `median_boost`, `conservative_median`, `conservative_pass`, weighted volatility/signal |

---

## 2. New task types (addable to all 23)

| Task type | Request `data` | Response `result` | Use |
|-----------|----------------|-------------------|-----|
| **regime_estimate** | `prices`, `returns`, `window` | `regime` (e.g. "trend"/"mean_revert"/"high_vol"), `confidence`, `language` | Regime-aware sizing; stats languages preferred. |
| **slippage_estimate** | `side`, `quantity`, `order_book` (bids/asks), `participation_rate` | `slippage_bps`, `language` | Pre-trade cost; speed languages preferred. |
| **position_sizing** | `capital`, `volatility_bps`, `confidence`, `max_risk_pct` | `size_pct`, `size_abs`, `language` | Per-language sizing view; correctness + stats. |
| **drawdown_check** | `current_equity`, `peak_equity`, `max_drawdown_pct` | `passed`, `current_drawdown_pct`, `language` | Circuit-breaker style; correctness preferred. |
| **correlation_estimate** | `series_a`, `series_b` (arrays) or `returns_matrix` | `correlation`, `language` | Pair/multi-asset; stats languages. |
| **liquidity_score** | `bids`, `asks`, `depth_levels` | `liquidity_score` (0–1), `depth_bps`, `language` | Execution quality; speed languages. |
| **market_impact** | `side`, `quantity`, `adv` (avg daily volume), `volatility` | `impact_bps`, `language` | Large orders; speed + stats. |
| **signal_filter** | `signal` (full object), `regime`, `volatility` | `accept` (bool), `filter_reason`, `language` | Pre-execution filter; all 23. |
| **confidence_calibration** | `historical_confidences`, `historical_pnl` | `calibrated_confidence`, `language` | Calibrate scores; stats. |
| **heartbeat** | `cycle_id`, `timestamp` | `ok`, `latency_ms`, `language` | Observability; all 23. |

---

## 3. New request/response fields (addable per existing task)

| Task | Add to `data` | Add to `result` |
|------|----------------|-----------------|
| **cycle_plan** | `regime`, `volatility_annual_bps`, `recent_pnl`, `open_positions_count` | `cycle_boost`, `sizing_hint` (e.g. "aggressive"/"neutral"/"reduce"), `uncertainty` |
| **order_book_processing** | `depth` (levels to use), `symbol`, `timestamp` | `spread_bps`, `imbalance`, `mid`, `depth_10_bps`, `vwap_estimate`, `microprice` |
| **risk_calculation** | `max_drawdown_pct`, `var_95_pct`, `per_symbol_limits` | `passed`, `exposure_ratio`, `max_ratio`, `var_contribution`, `tail_risk_flag` |
| **volatility_estimate** | `window_days`, `annualization_factor`, `method` ("close" / "garman_klass" / "parkinson") | `volatility_annual_bps`, `volatility_weight`, `realized_bps`, `confidence_interval` |
| **signal_score** | `symbol`, `strategy`, `regime`, `volatility` | `score_delta`, `signal_score_weight`, `base_score`, `recommendation` ("hold"/"size_up"/"size_down") |

---

## 4. New profile fields (addable per language)

| Profile key | Type | Meaning |
|-------------|------|---------|
| **preferred_tasks** | list of strings | Task types this language is preferred for (e.g. `["order_book_processing", "risk_calculation"]`). |
| **max_latency_ms** | number | Max acceptable latency; orchestrator can skip if slower. |
| **timeout_override** | number | Per-language timeout for `/execute`. |
| **regime_weight** | number | Weight when aggregating regime_estimate (stats > 1). |
| **slippage_weight** | number | Weight when aggregating slippage (speed > 1). |
| **drawdown_max_ratio** | number | Stricter drawdown limit for correctness languages. |
| **metadata** | object | Optional: version, compiler, runtime (e.g. "rust 1.75", "node 20"). |

---

## 5. New API endpoints (addable to every service)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/ready` | Readiness (dependencies up); 200 when ready to accept work. |
| GET | `/metrics` | Prometheus-style or JSON metrics (request count, latency percentiles, errors). |
| POST | `/batch` | Execute multiple tasks in one request: `{"tasks": [{task_type, data}, ...]}` → `{"results": [...]}`. |
| GET | `/capabilities` | Return `{"task_types": [...], "language": "...", "profile": {...}}`. |
| POST | `/warm` | Optional warm-up (e.g. JIT, connection pool); no required response. |

---

## 6. Observability and telemetry (addable)

| Item | Where | Description |
|------|--------|-------------|
| **Request ID** | Request body or header | `correlation_id` passed through and returned in response for tracing. |
| **Language tag** | Response / metrics | Every response includes `language`; metrics tagged by language. |
| **Latency histogram** | Per service | `took_ms` per task type and language; p50/p95/p99. |
| **Error rate** | Per service | Count of `ok: false` by task type and language. |
| **Ledger** | Orchestrator → trade ledger | Already have `record_language_call`; extend with task_type, correlation_id, latency. |
| **Health payload** | GET /health | Optional `{"language": "rust", "version": "...", "uptime_s": ...}`. |

---

## 7. Config and orchestration (addable)

| Config key | Scope | Description |
|------------|--------|-------------|
| **task_timeouts** | Per task_type | e.g. `order_book_processing: 0.5`, `volatility_estimate: 2.0`. |
| **language_overrides** | Per language | Override profile (e.g. `rust.risk_max_ratio: 0.46`) without code change. |
| **disable_tasks** | Per language | e.g. `matlab: [order_book_processing]` to skip OB for MATLAB. |
| **batch_size** | Orchestrator | Max tasks per `/batch` call. |
| **use_regime_estimate** | Global | When true, call regime_estimate (all 23 or stats-only) and pass into cycle_plan/sizing. |
| **use_slippage_estimate** | Global | When true, call slippage before execution and reject/cap size if too high. |
| **use_drawdown_check** | Global | When true, call drawdown_check (all 23 or correctness-only) and gate execution. |

---

## 8. Aggregation extensions (addable)

| Aggregate | Source | Output | Use |
|-----------|--------|--------|-----|
| **regime_consensus** | regime_estimate (all 23 or stats) | `regime`, `confidence`, `stats_median` | Feed into cycle_plan and position_sizing. |
| **slippage_consensus** | slippage_estimate (speed preferred) | `slippage_bps_median`, `slippage_bps_p95` | Pre-trade cost check. |
| **sizing_consensus** | position_sizing (all 23) | `size_pct_median`, `size_pct_conservative` (correctness), `size_pct_stats` (stats) | Final size = f(median, conservative). |
| **drawdown_consensus** | drawdown_check (all 23) | `passed`, `conservative_pass` | Gate execution like risk_all. |
| **filter_consensus** | signal_filter (all 23) | `accept_count`, `reject_count`; accept only if majority or unanimity. | Reduce bad signals. |

---

## 9. Data passed into every task (addable)

Global context the orchestrator can inject into every `data` payload (so each language sees the same world):

| Field | Type | Description |
|-------|------|-------------|
| **cycle_id** | string/number | Current cycle id. |
| **timestamp_utc** | string (ISO) | Request time. |
| **primary_exchange** | string | Already in cycle_plan; can add to all. |
| **portfolio_value_aud** | number | For sizing/risk context. |
| **regime** | string | If regime_estimate was run. |
| **volatility_annual_bps** | number | If volatility was computed. |
| **correlation_id** | string | For tracing. |

---

## 10. Language-specific extensions (addable per language)

Implementations can expose extra behavior per language without changing the protocol:

| Language | Extension idea |
|----------|----------------|
| **Rust / C++ / Go** | Native order book parsing, SIMD for spread/imbalance; sub-ms latency. |
| **CUDA** | Batch volatility/regime over many symbols; GPU kernels for correlation matrix. |
| **Java / Kotlin / Scala** | JVM risk engine (VaR, CVaR); rich type guarantees in result. |
| **Haskell / F# / Clojure** | Formal invariants (e.g. exposure always ≤ max_ratio); proof-friendly outputs. |
| **R / Julia / MATLAB** | Full stats (GARCH, regime models, calibration); return extra diagnostics. |
| **Elixir / Erlang** | Many concurrent small tasks; aggregate results in process. |
| **JavaScript / TypeScript** | Same logic in Node and browser; dashboard or client-side checks. |
| **Swift** | Same protocol for iOS/macOS tools. |
| **WebAssembly** | Same logic in WASM for portability and sandboxing. |
| **Mojo** | High-performance numerics with Python-like API. |

---

## 11. Summary table: addable by area

| Area | Addable items |
|------|----------------|
| **Task types** | regime_estimate, slippage_estimate, position_sizing, drawdown_check, correlation_estimate, liquidity_score, market_impact, signal_filter, confidence_calibration, heartbeat |
| **Request/response fields** | Extra context in cycle_plan; depth/vwap in order book; VaR/drawdown in risk; method/confidence in volatility; recommendation in signal_score |
| **Profile** | preferred_tasks, max_latency_ms, timeout_override, regime_weight, slippage_weight, drawdown_max_ratio, metadata |
| **API** | GET /ready, GET /metrics, POST /batch, GET /capabilities, POST /warm |
| **Observability** | correlation_id, latency histograms, error rates, extended ledger, health payload |
| **Config** | task_timeouts, language_overrides, disable_tasks, batch_size, use_regime_estimate, use_slippage_estimate, use_drawdown_check |
| **Aggregation** | regime_consensus, slippage_consensus, sizing_consensus, drawdown_consensus, filter_consensus |
| **Global context** | cycle_id, timestamp_utc, regime, volatility_annual_bps, correlation_id in every `data` |

All of the above can be added so that **every one of the 23 languages** supports the same extended protocol and profile, with strength-based routing and aggregation applied the same way as today.
