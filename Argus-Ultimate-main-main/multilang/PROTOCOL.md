# Multi-Language Service Protocol (23 Languages)

Every language service in the Argus mesh must implement this HTTP API. The orchestrator POSTs to `/execute` with a JSON body and expects a JSON response.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness; return 200 OK. |
| GET | `/ready` | Readiness; 200 when ready to accept work. |
| GET | `/metrics` | JSON: request_count, total_latency_ms, last_latency_ms, error_count, uptime_s. |
| GET | `/capabilities` | JSON: task_types, language, profile. |
| POST | `/execute` | Run one task; body and response below. |
| POST | `/batch` | Body: `{"tasks": [{task_type, data}, ...], "timeout": 2, "correlation_id": "..."}`; returns `{"results": [...], "took_ms": ...}`. |
| POST | `/warm` | Optional warm-up; 200 OK. |

## POST /execute

### Request body

```json
{
  "task_type": "cycle_plan" | "order_book_processing" | "risk_calculation" | "volatility_estimate" | "signal_score" | "regime_estimate" | "slippage_estimate" | "position_sizing" | "drawdown_check" | "correlation_estimate" | "liquidity_score" | "market_impact" | "signal_filter" | "confidence_calibration" | "heartbeat",
  "data": { ... },
  "timeout": 1.0,
  "correlation_id": "optional-trace-id"
}
```

### Response body (success)

```json
{
  "ok": true,
  "result": { ... },
  "took_ms": 0.12
}
```

### Response body (error)

```json
{
  "ok": false,
  "error": "optional message",
  "result": {}
}
```

---

## Task types and payloads

### 1. `cycle_plan`

**Request `data`:**
- `portfolio_value_aud` (number)
- `cash_balance_aud` (number)
- `signals` (integer)
- `primary_exchange` (string)

**Response `result`:**
- `language` (string): e.g. `"rust"`
- `cycle_boost` (number): in range about [-0.015, 0.015]; used for consensus
- `cycle_boost_scale` (number, optional): from language profile
- `ok` (boolean): true

---

### 2. `order_book_processing`

**Request `data`:**
- `bids` (array of [price, size] arrays)
- `asks` (array of [price, size] arrays)

**Response `result`:**
- `spread_bps` (number): (best_ask - best_bid) / mid * 1e4
- `imbalance` (number): (bid_vol - ask_vol) / total for top 5 levels
- `mid` (number, optional)
- `language` (string)
- `spread_mult` (number, optional): from language profile

---

### 3. `risk_calculation`

**Request `data`:**
- `position_value` (number)
- `capital` (number)

**Response `result`:**
- `passed` (boolean): true if position_value / capital <= language's max_ratio
- `exposure_ratio` (number): position_value / capital
- `max_ratio` (number, optional): from language profile
- `language` (string)

---

### 4. `volatility_estimate`

**Request `data`:**
- `prices` or `ohlcv_close` (array of numbers), or
- `returns` (array of numbers)

**Response `result`:**
- `volatility_annual_bps` (number): annualized volatility in basis points
- `volatility_weight` (number, optional): from language profile
- `language` (string)
- `ok` (boolean): true

---

### 5. `signal_score`

**Request `data`:**
- `confidence` (number)
- `score` (number, optional)
- any other signal fields

**Response `result`:**
- `score_delta` (number): small adjustment from this language
- `signal_score_weight` (number, optional): from language profile
- `base_score` (number, optional)
- `language` (string)
- `ok` (boolean): true

---

## Per-language profiles (in-process reference)

The orchestrator uses `LANGUAGE_PROFILES` in `unified_language_orchestrator.py` for in-process fallback. Real services can implement stricter/looser logic per language:

- **risk_max_ratio**: max position/capital (e.g. 0.40 Haskell, 0.48 Rust)
- **cycle_boost_scale**: scale for cycle_boost (e.g. 1.05 R/Julia for stats)
- **volatility_weight**: weight when aggregating volatility (stats languages > 1)
- **signal_score_weight**: weight for signal score delta
- **spread_mult**: multiplier on effective spread (>= 1 = more conservative)
- **role**: speed | correctness | stats | concurrency | ecosystem

Implement each language with its profile behavior for full consistency with the improved in-process logic.

---

## New task types (6–15)

### 6. `regime_estimate`
**Request `data`:** `prices` or `returns`, optional `window`.  
**Response `result`:** `regime` ("trend"|"mean_revert"|"high_vol"), `confidence`, `language`, `regime_weight`, `ok`.

### 7. `slippage_estimate`
**Request `data`:** `side`, `quantity`, `order_book` or `bids`/`asks`, optional `participation_rate`.  
**Response `result`:** `slippage_bps`, `language`, `ok`.

### 8. `position_sizing`
**Request `data`:** `capital`, `volatility_bps` or `volatility_annual_bps`, `confidence`, optional `max_risk_pct`.  
**Response `result`:** `size_pct`, `size_abs`, `language`, `ok`.

### 9. `drawdown_check`
**Request `data`:** `current_equity`, `peak_equity`, optional `max_drawdown_pct`.  
**Response `result`:** `passed`, `current_drawdown_pct`, `language`, `ok`.

### 10. `correlation_estimate`
**Request `data`:** `series_a`/`series_b` or `returns_a`/`returns_b`.  
**Response `result`:** `correlation`, `language`, `ok`.

### 11. `liquidity_score`
**Request `data`:** `bids`, `asks`, optional `depth_levels`.  
**Response `result`:** `liquidity_score` (0–1), `depth_bps`, `language`, `ok`.

### 12. `market_impact`
**Request `data`:** `side`, `quantity`, `adv`, `volatility`.  
**Response `result`:** `impact_bps`, `language`, `ok`.

### 13. `signal_filter`
**Request `data`:** `signal` (object with `confidence`), optional `regime`, `volatility`.  
**Response `result`:** `accept`, `filter_reason`, `language`, `ok`.

### 14. `confidence_calibration`
**Request `data`:** `historical_confidences`, `historical_pnl` (arrays).  
**Response `result`:** `calibrated_confidence`, `language`, `ok`.

### 15. `heartbeat`
**Request `data`:** optional `cycle_id`, `timestamp`.  
**Response `result`:** `ok`, `latency_ms`, `language`, `cycle_id`.
