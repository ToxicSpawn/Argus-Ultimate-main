# What Can Be Done to Improve the Quantum Stack

Focused list of **actionable** improvements for the Argus quantum layer: stubs, algorithms, wiring, observability, and optional hardware.

**Implemented:** Classical QMC in `quantum/algorithms/quantum_monte_carlo.py` with richer output (var_95, cvar_95, expected_shortfall_bps, n_samples_used); optional `quantum_amplitude_estimation` path in stub; QuantumWalkLite wired into cycle confidence; `production_simulator_available` in quantum-status. **Quantum bot:** All quantum_bot flags mapped; use_multi_timeframe_confirmation; walk params; full quantum walk when available; annealing selection; VaR position cap; circuit breaker cooldown/threshold; Monte Carlo INFO log and metrics (quantum_var_95, quantum_cvar_95, quantum_circuit_breaker_trips); cloud result used in risk when returned; use_hybrid_switcher wired (QuantumClassicalSwitcher); quantum_improvements_integration placeholder-safe; production simulator docs + use_production + use_quantum_portfolio_weights (optional weights every 10 cycles); self-improver applies use_quantum_monte_carlo_risk and use_quantum_walk with quantum_features_enabled; evolution use_quantum_bot documented; `main.py quantum` and `scripts/start_quantum_bot.py` entrypoints.

---

## 1. Restore or Implement Real Quantum Monte Carlo (High impact)

**Current:** `quantum/algorithms/quantum_monte_carlo.py` is a **placeholder** (raises at runtime). `quantum/quantum_unified_stubs.py` always falls back to **classical** NumPy VaR/CVaR.

**Improvements:**

- **Option A – Classical implementation in `quantum_monte_carlo`:** Implement a `run(returns, n_samples, confidence)` in `quantum/algorithms/quantum_monte_carlo.py` that returns `{"var", "cvar", "from_classical": True}` (same contract as the stub). Then the stub can call it and stop duplicating NumPy logic; you keep one place for tuning (e.g. bootstrap, block size).
- **Option B – Quantum amplitude estimation (when deps available):** Add an optional path that uses Qiskit/PennyLane amplitude estimation for VaR/CVaR when installed, with classical fallback. Document as “quantum-inspired / simulated” unless real backend is connected.
- **Option C – Richer risk output:** Extend the stub (and hook) to return and log e.g. `var_95`, `cvar_95`, `expected_shortfall_bps`, and optionally `n_samples_used`, so monitoring and circuit breaker can use the same numbers.

---

## 2. Wire Quantum Walk Into the Trading Cycle (High impact)

**Current:** `quantum_bot.use_quantum_walk` and `use_multi_timeframe_walk` are in config and mapped in `core/config_manager.py`, but **the main loop never calls `QuantumWalkLite`**. Only `main.py`’s `quantum-status` checks that the module imports.

**Improvements:**

- In the **unified trading system**, in the same place where signal confidence is adjusted (e.g. 23-language consensus boost around line ~2098), when `getattr(self.config, "use_quantum_walk", False)`:
  - Build `returns_history` or `price_history` from the current universe / cycle context (e.g. last N bars per symbol).
  - Call `QuantumWalkLite(...).analyze(returns_history=..., start_symbol=...)` and get `visitation_probabilities`.
  - Use visitation as a **confidence modifier**: e.g. `confidence *= (1.0 + 0.1 * (visitation[symbol] - 1/len(symbols)))` so higher visitation slightly boosts confidence (cap so it stays in [0, 1]).
- **Multi-timeframe:** If `use_multi_timeframe_walk` is true, run the walk on 1h and 1m (or available) returns separately, then combine (e.g. average or min of visitation) before applying to confidence.
- Keep it **quantum-inspired** and lightweight: no heavy Qiskit in the hot path unless `use_full_quantum_walk` is true and a heavy simulator is explicitly enabled.

---

## 3. Production Simulator Integration (Medium impact)

**Current:** `quantum/production_quantum_simulator.py` imports an external package from `ARGUS_QUANTUM_SIMULATOR_PATH` (e.g. `quantum_simulator`, `tensor_network_simulator`). If that path is missing, the module is unused.

**Improvements:**

- Document in `quantum/README.md` or `docs/`: required env var, expected package names, and how to enable “production” mode in config.
- Add an optional **config flag** (e.g. `quantum_simulator.use_production: true`) so the unified system or a dedicated script can call into `production_quantum_simulator` for portfolio optimization or strategy discovery when enabled, without changing code paths when the simulator is absent.
- Optionally expose a **quantum-status** line: `production_simulator_available: true/false` depending on whether the import from that path succeeds.

---

## 4. Stub Consistency and Observability (Medium impact)

- **Return shape:** Ensure `quantum_monte_carlo_risk` always returns the same keys (`var`, `cvar`, and optionally `from_classical` / `from_quantum`) so the risk hook and any dashboard can rely on them.
- **Logging:** Add a **debug** or **info** log line when the Monte Carlo hook runs (e.g. “Quantum Monte Carlo risk: VaR_95=… CVaR_95=…”), and when the circuit breaker is **not** tripped, so operators can confirm the feature is active.
- **Metrics:** If you have a metrics endpoint (e.g. Prometheus), expose `quantum_var_95`, `quantum_cvar_95`, and a counter for `quantum_circuit_breaker_trips` when the breaker is enabled.

---

## 5. Restore Placeholder Modules Only When Needed (Lower priority)

Many files under `quantum/` are **explicit placeholders** (e.g. `enhanced_quantum_system.py`, `quantum_unified.py`, vendor providers, `quantum_monte_carlo`). The bot runs fine without them because stubs fall back to classical or no-op.

**Improvements:**

- **Do not** restore everything at once. Prioritize:
  1. **Quantum Monte Carlo** (see §1) – only one used in the main loop.
  2. **Quantum walk** wiring (see §2) – config already exists; implementation is in-repo (`quantum_walk.py`).
- For **cloud and annealing**: keep stubs; when you want real hardware, restore or implement one provider (e.g. IBM) and wire `QUANTUM_API_KEY` / `IBM_QUANTUM_TOKEN` behind a feature flag.
- **Enhanced quantum system:** `quantum_improvements_integration.py` expects `EnhancedQuantumSystem` from `quantum.enhanced_quantum_system`, which is currently a placeholder. Either restore that module or make the integration optional (try/except and skip if placeholder), so that “integrate enhanced quantum” does not crash.

---

## 6. Transparency and Safety (Already in place – keep it)

- **quantum_simulated_disclosure: true** in config and docs: keep it. Clearly state that quantum is simulated/quantum-inspired unless a real backend is configured.
- **Circuit breaker:** `use_quantum_var_circuit_breaker` is off by default; keep it optional and document that enabling it can trigger emergency stop when drawdown > CVaR_95.
- **quantum-status** command: already reports flags and availability; consider adding one line for “Monte Carlo: classical fallback” vs “Monte Carlo: quantum path” when you have two paths.

---

## 7. Optional: Self-Improvement and Evolution

**Current:** `self_improvement_try_quantum_on_off` and `self_improvement_apply_quantum_choice` let the self-improver toggle “quantum” in shadow tuning.

**Implemented:**

- The self-improver’s “quantum” choice now sets `quantum_features_enabled`, `use_quantum_monte_carlo_risk`, and `use_quantum_walk` together when applying the chosen baseline/candidate, so shadow runs use the same flags as the main loop.
- **Evolution `use_quantum_bot`:** When evolution runs with `use_quantum_bot=True`, it enables the same quantum behaviour as the main bot when `quantum_bot` section is used: Monte Carlo risk and quantum walk (and any other quantum_bot flags in config). Backtest results are therefore comparable: “quantum on” = walk + Monte Carlo risk active (when so configured in YAML).

---

## Summary Table

| Improvement | Impact | Effort | Notes |
|-------------|--------|--------|--------|
| Implement/restore QMC in `quantum/algorithms/quantum_monte_carlo` | High | Low (classical) / Medium (amplitude est.) | Single place for VaR/CVaR; optional quantum path later |
| Wire QuantumWalkLite into cycle confidence | High | Low | Config exists; add one call site and visitation → confidence |
| Production simulator config + docs + status | Medium | Low | Env, flag, one line in quantum-status |
| Stub return shape + logging + metrics | Medium | Low | Consistency and observability |
| Restore other placeholders | Low | High | Only when needed; start with QMC |
| Self-improvement / evolution clarity | Low | Low | Document what “quantum” toggles |

Implementing **§1 (classical QMC in one place)** and **§2 (quantum walk in the cycle)** gives the largest benefit for the least risk; the rest can be done incrementally.

---

## What Else Can Be Added to the Quantum Bot

Concrete additions beyond the items above. The **quantum_bot** section in `unified_config.yaml` already defines several flags; the list below is what can be wired or added next.

### Config flags in YAML but not wired to behaviour

| Flag | Purpose | What to add |
|------|--------|-------------|
| **use_full_quantum_walk** | Use heavy QuantumWalkSimulator (Qiskit/PennyLane) instead of QuantumWalkLite | Map in `config_manager`; in the walk block, when true and simulator importable, call heavy walk and use its visitation for confidence; else keep Lite. |
| **use_hybrid_switcher** | Hybrid quantum–classical switching for signal/risk | Map in config; optionally call `quantum.hybrid_quantum_classical` (or a thin wrapper) to decide whether to use quantum or classical path for risk/signal this cycle; use result to toggle which path runs. |
| **use_cloud_quantum** | Use cloud backend for risk/sizing when API key set | Map in config; in risk hook or sizing, when true and `QUANTUM_API_KEY`/`IBM_QUANTUM_TOKEN` set, call `cloud_quantum_run` with a small risk/sizing problem and use result; fallback to classical on failure. |
| **use_quantum_annealing_selection** | QUBO + annealing for which signals/symbols to take | Map in config; before execution, build a small QUBO (e.g. reward per signal vs risk/correlation), call `quantum_annealing_solve(qubo)`; use solution to filter or rank signals (or select subset of symbols). Stub returns dummy; real D-Wave would need API key. |
| **volatility_adjusted_limits** / **correlation_aware_sizing** (under quantum_bot) | Same as main risk/execution flags | In `config_manager`, when loading quantum_bot section, optionally set `config.use_volatility_adjusted_limits` and `config.use_correlation_aware_sizing` from `quantum_bot.volatility_adjusted_limits` and `quantum_bot.correlation_aware_sizing` so one “quantum bot” block can turn on these existing features for peak mode. |

### Multi-timeframe confirmation (mapped but unused)

- **use_multi_timeframe_confirmation** is already mapped from `quantum_bot` to config but **never read in the main loop**.
- **Add:** When true, after the quantum walk block (or after 1h+1m visitation), require “alignment” before accepting a signal: e.g. same symbol has higher visitation on both 1h and 1m, or same direction (long/short) on 15m and 1h. If not aligned, set confidence to 0 or drop the signal so it is not executed.

### Observability

- **Monte Carlo:** Log at INFO (not only DEBUG) when the risk hook runs and report VaR_95 / CVaR_95; add a one-line “quantum path” vs “classical fallback” in `quantum-status` when the stub has both paths.
- **quantum-status:** List which quantum_bot features are **active** (e.g. “quantum_walk: on”, “quantum_monte_carlo_risk: on”, “circuit_breaker: off”) from current config so operators can confirm without reading YAML.

### New features (not in config yet)

| Feature | Description |
|--------|-------------|
| **Quantum-inspired position cap from VaR** | After the Monte Carlo hook runs, optionally cap position size or max new notional so that estimated tail loss (e.g. from CVaR) does not exceed a fraction of capital. |
| **Quantum portfolio weights** | When `production_quantum_simulator` is available and a flag is set, call it (or QAOA) once per cycle/period to get optimal weights for the current universe and use them in allocation instead of equal weight or a simple heuristic. |
| **Walk parameters in config** | Expose `quantum_walk_steps`, `quantum_walk_damping`, `quantum_walk_correlation_threshold` (and optionally `quantum_walk_boost_cap`) in `quantum_bot` or `quantum_features` so tuning does not require code changes. |
| **start_quantum_bot entrypoint** | Config comments refer to `start_quantum_bot.py` for “peak mode”; that script does not exist. Add a thin script (or `main.py quantum` subcommand) that loads config, forces `paper_trading_peak_mode` and quantum_bot flags on, and runs the same unified loop so “quantum bot” is just a preset. |

### Safety and tuning

- **Circuit breaker:** Optional cooldown (e.g. no retrip for N minutes after a trip) or threshold multiplier (e.g. trip only when drawdown > 1.2 × CVaR_95) so the breaker is less sensitive in volatile regimes.
- **Self-improvement:** Document that `use_quantum_bot` in evolution enables the same flags the main loop uses (walk + Monte Carlo when configured), and ensure the self-improver’s “quantum on/off” toggle writes back to the same config keys.

---

**Summary:** The highest-leverage next steps for the quantum bot are: **wire use_multi_timeframe_confirmation** (align 15m+1h before accept), **map quantum_bot volatility_adjusted_limits / correlation_aware_sizing** so one section drives both, **add observability** (INFO log for Monte Carlo, quantum-status list of active features), and **expose walk parameters** in config. After that, **use_full_quantum_walk**, **use_quantum_annealing_selection**, and **use_cloud_quantum** can be wired one by one when you want those paths.
