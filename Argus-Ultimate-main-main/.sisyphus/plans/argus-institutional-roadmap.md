# Argus Institutional Roadmap

## Executive Summary

Making Argus "institutional" means evolving it from a sophisticated retail/professional crypto trading system into one that could satisfy the requirements of a regulated fund manager, prime brokerage client, or compliance-conscious institution.

**Current state**: Argus has strong ML, execution, and risk foundations. What's missing is the institutional layer: cross-asset risk aggregation, FIX connectivity, regulatory-grade reporting, and multi-fund attribution.

**Approach**: Three phases, each independently deployable. Phase 1 delivers immediate value (risk visibility). Phase 2 enables institutional execution. Phase 3 satisfies compliance requirements.

---

## Phase 1: Enterprise Risk

**Goal**: Real-time, cross-strategy risk visibility with institutional-grade VaR/CVaR, stress testing, and risk attribution.

**Dependencies**: None — builds on existing `risk/` and `monitoring/` modules.

### 1.1 Real-Time VaR/CVaR Engine
**File**: `risk/enterprise_var.py`

**What**:
- Historical simulation VaR (configurable lookback: 1d, 7d, 30d)
- Parametric VaR using EWMA covariance matrix
- CVaR (Expected Shortfall) at 95% and 99% confidence
- Cross-asset correlation via rolling covariance
- Intraday calculation on configurable intervals (1min, 5min, 15min)

**Integration points**:
- Reads from `monitoring/trade_ledger.py` for position data
- Feeds into `monitoring/operator_dashboard.py` risk endpoint
- Uses `ml/volatility_estimators.py` for vol inputs

### 1.2 Stress Testing Engine
**File**: `risk/stress_engine.py`

**What**:
- Pre-built historical scenarios: 2008 GFC, March 2020 COVID, Luna/Celsius collapse (May 2022), FTX collapse (Nov 2022), SVB contagion (Mar 2023)
- Custom scenario builder (price shocks, correlation breaks, liquidity droughts)
- Portfolio-level P&L impact per scenario
- Recovery time estimation
- Stress test scheduler (daily, weekly, on-demand)

**Integration points**:
- Uses `risk/advanced_risk_engine.py` for existing risk calculations
- Results stored in `data/stress_tests/` as JSON
- Reportable via `monitoring/operator_dashboard.py`

### 1.3 Risk Attribution System
**File**: `risk/risk_attribution.py`

**What**:
- Marginal contribution to risk (MCTR) per strategy
- Component VaR per asset
- Regime-adjusted risk attribution (different weights in bull/bear/crisis)
- Risk budget tracking vs allocation

**Integration points**:
- Reads from `strategies/unified/strategy_engine.py` for strategy identification
- Uses `ml/volatility_adaptive_drl/regime_detector.py` for regime context
- Outputs to `monitoring/operator_dashboard.py`

### 1.4 Risk Limit Hierarchy
**File**: `risk/risk_limits.py`

**What**:
- Hierarchical limits: Global → Strategy → Asset → Position
- Limit types: notional, VaR, drawdown, concentration, leverage
- Real-time limit utilization monitoring
- Pre-trade limit checks (can this order be placed?)
- Breach handling: warn → reduce → halt (configurable per limit level)

**Integration points**:
- Hooks into `execution/` order submission flow
- Uses `risk/ultimate_defense.py` for circuit breaker escalation
- Config in `unified_config.yaml` under `risk.limits:`

### 1.5 Risk Metrics Dashboard
**File**: `monitoring/risk_dashboard.py`

**What**:
- Real-time risk feed via WebSocket (VaR, CVaR, exposure, utilization)
- Risk utilization gauges (per strategy, per limit)
- Historical risk chart data (30d, 90d, 1y)
- Risk alert integration (Telegram, Discord, PagerDuty)

**Integration points**:
- Extends `monitoring/operator_dashboard.py` with risk endpoints
- Uses `core/broadcast/ws_hub.py` for WebSocket delivery

### Phase 1 Effort Estimate
| Module | Effort | Priority |
|--------|--------|----------|
| enterprise_var.py | 2-3 days | 1 (foundation) |
| stress_engine.py | 2 days | 2 |
| risk_attribution.py | 2 days | 3 |
| risk_limits.py | 2 days | 4 |
| risk_dashboard.py | 1 day | 5 |
| Integration + tests | 2 days | 6 |
| **Total** | **~12 days** | |

---

## Phase 2: Institutional Execution

**Goal**: FIX connectivity, smart order routing, and execution algorithms suitable for institutional counterparties.

**Dependencies**: Phase 1 (risk limits must be in place before multi-venue routing).

### 2.1 FIX 4.4 Protocol Adapter
**File**: `execution/fix_adapter.py`

**What**:
- FIX 4.4 message builder/parser (NewOrderSingle, OrderCancelRequest, ExecutionReport)
- Session management (Logon, Logout, Heartbeat, TestRequest)
- Sequence number management and gap fill
- Pluggable transport layer (TCP, WebSocket)
- Target comp IDs for major crypto prime brokers

**Integration points**:
- Plugs into `execution/` as an additional venue adapter
- Uses `execution/async_order_submitter.py` patterns
- Config in `unified_config.yaml` under `execution.fix:`

### 2.2 Smart Order Router
**File**: `execution/smart_order_router.py`

**What**:
- Multi-venue order decomposition (split large orders across venues)
- Venue selection by: liquidity, fees, latency, fill rate history
- Anti-gaming logic (detect and avoid toxic flow)
- Real-time venue health monitoring
- DEX routing via aggregator protocols (1inch, Paraswap)

**Integration points**:
- Extends `execution/smart_order.py` (existing)
- Uses `monitoring/tca_enhanced.py` for venue quality ranking
- Reads from `risk/risk_limits.py` for position limits

### 2.3 Execution Algorithms
**File**: `execution/algo_orders_enhanced.py`

**What**:
- VWAP (Volume-Weighted Average Price) with historical volume profiles
- TWAP (Time-Weighted Average Price) with randomisation
- POV (Percentage of Volume) — already partially exists in `execution/pov_executor.py`
- Implementation Shortfall — minimise slippage vs decision price
- Adaptive execution — switch algo based on market conditions

**Integration points**:
- Builds on existing `execution/pov_executor.py`
- Uses `hft_engine/order_book_processor.py` for liquidity data
- Config in `unified_config.yaml` under `execution.algos:`

### 2.4 Co-location Awareness
**File**: `execution/latency_model.py`

**What**:
- Round-trip latency estimation per venue
- Network topology awareness (AWS regions, co-lo proximity)
- Latency-adjusted order timing
- Fill probability estimation based on latency

**Integration points**:
- Uses `hft_engine/latency_telemetry.py` (existing)
- Feeds into `execution/smart_order_router.py` for venue selection

### 2.5 Fill Quality Attribution
**File**: `monitoring/fill_quality.py`

**What**:
- Per-fill slippage measurement (vs arrival price, vs mid, vs best)
- Venue quality scoring (fill rate, speed, price improvement)
- Strategy execution quality ranking
- Historical fill quality trends

**Integration points**:
- Reads from `monitoring/trade_ledger.py`
- Uses `monitoring/tca_enhanced.py` (existing TCA)
- Feeds into `monitoring/operator_dashboard.py`

### Phase 2 Effort Estimate
| Module | Effort | Priority |
|--------|--------|----------|
| fix_adapter.py | 3-4 days | 1 |
| smart_order_router.py | 3 days | 2 |
| algo_orders_enhanced.py | 2 days | 3 |
| latency_model.py | 1 day | 4 |
| fill_quality.py | 1 day | 5 |
| Integration + tests | 2 days | 6 |
| **Total** | **~13 days** | |

---

## Phase 3: Compliance & Reporting

**Goal**: Regulatory-grade reporting and audit trail that would satisfy a compliance officer or fund administrator.

**Dependencies**: Phases 1+2 (needs risk data and execution quality data).

### 3.1 Enhanced TCA (Transaction Cost Analysis)
**File**: `monitoring/mifid_tca.py`

**What**:
- MiFID II-style execution quality metrics (price, cost, speed, likelihood)
- Venue comparison reports (which venue gives best fills?)
- Implementation shortfall breakdown (market impact vs timing vs spread)
- Periodic TCA reports (daily, weekly, monthly)

**Integration points**:
- Builds on existing `monitoring/tca_enhanced.py`
- Uses `monitoring/fill_quality.py` from Phase 2
- Report output to `data/reports/tca/`

### 3.2 Best Execution Reporting
**File**: `monitoring/best_execution.py`

**What**:
- Per-order best execution evidence (why this venue, why this algo?)
- RTS 28-style report (top 5 venues by volume)
- Execution policy compliance checking
- Defensible execution decision audit trail

**Integration points**:
- Reads from `execution/smart_order_router.py` decisions
- Uses `monitoring/fill_quality.py` for evidence
- Report output to `data/reports/best_execution/`

### 3.3 Immutable Audit Trail
**File**: `core/audit_trail_enhanced.py`

**What**:
- Event-sourced trade lifecycle (order → fill → settlement)
- Cryptographic hash chaining for tamper detection
- Immutable append-only log (append to file, never modify)
- Query interface for historical reconstruction
- Export for regulatory requests

**Integration points**:
- Extends `monitoring/audit_trail.py` (existing)
- Uses `core/event_store.py` (existing event sourcing)
- Storage in `data/audit/` as append-only JSONL

### 3.4 Multi-Fund P&L Attribution
**File**: `portfolio/fund_attribution.py`

**What**:
- Fund-level P&L aggregation (multiple strategies per fund)
- Strategy contribution to fund returns
- Fee attribution (management fee, performance fee, carry)
- NAV calculation per fund
- Investor-level reporting (capital account statements)

**Integration points**:
- Uses `monitoring/trade_ledger.py` for trade data
- Uses `risk/risk_attribution.py` from Phase 1
- Report output to `data/reports/fund/`

### 3.5 Compliance Report Generator
**File**: `monitoring/compliance_reporter.py`

**What**:
- PDF report generation (monthly/quarterly compliance reports)
- CSV export for fund administrator
- JSON API for automated compliance systems
- Report templates: TCA summary, best execution, risk summary, P&L attribution

**Integration points**:
- Aggregates data from all Phase 1-3 modules
- Uses `monitoring/pdf_reporter.py` (existing PDF generation)
- Report output to `data/reports/compliance/`

### Phase 3 Effort Estimate
| Module | Effort | Priority |
|--------|--------|----------|
| mifid_tca.py | 2 days | 1 |
| best_execution.py | 2 days | 2 |
| audit_trail_enhanced.py | 2 days | 3 |
| fund_attribution.py | 3 days | 4 |
| compliance_reporter.py | 2 days | 5 |
| Integration + tests | 2 days | 6 |
| **Total** | **~13 days** | |

---

## Dependency Graph

```
Phase 1 (Enterprise Risk)
├── enterprise_var.py ─────────────────────────────────┐
├── stress_engine.py ──────────────────────────────────┤
├── risk_attribution.py ───────────────────────────────┤
├── risk_limits.py ────────────────────────────────────┤
└── risk_dashboard.py ─────────────────────────────────┤
                                                       │
Phase 2 (Institutional Execution)                      │
├── fix_adapter.py ────────────────────────────────────┤
├── smart_order_router.py ─────────────────────────────┤──→ Phase 3
├── algo_orders_enhanced.py ───────────────────────────┤
├── latency_model.py ──────────────────────────────────┤
└── fill_quality.py ───────────────────────────────────┤
                                                       │
Phase 3 (Compliance & Reporting)                       │
├── mifid_tca.py ──────────────────────────────────────┤
├── best_execution.py ─────────────────────────────────┤
├── audit_trail_enhanced.py ───────────────────────────┤
├── fund_attribution.py ───────────────────────────────┘
└── compliance_reporter.py
```

**Critical path**: Phase 1 → Phase 2 → Phase 3

**Parallel opportunities within Phase 1**:
- `enterprise_var.py` and `stress_engine.py` can be built in parallel
- `risk_limits.py` depends on `enterprise_var.py` (for VaR-based limits)
- `risk_dashboard.py` is last (needs all risk data sources)

---

## Testing Strategy

### Unit Tests
- Each new module gets `tests/test_<module>.py`
- Mock dependencies using existing `unittest.mock` patterns
- Minimum coverage: 80%

### Integration Tests
- `tests/test_phase1_risk_integration.py` — VaR, stress, limits working together
- `tests/test_phase2_execution_integration.py` — FIX + SOR + algos
- `tests/test_phase3_compliance_integration.py` — Full reporting pipeline

### Regression Tests
- All existing 650+ tests must continue to pass
- New integration tests must not break existing `tests/test_advanced_features.py`

### Manual Verification
- Phase 1: Run stress test scenarios and verify P&L impact matches expectations
- Phase 2: Paper trade with FIX adapter against testnet
- Phase 3: Generate sample compliance report and verify formatting

---

## Migration Path

### Backward Compatibility
- All new modules use `try/except` graceful degradation (existing pattern)
- New config keys added to `unified_config.yaml` with sensible defaults
- Existing `risk/` and `execution/` modules remain unchanged
- New features are opt-in via config flags

### Deployment Order
1. Deploy Phase 1 modules (risk-only, no execution changes)
2. Verify risk metrics in dashboard for 1 week
3. Deploy Phase 2 modules (execution changes, risk limits active)
4. Paper trade with new execution for 1 week
5. Deploy Phase 3 modules (reporting only, no runtime changes)

### Config Migration
```yaml
# New section in unified_config.yaml
enterprise_risk:
  enabled: true
  var_confidence: 0.95
  var_lookback_days: 30
  stress_tests:
    enabled: true
    schedule: "daily"
  limits:
    max_portfolio_var_pct: 2.0
    max_strategy_var_pct: 1.0
    max_position_notional_usd: 100000

institutional_execution:
  fix:
    enabled: false
    host: ""
    port: 0
    sender_comp_id: "ARGUS"
  smart_order_router:
    enabled: true
    min_split_venues: 2
  algos:
    default: "adaptive"
    vwap:
      participation_rate: 0.10

compliance:
  mifid_tca:
    enabled: true
    report_schedule: "weekly"
  audit_trail:
    immutable: true
    retention_days: 2555  # 7 years
```

---

## Quick Wins (What to Implement First)

| Priority | Module | Why |
|----------|--------|-----|
| 1 | `risk/enterprise_var.py` | Immediate risk visibility, foundation for everything else |
| 2 | `risk/risk_limits.py` | Prevents catastrophic losses, hooks into existing execution |
| 3 | `execution/algo_orders_enhanced.py` | Better execution quality, builds on existing POV executor |
| 4 | `monitoring/fill_quality.py` | Easy wins, uses existing TCA data |
| 5 | `monitoring/compliance_reporter.py` | Impressive output, uses existing data sources |

---

## Success Criteria

### Phase 1 Complete
- [ ] Real-time VaR/CVaR displayed in operator dashboard
- [ ] Stress test scenarios produce plausible P&L impacts
- [ ] Risk limits halt trading when breached
- [ ] Risk attribution shows per-strategy contribution

### Phase 2 Complete
- [ ] FIX adapter connects to testnet prime broker
- [ ] Smart order router splits orders across 2+ venues
- [ ] VWAP/TWAP algorithms produce expected fill patterns
- [ ] Fill quality attribution ranks venues by performance

### Phase 3 Complete
- [ ] TCA report generated in MiFID II format
- [ ] Best execution report documents venue selection rationale
- [ ] Audit trail can reconstruct any trade from event log
- [ ] Compliance report PDF generated with all required sections

---

*Plan created: 2026-04-23*
*Total estimated effort: ~38 days*
*Recommended implementation order: Phase 1 → Phase 2 → Phase 3*
