# Beyond Placeholders (Institutional / Multi-Asset)

Placeholders and references for **spot–perpetual arbitrage**, **market making**, and **cross-asset** when you want to extend the bot. No code is required until you add the integrations.

---

## Spot–perpetual arbitrage

- **Idea:** Trade spot vs perpetual (e.g. Kraken futures); hedge when basis exceeds cost.
- **Config:** Add `kraken_futures` (or perp) credentials when using Kraken; use existing `multi_venue_enabled` for spot + perp as two venues.
- **Ref:** Hummingbot-style arb detection; EXTERNAL_SOURCES_INTEGRATION.md.

---

## Cross-exchange market making

- **Idea:** Two-venue order books; place orders on both; capture spread and skew.
- **Config:** Already have `multi_venue_enabled`, `use_venue_routing_by_spread`; extend with market-making strategy in `strategy_library` or a dedicated module when desired.
- **Ref:** Avellaneda-style inventory skew, reservation price (Hummingbot).

---

## Market making (single-venue)

- **Idea:** Place bid/ask around mid; adjust for inventory and volatility.
- **Config:** Strategy library has placeholder; add logic when venue supports maker rebates and you want MM.
- **Ref:** Avellaneda-Stoikov; EXTERNAL_SOURCES_INTEGRATION.md.

---

## Cross-asset (equities → crypto)

- **Idea:** Use equity indices or macro data as regime/sentiment input for crypto signals.
- **Config:** `correlation_matrix` and multi-symbol setup form the base; add an external feed (e.g. `external_alpha_url` returning regime score) or a small adapter that writes to config each cycle.
- **Ref:** Lead-lag or correlation as macro input; optional data pipeline.

---

## Options flow / volatility surface

- **Idea:** When options data is available, use flow or vol surface for regime/sentiment.
- **Code:** Stub in `ml/options_flow.py`: returns neutral when no data; when `options_flow_data` (or similar) is provided, return a score for strategy engine or regime.
- **Ref:** EVERYTHING_BEYOND_CAPABILITIES.md §12.

---

## Quick reference

| Area            | Placeholder / next step |
|-----------------|-------------------------|
| Spot–perp arb   | Kraken futures + multi_venue; arb detection module. |
| Market making   | Strategy library MM strategy; Avellaneda ref. |
| Cross-asset     | external_alpha or adapter → regime/correlation. |
| Options         | ml/options_flow.py stub; wire when data exists. |

See also: [EVERYTHING_BEYOND_CAPABILITIES.md](EVERYTHING_BEYOND_CAPABILITIES.md), [EXTERNAL_SOURCES_INTEGRATION.md](EXTERNAL_SOURCES_INTEGRATION.md).
