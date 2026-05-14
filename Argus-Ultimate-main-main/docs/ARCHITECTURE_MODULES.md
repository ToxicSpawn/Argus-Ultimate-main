# Argus Module Architecture & Canonical Ownership

> **Last updated:** 2026-04-16
> **Status:** Authoritative — do not create new top-level packages without updating this map.

This document prevents duplicate-package drift (Issue #18) by declaring the
**canonical** home for every domain. When in doubt, add to the canonical package,
not a new one.

---

## Canonical Packages

| Domain | Canonical Package | Deprecated / Tombstoned |
|---|---|---|
| Core orchestration | `core/` | — |
| Typed domain contracts | `domain/` | `unified_types.py` (legacy, kept for compat) |
| Execution engine | `execution/` | — |
| Risk management | `risk/` | — |
| Market data | `data/` | — |
| AI / ML brain | `ml/` | — |
| Strategy library | `strategies/` | `generated_strategies/` (auto-generated, ephemeral) |
| Alpha signals | `alpha/` | — |
| HFT engine | `hft_engine/` | `hft/` — tombstoned in batch 12 |
| Backtesting | `backtesting/` | `backtest/` — tombstoned in batch 12 |
| Live argus runner | `argus_live/` | `argus/` (legacy root), `argus_omega/` — deprecated batch 15 |
| Quantum | `quantum/` | top-level `quantum_*.py` stubs (kept for import compat) |
| Void Breaker | `void_breaker/` | — |
| Portfolio | `portfolio/` | — |
| Monitoring / metrics | `monitoring/` + `metrics/` | — |
| Infra / ops | `infra/` + `ops/` | — |
| Adaptive learning | `adaptive/` | — |
| Config (runtime) | `config/` | `config.yaml` — deprecated (batch 15 header) |
| CLI | `cli/` | — |
| Utils | `utils/` | `tools/` (maintenance scripts only) |

---

## Addition Protocol

Before creating a new top-level directory:

1. Check this table — does a canonical package already cover this domain?
2. If yes: add a module inside the existing package.
3. If no: add a row to this table **in the same PR** that creates the directory.
4. Never duplicate an existing canonical package with a synonym name
   (e.g. `hft2/`, `new_backtesting/`, `strategies_v2/`).

---

## Tombstoned Packages

Tombstoned packages contain only `__init__.py` with a `DeprecationWarning`
and an import redirect to the canonical location. They exist solely to avoid
breaking old import sites during the migration window.

```python
# Example tombstone pattern (already applied to hft/, backtest/, argus_omega/)
import warnings
warnings.warn(
    "'hft' is deprecated. Use 'hft_engine' instead.",
    DeprecationWarning,
    stacklevel=2,
)
from hft_engine import *  # noqa: F401, F403
```

Tombstoned packages will be **hard-deleted** once CI confirms zero imports
of the old path across all production entry points.
