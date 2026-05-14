"""Prometheus metric definitions — Regime + Bandit telemetry.
Push 99 — RegimeGrafana.

Import this module once at bot startup to register all metrics;
other modules call the helpers below to record observations.
"""
from __future__ import annotations

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    REGISTRY,
)

# ─── Guards: allow re-import without duplicate-registration errors ────────────
def _gauge(name: str, doc: str, labels: list[str] | None = None) -> Gauge:
    if name in REGISTRY._names_to_collectors:          # type: ignore[attr-defined]
        return REGISTRY._names_to_collectors[name]     # type: ignore[attr-defined]
    return Gauge(name, doc, labels or [])

def _counter(name: str, doc: str, labels: list[str] | None = None) -> Counter:
    if name in REGISTRY._names_to_collectors:          # type: ignore[attr-defined]
        return REGISTRY._names_to_collectors[name]     # type: ignore[attr-defined]
    return Counter(name, doc, labels or [])

def _histogram(name: str, doc: str, labels: list[str] | None = None, buckets: list[float] | None = None) -> Histogram:
    if name in REGISTRY._names_to_collectors:          # type: ignore[attr-defined]
        return REGISTRY._names_to_collectors[name]     # type: ignore[attr-defined]
    kwargs: dict = {}
    if buckets:
        kwargs["buckets"] = buckets
    return Histogram(name, doc, labels or [], **kwargs)


# ─── Regime metrics ───────────────────────────────────────────────────────────
regime_label = _gauge(
    "argus_regime_label",
    "Current market regime encoded as integer (0=RANGING,1=TRENDING,2=VOLATILE,3=CRISIS)",
    ["instance"],
)

regime_confidence = _gauge(
    "argus_regime_confidence",
    "Classifier confidence for the current regime prediction [0,1]",
    ["instance"],
)

regime_volatility_index = _gauge(
    "argus_regime_volatility_index",
    "Rolling normalised volatility index used by regime detector",
    ["instance", "symbol"],
)

regime_transitions_total = _counter(
    "argus_regime_transitions_total",
    "Total number of regime state transitions",
    ["from_regime", "to_regime"],
)


# ─── Bandit metrics ───────────────────────────────────────────────────────────
bandit_arm_pulls_total = _counter(
    "argus_bandit_arm_pulls_total",
    "Total number of times each bandit arm was selected",
    ["arm"],
)

bandit_arm_q_value = _gauge(
    "argus_bandit_arm_q_value",
    "Current Q-value estimate for each bandit arm",
    ["arm"],
)

bandit_epsilon = _gauge(
    "argus_bandit_epsilon",
    "Current epsilon (exploration rate) of the bandit",
    [],
)

bandit_cumulative_regret = _gauge(
    "argus_bandit_cumulative_regret",
    "Cumulative regret accumulated by the bandit policy",
    [],
)


# ─── Latency metrics ──────────────────────────────────────────────────────────
_LATENCY_BUCKETS = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

order_latency_ms = _histogram(
    "argus_order_latency_ms",
    "End-to-end order submission latency in milliseconds",
    ["venue", "side"],
    buckets=_LATENCY_BUCKETS,
)

latency_budget_ms = _gauge(
    "argus_latency_budget_ms",
    "Configured latency budget in milliseconds for the current regime",
    ["regime"],
)


# ─── Convenience helpers ──────────────────────────────────────────────────────
_REGIME_INT = {"RANGING": 0, "TRENDING": 1, "VOLATILE": 2, "CRISIS": 3}
_REGIME_STR = {v: k for k, v in _REGIME_INT.items()}


def record_regime_change(
    instance: str,
    new_regime: str,
    confidence: float,
    old_regime: str = "UNKNOWN",
) -> None:
    """Update all regime gauges and increment transition counter."""
    regime_label.labels(instance=instance).set(_REGIME_INT.get(new_regime, -1))
    regime_confidence.labels(instance=instance).set(confidence)
    if old_regime != "UNKNOWN" and old_regime != new_regime:
        regime_transitions_total.labels(
            from_regime=old_regime, to_regime=new_regime
        ).inc()


def record_bandit_pull(arm: str, q_value: float, epsilon: float, regret_delta: float = 0.0) -> None:
    """Record a single bandit arm pull and update related gauges."""
    bandit_arm_pulls_total.labels(arm=arm).inc()
    bandit_arm_q_value.labels(arm=arm).set(q_value)
    bandit_epsilon.set(epsilon)
    if regret_delta:
        # Increment the cumulative regret gauge (Gauge supports inc())
        bandit_cumulative_regret.inc(regret_delta)  # type: ignore[attr-defined]


def set_latency_budget(regime: str, budget_ms: float) -> None:
    """Publish the current regime-specific latency budget."""
    latency_budget_ms.labels(regime=regime).set(budget_ms)


def observe_order_latency(venue: str, side: str, latency_ms: float) -> None:
    """Record a single order latency observation."""
    order_latency_ms.labels(venue=venue, side=side).observe(latency_ms)
