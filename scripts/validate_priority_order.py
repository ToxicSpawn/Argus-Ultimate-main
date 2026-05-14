#!/usr/bin/env python3
"""
Validate Priority 1 (and optionally Priority 2) settings from docs/PRIORITY_ORDER.md.

Usage:
  python scripts/validate_priority_order.py --config unified_config.yaml [--strict]
Exits 0 if all checks pass, 1 otherwise. --strict fails if any optional check is missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def load_yaml(path: Path) -> dict:
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate priority-order config (alerts, live confidence, edge gate, etc.)")
    parser.add_argument("--config", default="unified_config.yaml", help="Config file path")
    parser.add_argument("--strict", action="store_true", help="Fail on optional checks too")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / args.config if not Path(args.config).is_absolute() else Path(args.config)
    if not config_path.exists():
        print("ERROR: Config not found:", config_path)
        return 1

    cfg = load_yaml(config_path)
    errors: list[str] = []
    warnings: list[str] = []

    # Priority 1: Alerts
    monitoring = cfg.get("monitoring") or {}
    alerts = monitoring.get("alerts") or {}
    if not alerts.get("enabled", False):
        errors.append("monitoring.alerts.enabled should be true for live (Priority 1)")
    telegram = (alerts.get("telegram") or {}) if isinstance(alerts.get("telegram"), dict) else {}
    if alerts.get("enabled") and not telegram.get("enabled") and not (monitoring.get("alerts") or {}).get("email", {}).get("enabled"):
        warnings.append("No alert channel enabled (telegram or email); set TELEGRAM_BOT_TOKEN and telegram.enabled or email")

    # Priority 1: Live confidence 0.78+
    ai = cfg.get("ai_brain") or {}
    live_conf = ai.get("live_min_signal_confidence")
    if live_conf is not None:
        if float(live_conf) < 0.78:
            errors.append("ai_brain.live_min_signal_confidence should be >= 0.78 for live (got %s)" % live_conf)
    else:
        warnings.append("ai_brain.live_min_signal_confidence not set (recommend 0.78 for live)")

    # Priority 1: Edge gate for live
    edge = cfg.get("edge_cost_gate") or {}
    modes = edge.get("modes") or []
    if isinstance(modes, list) and "live" not in [str(m) for m in modes]:
        errors.append("edge_cost_gate.modes should include 'live' (Priority 1)")
    if isinstance(modes, list) and "live" in [str(m) for m in modes]:
        if edge.get("live_min_edge_pct") is None and edge.get("min_edge_pct") is None:
            warnings.append("edge_cost_gate: set live_min_edge_pct (e.g. 1.0) for live")
        if edge.get("live_buffer_mult") is None and edge.get("buffer_mult") is None:
            warnings.append("edge_cost_gate: set live_buffer_mult (e.g. 2.2) for live")

    # Priority 1: Triggers include circuit_breaker
    triggers = alerts.get("triggers") or []
    trigger_types = [t.get("type") for t in triggers if isinstance(t, dict) and t.get("type")]
    if "circuit_breaker" not in trigger_types:
        warnings.append("monitoring.alerts.triggers should include type: circuit_breaker")

    # Priority 1: Pre-live and health check scripts exist
    pre_live = repo_root / "scripts" / "pre_live_check.py"
    health = repo_root / "scripts" / "health_check.py"
    kill_losers = repo_root / "scripts" / "kill_losers_review.py"
    if not pre_live.exists():
        warnings.append("scripts/pre_live_check.py not found")
    if not health.exists():
        warnings.append("scripts/health_check.py not found")
    if not kill_losers.exists():
        warnings.append("scripts/kill_losers_review.py not found")

    # Priority 2 (optional): IS gate, allocator, evolution load
    exec_eng = cfg.get("execution_engine") or {}
    if args.strict:
        if not exec_eng.get("use_is_gate"):
            warnings.append("execution_engine.use_is_gate not enabled (Priority 2)")
        evo = cfg.get("evolution") or {}
        if not evo.get("load_evolved"):
            warnings.append("evolution.load_evolved not true (Priority 2)")
        sa = cfg.get("strategy_allocator") or {}
        if not sa.get("enabled", True):
            warnings.append("strategy_allocator.enabled should be true (Priority 2)")
        # Priority 3: 23-lang risk gate and timeouts
        ml = cfg.get("multi_language") or {}
        if not ml.get("use_conservative_risk") and not ml.get("use_risk_all"):
            warnings.append("multi_language: enable use_conservative_risk or use_risk_all (Priority 3)")
        tt = ml.get("task_timeouts") or {}
        if not isinstance(tt, dict) or len(tt) < 3:
            warnings.append("multi_language.task_timeouts should define at least 3 task types (Priority 3)")
        risk = cfg.get("risk") or {}
        es = risk.get("emergency_shutdown") or {}
        if not es.get("enabled"):
            warnings.append("risk.emergency_shutdown.enabled should be true (Priority 3)")

    # Report
    for e in errors:
        print("FAIL:", e)
    for w in warnings:
        print("WARN:", w)
    if errors:
        print("\nFix FAIL items and re-run. See docs/PRIORITY_ORDER.md and docs/LIVE_CHECKLIST.md")
        return 1
    if warnings and args.strict:
        print("\nFix WARN items when using --strict. See docs/PRIORITY_ORDER.md")
        return 1
    if warnings:
        print("\nPriority 1 checks passed. Warnings (optional):", len(warnings))
    else:
        print("Priority 1 (and optional) checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
