#!/usr/bin/env python3
"""
Readiness score 0-100: config (alerts, live confidence, edge gate) + optional paper stats + priority validation.

Usage: python scripts/readiness_score.py [--config unified_config.yaml] [--include-paper]
Exits 0 and prints score and short report. Use for "am I ready for live?" snapshot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_yaml(path: Path) -> dict:
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute live readiness score 0-100")
    parser.add_argument("--config", default="unified_config.yaml", help="Config path")
    parser.add_argument("--include-paper", action="store_true", help="Add points for paper_results/allocator stats")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / args.config
    cfg = load_yaml(config_path) if config_path.exists() else {}
    score = 0
    max_score = 0
    report = []

    # Config: alerts (20)
    max_score += 20
    alerts = (cfg.get("monitoring") or {}).get("alerts") or {}
    if alerts.get("enabled"):
        score += 10
        report.append("alerts enabled +10")
    if (alerts.get("telegram") or {}).get("enabled") or (alerts.get("email") or {}).get("enabled"):
        score += 10
        report.append("alert channel (telegram/email) +10")
    else:
        report.append("no alert channel 0")

    # Config: live confidence 0.78+ (15)
    max_score += 15
    ai = cfg.get("ai_brain") or {}
    live_conf = ai.get("live_min_signal_confidence")
    if live_conf is not None and float(live_conf) >= 0.78:
        score += 15
        report.append("live_min_signal_confidence >= 0.78 +15")
    else:
        report.append("live_min_signal_confidence < 0.78 or unset 0")

    # Config: edge gate for live (15)
    max_score += 15
    edge = cfg.get("edge_cost_gate") or {}
    modes = edge.get("modes") or []
    if isinstance(modes, list) and "live" in [str(m) for m in modes]:
        score += 15
        report.append("edge_cost_gate.modes includes live +15")
    else:
        report.append("edge gate not for live 0")

    # Config: risk limits set (10)
    max_score += 10
    risk = cfg.get("risk") or {}
    if risk.get("max_daily_loss_pct") is not None and risk.get("max_drawdown_pct") is not None:
        score += 10
        report.append("risk limits set +10")
    else:
        report.append("risk limits missing 0")

    # Optional: paper evidence (20)
    max_score += 20
    if args.include_paper:
        paper_path = repo_root / "data" / "paper_results.json"
        alloc_path = repo_root / "data" / "strategy_allocator_stats.json"
        if paper_path.exists():
            try:
                data = json.loads(paper_path.read_text(encoding="utf-8"))
                trades = int(data.get("trades", 0) or 0)
                if trades >= 10:
                    score += 10
                    report.append("paper_results has >= 10 trades +10")
            except Exception:
                pass
        if alloc_path.exists():
            try:
                data = json.loads(alloc_path.read_text(encoding="utf-8"))
                buckets = data.get("buckets") or data.get("strategies") or {}
                n = sum(1 for _ in (buckets.values() if isinstance(buckets, dict) else []))
                if n >= 1:
                    score += 10
                    report.append("allocator stats present +10")
            except Exception:
                pass
    else:
        report.append("--include-paper not set (max +20 skipped)")

    # Normalize to 0-100 (if we didn't include paper, scale by max possible)
    total_max = 80 if not args.include_paper else 100
    scaled = int(round(score / max_score * total_max)) if max_score else 0
    scaled = min(100, scaled)

    print("Readiness score: %s/100" % scaled)
    for r in report:
        print("  %s" % r)
    if scaled >= 80:
        print("Verdict: Ready for live checklist. Complete docs/LIVE_CHECKLIST.md and run pre_live_check.")
    else:
        print("Verdict: Below 80. Fix config (alerts, live confidence, edge gate, risk). See docs/PRIORITY_ORDER.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
