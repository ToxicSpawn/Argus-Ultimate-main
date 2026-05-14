#!/usr/bin/env python3
"""
Pre-live checklist script: config, .env, alerts, risk limits.
Usage: python scripts/pre_live_check.py [--config unified_config.yaml]
Exit 0 if all checks pass, 1 otherwise.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo-root imports (e.g. monitoring.*) resolve when invoked as a script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_yaml(path: Path, *, profile: str | None = None) -> dict:
    try:
        from core.config_manager import load_unified_yaml

        data = load_unified_yaml(str(path), profile=profile)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"ERROR: Could not load config: {e}")
        return {}


def attempt_refresh_soak_gate(repo_root: Path, *, config_arg: str, profile: str | None) -> bool:
    script = repo_root / "scripts" / "soak_gate.py"
    if not script.exists():
        print(f"WARN: soak gate refresh skipped (missing script: {script})")
        return False
    cmd = [sys.executable, str(script), "--config", str(config_arg)]
    if profile:
        cmd.extend(["--profile", str(profile)])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except Exception as e:
        print(f"WARN: soak gate refresh failed to execute: {e}")
        return False
    if proc.returncode == 0:
        print("PASS: Soak gate report refreshed")
        return True
    tail = ((proc.stdout or "") + (proc.stderr or ""))[-400:]
    print(f"WARN: soak gate refresh returned non-zero ({proc.returncode}): {tail}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-live checklist")
    parser.add_argument("--config", default="unified_config.yaml", help="Config file path")
    parser.add_argument("--profile", default=None, help="Config profile name or YAML path overlay")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / args.config
    env_path = repo_root / ".env"

    failed = 0

    # Config exists
    if not config_path.exists():
        print("FAIL: Config file not found:", config_path)
        failed += 1
    else:
        print("PASS: Config file exists")

    cfg = load_yaml(config_path, profile=args.profile) if config_path.exists() else {}

    # .env exists (recommended for live)
    if not env_path.exists():
        print("WARN: .env not found. For live you need KRAKEN_API_KEY and KRAKEN_SECRET_KEY in .env")
    else:
        print("PASS: .env exists")
        env_content = env_path.read_text()
        if "KRAKEN_API_KEY" not in env_content or "KRAKEN_SECRET" not in env_content:
            print("WARN: .env may be missing KRAKEN_API_KEY or KRAKEN_SECRET_KEY")
        else:
            print("PASS: .env appears to have Kraken keys (values not checked)")

    # Alerts enabled
    alerts = (cfg.get("monitoring") or {}).get("alerts") or {}
    if not alerts.get("enabled"):
        print("FAIL: monitoring.alerts.enabled should be true for live")
        failed += 1
    else:
        print("PASS: Alerts enabled")
    telegram = (alerts.get("telegram") or {})
    if telegram.get("enabled") and (not os.environ.get("TELEGRAM_BOT_TOKEN") and not (repo_root / ".env").exists()):
        print("WARN: Telegram enabled but TELEGRAM_BOT_TOKEN not set (set in .env)")
    elif telegram.get("enabled"):
        print("PASS: Telegram alert channel configured (or .env present)")

    # Risk limits present
    risk = cfg.get("risk") or {}
    if not risk.get("max_daily_loss_pct"):
        print("WARN: risk.max_daily_loss_pct not set")
    else:
        print("PASS: risk.max_daily_loss_pct set")
    if not risk.get("max_drawdown_pct"):
        print("WARN: risk.max_drawdown_pct not set")
    else:
        print("PASS: risk.max_drawdown_pct set")

    # Confidence for live
    ai = cfg.get("ai_brain") or {}
    conf = ai.get("min_signal_confidence")
    if conf is not None and conf < 0.65:
        print("WARN: ai_brain.min_signal_confidence < 0.65; consider 0.72+ for live")
    elif conf is not None:
        print("PASS: min_signal_confidence acceptable for live")

    # Edge-cost gate includes live
    edge = cfg.get("edge_cost_gate") or {}
    modes = edge.get("modes") or []
    if isinstance(modes, list) and "live" not in modes:
        print("WARN: edge_cost_gate.modes should include 'live' for production")
    else:
        print("PASS: edge_cost_gate.modes includes live (or not applicable)")

    # Edge validation gate: require paper evidence before live (10/10 safety)
    runtime = cfg.get("runtime") or {}
    if runtime.get("live_require_paper_edge"):
        min_trades = int(runtime.get("live_min_trades_paper", 20) or 20)
        min_win_rate = float(runtime.get("live_min_win_rate_pct", 45) or 45) / 100.0
        paper_path = repo_root / "data" / "paper_results.json"
        alloc_path = repo_root / "data" / "strategy_allocator_stats.json"
        edge_ok = False
        fail_reasons: list[str] = []
        if paper_path.exists():
            try:
                import json as _json
                data = _json.loads(paper_path.read_text(encoding="utf-8"))
                # Gate uses closed trades when available (wins + losses / closed_trades).
                trades = int(data.get("trades", 0) or 0)
                wins = int(data.get("wins", data.get("winning_trades", 0)) or 0)
                losses = int(data.get("losses", data.get("losing_trades", 0)) or 0)
                closed_trades = int(data.get("closed_trades", 0) or 0)
                if closed_trades <= 0 and (wins + losses) > 0:
                    closed_trades = wins + losses
                eval_trades = closed_trades if closed_trades > 0 else trades
                eval_wr = (wins / max(eval_trades, 1))
                if eval_trades >= min_trades and eval_wr >= min_win_rate:
                    edge_ok = True
                else:
                    unit = "closed trades" if closed_trades > 0 else "trades"
                    fail_reasons.append(
                        "paper has %s %s (need %s) and win rate %.1f%% (need %.0f%%)"
                        % (eval_trades, unit, min_trades, 100 * eval_wr, min_win_rate * 100)
                    )
            except Exception as e:
                print("WARN: Could not read paper_results.json for edge gate:", e)
        if not edge_ok and alloc_path.exists():
            try:
                import json as _json
                data = _json.loads(alloc_path.read_text(encoding="utf-8"))
                buckets = data.get("buckets") or data.get("strategies") or {}
                total_t, total_w = 0, 0
                for v in buckets.values() if isinstance(buckets, dict) else []:
                    if isinstance(v, dict):
                        total_t += int(v.get("trades", v.get("n_trades", 0)) or 0)
                        total_w += int(v.get("wins", 0) or 0)
                if total_t >= min_trades and (total_w / max(total_t, 1)) >= min_win_rate:
                    edge_ok = True
                else:
                    fail_reasons.append(
                        "allocator has %s trades (need %s), win rate %.1f%% (need %.0f%%)"
                        % (total_t, min_trades, 100 * total_w / max(total_t, 1), min_win_rate * 100)
                    )
            except Exception:
                pass
        if runtime.get("live_require_paper_edge") and not edge_ok and not (paper_path.exists() or alloc_path.exists()):
            print("FAIL: live_require_paper_edge but no paper_results.json or strategy_allocator_stats.json (run paper first)")
            failed += 1
        elif edge_ok:
            print("PASS: Paper edge gate met (min_trades / min_win_rate)")
        else:
            msg = " | ".join(fail_reasons) if fail_reasons else "insufficient qualifying paper evidence"
            print(f"FAIL: live_require_paper_edge: {msg}")
            failed += 1

    # Soak promotion gate: require latest PASS report when enabled
    soak_gate = runtime.get("soak_gate") or {}
    if bool(soak_gate.get("enabled", True)):
        attempt_refresh_soak_gate(repo_root, config_arg=str(args.config), profile=args.profile)
        report_path = repo_root / str(soak_gate.get("report_path", "reports/soak_gate_latest.json"))
        max_age_h = float(soak_gate.get("max_age_hours", 24.0) or 24.0)
        if not report_path.exists():
            print(f"FAIL: soak gate enabled but report not found: {report_path}")
            failed += 1
        else:
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                soak_ok = True
                status = str(report.get("status", "")).upper()
                if status != "PASS":
                    print(f"FAIL: soak gate status is {status or 'UNKNOWN'}")
                    failed += 1
                    soak_ok = False
                from monitoring.soak_gate import load_thresholds_from_runtime

                t = load_thresholds_from_runtime({"soak_gate": soak_gate})
                expected_thresholds = {
                    "min_duration_seconds": float(t.min_duration_seconds),
                    "min_decision_count": int(t.min_decision_count),
                    "max_error_rate": float(t.max_error_rate),
                    "max_timeout_rate": float(t.max_timeout_rate),
                    "max_reconciliation_halts": int(t.max_reconciliation_halts),
                    "max_duplicate_intents": int(t.max_duplicate_intents),
                }
                if t.min_trade_count is not None:
                    expected_thresholds["min_trade_count"] = int(t.min_trade_count)
                if t.max_drawdown_pct is not None:
                    expected_thresholds["max_drawdown_pct"] = float(t.max_drawdown_pct)
                if t.max_cycle_latency_p90_ms is not None:
                    expected_thresholds["max_cycle_latency_p90_ms"] = float(t.max_cycle_latency_p90_ms)
                reported_thresholds = dict(report.get("thresholds") or {})
                int_keys = {"min_decision_count", "max_reconciliation_halts", "max_duplicate_intents", "min_trade_count"}
                for key, expected in expected_thresholds.items():
                    got = reported_thresholds.get(key)
                    if got is None:
                        print(f"FAIL: soak gate report missing threshold '{key}'")
                        failed += 1
                        soak_ok = False
                        continue
                    if key in int_keys:
                        if int(got) != expected:
                            print(f"FAIL: soak gate threshold mismatch for {key}: got {got}, expected {expected}")
                            failed += 1
                            soak_ok = False
                    else:
                        if abs(float(got) - float(expected)) > 1e-9:
                            print(f"FAIL: soak gate threshold mismatch for {key}: got {got}, expected {expected}")
                            failed += 1
                            soak_ok = False
                checked_at = str(report.get("checked_at") or "")
                if not checked_at:
                    print("FAIL: soak gate report missing checked_at")
                    failed += 1
                    soak_ok = False
                else:
                    dt = datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
                    age_h = (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 3600.0
                    if age_h > max_age_h:
                        print(f"FAIL: soak gate report is stale ({age_h:.2f}h > {max_age_h:.2f}h)")
                        failed += 1
                        soak_ok = False
                    elif soak_ok:
                        print("PASS: Soak gate status PASS and fresh")
            except Exception as e:
                print(f"FAIL: could not parse soak gate report: {e}")
                failed += 1

    # Infrastructure preflight gate: require fresh PASS from host determinism checks.
    infra_gate = runtime.get("infra_preflight") or {}
    if bool(infra_gate.get("enabled", False)):
        report_path = repo_root / str(infra_gate.get("report_path", "reports/infra/infra_preflight_latest.json"))
        max_age_h = float(infra_gate.get("max_age_hours", 24.0) or 24.0)
        if not report_path.exists():
            print(f"FAIL: infra preflight enabled but report not found: {report_path}")
            failed += 1
        else:
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                status = str(report.get("status", "")).upper()
                if status != "PASS":
                    reasons = report.get("fail_reasons") or []
                    detail = "; ".join(str(x) for x in reasons) if isinstance(reasons, list) and reasons else "unknown"
                    print(f"FAIL: infra preflight status is {status or 'UNKNOWN'} ({detail})")
                    failed += 1
                checked_at = str(report.get("checked_at") or "")
                if not checked_at:
                    print("FAIL: infra preflight report missing checked_at")
                    failed += 1
                else:
                    dt = datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
                    age_h = (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 3600.0
                    if age_h > max_age_h:
                        print(f"FAIL: infra preflight report is stale ({age_h:.2f}h > {max_age_h:.2f}h)")
                        failed += 1
                    elif status == "PASS":
                        print("PASS: Infra preflight status PASS and fresh")
            except Exception as e:
                print(f"FAIL: could not parse infra preflight report: {e}")
                failed += 1

    # Walk-forward promotion gate: require positive walk-forward summary when enabled.
    walk_forward_gate = runtime.get("walk_forward_gate") or {}
    if bool(walk_forward_gate.get("enabled", False)):
        wf_path = repo_root / str(walk_forward_gate.get("report_path", "reports/walk_forward_latest.json"))
        max_age_h = float(walk_forward_gate.get("max_age_hours", 168.0) or 168.0)
        min_avg_return_pct = float(walk_forward_gate.get("min_avg_return_pct", 0.0) or 0.0)
        min_avg_sharpe = float(walk_forward_gate.get("min_avg_sharpe", 0.0) or 0.0)
        if not wf_path.exists():
            print(f"FAIL: walk-forward gate enabled but report not found: {wf_path}")
            failed += 1
        else:
            try:
                report = json.loads(wf_path.read_text(encoding="utf-8"))
                summary = dict(report.get("summary") or {})
                avg_return = float(summary.get("avg_return_pct", 0.0) or 0.0)
                avg_sharpe = float(summary.get("avg_sharpe", 0.0) or 0.0)
                if avg_return < min_avg_return_pct:
                    print(
                        "FAIL: walk-forward avg_return_pct %.4f < required %.4f"
                        % (avg_return, min_avg_return_pct)
                    )
                    failed += 1
                if avg_sharpe < min_avg_sharpe:
                    print(
                        "FAIL: walk-forward avg_sharpe %.4f < required %.4f"
                        % (avg_sharpe, min_avg_sharpe)
                    )
                    failed += 1
                stamp = str(report.get("checked_at") or report.get("generated_at") or report.get("timestamp") or "")
                if stamp:
                    if "T" in stamp:
                        dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                    else:
                        dt = datetime.strptime(stamp, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
                    age_h = (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 3600.0
                    if age_h > max_age_h:
                        print("FAIL: walk-forward report is stale (%.2fh > %.2fh)" % (age_h, max_age_h))
                        failed += 1
                    else:
                        print("PASS: Walk-forward report is fresh")
            except Exception as e:
                print(f"FAIL: could not parse walk-forward report: {e}")
                failed += 1

    # Reconciliation ownership gate: freeze must be acknowledged before live.
    reconciliation = cfg.get("reconciliation") or {}
    if bool(reconciliation.get("require_operator_ack", True)):
        freeze_path = repo_root / str(reconciliation.get("freeze_file", "data/reconciliation_freeze.json"))
        if freeze_path.exists():
            try:
                freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
                if isinstance(freeze, dict):
                    active = bool(freeze.get("active", True))
                    acknowledged = bool(freeze.get("acknowledged", False))
                    if active and not acknowledged:
                        print(f"FAIL: reconciliation freeze is active and unacknowledged: {freeze_path}")
                        print("      Run: python main.py reconcile-ack --config unified_config.yaml --operator-id <name>")
                        failed += 1
                    else:
                        print("PASS: Reconciliation freeze not blocking live")
                else:
                    print(f"FAIL: reconciliation freeze file malformed (not object): {freeze_path}")
                    failed += 1
            except Exception as e:
                print(f"FAIL: could not parse reconciliation freeze file: {e}")
                failed += 1
        else:
            print("PASS: No active reconciliation freeze file")

    if failed > 0:
        print("\nSome checks failed. Fix before going live. See docs/LIVE_CHECKLIST.md")
        return 1
    print("\nPre-live checks passed. Still complete docs/LIVE_CHECKLIST.md before live.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
