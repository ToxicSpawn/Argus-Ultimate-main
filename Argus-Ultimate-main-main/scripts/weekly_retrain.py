"""
ARGUS Weekly Model Retrainer
============================
Retrains all (or selected) ARGUS ML models. Designed to be run weekly, either
manually or via Windows Task Scheduler.

Usage::

    py scripts/weekly_retrain.py                     # retrain all models
    py scripts/weekly_retrain.py --model rl_agent     # single model
    py scripts/weekly_retrain.py --model tft --gpu    # TFT with GPU hint
    py scripts/weekly_retrain.py --model all --notify # all + Discord summary

Exit codes:
  0 — all selected models succeeded
  1 — one or more models failed
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

KNOWN_MODELS = ("regime", "rl_agent", "tft")

# Subprocess timeouts in seconds
TIMEOUTS: Dict[str, int] = {
    "regime": 600,
    "rl_agent": 3600,
    "tft": 7200,
}

# Output artefacts (best-effort; shown in summary if they exist)
OUTPUT_PATHS: Dict[str, str] = {
    "regime": "models/regime_classifier.pkl",
    "rl_agent": "models/rl_agent.zip",
    "tft": "models/tft_model.pth",
}


# ---------------------------------------------------------------------------
# Config loader (minimal — only needs discord_webhook_url)
# ---------------------------------------------------------------------------

def _load_discord_webhook() -> str:
    """Load discord_webhook_url from unified_config.yaml. Returns '' on failure."""
    try:
        import yaml  # type: ignore
        cfg_path = REPO_ROOT / "unified_config.yaml"
        if not cfg_path.exists():
            return ""
        with cfg_path.open("r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        # Try monitoring.alerts.discord.webhook_url or discord_webhook.url
        monitoring = y.get("monitoring") or {}
        alerts = monitoring.get("alerts") or {}
        discord = alerts.get("discord") or {}
        url = str(discord.get("webhook_url") or "").strip()
        if url:
            return url
        # Fallback: top-level discord_webhook section
        dw = y.get("discord_webhook") or {}
        url = str(dw.get("url") or dw.get("webhook_url") or "").strip()
        if url:
            return url
        # Fallback: env var
        return os.environ.get("DISCORD_WEBHOOK_URL", "")
    except Exception:
        return os.environ.get("DISCORD_WEBHOOK_URL", "")


# ---------------------------------------------------------------------------
# Per-model trainers
# ---------------------------------------------------------------------------

def _train_regime(gpu: bool) -> Dict:
    """Train RegimeClassifier in-process (no subprocess needed — fast fit)."""
    start = time.time()
    status = "ok"
    error_msg = ""
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from ml.regime_classifier import RegimeClassifier  # type: ignore

        clf = RegimeClassifier()

        # Check if we have training data
        data_path = REPO_ROOT / "data" / "regime_training.csv"
        if data_path.exists():
            try:
                import pandas as pd  # type: ignore
                df = pd.read_csv(str(data_path))
                clf.fit(df)
                print(f"  [regime] Fitted on {len(df)} rows from {data_path.name}")
            except Exception as exc:
                print(f"  [regime] fit() failed: {exc} — training data may be malformed")
                status = "failed"
                error_msg = str(exc)
        else:
            print("  [regime] No training data found — skipping fit (model uses online learning)")
            status = "skipped"

    except ImportError as exc:
        print(f"  [regime] Import error: {exc}")
        status = "failed"
        error_msg = str(exc)
    except Exception as exc:
        print(f"  [regime] Unexpected error: {exc}")
        status = "failed"
        error_msg = str(exc)

    duration = time.time() - start
    output_path = str(REPO_ROOT / OUTPUT_PATHS["regime"])
    return {
        "model": "regime",
        "status": status,
        "duration_s": duration,
        "output_path": output_path if Path(output_path).exists() else "n/a",
        "error": error_msg,
    }


def _run_subprocess_model(
    model: str, module: str, gpu: bool, timeout: int
) -> Dict:
    """Run a model training module as a subprocess and return a result dict."""
    start = time.time()
    status = "ok"
    error_msg = ""

    cmd = [sys.executable, "-m", module]
    env = os.environ.copy()
    if gpu:
        env["ARGUS_GPU"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = "0"

    print(f"  [{model}] Running: {' '.join(cmd)} (timeout={timeout}s)")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=False,  # stream to stdout so the user sees progress
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            status = "failed"
            error_msg = f"exit code {result.returncode}"
            print(f"  [{model}] FAILED with exit code {result.returncode}")
        else:
            print(f"  [{model}] Completed successfully")
    except subprocess.TimeoutExpired:
        status = "timeout"
        error_msg = f"timed out after {timeout}s"
        print(f"  [{model}] TIMEOUT after {timeout}s")
    except FileNotFoundError as exc:
        status = "failed"
        error_msg = str(exc)
        print(f"  [{model}] Could not launch subprocess: {exc}")
    except Exception as exc:
        status = "failed"
        error_msg = str(exc)
        print(f"  [{model}] Unexpected error: {exc}")

    duration = time.time() - start
    output_path = str(REPO_ROOT / OUTPUT_PATHS[model])
    return {
        "model": model,
        "status": status,
        "duration_s": duration,
        "output_path": output_path if Path(output_path).exists() else "n/a",
        "error": error_msg,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def train_model(model: str, gpu: bool) -> Dict:
    """Train a single model. Returns result dict. Never raises."""
    print(f"\n{'='*55}")
    print(f"  Training: {model.upper()}")
    print(f"{'='*55}")
    try:
        if model == "regime":
            return _train_regime(gpu=gpu)
        elif model == "rl_agent":
            return _run_subprocess_model(
                "rl_agent", "ml.training.train_rl_agent", gpu=gpu, timeout=TIMEOUTS["rl_agent"]
            )
        elif model == "tft":
            return _run_subprocess_model(
                "tft", "ml.training.train_tft", gpu=gpu, timeout=TIMEOUTS["tft"]
            )
        else:
            return {
                "model": model,
                "status": "unknown",
                "duration_s": 0.0,
                "output_path": "n/a",
                "error": f"Unknown model name: {model}",
            }
    except Exception as exc:
        return {
            "model": model,
            "status": "failed",
            "duration_s": 0.0,
            "output_path": "n/a",
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(results: List[Dict]) -> None:
    print(f"\n{'='*65}")
    print(f"  ARGUS WEEKLY RETRAIN SUMMARY")
    print(f"{'='*65}")
    fmt = "  {:<12}  {:<10}  {:>10}  {}"
    print(fmt.format("Model", "Status", "Duration", "Output"))
    print("  " + "-" * 61)
    for r in results:
        dur = f"{r['duration_s']:.0f}s"
        out = r["output_path"]
        if len(out) > 40:
            out = "..." + out[-37:]
        print(fmt.format(r["model"], r["status"], dur, out))
        if r.get("error"):
            print(f"    ERROR: {r['error']}")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Discord notification
# ---------------------------------------------------------------------------

def _post_discord(webhook_url: str, results: List[Dict]) -> None:
    """POST a summary embed to Discord. Best-effort — never raises."""
    if not webhook_url:
        return

    all_ok = all(r["status"] in ("ok", "skipped") for r in results)
    color = 0x2ECC71 if all_ok else 0xE74C3C  # green / red
    title = "ARGUS Weekly Retrain: OK" if all_ok else "ARGUS Weekly Retrain: FAILURES"

    fields = []
    for r in results:
        emoji = {"ok": "✅", "skipped": "⏭", "failed": "❌", "timeout": "⏱"}.get(r["status"], "❓")
        val = f"{emoji} {r['status']} — {r['duration_s']:.0f}s"
        if r.get("error"):
            val += f"\n`{r['error'][:80]}`"
        fields.append({"name": r["model"].upper(), "value": val, "inline": True})

    payload = {
        "username": "ARGUS Bot",
        "embeds": [
            {
                "title": title,
                "color": color,
                "fields": fields,
                "footer": {"text": "py scripts/weekly_retrain.py"},
            }
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status not in (200, 204):
                print(f"  [discord] Unexpected response: {resp.status}")
            else:
                print("  [discord] Summary posted to Discord.")
    except urllib.error.HTTPError as exc:
        print(f"  [discord] HTTP error {exc.code}: {exc.reason}")
    except Exception as exc:
        print(f"  [discord] Notification failed: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="weekly_retrain",
        description="Retrain ARGUS ML models (regime, rl_agent, tft).",
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=list(KNOWN_MODELS) + ["all"],
        help="Which model to retrain (default: all)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Pass GPU hint to subprocess trainers (sets ARGUS_GPU=1)",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        default=False,
        help="Post completion summary to Discord webhook (from unified_config.yaml or DISCORD_WEBHOOK_URL)",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        default=False,
        help="Register a Windows Task Scheduler job to run weekly_retrain every Sunday at 03:00 (Phase X)",
    )
    parser.add_argument(
        "--unschedule",
        action="store_true",
        default=False,
        help="Remove the ARGUS_Weekly_Retrain scheduled task",
    )
    return parser.parse_args(argv)


def _schedule_task() -> int:
    """Register ARGUS_Weekly_Retrain in Windows Task Scheduler."""
    import shutil
    script_path = os.path.abspath(__file__)
    py_exe = shutil.which("py") or shutil.which("python") or "py"
    cmd = f'"{py_exe}" "{script_path}" --model all --gpu --notify'
    task_name = "ARGUS_Weekly_Retrain"
    try:
        result = subprocess.run(
            [
                "schtasks", "/create",
                "/tn", task_name,
                "/tr", cmd,
                "/sc", "weekly",
                "/d", "SUN",
                "/st", "03:00",
                "/f",  # force overwrite if exists
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"Scheduled task '{task_name}' created: every Sunday at 03:00")
            print(f"  Command: {cmd}")
            return 0
        else:
            print(f"schtasks failed: {result.stderr.strip()}")
            return 1
    except FileNotFoundError:
        print("schtasks not found — this command is Windows-only.")
        print(f"On Linux/Mac, add this to crontab: 0 3 * * 0 {cmd}")
        return 1


def _unschedule_task() -> int:
    """Remove ARGUS_Weekly_Retrain from Windows Task Scheduler."""
    try:
        result = subprocess.run(
            ["schtasks", "/delete", "/tn", "ARGUS_Weekly_Retrain", "/f"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print("Scheduled task 'ARGUS_Weekly_Retrain' removed.")
            return 0
        else:
            print(f"schtasks delete failed: {result.stderr.strip()}")
            return 1
    except FileNotFoundError:
        print("schtasks not found — this command is Windows-only.")
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    if args.schedule:
        return _schedule_task()
    if args.unschedule:
        return _unschedule_task()

    models_to_train: List[str] = (
        list(KNOWN_MODELS) if args.model == "all" else [args.model]
    )

    print(f"ARGUS Weekly Retrain — models: {', '.join(models_to_train)}")
    print(f"GPU hint: {args.gpu}  |  Notify: {args.notify}")

    results: List[Dict] = []
    for model in models_to_train:
        result = train_model(model, gpu=args.gpu)
        results.append(result)

    _print_summary(results)

    if args.notify:
        webhook_url = _load_discord_webhook()
        if webhook_url:
            _post_discord(webhook_url, results)
        else:
            print("  [notify] No Discord webhook configured — skipping notification.")

    failed = [r for r in results if r["status"] == "failed" or r["status"] == "timeout"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
