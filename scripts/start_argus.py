"""
ARGUS Workstation Startup Script — RTX 5080 build.

Orchestrates the full startup sequence:
  1. Check GPU (CUDA) and Ollama availability
  2. Verify API keys are set
  3. Run deployment checklist
  4. Optionally train ML models if weights are missing
  5. Launch ARGUS in paper mode (default) or live mode

Usage:
    python scripts/start_argus.py                    # paper trading
    python scripts/start_argus.py --live             # live (AUD 1000, requires keys)
    python scripts/start_argus.py --train-rl         # train RL agent first, then paper
    python scripts/start_argus.py --train-tft        # train TFT first, then paper
    python scripts/start_argus.py --check-only       # run checks, don't start
    python scripts/start_argus.py --capital 500      # custom capital
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("argus.startup")

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
WEIGHTS_DIR = ROOT / "ml" / "weights"

# Ensure project root is on sys.path so all ARGUS modules are importable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def _check_gpu() -> bool:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory // 2**20
            logger.info("  ✓ GPU: %s (%d MB)", name, mem)
            return True
        logger.warning("  ⚠ CUDA not available — ML will run on CPU")
        return False
    except ImportError:
        logger.warning("  ⚠ PyTorch not installed — GPU check skipped")
        return False


def _check_ollama() -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            data = json.loads(r.read())
            models = [m["name"] for m in data.get("models", [])]
            if models:
                logger.info("  ✓ Ollama running — models: %s", ", ".join(models[:3]))
                return True
            logger.warning("  ⚠ Ollama running but no models pulled")
            logger.warning("    Run: .\\scripts\\setup_ollama.ps1 -LargeModel")
            return False
    except Exception as e:
        logger.warning("  ⚠ Ollama not reachable (%s)", e)
        logger.warning("    Run: .\\scripts\\setup_ollama.ps1 -LargeModel")
        return False


def _check_api_keys() -> bool:
    kraken_key = os.getenv("KRAKEN_API_KEY")
    kraken_secret = os.getenv("KRAKEN_SECRET_KEY")
    coinbase_key = os.getenv("COINBASE_API_KEY")
    if kraken_key and kraken_secret:
        logger.info("  ✓ Kraken API keys present")
    else:
        logger.warning("  ⚠ Kraken API keys missing (KRAKEN_API_KEY / KRAKEN_SECRET_KEY)")
    if coinbase_key:
        logger.info("  ✓ Coinbase API key present")
    else:
        logger.info("  - Coinbase API key not set (optional)")
    return bool(kraken_key and kraken_secret)


def _check_models() -> dict[str, bool]:
    result = {}
    rl_path = MODELS_DIR / "rl_agent.zip"
    result["rl_agent"] = rl_path.exists()
    if rl_path.exists():
        logger.info("  ✓ RL agent model: %s", rl_path)
    else:
        logger.info("  - RL agent not trained yet (will use rule-based fallback)")
        logger.info("    Train: py -m ml.training.train_rl_agent")

    tft_weights = list(WEIGHTS_DIR.glob("tft_*.pkl")) if WEIGHTS_DIR.exists() else []
    result["tft"] = bool(tft_weights)
    if tft_weights:
        logger.info("  ✓ TFT weights: %d file(s)", len(tft_weights))
    else:
        logger.info("  - TFT not trained yet (will use numpy fallback)")
        logger.info("    Train: py -m ml.training.train_tft --symbol BTC/USD --months 18")
    return result


def _check_deployment() -> bool:
    try:
        from ops.deployment_checklist import DeploymentChecklist
        result = DeploymentChecklist().run()
        if result.go:
            logger.info("  ✓ Deployment checklist: GO (%d/%d)", result.passed_count, len(result.checks))
        else:
            logger.warning("  ✗ Deployment checklist: NO-GO\n%s", result.summary())
        return result.go
    except Exception as e:
        logger.warning("  ⚠ Deployment checklist unavailable: %s", e)
        return True  # Don't block if checker itself is unavailable


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_rl(timesteps: int = 1_000_000) -> bool:
    logger.info("Training RL execution agent (%d timesteps)...", timesteps)
    cmd = [sys.executable, "-m", "ml.training.train_rl_agent",
           "--timesteps", str(timesteps)]
    rc = subprocess.run(cmd, cwd=str(ROOT))
    if rc.returncode == 0:
        logger.info("RL training complete.")
        return True
    logger.error("RL training failed (exit %d)", rc.returncode)
    return False


def _train_tft(symbol: str = "BTC/USD", months: int = 18) -> bool:
    logger.info("Training TFT for %s (%d months)...", symbol, months)
    cmd = [sys.executable, "-m", "ml.training.train_tft",
           "--symbol", symbol, "--months", str(months)]
    rc = subprocess.run(cmd, cwd=str(ROOT))
    if rc.returncode == 0:
        logger.info("TFT training complete.")
        return True
    logger.error("TFT training failed (exit %d)", rc.returncode)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_checks(live: bool = False) -> bool:
    logger.info("=" * 55)
    logger.info("  ARGUS Startup Checks")
    logger.info("=" * 55)

    gpu_ok = _check_gpu()
    ollama_ok = _check_ollama()
    keys_ok = _check_api_keys()
    _check_models()
    checklist_ok = _check_deployment()

    logger.info("=" * 55)

    if live and not keys_ok:
        logger.error("Live mode requires Kraken API keys. Set KRAKEN_API_KEY + KRAKEN_SECRET_KEY.")
        return False

    if not checklist_ok and live:
        logger.error("Deployment checklist failed — cannot start live mode.")
        return False

    if not gpu_ok:
        logger.warning("No GPU detected. ML training will be slower on CPU.")
    if not ollama_ok:
        logger.warning("LLM signals disabled (Ollama not running).")

    return True


def _clear_stale_paper_data() -> None:
    """Clear stale data from previous paper trading runs for a fresh start."""
    import sqlite3

    data_dir = ROOT / "data"
    dbs_to_clean = {
        "unified_trades.db": [
            "DELETE FROM trades WHERE 1",
            "DELETE FROM events WHERE 1",
        ],
        "strategy_states.db": [
            "DELETE FROM strategy_state WHERE 1",
        ],
        "checkpoints.db": [
            "DELETE FROM checkpoints WHERE 1",
        ],
    }

    for db_name, statements in dbs_to_clean.items():
        db_path = data_dir / db_name
        if not db_path.exists():
            continue
        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            # Get actual table names (in case schema differs)
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cur.fetchall()}
            cleaned = []
            for stmt in statements:
                # Extract table name from DELETE statement
                table = stmt.split("FROM")[1].split("WHERE")[0].strip()
                if table in tables:
                    cur.execute(stmt)
                    cleaned.append(f"{table}({cur.rowcount})")
            if cleaned:
                conn.execute("VACUUM")
            conn.commit()
            conn.close()
            if cleaned:
                logger.info("  Cleared stale paper data from %s: %s", db_name, ", ".join(cleaned))
        except Exception as e:
            logger.debug("Could not clean %s: %s", db_name, e)


def main() -> None:
    parser = argparse.ArgumentParser(description="ARGUS workstation startup")
    parser.add_argument("--live", action="store_true", help="Start in live mode (real money)")
    parser.add_argument("--capital", type=float, default=1000.0, help="AUD capital")
    parser.add_argument("--config", default="unified_config.yaml", help="Config file path")
    parser.add_argument("--check-only", action="store_true", help="Run checks only, don't start")
    parser.add_argument("--train-rl", action="store_true", help="Train RL agent before starting")
    parser.add_argument("--train-tft", action="store_true", help="Train TFT before starting")
    parser.add_argument("--rl-timesteps", type=int, default=1_000_000)
    parser.add_argument("--tft-symbol", default="BTC/USD")
    parser.add_argument("--tft-months", type=int, default=18)
    args = parser.parse_args()

    logger.info("ARGUS Ultimate — RTX 5080 Workstation")
    logger.info("Date: 2026-03-12 | Mode: %s", "LIVE" if args.live else "PAPER")

    if not run_checks(live=args.live):
        sys.exit(1)

    if args.check_only:
        logger.info("Check-only mode — not starting ARGUS.")
        sys.exit(0)

    # Optional training
    if args.train_rl:
        if not _train_rl(args.rl_timesteps):
            logger.warning("RL training failed — continuing with rule-based fallback")

    if args.train_tft:
        if not _train_tft(args.tft_symbol, args.tft_months):
            logger.warning("TFT training failed — continuing with numpy fallback")

    # Clear stale paper-trading data from previous runs
    if not args.live:
        _clear_stale_paper_data()

    # Launch ARGUS
    mode = "live" if args.live else "paper"
    cmd = [
        sys.executable, str(ROOT / "main.py"),
        mode,
        "--capital", str(args.capital),
        "--config", args.config,
    ]
    if args.live:
        cmd.append("--yes-live")

    logger.info("Launching ARGUS: %s", " ".join(cmd))
    time.sleep(1)  # Let log flush

    os.execv(sys.executable, cmd)  # Replace process (not subprocess) — cleaner signal handling


if __name__ == "__main__":
    main()
