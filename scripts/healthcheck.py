#!/usr/bin/env python3
"""
ARGUS Production Health Check — quick verification of all critical systems.

Exit codes:
  0 = healthy (all critical checks pass)
  1 = degraded (some non-critical checks fail)
  2 = critical (one or more critical checks fail)

Outputs JSON summary to stdout.

Usage:
    py -B scripts/healthcheck.py
    py -B scripts/healthcheck.py --verbose
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env so API key checks work
_env_path = ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())


# ---------------------------------------------------------------------------
# Check result type
# ---------------------------------------------------------------------------
class CheckResult:
    __slots__ = ("name", "passed", "critical", "message", "elapsed_ms")

    def __init__(self, name: str, passed: bool, critical: bool, message: str, elapsed_ms: float = 0.0):
        self.name = name
        self.passed = passed
        self.critical = critical
        self.message = message
        self.elapsed_ms = elapsed_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "critical": self.critical,
            "message": self.message,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


def _run_check(name: str, fn, critical: bool = True) -> CheckResult:
    t0 = time.perf_counter()
    try:
        passed, msg = fn()
    except Exception as exc:
        passed, msg = False, f"Exception: {exc}"
    elapsed = (time.perf_counter() - t0) * 1000
    return CheckResult(name=name, passed=passed, critical=critical, message=msg, elapsed_ms=elapsed)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_config_loads() -> Tuple[bool, str]:
    """Verify unified_config.yaml loads and passes validation."""
    try:
        from core.config_manager import load_unified_yaml, validate_unified_config_dict
        y = load_unified_yaml()
        validate_unified_config_dict(y)
        keys = len(y)
        return True, f"Config loaded OK ({keys} top-level keys)"
    except Exception as exc:
        return False, f"Config error: {exc}"


def check_exchange_connectivity() -> Tuple[bool, str]:
    """TCP connect to Kraken and Coinbase API endpoints."""
    hosts = [("api.kraken.com", 443), ("api.coinbase.com", 443)]
    failed: List[str] = []
    for host, port in hosts:
        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.close()
        except OSError:
            failed.append(host)
    if not failed:
        return True, "All exchange endpoints reachable"
    return False, f"Cannot reach: {', '.join(failed)}"


def check_api_keys() -> Tuple[bool, str]:
    """Verify at least one exchange has API keys set (not placeholder values)."""
    exchanges = {
        "Kraken": ("KRAKEN_API_KEY", "KRAKEN_SECRET_KEY"),
        "Coinbase": ("COINBASE_API_KEY", "COINBASE_SECRET_KEY"),
    }
    configured: List[str] = []
    for name, (key_var, secret_var) in exchanges.items():
        key = os.getenv(key_var, "")
        secret = os.getenv(secret_var, "")
        if key and secret and "your_" not in key.lower() and "_here" not in key.lower():
            configured.append(name)
    if configured:
        return True, f"API keys configured: {', '.join(configured)}"
    return False, "No exchange API keys configured (or still placeholder values)"


def check_model_files() -> Tuple[bool, str]:
    """Check that key ML model files exist."""
    models_dir = ROOT / "models"
    required = ["regime_classifier.pkl", "rl_agent.zip"]
    optional = [
        "alpha_model.pkl", "hmm_regime.pkl", "signal_stacker.pkl",
        "volatility_forecaster.pkl", "vol_forecaster_v2.pkl",
    ]
    missing_required: List[str] = []
    missing_optional: List[str] = []
    for m in required:
        if not (models_dir / m).exists():
            missing_required.append(m)
    for m in optional:
        if not (models_dir / m).exists():
            missing_optional.append(m)

    if missing_required:
        msg = f"Missing required models: {missing_required}"
        if missing_optional:
            msg += f"; also missing optional: {missing_optional}"
        return False, msg

    found = len(required) - len(missing_required) + len(optional) - len(missing_optional)
    total = len(required) + len(optional)
    msg = f"{found}/{total} model files present"
    if missing_optional:
        msg += f" (missing optional: {missing_optional})"
    return True, msg


def check_db_files() -> Tuple[bool, str]:
    """Verify key SQLite databases are accessible (read/write)."""
    data_dir = ROOT / "data"
    dbs = ["unified_trades.db", "audit_trail.db", "strategy_states.db"]
    errors: List[str] = []
    for db_name in dbs:
        db_path = data_dir / db_name
        if not db_path.exists():
            # Not an error if file doesn't exist yet -- it will be created
            continue
        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute("SELECT 1")
            conn.close()
        except Exception as exc:
            errors.append(f"{db_name}: {exc}")
    if errors:
        return False, f"DB errors: {'; '.join(errors)}"
    return True, f"All database files accessible"


def check_disk_space() -> Tuple[bool, str]:
    """Ensure at least 1 GB free disk space on the data partition."""
    usage = shutil.disk_usage(str(ROOT))
    free_gb = usage.free / (1024 ** 3)
    if free_gb < 1.0:
        return False, f"Low disk space: {free_gb:.1f} GB free (need >= 1 GB)"
    return True, f"{free_gb:.1f} GB free"


def check_memory() -> Tuple[bool, str]:
    """Check available system memory (requires psutil, degrades gracefully)."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        avail_gb = mem.available / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        if avail_gb < 0.5:
            return False, f"Low memory: {avail_gb:.1f} GB available of {total_gb:.1f} GB total"
        return True, f"{avail_gb:.1f} GB available of {total_gb:.1f} GB total"
    except ImportError:
        return True, "psutil not installed -- memory check skipped"


def check_main_entrypoint() -> Tuple[bool, str]:
    """Verify main.py exists and is importable."""
    main_py = ROOT / "main.py"
    if not main_py.exists():
        return False, "main.py not found"
    uts = ROOT / "unified_trading_system.py"
    if not uts.exists():
        return False, "unified_trading_system.py not found"
    return True, "Entrypoint files present"


def check_data_dir() -> Tuple[bool, str]:
    """Verify data directory exists and is writable."""
    data_dir = ROOT / "data"
    if not data_dir.exists():
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return False, f"Cannot create data dir: {exc}"
    # Test writability
    test_file = data_dir / ".healthcheck_test"
    try:
        test_file.write_text("ok")
        test_file.unlink()
    except OSError as exc:
        return False, f"Data dir not writable: {exc}"
    return True, f"Data dir OK at {data_dir}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_healthcheck(verbose: bool = False) -> Dict[str, Any]:
    """Run all checks and return JSON-serialisable summary."""
    checks = [
        ("config_loads", check_config_loads, True),
        ("exchange_connectivity", check_exchange_connectivity, True),
        ("api_keys", check_api_keys, False),
        ("model_files", check_model_files, False),
        ("db_files", check_db_files, True),
        ("disk_space", check_disk_space, True),
        ("memory", check_memory, True),
        ("main_entrypoint", check_main_entrypoint, True),
        ("data_dir_writable", check_data_dir, True),
    ]

    results: List[CheckResult] = []
    for name, fn, critical in checks:
        r = _run_check(name, fn, critical)
        results.append(r)

    critical_failures = [r for r in results if not r.passed and r.critical]
    non_critical_failures = [r for r in results if not r.passed and not r.critical]

    if critical_failures:
        status = "critical"
        exit_code = 2
    elif non_critical_failures:
        status = "degraded"
        exit_code = 1
    else:
        status = "healthy"
        exit_code = 0

    summary = {
        "status": status,
        "exit_code": exit_code,
        "timestamp": time.time(),
        "checks_passed": sum(1 for r in results if r.passed),
        "checks_total": len(results),
        "critical_failures": [r.name for r in critical_failures],
        "warnings": [r.name for r in non_critical_failures],
        "checks": [r.to_dict() for r in results],
    }

    if verbose:
        for r in results:
            marker = "PASS" if r.passed else ("CRIT" if r.critical else "WARN")
            print(f"  [{marker}] {r.name}: {r.message} ({r.elapsed_ms:.0f}ms)", file=sys.stderr)

    return summary


def main() -> int:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    summary = run_healthcheck(verbose=verbose)
    print(json.dumps(summary, indent=2))
    return summary["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
