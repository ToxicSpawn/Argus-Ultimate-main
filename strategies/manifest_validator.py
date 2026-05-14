"""Argus Ultimate — Strategy MANIFEST.yaml validator.

Validates one or more MANIFEST.yaml files against the Argus strategy manifest
schema.  Can be used as a CLI tool or imported as a library.

Usage::

    # Validate a single file
    python strategies/manifest_validator.py strategies/MANIFEST.yaml

    # Validate all MANIFEST.yaml files recursively under strategies/
    python strategies/manifest_validator.py strategies/

    # Strict mode — fail on warnings too
    python strategies/manifest_validator.py strategies/ --strict

    # JSON output for CI pipelines
    python strategies/manifest_validator.py strategies/ --json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

# Required top-level keys
REQUIRED_KEYS = {"name", "version", "author"}

# Recommended (warn if missing)
RECOMMENDED_KEYS = {
    "description",
    "expected_regime",
    "timeframes",
    "backtest_summary",
    "parameters",
    "risk",
    "tags",
}

# backtest_summary required sub-keys
BACKTEST_REQUIRED = {
    "period",
    "total_return_pct",
    "sharpe_ratio",
    "max_drawdown_pct",
    "win_rate_pct",
    "total_trades",
}

# Version format: simple semver check
import re
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")

# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------


class ValidationResult:
    def __init__(self, path: Path):
        self.path = path
        self.errors: list[str] = []
        self.warnings: list[str] = []

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    @property
    def clean(self) -> bool:
        return self.ok and len(self.warnings) == 0

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def validate_manifest(path: Path) -> ValidationResult:
    """Validate a single MANIFEST.yaml file and return a ValidationResult."""
    result = ValidationResult(path)

    if not path.exists():
        result.error(f"File not found: {path}")
        return result

    # Parse YAML
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        result.error("PyYAML is not installed: pip install pyyaml")
        return result

    try:
        with path.open() as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        result.error(f"YAML parse error: {exc}")
        return result

    if not isinstance(data, dict):
        result.error("MANIFEST.yaml must be a YAML mapping at the top level")
        return result

    # --- Required keys ---
    for key in sorted(REQUIRED_KEYS):
        if key not in data:
            result.error(f"Missing required field: '{key}'")
        elif not data[key]:
            result.error(f"Required field '{key}' must not be empty")

    # --- Version format ---
    if "version" in data:
        if not _SEMVER_RE.match(str(data["version"])):
            result.error(
                f"'version' must be SemVer (e.g. 1.0.0), got: {data['version']!r}"
            )

    # --- Date fields ---
    for date_field in ("created", "updated"):
        if date_field in data and data[date_field]:
            if not _DATE_RE.match(str(data[date_field])):
                result.warn(
                    f"'{date_field}' should be ISO-8601 (YYYY-MM-DD), got: {data[date_field]!r}"
                )

    # --- Recommended keys ---
    for key in sorted(RECOMMENDED_KEYS):
        if key not in data:
            result.warn(f"Recommended field missing: '{key}'")

    # --- backtest_summary sub-keys ---
    if "backtest_summary" in data:
        bs = data["backtest_summary"]
        if isinstance(bs, dict):
            for sub in sorted(BACKTEST_REQUIRED):
                if sub not in bs:
                    result.warn(f"backtest_summary missing recommended field: '{sub}'")
        else:
            result.error("'backtest_summary' must be a mapping")

    # --- parameters consistency ---
    if "parameters" in data:
        params = data["parameters"]
        if isinstance(params, dict):
            for pname, pdef in params.items():
                if not isinstance(pdef, dict):
                    result.warn(f"parameter '{pname}' should be a mapping with type/default")
                    continue
                if "type" not in pdef:
                    result.warn(f"parameter '{pname}' missing 'type'")
                if "default" not in pdef:
                    result.warn(f"parameter '{pname}' missing 'default'")
                ptype = pdef.get("type")
                if ptype == "categorical" and "choices" not in pdef:
                    result.error(
                        f"parameter '{pname}' is categorical but missing 'choices'"
                    )
                if ptype in ("int", "float"):
                    if "low" in pdef and "high" in pdef:
                        if float(pdef["low"]) >= float(pdef["high"]):
                            result.error(
                                f"parameter '{pname}': low ({pdef['low']}) must be < high ({pdef['high']})"
                            )
        else:
            result.error("'parameters' must be a mapping")

    # --- risk profile ---
    if "risk" in data:
        risk = data["risk"]
        if isinstance(risk, dict):
            if "max_drawdown_limit_pct" in risk:
                try:
                    dd = float(risk["max_drawdown_limit_pct"])
                    if dd <= 0 or dd > 100:
                        result.warn(
                            f"risk.max_drawdown_limit_pct should be 0-100, got {dd}"
                        )
                except (TypeError, ValueError):
                    result.error("risk.max_drawdown_limit_pct must be numeric")
            if "leverage" in risk:
                try:
                    lev = float(risk["leverage"])
                    if lev < 1:
                        result.warn(f"risk.leverage < 1 is unusual, got {lev}")
                except (TypeError, ValueError):
                    result.error("risk.leverage must be numeric")
        else:
            result.error("'risk' must be a mapping")

    # --- timeframes ---
    if "timeframes" in data:
        tf = data["timeframes"]
        if isinstance(tf, dict):
            if "primary" not in tf:
                result.warn("timeframes.primary is recommended")
        else:
            result.error("'timeframes' must be a mapping")

    return result


def validate_directory(directory: Path) -> list[ValidationResult]:
    """Find and validate all MANIFEST.yaml files under a directory."""
    manifests = sorted(directory.rglob("MANIFEST.yaml"))
    if not manifests:
        logger.warning("No MANIFEST.yaml files found under %s", directory)
        return []
    return [validate_manifest(m) for m in manifests]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_result(result: ValidationResult, verbose: bool = True) -> None:
    status = "\033[32m✓ OK\033[0m" if result.ok else "\033[31m✗ FAIL\033[0m"
    print(f"{status}  {result.path}")
    if result.errors:
        for e in result.errors:
            print(f"  \033[31m  ERROR\033[0m  {e}")
    if result.warnings and verbose:
        for w in result.warnings:
            print(f"  \033[33m  WARN \033[0m  {w}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Validate Argus strategy MANIFEST.yaml files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python strategies/manifest_validator.py strategies/MANIFEST.yaml\n"
            "  python strategies/manifest_validator.py strategies/\n"
            "  python strategies/manifest_validator.py strategies/ --strict --json"
        ),
    )
    p.add_argument(
        "target",
        type=Path,
        help="Path to a MANIFEST.yaml file or a directory to scan recursively",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (non-zero exit if any warnings)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (useful for CI)",
    )
    p.add_argument(
        "--no-warnings",
        action="store_true",
        help="Suppress warning output",
    )
    args = p.parse_args(argv)

    target: Path = args.target

    if target.is_dir():
        results = validate_directory(target)
        if not results:
            print(f"No MANIFEST.yaml files found under {target}")
            return 0
    elif target.is_file():
        results = [validate_manifest(target)]
    else:
        print(f"ERROR: {target} is not a file or directory", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        for r in results:
            _print_result(r, verbose=not args.no_warnings)
        print()
        total = len(results)
        passed = sum(1 for r in results if r.ok)
        warned = sum(1 for r in results if r.ok and r.warnings)
        failed = total - passed
        print(f"Results: {passed}/{total} passed", end="")
        if warned:
            print(f", {warned} with warnings", end="")
        if failed:
            print(f", {failed} failed", end="")
        print()

    all_ok = all(r.ok for r in results)
    all_clean = all(r.clean for r in results)

    if args.strict:
        return 0 if all_clean else 1
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
