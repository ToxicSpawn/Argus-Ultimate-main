#!/usr/bin/env python3
"""Batch 9 — CI Coverage Gate Enforcer.

Runs pytest with coverage over the target packages and fails hard if
any package falls below its configured threshold.

Usage:
    python tools/check_coverage_gate.py                     # full run
    python tools/check_coverage_gate.py --report-only       # just show, never fail
    python tools/check_coverage_gate.py --package core      # single package
    python tools/check_coverage_gate.py --min-coverage 80   # override threshold

Thresholds (per package):
    core/      → 80%
    risk/      → 75%
    utils/     → 70%
    exchanges/ → 70%
    Overall    → 75%

Exit codes:
    0  all thresholds met
    1  one or more packages below threshold
    2  pytest itself failed (test errors / collection errors)
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Per-package minimum coverage %
PACKAGE_THRESHOLDS: dict[str, int] = {
    "core": 80,
    "risk": 75,
    "utils": 70,
    "exchanges": 70,
}
OVERALL_THRESHOLD = 75

# Packages whose absence is non-fatal (may not exist in all environments)
OPTIONAL_PACKAGES: frozenset[str] = frozenset()


@dataclass
class PackageResult:
    package: str
    covered_lines: int
    total_lines: int
    pct: float
    threshold: int
    passed: bool


def _parse_coverage_json(json_path: Path) -> dict[str, PackageResult]:
    """Parse coverage.json produced by pytest-cov --cov-report=json."""
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    files: dict = raw.get("files", {})

    results: dict[str, PackageResult] = {}
    totals_by_pkg: dict[str, list[tuple[int, int]]] = {p: [] for p in PACKAGE_THRESHOLDS}

    for filepath, data in files.items():
        p = Path(filepath)
        # Determine which tracked package this file belongs to
        for pkg in PACKAGE_THRESHOLDS:
            # Match "core/foo.py" or "core\\foo.py" etc.
            if p.parts and p.parts[0] == pkg:
                summary = data.get("summary", {})
                covered = int(summary.get("covered_lines", 0))
                total = int(summary.get("num_statements", 0))
                totals_by_pkg[pkg].append((covered, total))
                break

    for pkg, pairs in totals_by_pkg.items():
        covered = sum(c for c, _ in pairs)
        total = sum(t for _, t in pairs)
        pct = (covered / total * 100) if total else 0.0
        threshold = PACKAGE_THRESHOLDS[pkg]
        results[pkg] = PackageResult(
            package=pkg,
            covered_lines=covered,
            total_lines=total,
            pct=round(pct, 1),
            threshold=threshold,
            passed=pct >= threshold,
        )

    return results


def _run_pytest_with_coverage(packages: list[str]) -> tuple[int, Path]:
    """Run pytest and emit coverage JSON. Returns (returncode, json_path)."""
    json_path = Path(".coverage_gate.json")
    cov_args = []
    for pkg in packages:
        cov_args += [f"--cov={pkg}"]
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/", "tests_unified/",
        "-q", "--tb=no",
        "--cov-report=json:.coverage_gate.json",
        "--no-header",
        "--cov-report=term-missing:skip-covered",
    ] + cov_args
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode, json_path


def _print_table(results: dict[str, PackageResult]) -> None:
    print("\n" + "=" * 58)
    print(f"  {'Package':<12} {'Covered':>8} {'Total':>8} {'Pct':>7} {'Min':>6} {'Status':>8}")
    print("  " + "-" * 54)
    for pkg, r in sorted(results.items()):
        status = "PASS ✅" if r.passed else "FAIL ❌"
        print(f"  {r.package:<12} {r.covered_lines:>8} {r.total_lines:>8} {r.pct:>6.1f}% {r.threshold:>5}% {status:>8}")
    print("=" * 58 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="CI Coverage Gate Enforcer")
    parser.add_argument("--report-only", action="store_true", help="Print report but never exit non-zero")
    parser.add_argument("--package", default=None, help="Check a single package only")
    parser.add_argument(
        "--min-coverage", type=int, default=None,
        help="Override all per-package thresholds with this value"
    )
    parser.add_argument("--skip-pytest", action="store_true",
                        help="Skip running pytest; just re-parse existing .coverage_gate.json")
    args = parser.parse_args()

    # Determine packages to check
    if args.package:
        if args.package not in PACKAGE_THRESHOLDS:
            print(f"Unknown package '{args.package}'. Valid: {list(PACKAGE_THRESHOLDS)}")
            sys.exit(1)
        packages = [args.package]
    else:
        packages = list(PACKAGE_THRESHOLDS)

    # Override thresholds if requested
    thresholds = dict(PACKAGE_THRESHOLDS)
    if args.min_coverage is not None:
        thresholds = {pkg: args.min_coverage for pkg in packages}

    # Run pytest (or skip)
    json_path = Path(".coverage_gate.json")
    if not args.skip_pytest:
        rc, json_path = _run_pytest_with_coverage(packages)
        if rc == 2:  # pytest collection error
            print(f"\npytest exited with code {rc} (collection/import error)")
            sys.exit(2)
    else:
        if not json_path.exists():
            print(".coverage_gate.json not found. Run without --skip-pytest first.")
            sys.exit(1)

    if not json_path.exists():
        print("Coverage JSON not generated. Ensure pytest-cov is installed.")
        sys.exit(2)

    results = _parse_coverage_json(json_path)

    # Apply overrides
    for pkg, r in results.items():
        r.threshold = thresholds.get(pkg, OVERALL_THRESHOLD)
        r.passed = r.pct >= r.threshold

    _print_table(results)

    any_failed = any(not r.passed for r in results.values() if r.package not in OPTIONAL_PACKAGES)

    if any_failed and not args.report_only:
        failing = [r.package for r in results.values() if not r.passed]
        print(f"Coverage gate FAILED for: {', '.join(failing)}")
        print("Increase test coverage or lower thresholds in tools/check_coverage_gate.py")
        sys.exit(1)
    elif any_failed:
        print("Coverage gate would fail (--report-only; no exit 1)")
    else:
        print("Coverage gate PASSED ✅")


if __name__ == "__main__":
    main()
