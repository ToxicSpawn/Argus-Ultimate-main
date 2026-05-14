#!/usr/bin/env python3
"""annotation_scanner.py — find all unannotated functions and generate a coverage report.

Usage:
    python scripts/annotation_scanner.py                  # scan everything
    python scripts/annotation_scanner.py core/ risk/      # scan specific dirs
    python scripts/annotation_scanner.py --json           # output JSON report
    python scripts/annotation_scanner.py --summary        # print summary table
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules"}


@dataclass
class FunctionInfo:
    name: str
    lineno: int
    has_return_type: bool
    missing_param_types: List[str] = field(default_factory=list)


@dataclass
class FileReport:
    path: str
    total_functions: int
    annotated_functions: int
    unannotated: List[FunctionInfo] = field(default_factory=list)

    @property
    def coverage_pct(self) -> float:
        if self.total_functions == 0:
            return 100.0
        return self.annotated_functions / self.total_functions * 100.0


def scan_file(path: Path) -> Optional[FileReport]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return None

    total = 0
    annotated = 0
    unannotated: List[FunctionInfo] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_test_") or node.name == "__init__":
            continue
        total += 1
        has_return = node.returns is not None

        missing_params = []
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            if arg.arg == "self" or arg.arg == "cls":
                continue
            if arg.annotation is None:
                missing_params.append(arg.arg)

        is_annotated = has_return and len(missing_params) == 0
        if is_annotated:
            annotated += 1
        else:
            unannotated.append(FunctionInfo(
                name=node.name,
                lineno=node.lineno,
                has_return_type=has_return,
                missing_param_types=missing_params,
            ))

    return FileReport(
        path=str(path),
        total_functions=total,
        annotated_functions=annotated,
        unannotated=unannotated,
    )


def collect_files(targets: List[str]) -> List[Path]:
    paths: List[Path] = []
    for t in targets:
        p = Path(t)
        if p.is_file() and p.suffix == ".py":
            paths.append(p)
        elif p.is_dir():
            for f in p.rglob("*.py"):
                if not any(skip in f.parts for skip in SKIP_DIRS):
                    paths.append(f)
    return sorted(set(paths))


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan for missing type annotations")
    parser.add_argument("targets", nargs="*", default=["."])
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    parser.add_argument("--summary", action="store_true", help="Print summary table only")
    parser.add_argument("--min-coverage", type=float, default=0.0, help="Fail if overall coverage below this %")
    args = parser.parse_args()

    files = collect_files(args.targets)
    reports: List[FileReport] = []
    for f in files:
        r = scan_file(f)
        if r and r.total_functions > 0:
            reports.append(r)

    reports.sort(key=lambda r: r.coverage_pct)

    total_fns = sum(r.total_functions for r in reports)
    annotated_fns = sum(r.annotated_functions for r in reports)
    overall_pct = (annotated_fns / total_fns * 100.0) if total_fns > 0 else 100.0

    if args.json:
        data = {
            "overall_coverage_pct": round(overall_pct, 1),
            "total_functions": total_fns,
            "annotated_functions": annotated_fns,
            "files": [
                {
                    "path": r.path,
                    "coverage_pct": round(r.coverage_pct, 1),
                    "total": r.total_functions,
                    "annotated": r.annotated_functions,
                    "unannotated": [
                        {"name": f.name, "line": f.lineno, "missing_params": f.missing_param_types}
                        for f in r.unannotated
                    ],
                }
                for r in reports
            ],
        }
        print(json.dumps(data, indent=2))
    else:
        print(f"\n{'File':<60} {'Coverage':>10} {'Total':>7} {'Annotated':>10}")
        print("-" * 90)
        for r in reports[:40]:  # show worst 40
            print(f"{r.path:<60} {r.coverage_pct:>9.1f}% {r.total_functions:>7} {r.annotated_functions:>10}")
        print("-" * 90)
        print(f"{'TOTAL':<60} {overall_pct:>9.1f}% {total_fns:>7} {annotated_fns:>10}")
        print()

    if args.min_coverage > 0 and overall_pct < args.min_coverage:
        print(f"FAIL: coverage {overall_pct:.1f}% < required {args.min_coverage:.1f}%")
        sys.exit(1)


if __name__ == "__main__":
    main()
