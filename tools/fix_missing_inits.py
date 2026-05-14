#!/usr/bin/env python3
"""Batch 8 — Generate missing __init__.py files for all Python packages.

Usage:
    python tools/fix_missing_inits.py              # apply
    python tools/fix_missing_inits.py --dry-run    # preview only
    python tools/fix_missing_inits.py --path adaptive  # single package tree

A directory is treated as a package if it:
  - contains at least one .py file, OR
  - contains a subdirectory that already has an __init__.py
and does NOT already have an __init__.py itself.

Generated __init__.py files contain only a module docstring — they do not
import anything so they cannot cause circular-import regressions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Directories we never want to touch
SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".github", ".trae", ".claude",
    "__pycache__", ".mypy_cache", ".pytest_cache",
    "node_modules", "dist", "build", ".venv", "venv", "env",
    "artifacts", "reports", "docs", "grafana", "deploy",
    "data", "generated_strategies",
})


def _is_package_dir(path: Path) -> bool:
    """Return True if *path* looks like a Python package that needs an __init__."""
    if not path.is_dir():
        return False
    if path.name.startswith("."):
        return False
    if path.name in SKIP_DIRS:
        return False
    # Already has one
    if (path / "__init__.py").exists():
        return False
    # Must have at least one .py file (excluding __init__.py)
    py_files = [f for f in path.iterdir() if f.suffix == ".py" and f.name != "__init__.py"]
    return len(py_files) > 0


def _make_docstring(package_path: Path, root: Path) -> str:
    rel = package_path.relative_to(root)
    dotted = str(rel).replace("/", ".")
    return f'"""Package: {dotted}\n\nAuto-generated __init__.py - safe to extend.\n"""\n'


def find_missing(root: Path) -> list[Path]:
    missing: list[Path] = []
    for dirpath in sorted(root.rglob("*")):
        if not dirpath.is_dir():
            continue
        if any(part in SKIP_DIRS for part in dirpath.parts):
            continue
        if _is_package_dir(dirpath):
            missing.append(dirpath)
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate missing __init__.py files.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--path", default=".", help="Root path to scan (default: repo root)")
    args = parser.parse_args()

    root = Path(args.path).resolve()
    missing = find_missing(root)

    if not missing:
        print("No missing __init__.py files found. Nothing to do.")
        sys.exit(0)

    verb = "[DRY-RUN] Would create" if args.dry_run else "Created"
    for pkg in missing:
        init_file = pkg / "__init__.py"
        content = _make_docstring(pkg, root)
        if args.dry_run:
            print(f"{verb} {init_file}")
        else:
            init_file.write_text(content, encoding="utf-8")
            print(f"{verb} {init_file}")

    total = len(missing)
    action = "Would create" if args.dry_run else "Created"
    print(f"\n{action} {total} __init__.py file(s) across {total} package(s).")


if __name__ == "__main__":
    main()
