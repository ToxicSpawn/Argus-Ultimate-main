#!/usr/bin/env python3
"""tools/fix_print_calls.py — AST-based print() → logger.info() rewriter.

Replaces bare `print(...)` calls in production source with `logger.info(...)`
so all output flows through the structured logger.

Usage:
    python tools/fix_print_calls.py --dry-run   # preview
    python tools/fix_print_calls.py              # apply in-place
    python tools/fix_print_calls.py --path argus/adaptive

Skips:
  - tests/, tools/, scripts/, archive/ directories
  - print() calls that use file= or flush= kwargs (intentional stdout)
  - Files that are not valid Python
"""
from __future__ import annotations

import argparse
import ast
import logging
import sys
import textwrap
from pathlib import Path
from typing import List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    "archive", "tests", "tools", "scripts",
}

_LOGGER_BOILERPLATE = textwrap.dedent("""\
import logging
logger = logging.getLogger(__name__)
""")


def _is_intentional_print(call: ast.Call) -> bool:
    """Return True if the print call has file= or flush= kwargs (keep as-is)."""
    for kw in call.keywords:
        if kw.arg in ("file", "flush"):
            return True
    return False


def _has_logger_import(source: str) -> bool:
    return "logging.getLogger" in source


def _inject_logger_if_needed(source: str) -> str:
    if _has_logger_import(source):
        return source
    lines = source.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("#", '"""', "'''", "from __future__")):
            insert_at = i + 1
        elif stripped:
            break
    lines.insert(insert_at, _LOGGER_BOILERPLATE)
    return "".join(lines)


def _collect_print_calls(tree: ast.AST) -> List[ast.Call]:
    calls = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
            and not _is_intentional_print(node)
        ):
            calls.append(node)
    return calls


def rewrite_source(source: str, *, filename: str = "<unknown>") -> Tuple[str, int]:
    """
    Rewrite source: replace print() calls with logger.info().
    Returns (new_source, count).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        logger.debug("Skipping %s (SyntaxError): %s", filename, exc)
        return source, 0

    print_calls = _collect_print_calls(tree)
    if not print_calls:
        return source, 0

    # Simple text substitution: replace `print(` with `logger.info(` on matching lines
    # We do a token-safe replace by working line-by-line and tracking line numbers.
    lines = source.splitlines(keepends=True)
    patched_lines = set(n.lineno for n in print_calls)

    new_lines = []
    count = 0
    for i, line in enumerate(lines, start=1):
        if i in patched_lines:
            # Replace first occurrence of `print(` on this line
            # Handle both `print(` and `print (` (with space)
            import re
            new_line = re.sub(r'\bprint\s*\(', 'logger.info(', line, count=1)
            new_lines.append(new_line)
            count += 1
        else:
            new_lines.append(line)

    new_source = "".join(new_lines)
    new_source = _inject_logger_if_needed(new_source)
    return new_source, count


def collect_python_files(
    root: Path,
    skip_dirs: Set[str],
    extra_skips: List[str],
) -> List[Path]:
    files = []
    for p in root.rglob("*.py"):
        if any(part in skip_dirs for part in p.parts):
            continue
        if any(skip in str(p) for skip in extra_skips):
            continue
        files.append(p)
    return sorted(files)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Replace print() with logger.info() in production code.")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing files")
    parser.add_argument("--path", default=".", help="Root path to scan")
    parser.add_argument("--skip", action="append", default=[], help="Extra skip patterns")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    root = Path(args.path).resolve()
    skip_dirs = _DEFAULT_SKIP_DIRS | set(args.skip)
    files = collect_python_files(root, skip_dirs, args.skip)

    total_files = 0
    total_count = 0

    for fpath in files:
        try:
            source = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.warning("Cannot read %s: %s", fpath, e)
            continue

        new_source, count = rewrite_source(source, filename=str(fpath))
        if count == 0:
            continue

        total_files += 1
        total_count += count
        rel = fpath.relative_to(root)

        if args.dry_run:
            logger.info("[DRY-RUN] %s — %d print() call(s) would be replaced", rel, count)
        else:
            fpath.write_text(new_source, encoding="utf-8")
            logger.info("PATCHED %s — %d print() → logger.info()", rel, count)

    action = "Would replace" if args.dry_run else "Replaced"
    logger.info("%s %d print() call(s) across %d file(s).", action, total_count, total_files)
    return 0


if __name__ == "__main__":
    sys.exit(main())
