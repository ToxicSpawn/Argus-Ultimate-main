#!/usr/bin/env python3
"""tools/fix_silent_excepts.py — AST-based silent except rewriter.

Scans every .py file in the repo for bare `except:` and `except Exception:`
blocks whose bodies contain no logging call and rewrites them to emit
`logger.exception(...)` so errors are never silently swallowed.

Usage:
    # Preview (no files written)
    python tools/fix_silent_excepts.py --dry-run

    # Apply in-place
    python tools/fix_silent_excepts.py

    # Target a single directory
    python tools/fix_silent_excepts.py --path argus/adaptive

    # Skip certain paths
    python tools/fix_silent_excepts.py --skip tests --skip tools
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOGGER_METHODS = {"debug", "info", "warning", "error", "critical", "exception"}
_DEFAULT_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "archive"}


def _body_has_logging(body: list) -> bool:
    """Return True if the body already contains a logger.* call."""
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in _LOGGER_METHODS:
                return True
            if isinstance(func, ast.Name) and func.id in _LOGGER_METHODS:
                return True
    return False


def _body_is_trivial(body: list) -> bool:
    """Return True if body is pass / ellipsis / bare string (docstring)."""
    if len(body) == 1:
        node = body[0]
        if isinstance(node, ast.Pass):
            return True
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            return True  # pass-through string or ellipsis
    return all(isinstance(n, (ast.Pass,)) for n in body)


def _catch_is_bare_or_exception(handler: ast.ExceptHandler) -> bool:
    """Return True if handler catches bare except or `except Exception`."""
    if handler.type is None:
        return True  # bare `except:`
    if isinstance(handler.type, ast.Name) and handler.type.id == "Exception":
        return True
    return False


def _find_silent_handlers(
    tree: ast.AST,
) -> List[Tuple[ast.ExceptHandler, bool]]:
    """Return list of (handler, is_trivial_body) for silent handlers."""
    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                if not _catch_is_bare_or_exception(handler):
                    continue
                if _body_has_logging(handler.body):
                    continue  # already logs — skip
                results.append((handler, _body_is_trivial(handler.body)))
    return results


def _source_has_logger_import(source: str) -> bool:
    """Check if source already imports logging and defines a logger."""
    return "logging.getLogger" in source


_LOGGER_BOILERPLATE = textwrap.dedent("""\
import logging
logger = logging.getLogger(__name__)
""")


def _inject_logger_if_needed(source: str) -> str:
    if _source_has_logger_import(source):
        return source
    # Inject after module docstring / future imports / encoding lines
    lines = source.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("#", '"""', "'''", "from __future__", "# -*-")):
            insert_at = i + 1
        elif stripped:
            break
    lines.insert(insert_at, _LOGGER_BOILERPLATE)
    return "".join(lines)


def _build_logger_exception_stmt(handler: ast.ExceptHandler, source_lines: List[str]) -> str:
    """Build a logger.exception() call string for this handler."""
    # Derive a human-readable context label from the surrounding code
    lineno = handler.lineno
    context = source_lines[lineno - 1].strip() if lineno <= len(source_lines) else "except block"
    exc_var = handler.name or "exc"
    if handler.name:
        return f'logger.exception("Unexpected error in {context!r}: %s", {exc_var}, exc_info=True)\n'
    return f'logger.exception("Unexpected error", exc_info=True)\n'


def rewrite_source(source: str, *, filename: str = "<unknown>") -> Tuple[str, int]:
    """
    Rewrite source: inject logger.exception into silent except blocks.

    Returns (new_source, count_of_rewrites).
    If no rewrites needed, returns (source, 0).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        logger.debug("Skipping %s (SyntaxError): %s", filename, exc)
        return source, 0

    handlers = _find_silent_handlers(tree)
    if not handlers:
        return source, 0

    source = _inject_logger_if_needed(source)
    lines = source.splitlines(keepends=True)
    rewrites = 0

    # Process handlers in reverse line order so insertions don't shift indices
    handlers_sorted = sorted(handlers, key=lambda h: h[0].lineno, reverse=True)

    for handler, is_trivial in handlers_sorted:
        # Re-parse to get fresh line numbers after each rewrite
        try:
            fresh_tree = ast.parse("".join(lines))
        except SyntaxError:
            break
        fresh_handlers = _find_silent_handlers(fresh_tree)
        if not fresh_handlers:
            break

        # Match by offset (use the first unpatched handler)
        fh, _ = fresh_handlers[0]
        body_start = fh.body[0].lineno  # first line of the handler body
        indent = len(lines[body_start - 1]) - len(lines[body_start - 1].lstrip())
        indent_str = " " * indent

        exc_var = fh.name or "exc"
        if fh.name:
            log_line = f'{indent_str}logger.exception("Unexpected error: %s", {exc_var}, exc_info=True)\n'
        else:
            log_line = f'{indent_str}logger.exception("Unexpected error", exc_info=True)\n'

        if is_trivial:
            # Replace pass/ellipsis with logger call
            lines[body_start - 1] = log_line
        else:
            # Prepend logger call before existing body
            lines.insert(body_start - 1, log_line)

        rewrites += 1

    return "".join(lines), rewrites


def collect_python_files(
    root: Path,
    skip_dirs: Set[str],
    skip_paths: List[str],
) -> List[Path]:
    files = []
    for p in root.rglob("*.py"):
        if any(part in skip_dirs for part in p.parts):
            continue
        if any(skip in str(p) for skip in skip_paths):
            continue
        files.append(p)
    return sorted(files)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rewrite silent except blocks to log errors.")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing files")
    parser.add_argument("--path", default=".", help="Root path to scan (default: repo root)")
    parser.add_argument("--skip", action="append", default=[], help="Skip paths containing this string")
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
    total_rewrites = 0

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
        total_rewrites += count

        rel = fpath.relative_to(root)
        if args.dry_run:
            logger.info("[DRY-RUN] %s — %d silent except(s) would be patched", rel, count)
        else:
            fpath.write_text(new_source, encoding="utf-8")
            logger.info("PATCHED %s — %d silent except(s)", rel, count)

    action = "Would patch" if args.dry_run else "Patched"
    logger.info("%s %d file(s), %d silent except block(s) total.", action, total_files, total_rewrites)
    return 0


if __name__ == "__main__":
    sys.exit(main())
