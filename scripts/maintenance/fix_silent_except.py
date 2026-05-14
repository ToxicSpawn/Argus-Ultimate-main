"""
Maintenance utility: fix silent except:pass blocks across the codebase.
Adds logger.debug() calls so errors are not silently swallowed.

Usage (from repo root):
    python scripts/maintenance/fix_silent_except.py
"""
import re
import os
import sys

# Always run relative to repo root (two levels up from this script)
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(BASE)

SKIP_DIRS = {
    '.claude', '__pycache__', 'argus_omega', 'hft', 'hft_engine', 'quantum',
    'tests', 'tests_unified', 'scripts', '.git', 'node_modules', 'venv',
    'web', 'docs', 'infra', 'docker', 'config', '.venv', 'archive',
    'build', 'dist', '.mypy_cache', '.pytest_cache', '.ruff_cache',
    'memory', 'models', 'reports', 'logs', 'data'
}


def get_component_name(filepath: str) -> str:
    parts = filepath.replace('\\', '/').replace('./', '').split('/')
    return parts[-1].replace('.py', '') if parts else 'unknown'


def has_logger_setup(content: str) -> bool:
    return bool(re.search(r'logger\s*=\s*logging\.getLogger', content))


def has_logging_import(content: str) -> bool:
    return bool(re.search(r'^import\s+logging\b', content, re.MULTILINE))


def fix_file(filepath: str) -> int:
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except Exception:
        return 0

    component = get_component_name(filepath)
    changes = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^(\s*)except\s*(Exception)?\s*:\s*\n?$', line)
        if m and i + 1 < len(lines):
            next_stripped = lines[i + 1].strip()
            if next_stripped == 'pass':
                indent = m.group(1)
                exc_type = m.group(2) or 'Exception'
                lines[i] = f'{indent}except {exc_type} as _e:\n'
                lines[i + 1] = f'{indent}    logger.debug("{component} error: %s", _e)\n'
                changes += 1
        i += 1

    if changes == 0:
        return 0

    content = ''.join(lines)

    if not has_logging_import(content):
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                lines.insert(idx, 'import logging\n')
                break
        content = ''.join(lines)

    if not has_logger_setup(content):
        last_import = 0
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith('import ') or stripped.startswith('from ')):
                last_import = idx
        insert_at = last_import + 1
        while insert_at < len(lines) and lines[insert_at].strip() == '':
            insert_at += 1
        lines.insert(insert_at, 'logger = logging.getLogger(__name__)\n')
        if insert_at + 1 < len(lines) and lines[insert_at + 1].strip() != '':
            lines.insert(insert_at + 1, '\n')

    try:
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            f.writelines(lines)
    except PermissionError:
        print(f'  SKIP (permission denied): {filepath}')
        return 0

    return changes


TARGET_DIRS = [
    '.', 'adaptive', 'api', 'alpha', 'backtest', 'backtesting', 'cli',
    'compliance', 'core', 'core/connectors', 'core/execution',
    'data', 'data/onchain', 'data/macro', 'data/orderbook',
    'data/sentiment', 'data/defi', 'evolution', 'execution',
    'ml', 'ml/training', 'monitoring', 'ops', 'research',
    'risk', 'risk/position_sizing', 'services', 'strategies',
    'strategies/unified', 'utils',
]


def main() -> None:
    total_changes = 0
    files_changed = 0
    for d in TARGET_DIRS:
        dirpath = os.path.join(BASE, d)
        if not os.path.isdir(dirpath):
            continue
        for f in os.listdir(dirpath):
            if not f.endswith('.py') or f.startswith('test_'):
                continue
            filepath = os.path.join(dirpath, f)
            if not os.path.isfile(filepath):
                continue
            changes = fix_file(filepath)
            if changes > 0:
                total_changes += changes
                files_changed += 1
                print(f'  Fixed {changes:3d} blocks in {os.path.relpath(filepath, BASE)}')
    print(f'\nTotal: {total_changes} silent except:pass blocks fixed in {files_changed} files')


if __name__ == '__main__':
    main()
