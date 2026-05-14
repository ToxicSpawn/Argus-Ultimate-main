"""
Maintenance utility: fix silent except:pass in a single file.

Usage (from repo root):
    python scripts/maintenance/fix_quick.py <path/to/file.py>
"""
import os
import sys

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def get_component_name(filepath: str) -> str:
    return os.path.basename(filepath).replace('.py', '')


def fix_file(filepath: str) -> int:
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    lines = content.split('\n')
    component = get_component_name(filepath)
    changes = 0
    i = 0
    while i < len(lines) - 1:
        line = lines[i]
        stripped = line.strip()
        next_stripped = lines[i + 1].strip()
        if next_stripped == 'pass' and (stripped == 'except Exception:' or stripped == 'except:'):
            indent = line[:len(line) - len(line.lstrip())]
            lines[i] = f'{indent}except Exception as _e:'
            lines[i + 1] = f'{indent}    logger.debug("{component} error: %s", _e)'
            changes += 1
        i += 1

    if changes == 0:
        return 0

    new_content = '\n'.join(lines)

    if 'logger = logging.getLogger' not in new_content:
        for idx in range(len(lines) - 1, -1, -1):
            s = lines[idx].strip()
            if (s.startswith('import ') or s.startswith('from ')) and not lines[idx].startswith((' ', '\t')):
                insert_at = idx + 1
                while insert_at < len(lines) and lines[insert_at].strip() == '':
                    insert_at += 1
                lines.insert(insert_at, 'logger = logging.getLogger(__name__)')
                lines.insert(insert_at + 1, '')
                break

    new_content = '\n'.join(lines)
    if 'import logging' not in new_content:
        for idx in range(len(lines)):
            s = lines[idx].strip()
            if s.startswith('import ') or s.startswith('from '):
                lines.insert(idx, 'import logging')
                break

    try:
        with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(lines))
    except PermissionError:
        print(f'SKIP (permission denied): {filepath}')
        return 0

    return changes


if __name__ == '__main__':
    if len(sys.argv) > 1:
        fp = sys.argv[1]
        c = fix_file(fp)
        print(f'Fixed {c} blocks in {fp}')
    else:
        print('Usage: python scripts/maintenance/fix_quick.py <filepath>')
