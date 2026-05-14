# Maintenance Scripts

One-shot utilities for codebase hygiene. Run from the **repo root**.

| Script | Purpose |
|---|---|
| `fix_silent_except.py` | Scans all production `.py` files and replaces silent `except: pass` blocks with `logger.debug()` calls |
| `fix_quick.py` | Same fix but for a single file: `python scripts/maintenance/fix_quick.py <path>` |

## Usage

```bash
# Fix all silent excepts across the codebase
python scripts/maintenance/fix_silent_except.py

# Fix a single file
python scripts/maintenance/fix_quick.py unified_execution_engine.py
```
