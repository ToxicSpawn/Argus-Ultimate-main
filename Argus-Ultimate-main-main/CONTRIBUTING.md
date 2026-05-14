# Contributing to Argus Ultimate

Thank you for contributing. This guide covers the workflow, standards, and rules for working on this codebase.

---

## Workflow

1. **Branch from `main`** using the naming convention:
   - `feat/short-description` — new feature
   - `fix/short-description` — bug fix
   - `refactor/short-description` — code restructure
   - `chore/short-description` — tooling, deps, CI
   - `docs/short-description` — documentation only

2. **Keep PRs focused.** One concern per PR. Avoid mixing feature work with refactoring.

3. **Pre-commit hooks must pass** before pushing. Install them once:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

4. **Tests are required** for all trading logic changes. Run:
   ```bash
   pytest tests/ -m "not slow"
   pytest tests_unified/
   ```

5. **PR title format** (conventional commits preferred):
   ```
   feat: add Kelly criterion sizing to HFT engine
   fix: correct pytz version cap in pyproject.toml
   chore: update pre-commit hooks to latest
   ```

---

## Code Style

```bash
# Format
black .
isort .

# Lint
ruff check . --fix

# Type check (core modules)
mypy core/ risk/ execution/ monitoring/ --ignore-missing-imports

# Security scan
bandit -r . --exclude archive,.venv -ll
```

All of the above run automatically via pre-commit. CI will fail if ruff or black report errors.

---

## Architecture Rules

### Entrypoint
- **`main.py` is the canonical entrypoint.** Do not add new root-level `run_*.py` scripts.
  - Use `python main.py paper`, `python main.py live`, `python main.py backtest`.
  - If a new mode is needed, add a subcommand to `main.py`.

### Configuration
- **Edit `config/` subdirectory files** for domain-specific config.
- **Do not add new keys directly to `unified_config.yaml`** without also adding validation in the config loader.
- **Never hardcode exchange-specific symbols, thresholds, or credentials** in source files.

### Risk
- Every strategy that generates trade signals **must call the risk manager** before submitting orders.
- Position limits, stop-loss, and max drawdown checks are **not optional**.

### Async
- All I/O-bound code must be `async`. No blocking calls (`time.sleep`, `requests.get`) in the hot path.
- Use `asyncio.get_running_loop()` or `asyncio.run()`. **Never `asyncio.get_event_loop()`** (deprecated in Python 3.12).

### Security
- **No hardcoded secrets.** All credentials via environment variables or `.env`.
- No committing `.env` files. The `detect-secrets` hook will block this.
- Run `bandit -r . -ll` before submitting a PR if your changes touch auth, database, or network code.

---

## Testing Guidelines

- Unit tests live in `tests/`.
- Integration tests live in `tests_unified/`.
- Mark slow tests with `@pytest.mark.slow`.
- Mark live-exchange tests with `@pytest.mark.live` (never run in CI).
- Mocks for exchange API calls are required. Do not make real network calls in unit tests.

---

## CHANGELOG

Update `CHANGELOG.md` under `[Unreleased]` for any user-visible change. Include:
- What changed
- Why
- Any migration steps for users

---

## Batch Tool Workflow

Argus development is organised into numbered batches (B14, B15, …). Each batch is a set of related improvements that are implemented together in a single working session.

### Batch numbering

Batches are numbered sequentially (B14, B15, …). Tasks within a batch are prefixed with **M** followed by a two-digit number (e.g. M11, M21, M25). The M-number identifies the improvement item; the B-prefix identifies the batch it was shipped in.

When starting a new batch, check `IMPROVEMENTS.md` for the next available M-number and the batch queue.

### Running the fixers

Several automated fixer scripts live in `tools/` and the repo root. Run them in this order when setting up or after large changes:

```bash
# 1. Ensure every package directory has an __init__.py
python tools/fix_missing_inits.py

# 2. Broad "all fixes" runner (format + lint + import sort)
python fix_all.py

# 3. Silent-except fixer (finds bare `except: pass` blocks)
python fix_silent_except.py
```

After running the fixers, verify no new import errors were introduced:

```bash
python -c "import main; print('OK')"
```

### Updating IMPROVEMENTS.md

Every batch task should have an entry in `IMPROVEMENTS.md`. To add one:

1. Open `IMPROVEMENTS.md`.
2. Find the `## Backlog` or `## In Progress` section.
3. Add a line in the format:
   ```
   - **M42** — Short description of the improvement (B15)
   ```
4. Move the item to `## Completed` with its batch label when the work is merged.

### Running the test suite before committing

Always run the full test suite before opening a PR:

```bash
# Fast unit tests only (no network, no live exchange)
pytest tests/ -m "not slow and not live" -q

# Integration tests
pytest tests_unified/ -q

# Everything (slower)
pytest tests/ tests_unified/ -q
```

For targeted runs during development:

```bash
# Run a single file
pytest tests_unified/test_adaptive_coverage.py -v

# Run tests matching a keyword
pytest -k "liquidity" -v

# Show coverage (requires pytest-cov)
pytest tests_unified/ --cov=services --cov=adaptive --cov-report=term-missing
```

Mark tests appropriately:

- `@pytest.mark.slow` — tests that take > 10 s
- `@pytest.mark.live` — tests that require a real exchange connection (never run in CI)
- `@pytest.mark.asyncio` — async tests (requires `pytest-asyncio`)
