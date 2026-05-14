# Argus Ultimate – Upgrade Guide

This document summarizes the **full bot upgrade** to v3.0 and how to get the most from it.

## Version 3.0.0 – Full Bot Upgrade

### What’s New

- **Single version source**  
  Version is defined in `core/version.py` and reported by `python main.py --version` and in system status.

- **Config versioning**  
  `unified_config.yaml` supports optional top-level `config_version: 1` for future schema migrations. The system logs config version at startup.

- **Graceful shutdown**  
  Paper/live runs handle Ctrl+C and task cancellation: the trading loop is cancelled and `shutdown()` is always called so resources and state are cleaned up.

- **CLI entry point**  
  After `pip install -e .`, you can run `argus paper` or `argus live` etc. The `main:cli` entry point forwards to the same logic as `python main.py`.

- **Dependency bumps**  
  Requirements and pyproject use newer compatible versions (numpy 1.26+, pandas 2.2+, scikit-learn 1.5+, scipy 1.12+, prometheus-client 0.20+, rich 13.7+). Python 3.10–3.13 is supported.

- **Startup and status**  
  On startup the system logs Argus version and config version. `get_status()` includes a `version` field for monitoring and dashboards.

### How to Upgrade

1. **Pull the latest code** and ensure you’re on the branch that includes the v3.0 changes.

2. **Reinstall dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
   Or, for editable install with CLI:
   ```bash
   pip install -e .
   ```

3. **Config (optional)**  
   Add at the top of `unified_config.yaml` (after comments):
   ```yaml
   config_version: 1
   ```
   If you omit it, the default is `1`.

4. **Run as before**  
   - `python main.py paper --capital 1000`  
   - `python main.py --version`  
   - With editable install: `argus paper`, `argus --version`

### Integration Recap (from previous work)

The bot is wired so that:

- **Evolved params** load at startup when `evolution.load_evolved` is true.
- **Strategy allocator** ranks signals by PnL and records trades for future ranking.
- **23-language gates** (regime, drawdown, slippage) are driven by config and feed the strategy engine and execution path.

All of this is part of the “entire bot” upgrade and works together in v3.0.

### Troubleshooting

- **ImportError for core.version**  
  Run from the project root so `core` is on the path, or use `pip install -e .`.

- **Old dependency conflicts**  
  Use a fresh venv and `pip install -r requirements.txt` (or `pip install -e .`).

- **Config schema changes later**  
  When we introduce breaking config changes, we’ll bump `config_version` and document migration in this file.
