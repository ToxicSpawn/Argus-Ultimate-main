#!/usr/bin/env python3
"""
Start the unified trading system in quantum bot mode (paper + peak + quantum walk + Monte Carlo risk).

Usage:
  python scripts/start_quantum_bot.py [--config path] [--capital 1000] [--cycle-seconds 60] [--cycles 0] [--paper-days 0]
  # or from repo root:
  python main.py quantum [--config path] [--capital 1000] ...
"""

import os
import sys

# Run from repo root so imports resolve
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

# Delegate to main
from main import run_quantum_bot, logger
import argparse

def main():
    p = argparse.ArgumentParser(description="Argus quantum bot (paper + peak + quantum)")
    p.add_argument("--config", default=None, help="Config file path")
    p.add_argument("--capital", "-c", type=float, default=1000.0, help="Starting capital")
    p.add_argument("--cycle-seconds", type=float, default=60.0, help="Seconds between cycles")
    p.add_argument("--cycles", type=int, default=0, help="Max cycles (0 = unlimited)")
    p.add_argument("--paper-days", type=float, default=0.0, help="Approximate paper run days")
    p.add_argument("--no-multilang", action="store_true", help="Disable multilang")
    args = p.parse_args()
    logger.info("Quantum bot entrypoint: starting unified system with quantum preset")
    run_quantum_bot(
        args.capital,
        config_file=args.config,
        cycle_seconds=args.cycle_seconds,
        cycles=args.cycles,
        paper_days=args.paper_days,
        no_multilang=args.no_multilang,
    )

if __name__ == "__main__":
    main()
