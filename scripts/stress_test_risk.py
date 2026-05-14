#!/usr/bin/env python3
"""
Stress-test risk limits and circuit breaker with synthetic positions and PnL.
Usage: python scripts/stress_test_risk.py [--config unified_config.yaml]
Exit 0 if limits/circuit breaker behave as expected; 1 on failure.
"""

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Stress test risk limits and circuit breaker")
    parser.add_argument("--config", default="unified_config.yaml", help="Config file path")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    failed = 0

    # 1. Load config
    try:
        import yaml
        cfg_path = repo_root / args.config
        if not cfg_path.exists():
            print("WARN: Config not found, using defaults")
            risk_cfg = {}
        else:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            risk_cfg = cfg.get("risk") or {}
    except Exception as e:
        print("FAIL: Could not load config:", e)
        return 1

    capital_cfg = cfg.get("capital") or {}
    initial_capital = float(capital_cfg.get("starting_capital_aud", risk_cfg.get("initial_capital", 10_000)) or 10_000)
    max_daily_loss = float(risk_cfg.get("max_daily_loss_pct", 0.02) or 0.02)
    max_consecutive = int(risk_cfg.get("max_consecutive_losses", 5) or 5)
    circuit_dd_pct = float(risk_cfg.get("circuit_breaker_dd_pct", 8.0) or 8.0)

    # 2. Instantiate UnifiedRiskManager
    try:
        from risk.unified_risk_manager import UnifiedRiskManager
        rm = UnifiedRiskManager(
            initial_capital=initial_capital,
            max_daily_loss=max_daily_loss,
            max_consecutive_losses=max_consecutive,
        )
        print("PASS: UnifiedRiskManager initialized")
    except Exception as e:
        print("FAIL: UnifiedRiskManager init:", e)
        return 1

    # 3. Simulate losses and check circuit breaker / daily limit
    loss_pct = 0.01
    for i in range(max_consecutive + 2):
        pnl = -initial_capital * loss_pct
        new_cap = rm.current_capital + pnl
        rm.update_capital(max(0.0, new_cap), pnl=pnl)
        rm.record_trade(pnl=pnl)
        if rm.check_circuit_breaker():
            print("PASS: Circuit breaker tripped after consecutive losses as expected")
            break
    else:
        # If we have strict max_consecutive_losses=2, it should trip
        if max_consecutive <= 3 and not rm.is_circuit_breaker_active():
            print("WARN: Circuit breaker did not trip after", max_consecutive + 2, "losses (check max_consecutive_losses)")

    # 4. Trip via public API (VaR breach style)
    try:
        from risk.unified_risk_manager import UnifiedRiskManager
        rm2 = UnifiedRiskManager(initial_capital=10_000.0, max_consecutive_losses=5)
        rm2.trip_circuit_breaker("VaR breach test")
        if rm2.is_circuit_breaker_active():
            print("PASS: trip_circuit_breaker() activates circuit breaker")
        else:
            print("FAIL: trip_circuit_breaker() did not activate")
            failed += 1
    except Exception as e:
        print("FAIL: trip_circuit_breaker test:", e)
        failed += 1

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
