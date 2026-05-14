"""
Position reconciliation — periodically compares internal positions
against exchange balances and corrects drift.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PositionReconciler:
    """
    Periodic reconciliation of internal position state vs exchange balances.

    Runs every `interval_seconds` (default 300 = 5 minutes).
    Detects and logs drift. Can auto-correct if enabled.
    """

    def __init__(
        self,
        portfolio_manager: Any,
        exchange_manager: Any,
        interval_seconds: float = 300.0,
        auto_correct: bool = True,
        drift_threshold_pct: float = 0.01,  # 1% drift triggers correction
    ):
        self._pm = portfolio_manager
        self._em = exchange_manager
        self._interval = interval_seconds
        self._auto_correct = auto_correct
        self._drift_threshold = drift_threshold_pct
        self._task: Optional[asyncio.Task] = None
        self._last_reconcile: float = 0.0
        self._drift_count = 0
        self._reconcile_count = 0

    async def reconcile_once(self) -> Dict[str, Any]:
        """Run one reconciliation cycle. Returns report."""
        report = {
            "timestamp": time.time(),
            "drifts": [],
            "corrected": [],
            "errors": [],
        }

        try:
            # Get exchange balances
            balances = await self._em.get_balances()
            if balances is None:
                report["errors"].append("exchange returned None balances")
                return report

            # Get internal positions
            internal = self._pm.get_positions()

            # Check each internal position against exchange
            all_symbols = set(internal.keys())
            for asset, balance in balances.items():
                bal = float(balance) if balance else 0.0
                if bal > 0.0001:
                    # Find matching symbol (e.g., BTC → BTC/USD)
                    for sym in list(internal.keys()) + [f"{asset}/USD", f"{asset}/AUD"]:
                        if sym.startswith(asset):
                            all_symbols.add(sym)

            for symbol in all_symbols:
                int_pos = internal.get(symbol, {})
                int_qty = float(int_pos.get("quantity", 0) or 0)

                # Extract base asset from symbol
                base = symbol.split("/")[0] if "/" in symbol else symbol
                exch_qty = float(balances.get(base, 0) or 0)

                if abs(int_qty - exch_qty) > 0.0001:
                    drift_pct = abs(int_qty - exch_qty) / max(int_qty, exch_qty, 0.0001)

                    drift_info = {
                        "symbol": symbol,
                        "internal_qty": int_qty,
                        "exchange_qty": exch_qty,
                        "drift_pct": round(drift_pct * 100, 2),
                    }
                    report["drifts"].append(drift_info)
                    self._drift_count += 1

                    if self._auto_correct and drift_pct > self._drift_threshold:
                        price = float(int_pos.get("current_price", 0) or 0)
                        self._pm.reconcile_position(symbol, exch_qty, price)
                        report["corrected"].append(symbol)
                        logger.warning(
                            "PositionReconciler: corrected %s — internal=%.8f → exchange=%.8f (drift=%.1f%%)",
                            symbol, int_qty, exch_qty, drift_pct * 100,
                        )

            self._reconcile_count += 1
            self._last_reconcile = time.time()

        except Exception as e:
            report["errors"].append(str(e))
            logger.error("PositionReconciler: reconciliation failed: %s", e)

        return report

    async def start(self) -> None:
        """Start periodic reconciliation loop."""
        if self._task is not None:
            return

        async def _loop():
            while True:
                try:
                    await asyncio.sleep(self._interval)
                    report = await self.reconcile_once()
                    if report["drifts"]:
                        logger.info(
                            "PositionReconciler: found %d drifts, corrected %d",
                            len(report["drifts"]), len(report["corrected"]),
                        )
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("PositionReconciler: loop error: %s", e)

        self._task = asyncio.create_task(_loop())
        logger.info("PositionReconciler: started (interval=%.0fs)", self._interval)

    async def stop(self) -> None:
        """Stop periodic reconciliation."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "reconcile_count": self._reconcile_count,
            "drift_count": self._drift_count,
            "last_reconcile": self._last_reconcile,
            "auto_correct": self._auto_correct,
        }
