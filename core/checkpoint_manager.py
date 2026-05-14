"""
Batch 3 – Checkpoint Manager
==============================
Persists and restores system state (portfolio, positions, trade history,
strategy allocator weights, evolved params) so the bot can resume after
a restart without losing context.

Checkpoints are written atomically (write-then-rename) to:
    data/checkpoints/checkpoint_<timestamp>.json

The latest checkpoint is symlinked / copied to:
    data/checkpoints/latest.json

Rotation: keeps the last N checkpoints (configurable); older files are pruned.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from unified_trading_system import UnifiedSystemArchitecture

logger = logging.getLogger(__name__)

_DEFAULT_DIR = "data/checkpoints"
_DEFAULT_KEEP = 10
_DEFAULT_INTERVAL = 300   # seconds between auto-checkpoints


class CheckpointManager:
    """
    Periodic + on-demand checkpoint writer/restorer for the unified trading
    system state.
    """

    def __init__(self, system: "UnifiedSystemArchitecture") -> None:
        self._sys = system
        cfg = getattr(system, "config", None)
        self._dir = Path(
            str(getattr(cfg, "checkpoint_dir", _DEFAULT_DIR) or _DEFAULT_DIR)
        )
        self._keep = int(getattr(cfg, "checkpoint_keep", _DEFAULT_KEEP) or _DEFAULT_KEEP)
        self._interval = float(
            getattr(cfg, "checkpoint_interval_seconds", _DEFAULT_INTERVAL) or _DEFAULT_INTERVAL
        )
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_checkpoint_ts: float = 0.0
        self._total_written: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._dir.mkdir(parents=True, exist_ok=True)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._task = loop.create_task(
                    self._checkpoint_loop(), name="checkpoint_manager"
                )
        except Exception as exc:
            logger.debug("CheckpointManager: could not create async task: %s", exc)

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    # ------------------------------------------------------------------
    # Checkpoint write
    # ------------------------------------------------------------------

    def save(self, label: str = "") -> Optional[str]:
        """Write a checkpoint synchronously.  Returns path or None on error."""
        try:
            snapshot = self._build_snapshot()
            ts = int(time.time())
            suffix = f"_{label}" if label else ""
            name = f"checkpoint_{ts}{suffix}.json"
            tmp_path = self._dir / (name + ".tmp")
            final_path = self._dir / name
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, default=str)
            tmp_path.rename(final_path)
            # Overwrite latest.json atomically
            latest = self._dir / "latest.json"
            shutil.copy2(final_path, latest)
            self._last_checkpoint_ts = time.time()
            self._total_written += 1
            self._prune()
            logger.info("Checkpoint saved: %s", final_path.name)
            return str(final_path)
        except Exception as exc:
            logger.warning("CheckpointManager.save error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Checkpoint restore
    # ------------------------------------------------------------------

    def restore(self, path: Optional[str] = None) -> bool:
        """
        Restore system state from a checkpoint file.
        If *path* is None, uses latest.json.
        Returns True on success.
        """
        target = Path(path) if path else self._dir / "latest.json"
        if not target.exists():
            logger.warning("CheckpointManager.restore: %s not found", target)
            return False
        try:
            with open(target, "r", encoding="utf-8") as f:
                snap = json.load(f)
            self._apply_snapshot(snap)
            logger.info("Checkpoint restored from %s", target.name)
            return True
        except Exception as exc:
            logger.warning("CheckpointManager.restore error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _checkpoint_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._interval)
                self.save()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("CheckpointManager loop error: %s", exc)

    def _build_snapshot(self) -> Dict[str, Any]:
        s = self._sys
        return {
            "ts": time.time(),
            "run_id": getattr(s, "run_id", ""),
            "run_mode": str(getattr(getattr(s, "config", None), "run_mode", "paper")),
            "portfolio_value_aud": getattr(s, "portfolio_value_aud", 0.0),
            "cash_balance_aud": getattr(s, "cash_balance_aud", 0.0),
            "peak_equity_aud": getattr(s, "peak_equity_aud", 0.0),
            "total_pnl_aud": getattr(s, "total_pnl_aud", 0.0),
            "realized_pnl_aud": getattr(s, "realized_pnl_aud", 0.0),
            "total_fees_aud": getattr(s, "total_fees_aud", 0.0),
            "total_trades": getattr(s, "total_trades", 0),
            "winning_trades": getattr(s, "winning_trades", 0),
            "losing_trades": getattr(s, "losing_trades", 0),
            "consecutive_losses": getattr(s, "consecutive_losses", 0),
            "completed_cycles": getattr(s, "_completed_cycles", 0),
            "positions": dict(getattr(s, "positions", {}) or {}),
            "latest_regime": getattr(s, "_latest_regime_label", ""),
            "strategy_allocator": self._dump_allocator(),
        }

    def _dump_allocator(self) -> Dict[str, Any]:
        alloc = getattr(self._sys, "strategy_allocator", None)
        if alloc is None:
            return {}
        try:
            if hasattr(alloc, "get_stats"):
                return alloc.get_stats()
            if hasattr(alloc, "stats"):
                return alloc.stats()
        except Exception:
            pass
        return {}

    def _apply_snapshot(self, snap: Dict[str, Any]) -> None:
        s = self._sys
        for attr in (
            "portfolio_value_aud",
            "cash_balance_aud",
            "peak_equity_aud",
            "total_pnl_aud",
            "realized_pnl_aud",
            "total_fees_aud",
            "total_trades",
            "winning_trades",
            "losing_trades",
            "consecutive_losses",
        ):
            if attr in snap:
                try:
                    setattr(s, attr, snap[attr])
                except Exception:
                    pass
        if "positions" in snap and isinstance(snap["positions"], dict):
            try:
                s.positions.update(snap["positions"])
            except Exception:
                pass
        if "latest_regime" in snap:
            try:
                s._latest_regime_label = str(snap["latest_regime"])
            except Exception:
                pass

    def _prune(self) -> None:
        """Delete old checkpoint files beyond the keep limit."""
        try:
            files = sorted(
                [f for f in self._dir.glob("checkpoint_*.json") if not f.name.startswith("latest")],
                key=lambda f: f.stat().st_mtime,
            )
            excess = files[: max(0, len(files) - self._keep)]
            for f in excess:
                try:
                    f.unlink()
                except Exception:
                    pass
        except Exception as exc:
            logger.debug("CheckpointManager._prune error: %s", exc)

    def stats(self) -> Dict[str, Any]:
        return {
            "total_written": self._total_written,
            "last_checkpoint_ts": self._last_checkpoint_ts,
            "checkpoint_dir": str(self._dir),
        }
